
import torch
import torch.nn as nn
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils import build_transformer


@ATTENTION.register_module()
class QueryUpdate(nn.Module):
    """
    CLRNet ROIGather module to process pooled features
    and make them interact with global information.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/roi_gather.py
    """

    def __init__(
        self,
        in_channels,
        num_priors,
        sample_points,
        fc_hidden_dim,
        refine_layers,
        mid_channels=64,
        cross_attention_weight=1.0,
    ):
        """
        Args:
            in_channels (int): Number of input feature channels.
            num_priors (int): Number of priors (anchors).
            sample_points (int): Number of pooling sample points (rows).
            fc_hidden_dim (int): FC middle channel number.
            refine_layers (int): The number of refine levels.
            mid_channels (int): The number of input channels to catconv.
            cross_attention_weight (float): Weight to fuse cross attention result.
        """
        super(QueryUpdate, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.cross_attention_weight = cross_attention_weight

        if self.cross_attention_weight > 0:
            self.attention = DII()

        # learnable layers
        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(
                    in_channels,
                    mid_channels,
                    (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm_cfg=dict(type='BN'),
                )
            )

            self.catconv.append(
                ConvModule(
                    mid_channels * (i + 1),
                    in_channels,
                    (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm_cfg=dict(type='BN'),
                )
            )

        self.fc = nn.Linear(sample_points * in_channels, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        """
        Args:
            x (List[torch.Tensor]): List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, C, Ns, 1).
            layer_index (int): Current refine layer index.
        Returns:
            cat_feat (torch.Tensor): Fused feature tensor, shape (B * Np, C, Ns, 1).
        B: batch size, Np: number of priors (anchors), Ns: number of sample points (rows).
        """
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, proposal_features, layer_index):
        """
        ROIGather forward function.
        Args:
            roi_features (List[torch.Tensor]): List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, Cin, Ns, 1).
            fmap_pyramid (List[torch.Tensor]): Multi-level feature pyramid.
                Each tensor has a shape (B, Cin, H_i, W_i) where i is the pyramid level.
            layer_index (int): The current refine layer index.
        Returns:
            roi (torch.Tensor): Output feature tensors, shape (B, Np, Ch).
        B: batch size, Np: number of priors (anchors), Ns: number of sample points (rows),
        Cin: input channel number, Ch: hidden channel number.
        """
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            fmap_pyramid: feature map pyramid
            layer_index: currently on which layer to refine
        Return:
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        # [B * Np, Cin, Ns, 1] * N -> [B * Np, Cin, Ns, 1]
        roi = self.roi_fea(roi_features, layer_index)
        # [B * Np, Cin, Ns, 1] -> [B * Np, Cin, Ns]
        roi = roi.squeeze(-1)

        # if self.cross_attention_weight > 0:
        #     context = self.attention(roi, fmap)
            # roi = roi + self.cross_attention_weight * F.dropout(
            #     context, p=0.1, training=self.training
            # )

        return self.attention(roi, proposal_features)
    
class DII(nn.Module):
    r"""Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (dict): The activation config for FFNs.
        dynamic_conv_cfg (dict): The convolution config
            for DynamicConv.
        loss_iou (dict): The config for iou or giou loss.

    """

    def __init__(self,
                #  num_classes=80,
                 num_offsets=72,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=256,
                 in_channels=64,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                #  loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(DII, self).__init__()

        self.in_channels = in_channels
        self.n_offsets = num_offsets
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.instance_interactive_conv = DynamicConv(
            in_channels=64,
            feat_channels=36, # Np, num of sample points
            out_channels=64,
            input_feat_shape=7,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'))
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(
            dict(type='LN'), in_channels)[1]

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        self.fc_cls = nn.Linear(in_channels, 2)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        self.fc_reg = nn.Linear(in_channels, self.n_offsets + 4)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(DII, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size, num_proposals, feature_dimensions).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        batch_size, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2) # (N, B, C)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        attn_feats = proposal_feat.permute(1, 0, 2)

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat)
        bbox_delta = self.fc_reg(reg_feat)

        return cls_score, bbox_delta, obj_feat.view(
            batch_size, num_proposals, self.in_channels), attn_feats
    

class DynamicConv(BaseModule):
    """Implements Dynamic Convolution.

    This module generate parameters for each sample and
    use bmm to implement 1*1 convolution. Code is modified
    from the `official github repo <https://github.com/PeizeSun/
    SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py#L258>`_ .

    Args:
        in_channels (int): The input feature channel.
            Defaults to 256.
        feat_channels (int): The inner feature channel.
            Defaults to 64.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        input_feat_shape (int): The shape of input feature.
            Defaults to 7.
        with_proj (bool): Project two-dimentional feature to
            one-dimentional feature. Default to True.
        act_cfg (dict): The activation config for DynamicConv.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=36,
                 out_channels=256,
                 input_feat_shape=7,
                 with_proj=True,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(DynamicConv, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * feat_channels
        if self.with_proj:
            self.fc_layer = nn.Linear(num_output, self.out_channels)
            self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, param_feature, input_feature):
        """Forward function for `DynamicConv`.

        Args:
            param_feature (Tensor): The feature can be used
                to generate the parameter, has shape
                (num_all_proposals, in_channels).
            input_feature (Tensor): Feature that
                interact with parameters, has shape
                (num_all_proposals, in_channels, Ns).

        Returns:
            Tensor: The output feature has shape
            (num_all_proposals, out_channels).
        """
        input_feature = input_feature.permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        if self.with_proj:
            features = features.flatten(1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features
