from mmcv.cnn.bricks.registry import ATTENTION
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


@ATTENTION.register_module()
class QueryFeat(nn.Module):
    def __init__(
        self,
        in_channels,
        num_priors,
        # sample_points,
        # fc_hidden_dim,
        # refine_layers,
        # mid_channels=48,
        cross_attention_weight=1.0,
    ):
        super(QueryFeat, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.cross_attention_weight = cross_attention_weight

        if self.cross_attention_weight > 0:
            self.attention = AnchorVecFeatureMapAttention(num_priors, in_channels)

        # learnable layers
        # self.convs = nn.ModuleList()
        # self.catconv = nn.ModuleList()
        # for i in range(refine_layers):
        #     self.convs.append(
        #         ConvModule(
        #             in_channels,
        #             mid_channels,
        #             (9, 1),
        #             padding=(4, 0),
        #             bias=False,
        #             norm_cfg=dict(type='BN'),
        #         )
        #     )

        #     self.catconv.append(
        #         ConvModule(
        #             mid_channels * (i + 1),
        #             in_channels,
        #             (9, 1),
        #             padding=(4, 0),
        #             bias=False,
        #             norm_cfg=dict(type='BN'),
        #         )
        #     )

        # self.fc = nn.Linear(sample_points * in_channels, fc_hidden_dim)
        # self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def forward(self, prefusion_feature, proposal_features):

        if self.cross_attention_weight > 0:
            context = self.attention(proposal_features, prefusion_feature)
            proposal_features = proposal_features + self.cross_attention_weight * F.dropout(
                context, p=0.1, training=self.training
            )

        return proposal_features


class AnchorVecFeatureMapAttention(nn.Module):
    def __init__(self, n_query, dim):
        """
        Args:
            n_query (int): Number of queries (priors, anchors).
            dim (int): Key and Value dim.
        """
        super(AnchorVecFeatureMapAttention, self).__init__()
        self.dim = dim
        self.resize = FeatureResize()
        self.f_key = ConvModule(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
        )
        self.f_query = nn.Sequential(
            nn.Conv1d(
                in_channels=n_query,
                out_channels=n_query,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=n_query,
            ),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0
        )
        self.W = nn.Conv1d(
            in_channels=n_query,
            out_channels=n_query,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=n_query,
        )
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, query, feat):
        """
        Forward function for cross attention.
        Args:
            query (torch.Tensor): Feature maps at the current refine level, shape (B, Np, C).
            feat (torch.Tensor): Feature maps at the current refine level, shape (B, C, H, W).
        Returns:
            context (torch.Tensor): Output global context, shape (B, Np, C).
        B: batch size, Np: number of priors (anchors)
        """
        query = self.f_query(query)  # [B, Np, C] ->  [B, Np, C]
        key = self.f_key(feat)  # [B, C, H, W]
        key = self.resize(key)  # [B, C, H'W']
        value = self.f_value(feat)  # [B, C, H, W]
        value = self.resize(value)  # [B, C, H'W']
        value = value.permute(0, 2, 1)  # [B, H'W', C]

        # attention: [B, Np, C] x [B, C, H'W'] -> [B, Np, H'W']
        sim_map = torch.matmul(query, key)
        sim_map = (self.dim**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)  #  [B, Np, C]
        context = self.W(context)  #  [B, Np, C]
        return context


class FeatureResize(nn.Module):
    """Resize the feature map by interpolation."""
    def __init__(self, size=(10, 25)):
        """
        Args:
            size (tuple): Target size (H', W').
        """
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        """
        Forward function.
        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W).
        Returns:
            out (torch.Tensor): Resized tensor with shape (B, C, H'W').
        """
        x = F.interpolate(x, self.size)
        return x.flatten(2)