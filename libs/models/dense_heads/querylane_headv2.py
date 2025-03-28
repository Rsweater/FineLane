"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_attention
from mmdet.core import build_assigner
from mmdet.models.builder import build_loss
from mmdet.models.builder import HEADS

from libs.models.dense_heads.seg_decoder import SegDecoder
from libs.models.layers.fusion_blocks import CSAM2d
from libs.utils.lane_utils import Lane
from libs.core.lane.bezier_curve import BezierCurve

@HEADS.register_module
class QueryLaneHeadV2(nn.Module):
    def __init__(
        self,
        img_w=800,
        img_h=320,
        prior_topk=25,
        num_priors=64,
        prior_feat_channels=128,
        attention_in_channels=128,
        fc_hidden_dim=64,
        seg_channel=64+64+64,
        num_fc=2,
        feat_sample_points=30,
        loss_sample_points=100,
        attention=None,
        loss_cls=None,
        loss_dist=None,
        loss_seg=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super(QueryLaneHeadV2, self).__init__()
        self.bezier_curve = BezierCurve(order=3)
        self.img_w = img_w
        self.img_h = img_h
        self.prior_topk = prior_topk
        self.refine_layers = attention.refine_layers + 1
        self.num_priors = attention.num_priors = num_priors
        self.feat_sample_points = attention.sample_points = feat_sample_points
        self.fc_hidden_dim = attention.fc_hidden_dim = fc_hidden_dim
        attention.in_channels = attention_in_channels 
        self.prior_feat_channels = prior_feat_channels
        self.loss_sample_points = loss_sample_points

        self.channel_attention = CSAM2d(prior_feat_channels)

        # proposal fc stage
        self.pro_num_fixed_fc = nn.Linear(250, self.num_priors)
        self.pro_num_fixed_fc_norm = nn.LayerNorm(self.num_priors)

        pro_shared_branchs_fc = list()
        for _ in range(num_fc):
            pro_shared_branchs_fc.append(nn.Sequential(
                nn.Linear(prior_feat_channels, prior_feat_channels),
                nn.ReLU(inplace=True)
            ))
        self.pro_shared_branchs_fc = nn.Sequential(*pro_shared_branchs_fc)
        self.pro_cls_layers = nn.Conv1d(prior_feat_channels, 1, 1)
        self.pro_reg_layers = nn.Conv1d(prior_feat_channels, 8, 1)

        self.attention = build_attention(attention)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
            ]
            cls_modules += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
            ]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.reg_layers = nn.Linear(self.fc_hidden_dim, 8)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 1)
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_dist = build_loss(loss_dist)
        self.loss_seg = build_loss(loss_seg) if loss_seg["loss_weight"] > 0 else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
        # Auxiliary head
        if self.loss_seg:
            self.seg_decoder = SegDecoder(
                self.img_h, self.img_w,
                seg_channel, loss_seg.num_classes,
            )

        self.init_weights()

    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)

        for m in self.pro_cls_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        for m in self.pro_reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)

    def pool_prior_features(self, batch_features, prior_xs):
        """
        Pool features from the feature map along the prior points.
        Args:
            batch_features (torch.Tensor): Input feature maps, shape: (B, C, H, W)
            prior_xs (torch.Tensor):. Prior points, shape (B, Np, Ns)
                where Np is the number of priors and Ns is the number of sample points.
        Returns:
            # feature (torch.Tensor): Pooled features with shape (B * Np, C, Ns, 1).
            feature (torch.Tensor): Pooled features with shape (B * Np, C, Ns)
        """

        batch_size = batch_features.shape[0]
        prior_xs = prior_xs.view(batch_size, self.num_priors, -1, 2)
        prior_xs = prior_xs * 2.0 - 1.0
        feature = F.grid_sample(batch_features, prior_xs, align_corners=True).permute(
            0, 2, 1, 3
        ).contiguous()
        feature = feature.reshape(
            batch_size * self.num_priors,
            self.fc_hidden_dim,
            self.feat_sample_points,
            1,
        )
        return feature

    def forward(self, x, **kwargs):
        """
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            feature_pyramid: Input features (list[Tensor]). Each tensor has a shape (B, C, H_i, W_i),
                where i is the pyramid level.
                Example of shapes: ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        Returns:
            pred_dict (List[dict]): List of prediction dicts each of which containins multiple lane predictions.
                cls_logits (torch.Tensor): 2-class logits with shape (B, Np, 2).
                anchor_params (torch.Tensor): anchor parameters with shape (B, Np, 3).
                lengths (torch.Tensor): lane lengths in row numbers with shape (B, Np, 1).
                xs (torch.Tensor): x coordinates of the lane points with shape (B, Np, Nr).

        B: batch size, Np: number of priors (anchors), Nr: num_points (rows).
        """
        batch_size = x[0].shape[0]
        feature_pyramid = list(x[len(x) - self.refine_layers :])
        feature_pyramid.reverse()  # (1, 64, 10, 25) (1, 64, 20, 50) (1, 64, 40, 100)
        proposal_feature = feature_pyramid[0]  # (1, 64, 10, 25)
        prefusion_feature = feature_pyramid[1:]  # (1, 64, 20, 50) (1, 64, 40, 100)

        # lane preselector
        pro_feature = self.channel_attention(proposal_feature).reshape(batch_size*self.prior_feat_channels, -1) # (B*C, H*W)
        pro_num_fixed = self.pro_num_fixed_fc_norm(self.pro_num_fixed_fc(pro_feature)).reshape(batch_size*self.num_priors, -1) # (B*Np, C)

        # proposal mlp stage
        pro_fc_feature = self.pro_shared_branchs_fc(pro_num_fixed) # (B*Np, C)
        pro_fc_feature = pro_fc_feature.reshape(batch_size, self.num_priors, -1).permute(0, 2, 1).contiguous() # (B, C, Np)

        # proposal detection head
        pro_reg = self.pro_reg_layers(pro_fc_feature).permute(0, 2, 1).contiguous() # (B, Np, 4 + Nr)
        pro_cls = self.pro_cls_layers(pro_fc_feature.clone()).permute(0, 2, 1).contiguous() # (B, Np, 1)

        # pro_cls_scores = pro_cls[:, :, 0]
        # _, topk_inds = torch.topk(pro_cls[:, :, 0], k=self.prior_topk, dim=, largest=True, sorted=False)
        
        # select topk by topk_ind
        # pred_dict = {
        #     "cls_logits": pro_cls[torch.arange(batch_size)[:, None], topk_inds], # (B, 20, 2)
        #     # "anchor_params": anchor_xyt[torch.arange(batch_size)[:, None], topk_inds], # (B, 20, 3)
        #     # "lengths": pro_reg[torch.arange(batch_size)[:, None], topk_inds, 3:4], # (B, 20, 1)
        #     "control_points": pro_reg[torch.arange(batch_size)[:, None], topk_inds], # (B, 20, 8)
        #     "proposal": torch.ones(batch_size, dtype=torch.bool) # proposal flag(B, )
        # }
        pred_dict = {
            "cls_logits": pro_cls,
            "control_points": pro_reg.reshape(batch_size, self.num_priors, 4, 2), 
            "proposal": torch.ones(batch_size, dtype=torch.bool)
        }
        predictions_list = []
        predictions_list.append(pred_dict)

        pred_sample_points = self.bezier_curve.get_sample_points(
            pro_reg.reshape(-1, 4, 2), num_sample_points=self.feat_sample_points
        )

        query_feats = pro_fc_feature.permute(0, 2, 1).contiguous()
        pooled_features = []
        for stage, feature in enumerate(prefusion_feature):
            pooled_feature = self.pool_prior_features(feature, pred_sample_points)
            pooled_features.append(pooled_feature)
            query_feats = self.attention(
                pooled_features, query_feats, stage
            )
        fc_features = query_feats.view(self.num_priors, batch_size, -1).reshape(
            batch_size * self.num_priors, self.fc_hidden_dim
        )  # [B * Np, Ch]

        # 3. cls and reg heads
        cls_features = fc_features.clone()
        reg_features = fc_features.clone()
        for cls_layer in self.cls_modules:
            cls_features = cls_layer(cls_features)
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(reg_features)

        cls_logits = self.cls_layers(cls_features)
        cls_logits = cls_logits.reshape(
            batch_size, -1, cls_logits.shape[1]
        )  # (B, Np, 2)

        reg = self.reg_layers(reg_features)
        reg = reg.reshape(batch_size, -1, reg.shape[1])  # (B, Np, 4 + Nr)

        # 4. reg processing
        pred_dict = {
            "cls_logits": cls_logits,
            "control_points": reg.reshape(batch_size, self.num_priors, 4, 2),
            "proposal": torch.zeros(batch_size, dtype=torch.bool)
        }

        predictions_list.append(pred_dict)

        return predictions_list

    def loss(self, out_dict, img_metas):
        """Loss calculation from the network output.

        Args:
            out_dict (dict[torch.Tensor]): Output dict from the network containing:
                predictions (List[dict]): 3-layer prediction dicts each of which contains:
                    cls_logits: shape (B, Np, 2), anchor_params: shape (B, Np, 3),
                    lengths: shape (B, Np, 1) and xs: shape (B, Np, Nr).
                seg (torch.Tensor): segmentation maps, shape (B, C, H, W).
                where
                B: batch size, Np: number of priors (anchors), Nr: number of rows,
                C: segmentation channels, H and W: the largest feature's spatial shape.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = len(img_metas)
        device = out_dict["predictions"][0]["cls_logits"].device
        cls_loss = torch.tensor(0.0).to(device)
        dist_loss = torch.tensor(0.0).to(device)

        for stage in range(len(out_dict["predictions"])):
            for b, img_meta in enumerate(img_metas):
                pred_dict = {
                    k: v[b] for k, v in out_dict["predictions"][stage].items()
                }
                cls_pred = pred_dict["cls_logits"]
                reg_pred = pred_dict['control_points'] # (W, 4, 2)
                # get target results
                gt_lanes = img_meta['gt_lanes'].clone().to(device) # (N_lanes, 4, 2)
                cls_target = torch.zeros_like(cls_pred) # (W, 1)

                if len(gt_lanes) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_loss = (
                        cls_loss + self.loss_cls(cls_pred, cls_target).sum()
                    )
                    continue

                with torch.no_grad():
                    (
                        matched_row_inds,
                        matched_col_inds,
                    ) = self.assigner.assign(
                        pred_dict, gt_lanes.clone(), img_meta
                    )

                # classification targets
                cls_target[matched_row_inds, :] = 1
                cls_loss = (
                    cls_loss
                    + self.loss_cls(cls_pred, cls_target)
                )

                # regression loss
                pred_control_points = reg_pred[matched_row_inds] # (N_matched, 4, 2)
                gt_control_points = gt_lanes[matched_col_inds] # (N_matched, 4, 2)
                pred_sample_points = self.bezier_curve.get_sample_points(
                    pred_control_points, num_sample_points=self.loss_sample_points
                )
                gt_sample_points = self.bezier_curve.get_sample_points(
                    gt_control_points, num_sample_points=self.loss_sample_points
                )
                dist_loss = (
                    dist_loss + self.loss_dist(pred_sample_points, gt_sample_points)
                )

        cls_loss = cls_loss / batch_size * len(out_dict["predictions"])
        dist_loss = dist_loss / batch_size * len(out_dict["predictions"])
        loss_dict = {
            "loss_cls": cls_loss,
            "loss_dist": dist_loss,
        }

        # extra segmentation loss
        if self.loss_seg:
            tgt_masks = np.array([t["gt_masks"].data[0] for t in img_metas])
            tgt_masks = torch.tensor(tgt_masks).long().to(device)  # (B, H, W)
            pred_masks = F.interpolate(
                out_dict["seg"], mode="bilinear", align_corners=False,
                size=[tgt_masks.shape[1], tgt_masks.shape[2]], 
            ) # (B, n, H, W) -> (B, n, img_H, img_W)
            loss_dict["loss_seg"] = self.loss_seg(pred_masks, tgt_masks)

        return loss_dict

    def forward_train(self, x, img_metas, **kwargs):
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        predictions = self(x)
        out_dict = {"predictions": predictions}
        if self.loss_seg:
            out_dict["seg"] = self.forward_seg(x)

        losses = self.loss(out_dict, img_metas)
        return losses

    def forward_seg(self, x):
        """Forward function for training mode.
        Args:
            x (list[torch.tensor]): Features from backbone.
        Returns:
            torch.tensor: segmentation maps, shape (B, C, H, W), where
            B: batch size, C: segmentation channels, H and W: the largest feature's spatial shape.
        """
        batch_features = list(x[len(x) - self.refine_layers :])
        batch_features.reverse()

        seg_features = torch.cat(
            [
                F.interpolate(
                    feature,
                    size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                    mode="bilinear",
                    align_corners=False,
                )
                for feature in batch_features
            ],
            dim=1,
        )
        seg = self.seg_decoder(seg_features)
        return seg

    def get_lanes(self, pred_dict, as_lane=True):
        """Get lanes from prediction results.
        Args:
            pred_dict (dict): A dictionary of prediction results.
            as_lane (bool): Whether to return lanes as a list of points or as a list of Bezier curves.
        Returns:
            list: A list of lanes.
        """
        assert (
            len(pred_dict["cls_logits"]) == 1
        ), "Only single-image prediction is available!"
        # filter out the conf lower than conf threshold
        scores = pred_dict["cls_logits"].squeeze() # (W)
        scores = scores.sigmoid() # (W)
        existences = scores > self.test_cfg.conf_threshold

        pred_control_points = pred_dict["control_points"].squeeze(dim=0) # (W, 4, 2)
        num_pred = pred_control_points.shape[0]

        if self.test_cfg.window_size > 0:
            _, max_indices = F.max_pool1d(
                scores.unsqueeze(0).unsqueeze(0).contiguous(),
                kernel_size=self.test_cfg.window_size,
                stride=1,
                padding=(self.test_cfg.window_size - 1) // 2,
                return_indices=True
            ) # (1, 1, W)
            max_indices = max_indices.squeeze(dim=1) # (1, W)
            indices = torch.arange(0, num_pred, dtype=scores.dtype, 
                device=scores.device).unsqueeze(dim=0).expand_as(max_indices) 
            local_maximas = (max_indices == indices) # (B, W)
            existences = existences * local_maximas

        valid_score = scores * existences[0]
        sorted_score, sorted_indices = torch.sort(valid_score, dim=0, descending=True)
        valid_indices = torch.nonzero(sorted_score, as_tuple=True)[0][:self.test_cfg.max_num_lanes]

        keep_index = sorted_indices[valid_indices] # (N_lanes, )
        scores = scores[keep_index] # (N_lanes, )
        pred_control_points = pred_control_points[keep_index] # (N_lanes, 4, 2)

        if len(keep_index) == 0:
            return [], []
        
        preds = self.predictions_to_lanes(scores, pred_control_points, as_lane)

        return preds, scores

    def predictions_to_lanes(self, scores, pred_control_points, as_lane=True):
        """Convert predictions to lanes.
        Args:
            pred_control_points (torch.Tensor): Predicted control points, shape (N_lanes, 4, 2).
            scores (torch.Tensor): Predicted scores, shape (N_lanes, ).
            as_lane (bool): Whether to return lanes as a list of points or as a list of Bezier curves.
        Returns:
            list: A list of lanes.
        """
        dataset = self.test_cfg.get("dataset", None)

        lanes = []
        for score, control_point in zip(scores, pred_control_points):
            # score: (1, )
            # control_point: (4, 2)
            score = score.detach().cpu().numpy()
            control_point = control_point.detach().cpu().numpy()

            if dataset == 'tusimple':
                ppl = 56
                gap = 10
                bezier_threshold = 5.0 / self.test_cfg.ori_img_h
                h_samples = np.array(
                    [1.0 - (ppl - i) * gap / self.test_cfg.ori_img_h for i in range(ppl)], dtype=np.float32
                )   # (56, )

                sample_point = self.bezier_curve.get_sample_points(
                    control_points_matrix=control_point,
                    num_sample_points=self.test_cfg.ori_img_h)  # (N_sample_points-720, 2)  2: (x, y)

                ys = (
                    sample_point[:, 1] * (self.test_cfg.ori_img_h - self.test_cfg.cut_height)
                     + self.test_cfg.cut_height
                ) / self.test_cfg.ori_img_h   # (720, )
                dis = np.abs(h_samples.reshape(ppl, -1) - ys)    # (56, 720)
                idx = np.argmin(dis, axis=-1)  # (56, )
                temp = []
                for i in range(ppl):
                    h = self.test_cfg.ori_img_h - (ppl - i) * gap
                    if dis[i][idx[i]] > bezier_threshold or sample_point[idx[i]][0] > 1 \
                            or sample_point[idx[i]][0] < 0:
                        temp.append([-2, h])
                    else:
                        temp.append([sample_point[idx[i]][0] * self.test_cfg.ori_img_w, h])
                temp = np.array(temp, dtype=np.float32)
                lanes.append(temp)
            else:
                sample_point = self.bezier_curve.get_sample_points(
                    control_points_matrix=control_point,
                    num_sample_points=self.test_cfg['num_sample_points'])      # (N_sample_points, 2)  2: (x, y)

                lane_xs = sample_point[:, 0]      # 由上向下
                lane_ys = sample_point[:, 1]

                x_mask = np.logical_and(lane_xs >= 0, lane_xs < 1)
                y_mask = np.logical_and(lane_ys >= 0, lane_ys < 1)
                mask = np.logical_and(x_mask, y_mask)

                lane_xs = lane_xs[mask]
                lane_ys = lane_ys[mask]
                lane_ys = (
                    lane_ys * (self.test_cfg.ori_img_h - self.test_cfg.cut_height) 
                    + self.test_cfg.cut_height
                ) / self.test_cfg.ori_img_h
                if len(lane_xs) <= 1:
                    continue
                points = np.stack((lane_xs, lane_ys), axis=1)  # (N_sample_points, 2)  normalized

                points = sorted(points, key=lambda x: x[1])
                filtered_points = []
                used = set()
                for p in points:
                    if p[1] not in used:
                        filtered_points.append(p)
                        used.add(p[1])
                points = np.array(filtered_points)
                if as_lane:
                    lane = Lane(points=points,
                                metadata={
                                    'conf': score,
                                })
                else:
                    lane = points
                lanes.append(lane)
        return lanes

    def simple_test(self, feats):
        """Test function without test-time augmentation.
        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the FPN.
        Returns:
            result_dict (dict): Inference result containing
                lanes (List[torch.Tensor]): List of lane tensors (shape: (N, 2))
                    or `Lane` objects, where N is the number of rows.
                scores (torch.Tensor): Confidence scores of the lanes.
        """
        pred_dict = self(feats)[-1]
        lanes, scores = self.get_lanes(pred_dict, as_lane=self.test_cfg.as_lane)
        result_dict = {
            "lanes": lanes,
            "scores": scores,
        }
        return result_dict
