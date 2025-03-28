
import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost

from ..bezier_curve import BezierCurve


@BBOX_ASSIGNERS.register_module()
class BezierDynamicTopkAssigner(BaseAssigner):
    def __init__(
            self, 
            order=3, topk=1,
            num_sample_points=100, 
            cls_cost=None, iou_cost=None, 
            length_cost=None, endpoint_cost=None,
            ):
        self.topk = topk
        self.bezier_curve = BezierCurve(order=order)
        self.num_sample_points = num_sample_points
        self.cls_cost = build_match_cost(cls_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.length_cost = build_match_cost(length_cost)
        self.endpoint_cost = build_match_cost(endpoint_cost)

    def dynamic_k_assign(self, cost, ious_matrix, max_topk):
        """
        Assign grouth truths with priors dynamically.
        Args:
            cost: the assign cost, shape (Np, Ng).
            ious_matrix: iou of grouth truth and priors, shape (Np, Ng).
        Returns:
            torch.Tensor: the indices of assigned prior.
            torch.Tensor: the corresponding ground truth indices.
        Np: number of priors (anchors), Ng: number of GT lanes.
        """
        matching_matrix = torch.zeros_like(cost)
        ious_matrix[ious_matrix < 0] = 0.0
        topk_ious, _ = torch.topk(ious_matrix, max_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=max_topk)
        num_gt = cost.shape[1]
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[pos_idx, gt_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        matched_gt = matching_matrix.sum(1)
        if (matched_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
            matching_matrix[matched_gt > 1, 0] *= 0.0
            matching_matrix[matched_gt > 1, cost_argmin] = 1.0

        prior_idx = matching_matrix.sum(1).nonzero()
        gt_idx = matching_matrix[prior_idx].argmax(-1)
        return prior_idx.flatten(), gt_idx.flatten()

    def assign(
        self, 
        pred_dict,
        gt_lanes,
        # max_topk=1,
    ):
        # control points to sample points
        # (Np, N_sample_points, 2)
        pred_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=pred_dict['control_points'].detach().clone(),
                                                                 num_sample_points=self.num_sample_points)
        # (Ng, N_sample_points, 2)
        gt_sample_points = self.bezier_curve.get_sample_points(control_points_matrix=gt_lanes.detach().clone(),
                                                               num_sample_points=self.num_sample_points)

        # compute iou between predictions and targets, shape (Np, Ng)
        ious_matrix = self.iou_cost(pred_sample_points, gt_sample_points)
        ious_matrix = 1 - (1 - ious_matrix) / torch.max(1-ious_matrix) + 1e-2
        # compute endpoint cost between predictions and targets, shape (Np, Ng)
        # endpoints_matrix = self.endpoint_cost(pred_dict['control_points'], gt_lanes)
        # # compute length cost between predictions and targets, shape (Np, Ng)
        # length_matrix = self.length_cost(pred_sample_points, gt_sample_points)
        # compute cls cost between predictions and targets, shape (Np, Ng)
        cls_target = torch.ones(gt_lanes.shape[0], dtype=torch.long, device=gt_lanes.device)
        cls_martix = self.cls_cost(pred_dict["cls_logits"].detach().clone(), cls_target)

        cost = cls_martix + ious_matrix * 3.0 # + endpoints_matrix + length_matrix
        # assign grouth truths with priors dynamically
        matched_row_inds, matched_col_inds = self.dynamic_k_assign(
            cost, ious_matrix, max_topk=self.topk,
        )

        return matched_row_inds, matched_col_inds