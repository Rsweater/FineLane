model = dict(
    type="Detector",
    backbone=dict(type="DLANet"),
    # neck=dict(
    #     type="CLRerNetFPN",
    #     in_channels=[128, 256, 512],
    #     out_channels=64,
    #     num_outs=3,
    # ),
    neck=dict(
        type='SPGChannelMapper',
        in_channels=[128, 256, 512],
        kernel_size=1,
        out_channels=[128, 128, 256],
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32)),
    lane_head=dict(
        type="SPGLaneHead",
        anchor_generator=dict(
            type="RefGenerator",
            num_priors=64,
            num_points=72,
        ),
        img_w=800,
        img_h=320,
        prior_feat_channels=256,
        fc_hidden_dim=128,
        seg_channel=128+128+256,
        attention_in_channels=128,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        attention=dict(type="MsPFusion"),
        loss_cls=dict(type="FocalLoss", alpha=0.25, gamma=2, loss_weight=2.0),
        loss_bbox=dict(type="SmoothL1Loss", reduction="none", loss_weight=0.2),
        loss_iou=dict(
            type="LaneIoULoss",
            lane_width=7.5 / 800,
            loss_weight=4.0,
        ),
        loss_seg=dict(
            type="CLRNetSegLoss",
            loss_weight=2.0,
            num_classes=9,  # 8 lanes + 1 background
            ignore_label=255,
            bg_weight=0.4,
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="DynamicTopkAssigner",
            min_topk=1,
            cost_combination=1,
            cls_cost=dict(type="FocalCost", weight=1.0),
            reg_cost=dict(type="DistanceCost", weight=0.0),
            iou_dynamick=dict(
                type="LaneIoUCost",
                lane_width=7.5 / 800,
                use_pred_start_end=False,
                use_giou=True,
            ),
            iou_cost=dict(
                type="LaneIoUCost",
                lane_width=30 / 800,
                use_pred_start_end=True,
                use_giou=True,
            ),
        ),
        assigner_proposals_topk=4,
        assigner_lane_topk=4,
    ),
    test_cfg=dict(
        # conf threshold is obtained from cross-validation
        # of the train set. The following value is
        # for CLRerNet w/ DLA34 & EMA model.
        conf_threshold=0.41,
        use_nms=True,
        as_lanes=True,
        nms_thres=50,
        nms_topk=8,
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
    ),
)
