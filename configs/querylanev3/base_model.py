model = dict(
    type="Detector",
    backbone=dict(type="DLANet"),
    neck=dict(
        type="SA_FPN",
        in_channels=[128, 256, 512],
        out_channels=256,
        num_outs=3,
    ),
    lane_head=dict(
        type="QueryLaneHeadV31",
        prior_feat_channels=256,
        attention_in_channels=256,
        fc_hidden_dim=256,
        seg_channel=256*3,
        num_priors=24,
        attention=dict(type="QueryFeat"),
        # loss_cls=dict(type="FocalLoss", alpha=0.25, gamma=2, loss_weight=0.1),
        loss_dist=dict(
            type='ChamferLoss',
            # reduction='mean',
            loss_weight=2.0,
            r = 10/800,
        ),
        loss_length=dict(
            type='LengthLoss',
            # reduction='mean',
            loss_weight=2.0,
        ),
        loss_endpoint=dict(
            type='EndpointLoss',
            # reduction='mean',
            loss_weight=2.0,
        ),
        loss_cls=dict(
            type='FocalLoss', 
            alpha=0.25, 
            gamma=2, 
            loss_weight=0.1
        ),
        loss_seg=dict(
            type="CLRNetSegLoss",
            loss_weight=0.1,
            num_classes=5,  # 4 lanes + 1 background
            ignore_label=255,
            bg_weight=0.4,
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="BezierDynamicTopkAssigner",
            order=3, topk=1,
            num_sample_points=100,
            cls_cost=dict(type="FocalCost", weight=1.0), 
            iou_cost=dict(type="CIoUCost", weight=3.0, r=10/800), 
            length_cost=dict(type="LengthCost", weight=1.0), 
            endpoint_cost=dict(type="EndpointCost", weight=1.0),
        )
    ),
    test_cfg=dict(
        # dataset info
        dataset="culane",
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
        # inference settings
        conf_threshold=0.4,
        window_size=-1,
        max_num_lanes=4,
        num_sample_points=50,
        as_lane=True
    ),
)
