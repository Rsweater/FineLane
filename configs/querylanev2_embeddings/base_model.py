model = dict(
    type="Detector",
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth'
        )),
    neck=dict(
        type='SA_FPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        num_outs=3),
    lane_head=dict(
        type="QueryLaneHeadEmbeddings",
        # num_priors=10,
        prior_feat_channels=256,
        attention_in_channels=256,
        fc_hidden_dim=256,
        seg_channel=256*3,
        attention=dict(type="QueryROI", refine_layers=2),
        # loss_cls=dict(type="KorniaFocalLoss", alpha=0.25, gamma=2, loss_weight=0.1),
        loss_dist=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=10.0,
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            class_weight=1.0 / 0.4,
            reduction='mean',
            loss_weight=0.1
        ),
        loss_seg=dict(
            type='CLRNetSegLoss',
            loss_weight=0.1,
            num_classes=9,
            ignore_label=255,
            bg_weight=0.4
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='BezierHungarianAssigner',
            order=3,
            num_sample_points=100,
            alpha=0.8,
            window_size=9
        )
    ),
    test_cfg=dict(
        # dataset info
        dataset="vil100",
        ori_img_w='no fixed size',
        ori_img_h='no fixed size',
        cut_height='no fixed size',
        conf_threshold=0.4,
        window_size=9,
        max_num_lanes=6,
        num_sample_points=50,
        as_lane=True
    ),
)
