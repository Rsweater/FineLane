_base_ = [
    "../base_model.py", "dataset.py",
    "../../_base_/default_runtime.py",
]

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.lane",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

cfg_name = "querylanev2_tusimple_r34.py"

model = dict(
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'torchvision://resnet34'
        )
    ),
    lane_head=dict(
        loss_seg=dict(
            loss_weight=0.75,
            num_classes=7,  # 6 lane + 1 background
        )
    ),
    train_cfg=dict(
        assigner=dict(
            type='BezierHungarianAssigner',
            order=3,
            num_sample_points=100,
            alpha=0.8,
            window_size=11
        )
    ),
    # training and testing settings
    test_cfg=dict(
        # dataset info
        # dataset="tusimple",
        ori_img_w=1280,
        ori_img_h=720,
        cut_height=160,
        # inference settings
        conf_threshold=0.4,
        window_size=11,
        max_num_lanes=4,
        num_sample_points=50,
    ),
)

total_epochs = 1000
evaluation = dict(start=10, interval=10)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

data = dict(samples_per_gpu=48, workers_per_gpu=8)  # single GPU setting

# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='Poly',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1,
    min_lr=1e-05)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
