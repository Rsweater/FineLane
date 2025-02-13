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

cfg_name = "querylanev2_culane_r18.py"

model = dict(
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
    lane_head=dict(
        # type="BezierLaneHeadV2",
        loss_cls=dict(
            loss_weight=2.0,
        ),
        loss_dist=dict(
            loss_weight=3.0,
        ),
        loss_seg=dict(
            loss_weight=0.50,
        ),
    ),
     # training and testing settings
    test_cfg=dict(
        # dataset info
        # dataset="tusimple",
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
        # inference settings
        conf_threshold=0.95,
        window_size=9,
        max_num_lanes=4,
        num_sample_points=50,
    ),
)



total_epochs = 36
evaluation = dict(start=3, interval=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
custom_hooks = [dict(type="ExpMomentumEMAHook", momentum=0.0001, priority=5)]

data = dict(samples_per_gpu=24, workers_per_gpu=4)  # single GPU setting

# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-3,
    paramwise_cfg=dict(
        custom_keys={
            'conv_offset': dict(lr_mult=0.1),
        }),
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    by_epoch=True
)

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
