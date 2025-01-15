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

cfg_name = "querylanev2_vil100_dla34.py"

model = dict(
    backbone=dict(
        type="DLANet",
        dla="dla34",
        pretrained=True,
    ),
    lane_head=dict(
        loss_seg=dict(
            # loss_weight=2.0,
            num_classes=9,  # 8 lane + 1 background
        ),
    ),
    test_cfg=dict(
        # dataset info
        ori_img_w=1280,
        ori_img_h=720,
        cut_height=160,
        # inference settings
        conf_threshold=0.4,
        window_size=5,
        max_num_lanes=6,
        num_sample_points=50,
    ),
)

total_epochs = 400
evaluation = dict(start=10, interval=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
custom_hooks = [dict(type="ExpMomentumEMAHook", momentum=0.0001, priority=20)]


data = dict(samples_per_gpu=48, workers_per_gpu=4)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=7e-4)
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
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
