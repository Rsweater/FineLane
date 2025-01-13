_base_ = [
    "../base_spglane.py",
    "dataset_vil100_spglane.py",
    "../../_base_/default_runtime.py",
]

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core",
    ],
    allow_failed_imports=False,
)

cfg_name = "spglane_vil100_dla34.py"

model = dict(
    backbone=dict(
        type="DLANet",
        dla="dla34",
        pretrained=True,
    ),
    lane_head=dict(
        loss_iou=dict(loss_weight=4.0),
        loss_seg=dict(
            loss_weight=2.0,
            num_classes=9,  # 8 lane + 1 background
        ),
    ),
    test_cfg=dict(
        conf_threshold=0.43,
        use_nms=True,
        as_lanes=True,
        nms_thres=50,
        nms_topk=8,
    ),
)

custom_hooks = [dict(type="ExpMomentumEMAHook", momentum=0.0001, priority=49)]

total_epochs = 150
evaluation = dict(start=20, interval=3)
checkpoint_config = dict(interval=3, max_keep_ckpts=10)

data = dict(samples_per_gpu=36)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=7e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=7e-4, by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
# find_unused_parameters=True
