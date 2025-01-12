_base_ = [
    "../base_spglane.py",
    "dataset_tusimple_spglane.py",
    "../../_base_/default_runtime.py",
]

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

cfg_name = "spglane_tusimple_r18.py"

model = dict(
    backbone=dict(
        type='ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    bbox_head=dict(
        loss_cls=dict(type="KorniaFocalLoss", alpha=0.25, gamma=2, loss_weight=6.0),
        loss_bbox=dict(type="SmoothL1Loss", reduction="none", loss_weight=0.5),
        loss_iou=dict(
            type="LaneIoULoss",
            lane_width=7.5 / 800,
            loss_weight=4.0,
        ),
        loss_seg=dict(
            type="CLRNetSegLoss",
            loss_weight=1.0,
            num_classes=7,  # 6 lanes + 1 background
            ignore_label=255,
            bg_weight=0.4,),
    ),
    test_cfg=dict(conf_threshold=0.40, nms_thres=50, nms_topk=6)
)

total_epochs = 150
evaluation = dict(interval=3)
checkpoint_config = dict(interval=2)

data = dict(samples_per_gpu=48)  # single GPU setting

# optimizer
optimizer = dict(type='AdamW', lr=8e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.0, by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
# find_unused_parameters=True
