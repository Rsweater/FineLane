_base_ = [
    "../base_spglane.py",
    "dataset_curvelanes_spglane.py",
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

cfg_name = "spglane_curvelanes_dla34.py"

model = dict(
    backbone=dict(
        type="DLANet",
        dla="dla34",
        pretrained=True,
    ),
    bbox_head=dict(
        loss_iou=dict(
            type="LaneIoULoss",
            lane_width=2.5 / 224,
            loss_weight=4.0,
        ),
        loss_seg=dict(
            loss_weight=2.0,
            num_classes=2,  # 1 lane + 1 background
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            iou_dynamick=dict(
                type="LaneIoUCost",
                lane_width=2.5 / 224,
                use_pred_start_end=False,
                use_giou=True,
            ),
            iou_cost=dict(
                type="LaneIoUCost",
                lane_width=10 / 224,
                use_pred_start_end=True,
                use_giou=True,
            ),
        )
    ),
    test_cfg=dict(
        conf_threshold=0.465,
        use_nms=True,
        as_lanes=True,
        nms_thres=15,
        nms_topk=16,
    ),
)

total_epochs = 27
evaluation = dict(interval=3)
checkpoint_config = dict(interval=total_epochs)

data = dict(samples_per_gpu=40)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=8e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=0.0, by_epoch=False)

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
