_base_ = [
    "../base_spglane.py",
    "dataset_culane_spglane.py",
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

cfg_name = "spglane_culane_dla34.py"

model = dict(
    backbone=dict(
        type="DLANet",
        dla="dla34",
        pretrained=True,
    ),
    test_cfg=dict(conf_threshold=0.465))

total_epochs = 33
evaluation = dict(interval=3)
checkpoint_config = dict(interval=total_epochs)
custom_hooks = [dict(type='ExpMomentumEMAHook', momentum=0.0001, priority=5)]

data = dict(samples_per_gpu=30)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=8e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=6e-4, by_epoch=False)

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
# find_unused_parameters=True
