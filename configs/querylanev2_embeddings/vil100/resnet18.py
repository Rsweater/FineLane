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

cfg_name = "querylanev2_vil100_r18.py"

# model=dict(
  
#  )

total_epochs = 400
evaluation = dict(start=10, interval=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
custom_hooks = [dict(type="ExpMomentumEMAHook", momentum=0.0001, priority=20)]


data = dict(samples_per_gpu=48, workers_per_gpu=8)  # single GPU setting

# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='Poly',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    min_lr=1e-05)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
