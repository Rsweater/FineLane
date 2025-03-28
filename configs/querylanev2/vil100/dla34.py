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

cfg_name = "querylanev2_vil100_r50.py"
ckpt_timm = "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth"
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=ckpt_timm)),
    neck=dict(
        type='SA_FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3),
    # lane_head=dict(
    #     loss_seg=dict(
    #         # loss_weight=2.0,
    #         num_classes=9,  # 8 lane + 1 background
    #     ),
    # ),
    # test_cfg=dict(
    #     # dataset info
    #     ori_img_w=1280,
    #     ori_img_h=720,
    #     cut_height=160,
    #     # inference settings
    #     conf_threshold=0.4,
    #     window_size=5,
    #     max_num_lanes=6,
    #     num_sample_points=50,
    # ),
)

total_epochs = 250
evaluation = dict(start=1, interval=1)
checkpoint_config = dict(interval=1)
custom_hooks = [dict(type="ExpMomentumEMAHook", momentum=0.0001, priority=20)]


data = dict(samples_per_gpu=24, workers_per_gpu=4)  # single GPU setting

# optimizer
optimizer = dict(type='Adam', lr=0.0006, betas=(0.9, 0.999), eps=1e-08)
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
