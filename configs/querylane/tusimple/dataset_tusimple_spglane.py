dataset_type = "TuSimpleDataset"
data_root = "dataset/tusimple"
crop_bbox = [0, 160, 1280, 720]
img_scale = (800, 320)
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)
compose_cfg = dict(bboxes=False, keypoints=True, masks=True)

# data pipeline settings
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(type="HorizontalFlip", p=0.5),
    dict(type="ChannelShuffle", p=0.1),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.04,
        contrast_limit=0.15,
        p=0.6,
    ),
    dict(
        type="HueSaturationValue",
        hue_shift_limit=(-10, 10),
        sat_shift_limit=(-10, 10),
        val_shift_limit=(-10, 10),
        p=0.7,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="MotionBlur", blur_limit=5, p=1.0),
            dict(type="MedianBlur", blur_limit=5, p=1.0),
        ],
        p=0.2,
    ),
    dict(
        type="IAAAffine",
        scale=(0.8, 1.2),
        rotate=(-10.0, 10.0),  # this sometimes breaks lane sorting
        translate_percent=0.1,
        p=0.7,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="albumentation", pipelines=train_al_pipeline),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectCLRNet",
        keys=["img"], max_lanes=6,
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "ori_shape",
            "eval_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "lanes",
        ],
    ),
]

val_pipeline = [
    dict(type="albumentation", pipelines=val_al_pipeline),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectCLRNet",
        keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
        ],
    ),
]

data = dict(
    samples_per_gpu=32,  # medium
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=[
            data_root + '/label_data_0313.json',
            data_root + '/label_data_0531.json',
            data_root + '/label_data_0601.json'
        ],
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=[data_root + '/test_label.json'], # test_baseline,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=[data_root + '/test_label.json'],
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
