_base_ = './universenet50_fp16_4x4_lr0001_mstrain_640_1280_from_waymo_7e_nightowls.py'  # noqa

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1280, 800), (1536, 960)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(test=dict(pipeline=test_pipeline))

model = dict(
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.02,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=1000))
