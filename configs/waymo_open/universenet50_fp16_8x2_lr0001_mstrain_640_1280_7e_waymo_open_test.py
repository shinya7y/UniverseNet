_base_ = [
    '../_base_/models/universenet50.py',
    '../_base_/datasets/waymo_open_2d_detection_mstrain_640_1280.py',
    '../_base_/schedules/schedule_7e.py', '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=3))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_img_scale_short = [960 - 8, 1600 - 8, 2240 - 8]
test_img_scale = [(int(short * 1.5), short) for short in test_img_scale_short]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=test_img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='soft_nms', iou_thr=0.6, min_score=0.01),
    max_per_img=1000)

fp16 = dict(loss_scale=512.)

load_from = '../data/checkpoints/universenet50_fp16_8x2_lr0001_mstrain_640_1280_7e_waymo_open_20200526_080330/epoch_7.pth'  # noqa
