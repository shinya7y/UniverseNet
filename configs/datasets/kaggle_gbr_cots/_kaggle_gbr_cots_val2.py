# dataset settings
dataset_type = 'GBRCOTSDataset'
img_root = '/kaggle/input/tensorflow-great-barrier-reef/train_images/'
ann_root = '/kaggle/data/tensorflow-great-barrier-reef/annotations/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 720), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
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
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_root + 'instances_full_not_video_2.json',
        img_prefix=img_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'instances_full_video_2.json',
        img_prefix=img_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'instances_full_video_2.json',
        img_prefix=img_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
