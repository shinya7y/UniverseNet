# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/Manga109s/'
classes = ('body', 'face', 'frame', 'text')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1344, 480), (1344, 960)],
        multiscale_mode='range',
        keep_ratio=True),
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
        img_scale=(1216, 864),
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
        classes=classes,
        ann_file=data_root +
        'annotations_coco_format/manga109s_coco_68train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root +
        'annotations_coco_format/manga109s_coco_4val.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        force_filter_imgs=True),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root +
        'annotations_coco_format/manga109s_coco_15test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        force_filter_imgs=True))
evaluation = dict(interval=1, metric='bbox')
