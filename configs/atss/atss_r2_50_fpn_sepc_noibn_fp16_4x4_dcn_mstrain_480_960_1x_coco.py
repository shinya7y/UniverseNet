_base_ = './atss_r50_fpn_sepc_noibn_1x_coco.py'
model = dict(
    pretrained='../data/checkpoints/res2net50_v1b_26w_4s-3cf99910_mmdetv2.pth',
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=4,
        base_width=26,
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(samples_per_gpu=4, train=dict(pipeline=train_pipeline))

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# fp16 settings
fp16 = dict(loss_scale=512.)
