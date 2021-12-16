_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='poolformer_s24_feat',
        out_indices=(2, 4, 6),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[128, 320, 512],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
