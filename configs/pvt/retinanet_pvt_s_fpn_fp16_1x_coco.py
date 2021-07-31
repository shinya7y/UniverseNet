_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained_ckpt = 'https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth'  # noqa
model = dict(
    pretrained=pretrained_ckpt,
    backbone=dict(_delete_=True, type='pvt_small'),
    neck=dict(in_channels=[64, 128, 320, 512]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)

fp16 = dict(loss_scale='dynamic')
