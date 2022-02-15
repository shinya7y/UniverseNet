_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '_kaggle_gbr_cots_val2_mixup_affine_480_960.py',
    '../../_base_/schedules/schedule_7e.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')),
    roi_head=dict(bbox_head=dict(num_classes=1)))

data = dict(samples_per_gpu=4, val=dict(samples_per_gpu=4))

optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=500)

fp16 = dict(loss_scale=dict(init_scale=512))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth'  # noqa

evaluation = dict(interval=7)
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=0, priority=48)
]
