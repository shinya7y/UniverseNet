_base_ = [
    '_cascade_rcnn_r50_fpn_1class.py',
    '_kaggle_gbr_cots_val2_mixup_affine_hsv_1440.py',
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
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')))

data = dict(samples_per_gpu=2, val=dict(samples_per_gpu=2))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=500)

fp16 = dict(loss_scale=dict(init_scale=512))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco/cascade_mask_rcnn_r2_101_fpn_20e_coco-8a7b41e1.pth'  # noqa

evaluation = dict(interval=7)
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=0, priority=48)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
