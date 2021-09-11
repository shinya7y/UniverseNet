_base_ = [
    '../swin_original/models/atss_swint_fpn.py',
    '../_base_/datasets/manga109s.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=4))

data = dict(samples_per_gpu=4)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=500)

evaluation = dict(classwise=True)

fp16 = dict(loss_scale='dynamic')

load_from = 'https://github.com/shinya7y/weights/releases/download/v1.0.0/atss_swint_fpn_fp16_4x4_adamw_1x_coco_20210502_epoch_12-3c37c44b.pth'  # noqa
