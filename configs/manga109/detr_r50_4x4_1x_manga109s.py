_base_ = [
    '../detr/_detr_r50.py', '../_base_/datasets/manga109s.py',
    '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=4))

# use default setting (size_divisor=32) for simplicity
data = dict(samples_per_gpu=4)

optimizer = dict(
    type='AdamW',
    lr=8e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

lr_config = dict(policy='step', step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

evaluation = dict(classwise=True)

load_from = 'https://github.com/shinya7y/weights/releases/download/v1.0.0/detr_r50_4x4_1x_coco_20220705_epoch_12-76992e99.pth'  # noqa
