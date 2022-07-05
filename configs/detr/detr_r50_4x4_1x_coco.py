_base_ = [
    '_detr_r50.py', '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# use default setting (size_divisor=32) for simplicity
data = dict(samples_per_gpu=4)

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

lr_config = dict(policy='step', step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
