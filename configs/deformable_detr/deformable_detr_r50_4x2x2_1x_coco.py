_base_ = [
    '_deformable_detr_r50.py', '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# use default setting (size_divisor=32) for simplicity
data = dict(train=dict(filter_empty_gt=False))

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(
    grad_clip=dict(max_norm=0.1, norm_type=2), cumulative_iters=2)

lr_config = dict(policy='step', step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
