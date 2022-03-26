_base_ = [
    '../focalnet/models/mask_rcnn_focalnet_fpn.py',
    '../_base_/datasets/coco_instance_detraug.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_srf.pth'  # noqa
model = dict(
    backbone=dict(
        type='FocalNet',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        focal_windows=[9, 9, 9, 9],
        focal_levels=[2, 2, 2, 2],
        use_conv_embed=False,
        pretrained=pretrained),
    neck=dict(in_channels=[128, 256, 512, 1024]))

data = dict(samples_per_gpu=1)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

fp16 = dict(loss_scale=dict(init_scale=512))
