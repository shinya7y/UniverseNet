_base_ = 'htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py'  # noqa

model = dict(
    backbone=dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False),
    neck=dict(in_channels=[192, 384, 768, 1536]))

lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
