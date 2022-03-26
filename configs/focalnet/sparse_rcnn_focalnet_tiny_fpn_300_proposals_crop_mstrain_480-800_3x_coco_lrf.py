_base_ = [
    '../focalnet/models/sparse_rcnn_focalnet_fpn.py',
    '../_base_/datasets/coco_detection_detraug.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_lrf.pth'  # noqa
num_proposals = 300
model = dict(
    backbone=dict(
        drop_path_rate=0.3,
        focal_levels=[3, 3, 3, 3],
        focal_windows=[11, 9, 9, 7],
        pretrained=pretrained),
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(
        _delete_=True, rpn=None, rcnn=dict(max_per_img=num_proposals)))

optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
