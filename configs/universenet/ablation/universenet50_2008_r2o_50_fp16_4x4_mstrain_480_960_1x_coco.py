_base_ = [
    '../../universenet/models/universenet50_2008.py',
    '../../_base_/datasets/coco_detection_mstrain_480_960.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
pretrained_ckpt = 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth'  # noqa
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        deep_stem=False,
        avg_down=False,
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, False, True),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_ckpt)))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)
