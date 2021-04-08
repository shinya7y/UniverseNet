_base_ = [
    '../../universenet/models/universenet50_2008.py',
    '../../_base_/datasets/coco_detection_mstrain_480_960.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='SEPC',
            out_channels=256,
            stacked_convs=4,
            pconv_deform=False,
            lcconv_deform=True,
            ibn=False,  # please set imgs/gpu >= 4
            lcconv_padding=1)
    ])

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)
