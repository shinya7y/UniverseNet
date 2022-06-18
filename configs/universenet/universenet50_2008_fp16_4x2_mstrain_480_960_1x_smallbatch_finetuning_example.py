# This config shows an example for small-batch fine-tuning from a COCO model.
# Please see also the MMDetection tutorial below.
# https://github.com/shinya7y/UniverseNet/blob/master/docs/en/tutorials/finetune.md

_base_ = [
    '../universenet/models/universenet50_2008.py',
    # Please change to your dataset config.
    '../_base_/datasets/coco_detection_mstrain_480_960.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    # SyncBN is used in universenet50_2008.py
    # If total batch size < 16, please change BN settings of backbone.
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True),
    # iBN of SEPC is used in universenet50_2008.py
    # If samples_per_gpu < 4, please change BN settings of SEPC.
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
            ibn=True,
            pnorm_eval=True,  # please set True if samples_per_gpu < 4
            lcnorm_eval=True,  # please set True if samples_per_gpu < 4
            lcconv_padding=1)
    ],
    bbox_head=dict(num_classes=80))  # please change for your dataset

# Optimal total batch size depends on dataset size and learning rate.
# If image sizes are not so large and you have enough GPU memory,
# larger samples_per_gpu will be preferable.
data = dict(samples_per_gpu=2)

# This config assumes that total batch size is 8 (4 GPUs * 2 samples_per_gpu).
# Since the batch size is half of other configs,
# the learning rate is also halved according to the Linear Scaling Rule.
# Tuning learning rate around it will be important on other datasets.
# For example, you can try 0.005 first, then 0.002, 0.01, 0.001, and 0.02.
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

# If fine-tuning from COCO, gradients should not be so large.
# It is natural to train models without gradient clipping.
optimizer_config = dict(_delete_=True, grad_clip=None)

# If fine-tuning from COCO, a warmup_iters of 500 or less may be enough.
# This setting is not so important unless losses are unstable during warmup.
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)

# Please set `load_from` to use a COCO pre-trained model.
load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
