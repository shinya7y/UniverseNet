_base_ = [
    '../universenet/models/universenet50.py',
    '../_base_/datasets/manga109s_mstrain_480_960.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=4))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=500)

evaluation = dict(classwise=True)

fp16 = dict(loss_scale=512.)

load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_4x4_mstrain_480_960_1x_coco_20200520_epoch_12-838b7baa.pth'  # noqa
# when RuntimeError: Only one file(not dir) is allowed in the zipfile
# load_from = '../.cache/torch/hub/checkpoints/universenet50_fp16_4x4_mstrain_480_960_1x_coco_20200520_epoch_12-838b7baa.pth'  # noqa
