_base_ = [
    '../universenet/models/universenet50_2008.py',
    '../_base_/datasets/manga109s_mstrain_480_960.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=4))

data = dict(samples_per_gpu=4)

# 1 GPU * 4 samples_per_gpu * 4 cumulative_iters
# to simulate 4 GPUs * 4 samples_per_gpu
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(cumulative_iters=4)
lr_config = dict(warmup_iters=2000)  # 500 * cumulative_iters

evaluation = dict(classwise=True)

fp16 = dict(loss_scale=dict(init_scale=512.))

load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_1x_coco_20200812_epoch_12-f522ede5.pth'  # noqa
