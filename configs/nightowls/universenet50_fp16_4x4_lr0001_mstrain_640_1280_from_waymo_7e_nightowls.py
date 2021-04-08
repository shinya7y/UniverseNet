_base_ = [
    '../universenet/models/universenet50.py',
    '../_base_/datasets/nightowls_mstrain_640_1280.py',
    '../_base_/schedules/schedule_7e.py', '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=3))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=512.)

load_from = '../data/checkpoints/universenet50_fp16_8x2_lr0001_mstrain_640_1280_7e_waymo_open_20200526_080330/epoch_7_for_nightowls.pth'  # noqa
