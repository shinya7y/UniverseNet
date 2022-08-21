_base_ = [
    '../universenet/models/universenet50_2008.py',
    '../_base_/datasets/waymo_open_2d_detection_f0.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=3))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=dict(init_scale=512))

load_from = 'https://github.com/shinya7y/weights/releases/download/v1.0.0/universenet50_2008_fp16_4x4_1x_coco_20201009_epoch_12-cbd3958a.pth'  # noqa
