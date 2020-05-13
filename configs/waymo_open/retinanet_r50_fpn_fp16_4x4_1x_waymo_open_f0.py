_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/waymo_open_2d_detection_f0.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(bbox_head=dict(num_classes=3))
# data
data = dict(samples_per_gpu=4)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'  # noqa
# fp16 settings
fp16 = dict(loss_scale=512.)
