_base_ = [
    '../dyhead/_atss_r50_fpn_dyhead.py', '../_base_/datasets/manga109s.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=4))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=500)

evaluation = dict(classwise=True)

fp16 = dict(loss_scale=dict(init_scale=512))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_4x4_1x_coco/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth'  # noqa
