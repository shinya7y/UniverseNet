_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py', '_mva2023_sod4bird_2160.py',
    '../../_base_/schedules/schedule_7e.py', '../../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=1)),
    test_cfg=dict(rcnn=dict(score_thr=0.001)))

data = dict(samples_per_gpu=2, val=dict(samples_per_gpu=2))

# 1 GPU * 2 samples_per_gpu * 8 cumulative_iters
# to simulate 8 GPUs * 2 samples_per_gpu
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(cumulative_iters=8)
lr_config = dict(warmup_iters=4000)  # 500 * cumulative_iters

fp16 = dict(loss_scale=dict(init_scale=512))
custom_hooks = []

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
