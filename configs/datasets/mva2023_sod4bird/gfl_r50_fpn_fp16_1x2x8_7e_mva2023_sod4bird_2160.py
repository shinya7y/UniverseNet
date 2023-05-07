_base_ = [
    '../../gfl/_gfl_r50_fpn.py', '_mva2023_sod4bird_2160.py',
    '../../_base_/schedules/schedule_7e.py', '../../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=1), test_cfg=dict(score_thr=0.001))

data = dict(samples_per_gpu=2, val=dict(samples_per_gpu=2))

# 1 GPU * 2 samples_per_gpu * 8 cumulative_iters
# to simulate 8 GPUs * 2 samples_per_gpu
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(cumulative_iters=8)
lr_config = dict(warmup_iters=4000)  # 500 * cumulative_iters

fp16 = dict(loss_scale=dict(init_scale=512))
custom_hooks = []

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'  # noqa
