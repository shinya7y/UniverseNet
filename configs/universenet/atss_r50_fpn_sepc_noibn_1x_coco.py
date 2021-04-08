_base_ = [
    '../universenet/models/atss_r50_fpn_sepc_noibn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
