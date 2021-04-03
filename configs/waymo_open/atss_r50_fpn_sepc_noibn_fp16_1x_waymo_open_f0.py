_base_ = [
    '../universenet/models/atss_r50_fpn_sepc_noibn.py',
    '../_base_/datasets/waymo_open_2d_detection_f0.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=3))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=512.)

load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.06/atss_r50_fpn_sepc_noibn_1x_coco_20200518_epoch_12-e1725b92.pth'  # noqa
# when RuntimeError: Only one file(not dir) is allowed in the zipfile
# load_from = '../.cache/torch/hub/checkpoints/atss_r50_fpn_sepc_noibn_1x_coco_20200518_epoch_12-e1725b92.pth'  # noqa
