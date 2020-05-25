_base_ = [
    '../_base_/models/atss_r2_50_fpn_sepc_noibn_dcn.py',
    '../_base_/datasets/waymo_open_2d_detection_f0_mstrain_640_1280.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=2)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=512.)

load_from = '../data/checkpoints/atss_r2_50_fpn_sepc_noibn_fp16_8x2_dcn_mstrain_480_960_2x_coco_20200523_114137/epoch_24.pth'  # noqa
