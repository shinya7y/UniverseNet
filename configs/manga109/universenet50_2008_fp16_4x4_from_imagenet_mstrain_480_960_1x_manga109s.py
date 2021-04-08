_base_ = [
    '../universenet/models/universenet50_2008.py',
    '../_base_/datasets/manga109s_mstrain_480_960.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=4))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=500)

evaluation = dict(classwise=True)

fp16 = dict(loss_scale=512.)
