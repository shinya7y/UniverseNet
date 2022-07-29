_base_ = [
    '../convnext/models/atss_convnext-t_p4_w7_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# drop_path_rate could be tuned for better results
# https://github.com/facebookresearch/ConvNeXt/issues/69
model = dict(backbone=dict(drop_path_rate=0.2))

data = dict(samples_per_gpu=4)

optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=dict(init_scale=512))
