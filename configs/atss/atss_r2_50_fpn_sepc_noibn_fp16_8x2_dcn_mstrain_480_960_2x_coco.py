_base_ = './atss_r2_50_fpn_sepc_noibn_fp16_4x4_dcn_mstrain_480_960_1x_coco.py'

data = dict(samples_per_gpu=2)

optimizer_config = dict(_delete_=True, grad_clip=None)
lr_config = dict(step=[16, 22])
total_epochs = 24

resume_from = '../data/checkpoints/atss_r2_50_fpn_sepc_noibn_fp16_4x4_dcn_mstrain_480_960_1x_coco_20200520_202205/epoch_8.pth'  # noqa
