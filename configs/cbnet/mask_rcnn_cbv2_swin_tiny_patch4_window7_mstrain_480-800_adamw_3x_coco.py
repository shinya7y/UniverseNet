_base_ = '../swin_original/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'  # noqa

model = dict(backbone=dict(type='CBSwinTransformer'), neck=dict(type='CBFPN'))
