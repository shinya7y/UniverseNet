_base_ = '../swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'  # noqa

model = dict(backbone=dict(type='CBSwinTransformer'), neck=dict(type='CBFPN'))
