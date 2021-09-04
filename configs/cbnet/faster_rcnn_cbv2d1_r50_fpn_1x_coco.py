_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='CBResNet',
        cb_del_stages=1,
        cb_inplanes=[64, 256, 512, 1024, 2048]),
    neck=dict(type='CBFPN'))
