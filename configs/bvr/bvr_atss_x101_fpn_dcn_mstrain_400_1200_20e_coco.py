_base_ = [
    '../_base_/datasets/coco_detection_mstrain_400_1200.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

model = dict(
    type='BVR',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='BVRHead',
        bbox_head_cfg=dict(
            type='ATSSHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            dcn_on_last_conv=True),
        keypoint_pos='input',
        keypoint_head_cfg=dict(
            type='KeypointHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            strides=[8, 16, 32, 64, 128],
            shared_stacked_convs=0,
            logits_convs=1,
            head_types=['top_left_corner', 'bottom_right_corner', 'center'],
            corner_pooling=False,
            loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_cls=dict(type='GaussianFocalLoss', loss_weight=0.25)),
        cls_keypoint_cfg=dict(
            keypoint_types=['center'],
            with_key_score=False,
            with_relation=True),
        reg_keypoint_cfg=dict(
            keypoint_types=['top_left_corner', 'bottom_right_corner'],
            with_key_score=False,
            with_relation=True),
        keypoint_cfg=dict(max_keypoint_num=20, keypoint_score_thr=0.0),
        feature_selection_cfg=dict(
            selection_method='index',
            cross_level_topk=50,
            cross_level_selection=True),
        num_attn_heads=8,
        scale_position=False,
        pos_cfg=dict(base_size=[300, 300], log_scale=True, num_layers=2),
        shared_positional_encoding_outer=True),
    train_cfg=dict(
        bbox=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        keypoint=dict(
            assigner=dict(type='PointKptAssigner'),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
