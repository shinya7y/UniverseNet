# CBNetV2 Benchmark and Model Zoo

## CNN-based (w/ ImageNet-1k pretrained)

### Faster R-CNN

|  Backbone   | Lr Schd | box mAP (minival) | #params | FLOPs |                     config                      |                                                         log                                                         |                                                       model                                                        |
| :---------: | :-----: | :---------------: | :-----: | :---: | :---------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| DB-ResNet50 |   1x    |       40.8        |   69M   | 284G  | [config](faster_rcnn_cbv2d1_r50_fpn_1x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/faster_rcnn_cbv2d1_r50_fpn_1x_coco.log.json) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/faster_rcnn_cbv2d1_r50_fpn_1x_coco.pth.zip) |

### Cascade R-CNN (1600x1400)

|     Backbone      |    Lr Schd     | box mAP (minival/test-dev) | #params | FLOPs |                                         config                                          |                                                                       model                                                                        |
| :---------------: | :------------: | :------------------------: | :-----: | :---: | :-------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Res2Net101-DCN |      20e       |           53.7/-           |  141M   | 429G  |       [config](cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_coco.py)       |   [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_coco.pth.zip)   |
| DB-Res2Net101-DCN | 20e + 1x (swa) |         54.8/55.3          |  141M   | 429G  | [config (test only)](cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_coco_swa.pth.zip) |

### Cascade R-CNN w/ 4conv1fc (1600x1400)

|     Backbone      |    Lr Schd     | box mAP (minival/test-dev) | #params | FLOPs |                                                config                                                |                                                                              model                                                                              |
| :---------------: | :------------: | :------------------------: | :-----: | :---: | :--------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Res2Net101-DCN |      20e       |           54.1/-           |  146M   | 774G  |       [config](cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_giou_4conv1f_coco.py)       |   [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_giou_4conv1f_coco.pth.zip)   |
| DB-Res2Net101-DCN | 20e + 1x (swa) |         55.3/55.6          |  146M   | 774G  | [config (test only)](cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_giou_4conv1f_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_giou_4conv1f_coco_swa.pth.zip) |

**Notes**:

- For SWA training, please refer to [SWA Object Detection](https://github.com/hyz-xmaster/swa_object_detection)

## Transformer-based (w/ ImageNet-1k pretrained)

### Mask R-CNN

| Backbone  | Lr Schd | box mAP (minival) | mask mAP (minival) | #params | FLOPs |                                       config                                       |                                                                          log                                                                           |                                                                         model                                                                         |
| :-------: | :-----: | :---------------: | :----------------: | :-----: | :---: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Swin-T |   3x    |       50.2        |        44.5        |   76M   | 357G  | [config](mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.log.json) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip) |

### Cascade Mask R-CNN w/ 4conv1fc

| Backbone  | Lr Schd | box mAP (minival) | mask mAP (minival) | #params | FLOPs |                                           config                                           |                                                                              log                                                                               |                                                                             model                                                                             |
| :-------: | :-----: | :---------------: | :----------------: | :-----: | :---: | :----------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Swin-T |   3x    |       53.6        |        46.2        |  114M   | 836G  | [config](cascade_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.log.json) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip) |

### Cascade Mask R-CNN w/ 4conv1fc (1600x1400)

| Backbone  | Lr Schd | box mAP (minival/test-dev) | mask mAP (minival/test-dev) | #params | FLOPs |                                            config                                            |                                                                              model                                                                              |
| :-------: | :-----: | :------------------------: | :-------------------------: | :-----: | :---: | :------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Swin-S |   3x    |         56.3/56.9          |          48.6/49.1          |  156M   | 1016G | [config](cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip) |

## Transformer-based (w/ ImageNet-22k pretrained)

### HTC (1600x1400)

| Backbone  |    Lr Schd     | box mAP (minival/test-dev) | mask mAP (minival/test-dev) | #params | FLOPs |                                           config                                           |                                                                          model                                                                           |
| :-------: | :------------: | :------------------------: | :-------------------------: | :-----: | :---: | :----------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Swin-B |      20e       |           57.9/-           |           50.2/-            |  231M   | 1004G |       [config](htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco.py)       |   [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_adamw_20e_coco.pth.zip)   |
| DB-Swin-B | 20e + 1x (swa) |         58.2/58.6          |          50.4/51.1          |  231M   | 1004G | [config (test only)](htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_adamw_20e_coco_swa.pth.zip) |

### HTC (bbox head w/ 4conv1fc) (1600x1400)

*Compared to regular HTC, our HTC uses 4conv1fc in bbox head.*

|      Backbone      | Lr Schd | box mAP (minival/test-dev) | mask mAP (minival/test-dev) | #params | FLOPs |                                                 config                                                  |                                                                               model                                                                               |
| :----------------: | :-----: | :------------------------: | :-------------------------: | :-----: | :---: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     DB-Swin-B      |   20e   |         58.4/58.7          |          50.7/51.1          |  235M   | 1348G |       [config](htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py)       | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip) |
| DB-Swin-B (BigDet) |   20e   |         59.1/59.5          |           51.1/-            |    -    |   -   |       [config](htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py)       |            [AWS](https://big-detection.s3.us-west-2.amazonaws.com/bigdet_cpts/mmdetection_cpts/htc_cbv2_swin_base_giou_4conv1f_bigdet_coco-ft_20e.pth)            |
|     DB-Swin-L      |   1x    |         59.1/59.4          |          51.0/51.6          |  453M   | 2162G | [config (test only)](htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth.zip) |
|  DB-Swin-L (TTA)   |   1x    |         59.6/60.1          |          51.8/52.3          |  453M   |   -   | [config (test only)](htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth.zip) |

- BigDet denotes [BigDetection](https://github.com/amazon-research/bigdetection) pre-training.\
  `load_from=https://big-detection.s3.us-west-2.amazonaws.com/bigdet_cpts/mmdetection_cpts/htc_cbv2_swin_base_giou_4conv1f_bigdet.pth`
- TTA denotes test time augmentation.

**Notes**:

- **Pre-trained models of Swin Transformer can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
