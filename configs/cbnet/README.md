# CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cbnetv2-a-composite-backbone-network/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=cbnetv2-a-composite-backbone-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cbnetv2-a-composite-backbone-network/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=cbnetv2-a-composite-backbone-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cbnetv2-a-composite-backbone-network/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=cbnetv2-a-composite-backbone-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cbnetv2-a-composite-backbone-network/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=cbnetv2-a-composite-backbone-network)

By [Tingting Liang](https://github.com/tingtingliangvs)\*, [Xiaojie Chu](https://github.com/chuxiaojie)\*, [Yudong Liu](https://github.com/PKUbahuangliuhe)\*, Yongtao Wang, Zhi Tang, Wei Chu, Jingdong Chen, Haibin Ling.

This repo is the official implementation of [CBNetV2](http://arxiv.org/abs/2107.00420). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

Contact us with tingtingliang@pku.edu.cn, chuxiaojie@stu.pku.edu.cn, wyt@pku.edu.cn.
## Introduction
*CBNetV2* achieves strong single-model performance on COCO object detection (`60.1 box AP` and `52.3 mask AP` on test-dev) without extra training data.

![teaser](figures/cbnetv2.png)


## Partial Results and Models
**More results and models can be found in [model zoo](model_zoo.md)**

### Faster R-CNN
|  Backbone   | Lr Schd | box mAP (minival) | #params | FLOPs |                            config                             |                                                         log                                                         |                                                       model                                                        |
| :---------: | :-----: | :---------------: | :-----: | :---: | :-----------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| DB-ResNet50 |   1x    |       40.8        |   69M   | 284G  | [config](configs/cbnet/faster_rcnn_cbv2d1_r50_fpn_1x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/faster_rcnn_cbv2d1_r50_fpn_1x_coco.log.json) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/faster_rcnn_cbv2d1_r50_fpn_1x_coco.pth.zip) |


### Mask R-CNN

| Backbone  | Lr Schd | box mAP (minival) | mask mAP (minival) | #params | FLOPs |                                              config                                              |                                                                          log                                                                           |                                                                         model                                                                         |
| :-------: | :-----: | :---------------: | :----------------: | :-----: | :---: | :----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Swin-T |   3x    |       50.2        |        44.5        |   76M   | 357G  | [config](configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.log.json) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip) |

### Cascade Mask R-CNN (1600x1400)
| Backbone  | Lr Schd | box mAP (minival/test-dev) | mask mAP (minival/test-dev) | #params | FLOPs |                                                   config                                                   |                                                                              model                                                                              |
| :-------: | :-----: | :------------------------: | :-------------------------: | :-----: | :---: | :--------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DB-Swin-S |   3x    |         56.3/56.9          |          48.6/49.1          |  156M   | 1016G | [config](configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip) |

### Improved HTC (1600x1400)
*We use ImageNet-22k pretrained checkpoints of Swin-B and Swin-L. Compared to regular HTC, our HTC uses 4conv1fc in bbox head.*
|    Backbone     | Lr Schd | box mAP (minival/test-dev) | mask mAP (minival/test-dev) | #params | FLOPs |                                                        config                                                         |                                                                               model                                                                               |
| :-------------: | :-----: | :------------------------: | :-------------------------: | :-----: | :---: | :-------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    DB-Swin-B    |   20e   |         58.4/58.7          |          50.7/51.1          |  235M   | 1348G |       [config](configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py)       | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip) |
|    DB-Swin-L    |   1x    |         59.1/59.4          |          51.0/51.6          |  453M   | 2162G | [config (test only)](configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth.zip) |
| DB-Swin-L (TTA) |   1x    |         59.6/60.1          |          51.8/52.3          |  453M   |   -   | [config (test only)](configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth.zip) |

TTA denotes test time augmentation.

**Notes**:

- **Pre-trained models of Swin Transformer can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing (w/o segm result)
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox

# multi-gpu testing (w/ segm result)
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>
```
For example, to train a Faster R-CNN model with a `Duel-ResNet50` backbone and 8 gpus, run:
```
# path of pre-training model (resnet50) is already in config
tools/dist_train.sh configs/cbnet/faster_rcnn_cbv2d1_r50_fpn_1x_coco.py 8
```

Another  example, to train a Mask R-CNN model with a `Duel-Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL>
```



### Apex (optional):
Following [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), we use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Documents and Tutorials
*We list some documents and tutorials from [MMDetection](https://github.com/open-mmlab/mmdetection), which may be helpful to you.*
* [Learn about Configs](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md)
* [Train with customized datasets](https://github.com/open-mmlab/mmdetection/blob/master/docs/2_new_data_model.md)
* [Finetuning Models](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/finetune.md)


## Citation
If you use our code/model, please consider to cite our paper [CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection](http://arxiv.org/abs/2107.00420).
```
@article{liang2021cbnetv2,
  title={CBNetV2: A Composite Backbone Network Architecture for Object Detection},
  author={Tingting Liang and Xiaojie Chu and Yudong Liu and Yongtao Wang and Zhi Tang and Wei Chu and Jingdong Chen and Haibing Ling},
  journal={arXiv preprint arXiv:2107.00420},
  year={2021}
}
```

## License
The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.


## Other Links
> **Original CBNet**: See [CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://github.com/VDIGPKU/CBNet).
