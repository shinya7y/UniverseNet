# CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection

## Introduction

<!-- [ALGORITHM] -->

This directory contains the configs and results of [CBNetV2](http://arxiv.org/abs/2107.00420).

**Results and models can be found in [model zoo](model_zoo.md).**

## Usage

### Inference

```
# single-gpu testing (w/o segm result)
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox

# multi-gpu testing (w/ segm result)
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

Example 1:
To train a Faster R-CNN model with a `Dual-ResNet50` backbone and 8 gpus, run:

```
# path of pre-training model (resnet50) is already in config
tools/dist_train.sh configs/cbnet/faster_rcnn_cbv2d1_r50_fpn_1x_coco.py 8
```

Example 2:
To train a Mask R-CNN model with a `Dual-Swin-T` backbone and 8 gpus, run:

```
tools/dist_train.sh configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL>
```

### Mixed Precision Training

The current configs use mixed precision training via MMCV by default.
Please install PyTorch >= 1.6.0 to use torch.cuda.amp.

If you find performance difference from apex (used by the original authors), please raise an issue.
Otherwise, we will clean code for apex.

## Citation

```latex
@article{liang2021cbnetv2,
  title={CBNetV2: A Composite Backbone Network Architecture for Object Detection},
  author={Tingting Liang and Xiaojie Chu and Yudong Liu and Yongtao Wang and Zhi Tang and Wei Chu and Jingdong Chen and Haibing Ling},
  journal={arXiv preprint arXiv:2107.00420},
  year={2021}
}
```
