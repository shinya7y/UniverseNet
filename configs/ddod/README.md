# Disentangle Your Dense Object Detector

## Introduction

<!-- [ALGORITHM] -->

This directory contains the configs and results of [Disentangle Your Dense Object Detector](https://arxiv.org/abs/2107.02963).

## Results and Models

|   Method   | Backbone | Lr schd | box AP | box AP by authors |                     Config                     |                                                              Download                                                              |
| :--------: | :------: | :-----: | :----: | :---------------: | :--------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
| ATSS (IoU) |   R-50   |   1x    |   -    |       39.4        | [config](atss_iou_r50_fpn_fp16_4x4_1x_coco.py) |                                                                 -                                                                  |
| ATSS+DDOD  |   R-50   |   1x    |  41.3  |       41.6        |   [config](ddod_r50_fpn_fp16_4x4_1x_coco.py)   | [model](https://github.com/shinya7y/weights/releases/download/v1.0.0/ddod_r50_fpn_fp16_4x4_1x_coco_20210715_epoch_12-afcf6d59.pth) |

- DDOD adopts a variant of ATSS as a baseline, which predicts IoU instead of centerness.
- We use a total batch size of 16 (in 4 GPUs) with `lr=0.01`,
  while the authors use a total batch size of 32 (in 8 GPUs) with `lr=0.02`.
- Other possible reasons for AP difference are randomness, instability due to fp16, and implementation change of mixed-precision training (MMCV >= 1.3.2).

## Citation

```latex
@article{chen2021disentangle,
  title={Disentangle Your Dense Object Detector},
  author={Zehui Chen and Chenhongyi Yang and Qiaofei Li and Feng Zhao and Zhengjun Zha and Feng Wu},
  journal={arXiv preprint arXiv:2107.02963},
  year={2021}
}
```
