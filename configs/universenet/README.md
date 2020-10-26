# UniverseNet


## Introduction

UniverseNet is a single-stage detector for universal scale detection. Unlike EfficientDet and YOLOv4, it is trained using universal training settings (e.g., 2x schedule, initial lr: 0.01).

UniverseNet is the SOTA single-stage detector on the Waymo Open Dataset 2D detection, and achieves the 1st place in the NightOwls Detection Challenge 2020 all objects track.


## Example for fine-tuning

For fine-tuning from a COCO pre-trained model, please see [this example](universenet50_2008_fp16_4x2_mstrain_480_960_1x_smallbatch_finetuning_example.py).


## Results and Models

### Main results

|       Method       | Backbone | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                                            Download                                                                            |
| :----------------: | :------: | :-----: | :------: | :------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     ATSS+SEPC      |   R-50   |   1x    |    -     |       -        |  42.1  |                                                                               -                                                                                |
|    UniverseNet     |  R2-50   |   1x    |   5.1    |      16.1      |  46.7  |                                                                               -                                                                                |
|    UniverseNet     |  R2-50   |   2x    |   5.1    |      16.1      |  48.9  |     [model](https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_23-f9f426a3.pth)     |
|  UniverseNet+GFL   |  R2-50   |   1x    |   5.3    |      16.9      |  47.5  |   [model](https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet50_gfl_fp16_4x4_mstrain_480_960_1x_coco_20200708_epoch_12-68bb73b9.pth)   |
|  UniverseNet+GFL   |  R2-50   |   2x    |   5.3    |      16.9      |  49.4  |   [model](https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet50_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200729_epoch_24-c9308e66.pth)   |
|  UniverseNet+GFL   |  R2-101  |   2x    |   8.5    |      11.7      |  50.8  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth)   |
| UniverseNet 20.08d |  R2-50   |   1x    |   5.8    |       -        |  48.6  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet50_2008d_fp16_4x4_mstrain_480_960_1x_coco_20201013_epoch_12-8d9334a9.pth)  |
| UniverseNet 20.08d |  R2-101  |   20e   |   9.1    |       -        |  50.9  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_20201023_epoch_20-3e0d236a.pth) |
| UniverseNet 20.08d |  R2-101  |   2x    |   9.1    |       -        |  50.6  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet101_2008d_fp16_4x4_mstrain_480_960_2x_coco_20201013_epoch_24-1f70df0b.pth)  |
| UniverseNet 20.08  |  R2-50   |   1x    |   5.5    |      23.6      |  47.5  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_1x_coco_20200812_epoch_12-f522ede5.pth)   |
| UniverseNet 20.08  |  R2-50   |   2x    |   5.5    |      23.6      |  48.5  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth)   |

- In addition to ATSS+SEPC, UniverseNet uses Res2Net-v1b-50, DCN, and multi-scale training (480-960).
- The settings for normalization layers (including whether to use iBN of SEPC) depend on the config files.
- All models except for ATSS+SEPC were trained and evaluated using fp16 (mixed precision).
- The above UniverseNet (2x) model is a checkpoint at epoch 23. The AP of [a checkpoint at epoch 24](https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_24-726c5c93.pth) is quite similar (48.9) but slightly worse.
- Inference time (fps) is measured using the DCN ops in mmdet/ops (not mmcv/ops).


### Faster models

|       Method       | Test scale | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                                           Download                                                                            |
| :----------------: | :--------: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| UniverseNet 20.08f | (512, 512) |   1x    |   7.0    |       -        |  42.0  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet50_2008f_fp16_4x16_mstrain_320_640_1x_coco_20201023_epoch_12-869a669c.pth) |
| UniverseNet 20.08f | (512, 512) |   2x    |   7.0    |       -        |  43.3  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet50_2008f_fp16_4x16_mstrain_320_640_2x_coco_20201023_epoch_24-bf6a623b.pth) |

- 4 GPUs x 16 `samples_per_gpu`. You will be able to use BatchNorm with `norm_eval=False` even on 1 GPU.
- Good to detect large objects (APL > 60%).


### Test scale and test-dev AP

UniverseNet can achieve the EfficientDet-D4 level AP (val AP: 49.0, test-dev AP: 49.4) with 24 epochs training.

|   Method    | Test scale  | Inf time (fps) | box AP (val) | box AP (test-dev) |
| :---------: | :---------: | :------------: | :----------: | :---------------: |
| UniverseNet | (1333, 800) |      15.8      |     48.9     |       49.2        |
| UniverseNet | (1600, 960) |      14.5      |     49.2     |       49.5        |

<!-- (1333, 800)
0.489 0.675 0.535 0.323 0.534 0.633
0.492 0.679 0.535 0.306 0.528 0.621
-->
<!-- (1600, 960)
0.492 0.677 0.538 0.342 0.535 0.624
0.495 0.683 0.540 0.320 0.530 0.603
-->


### Other hyperparameters and details for reproduction

|   Method    | warmup_iters | lcconv_padding | GPUs x samples_per_gpu | box AP |
| :---------: | :----------: | :------------: | :--------------------: | :----: |
| UniverseNet |     500      |       0        |       4x4 -> 8x2       |  48.9  |
| UniverseNet |     1000     |       1        |          4x4           |  48.9  |
| UniverseNet |     3665     |       0        |          4x4           |  48.8  |

- The checkpoints in [release 20.06](https://github.com/shinya7y/UniverseNet/releases/tag/20.06) were trained with a `warmup_iters` of 500.
  To make training more stable, the current config sets `warmup_iters` to 1000. The difference will not affect the final accuracy so much.
- In the official SEPC implementation, padding values in lconv and cconv (we call `lcconv_padding`) are [set to 0](https://github.com/jshilong/SEPC/issues/13).
  Setting `lcconv_padding` to 1 doesn't affect accuracy.
- To accelerate training for CVPR competitions, we used 8 GPUs for 9-24 epochs, after using 4 GPUs for 1-8 epochs.
