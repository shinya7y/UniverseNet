# UniverseNet

## Introduction

<!-- [ALGORITHM] -->

UniverseNets are state-of-the-art detectors for universal-scale object detection.
Please refer to our paper for details.
https://arxiv.org/abs/2103.14027

```
@inproceedings{USB_shinya_BMVC2022,
  title={{USB}: Universal-Scale Object Detection Benchmark},
  author={Shinya, Yosuke},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
```

## Example for fine-tuning

For fine-tuning from a COCO pre-trained model, please see [this example](universenet50_2008_fp16_4x2_mstrain_480_960_1x_smallbatch_finetuning_example.py).

## Results and Models

### Main results

|       Method       | Backbone | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                                            Download                                                                            |
| :----------------: | :------: | :-----: | :------: | :------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     ATSS+SEPC      |   R-50   |   1x    |    -     |      25.0      |  42.1  |            [model](https://github.com/shinya7y/UniverseNet/releases/download/20.06/atss_r50_fpn_sepc_noibn_1x_coco_20200518_epoch_12-e1725b92.pth)             |
|    UniverseNet     |  R2-50   |   1x    |   5.1    |      17.3      |  46.7  |     [model](https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_4x4_mstrain_480_960_1x_coco_20200520_epoch_12-838b7baa.pth)     |
|    UniverseNet     |  R2-50   |   2x    |   5.1    |      17.3      |  48.9  |     [model](https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_23-f9f426a3.pth)     |
|  UniverseNet+GFL   |  R2-50   |   1x    |   5.3    |      17.6      |  47.5  |   [model](https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet50_gfl_fp16_4x4_mstrain_480_960_1x_coco_20200708_epoch_12-68bb73b9.pth)   |
|  UniverseNet+GFL   |  R2-50   |   2x    |   5.3    |      17.6      |  49.4  |   [model](https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet50_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200729_epoch_24-c9308e66.pth)   |
|  UniverseNet+GFL   |  R2-101  |   2x    |   8.5    |      11.9      |  50.8  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth)   |
| UniverseNet 20.08d |  R2-50   |   1x    |   5.8    |      17.3      |  48.6  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet50_2008d_fp16_4x4_mstrain_480_960_1x_coco_20201013_epoch_12-8d9334a9.pth)  |
| UniverseNet 20.08d |  R2-101  |   20e   |   9.1    |      11.7      |  50.9  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_20201023_epoch_20-3e0d236a.pth) |
| UniverseNet 20.08d |  R2-101  |   2x    |   9.1    |      11.7      |  50.6  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet101_2008d_fp16_4x4_mstrain_480_960_2x_coco_20201013_epoch_24-1f70df0b.pth)  |
| UniverseNet 20.08  |  R2-50   |   1x    |   5.5    |      24.9      |  47.5  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_1x_coco_20200812_epoch_12-f522ede5.pth)   |
| UniverseNet 20.08  |  R2-50   |   2x    |   5.5    |      24.9      |  48.5  |  [model](https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth)   |

- In addition to ATSS+SEPC, UniverseNet uses Res2Net-v1b-50, DCN, and multi-scale training (480-960).
- The settings for normalization layers (including whether to use iBN of SEPC) depend on the config files.
- All models except for ATSS+SEPC were trained and evaluated using fp16 (mixed precision).
- The above UniverseNet (2x) model is a checkpoint at epoch 23. The AP of [a checkpoint at epoch 24](https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_24-726c5c93.pth) is quite similar (48.9) but slightly worse.

### Faster models

|       Method       | Backbone | Test scale  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                                              Download                                                                               |
| :----------------: | :------: | :---------: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| UniverseNet 20.08s |  R-50-C  | (1333, 800) |   2x    |   4.7    |      31.6      |  46.9  |    [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/universenet50_2008s_fp16_4x4_mstrain_480_960_2x_coco_20201106_epoch_24-3b6cad5b.pth)     |
| UniverseNet 20.08s |  R-50-C  | (512, 512)  |   2x    |   5.7    |      36.0      |  41.8  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/universenet50_2008s_fp16_4x16_mini_mstrain_320_640_2x_coco_20201110_epoch_24-d2655d05.pth)  |
| UniverseNet 20.08s |  R-50-C  | (224, 224)  |   2x    |   1.6    |       -        |  29.4  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/universenet50_2008s_fp16_4x16_micro_mstrain_128_256_2x_coco_20201111_epoch_24-2655e5d3.pth) |

- 4 GPUs x 16 `samples_per_gpu` for small test scales ((512, 512) and (224, 224)).
  You will be able to use BatchNorm with `norm_eval=False` even on 1 GPU.

### Test scale and test-dev AP

|       Method       | Backbone | Max test scale | TTA | Inf time (fps) | box AP (val) | box AP (test-dev) |
| :----------------: | :------: | :------------: | :-: | :------------: | :----------: | :---------------: |
|    UniverseNet     |  R2-50   |  (1333, 800)   |  -  |      15.8      |     48.9     |       49.2        |
|    UniverseNet     |  R2-50   |  (1600, 960)   |  -  |      14.5      |     49.2     |       49.5        |
| UniverseNet 20.08s |  R-50-C  |  (1333, 800)   |  -  |      31.6      |     46.9     |       47.4        |
| UniverseNet 20.08  |  R2-50   |  (1333, 800)   |  -  |      24.9      |     48.5     |       48.8        |
| UniverseNet 20.08d |  R2-101  |  (1333, 800)   |  -  |      11.7      |     50.9     |       51.3        |
| UniverseNet 20.08d |  R2-101  |  (2000, 1200)  |  5  |       -        |     53.1     |       53.8        |
| UniverseNet 20.08d |  R2-101  |  (3000, 1800)  | 13  |       -        |     53.5     |       54.1        |

- TTA: test-time augmentation including horizontal flip and multi-scale testing (numbers denote scales).

<!-- box AP (val)
0.469 0.652 0.511 0.297 0.508 0.617
0.485 0.670 0.526 0.306 0.527 0.627
0.509 0.695 0.554 0.335 0.555 0.658
0.531 0.707 0.586 0.374 0.574 0.680
0.535 0.708 0.589 0.369 0.575 0.681
-->

### Misc.

<details>
<summary>Other hyperparameters and details for reproduction</summary>

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

</details>
