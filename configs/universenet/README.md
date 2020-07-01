# UniverseNet


## Introduction

UniverseNet is a single-stage detector for universal scale detection. Unlike EfficientDet and YOLOv4, it is trained using universal training settings (e.g., 2x schedule, initial lr: 0.01).

UniverseNet is the SOTA single-stage detector on the Waymo Open Dataset 2D detection, and achieves the 1st place in the NightOwls Detection Challenge 2020 all objects track.


## Results and Models

### Main results

|   Method    | Backbone | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                                        Download                                                                        |
| :---------: | :------: | :-----: | :------: | :------------: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
|  ATSS+SEPC  |   R-50   |   1x    |    -     |       -        |  42.1  |                                                                           -                                                                            |
| UniverseNet |  R2-50   |   1x    |   5.1    |      15.8      |  46.7  |                                                                           -                                                                            |
| UniverseNet |  R2-50   |   2x    |   5.1    |      15.8      |  48.9  | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_23-f9f426a3.pth) |

- In addition to ATSS+SEPC, UniverseNet uses Res2Net-v1b-50, DCN, and multi-scale training (480-960).
- iBN of SEPC is set to False to allow for batch sizes less than 4.
- All models except for ATSS+SEPC were trained and evaluated using fp16 (mixed precision).
- The above UniverseNet (2x) model is a checkpoint at epoch 23. The AP of [a checkpoint at epoch 24](https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_24-726c5c93.pth) is quite similar (48.9) but slightly worse.


### Test scale and test-dev AP

UniverseNet can achieve the EfficientDet-D4 level AP (val AP: 49.0, test-dev AP: 49.4) with 24 epochs training.

| Test scale  | Inf time (fps) | box AP (val) | box AP (test-dev) |
| :---------: | :------------: | :----------: | :---------------: |
| (1333, 800) |      15.8      |     48.9     |       49.2        |
| (1600, 960) |      14.5      |     49.2     |       49.5        |

<!-- (1333, 800)
0.489 0.675 0.535 0.323 0.534 0.633
0.492 0.679 0.535 0.306 0.528 0.621
-->
<!-- (1600, 960)
0.492 0.677 0.538 0.342 0.535 0.624
0.495 0.683 0.540 0.320 0.530 0.603
-->


### Other hyperparameters and details for reproduction

| warmup_iters | lcconv_padding | GPUs x samples_per_gpu | box AP |
| :----------: | :------------: | :--------------------: | :----: |
|     500      |       0        |       4x4 -> 8x2       |  48.9  |
|     1000     |       1        |          4x4           |  48.9  |
|     3665     |       0        |          4x4           |  48.8  |

- The above checkpoints were trained with a warmup_iters of 500.
  To make training more stable, the current config sets warmup_iters to 1000. The difference will not affect the final accuracy so much.
  Your training is going well if the AP of the first epoch model is around 20-22.
- In the official SEPC implementation, [padding=0](https://github.com/jshilong/SEPC/issues/13) in lconv and cconv (lcconv_padding).
  Setting lcconv_padding to 1 doesn't affect accuracy.
- To accelerate training for CVPR competitions, we used 8 GPUs for 9-24 epochs, after using 4 GPUs for 1-8 epochs.
