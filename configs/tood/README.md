# TOOD: Task-aligned One-stage Object Detection (**ICCV 2021 Oral**)
[Paper](https://arxiv.org/abs/2108.07755)

## Introduction

One-stage object detection is commonly implemented by optimizing two sub-tasks: object classification and localization, using heads with two parallel branches, which might lead to a certain level of spatial misalignment in predictions between the two tasks. In this work, we propose a Task-aligned One-stage Object Detection (TOOD) that explicitly aligns the two tasks in a learning-based manner. First, we design a novel Task-aligned Head (T-Head) which offers a better balance between learning task-interactive and task-specific features, as well as a greater flexibility to learn the alignment via a task-aligned predictor. Second, we propose Task Alignment Learning (TAL) to explicitly pull closer (or even unify) the optimal anchors for the two tasks during training via a designed sample assignment scheme and a task-aligned loss. Extensive experiments are conducted on MS-COCO, where TOOD achieves a **51.1 AP** at single-model single-scale testing. This surpasses the recent one-stage detectors by a large margin, such as ATSS (47.7 AP), GFL (48.2 AP), and PAA (49.0 AP), with fewer parameters and FLOPs. Qualitative results also demonstrate the effectiveness of TOOD for better aligning the tasks of object classification and localization.
### Method overview

<div align=center><img src="resources/overview.png" width="500px"/></div>

### Parallel head vs. T-head
![method overview](resources/T-head.png)

## Prerequisites

- MMDetection version 2.14.0.

- Please see [get_started.md](docs/get_started.md) for installation and the basic usage of MMDetection.

## Train

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'.

./tools/dist_train.sh configs/tood/tood_r50_fpn_1x_coco.py 4
```

## Inference

```python
./tools/dist_test.sh configs/tood/tood_r50_fpn_1x_coco.py work_dirs/tood_r50_fpn_1x_coco/epoch_12.pth 4 --eval bbox
```

## Models

For your convenience, we provide the following trained models (TOOD). All models are trained with 16 images in a mini-batch.

| Model                         |    Anchor    | MS train |  DCN  | Lr schd | AP (minival) | AP (test-dev) |                                  Config                                   |                                                                        Download                                                                         |
| ----------------------------- | :----------: | :------: | :---: | :-----: | :----------: | :-----------: | :-----------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: |
| TOOD_R_50_FPN_1x              | Anchor-free  |    No    |   N   |   1x    |     42.5     |     42.7      |              [config](configs/tood/tood_r50_fpn_1x_coco.py)               | [google](https://drive.google.com/file/d/1M7ccIsfQKA5pEtgMlRSadokLu_cFKO4B/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1rjAwcX2rq5xTm7_9AdWR2Q) |
| TOOD_R_50_FPN_anchor_based_1x | Anchor-based |    No    |   N   |   1x    |     42.4     |     42.8      |        [config](configs/tood/tood_r50_fpn_anchor_based_1x_coco.py)        | [google](https://drive.google.com/file/d/1G3Waqs3Xh7h1bfwcUfek91S1JKRCTAdV/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1E_Lsxj4GXhe7iPL6feVa5Q) |
| TOOD_R_101_FPN_2x             | Anchor-free  |   Yes    |   N   |   2x    |     46.2     |     46.7      |          [config](configs/tood/tood_r101_fpn_mstrain_2x_coco.py)          | [google](https://drive.google.com/file/d/14NTtLVpG0I75jb55hB6smnibknkQ4wdb/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1Py-73Xysv5w5Gvqysc_RxA) |
| TOOD_X_101_FPN_2x             | Anchor-free  |   Yes    |   N   |   2x    |     47.6     |     48.5      |       [config](configs/tood/tood_x101_64x4d_fpn_mstrain_2x_coco.py)       | [google](https://drive.google.com/file/d/1IbCZ5Lim_vkgRctsJ7Sb8czrOFQpmuRF/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1Y-CNmcHZtrWUFCrocSjiEA) |
| TOOD_R_101_dcnv2_FPN_2x       | Anchor-free  |   Yes    |   Y   |   2x    |     49.2     |     49.6      |    [config](configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py)    | [google](https://drive.google.com/file/d/1ufipVoODv-NgthQ8ZvLeW12TEIsCgWl5/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1BfgMtKprAzoTBm91XEQk4Q) |
| TOOD_X_101_dcnv2_FPN_2x       | Anchor-free  |   Yes    |   Y   |   2x    |     50.5     |     51.1      | [config](configs/tood/tood_x101_64x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py) | [google](https://drive.google.com/file/d/1xYSuZF5RfK81rJImNlTZWbIhPWfb5S5-/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1g2qiGJVV_dZmVF5D20SsNg) |

[0] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[1] *`dcnv2` denotes deformable convolutional networks v2.* \
[2] *Refer to more details in config files in `config/tood/`.* \
[3] *Extraction code of baidu netdisk: tood.*


## Acknowledgement

Thanks MMDetection team for the wonderful open source project!


## Citation

If you find TOOD useful in your research, please consider citing:

```
@inproceedings{feng2021tood,
    title={TOOD: Task-aligned One-stage Object Detection},
    author={Feng, Chengjian and Zhong, Yujie and Gao, Yu and Scott, Matthew R and Huang, Weilin},
    booktitle={ICCV},
    year={2021}
}
```
