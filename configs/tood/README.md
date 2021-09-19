# TOOD: Task-aligned One-stage Object Detection

## Introduction

<!-- [ALGORITHM] -->

[Paper](https://arxiv.org/abs/2108.07755)

## Results and Models

| Backbone |    Anchor    | MS train |  DCN  | Lr schd | AP (minival) | AP (test-dev) |                            Config                            |                                                                        Download                                                                         |
| :------: | :----------: | :------: | :---: | :-----: | :----------: | :-----------: | :----------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | Anchor-free  |    No    |   N   |   1x    |     42.5     |     42.7      |              [config](tood_r50_fpn_1x_coco.py)               | [google](https://drive.google.com/file/d/1M7ccIsfQKA5pEtgMlRSadokLu_cFKO4B/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1rjAwcX2rq5xTm7_9AdWR2Q) |
|   R-50   | Anchor-based |    No    |   N   |   1x    |     42.4     |     42.8      |        [config](tood_r50_fpn_anchor_based_1x_coco.py)        | [google](https://drive.google.com/file/d/1G3Waqs3Xh7h1bfwcUfek91S1JKRCTAdV/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1E_Lsxj4GXhe7iPL6feVa5Q) |
|  R-101   | Anchor-free  |   Yes    |   N   |   2x    |     46.2     |     46.7      |          [config](tood_r101_fpn_mstrain_2x_coco.py)          | [google](https://drive.google.com/file/d/14NTtLVpG0I75jb55hB6smnibknkQ4wdb/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1Py-73Xysv5w5Gvqysc_RxA) |
|  X-101   | Anchor-free  |   Yes    |   N   |   2x    |     47.6     |     48.5      |       [config](tood_x101_64x4d_fpn_mstrain_2x_coco.py)       | [google](https://drive.google.com/file/d/1IbCZ5Lim_vkgRctsJ7Sb8czrOFQpmuRF/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1Y-CNmcHZtrWUFCrocSjiEA) |
|  R-101   | Anchor-free  |   Yes    |   Y   |   2x    |     49.2     |     49.6      |    [config](tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py)    | [google](https://drive.google.com/file/d/1ufipVoODv-NgthQ8ZvLeW12TEIsCgWl5/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1BfgMtKprAzoTBm91XEQk4Q) |
|  X-101   | Anchor-free  |   Yes    |   Y   |   2x    |     50.5     |     51.1      | [config](tood_x101_64x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py) | [google](https://drive.google.com/file/d/1xYSuZF5RfK81rJImNlTZWbIhPWfb5S5-/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1g2qiGJVV_dZmVF5D20SsNg) |

[1] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[2] *All models are trained with 16 images in a mini-batch.* \
[3] *`DCN` denotes deformable convolutional networks v2.* \
[4] *Refer to more details in config files.* \
[5] *Extraction code of baidu netdisk: tood.*

## Citation

```latex
@inproceedings{feng2021tood,
    title={TOOD: Task-aligned One-stage Object Detection},
    author={Feng, Chengjian and Zhong, Yujie and Gao, Yu and Scott, Matthew R and Huang, Weilin},
    booktitle={ICCV},
    year={2021}
}
```
