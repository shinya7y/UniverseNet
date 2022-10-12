# Manga109 Dataset

## Introduction

<!-- [DATASET] -->

[The Manga109 dataset](http://www.manga109.org/en/index.html) contains artificial images of manga (Japanese comics) and annotations for four categories (body, face, frame, and text).
Many characteristics are different from natural images.

The *Manga109-s* dataset (87 volumes) is a subset of the full *Manga109* dataset (109 volumes).
Unlike the full Manga109 dataset, the Manga109-s dataset can be used by commercial organizations.
For a wide range of users, we conduct experiments on Manga109-s.

Please see [this page](http://www.manga109.org/en/download_s.html) to download Manga109-s.
Please see [our manga109api fork](https://github.com/shinya7y/manga109api) to convert the dataset to COCO format.
We use [68train, 4val, and 15test splits](#dataset-splits).
The 15test set was selected to be well-balanced for reliable evaluation.

## Results

### 68train, 15test (Manga109-s v2020.12.18)

|      Method       |  Backbone  | Lr schd |  AP  |                                                                             Download                                                                             |
| :---------------: | :--------: | :-----: | :--: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   Faster R-CNN    |    R-50    |   1x    | 65.8 |        [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/faster_rcnn_r50_fpn_fp16_4x4_1x_manga109s_20201219_epoch_12-264d9f31.pth)         |
|   Cascade R-CNN   |    R-50    |   1x    | 67.6 |        [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/cascade_rcnn_r50_fpn_fp16_4x4_1x_manga109s_20201219_epoch_12-aece91e1.pth)        |
|     RetinaNet     |    R-50    |   1x    | 65.3 |         [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/retinanet_r50_fpn_fp16_4x4_1x_manga109s_20201219_epoch_12-9fa45ba4.pth)          |
|       ATSS        |    R-50    |   1x    | 66.5 |            [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/atss_r50_fpn_fp16_4x4_1x_manga109s_20201219_epoch_12-c3e34e96.pth)            |
|        GFL        |    R-50    |   1x    | 67.3 |            [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/gfl_r50_fpn_fp16_4x4_1x_manga109s_20201219_epoch_12-49659797.pth)             |
|       DETR        |    R-50    |   1x    | 31.2 |                  [model](https://github.com/shinya7y/weights/releases/download/v1.0.1/detr_r50_4x4_1x_manga109s_20220706_epoch_12-6d9a0c64.pth)                  |
|  Deformable DETR  |    R-50    |   1x    | 64.1 |           [model](https://github.com/shinya7y/weights/releases/download/v1.0.1/deformable_detr_r50_4x2x2_1x_manga109s_20220630_epoch_12-d930c644.pth)            |
|   Sparse R-CNN    |    R-50    |   1x    | 63.1 |          [model](https://github.com/shinya7y/weights/releases/download/v1.0.1/sparse_rcnn_r50_fpn_fp16_4x4_1x_manga109s_20220624_epoch_12-0d35864c.pth)          |
|       ATSS        |   Swin-T   |   1x    | 66.2 |         [model](https://github.com/shinya7y/weights/releases/download/v1.0.1/atss_swint_fpn_fp16_4x4_adamw_1x_manga109s_20210504_epoch_12-c96ddec3.pth)          |
|       ATSS        | ConvNeXt-T |   1x    | 67.4 |       [model](https://github.com/shinya7y/weights/releases/download/v1.0.1/atss_convnext-t_p4_w7_fpn_fp16_4x4_1x_manga109s_20220717_epoch_12-def8032b.pth)       |
|     ATSS+SEPC     |    R-50    |   1x    | 67.1 |      [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/atss_r50_fpn_sepc_noibn_fp16_4x4_1x_manga109s_20201219_epoch_12-b9eef036.pth)       |
|    ATSS+DyHead    |    R-50    |   1x    | 67.9 |          [model](https://github.com/shinya7y/weights/releases/download/v1.0.1/atss_r50_fpn_dyhead_fp16_4x4_1x_manga109s_20220626_epoch_12-9c75a796.pth)          |
|      YOLOX-L      |   CSP v5   |   1x    | 70.2 |               [model](https://github.com/shinya7y/weights/releases/download/v1.0.1/yolox_l_fp16_4x4_12e_manga109s_20220722_epoch_12-a19f80f6.pth)                |
|    UniverseNet    |   R2-50    |   1x    | 68.9 |   [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/universenet50_fp16_4x4_mstrain_480_960_1x_manga109s_20201220_epoch_12-ae4e7451.pth)    |
| UniverseNet 20.08 |   R2-50    |   1x    | 69.9 | [model](https://github.com/shinya7y/UniverseNet/releases/download/20.12/universenet50_2008_fp16_4x4_mstrain_480_960_1x_manga109s_20201220_epoch_12-6af914a4.pth) |

- In addition to ATSS+SEPC, UniverseNet uses Res2Net-v1b-50, DCN, and multi-scale training (480-960).
- The settings for normalization layers (including whether to use iBN of SEPC) depend on the config files.
- Most models were trained and evaluated using fp16 (mixed precision).
- Each model was fine-tuned from a corresponding COCO pre-trained model.

## Dataset splits

- 15test: `["Akuhamu", "BakuretsuKungFuGirl", "DollGun", "EvaLady", "HinagikuKenzan", "KyokugenCyclone", "LoveHina_vol01", "MomoyamaHaikagura", "TennenSenshiG", "UchiNoNyan'sDiary", "UnbalanceTokyo", "YamatoNoHane", "YoumaKourin", "YumeNoKayoiji", "YumeiroCooking"]`
- 4val: `["HealingPlanet", "LoveHina_vol14", "SeisinkiVulnus", "That'sIzumiko"]`
- 68train: All the other volumes

## Notes

- Please check the dataset licenses ([Manga109](http://www.manga109.org/en/download.html), [Manga109-s](http://www.manga109.org/en/download_s.html)).
- The typical scale of the original images is (1654, 1170).
  The number of maximum total pixels of (1216, 864) for Manga109 is almost the same as that of (1333, 800) for COCO.

## Citations

Users must cite the two papers below for use in academic papers.

```
@article{mtap_matsui_2017,
    author={Yusuke Matsui and Kota Ito and Yuji Aramaki and Azuma Fujimoto and Toru Ogawa and Toshihiko Yamasaki and Kiyoharu Aizawa},
    title={Sketch-based Manga Retrieval using Manga109 Dataset},
    journal={Multimedia Tools and Applications},
    volume={76},
    number={20},
    pages={21811--21838},
    doi={10.1007/s11042-016-4020-z},
    year={2017}
}
```

```
@article{multimedia_aizawa_2020,
    author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
    title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
    journal={IEEE MultiMedia},
    volume={27},
    number={2},
    pages={8--18},
    doi={10.1109/mmul.2020.2987895},
    year={2020}
}
```

Please cite the following paper for the benchmark results.
https://arxiv.org/abs/2103.14027

```
@inproceedings{USB_shinya_BMVC2022,
  title={{USB}: Universal-Scale Object Detection Benchmark},
  author={Shinya, Yosuke},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
```
