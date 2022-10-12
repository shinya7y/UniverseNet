# Waymo Open Dataset

## Introduction

<!-- [DATASET] -->

The Waymo Open Dataset is a large-scale diverse dataset for autonomous driving.
Although the KITTI dataset has been popular in this field, it is a small-scale non-diverse dataset.
Especially for object detection, the Waymo Open Dataset (or other large-scale datasets) should be used as a new standard.

Please see [WaymoCOCO](https://github.com/shinya7y/WaymoCOCO) (Waymo Open Dataset converter to COCO format) to prepare and convert the dataset.
The converter supports to extract 1/10 size dataset based on the ones place of frame index.
For example, *f0* (frame 0) subsets are extracted from frames 0, 10, 20, ..., 190.
Using 1/10 size subsets is useful when you would like to:

- do much trial and error before full training.
- evaluate the generalization of your method on the second dataset other than COCO.

## Results

### f0train, f0val

<details open>
<summary>Fine-tuning from COCO 1x models</summary>

|      Method       |  Backbone  | Lr schd | CAP @832 | CAP @1280 | KAP @832 | KAP @1280 |
| :---------------: | :--------: | :-----: | :------: | :-------: | :------: | :-------: |
|   Faster R-CNN    |    R-50    |   1x    |  0.345   |   0.362   |  0.4984  |  0.5265   |
|   Cascade R-CNN   |    R-50    |   1x    |  0.364   |   0.382   |  0.5107  |  0.5388   |
|     RetinaNet     |    R-50    |   1x    |  0.325   |   0.353   |  0.4618  |  0.5066   |
|       ATSS        |    R-50    |   1x    |  0.354   |   0.380   |  0.5056  |  0.5438   |
|        GFL        |    R-50    |   1x    |  0.357   |   0.383   |  0.5034  |  0.5423   |
|       DETR        |    R-50    |   1x    |  0.178   |     -     |  0.2935  |     -     |
|  Deformable DETR  |    R-50    |   1x    |  0.327   |     -     |  0.4852  |     -     |
|   Sparse R-CNN    |    R-50    |   1x    |  0.328   |     -     |  0.4792  |     -     |
|       ATSS        |   Swin-T   |   1x    |  0.372   |   0.388   |  0.5291  |  0.5549   |
|       ATSS        | ConvNeXt-T |   1x    |  0.383   |     -     |  0.5426  |     -     |
|     ATSS+SEPC     |    R-50    |   1x    |  0.350   |   0.385   |  0.4968  |  0.5494   |
|    ATSS+DyHead    |    R-50    |   1x    |  0.371   |     -     |  0.5227  |     -     |
|    UniverseNet    |   R2-50    |   1x    |  0.386   |   0.435   |  0.5430  |  0.6061   |
| UniverseNet-20.08 |   R2-50    |   1x    |  0.390   |   0.437   |  0.5459  |  0.6091   |

</details>

<details>
<summary>Fine-tuning from COCO 2x models</summary>

|      Method       | Backbone | Lr schd | CAP @832 | KAP @832 |
| :---------------: | :------: | :-----: | :------: | :------: |
|   Faster R-CNN    |   R-50   |   1x    |  0.347   |  0.4997  |
|     RetinaNet     |   R-50   |   1x    |  0.326   |  0.4630  |
|      YOLOX-L      |  CSP v5  |   1x    |  0.410   |  0.5756  |
|    UniverseNet    |  R2-50   |   1x    |  0.391   |  0.5475  |
|    UniverseNet    |  R2-50   |   2x    |  0.390   |  0.5505  |
| UniverseNet-20.08 |  R2-50   |   1x    |  0.397   |  0.5539  |

</details>

<details open>
<summary>Fine-tuning from COCO 3x models</summary>

| Method  | Backbone | Lr schd | CAP @832 | KAP @832 |
| :-----: | :------: | :-----: | :------: | :------: |
| YOLOX-L |  CSP v5  |   1x    |  0.416   |  0.5824  |

</details>

- Test scales are shown after @ by shorter side pixels.
- CAP denotes COCO-style AP.
- KAP denotes KITTI-style AP (IoU thresholds: 0.7 for vehicles, 0.5 for pedestrians and cyclists).
- In addition to ATSS+SEPC, UniverseNet uses Res2Net-v1b-50, DCN, and multi-scale training (640-1280).
- The settings for normalization layers (including whether to use iBN of SEPC) depend on the config files.
- Most models were trained and evaluated using fp16 (mixed precision).
- Each model was fine-tuned from a corresponding COCO pre-trained model.

### full train, f0val

|   Method    | Backbone |  Lr   | Lr schd | soft-NMS |     Test scale      |  CAP  |  KAP   |
| :---------: | :------: | :---: | :-----: | :------: | :-----------------: | :---: | :----: |
| UniverseNet |  R2-50   | 0.001 |   1e    |    N     |        1344         | 0.425 |   -    |
| UniverseNet |  R2-50   | 0.001 | **7e**  |    N     |        1344         | 0.446 | 0.6160 |
| UniverseNet |  R2-50   | 0.001 |   7e    |  **Y**   |        1344         | 0.448 | 0.6217 |
| UniverseNet |  R2-50   | 0.001 |   7e    |    Y     |      **1920**       | 0.458 | 0.6383 |
| UniverseNet |  R2-50   | 0.001 |   7e    |    Y     | **960, 1600, 2240** | 0.467 | 0.6502 |
| UniverseNet |  R2-50   | 0.001 |   7e    |    Y     | **952, 1592, 2232** | 0.468 | 0.6510 |

- Changed values are shown ​​in bold.
- 7e: 7 epochs training, lr decay at 6 epoch.
- Higher learning rates and/or longer training will be preferable for better AP.
- A machine with 208-416 GB of CPU memory is needed for full training as of MMDetection v2.0.

## Training memory and inference time

|    Method     | Train scale | Mem (GB) | Inf time @832 (fps) |
| :-----------: | :---------: | :------: | :-----------------: |
| Faster R-CNN  |     832     |   4.2    |        33.7         |
| Cascade R-CNN |     832     |   4.7    |        26.4         |
|   RetinaNet   |     832     |   3.8    |        35.2         |
|     ATSS      |     832     |   4.5    |        31.5         |
|   ATSS+SEPC   |     832     |   3.3    |        22.9         |
|  UniverseNet  |     832     |   4.2    |        16.2         |
|  UniverseNet  |  640-1280   |   9.1    |        16.2         |

- samples_per_gpu=4 for training.
- samples_per_gpu=1 for measuring inference time on V100.
- All models were trained and evaluated using fp16 (mixed precision).

## Notes

- Models trained on the Waymo Open Dataset cannot be published due to [the dataset license](https://waymo.com/open/terms/).
  If you need pre-trained UniverseNet models, please send a evidence for Waymo Open Dataset registration to shinya7y via [Twitter](https://twitter.com/shinya7y), [LinkedIn](https://www.linkedin.com/in/yosukeshinya), or other media.
- In the tables above, test scales are shown by shorter side pixels. Longer side pixels are 1.5x.
  The number of maximum total pixels of (1248, 832) for Waymo Open is almost the same as that of (1333, 800) for COCO.

## Citations

```
@inproceedings{waymo_open_dataset_cvpr2020,
  author = {Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and Vasudevan, Vijay and Han, Wei and Ngiam, Jiquan and Zhao, Hang and Timofeev, Aleksei and Ettinger, Scott and Krivokon, Maxim and Gao, Amy and Joshi, Aditya and Zhang, Yu and Shlens, Jonathon and Chen, Zhifeng and Anguelov, Dragomir},
  title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

```
@misc{waymo_open_dataset,
  title = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year = {2019}
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
