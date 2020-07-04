# Waymo Open Dataset

## Introduction

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

|    Method     | Backbone | Lr schd | AP @832 | AP @1088 | AP @1344 | AP @1600 | KAP @832 | KAP @1088 | KAP @1344 | KAP @1600 |
| :-----------: | :------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :-------: | :-------: | :-------: |
| Faster R-CNN  |   R-50   |   1x    |  0.347  |  0.366   |  0.352   |  0.333   |  0.4997  |  0.5298   |  0.5115   |  0.4877   |
| Cascade R-CNN |   R-50   |   1x    |  0.364  |  0.385   |  0.374   |  0.360   |  0.5123  |  0.5425   |  0.5286   |  0.5124   |
|   RetinaNet   |   R-50   |   1x    |  0.326  |  0.348   |  0.351   |  0.344   |  0.4630  |  0.4956   |  0.5021   |  0.4960   |
|     ATSS      |   R-50   |   1x    |  0.354  |  0.377   |  0.370   |  0.361   |  0.5056  |  0.5366   |  0.5295   |  0.5201   |
|   ATSS+SEPC   |   R-50   |   1x    |  0.350  |  0.378   |  0.383   |  0.378   |  0.4968  |  0.5368   |  0.5450   |  0.5401   |
|  UniverseNet  |  R2-50   |   1x    |  0.391  |  0.424   |  0.439   |  0.444   |  0.5475  |  0.5891   |  0.6075   |  0.6160   |

- Test scales are shown after @ by shorter side pixels.
- KAP denotes KITTI-like mAP (IoU thresholds: 0.7 for vehicles, 0.5 for pedestrians and cyclists).
- In addition to ATSS+SEPC, UniverseNet uses Res2Net-v1b-50, DCN, and multi-scale training (640-1280).
- iBN of SEPC is set to False to allow for batch sizes less than 4.
- All models were trained and evaluated using fp16 (mixed precision).
- Each model was fine-tuned from a corresponding COCO pre-trained model.


### full train, f0val

|   Method    | Backbone |  Lr   | Lr schd | soft-NMS |     Test scale      |  AP   | KITTI-like mAP |
| :---------: | :------: | :---: | :-----: | :------: | :-----------------: | :---: | :------------: |
| UniverseNet |  R2-50   | 0.001 |   1e    |    N     |        1344         | 0.425 |       -        |
| UniverseNet |  R2-50   | 0.001 | **7e**  |    N     |        1344         | 0.446 |     0.616      |
| UniverseNet |  R2-50   | 0.001 |   7e    |  **Y**   |        1344         | 0.448 |     0.622      |
| UniverseNet |  R2-50   | 0.001 |   7e    |    Y     |      **1920**       | 0.458 |     0.638      |
| UniverseNet |  R2-50   | 0.001 |   7e    |    Y     | **960, 1600, 2240** | 0.467 |     0.650      |
| UniverseNet |  R2-50   | 0.001 |   7e    |    Y     | **952, 1592, 2232** | 0.468 |     0.651      |

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
  If you need pre-trained UniverseNet models, please take a screenshot of the confirmation mail of the dataset registration, and send it to shinya7y via [Twitter](https://twitter.com/shinya7y), [LinkedIn](https://www.linkedin.com/in/yosukeshinya), or other media.
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
