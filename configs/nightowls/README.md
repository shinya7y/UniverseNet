# NightOwls Dataset

## Introduction

<!-- [DATASET] -->

The NightOwls dataset is a dataset for pedestrian detection at night.
It contains three classes ('pedestrian', 'bicycledriver', 'motorbikedriver') except for ignore areas.

## Results

### UniverseNet

Approach:

- We fine-tuned a model from a Waymo Open pre-trained model that fine-tuned from a COCO pre-trained model.
- The weights and biases of the classification layer were transferred using [this script](../../tools/convert_waymo_checkpoint_for_nightowls.py).
- Early stopping was used for the driver classes.
  - 2 epochs w/o background images (4554 iterations)
  - Too early for the pedestrian class

|        Test scale         | Flip | f0val MR<br>pedestrian | test MR<br>pedestrian | test MR<br>bicycledriver | test MR<br>motorbikedriver | test MR<br>mean |
| :-----------------------: | :--: | :--------------------: | :-------------------: | :----------------------: | :------------------------: | :-------------: |
|        (1280, 800)        |  N   |          9.76          |         11.92         |           7.33           |            3.24            |      7.49       |
| (1280, 800), (1536, 960)  |  N   |          9.51          |           -           |            -             |             -              |        -        |
| (1280, 800), (1536, 960)  |  Y   |          9.28          |         10.85         |           4.81           |            1.35            |      5.67       |
| (1280, 800), (2048, 1280) |  N   |          9.70          |           -           |            -             |             -              |        -        |

- MR is Miss Rate (%) at Reasonable setting.
- The model was trained and evaluated using fp16 (mixed precision).

### NightOwls Detection Challenge 2020

UniverseNet achieves the 1st place in the NightOwls Detection Challenge 2020 all objects track.

|   Team name (Method)   |  Backbone   | Pre-training  |         Test scale         | test MR<br>pedestrian | test MR<br>bicycledriver | test MR<br>motorbikedriver | test MR<br>mean |
| :--------------------: | :---------: | :-----------: | :------------------------: | :-------------------: | :----------------------: | :------------------------: | :-------------: |
| shinya7y (UniverseNet) |    R2-50    | COCO -> Waymo |  (1280, 800), (1536, 960)  |         10.85         |           4.81           |            1.35            |      5.67       |
|       DeepBlueAI       | X-101-64x4d |     COCO      | (1920, 1280), (2048, 1280) |         10.43         |           4.17           |            9.59            |      8.06       |
|        dereyly         |      -      |       -       |             -              |         12.18         |           5.81           |           12.89            |      10.29      |

- MR is Miss Rate (%) at Reasonable setting. The values are from the organizer's presentation.
- The DeepBlueAI team used Cascade R-CNN, CBNet, Double-Head, DCN, and soft-NMS according to their presentation.
- Considering the method and test scale, UniverseNet should be faster than the runner-up.
- The above-mentioned approach of UniverseNet seems remarkably powerful for motorbikedriver.

## Notes

- Models pre-trained on the Waymo Open Dataset cannot be published due to its [license](https://waymo.com/open/terms/).
  If you need pre-trained UniverseNet models, please send a evidence for Waymo Open Dataset registration to shinya7y via [Twitter](https://twitter.com/shinya7y), [LinkedIn](https://www.linkedin.com/in/yosukeshinya), or other media.

## Citations

```
@inproceedings{Nightowls,
  title={NightOwls: A pedestrians at night dataset},
  author={Neumann, Luk{\'a}{\v{s}} and Karg, Michelle and Zhang, Shanshan and Scharfenberger, Christian and Piegert, Eric and Mistr, Sarah and Prokofyeva, Olga and Thiel, Robert and Vedaldi, Andrea and Zisserman, Andrew and Schiele, Bernt},
  booktitle={Asian Conference on Computer Vision},
  pages={691--705},
  year={2018},
  organization={Springer}
}
```

For the results of UniverseNet, please cite the arXiv v2 version of our paper.
https://arxiv.org/abs/2103.14027v2

```
@article{USB_shinya_2021,
  title={{USB}: Universal-Scale Object Detection Benchmark},
  author={Shinya, Yosuke},
  journal={arXiv:2103.14027v2},
  year={2021}
}
```
