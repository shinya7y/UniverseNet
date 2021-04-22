# UniverseNet

UniverseNets are state-of-the-art detectors for universal-scale object detection.
Please refer to our paper for details.
https://arxiv.org/abs/2103.14027

![universal-scale object detection](https://user-images.githubusercontent.com/42844407/113513063-b5aa2780-95a2-11eb-8413-2fb470256a1a.png)
![COCO AP](https://user-images.githubusercontent.com/42844407/112571346-317dd480-8e2b-11eb-8b1e-a97ba2e97997.png)

## Changelog

- master
  - Propose [Universal-Scale object detection Benchmark (USB)](https://arxiv.org/abs/2103.14027)
  - Add codes for Swin Transformer, GFLv2, RelationNet++ (BVR)
  - Update and fix codes for PyTorch 1.7.1, mmdet 2.11.0, mmcv-full 1.3.0
- 20.12 (Dec. 2020):
  - Add configs for Manga109-s dataset
  - Add ATSS-style TTA for SOTA accuracy (COCO test-dev AP 54.1)
  - Add UniverseNet 20.08s for realtime speed (> 30 fps)
- 20.10 (Oct. 2020):
  - Add variants of UniverseNet 20.08
  - Update and fix codes for PyTorch 1.6.0, mmdet 2.4.0, mmcv-full 1.1.2
- 20.08 (Aug. 2020): **UniverseNet 20.08**
  - Improve usage of batchnorm
  - Use DCN modestly by default for faster training and inference
- 20.07 (July 2020): **UniverseNet+GFL**
  - Add GFL to improve accuracy and speed
  - Provide stronger pre-trained model (backbone: Res2Net-101)
- 20.06 (June 2020): **UniverseNet**
  - Achieve SOTA single-stage detector on Waymo Open Dataset 2D detection
  - Win 1st place in NightOwls Detection Challenge 2020 all objects track

## Features not in the original MMDetection

Methods and architectures:

- [x] [UniverseNets (arXiv 2021)](configs/universenet/)
- [x] [Swin Transformer (arXiv 2021)](configs/swin/)
- [x] [GFLv2 (CVPR 2021)](configs/gflv2/)
- [x] [RelationNet++ (BVR) (NeurIPS 2020)](configs/bvr/)
- [x] SEPC (CVPR 2020)
- [x] [ATSS-style TTA (CVPR 2020)](configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_test_vote.py)
- [x] ~~Test-time augmentation for ATSS and GFL~~ [merged](https://github.com/open-mmlab/mmdetection/commits?author=shinya7y)

Benchmarks and datasets:

- [x] USB (arXiv 2021)
- [x] [Waymo Open Dataset (CVPR 2020)](configs/waymo_open/)
- [x] [Manga109-s dataset (MTAP 2017, IEEE MultiMedia 2020)](configs/manga109/)
- [x] [NightOwls dataset (ACCV 2018)](configs/nightowls/)

## Citation

```
@article{USB_shinya_2021,
  title={{USB}: Universal-Scale Object Detection Benchmark},
  author={Shinya, Yosuke},
  journal={arXiv:2103.14027},
  year={2021}
}
```

## License

Major parts of the code are released under the [Apache 2.0 license](LICENSE).
Plsease check [NOTICE](NOTICE) for exceptions.

## Acknowledgements

Some codes are modified from the repositories of
[Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection),
[RelationNet++](https://github.com/microsoft/RelationNet2),
[SEPC](https://github.com/jshilong/SEPC),
[GFLv2](https://github.com/implus/GFocalV2),
and [NightOwls](https://gitlab.com/vgg/nightowlsapi).
When merging, please note that there are some minor differences from the above repositories and [the original MMDetection repository](https://github.com/open-mmlab/mmdetection).

<br><br>


<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
</div>

**News**: We released the technical report on [ArXiv](https://arxiv.org/abs/1906.07155).

Documentation: https://mmdetection.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.3+**.
The old v1.x branch works with PyTorch 1.1 to 1.4, but v2.0 is strongly recommended for faster speed, higher performance, better design and more friendly usage.

![demo image](resources/coco_test_12510.jpg)

### Major features

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## Changelog

v2.11.0 was released in 01/04/2021.
Please refer to [changelog.md](docs/changelog.md) for details and release history.
A comparison between v1.x and v2.0 codebases can be found in [compatibility.md](docs/compatibility.md).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] VGG (ICLR'2015)
- [x] HRNet (CVPR'2019)
- [x] RegNet (CVPR'2020)
- [x] Res2Net (TPAMI'2020)
- [x] ResNeSt (ArXiv'2020)

Supported methods:

- [x] [RPN (NeurIPS'2015)](configs/rpn)
- [x] [Fast R-CNN (ICCV'2015)](configs/fast_rcnn)
- [x] [Faster R-CNN (NeurIPS'2015)](configs/faster_rcnn)
- [x] [Mask R-CNN (ICCV'2017)](configs/mask_rcnn)
- [x] [Cascade R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [Cascade Mask R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [SSD (ECCV'2016)](configs/ssd)
- [x] [RetinaNet (ICCV'2017)](configs/retinanet)
- [x] [GHM (AAAI'2019)](configs/ghm)
- [x] [Mask Scoring R-CNN (CVPR'2019)](configs/ms_rcnn)
- [x] [Double-Head R-CNN (CVPR'2020)](configs/double_heads)
- [x] [Hybrid Task Cascade (CVPR'2019)](configs/htc)
- [x] [Libra R-CNN (CVPR'2019)](configs/libra_rcnn)
- [x] [Guided Anchoring (CVPR'2019)](configs/guided_anchoring)
- [x] [FCOS (ICCV'2019)](configs/fcos)
- [x] [RepPoints (ICCV'2019)](configs/reppoints)
- [x] [Foveabox (TIP'2020)](configs/foveabox)
- [x] [FreeAnchor (NeurIPS'2019)](configs/free_anchor)
- [x] [NAS-FPN (CVPR'2019)](configs/nas_fpn)
- [x] [ATSS (CVPR'2020)](configs/atss)
- [x] [FSAF (CVPR'2019)](configs/fsaf)
- [x] [PAFPN (CVPR'2018)](configs/pafpn)
- [x] [Dynamic R-CNN (ECCV'2020)](configs/dynamic_rcnn)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CARAFE (ICCV'2019)](configs/carafe/README.md)
- [x] [DCNv2 (CVPR'2019)](configs/dcn/README.md)
- [x] [Group Normalization (ECCV'2018)](configs/gn/README.md)
- [x] [Weight Standardization (ArXiv'2019)](configs/gn+ws/README.md)
- [x] [OHEM (CVPR'2016)](configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py)
- [x] [Soft-NMS (ICCV'2017)](configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py)
- [x] [Generalized Attention (ICCV'2019)](configs/empirical_attention/README.md)
- [x] [GCNet (ICCVW'2019)](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training (ArXiv'2017)](configs/fp16/README.md)
- [x] [InstaBoost (ICCV'2019)](configs/instaboost/README.md)
- [x] [GRoIE (ICPR'2020)](configs/groie/README.md)
- [x] [DetectoRS (ArXix'2020)](configs/detectors/README.md)
- [x] [Generalized Focal Loss (NeurIPS'2020)](configs/gfl/README.md)
- [x] [CornerNet (ECCV'2018)](configs/cornernet/README.md)
- [x] [Side-Aware Boundary Localization (ECCV'2020)](configs/sabl/README.md)
- [x] [YOLOv3 (ArXiv'2018)](configs/yolo/README.md)
- [x] [PAA (ECCV'2020)](configs/paa/README.md)
- [x] [YOLACT (ICCV'2019)](configs/yolact/README.md)
- [x] [CentripetalNet (CVPR'2020)](configs/centripetalnet/README.md)
- [x] [VFNet (ArXix'2020)](configs/vfnet/README.md)
- [x] [DETR (ECCV'2020)](configs/detr/README.md)
- [x] [CascadeRPN (NeurIPS'2019)](configs/cascade_rpn/README.md)
- [x] [SCNet (AAAI'2021)](configs/scnet/README.md)

Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
