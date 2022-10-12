# UniverseNet

This is the official repository of "[USB: Universal-Scale Object Detection Benchmark](https://arxiv.org/abs/2103.14027)" (BMVC 2022).

We established a new benchmark *USB* with fair protocols and designed state-of-the-art detectors *UniverseNets* for universal-scale object detection.
This repository extends MMDetection with more features and allows for more comprehensive benchmarking and development.

## Introduction

![universal-scale object detection](https://user-images.githubusercontent.com/42844407/113513063-b5aa2780-95a2-11eb-8413-2fb470256a1a.png)

Benchmarks, such as COCO, play a crucial role in object detection. However, existing benchmarks are insufficient in scale variation, and their protocols are inadequate for fair comparison. In this paper, we introduce the Universal-Scale object detection Benchmark (USB). USB has variations in object scales and image domains by incorporating COCO with the recently proposed Waymo Open Dataset and Manga109-s dataset. To enable fair comparison and inclusive research, we propose training and evaluation protocols. They have multiple divisions for training epochs and evaluation image resolutions, like weight classes in sports, and compatibility across training protocols, like the backward compatibility of the Universal Serial Bus. Specifically, we request participants to report results with not only higher protocols (longer training) but also lower protocols (shorter training). Using the proposed benchmark and protocols, we conducted extensive experiments using 15 methods and found weaknesses of existing COCO-biased methods.

Please refer to our paper for details.
https://arxiv.org/abs/2103.14027

## Changelog

- recent:
  - Our paper got accepted to BMVC 2022!
- 22.06 (June 2022):
  - Add SwinV2, FocalNet, GBR COTS dataset
  - Update codes for mmdet 2.25.0, mmcv-full 1.4.4
- 21.12 (Dec. 2021):
  - Support [finer scale-wise AP metrics](docs/en/tutorials/finer_scale_ap.md)
  - Add codes for TOOD, ConvMLP, PoolFormer
  - Update codes for PyTorch 1.9.0, mmdet 2.17.0, mmcv-full 1.3.13
- 21.09 (Sept. 2021):
  - Support gradient accumulation to simulate large batch size with few GPUs ([example](configs/manga109/universenet50_2008_fp16_1x4x4_mstrain_480_960_1x_manga109s.py))
  - Add codes for CBNetV2, PVT, PVTv2, DDOD
  - Update and fix codes for mmdet 2.14.0, mmcv-full 1.3.9
- 21.04 (Apr. 2021):
  - Propose [Universal-Scale object detection Benchmark (USB)](https://arxiv.org/abs/2103.14027)
  - Add codes for Swin Transformer, GFLv2, RelationNet++ (BVR)
  - Update and fix codes for PyTorch 1.7.1, mmdet 2.11.0, mmcv-full 1.3.2
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

- [x] [UniverseNets (BMVC 2022)](configs/universenet/)
- [x] [FocalNet (NeurIPS 2022)](configs/focalnet/)
- [x] [Swin Transformer V2 (CVPR 2022)](configs/swinv2/)
- [x] [PoolFormer (CVPR 2022)](configs/poolformer/)
- [x] [PVTv2 (CVMJ 2022)](configs/pvtv2_original/) stronger models
- [x] [ConvMLP (arXiv 2021)](configs/convmlp/)
- [x] [CBNetV2 (arXiv 2021)](configs/cbnet/)
- [x] ~~TOOD (ICCV 2021)~~ supported
- [x] ~~PVT (ICCV 2021)~~ supported
- [x] [Swin Transformer (ICCV 2021)](configs/swin_original/) stronger models
- [x] ~~DDOD (ACMMM 2021)~~ supported
- [x] [GFLv2 (CVPR 2021)](configs/gflv2/)
- [x] [RelationNet++ (BVR) (NeurIPS 2020)](configs/bvr/)
- [x] SEPC (CVPR 2020)
- [x] [ATSS-style TTA (CVPR 2020)](configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_test_vote.py)
- [x] ~~Test-time augmentation for ATSS and GFL~~ merged

Benchmarks and datasets:

- [x] USB (BMVC 2022)
- [x] [Waymo Open Dataset (CVPR 2020)](configs/waymo_open/)
- [x] [Manga109-s dataset (MTAP 2017, IEEE MultiMedia 2020)](configs/manga109/)
- [x] [NightOwls dataset (ACCV 2018)](configs/nightowls/)
- [x] [GBR COTS dataset (arXiv 2021)](configs/datasets/kaggle_gbr_cots/)

## Usage

### Installation

See [get_started.md](docs/en/get_started.md).

### Basic Usage

See [MMDetection documents](#getting-started).
Especially, see [this document](docs/en/1_exist_data_model.md) to evaluate and train existing models on COCO.

### Examples

We show examples to evaluate and train UniverseNet-20.08 on COCO with 4 GPUs.

```bash
# evaluate pre-trained model
mkdir -p ${HOME}/data/checkpoints/
wget -P ${HOME}/data/checkpoints/ https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth
CONFIG_FILE=configs/universenet/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco.py
CHECKPOINT_FILE=${HOME}/data/checkpoints/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth
GPU_NUM=4
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval bbox

# train model
CONFIG_FILE=configs/universenet/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco.py
CONFIG_NAME=$(basename ${CONFIG_FILE} .py)
WORK_DIR="${HOME}/logs/coco/${CONFIG_NAME}_`date +%Y%m%d_%H%M%S`"
GPU_NUM=4
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --work-dir ${WORK_DIR} --seed 0
```

Even if you have one GPU, we recommend using `tools/dist_train.sh` and `tools/dist_test.sh` to avoid [a SyncBN issue](https://github.com/open-mmlab/mmdetection/issues/847#issuecomment-504421173).

## Citation

```
@inproceedings{USB_shinya_BMVC2022,
  title={{USB}: Universal-Scale Object Detection Benchmark},
  author={Shinya, Yosuke},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
```

## License

Major parts of the code are released under the [Apache 2.0 license](LICENSE).
Plsease check [NOTICE](NOTICE) for exceptions.

## Acknowledgements

Some codes are modified from the repositories of
[FocalNet](https://github.com/microsoft/FocalNet),
[PoolFormer](https://github.com/sail-sg/poolformer),
[ConvMLP](https://github.com/SHI-Labs/Convolutional-MLPs),
[Swin Transformer](https://github.com/microsoft/Swin-Transformer),
[Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection),
[RelationNet++](https://github.com/microsoft/RelationNet2),
[SEPC](https://github.com/jshilong/SEPC),
[PVT](https://github.com/whai362/PVT),
[CBNetV2](https://github.com/VDIGPKU/CBNetV2),
[GFLv2](https://github.com/implus/GFocalV2),
and [NightOwls](https://gitlab.com/vgg/nightowlsapi).
When merging, please note that there are some minor differences from the above repositories and [the original MMDetection repository](https://github.com/open-mmlab/mmdetection).

<br><br>

<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)

[📘Documentation](https://mmdetection.readthedocs.io/en/stable/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/stable/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/en/stable/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/stable/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

<img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

</details>

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## What's New

**2.25.0** was released in 1/6/2022:

- Support dedicated `MMDetWandbHook` hook
- Support [ConvNeXt](configs/convnext), [DDOD](configs/ddod), [SOLOv2](configs/solov2)
- Support [Mask2Former](configs/mask2former) for instance segmentation
- Rename [config files of Mask2Former](configs/mask2former)

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

For compatibility changes between different versions of MMDetection, please refer to [compatibility.md](docs/en/compatibility.md).

## Installation

Please refer to [Installation](docs/en/get_started.md/#Installation) for installation instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection. We provide [colab tutorial](demo/MMDet_Tutorial.ipynb) and [instance segmentation colab tutorial](demo/MMDet_InstanceSeg_Tutorial.ipynb), and other tutorials for:

- [with existing dataset](docs/en/1_exist_data_model.md)
- [with new dataset](docs/en/2_new_data_model.md)
- [with existing dataset_new_model](docs/en/3_exist_data_new_model.md)
- [learn about configs](docs/en/tutorials/config.md)
- [customize_datasets](docs/en/tutorials/customize_dataset.md)
- [customize data pipelines](docs/en/tutorials/data_pipeline.md)
- [customize_models](docs/en/tutorials/customize_models.md)
- [customize runtime settings](docs/en/tutorials/customize_runtime.md)
- [customize_losses](docs/en/tutorials/customize_losses.md)
- [finetuning models](docs/en/tutorials/finetune.md)
- [export a model to ONNX](docs/en/tutorials/pytorch2onnx.md)
- [export ONNX to TRT](docs/en/tutorials/onnx2tensorrt.md)
- [weight initialization](docs/en/tutorials/init_cfg.md)
- [how to xxx](docs/en/tutorials/how_to.md)

## Overview of Benchmark and Model Zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Panoptic Segmentation</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/fast_rcnn">Fast R-CNN (ICCV'2015)</a></li>
            <li><a href="configs/faster_rcnn">Faster R-CNN (NeurIPS'2015)</a></li>
            <li><a href="configs/rpn">RPN (NeurIPS'2015)</a></li>
            <li><a href="configs/ssd">SSD (ECCV'2016)</a></li>
            <li><a href="configs/retinanet">RetinaNet (ICCV'2017)</a></li>
            <li><a href="configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a></li>
            <li><a href="configs/yolo">YOLOv3 (ArXiv'2018)</a></li>
            <li><a href="configs/cornernet">CornerNet (ECCV'2018)</a></li>
            <li><a href="configs/grid_rcnn">Grid R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/guided_anchoring">Guided Anchoring (CVPR'2019)</a></li>
            <li><a href="configs/fsaf">FSAF (CVPR'2019)</a></li>
            <li><a href="configs/centernet">CenterNet (CVPR'2019)</a></li>
            <li><a href="configs/libra_rcnn">Libra R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/tridentnet">TridentNet (ICCV'2019)</a></li>
            <li><a href="configs/fcos">FCOS (ICCV'2019)</a></li>
            <li><a href="configs/reppoints">RepPoints (ICCV'2019)</a></li>
            <li><a href="configs/free_anchor">FreeAnchor (NeurIPS'2019)</a></li>
            <li><a href="configs/cascade_rpn">CascadeRPN (NeurIPS'2019)</a></li>
            <li><a href="configs/foveabox">Foveabox (TIP'2020)</a></li>
            <li><a href="configs/double_heads">Double-Head R-CNN (CVPR'2020)</a></li>
            <li><a href="configs/atss">ATSS (CVPR'2020)</a></li>
            <li><a href="configs/nas_fcos">NAS-FCOS (CVPR'2020)</a></li>
            <li><a href="configs/centripetalnet">CentripetalNet (CVPR'2020)</a></li>
            <li><a href="configs/autoassign">AutoAssign (ArXiv'2020)</a></li>
            <li><a href="configs/sabl">Side-Aware Boundary Localization (ECCV'2020)</a></li>
            <li><a href="configs/dynamic_rcnn">Dynamic R-CNN (ECCV'2020)</a></li>
            <li><a href="configs/detr">DETR (ECCV'2020)</a></li>
            <li><a href="configs/paa">PAA (ECCV'2020)</a></li>
            <li><a href="configs/vfnet">VarifocalNet (CVPR'2021)</a></li>
            <li><a href="configs/sparse_rcnn">Sparse R-CNN (CVPR'2021)</a></li>
            <li><a href="configs/yolof">YOLOF (CVPR'2021)</a></li>
            <li><a href="configs/yolox">YOLOX (CVPR'2021)</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR (ICLR'2021)</a></li>
            <li><a href="configs/tood">TOOD (ICCV'2021)</a></li>
            <li><a href="configs/ddod">DDOD (ACM MM'2021)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/mask_rcnn">Mask R-CNN (ICCV'2017)</a></li>
          <li><a href="configs/cascade_rcnn">Cascade Mask R-CNN (CVPR'2018)</a></li>
          <li><a href="configs/ms_rcnn">Mask Scoring R-CNN (CVPR'2019)</a></li>
          <li><a href="configs/htc">Hybrid Task Cascade (CVPR'2019)</a></li>
          <li><a href="configs/yolact">YOLACT (ICCV'2019)</a></li>
          <li><a href="configs/instaboost">InstaBoost (ICCV'2019)</a></li>
          <li><a href="configs/solo">SOLO (ECCV'2020)</a></li>
          <li><a href="configs/point_rend">PointRend (CVPR'2020)</a></li>
          <li><a href="configs/detectors">DetectoRS (ArXiv'2020)</a></li>
          <li><a href="configs/solov2">SOLOv2 (NeurIPS'2020)</a></li>
          <li><a href="configs/scnet">SCNet (AAAI'2021)</a></li>
          <li><a href="configs/queryinst">QueryInst (ICCV'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/panoptic_fpn">Panoptic FPN (CVPR'2019)</a></li>
          <li><a href="configs/maskformer">MaskFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
        </ul>
      </td>
      <td>
        </ul>
          <li><b>Contrastive Learning</b></li>
        <ul>
        <ul>
          <li><a href="configs/selfsup_pretrain">SwAV (NeurIPS'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCo (CVPR'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCov2 (ArXiv'2020)</a></li>
        </ul>
        </ul>
        </ul>
          <li><b>Distillation</b></li>
        <ul>
        <ul>
          <li><a href="configs/ld">Localization Distillation (CVPR'2022)</a></li>
          <li><a href="configs/lad">Label Assignment Distillation (WACV'2022)</a></li>
        </ul>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>VGG (ICLR'2015)</li>
        <li>ResNet (CVPR'2016)</li>
        <li>ResNeXt (CVPR'2017)</li>
        <li>MobileNetV2 (CVPR'2018)</li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/empirical_attention">Generalized Attention (ICCV'2019)</a></li>
        <li><a href="configs/gcnet">GCNet (ICCVW'2019)</a></li>
        <li><a href="configs/res2net">Res2Net (TPAMI'2020)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/pvt">PVT (ICCV'2021)</a></li>
        <li><a href="configs/swin">Swin (CVPR'2021)</a></li>
        <li><a href="configs/pvt">PVTv2 (ArXiv'2021)</a></li>
        <li><a href="configs/resnet_strikes_back">ResNet strikes back (ArXiv'2021)</a></li>
        <li><a href="configs/efficientnet">EfficientNet (ArXiv'2021)</a></li>
        <li><a href="configs/convnext">ConvNeXt (CVPR'2022)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/pafpn">PAFPN (CVPR'2018)</a></li>
        <li><a href="configs/nas_fpn">NAS-FPN (CVPR'2019)</a></li>
        <li><a href="configs/carafe">CARAFE (ICCV'2019)</a></li>
        <li><a href="configs/fpg">FPG (ArXiv'2020)</a></li>
        <li><a href="configs/groie">GRoIE (ICPR'2020)</a></li>
        <li><a href="configs/dyhead">DyHead (CVPR'2021)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/ghm">GHM (AAAI'2019)</a></li>
          <li><a href="configs/gfl">Generalized Focal Loss (NeurIPS'2020)</a></li>
          <li><a href="configs/seesaw_loss">Seasaw Loss (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py">OHEM (CVPR'2016)</a></li>
          <li><a href="configs/gn">Group Normalization (ECCV'2018)</a></li>
          <li><a href="configs/dcn">DCN (ICCV'2017)</a></li>
          <li><a href="configs/dcnv2">DCNv2 (CVPR'2019)</a></li>
          <li><a href="configs/gn+ws">Weight Standardization (ArXiv'2019)</a></li>
          <li><a href="configs/pisa">Prime Sample Attention (CVPR'2020)</a></li>
          <li><a href="configs/strong_baselines">Strong Baselines (CVPR'2021)</a></li>
          <li><a href="configs/resnet_strikes_back">Resnet strikes back (ArXiv'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

Some other methods are also supported in [projects using MMDetection](./docs/en/projects.md).

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

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

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
