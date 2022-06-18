# Swin Transformer for Object Detection

## Introduction

<!-- [ALGORITHM] -->

This directory contains the configs and results of [Swin Transformer](https://arxiv.org/abs/2103.14030).
Most configs and results are based on the [official repository](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

Please consider using the [mmdet's configs](../swin/) when you train new models.

## Results and Models

### ATSS

| Backbone |  Pretrain   | Lr schd | box AP |                       config                       |                                                                    model                                                                    |
| :------: | :---------: | :-----: | :----: | :------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: |
|  Swin-T  | ImageNet-1K |   1x    |  43.7  | [config](atss_swint_fpn_fp16_4x4_adamw_1x_coco.py) | [github](https://github.com/shinya7y/weights/releases/download/v1.0.0/atss_swint_fpn_fp16_4x4_adamw_1x_coco_20210502_epoch_12-3c37c44b.pth) |

### Mask R-CNN

| Backbone |  Pretrain   | Lr schd | box AP | mask AP | #params | FLOPs |                                     config                                     |                                                                                          log                                                                                          |                                                                                      model                                                                                       |
| :------: | :---------: | :-----: | :----: | :-----: | :-----: | :---: | :----------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Swin-T  | ImageNet-1K |   1x    |  43.7  |  39.8   |   48M   | 267G  | [config](mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py)  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/19UOW0xl0qc-pXQ59aFKU5w) |
|  Swin-T  | ImageNet-1K |   3x    |  46.0  |  41.6   |   48M   | 267G  | [config](mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py)  |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1Te-Ovk4yaavmE4jcIOPAaw)   |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1YpauXYAFOohyMi3Vkb6DBg)   |
|  Swin-S  | ImageNet-1K |   3x    |  48.5  |  43.3   |   69M   | 359G  | [config](mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py) |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1ymCK7378QS91yWlxHMf1yw)  |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1V4w4aaV7HSjXNFTOSA6v6w)  |

### Cascade Mask R-CNN

| Backbone |  Pretrain   | Lr schd | box AP | mask AP | #params | FLOPs |                                               config                                                |                                                                                              log                                                                                              |                                                                                          model                                                                                           |
| :------: | :---------: | :-----: | :----: | :-----: | :-----: | :---: | :-------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Swin-T  | ImageNet-1K |   1x    |  48.1  |  41.7   |   86M   | 745G  | [config](cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py)  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1x4vnorYZfISr-d_VUSVQCA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1vFwbN1iamrtwnQSxMIW4BA) |
|  Swin-T  | ImageNet-1K |   3x    |  50.4  |  43.7   |   86M   | 745G  | [config](cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py)  |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1GW_ic617Ak_NpRayOqPSOA)   |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1i-izBrODgQmMwTv6F6-x3A)   |
|  Swin-S  | ImageNet-1K |   3x    |  51.9  |  45.0   |  107M   | 838G  | [config](cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/17Vyufk85vyocxrBT1AbavQ)  |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1Sv9-gP1Qpl6SGOF6DBhUbw)  |
|  Swin-B  | ImageNet-1K |   3x    |  51.9  |  45.0   |  145M   | 982G  | [config](cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py)  |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1UZAR39g-0kE_aGrINwfVHg)   |  [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1tHoC9PMVnldQUAfcF6FT3A)   |

<!--

### RepPoints V2

| Backbone |  Pretrain   | Lr schd | box AP | mask AP | #params | FLOPs |
| :------: | :---------: | :-----: | :----: | :-----: | :-----: | :---: |
|  Swin-T  | ImageNet-1K |   3x    |  50.0  |    -    |   45M   | 283G  |

### Mask RepPoints V2

| Backbone |  Pretrain   | Lr schd | box AP | mask AP | #params | FLOPs |
| :------: | :---------: | :-----: | :----: | :-----: | :-----: | :---: |
|  Swin-T  | ImageNet-1K |   3x    |  50.3  |  43.6   |   47M   | 292G  |

-->

**Notes**:

- **Pre-trained models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
- Access code for `baidu` is `swin`.

## Usage

### Inference

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:

```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]
```

For example, to train a Cascade Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:

```
tools/dist_train.sh configs/swin_original/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL>
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.

### Mixed Precision Training

The current configs use mixed precision training via MMCV by default.
Please install PyTorch >= 1.6.0 to use torch.cuda.amp.

If you find performance difference from apex (used by the original authors), please raise an issue.
Otherwise, we will clean code for apex.

<details>
<summary>Click me to use apex</summary>

To install apex, run:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Modify configs with the following code:

```python
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
fp16 = None
optimizer_config = dict(
    type='ApexOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

</details>

## Citation

```latex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Other Links

> **Image Classification**: See [Swin Transformer for Image Classification](https://github.com/microsoft/Swin-Transformer).

> **Semantic Segmentation**: See [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation).
