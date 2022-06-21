# SwinV2

> [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)

<!-- [BACKBONE] -->

<!--
## Results and Models

https://arxiv.org/abs/2111.09883 A1.2
Cascade Mask R-CNN
window size 16x16
detraug
3x schedule

https://arxiv.org/abs/2206.03382
Cascade Mask R-CNN
SwinV2-B
ImageNet-22K pre-training
box / mask
53.0 / 45.8
-->

## Notice

- Since the authors have not published the code for SwinV2 detection,
  our implementation is based on [SwinV1 detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py) and [SwinV2 classification](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py).
- We have not confirmed the reproduction of the paper results.
- Please check for updates of the official repositories and [mmcls](https://github.com/open-mmlab/mmclassification/pull/799).

## Citation

```latex
@inproceedings{liu2021swinv2,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution},
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
