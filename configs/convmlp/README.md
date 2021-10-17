# Convolutional MLP

## Introduction

<!-- [ALGORITHM] -->

Preprint link: [ConvMLP: Hierarchical Convolutional MLPs for Vision](https://arxiv.org/abs/2109.04454)

## Results and Models

|   Method   | Backbone  | Lr schd | box AP | mask AP |                    Config                    |                                              Download                                               |
| :--------: | :-------: | :-----: | :----: | :-----: | :------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| RetinaNet  | ConvMLP-S |   1x    |  37.2  |    -    | [config](retinanet_convmlp_s_fpn_1x_coco.py) | [model](http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/detection/retinanet_convmlp_s_coco.pth) |
| RetinaNet  | ConvMLP-M |   1x    |  39.4  |    -    | [config](retinanet_convmlp_m_fpn_1x_coco.py) | [model](http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/detection/retinanet_convmlp_m_coco.pth) |
| RetinaNet  | ConvMLP-L |   1x    |  40.2  |    -    | [config](retinanet_convmlp_l_fpn_1x_coco.py) | [model](http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/detection/retinanet_convmlp_l_coco.pth) |
| Mask R-CNN | ConvMLP-S |   1x    |  38.4  |  35.7   | [config](maskrcnn_convmlp_s_fpn_1x_coco.py)  | [model](http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/detection/maskrcnn_convmlp_s_coco.pth)  |
| Mask R-CNN | ConvMLP-M |   1x    |  40.6  |  37.2   | [config](maskrcnn_convmlp_m_fpn_1x_coco.py)  | [model](http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/detection/maskrcnn_convmlp_m_coco.pth)  |
| Mask R-CNN | ConvMLP-L |   1x    |  41.7  |  38.2   | [config](maskrcnn_convmlp_l_fpn_1x_coco.py)  | [model](http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/detection/maskrcnn_convmlp_l_coco.pth)  |

## Citation

```bibtex
@article{li2021convmlp,
    title={ConvMLP: Hierarchical Convolutional MLPs for Vision},
    author={Jiachen Li and Ali Hassani and Steven Walton and Humphrey Shi},
    year={2021},
    eprint={2109.04454},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
