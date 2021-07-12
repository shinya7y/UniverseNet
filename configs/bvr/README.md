# RelationNet++: Bridging Visual Representations for Object Detection via Transformer Decoder

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{relationnetplusplus2020,
  title={RelationNet++: Bridging Visual Representations for Object Detection via Transformer Decoder},
  author={Chi, Cheng and Wei, Fangyun and Hu, Han},
  booktitle={NeurIPS},
  year={2020}
}
```

## Results and models

|    Method     | Backbone  | MS train | Lr schd | box AP |                              Config                               |                                           Download                                           |
| :-----------: | :-------: | :------: | :-----: | :----: | :---------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| RetinaNet+BVR |   R-50    |    N     |   1x    |  38.5  |           [config](bvr_retinanet_r50_fpn_gn_1x_coco.py)           | [Google](https://drive.google.com/file/d/1iKygKRi6EmqRsEQgBhJTfToWweVXEltB/view?usp=sharing) |
| RetinaNet+BVR | X-101-DCN |    Y     |   20e   |  46.5  | [config](bvr_retinanet_x101_fpn_dcn_mstrain_400_1200_20e_coco.py) | [Google](https://drive.google.com/file/d/1YyAG9OAjkeWStGkM5kLy6l95tXEa_E_b/view?usp=sharing) |
|   FCOS+BVR    | X-101-DCN |    Y     |   20e   |  48.9  |   [config](bvr_fcos_x101_fpn_dcn_mstrain_400_1200_20e_coco.py)    | [Google](https://drive.google.com/file/d/1IT1YBnNLrGQs-Be_drfF2ntEq4OjtCaO/view?usp=sharing) |
|   ATSS+BVR    | X-101-DCN |    Y     |   20e   |  50.7  |   [config](bvr_atss_x101_fpn_dcn_mstrain_400_1200_20e_coco.py)    | [Google](https://drive.google.com/file/d/16kTxTPGIN4O4wFHKhJMdFP_rlZ7eXde9/view?usp=sharing) |
