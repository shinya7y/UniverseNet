# Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection

## Introduction

<!-- [ALGORITHM] -->

GFocalV2 (GFLV2) is a next generation of GFocalV1 (GFLV1), which utilizes the statistics of learned bounding box distributions to guide the reliable localization quality estimation.
For details see [GFocalV2](https://arxiv.org/abs/2011.12885).

## Results and Models

| Backbone   | Lr schd | Multi-scale training | box AP | Inf time (fps) |                                          Download                                           |
| ---------- | :-----: | :------------------: | :----: | :------------: | :-----------------------------------------------------------------------------------------: |
| R-50       |   1x    |          No          |  41.0  |      19.4      | [model](https://drive.google.com/file/d/1wSE9-c7tcQwIDPC6Vm_yfOokdPfmYmy7/view?usp=sharing) |
| R-50       |   2x    |         Yes          |  43.9  |      19.4      | [model](https://drive.google.com/file/d/17-1cKRdR5J3SfZ9NBCwe6QE554uTS30F/view?usp=sharing) |
| R-101      |   2x    |         Yes          |  45.8  |      14.6      | [model](https://drive.google.com/file/d/1qomgA7mzKW0bwybtG4Avqahv67FUxmNx/view?usp=sharing) |
| R-101-dcn  |   2x    |         Yes          |  48.0  |      12.7      | [model](https://drive.google.com/file/d/1xsBjxmqsJoYZYPMr0k06X5K9nnPrexcx/view?usp=sharing) |
| X-101-dcn  |   2x    |         Yes          |  48.8  |      10.7      | [model](https://drive.google.com/file/d/1AHDVQoclYPSP0Ync2a5FCsr_rhq2QdMH/view?usp=sharing) |
| R2-101-dcn |   2x    |         Yes          |  49.9  |      10.9      | [model](https://drive.google.com/file/d/1sAXfYLXIxZgMrC44LBqDgfYImThZ_kud/view?usp=sharing) |

\[1\] *The reported numbers here are from new experimental trials (in the cleaned repo), which may be slightly different from the original paper.* \
\[2\] *Note that the 1x performance may be slightly unstable due to insufficient training. In practice, the 2x results are considerably stable between multiple runs.* \
\[3\] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
\[4\] *`dcn` denotes deformable convolutional networks.* \
\[5\] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.*

## Citation

```latex
@article{li2020gfl,
  title={Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}
```

```latex
@article{li2020gflv2,
  title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2011.12885},
  year={2020}
}
```
