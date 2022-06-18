# Pyramid Vision Transformer (PVT)

## Introduction

<!-- [ALGORITHM] -->

This directory contains the configs and results of [PVTv2](https://arxiv.org/abs/2106.13797).
You can find more examples in [the original repository](https://github.com/whai362/PVT/tree/v2/detection).

Please consider using the [mmdet's configs](../pvt/) when you train new models.

## Results and Models

|       Method       | Backbone    |  Pretrain   | Lr schd | Aug | box AP | mask AP | Config                                                                  | Download                                                                                                                                                                                |
| :----------------: | ----------- | :---------: | :-----: | :-: | :----: | :-----: | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|        ATSS        | PVTv2-B2-Li | ImageNet-1K |   3x    | Yes |  48.9  |    -    | [config](atss_pvt_v2_b2_li_fpn_fp16_detraug_3x_coco.py)                 | [log](https://drive.google.com/file/d/1pg2O6gC5zKvFnAuC98wsexcx7mAqmv16/view?usp=sharing) & [model](https://drive.google.com/file/d/1CB4teTBwOpofCrHM91QvKKFLKcVMzZVi/view?usp=sharing) |
|        ATSS        | PVTv2-B2    | ImageNet-1K |   3x    | Yes |  49.9  |    -    | [config](atss_pvt_v2_b2_fpn_fp16_detraug_3x_coco.py)                    | [log](https://drive.google.com/file/d/1Vnf8-BszhTEkOQqwLA2-XLeuMT9n5ceR/view?usp=sharing) & [model](https://drive.google.com/file/d/1TKbj-i7oLgC7zstFuV0Neumu4iKMpBGh/view?usp=sharing) |
|        GFL         | PVTv2-B2-Li | ImageNet-1K |   3x    | Yes |  49.2  |    -    | [config](gfl_pvt_v2_b2_li_fpn_fp16_detraug_3x_coco.py)                  | [log](https://drive.google.com/file/d/1hqieuwCe79HAVMMVz8sEZsnG-R74Z_AO/view?usp=sharing) & [model](https://drive.google.com/file/d/1CnXlOEs9g7-LAoaDFcukTh5x0R4popZp/view?usp=sharing) |
|        GFL         | PVTv2-B2    | ImageNet-1K |   3x    | Yes |  50.2  |    -    | [config](gfl_pvt_v2_b2_fpn_fp16_detraug_3x_coco.py)                     | [log](https://drive.google.com/file/d/1AEMecyBnsomn4bxj1ySMxFdCsi8KCzGT/view?usp=sharing) & [model](https://drive.google.com/file/d/1XODtTQ3UAQz75vqhXBddqn7JpQke0vn6/view?usp=sharing) |
|    Sparse R-CNN    | PVTv2-B2-Li | ImageNet-1K |   3x    | Yes |  48.9  |    -    | [config](sparse_rcnn_pvt_v2_b2_li_fpn_300_proposals_detraug_3x_coco.py) | [log](https://drive.google.com/file/d/1uVHEwr5FDqlL3UvstpncCuaClU54lig6/view?usp=sharing) & [model](https://drive.google.com/file/d/1W8Wt2WbyhEi0JOUblaEcH9gx0I6z1wAv/view?usp=sharing) |
|    Sparse R-CNN    | PVTv2-B2    | ImageNet-1K |   3x    | Yes |  50.1  |    -    | [config](sparse_rcnn_pvt_v2_b2_fpn_300_proposals_detraug_3x_coco.py)    | [log](https://drive.google.com/file/d/1hDJwwICMmFqqF0A2Z35uNR6C5nI-2m22/view?usp=sharing) & [model](https://drive.google.com/file/d/1xtn-wD_nYSwudF1SqsSHl7opEXM4dhPN/view?usp=sharing) |
| Cascade Mask R-CNN | PVTv2-B2-Li | ImageNet-1K |   3x    | Yes |  50.9  |  44.0   | [config](cascade_mask_rcnn_pvt_v2_b2_li_fpn_fp16_detraug_3x_coco.py)    | [log](https://drive.google.com/file/d/1X_DC4yd89t4MJjQt9XmuCwx1hRmklN3z/view?usp=sharing) & [model](https://drive.google.com/file/d/1dG4O-M0EqKYdTtZqJdRjJopiwwnoakee/view?usp=sharing) |
| Cascade Mask R-CNN | PVTv2-B2    | ImageNet-1K |   3x    | Yes |  51.1  |  44.4   | [config](cascade_mask_rcnn_pvt_v2_b2_fpn_fp16_detraug_3x_coco.py)       | [log](https://drive.google.com/file/d/1gKEa_lUvm3Okonk33wgUjzfzVijEch-2/view?usp=sharing) & [model](https://drive.google.com/file/d/11jmqwLQSqQ1zin9D2sRYqeC8YfzGaN3V/view?usp=sharing) |

## Usage

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
python setup.py install --cpp_ext --cuda_ext --user
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

PVTv1

```latex
@misc{wang2021pyramid,
      title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions},
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2102.12122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

PVTv2

```latex
@misc{wang2021pvtv2,
      title={PVTv2: Improved Baselines with Pyramid Vision Transformer},
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2106.13797},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
