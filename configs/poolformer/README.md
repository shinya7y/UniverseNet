# PoolFormer

## Introduction

<!-- [ALGORITHM] -->

For details see [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418).

## Results and Models

| Method     | Backbone       | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                            | Download                                                                                                                                                                                |
| ---------- | -------------- | ----------- | :-----: | :-: | :----: | :-----: | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RetinaNet  | PoolFormer-S12 | ImageNet-1K |   1x    | No  |  36.2  |    -    | [config](retinanet_poolformer_s12_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1wdpzEmthjj8WJ99SnCLb32sF38FBbod7/view?usp=sharing) & [model](https://drive.google.com/file/d/1GKx4jbxdO4ClagPXXt7CoomrV4pOpqul/view?usp=sharing) |
| RetinaNet  | PoolFormer-S24 | ImageNet-1K |   1x    | No  |  38.9  |    -    | [config](retinanet_poolformer_s24_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1eNlNM1HDBLWejhMgMETvkPxLvUcP0OZ9/view?usp=sharing) & [model](https://drive.google.com/file/d/1EjsWpdopem-xeLndPQnQcHp8aoEUHQXR/view?usp=sharing) |
| RetinaNet  | PoolFormer-S36 | ImageNet-1K |   1x    | No  |  39.5  |    -    | [config](retinanet_poolformer_s36_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1qk-dSgfgYqFbo4zPu3Z3WdV7Kzm28_Xf/view?usp=sharing) & [model](https://drive.google.com/file/d/1EgJDCg7LXXnHdGdJaHyEnoBPm-fNG2bt/view?usp=sharing) |
| Mask R-CNN | PoolFormer-S12 | ImageNet-1K |   1x    | No  |  37.3  |  34.6   | [config](mask_rcnn_poolformer_s12_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1UfIP32QmT7MxBL_AQ3z1h7L21xYlB6aJ/view?usp=sharing) & [model](https://drive.google.com/file/d/1-GSkqaS3SovfCVDsH8CzS1DikPX3cFTY/view?usp=sharing) |
| Mask R-CNN | PoolFormer-S24 | ImageNet-1K |   1x    | No  |  40.1  |  37.0   | [config](mask_rcnn_poolformer_s24_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1yz6NPJ63ZlN02Oj2TY6KnjxK2Xg03BBa/view?usp=sharing) & [model](https://drive.google.com/file/d/10Br62EU-VErQq6rP67sf4qXJIBLOnmLT/view?usp=sharing) |
| Mask R-CNN | PoolFormer-S36 | ImageNet-1K |   1x    | No  |  41.0  |  37.7   | [config](mask_rcnn_poolformer_s36_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1oac1AVJ9skQZp0yXjTYY9_IhM8AxHVjT/view?usp=sharing) & [model](https://drive.google.com/file/d/1LyJxcO0fw2hwZg9Z--Zbjbw3W7U4JyqT/view?usp=sharing) |

All the models can also be downloaded by [BaiDu Yun](https://pan.baidu.com/s/1HSaJtxgCkUlawurQLq87wQ) (password: esac).

Please note that we just simply follow the hyper-parameters of PVT which may not be the optimal ones for PoolFormer.
Feel free to tune the hyper-parameters to get better performance.

## Citation

```latex
@article{yu2021metaformer,
  title={MetaFormer is Actually What You Need for Vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2111.11418},
  year={2021}
}
```
