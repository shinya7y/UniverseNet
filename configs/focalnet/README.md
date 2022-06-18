# FocalNet

> [Focal Modulation Networks](https://arxiv.org/abs/2203.11926)

<!-- [BACKBONE] -->

## Abstract

In this work, we propose focal modulation network (FocalNet in short), where self-attention (SA) is completely replaced by a focal modulation module that is more effective and efficient for modeling token interactions. Focal modulation comprises three components: (i) hierarchical contextualization, implemented using a stack of depth-wise convolutional layers, to encode visual contexts from short to long ranges at different granularity levels, (ii) gated aggregation to selectively aggregate context features for each visual token (query) based on its content, and (iii) modulation or element-wise affine transformation to fuse the aggregated features into the query vector. Extensive experiments show that FocalNets outperform the state-of-the-art SA counterparts (e.g., Swin Transformers) with similar time and memory cost on the tasks of image classification, object detection, and semantic segmentation. Specifically, our FocalNets with tiny and base sizes achieve 82.3% and 83.9% top-1 accuracy on ImageNet-1K. After pretrained on ImageNet-22K, it attains 86.5% and 87.3% top-1 accuracy when finetuned with resolution 224×224 and 384×384, respectively. FocalNets exhibit remarkable superiority when transferred to downstream tasks. For object detection with Mask R-CNN, our FocalNet base trained with 1× already surpasses Swin trained with 3× schedule (49.0 v.s. 48.5). For semantic segmentation with UperNet, FocalNet base evaluated at single-scale outperforms Swin evaluated at multi-scale (50.5 v.s. 49.7). These results render focal modulation a favorable alternative to SA for effective and efficient visual modeling in real-world applications.

<!--
<div align=center>
<img src=""/>
</div>
-->

## Results and Models

See the [official repository](https://github.com/microsoft/FocalNet/tree/ecfa580e252a106899e03de13af543f580f43da2#object-detection-on-coco).

### Notes

- We may need key conversion for the official Sparse R-CNN weights.
- Although most configs set `samples_per_gpu=1`, the authors overwrite it with `samples_per_gpu=2`.
  See [detection/README.md](https://github.com/microsoft/FocalNet/blob/main/detection/README.md) and `"iter": 7330` in [logs](https://github.com/microsoft/FocalNet/tree/ecfa580e252a106899e03de13af543f580f43da2#object-detection-on-coco).
  Thus we keep `base_batch_size=16` in `auto_scale_lr`.
- If you find performance difference from apex (used by the original authors), please raise an issue.

## Citation

```latex
@misc{yang2022focal,
      title={Focal Modulation Networks},
      author={Jianwei Yang and Chunyuan Li and Jianfeng Gao},
      year={2022},
      eprint={2203.11926},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
