# Finer scale-wise AP metrics

## Introduction

As defined in [TinyPerson benchmark](https://arxiv.org/abs/1912.10664), we can consider two types of object scales.

- Absolute Scale: `sqrt{wh}`
- Relative Scale: `sqrt{wh/WH}`

where `w` and `h` denote the object's width and height, respectively,
and `W` and `H` denote the image's width and height, respectively.
In other words, `wh` means the total number of pixels occupied by the object in the original (non-resized) image,
and `wh/WH` means the ratio of area occupied by the object in the image, which is constant regardless of image resizing.

The scale-wise AP metrics of COCO and other existing benchmarks are insufficient for datasets with large scale variations and benchmarks with multiple datasets.

- They are too coarse for detailed scale-wise analysis.
  COCO adopts only three scale-wise metrics (small, medium, and large).
  If we use them on other datasets, they confuse objects of significantly different scales.
  For example, the Absolute Scale of a small object might be 1 or 30, and that of a large object might be 100 or 1600.
- They consider Absolute Scale only.
  To limit inference time and GPU memory consumption and for fair comparison, the input image scales are typically resized.
  If they are smaller than the original image scales, Relative Scale will have a greater effect on accuracy than Absolute Scale.
  When we use only COCO, there is no large difference in the evaluation based on either.
  However, there are large variations in image scales when we use multiple datasets.

To resolve these issues, we define two types of finer scale-wise AP metrics using exponential thresholds.

- Absolute Scale AP (ASAP) partitions object scales based on Absolute Scale (0, 8, 16, 32, ..., 1024, ∞).
- Relative Scale AP (RSAP) partitions object scales based on Relative Scale (0, 1/256, 1/128, 1/64, ..., 1/2, 1).

## Usage

Absolute Scale AP (ASAP)

```bash
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval bbox \
  --eval-options "area_range_type=absolute_scale_ap"
```

Relative Scale AP (RSAP)

```bash
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval bbox \
  --eval-options "area_range_type=relative_scale_ap"
```

## Citation

Please cite the following paper.
https://arxiv.org/abs/2103.14027

```
@inproceedings{USB_shinya_BMVC2022,
  title={{USB}: Universal-Scale Object Detection Benchmark},
  author={Shinya, Yosuke},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
```
