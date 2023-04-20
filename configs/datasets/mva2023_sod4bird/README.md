# MVA2023 SOD4Bird

> [MVA2023 Small Object Detection Challenge for Spotting Birds](https://www.mva-org.jp/mva2023/challenge)

<!-- [DATASET] -->

<!--
## Abstract

<div align=center>
<img src=""/>
</div>
-->

## Results and Models

| Method | Backbone | Lr schd | Scale | Flip | val AP | val AP50 | val AP50 (max 300) | public test AP50 |                               Config                                |                                                                      Download                                                                       |
| :----: | :------: | :-----: | :---: | :--: | :----: | :------: | :----------------: | :--------------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: |
|  GFL   |   R-50   |   7e    | 2160  |  N   | 0.465  |  0.886   |       0.890        |      0.724       |   [config](./gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160.py)    | [model](https://github.com/shinya7y/weights/releases/download/v1.0.3/gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_20230415_epoch_7-383174fb.pth) |
|  GFL   |   R-50   |   7e    | 2160  |  Y   | 0.470  |  0.889   |       0.896        |      0.731       | [config](./gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_test.py) | [model](https://github.com/shinya7y/weights/releases/download/v1.0.3/gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_20230415_epoch_7-383174fb.pth) |

- Scale: shorter side pixels.
- Flip: horizontal flip for test-time augmentation.
- max 300: `max_per_img=300`.
  The maximum number of detections per image for public test/private test evaluation is unknown.

## Usage

### Installation

See [get_started.md](../../../docs/en/get_started.md) to install Miniconda and CUDA toolkit.

```bash
# change channel setting of conda
conda config --remove channels defaults
conda config --add channels conda-forge
# create and activate conda environment
conda create --name univmva2023 python=3.9 -y
conda activate univmva2023
# install major requirements
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.4.4
# install UniverseNet
git clone https://github.com/shinya7y/UniverseNet.git
cd UniverseNet
pip install -v -e .
```

My testing environment is as follows.
Version details can be found in [this conda environment file](../../../docs/en/envs/conda_env_universenet_mva2023.yaml).

- GPU: NVIDIA GeForce RTX 3090
- Ubuntu 22.04 (on WSL2)
- Python 3.9
- PyTorch 1.9.0
- cudatoolkit 11.1
- mmcv-full 1.4.4
- nvcc 11.7
- GCC 11.3

If you have any problems with the above installation, `apt install` might be required beforehand.
For example, `sudo apt -y install build-essential libgl1-mesa-glx` and other libraries written in [Dockerfile](../../../docker/Dockerfile).

To resolve installation problems, please see the documents and issues of MMDetection and MMCV.
If problems persist, please contact shinya7y via the challenge's Discord, this repository's issue, or email.

### Preparation

Based on your data directory and evaluation dataset (val / public test / private test),
please edit `data_root`, `data.test.ann_file`, and `data.test.img_prefix` in the configs in this directory.

### Testing

```bash
# download trained model
mkdir -p ${HOME}/data/checkpoints/
wget -P ${HOME}/data/checkpoints/ https://github.com/shinya7y/weights/releases/download/v1.0.3/gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_20230415_epoch_7-383174fb.pth

CONFIG_FILE=configs/datasets/mva2023_sod4bird/gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_test.py
CHECKPOINT_FILE=${HOME}/data/checkpoints/gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_20230415_epoch_7-383174fb.pth
# for evaluation
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval bbox
# for submission
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --eval-options jsonfile_prefix=results
```

To help reproduction with other repositories that are based on mmdet (around 2.25.0), I provide [another config](./gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_test_whole_config.py).
Note that this config is only for reproduction of testing.

### Training

```bash
CONFIG_FILE=configs/datasets/mva2023_sod4bird/gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160.py
CONFIG_NAME=$(basename ${CONFIG_FILE} .py)
WORK_DIR="${HOME}/logs/mva2023_sod4bird/${CONFIG_NAME}_`date +%Y%m%d_%H%M%S`"
python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --seed 0
```

This training takes 5 hours with one RTX 3090.

### Used pre-trained model

I used [a COCO pre-trained model](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth),
which is publicly available from [mmdetection/configs/gfl](https://github.com/open-mmlab/mmdetection/tree/v2.25.0/configs/gfl).

## Citation

```latex
@misc{sodbchallenge2023misc,
  title={{MVA2023 Small Object Detection Challenge for Spotting Birds}},
  author={Yuki Kondo and Norimichi Ukita and Takayuki Yamaguchi},
  howpublished={\url{https://www.mva-org.jp/mva2023/challenge}},
  year={2023}}
```
