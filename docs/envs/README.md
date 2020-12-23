## Testing environment

### UniverseNet 20.10 (Oct. 2020)

- Ubuntu 18.04
- Python 3.7
- PyTorch 1.6.0
- CUDA 10.1
- GCC 7.5
- mmcv-full 1.1.2

### UniverseNet 20.08 (Aug. 2020)

- Ubuntu 18.04
- Python 3.7
- PyTorch 1.5.0
- CUDA 10.1
- GCC 7.5
- mmcv-full 1.0.4

### UniverseNet 20.06 (June 2020)

- Ubuntu 18.04
- Python 3.7
- PyTorch 1.5.0
- CUDA 10.1
- GCC 7.5
- mmcv 0.5.5

### Details

More details can be found in the conda environment file in this directory.
Installation via `conda env create --file conda_env.yaml` is not supported.
Some modifications (e.g., CUDA version, cocoapi from github) will be needed to do that.

Versions used for measuring inference time may differ.
Some values are new and others are outdated.
