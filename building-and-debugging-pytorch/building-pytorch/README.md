# Building PyTorch

## Dependencies

### PyTorch source code

```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
```

### Conda

We _strongly_ recommend using Conda to manage your PyTorch dependencies and Python environment. You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your operating system.&#x20;

You can either install the standard [Anaconda distribution](https://www.anaconda.com/products/individual), or if you want manual control over your dependencies you can install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### CUDA (optional)

If you want to compile with CUDA support, install:

* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 10.2 or above
* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 or above
* [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardwares

### ROCm (optional)

If you want to compile with ROCm support, install AMD ROCm 4.0 or above.&#x20;

* ROCm is currently supported only for Linux system.

### Conda setup

First, create a Conda environment and install the dependencies common to all operating systems. From the root of your PyTorch repo, run:

```
conda create --name pytorch-dev-env
conda activate pytorch-dev-env
conda install --file requirements.txt
```

\
