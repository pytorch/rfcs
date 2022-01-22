---
description: This page gets you set up to do a build of PyTorch from multiple platforms.
---

# Building PyTorch

## Prerequisites

### PyTorch source code

```bash
git clone --recursive git@github.com:pytorch/pytorch.git
cd pytorch

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive --jobs 0
```

### Conda

We _strongly_ recommend using Conda to manage your PyTorch dependencies and Python environment. You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your operating system.

You can either install the standard [Anaconda distribution](https://www.anaconda.com/products/individual), or if you want manual control over your dependencies you can install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### CUDA (optional)

If you want to compile with CUDA support, install:

* [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 10.2 or above
* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 or above
* [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardwares

### ROCm (optional)

If you want to compile with ROCm support, install AMD ROCm 4.0 or above.

* ROCm is currently supported only for Linux system.

### Conda setup

First, create a Conda environment and install the dependencies common to all operating systems. From the root of your PyTorch repo, run:

```
conda create --name pytorch-dev-env
conda activate pytorch-dev-env
conda install --file requirements.txt
```

Then, install OS-specific dependencies.

#### Linux

```bash
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```

#### MacOS

```bash
# Add these packages if torch.distributed is needed
conda install pkg-config libuv
```

#### Windows

```bash
# Add these packages if torch.distributed is needed.
# Distributed package support on Windows is a prototype feature and is subject to changes.
conda install -c conda-forge libuv=1.39
```

### Building PyTorch

#### Linux

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

If you are compiling for ROCm, you must run this command first:

```bash
python tools/amd_build/build_amd.py
```

#### MacOS

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

CUDA is not supported on macOS.

#### Windows

Choose Correct Visual Studio Version.

Sometimes there are regressions in new versions of Visual Studio, so it's best to use the same Visual Studio Version [16.8.5](https://github.com/pytorch/pytorch/blob/master/.circleci/scripts/vs\_install.ps1) as Pytorch CI's.

PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise, Professional, or Community Editions. You can also install the build tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools _do not_ come with Visual Studio Code by default.

If you want to build legacy python code, please refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

Build with CPU

It's fairly easy to build with CPU.

```
conda activate
python setup.py install
```

Note on OpenMP: The desired OpenMP implementation is Intel OpenMP (iomp). In order to link against iomp, you'll need to manually download the library and set up the building environment by tweaking `CMAKE_INCLUDE_PATH` and `LIB`. The instruction [here](https://github.com/pytorch/pytorch/blob/master/docs/source/notes/windows.rst#building-from-source) is an example for setting up both MKL and Intel OpenMP. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.

Build with CUDA

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia\_tools\_extension\_library\_nvtx.htm) is needed to build Pytorch with CUDA. NVTX is a part of CUDA distributive, where it is called "Nsight Compute". To install it onto already installed CUDA run CUDA installation once again and check the corresponding checkbox. Make sure that CUDA with Nsight Compute is installed after Visual Studio.

Currently, VS 2017 / 2019, and Ninja are supported as the generator of CMake. If `ninja.exe` is detected in `PATH`, then Ninja will be used as the default generator, otherwise, it will use VS 2017 / 2019.\
If Ninja is selected as the generator, the latest MSVC will get selected as the underlying toolchain.

Additional libraries such as [Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) are often needed. Please refer to the [installation-helper](https://github.com/pytorch/pytorch/tree/master/.jenkins/pytorch/win-test-helpers/installation-helpers) to install them.

You can refer to the [build\_pytorch.bat](https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/win-test-helpers/build\_pytorch.bat) script for some other environment variables configurations

```
cmd
:: Set the environment variables after you have downloaded and upzipped the mkl package,
:: else CMake would throw error as `Could NOT find OpenMP`.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%
:: Read the content in the previous section carefully before you proceed.
:: [Optional] If you want to override the underlying toolset used by Ninja and Visual Studio with CUDA, please run the following script block.
:: "Visual Studio 2019 Developer Command Prompt" will be run automatically.
:: Make sure you have CMake >= 3.12 before you do this when you use the Visual Studio generator.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
:: [Optional] If you want to override the CUDA host compiler
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe
python setup.py install
```
