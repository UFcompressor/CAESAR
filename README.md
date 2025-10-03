markdown# CAESAR_C

C++ implementation of [CAESAR](https://github.com/Shaw-git/CAESAR) using **LibTorch**.  
The goal is to provide a C++ version of the CAESAR foundation model for efficient compression of scientific data.

---

## Overview

**CAESAR (Conditional AutoEncoder with Super-resolution for Augmented Reduction)** is a unified framework for spatio-temporal scientific data reduction.

- The baseline model, **CAESAR-V**, is built on a **variational autoencoder (VAE)** with scale hyperpriors and super-resolution modules to achieve high compression.  
- It encodes data into a latent space and uses learned priors for compact, information-rich representation.  
- This repository ports CAESAR into **C++ with LibTorch** for use in high-performance scientific applications.  

**Reference:**  
[Shaw et al., CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data](https://github.com/Shaw-git/CAESAR)

---

## Build Instructions
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake \
  -DCMAKE_PREFIX_PATH="/blue/ranka/eklasky/caesar_venv/lib/python3.11/site-packages/torch/share/cmake" \
  -DCMAKE_CXX_FLAGS="-I$HOME/local/nvcomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/nvcomp/lib" \
  ..

# Compile with all available cores
cmake --build . --config Release -- -j$(nproc)

cd ..

# Download pretrained models
./download_models.sh

# Compress models
cd models/compress_models/
python3 compress_model.py

Dependencies
Required

LibTorch (PyTorch C++ API)
CMake (3.10+)
Zstandard (zstd) - For compression support

⚡ Optional (GPU Support)

CUDA Toolkit (ensure nvcc is in PATH)
nvCOMP (NVIDIA Compression Library)
MPI (optional, for distributed runs)


Installing nvCOMP
1. Download the CUDA 12 archive:
bashwget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz
2. Extract to a local folder (example: ~/local/nvcomp):
bashmkdir -p ~/local/nvcomp
tar -xJf nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz -C ~/local/nvcomp --strip-components=1
3. Set environment variables:
bashexport CMAKE_PREFIX_PATH=$HOME/local/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/nvcomp/lib:$LD_LIBRARY_PATH

Python Environment
This project relies on a Python environment for preprocessing and model compression.
CPU Installation
Tested with Python 3.10+:
bashpip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu

pip install numpy<2 scipy==1.15.3 tqdm==4.67.1 einops==0.8.1 pyyaml==6.0.2 \
  bsconv==0.4.0 einops-exts==0.0.4 rotary-embedding-torch==0.8.6 \
  compressai==1.2.6 imageio==2.37.0 zstandard==0.23.0
GPU Installation (NVIDIA only)
Tested with CUDA 11.8/12.x:
bashpip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install numpy<2 scipy==1.15.3 tqdm==4.67.1 einops==0.8.1 pyyaml==6.0.2 \
  bsconv==0.4.0 einops-exts==0.0.4 rotary-embedding-torch==0.8.6 \
  compressai==1.2.6 imageio==2.37.0 zstandard==0.23.0

pip install cupy-cuda11x==13.4.1 nvidia-nvcomp-cu11==4.2.0.11

References

Original CAESAR repository: https://github.com/Shaw-git/CAESAR
NVIDIA nvCOMP: https://developer.nvidia.com/nvcomp
CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
PyTorch: https://pytorch.org/
Zstandard (zstd): https://facebook.github.io/zstd/


Notes

GPU support currently tested only with NVIDIA GPUs
MPI support is optional and not required for basic use
Model compression requires Python environment to be set up correctly

RetryClaude does not have the ability to run the code it generates yet.
