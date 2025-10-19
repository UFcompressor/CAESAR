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

git clone https://github.com/Shaw-git/CAESAR/tree/main
export CAESAR:$PYTHONPATH
cd CAESAR
pip install -r requirements.txt

cd ..

git clone https://github.com/E53klasky/CAESAR_C.git
cd CAECAR_C 
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

Optional (GPU Support)

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
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu

GPU Installation (NVIDIA only)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

sudo apt-get install libzstd-dev


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


build with adios2: cmake ..   -DCMAKE_INSTALL_PREFIX=~/Programs/CAESAR_C/install   -DTorch_DIR=/home/adios/.local/lib/python3.11/site-packages/torch/share/cmake/Torch

