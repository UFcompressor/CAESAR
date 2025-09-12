# CAESAR_C

This repository contains my work on converting [CAESAR](https://github.com/Shaw-git/CAESAR) into C++ code using **LibTorch**.  
The goal is to provide a C++ implementation of the CAESAR foundation model.  

---

## Build Instructions

```bash

mkdir build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH="/home/eklasky/Software/libtorch;/home/eklasky/local/nvcomp" \
  -DCMAKE_CXX_FLAGS="-I/home/eklasky/local/nvcomp/include"


make

../download_models.sh


---
Required Dependencies
1. LibTorch (PyTorch C++ API)

PyTorch official website

Choose the LibTorch version that matches your CUDA version (e.g., CUDA 12.0).

2. nvCOMP (NVIDIA Compression Library)

Download the CUDA 12 archive:

wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz


Extract to a local folder (e.g., ~/local/nvcomp):

mkdir -p ~/local/nvcomp
tar -xJf nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz -C ~/local/nvcomp --strip-components=1


Set environment variables:

export CMAKE_PREFIX_PATH=$HOME/local/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/nvcomp/lib:$LD_LIBRARY_PATH

4. CUDA Toolkit

Required for GPU support.

Ensure nvcc is in your PATH.

5. Zstandard (zstd)

For compression support. Can be installed locally if needed.






