# CAESAR_C

This repository contains my work on converting [CAESAR](https://github.com/Shaw-git/CAESAR) into C++ code using **LibTorch**.  
The goal is to provide a C++ implementation of the CAESAR foundation model.  

---

## Build Instructions

```bash

mkdir build && cd build

  cmake   -DCMAKE_PREFIX_PATH="/blue/ranka/eklasky/caesar_venv/lib/python3.11/site-packages/torch/share/cmake"   -DCMAKE_CXX_FLAGS="-I/home/eklasky/local/nvcomp/include"   -DCMAKE_EXE_LINKER_FLAGS="-L/home/eklasky/local/nvcomp/lib"   ..

cmake --build . --config Release -- -j$(nproc)

cd ..

./download_models.sh

cd models/compress_models/

python3 compress_model.py

---
Required Dependencies
1. PyTorch

PyTorch official website

Use pytorch 2.8+
INVIDA GPUU
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
CPU
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

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

link to install https://developer.nvidia.com/cuda-12-9-0-download-archive?target_os=Linux






