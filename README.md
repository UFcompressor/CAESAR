CAESAR_C

C++ implementation of CAESAR using LibTorch. The goal is to provide a C++ version of the CAESAR foundation model for efficient compression of scientific data.

Overview

CAESAR (Conditional AutoEncoder with Super-resolution for Augmented Reduction) is a unified framework for spatio-temporal scientific data reduction.

The baseline model, CAESAR-V, is built on a variational autoencoder (VAE) with scale hyperpriors and super-resolution modules to achieve high compression. It encodes data into a latent space and uses learned priors for compact, information-rich representation. This repository ports CAESAR into C++ with LibTorch for use in high-performance scientific applications.

Reference: Shaw et al., CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data

Notes

GPU support currently tested only with NVIDIA GPUs. MPI support is optional and not required for basic use. OpenMP is required. Model compression requires Python environment to be set up correctly.

Build Instructions

Clone the repository:

```bash
git clone https://github.com/E53klasky/CAESAR_C.git
cd CAESAR_C
```

Create and activate Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

Install dependencies based on your platform:

For Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ zstd libzstd-dev libomp-dev

grep -v "^torch" requirements.txt | \
  grep -v "^torchvision" | \
  grep -v "^--extra-index-url" | \
  grep -v "^cupy" | \
  grep -v "^nvidia" | \
  grep -v "^$" > temp_requirements.txt

pip install --no-cache-dir -r temp_requirements.txt
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install compressai==1.2.6 imageio==2.37.0
rm temp_requirements.txt
```

For macOS:

```bash
brew install cmake zstd gcc libomp

grep -v "^torch" requirements.txt | \
  grep -v "^torchvision" | \
  grep -v "^--extra-index-url" | \
  grep -v "^cupy" | \
  grep -v "^nvidia" | \
  grep -v "^$" > temp_requirements.txt

pip install -r temp_requirements.txt
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install compressai==1.2.6 imageio==2.37.0
rm temp_requirements.txt
```

Download and prepare pretrained models:

```bash
chmod +x download_models.sh
./download_models.sh

python3 CAESAR_compressor.py cpu
python3 CAESAR_hyper_decompressor.py cpu
python3 CAESAR_decompressor.py cpu
```

Configure and build with CMake:

```bash
mkdir -p build
cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

For macOS with OpenMP:

```bash
export OpenMP_ROOT=$(brew --prefix libomp)

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OpenMP_ROOT/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OpenMP_ROOT/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="$OpenMP_ROOT/lib/libomp.dylib"

cmake --build . --config Release --parallel
```

For Debug builds:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build . --config Debug --parallel
```

Dependencies

Core Dependencies

LibTorch (PyTorch C++ API)
CMake (3.10+)
Zstandard (zstd) - For compression support
OpenMP - Required for parallel processing
Python 3.10+ - For model preparation and preprocessing

GPU Support (Optional)

CUDA Toolkit - Ensure nvcc is in PATH
nvCOMP - NVIDIA Compression Library
MPI - Optional, for distributed runs

Installing nvCOMP

Download the CUDA 12 archive:

```bash
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz
```

Extract to a local folder:

```bash
mkdir -p ~/local/nvcomp
tar -xJf nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz -C ~/local/nvcomp --strip-components=1
```

Set environment variables:

```bash
export CMAKE_PREFIX_PATH=$HOME/local/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/nvcomp/lib:$LD_LIBRARY_PATH
```

Build with GPU support:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH;$HOME/local/nvcomp" \
  -DCMAKE_CXX_FLAGS="-I$HOME/local/nvcomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/nvcomp/lib" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
```

Install PyTorch with CUDA support:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Model Directory Configuration

CAESAR automatically finds model files in the following order:

1. Custom location (if set): `export CAESAR_MODEL_DIR=/path/to/your/models`
2. Development build: Automatically finds `../exported_model/` relative to executable
3. After installation: Automatically finds models at `/usr/local/share/caesar/models`

For most users, no configuration is needed. CAESAR will find the models automatically.

If you have multiple model versions or want to use models from a different location:

```bash
export CAESAR_MODEL_DIR=/path/to/custom/models
./your_program
```

Installation with ADIOS2

To build with ADIOS2 support:

```bash
cmake .. \
  -DCMAKE_INSTALL_PREFIX=~/Programs/CAESAR_C/install \
  -DTorch_DIR=/path/to/python/site-packages/torch/share/cmake/Torch
```

References

Original CAESAR repository: https://github.com/Shaw-git/CAESAR
NVIDIA nvCOMP: https://developer.nvidia.com/nvcomp
CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
PyTorch: https://pytorch.org/
Zstandard (zstd): https://facebook.github.io/zstd/
