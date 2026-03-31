<div align="center">

# CAESAR — GPU Build Installation Guide

_Standalone installation instructions for the GPU build target_

</div>

---

## Before You Begin

Read [`README.md`](README.md) in this directory before proceeding. It documents the precision configuration and cross-platform compatibility rules for this build. Skipping it may result in data loss or silent corruption.

**Requirements:**

| Requirement       | Minimum Version         |
| ----------------- | ----------------------- |
| Python            | 3.10                    |
| CMake             | 3.10                    |
| Zstandard (zstd)  | 1.5                     |
| CUDA Toolkit      | 12.0                    |
| nvCOMP            | 5.0                     |
| NVIDIA GPU Driver | Compatible with CUDA 12 |

Verify your CUDA installation before proceeding:

```bash
nvcc --version
nvidia-smi
```

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/E53klasky/CAESAR_C.git
cd CAESAR_C
```

---

## Step 2 — Apply the GPU Patch

```bash
git apply GPU/gpu.patch
```

Verify the patch applied cleanly:

```bash
git status
# Expected: relevant source files listed as modified
```

---

## Step 3 — Create and Activate a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

---

## Step 4 — Install System Dependencies (Linux)

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ zstd libzstd-dev
```

---

## Step 5 — Install nvCOMP

```bash
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz

mkdir -p ~/local/nvcomp
tar -xJf nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz -C ~/local/nvcomp --strip-components=1
```

Set the required environment variables. Add these to your shell profile (`.bashrc` or `.zshrc`) to make them persistent:

```bash
export CMAKE_PREFIX_PATH=$HOME/local/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/nvcomp/lib:$LD_LIBRARY_PATH
```

---

## Step 6 — Install Python Dependencies

```bash
source venv/bin/activate

grep -v "^torch" requirements.txt | \
  grep -v "^torchvision" | \
  grep -v "^--extra-index-url" | \
  grep -v "^cupy" | \
  grep -v "^nvidia" | \
  grep -v "^$" > temp_requirements.txt

pip install --no-cache-dir -r temp_requirements.txt
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install compressai==1.2.6 imageio==2.37.0
rm temp_requirements.txt
```

---

## Step 7 — Download and Prepare Pretrained Models

```bash
chmod +x download_models.sh
./download_models.sh

python3 CAESAR_compressor.py cuda
python3 CAESAR_hyper_decompressor.py cuda
python3 CAESAR_decompressor.py cuda
```

---

## Step 8 — Configure and Build with CMake

```bash
mkdir -p build && cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH;$HOME/local/nvcomp" \
  -DCMAKE_CXX_FLAGS="-I$HOME/local/nvcomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/nvcomp/lib" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

For a debug build:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH;$HOME/local/nvcomp" \
  -DCMAKE_CXX_FLAGS="-I$HOME/local/nvcomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/nvcomp/lib" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build . --config Debug --parallel
```

---

## Step 9 — Verify the Build

```bash
cd build/tests
./test_caesarCD
./test_runGaeCuda
./test_padding
```

All three tests should complete without error.

---

## Reminder

Data compressed with this GPU build can be decompressed on any machine using the same float16 / double / double configuration. It cannot be decompressed with the CPU build. See [`README.md`](README.md) for the full explanation.
