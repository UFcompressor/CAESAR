<div align="center">

# CAESAR — CPU Build Installation Guide

_Standalone installation instructions for the CPU build target_

</div>

---

## Before You Begin

Read [`README.md`](README.md) in this directory before proceeding. It documents the precision configuration and cross-platform limitations of this build. Skipping it may result in data loss or silent corruption.

**Requirements:**

| Requirement      | Minimum Version    |
| ---------------- | ------------------ |
| Python           | 3.10               |
| CMake            | 3.10               |
| Zstandard (zstd) | 1.5                |
| Git              | Any recent version |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/E53klasky/CAESAR_C.git
cd CAESAR_C
```

---

## Step 2 — Apply the CPU Patch

```bash
git apply CPU/cpu_build.patch
```

This modifies the compressor to use float32 instead of float16 and updates the corresponding C++ precision flag. Verify the patch applied cleanly before continuing.

```bash
git status
# Expected: CAESAR/models/caesar_compress.cpp and CAESAR_compressor.py listed as modified
```

---

## Step 3 — Create and Activate a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

---

## Step 4 — Install System Dependencies

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ zstd libzstd-dev

source venv/bin/activate

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

### macOS

```bash
brew install cmake zstd gcc

source venv/bin/activate

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

---

## Step 5 — Download and Prepare Pretrained Models

```bash
chmod +x download_models.sh
./download_models.sh

python3 CAESAR_compressor.py cpu
python3 CAESAR_hyper_decompressor.py cpu
python3 CAESAR_decompressor.py cpu
```

---

## Step 6 — Configure and Build with CMake

```bash
mkdir -p build && cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

For a debug build:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build . --config Debug --parallel
```

---

## Step 7 — Verify the Build

```bash
cd build/tests
./test_caesarCD
./test_runGaeCuda
./test_padding
```

All three tests should complete without error.

---

## Reminder

Data compressed with this CPU build must be decompressed with this CPU build. See [`README.md`](README.md) for the full explanation of why mixing builds produces incorrect results.
