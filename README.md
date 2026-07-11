<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=800&size=64&duration=1&pause=1000&color=E8500A&center=true&vCenter=true&repeat=false&width=900&height=120&lines=CAESAR"/>
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=800&size=64&duration=1&pause=1000&color=E8500A&center=true&vCenter=true&repeat=false&width=900&height=120&lines=CAESAR" alt="CAESAR"/>
</picture>

**Conditional AutoEncoder with Super-resolution for Augmented Reduction**

_A C++ / LibTorch foundation model for efficient compression of scientific data_

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-1B4FA8)
![C++](https://img.shields.io/badge/C++-17-E8500A)
![LibTorch](https://img.shields.io/badge/LibTorch-2.8%2B-1B4FA8)
![zstd](https://img.shields.io/badge/zstd-1.5%2B-E8500A)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%20%7C%20Apple%20Silicon-1B4FA8)
[![License](https://img.shields.io/badge/license-Apache--2.0-1B4FA8)](https://github.com/UFcomrpessor/CAESAR/tree/master#Apache-2.0-1-ov-file)

</div>

---

## Overview

CAESAR is a unified framework for spatio-temporal scientific data reduction. The baseline model, **CAESAR-V**, is built on a variational autoencoder (VAE) with scale hyperpriors and super-resolution modules to achieve high compression ratios while preserving scientific fidelity.

It encodes data into a compact latent space and uses learned priors for information-rich representation. This repository ports CAESAR into **C++ with LibTorch** for deployment in high-performance computing (HPC) environments and scientific workflows.

CAESAR runs on CPU by default, and supports GPU acceleration on both NVIDIA (CUDA) and Apple Silicon (Metal / MPS) platforms.

> **Reference:** Shaw et al., _CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data_

---

## Build Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/UFcompressor/CAESAR
cd CAESAR
```

### 2. Create and Activate a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### 3. Install Platform Dependencies

<details>
<summary>Linux (Ubuntu/Debian)</summary>

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
pip install compressai==1.2.6
rm temp_requirements.txt
```

</details>

<details>
<summary>macOS</summary>

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
pip install compressai==1.2.6 
rm temp_requirements.txt
```

> On Apple Silicon (M1/M2/M3/M4), the standard CPU wheel above also enables the Metal Performance Shaders (MPS) backend for CPU-only use, but MPS acceleration for this project has only been verified on the nightly build. To enable MPS, install the nightly build instead of the pinned CPU wheel above:
>
> ```bash
> pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
> ```
>
> This project has been verified on Apple Silicon (M1) with at least the following versions:
>
> | Package     | Minimum Version Verified |
> | ----------- | ------------------------- |
> | torch       | 2.14.0.dev20260702         |
> | torchvision | 0.29.0.dev20260702         |
> | torchaudio  | 2.11.0                     |
>
> Check your installed versions with:
>
> ```bash
> python3 -c "import torch, torchvision, torchaudio; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('torchaudio:', torchaudio.__version__)"
> ```
>
> See [GPU Support (Apple Silicon)](#gpu-support-apple-silicon) below to build with MPS acceleration enabled.

</details>

<details>
<summary>Windows</summary>

```powershell
# Install CMake, zstd, and a recent MSVC toolchain (Visual Studio Build Tools) first

venv\Scripts\activate

findstr /v /b "torch torchvision --extra-index-url cupy nvidia" requirements.txt > temp_requirements.txt

pip install --no-cache-dir -r temp_requirements.txt
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install compressai==1.2.6 
del temp_requirements.txt
```

</details>

### 4. Download and Prepare Pretrained Models

```bash
./download_models.sh

python3 CAESAR_compressor.py cpu
python3 CAESAR_hyper_decompressor.py cpu
python3 CAESAR_decompressor.py cpu
```

### 5. Configure and Build with CMake

```bash
mkdir -p build && cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j6
```

For debug builds, replace `-DCMAKE_BUILD_TYPE=Release` with `-DCMAKE_BUILD_TYPE=Debug`.

---

## GPU Support (NVIDIA)

GPU support requires CUDA and nvCOMP.

<details>
<summary>Install nvCOMP</summary>

```bash
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz

mkdir -p ~/local/nvcomp
tar -xJf nvcomp-linux-x86_64-5.0.0.6_cuda12-archive.tar.xz -C ~/local/nvcomp --strip-components=1

export CMAKE_PREFIX_PATH=$HOME/local/nvcomp:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/nvcomp/lib:$LD_LIBRARY_PATH
```

</details>

<details>
<summary>Build with GPU support</summary>

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH;$HOME/local/nvcomp" \
  -DCMAKE_CXX_FLAGS="-I$HOME/local/nvcomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/nvcomp/lib" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
```

</details>

---

## GPU Support (Apple Silicon)

CAESAR supports GPU acceleration on Apple Silicon (M1/M2/M3/M4) through PyTorch's Metal Performance Shaders (MPS) backend. No additional compression library equivalent to nvCOMP is required for this path.

<details>
<summary>Requirements</summary>

- Apple Silicon Mac (M1 or newer)
- LibTorch build with MPS support (the standard macOS LibTorch distribution includes this)
- torch >= 2.14.0.dev20260702, torchvision >= 0.29.0.dev20260702, torchaudio >= 2.11.0 (nightly build; see installation note above)

</details>

<details>
<summary>Build with MPS support</summary>

```bash
source venv/bin/activate

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j6
```

Verify that MPS is available before running:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

At runtime, select the MPS device the same way you would select `cuda` on NVIDIA systems (refer to the relevant CLI flag or configuration option for device selection).

</details>

---

## Environment Variables

CAESAR resolves model files in the following priority order:

| Priority | Location                                                             |
| -------- | -------------------------------------------------------------------- |
| 1        | `$CAESAR_MODEL_DIR` environment variable (if set)                    |
| 2        | `../exported_model/` relative to the executable (development builds) |
| 3        | `/usr/local/share/caesar/models` (installed builds)                  |

```bash
export CAESAR_MODEL_DIR=/path/to/your/models
```

## Dependencies

### Core

| Dependency                 | Minimum Version |
| -------------------------- | --------------- |
| LibTorch (PyTorch C++ API) | 2.8             |
| CMake                      | 3.10            |
| Zstandard (zstd)           | 1.5 (required)  |
| Python                     | 3.10            |

GPU-specific dependencies (CUDA, nvCOMP, MPS-compatible torch builds) are covered in the respective [GPU Support](#gpu-support-nvidia) sections above.

---

## Citation

If you use CAESAR in your research, please cite the following works:

```bibtex
@inproceedings{li2025foundation,
  title        = {Foundation Model for Lossy Compression of Spatiotemporal Scientific Data},
  author       = {Li, Xiao and Lee, Jaemoon and Rangarajan, Anand and Ranka, Sanjay},
  booktitle    = {Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages        = {368--380},
  year         = {2025},
  organization = {Springer}
}
```

```bibtex
@article{li2025generative,
  title   = {Generative Latent Diffusion for Efficient Spatiotemporal Data Reduction},
  author  = {Li, Xiao and Zhu, Liangji and Rangarajan, Anand and Ranka, Sanjay},
  journal = {arXiv preprint arXiv:2507.02129},
  year    = {2025}
}
```

---

## Contact

For questions, bug reports, or contributions, please open an issue on [GitHub](https://github.com/UFcomrpessor/CAESAR/issues).

---

## References

| Resource                 | Link                                                                           |
| ------------------------ | ------------------------------------------------------------------------------ |
| Original CAESAR (Python) | [Shaw-git/CAESAR](https://github.com/Shaw-git/CAESAR)                          |
| NVIDIA nvCOMP            | [developer.nvidia.com/nvcomp](https://developer.nvidia.com/nvcomp)             |
| CUDA Toolkit             | [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) |
| PyTorch                  | [pytorch.org](https://pytorch.org)                                             |
| Zstandard                | [facebook.github.io/zstd](https://facebook.github.io/zstd)                     |
| CompressAI               | [InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI)    |

---
<div align="center">
<sub>Built for science. Engineered for performance.</sub>
</div>