<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=800&size=64&duration=1&pause=1000&color=E8500A&center=true&vCenter=true&repeat=false&width=900&height=120&lines=CAESAR"/>
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=800&size=64&duration=1&pause=1000&color=E8500A&center=true&vCenter=true&repeat=false&width=900&height=120&lines=CAESAR" alt="CAESAR"/>
</picture>

**Conditional AutoEncoder with Super-resolution for Augmented Reduction**

_A C++ / LibTorch foundation model for efficient compression of scientific data_

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-1B4FA8)
![C++](https://img.shields.io/badge/C++-17-E8500A)
![LibTorch](https://img.shields.io/badge/LibTorch-2.8%2B-1B4FA8)
![zstd](https://img.shields.io/badge/zstd-1.5%2B-E8500A)
![License](https://img.shields.io/badge/license-MIT-1B4FA8)

</div>

---

## Overview

CAESAR is a unified framework for spatio-temporal scientific data reduction. The baseline model, **CAESAR-V**, is built on a variational autoencoder (VAE) with scale hyperpriors and super-resolution modules to achieve high compression ratios while preserving scientific fidelity.

It encodes data into a compact latent space and uses learned priors for information-rich representation. This repository ports CAESAR into **C++ with LibTorch** for deployment in high-performance computing (HPC) environments and scientific workflows.

> **Reference:** Shaw et al., _CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data_

---

## Build Targets: CPU vs. GPU

Before following the build instructions, apply the patch for your target platform. Each build is an **optimized configuration** — please read the assumptions documented in each directory before proceeding.

### CPU Build

```bash
git apply CPU/cpu_build.patch
```

### GPU Build (NVIDIA only)

```bash
git apply GPU/gpu.patch
```

Each folder contains two documents:

| Document     | Purpose                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------- |
| `README.md`  | Architecture assumptions, known limitations, and important behavioral notes for that build target |
| `INSTALL.md` | Step-by-step installation guide specific to that build target                                     |

**Read both before building.** The CPU and GPU builds make different assumptions about floating-point precision, memory layout, and hardware availability. Mixing configurations — for example, compressing with a GPU build and decompressing with a CPU build — will produce incorrect results.

---

## Build Instructions

### 1. Clone the Repository

```bash
git https://github.com/UFcomrpessor/CAESAR.git
cd CAESAR
```

### 2. Apply Your Build Patch

See [Build Targets: CPU vs. GPU](#build-targets-cpu-vs-gpu) above, and read the corresponding `README.md` and `INSTALL.md` in the `CPU/` or `GPU/` directory before continuing.

### 3. Create and Activate a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### 4. Install Platform Dependencies

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
pip install compressai==1.2.6 imageio==2.37.0
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
pip install compressai==1.2.6 imageio==2.37.0
rm temp_requirements.txt
```

</details>

### 5. Download and Prepare Pretrained Models

```bash
chmod +x download_models.sh
./download_models.sh

python3 CAESAR_compressor.py cpu
python3 CAESAR_hyper_decompressor.py cpu
python3 CAESAR_decompressor.py cpu
```

### 6. Configure and Build with CMake

```bash
mkdir -p build && cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel
```

For debug builds, replace `-DCMAKE_BUILD_TYPE=Release` with `-DCMAKE_BUILD_TYPE=Debug`.

---

## GPU Support (NVIDIA)

> See [`GPU/README.md`](GPU/README.md) and [`GPU/INSTALL.md`](GPU/INSTALL.md) for the full GPU setup guide and build assumptions.

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

The following environment variables can be set to tune runtime behavior:

| Variable                         | Description                                                                                                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CAESAR_GAE_ZSTD_LEVEL`          | Sets the Zstandard compression level for the GAE stage. Accepts an integer from **1–19**. Higher levels produce better compression ratios but run more slowly.    |
| `CAESAR_DO_NOT_EMPTY_CUDA_CACHE` | When set, CAESAR skips clearing the CUDA cache between operations. This can provide a minor speedup but increases memory pressure and may cause issues on larger datasets. |

```bash
export CAESAR_GAE_ZSTD_LEVEL=9
export CAESAR_DO_NOT_EMPTY_CUDA_CACHE=1
```

---

## Dependencies

### Core

| Dependency                 | Minimum Version |
| -------------------------- | --------------- |
| LibTorch (PyTorch C++ API) | 2.8             |
| CMake                      | 3.10            |
| Zstandard (zstd)           | 1.5 (required)  |
| Python                     | 3.10            |

### Optional — GPU

| Dependency   | Notes                      |
| ------------ | -------------------------- |
| CUDA Toolkit | Ensure `nvcc` is in `PATH` |
| nvCOMP       | NVIDIA compression library |

---

## Platform Support

| Platform              |           CPU           |      GPU      |
| --------------------- | :---------------------: | :-----------: |
| Linux (Ubuntu/Debian) | Supported — recommended |  NVIDIA only  |
| macOS                 |        Supported        | Not supported |
| Windows               |      Not supported      | Not supported |

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

For questions, bug reports, or contributions, please open an issue on [GitHub](https://github.com/E53klasky/CAESAR_C/issues).

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
