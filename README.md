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
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%20%7C%20AMD%20ROCm%20%7C%20Apple%20Silicon-1B4FA8)
[![License](https://img.shields.io/badge/license-Apache--2.0-1B4FA8)](https://github.com/UFcomrpessor/CAESAR/tree/master#Apache-2.0-1-ov-file)

</div>

---

## Overview

CAESAR is a unified framework for spatio-temporal scientific data reduction. The baseline model, **CAESAR-V**, is built on a variational autoencoder (VAE) with scale hyperpriors and super-resolution modules to achieve high compression ratios while preserving scientific fidelity.

Developer notes are in [Things to know about the codebase](docs/codebase.md).
The same Markdown is the source for a Read the Docs site once this repository
is connected to a Read the Docs project.

It encodes data into a compact latent space and uses learned priors for information-rich representation. This repository ports CAESAR into **C++ with LibTorch** for deployment in high-performance computing (HPC) environments and scientific workflows.

CAESAR runs on CPU by default, and supports GPU acceleration on both NVIDIA (CUDA) and Apple Silicon (Metal / MPS) platforms.

> **Reference:** Shaw et al., _CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data_

---

## Build Instructions

CAESAR requires a C++17 compiler. When using GCC, use GCC 9 or newer;
GCC versions older than 9 (including 7.5) are unsupported.

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

The Linux, macOS, and Windows instructions below all install
`PyYAML>=6.0` from `requirements.txt`. PyYAML provides the Python `yaml` module
used by the CAESAR model and training configuration code.

<details>
<summary>Linux (Ubuntu/Debian)</summary>

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ zstd libzstd-dev

# Verify that the installed GCC version is 9 or newer.
g++ --version

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
python3 model_registry.py --list
python3 model_registry.py caesar_v2
python3 compile_model.py cpu
```

CAESAR v1 is the original foundation model (`caesar_v.pt`); CAESAR v2 is
its newer optimized foundation model (`model_bs64_ep100k.pt`) and the default.
Checkpoints must be registered in the UFL catalog and match their registered
SHA-256 before compilation. See [model installation and identity](docs/models.md)
for offline downloads, registration, devices, caching, and the ADIOS contract.

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

## Quick start: compress and reconstruct

After building, run from the repository root with a matching compiled model:

```bash
export CAESAR_MODEL_DIR="$PWD/exported_model"
# input.bin contains contiguous float32 values in the specified axis order.
build/CAESAR/caesar compress input.bin --shape 1,1,20,256,256 \
  --n-frame 8 --error-bound 0.001 --correction gae --output sample.cae
build/CAESAR/caesar decompress sample.cae --output restored.bin \
  --verify --original input.bin
```

The CLI writes `sample.cae.latents`, `sample.cae.hyper`, and `sample.cae.meta`;
keep all three. `restored.bin` contains raw float32 values. The C++ `caesar::save`
API instead writes one complete versioned file; these formats are different.

For in-memory use, no intermediate files are required:

```cpp
#include <caesar/models/caesar_compress.h>
#include <caesar/models/caesar_decompress.h>

// field is a nonempty floating-point torch::Tensor in [T,H,W] order.
CompressionConfig config;
config.memory_data = field;
config.n_frame = 8; // Required; current compiled architecture supports only 8.
Compressor compressor;
CompressionResult result = compressor.compress(config, 0.001f);
Decompressor decompressor;
torch::Tensor restored = decompressor.decompress(result);
auto cpu_values = restored.cpu().contiguous();
```

`CompressionResult` owns binary latent streams, normalization and shape metadata,
and correction payloads. See the [complete field/type reference](docs/public_api.md#compressionresult-field-reference)
for every member and nested structure, plus buffer serialization for applications
that transport results without writing files. Binary strings can contain zero
bytes: use `.data()` and `.size()`, never `strlen()`.

### Input and runtime constraints

- Accepted shapes are `[T,H,W]`, `[S,T,H,W]`, and `[V,S,T,H,W]`
  (variable, section, time, height, width). **5D input encodes variable 0 only**
  and returns `[1,S,T,H,W]`; compress other variables separately.
- Padding, blocking, and shape restoration are internal. Output is float32 on
  the installed model's device; CPU float64 input is converted internally.
- GAE is the default correction; LBRC and NGLR are alternatives. The relative
  error bound is a target: validate reconstructed error for your data. See
  [API constraints](docs/public_api.md) and [known limitations](docs/TODO.txt).
- Decoding requires the matching model installation, including its probability
  tables. Results record checkpoint identity but do not embed model packages.
  GPU-to-CPU entropy-decoding portability is still pending.
- Set `CAESAR_MODEL_DIR` before first use. Models and tables are cached per thread;
  reuse objects on their creating thread, and restart to switch installations.
  See [cache behavior and clearing](docs/models.md#runtime-cache-lifetime-and-clearing).

For a runnable C++ example, configure with `-DBUILD_EXAMPLES=ON`, build, and run
`build/examples/hello_caesar gae`. See [examples](examples/README.md),
[model installation](docs/models.md), and the [documentation index](docs/index.md).

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

TORCH_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
GPU_ARCH=$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}{b}")')

# Run from the repository root; target the current NVIDIA GPU architecture.
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH;$HOME/local/nvcomp" \
  -DCMAKE_CXX_FLAGS="-I$HOME/local/nvcomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/local/nvcomp/lib" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j6
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
| GCC (when using GCC)       | 9 (C++17)       |
| LibTorch (PyTorch C++ API) | 2.8             |
| CMake                      | 3.10            |
| Zstandard (zstd)           | 1.5 (required)  |
| Python                     | 3.10            |
| PyYAML                     | 6.0             |

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
