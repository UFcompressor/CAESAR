<div align="center">

<img src="docs/caesar_title.svg" width="900" height="120" alt="CAESAR"/>

**Conditional AutoEncoder with Super-resolution for Augmented Reduction**

_A C++ / LibTorch foundation model for efficient compression of scientific data_

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-1B4FA8)
![C++](https://img.shields.io/badge/C++-17-E8500A)
![LibTorch](https://img.shields.io/badge/LibTorch-2.8%2B-1B4FA8)
![zstd](https://img.shields.io/badge/zstd-1.5%2B-E8500A)
[![GPU validation](https://img.shields.io/badge/GPU-validation%20varies%20by%20backend-1B4FA8)](docs/platform_support.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-1B4FA8)](LICENSE)

</div>

---

## Overview

CAESAR is a unified framework for spatio-temporal scientific data reduction. The baseline model, **CAESAR-V**, is built on a variational autoencoder (VAE) with scale hyperpriors and super-resolution modules to achieve high compression ratios while preserving scientific fidelity.

Developer notes are in [Things to know about the codebase](docs/codebase.md).
The same Markdown is the source for a Read the Docs site once this repository
is connected to a Read the Docs project.

It encodes data into a compact latent space and uses learned priors for information-rich representation. This repository ports CAESAR into **C++ with LibTorch** for deployment in high-performance computing (HPC) environments and scientific workflows.

GAE and LBRC are extensively tested on Linux CPU and Linux NVIDIA CUDA,
which is the primary GPU deployment platform. CPU builds and GAE/LBRC tests
are included in Linux, macOS, and Windows CI. NGLR has more limited validation.
AMD ROCm has preliminary GAE validation; **LBRC testing is in progress**.
Apple MPS is experimental, Intel XPU is unverified, and Windows GPU execution
has not been verified. See the [platform compatibility matrix](docs/platform_support.md)
for method-specific confidence, CI scope, and known limitations.

Model training has been tested on **NVIDIA GPUs only**, as reported by the
maintainer. Training has not been tested on CPU, AMD GPUs, Apple MPS, Intel XPU,
or any other backend. Training validation is separate from the compression and
decompression testing summarized above.

> **Reference:** Shaw et al., _CAESAR: A Unified Framework of Foundation and Generative Models for Efficient Compression of Scientific Data_

---

## Build Instructions

CAESAR requires a C++17 compiler. When using GCC, use GCC 9 or newer;
GCC versions older than 9 (including 7.5) are unsupported.

### Optional: Set Up GitHub CLI (`gh`)

GitHub CLI is useful for authenticated Git operations and pull requests; it is
optional for cloning this public repository. Install Git as well if needed.

**With Conda**, activate the environment where you want `gh` available, then run:

```bash
conda install --channel conda-forge gh git
```

**Without Conda**, use the command for your platform:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y gh git

# macOS (Homebrew)
brew install gh git
```

```powershell
# Windows (WinGet); reopen your terminal after installation.
winget install --id GitHub.cli --source winget
winget install --id Git.Git --source winget
```

**On HPC systems without sudo or Conda**, follow the
[local binary installation instructions](docs/model_registration.md#install-on-linux-hpc-systems-without-sudo-or-conda)
to download an official release into `~/.local/opt`, symlink `gh` into
`~/.local/bin`, and configure PATH.

Then authenticate with GitHub and configure Git to use `gh` for HTTPS credentials:

```bash
gh --version
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

Follow the browser prompts to finish signing in. On a remote machine, open the
shown URL on your local computer and enter the displayed one-time code. With a
Conda installation, keep that environment active when using `gh` or Git's `gh`
credential helper.

See the [official installation instructions](https://github.com/cli/cli#installation)
for other platforms or newer packages, and the
[authentication manual](https://cli.github.com/manual/gh_auth_login) for details.

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

### 3. Install Dependencies and Compile a Model

#### Install Platform Dependencies

The Linux, macOS, and Windows instructions below all install
`PyYAML>=6.0` from `requirements.txt`. PyYAML provides the Python `yaml` module
used by the CAESAR model and training configuration code.

<details>
<summary>Linux (Ubuntu/Debian)</summary>

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ zstd libzstd-dev libomp-dev

# Verify that the installed GCC version is 9 or newer.
g++ --version

source venv/bin/activate

grep -v "^torch" requirements.txt | \
  grep -v "^torchvision" | \
  grep -v "^--extra-index-url" | \
  grep -v "^cupy" | \
  grep -v "^nvidia" | \
  grep -v "^compressai" | \
  grep -v "^$" > temp_requirements.txt

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r temp_requirements.txt
pip install "pybind11>=2.6.0"
pip install --no-build-isolation compressai==1.2.6
rm temp_requirements.txt
```

</details>

<details>
<summary>macOS</summary>

```bash
brew install cmake zstd gcc libomp

source venv/bin/activate

grep -v "^torch" requirements.txt | \
  grep -v "^torchvision" | \
  grep -v "^--extra-index-url" | \
  grep -v "^cupy" | \
  grep -v "^nvidia" | \
  grep -v "^compressai" | \
  grep -v "^$" > temp_requirements.txt

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r temp_requirements.txt
pip install "pybind11>=2.6.0"
CC=clang CXX=clang++ pip install --no-build-isolation compressai==1.2.6
rm temp_requirements.txt
```

> Apple MPS is experimental. GAE has worked for selected, less stringent
> error targets; stringent targets, LBRC, and NGLR remain insufficiently validated.
> Compatibility depends on the PyTorch version, AOTI export, and available MPS
> operations. Earlier instructions recorded an M1 trial with torch
> `2.14.0.dev20260702`, torchvision `0.29.0.dev20260702`, and torchaudio `2.11.0`;
> these are trial versions, not minimum compatibility guarantees. CPU CI uses
> torch `2.8.0` and does not establish MPS reliability. See
> [Apple GPU guidance](#gpu-support-apple-silicon).

</details>

<details>
<summary>Windows</summary>

Windows uses a separate **Miniconda / Python 3.10 / MSVC / CPU PyTorch**
procedure. Follow the [Windows CI installation procedure](docs/platform_support.md#windows-cpu-installation)
for the exact dependency filtering, CompressAI wheel workarounds, MSVC model
export, and NMake build. The POSIX virtual-environment and `make` commands on
this page are for Linux/macOS. Windows GPU execution is not verified.

</details>

#### Download and Compile a Pretrained Model

After installing the dependencies, run these commands from the repository root.
Compile a model before configuring and building CAESAR with CMake.

```bash
# Aligned available-model table, descriptions, and checkpoint identities
python3 model_registry.py --list
python3 model_registry.py caesar_v2
python3 compile_model.py cpu
```

CAESAR v1 is the original foundation model (`caesar_v.pt`); CAESAR v2 is
its newer optimized foundation model (`model_bs64_ep100k.pt`) and the default.
Checkpoints must be registered in the UFL catalog and match their registered
SHA-256 before compilation. See [model installation and identity](docs/models.md)
for offline downloads, registration, devices, caching, and the ADIOS contract.

### 4. Configure and Build with CMake

```bash
mkdir -p build && cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake .. \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j6
cd ..
```

On macOS, use the explicit Homebrew OpenMP CMake flags in the
[cross-platform workflow](.github/workflows/cross-platform-test.yml); the CI
also sets `DYLD_LIBRARY_PATH` to the Homebrew libomp directory and
`OMP_NUM_THREADS=1` when running tests.

For debug builds, replace `-DCMAKE_BUILD_TYPE=Release` with `-DCMAKE_BUILD_TYPE=Debug`.

After the build completes, inspect the executable's selected model, compiled
backend, and recorded compilation CPU/GPU name from the repository root:

```sh
./build/CAESAR/caesar --model-info
```

See [model inspection and output examples](docs/models.md#inspect-the-selected-model).

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

Linux NVIDIA CUDA is the primary, extensively tested GPU configuration for GAE
and LBRC; NGLR has limited validation. Export the model with
`python3 compile_model.py cuda` before configuring the build.

NVIDIA GPU builds use nvCOMP **5.0.0.6**. CMake reuses an exact matching SDK or downloads the checksum-verified NVIDIA CUDA 12 binary SDK into the build directory. Automatic downloads support Linux x86_64 with CUDA 12; CPU and ROCm builds do not download nvCOMP. Set `-DCAESAR_DOWNLOAD_NVCOMP=OFF` to require a local SDK (provided through `CMAKE_PREFIX_PATH`). nvCOMP is distributed prebuilt, so no source compilation is needed.

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
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j6
```

Quick nvCOMP verification (requires CUDA LibTorch, the CUDA 12 toolkit, and an NVIDIA GPU; the normal configure also requires the registered model installation described above):

```bash
# Configure with BUILD_TESTS=ON as above. Look for "Using nvCOMP 5.0.0.6".
cmake --build build --target test_nvcomp -j2
./build/tests/test_nvcomp
```

The test requires no model inference and checks exact byte recovery for single-chunk and multi-chunk Zstd GPU compression. It exits nonzero on failure. For offline builds, keep the local SDK and pass its prefix with `-DCMAKE_PREFIX_PATH="$TORCH_PATH;$HOME/local/nvcomp"`.

</details>

---

## GPU Support (Apple Silicon)

Apple Silicon MPS is an experimental accelerator path. GAE has worked for
selected, less stringent NRMSE targets. More stringent targets and other correction
methods have not been sufficiently validated. Float64 input is unsupported.
This path uses CPU entropy coding and does not use nvCOMP. See the
[platform matrix](docs/platform_support.md).

<details>
<summary>Requirements</summary>

- Apple Silicon Mac (M1 or newer)
- A PyTorch/LibTorch build whose MPS operations and AOTI export work for the selected model
- Matching model artifacts exported for `mps`; a version number alone does not establish compatibility

</details>

<details>
<summary>Build with MPS support</summary>

```bash
# Run from the repository root with an MPS-compatible PyTorch environment.
source venv/bin/activate
python3 compile_model.py mps

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$TORCH_PATH" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel 6
```

Verify that MPS is available before running:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

Use the Homebrew OpenMP configuration described in the CPU build instructions.
Export for the intended device before building: `python3 compile_model.py mps`.
Use `cpu` for CPU execution or `cuda` for CUDA/ROCm export with the appropriate
PyTorch build. The runtime derives its device from the compiled installation;
there is no runtime CLI device override. A positive availability check alone
does not establish successful export or an accurate round trip. See
[model device constraints](docs/models.md#download-and-compile).

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
| CMake                      | 3.18            |
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

For questions, bug reports, or contributions, please open an issue on [GitHub](https://github.com/UFcompressor/CAESAR/issues).

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
