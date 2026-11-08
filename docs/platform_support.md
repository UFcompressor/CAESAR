# Platform compatibility and validation

This matrix separates implemented execution paths from testing evidence. The
Linux CPU/NVIDIA, preliminary ROCm GAE, and experimental MPS statuses below are
maintainer-reported. CPU CI scope is derived from the checked-in
[cross-platform workflow](../.github/workflows/cross-platform-test.yml) and its
test sources. No hosted CI run was independently checked for this documentation
update; “CI-tested” describes the reported CPU testing and configured coverage,
not a claim that a particular revision's run passed.

## Training validation

The maintainer reports that model training has been tested on **NVIDIA GPUs
only**. Training has not been tested on CPU, AMD ROCm, Apple MPS, Intel XPU, or
any other backend. This records testing scope, not a claim that training on
other backends is impossible.

The matrix below describes compression/decompression execution and residual
correction validation. Extensive CPU inference testing and CPU CI coverage do
not establish model-training validation on CPU. NGLR's per-compression correction
training is distinct from foundation-model training in this statement.

## Compatibility matrix

| Operating system      | Compute backend | GAE                    | LBRC                      | NGLR                              | Verification scope                                                                                           |
| --------------------- | --------------- | ---------------------- | ------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Linux                 | CPU             | Extensively tested     | Extensively tested        | Experimental / limited validation | Maintainer reports reliable GAE/LBRC in normal configurations; CPU CI also exercises both.                   |
| Linux                 | NVIDIA CUDA     | Extensively tested     | Extensively tested        | Experimental / limited validation | Primary, most extensively tested GPU deployment; maintainer-reported use, separate from CPU CI.              |
| Windows               | CPU             | CI-tested              | CI-tested                 | Not established by CI             | MSVC/Conda CPU workflow; Release assertions and model-cache exclusion limit coverage.                        |
| Windows               | GPU             | Not verified           | Not verified              | Not verified                      | No verified Windows GPU execution.                                                                           |
| macOS                 | CPU             | CI-tested              | CI-tested                 | Not established by CI             | CPU model export and Debug build in the workflow.                                                            |
| macOS / Apple Silicon | MPS             | Experimental / limited | Not sufficiently verified | Not sufficiently verified         | GAE has worked for selected, less stringent targets; more stringent targets remain insufficiently validated. |
| Linux                 | AMD ROCm        | Preliminary validation | **Testing in progress**   | Not sufficiently verified         | Maintainer reports successful GAE GPU execution; LBRC result is not established.                             |
| Backend-dependent OS  | Intel XPU       | Not verified           | Not verified              | Not verified                      | Device/export path exists; verified end-to-end execution is not established.                                 |

“Extensively tested” describes substantial use and testing, not a mathematical
error guarantee for every input. “CI-tested” has the narrower scope described
below. “Preliminary validation” describes selected execution evidence, not a
representative dataset/error-target campaign. “Experimental / limited” and
“Not sufficiently verified” require further numerical and compatibility checks.
“Testing in progress” records a pending result. “Not verified” does not assert
that an implementation is absent or that execution must fail.

NGLR is implemented but has more limited validation even on Linux CPU and NVIDIA;
issues may occur. Keep it available as an API method without assuming parity
with GAE and LBRC. See [NGLR](nglr.md).

## Inference, entropy coding, and model artifacts

- **CPU inference** executes the compiled neural components through LibTorch on
  CPU. GAE and LBRC are established CPU implementations, not GPU-only methods.
- **Accelerator inference** runs eligible tensor/model operations on the selected
  accelerator. It does not imply that every correction or coding operation runs
  there, or that every correction method is validated.
- **Entropy coding** has CPU rANS and NVIDIA-specific custom CUDA rANS paths.
  [CMake](../CMakeLists.txt) excludes ROCm from `CAESAR_HAS_CUDA_RANS`, and
  [the runtime guard](../CAESAR/models/range_coder/rans_cuda.cpp) excludes
  `USE_ROCM`. ROCm exposes its device through PyTorch's `cuda` API but uses CPU
  entropy coding in place of NVIDIA CUDA rANS. CPU and MPS also use CPU entropy
  coding. This fallback does not establish ROCm LBRC or NGLR correctness.
- **nvCOMP** is an NVIDIA-specific lossless acceleration library, distinct from
  neural inference and rANS. Its use depends on build/runtime paths. Although
  optional in the broader architecture, the current NVIDIA CUDA CMake path
  requires nvCOMP **5.0.0.6**, reusing an exact SDK or downloading it. Setting
  `CAESAR_DOWNLOAD_NVCOMP=OFF` requires a local SDK; it does not disable nvCOMP.
  Automatic download requires Linux x86_64 and CUDA 12. CPU/ROCm builds do not
  download it. See [NVIDIA build instructions](../README.md#gpu-support-nvidia).

Export the model for the intended backend (`cpu`, `cuda` for NVIDIA/ROCm with the
appropriate PyTorch build, `mps`, or `xpu`). The device/export choices are
implementation paths, not equal compatibility promises. AOTI compilation and
required backend operations must work on the actual platform and PyTorch version.
The runtime reads the compiled installation's device; changing a runtime flag
cannot retarget CPU artifacts to a GPU. An unavailable compiled backend raises
an error rather than automatically falling back to CPU. Constructors and the current CLI do not
accept a device override. Keep all three `.pt2` packages and six probability
tables from the matching installation. See [models](models.md).

MPS compatibility depends on PyTorch and available backend operations. Float64
input on MPS is explicitly rejected. CPU CI and a successful
`torch.backends.mps.is_available()` check do not establish MPS numerical accuracy.

## CPU CI scope and limitations

The workflow downloads `caesar_v2`, checks registry/checkpoint metadata, and
exports a CPU model. Linux/macOS use Python 3.11 and Debug builds; Windows uses
Python 3.10 and Release. The configured PyTorch versions are Linux `2.9.0`,
macOS `2.8.0`, and Windows `2.12.0` (CPU installation for Windows/Linux).
These are workflow pins, not an assertion that every PyTorch version works.

All three jobs run `test_caesarCD`, `test_runGaeCuda`, `test_padding`,
`test_model_metadata`, `test_LBRC`, `test_serialization`, `test_rans_kernels`,
and CLI GAE/LBRC round trips. The generated float32 fixture has shape
`1,1,20,256,256`. Despite its name, `test_runGaeCuda` exercises CPU operations
in these CPU jobs. `test_caesarCD` measures GAE NRMSE at target `0.0001`
(with float epsilon tolerance); the CLI checks GAE/LBRC NRMSE against `0.001`
and verifies reconstructed byte counts. These selected tests do not exhaust
shapes, models, datasets, or error targets.

Windows differs from Linux/macOS:

- `test_model_cache` is explicitly skipped: multiple AOTI runners encounter a
  locked extracted `.pyd` file (Windows error 32).
- Release defines `NDEBUG`, so assertion-based checks (including checks in
  `test_LBRC` and GAE unit tests) do not provide the same validation as Debug.
  The CLI still explicitly checks NRMSE and file sizes.
- Some PowerShell test steps invoke native executables without explicitly
  propagating `$LASTEXITCODE`; a green job alone is weaker evidence than each
  test's logs. Review individual outputs before claiming revision-level success.
- Windows runs the CLI shell test through Git for Windows Bash and adds
  PyTorch's `lib` directory to `PATH` for serialization/rANS and CLI tests.

`test_NGLR` is built but not invoked by these workflow jobs. The quick CLI test
intentionally excludes NGLR because of its per-compression training cost.
Serialization coverage of NGLR fields is not NGLR inference validation. The
standalone `test_caesarC` and `test_caesarD` binaries are also not invoked as
separate workflow steps. No accelerator CI job is defined here.

## Windows CPU installation

Windows dependency installation is more involved than Linux, particularly for
CompressAI. Use the existing [Windows workflow](../.github/workflows/cross-platform-test.yml)
as the executable reference. It uses `windows-latest`, Visual Studio MSVC x64
(discovered with `vswhere`), Miniconda environment `compress_env` with Python
3.10, CMake/NMake, Conda zstd, and CPU PyTorch. Do not substitute the POSIX
virtual environment or an assumed CompressAI pip-only recipe.

1. Install Git for Windows, Miniconda, and Visual Studio C++ Build Tools with
   the x64 compiler and Windows SDK. CI enables Windows long paths. Its short
   temporary/source directories (`D:\T`, `D:\cai`) also avoid long build paths;
   adapt these to a writable local drive when reproducing locally.
2. Create `compress_env` with Python 3.10. From PowerShell in the repository,
   reproduce the workflow's filtering and CPU dependencies:

   ```powershell
   conda create -n compress_env python=3.10 -y
   $lines = Get-Content requirements.txt | Where-Object {
       $_ -notmatch "^torch" -and
       $_ -notmatch "^torchvision" -and
       $_ -notmatch "^--extra-index-url" -and
       $_ -notmatch "^cupy" -and
       $_ -notmatch "^nvidia" -and
       $_ -notmatch "^compressai" -and
       $_ -notmatch "^imageio" -and
       $_ -notmatch "^\s*$"
   }
   $lines | Set-Content temp_requirements.txt
   conda run -n compress_env python -m pip install --upgrade pip wheel setuptools
   conda run -n compress_env python -m pip install --no-cache-dir -r temp_requirements.txt
   Remove-Item temp_requirements.txt
   conda run -n compress_env python -m pip install torch==2.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   conda install -n compress_env -y -c conda-forge cmake zstd
   ```

3. Build CompressAI from its source as CI does. The workflow clones its default
   branch without a commit pin, so dependency reproducibility is limited. It
   excludes `imageio` from the initial requirements; despite the step name,
   it does not issue a separate explicit imageio installation command.

   ```powershell
   $env:TEMP = "D:\T"
   $env:TMP = "D:\T"
   New-Item -ItemType Directory -Force -Path D:\T
   git clone https://github.com/InterDigitalInc/CompressAI.git D:\cai
   Push-Location D:\cai
   Get-ChildItem -Recurse -Filter "*.py" | ForEach-Object {
       $content = Get-Content $_.FullName -Raw -Encoding UTF8
       if ($content -match '"-std=c\+\+17"' -or $content -match '"-O3"') {
           $content = $content -replace '"-std=c\+\+17"', '"/std:c++17"'
           $content = $content -replace '"-O3"', '"/O2"'
           Set-Content $_.FullName $content -Encoding UTF8
       }
   }
   conda run -n compress_env python -m pip install pybind11 wheel setuptools
   conda run -n compress_env python setup.py bdist_wheel --dist-dir D:\cai\dist
   $wheel = Get-ChildItem D:\cai\dist\compressai-*.whl | Select-Object -First 1
   conda run -n compress_env python -m pip install $wheel.FullName
   Pop-Location
   ```

4. Download/check the model and export under the MSVC environment. In CI,
   `$env:GITHUB_WORKSPACE` is the checkout; locally set it to the repository
   root first. Check each native command's exit code before proceeding.

   ```powershell
   $env:GITHUB_WORKSPACE = (Get-Location).Path
   $env:CAESAR_MODEL_DIR = "$env:GITHUB_WORKSPACE\exported_model"
   $env:PYTHONPATH = $env:GITHUB_WORKSPACE
   conda run -n compress_env python model_registry.py caesar_v2 --output "$env:GITHUB_WORKSPACE\pretrained"
   conda run -n compress_env python tests/test_model_registry.py
   conda run -n compress_env python tests/check_registered_checkpoints.py
   $vsPath = & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
   $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
   $python = (conda run -n compress_env python -c "import sys; print(sys.executable)").Trim()
   $selection = Join-Path $env:GITHUB_WORKSPACE "pretrained\selected_model.json"
   cmd /c "`"$vcvars`" && `"$python`" compile_model.py cpu --selection `"$selection`" --output `"$env:CAESAR_MODEL_DIR`""
   ```

   CI requires nonempty `model_metadata.txt`, all three `caesar_*.pt2` packages
   (compressor, hyper-decompressor, decompressor), and the six `vbr_*` / `gs_*`
   `.bin` tables before CMake configuration.

5. Configure and build using MSVC/NMake and the same environment's Torch:

   ```powershell
   $TORCH_PATH = (conda run -n compress_env python -c "import torch; print(torch.utils.cmake_prefix_path)").Trim()
   New-Item -ItemType Directory -Force -Path build
   Push-Location build
   cmd /c "`"$vcvars`" && cmake .. -G `"NMake Makefiles`" -DCMAKE_PREFIX_PATH=`"$TORCH_PATH`" -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release"
   cmd /c "`"$vcvars`" && cmake --build ."
   Pop-Location
   conda run -n compress_env python tests/gen_data.py build/tests --filename TCf48.bin.f32
   ```

Follow the workflow's per-test working directories, DLL paths, output-directory
creation, MSVC setup for `test_caesarCD.exe`, and Git Bash invocation for CLI
round trips. The CLI executable is `build/CAESAR/caesar_cli.exe`. The model-cache
skip and Release limitations above remain applicable. This procedure documents
the checked-in CPU workflow; it does not verify Windows GPU execution.

## Numerical accuracy and decoding portability

A requested NRMSE is a **target whose achieved error must be checked** against
the original data. It is not an unconditional guarantee. Extensive CPU/NVIDIA
testing does not remove the documented filtering failures, possible padding
issues, GAE/LBRC block artifacts, or float32 precision limitations. Keep the
recorded examples and unresolved hypotheses in [known limitations](TODO.txt).

GPU-generated compressed data is **not currently guaranteed to decode correctly
on CPU** and can reconstruct garbled data. Latent indexes are regenerated by
the hyper-decoder rather than serialized; matching probability tables are also
required separately. A common serialization format or CPU-compatible rANS
kernel test does not establish full cross-device numerical/entropy-decoding
portability. Validate reconstruction and achieved error for the actual producer
and consumer configurations.

## Pending AMD LBRC evidence

**AMD ROCm LBRC: Testing in progress.** No success or failure result is recorded
here. Preserve the CPU, NVIDIA, Apple, and Windows classifications unless new
evidence directly warrants a change. Use the following record for each trial:

| Field             | Value to record                                                 |
| ----------------- | --------------------------------------------------------------- |
| Software revision | CAESAR commit and build options                                 |
| Hardware and OS   | GPU model, Linux distribution/version                           |
| Runtime           | ROCm version, PyTorch version, relevant dependency versions     |
| Model             | Checkpoint name/identity/hash and export backend                |
| Input             | Dataset identifier, dimensions/axis order, dtype                |
| Correction        | LBRC settings and requested NRMSE                               |
| Compression       | Success/failure, exit status, log location                      |
| Decompression     | Success/failure, exit status, log location                      |
| Reconstruction    | Expected/actual shape and byte count                            |
| Accuracy          | Measured NRMSE against original data and whether target was met |

Record missing fields as unknown. Distinguish successful execution, shape
preservation, and achieved accuracy. A single successful end-to-end trial may
justify “Preliminary validation” for that configuration; it cannot establish
comprehensive validation. On failure, record its conditions and keep compatibility
under investigation. Stronger wording requires representative datasets and error
targets. SoftwareX manuscript platform revisions remain deferred until this
evidence has been reviewed; no conditional success claim belongs in the paper yet.
