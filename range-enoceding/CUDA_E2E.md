# CUDA rANS end-to-end check

Branch: `cuda-rans-e2e`. This integrates GPU entropy coding into the existing
compressor and decompressor. CUDA models produce symbols/indexes consumed on
GPU; decoding produces GPU tensors used directly by the hyper decoder and
main decoder. Compressed strings still live on CPU for the existing file/API
format. It is not an entirely GPU-resident compressed-file pipeline.

## Build on your allocated HiPerGator GPU node

Use your existing CUDA LibTorch environment and CUDA-exported models. From the
repository root, reuse your working build configuration:

```bash
module load cuda
cmake -S . -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCAESAR_ENABLE_CUDA_RANS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target test_rans_cuda test_caesarCD -j6
```

89 targets the NVIDIA L4 you tested. Preserve your existing Torch/nvcomp CMake
paths. A new build directory additionally needs your usual CMAKE_PREFIX_PATH.
The build explicitly rejects CPU-only LibTorch for this option.

## First test: GPU codec, no models/data needed

```bash
./build/tests/test_rans_cuda
```

This checks the integrated LibTorch wrapper and actual CUDA kernels, exact CPU
encoded-byte compatibility, decoded GPU values, noncontiguous inputs, a
nondefault CUDA stream and truncated-input rejection. Expected final output:
`PASS: CUDA/LibTorch codec, ...`.

Optional device memory check:

```bash
compute-sanitizer --tool memcheck --error-exitcode=1 ./build/tests/test_rans_cuda
```

`test_rans_kernels` is a separate CPU-emulated arithmetic/sanitizer test, not the
GPU test. It passed 15 local cases before this branch was prepared. The integrated
CUDA build and model end-to-end run still need validation on HiPerGator.

## Second test: your existing test_caesarCD

Run from the directory you normally use, containing the raw file named in your
`test_caesarCD.cpp`. Use identical data, shape, batch size, error bound and
CUDA-exported models for all three runs. `CAESAR_MODEL_DIR` can point to those
models; `model_device.txt` must describe the actual CUDA export. Do not relabel
CPU-exported models as CUDA.

For example, from `build/tests`:

```bash
set -o pipefail
CAESAR_RANS=cpu ./test_caesarCD 2>&1 | tee cpu-rans.log
cp output/encoded_latents.bin output/encoded_latents.cpu.bin
cp output/encoded_hyper_latents.bin output/encoded_hyper_latents.cpu.bin

CAESAR_RANS=verify ./test_caesarCD 2>&1 | tee verify-rans.log
cmp output/encoded_latents.cpu.bin output/encoded_latents.bin
cmp output/encoded_hyper_latents.cpu.bin output/encoded_hyper_latents.bin

CAESAR_RANS=cuda ./test_caesarCD 2>&1 | tee cuda-rans.log
cmp output/encoded_latents.cpu.bin output/encoded_latents.bin
cmp output/encoded_hyper_latents.cpu.bin output/encoded_hyper_latents.bin
```

- `cpu` (default): existing CPU entropy path, models still run on CUDA.
- `verify`: GPU entropy path plus exact CPU comparisons for every encoded
  stream and decoded symbol; throws on mismatch. Expect `[rANS verify]` lines
  for both encode and decode. These additional copies are for verification.
- `cuda`: GPU entropy path without CPU comparisons or symbol/index downloads.

Each run should show `TEST PASSED`. Compare CR and NRMSE across the three logs.
`cmp` should print nothing and return success. Exact entropy verification is
stronger than comparing rounded NRMSE/CR alone. If all data are filtered, there
may be no streams: use a nonconstant input to exercise both CUDA paths.

The CPU codec now reserves two extra flush words for tiny/empty streams; this
does not change encoded bytes for normal inputs. CUDA rejects CDF/index errors,
truncated streams, and escapes needing 8+ bypass nibbles (outside the existing
CPU implementation's defined shift range). It uses the current LibTorch CUDA
stream and bounds encoding scratch by processing at most 1024 streams at once.
Performance optimization is deferred until correctness is confirmed.
