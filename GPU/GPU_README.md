<div align="center">

# CAESAR — GPU Build

_Assumptions, limitations, and precision configuration for the GPU build target_

</div>

---

## What This Document Covers

This README documents the assumptions, precision configuration, and known limitations of the **GPU build** of CAESAR. Read this in full before applying the patch or building.

For step-by-step installation instructions, see [`INSTALL.md`](INSTALL.md) in this directory.

---

## Applying the Patch

```bash
git apply GPU/gpu.patch
```

Apply this before running any model export scripts or CMake configuration. Do not build without applying it first.

---

## Hardware Requirements

- NVIDIA GPU only. AMD and Apple Silicon GPUs are not supported.
- CUDA 12+ is required. Ensure `nvcc` is available in your `PATH`.
- nvCOMP 5.0+ must be installed. See [`INSTALL.md`](INSTALL.md) for setup instructions.

---

## Precision Configuration

The GPU build uses the following precision for each model component:

| Component          | Precision        |
| ------------------ | ---------------- |
| Compressor         | float16 (half)   |
| Hyper-Decompressor | float64 (double) |
| Decompressor       | float64 (double) |

The compressor runs in float16 to maximize GPU throughput. The hyper-decompressor and decompressor run in float64 for numerical stability during reconstruction.

---

## Cross-Platform Compatibility

**The GPU build is cross-platform compatible — provided the receiving machine uses the same precision configuration (float16 / double / double).**

Because the hyper-decompressor and decompressor use float64 on both machines, compressed data produced by this GPU build can be correctly decompressed on any other machine that also uses a float16 compressor and float64 hyper-decompressor/decompressor. The compressed representation is consistent across machines with the same configuration.

Compatibility rules:

| Compressed with                       | Can decompress with                       |
| ------------------------------------- | ----------------------------------------- |
| GPU build (float16 / double / double) | Any GPU build (float16 / double / double) |
| CPU build (float32 / double / double) | CPU build only — not cross-platform       |

---

## Critical Warning

> **DO NOT use this build to decompress data that was compressed with a different build.**
>
> If a dataset was compressed with the standard CPU build or any other configuration, it must be decompressed with that same build. Using this GPU build to decompress data from a different build will produce incorrect results without any error or warning.
>
> **Only compress and decompress with the same build configuration.**

---

## Known Limitations

- GPU memory limits may affect very large input tensors. If you encounter out-of-memory errors, reduce input chunk size.
- Only tested with NVIDIA GPUs. No support for AMD ROCm or Apple Metal.
- CUDA driver and toolkit versions must be compatible. Mismatched versions are a common source of build failures — verify with `nvcc --version` and `nvidia-smi`.

---

## See Also

- [`INSTALL.md`](INSTALL.md) — Full standalone installation guide for the GPU build
- [`CPU/README.md`](../CPU/README.md) — CPU build assumptions and precision configuration
- [Main README](../README.md) — Project overview, model directory configuration, and citation
