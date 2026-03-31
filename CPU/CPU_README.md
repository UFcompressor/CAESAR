<div align="center">

# CAESAR — CPU Build

_Assumptions, limitations, and precision configuration for the CPU build target_

</div>

---

## What This Document Covers

This README documents the assumptions, precision configuration, and known limitations of the **CPU build** of CAESAR. Read this in full before applying the patch or building.

For step-by-step installation instructions, see [`INSTALL.md`](INSTALL.md) in this directory.

---

## Applying the Patch

```bash
git apply CPU/cpu_build.patch
```

This patch modifies `CAESAR_compressor.py` and the C++ compressor source to switch from float16 to float32. Do not build without applying it first.

---

## Precision Configuration

The CPU build uses the following precision for each model component:

| Component          | Precision        |
| ------------------ | ---------------- |
| Compressor         | float32 (single) |
| Hyper-Decompressor | float64 (double) |
| Decompressor       | float64 (double) |

This differs from the GPU build, which uses float16 for the compressor. The CPU build has not been precision-optimized — float32 is used for the compressor because float16 is not efficiently supported on CPU.

---

## Cross-Platform Compatibility

**The CPU build is not cross-platform compatible with the GPU build.**

Because the CPU compressor runs in float32 and the GPU compressor runs in float16, compressed outputs from one build cannot be correctly decompressed by the other. The hyper-decompressor and decompressor use float64 in both builds, but this does not make them interchangeable — the compressed representation itself differs.

Compatibility rules:

| Compressed with                       | Must decompress with                  |
| ------------------------------------- | ------------------------------------- |
| CPU build (float32 / double / double) | CPU build (float32 / double / double) |
| GPU build (float16 / double / double) | GPU build (float16 / double / double) |

---

## Critical Warning

> **DO NOT use this build to decompress data that was compressed with a different build.**
>
> If a dataset was compressed with the standard or GPU build, it must be decompressed with that same build. Using this CPU build to decompress data from another build will produce incorrect results without any error or warning.
>
> **Only compress and decompress with the same build configuration.**

---

## Known Limitations

- The CPU compressor has not been performance-optimized. Compression will be significantly slower than the GPU build, particularly for large datasets.
- float16 inference is not used on CPU. If you need float16 behavior, use the GPU build on a CUDA-capable machine.
- This build has been tested on Linux and macOS. Windows is not supported.

---

## See Also

- [`INSTALL.md`](INSTALL.md) — Full standalone installation guide for the CPU build
- [`GPU/README.md`](../GPU/README.md) — GPU build assumptions and precision configuration
- [Main README](../README.md) — Project overview, model directory configuration, and citation
