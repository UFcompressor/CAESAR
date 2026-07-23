#pragma once

#ifdef USE_CUDA
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#include <c10/hip/HIPCachingAllocator.h>
#include <hip/hip_runtime.h>
#else
#include <c10/cuda/CUDACachingAllocator.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#ifdef ENABLE_NVCOMP
#include <nvcomp/cascaded.h>
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>
#endif
#endif
#endif

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
#include <torch/script.h>

#include <cstddef>
#include <cstdint>
#include <vector>

struct NvcompBatchCompressResult {
  torch::Tensor compressed;  // CPU kUInt8 tensor
  size_t rawBytes;
};

std::vector<NvcompBatchCompressResult> nvcomp_batch_compress(
    const std::vector<torch::Tensor>& inputs);

std::vector<std::vector<uint8_t>> nvcomp_batch_decompress(
    const std::vector<const uint8_t*>& comp_ptrs,
    const std::vector<size_t>& comp_sizes,
    const std::vector<size_t>& decomp_sizes);
#endif  // USE_CUDA && ENABLE_NVCOMP
