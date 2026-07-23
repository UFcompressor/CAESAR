#pragma once
#include <torch/script.h>
#include <zstd.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "model_utils.h"

#ifdef USE_CUDA
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#include <c10/hip/HIPCachingAllocator.h>
#include <hip/hip_runtime.h>
#else
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#ifdef ENABLE_NVCOMP
#include <nvcomp/zstd.h>
#endif
#endif
#endif

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP) && !defined(USE_ROCM) && \
    !defined(__HIP_PLATFORM_AMD__)
#define LBRC_HAS_GPU_COMPRESS 1
#else
#define LBRC_HAS_GPU_COMPRESS 0
#endif

struct LBRCBlock {
  double step = 1.0;
  uint32_t bit_count = 1;
  std::vector<std::vector<uint8_t>> streams;
};

struct LBRCMetaData {
  bool lbrc_correction_occur = false;
  float x_mean = 0.f;
  float scale = 0.f;
  std::array<int64_t, 3> block_size = {60, 120, 120};  // bt, bh, bw
};

namespace caesar::lbrc {

void compress(const torch::Tensor& original, const torch::Tensor& recons,
              double target_nrmse, LBRCMetaData& meta,
              std::vector<LBRCBlock>& blocks, int workers = 0);

torch::Tensor decompress(const torch::Tensor& recons, const LBRCMetaData& meta,
                         const std::vector<LBRCBlock>& blocks, int workers = 0);

}  // namespace caesar::lbrc
