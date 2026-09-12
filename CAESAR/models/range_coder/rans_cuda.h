#pragma once
#include <string>
#include <torch/torch.h>
#include <vector>

namespace caesar::rans_cuda {
// Use CUDA entropy coding for NVIDIA CUDA tensors; CPU/ROCm use CPU rANS.
bool enabled(const torch::Device &device);

class Codec {
public:
  Codec(const std::vector<std::vector<int32_t>> &cdfs,
        const std::vector<int32_t> &lengths,
        const std::vector<int32_t> &offsets, torch::Device device);
  std::vector<std::string> encode(const torch::Tensor &symbols,
                                  const torch::Tensor &indexes) const;
  // Input strings retain the existing CPU/file format. Output stays on GPU.
  torch::Tensor decode(const std::vector<std::string> &strings, size_t begin,
                       const torch::Tensor &indexes) const;

private:
  torch::Device device_;
  int row_count_;
  torch::Tensor cdf_gpu_, rows_gpu_, offsets_gpu_;
};
} // namespace caesar::rans_cuda
