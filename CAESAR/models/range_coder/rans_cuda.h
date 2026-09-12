#pragma once
#include <string>
#include <torch/torch.h>
#include <vector>

namespace caesar::rans_cuda {
// CAESAR_RANS=cpu (default), cuda, or verify (GPU + exact CPU comparisons).
bool enabled(const torch::Device &device);
bool verification_enabled();

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
  std::vector<std::vector<int32_t>> cdfs_;
  std::vector<int32_t> lengths_, offsets_;
  torch::Tensor cdf_gpu_, rows_gpu_, offsets_gpu_;
};
} // namespace caesar::rans_cuda
