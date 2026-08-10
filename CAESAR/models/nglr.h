#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <zstd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "model_utils.h"

namespace nglr {

// Values written by the training/export step in model_dir/nglr_meta.txt.
struct NGLRMeta {
  double x_mean = 0.0;
  double scale = 1.0;
  double step = 1.0;
  double q_context_scale = 1.0;
  double delta_scale = 1.0;
  int64_t block_t = 1;
  int64_t block_h = 1;
  int64_t block_w = 1;
};

NGLRMeta load_meta_from_model_dir(const std::string &model_dir);

class NGLRModel {
public:
  NGLRModel(const std::string &scripted_model_path,
            std::optional<c10::Device> device = std::nullopt);

  const c10::Device &device() const noexcept { return device_; }
  torch::Tensor encode_recons(const torch::Tensor &recons) const;
  torch::Tensor
  forward_from_recons_feature(const torch::Tensor &recons_feature,
                              const torch::Tensor &q_context) const;

private:
  torch::jit::script::Module module_;
  c10::Device device_;
};

struct NGLRBundle {
  NGLRModel model;
  NGLRMeta meta;

  static NGLRBundle load(const std::string &model_dir,
                         std::optional<c10::Device> device = std::nullopt);
};

struct NGLREncodeStats {
  uint64_t correction_bytes = 0;
  uint64_t uncompressed_bitplane_bytes = 0;
  uint64_t block_count = 0;
  double delta_abs_mean = 0.0;
  uint64_t delta_zero_count = 0;
};

// The correction payload is fully in memory.  No correction file is opened,
// written, read, renamed, or otherwise created by these functions.
NGLREncodeStats nglr_encode(const torch::Tensor &original,
                            const torch::Tensor &recons, const NGLRModel &model,
                            const NGLRMeta &meta,
                            std::vector<uint8_t> &correction_out,
                            int zstd_level = 3);

torch::Tensor nglr_decode(const torch::Tensor &recons, const NGLRModel &model,
                          const NGLRMeta &meta,
                          const std::vector<uint8_t> &correction);

} // namespace nglr
