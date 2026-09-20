#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nglr_network.h"

namespace nglr {

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

void validate_meta(const NGLRMeta &meta);

struct NGLRTrainOptions {
  int64_t block_t = 60, block_h = 120, block_w = 120;
  int32_t hidden = 32, q_hidden = 16, model_blocks = 4;
  int32_t epochs = 50, batch_size = 8, quant_iters = 24;
  double learning_rate = 5e-4, weight_decay = 1e-6, grad_clip = 1.0;
  int32_t zstd_level = 3;
};

struct NGLRWeight {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> values;
};

// Plain metadata and weights for the existing ADIOS/file writers. The codec
// and network layout are identified separately from the foundation model ID.
struct NGLRMetaData {
  uint32_t schema_version = 1;
  bool correction_occurred = false;
  bool constant_input = false;
  NGLRMeta quantization;
  int32_t hidden = 32, q_hidden = 16, model_blocks = 4;
  std::vector<int64_t> shape;
  std::vector<NGLRWeight> weights;
};

void validate_options(const NGLRTrainOptions &options);
void validate_metadata(const NGLRMetaData &metadata);

class NGLRModel {
public:
  explicit NGLRModel(int32_t hidden = 32, int32_t q_hidden = 16,
                     int32_t blocks = 4,
                     c10::Device device = c10::Device(c10::kCPU));
  explicit NGLRModel(const NGLRMetaData &metadata);
  const c10::Device &device() const noexcept { return device_; }
  torch::Tensor encode_recons(const torch::Tensor &recons) const;
  torch::Tensor forward_from_recons_feature(const torch::Tensor &features,
                                            const torch::Tensor &context) const;
  CausalNeuralLorenzoNet network() const { return network_; }
  std::vector<NGLRWeight> weights() const;
  void load_weights(const std::vector<NGLRWeight> &weights);

private:
  CausalNeuralLorenzoNet network_{nullptr};
  c10::Device device_;
};

// Fits a fresh model and produces correction data. Inputs are the unpadded
// original and foundation reconstruction in [B,C,T,H,W] order.
void compress(const torch::Tensor &original, const torch::Tensor &recons,
              double target_nrmse, NGLRMetaData &metadata,
              std::vector<uint8_t> &correction,
              const NGLRTrainOptions &options = {},
              c10::Device training_device = c10::Device(c10::kCPU));
torch::Tensor decompress(const torch::Tensor &recons,
                         const NGLRMetaData &metadata,
                         const std::vector<uint8_t> &correction);

struct NGLREncodeStats {
  uint64_t correction_bytes = 0;
  uint64_t uncompressed_bitplane_bytes = 0;
  uint64_t block_count = 0;
  double delta_abs_mean = 0.0;
  uint64_t delta_zero_count = 0;
};

NGLREncodeStats nglr_encode(const torch::Tensor &original,
                            const torch::Tensor &recons, const NGLRModel &model,
                            const NGLRMeta &meta,
                            std::vector<uint8_t> &correction_out,
                            int zstd_level = 3);

torch::Tensor nglr_decode(const torch::Tensor &recons, const NGLRModel &model,
                          const NGLRMeta &meta,
                          const std::vector<uint8_t> &correction);

} // namespace nglr
