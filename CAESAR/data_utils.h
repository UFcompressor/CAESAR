#pragma once
#include <torch/torch.h>

#include <cstdint>

#include "models/model_utils.h"

/**
 * Structure to hold padding metadata for 5D tensor conversion
 */
struct PaddingInfo {
  std::vector<int64_t> original_shape;
  int64_t original_length;
  std::vector<int64_t> padded_shape;
  int64_t H;
  int64_t W;
  bool was_padded;
};

std::pair<torch::Tensor, PaddingInfo> to_5d(torch::Tensor &arr);

/**
 * Restore original tensor from padded 5D format using metadata
 *
 * @param padded_5d The padded 5D tensor (1, 1, D, H, W)
 * @param info PaddingInfo metadata from the to_5d_and_pad function
 * @return Restored tensor with original shape and size
 */
torch::Tensor restore_from_5d(torch::Tensor &padded_5d,
                              const PaddingInfo &info);
