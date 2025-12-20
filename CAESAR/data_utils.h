#pragma once
#include <torch/torch.h>
#include <vector>
#include <cstdint>

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

/**
 * Convert any N-dimensional tensor to a padded 5D tensor (1, 1, D, H, W)
 * 
 * Fast path: If input is 3D/4D with dimensions >= [8, 128, 128], just reshapes to 5D
 * Slow path: Otherwise flattens and pads to H×W patches
 *
 * @param arr Input tensor of any shape
 * @param H Height of the spatial blocks (default 256)
 * @param W Width of the spatial blocks (default 256)
 * @param force_padding If true, always use slow path (default false)
 * @return std::pair containing the padded 5D tensor and PaddingInfo metadata
 */
std::pair<torch::Tensor, PaddingInfo> to_5d_and_pad(
    const torch::Tensor& arr,
    int64_t H = 256,
    int64_t W = 256,
    bool force_padding = false
);

/**
 * Restore original tensor from padded 5D format using metadata
 *
 * @param padded_5d The padded 5D tensor (1, 1, D, H, W)
 * @param info PaddingInfo metadata from the to_5d_and_pad function
 * @return Restored tensor with original shape and size
 */
torch::Tensor restore_from_5d(
    const torch::Tensor& padded_5d,
    const PaddingInfo& info
);
