#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace caesar::nglr {

struct NGLRMetaData {
    bool nglr_correction_occur = false;

    double target = 0.0;
    double mean = 0.0;
    double scale = 1.0;
    double step = 1.0;
    double q_scale = 1.0;
    double d_scale = 1.0;

    int block_t = 60;
    int block_h = 128;
    int block_w = 128;

    int hidden = 32;
    int q_hidden = 16;
    int model_blocks = 4;
    int train_epochs = 60;
    int zstd_level = 3;

    std::vector<int64_t> shape;

    int64_t original_bytes = 0;
    int64_t latent_bit = 0;
    int64_t correction_bytes = 0;
    int64_t model_bytes = 0;
};

struct NGLRBlockStream {
    int bit_count = 0;
    int64_t T = 0;
    int64_t H = 0;
    int64_t W = 0;
    std::vector<std::vector<uint8_t>> streams;
};

struct NGLRCompressedData {
    std::vector<NGLRBlockStream> blocks;
};

struct NGLRResult {
    NGLRMetaData meta;
    NGLRCompressedData compressed;
};

torch::Tensor lorenzo_pred(const torch::Tensor& q);
torch::Tensor lorenzo_delta(const torch::Tensor& q);
torch::Tensor zigzag_encode(const torch::Tensor& delta);
torch::Tensor zigzag_decode(const torch::Tensor& zz);

NGLRResult compress(
    const torch::Tensor& original,
    const torch::Tensor& reconstruction,
    double target_nrmse,
    torch::Device device
);

torch::Tensor decompress(
    const torch::Tensor& reconstruction,
    const NGLRMetaData& meta,
    const NGLRCompressedData& compressed,
    torch::Device device
);

} // namespace caesar::nglr
