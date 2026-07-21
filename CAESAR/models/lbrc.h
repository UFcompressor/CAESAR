#pragma once
#include "model_utils.h"
#include <zstd.h>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <cstdint>
#include <array>

struct LBRCBlock {
    double   step      = 1.0;
    uint32_t bit_count = 1;

    // one zstd-compressed bit-plane stream per bit
    std::vector<std::vector<uint8_t>> streams;
};

struct LBRCMetaData {
    bool    lbrc_correction_occur = false;
    float   x_mean                = 0.f;
    float   scale                 = 0.f;
    std::array<int64_t, 3> block_size = {60, 120, 120}; // bt, bh, bw
};

namespace caesar::lbrc {
// hard code for cpu for now
void compress(
    const torch::Tensor& original,   // CPU float32
    const torch::Tensor& recons,     // CPU float32
    double                target_nrmse,
    LBRCMetaData&         meta,
    std::vector<LBRCBlock>& blocks,
    int                   workers = 0);

torch::Tensor decompress(
    const torch::Tensor&            recons,   // CPU float32
    const LBRCMetaData&             meta,
    const std::vector<LBRCBlock>&   blocks,
    int                             workers = 0);

} // namespace caesar::lbrc