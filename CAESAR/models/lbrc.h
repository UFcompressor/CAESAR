#pragma once
#include "model_utils.h"
#include <zstd.h>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdint>
#include <array>

struct LBRCBlock {
    int64_t b  = 0, c  = 0;
    int64_t t0 = 0, t1 = 0;
    int64_t h0 = 0, h1 = 0;
    int64_t w0 = 0, w1 = 0;

    double   step      = 1.0;
    uint32_t bit_count = 1;
    int64_t  num       = 0;                           // T*H*W voxels

    // one zstd-compressed bit-plane stream per bit
    std::vector<std::vector<uint8_t>> streams;
};

struct LBRCMetaData {
    bool    lbrc_correction_occur = false;
    float   x_mean                = 0.f;
    float   scale                 = 0.f;
    double  encoded_nrmse         = 0.0;
    // B,C,T,H,W of the tensor that was compressed (after deblockHW, pre-pad)
    std::vector<int32_t>   shape;        // size 5
    std::array<int64_t, 3> block_size    = {60, 120, 120}; // bt, bh, bw
    int                    zstd_level    = 21;
    int                    quant_iter    = 16;
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