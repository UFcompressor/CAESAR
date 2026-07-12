#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace caesar::nglr {

torch::Tensor recons_features(const torch::Tensor& x);

struct ResBlock3DImpl : torch::nn::Module {
    torch::nn::Conv3d c1{nullptr};
    torch::nn::Conv3d c2{nullptr};
    torch::nn::GroupNorm n1{nullptr};
    torch::nn::GroupNorm n2{nullptr};

    explicit ResBlock3DImpl(int64_t ch);

    torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(ResBlock3D);

struct CausalNeuralLorenzoNetImpl : torch::nn::Module {
    torch::nn::Sequential recons_in{nullptr};
    torch::nn::Sequential recons_blocks{nullptr};
    torch::nn::Sequential q_branch{nullptr};
    torch::nn::Sequential fusion{nullptr};

    CausalNeuralLorenzoNetImpl(
        int64_t hidden = 32,
        int64_t q_hidden = 16,
        int64_t blocks = 4
    );

    torch::Tensor encode_recons(const torch::Tensor& r);
    torch::Tensor forward_from_feature(
        const torch::Tensor& rf,
        const torch::Tensor& qctx
    );
    torch::Tensor forward(
        const torch::Tensor& r,
        const torch::Tensor& qctx
    );
};

TORCH_MODULE(CausalNeuralLorenzoNet);
struct NGLRMetaData {
    bool nglr_correction_occur = false;

    double target = 0.0;
    double mean = 0.0;
    double scale = 1.0;
    double step = 1.0;
    double q_scale = 1.0;
    double d_scale = 1.0;

    double base_nrmse = 0.0;
    double quant_nrmse = 0.0;
    double best_loss = 0.0;
    int best_epoch = 0;

    int block_t = 60;
    int block_h = 120;
    int block_w = 120;

    int hidden = 32;
    int q_hidden = 16;
    int model_blocks = 4;
    int train_epochs = 1;
    int zstd_level = 3;

    std::vector<int64_t> shape;

    int64_t original_bytes = 0;
    int64_t latent_bit = 0;
    int64_t correction_bytes = 0;
};

struct NGLRBlockStream {
    int bit_count = 0;
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

NGLRResult encode_correction(
    const torch::Tensor& original,
    const torch::Tensor& reconstruction,
    CausalNeuralLorenzoNet model,
    NGLRMetaData meta,
    torch::Device device
);

torch::Tensor decompress(
    const torch::Tensor& reconstruction,
    const NGLRMetaData& meta,
    const NGLRCompressedData& compressed,
    torch::Device device
);

} // namespace caesar::nglr
