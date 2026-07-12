#pragma once

#include "nglr_model.h"

#include <torch/torch.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace caesar::nglr {

// Training configuration is intentionally generic.
// Nothing here is tied to JHTDB, one tensor shape, or one error bound.
struct NGLRTrainConfig {
    double target_nrmse = 1e-5;

    // A value <= 0 means "choose automatically from the input shape".
    int block_t = 0;
    int block_h = 0;
    int block_w = 0;

    int hidden = 32;
    int q_hidden = 16;
    int model_blocks = 4;

    int train_epochs = 1;
    double lr = 5e-4;
    double weight_decay = 1e-6;
    double grad_clip = 1.0;

    int quant_iters = 24;
    int zstd_level = 3;
    int seed = 2026;

    bool verbose = true;
};

struct NGLRTrainResult {
    CausalNeuralLorenzoNet model = nullptr;
    NGLRMetaData meta;

    double base_nrmse = 0.0;
    double quant_nrmse = 0.0;
    double best_loss = 0.0;
    int best_epoch = 0;
};

NGLRTrainConfig default_train_config(double target_nrmse);

NGLRTrainResult train_nglr_model(
    const torch::Tensor& original,
    const torch::Tensor& reconstruction,
    const NGLRTrainConfig& config,
    torch::Device device
);

} // namespace caesar::nglr
