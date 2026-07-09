#include "nglr_train.h"

#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <sstream>


#if !defined(USE_CUDA)

// Training NGLR also needs CUDA. For CPU-only builds, these stubs make the limitation
// obvious and keep portability builds from spending time compiling unused GPU code.
#include <stdexcept>

namespace caesar::nglr {

NGLRTrainConfig default_train_config(double target_nrmse) {
    NGLRTrainConfig cfg;
    cfg.target_nrmse = target_nrmse;
    return cfg;
}

// Train the NGLR predictor. The predictor's job is to make residuals smaller so the
// bitplane compressor has less information to store.
NGLRTrainResult train_nglr_model(
    const torch::Tensor&,
    const torch::Tensor&,
    const NGLRTrainConfig&,
    torch::Device
) {
    // macOS GitHub runners are CPU-only and cannot provide CUDA/nvCOMP.
    // Avoid compiling the heavy LibTorch autograd training implementation there;
    // CUDA/nvCOMP builds compile the full implementation below unchanged.
    throw std::runtime_error(
        "NGLR training requires a CUDA/nvCOMP-enabled build."
    );
}

} // namespace caesar::nglr

#else
namespace caesar::nglr {
namespace {

constexpr int kDefaultBlockT = 60;
constexpr int kDefaultBlockH = 128;
constexpr int kDefaultBlockW = 128;

struct ResolvedBlockConfig {
    int block_t = kDefaultBlockT;
    int block_h = kDefaultBlockH;
    int block_w = kDefaultBlockW;
};

struct BlockSlice {
    int64_t b = 0;
    int64_t c = 0;
    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t h0 = 0;
    int64_t h1 = 0;
    int64_t w0 = 0;
    int64_t w1 = 0;
};

ResolvedBlockConfig resolve_block_config(
    const NGLRTrainConfig& config,
    const std::vector<int64_t>& shape
) {
    if (shape.size() != 5) {
        throw std::runtime_error("NGLR training expects 5D tensor [B,C,T,H,W]");
    }

    const int64_t T = shape[2];
    const int64_t H = shape[3];
    const int64_t W = shape[4];

    ResolvedBlockConfig out;
    out.block_t = static_cast<int>(std::min<int64_t>(
        config.block_t > 0 ? config.block_t : kDefaultBlockT,
        T
    ));
    out.block_h = static_cast<int>(std::min<int64_t>(
        config.block_h > 0 ? config.block_h : kDefaultBlockH,
        H
    ));
    out.block_w = static_cast<int>(std::min<int64_t>(
        config.block_w > 0 ? config.block_w : kDefaultBlockW,
        W
    ));

    out.block_t = std::max(1, out.block_t);
    out.block_h = std::max(1, out.block_h);
    out.block_w = std::max(1, out.block_w);

    return out;
}

std::vector<BlockSlice> block_slices(
    const std::vector<int64_t>& shape,
    int64_t block_t,
    int64_t block_h,
    int64_t block_w
) {
    if (shape.size() != 5) {
        throw std::runtime_error("NGLR training expects 5D tensor [B,C,T,H,W]");
    }

    std::vector<BlockSlice> out;

    const int64_t B = shape[0];
    const int64_t C = shape[1];
    const int64_t T = shape[2];
    const int64_t H = shape[3];
    const int64_t W = shape[4];

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t t0 = 0; t0 < T; t0 += block_t) {
                const int64_t t1 = std::min(t0 + block_t, T);

                for (int64_t h0 = 0; h0 < H; h0 += block_h) {
                    const int64_t h1 = std::min(h0 + block_h, H);

                    for (int64_t w0 = 0; w0 < W; w0 += block_w) {
                        const int64_t w1 = std::min(w0 + block_w, W);
                        out.push_back({b, c, t0, t1, h0, h1, w0, w1});
                    }
                }
            }
        }
    }

    return out;
}

torch::Tensor slice_5d_to_3d_train(
    const torch::Tensor& x,
    const BlockSlice& sl
) {
    return x.index({
        sl.b,
        sl.c,
        torch::indexing::Slice(sl.t0, sl.t1),
        torch::indexing::Slice(sl.h0, sl.h1),
        torch::indexing::Slice(sl.w0, sl.w1)
    }).contiguous();
}

double tensor_mean_cpu(const torch::Tensor& x) {
    return x.to(torch::kCPU).to(torch::kFloat32).mean().item<double>();
}

double tensor_range_cpu(const torch::Tensor& x) {
    torch::Tensor xc = x.to(torch::kCPU).to(torch::kFloat32);
    return (xc.max() - xc.min()).item<double>();
}

double decoded_nrmse(
    const torch::Tensor& original,
    const torch::Tensor& r_norm,
    const torch::Tensor& q,
    double step,
    double mean,
    double scale
) {
    torch::Tensor decoded =
        (r_norm + q.to(torch::kFloat32) * step) * scale + mean;

    torch::Tensor err =
        (original.to(torch::kFloat32) - decoded.to(torch::kFloat32)) / scale;

    return std::sqrt(err.pow(2).mean().item<double>());
}

double zero_nrmse(
    const torch::Tensor& original,
    const torch::Tensor& reconstruction,
    double scale
) {
    torch::Tensor err =
        (original.to(torch::kFloat32) - reconstruction.to(torch::kFloat32)) / scale;

    return std::sqrt(err.pow(2).mean().item<double>());
}

torch::Tensor quantize_with_step(
    const torch::Tensor& residual,
    double step
) {
    return torch::round(residual / step).to(torch::kInt32).contiguous();
}

double decode_sse(
    const torch::Tensor& original,
    const torch::Tensor& r_norm,
    const torch::Tensor& residual,
    double step,
    double mean,
    double scale
) {
    torch::Tensor q = torch::round(residual / step);
    torch::Tensor decoded = (r_norm + q * step) * scale + mean;
    torch::Tensor e = (original.to(torch::kFloat32) - decoded.to(torch::kFloat32)) / scale;
    return e.pow(2).sum().item<double>();
}

std::pair<double, torch::Tensor> safe_global_quantize(
    const torch::Tensor& original,
    const torch::Tensor& reconstruction,
    const torch::Tensor& r_norm,
    const torch::Tensor& residual,
    double target,
    int iters,
    double mean,
    double scale
) {
    const int64_t n = residual.numel();
    const double target_sse = target * target * static_cast<double>(n);

    const double base_sse =
        ((original.to(torch::kFloat32) - reconstruction.to(torch::kFloat32)) / scale)
            .pow(2)
            .sum()
            .item<double>();

    if (base_sse <= target_sse) {
        return {
            1.0,
            torch::zeros_like(residual, torch::TensorOptions().dtype(torch::kInt32))
        };
    }

    double low = 0.0;
    double high = std::max(target * std::sqrt(12.0), 1e-12);

    while (decode_sse(original, r_norm, residual, high, mean, scale) <= target_sse) {
        low = high;
        high *= 2.0;
    }

    for (int i = 0; i < std::max(1, iters); ++i) {
        const double mid = 0.5 * (low + high);

        if (decode_sse(original, r_norm, residual, mid, mean, scale) <= target_sse) {
            low = mid;
        } else {
            high = mid;
        }
    }

    double step = std::max(low, 1e-12);
    torch::Tensor q = quantize_with_step(residual, step);

    while (decoded_nrmse(original, r_norm, q, step, mean, scale) > target) {
        step *= 0.999999;
        q = quantize_with_step(residual, step);
    }

    return {step, q};
}

// Build the basic Lorenzo residual used as a training target. It is simply the current
// q value minus what causal Lorenzo prediction would have guessed.
torch::Tensor lorenzo_delta_3d_train(const torch::Tensor& q_in) {
    torch::Tensor q = q_in.to(torch::kCPU).to(torch::kInt64).contiguous();

    if (q.dim() != 3) {
        throw std::runtime_error("lorenzo_delta_3d_train expects [T,H,W]");
    }

    torch::Tensor pred = torch::zeros_like(q);

    const int64_t T = q.size(0);
    const int64_t H = q.size(1);
    const int64_t W = q.size(2);

    auto qa = q.accessor<int64_t, 3>();
    auto pa = pred.accessor<int64_t, 3>();

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                const int64_t v1 = (t > 0) ? qa[t - 1][h][w] : 0;
                const int64_t v2 = (h > 0) ? qa[t][h - 1][w] : 0;
                const int64_t v3 = (w > 0) ? qa[t][h][w - 1] : 0;

                const int64_t v4 = (t > 0 && h > 0) ? qa[t - 1][h - 1][w] : 0;
                const int64_t v5 = (t > 0 && w > 0) ? qa[t - 1][h][w - 1] : 0;
                const int64_t v6 = (h > 0 && w > 0) ? qa[t][h - 1][w - 1] : 0;
                const int64_t v7 =
                    (t > 0 && h > 0 && w > 0) ? qa[t - 1][h - 1][w - 1] : 0;

                pa[t][h][w] = v1 + v2 + v3 - v4 - v5 - v6 + v7;
            }
        }
    }

    return (q - pred).contiguous();
}

// Create neighbor-context channels from q. These are the same kind of causal clues the
// model will have during decompression, so training stays honest.
torch::Tensor q_context_3d_train(
    const torch::Tensor& q_in,
    double q_scale
) {
    torch::Tensor q = q_in.to(torch::kCPU).to(torch::kInt64).contiguous();

    const int64_t T = q.size(0);
    const int64_t H = q.size(1);
    const int64_t W = q.size(2);

    torch::Tensor ctx =
        torch::zeros({8, T, H, W}, torch::TensorOptions().dtype(torch::kFloat32));

    auto qa = q.accessor<int64_t, 3>();
    auto ca = ctx.accessor<float, 4>();

    const double denom = std::max(q_scale, 1.0);

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                const int64_t v1 = (t > 0) ? qa[t - 1][h][w] : 0;
                const int64_t v2 = (h > 0) ? qa[t][h - 1][w] : 0;
                const int64_t v3 = (w > 0) ? qa[t][h][w - 1] : 0;

                const int64_t v4 = (t > 0 && h > 0) ? qa[t - 1][h - 1][w] : 0;
                const int64_t v5 = (t > 0 && w > 0) ? qa[t - 1][h][w - 1] : 0;
                const int64_t v6 = (h > 0 && w > 0) ? qa[t][h - 1][w - 1] : 0;
                const int64_t v7 =
                    (t > 0 && h > 0 && w > 0) ? qa[t - 1][h - 1][w - 1] : 0;

                const int64_t pred = v1 + v2 + v3 - v4 - v5 - v6 + v7;

                ca[0][t][h][w] = static_cast<float>(v1 / denom);
                ca[1][t][h][w] = static_cast<float>(v2 / denom);
                ca[2][t][h][w] = static_cast<float>(v3 / denom);
                ca[3][t][h][w] = static_cast<float>(v4 / denom);
                ca[4][t][h][w] = static_cast<float>(v5 / denom);
                ca[5][t][h][w] = static_cast<float>(v6 / denom);
                ca[6][t][h][w] = static_cast<float>(v7 / denom);
                ca[7][t][h][w] = static_cast<float>(pred / denom);
            }
        }
    }

    return ctx.contiguous();
}

std::pair<double, double> estimate_scales(
    const torch::Tensor& q,
    const NGLRTrainConfig& config,
    int block_t,
    int block_h,
    int block_w
) {
    const std::vector<int64_t> shape = q.sizes().vec();

    double q_sum = 0.0;
    double d_sum = 0.0;
    int64_t n = 0;

    for (const BlockSlice& sl : block_slices(shape, block_t, block_h, block_w)) {
        torch::Tensor qb = slice_5d_to_3d_train(q, sl);
        torch::Tensor db = lorenzo_delta_3d_train(qb).to(torch::kInt64);

        q_sum += qb.to(torch::kInt64).abs().sum().item<double>();
        d_sum += db.abs().sum().item<double>();
        n += qb.numel();
    }

    if (n == 0) {
        return {1.0, 1.0};
    }

    return {
        std::max(1.0, q_sum / static_cast<double>(n)),
        std::max(1.0, d_sum / static_cast<double>(n))
    };
}

torch::Tensor charbonnier_loss(
    const torch::Tensor& x,
    double eps = 1e-6
) {
    return torch::sqrt(x * x + eps * eps).mean();
}
void copy_model_weights(
    CausalNeuralLorenzoNet src,
    CausalNeuralLorenzoNet dst
) {
    torch::serialize::OutputArchive out;
    src->save(out);

    std::stringstream buffer;
    out.save_to(buffer);

    torch::serialize::InputArchive in;
    in.load_from(buffer);

    dst->load(in);
}

} // namespace

NGLRTrainConfig default_train_config(double target_nrmse) {
    NGLRTrainConfig cfg;
    cfg.target_nrmse = target_nrmse;
    return cfg;
}

NGLRTrainResult train_nglr_model(
    const torch::Tensor& original,
    const torch::Tensor& reconstruction,
    const NGLRTrainConfig& config,
    torch::Device device
) {
    if (!original.defined() || !reconstruction.defined()) {
        throw std::runtime_error("NGLR training received undefined tensors");
    }

    c10::InferenceMode inference_mode(false);
    torch::AutoGradMode grad_mode(true);

    torch::manual_seed(config.seed);
    if (device.is_cuda()) {
        torch::cuda::manual_seed_all(config.seed);
    }

    torch::Tensor x = original.to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor r = reconstruction.to(torch::kCPU).to(torch::kFloat32).contiguous();

    if (x.sizes() != r.sizes()) {
        throw std::runtime_error("NGLR training shape mismatch");
    }

    if (x.dim() != 5) {
        throw std::runtime_error("NGLR training expects 5D tensor [B,C,T,H,W]");
    }

    const std::vector<int64_t> shape = x.sizes().vec();
    ResolvedBlockConfig blocks = resolve_block_config(config, shape);

    const double mean = tensor_mean_cpu(x);
    const double scale = tensor_range_cpu(x);

    if (scale <= 0.0) {
        throw std::runtime_error("NGLR training invalid scale");
    }

    torch::Tensor x_norm = (x - mean) / scale;
    torch::Tensor r_norm = (r - mean) / scale;
    torch::Tensor residual = (x_norm - r_norm).contiguous();

    auto quantized =
        safe_global_quantize(
            x,
            r,
            r_norm,
            residual,
            config.target_nrmse,
            config.quant_iters,
            mean,
            scale
        );

    const double step = quantized.first;
    torch::Tensor q = quantized.second.to(torch::kCPU).to(torch::kInt32).contiguous();

    auto scales =
        estimate_scales(
            q,
            config,
            blocks.block_t,
            blocks.block_h,
            blocks.block_w
        );

    NGLRMetaData meta;
    meta.nglr_correction_occur = true;
    meta.target = config.target_nrmse;
    meta.mean = mean;
    meta.scale = scale;
    meta.step = step;
    meta.q_scale = scales.first;
    meta.d_scale = scales.second;
    meta.block_t = blocks.block_t;
    meta.block_h = blocks.block_h;
    meta.block_w = blocks.block_w;
    meta.hidden = config.hidden;
    meta.q_hidden = config.q_hidden;
    meta.model_blocks = config.model_blocks;
    meta.train_epochs = config.train_epochs;
    meta.shape = shape;
    meta.original_bytes = x.numel() * x.element_size();

    const double base_nrmse = zero_nrmse(x, r, scale);
    const double quant_nrmse = decoded_nrmse(x, r_norm, q, step, mean, scale);
    if (base_nrmse <= config.target_nrmse) {
        CausalNeuralLorenzoNet model =
        CausalNeuralLorenzoNet(
            config.hidden,
            config.q_hidden,
            config.model_blocks
        );

    model->to(device);
    model->eval();

    NGLRTrainResult result;
    result.model = model;
    result.meta = meta;
    result.base_nrmse = base_nrmse;
    result.quant_nrmse = quant_nrmse;
    result.best_loss = 0.0;
    result.best_epoch = 0;

    if (config.verbose) {
        std::cout << "NGLR skipped training: base NRMSE already within target"
                  << std::endl;
    }

    return result;
}

    std::vector<BlockSlice> slices =
        block_slices(meta.shape, meta.block_t, meta.block_h, meta.block_w);

    CausalNeuralLorenzoNet model =
    CausalNeuralLorenzoNet(
        config.hidden,
        config.q_hidden,
        config.model_blocks
    );

CausalNeuralLorenzoNet best_model =
    CausalNeuralLorenzoNet(
        config.hidden,
        config.q_hidden,
        config.model_blocks
    );

model->to(device);
best_model->to(device);

model->train();
best_model->eval();

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(config.lr)
            .weight_decay(config.weight_decay)
    );

    if (config.verbose) {
        std::cout << "NGLR C++ training target: "
                  << config.target_nrmse
                  << " blocks: "
                  << slices.size()
                  << " step: "
                  << step
                  << std::endl;

        std::cout << "NGLR block shape: "
                  << meta.block_t << "x"
                  << meta.block_h << "x"
                  << meta.block_w
                  << std::endl;

        std::cout << "NGLR base NRMSE: "
                  << base_nrmse
                  << " quant NRMSE: "
                  << quant_nrmse
                  << std::endl;

        std::cout << "NGLR q_scale: "
                  << meta.q_scale
                  << " d_scale: "
                  << meta.d_scale
                  << std::endl;
    }

    std::vector<size_t> order(slices.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    std::mt19937 rng(config.seed);

    double best_loss = std::numeric_limits<double>::infinity();
    int best_epoch = 0;

    for (int epoch = 1; epoch <= config.train_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        model->train();

        double loss_sum = 0.0;
        double remain_sum = 0.0;

        for (size_t idx : order) {
            const BlockSlice& sl = slices[idx];

            torch::Tensor qb = slice_5d_to_3d_train(q, sl);
            torch::Tensor rb = slice_5d_to_3d_train(r_norm, sl);

            torch::Tensor d =
                lorenzo_delta_3d_train(qb).to(torch::kFloat32) /
                meta.d_scale;

            torch::Tensor r_t =
                rb.unsqueeze(0)
                  .unsqueeze(0)
                  .to(device)
                  .to(torch::kFloat32)
                  .contiguous();

            torch::Tensor c_t =
                q_context_3d_train(qb, meta.q_scale)
                    .unsqueeze(0)
                    .to(device)
                    .to(torch::kFloat32)
                    .contiguous();

            torch::Tensor d_t =
                d.unsqueeze(0)
                 .unsqueeze(0)
                 .to(device)
                 .to(torch::kFloat32)
                 .contiguous();

            torch::Tensor pred = model->forward(r_t, c_t);
            torch::Tensor remain = d_t - pred;
            torch::Tensor loss = charbonnier_loss(remain);

            optimizer.zero_grad();
            loss.backward();

            torch::nn::utils::clip_grad_norm_(
                model->parameters(),
                config.grad_clip
            );

            optimizer.step();

            loss_sum += loss.item<double>();
            remain_sum += remain.abs().mean().item<double>() * meta.d_scale;
        }

        const double denom =
            static_cast<double>(std::max<size_t>(1, slices.size()));

        const double avg_loss = loss_sum / denom;
        const double avg_remain = remain_sum / denom;

        if (avg_loss < best_loss) {
    best_loss = avg_loss;
    best_epoch = epoch;
    copy_model_weights(model, best_model);
}

        if (config.verbose) {
            std::cout << "NGLR epoch "
                      << epoch
                      << " loss "
                      << avg_loss
                      << " remain_abs "
                      << avg_remain
                      << " best_epoch "
                      << best_epoch
                      << std::endl;
        }
    }

    best_model->eval();

    NGLRTrainResult result;
    result.model = best_model;
    result.meta = meta;
    result.base_nrmse = base_nrmse;
    result.quant_nrmse = quant_nrmse;
    result.best_loss = best_loss;
    result.best_epoch = best_epoch;

    return result;
}

} // namespace caesar::nglr

// End of the CUDA training code. CPU-only builds use the simple stubs at the top.
#endif // !defined(USE_CUDA)
