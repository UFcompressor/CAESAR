#include "nglr.h"
#include "nglr_training_stop.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <zstd.h>

namespace nglr {
namespace {
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::invalid_argument("nglr: " + message);
}

void validate_architecture(int32_t hidden, int32_t q_hidden, int32_t blocks) {
  require(hidden >= 8 && hidden <= 256 && hidden % 4 == 0 && q_hidden > 0 &&
              q_hidden <= 256 && blocks >= 0 && blocks <= 32,
          "invalid network architecture");
}

void validate_input(const torch::Tensor &x) {
  require(x.defined() && x.dim() == 5 && x.numel() > 0,
          "expected a nonempty [B,C,T,H,W] tensor");
  require(x.scalar_type() == torch::kFloat32,
          "NGLR currently requires float32 input");
  require(torch::isfinite(x).all().item<bool>(), "input must be finite");
}

struct Block {
  int64_t b, c, t, h, w, nt, nh, nw;
};
std::vector<Block> blocks(const torch::Tensor &x, const NGLRMeta &m) {
  std::vector<Block> out;
  for (int64_t b = 0; b < x.size(0); ++b)
    for (int64_t c = 0; c < x.size(1); ++c)
      for (int64_t t = 0; t < x.size(2); t += m.block_t)
        for (int64_t h = 0; h < x.size(3); h += m.block_h)
          for (int64_t w = 0; w < x.size(4); w += m.block_w)
            out.push_back({b, c, t, h, w, std::min(m.block_t, x.size(2) - t),
                           std::min(m.block_h, x.size(3) - h),
                           std::min(m.block_w, x.size(4) - w)});
  return out;
}
torch::Tensor block(const torch::Tensor &x, const Block &b) {
  using torch::indexing::Slice;
  return x
      .index({b.b, b.c, Slice(b.t, b.t + b.nt), Slice(b.h, b.h + b.nh),
              Slice(b.w, b.w + b.nw)})
      .contiguous();
}

// Teacher-forced causal channels, in the same order as strict diagonal decode.
torch::Tensor context(const torch::Tensor &q) {
  auto padded =
      torch::zeros({q.size(0) + 1, q.size(1) + 1, q.size(2) + 1}, q.options());
  padded.slice(0, 1).slice(1, 1).slice(2, 1).copy_(q);
  std::vector<torch::Tensor> channels;
  for (auto offset : {std::array<int, 3>{0, 1, 1},
                      {1, 0, 1},
                      {1, 1, 0},
                      {0, 0, 1},
                      {0, 1, 0},
                      {1, 0, 0},
                      {0, 0, 0}})
    channels.push_back(padded.slice(0, offset[0], offset[0] + q.size(0))
                           .slice(1, offset[1], offset[1] + q.size(1))
                           .slice(2, offset[2], offset[2] + q.size(2)));
  channels.push_back(channels[0] + channels[1] + channels[2] - channels[3] -
                     channels[4] - channels[5] + channels[6]);
  return torch::stack(channels);
}

torch::Tensor quantize(const torch::Tensor &residual, double step) {
  auto q = torch::round(residual / step);
  require(torch::isfinite(q).all().item<bool>() &&
              q.to(torch::kFloat64).abs().max().item<double>() <= INT32_MAX,
          "quantized residual exceeds int32 range");
  return q.to(torch::kInt32);
}

double error(const torch::Tensor &x, const torch::Tensor &decoded,
             double scale) {
  return ((x.to(torch::kFloat64) - decoded.to(torch::kFloat64)) / scale)
      .square()
      .mean()
      .sqrt()
      .item<double>();
}
} // namespace

void validate_options(const NGLRTrainOptions &o) {
  validate_architecture(o.hidden, o.q_hidden, o.model_blocks);
  require(o.block_t > 0 && o.block_h > 0 && o.block_w > 0,
          "training blocks must be positive");
  require(o.epochs > 0 && o.batch_size > 0 && o.quant_iters > 0 &&
              o.quant_iters <= 64,
          "epochs, batch size and quantization iterations must be positive "
          "(iterations <= 64)");
  require(std::isfinite(o.learning_rate) && o.learning_rate > 0 &&
              std::isfinite(o.weight_decay) && o.weight_decay >= 0 &&
              std::isfinite(o.grad_clip) && o.grad_clip > 0,
          "invalid optimizer settings");
  require(o.zstd_level >= ZSTD_minCLevel() && o.zstd_level <= ZSTD_maxCLevel(),
          "invalid Zstd level");
}

void validate_metadata(const NGLRMetaData &m) {
  require(m.schema_version == 1, "unsupported NGLR metadata version");
  validate_meta(m.quantization);
  validate_architecture(m.hidden, m.q_hidden, m.model_blocks);
  require(m.shape.size() == 5 && std::all_of(m.shape.begin(), m.shape.end(),
                                             [](int64_t n) { return n > 0; }),
          "invalid NGLR input shape");
  require(m.correction_occurred ? !m.weights.empty() : m.weights.empty(),
          "NGLR weights disagree with correction flag");
  require(!m.constant_input || !m.correction_occurred,
          "constant input cannot have a neural correction");
}

NGLRModel::NGLRModel(int32_t hidden, int32_t q_hidden, int32_t blocks,
                     c10::Device device)
    : device_(device) {
  validate_architecture(hidden, q_hidden, blocks);
  network_ = CausalNeuralLorenzoNet(hidden, q_hidden, blocks);
  network_->to(device_);
  network_->eval();
}
NGLRModel::NGLRModel(const NGLRMetaData &m, c10::Device device)
    : NGLRModel(m.hidden, m.q_hidden, m.model_blocks, device) {
  validate_metadata(m);
  require(m.correction_occurred, "no trained model in this correction");
  load_weights(m.weights);
}
torch::Tensor NGLRModel::encode_recons(const torch::Tensor &r) const {
  torch::NoGradGuard guard;
  return network()->encode_recons(r);
}
torch::Tensor
NGLRModel::forward_from_recons_feature(const torch::Tensor &rf,
                                       const torch::Tensor &ctx) const {
  torch::NoGradGuard guard;
  return network()->forward_from_feature(rf, ctx);
}
std::vector<NGLRWeight> NGLRModel::weights() const {
  std::vector<NGLRWeight> out;
  for (const auto &p : network_->named_parameters()) {
    auto cpu = p.value().detach().to(torch::kCPU).contiguous();
    const auto *ptr = cpu.data_ptr<float>();
    out.push_back({p.key(), cpu.sizes().vec(), {ptr, ptr + cpu.numel()}});
  }
  return out;
}
void NGLRModel::load_weights(const std::vector<NGLRWeight> &weights) {
  torch::NoGradGuard guard;
  auto parameters = network_->named_parameters();
  require(weights.size() == parameters.size(),
          "model parameter count mismatch");
  std::set<std::string> seen;
  for (const auto &w : weights) {
    require(parameters.contains(w.name) && seen.insert(w.name).second,
            "unknown or duplicate model parameter: " + w.name);
    auto p = parameters[w.name];
    require(p.sizes().vec() == w.shape &&
                static_cast<size_t>(p.numel()) == w.values.size(),
            "model parameter shape mismatch: " + w.name);
    require(std::all_of(w.values.begin(), w.values.end(),
                        [](float v) { return std::isfinite(v); }),
            "nonfinite model parameter: " + w.name);
    p.copy_(torch::from_blob(const_cast<float *>(w.values.data()), w.shape,
                             torch::kFloat32)
                .to(device_));
  }
}

void compress(const torch::Tensor &original, const torch::Tensor &recons,
              double target, NGLRMetaData &metadata,
              std::vector<uint8_t> &correction, const NGLRTrainOptions &options,
              c10::Device training_device, c10::Device codec_device) {
  validate_options(options);
  require(std::isfinite(target) && target > 0,
          "target NRMSE must be finite and positive");
  validate_input(original);
  validate_input(recons);
  require(original.sizes() == recons.sizes(),
          "original/reconstruction shape mismatch");
  // CAESAR calls correction under inference mode. Clone after disabling it so
  // training never saves inference tensors for backward.
  c10::InferenceMode inference(false);
  torch::AutoGradMode grad(true);
  auto x = original.detach().to(torch::kCPU).clone();
  auto r = recons.detach().to(torch::kCPU).clone();
  NGLRMetaData m;
  m.shape = x.sizes().vec();
  m.hidden = options.hidden;
  m.q_hidden = options.q_hidden;
  m.model_blocks = options.model_blocks;
  auto &qm = m.quantization;
  qm.x_mean = x.to(torch::kFloat64).mean().item<double>();
  qm.scale = x.max().item<double>() - x.min().item<double>();
  qm.block_t = std::min(options.block_t, x.size(2));
  qm.block_h = std::min(options.block_h, x.size(3));
  qm.block_w = std::min(options.block_w, x.size(4));
  if (qm.scale == 0) {
    qm.scale = 1;
    m.constant_input = true;
    metadata = std::move(m);
    correction.clear();
    return;
  }
  if (error(x, r, qm.scale) <= target) {
    metadata = std::move(m);
    correction.clear();
    return;
  }
  const auto rn = ((r - qm.x_mean) / qm.scale).contiguous();
  const auto residual = (((x - qm.x_mean) / qm.scale) - rn).contiguous();
  auto decoded_error = [&](double step) {
    auto q = quantize(residual, step);
    auto decoded = (rn + q.to(torch::kFloat32) * step) * qm.scale + qm.x_mean;
    return error(x, decoded, qm.scale);
  };
  // Find a verified feasible step first. Finite iteration limits prevent a
  // rounding floor or unrepresentable residual from hanging compression.
  double low = target * 2.0;
  int attempts = 0;
  while (decoded_error(low) > target && attempts++ < 32)
    low *= 0.5;
  require(decoded_error(low) <= target,
          "requested bound is below float32 reconstruction precision");
  double high = std::max(low * 2.0, target * std::sqrt(12.0));
  for (int i = 0; i < 32 && decoded_error(high) <= target; ++i) {
    low = high;
    high *= 2.0;
  }
  for (int i = 0; i < options.quant_iters; ++i) {
    const double mid = low + (high - low) * 0.5;
    if (decoded_error(mid) <= target)
      low = mid;
    else
      high = mid;
  }
  qm.step = low;
  auto q = quantize(residual, qm.step).to(torch::kInt64);
  const auto slices = blocks(q, qm);
  double delta_sum = 0;
  std::map<std::array<int64_t, 3>, std::vector<size_t>> groups;
  for (size_t i = 0; i < slices.size(); ++i) {
    auto qb = block(q, slices[i]);
    delta_sum +=
        (qb - context(qb)[7]).to(torch::kFloat64).abs().sum().item<double>();
    groups[{slices[i].nt, slices[i].nh, slices[i].nw}].push_back(i);
  }
  qm.q_context_scale =
      std::max(1.0, q.to(torch::kFloat64).abs().mean().item<double>());
  qm.delta_scale = std::max(1.0, delta_sum / q.numel());
  NGLRModel fitted(m.hidden, m.q_hidden, m.model_blocks, training_device);
  auto net = fitted.network();
  torch::optim::AdamW optimizer(
      net->parameters(), torch::optim::AdamWOptions(options.learning_rate)
                             .weight_decay(options.weight_decay));
  std::mt19937 rng(
      2026); // local shuffle; do not reseed the process-wide Torch RNG
  double best_loss = std::numeric_limits<double>::infinity();
  std::vector<NGLRWeight> best;
  detail::ZeroLossStop zero_loss_stop;
  for (int epoch = 0; epoch < options.epochs; ++epoch) {
    net->train();
    double loss_sum = 0;
    int64_t count = 0;
    for (auto &group : groups) {
      auto &order = group.second;
      std::shuffle(order.begin(), order.end(), rng);
      for (size_t first = 0; first < order.size();
           first += options.batch_size) {
        std::vector<torch::Tensor> rs, cs, ds;
        for (size_t j = first;
             j < std::min(order.size(), first + options.batch_size); ++j) {
          const auto &s = slices[order[j]];
          auto qb = block(q, s);
          auto ctx = context(qb);
          rs.push_back(block(rn, s).unsqueeze(0));
          cs.push_back(ctx.to(torch::kFloat32) / qm.q_context_scale);
          ds.push_back(((qb - ctx[7]).to(torch::kFloat32) / qm.delta_scale)
                           .unsqueeze(0));
        }
        auto rb = torch::stack(rs).to(training_device);
        auto cb = torch::stack(cs).to(training_device);
        auto db = torch::stack(ds).to(training_device);
        optimizer.zero_grad();
        auto remain = db - net->forward(rb, cb);
        auto loss = ((remain.square() + 1e-12).sqrt() - 1e-6).mean();
        require(torch::isfinite(loss).item<bool>(), "nonfinite training loss");
        loss.backward();
        torch::nn::utils::clip_grad_norm_(net->parameters(), options.grad_clip,
                                          2.0, true);
        optimizer.step();
        loss_sum += loss.item<double>() * db.numel();
        count += db.numel();
      }
    }
    if (loss_sum / count < best_loss) {
      best_loss = loss_sum / count;
      best = fitted.weights();
    }
    if (zero_loss_stop.observe(loss_sum / count))
      break;
  }
  m.weights = std::move(best);
  m.correction_occurred = true;
  // Decode must use a compatible predictor device with these exact weights.
  NGLRModel codec(m, codec_device);
  std::vector<uint8_t> encoded;
  nglr_encode(x, r, codec, qm, encoded, options.zstd_level);
  metadata = std::move(m);
  correction = std::move(encoded);
}

torch::Tensor decompress(const torch::Tensor &recons,
                         const NGLRMetaData &metadata,
                         const std::vector<uint8_t> &correction,
                         c10::Device codec_device) {
  validate_metadata(metadata);
  validate_input(recons);
  require(recons.sizes().vec() == metadata.shape,
          "reconstruction shape differs from NGLR metadata");
  if (!metadata.correction_occurred) {
    require(correction.empty(),
            "unexpected correction bytes for an uncorrected input");
    return metadata.constant_input
               ? torch::full_like(recons, metadata.quantization.x_mean)
               : recons;
  }
  require(!correction.empty(), "missing NGLR correction bytes");
  NGLRModel model(metadata, codec_device);
  return nglr_decode(recons, model, metadata.quantization, correction);
}
} // namespace nglr
