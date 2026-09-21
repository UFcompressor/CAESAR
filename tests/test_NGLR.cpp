#include "nglr.h"
#include "nglr_training_stop.h"

#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void rejects(const std::function<void()> &call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception &) {
    rejected = true;
  }
  check(rejected, "invalid NGLR input was accepted");
}
} // namespace

int main() {
  try {
    nglr::detail::ZeroLossStop stop;
    check(!stop.observe(0.0), "stopped after one zero epoch");
    check(!stop.observe(0.0), "stopped after two zero epochs");
    check(!stop.observe(0.1), "nonzero epoch did not reset stopping");
    check(!stop.observe(0.0) && !stop.observe(0.0) && stop.observe(0.0),
          "did not stop after three consecutive zero epochs");
    nglr::detail::ZeroLossStop tiny;
    check(!tiny.observe(1e-20) && !tiny.observe(1e-20) && !tiny.observe(1e-20),
          "treated a nonzero loss as zero");
    torch::set_num_threads(1);
    torch::manual_seed(7);
    nglr::NGLRTrainOptions options;
    options.hidden = 8;
    options.q_hidden = 4;
    options.model_blocks = 1;
    options.epochs = 1;
    options.batch_size = 2;
    options.block_t = 2;
    options.block_h = 2;
    options.block_w = 2;
    auto x = torch::arange(54, torch::kFloat32).sin().reshape({1, 2, 3, 3, 3});
    auto r = torch::zeros_like(x);
    nglr::NGLRMetaData meta;
    std::vector<uint8_t> bytes;
    // Deliberately match Compressor's enclosing inference mode. Training must
    // re-enable autograd and own normal tensors without changing caller mode.
    {
      c10::InferenceMode inference;
      nglr::compress(x, r, 0.01, meta, bytes, options);
      check(c10::InferenceMode::is_enabled(),
            "training changed caller inference mode");
    }
    check(meta.correction_occurred && !meta.weights.empty() && !bytes.empty(),
          "training did not produce a model and correction");
    bool updated = false;
    for (const auto &w : meta.weights)
      if (w.name == "fusion.4.weight")
        for (float v : w.values)
          updated |= v != 0;
    check(updated, "optimizer did not update zero-initialized output weights");
    // Unit test of the correction codec, independent of any foundation model.
    auto decoded = nglr::decompress(r, meta, bytes);
    const auto &m = meta.quantization;
    auto rn = (r - m.x_mean) / m.scale;
    auto q = torch::round((((x - m.x_mean) / m.scale) - rn) / m.step);
    auto expected = (rn + q * m.step) * m.scale + m.x_mean;
    check(torch::equal(decoded, expected),
          "strict codec did not recover the quantized residual");
    auto trained = nglr::NGLRModel(meta);
    auto features = trained.encode_recons(r.slice(1, 0, 1));
    auto weights = meta.weights;
    weights[0].name = "missing";
    rejects([&] { trained.load_weights(weights); });
    weights = meta.weights;
    weights[0].values.pop_back();
    rejects([&] { trained.load_weights(weights); });
    auto truncated = bytes;
    truncated.pop_back();
    rejects([&] { nglr::decompress(r, meta, truncated); });
    auto extra = bytes;
    extra.push_back(0);
    rejects([&] { nglr::decompress(r, meta, extra); });
    auto badmeta = meta;
    ++badmeta.schema_version;
    rejects([&] { nglr::decompress(r, badmeta, bytes); });
    badmeta = meta;
    badmeta.weights.clear();
    rejects([&] { nglr::decompress(r, badmeta, bytes); });
    rejects([&] { nglr::compress(x, r, 0, meta, bytes, options); });
    auto badoptions = options;
    badoptions.epochs = 0;
    rejects([&] { nglr::compress(x, r, 0.01, meta, bytes, badoptions); });
    nglr::compress(x, x, 0.01, meta, bytes, options);
    check(!meta.correction_occurred && meta.weights.empty() && bytes.empty(),
          "no-correction case retained a model");
    check(torch::equal(nglr::decompress(x, meta, bytes), x),
          "no-correction path changed input");
    auto constant = torch::full_like(x, 3.25);
    nglr::compress(constant, r, 0.01, meta, bytes, options);
    check(meta.constant_input && bytes.empty(),
          "constant field was not handled");
    check(torch::equal(nglr::decompress(r, meta, bytes), constant),
          "constant field changed");
    std::cout << "NGLR training, weights and correction unit tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
