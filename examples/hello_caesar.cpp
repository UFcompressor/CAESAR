#include "models/caesar_compress.h"
#include "models/caesar_decompress.h"

#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
  try {
    const std::string method = argc > 1 ? argv[1] : "gae";
    if (method != "gae" && method != "lbrc")
      throw std::invalid_argument("Usage: hello_caesar [gae|lbrc]");

    // A simple scientific field with shape [time, height, width].
    auto t = torch::linspace(0, 1, 8).view({8, 1, 1});
    auto y = torch::linspace(0, 6.2831853, 256).view({1, 256, 1});
    auto x = torch::linspace(0, 6.2831853, 256).view({1, 1, 256});
    auto data = (1 + t) * torch::sin(y) * torch::cos(x);

    CompressionConfig config;
    config.memory_data = data;
    config.n_frame = 8;
    config.correction_method = method == "gae" ? caesar::CorrectionMethod::GAE
                                               : caesar::CorrectionMethod::LBRC;
    // Mean/range normalization and its inverse are handled internally.
    const float target_nrmse = 1e-3f;
    Compressor compressor;
    auto compressed = compressor.compress(config, target_nrmse);
    Decompressor decompressor;
    auto reconstructed = decompressor.decompress(compressed).cpu();

    if (reconstructed.sizes() != data.sizes())
      throw std::runtime_error("Reconstructed shape differs from input");
    auto reference = data.to(torch::kFloat64);
    auto difference = reconstructed.to(torch::kFloat64) - reference;
    const double range = (reference.max() - reference.min()).item<double>();
    const double nrmse =
        std::sqrt(difference.square().mean().item<double>()) / range;
    const bool passed = std::isfinite(nrmse) && nrmse <= target_nrmse;
    std::cout << "Correction: " << method
              << "\nShape: " << reconstructed.sizes()
              << "\nTarget NRMSE: " << std::scientific << std::setprecision(8)
              << target_nrmse << "\nMeasured NRMSE: " << nrmse << "\n"
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
