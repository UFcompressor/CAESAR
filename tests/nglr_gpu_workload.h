#pragma once
#include "benchmark_options.h"
#include "nglr.h"

// A realistic correction-codec workload, separate from the tiny unit tests.
inline void test_nglr_gpu_workload() {
  const auto device = caesar_test::test_device();
  if (!device.is_cuda())
    throw std::runtime_error("NGLR GPU workload requires CAESAR_REQUIRE_GPU=1");
  torch::manual_seed(7);
  const std::vector<int64_t> shape{1, 1, 100, 500, 500};
  auto original = (torch::arange(100 * 500 * 500, torch::kFloat32) * 0.001)
                      .sin()
                      .reshape(shape);
  auto reconstruction = original * 0.95;
  nglr::NGLRTrainOptions options;
  options.hidden = 8;
  options.q_hidden = 4;
  options.model_blocks = 1;
  options.epochs = 1;
  options.batch_size = 1;
  options.block_t = 8;
  options.block_h = 256;
  options.block_w = 256;
  nglr::NGLRMetaData metadata;
  std::vector<uint8_t> bytes;
  constexpr double bound = 0.01;
  std::cout << "NGLR GPU workload: 100x500x500, blocks 8x256x256, one epoch\n";
  nglr::compress(original, reconstruction, bound, metadata, bytes, options,
                 device, device);
  auto decoded = nglr::decompress(reconstruction, metadata, bytes, device);
  if (decoded.sizes() != original.sizes() ||
      !torch::isfinite(decoded).all().item<bool>())
    throw std::runtime_error("NGLR GPU workload produced invalid output");
  const double range = (original.max() - original.min()).item<double>();
  const double nrmse =
      (decoded.cpu() - original).square().mean().sqrt().item<double>() / range;
  if (!std::isfinite(nrmse) || nrmse > bound)
    throw std::runtime_error("NGLR GPU workload exceeded its error bound");
  if (!metadata.correction_occurred || metadata.weights.empty() ||
      bytes.empty())
    throw std::runtime_error(
        "NGLR GPU workload did not exercise training and coding");
  std::cout << "NGLR GPU workload passed; NRMSE=" << nrmse << "\n";
}
