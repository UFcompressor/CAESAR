#include "../CAESAR/models/nglr_model.h"
#include "../CAESAR/models/nglr_train.h"

#include <torch/torch.h>

#if defined(USE_CUDA)
#include <torch/cuda.h>
#endif

#if __has_include(<torch/mps.h>)
#include <torch/mps.h>
#endif

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void run_device(const torch::Device& device) {
  torch::manual_seed(2026);

  torch::Tensor x =
      torch::linspace(-1.0, 1.0, 1 * 1 * 4 * 8 * 8, torch::kFloat32)
          .reshape({1, 1, 4, 8, 8});
  torch::Tensor r = (x + 0.02 * torch::sin(x * 17.0)).contiguous();

  caesar::nglr::NGLRTrainConfig cfg =
      caesar::nglr::default_train_config(1e-3);
  cfg.block_t = 4;
  cfg.block_h = 8;
  cfg.block_w = 8;
  cfg.hidden = 8;
  cfg.q_hidden = 4;
  cfg.model_blocks = 1;
  cfg.train_epochs = 1;
  cfg.verbose = false;

  auto trained = caesar::nglr::train_nglr_model(x, r, cfg, device);
  if (!trained.correction_required) {
    throw std::runtime_error("NGLR smoke skipped correction");
  }

  auto encoded =
      caesar::nglr::encode_correction(x, r, trained.model, trained.meta, device);
  torch::Tensor decoded =
      caesar::nglr::decompress(r, encoded.meta, encoded.compressed, device);

  const double rmse =
      (decoded.to(torch::kCPU) - x).pow(2).mean().sqrt().item<double>();
  if (!std::isfinite(rmse)) {
    throw std::runtime_error("NGLR smoke produced non-finite RMSE");
  }

  std::cout << "NGLR smoke " << device << " RMSE=" << rmse << std::endl;
}

} // namespace

int main() {
  std::vector<torch::Device> devices{torch::Device(torch::kCPU)};

#if defined(USE_CUDA)
  if (torch::cuda::is_available()) {
    devices.emplace_back(torch::kCUDA, 0);
  }
#endif

#if __has_include(<torch/mps.h>)
  if (torch::mps::is_available()) {
    devices.emplace_back(torch::kMPS);
  }
#endif

  for (const torch::Device& device : devices) {
    run_device(device);
  }

  return 0;
}
