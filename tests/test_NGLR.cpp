#include "../CAESAR/models/nglr_model.h"

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

  torch::Tensor q =
      (torch::arange(1 * 1 * 4 * 8 * 8, torch::kInt32).reshape({1, 1, 4, 8, 8}) %
       7) -
      3;
  torch::Tensor r = torch::zeros({1, 1, 4, 8, 8}, torch::kFloat32);
  torch::Tensor x = q.to(torch::kFloat32) * 0.01f;

  caesar::nglr::NGLRMetaData meta;
  meta.mean = 0.0;
  meta.scale = 1.0;
  meta.step = 0.01;
  meta.q_scale = 3.0;
  meta.d_scale = 1.0;
  meta.block_t = 4;
  meta.block_h = 8;
  meta.block_w = 8;
  meta.hidden = 8;
  meta.q_hidden = 4;
  meta.model_blocks = 1;
  meta.shape = x.sizes().vec();

  caesar::nglr::CausalNeuralLorenzoNet model =
      caesar::nglr::CausalNeuralLorenzoNet(
          meta.hidden,
          meta.q_hidden,
          meta.model_blocks);

  auto encoded =
      caesar::nglr::encode_correction(x, r, model, meta, device);
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
