#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nglr.h"

namespace {

torch::Tensor load_tensor(const std::string &path) {
  torch::Tensor tensor;
  try {
    torch::load(tensor, path);
  } catch (const c10::Error &e) {
    throw std::runtime_error("could not load tensor '" + path +
                             "': " + e.what());
  }
  if (!tensor.defined())
    throw std::runtime_error("file did not contain a tensor: " + path);
  return tensor;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <model_dir> <original_tensor.pt> <recons_tensor.pt>\n";
    return 2;
  }

  try {
    const auto bundle = nglr::NGLRBundle::load(argv[1]);
    const auto original = load_tensor(argv[2]).to(torch::kFloat32).contiguous();
    const auto recons = load_tensor(argv[3]).to(torch::kFloat32).contiguous();

    std::vector<uint8_t> correction;
    const auto stats = nglr::nglr_encode(original, recons, bundle.model,
                                         bundle.meta, correction);
    const auto decoded =
        nglr::nglr_decode(recons, bundle.model, bundle.meta, correction);

    const auto error = (decoded - original).abs();
    std::cout << "correction_bytes: " << stats.correction_bytes << '\n'
              << "uncompressed_bitplane_bytes: "
              << stats.uncompressed_bitplane_bytes << '\n'
              << "blocks: " << stats.block_count << '\n'
              << "max_abs_error: " << error.max().item<double>() << '\n'
              << "mean_abs_error: " << error.mean().item<double>() << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "test_NGLR: " << e.what() << '\n';
    return 1;
  }
}
