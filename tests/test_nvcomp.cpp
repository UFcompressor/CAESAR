#include "models/gpu_lossess.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    if (!torch::cuda::is_available())
      throw std::runtime_error("An NVIDIA CUDA GPU is required");
    // Exercise both a single chunk and CAESAR's multi-chunk frame.
    std::vector<torch::Tensor> inputs;
    for (int64_t bytes : {int64_t(4096), int64_t(200000)})
      inputs.push_back(torch::randint(0, 256, {bytes},
          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)));
    auto compressed = nvcomp_batch_compress(inputs);
    std::vector<const uint8_t *> pointers;
    std::vector<size_t> sizes, raw_sizes;
    for (const auto &item : compressed) {
      pointers.push_back(item.compressed.data_ptr<uint8_t>());
      sizes.push_back(item.compressed.numel());
      raw_sizes.push_back(item.rawBytes);
    }
    auto decoded = nvcomp_batch_decompress(pointers, sizes, raw_sizes);
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto original = inputs[i].cpu();
      if (decoded[i].size() != raw_sizes[i] ||
          !std::equal(decoded[i].begin(), decoded[i].end(), original.data_ptr<uint8_t>()))
        throw std::runtime_error("nvCOMP round-trip mismatch");
    }
    std::cout << "PASS: nvCOMP Zstd GPU round trips (single and multiple chunks)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
