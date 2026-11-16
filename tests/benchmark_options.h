#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace caesar_test {
inline bool gpu_required() {
  const char *value = std::getenv("CAESAR_REQUIRE_GPU");
  return value && std::string(value) == "1";
}
inline torch::Device test_device() {
  if (!gpu_required())
    return torch::Device(torch::kCPU);
  if (!torch::cuda::is_available())
    throw std::runtime_error(
        "GPU testing requires CUDA/ROCm; CPU fallback rejected");
  return torch::Device(torch::kCUDA);
}
inline void require_gpu(const torch::Device &device) {
  if (gpu_required() && (!device.is_cuda() || !torch::cuda::is_available()))
    throw std::runtime_error(
        "Compiled model must use CUDA/ROCm for GPU testing");
}
inline void synchronize(const torch::Device &device) {
  if (device.is_cuda())
    torch::cuda::synchronize(device.index());
}
inline std::vector<int64_t> shape() {
  const char *value = std::getenv("CAESAR_BENCH_SHAPE");
  if (!value)
    return {1, 1, 20, 256, 256};
  std::vector<int64_t> dims;
  std::istringstream input(value);
  std::string part;
  uint64_t elements = 1;
  while (std::getline(input, part, ',')) {
    size_t used = 0;
    auto dim = std::stoll(part, &used);
    if (used != part.size() || dim <= 0 ||
        static_cast<uint64_t>(dim) > (UINT64_MAX / sizeof(float)) / elements)
      throw std::runtime_error("Invalid benchmark shape");
    elements *= dim;
    dims.push_back(dim);
  }
  if (dims.size() < 2 || dims.size() > 5)
    throw std::runtime_error("Benchmark requires 2 through 5 dimensions");
  return dims;
}
inline std::string input_path() {
  const char *value = std::getenv("CAESAR_BENCH_DATA");
  return value ? value : "TCf48.bin.f32";
}
inline void metric(const char *operation, double seconds, uint64_t bytes) {
  if (!(seconds > 0) || !std::isfinite(seconds))
    throw std::runtime_error("Invalid benchmark duration");
  std::cout << "CAESAR_BENCHMARK " << operation << " " << std::setprecision(17)
            << seconds << " " << bytes << "\n";
}
} // namespace caesar_test
