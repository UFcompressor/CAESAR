#include "rans_cuda.h"
#include "rans_coder.hpp"
#include "rans_cuda_kernels.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#ifdef CAESAR_CUDA_RANS
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace caesar::rans_cuda {
namespace {
std::string mode() {
  const char *value = std::getenv("CAESAR_RANS");
  std::string result = value ? value : "cpu";
  TORCH_CHECK(result == "cpu" || result == "cuda" || result == "verify",
              "CAESAR_RANS must be cpu, cuda, or verify");
  return result;
}
#ifdef CAESAR_CUDA_RANS
void check_errors(const torch::Tensor &errors) {
  auto host = errors.cpu(); // also completes the current-stream kernel
  const int32_t *p = host.data_ptr<int32_t>();
  for (int64_t i = 0; i < host.numel(); ++i)
    TORCH_CHECK(p[i] == 0, "CUDA rANS failure in stream ", i, ": ",
                p[i] == 1   ? "invalid CDF index"
                : p[i] == 2 ? "symbol outside CPU-compatible bypass range"
                            : "invalid/truncated encoded stream");
}
void check_shape(const torch::Tensor &t, torch::Device device) {
  TORCH_CHECK(t.device() == device && t.scalar_type() == torch::kInt32,
              "CUDA rANS expects int32 tensors on ", device);
  TORCH_CHECK(t.dim() >= 2 && t.size(0) > 0,
              "CUDA rANS needs nonempty stream rows");
  TORCH_CHECK(t.size(0) <= INT32_MAX &&
                  t.numel() / t.size(0) <= (INT32_MAX - 4) / 2,
              "CUDA rANS tensor too large");
}
#endif
} // namespace
bool verification_enabled() { return mode() == "verify"; }
bool enabled(const torch::Device &device) {
  if (mode() == "cpu")
    return false;
#ifdef CAESAR_CUDA_RANS
  TORCH_CHECK(device.is_cuda(), "CAESAR_RANS=", mode(),
              " requires CUDA models/device");
  return true;
#else
  TORCH_CHECK(
      false,
      "CUDA rANS was not built. Configure with -DCAESAR_ENABLE_CUDA_RANS=ON");
#endif
}

Codec::Codec(const std::vector<std::vector<int32_t>> &cdfs,
             const std::vector<int32_t> &lengths,
             const std::vector<int32_t> &offsets, torch::Device device)
    : device_(device), cdfs_(cdfs), lengths_(lengths), offsets_(offsets) {
#ifdef CAESAR_CUDA_RANS
  TORCH_CHECK(device.is_cuda(), "CUDA rANS requires CUDA device");
  c10::cuda::CUDAGuard guard(device);
  device_ = torch::Device(torch::kCUDA, c10::cuda::current_device());
  TORCH_CHECK(!cdfs.empty() && cdfs.size() == lengths.size() &&
                  cdfs.size() == offsets.size(),
              "Invalid rANS table dimensions");
  std::vector<int32_t> flat, rows{0};
  for (size_t r = 0; r < cdfs.size(); ++r) {
    int n = lengths[r];
    TORCH_CHECK(n >= 3 && size_t(n) <= cdfs[r].size(), "Invalid CDF length");
    TORCH_CHECK(cdfs[r][0] == 0 && cdfs[r][n - 1] == 65536,
                "Invalid CDF endpoints");
    for (int j = 1; j < n; ++j)
      TORCH_CHECK(cdfs[r][j] > cdfs[r][j - 1],
                  "CDF must be strictly increasing");
    TORCH_CHECK(flat.size() + n <= INT32_MAX, "CDF table too large");
    flat.insert(flat.end(), cdfs[r].begin(), cdfs[r].begin() + n);
    rows.push_back(int32_t(flat.size()));
  }
  cdf_gpu_ = torch::tensor(flat, torch::kInt32).to(device_);
  rows_gpu_ = torch::tensor(rows, torch::kInt32).to(device_);
  offsets_gpu_ = torch::tensor(offsets, torch::kInt32).to(device_);
#else
  TORCH_CHECK(false, "CUDA rANS unavailable in this build");
#endif
}

std::vector<std::string> Codec::encode(const torch::Tensor &symbols,
                                       const torch::Tensor &indexes) const {
#ifdef CAESAR_CUDA_RANS
  c10::cuda::CUDAGuard guard(device_);
  check_shape(symbols, device_);
  check_shape(indexes, device_);
  TORCH_CHECK(symbols.sizes() == indexes.sizes(),
              "rANS symbols/indexes shape mismatch");
  // Bound scratch usage when compression collects many model batches.
  if (symbols.size(0) > 1024) {
    std::vector<std::string> result;
    result.reserve(symbols.size(0));
    for (int64_t start = 0; start < symbols.size(0); start += 1024) {
      int64_t count = std::min<int64_t>(1024, symbols.size(0) - start);
      auto part = encode(symbols.narrow(0, start, count),
                         indexes.narrow(0, start, count));
      result.insert(result.end(), std::make_move_iterator(part.begin()),
                    std::make_move_iterator(part.end()));
    }
    return result;
  }
  auto syms = symbols.contiguous(), idx = indexes.contiguous();
  int n = int(syms.size(0)), size = int(syms.numel() / n),
      capacity = 2 * size + 4;
  auto words = torch::empty({n, capacity}, syms.options());
  auto lengths = torch::empty({n}, syms.options());
  auto errors = torch::empty({n}, syms.options());
  launch_encode(syms.data_ptr<int32_t>(), idx.data_ptr<int32_t>(),
                cdf_gpu_.data_ptr<int32_t>(), rows_gpu_.data_ptr<int32_t>(),
                offsets_gpu_.data_ptr<int32_t>(), int(cdfs_.size()),
                reinterpret_cast<uint32_t *>(words.data_ptr<int32_t>()),
                lengths.data_ptr<int32_t>(), errors.data_ptr<int32_t>(), n,
                size, capacity,
                c10::cuda::getCurrentCUDAStream(device_.index()).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  check_errors(errors);
  auto host_words = words.cpu(), host_lengths = lengths.cpu();
  const int32_t *data = host_words.data_ptr<int32_t>();
  const int32_t *lens = host_lengths.data_ptr<int32_t>();
  std::vector<std::string> result(n);
  for (int i = 0; i < n; ++i) {
    TORCH_CHECK(lens[i] >= 2 && lens[i] <= capacity, "Invalid encoded length");
    result[i].assign(reinterpret_cast<const char *>(
                         data + int64_t(i) * capacity + capacity - lens[i]),
                     size_t(lens[i]) * 4);
  }
  if (verification_enabled()) {
    auto hs = syms.cpu(), hi = idx.cpu();
    const auto *sp = hs.data_ptr<int32_t>();
    const auto *ip = hi.data_ptr<int32_t>();
    for (int i = 0; i < n; ++i) {
      RansEncoder cpu;
      int64_t start = int64_t(i) * size;
      TORCH_CHECK(result[i] ==
                      cpu.encode_with_indexes(
                          std::vector<int32_t>(sp + start, sp + start + size),
                          std::vector<int32_t>(ip + start, ip + start + size),
                          cdfs_, lengths_, offsets_),
                  "CUDA rANS encoded bytes differ from CPU at stream ", i);
    }
    std::cout << "[rANS verify] encode: " << n
              << " streams match CPU byte-for-byte\n";
  }
  return result;
#else
  TORCH_CHECK(false, "CUDA rANS unavailable in this build");
#endif
}

torch::Tensor Codec::decode(const std::vector<std::string> &strings,
                            size_t begin, const torch::Tensor &indexes) const {
#ifdef CAESAR_CUDA_RANS
  c10::cuda::CUDAGuard guard(device_);
  check_shape(indexes, device_);
  auto idx = indexes.contiguous();
  int n = int(idx.size(0)), size = int(idx.numel() / n);
  TORCH_CHECK(begin <= strings.size() && size_t(n) <= strings.size() - begin,
              "Missing rANS streams");
  std::vector<int64_t> positions{0};
  for (int i = 0; i < n; ++i) {
    const auto &s = strings[begin + i];
    TORCH_CHECK(s.size() >= 8 && s.size() % 4 == 0,
                "Invalid rANS stream byte length");
    TORCH_CHECK(s.size() / 4 <= size_t(INT64_MAX - positions.back()),
                "rANS input too large");
    positions.push_back(positions.back() + int64_t(s.size() / 4));
  }
  auto host = torch::empty({positions.back()}, torch::kInt32);
  for (int i = 0; i < n; ++i)
    std::memcpy(host.data_ptr<int32_t>() + positions[i],
                strings[begin + i].data(), strings[begin + i].size());
  auto words = host.to(device_);
  auto pos = torch::tensor(positions, torch::kInt64).to(device_);
  auto result = torch::empty_like(idx);
  auto errors = torch::empty({n}, idx.options());
  launch_decode(reinterpret_cast<const uint32_t *>(words.data_ptr<int32_t>()),
                pos.data_ptr<int64_t>(), idx.data_ptr<int32_t>(),
                cdf_gpu_.data_ptr<int32_t>(), rows_gpu_.data_ptr<int32_t>(),
                offsets_gpu_.data_ptr<int32_t>(), int(cdfs_.size()),
                result.data_ptr<int32_t>(), errors.data_ptr<int32_t>(), n, size,
                c10::cuda::getCurrentCUDAStream(device_.index()).stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  check_errors(errors);
  if (verification_enabled()) {
    auto hi = idx.cpu(), decoded = result.cpu();
    const auto *ip = hi.data_ptr<int32_t>();
    const auto *dp = decoded.data_ptr<int32_t>();
    for (int i = 0; i < n; ++i) {
      RansDecoder cpu;
      int64_t start = int64_t(i) * size;
      auto expected = cpu.decode_with_indexes(
          strings[begin + i],
          std::vector<int32_t>(ip + start, ip + start + size), cdfs_, lengths_,
          offsets_);
      TORCH_CHECK(std::equal(expected.begin(), expected.end(), dp + start),
                  "CUDA rANS decoded symbols differ from CPU at stream ", i);
    }
    std::cout << "[rANS verify] decode: " << n
              << " streams match CPU exactly\n";
  }
  return result;
#else
  TORCH_CHECK(false, "CUDA rANS unavailable in this build");
#endif
}
} // namespace caesar::rans_cuda
