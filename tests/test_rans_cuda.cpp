#include "../CAESAR/models/range_coder/rans_coder.hpp"
#include "../CAESAR/models/range_coder/rans_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>

int main() {
  try {
    torch::Device device(torch::kCUDA, 0);
    // Catch accidental launches on CUDA's default stream.
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromPool(false, 0));
    caesar::rans_cuda::Codec codec(
        {{0, 1, 32000, 65536}, {0, 32768, 65535, 65536}}, {4, 4}, {-5, 10},
        device);
    for (int n : {1, 6, 129})
      for (int size : {1, 2, 1024}) {
        auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
        auto symbols = torch::arange(n * size, opts).remainder(100) - 50;
        symbols =
            symbols.reshape({size, n}).transpose(0, 1); // noncontiguous input
        auto idx =
            torch::arange(n * size, opts).remainder(2).reshape({n, size});
        auto encoded = codec.encode(symbols, idx);
        auto hs = symbols.cpu().contiguous(), hi = idx.cpu();
        for (int i = 0; i < n; ++i) {
          auto *sp = hs.data_ptr<int32_t>() + i * size;
          auto *ip = hi.data_ptr<int32_t>() + i * size;
          RansEncoder cpu;
          TORCH_CHECK(encoded[i] ==
                          cpu.encode_with_indexes(
                              std::vector<int32_t>(sp, sp + size),
                              std::vector<int32_t>(ip, ip + size),
                              {{0, 1, 32000, 65536}, {0, 32768, 65535, 65536}},
                              {4, 4}, {-5, 10}),
                      "Byte mismatch");
        }
        // Nonzero begin tests slices from the saved stream collection.
        encoded.insert(encoded.begin(), std::string());
        auto decoded = codec.decode(encoded, 1, idx);
        TORCH_CHECK(decoded.is_cuda() && torch::equal(decoded, symbols),
                    "Round-trip mismatch");
        encoded[1].resize(4);
        bool rejected = false;
        try {
          codec.decode(encoded, 1, idx);
        } catch (const c10::Error &) {
          rejected = true;
        }
        TORCH_CHECK(rejected, "Truncated input not rejected");
      }
    std::cout << "PASS: CUDA/LibTorch codec, nondefault stream, CPU bytes, "
                 "device output, malformed input\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
