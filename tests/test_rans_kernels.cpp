// CPU sanitizer harness for the production CUDA kernel arithmetic.
// nvcc is not used here; test_rans_cuda exercises device launches + LibTorch.
#include "../CAESAR/models/range_coder/rans_coder.hpp"
#include "../CAESAR/models/range_coder/rans_cuda_kernels.cu"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

void require(bool ok) {
  if (!ok)
    throw std::runtime_error("rANS kernel test failed");
}
int main() {
  using namespace caesar::rans_cuda;
  std::vector<std::vector<int32_t>> cdfs{{0, 1, 32000, 65536},
                                         {0, 32768, 65535, 65536}};
  std::vector<int32_t> lengths{4, 4}, offsets{-5, 10}, rows{0, 4, 8};
  std::vector<int32_t> flat{0, 1, 32000, 65536, 0, 32768, 65535, 65536};
  std::mt19937 rng(123);
  for (int n : {1, 6, 129})
    for (int size : {0, 1, 2, 17, 1024}) {
      int cap = 2 * size + 4;
      std::vector<int32_t> syms(n * size), idx(n * size), out(n * size),
          errors(n), lens(n);
      for (int i = 0; i < n * size; ++i) {
        idx[i] = rng() % 2;
        int choices[]{-1000000, -1, 0, 1, 2, 3, 1000000};
        syms[i] = choices[rng() % 7] + offsets[idx[i]];
      }
      std::vector<uint32_t> words(n * cap);
      launch_encode(syms.data(), idx.data(), flat.data(), rows.data(),
                    offsets.data(), 2, words.data(), lens.data(), errors.data(),
                    n, size, cap, nullptr);
      std::vector<uint32_t> packed;
      std::vector<int64_t> pos{0};
      for (int i = 0; i < n; ++i) {
        require(errors[i] == 0 && lens[i] >= 2 && lens[i] <= cap);
        const uint32_t *p = words.data() + (i + 1) * cap - lens[i];
        std::string encoded(reinterpret_cast<const char *>(p), lens[i] * 4);
        RansEncoder cpu;
        std::vector<int32_t> is(idx.begin() + i * size,
                                idx.begin() + (i + 1) * size);
        std::vector<int32_t> ss(syms.begin() + i * size,
                                syms.begin() + (i + 1) * size);
        require(encoded ==
                cpu.encode_with_indexes(ss, is, cdfs, lengths, offsets));
        RansDecoder decoder;
        require(decoder.decode_with_indexes(encoded, is, cdfs, lengths,
                                            offsets) == ss);
        packed.insert(packed.end(), p, p + lens[i]);
        pos.push_back(packed.size());
      }
      launch_decode(packed.data(), pos.data(), idx.data(), flat.data(),
                    rows.data(), offsets.data(), 2, out.data(), errors.data(),
                    n, size, nullptr);
      require(out == syms && std::all_of(errors.begin(), errors.end(),
                                         [](int e) { return e == 0; }));
      if (size) {
        idx[0] = 2;
        launch_decode(packed.data(), pos.data(), idx.data(), flat.data(),
                      rows.data(), offsets.data(), 2, out.data(), errors.data(),
                      n, size, nullptr);
        require(errors[0] == 1);
        launch_encode(syms.data(), idx.data(), flat.data(), rows.data(),
                      offsets.data(), 2, words.data(), lens.data(),
                      errors.data(), n, size, cap, nullptr);
        require(errors[0] == 1);
        idx[0] = 0;
        syms[0] = INT32_MAX;
        launch_encode(syms.data(), idx.data(), flat.data(), rows.data(),
                      offsets.data(), 2, words.data(), lens.data(),
                      errors.data(), n, size, cap, nullptr);
        require(errors[0] == 2);
        packed[0] = 0;
        packed[1] = 0;
        launch_decode(packed.data(), pos.data(), idx.data(), flat.data(),
                      rows.data(), offsets.data(), 2, out.data(), errors.data(),
                      n, size, nullptr);
        require(errors[0] == 3);
      }
    }
  std::cout << "PASS: 15 production-kernel cases, CPU byte compatibility, "
               "round trips, invalid inputs\n";
}
