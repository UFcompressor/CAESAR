#include "rans_cuda_kernels.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define DEVICE __device__ __forceinline__
#define KERNEL __global__
#else
// CPU emulation for sanitizer tests of the exact kernel arithmetic.
#define DEVICE inline
#define KERNEL
#endif

namespace caesar::rans_cuda {
constexpr uint64_t lower = 1ull << 31;
DEVICE void put(uint64_t &x, uint32_t *&p, uint32_t start, uint32_t freq) {
  if (x >= ((lower >> 16) << 32) * freq) {
    *--p = uint32_t(x);
    x >>= 32;
  }
  x = ((x / freq) << 16) + x % freq + start;
}
DEVICE void put_bits(uint64_t &x, uint32_t *&p, uint32_t v) {
  if (x >= ((lower >> 16) << 32) * (1u << 12)) {
    *--p = uint32_t(x);
    x >>= 32;
  }
  x = (x << 4) | v;
}
DEVICE bool renorm(uint64_t &x, const uint32_t *&p, const uint32_t *end) {
  if (x < lower) {
    if (p == end)
      return false;
    x = (x << 32) | *p++;
  }
  return true;
}
DEVICE bool bits(uint64_t &x, const uint32_t *&p, const uint32_t *end,
                 uint32_t &v) {
  v = x & 15;
  x >>= 4;
  return renorm(x, p, end);
}

KERNEL void encode_kernel(const int32_t *symbols, const int32_t *indexes,
                          const int32_t *cdf, const int32_t *rows,
                          const int32_t *offsets, int row_count,
                          uint32_t *words, int32_t *lengths, int32_t *errors,
                          int streams, int size, int capacity) {
#ifdef __CUDACC__
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= streams)
    return;
#else
  for (int t = 0; t < streams; ++t) {
#endif
  errors[t] = 0;
  uint32_t *base = words + int64_t(t) * capacity;
  uint32_t *p = base + capacity;
  uint64_t x = lower;
  for (int j = size - 1; j >= 0; --j) {
    int64_t i = int64_t(t) * size + j;
    int r = indexes[i];
    if (r < 0 || r >= row_count) {
      errors[t] = 1;
      break;
    }
    int max_value = rows[r + 1] - rows[r] - 2;
    int64_t v = int64_t(symbols[i]) - offsets[r];
    uint64_t raw = 0;
    if (v < 0) {
      raw = uint64_t(-2 * v - 1);
      v = max_value;
    } else if (v >= max_value) {
      raw = uint64_t(2 * (v - max_value));
      v = max_value;
    }
    // The existing CPU codec has undefined shifts for >=8 bypass nibbles.
    // Reject such inputs explicitly rather than silently producing different
    // bytes.
    if (raw >= (1ull << 28)) {
      errors[t] = 2;
      break;
    }
    if (v == max_value) {
      int n = 0;
      for (uint64_t z = raw; z; z >>= 4)
        ++n;
      for (int b = n - 1; b >= 0; --b)
        put_bits(x, p, (raw >> (4 * b)) & 15);
      put_bits(x, p, n);
    }
    int c = rows[r] + int(v);
    put(x, p, cdf[c], cdf[c + 1] - cdf[c]);
  }
  p -= 2;
  p[0] = uint32_t(x);
  p[1] = uint32_t(x >> 32);
  lengths[t] = int(base + capacity - p);
#ifndef __CUDACC__
}
#endif
}

KERNEL void decode_kernel(const uint32_t *words, const int64_t *positions,
                          const int32_t *indexes, const int32_t *cdf,
                          const int32_t *rows, const int32_t *offsets,
                          int row_count, int32_t *output, int32_t *errors,
                          int streams, int size) {
#ifdef __CUDACC__
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= streams)
    return;
#else
    for (int t = 0; t < streams; ++t) {
#endif
  errors[t] = 0;
  const uint32_t *p = words + positions[t];
  const uint32_t *end = words + positions[t + 1];
  uint64_t x = uint64_t(p[0]) | (uint64_t(p[1]) << 32);
  p += 2;
  if (x < lower || x >= (1ull << 63))
    errors[t] = 3;
  for (int j = 0; j < size && !errors[t]; ++j) {
    int64_t i = int64_t(t) * size + j;
    int r = indexes[i];
    if (r < 0 || r >= row_count) {
      errors[t] = 1;
      break;
    }
    const int32_t *row = cdf + rows[r];
    int max_value = rows[r + 1] - rows[r] - 2;
    uint32_t cum = x & 65535;
    int lo = 0, hi = max_value;
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (uint32_t(row[mid]) <= cum)
        lo = mid;
      else
        hi = mid - 1;
    }
    x = uint64_t(row[lo + 1] - row[lo]) * (x >> 16) + cum - row[lo];
    if (!renorm(x, p, end)) {
      errors[t] = 3;
      break;
    }
    int64_t value = lo;
    if (lo == max_value) {
      uint32_t n;
      if (!bits(x, p, end, n) || n > 7) {
        errors[t] = 3;
        break;
      }
      uint32_t raw = 0;
      for (uint32_t b = 0; b < n; ++b) {
        uint32_t v;
        if (!bits(x, p, end, v)) {
          errors[t] = 3;
          break;
        }
        raw |= v << (4 * b);
      }
      value =
          (raw & 1) ? -int64_t(raw >> 1) - 1 : int64_t(raw >> 1) + max_value;
    }
    value += offsets[r];
    if (value < INT32_MIN || value > INT32_MAX) {
      errors[t] = 2;
      break;
    }
    output[i] = int32_t(value);
  }
  if (!errors[t] && (p != end || x != lower))
    errors[t] = 3;
#ifndef __CUDACC__
}
#endif
}

void launch_encode(const int32_t *symbols, const int32_t *indexes,
                   const int32_t *cdf, const int32_t *rows,
                   const int32_t *offsets, int row_count, uint32_t *words,
                   int32_t *lengths, int32_t *errors, int streams, int size,
                   int capacity, void *stream) {
#ifdef __CUDACC__
  encode_kernel<<<(streams + 127) / 128, 128, 0,
                  static_cast<cudaStream_t>(stream)>>>(
#else
      (void)stream;
      encode_kernel(
#endif
      symbols, indexes, cdf, rows, offsets, row_count, words, lengths, errors,
      streams, size, capacity);
}
void launch_decode(const uint32_t *words, const int64_t *positions,
                   const int32_t *indexes, const int32_t *cdf,
                   const int32_t *rows, const int32_t *offsets, int row_count,
                   int32_t *output, int32_t *errors, int streams, int size,
                   void *stream) {
#ifdef __CUDACC__
  decode_kernel<<<(streams + 127) / 128, 128, 0,
                  static_cast<cudaStream_t>(stream)>>>(
#else
      (void)stream;
      decode_kernel(
#endif
      words, positions, indexes, cdf, rows, offsets, row_count, output, errors,
      streams, size);
}
} // namespace caesar::rans_cuda
