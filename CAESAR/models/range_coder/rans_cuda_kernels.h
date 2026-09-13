#pragma once
#include <cstdint>

namespace caesar::rans_cuda {
// Tables are flattened on device; offsets has rows+1 entries.
void launch_encode(const int32_t *symbols, const int32_t *indexes,
                   const int32_t *cdf, const int32_t *rows,
                   const int32_t *offsets, int row_count, uint32_t *words,
                   int32_t *lengths, int32_t *errors, int streams, int size,
                   int capacity, void *stream);
void launch_decode(const uint32_t *words, const int64_t *positions,
                   const int32_t *indexes, const int32_t *cdf,
                   const int32_t *rows, const int32_t *offsets, int row_count,
                   int32_t *output, int32_t *errors, int streams, int size,
                   void *stream);
} // namespace caesar::rans_cuda
