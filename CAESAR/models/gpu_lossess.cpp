#include "gpu_lossess.h"

#ifdef USE_CUDA
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#define CHECK_CUDA(cmd)                                                        \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      throw std::runtime_error(std::string("HIP error: ") +                    \
                               hipGetErrorString(e));                          \
    }                                                                          \
  } while (0)
#else
#define CHECK_CUDA(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(e));                         \
    }                                                                          \
  } while (0)
#endif

#ifdef ENABLE_NVCOMP
#define CHECK_NVCOMP(cmd)                                                      \
  do {                                                                         \
    nvcompStatus_t s = (cmd);                                                  \
    if (s != nvcompSuccess) {                                                  \
      throw std::runtime_error("nvCOMP error in " #cmd);                       \
    }                                                                          \
  } while (0)
#endif
#endif

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)

// We chunk all inputs down to this size before submitting as one big batch.
static constexpr size_t NVCOMP_ZSTD_MAX_CHUNK = 16ULL * 1024 * 1024; // 16 MB

// Skips all host-to-device uploads for input data — tensors are used directly,
// so this function pipelines seamlessly after GPU-side operations like
// bitsToBytes.
std::vector<NvcompBatchCompressResult>
nvcomp_batch_compress(const std::vector<torch::Tensor> &inputs) {
  const size_t N = inputs.size();
  std::vector<NvcompBatchCompressResult> results(N);

  struct ChunkInfo {
    size_t buf_idx;
    const uint8_t *src_ptr; // points directly into the GPU tensor
    size_t chunk_size;
  };
  std::vector<ChunkInfo> chunks;
  std::vector<size_t> chunk_start_idx(N);

  for (size_t i = 0; i < N; i++) {
    size_t sz = (size_t)inputs[i].numel();
    results[i].rawBytes = sz;
    chunk_start_idx[i] = chunks.size();
    if (sz == 0)
      continue;
    const uint8_t *ptr = inputs[i].data_ptr<uint8_t>();
    size_t offset = 0;
    while (offset < sz) {
      size_t this_chunk = std::min(NVCOMP_ZSTD_MAX_CHUNK, sz - offset);
      chunks.push_back({i, ptr + offset, this_chunk});
      offset += this_chunk;
    }
  }

  const size_t totalChunks = chunks.size();
  if (totalChunks == 0)
    return results;

  nvcompBatchedZstdCompressOpts_t comp_opts =
      nvcompBatchedZstdCompressDefaultOpts;

  size_t maxOutPerChunk = 0;
  CHECK_NVCOMP(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
      NVCOMP_ZSTD_MAX_CHUNK, comp_opts, &maxOutPerChunk));

  size_t totalTempBytes = 0;
  size_t totalUncompressed = 0;
  for (auto &c : chunks)
    totalUncompressed += c.chunk_size;

  CHECK_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
      totalChunks, NVCOMP_ZSTD_MAX_CHUNK, comp_opts, &totalTempBytes,
      totalUncompressed));

  // remove size and input pool and its allocation
  void *d_output_pool = nullptr;
  void *d_temp = nullptr;
  void *d_input_ptrs = nullptr;
  void *d_output_ptrs = nullptr;
  void *d_input_sizes = nullptr;
  void *d_output_sizes = nullptr;
  void *d_statuses = nullptr;

  CHECK_CUDA(cudaMalloc(&d_output_pool, totalChunks * maxOutPerChunk));
  if (totalTempBytes > 0)
    CHECK_CUDA(cudaMalloc(&d_temp, totalTempBytes));
  CHECK_CUDA(cudaMalloc(&d_input_ptrs, totalChunks * sizeof(void *)));
  CHECK_CUDA(cudaMalloc(&d_output_ptrs, totalChunks * sizeof(void *)));
  CHECK_CUDA(cudaMalloc(&d_input_sizes, totalChunks * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_output_sizes, totalChunks * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_statuses, totalChunks * sizeof(nvcompStatus_t)));

  std::vector<void *> h_input_ptrs(totalChunks);
  std::vector<void *> h_output_ptrs(totalChunks);
  std::vector<size_t> h_input_sizes(totalChunks);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Point directly into GPU tensor memory
  for (size_t c = 0; c < totalChunks; c++) {
    h_input_ptrs[c] = (void *)chunks[c].src_ptr;
    h_output_ptrs[c] = (uint8_t *)d_output_pool + c * maxOutPerChunk;
    h_input_sizes[c] = chunks[c].chunk_size;
  }

  // H2D: only pointer/size metadata arrays (no need to upload data)
  CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(),
                             totalChunks * sizeof(void *),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(),
                             totalChunks * sizeof(void *),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(),
                             totalChunks * sizeof(size_t),
                             cudaMemcpyHostToDevice, stream));

  // Compress entire batch — single call, fully async
  CHECK_NVCOMP(nvcompBatchedZstdCompressAsync(
      (const void *const *)d_input_ptrs, (const size_t *)d_input_sizes,
      NVCOMP_ZSTD_MAX_CHUNK, totalChunks, d_temp, totalTempBytes,
      (void *const *)d_output_ptrs, (size_t *)d_output_sizes, comp_opts,
      (nvcompStatus_t *)d_statuses, stream));

  // D2H: read back sizes and statuses on the same stream — no mid-sync
  std::vector<size_t> h_output_sizes(totalChunks);
  std::vector<nvcompStatus_t> h_statuses(totalChunks);

  CHECK_CUDA(cudaMemcpyAsync(h_output_sizes.data(), d_output_sizes,
                             totalChunks * sizeof(size_t),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(h_statuses.data(), d_statuses,
                             totalChunks * sizeof(nvcompStatus_t),
                             cudaMemcpyDeviceToHost, stream));

  // Single sync — everything above overlaps on the stream
  CHECK_CUDA(cudaStreamSynchronize(stream));

  for (size_t c = 0; c < totalChunks; c++) {
    if (h_statuses[c] != nvcompSuccess)
      throw std::runtime_error("nvcomp Zstd compress failed on chunk " +
                               std::to_string(c) + " (buffer " +
                               std::to_string(chunks[c].buf_idx) + ")");
  }

  // Assemble per-buffer results (same framing as original)
  for (size_t i = 0; i < N; i++) {
    if (inputs[i].numel() == 0)
      continue;

    size_t first = chunk_start_idx[i];
    size_t count = 0;
    for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++)
      count++;

    if (count == 1) {
      results[i].compressed =
          torch::empty({(int64_t)h_output_sizes[first]}, torch::kUInt8);
      CHECK_CUDA(cudaMemcpy(results[i].compressed.data_ptr<uint8_t>(),
                            (uint8_t *)d_output_pool + first * maxOutPerChunk,
                            h_output_sizes[first], cudaMemcpyDeviceToHost));
    } else {
      size_t headerSize = 8 + count * 8 + count * 8;
      size_t totalCompressed = 0;
      for (size_t c = first; c < first + count; c++)
        totalCompressed += h_output_sizes[c];

      results[i].compressed = torch::empty(
          {(int64_t)(headerSize + totalCompressed)}, torch::kUInt8);
      uint8_t *p = results[i].compressed.data_ptr<uint8_t>();

      uint64_t nc = count;
      memcpy(p, &nc, 8);
      p += 8;
      for (size_t c = first; c < first + count; c++) {
        uint64_t us = chunks[c].chunk_size;
        memcpy(p, &us, 8);
        p += 8;
      }
      for (size_t c = first; c < first + count; c++) {
        uint64_t cs = h_output_sizes[c];
        memcpy(p, &cs, 8);
        p += 8;
      }
      for (size_t c = first; c < first + count; c++) {
        CHECK_CUDA(cudaMemcpy(p, (uint8_t *)d_output_pool + c * maxOutPerChunk,
                              h_output_sizes[c], cudaMemcpyDeviceToHost));
        p += h_output_sizes[c];
      }
    }
  }

  cudaFree(d_output_pool);
  if (d_temp)
    cudaFree(d_temp);
  cudaFree(d_input_ptrs);
  cudaFree(d_output_ptrs);
  cudaFree(d_input_sizes);
  cudaFree(d_output_sizes);
  cudaFree(d_statuses);
  cudaStreamDestroy(stream);

  return results;
}

// Batched GPU decompress — handles both single-chunk and multi-chunk framing
std::vector<std::vector<uint8_t>>
nvcomp_batch_decompress(const std::vector<const uint8_t *> &comp_ptrs,
                        const std::vector<size_t> &comp_sizes,
                        const std::vector<size_t> &decomp_sizes) {
  const size_t N = comp_ptrs.size();
  std::vector<std::vector<uint8_t>> results(N);

  struct DecompChunkInfo {
    size_t buf_idx;
    const uint8_t *comp_ptr; // points into host compressed data
    size_t comp_size;
    size_t decomp_size;
  };
  std::vector<DecompChunkInfo> chunks;
  std::vector<size_t> chunk_start_idx(N);

  for (size_t i = 0; i < N; i++) {
    chunk_start_idx[i] = chunks.size();
    if (comp_sizes[i] == 0 || decomp_sizes[i] == 0)
      continue;

    const uint8_t *p = comp_ptrs[i];

    // Read first 8 bytes as potential num_chunks
    uint64_t potential_nc = 0;
    memcpy(&potential_nc, p, 8);

    size_t headerSize = 8 + potential_nc * 8 + potential_nc * 8;
    bool is_multi =
        (potential_nc > 1 && potential_nc < 1000 && headerSize < comp_sizes[i]);

    if (is_multi) {
      const uint8_t *hp = p + 8;
      std::vector<size_t> unc_sizes(potential_nc), cmp_sizes(potential_nc);
      size_t total_unc = 0;
      for (size_t c = 0; c < potential_nc; c++) {
        memcpy(&unc_sizes[c], hp, 8);
        hp += 8;
        total_unc += unc_sizes[c];
      }
      for (size_t c = 0; c < potential_nc; c++) {
        memcpy(&cmp_sizes[c], hp, 8);
        hp += 8;
      }

      if (total_unc == decomp_sizes[i]) {
        // Valid multi-chunk
        const uint8_t *data_ptr = p + headerSize;
        for (size_t c = 0; c < potential_nc; c++) {
          chunks.push_back({i, data_ptr, cmp_sizes[c], unc_sizes[c]});
          data_ptr += cmp_sizes[c];
        }
        continue;
      }
    }

    // Single chunk — entire compressed buffer is one chunk
    chunks.push_back({i, p, comp_sizes[i], decomp_sizes[i]});
  }

  const size_t totalChunks = chunks.size();
  if (totalChunks == 0)
    return results;

  size_t maxCompChunk = 0;
  size_t maxDecompChunk = 0;
  size_t totalDecomp = 0;
  for (auto &c : chunks) {
    maxCompChunk = std::max(maxCompChunk, c.comp_size);
    maxDecompChunk = std::max(maxDecompChunk, c.decomp_size);
    totalDecomp += c.decomp_size;
  }

  nvcompBatchedZstdDecompressOpts_t decomp_opts =
      nvcompBatchedZstdDecompressDefaultOpts;

  size_t totalTempBytes = 0;
  CHECK_NVCOMP(nvcompBatchedZstdDecompressGetTempSizeAsync(
      totalChunks, maxDecompChunk, decomp_opts, &totalTempBytes, totalDecomp));

  void *d_comp_pool = nullptr;
  void *d_decomp_pool = nullptr;
  void *d_temp = nullptr;

  CHECK_CUDA(cudaMalloc(&d_comp_pool, totalChunks * maxCompChunk));
  CHECK_CUDA(cudaMalloc(&d_decomp_pool, totalChunks * maxDecompChunk));
  if (totalTempBytes > 0)
    CHECK_CUDA(cudaMalloc(&d_temp, totalTempBytes));

  void *d_input_ptrs = nullptr;
  void *d_output_ptrs = nullptr;
  void *d_input_sizes = nullptr;
  void *d_output_sizes = nullptr;
  void *d_statuses = nullptr;

  CHECK_CUDA(cudaMalloc(&d_input_ptrs, totalChunks * sizeof(void *)));
  CHECK_CUDA(cudaMalloc(&d_output_ptrs, totalChunks * sizeof(void *)));
  CHECK_CUDA(cudaMalloc(&d_input_sizes, totalChunks * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_output_sizes, totalChunks * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_statuses, totalChunks * sizeof(nvcompStatus_t)));

  std::vector<void *> h_input_ptrs(totalChunks), h_output_ptrs(totalChunks);
  std::vector<size_t> h_input_sizes(totalChunks), h_output_sizes(totalChunks);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  for (size_t c = 0; c < totalChunks; c++) {
    uint8_t *comp_slot = (uint8_t *)d_comp_pool + c * maxCompChunk;
    uint8_t *decomp_slot = (uint8_t *)d_decomp_pool + c * maxDecompChunk;

    h_input_ptrs[c] = comp_slot;
    h_output_ptrs[c] = decomp_slot;
    h_input_sizes[c] = chunks[c].comp_size;
    h_output_sizes[c] = chunks[c].decomp_size;

    CHECK_CUDA(cudaMemcpyAsync(comp_slot, chunks[c].comp_ptr,
                               chunks[c].comp_size, cudaMemcpyHostToDevice,
                               stream));
  }

  CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(),
                             totalChunks * sizeof(void *),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(),
                             totalChunks * sizeof(void *),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(),
                             totalChunks * sizeof(size_t),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_output_sizes, h_output_sizes.data(),
                             totalChunks * sizeof(size_t),
                             cudaMemcpyHostToDevice, stream));

  CHECK_NVCOMP(nvcompBatchedZstdDecompressAsync(
      (const void *const *)d_input_ptrs, (const size_t *)d_input_sizes,
      (const size_t *)d_output_sizes, // buffer sizes (max)
      (size_t *)d_output_sizes,       // actual sizes written back
      totalChunks, d_temp, totalTempBytes, (void *const *)d_output_ptrs,
      decomp_opts, (nvcompStatus_t *)d_statuses, stream));

  // see if this can be removed
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<nvcompStatus_t> h_statuses(totalChunks);
  CHECK_CUDA(cudaMemcpy(h_statuses.data(), d_statuses,
                        totalChunks * sizeof(nvcompStatus_t),
                        cudaMemcpyDeviceToHost));

  for (size_t c = 0; c < totalChunks; c++) {
    if (h_statuses[c] != nvcompSuccess)
      throw std::runtime_error("nvcomp Zstd decompress failed on chunk " +
                               std::to_string(c) + " (buffer " +
                               std::to_string(chunks[c].buf_idx) + ")");
  }

  for (size_t i = 0; i < N; i++) {
    if (decomp_sizes[i] == 0)
      continue;

    results[i].resize(decomp_sizes[i]);
    uint8_t *out = results[i].data();

    size_t first = chunk_start_idx[i];
    size_t count = 0;
    for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++)
      count++;

    for (size_t c = first; c < first + count; c++) {
      CHECK_CUDA(cudaMemcpy(out, (uint8_t *)d_decomp_pool + c * maxDecompChunk,
                            chunks[c].decomp_size, cudaMemcpyDeviceToHost));
      out += chunks[c].decomp_size;
    }
  }

  cudaFree(d_comp_pool);
  cudaFree(d_decomp_pool);
  if (d_temp)
    cudaFree(d_temp);
  cudaFree(d_input_ptrs);
  cudaFree(d_output_ptrs);
  cudaFree(d_input_sizes);
  cudaFree(d_output_sizes);
  cudaFree(d_statuses);
  cudaStreamDestroy(stream);

  return results;
}

#endif // USE_CUDA && ENABLE_NVCOMP
