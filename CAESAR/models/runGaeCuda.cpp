#include "runGaeCuda.h"
#include "gaeFusedKernels.h"
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstring>

#ifdef USE_CUDA
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#define CHECK_CUDA(cmd) do { \
          hipError_t e = (cmd); \
          if (e != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e)); \
          } \
        } while(0)
#else
#define CHECK_CUDA(cmd) do { \
          cudaError_t e = (cmd); \
          if (e != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e)); \
          } \
        } while(0)
#endif

#ifdef ENABLE_NVCOMP
#define CHECK_NVCOMP(cmd) do { \
          nvcompStatus_t s = (cmd); \
          if (s != nvcompSuccess) { \
            throw std::runtime_error("nvCOMP error in " #cmd); \
          } \
        } while(0)
#endif
#endif



#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)

struct NvcompBatchCompressResult {
    std::vector<uint8_t> compressed;
    size_t               rawBytes;
};

// We chunk all inputs down to this size before submitting as one big batch.
static constexpr size_t NVCOMP_ZSTD_MAX_CHUNK = 16ULL * 1024 * 1024; // 16 MB

static void parallel_zero_fill_payload(uint8_t* data, size_t size)
{
    if (data == nullptr || size == 0) return;

    unsigned nThreads = std::thread::hardware_concurrency();
    if (const char* envThreads = std::getenv("CAESAR_GAE_PAYLOAD_TOUCH_THREADS")) {
        int parsed = std::atoi(envThreads);
        if (parsed > 0) nThreads = static_cast<unsigned>(parsed);
    }
    if (nThreads == 0) nThreads = 8;
    nThreads = std::min<unsigned>(nThreads, 32);

    const size_t minChunk = 64ULL * 1024 * 1024;
    if (size < minChunk || nThreads == 1) {
        std::memset(data, 0, size);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nThreads);
    const size_t chunk = (size + nThreads - 1) / nThreads;

    for (unsigned t = 0; t < nThreads; ++t) {
        size_t begin = static_cast<size_t>(t) * chunk;
        if (begin >= size) break;
        size_t end = std::min(size, begin + chunk);
        workers.emplace_back([data, begin, end]() {
            std::memset(data + begin, 0, end - begin);
        });
    }

    for (auto& worker : workers) worker.join();
}

                                                                     //
static std::vector<NvcompBatchCompressResult>
nvcomp_batch_compress(
    const std::vector<const uint8_t*>& inputs,
    const std::vector<size_t>&         sizes)
{
    const size_t N = inputs.size();
    std::vector<NvcompBatchCompressResult> results(N);

    struct ChunkInfo {
        size_t buf_idx;
        size_t offset;
        size_t chunk_size;
    };
    std::vector<ChunkInfo> chunks;
    std::vector<size_t>    chunk_start_idx(N);

    for (size_t i = 0; i < N; i++) {
        results[i].rawBytes = sizes[i];
        chunk_start_idx[i]  = chunks.size();
        if (sizes[i] == 0) continue;
        size_t offset = 0;
        while (offset < sizes[i]) {
            size_t this_chunk = std::min(NVCOMP_ZSTD_MAX_CHUNK, sizes[i] - offset);
            chunks.push_back({i, offset, this_chunk});
            offset += this_chunk;
        }
    }

    const size_t totalChunks = chunks.size();
    if (totalChunks == 0) return results;

    nvcompBatchedZstdCompressOpts_t comp_opts = nvcompBatchedZstdCompressDefaultOpts;

    size_t maxOutPerChunk = 0;
    CHECK_NVCOMP(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        NVCOMP_ZSTD_MAX_CHUNK, comp_opts, &maxOutPerChunk));

    size_t totalTempBytes    = 0;
    size_t totalUncompressed = 0;
    for (auto& c : chunks) totalUncompressed += c.chunk_size;

    CHECK_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
        totalChunks, NVCOMP_ZSTD_MAX_CHUNK, comp_opts,
        &totalTempBytes, totalUncompressed));

    std::cout << "[NVCOMP pipeline] totalChunks=" << totalChunks
              << " maxOutPerChunk=" << maxOutPerChunk
              << " tempBytes=" << totalTempBytes << "\n";

    // Input pool sized to actual data, not totalChunks * 16MB
    size_t inputPoolSize = totalUncompressed + totalChunks * 255;
    size_t outputPoolSize = totalChunks * maxOutPerChunk;

    void* d_input_pool   = nullptr;
    void* d_output_pool  = nullptr;
    void* d_temp         = nullptr;
    void* d_input_ptrs   = nullptr;
    void* d_output_ptrs  = nullptr;
    void* d_input_sizes  = nullptr;
    void* d_output_sizes = nullptr;
    void* d_statuses     = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input_pool,   inputPoolSize));
    CHECK_CUDA(cudaMalloc(&d_output_pool,  outputPoolSize));
    if (totalTempBytes > 0)
        CHECK_CUDA(cudaMalloc(&d_temp, totalTempBytes));
    CHECK_CUDA(cudaMalloc(&d_input_ptrs,   totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_output_ptrs,  totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_input_sizes,  totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_output_sizes, totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_statuses,     totalChunks * sizeof(nvcompStatus_t)));

    // Build host-side pointer arrays using actual offsets into input pool
    std::vector<void*>  h_input_ptrs(totalChunks);
    std::vector<void*>  h_output_ptrs(totalChunks);
    std::vector<size_t> h_input_sizes(totalChunks);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Compute actual byte offset of each chunk in the input pool
    size_t pool_offset = 0;
    for (size_t c = 0; c < totalChunks; c++) {
        h_input_ptrs[c]  = (uint8_t*)d_input_pool  + pool_offset;
        h_output_ptrs[c] = (uint8_t*)d_output_pool + c * maxOutPerChunk;
        h_input_sizes[c] = chunks[c].chunk_size;
        pool_offset += (chunks[c].chunk_size + 255) & ~size_t(255);  // round up to 256-byte
    }

    // H2D: upload all chunks async — no sync
    for (size_t c = 0; c < totalChunks; c++) {
        CHECK_CUDA(cudaMemcpyAsync(
            h_input_ptrs[c],
            inputs[chunks[c].buf_idx] + chunks[c].offset,
            chunks[c].chunk_size,
            cudaMemcpyHostToDevice, stream));
    }

    // H2D: upload pointer/size metadata async
    CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs,  h_input_ptrs.data(),  totalChunks * sizeof(void*),  cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(), totalChunks * sizeof(void*),  cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(), totalChunks * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Compress entire batch — single call, fully async
    CHECK_NVCOMP(nvcompBatchedZstdCompressAsync(
        (const void* const*)d_input_ptrs,
        (const size_t*)d_input_sizes,
        NVCOMP_ZSTD_MAX_CHUNK,
        totalChunks,
        d_temp,
        totalTempBytes,
        (void* const*)d_output_ptrs,
        (size_t*)d_output_sizes,
        comp_opts,
        (nvcompStatus_t*)d_statuses,
        stream));

    // D2H: read back sizes and statuses async on same stream — no mid-sync
    std::vector<size_t>         h_output_sizes(totalChunks);
    std::vector<nvcompStatus_t> h_statuses(totalChunks);

    CHECK_CUDA(cudaMemcpyAsync(h_output_sizes.data(), d_output_sizes,
        totalChunks * sizeof(size_t),         cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_statuses.data(),     d_statuses,
        totalChunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream));

    // Single sync — everything above overlaps on the stream
    CHECK_CUDA(cudaStreamSynchronize(stream));

    for (size_t c = 0; c < totalChunks; c++) {
        if (h_statuses[c] != nvcompSuccess)
            throw std::runtime_error("nvcomp Zstd compress failed on chunk "
                + std::to_string(c) + " (buffer "
                + std::to_string(chunks[c].buf_idx) + ")");
    }

    // Assemble per-buffer results (same framing as original)
    for (size_t i = 0; i < N; i++) {
        if (sizes[i] == 0) continue;

        size_t first = chunk_start_idx[i];
        size_t count = 0;
        for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++) count++;

        std::vector<uint8_t>& out = results[i].compressed;

        if (count == 1) {
            out.resize(h_output_sizes[first]);
            CHECK_CUDA(cudaMemcpy(out.data(),
                (uint8_t*)d_output_pool + first * maxOutPerChunk,
                h_output_sizes[first], cudaMemcpyDeviceToHost));
        } else {
            size_t headerSize      = 8 + count * 8 + count * 8;
            size_t totalCompressed = 0;
            for (size_t c = first; c < first + count; c++) totalCompressed += h_output_sizes[c];

            out.resize(headerSize + totalCompressed);
            uint8_t* p = out.data();

            uint64_t nc = count;
            memcpy(p, &nc, 8); p += 8;
            for (size_t c = first; c < first + count; c++) {
                uint64_t us = chunks[c].chunk_size; memcpy(p, &us, 8); p += 8;
            }
            for (size_t c = first; c < first + count; c++) {
                uint64_t cs = h_output_sizes[c];    memcpy(p, &cs, 8); p += 8;
            }
            for (size_t c = first; c < first + count; c++) {
                CHECK_CUDA(cudaMemcpy(p,
                    (uint8_t*)d_output_pool + c * maxOutPerChunk,
                    h_output_sizes[c], cudaMemcpyDeviceToHost));
                p += h_output_sizes[c];
            }
        }
    }

    cudaFree(d_input_pool);
    cudaFree(d_output_pool);
    if (d_temp) cudaFree(d_temp);
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_sizes);
    cudaFree(d_statuses);
    cudaStreamDestroy(stream);

    return results;
}


// Batched GPU compress for inputs that already live on device.
static std::vector<NvcompBatchCompressResult>
nvcomp_batch_compress_device(
    const std::vector<const uint8_t*>& device_inputs,
    const std::vector<size_t>&         sizes)
{
    const size_t N = device_inputs.size();
    std::vector<NvcompBatchCompressResult> results(N);

    struct ChunkInfo {
        size_t buf_idx;
        size_t offset;
        size_t chunk_size;
    };

    std::vector<ChunkInfo> chunks;
    std::vector<size_t> chunk_start_idx(N);

    for (size_t i = 0; i < N; i++) {
        results[i].rawBytes = sizes[i];
        chunk_start_idx[i] = chunks.size();
        if (sizes[i] == 0) continue;

        size_t offset = 0;
        while (offset < sizes[i]) {
            size_t this_chunk = std::min(NVCOMP_ZSTD_MAX_CHUNK, sizes[i] - offset);
            chunks.push_back({i, offset, this_chunk});
            offset += this_chunk;
        }
    }

    const size_t totalChunks = chunks.size();
    if (totalChunks == 0) return results;

    nvcompBatchedZstdCompressOpts_t comp_opts = nvcompBatchedZstdCompressDefaultOpts;

    size_t maxOutPerChunk = 0;
    CHECK_NVCOMP(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        NVCOMP_ZSTD_MAX_CHUNK, comp_opts, &maxOutPerChunk));

    size_t totalTempBytes = 0;
    size_t totalUncompressed = 0;
    for (const auto& c : chunks) totalUncompressed += c.chunk_size;

    CHECK_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
        totalChunks, NVCOMP_ZSTD_MAX_CHUNK, comp_opts,
        &totalTempBytes, totalUncompressed));

    size_t outputStride = (maxOutPerChunk + 255) & ~size_t(255);
    size_t outputPoolSize = totalChunks * outputStride;

    void* d_output_pool = nullptr;
    void* d_temp = nullptr;
    void* d_input_ptrs = nullptr;
    void* d_output_ptrs = nullptr;
    void* d_input_sizes = nullptr;
    void* d_output_sizes = nullptr;
    void* d_statuses = nullptr;

    CHECK_CUDA(cudaMalloc(&d_output_pool, outputPoolSize));
    if (totalTempBytes > 0)
        CHECK_CUDA(cudaMalloc(&d_temp, totalTempBytes));

    CHECK_CUDA(cudaMalloc(&d_input_ptrs, totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_output_ptrs, totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_input_sizes, totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_output_sizes, totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_statuses, totalChunks * sizeof(nvcompStatus_t)));

    std::vector<void*> h_input_ptrs(totalChunks);
    std::vector<void*> h_output_ptrs(totalChunks);
    std::vector<size_t> h_input_sizes(totalChunks);

    for (size_t c = 0; c < totalChunks; c++) {
        const auto& chunk = chunks[c];
        h_input_ptrs[c] = const_cast<uint8_t*>(
            device_inputs[chunk.buf_idx] + chunk.offset);
        h_output_ptrs[c] = static_cast<uint8_t*>(d_output_pool) + c * outputStride;
        h_input_sizes[c] = chunk.chunk_size;
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(),
        totalChunks * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(),
        totalChunks * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(),
        totalChunks * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    CHECK_NVCOMP(nvcompBatchedZstdCompressAsync(
        (const void* const*)d_input_ptrs,
        (const size_t*)d_input_sizes,
        NVCOMP_ZSTD_MAX_CHUNK,
        totalChunks,
        d_temp,
        totalTempBytes,
        (void* const*)d_output_ptrs,
        (size_t*)d_output_sizes,
        comp_opts,
        (nvcompStatus_t*)d_statuses,
        stream));

    std::vector<size_t> h_output_sizes(totalChunks);
    std::vector<nvcompStatus_t> h_statuses(totalChunks);

    CHECK_CUDA(cudaMemcpyAsync(h_output_sizes.data(), d_output_sizes,
        totalChunks * sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_statuses.data(), d_statuses,
        totalChunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    for (size_t c = 0; c < totalChunks; c++) {
        if (h_statuses[c] != nvcompSuccess) {
            throw std::runtime_error("nvcomp device Zstd compress failed on chunk "
                + std::to_string(c) + " (buffer "
                + std::to_string(chunks[c].buf_idx) + ")");
        }
        if (h_output_sizes[c] > maxOutPerChunk) {
            throw std::runtime_error("nvcomp device Zstd output exceeded bound");
        }
    }

    for (size_t i = 0; i < N; i++) {
        if (sizes[i] == 0) continue;

        size_t first = chunk_start_idx[i];
        size_t count = 0;
        for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++) count++;

        std::vector<uint8_t>& out = results[i].compressed;

        if (count == 1) {
            out.resize(h_output_sizes[first]);
            CHECK_CUDA(cudaMemcpy(out.data(),
                static_cast<uint8_t*>(d_output_pool) + first * outputStride,
                h_output_sizes[first], cudaMemcpyDeviceToHost));
        } else {
            size_t headerSize = 8 + count * 8 + count * 8;
            size_t totalCompressed = 0;
            for (size_t c = first; c < first + count; c++) totalCompressed += h_output_sizes[c];

            out.resize(headerSize + totalCompressed);
            uint8_t* q = out.data();

            uint64_t nc = count;
            std::memcpy(q, &nc, 8); q += 8;
            for (size_t c = first; c < first + count; c++) {
                uint64_t us = chunks[c].chunk_size;
                std::memcpy(q, &us, 8); q += 8;
            }
            for (size_t c = first; c < first + count; c++) {
                uint64_t cs = h_output_sizes[c];
                std::memcpy(q, &cs, 8); q += 8;
            }
            for (size_t c = first; c < first + count; c++) {
                CHECK_CUDA(cudaMemcpy(q,
                    static_cast<uint8_t*>(d_output_pool) + c * outputStride,
                    h_output_sizes[c], cudaMemcpyDeviceToHost));
                q += h_output_sizes[c];
            }
        }
    }

    cudaFree(d_output_pool);
    if (d_temp) cudaFree(d_temp);
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_sizes);
    cudaFree(d_statuses);
    cudaStreamDestroy(stream);

    return results;
}



// Batched GPU compress directly into the final contiguous CAESAR payload.
// Payload format is unchanged:
// [4 x uint64_t sizes][processMask][prefixMask][maskLength][coeffInt]
static std::vector<size_t>
nvcomp_batch_compress_device_to_payload(
    const std::vector<const uint8_t*>& device_inputs,
    const std::vector<size_t>& sizes,
    std::vector<uint8_t>& payload)
{
    const bool profileDirect =
        std::getenv("CAESAR_GAE_LOSSLESS_PROFILE") != nullptr;
    auto directStart = std::chrono::high_resolution_clock::now();
    auto directLast = directStart;
    auto printDirectProfile = [&](const char* name) {
        if (!profileDirect) return;
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - directLast).count();
        double total = std::chrono::duration<double>(now - directStart).count();
        std::cout << "[GAE DIRECT PAYLOAD PROFILE] " << name
                  << " time=" << dt << " s total=" << total << " s\n";
        directLast = now;
    };

    const size_t N = device_inputs.size();
    std::vector<size_t> compressedSizes(N, 0);

    struct ChunkInfo {
        size_t buf_idx;
        size_t offset;
        size_t chunk_size;
    };

    std::vector<ChunkInfo> chunks;
    std::vector<size_t> chunk_start_idx(N);

    for (size_t i = 0; i < N; i++) {
        chunk_start_idx[i] = chunks.size();
        if (sizes[i] == 0) continue;

        size_t offset = 0;
        while (offset < sizes[i]) {
            size_t this_chunk = std::min(NVCOMP_ZSTD_MAX_CHUNK, sizes[i] - offset);
            chunks.push_back({i, offset, this_chunk});
            offset += this_chunk;
        }
    }

    const size_t totalChunks = chunks.size();
    if (totalChunks == 0) {
        payload.clear();
        return compressedSizes;
    }

    nvcompBatchedZstdCompressOpts_t comp_opts = nvcompBatchedZstdCompressDefaultOpts;

    size_t maxOutPerChunk = 0;
    CHECK_NVCOMP(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        NVCOMP_ZSTD_MAX_CHUNK, comp_opts, &maxOutPerChunk));

    size_t totalTempBytes = 0;
    size_t totalUncompressed = 0;
    for (const auto& c : chunks) totalUncompressed += c.chunk_size;

    CHECK_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
        totalChunks, NVCOMP_ZSTD_MAX_CHUNK, comp_opts,
        &totalTempBytes, totalUncompressed));

    size_t outputStride = (maxOutPerChunk + 255) & ~size_t(255);
    size_t outputPoolSize = totalChunks * outputStride;
    printDirectProfile("setup_chunks_queries");

    void* d_output_pool = nullptr;
    void* d_temp = nullptr;
    void* d_input_ptrs = nullptr;
    void* d_output_ptrs = nullptr;
    void* d_input_sizes = nullptr;
    void* d_output_sizes = nullptr;
    void* d_statuses = nullptr;

    CHECK_CUDA(cudaMalloc(&d_output_pool, outputPoolSize));
    if (totalTempBytes > 0)
        CHECK_CUDA(cudaMalloc(&d_temp, totalTempBytes));

    CHECK_CUDA(cudaMalloc(&d_input_ptrs, totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_output_ptrs, totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_input_sizes, totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_output_sizes, totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_statuses, totalChunks * sizeof(nvcompStatus_t)));
    printDirectProfile("cuda_mallocs");

    std::vector<void*> h_input_ptrs(totalChunks);
    std::vector<void*> h_output_ptrs(totalChunks);
    std::vector<size_t> h_input_sizes(totalChunks);

    for (size_t c = 0; c < totalChunks; c++) {
        const auto& chunk = chunks[c];
        h_input_ptrs[c] = const_cast<uint8_t*>(
            device_inputs[chunk.buf_idx] + chunk.offset);
        h_output_ptrs[c] = static_cast<uint8_t*>(d_output_pool) + c * outputStride;
        h_input_sizes[c] = chunk.chunk_size;
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(),
        totalChunks * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(),
        totalChunks * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(),
        totalChunks * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    CHECK_NVCOMP(nvcompBatchedZstdCompressAsync(
        (const void* const*)d_input_ptrs,
        (const size_t*)d_input_sizes,
        NVCOMP_ZSTD_MAX_CHUNK,
        totalChunks,
        d_temp,
        totalTempBytes,
        (void* const*)d_output_ptrs,
        (size_t*)d_output_sizes,
        comp_opts,
        (nvcompStatus_t*)d_statuses,
        stream));

    std::vector<size_t> h_output_sizes(totalChunks);
    std::vector<nvcompStatus_t> h_statuses(totalChunks);

    CHECK_CUDA(cudaMemcpyAsync(h_output_sizes.data(), d_output_sizes,
        totalChunks * sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_statuses.data(), d_statuses,
        totalChunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    printDirectProfile("nvcomp_and_size_sync");

    for (size_t c = 0; c < totalChunks; c++) {
        if (h_statuses[c] != nvcompSuccess) {
            throw std::runtime_error("nvcomp direct payload failed on chunk "
                + std::to_string(c) + " (buffer "
                + std::to_string(chunks[c].buf_idx) + ")");
        }
        if (h_output_sizes[c] > maxOutPerChunk) {
            throw std::runtime_error("nvcomp direct payload output exceeded bound");
        }
    }

    for (size_t i = 0; i < N; i++) {
        if (sizes[i] == 0) continue;

        size_t first = chunk_start_idx[i];
        size_t count = 0;
        for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++) count++;

        if (count == 1) {
            compressedSizes[i] = h_output_sizes[first];
        } else {
            size_t totalCompressed = 0;
            for (size_t c = first; c < first + count; c++) totalCompressed += h_output_sizes[c];
            compressedSizes[i] = 8 + count * 8 + count * 8 + totalCompressed;
        }
    }

    size_t totalCompressedPayloadBytes = 0;
    for (size_t sz : compressedSizes) totalCompressedPayloadBytes += sz;

    const size_t totalPayloadBytes = 4 * sizeof(uint64_t) + totalCompressedPayloadBytes;
    bool parallelTouchPayload = true;
    if (const char* disableParallelTouch = std::getenv("CAESAR_GAE_DISABLE_PARALLEL_TOUCH_PAYLOAD")) {
        if (std::string(disableParallelTouch) == "1") parallelTouchPayload = false;
    }

    payload.clear();
    if (parallelTouchPayload) {
        payload.reserve(totalPayloadBytes);

        struct VecHack : public std::vector<uint8_t> {
            using std::vector<uint8_t>::_M_impl;
        };
        reinterpret_cast<VecHack&>(payload)._M_impl._M_finish =
            reinterpret_cast<VecHack&>(payload)._M_impl._M_start + totalPayloadBytes;

        parallel_zero_fill_payload(payload.data(), payload.size());
        printDirectProfile("payload_parallel_touch_resize");
    } else {
        payload.resize(totalPayloadBytes);
        printDirectProfile("payload_resize");
    }

    uint8_t* out = payload.data();
    for (size_t sz : compressedSizes) {
        uint64_t v = static_cast<uint64_t>(sz);
        std::memcpy(out, &v, sizeof(uint64_t));
        out += sizeof(uint64_t);
    }

    for (size_t i = 0; i < N; i++) {
        if (sizes[i] == 0) continue;

        size_t first = chunk_start_idx[i];
        size_t count = 0;
        for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++) count++;

        if (count == 1) {
            CHECK_CUDA(cudaMemcpy(out,
                static_cast<uint8_t*>(d_output_pool) + first * outputStride,
                h_output_sizes[first],
                cudaMemcpyDeviceToHost));
            out += h_output_sizes[first];
        } else {
            uint64_t nc = static_cast<uint64_t>(count);
            std::memcpy(out, &nc, sizeof(uint64_t));
            out += sizeof(uint64_t);

            for (size_t c = first; c < first + count; c++) {
                uint64_t us = static_cast<uint64_t>(chunks[c].chunk_size);
                std::memcpy(out, &us, sizeof(uint64_t));
                out += sizeof(uint64_t);
            }

            for (size_t c = first; c < first + count; c++) {
                uint64_t cs = static_cast<uint64_t>(h_output_sizes[c]);
                std::memcpy(out, &cs, sizeof(uint64_t));
                out += sizeof(uint64_t);
            }

            for (size_t c = first; c < first + count; c++) {
                CHECK_CUDA(cudaMemcpy(out,
                    static_cast<uint8_t*>(d_output_pool) + c * outputStride,
                    h_output_sizes[c],
                    cudaMemcpyDeviceToHost));
                out += h_output_sizes[c];
            }
        }
    }
    printDirectProfile("direct_d2h_into_payload");

    cudaFree(d_output_pool);
    if (d_temp) cudaFree(d_temp);
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_sizes);
    cudaFree(d_statuses);
    cudaStreamDestroy(stream);
    printDirectProfile("cleanup");

    return compressedSizes;
}


// Batched GPU decompress — handles both single-chunk and multi-chunk framing
static std::vector<std::vector<uint8_t>>
nvcomp_batch_decompress(
    const std::vector<const uint8_t*>& comp_ptrs,
    const std::vector<size_t>&         comp_sizes,
    const std::vector<size_t>&         decomp_sizes)
{
    const size_t N = comp_ptrs.size();
    std::vector<std::vector<uint8_t>> results(N);

    struct DecompChunkInfo {
        size_t buf_idx;
        const uint8_t* comp_ptr;   // points into host compressed data
        size_t comp_size;
        size_t decomp_size;
    };
    std::vector<DecompChunkInfo> chunks;
    std::vector<size_t> chunk_start_idx(N);

    for (size_t i = 0; i < N; i++) {
        chunk_start_idx[i] = chunks.size();
        if (comp_sizes[i] == 0 || decomp_sizes[i] == 0) continue;

        const uint8_t* p = comp_ptrs[i];

        // Read first 8 bytes as potential num_chunks
        uint64_t potential_nc = 0;
        memcpy(&potential_nc, p, 8);

        size_t headerSize = 8 + potential_nc * 8 + potential_nc * 8;
        bool is_multi = (potential_nc > 1 && potential_nc < 1000 && headerSize < comp_sizes[i]);

        if (is_multi) {
            const uint8_t* hp = p + 8;
            std::vector<size_t> unc_sizes(potential_nc), cmp_sizes(potential_nc);
            size_t total_unc = 0;
            for (size_t c = 0; c < potential_nc; c++) {
                memcpy(&unc_sizes[c], hp, 8); hp += 8;
                total_unc += unc_sizes[c];
            }
            for (size_t c = 0; c < potential_nc; c++) {
                memcpy(&cmp_sizes[c], hp, 8); hp += 8;
            }

            if (total_unc == decomp_sizes[i]) {
                // Valid multi-chunk
                const uint8_t* data_ptr = p + headerSize;
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
    if (totalChunks == 0) return results;

    size_t maxCompChunk   = 0;
    size_t maxDecompChunk = 0;
    size_t totalDecomp    = 0;
    for (auto& c : chunks) {
        maxCompChunk   = std::max(maxCompChunk,   c.comp_size);
        maxDecompChunk = std::max(maxDecompChunk, c.decomp_size);
        totalDecomp   += c.decomp_size;
    }

    nvcompBatchedZstdDecompressOpts_t decomp_opts = nvcompBatchedZstdDecompressDefaultOpts;

    size_t totalTempBytes = 0;
    CHECK_NVCOMP(nvcompBatchedZstdDecompressGetTempSizeAsync(
        totalChunks, maxDecompChunk, decomp_opts, &totalTempBytes, totalDecomp));

    void* d_comp_pool   = nullptr;
    void* d_decomp_pool = nullptr;
    void* d_temp        = nullptr;

    CHECK_CUDA(cudaMalloc(&d_comp_pool,   totalChunks * maxCompChunk));
    CHECK_CUDA(cudaMalloc(&d_decomp_pool, totalChunks * maxDecompChunk));
    if (totalTempBytes > 0)
        CHECK_CUDA(cudaMalloc(&d_temp, totalTempBytes));

    void* d_input_ptrs   = nullptr;
    void* d_output_ptrs  = nullptr;
    void* d_input_sizes  = nullptr;
    void* d_output_sizes = nullptr;
    void* d_statuses     = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input_ptrs,   totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_output_ptrs,  totalChunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_input_sizes,  totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_output_sizes, totalChunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_statuses,     totalChunks * sizeof(nvcompStatus_t)));

    std::vector<void*>  h_input_ptrs(totalChunks), h_output_ptrs(totalChunks);
    std::vector<size_t> h_input_sizes(totalChunks), h_output_sizes(totalChunks);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (size_t c = 0; c < totalChunks; c++) {
        uint8_t* comp_slot   = (uint8_t*)d_comp_pool   + c * maxCompChunk;
        uint8_t* decomp_slot = (uint8_t*)d_decomp_pool + c * maxDecompChunk;

        h_input_ptrs[c]   = comp_slot;
        h_output_ptrs[c]  = decomp_slot;
        h_input_sizes[c]  = chunks[c].comp_size;
        h_output_sizes[c] = chunks[c].decomp_size;

        CHECK_CUDA(cudaMemcpyAsync(comp_slot, chunks[c].comp_ptr,
            chunks[c].comp_size, cudaMemcpyHostToDevice, stream));
    }

    CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs,   h_input_ptrs.data(),   totalChunks * sizeof(void*),  cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs,  h_output_ptrs.data(),  totalChunks * sizeof(void*),  cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_input_sizes,  h_input_sizes.data(),  totalChunks * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_output_sizes, h_output_sizes.data(), totalChunks * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    CHECK_NVCOMP(nvcompBatchedZstdDecompressAsync(
        (const void* const*)d_input_ptrs,
        (const size_t*)d_input_sizes,
        (const size_t*)d_output_sizes,   // buffer sizes (max)
        (size_t*)d_output_sizes,         // actual sizes written back
        totalChunks,
        d_temp, totalTempBytes,
        (void* const*)d_output_ptrs,
        decomp_opts,
        (nvcompStatus_t*)d_statuses,
        stream));

    // see if this can be rmeoved
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<nvcompStatus_t> h_statuses(totalChunks);
    CHECK_CUDA(cudaMemcpy(h_statuses.data(), d_statuses, totalChunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));

    for (size_t c = 0; c < totalChunks; c++) {
        if (h_statuses[c] != nvcompSuccess)
            throw std::runtime_error("nvcomp Zstd decompress failed on chunk " + std::to_string(c)
                + " (buffer " + std::to_string(chunks[c].buf_idx) + ")");
    }

    for (size_t i = 0; i < N; i++) {
        if (decomp_sizes[i] == 0) continue;

        results[i].resize(decomp_sizes[i]);
        uint8_t* out = results[i].data();

        size_t first = chunk_start_idx[i];
        size_t count = 0;
        for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++) count++;

        for (size_t c = first; c < first + count; c++) {
            CHECK_CUDA(cudaMemcpy(out,
                (uint8_t*)d_decomp_pool + c * maxDecompChunk,
                chunks[c].decomp_size, cudaMemcpyDeviceToHost));
            out += chunks[c].decomp_size;
        }
    }

    cudaFree(d_comp_pool);
    cudaFree(d_decomp_pool);
    if (d_temp) cudaFree(d_temp);
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_sizes);
    cudaFree(d_statuses);
    cudaStreamDestroy(stream);

    return results;
}

#endif // USE_CUDA && ENABLE_NVCOMP


PCA::PCA(int numComponents , const std::string& device)
    : numComponents_(numComponents) , device_(torch::Device(device)) {
}

PCA& PCA::fit(const torch::Tensor& x) {
    auto xDevice = x.to(device_);
    mean_ = torch::mean(xDevice, 0);
    auto xCentered = xDevice - mean_;

    // Normalize by std to handle tiny value ranges
    auto scale = torch::std(xCentered) + 1e-8f;
    xCentered = xCentered / scale;

    auto C = torch::matmul(xCentered.transpose(0, 1), xCentered) / (xCentered.size(0) - 1);

    auto eigen = torch::linalg_eigh(C);
    auto evals = std::get<0>(eigen);
    auto evecs = std::get<1>(eigen);
    auto idx = torch::argsort(evals, 0, true);
    auto Vt = torch::index_select(evecs, 1, idx).transpose(0, 1);

    if (numComponents_ > 0) {
        Vt = Vt.slice(0, 0, numComponents_);
    }
    components_ = Vt;
    return *this;
}

torch::Tensor block2Vector(const torch::Tensor& blockData , std::pair<int , int> patchSize) {
    int patchH = patchSize.first;
    int patchW = patchSize.second;

    auto sizes = blockData.sizes();
    int dims = sizes.size();

    int T = sizes[dims - 3];
    int H = sizes[dims - 2];
    int W = sizes[dims - 1];

    int nH = H / patchH;
    int nW = W / patchW;

    std::vector<int64_t> newShape;
    for (int i = 0; i < dims - 3; ++i)
        newShape.push_back(sizes[i]);
    newShape.push_back(T);
    newShape.push_back(nH);
    newShape.push_back(patchH);
    newShape.push_back(nW);
    newShape.push_back(patchW);

    auto reshaped = blockData.reshape(newShape);

    std::vector<int64_t> permuteOrder;
    int batchDims = dims - 3;

    for (int i = 0; i < batchDims; ++i) {
        permuteOrder.push_back(i);
    }

    permuteOrder.push_back(batchDims + 0);
    permuteOrder.push_back(batchDims + 3);
    permuteOrder.push_back(batchDims + 1);
    permuteOrder.push_back(batchDims + 2);
    permuteOrder.push_back(batchDims + 4);

    auto permuted = reshaped.permute(permuteOrder);

    int64_t finalDim = patchH * patchW;
    return permuted.reshape({ -1, finalDim });
}
torch::Tensor vector2Block(const torch::Tensor& vectors ,
    const std::vector<int64_t>& originalShape ,
    std::pair<int , int> patchSize) {
    int patchH = patchSize.first;
    int patchW = patchSize.second;

    int dims = originalShape.size();
    int T = originalShape[dims - 3];
    int H = originalShape[dims - 2];
    int W = originalShape[dims - 1];

    int nH = H / patchH;
    int nW = W / patchW;
    int batchDims = dims - 3;
    std::vector<int64_t> reshapedShape;
    for (int i = 0; i < batchDims; ++i) {
        reshapedShape.push_back(originalShape[i]);
    }
    reshapedShape.push_back(T);
    reshapedShape.push_back(nW);
    reshapedShape.push_back(nH);
    reshapedShape.push_back(patchH);
    reshapedShape.push_back(patchW);

    auto reshaped = vectors.reshape(reshapedShape);

    std::vector<int64_t> permuteOrder;

    for (int i = 0; i < batchDims; ++i) {
        permuteOrder.push_back(i);
    }


    permuteOrder.push_back(batchDims + 0);
    permuteOrder.push_back(batchDims + 2);
    permuteOrder.push_back(batchDims + 3);
    permuteOrder.push_back(batchDims + 1);
    permuteOrder.push_back(batchDims + 4);

    auto permuted = reshaped.permute(permuteOrder).contiguous();
    return permuted.reshape(originalShape);
}


std::pair<torch::Tensor , torch::Tensor> indexMaskPrefix(const torch::Tensor& arr2d) {
    int64_t numCols = arr2d.size(1);

    auto reversedArr = torch::flip(arr2d , { 1 });
    auto lastOneFromRight = reversedArr.to(torch::kInt32).argmax(1 , false);
    auto maskLen = numCols - lastOneFromRight - 1;

    auto arange = torch::arange(numCols , arr2d.options().dtype(torch::kLong));
    auto mask = arange.unsqueeze(0).le(maskLen.unsqueeze(1));

    auto result = arr2d.masked_select(mask);
    auto maskLenUint8 = maskLen.to(torch::kUInt8);

    return { result, maskLenUint8 };
}

torch::Tensor indexMaskReverse(const torch::Tensor& prefixMask ,
    const torch::Tensor& maskLength ,
    int64_t numCols) {
    auto device = prefixMask.device();
    auto arange = torch::arange(numCols , torch::dtype(torch::kLong).device(device));
    auto maskLength_d = maskLength.to(prefixMask.device());
    auto mask = arange.unsqueeze(0).le(maskLength_d.unsqueeze(1));

    auto arr2d = torch::zeros({ maskLength_d.size(0), numCols } ,
        torch::dtype(torch::kBool).device(device));


    arr2d.index_put_({ mask } , prefixMask.to(torch::kBool).reshape({ -1 }));
    return arr2d;
}

std::vector<uint8_t> BitUtils::bitsToBytes(const torch::Tensor& bitArray) {
    torch::Tensor bits = bitArray.dtype() == torch::kUInt8
        ? bitArray.flatten()
        : bitArray.to(torch::kUInt8).flatten();

    int64_t numBits = bits.numel();
    int64_t numBytes = (numBits + 7) / 8;
    int64_t paddedBits = numBytes * 8;

    if (paddedBits != numBits) {
        torch::Tensor padded = torch::zeros({paddedBits}, bits.options());
        padded.narrow(0, 0, numBits).copy_(bits);
        bits = padded;
    }

    torch::Tensor weights = torch::tensor(
        {128, 64, 32, 16, 8, 4, 2, 1},
        torch::TensorOptions().dtype(torch::kUInt8).device(bits.device()));

    torch::Tensor packed = (bits.reshape({numBytes, 8}) * weights)
        .sum(1)
        .to(torch::kUInt8)
        .contiguous()
        .cpu();

    std::vector<uint8_t> out(numBytes);
    std::memcpy(out.data(), packed.data_ptr<uint8_t>(), numBytes);
    return out;
}


static torch::Tensor bitsToBytesTensor(const torch::Tensor& bitArray) {
    torch::Tensor bits = bitArray.dtype() == torch::kUInt8
        ? bitArray.flatten()
        : bitArray.to(torch::kUInt8).flatten();

    int64_t numBits = bits.numel();
    int64_t numBytes = (numBits + 7) / 8;
    int64_t paddedBits = numBytes * 8;

    if (paddedBits != numBits) {
        torch::Tensor padded = torch::zeros({paddedBits}, bits.options());
        padded.narrow(0, 0, numBits).copy_(bits);
        bits = padded;
    }

    torch::Tensor weights = torch::tensor(
        {128, 64, 32, 16, 8, 4, 2, 1},
        torch::TensorOptions().dtype(torch::kUInt8).device(bits.device()));

    return (bits.reshape({numBytes, 8}) * weights)
        .sum(1)
        .to(torch::kUInt8)
        .contiguous();
}

static size_t tensorByteSize(const torch::Tensor& tensor) {
    return static_cast<size_t>(tensor.numel() * tensor.element_size());
}

static const uint8_t* tensorBytePtr(const torch::Tensor& tensor) {
    return reinterpret_cast<const uint8_t*>(tensor.data_ptr());
}


torch::Tensor BitUtils::bytesToBits(const std::vector<uint8_t>& byteSeq , int64_t numBits) {
    int64_t totalBits = byteSeq.size() * 8;

    if (numBits == -1) {
        numBits = totalBits;
    }

    numBits = std::min(numBits , totalBits);

    torch::Tensor unpacked = torch::zeros({ numBits } , torch::kBool);
    auto data = unpacked.data_ptr<bool>();

    for (int64_t bitIdx = 0; bitIdx < numBits; ++bitIdx) {
        int64_t byteIdx = bitIdx / 8;
        int64_t bitInByte = bitIdx % 8;

        uint8_t byte = byteSeq[byteIdx];
        bool bitValue = (byte >> (7 - bitInByte)) & 1;
        data[bitIdx] = bitValue;
    }

    return unpacked;
}

uint8_t BitUtils::packByte(const uint8_t* bits) {
    uint8_t byte = 0;
    for (int i = 0; i < 8; ++i) {
        if (bits[i]) {
            byte |= (1 << (7 - i));
        }
    }
    return byte;
}

void BitUtils::unpackByte(uint8_t byte , uint8_t* bits) {
    for (int i = 0; i < 8; ++i) {
        bits[i] = (byte >> (7 - i)) & 1;
    }
}

PCACompressor::PCACompressor(double nrmse ,
    double quanFactor ,
    const std::string& device ,
    const std::string& codecAlgorithm ,
    std::pair<int , int> patchSize)
    : quanBin_(nrmse* quanFactor) ,
    device_(device.rfind("cuda", 0) == 0 ? torch::kCUDA : torch::kCPU) ,
    codecAlgorithm_(codecAlgorithm) ,
    patchSize_(patchSize) ,
    vectorSize_(patchSize.first* patchSize.second) ,
    errorBound_(nrmse* std::sqrt(vectorSize_)) ,
    error_(nrmse) {

    std::cout << "PCACompressor initialized with:" << std::endl;
    std::cout << "  NRMSE: " << nrmse << std::endl;
    std::cout << "  Quantization factor: " << quanFactor << std::endl;
    std::cout << "  Device: " << device << std::endl;
    std::cout << "  Patch size: (" << patchSize.first << ", " << patchSize.second << ")" << std::endl;
    std::cout << "  Error bound: " << errorBound_ << std::endl;
}

PCACompressor::~PCACompressor() {
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif
}

GAECompressionResult PCACompressor::compress(torch::Tensor originalData ,
    torch::Tensor reconsData) {

    const bool profileGae =
        std::getenv("CAESAR_GAE_COMPRESS_PROFILE") != nullptr;
    auto gaeProfileStart = std::chrono::high_resolution_clock::now();
    auto gaeProfileLast = gaeProfileStart;
    auto printGaeProfile = [&](const char* name) {
        if (!profileGae) return;
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - gaeProfileLast).count();
        double total = std::chrono::duration<double>(now - gaeProfileStart).count();
        std::cout << "[GAE COMPRESS PROFILE] " << name
                  << " time=" << dt << " s total=" << total << " s\n";
        gaeProfileLast = now;
    };

    auto inputShape = originalData.sizes();

    int64_t totalVectors;
    if (inputShape.size() == 2) {
        totalVectors = originalData.size(0);
    }
    else {
        int T = inputShape[inputShape.size() - 3];
        int H = inputShape[inputShape.size() - 2];
        int W = inputShape[inputShape.size() - 1];
        int nH = H / patchSize_.first;
        int nW = W / patchSize_.second;
        totalVectors = T * nH * nW;
        for (int i = 0; i < (int)inputShape.size() - 3; ++i) {
            totalVectors *= inputShape[i];
        }
    }

    torch::Tensor originalDataDevice = originalData.device() == device_
        ? originalData
        : originalData.to(device_ , true);

    torch::Tensor reconsDataDevice = reconsData.device() == device_
        ? reconsData
        : reconsData.to(device_ , true);

    if (inputShape.size() == 2) {
        assert(originalDataDevice.size(1) == vectorSize_);
    }
    else {
        originalDataDevice = block2Vector(originalDataDevice , patchSize_);
        reconsDataDevice = block2Vector(reconsDataDevice , patchSize_);
    printGaeProfile("prepare_vectors");
    }

    torch::Tensor residualPca = originalDataDevice - reconsDataDevice;
    printGaeProfile("residual");

    originalDataDevice = torch::Tensor();
    reconsDataDevice = torch::Tensor();
    originalData = torch::Tensor();
    reconsData = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    torch::Tensor norms = torch::linalg_norm(residualPca , c10::nullopt , { 1 });

    MainData mainData;
    mainData.processMask = norms > errorBound_;
    printGaeProfile("process_mask");
    norms = torch::Tensor();

    if (torch::sum(mainData.processMask).item<int64_t>() <= 0) {
        MetaData metaData;
        metaData.GAE_correction_occur = false;
        metaData.pcaBasis = torch::empty({ 0, vectorSize_ } , torch::kFloat32);
        metaData.uniqueVals = torch::empty({ 0 } , torch::kFloat32);
        metaData.quanBin = quanBin_;
        metaData.nVec = mainData.processMask.size(0);
        metaData.prefixLength = 0;
        metaData.dataBytes = 0;

        auto compressedData = std::make_unique<CompressedData>();
        compressedData->data.clear();
        compressedData->dataBytes = 0;
        compressedData->coeffIntBytes = 0;

        return { metaData, std::move(compressedData), 0 };
    }

    auto indices = torch::nonzero(mainData.processMask).squeeze(1);
    residualPca = torch::index_select(residualPca , 0 , indices);
    printGaeProfile("filter_residual");
    indices = torch::Tensor();

    if (residualPca.size(0) < 2) {
        MetaData metaData;
        metaData.GAE_correction_occur = false;
        metaData.pcaBasis = torch::empty({0, vectorSize_} , torch::kFloat32);
        metaData.uniqueVals = torch::empty({0} , torch::kFloat32);
        metaData.quanBin = quanBin_;
        metaData.nVec = mainData.processMask.size(0);
        metaData.prefixLength = 0;
        metaData.dataBytes = 0;

        auto compressedData = std::make_unique<CompressedData>();
        compressedData->data.clear();
        compressedData->dataBytes = 0;
        compressedData->coeffIntBytes = 0;

        return {metaData, std::move(compressedData), 0};
    }

    PCA pca(-1 , device_.str());
    pca.fit(residualPca);
    printGaeProfile("pca_fit_float");
    torch::Tensor pcaBasis = pca.components();

    if (pcaBasis.size(0) == 0 || pcaBasis.size(1) == 0) {
        MetaData metaData;
        metaData.GAE_correction_occur = false;
        metaData.pcaBasis = torch::empty({ 0, vectorSize_ } , torch::kFloat32);
        metaData.uniqueVals = torch::empty({ 0 } , torch::kFloat32);
        metaData.quanBin = quanBin_;
        metaData.nVec = mainData.processMask.size(0);
        metaData.prefixLength = 0;
        metaData.dataBytes = 0;

        auto compressedData = std::make_unique<CompressedData>();
        compressedData->data.clear();
        compressedData->dataBytes = 0;
        compressedData->coeffIntBytes = 0;

        return { metaData, std::move(compressedData), 0 };
    }

    torch::Tensor allCoeff = torch::matmul(residualPca , pcaBasis.transpose(0 , 1));
    printGaeProfile("coeff_matmul");
    torch::Tensor reconstructedResidual = torch::matmul(allCoeff , pcaBasis);
    torch::Tensor reconError = torch::abs(reconstructedResidual - residualPca);
    double reconErrorMax = reconError.max().item<double>();
    printGaeProfile("pca_error_check");
    reconstructedResidual = torch::Tensor();
    reconError = torch::Tensor();
    allCoeff = torch::Tensor();

    if (reconErrorMax > error_ || error_ < 10.0 * static_cast<double>(std::numeric_limits<float>::epsilon())) {
#ifdef USE_CUDA
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        double leftGiB = gpu_used_gb();
        if ((prop.totalGlobalMem / (1024.0 * 1024 * 1024)) - 2.0 < leftGiB) {
            std::cerr << "[WARN] GAE near memory limit: "
                      << leftGiB << " GiB used on GPU "
                      << " GiB left on GPU " << (prop.totalGlobalMem / (1024.0 * 1024 * 1024)) - leftGiB << " GiB.\n"
                      << "[WARN] Consider: larger error bound, smaller dataset chunks,\n"
                      << "[WARN] or multiple GPUs. Attempting anyway...\n";
        }
#endif
        residualPca = residualPca.to(torch::kDouble);
        pca.fit(residualPca);
        pcaBasis = pca.components();
    }

    allCoeff = torch::matmul(residualPca , pcaBasis.transpose(0 , 1));
    residualPca = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    torch::Tensor allCoeffPower = allCoeff.pow(2);
    torch::Tensor sortIndex     = torch::argsort(allCoeffPower, 1, true).to(torch::kInt32);
    printGaeProfile("argsort");

    torch::Tensor finalMask;
    const bool useFusedMask =
        std::getenv("CAESAR_GAE_FUSED_MASK") != nullptr &&
        std::string(std::getenv("CAESAR_GAE_FUSED_MASK")) == "1";

#if defined(USE_CUDA)
    if (useFusedMask && allCoeff.is_cuda() && sortIndex.is_cuda() &&
        allCoeff.scalar_type() == torch::kFloat32 &&
        sortIndex.scalar_type() == torch::kInt32) {
        finalMask = torch::empty(
            {allCoeff.size(0), allCoeff.size(1)},
            torch::TensorOptions().dtype(torch::kBool).device(device_));

        bool ok = caesarGaeBuildFinalMask(
            allCoeff.data_ptr<float>(),
            sortIndex.data_ptr<int32_t>(),
            finalMask.data_ptr<bool>(),
            allCoeff.size(0),
            allCoeff.size(1),
            quanBin_,
            errorBound_);

        if (!ok) {
            throw std::runtime_error("caesarGaeBuildFinalMask launch failed");
        }

        printGaeProfile("fused_final_mask");

        const bool compareFusedMask =
            std::getenv("CAESAR_GAE_FUSED_MASK_COMPARE") != nullptr &&
            std::string(std::getenv("CAESAR_GAE_FUSED_MASK_COMPARE")) == "1";

        if (compareFusedMask) {
            int64_t compareRows = std::min<int64_t>(allCoeff.size(0), 4096);
            if (const char* envRows = std::getenv("CAESAR_GAE_FUSED_MASK_COMPARE_ROWS")) {
                int64_t parsed = std::atoll(envRows);
                if (parsed > 0) compareRows = std::min<int64_t>(allCoeff.size(0), parsed);
            }

            torch::Tensor allCoeffCmp = allCoeff.narrow(0, 0, compareRows);
            torch::Tensor sortIndexCmp = sortIndex.narrow(0, 0, compareRows);
            torch::Tensor allCoeffPowerCmp = allCoeffPower.narrow(0, 0, compareRows);
            torch::Tensor finalMaskCmp = finalMask.narrow(0, 0, compareRows);

            torch::Tensor allCoeffSorted = torch::gather(allCoeffCmp, 1, sortIndexCmp.to(torch::kLong));
            torch::Tensor quanCoeffSorted = allCoeffSorted.div(quanBin_).round();

            torch::Tensor quantError = quanCoeffSorted.mul(quanBin_);
            quantError.sub_(allCoeffSorted).pow_(2);

            torch::Tensor allCoeffPowerDesc = torch::gather(allCoeffPowerCmp, 1, sortIndexCmp.to(torch::kLong));
            allCoeffPowerDesc.sub_(quantError);

            torch::Tensor totalPower = torch::sum(allCoeffPowerCmp, 1).unsqueeze(1);
            torch::Tensor stepErrors = totalPower - torch::cumsum(allCoeffPowerDesc, 1);
            torch::Tensor mask = stepErrors > (errorBound_ * errorBound_);

            torch::Tensor firstFalseIdx = torch::argmin(mask.to(torch::kInt), 1);
            auto batchIndices = torch::arange(mask.size(0),
                torch::TensorOptions().device(device_));
            mask.index_put_({batchIndices.unsqueeze(1), firstFalseIdx.unsqueeze(1)}, true);

            torch::Tensor selectedCoeffQBool = (quanCoeffSorted != 0) & mask;
            torch::Tensor oldFinalMask = torch::zeros(
                {selectedCoeffQBool.size(0), selectedCoeffQBool.size(1)},
                torch::TensorOptions().dtype(torch::kBool).device(device_));
            oldFinalMask.scatter_(1, sortIndexCmp.to(torch::kLong), selectedCoeffQBool);

            int64_t diffCount = torch::sum(finalMaskCmp != oldFinalMask).item<int64_t>();
            int64_t fusedCount = torch::sum(finalMaskCmp).item<int64_t>();
            int64_t oldCount = torch::sum(oldFinalMask).item<int64_t>();

            std::cout << "[GAE FUSED MASK COMPARE] rows=" << compareRows
                      << " diff=" << diffCount
                      << " fused_count=" << fusedCount
                      << " old_count=" << oldCount << "\n";

            if (diffCount > 0) {
                torch::Tensor mismatch = torch::nonzero(finalMaskCmp != oldFinalMask);
                auto first = mismatch.index({0}).to(torch::kCPU);
                int64_t row0 = first.index({0}).item<int64_t>();
                int64_t col0 = first.index({1}).item<int64_t>();

                torch::Tensor rowSort = sortIndexCmp.index({row0}).to(torch::kCPU);
                int64_t rank0 = -1;
                for (int64_t r = 0; r < rowSort.numel(); ++r) {
                    if (rowSort.index({r}).item<int64_t>() == col0) {
                        rank0 = r;
                        break;
                    }
                }

                bool fusedVal = finalMaskCmp.index({row0, col0}).item<bool>();
                bool oldVal = oldFinalMask.index({row0, col0}).item<bool>();

                std::cout << "[GAE FUSED MASK FIRST_DIFF] row=" << row0
                          << " col=" << col0
                          << " rank=" << rank0
                          << " fused=" << fusedVal
                          << " old=" << oldVal << "\n";

                if (rank0 >= 0) {
                    auto stepCpu = stepErrors.index({row0}).to(torch::kCPU);
                    auto maskCpu = mask.index({row0}).to(torch::kCPU);
                    auto sortCpu = sortIndexCmp.index({row0}).to(torch::kCPU);
                    auto coeffCpu = allCoeffCmp.index({row0}).to(torch::kCPU);
                    auto qCpu = quanCoeffSorted.index({row0}).to(torch::kCPU);

                    int64_t firstFalse = firstFalseIdx.index({row0}).item<int64_t>();
                    int64_t sortedCol = sortCpu.index({rank0}).item<int64_t>();
                    double coeff = coeffCpu.index({sortedCol}).item<float>();
                    double q = qCpu.index({rank0}).item<float>();
                    double step = stepCpu.index({rank0}).item<float>();
                    bool oldMaskAtRank = maskCpu.index({rank0}).item<bool>();

                    std::cout << "[GAE FUSED MASK DIFF_DETAIL] row=" << row0
                              << " rank=" << rank0
                              << " sorted_col=" << sortedCol
                              << " firstFalse=" << firstFalse
                              << " coeff=" << coeff
                              << " q=" << q
                              << " stepError=" << step
                              << " threshold=" << (errorBound_ * errorBound_)
                              << " oldMaskAtRank=" << oldMaskAtRank << "\n";
                }
            }

            printGaeProfile("fused_mask_compare_old_path");
        }

        allCoeffPower = torch::Tensor();
        sortIndex = torch::Tensor();
    } else
#endif
    {
        if (useFusedMask) {
            std::cout << "[GAE FUSED MASK SKIP] allCoeff_cuda=" << allCoeff.is_cuda()
                      << " sortIndex_cuda=" << sortIndex.is_cuda()
                      << " allCoeff_dtype=" << allCoeff.scalar_type()
                      << " sortIndex_dtype=" << sortIndex.scalar_type()
                      << " rows=" << allCoeff.size(0)
                      << " cols=" << allCoeff.size(1)
                      << "\n";
        }
        torch::Tensor allCoeffSorted = torch::gather(allCoeff, 1, sortIndex.to(torch::kLong));
        torch::Tensor quanCoeffSorted = allCoeffSorted.div(quanBin_).round();

        torch::Tensor quantError = quanCoeffSorted.mul(quanBin_);
        quantError.sub_(allCoeffSorted).pow_(2);

        torch::Tensor allCoeffPowerDesc = torch::gather(allCoeffPower, 1, sortIndex.to(torch::kLong));
        allCoeffPowerDesc.sub_(quantError);
        printGaeProfile("coeff_quant_error");
        allCoeffSorted = torch::Tensor();
        quantError = torch::Tensor();

        torch::Tensor totalPower = torch::sum(allCoeffPower, 1).unsqueeze(1);
        allCoeffPower = torch::Tensor();
#ifdef USE_CUDA
        cleanupGPUMemory();
#endif

        torch::Tensor stepErrors = totalPower - torch::cumsum(allCoeffPowerDesc, 1);
        printGaeProfile("cumsum_step_errors");
        allCoeffPowerDesc = torch::Tensor();
        totalPower        = torch::Tensor();

        torch::Tensor mask = stepErrors > (errorBound_ * errorBound_);
        printGaeProfile("final_mask_threshold");
        stepErrors = torch::Tensor();

        torch::Tensor firstFalseIdx = torch::argmin(mask.to(torch::kInt), 1);
        auto batchIndices = torch::arange(mask.size(0),
            torch::TensorOptions().device(device_));
        mask.index_put_({batchIndices.unsqueeze(1), firstFalseIdx.unsqueeze(1)}, true);
        printGaeProfile("final_mask_index_put");
        firstFalseIdx = torch::Tensor();
        batchIndices  = torch::Tensor();

        torch::Tensor selectedCoeffQBool = (quanCoeffSorted != 0) & mask;
        printGaeProfile("selected_coeff_bool");
        mask             = torch::Tensor();

        quanCoeffSorted  = torch::Tensor();

        finalMask = torch::zeros(
            {selectedCoeffQBool.size(0), selectedCoeffQBool.size(1)},
            torch::TensorOptions().dtype(torch::kBool).device(device_));
        finalMask.scatter_(1, sortIndex.to(torch::kLong), selectedCoeffQBool);
        printGaeProfile("final_mask_scatter");
        selectedCoeffQBool = torch::Tensor();
        sortIndex          = torch::Tensor();
    }
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    torch::Tensor coeffIntFlatten = torch::round(
        allCoeff.masked_select(finalMask) / quanBin_
    );
    printGaeProfile("coeff_masked_select");
    allCoeff        = torch::Tensor();   
#ifdef USE_CUDA
cleanupGPUMemory();
#endif

    const bool directCoeffInt =
        std::getenv("CAESAR_GAE_DIRECT_COEFFINT") != nullptr &&
        std::string(std::getenv("CAESAR_GAE_DIRECT_COEFFINT")) == "1";
    const bool requestDirectCoeffInt16 =
        std::getenv("CAESAR_GAE_DIRECT_COEFFINT16") != nullptr &&
        std::string(std::getenv("CAESAR_GAE_DIRECT_COEFFINT16")) == "1";
    bool usedDirectCoeffInt16 = false;

    int64_t chunk_size = 1LL << 30;
    int64_t numel      = coeffIntFlatten.numel();

    torch::Tensor uniqueVals;
    torch::Tensor inverseIndices;

    if (requestDirectCoeffInt16) {
        double coeffMin = coeffIntFlatten.min().item<double>();
        double coeffMax = coeffIntFlatten.max().item<double>();
        if (coeffMin >= -32768.0 && coeffMax <= 32767.0) {
            uniqueVals = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            inverseIndices = coeffIntFlatten.to(torch::kInt16).contiguous();
            usedDirectCoeffInt16 = true;
        } else {
            std::cout << "[GAE DIRECT COEFFINT16] disabled: quantized coeff range=["
                      << coeffMin << "," << coeffMax << "] exceeds int16\n";
        }
    }

    if (!usedDirectCoeffInt16 && directCoeffInt) {
        uniqueVals = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        inverseIndices = coeffIntFlatten.to(torch::kInt32).contiguous();
    }
    else if (!usedDirectCoeffInt16 && numel <= chunk_size) {
        auto unique_result = at::_unique(coeffIntFlatten, true, true);
        uniqueVals     = std::get<0>(unique_result);
        inverseIndices = std::get<1>(unique_result);
    }
    else {
        std::vector<at::Tensor> inverse_parts;
        std::vector<at::Tensor> unique_parts;
        int64_t offset = 0;

        for (int64_t start = 0; start < numel; start += chunk_size) {
            int64_t current_chunk_size = std::min(chunk_size, numel - start);
            auto chunk          = coeffIntFlatten.narrow(0, start, current_chunk_size);
            auto partial_unique = at::_unique(chunk, true, true);

            unique_parts.push_back(std::get<0>(partial_unique));
            auto inv = std::get<1>(partial_unique) + offset;
            inverse_parts.push_back(inv);
            offset += std::get<0>(partial_unique).size(0);
        }

        torch::Tensor all_uniques = torch::cat(unique_parts, 0);
        unique_parts.clear();
        unique_parts.shrink_to_fit();
#ifdef USE_CUDA
        cleanupGPUMemory();
#endif

        torch::Tensor all_inverses = torch::cat(inverse_parts, 0);
        inverse_parts.clear();
        inverse_parts.shrink_to_fit();
#ifdef USE_CUDA
        cleanupGPUMemory();
#endif

        auto final_unique   = at::_unique(all_uniques, true, true);
        uniqueVals          = std::get<0>(final_unique);
        torch::Tensor remap = std::get<1>(final_unique);
        all_uniques         = torch::Tensor();

        inverseIndices = remap.index_select(0, all_inverses);
        remap          = torch::Tensor();
        all_inverses   = torch::Tensor();
    }

    coeffIntFlatten   = torch::Tensor();  
    mainData.coeffInt = inverseIndices;
    printGaeProfile(usedDirectCoeffInt16 ? "direct_coeffInt16" : (directCoeffInt ? "direct_coeffInt" : "unique_inverse"));
#ifdef USE_CUDA
cleanupGPUMemory();
#endif

    auto prefixResult = indexMaskPrefix(finalMask);
    printGaeProfile("indexMaskPrefix");
    mainData.prefixMask = prefixResult.first;
    mainData.maskLength = prefixResult.second;

    finalMask = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    MetaData metaData;
    metaData.GAE_correction_occur = true;
    metaData.pcaBasis = pcaBasis.to(torch::kFloat32).to(device_);
    metaData.uniqueVals = uniqueVals.to(torch::kFloat32).to(device_);
    metaData.quanBin = quanBin_;
    metaData.nVec = mainData.processMask.size(0);
    metaData.prefixLength = mainData.prefixMask.size(0);

    pcaBasis   = torch::Tensor();
    uniqueVals = torch::Tensor();

    auto compressResult = compressLossless(metaData , mainData);
    printGaeProfile("compressLossless");
    metaData.dataBytes = compressResult.second;

    return { metaData, std::move(compressResult.first), compressResult.second };
}

torch::Tensor PCACompressor::decompress(const torch::Tensor& reconsData ,
    const MetaData& metaData ,
    const CompressedData& compressedData) {

    if (metaData.dataBytes == 0 || metaData.pcaBasis.numel() == 0) {
        return reconsData;
    }

    auto inputShape = reconsData.sizes();

    torch::Tensor reconsDevice = reconsData.to(device_);

    bool needsReshape = (inputShape.size() != 2);
    if (needsReshape) {
        reconsDevice = block2Vector(reconsDevice , patchSize_);
    }

    MainData mainData = decompressLossless(metaData , compressedData);

    torch::Tensor indexMask = indexMaskReverse(mainData.prefixMask ,
        mainData.maskLength ,
        metaData.pcaBasis.size(0));

    torch::Tensor coeffInt;
    if (metaData.uniqueVals.numel() == 0) {
        coeffInt = mainData.coeffInt.to(torch::kFloat32);
    } else {
        coeffInt = metaData.uniqueVals.index({ mainData.coeffInt.to(torch::kLong) });
    }

    torch::Tensor coeff = torch::zeros(indexMask.sizes() ,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    coeff.masked_scatter_(indexMask , coeffInt * metaData.quanBin);
    coeffInt = torch::Tensor();
    indexMask = torch::Tensor();

    torch::Tensor pcaReconstruction = torch::matmul(coeff , metaData.pcaBasis.to(torch::kFloat32));
    coeff = torch::Tensor();

    reconsDevice.index_put_({ mainData.processMask } ,
        reconsDevice.index({ mainData.processMask }) + pcaReconstruction);
    pcaReconstruction = torch::Tensor();

    if (needsReshape) {
        reconsDevice = vector2Block(reconsDevice , inputShape.vec() , patchSize_);
    }

    return reconsDevice;
}


struct ByteView {
    const uint8_t* data;
    size_t size;
};

std::pair<std::unique_ptr<CompressedData> , int64_t>
PCACompressor::compressLossless(const MetaData& metaData , const MainData& mainData)
{
    auto compressedData = std::make_unique<CompressedData>();

    const bool profileLossless =
        std::getenv("CAESAR_GAE_LOSSLESS_PROFILE") != nullptr;
    auto losslessStart = std::chrono::high_resolution_clock::now();
    auto lastProfilePoint = losslessStart;
    auto printLosslessProfile = [&](const char* name) {
        if (!profileLossless) return;
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - lastProfilePoint).count();
        double total = std::chrono::duration<double>(now - losslessStart).count();
        std::cout << "[GAE LOSSLESS PROFILE] " << name
                  << " time=" << dt << " s total=" << total << " s\n";
        lastProfilePoint = now;
    };

    torch::Tensor processMaskPacked = bitsToBytesTensor(mainData.processMask);
    printLosslessProfile("pack_processMask");
    torch::Tensor prefixMaskPacked = bitsToBytesTensor(mainData.prefixMask);
    printLosslessProfile("pack_prefixMask");
    torch::Tensor maskLengthPacked = mainData.maskLength.contiguous();
    printLosslessProfile("pack_maskLength_contiguous");

    torch::Tensor coeffIntConverted;
    int64_t nUniqueVals = metaData.uniqueVals.size(0);
    if (nUniqueVals == 0)
        coeffIntConverted = mainData.coeffInt.contiguous();
    else if (nUniqueVals < 256)
        coeffIntConverted = mainData.coeffInt.to(torch::kUInt8).contiguous();
    else if (nUniqueVals < 32768)
        coeffIntConverted = mainData.coeffInt.to(torch::kInt16).contiguous();
    else
        coeffIntConverted = mainData.coeffInt.to(torch::kInt32).contiguous();

    printLosslessProfile("convert_coeffInt");
    const size_t raw_process_mask_bytes = tensorByteSize(processMaskPacked);
    const size_t raw_prefix_mask_bytes = tensorByteSize(prefixMaskPacked);
    const size_t raw_mask_length_bytes = tensorByteSize(maskLengthPacked);
    const size_t raw_coeff_int_bytes = tensorByteSize(coeffIntConverted);

    std::vector<uint8_t> processMaskCompressed, prefixMaskCompressed, maskLengthCompressed, coeffIntCompressed;
    std::vector<size_t> compressedSizes;

    bool use_nvcomp = false;
    bool use_hybrid = false;

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    use_nvcomp = device_.is_cuda();
    if (const char* useCpuZstd = std::getenv("CAESAR_GAE_USE_CPU_ZSTD")) {
        if (std::string(useCpuZstd) == "1") use_nvcomp = false;
    }
    if (const char* hybridNvcomp = std::getenv("CAESAR_GAE_HYBRID_NVCOMP")) {
        if (std::string(hybridNvcomp) == "1") {
            use_hybrid = device_.is_cuda();
            use_nvcomp = false;
        }
    }
#endif
    if (const char* forceCpu = std::getenv("CAESAR_GAE_FORCE_CPU_ZSTD")) {
        if (std::string(forceCpu) == "1") {
            use_nvcomp = false;
            use_hybrid = false;
        }
    }

    std::vector<uint8_t> cpuProcess, cpuPrefix, cpuMaskLength, cpuCoeffInt;
    std::vector<size_t> cpuSizes;

    auto run_cpu_zstd = [&]() {
        torch::Tensor maskLengthBytes = torch::from_blob(
            maskLengthPacked.data_ptr(),
            {static_cast<int64_t>(raw_mask_length_bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(maskLengthPacked.device()));

        torch::Tensor coeffIntBytes = torch::from_blob(
            coeffIntConverted.data_ptr(),
            {static_cast<int64_t>(raw_coeff_int_bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(coeffIntConverted.device()));

        torch::Tensor allBytesCpu = torch::cat(
            {processMaskPacked, prefixMaskPacked, maskLengthBytes, coeffIntBytes}, 0)
            .cpu()
            .contiguous();

        const uint8_t* base = allBytesCpu.data_ptr<uint8_t>();
        ByteView processMaskView{base, raw_process_mask_bytes};
        ByteView prefixMaskView{processMaskView.data + processMaskView.size, raw_prefix_mask_bytes};
        ByteView maskLengthView{prefixMaskView.data + prefixMaskView.size, raw_mask_length_bytes};
        ByteView coeffIntView{maskLengthView.data + maskLengthView.size, raw_coeff_int_bytes};

        int compressionLevel = 2;
        if (const char* envLevel = std::getenv("CAESAR_GAE_ZSTD_LEVEL")) {
            int parsedLevel = std::atoi(envLevel);
            if (parsedLevel >= 1 && parsedLevel <= 19) compressionLevel = parsedLevel;
        }

        auto zstd_compress_mt = [&](ByteView in, std::vector<uint8_t>& out,
                                    int level, int workers) -> size_t
        {
            if (in.size == 0) {
                out.clear();
                return 0;
            }

#if !defined(ZSTD_VERSION_NUMBER) || (ZSTD_VERSION_NUMBER < 10400)
            throw std::runtime_error("zstd too old: need >= 1.4.0 for multithread");
#endif

            ZSTD_CCtx* cctx = ZSTD_createCCtx();
            if (!cctx) throw std::runtime_error("ZSTD_createCCtx failed");

            size_t s1 = ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers);
            if (ZSTD_isError(s1)) {
                ZSTD_freeCCtx(cctx);
                throw std::runtime_error(std::string("ZSTD_c_nbWorkers set failed: ") + ZSTD_getErrorName(s1));
            }

            size_t s2 = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
            if (ZSTD_isError(s2)) {
                ZSTD_freeCCtx(cctx);
                throw std::runtime_error(std::string("ZSTD_c_compressionLevel set failed: ") + ZSTD_getErrorName(s2));
            }

            out.resize(ZSTD_compressBound(in.size));
            size_t compSize = ZSTD_compress2(cctx, out.data(), out.size(), in.data, in.size);
            ZSTD_freeCCtx(cctx);

            if (ZSTD_isError(compSize))
                throw std::runtime_error(std::string("zstd compress2 failed: ") + ZSTD_getErrorName(compSize));

            out.resize(compSize);
            return compSize;
        };

        int workers = std::min(get_allocated_cores(), 5);
        if (const char* envWorkers = std::getenv("CAESAR_GAE_ZSTD_WORKERS")) {
            int parsedWorkers = std::atoi(envWorkers);
            if (parsedWorkers > 0) workers = std::min(parsedWorkers, get_allocated_cores());
        }

        size_t processMaskCompSize = zstd_compress_mt(processMaskView, cpuProcess, compressionLevel, workers);
        size_t prefixMaskCompSize = zstd_compress_mt(prefixMaskView, cpuPrefix, compressionLevel, workers);
        size_t maskLengthCompSize = zstd_compress_mt(maskLengthView, cpuMaskLength, compressionLevel, workers);
        size_t coeffIntCompSize = zstd_compress_mt(coeffIntView, cpuCoeffInt, compressionLevel, workers);

        cpuSizes = {processMaskCompSize, prefixMaskCompSize, maskLengthCompSize, coeffIntCompSize};
    };

    std::vector<uint8_t> nvProcess, nvPrefix, nvMaskLength, nvCoeffInt;
    std::vector<size_t> nvSizes;
    bool usedDirectPayload = false;

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    auto run_nvcomp = [&]() {
        torch::Tensor maskLengthBytes = torch::from_blob(
            maskLengthPacked.data_ptr(),
            {static_cast<int64_t>(raw_mask_length_bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(maskLengthPacked.device()));

        torch::Tensor coeffIntBytes = torch::from_blob(
            coeffIntConverted.data_ptr(),
            {static_cast<int64_t>(raw_coeff_int_bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(coeffIntConverted.device()));

        std::vector<const uint8_t*> ptrs = {
            processMaskPacked.data_ptr<uint8_t>(),
            prefixMaskPacked.data_ptr<uint8_t>(),
            maskLengthBytes.data_ptr<uint8_t>(),
            coeffIntBytes.data_ptr<uint8_t>()
        };

        std::vector<size_t> rawSizes = {
            raw_process_mask_bytes,
            raw_prefix_mask_bytes,
            raw_mask_length_bytes,
            raw_coeff_int_bytes
        };

        auto batchResults = nvcomp_batch_compress_device(ptrs, rawSizes);

        nvProcess = std::move(batchResults[0].compressed);
        nvPrefix = std::move(batchResults[1].compressed);
        nvMaskLength = std::move(batchResults[2].compressed);
        nvCoeffInt = std::move(batchResults[3].compressed);

        nvSizes = {nvProcess.size(), nvPrefix.size(), nvMaskLength.size(), nvCoeffInt.size()};
    };
      auto run_nvcomp_direct_payload = [&]() {
          torch::Tensor maskLengthBytes = torch::from_blob(
              maskLengthPacked.data_ptr(),
              {static_cast<int64_t>(raw_mask_length_bytes)},
              torch::TensorOptions().dtype(torch::kUInt8).device(maskLengthPacked.device()));

          torch::Tensor coeffIntBytes = torch::from_blob(
              coeffIntConverted.data_ptr(),
              {static_cast<int64_t>(raw_coeff_int_bytes)},
              torch::TensorOptions().dtype(torch::kUInt8).device(coeffIntConverted.device()));

          std::vector<const uint8_t*> ptrs = {
              processMaskPacked.data_ptr<uint8_t>(),
              prefixMaskPacked.data_ptr<uint8_t>(),
              maskLengthBytes.data_ptr<uint8_t>(),
              coeffIntBytes.data_ptr<uint8_t>()
          };

          std::vector<size_t> rawSizes = {
              raw_process_mask_bytes,
              raw_prefix_mask_bytes,
              raw_mask_length_bytes,
              raw_coeff_int_bytes
          };

          compressedSizes = nvcomp_batch_compress_device_to_payload(
              ptrs, rawSizes, compressedData->data);
          usedDirectPayload = true;
      };
#endif

    if (use_hybrid) {
        std::cout << "[GAE Coeff Compression] HYBRID CPU ZSTD + NVCOMP, choosing smaller blocks\n";
        run_cpu_zstd();
        printLosslessProfile("cpu_zstd_compress");
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
        run_nvcomp();
        printLosslessProfile("nvcomp_batch_compress");
#else
        throw std::runtime_error("Hybrid nvCOMP requested but nvCOMP is not enabled");
#endif

        std::vector<std::vector<uint8_t>*> cpuBlocks = {&cpuProcess, &cpuPrefix, &cpuMaskLength, &cpuCoeffInt};
        std::vector<std::vector<uint8_t>*> nvBlocks = {&nvProcess, &nvPrefix, &nvMaskLength, &nvCoeffInt};

        std::vector<const char*> names = {"processMask", "prefixMask", "maskLength", "coeffInt"};
        for (size_t i = 0; i < 4; ++i) {
            const bool chooseNv = nvSizes[i] <= cpuSizes[i];
            std::cout << "[GAE HYBRID] " << names[i]
                      << " cpu=" << cpuSizes[i]
                      << " nvcomp=" << nvSizes[i]
                      << " chosen=" << (chooseNv ? "nvcomp" : "cpu")
                      << "\n";
        }

        processMaskCompressed = (nvSizes[0] <= cpuSizes[0]) ? std::move(nvProcess) : std::move(cpuProcess);
        prefixMaskCompressed = (nvSizes[1] <= cpuSizes[1]) ? std::move(nvPrefix) : std::move(cpuPrefix);
        maskLengthCompressed = (nvSizes[2] <= cpuSizes[2]) ? std::move(nvMaskLength) : std::move(cpuMaskLength);
        coeffIntCompressed = (nvSizes[3] <= cpuSizes[3]) ? std::move(nvCoeffInt) : std::move(cpuCoeffInt);

        compressedSizes = {
            processMaskCompressed.size(),
            prefixMaskCompressed.size(),
            maskLengthCompressed.size(),
            coeffIntCompressed.size()
        };
    }
    else if (use_nvcomp) {
        std::cout << "[GAE Coeff Compression] NVCOMP ZSTD batched with all raw inputs on GPU\n";
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
          bool directPayload = true;
          if (const char* disableDirectPayload = std::getenv("CAESAR_GAE_DISABLE_DIRECT_PAYLOAD")) {
              if (std::string(disableDirectPayload) == "1") directPayload = false;
          }
          if (directPayload) {
              run_nvcomp_direct_payload();
            printLosslessProfile("nvcomp_direct_payload");
        } else {
            run_nvcomp();
            printLosslessProfile("nvcomp_batch_compress");
            processMaskCompressed = std::move(nvProcess);
            prefixMaskCompressed = std::move(nvPrefix);
            maskLengthCompressed = std::move(nvMaskLength);
            coeffIntCompressed = std::move(nvCoeffInt);
            compressedSizes = nvSizes;
        }
#else
        throw std::runtime_error("nvCOMP requested but nvCOMP is not enabled");
#endif
    }
    else {
        std::cout << "[GAE Coeff Compression] CPU ZSTD is used (zstdmt)\n";
        run_cpu_zstd();
        printLosslessProfile("cpu_zstd_compress");
        std::cout << "Using "
                  << (std::getenv("CAESAR_GAE_ZSTD_WORKERS") ? std::getenv("CAESAR_GAE_ZSTD_WORKERS") : "5")
                  << " threads for zstd compression\n";
        processMaskCompressed = std::move(cpuProcess);
        prefixMaskCompressed = std::move(cpuPrefix);
        maskLengthCompressed = std::move(cpuMaskLength);
        coeffIntCompressed = std::move(cpuCoeffInt);
        compressedSizes = cpuSizes;
    }

    std::cout << "[GAE LOSSLESS SIZES] raw="
              << raw_process_mask_bytes << ","
              << raw_prefix_mask_bytes << ","
              << raw_mask_length_bytes << ","
              << raw_coeff_int_bytes << " comp="
              << compressedSizes[0] << ","
              << compressedSizes[1] << ","
              << compressedSizes[2] << ","
              << compressedSizes[3] << "\n";

    if (!usedDirectPayload) {
        const size_t totalCompressedPayloadBytes =
            processMaskCompressed.size() +
            prefixMaskCompressed.size() +
            maskLengthCompressed.size() +
            coeffIntCompressed.size();

        compressedData->data.clear();
        compressedData->data.resize(4 * sizeof(size_t) + totalCompressedPayloadBytes);
        printLosslessProfile("payload_resize");

        uint8_t* out = compressedData->data.data();
        for (size_t size : compressedSizes) {
            for (int i = 0; i < 8; ++i) {
                *out++ = static_cast<uint8_t>((size >> (i * 8)) & 0xFF);
            }
        }
        printLosslessProfile("payload_header");

        std::memcpy(out, processMaskCompressed.data(), processMaskCompressed.size());
        out += processMaskCompressed.size();
        printLosslessProfile("payload_copy_processMask");

        std::memcpy(out, prefixMaskCompressed.data(), prefixMaskCompressed.size());
        out += prefixMaskCompressed.size();
        printLosslessProfile("payload_copy_prefixMask");

        std::memcpy(out, maskLengthCompressed.data(), maskLengthCompressed.size());
        out += maskLengthCompressed.size();
        printLosslessProfile("payload_copy_maskLength");

        std::memcpy(out, coeffIntCompressed.data(), coeffIntCompressed.size());
        printLosslessProfile("payload_copy_coeffInt");
    } else {
        printLosslessProfile("payload_already_direct");
    }

    compressedData->coeffIntBytes = raw_coeff_int_bytes;
    compressedData->dataBytes = compressedData->data.size();
    printLosslessProfile("compressLossless_total");

    return {std::move(compressedData), compressedData->dataBytes};
}

MainData PCACompressor::decompressLossless(
    const MetaData& metaData , const CompressedData& compressedData)
{
    MainData mainData;
    size_t offset = 0;

    #define CHECK_ZSTD_BLOCK(name, off, csz) do { \
      if ((off) + (csz) > compressedData.data.size()) \
        throw std::runtime_error(std::string("OOB block: ") + (name)); \
      const uint8_t* p = compressedData.data.data() + (off); \
      std::printf("[DBG] %s off=%zu csz=%zu magic=%02x %02x %02x %02x\n", \
                  (name), (size_t)(off), (size_t)(csz), p[0], p[1], p[2], p[3]); \
    } while(0)

    std::vector<size_t> compressedSizes(4);
    for (int i = 0; i < 4; ++i) {
        size_t size = 0;
        for (int j = 0; j < 8; ++j)
            size |= (size_t)compressedData.data[offset++] << (j * 8);
        compressedSizes[i] = size;
    }

    bool use_nvcomp = false;
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    use_nvcomp = device_.is_cuda();
#endif
    if (const char* forceCpu = std::getenv("CAESAR_GAE_FORCE_CPU_ZSTD")) {
        if (std::string(forceCpu) == "1") use_nvcomp = false;
    }

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    if (use_nvcomp)
    {
        std::cout << "[GAE Coeff Decompression] NVCOMP ZSTD batched\n";

        size_t processMaskOrigSize = (metaData.nVec + 7) / 8;
        size_t prefixMaskOrigSize  = (metaData.prefixLength + 7) / 8;
        size_t coeffIntOrigSize    = compressedData.coeffIntBytes;

        // ── Step 1: decompress processMask alone (we need it to compute maskLength's size) ──
        {
            std::vector<const uint8_t*> ptrs         = { compressedData.data.data() + offset };
            std::vector<size_t>         comp_sizes   = { compressedSizes[0] };
            std::vector<size_t>         decomp_sizes = { processMaskOrigSize };

            auto res = nvcomp_batch_decompress(ptrs, comp_sizes, decomp_sizes);
            mainData.processMask = BitUtils::bytesToBits(res[0], metaData.nVec).to(device_);
        }
        offset += compressedSizes[0];

        int64_t numVecsProcessed = torch::sum(mainData.processMask).item<int64_t>();

        // ── Step 2: batch decompress the remaining 3 (prefixMask, maskLength, coeffInt) ──
        {
            std::vector<const uint8_t*> ptrs = {
                compressedData.data.data() + offset,
                compressedData.data.data() + offset + compressedSizes[1],
                compressedData.data.data() + offset + compressedSizes[1] + compressedSizes[2]
            };
            std::vector<size_t> comp_sizes   = { compressedSizes[1], compressedSizes[2], compressedSizes[3] };
            std::vector<size_t> decomp_sizes = { prefixMaskOrigSize, (size_t)numVecsProcessed, coeffIntOrigSize };

            auto res = nvcomp_batch_decompress(ptrs, comp_sizes, decomp_sizes);

            // prefixMask
            mainData.prefixMask = BitUtils::bytesToBits(res[0], metaData.prefixLength).to(device_);

            // maskLength
            mainData.maskLength = torch::from_blob(res[1].data(),
                { numVecsProcessed }, torch::kUInt8).clone().to(device_);

            // coeffInt
            int64_t nUniqueVals = metaData.uniqueVals.size(0);
            torch::ScalarType coeffDtype;
            size_t elementSize;
            if      (nUniqueVals == 0)    { coeffDtype = torch::kInt16;  elementSize = 2; }
            else if (nUniqueVals < 256)   { coeffDtype = torch::kUInt8;  elementSize = 1; }
            else if (nUniqueVals < 32768) { coeffDtype = torch::kInt16;  elementSize = 2; }
            else                          { coeffDtype = torch::kInt32;  elementSize = 4; }

            int64_t numElements = (int64_t)res[2].size() / (int64_t)elementSize;
            mainData.coeffInt = torch::empty({ numElements }, coeffDtype);
            std::memcpy(mainData.coeffInt.data_ptr(), res[2].data(), res[2].size());
            mainData.coeffInt = mainData.coeffInt.to(device_);
        }

        return mainData;  
    }
#endif

    std::cout << "[GAE Coeff Decompression] CPU ZSTD is used\n";

    // processMask
    size_t processMaskOrigSize = (metaData.nVec + 7) / 8;
    std::vector<uint8_t> processMaskVec(processMaskOrigSize);
    {
        CHECK_ZSTD_BLOCK("process_mask", offset, compressedSizes[0]);
        size_t sz = ZSTD_decompress(
            processMaskVec.data(), processMaskVec.size(),
            compressedData.data.data() + offset, compressedSizes[0]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("process_mask decompression failed");
    }
    mainData.processMask = BitUtils::bytesToBits(processMaskVec, metaData.nVec).to(device_);
    offset += compressedSizes[0];

    // prefixMask
    size_t prefixMaskOrigSize = (metaData.prefixLength + 7) / 8;
    std::vector<uint8_t> prefixMaskVec(prefixMaskOrigSize);
    {
        CHECK_ZSTD_BLOCK("prefix_mask", offset, compressedSizes[1]);
        size_t sz = ZSTD_decompress(
            prefixMaskVec.data(), prefixMaskVec.size(),
            compressedData.data.data() + offset, compressedSizes[1]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("prefix_mask decompression failed");
    }
    mainData.prefixMask = BitUtils::bytesToBits(prefixMaskVec, metaData.prefixLength).to(device_);
    offset += compressedSizes[1];

    // maskLength
    int64_t numVecsProcessed = torch::sum(mainData.processMask).item<int64_t>();
    std::vector<uint8_t> maskLengthVec(numVecsProcessed);
    {
        CHECK_ZSTD_BLOCK("mask_length", offset, compressedSizes[2]);
        size_t sz = ZSTD_decompress(
            maskLengthVec.data(), maskLengthVec.size(),
            compressedData.data.data() + offset, compressedSizes[2]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("mask_length decompression failed");
    }
    mainData.maskLength = torch::from_blob(maskLengthVec.data(),
        { numVecsProcessed }, torch::kUInt8).clone().to(device_);
    offset += compressedSizes[2];

    // coeffInt
    int64_t nUniqueVals = metaData.uniqueVals.size(0);
    torch::ScalarType coeffDtype;
    size_t elementSize;
    if      (nUniqueVals == 0)    { coeffDtype = torch::kInt16;  elementSize = sizeof(int16_t);  }
    else if (nUniqueVals < 256)   { coeffDtype = torch::kUInt8;  elementSize = sizeof(uint8_t);  }
    else if (nUniqueVals < 32768) { coeffDtype = torch::kInt16;  elementSize = sizeof(int16_t);  }
    else                          { coeffDtype = torch::kInt32;  elementSize = sizeof(int32_t);  }

    size_t coeffIntOrigSize = compressedData.coeffIntBytes;
    std::vector<uint8_t> coeffIntVec(coeffIntOrigSize);
    {
        CHECK_ZSTD_BLOCK("coeff_int", offset, compressedSizes[3]);
        auto p   = compressedData.data.data() + offset;
        auto fsz = ZSTD_getFrameContentSize(p, compressedSizes[3]);

        size_t sz = ZSTD_decompress(
            coeffIntVec.data(), coeffIntVec.size(),
            compressedData.data.data() + offset, compressedSizes[3]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("coeff_int decompression failed");
    }

    int64_t numElements = (int64_t)coeffIntVec.size() / (int64_t)elementSize;
    mainData.coeffInt = torch::empty({ numElements }, coeffDtype);
    std::memcpy(mainData.coeffInt.data_ptr(), coeffIntVec.data(), coeffIntVec.size());
    mainData.coeffInt = mainData.coeffInt.to(device_);

    return mainData;
}


// todo remove this to be better
torch::Tensor PCACompressor::toCPUContiguous(const torch::Tensor& tensor) {
    return tensor.cpu().contiguous();
}

std::vector<uint8_t> PCACompressor::serializeTensor(const torch::Tensor& tensor) {
    auto cpuTensor = toCPUContiguous(tensor);
    auto dataPtr = cpuTensor.data_ptr();
    auto numBytes = cpuTensor.numel() * cpuTensor.element_size();

    std::vector<uint8_t> bytes(numBytes);
    std::memcpy(bytes.data() , dataPtr , numBytes);

    return bytes;
}

torch::Tensor PCACompressor::deserializeTensor(const std::vector<uint8_t>& bytes ,
    const std::vector<int64_t>& shape ,
    torch::ScalarType dtype) {
    auto tensor = torch::empty(shape , dtype);
    std::memcpy(tensor.data_ptr() , bytes.data() , bytes.size());
    return tensor;
}

void PCACompressor::cleanupGPUMemory() {

    const char* disable =
        std::getenv("CAESAR_DO_NOT_EMPTY_CUDA_CACHE");

    if (disable && std::string(disable) == "1") {
        return;
    }

#ifdef USE_CUDA
    if (device_.is_cuda()) {
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        c10::hip::HIPCachingAllocator::emptyCache();
#else
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }
#endif
}
