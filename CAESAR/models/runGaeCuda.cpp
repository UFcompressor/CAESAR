#include "runGaeCuda.h"

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

// ===== nvcomp codec selector (edit here) =====
#define NVCOMP_CODEC_ZSTD     0
#define NVCOMP_CODEC_GDEFLATE 1
#define NVCOMP_CODEC_LZ4      2
#define NVCOMP_CODEC_SNAPPY   3
#define NVCOMP_CODEC_CASCADED 4

static int g_nvcomp_codec = NVCOMP_CODEC_ZSTD;

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)

struct NvcompBatchCompressResult {
    std::vector<uint8_t> compressed;
    size_t               rawBytes;
};

// We chunk all inputs down to this size before submitting as one big batch.
static constexpr size_t NVCOMP_ZSTD_MAX_CHUNK = 16ULL * 1024 * 1024; // 16 MB

static std::vector<NvcompBatchCompressResult>
nvcomp_batch_compress(
    const std::vector<const uint8_t*>& inputs,
    const std::vector<size_t>&         sizes)
{
    const size_t N = inputs.size();
    std::vector<NvcompBatchCompressResult> results(N);

    // Each logical buffer i gets split into ceil(sizes[i] / MAX_CHUNK) pieces.
    // We flatten all pieces across all buffers into one giant batch.
    struct ChunkInfo {
        size_t buf_idx;      // which of the N input buffers this came from
        size_t offset;       // byte offset within that buffer
        size_t chunk_size;   // bytes in this chunk
    };
    std::vector<ChunkInfo> chunks;
    std::vector<size_t>    chunk_start_idx(N); // index into chunks[] where buffer i starts

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
    for (auto& c : chunks) totalUncompressed += c.chunk_size;

    CHECK_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
        totalChunks, NVCOMP_ZSTD_MAX_CHUNK, comp_opts,
        &totalTempBytes, totalUncompressed));

    std::cout << "[NVCOMP] totalChunks=" << totalChunks
              << " maxOutPerChunk=" << maxOutPerChunk
              << " tempBytes=" << totalTempBytes << "\n";

    size_t inputPoolSize  = totalChunks * NVCOMP_ZSTD_MAX_CHUNK;
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

    std::vector<void*>  h_input_ptrs(totalChunks);
    std::vector<void*>  h_output_ptrs(totalChunks);
    std::vector<size_t> h_input_sizes(totalChunks);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (size_t c = 0; c < totalChunks; c++) {
        uint8_t* in_slot  = (uint8_t*)d_input_pool  + c * NVCOMP_ZSTD_MAX_CHUNK;
        uint8_t* out_slot = (uint8_t*)d_output_pool + c * maxOutPerChunk;

        h_input_ptrs[c]  = in_slot;
        h_output_ptrs[c] = out_slot;
        h_input_sizes[c] = chunks[c].chunk_size;

        CHECK_CUDA(cudaMemcpyAsync(in_slot,
            inputs[chunks[c].buf_idx] + chunks[c].offset,
            chunks[c].chunk_size,
            cudaMemcpyHostToDevice, stream));
    }

    CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs,  h_input_ptrs.data(),  totalChunks * sizeof(void*),  cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(), totalChunks * sizeof(void*),  cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(), totalChunks * sizeof(size_t), cudaMemcpyHostToDevice, stream));

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

    // see if this can be removed fro later
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<size_t>         h_output_sizes(totalChunks);
    std::vector<nvcompStatus_t> h_statuses(totalChunks);

    CHECK_CUDA(cudaMemcpy(h_output_sizes.data(), d_output_sizes, totalChunks * sizeof(size_t),         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_statuses.data(),     d_statuses,     totalChunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));

    for (size_t c = 0; c < totalChunks; c++) {
        if (h_statuses[c] != nvcompSuccess)
            throw std::runtime_error("nvcomp Zstd compress failed on chunk " + std::to_string(c)
                + " (buffer " + std::to_string(chunks[c].buf_idx) + ")");
    }

    //      Format per buffer: [num_chunks:uint64][uncomp_size_0:uint64]...[comp_size_0:uint64]...[data...]
    //      This lets decompression reconstruct without knowing original sizes externally.
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
            // Multi-chunk — write framing header then chunk data
            // Header: [num_chunks:8][uncompressed_sizes:8*count][compressed_sizes:8*count]
            size_t headerSize = 8 + count * 8 + count * 8;
            size_t totalCompressed = 0;
            for (size_t c = first; c < first + count; c++) totalCompressed += h_output_sizes[c];

            out.resize(headerSize + totalCompressed);
            uint8_t* p = out.data();

            // num_chunks (little-endian uint64)
            uint64_t nc = count;
            memcpy(p, &nc, 8); p += 8;

            // uncompressed chunk sizes
            for (size_t c = first; c < first + count; c++) {
                uint64_t us = chunks[c].chunk_size;
                memcpy(p, &us, 8); p += 8;
            }
            // compressed chunk sizes
            for (size_t c = first; c < first + count; c++) {
                uint64_t cs = h_output_sizes[c];
                memcpy(p, &cs, 8); p += 8;
            }
            // chunk data
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
    mean_ = torch::mean(xDevice , 0);
    auto xCentered = xDevice - mean_;

    auto C = torch::matmul(xCentered.transpose(0 , 1) , xCentered) / (xCentered.size(0) - 1);

    auto eigen = torch::linalg_eigh(C);
    auto evals = std::get<0>(eigen);
    auto evecs = std::get<1>(eigen);
    auto idx = torch::argsort(evals , 0 , true);
    auto Vt = torch::index_select(evecs , 1 , idx).transpose(0 , 1);

    if (numComponents_ > 0) {
        Vt = Vt.slice(0 , 0 , numComponents_);
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
    torch::Tensor bits;
    if (bitArray.dtype() != torch::kUInt8) {
        bits = bitArray.to(torch::kUInt8);
    }
    else {
        bits = bitArray;
    }

    bits = bits.contiguous().cpu();

    auto data = bits.data_ptr<uint8_t>();
    int64_t numBits = bits.numel();

    int64_t numBytes = (numBits + 7) / 8;

    std::vector<uint8_t> packed(numBytes , 0);

    for (int64_t byteIdx = 0; byteIdx < numBytes; ++byteIdx) {
        uint8_t byte = 0;
        for (int bitIdx = 0; bitIdx < 8; ++bitIdx) {
            int64_t globalBitIdx = byteIdx * 8 + bitIdx;
            if (globalBitIdx < numBits) {
                if (data[globalBitIdx]) {
                    byte |= (1 << (7 - bitIdx));
                }
            }
        }
        packed[byteIdx] = byte;
    }

    return packed;
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


GAECompressionResult PCACompressor::compress(const torch::Tensor& originalData ,
    const torch::Tensor& reconsData) {

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
        for (int i = 0; i < inputShape.size() - 3; ++i) {
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
    }

    torch::Tensor residualPca = originalDataDevice - reconsDataDevice;

    torch::Tensor norms = torch::linalg_norm(residualPca , c10::nullopt , { 1 });

    MainData mainData;
    mainData.processMask = norms > errorBound_;
    norms = torch::Tensor();

    if (torch::sum(mainData.processMask).item<int64_t>() <= 0) {
        MetaData metaData;
        metaData.GAE_correction_occur = false;
        metaData.pcaBasis = torch::empty({ 0, vectorSize_ } , torch::kFloat32);
        metaData.uniqueVals = torch::empty({ 0 } , torch::kFloat32);
        metaData.quanBin = quanBin_;
        metaData.nVec = originalData.size(0);
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
    indices = torch::Tensor();

    PCA pca(-1 , device_.str());
    pca.fit(residualPca);
    torch::Tensor pcaBasis = pca.components();
    std::cout << "finished pca\n";

    if (pcaBasis.size(0) == 0 || pcaBasis.size(1) == 0) {
        MetaData metaData;
        metaData.GAE_correction_occur = false;
        metaData.pcaBasis = torch::empty({ 0, vectorSize_ } , torch::kFloat32);
        metaData.uniqueVals = torch::empty({ 0 } , torch::kFloat32);
        metaData.quanBin = quanBin_;
        metaData.nVec = originalData.size(0);
        metaData.prefixLength = 0;
        metaData.dataBytes = 0;

        auto compressedData = std::make_unique<CompressedData>();
        compressedData->data.clear();
        compressedData->dataBytes = 0;
        compressedData->coeffIntBytes = 0;

        return { metaData, std::move(compressedData), 0 };
    }

    torch::Tensor allCoeff = torch::matmul(residualPca , pcaBasis.transpose(0 , 1));
    torch::Tensor reconstructedResidual = torch::matmul(allCoeff , pcaBasis);
    torch::Tensor reconError = torch::abs(reconstructedResidual - residualPca);
    double reconErrorMax = reconError.max().item<double>();

    reconstructedResidual = torch::Tensor();
    reconError = torch::Tensor();

    if (reconErrorMax > error_) {
        std::cout << "[WARN] High PCA reconstruction error (" << reconErrorMax
            << ") > " << error_ << " — switching to float64" << std::endl;

        residualPca = residualPca.to(torch::kDouble);
        pca.fit(residualPca);
        pcaBasis = pca.components();
        allCoeff = torch::matmul(residualPca , pcaBasis.transpose(0 , 1));
        
        pcaBasis = pcaBasis.to(torch::kFloat32);
        allCoeff = allCoeff.to(torch::kFloat32);
    }

    originalDataDevice = torch::Tensor();
    reconsDataDevice = torch::Tensor();
    residualPca = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    std::cout << "allCoeffpower\n";
    torch::Tensor allCoeffPower = allCoeff.pow(2);
    torch::Tensor sortIndex = torch::argsort(allCoeffPower , 1 , true);
    torch::Tensor allCoeffSorted = torch::gather(allCoeff , 1 , sortIndex);
    torch::Tensor quanCoeffSorted = torch::round(allCoeffSorted / quanBin_) * quanBin_;
    torch::Tensor resCoeffSorted = allCoeffSorted - quanCoeffSorted;
    allCoeffSorted = torch::Tensor();

    torch::Tensor tmp = resCoeffSorted.pow(2);
    torch::Tensor allCoeffPowerDesc = torch::gather(allCoeffPower , 1 , sortIndex) - tmp;
    tmp = torch::Tensor();
    resCoeffSorted = torch::Tensor();

    torch::Tensor stepErrors = torch::ones_like(allCoeffPowerDesc);
    torch::Tensor remainErrors = torch::sum(allCoeffPower , 1);
    allCoeffPower = torch::Tensor();

    std::cout << "before for loop of compress in gae\n";
    for (int64_t i = 0; i < stepErrors.size(1); ++i) {
        remainErrors = remainErrors - allCoeffPowerDesc.select(1 , i);
        stepErrors.select(1 , i) = remainErrors;
    }

    allCoeffPowerDesc = torch::Tensor();
    remainErrors = torch::Tensor();

    torch::Tensor mask = stepErrors > (errorBound_ * errorBound_);
    stepErrors = torch::Tensor();

    torch::Tensor firstFalseIdx = torch::argmin(mask.to(torch::kInt) , 1);
    auto batchIndices = torch::arange(mask.size(0) , torch::TensorOptions().device(device_));
    mask.index_put_({ batchIndices.unsqueeze(1), firstFalseIdx.unsqueeze(1) } , true);
    firstFalseIdx = torch::Tensor();
    batchIndices = torch::Tensor();

    torch::Tensor selectedCoeffQ = quanCoeffSorted * mask;
    quanCoeffSorted = torch::Tensor();

    torch::Tensor selectedCoeffUnsortQ = torch::zeros_like(selectedCoeffQ);
    auto idx = torch::arange(selectedCoeffQ.size(0) , torch::TensorOptions().device(device_)).unsqueeze(1);
    selectedCoeffUnsortQ.scatter_(1 , sortIndex , selectedCoeffQ);

    selectedCoeffQ = torch::Tensor();
    sortIndex = torch::Tensor();
    idx = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    mask = selectedCoeffUnsortQ != 0;
    selectedCoeffUnsortQ = torch::Tensor();

    torch::Tensor coeffIntFlatten = torch::round(
        allCoeff.masked_select(mask) / quanBin_
    );
    allCoeff = torch::Tensor();

    std::vector<at::Tensor> inverse_parts;
    std::vector<at::Tensor> unique_parts;
    int64_t chunk_size = 1LL << 30;
    int64_t numel = coeffIntFlatten.numel();
    int64_t offset = 0;

    for (int64_t start = 0; start < numel; start += chunk_size) {
        int64_t current_chunk_size = std::min(chunk_size , numel - start);
        auto chunk = coeffIntFlatten.narrow(0 , start , current_chunk_size);
        auto partial_unique = at::_unique(chunk , true , true);

        unique_parts.push_back(std::get<0>(partial_unique));

        auto inv = std::get<1>(partial_unique) + offset;
        inverse_parts.push_back(inv);
        offset += std::get<0>(partial_unique).size(0);
    }

    coeffIntFlatten = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    torch::Tensor all_uniques = torch::cat(unique_parts , 0);
    unique_parts.clear();
    unique_parts.shrink_to_fit();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    torch::Tensor all_inverses = torch::cat(inverse_parts , 0);
    inverse_parts.clear();
    inverse_parts.shrink_to_fit();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    auto final_unique = at::_unique(all_uniques , true , true);
    torch::Tensor uniqueVals = std::get<0>(final_unique);
    torch::Tensor remap = std::get<1>(final_unique);
    all_uniques = torch::Tensor();

    torch::Tensor inverseIndices = remap.index_select(0 , all_inverses);
    remap = torch::Tensor();
    all_inverses = torch::Tensor();

    mainData.coeffInt = inverseIndices;

#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    auto prefixResult = indexMaskPrefix(mask);
    mainData.prefixMask = prefixResult.first;
    mainData.maskLength = prefixResult.second;

    mask = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    MetaData metaData;
    metaData.GAE_correction_occur = true;
    metaData.pcaBasis = pcaBasis.to(torch::kFloat32).to(device_);
    metaData.uniqueVals = uniqueVals.to(device_);
    metaData.quanBin = quanBin_;
    metaData.nVec = mainData.processMask.size(0);
    metaData.prefixLength = mainData.prefixMask.size(0);

    pcaBasis = torch::Tensor();
    uniqueVals = torch::Tensor();

    std::cout << "made it to compress Lossless\n";
    auto compressResult = compressLossless(metaData , mainData);
    std::cout << "finished compress lossless\n";
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

    torch::Tensor coeffInt = metaData.uniqueVals.index({ mainData.coeffInt.to(torch::kLong) });

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

std::pair<std::unique_ptr<CompressedData> , int64_t>
PCACompressor::compressLossless(const MetaData& metaData , const MainData& mainData)
{
    // auto start = get_start_time();
    auto compressedData = std::make_unique<CompressedData>();
    int64_t totalBytes = 0;

    auto processMaskBytes = BitUtils::bitsToBytes(mainData.processMask.to(torch::kUInt8));
    auto prefixMaskBytes = BitUtils::bitsToBytes(mainData.prefixMask.to(torch::kUInt8));
    auto maskLengthBytes = serializeTensor(mainData.maskLength);

    torch::Tensor coeffIntConverted;
    int64_t nUniqueVals = metaData.uniqueVals.size(0);
    if (nUniqueVals < 256)
        coeffIntConverted = mainData.coeffInt.to(torch::kUInt8);
    else if (nUniqueVals < 32768)
        coeffIntConverted = mainData.coeffInt.to(torch::kInt16);
    else
        coeffIntConverted = mainData.coeffInt.to(torch::kInt32);
    auto coeffIntBytes = serializeTensor(coeffIntConverted);

    const int compressionLevel = 3;

    size_t raw_process_mask_bytes = 0;
    size_t raw_prefix_mask_bytes  = 0;
    size_t raw_mask_length_bytes  = 0;
    size_t raw_coeff_int_bytes    = 0;
    
    bool use_nvcomp = false;
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
use_nvcomp = device_.is_cuda();
#endif
    
std::cout << "USE_CUDA="
#ifdef USE_CUDA
          << 1
#else
          << 0
#endif
          << " ENABLE_NVCOMP="
#ifdef ENABLE_NVCOMP
          << 1
#else
          << 0
#endif
          << " is_cuda=" << device_.is_cuda() << "\n";
    std::vector<uint8_t> processMaskCompressed , prefixMaskCompressed , maskLengthCompressed , coeffIntCompressed;
    std::vector<size_t> compressedSizes;

    #if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    if (use_nvcomp)
    {
        std::cout << "[GAE Coeff Compression] NVCOMP ZSTD batched (4 buffers, 1 call)\n";

        std::vector<const uint8_t*> ptrs  = { processMaskBytes.data(), prefixMaskBytes.data(), maskLengthBytes.data(), coeffIntBytes.data() };
        std::vector<size_t>         sizes = { processMaskBytes.size(), prefixMaskBytes.size(), maskLengthBytes.size(), coeffIntBytes.size() };

        auto batchResults = nvcomp_batch_compress(ptrs, sizes );

        raw_process_mask_bytes = sizes[0];
        raw_prefix_mask_bytes  = sizes[1];
        raw_mask_length_bytes  = sizes[2];
        raw_coeff_int_bytes    = sizes[3];

        processMaskCompressed = std::move(batchResults[0].compressed);
        prefixMaskCompressed  = std::move(batchResults[1].compressed);
        maskLengthCompressed  = std::move(batchResults[2].compressed);
        coeffIntCompressed    = std::move(batchResults[3].compressed);

        // Free raw buffers
        processMaskBytes.clear(); processMaskBytes.shrink_to_fit();
        prefixMaskBytes.clear();  prefixMaskBytes.shrink_to_fit();
        maskLengthBytes.clear();  maskLengthBytes.shrink_to_fit();
        coeffIntBytes.clear();    coeffIntBytes.shrink_to_fit();

        compressedSizes = {
            processMaskCompressed.size(),
            prefixMaskCompressed.size(),
            maskLengthCompressed.size(),
            coeffIntCompressed.size()
        };
    }
#endif

//     if (!use_nvcomp)
//     {
//         std::cout << "[GAE Coeff Compression] CPU ZSTD is used\n";
//         size_t processMaskBound = ZSTD_compressBound(processMaskBytes.size());
//         processMaskCompressed.resize(processMaskBound);
//         size_t processMaskCompSize = ZSTD_compress(
//             processMaskCompressed.data() , processMaskCompressed.size() ,
//             processMaskBytes.data() , processMaskBytes.size() ,
//             compressionLevel);
//         processMaskBytes.clear();
//         processMaskBytes.shrink_to_fit();
//         if (ZSTD_isError(processMaskCompSize))
//             throw std::runtime_error("process_mask compression failed");
//         processMaskCompressed.resize(processMaskCompSize);

//         size_t prefixMaskBound = ZSTD_compressBound(prefixMaskBytes.size());
//         prefixMaskCompressed.resize(prefixMaskBound);
//         size_t prefixMaskCompSize = ZSTD_compress(
//             prefixMaskCompressed.data() , prefixMaskCompressed.size() ,
//             prefixMaskBytes.data() , prefixMaskBytes.size() ,
//             compressionLevel);
//         prefixMaskBytes.clear();
//         prefixMaskBytes.shrink_to_fit();
//         if (ZSTD_isError(prefixMaskCompSize))
//             throw std::runtime_error("prefix_mask compression failed");
//         prefixMaskCompressed.resize(prefixMaskCompSize);

//         size_t maskLengthBound = ZSTD_compressBound(maskLengthBytes.size());
//         maskLengthCompressed.resize(maskLengthBound);
//         size_t maskLengthCompSize = ZSTD_compress(
//             maskLengthCompressed.data() , maskLengthCompressed.size() ,
//             maskLengthBytes.data() , maskLengthBytes.size() ,
//             compressionLevel);
//         maskLengthBytes.clear();
//         maskLengthBytes.shrink_to_fit();
//         if (ZSTD_isError(maskLengthCompSize))
//             throw std::runtime_error("mask_length compression failed");
//         maskLengthCompressed.resize(maskLengthCompSize);

//         size_t coeffIntBound = ZSTD_compressBound(coeffIntBytes.size());
//         coeffIntCompressed.resize(coeffIntBound);
//         size_t coeffIntCompSize = ZSTD_compress(
//             coeffIntCompressed.data() , coeffIntCompressed.size() ,
//             coeffIntBytes.data() , coeffIntBytes.size() ,
//             compressionLevel);
//         if (ZSTD_isError(coeffIntCompSize))
//             throw std::runtime_error("coeff_int compression failed");
//         coeffIntCompressed.resize(coeffIntCompSize);

//         compressedSizes = {
//             processMaskCompSize,
//             prefixMaskCompSize,
//             maskLengthCompSize,
//             coeffIntCompSize
//         };
//     }
    if (!use_nvcomp)
    {
        std::cout << "[GAE Coeff Compression] CPU ZSTD is used (zstdmt)\n";

        // CHNAGE THIS make it work with 4 and N being the num of cpu allocate on linux
        // --- helper: single-buffer zstd multithread compression ---
        // workers=10 means zstd internally splits the input into jobs and compresses in parallel
        auto zstd_compress_mt = [&](const std::vector<uint8_t>& in,
                                    std::vector<uint8_t>& out,
                                    int level,
                                    int workers) -> size_t
        {
            if (in.empty()) {
                out.clear();
                return 0;
            }

            // sanity: require zstd >= 1.4.0 for ZSTD_c_nbWorkers
            #if !defined(ZSTD_VERSION_NUMBER) || (ZSTD_VERSION_NUMBER < 10400)
                throw std::runtime_error("zstd too old: need >= 1.4.0 for multithread (ZSTD_c_nbWorkers)");
            #endif

            ZSTD_CCtx* cctx = ZSTD_createCCtx();
            if (!cctx) throw std::runtime_error("ZSTD_createCCtx failed");

            // enable multithread
            size_t s1 = ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers);
            if (ZSTD_isError(s1)) {
                ZSTD_freeCCtx(cctx);
                throw std::runtime_error(std::string("ZSTD_c_nbWorkers set failed: ") + ZSTD_getErrorName(s1));
            }

            // set compression level
            size_t s2 = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
            if (ZSTD_isError(s2)) {
                ZSTD_freeCCtx(cctx);
                throw std::runtime_error(std::string("ZSTD_c_compressionLevel set failed: ") + ZSTD_getErrorName(s2));
            }

            // allocate output
            size_t bound = ZSTD_compressBound(in.size());
            out.resize(bound);

            // compress
            size_t compSize = ZSTD_compress2(cctx, out.data(), out.size(), in.data(), in.size());

            ZSTD_freeCCtx(cctx);

            if (ZSTD_isError(compSize)) {
                throw std::runtime_error(std::string("zstd compress2 failed: ") + ZSTD_getErrorName(compSize));
            }

            out.resize(compSize);
            return compSize;
        };

        const int workers = 10;  // you asked "split into 10 parts"

        size_t processMaskCompSize = zstd_compress_mt(processMaskBytes, processMaskCompressed, compressionLevel, workers);
        size_t prefixMaskCompSize  = zstd_compress_mt(prefixMaskBytes,  prefixMaskCompressed,  compressionLevel, workers);
        size_t maskLengthCompSize  = zstd_compress_mt(maskLengthBytes,  maskLengthCompressed,  compressionLevel, workers);
        size_t coeffIntCompSize    = zstd_compress_mt(coeffIntBytes,    coeffIntCompressed,    compressionLevel, workers);
        
        raw_process_mask_bytes = processMaskBytes.size();
        raw_prefix_mask_bytes  = prefixMaskBytes.size();
        raw_mask_length_bytes  = maskLengthBytes.size();
        raw_coeff_int_bytes    = coeffIntBytes.size();
        
        // free raw buffers after compression finished
        processMaskBytes.clear(); processMaskBytes.shrink_to_fit();
        prefixMaskBytes.clear();  prefixMaskBytes.shrink_to_fit();
        maskLengthBytes.clear();  maskLengthBytes.shrink_to_fit();
        coeffIntBytes.clear();    coeffIntBytes.shrink_to_fit();

        compressedSizes = {
            processMaskCompSize,
            prefixMaskCompSize,
            maskLengthCompSize,
            coeffIntCompSize
        };
    }

    std::cout << "meta_pca_basis_bytes      : "
              << (metaData.pcaBasis.numel() * metaData.pcaBasis.element_size()) << "\n";
    std::cout << "meta_unique_vals_bytes    : "
              << (metaData.uniqueVals.numel() * metaData.uniqueVals.element_size()) << "\n";

    // ====== comp bytes (压缩后) ======
    size_t comp_process_mask_bytes = processMaskCompressed.size();
    size_t comp_prefix_mask_bytes  = prefixMaskCompressed.size();
    size_t comp_mask_length_bytes  = maskLengthCompressed.size();
    size_t comp_coeff_int_bytes    = coeffIntCompressed.size();

    auto CR = [](size_t rawb, size_t compb) -> double {
        return compb ? (double)rawb / (double)compb : 0.0;
    };

    std::cout << "comp_process_mask_bytes   : " << comp_process_mask_bytes
              << "  (raw=" << raw_process_mask_bytes
              << ", CR=" << CR(raw_process_mask_bytes, comp_process_mask_bytes) << ")\n";
    std::cout << "comp_prefix_mask_bytes    : " << comp_prefix_mask_bytes
              << "  (raw=" << raw_prefix_mask_bytes
              << ", CR=" << CR(raw_prefix_mask_bytes,  comp_prefix_mask_bytes)  << ")\n";
    std::cout << "comp_mask_length_bytes    : " << comp_mask_length_bytes
              << "  (raw=" << raw_mask_length_bytes
              << ", CR=" << CR(raw_mask_length_bytes,  comp_mask_length_bytes)  << ")\n";
    std::cout << "comp_coeff_int_bytes      : " << comp_coeff_int_bytes
              << "  (raw=" << raw_coeff_int_bytes
              << ", CR=" << CR(raw_coeff_int_bytes,    comp_coeff_int_bytes)    << ")\n";
    for (size_t size : compressedSizes) {
        for (int i = 0; i < 8; ++i) {
            compressedData->data.push_back((size >> (i * 8)) & 0xFF);
        }
    }

    compressedData->data.insert(compressedData->data.end() ,
        processMaskCompressed.begin() , processMaskCompressed.end());
    compressedData->data.insert(compressedData->data.end() ,
        prefixMaskCompressed.begin() , prefixMaskCompressed.end());
    compressedData->data.insert(compressedData->data.end() ,
        maskLengthCompressed.begin() , maskLengthCompressed.end());
    compressedData->data.insert(compressedData->data.end() ,
        coeffIntCompressed.begin() , coeffIntCompressed.end());

    // compressedData->coeffIntBytes = coeffIntBytes.size();
    compressedData->coeffIntBytes = raw_coeff_int_bytes;
    totalBytes = compressedData->data.size();
    compressedData->dataBytes = totalBytes;

    // auto end = get_time(start);
    // std::cout << "Time spent in comrpesslossess: " << end.count() << " s\n";

    return { std::move(compressedData), totalBytes };
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

    // ═══════════════════════════════════════════════════════
    // GPU PATH — batched nvcomp, early return
    // ═══════════════════════════════════════════════════════
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
            if      (nUniqueVals < 256)   { coeffDtype = torch::kUInt8;  elementSize = 1; }
            else if (nUniqueVals < 32768) { coeffDtype = torch::kInt16;  elementSize = 2; }
            else                          { coeffDtype = torch::kInt32;  elementSize = 4; }

            int64_t numElements = (int64_t)res[2].size() / (int64_t)elementSize;
            mainData.coeffInt = torch::empty({ numElements }, coeffDtype);
            std::memcpy(mainData.coeffInt.data_ptr(), res[2].data(), res[2].size());
            mainData.coeffInt = mainData.coeffInt.to(device_);
        }

        return mainData;  // ← early return, CPU path below is never reached
    }
#endif

    // ═══════════════════════════════════════════════════════
    // CPU PATH — plain ZSTD, no gpu_decompress calls at all
    // ═══════════════════════════════════════════════════════
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
    if      (nUniqueVals < 256)   { coeffDtype = torch::kUInt8;  elementSize = sizeof(uint8_t);  }
    else if (nUniqueVals < 32768) { coeffDtype = torch::kInt16;  elementSize = sizeof(int16_t);  }
    else                          { coeffDtype = torch::kInt32;  elementSize = sizeof(int32_t);  }

    size_t coeffIntOrigSize = compressedData.coeffIntBytes;
    std::vector<uint8_t> coeffIntVec(coeffIntOrigSize);
    {
        CHECK_ZSTD_BLOCK("coeff_int", offset, compressedSizes[3]);
        auto p   = compressedData.data.data() + offset;
        auto fsz = ZSTD_getFrameContentSize(p, compressedSizes[3]);
        std::cout << "[DBG] coeff_int frameSize=" << (unsigned long long)fsz
                  << " expected=" << coeffIntOrigSize << "\n";

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
