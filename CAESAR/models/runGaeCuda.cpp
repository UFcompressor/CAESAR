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



#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)

struct NvcompBatchCompressResult {
    torch::Tensor compressed;  // CPU kUInt8 tensor
    size_t        rawBytes;
};

// We chunk all inputs down to this size before submitting as one big batch.
static constexpr size_t NVCOMP_ZSTD_MAX_CHUNK = 16ULL * 1024 * 1024; // 16 MB

// Skips all host-to-device uploads for input data — tensors are used directly,
// so this function pipelines seamlessly after GPU-side operations like bitsToBytes.
static std::vector<NvcompBatchCompressResult>
nvcomp_batch_compress(const std::vector<torch::Tensor>& inputs)
{
    const size_t N = inputs.size();
    std::vector<NvcompBatchCompressResult> results(N);

    struct ChunkInfo {
        size_t          buf_idx;
        const uint8_t*  src_ptr;   // points directly into the GPU tensor
        size_t          chunk_size;
    };
    std::vector<ChunkInfo> chunks;
    std::vector<size_t>    chunk_start_idx(N);

    for (size_t i = 0; i < N; i++) {
        size_t sz = (size_t)inputs[i].numel();
        results[i].rawBytes = sz;
        chunk_start_idx[i]  = chunks.size();
        if (sz == 0) continue;
        const uint8_t* ptr = inputs[i].data_ptr<uint8_t>();
        size_t offset = 0;
        while (offset < sz) {
            size_t this_chunk = std::min(NVCOMP_ZSTD_MAX_CHUNK, sz - offset);
            chunks.push_back({i, ptr + offset, this_chunk});
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


    // remove size and input pool and its allocation
    void* d_output_pool  = nullptr;
    void* d_temp         = nullptr;
    void* d_input_ptrs   = nullptr;
    void* d_output_ptrs  = nullptr;
    void* d_input_sizes  = nullptr;
    void* d_output_sizes = nullptr;
    void* d_statuses     = nullptr;

    CHECK_CUDA(cudaMalloc(&d_output_pool,  totalChunks * maxOutPerChunk));
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

    // Point directly into GPU tensor memory
    for (size_t c = 0; c < totalChunks; c++) {
        h_input_ptrs[c]  = (void*)chunks[c].src_ptr;
        h_output_ptrs[c] = (uint8_t*)d_output_pool + c * maxOutPerChunk;
        h_input_sizes[c] = chunks[c].chunk_size;
    }

    // H2D: only pointer/size metadata arrays (no need to upload data)
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

    // D2H: read back sizes and statuses on the same stream — no mid-sync
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
        if (inputs[i].numel() == 0) continue;

        size_t first = chunk_start_idx[i];
        size_t count = 0;
        for (size_t c = first; c < totalChunks && chunks[c].buf_idx == i; c++) count++;

        if (count == 1) {
            results[i].compressed = torch::empty({(int64_t)h_output_sizes[first]}, torch::kUInt8);
            CHECK_CUDA(cudaMemcpy(results[i].compressed.data_ptr<uint8_t>(),
                (uint8_t*)d_output_pool + first * maxOutPerChunk,
                h_output_sizes[first], cudaMemcpyDeviceToHost));
        } else {
            size_t headerSize      = 8 + count * 8 + count * 8;
            size_t totalCompressed = 0;
            for (size_t c = first; c < first + count; c++) totalCompressed += h_output_sizes[c];

            results[i].compressed = torch::empty({(int64_t)(headerSize + totalCompressed)}, torch::kUInt8);
            uint8_t* p = results[i].compressed.data_ptr<uint8_t>();

            uint64_t nc = count;
            memcpy(p, &nc, 8); p += 8;
            for (size_t c = first; c < first + count; c++) {
                uint64_t us = chunks[c].chunk_size; memcpy(p, &us, 8); p += 8;
            }
            for (size_t c = first; c < first + count; c++) {
                uint64_t cs = h_output_sizes[c]; memcpy(p, &cs, 8); p += 8;
            }
            for (size_t c = first; c < first + count; c++) {
                CHECK_CUDA(cudaMemcpy(p,
                    (uint8_t*)d_output_pool + c * maxOutPerChunk,
                    h_output_sizes[c], cudaMemcpyDeviceToHost));
                p += h_output_sizes[c];
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

torch::Tensor BitUtils::bitsToBytes(const torch::Tensor& bitArray) {
    torch::Tensor bits = bitArray.dtype() == torch::kUInt8
        ? bitArray.flatten()
        : bitArray.to(torch::kUInt8).flatten();

    int64_t numBits    = bits.numel();
    int64_t numBytes   = (numBits + 7) / 8;
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

torch::Tensor BitUtils::bytesToBits(const torch::Tensor& byteSeq , int64_t numBits) {
    torch::Tensor bytes = byteSeq.flatten().to(torch::kUInt8);
    int64_t numBytes  = bytes.numel();
    int64_t totalBits = numBytes * 8;

    if (numBits == -1) numBits = totalBits;
    numBits = std::min(numBits, totalBits);

    // Broadcast each byte against MSB-first weights to extract individual bits.
    torch::Tensor weights = torch::tensor(
        {128, 64, 32, 16, 8, 4, 2, 1},
        torch::TensorOptions().dtype(torch::kUInt8).device(bytes.device()));

    torch::Tensor bits = bytes.unsqueeze(1)  
        .bitwise_and(weights)      
        .ne(0)
        .reshape({-1});

    return bits.narrow(0, 0, numBits);
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

}

PCACompressor::~PCACompressor() {
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif
}

GAECompressionResult PCACompressor::compress(torch::Tensor originalData ,
    torch::Tensor reconsData) {

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
    }

    torch::Tensor residualPca = originalDataDevice - reconsDataDevice;

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
    torch::Tensor reconstructedResidual = torch::matmul(allCoeff , pcaBasis);
    torch::Tensor reconError = torch::abs(reconstructedResidual - residualPca);
    double reconErrorMax = reconError.max().item<double>();
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

    torch::Tensor allCoeffSorted = torch::gather(allCoeff, 1, sortIndex.to(torch::kLong));
    torch::Tensor quanCoeffSorted = torch::round(allCoeffSorted / quanBin_); 
{
    torch::Tensor diff = allCoeffSorted - quanCoeffSorted * quanBin_;
    allCoeffSorted = diff.pow(2);  
}
    torch::Tensor allCoeffPowerDesc = torch::gather(allCoeffPower, 1, sortIndex.to(torch::kLong));
    allCoeffPowerDesc.sub_(allCoeffSorted);
    allCoeffSorted = torch::Tensor();

    torch::Tensor totalPower = torch::sum(allCoeffPower, 1).unsqueeze(1);
    allCoeffPower = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    torch::Tensor stepErrors = totalPower - torch::cumsum(allCoeffPowerDesc, 1);
    allCoeffPowerDesc = torch::Tensor();
    totalPower        = torch::Tensor();

    torch::Tensor mask = stepErrors > (errorBound_ * errorBound_);
    stepErrors = torch::Tensor();

    torch::Tensor firstFalseIdx = torch::argmin(mask.to(torch::kInt), 1);
    auto batchIndices = torch::arange(mask.size(0),
        torch::TensorOptions().device(device_));
    mask.index_put_({batchIndices.unsqueeze(1), firstFalseIdx.unsqueeze(1)}, true);
    firstFalseIdx = torch::Tensor();
    batchIndices  = torch::Tensor();

    torch::Tensor selectedCoeffQBool = (quanCoeffSorted != 0) & mask;
    mask             = torch::Tensor();

    quanCoeffSorted  = torch::Tensor();

    torch::Tensor finalMask = torch::zeros(
        {selectedCoeffQBool.size(0), selectedCoeffQBool.size(1)},
        torch::TensorOptions().dtype(torch::kBool).device(device_));
    finalMask.scatter_(1, sortIndex.to(torch::kLong), selectedCoeffQBool);
    selectedCoeffQBool = torch::Tensor();
    sortIndex          = torch::Tensor();
#ifdef USE_CUDA
    cleanupGPUMemory();
#endif

    torch::Tensor coeffIntFlatten = torch::round(
        allCoeff.masked_select(finalMask) / quanBin_
    );
    allCoeff        = torch::Tensor();   
#ifdef USE_CUDA
cleanupGPUMemory();
#endif

    torch::Tensor uniqueVals, inverseIndices;
    int64_t chunk_size = 1LL << 30;
    int64_t numel      = coeffIntFlatten.numel();

    if (numel <= chunk_size) {
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
#ifdef USE_CUDA
cleanupGPUMemory();
#endif

    auto prefixResult = indexMaskPrefix(finalMask);
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

    int64_t n_processed = torch::sum(mainData.processMask).item<int64_t>();
    int64_t n_total = mainData.processMask.size(0);

    if (needsReshape) {
        reconsDevice = vector2Block(reconsDevice , inputShape.vec() , patchSize_);
    }

    return reconsDevice;
}

std::pair<std::unique_ptr<CompressedData> , int64_t>
PCACompressor::compressLossless(const MetaData& metaData , const MainData& mainData)
{
    auto compressedData = std::make_unique<CompressedData>();
    int64_t totalBytes = 0;

    torch::Tensor processMaskBytes = BitUtils::bitsToBytes(mainData.processMask.to(torch::kUInt8));
    torch::Tensor prefixMaskBytes  = BitUtils::bitsToBytes(mainData.prefixMask.to(torch::kUInt8));
    torch::Tensor maskLengthBytes = mainData.maskLength.contiguous().view(torch::kUInt8);

    torch::Tensor coeffIntConverted;
    int64_t nUniqueVals = metaData.uniqueVals.size(0);
    if (nUniqueVals < 256)
        coeffIntConverted = mainData.coeffInt.to(torch::kUInt8);
    else if (nUniqueVals < 32768)
        coeffIntConverted = mainData.coeffInt.to(torch::kInt16);
    else
        coeffIntConverted = mainData.coeffInt.to(torch::kInt32);
    torch::Tensor coeffIntBytes = coeffIntConverted.contiguous().view(torch::kUInt8);
    const int compressionLevel = 3;

    size_t raw_process_mask_bytes = (size_t)processMaskBytes.numel();
    size_t raw_prefix_mask_bytes  = (size_t)prefixMaskBytes.numel();
    size_t raw_mask_length_bytes  = (size_t)maskLengthBytes.numel();
    size_t raw_coeff_int_bytes    = (size_t)coeffIntBytes.numel();

    bool use_nvcomp = false;
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    use_nvcomp = device_.is_cuda();
#endif

    // Change to tensors
    torch::Tensor processMaskCompressed, prefixMaskCompressed,
                  maskLengthCompressed,  coeffIntCompressed;
    std::vector<size_t> compressedSizes;

    #if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    if (use_nvcomp)
    {

        // processMask/prefixMask are already on GPU (from bitsToBytes).
        std::vector<torch::Tensor> inputs = {
            processMaskBytes.contiguous(),
            prefixMaskBytes.contiguous(),
            maskLengthBytes.contiguous(),
            coeffIntBytes.contiguous()
        };

        auto batchResults = nvcomp_batch_compress(inputs);

        // Release input tensors now that compression is done.
        processMaskBytes = torch::Tensor();
        prefixMaskBytes  = torch::Tensor();
        maskLengthBytes  = torch::Tensor();
        coeffIntBytes    = torch::Tensor();

        processMaskCompressed = std::move(batchResults[0].compressed);
        prefixMaskCompressed  = std::move(batchResults[1].compressed);
        maskLengthCompressed  = std::move(batchResults[2].compressed);
        coeffIntCompressed    = std::move(batchResults[3].compressed);

        compressedSizes = {
            (size_t)processMaskCompressed.numel(),
            (size_t)prefixMaskCompressed.numel(),
            (size_t)maskLengthCompressed.numel(),
            (size_t)coeffIntCompressed.numel()
        };
    }
#endif

    if (!use_nvcomp)
    {

        auto zstd_compress_mt = [&](const torch::Tensor& in_tensor,
                                    std::vector<uint8_t>& out,
                                    int level,
                                    int workers) -> size_t
        {
            if (in_tensor.numel() == 0) { out.clear(); return 0; }

            const uint8_t* data      = in_tensor.data_ptr<uint8_t>();
            size_t         data_size = (size_t)in_tensor.numel();

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
            size_t bound = ZSTD_compressBound(data_size);
            out.resize(bound);

            // compress
            size_t compSize = ZSTD_compress2(cctx, out.data(), out.size(), data, data_size);

            ZSTD_freeCCtx(cctx);

            if (ZSTD_isError(compSize)) {
                throw std::runtime_error(std::string("zstd compress2 failed: ") + ZSTD_getErrorName(compSize));
            }

            out.resize(compSize);
            return compSize;
        };


        torch::Tensor pmbCpu  = processMaskBytes.cpu().contiguous();
        torch::Tensor pfmbCpu = prefixMaskBytes.cpu().contiguous();
        torch::Tensor mlbCpu  = maskLengthBytes.cpu().contiguous();
        torch::Tensor cibCpu  = coeffIntBytes.cpu().contiguous();
        
        processMaskBytes = torch::Tensor();
        prefixMaskBytes  = torch::Tensor();
        maskLengthBytes  = torch::Tensor();
        coeffIntBytes    = torch::Tensor();

        const int workers = get_allocated_cores();

        std::vector<uint8_t> pmc, pfmc, mlc, cic;
        size_t processMaskCompSize = zstd_compress_mt(pmbCpu,         pmc, compressionLevel, workers);
        size_t prefixMaskCompSize  = zstd_compress_mt(pfmbCpu,        pfmc, compressionLevel, workers);
        size_t maskLengthCompSize  = zstd_compress_mt(mlbCpu,   mlc, compressionLevel, workers);
        size_t coeffIntCompSize    = zstd_compress_mt(cibCpu,   cic, compressionLevel, workers);

        maskLengthBytes = torch::Tensor();
        coeffIntBytes   = torch::Tensor();

        // Wrap compressed vectors into tensors for uniform assembly below.
        processMaskCompressed = torch::tensor(pmc, torch::kUInt8);
        prefixMaskCompressed  = torch::tensor(pfmc, torch::kUInt8);
        maskLengthCompressed  = torch::tensor(mlc, torch::kUInt8);
        coeffIntCompressed    = torch::tensor(cic, torch::kUInt8);

        compressedSizes = {
            processMaskCompSize,
            prefixMaskCompSize,
            maskLengthCompSize,
            coeffIntCompSize
        };
    }

    size_t comp_process_mask_bytes = (size_t)processMaskCompressed.numel();
    size_t comp_prefix_mask_bytes  = (size_t)prefixMaskCompressed.numel();
    size_t comp_mask_length_bytes  = (size_t)maskLengthCompressed.numel();
    size_t comp_coeff_int_bytes    = (size_t)coeffIntCompressed.numel();

    auto CR = [](size_t rawb, size_t compb) -> double {
        return compb ? (double)rawb / (double)compb : 0.0;
    };

    const size_t totalCompressedBytes =
        comp_process_mask_bytes + comp_prefix_mask_bytes +
        comp_mask_length_bytes  + comp_coeff_int_bytes;

    compressedData->data.clear();
    compressedData->data.reserve(4 * sizeof(size_t) + totalCompressedBytes);

    for (size_t sz : compressedSizes) {
        for (int i = 0; i < 8; ++i)
            compressedData->data.push_back((sz >> (i * 8)) & 0xFF);
    }

    auto append_tensor = [&](const torch::Tensor& t) {
        const uint8_t* p = t.data_ptr<uint8_t>();
        compressedData->data.insert(compressedData->data.end(), p, p + t.numel());
    };
    append_tensor(processMaskCompressed);
    append_tensor(prefixMaskCompressed);
    append_tensor(maskLengthCompressed);
    append_tensor(coeffIntCompressed);

    compressedData->coeffIntBytes = raw_coeff_int_bytes;
    totalBytes = compressedData->data.size();
    compressedData->dataBytes = totalBytes;
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

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    if (use_nvcomp)
    {

        size_t processMaskOrigSize = (metaData.nVec + 7) / 8;
        size_t prefixMaskOrigSize  = (metaData.prefixLength + 7) / 8;
        size_t coeffIntOrigSize    = compressedData.coeffIntBytes;

        // ── Step 1: decompress processMask alone (we need it to compute maskLength's size) ──
        {
            std::vector<const uint8_t*> ptrs         = { compressedData.data.data() + offset };
            std::vector<size_t>         comp_sizes   = { compressedSizes[0] };
            std::vector<size_t>         decomp_sizes = { processMaskOrigSize };

            auto res = nvcomp_batch_decompress(ptrs, comp_sizes, decomp_sizes);
            mainData.processMask = BitUtils::bytesToBits(
                torch::from_blob(res[0].data(), {(int64_t)res[0].size()}, torch::kUInt8).clone(),
                metaData.nVec).to(device_);
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
            mainData.prefixMask = BitUtils::bytesToBits(
                torch::from_blob(res[0].data(), {(int64_t)res[0].size()}, torch::kUInt8).clone(),
                metaData.prefixLength).to(device_);

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

        return mainData;  
    }
#endif


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
    mainData.processMask = BitUtils::bytesToBits(
        torch::from_blob(processMaskVec.data(), {(int64_t)processMaskVec.size()}, torch::kUInt8).clone(),
        metaData.nVec).to(device_);
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
    mainData.prefixMask = BitUtils::bytesToBits(
        torch::from_blob(prefixMaskVec.data(), {(int64_t)prefixMaskVec.size()}, torch::kUInt8).clone(),
        metaData.prefixLength).to(device_);
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
