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

PCA::PCA(int numComponents , const std::string& device)
    : numComponents_(numComponents) , device_(torch::Device(device)) {
}

PCA& PCA::fit(const torch::Tensor& x) {
    auto xDevice = x.to(device_);
    const auto N = xDevice.size(0);
    const auto D = xDevice.size(1);

    mean_ = torch::mean(xDevice, 0);

    if (N < 2) {
        auto eye = torch::eye(
            D,
            torch::TensorOptions().dtype(x.dtype()).device(device_)
        );

        if (numComponents_ > 0) {
            components_ = eye.slice(0, 0, numComponents_);
        } else {
            components_ = eye;
        }

        return *this;
    }

    auto xCentered = xDevice - mean_;

    auto C = torch::matmul(xCentered.transpose(0, 1), xCentered) / (N - 1);

    C = 0.5 * (C + C.transpose(0, 1));
    auto trace = torch::trace(C);
    auto eps = 1e-6 * trace / D;
    C = C + eps * torch::eye(
        D,
        torch::TensorOptions().dtype(C.dtype()).device(C.device())
    );

    torch::Tensor evals, evecs;
    try {
        auto eigen = torch::linalg_eigh(C);
        evals = std::get<0>(eigen);
        evecs = std::get<1>(eigen);
    }
    catch (const c10::Error&) {
        auto svd = torch::linalg_svd(xCentered, false);
        auto Vh = std::get<2>(svd);

        if (numComponents_ > 0) {
            components_ = Vh.slice(0, 0, numComponents_);
        } else {
            components_ = Vh;
        }
        return *this;
    }

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
    device_(device == "cuda" ? torch::kCUDA : torch::kCPU) ,
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
    const torch::Tensor& reconsData, const torch::Tensor& externalBasis) {

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

    torch::Tensor pcaBasis = externalBasis;   // 或成员变量 basis_
    TORCH_CHECK(pcaBasis.defined(), "externalBasis is not defined");
    pcaBasis = pcaBasis.to(device_, true).to(torch::kFloat32).contiguous();
    
    torch::Tensor allCoeff = torch::matmul(residualPca , pcaBasis.transpose(0 , 1));
    torch::Tensor reconstructedResidual = torch::matmul(allCoeff , pcaBasis);
    torch::Tensor reconError = torch::abs(reconstructedResidual - residualPca);
    double reconErrorMax = reconError.max().item<double>();

    reconstructedResidual = torch::Tensor();
    reconError = torch::Tensor();

//     if (reconErrorMax > error_) {
//         std::cout << "[WARN] High PCA reconstruction error (" << reconErrorMax
//             << ") > " << error_ << " — switching to float64" << std::endl;

//         residualPca = residualPca.to(torch::kDouble);
//         pca.fit(residualPca);
//         pcaBasis = pca.components();
//         allCoeff = torch::matmul(residualPca , pcaBasis.transpose(0 , 1));
        
//         pcaBasis = pcaBasis.to(torch::kFloat32);
//         allCoeff = allCoeff.to(torch::kFloat32);
//     }

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
    auto start = get_start_time();
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

    const int compressionLevel = 21;

    bool use_nvcomp = false;
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    use_nvcomp = device_.is_cuda();
#endif

    std::vector<uint8_t> processMaskCompressed , prefixMaskCompressed , maskLengthCompressed , coeffIntCompressed;
    std::vector<size_t> compressedSizes;

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    if (use_nvcomp)
    {
        auto gpu_compress = [&](const std::vector<uint8_t>& input) -> std::vector<uint8_t>
            {
                if (input.empty()) return {};

                const size_t MAX_CHUNK_SIZE = 256 * 1024 * 1024; // 256 MB chunks
                const size_t input_bytes = input.size();
                size_t num_chunks = (input_bytes + MAX_CHUNK_SIZE - 1) / MAX_CHUNK_SIZE;

                if (num_chunks > 1) {
                    std::cout << "[DEBUG] Splitting " << input_bytes / (1024.0 * 1024.0 * 1024.0)
                        << " GiB into " << num_chunks << " chunks of max "
                        << MAX_CHUNK_SIZE / (1024.0 * 1024.0) << " MB\n";
                }

                cudaStream_t stream;
                CHECK_CUDA(cudaStreamCreate(&stream));

                std::vector<size_t> h_input_sizes(num_chunks);
                std::vector<size_t> h_output_sizes(num_chunks);
                std::vector<std::vector<uint8_t>> compressed_chunks(num_chunks);

                nvcompBatchedZstdCompressOpts_t comp_opts{};

                // Process each chunk independently to avoid huge memory allocation
                for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                    size_t offset = chunk_idx * MAX_CHUNK_SIZE;
                    size_t chunk_size = std::min(MAX_CHUNK_SIZE , input_bytes - offset);
                    h_input_sizes[chunk_idx] = chunk_size;

                    // Allocate GPU memory for this chunk only
                    void* d_input = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_input , chunk_size));
                    CHECK_CUDA(cudaMemcpyAsync(d_input , input.data() + offset , chunk_size ,
                        cudaMemcpyHostToDevice , stream));

                    void** d_inputs = nullptr;
                    size_t* d_input_sizes = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_inputs , sizeof(void*)));
                    CHECK_CUDA(cudaMalloc(&d_input_sizes , sizeof(size_t)));
                    CHECK_CUDA(cudaMemcpyAsync(d_inputs , &d_input , sizeof(void*) ,
                        cudaMemcpyHostToDevice , stream));
                    CHECK_CUDA(cudaMemcpyAsync(d_input_sizes , &chunk_size , sizeof(size_t) ,
                        cudaMemcpyHostToDevice , stream));

                    size_t temp_bytes = 0;
                    CHECK_NVCOMP(nvcompBatchedZstdCompressGetTempSizeAsync(
                        1 , chunk_size , comp_opts , &temp_bytes , chunk_size));

                    size_t max_out_bytes = 0;
                    CHECK_NVCOMP(nvcompBatchedZstdCompressGetMaxOutputChunkSize(
                        chunk_size , comp_opts , &max_out_bytes));

                    void* d_temp = nullptr;
                    if (temp_bytes > 0) CHECK_CUDA(cudaMalloc(&d_temp , temp_bytes));

                    void* d_output = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_output , max_out_bytes));

                    void** d_outputs = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_outputs , sizeof(void*)));
                    CHECK_CUDA(cudaMemcpyAsync(d_outputs , &d_output , sizeof(void*) ,
                        cudaMemcpyHostToDevice , stream));

                    size_t* d_output_size = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_output_size , sizeof(size_t)));

                    nvcompStatus_t* d_status = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_status , sizeof(nvcompStatus_t)));

                    CHECK_NVCOMP(nvcompBatchedZstdCompressAsync(
                        (const void* const*)d_inputs ,
                        d_input_sizes ,
                        chunk_size ,
                        1 ,
                        d_temp ,
                        temp_bytes ,
                        (void* const*)d_outputs ,
                        d_output_size ,
                        comp_opts ,
                        d_status ,
                        stream));

                    CHECK_CUDA(cudaStreamSynchronize(stream));

                    nvcompStatus_t h_status;
                    CHECK_CUDA(cudaMemcpy(&h_status , d_status , sizeof(nvcompStatus_t) ,
                        cudaMemcpyDeviceToHost));
                    if (h_status != nvcompSuccess)
                        throw std::runtime_error("Chunk " + std::to_string(chunk_idx) + " compression failed");

                    size_t compressed_size;
                    CHECK_CUDA(cudaMemcpy(&compressed_size , d_output_size , sizeof(size_t) ,
                        cudaMemcpyDeviceToHost));
                    h_output_sizes[chunk_idx] = compressed_size;

                    compressed_chunks[chunk_idx].resize(compressed_size);
                    CHECK_CUDA(cudaMemcpy(compressed_chunks[chunk_idx].data() , d_output ,
                        compressed_size , cudaMemcpyDeviceToHost));

                    cudaFree(d_input);
                    cudaFree(d_inputs);
                    cudaFree(d_input_sizes);
                    cudaFree(d_output);
                    cudaFree(d_outputs);
                    cudaFree(d_output_size);
                    if (d_temp) cudaFree(d_temp);
                    cudaFree(d_status);

                    std::cout << "[DEBUG] Chunk " << chunk_idx + 1 << "/" << num_chunks
                        << " compressed: " << chunk_size / (1024.0 * 1024.0) << " MB -> "
                        << compressed_size / (1024.0 * 1024.0) << " MB\n";
                }

                cudaStreamDestroy(stream);

                std::vector<uint8_t> output;

                if (num_chunks > 1) {
                    for (int i = 0; i < 8; ++i) {
                        output.push_back((num_chunks >> (i * 8)) & 0xFF);
                    }

                    for (size_t i = 0; i < num_chunks; i++) {
                        size_t uncompressed_chunk_size = h_input_sizes[i];
                        for (int j = 0; j < 8; ++j) {
                            output.push_back((uncompressed_chunk_size >> (j * 8)) & 0xFF);
                        }
                    }

                    for (size_t chunk_size : h_output_sizes) {
                        for (int j = 0; j < 8; ++j) {
                            output.push_back((chunk_size >> (j * 8)) & 0xFF);
                        }
                    }
                }

                for (size_t i = 0; i < num_chunks; i++) {
                    output.insert(output.end() ,
                        compressed_chunks[i].begin() ,
                        compressed_chunks[i].end());
                }

                return output;
            };

        std::cout << "[GAE Coeff Compression] NVCOMP ZSTD is used\n";
        processMaskCompressed = gpu_compress(processMaskBytes);
        processMaskBytes.clear();
        processMaskBytes.shrink_to_fit();
        prefixMaskCompressed = gpu_compress(prefixMaskBytes);
        prefixMaskBytes.clear();
        prefixMaskBytes.shrink_to_fit();
        maskLengthCompressed = gpu_compress(maskLengthBytes);
        maskLengthBytes.clear();
        maskLengthBytes.shrink_to_fit();
        coeffIntCompressed = gpu_compress(coeffIntBytes);

        compressedSizes = {
            processMaskCompressed.size(),
            prefixMaskCompressed.size(),
            maskLengthCompressed.size(),
            coeffIntCompressed.size()
        };
    }
#endif

    if (!use_nvcomp)
    {
        std::cout << "[GAE Coeff Compression] CPU ZSTD is used\n";
        size_t processMaskBound = ZSTD_compressBound(processMaskBytes.size());
        processMaskCompressed.resize(processMaskBound);
        size_t processMaskCompSize = ZSTD_compress(
            processMaskCompressed.data() , processMaskCompressed.size() ,
            processMaskBytes.data() , processMaskBytes.size() ,
            compressionLevel);
        processMaskBytes.clear();
        processMaskBytes.shrink_to_fit();
        if (ZSTD_isError(processMaskCompSize))
            throw std::runtime_error("process_mask compression failed");
        processMaskCompressed.resize(processMaskCompSize);

        size_t prefixMaskBound = ZSTD_compressBound(prefixMaskBytes.size());
        prefixMaskCompressed.resize(prefixMaskBound);
        size_t prefixMaskCompSize = ZSTD_compress(
            prefixMaskCompressed.data() , prefixMaskCompressed.size() ,
            prefixMaskBytes.data() , prefixMaskBytes.size() ,
            compressionLevel);
        prefixMaskBytes.clear();
        prefixMaskBytes.shrink_to_fit();
        if (ZSTD_isError(prefixMaskCompSize))
            throw std::runtime_error("prefix_mask compression failed");
        prefixMaskCompressed.resize(prefixMaskCompSize);

        size_t maskLengthBound = ZSTD_compressBound(maskLengthBytes.size());
        maskLengthCompressed.resize(maskLengthBound);
        size_t maskLengthCompSize = ZSTD_compress(
            maskLengthCompressed.data() , maskLengthCompressed.size() ,
            maskLengthBytes.data() , maskLengthBytes.size() ,
            compressionLevel);
        maskLengthBytes.clear();
        maskLengthBytes.shrink_to_fit();
        if (ZSTD_isError(maskLengthCompSize))
            throw std::runtime_error("mask_length compression failed");
        maskLengthCompressed.resize(maskLengthCompSize);

        size_t coeffIntBound = ZSTD_compressBound(coeffIntBytes.size());
        coeffIntCompressed.resize(coeffIntBound);
        size_t coeffIntCompSize = ZSTD_compress(
            coeffIntCompressed.data() , coeffIntCompressed.size() ,
            coeffIntBytes.data() , coeffIntBytes.size() ,
            compressionLevel);
        if (ZSTD_isError(coeffIntCompSize))
            throw std::runtime_error("coeff_int compression failed");
        coeffIntCompressed.resize(coeffIntCompSize);

        compressedSizes = {
            processMaskCompSize,
            prefixMaskCompSize,
            maskLengthCompSize,
            coeffIntCompSize
        };
    }

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

    compressedData->coeffIntBytes = coeffIntBytes.size();
    totalBytes = compressedData->data.size();
    compressedData->dataBytes = totalBytes;

    auto end = get_time(start);
    std::cout << "Time spent in comrpesslossess: " << end.count() << " s\n";

    return { std::move(compressedData), totalBytes };
}

MainData PCACompressor::decompressLossless(
    const MetaData& metaData , const CompressedData& compressedData)
{
    auto start = get_start_time(); 
    MainData mainData;
    size_t offset = 0;

    std::vector<size_t> compressedSizes(4);
    for (int i = 0; i < 4; ++i) {
        size_t size = 0;
        for (int j = 0; j < 8; ++j)
            size |= (size_t)compressedData.data[offset++] << (j * 8);
        compressedSizes[i] = size;
    }

    //** JL modified **//
    // ZSTD is on GPU only if nvcomp is enabled
    bool use_nvcomp = false;
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
    use_nvcomp = device_.is_cuda();
#endif

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)

    auto gpu_decompress = [&](const uint8_t* comp_ptr , size_t comp_size , size_t decomp_size) -> std::vector<uint8_t>
        {
            if (comp_size == 0 || decomp_size == 0)
                return {};

            size_t num_chunks = 1;
            std::vector<size_t> chunk_uncompressed_sizes;
            std::vector<size_t> chunk_compressed_sizes;
            const uint8_t* actual_comp_ptr = comp_ptr;
            size_t actual_comp_size = comp_size;

            if (comp_size >= 16) {
                size_t potential_chunk_count = 0;
                for (int i = 0; i < 8; ++i) {
                    potential_chunk_count |= (size_t)comp_ptr[i] << (i * 8);
                }

                // If chunk count looks reasonable (> 1 and metadata fits in comp_size)
                // Metadata: 8 bytes (num_chunks) + 8*num_chunks (uncompressed sizes) + 8*num_chunks (compressed sizes)
                size_t metadata_size = 8 + potential_chunk_count * 8 + potential_chunk_count * 8;
                if (potential_chunk_count > 1 && potential_chunk_count < 1000 && metadata_size < comp_size) {
                    num_chunks = potential_chunk_count;

                    chunk_uncompressed_sizes.resize(num_chunks);
                    size_t meta_offset = 8;
                    for (size_t i = 0; i < num_chunks; i++) {
                        size_t chunk_size = 0;
                        for (int j = 0; j < 8; ++j) {
                            chunk_size |= (size_t)comp_ptr[meta_offset++] << (j * 8);
                        }
                        chunk_uncompressed_sizes[i] = chunk_size;
                    }

                    chunk_compressed_sizes.resize(num_chunks);
                    for (size_t i = 0; i < num_chunks; i++) {
                        size_t chunk_size = 0;
                        for (int j = 0; j < 8; ++j) {
                            chunk_size |= (size_t)comp_ptr[meta_offset++] << (j * 8);
                        }
                        chunk_compressed_sizes[i] = chunk_size;
                    }

                    // Verify total uncompressed size matches
                    size_t total_uncompressed = 0;
                    for (size_t size : chunk_uncompressed_sizes) {
                        total_uncompressed += size;
                    }

                    if (total_uncompressed == decomp_size) {
                        actual_comp_ptr = comp_ptr + metadata_size;
                        actual_comp_size = comp_size - metadata_size;

                        std::cout << "[DEBUG] Detected multi-chunk decompression: " << num_chunks << " chunks, total uncompressed: "
                            << total_uncompressed << " bytes (expected: " << decomp_size << " bytes)\n";
                    }
                    else {
                        std::cout << "[DEBUG] Size mismatch in multi-chunk detection (total: " << total_uncompressed
                            << ", expected: " << decomp_size << "), treating as single chunk\n";
                        num_chunks = 1;
                        chunk_uncompressed_sizes.clear();
                        chunk_compressed_sizes.clear();
                    }
                }
            }

            cudaStream_t stream;
            CHECK_CUDA(cudaStreamCreate(&stream));

            // For multi-chunk, we need to handle each chunk separately
            // For multi-chunk, process each chunk sequentially to avoid OOM
            if (num_chunks > 1) {
                std::cout << "[DEBUG] Sequential multi-chunk decompression: " << num_chunks << " chunks\n";

                std::vector<uint8_t> output(decomp_size);
                size_t output_offset = 0;
                size_t compressed_offset = 0;

                cudaStream_t stream;
                CHECK_CUDA(cudaStreamCreate(&stream));

                // Process each chunk individually
                for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                    size_t chunk_comp_size = chunk_compressed_sizes[chunk_idx];
                    size_t chunk_decomp_size = chunk_uncompressed_sizes[chunk_idx];

                    // Allocate for this chunk only
                    void* d_comp_input = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_comp_input , chunk_comp_size));
                    CHECK_CUDA(cudaMemcpyAsync(d_comp_input , actual_comp_ptr + compressed_offset ,
                        chunk_comp_size , cudaMemcpyHostToDevice , stream));

                    void** d_inputs = nullptr;
                    size_t* d_input_sizes = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_inputs , sizeof(void*)));
                    CHECK_CUDA(cudaMalloc(&d_input_sizes , sizeof(size_t)));
                    CHECK_CUDA(cudaMemcpyAsync(d_inputs , &d_comp_input , sizeof(void*) ,
                        cudaMemcpyHostToDevice , stream));
                    CHECK_CUDA(cudaMemcpyAsync(d_input_sizes , &chunk_comp_size , sizeof(size_t) ,
                        cudaMemcpyHostToDevice , stream));

                    void* d_output = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_output , chunk_decomp_size));

                    void** d_outputs = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_outputs , sizeof(void*)));
                    CHECK_CUDA(cudaMemcpyAsync(d_outputs , &d_output , sizeof(void*) ,
                        cudaMemcpyHostToDevice , stream));

                    size_t* d_output_sizes = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_output_sizes , sizeof(size_t)));
                    CHECK_CUDA(cudaMemcpyAsync(d_output_sizes , &chunk_decomp_size , sizeof(size_t) ,
                        cudaMemcpyHostToDevice , stream));

                    nvcompBatchedZstdDecompressOpts_t decomp_opts{};
                    size_t temp_bytes = 0;
                    CHECK_NVCOMP(nvcompBatchedZstdDecompressGetTempSizeAsync(
                        1 , chunk_decomp_size , decomp_opts , &temp_bytes , chunk_decomp_size));

                    void* d_temp = nullptr;
                    if (temp_bytes > 0) CHECK_CUDA(cudaMalloc(&d_temp , temp_bytes));

                    nvcompStatus_t* d_status = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_status , sizeof(nvcompStatus_t)));

                    CHECK_NVCOMP(nvcompBatchedZstdDecompressAsync(
                        (const void* const*)d_inputs ,
                        d_input_sizes ,
                        d_output_sizes ,
                        d_output_sizes ,
                        1 ,
                        d_temp , temp_bytes ,
                        (void* const*)d_outputs ,
                        decomp_opts ,
                        d_status ,
                        stream));

                    CHECK_CUDA(cudaStreamSynchronize(stream));

                    nvcompStatus_t h_status;
                    CHECK_CUDA(cudaMemcpy(&h_status , d_status , sizeof(nvcompStatus_t) ,
                        cudaMemcpyDeviceToHost));
                    if (h_status != nvcompSuccess)
                        throw std::runtime_error("Chunk " + std::to_string(chunk_idx) + " decompression failed");

                    CHECK_CUDA(cudaMemcpy(output.data() + output_offset , d_output ,
                        chunk_decomp_size , cudaMemcpyDeviceToHost));

                    cudaFree(d_comp_input);
                    cudaFree(d_inputs);
                    cudaFree(d_input_sizes);
                    cudaFree(d_output);
                    cudaFree(d_outputs);
                    cudaFree(d_output_sizes);
                    if (d_temp) cudaFree(d_temp);
                    cudaFree(d_status);

                    output_offset += chunk_decomp_size;
                    compressed_offset += chunk_comp_size;

                    if ((chunk_idx + 1) % 10 == 0 || chunk_idx == num_chunks - 1) {
                        std::cout << "[DEBUG] Decompressed chunk " << chunk_idx + 1 << "/" << num_chunks << "\n";
                    }
                }

                cudaStreamDestroy(stream);
                auto end = get_time(start);
                std::cout << "End of decomrpesslosssess: " << end.count() << " s\n";
                return output;
            }

            void* d_input = nullptr;
            CHECK_CUDA(cudaMalloc(&d_input , actual_comp_size));
            CHECK_CUDA(cudaMemcpyAsync(d_input , actual_comp_ptr , actual_comp_size , cudaMemcpyHostToDevice , stream));

            void** d_inputs = nullptr;
            size_t* d_input_sizes = nullptr;
            CHECK_CUDA(cudaMalloc(&d_inputs , sizeof(void*)));
            CHECK_CUDA(cudaMalloc(&d_input_sizes , sizeof(size_t)));
            CHECK_CUDA(cudaMemcpyAsync(d_inputs , &d_input , sizeof(void*) , cudaMemcpyHostToDevice , stream));
            CHECK_CUDA(cudaMemcpyAsync(d_input_sizes , &actual_comp_size , sizeof(size_t) , cudaMemcpyHostToDevice , stream));

            void* d_output = nullptr;
            CHECK_CUDA(cudaMalloc(&d_output , decomp_size));
            void** d_outputs = nullptr;
            CHECK_CUDA(cudaMalloc(&d_outputs , sizeof(void*)));
            CHECK_CUDA(cudaMemcpyAsync(d_outputs , &d_output , sizeof(void*) , cudaMemcpyHostToDevice , stream));

            size_t* d_output_sizes = nullptr;
            CHECK_CUDA(cudaMalloc(&d_output_sizes , sizeof(size_t)));
            CHECK_CUDA(cudaMemcpyAsync(d_output_sizes , &decomp_size , sizeof(size_t) , cudaMemcpyHostToDevice , stream));

            nvcompBatchedZstdDecompressOpts_t decomp_opts{};
            size_t temp_bytes = 0;
            CHECK_NVCOMP(nvcompBatchedZstdDecompressGetTempSizeAsync(
                num_chunks ,
                decomp_size ,
                decomp_opts ,
                &temp_bytes ,
                decomp_size));

            void* d_temp = nullptr;
            if (temp_bytes > 0)
                CHECK_CUDA(cudaMalloc(&d_temp , temp_bytes));

            nvcompStatus_t* d_statuses = nullptr;
            CHECK_CUDA(cudaMalloc(&d_statuses , sizeof(nvcompStatus_t)));

            CHECK_NVCOMP(nvcompBatchedZstdDecompressAsync(
                (const void* const*)d_inputs ,
                d_input_sizes ,
                d_output_sizes ,
                d_output_sizes ,
                num_chunks ,
                d_temp , temp_bytes ,
                (void* const*)d_outputs ,
                decomp_opts ,
                d_statuses ,
                stream));

            CHECK_CUDA(cudaStreamSynchronize(stream));

            nvcompStatus_t h_status{};
            CHECK_CUDA(cudaMemcpy(&h_status , d_statuses , sizeof(nvcompStatus_t) , cudaMemcpyDeviceToHost));
            if (h_status != nvcompSuccess)
                throw std::runtime_error("nvcompBatchedZstdDecompressAsync status != nvcompSuccess");

            std::vector<uint8_t> output(decomp_size);
            CHECK_CUDA(cudaMemcpy(output.data() , d_output , decomp_size , cudaMemcpyDeviceToHost));

            cudaFree(d_input);
            cudaFree(d_inputs);
            cudaFree(d_input_sizes);
            cudaFree(d_outputs);
            cudaFree(d_output_sizes);
            if (d_temp) cudaFree(d_temp);
            cudaFree(d_output);
            cudaFree(d_statuses);
            cudaStreamDestroy(stream);

            return output;
        };
#endif 

    size_t processMaskOrigSize = (metaData.nVec + 7) / 8;
    std::vector<uint8_t> processMaskVec(processMaskOrigSize);

    if (use_nvcomp) {
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
        std::cout << "[GAE Coeff Decompression] NVCOMP ZSTD is used\n";
        processMaskVec = gpu_decompress(
            compressedData.data.data() + offset ,
            compressedSizes[0] ,
            processMaskOrigSize);
#endif
    }
    else {
        std::cout << "[GAE Coeff Decompression] CPU ZSTD is used\n";
        size_t sz = ZSTD_decompress(
            processMaskVec.data() , processMaskVec.size() ,
            compressedData.data.data() + offset , compressedSizes[0]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("process_mask decompression failed");
    }
    mainData.processMask = BitUtils::bytesToBits(processMaskVec , metaData.nVec).to(device_);
    offset += compressedSizes[0];

    size_t prefixMaskOrigSize = (metaData.prefixLength + 7) / 8;
    std::vector<uint8_t> prefixMaskVec(prefixMaskOrigSize);

    if (use_nvcomp) {
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
        prefixMaskVec = gpu_decompress(
            compressedData.data.data() + offset ,
            compressedSizes[1] ,
            prefixMaskOrigSize);
#endif
    }
    else {
        size_t sz = ZSTD_decompress(
            prefixMaskVec.data() , prefixMaskVec.size() ,
            compressedData.data.data() + offset , compressedSizes[1]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("prefix_mask decompression failed");
    }
    mainData.prefixMask = BitUtils::bytesToBits(prefixMaskVec , metaData.prefixLength).to(device_);
    offset += compressedSizes[1];

    int64_t numVecsProcessed = torch::sum(mainData.processMask).item<int64_t>();
    std::vector<uint8_t> maskLengthVec(numVecsProcessed);

    if (use_nvcomp) {
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
        maskLengthVec = gpu_decompress(
            compressedData.data.data() + offset ,
            compressedSizes[2] ,
            numVecsProcessed);
#endif
    }
    else {
        size_t sz = ZSTD_decompress(
            maskLengthVec.data() , maskLengthVec.size() ,
            compressedData.data.data() + offset , compressedSizes[2]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("mask_length decompression failed");
    }
    mainData.maskLength = torch::from_blob(maskLengthVec.data() ,
        { numVecsProcessed } , torch::kUInt8).clone().to(device_);
    offset += compressedSizes[2];

    int64_t nUniqueVals = metaData.uniqueVals.size(0);
    torch::ScalarType coeffDtype;
    size_t elementSize;
    if (nUniqueVals < 256) {
        coeffDtype = torch::kUInt8;
        elementSize = sizeof(uint8_t);
    }
    else if (nUniqueVals < 32768) {
        coeffDtype = torch::kInt16;
        elementSize = sizeof(int16_t);
    }
    else {
        coeffDtype = torch::kInt32;
        elementSize = sizeof(int32_t);
    }

    size_t coeffIntOrigSize = compressedData.coeffIntBytes;
    std::vector<uint8_t> coeffIntVec(coeffIntOrigSize);

    if (use_nvcomp) {
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
        coeffIntVec = gpu_decompress(
            compressedData.data.data() + offset ,
            compressedSizes[3] ,
            coeffIntOrigSize);
#endif
    }
    else {
        size_t sz = ZSTD_decompress(
            coeffIntVec.data() , coeffIntVec.size() ,
            compressedData.data.data() + offset , compressedSizes[3]);
        if (ZSTD_isError(sz))
            throw std::runtime_error("coeff_int decompression failed");
    }

    int64_t numElements = coeffIntVec.size() / elementSize;
    mainData.coeffInt = torch::empty({ numElements } , coeffDtype);
    std::memcpy(mainData.coeffInt.data_ptr() , coeffIntVec.data() , coeffIntVec.size());
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
