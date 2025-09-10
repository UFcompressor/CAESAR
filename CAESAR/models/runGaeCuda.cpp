#include "runGaeCuda.h"


PCA::PCA(int numComponents, const std::string& device)
    : numComponents_(numComponents), device_(torch::Device(device)) {}

PCA& PCA::fit(const torch::Tensor& x) {
    auto xDevice = x.to(device_);
    mean_ = torch::mean(xDevice, 0);
    auto xCentered = xDevice - mean_;

    auto C = torch::matmul(xCentered.t(), xCentered) / (xCentered.size(0) - 1);

    auto eigen = torch::linalg_eigh(C);
    auto evals = std::get<0>(eigen);
    auto evecs = std::get<1>(eigen);

    auto idx = std::get<1>(torch::sort(evals, /*dim=*/0, /*descending=*/true));
    auto Vt = torch::index_select(evecs, 1, idx).t();

    if (numComponents_ > 0) {
        Vt = Vt.slice(0, 0, numComponents_);
    }

    components_ = Vt;

    return *this;
}

// this may work correly
torch::Tensor block2Vector(const torch::Tensor& blockData, std::pair<int, int> patchSize) {
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
    for (int i = 0; i < dims - 3; ++i) permuteOrder.push_back(i);
    permuteOrder.push_back(dims - 3);
    permuteOrder.push_back(dims - 1); 
    permuteOrder.push_back(dims - 2);
    permuteOrder.push_back(dims);   
    permuteOrder.push_back(dims + 1); 

    auto permuted = reshaped.permute(permuteOrder);

    int64_t finalDim = patchH * patchW;

    std::vector<int64_t> finalShape;
    for (int i = 0; i < dims - 3; ++i) finalShape.push_back(sizes[i]);
    finalShape.push_back(T * nH * nW);
    finalShape.push_back(finalDim);

    return permuted.reshape(finalShape);
}
// idk if this works properly
torch::Tensor vector2Block(const torch::Tensor& vectors,
                           const std::vector<int64_t>& originalShape,
                           std::pair<int, int> patchSize) {
    int patchH = patchSize.first;
    int patchW = patchSize.second;

    int dims = originalShape.size();
    int T = originalShape[dims - 3];
    int H = originalShape[dims - 2];
    int W = originalShape[dims - 1];

    int nH = H / patchH;
    int nW = W / patchW;


    std::vector<int64_t> reshapedShape;
    for (int i = 0; i < dims - 3; ++i) reshapedShape.push_back(originalShape[i]);
    reshapedShape.push_back(nH);
    reshapedShape.push_back(nW);
    reshapedShape.push_back(T); 
    reshapedShape.push_back(patchH);
    reshapedShape.push_back(patchW);

    auto reshaped = vectors.reshape(reshapedShape);

std::vector<int64_t> permuteOrder;
int batchDims = dims - 3;
for (int i = 0; i < batchDims; ++i) permuteOrder.push_back(i);


permuteOrder.push_back(batchDims + 2); 
permuteOrder.push_back(batchDims + 0);
permuteOrder.push_back(batchDims + 3);
permuteOrder.push_back(batchDims + 1);
permuteOrder.push_back(batchDims + 4);


    auto permuted = reshaped.permute(permuteOrder).contiguous();

    return permuted.reshape(originalShape);
}

std::pair<torch::Tensor, torch::Tensor> indexMaskPrefix(const torch::Tensor& arr2d){
    int64_t numCols =  arr2d.size(1);

    auto reversedArr = torch::flip(arr2d, {1});

    auto lastOneFromRight = reversedArr.to(torch::kInt32).argmax(1, false);

    auto maskLen = numCols - lastOneFromRight - 1;

    auto arange = torch::arange(numCols, arr2d.options().dtype(torch::kLong));
auto mask = arange.unsqueeze(0).le(maskLen.unsqueeze(1));

   
    auto result =  arr2d.masked_select(mask);

    auto maskLenUint8 = maskLen.to(torch::kUInt8);
    
    return {result, maskLenUint8};
}

torch::Tensor indexMaskReverse(const torch::Tensor& prefixMask,
                                 const torch::Tensor& maskLength,
                                 int64_t numCols) {
    auto device = prefixMask.device();
    auto arange = torch::arange(numCols, torch::dtype(torch::kLong).device(device));
    auto mask = arange.unsqueeze(0).le(maskLength.unsqueeze(1));
    auto arr2d = torch::zeros({maskLength.size(0), numCols},
                               torch::dtype(torch::kBool).device(device));
    
    arr2d.index_put_({mask}, prefixMask.to(torch::kBool).reshape({-1}));
    return arr2d;
}


std::vector<uint8_t> BitUtils::bitsToBytes(const torch::Tensor& bitArray) {

    torch::Tensor bits;
    if (bitArray.dtype() != torch::kUInt8) {
        bits = bitArray.to(torch::kUInt8);
    } else {
        bits = bitArray.clone();
    }
    
    // Ensure tensor is contiguous and on CPU may change to gpu later???????
    bits = bits.contiguous().cpu();
    
    auto data = bits.data_ptr<uint8_t>();
    int64_t numBits = bits.numel();
    
    int64_t numBytes = (numBits + 7) / 8;
    
    std::vector<uint8_t> packed(numBytes, 0);
    
    for (int64_t byteIdx = 0; byteIdx < numBytes; ++byteIdx) {
        uint8_t byte = 0;
        for (int bitIdx = 0; bitIdx < 8; ++bitIdx) {
            int64_t globalBitIdx = byteIdx * 8 + bitIdx;
            if (globalBitIdx < numBits) {
                // Pack bits in big-endian order 
                if (data[globalBitIdx]) {
                    byte |= (1 << (7 - bitIdx));
                }
            }
        }
        packed[byteIdx] = byte;
    }
    
    return packed;
}

torch::Tensor BitUtils::bytesToBits(const std::vector<uint8_t>& byteSeq, int64_t numBits) {
    int64_t totalBits = byteSeq.size() * 8;
    
    if (numBits == -1) {
        numBits = totalBits;
    }
    
    numBits = std::min(numBits, totalBits);
    
    torch::Tensor unpacked = torch::zeros({numBits}, torch::kBool);
    auto data = unpacked.data_ptr<bool>();
    
    for (int64_t bitIdx = 0; bitIdx < numBits; ++bitIdx) {
        int64_t byteIdx = bitIdx / 8;
        int64_t bitInByte = bitIdx % 8;
        
        // Unpack bits in big-endian order
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

void BitUtils::unpackByte(uint8_t byte, uint8_t* bits) {
    for (int i = 0; i < 8; ++i) {
        bits[i] = (byte >> (7 - i)) & 1;
    }
}


PCACompressor::PCACompressor(double nrmse, 
                           double quanFactor, 
                           const std::string& device,
                           const std::string& codecAlgorithm,
                           std::pair<int, int> patchSize)
    : quanBin_(nrmse * quanFactor),
      device_(device == "cuda" ? torch::kCUDA : torch::kCPU),
      codecAlgorithm_(codecAlgorithm),
      patchSize_(patchSize),
      vectorSize_(patchSize.first * patchSize.second),
      errorBound_(nrmse * std::sqrt(vectorSize_)),
      error_(nrmse) {
    
    std::cout << "PCACompressor initialized with:" << std::endl;
    std::cout << "  NRMSE: " << nrmse << std::endl;
    std::cout << "  Quantization factor: " << quanFactor << std::endl;
    std::cout << "  Device: " << device << std::endl;
    std::cout << "  Patch size: (" << patchSize.first << ", " << patchSize.second << ")" << std::endl;
    std::cout << "  Error bound: " << errorBound_ << std::endl;
}

PCACompressor::~PCACompressor() {
    cleanupGPUMemory();
}

CompressionResult PCACompressor::compress(const torch::Tensor& originalData, 
                                        const torch::Tensor& reconsData) {
    
    auto inputShape = originalData.sizes();
    
    // Move data to target device
    torch::Tensor originalDataDevice = originalData.to(device_, /*non_blocking=*/true);
    torch::Tensor reconsDataDevice = reconsData.to(device_, /*non_blocking=*/true);
    
    // Handle different input shapes
    if (inputShape.size() == 2) {
        // Already in vector format
        assert(originalDataDevice.size(1) == vectorSize_);
    } else {
        // Convert blocks to vectors
        originalDataDevice = block2Vector(originalDataDevice, patchSize_);
        reconsDataDevice = block2Vector(reconsDataDevice, patchSize_);
    }
    
    // Calculate residual
    torch::Tensor residualPca = originalDataDevice - reconsDataDevice;
    torch::Tensor norms = torch::linalg_norm(residualPca, /*dim=*/1);
    torch::Tensor processMask = norms > errorBound_;
    
    // Early exit if nothing to process
    if (torch::sum(processMask).item<int64_t>() <= 0) {
        MetaData metaData;
        metaData.dataBytes = 0;
        return {metaData, nullptr, 0};
    }
    
    // Extract residuals that need processing
    residualPca = residualPca.index({processMask});
    
    // Perform PCA
    PCA pca(-1, device_.str()); // -1 means use all components
    pca.fit(residualPca);
    torch::Tensor pcaBasis = pca.components();
    torch::Tensor allCoeff = torch::matmul(residualPca, pcaBasis.transpose(0, 1));
    
    // Check reconstruction error and switch to double precision if needed
    torch::Tensor reconstructedResidual = torch::matmul(allCoeff, pcaBasis);
    torch::Tensor reconError = torch::abs(reconstructedResidual - residualPca);
    double reconErrorMax = reconError.max().item<double>();
    
    if (reconErrorMax > error_) {
        std::cout << "[PCA] Switching to float64 due to high PCA reconstruction error: " 
                  << reconErrorMax << std::endl;
        
        residualPca = residualPca.to(torch::kDouble);
        pca.fit(residualPca);
        pcaBasis = pca.components();
        allCoeff = torch::matmul(residualPca, pcaBasis.transpose(0, 1));
    }
    
    // Clean up intermediate tensors
    originalDataDevice = torch::Tensor();
    reconsDataDevice = torch::Tensor();
    residualPca = torch::Tensor();
    cleanupGPUMemory();
    
    // Calculate coefficient powers and sort
    torch::Tensor allCoeffPower = allCoeff.pow(2);
    torch::Tensor sortIndex = torch::argsort(allCoeffPower, /*dim=*/1, /*descending=*/true);
    
    torch::Tensor allCoeffSorted = torch::gather(allCoeff, 1, sortIndex);
    torch::Tensor quanCoeffSorted = torch::round(allCoeffSorted / quanBin_) * quanBin_;
    torch::Tensor resCoeffSorted = allCoeffSorted - quanCoeffSorted;
    
    // Clean up
    allCoeffSorted = torch::Tensor();
    cleanupGPUMemory();
    
    // Calculate step errors
    torch::Tensor allCoeffPowerDesc = torch::gather(allCoeffPower, 1, sortIndex) - resCoeffSorted.pow(2);
    torch::Tensor stepErrors = torch::ones_like(allCoeffPowerDesc);
    torch::Tensor remainErrors = torch::sum(allCoeffPower, /*dim=*/1);
    
    for (int64_t i = 0; i < stepErrors.size(1); ++i) {
        remainErrors = remainErrors - allCoeffPowerDesc.select(1, i);
        stepErrors.select(1, i) = remainErrors;
    }
    
    // Clean up
    allCoeffPowerDesc = torch::Tensor();
    remainErrors = torch::Tensor();
    cleanupGPUMemory();
    
    // Create mask for significant coefficients
    torch::Tensor mask = stepErrors > (errorBound_ * errorBound_);
    stepErrors = torch::Tensor();
    cleanupGPUMemory();
    
    // Find first false index and set it to true
    torch::Tensor firstFalseIdx = torch::argmin(mask.to(torch::kInt), /*dim=*/1);
    auto batchIndices = torch::arange(mask.size(0), torch::TensorOptions().device(device_));
    mask.index_put_({batchIndices.unsqueeze(1), firstFalseIdx.unsqueeze(1)}, true);
    
    // Apply mask to quantized coefficients
    torch::Tensor selectedCoeffQ = quanCoeffSorted * mask;
    quanCoeffSorted = torch::Tensor();
    cleanupGPUMemory();
    
    // Unsort the selected coefficients
    torch::Tensor selectedCoeffUnsortQ = torch::zeros_like(selectedCoeffQ);
    auto idx = torch::arange(selectedCoeffQ.size(0), torch::TensorOptions().device(device_)).unsqueeze(1);
    selectedCoeffUnsortQ.scatter_(1, sortIndex, selectedCoeffQ);
    
    selectedCoeffQ = torch::Tensor();
    sortIndex = torch::Tensor();
    cleanupGPUMemory();
    
    // Create final mask and extract coefficients
    mask = selectedCoeffUnsortQ != 0;
    
    torch::Tensor coeffIntFlatten = torch::round(
        allCoeff.reshape({-1}).index({mask.reshape({-1})}) / quanBin_
    );
    
    auto uniqueResult = at::_unique(coeffIntFlatten, true, true);
    torch::Tensor uniqueVals = std::get<0>(uniqueResult);
    coeffIntFlatten = std::get<1>(uniqueResult);
    
    allCoeff = torch::Tensor();
    cleanupGPUMemory();
    
    // Create prefix mask
    auto prefixResult = indexMaskPrefix(mask);
    torch::Tensor prefixMaskFlatten = prefixResult.first;
    torch::Tensor maskLength = prefixResult.second;
    
    mask = torch::Tensor();
    cleanupGPUMemory();
    
    // Prepare metadata and main data
    MetaData metaData;
    metaData.pcaBasis = pcaBasis;
    metaData.uniqueVals = uniqueVals;
    metaData.quanBin = quanBin_;
    metaData.nVec = processMask.size(0);
    metaData.prefixLength = prefixMaskFlatten.size(0);
    
    MainData mainData;
    mainData.processMask = processMask;
    mainData.prefixMask = prefixMaskFlatten;
    mainData.maskLength = maskLength;
    mainData.coeffInt = coeffIntFlatten;
    
    // Compress the data
    auto compressResult = compressLossless(metaData, mainData);
    metaData.dataBytes = compressResult.second;
    
    return {metaData, std::move(compressResult.first), compressResult.second};
}

std::pair<std::unique_ptr<CompressedData>, int64_t> PCACompressor::compressLossless(
    const MetaData& metaData, 
    const MainData& mainData) {
    
    std::vector<uint8_t> allData;
    
    // Serialize all tensors and combine
    auto pcaBasisBytes = serializeTensor(metaData.pcaBasis);
    auto uniqueValsBytes = serializeTensor(metaData.uniqueVals);
    auto processMaskBytes = BitUtils::bitsToBytes(mainData.processMask.to(torch::kUInt8).cpu());
    auto prefixMaskBytes = BitUtils::bitsToBytes(mainData.prefixMask.to(torch::kUInt8).cpu());
    auto maskLengthBytes = serializeTensor(mainData.maskLength);
    auto coeffIntBytes = serializeTensor(mainData.coeffInt);
    
    // Combine all bytes (in a real implementation, you might want to add headers/sizes)
    allData.insert(allData.end(), pcaBasisBytes.begin(), pcaBasisBytes.end());
    allData.insert(allData.end(), uniqueValsBytes.begin(), uniqueValsBytes.end());
    allData.insert(allData.end(), processMaskBytes.begin(), processMaskBytes.end());
    allData.insert(allData.end(), prefixMaskBytes.begin(), prefixMaskBytes.end());
    allData.insert(allData.end(), maskLengthBytes.begin(), maskLengthBytes.end());
    allData.insert(allData.end(), coeffIntBytes.begin(), coeffIntBytes.end());
    
    // TODO: Apply actual compression algorithm (Zstd, etc.)
    // For now, just return the raw bytes
    
    auto compressedData = std::make_unique<CompressedData>();
    compressedData->data = std::move(allData);
    compressedData->dataBytes = compressedData->data.size();
    
    return {std::move(compressedData), compressedData->dataBytes};
}

torch::Tensor PCACompressor::toCPUContiguous(const torch::Tensor& tensor) {
    return tensor.cpu().contiguous();
}

std::vector<uint8_t> PCACompressor::serializeTensor(const torch::Tensor& tensor) {
    auto cpuTensor = toCPUContiguous(tensor);
    auto dataPtr = cpuTensor.data_ptr();
    auto numBytes = cpuTensor.numel() * cpuTensor.element_size();
    
    std::vector<uint8_t> bytes(numBytes);
    std::memcpy(bytes.data(), dataPtr, numBytes);
    
    return bytes;
}

void PCACompressor::cleanupGPUMemory() {
    if (device_.is_cuda()) {
       c10::cuda::CUDACachingAllocator::emptyCache();
    }
}

