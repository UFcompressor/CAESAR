#include "runGaeCuda.h"

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
    auto mask = arange.unsqueeze(0).le(maskLength.unsqueeze(1));
    auto arr2d = torch::zeros({ maskLength.size(0), numCols } ,
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
        bits = bitArray.clone();
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
    cleanupGPUMemory();
}

CompressionResult PCACompressor::compress(const torch::Tensor& originalData ,
    const torch::Tensor& reconsData) {

    auto inputShape = originalData.sizes();

    // FIX: Save the total vector count BEFORE any transformations
    int64_t totalVectors;
    if (inputShape.size() == 2) {
        totalVectors = originalData.size(0);
    }
    else {
     // For block data, calculate number of vectors after block2Vector
        int T = inputShape[inputShape.size() - 3];
        int H = inputShape[inputShape.size() - 2];
        int W = inputShape[inputShape.size() - 1];
        int nH = H / patchSize_.first;
        int nW = W / patchSize_.second;
        totalVectors = T * nH * nW;
        // Account for batch dimensions if any
        for (int i = 0; i < inputShape.size() - 3; ++i) {
            totalVectors *= inputShape[i];
        }
    }

    torch::Tensor originalDataDevice = originalData.to(device_ , true);
    torch::Tensor reconsDataDevice = reconsData.to(device_ , true);

    if (inputShape.size() == 2) {
        assert(originalDataDevice.size(1) == vectorSize_);
    }
    else {
        originalDataDevice = block2Vector(originalDataDevice , patchSize_);
        reconsDataDevice = block2Vector(reconsDataDevice , patchSize_);
    }

    torch::Tensor residualPca = originalDataDevice - reconsDataDevice;

    torch::Tensor norms = torch::linalg_norm(residualPca , c10::nullopt , { 1 });
    torch::Tensor processMask = norms > errorBound_;
    if (torch::sum(processMask).item<int64_t>() <= 0) {
        std::cout << "[DEBUG] No elements exceed errorBound_. Returning empty result." << std::endl;

        MetaData metaData;
        // FIX: pcaBasis should be 2D [0, vectorSize_], not 1D [0]
        metaData.pcaBasis = torch::empty({ 0, vectorSize_ } , torch::kFloat32);
        metaData.uniqueVals = torch::empty({ 0 } , torch::kFloat32);
        metaData.quanBin = quanBin_;
        metaData.nVec = originalData.size(0);
        metaData.prefixLength = 0;
        metaData.dataBytes = 0;
        std::cout << "[DEBUG] Total vectors (no PCA needed): " << metaData.nVec << std::endl;

        auto compressedData = std::make_unique<CompressedData>();
        compressedData->data.clear();  // Better than = {}
        compressedData->dataBytes = 0;
        compressedData->coeffIntBytes = 0;
        std::cout << "[DEBUG] Created empty CompressedData." << std::endl;

        std::cout << "[DEBUG] Returning result..." << std::endl;
        // FIX: Use aggregate initialization or return directly
        return { metaData, std::move(compressedData), 0 };
    }

    auto indices = torch::nonzero(processMask).squeeze(1);
    std::cout << "[DEBUG] Selected " << indices.size(0) << " indices for PCA processing." << std::endl;

    residualPca = torch::index_select(residualPca , 0 , indices);

    PCA pca(-1 , device_.str());
    pca.fit(residualPca);
    torch::Tensor pcaBasis = pca.components();
    std::cout << "[DEBUG] PCA basis shape: " << pcaBasis.sizes() << std::endl;

    if (pcaBasis.size(0) == 0 || pcaBasis.size(1) == 0) {
        std::cout << "PCA failed - empty basis" << std::endl;

        MetaData metaData;
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
    std::cout << "[DEBUG] allCoeff shape: " << allCoeff.sizes() << std::endl;

    torch::Tensor reconstructedResidual = torch::matmul(allCoeff , pcaBasis);
    torch::Tensor reconError = torch::abs(reconstructedResidual - residualPca);
    double reconErrorMax = reconError.max().item<double>();
    std::cout << "[DEBUG] PCA reconstruction error max: " << reconErrorMax << std::endl;

    if (reconErrorMax > error_) {
        std::cout << "[WARN] High PCA reconstruction error (" << reconErrorMax
            << ") > " << error_ << " — switching to float64" << std::endl;

        residualPca = residualPca.to(torch::kDouble);
        pca.fit(residualPca);
        pcaBasis = pca.components();
        allCoeff = torch::matmul(residualPca , pcaBasis.transpose(0 , 1));
    }

    std::cout << "[DEBUG] Cleaning up intermediate tensors" << std::endl;
    originalDataDevice = torch::Tensor();
    reconsDataDevice = torch::Tensor();
    residualPca = torch::Tensor();
    cleanupGPUMemory();

    torch::Tensor allCoeffPower = allCoeff.pow(2);
    torch::Tensor sortIndex = torch::argsort(allCoeffPower , 1 , true);
    std::cout << "[DEBUG] Sorting coefficients by power" << std::endl;

    torch::Tensor allCoeffSorted = torch::gather(allCoeff , 1 , sortIndex);
    torch::Tensor quanCoeffSorted = torch::round(allCoeffSorted / quanBin_) * quanBin_;
    torch::Tensor resCoeffSorted = allCoeffSorted - quanCoeffSorted;
    std::cout << "[DEBUG] Quantization bin size: " << quanBin_ << std::endl;

    allCoeffSorted = torch::Tensor();
    cleanupGPUMemory();

    torch::Tensor allCoeffPowerDesc = torch::gather(allCoeffPower , 1 , sortIndex) - resCoeffSorted.pow(2);
    torch::Tensor stepErrors = torch::ones_like(allCoeffPowerDesc);
    torch::Tensor remainErrors = torch::sum(allCoeffPower , 1);
    std::cout << "[DEBUG] Starting step error loop..." << std::endl;

    for (int64_t i = 0; i < stepErrors.size(1); ++i) {
        remainErrors = remainErrors - allCoeffPowerDesc.select(1 , i);
        stepErrors.select(1 , i) = remainErrors;
    }

    std::cout << "[DEBUG] Finished step error loop" << std::endl;

    allCoeffPowerDesc = torch::Tensor();
    remainErrors = torch::Tensor();
    cleanupGPUMemory();

    torch::Tensor mask = stepErrors > (errorBound_ * errorBound_);
    std::cout << "[DEBUG] Generated mask shape: " << mask.sizes() << std::endl;

    stepErrors = torch::Tensor();
    cleanupGPUMemory();

    torch::Tensor firstFalseIdx = torch::argmin(mask.to(torch::kInt) , 1);
    auto batchIndices = torch::arange(mask.size(0) , torch::TensorOptions().device(device_));
    mask.index_put_({ batchIndices.unsqueeze(1), firstFalseIdx.unsqueeze(1) } , true);
    std::cout << "[DEBUG] Applied firstFalseIdx mask modification" << std::endl;

    torch::Tensor selectedCoeffQ = quanCoeffSorted * mask;
    std::cout << "[DEBUG] selectedCoeffQ shape: " << selectedCoeffQ.sizes() << std::endl;

    quanCoeffSorted = torch::Tensor();
    cleanupGPUMemory();

    torch::Tensor selectedCoeffUnsortQ = torch::zeros_like(selectedCoeffQ);
    auto idx = torch::arange(selectedCoeffQ.size(0) , torch::TensorOptions().device(device_)).unsqueeze(1);
    selectedCoeffUnsortQ.scatter_(1 , sortIndex , selectedCoeffQ);
    std::cout << "[DEBUG] Restored original coefficient order" << std::endl;

    selectedCoeffQ = torch::Tensor();
    sortIndex = torch::Tensor();
    cleanupGPUMemory();

    mask = selectedCoeffUnsortQ != 0;
    std::cout << "[DEBUG] Nonzero mask count: " << torch::sum(mask).item<int64_t>() << std::endl;

    torch::Tensor coeffIntFlatten = torch::round(
        allCoeff.masked_select(mask) / quanBin_
    );
    std::cout << "[DEBUG] Flattened coefficient count: " << coeffIntFlatten.size(0) << std::endl;

    selectedCoeffUnsortQ = torch::Tensor();
    allCoeff = torch::Tensor();
    cleanupGPUMemory();

    auto uniqueResult = at::_unique(coeffIntFlatten , true , true);
    torch::Tensor uniqueVals = std::get<0>(uniqueResult);
    torch::Tensor inverseIndices = std::get<1>(uniqueResult);
    coeffIntFlatten = inverseIndices;
    std::cout << "[DEBUG] Unique values count: " << uniqueVals.size(0) << std::endl;

    cleanupGPUMemory();

    auto prefixResult = indexMaskPrefix(mask);
    torch::Tensor prefixMaskFlatten = prefixResult.first;
    torch::Tensor maskLength = prefixResult.second;
    std::cout << "[DEBUG] Prefix mask length: " << prefixMaskFlatten.size(0) << std::endl;

    mask = torch::Tensor();
    cleanupGPUMemory();

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

    std::cout << "[DEBUG] Compressing lossless portion..." << std::endl;
    auto compressResult = compressLossless(metaData , mainData);
    metaData.dataBytes = compressResult.second;
    std::cout << "[DEBUG] Compression complete. Compressed bytes: " << compressResult.second << std::endl;

    std::cout << "[DEBUG] Exiting PCACompressor::compress()" << std::endl;
    return { metaData, std::move(compressResult.first), compressResult.second };
}


torch::Tensor PCACompressor::decompress(const torch::Tensor& reconsData ,
    const MetaData& metaData ,
    const CompressedData& compressedData) {

    if (metaData.dataBytes == 0 || metaData.pcaBasis.numel() == 0) {
        return reconsData.clone();  // Return reconstruction as-is
    }
    auto inputShape = reconsData.sizes();
    torch::Tensor reconsDevice = reconsData.clone().to(device_);

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
        torch::TensorOptions().dtype(metaData.pcaBasis.dtype()).device(device_));

    coeff.masked_scatter_(indexMask , coeffInt * metaData.quanBin);

    torch::Tensor pcaReconstruction = torch::matmul(coeff , metaData.pcaBasis);

    torch::Tensor reconsSubset = reconsDevice.index({ mainData.processMask });
    reconsSubset = reconsSubset + pcaReconstruction;

    reconsDevice.index_put_({ mainData.processMask } , reconsSubset);

    if (needsReshape) {
        reconsDevice = vector2Block(reconsDevice , inputShape.vec() , patchSize_);
    }

    return reconsDevice.cpu();
}

std::pair<std::unique_ptr<CompressedData> , int64_t>
PCACompressor::compressLossless(const MetaData& metaData , const MainData& mainData) {
    auto compressedData = std::make_unique<CompressedData>();
    int64_t totalBytes = 0;


    auto processMaskBytes = BitUtils::bitsToBytes(mainData.processMask.to(torch::kUInt8).cpu());
    auto prefixMaskBytes = BitUtils::bitsToBytes(mainData.prefixMask.to(torch::kUInt8).cpu());
    auto maskLengthBytes = serializeTensor(mainData.maskLength);

    torch::Tensor coeffIntConverted;
    int64_t nUniqueVals = metaData.uniqueVals.size(0);
    if (nUniqueVals < 256)       coeffIntConverted = mainData.coeffInt.to(torch::kUInt8);
    else if (nUniqueVals < 32768) coeffIntConverted = mainData.coeffInt.to(torch::kInt16);
    else                         coeffIntConverted = mainData.coeffInt.to(torch::kInt32);
    auto coeffIntBytes = serializeTensor(coeffIntConverted);

    const int compressionLevel = 21;


    size_t processMaskBound = ZSTD_compressBound(processMaskBytes.size());
    std::vector<uint8_t> processMaskCompressed(processMaskBound);
    size_t processMaskCompSize = ZSTD_compress(
        processMaskCompressed.data() , processMaskCompressed.size() ,
        processMaskBytes.data() , processMaskBytes.size() ,
        compressionLevel
    );
    if (ZSTD_isError(processMaskCompSize)) {
        throw std::runtime_error("process_mask compression failed");
    }
    processMaskCompressed.resize(processMaskCompSize);


    size_t prefixMaskBound = ZSTD_compressBound(prefixMaskBytes.size());
    std::vector<uint8_t> prefixMaskCompressed(prefixMaskBound);
    size_t prefixMaskCompSize = ZSTD_compress(
        prefixMaskCompressed.data() , prefixMaskCompressed.size() ,
        prefixMaskBytes.data() , prefixMaskBytes.size() ,
        compressionLevel
    );
    if (ZSTD_isError(prefixMaskCompSize)) {
        throw std::runtime_error("prefix_mask compression failed");
    }
    prefixMaskCompressed.resize(prefixMaskCompSize);


    size_t maskLengthBound = ZSTD_compressBound(maskLengthBytes.size());
    std::vector<uint8_t> maskLengthCompressed(maskLengthBound);
    size_t maskLengthCompSize = ZSTD_compress(
        maskLengthCompressed.data() , maskLengthCompressed.size() ,
        maskLengthBytes.data() , maskLengthBytes.size() ,
        compressionLevel
    );
    if (ZSTD_isError(maskLengthCompSize)) {
        throw std::runtime_error("mask_length compression failed");
    }
    maskLengthCompressed.resize(maskLengthCompSize);


    size_t coeffIntBound = ZSTD_compressBound(coeffIntBytes.size());
    std::vector<uint8_t> coeffIntCompressed(coeffIntBound);
    size_t coeffIntCompSize = ZSTD_compress(
        coeffIntCompressed.data() , coeffIntCompressed.size() ,
        coeffIntBytes.data() , coeffIntBytes.size() ,
        compressionLevel
    );
    if (ZSTD_isError(coeffIntCompSize)) {
        throw std::runtime_error("coeff_int compression failed");
    }
    coeffIntCompressed.resize(coeffIntCompSize);


    std::vector<size_t> compressedSizes = {
        processMaskCompSize,
        prefixMaskCompSize,
        maskLengthCompSize,
        coeffIntCompSize
    };


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

    return { std::move(compressedData), totalBytes };
}

MainData PCACompressor::decompressLossless(const MetaData& metaData ,
    const CompressedData& compressedData) {
    MainData mainData;
    size_t offset = 0;


    std::vector<size_t> compressedSizes(4);
    for (int i = 0; i < 4; ++i) {
        size_t size = 0;
        for (int j = 0; j < 8; ++j) {
            size |= (size_t)compressedData.data[offset++] << (j * 8);
        }
        compressedSizes[i] = size;
    }

    size_t processMaskOrigSize = (metaData.nVec + 7) / 8;
    std::vector<uint8_t> processMaskVec(processMaskOrigSize);
    size_t processMaskDecompSize = ZSTD_decompress(
        processMaskVec.data() , processMaskVec.size() ,
        compressedData.data.data() + offset , compressedSizes[0]
    );
    if (ZSTD_isError(processMaskDecompSize)) {
        throw std::runtime_error("process_mask decompression failed");
    }
    mainData.processMask = BitUtils::bytesToBits(processMaskVec , metaData.nVec).to(device_);
    offset += compressedSizes[0];

    size_t prefixMaskOrigSize = (metaData.prefixLength + 7) / 8;
    std::vector<uint8_t> prefixMaskVec(prefixMaskOrigSize);
    size_t prefixMaskDecompSize = ZSTD_decompress(
        prefixMaskVec.data() , prefixMaskVec.size() ,
        compressedData.data.data() + offset , compressedSizes[1]
    );
    if (ZSTD_isError(prefixMaskDecompSize)) {
        throw std::runtime_error("prefix_mask decompression failed");
    }
    mainData.prefixMask = BitUtils::bytesToBits(prefixMaskVec , metaData.prefixLength).to(device_);
    offset += compressedSizes[1];


    int64_t numVecsProcessed = torch::sum(mainData.processMask).item<int64_t>();
    std::vector<uint8_t> maskLengthVec(numVecsProcessed);
    size_t maskLengthDecompSize = ZSTD_decompress(
        maskLengthVec.data() , maskLengthVec.size() ,
        compressedData.data.data() + offset , compressedSizes[2]
    );
    if (ZSTD_isError(maskLengthDecompSize)) {
        throw std::runtime_error("mask_length decompression failed");
    }
    mainData.maskLength = torch::empty({ numVecsProcessed } , torch::kUInt8);
    std::memcpy(mainData.maskLength.data_ptr<uint8_t>() , maskLengthVec.data() , numVecsProcessed);
    mainData.maskLength = mainData.maskLength.to(device_);
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
    size_t coeffIntDecompSize = ZSTD_decompress(
        coeffIntVec.data() , coeffIntVec.size() ,
        compressedData.data.data() + offset , compressedSizes[3]
    );
    if (ZSTD_isError(coeffIntDecompSize)) {
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
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
#else
#endif
}