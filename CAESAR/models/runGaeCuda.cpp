#include "runGaeCuda.h"

PCA::PCA(int numComponents, const std::string &device)
    : numComponents_(numComponents), device_(torch::Device(device)) {}

PCA &PCA::fit(const torch::Tensor &x) {
  auto xDevice = x.to(device_);
  mean_ = torch::mean(xDevice, 0);
  auto xCentered = xDevice - mean_;

  auto C = torch::matmul(xCentered.transpose(0, 1), xCentered) /
           (xCentered.size(0) - 1);

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

torch::Tensor block2Vector(const torch::Tensor &blockData,
                           std::pair<int, int> patchSize) {
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
  return permuted.reshape({-1, finalDim});
}
torch::Tensor vector2Block(const torch::Tensor &vectors,
                           const std::vector<int64_t> &originalShape,
                           std::pair<int, int> patchSize) {
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

std::pair<torch::Tensor, torch::Tensor>
indexMaskPrefix(const torch::Tensor &arr2d) {
  int64_t numCols = arr2d.size(1);

  auto reversedArr = torch::flip(arr2d, {1});
  auto lastOneFromRight = reversedArr.to(torch::kInt32).argmax(1, false);
  auto maskLen = numCols - lastOneFromRight - 1;

  auto arange = torch::arange(numCols, arr2d.options().dtype(torch::kLong));
  auto mask = arange.unsqueeze(0).le(maskLen.unsqueeze(1));

  auto result = arr2d.masked_select(mask);
  auto maskLenUint8 = maskLen.to(torch::kUInt8);

  return {result, maskLenUint8};
}

torch::Tensor indexMaskReverse(const torch::Tensor &prefixMask,
                               const torch::Tensor &maskLength,
                               int64_t numCols) {
  auto device = prefixMask.device();
  auto arange =
      torch::arange(numCols, torch::dtype(torch::kLong).device(device));
  auto maskLength_d = maskLength.to(prefixMask.device());
  auto mask = arange.unsqueeze(0).le(maskLength_d.unsqueeze(1));

  auto arr2d = torch::zeros({maskLength_d.size(0), numCols},
                            torch::dtype(torch::kBool).device(device));

  arr2d.index_put_({mask}, prefixMask.to(torch::kBool).reshape({-1}));
  return arr2d;
}

torch::Tensor BitUtils::bitsToBytes(const torch::Tensor &bitArray) {
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

torch::Tensor BitUtils::bytesToBits(const torch::Tensor &byteSeq,
                                    int64_t numBits) {
  torch::Tensor bytes = byteSeq.flatten().to(torch::kUInt8);
  int64_t numBytes = bytes.numel();
  int64_t totalBits = numBytes * 8;

  if (numBits == -1)
    numBits = totalBits;
  numBits = std::min(numBits, totalBits);

  // Broadcast each byte against MSB-first weights to extract individual bits.
  torch::Tensor weights = torch::tensor(
      {128, 64, 32, 16, 8, 4, 2, 1},
      torch::TensorOptions().dtype(torch::kUInt8).device(bytes.device()));

  torch::Tensor bits =
      bytes.unsqueeze(1).bitwise_and(weights).ne(0).reshape({-1});

  return bits.narrow(0, 0, numBits);
}

uint8_t BitUtils::packByte(const uint8_t *bits) {
  uint8_t byte = 0;
  for (int i = 0; i < 8; ++i) {
    if (bits[i]) {
      byte |= (1 << (7 - i));
    }
  }
  return byte;
}

void BitUtils::unpackByte(uint8_t byte, uint8_t *bits) {
  for (int i = 0; i < 8; ++i) {
    bits[i] = (byte >> (7 - i)) & 1;
  }
}

PCACompressor::PCACompressor(double nrmse, double quanFactor,
                             const std::string &device,
                             const std::string &codecAlgorithm,
                             std::pair<int, int> patchSize)
    : quanBin_(nrmse * quanFactor),
      device_(device.rfind("cuda", 0) == 0  ? torch::kCUDA
              : device.rfind("mps", 0) == 0 ? torch::kMPS
              : device.rfind("xpu", 0) == 0 ? torch::kXPU
                                            : torch::kCPU),
      codecAlgorithm_(codecAlgorithm), patchSize_(patchSize),
      vectorSize_(patchSize.first * patchSize.second),
      errorBound_(nrmse * std::sqrt(vectorSize_)), error_(nrmse) {}

PCACompressor::~PCACompressor() {
#ifdef USE_CUDA
  cleanupGPUMemory();
#endif
}

GAECompressionResult PCACompressor::compress(torch::Tensor originalData,
                                             torch::Tensor reconsData) {
  auto inputShape = originalData.sizes();

  int64_t totalVectors;
  if (inputShape.size() == 2) {
    totalVectors = originalData.size(0);
  } else {
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
                                         : originalData.to(device_, true);

  torch::Tensor reconsDataDevice = reconsData.device() == device_
                                       ? reconsData
                                       : reconsData.to(device_, true);

  if (inputShape.size() == 2) {
    assert(originalDataDevice.size(1) == vectorSize_);
  } else {
    originalDataDevice = block2Vector(originalDataDevice, patchSize_);
    reconsDataDevice = block2Vector(reconsDataDevice, patchSize_);
  }

  torch::Tensor residualPca = originalDataDevice - reconsDataDevice;

  originalDataDevice = torch::Tensor();
  reconsDataDevice = torch::Tensor();
  originalData = torch::Tensor();
  reconsData = torch::Tensor();
#ifdef USE_CUDA
  cleanupGPUMemory();
#endif

  torch::Tensor norms = torch::linalg_norm(residualPca, c10::nullopt, {1});

  MainData mainData;
  mainData.processMask = norms > errorBound_;
  norms = torch::Tensor();

  if (torch::sum(mainData.processMask).item<int64_t>() <= 0) {
    MetaData metaData;
    metaData.GAE_correction_occur = false;
    metaData.pcaBasis = torch::empty({0, vectorSize_}, torch::kFloat32);
    metaData.uniqueVals = torch::empty({0}, torch::kFloat32);
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

  auto indices = torch::nonzero(mainData.processMask).squeeze(1);
  residualPca = torch::index_select(residualPca, 0, indices);
  indices = torch::Tensor();

  if (residualPca.size(0) < 2) {
    MetaData metaData;
    metaData.GAE_correction_occur = false;
    metaData.pcaBasis = torch::empty({0, vectorSize_}, torch::kFloat32);
    metaData.uniqueVals = torch::empty({0}, torch::kFloat32);
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

  PCA pca(-1, device_.str());
  pca.fit(residualPca);
  torch::Tensor pcaBasis = pca.components();

  if (pcaBasis.size(0) == 0 || pcaBasis.size(1) == 0) {
    MetaData metaData;
    metaData.GAE_correction_occur = false;
    metaData.pcaBasis = torch::empty({0, vectorSize_}, torch::kFloat32);
    metaData.uniqueVals = torch::empty({0}, torch::kFloat32);
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

  torch::Tensor allCoeff = torch::matmul(residualPca, pcaBasis.transpose(0, 1));
  torch::Tensor reconstructedResidual = torch::matmul(allCoeff, pcaBasis);
  torch::Tensor reconError = torch::abs(reconstructedResidual - residualPca);
  double reconErrorMax = reconError.max().item<double>();
  reconstructedResidual = torch::Tensor();
  reconError = torch::Tensor();
  allCoeff = torch::Tensor();

  if (reconErrorMax > error_ ||
      error_ <
          10.0 * static_cast<double>(std::numeric_limits<float>::epsilon())) {
#ifdef USE_CUDA
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    double leftGiB = gpu_used_gb();
    if ((prop.totalGlobalMem / (1024.0 * 1024 * 1024)) - 2.0 < leftGiB) {
      std::cerr
          << "[WARN] GAE near memory limit: " << leftGiB << " GiB used on GPU "
          << " GiB left on GPU "
          << (prop.totalGlobalMem / (1024.0 * 1024 * 1024)) - leftGiB
          << " GiB.\n"
          << "[WARN] Consider: larger error bound, smaller dataset chunks,\n"
          << "[WARN] or multiple GPUs. Attempting anyway...\n";
    }
#endif
    residualPca = residualPca.to(torch::kDouble);
    pca.fit(residualPca);
    pcaBasis = pca.components();
  }

  allCoeff = torch::matmul(residualPca, pcaBasis.transpose(0, 1));
  residualPca = torch::Tensor();
#ifdef USE_CUDA
  cleanupGPUMemory();
#endif

  torch::Tensor allCoeffPower = allCoeff.pow(2);
  torch::Tensor sortIndex =
      torch::argsort(allCoeffPower, 1, true).to(torch::kInt32);

  torch::Tensor allCoeffSorted =
      torch::gather(allCoeff, 1, sortIndex.to(torch::kLong));
  torch::Tensor quanCoeffSorted = torch::round(allCoeffSorted / quanBin_);
  {
    torch::Tensor diff = allCoeffSorted - quanCoeffSorted * quanBin_;
    allCoeffSorted = diff.pow(2);
  }
  torch::Tensor allCoeffPowerDesc =
      torch::gather(allCoeffPower, 1, sortIndex.to(torch::kLong));
  allCoeffPowerDesc.sub_(allCoeffSorted);
  allCoeffSorted = torch::Tensor();

  torch::Tensor totalPower = torch::sum(allCoeffPower, 1).unsqueeze(1);
  allCoeffPower = torch::Tensor();
#ifdef USE_CUDA
  cleanupGPUMemory();
#endif

  torch::Tensor stepErrors = totalPower - torch::cumsum(allCoeffPowerDesc, 1);
  allCoeffPowerDesc = torch::Tensor();
  totalPower = torch::Tensor();

  torch::Tensor mask = stepErrors > (errorBound_ * errorBound_);
  stepErrors = torch::Tensor();

  torch::Tensor firstFalseIdx = torch::argmin(mask.to(torch::kInt), 1);
  auto batchIndices =
      torch::arange(mask.size(0), torch::TensorOptions().device(device_));
  mask.index_put_({batchIndices.unsqueeze(1), firstFalseIdx.unsqueeze(1)},
                  true);
  firstFalseIdx = torch::Tensor();
  batchIndices = torch::Tensor();

  torch::Tensor selectedCoeffQBool = (quanCoeffSorted != 0) & mask;
  mask = torch::Tensor();

  quanCoeffSorted = torch::Tensor();

  torch::Tensor finalMask =
      torch::zeros({selectedCoeffQBool.size(0), selectedCoeffQBool.size(1)},
                   torch::TensorOptions().dtype(torch::kBool).device(device_));
  finalMask.scatter_(1, sortIndex.to(torch::kLong), selectedCoeffQBool);
  selectedCoeffQBool = torch::Tensor();
  sortIndex = torch::Tensor();
#ifdef USE_CUDA
  cleanupGPUMemory();
#endif

  torch::Tensor coeffIntFlatten =
      torch::round(allCoeff.masked_select(finalMask) / quanBin_);
  allCoeff = torch::Tensor();
#ifdef USE_CUDA
  cleanupGPUMemory();
#endif

  torch::Tensor uniqueVals, inverseIndices;
  int64_t chunk_size = 1LL << 30;
  int64_t numel = coeffIntFlatten.numel();

  if (numel <= chunk_size) {
    auto unique_result = at::_unique(coeffIntFlatten, true, true);
    uniqueVals = std::get<0>(unique_result);
    inverseIndices = std::get<1>(unique_result);
  } else {
    std::vector<at::Tensor> inverse_parts;
    std::vector<at::Tensor> unique_parts;
    int64_t offset = 0;

    for (int64_t start = 0; start < numel; start += chunk_size) {
      int64_t current_chunk_size = std::min(chunk_size, numel - start);
      auto chunk = coeffIntFlatten.narrow(0, start, current_chunk_size);
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

    auto final_unique = at::_unique(all_uniques, true, true);
    uniqueVals = std::get<0>(final_unique);
    torch::Tensor remap = std::get<1>(final_unique);
    all_uniques = torch::Tensor();

    inverseIndices = remap.index_select(0, all_inverses);
    remap = torch::Tensor();
    all_inverses = torch::Tensor();
  }

  coeffIntFlatten = torch::Tensor();
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

  pcaBasis = torch::Tensor();
  uniqueVals = torch::Tensor();

  auto compressResult = compressLossless(metaData, mainData);
  metaData.dataBytes = compressResult.second;

  return {metaData, std::move(compressResult.first), compressResult.second};
}

torch::Tensor PCACompressor::decompress(const torch::Tensor &reconsData,
                                        const MetaData &metaData,
                                        const CompressedData &compressedData) {
  if (metaData.dataBytes == 0 || metaData.pcaBasis.numel() == 0) {
    return reconsData;
  }

  auto inputShape = reconsData.sizes();

  torch::Tensor reconsDevice = reconsData.to(device_);

  bool needsReshape = (inputShape.size() != 2);
  if (needsReshape) {
    reconsDevice = block2Vector(reconsDevice, patchSize_);
  }

  MainData mainData = decompressLossless(metaData, compressedData);

  torch::Tensor indexMask = indexMaskReverse(
      mainData.prefixMask, mainData.maskLength, metaData.pcaBasis.size(0));
  indexMask = indexMask.to(device_);

  torch::Tensor indexIndices = mainData.coeffInt.to(torch::kLong);
  if (indexIndices.device() != device_) {
    indexIndices = indexIndices.to(device_);
  }

  torch::Tensor coeffInt = metaData.uniqueVals;
  if (coeffInt.device() != device_) {
    coeffInt = coeffInt.to(device_);
  }
  coeffInt = coeffInt.index({indexIndices}).to(torch::kFloat32).to(device_);

  torch::Tensor coeff = torch::zeros(
      indexMask.sizes(),
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  torch::Tensor scatterValues = coeffInt * metaData.quanBin;
  if (scatterValues.device() != device_) {
    scatterValues = scatterValues.to(device_);
  }

  coeff.masked_scatter_(indexMask, scatterValues);
  coeffInt = torch::Tensor();
  indexMask = torch::Tensor();

  torch::Tensor pcaBasisDevice = metaData.pcaBasis.to(torch::kFloat32);
  if (pcaBasisDevice.device() != device_) {
    pcaBasisDevice = pcaBasisDevice.to(device_);
  }
  torch::Tensor pcaReconstruction = torch::matmul(coeff, pcaBasisDevice);
  coeff = torch::Tensor();

  reconsDevice.index_put_({mainData.processMask},
                          reconsDevice.index({mainData.processMask}) +
                              pcaReconstruction);
  pcaReconstruction = torch::Tensor();

  int64_t n_processed = torch::sum(mainData.processMask).item<int64_t>();
  int64_t n_total = mainData.processMask.size(0);

  if (needsReshape) {
    reconsDevice = vector2Block(reconsDevice, inputShape.vec(), patchSize_);
  }

  return reconsDevice;
}

std::pair<std::unique_ptr<CompressedData>, int64_t>
PCACompressor::compressLossless(const MetaData &metaData,
                                const MainData &mainData) {
  auto compressedData = std::make_unique<CompressedData>();
  int64_t totalBytes = 0;

  torch::Tensor processMaskBytes =
      BitUtils::bitsToBytes(mainData.processMask.to(torch::kUInt8));
  torch::Tensor prefixMaskBytes =
      BitUtils::bitsToBytes(mainData.prefixMask.to(torch::kUInt8));
  torch::Tensor maskLengthBytes =
      mainData.maskLength.contiguous().view(torch::kUInt8);

  torch::Tensor coeffIntConverted;
  int64_t nUniqueVals = metaData.uniqueVals.size(0);
  if (nUniqueVals < 256)
    coeffIntConverted = mainData.coeffInt.to(torch::kUInt8);
  else if (nUniqueVals < 32768)
    coeffIntConverted = mainData.coeffInt.to(torch::kInt16);
  else
    coeffIntConverted = mainData.coeffInt.to(torch::kInt32);
  torch::Tensor coeffIntBytes =
      coeffIntConverted.contiguous().view(torch::kUInt8);
  const int compressionLevel = 3;

  size_t raw_process_mask_bytes = (size_t)processMaskBytes.numel();
  size_t raw_prefix_mask_bytes = (size_t)prefixMaskBytes.numel();
  size_t raw_mask_length_bytes = (size_t)maskLengthBytes.numel();
  size_t raw_coeff_int_bytes = (size_t)coeffIntBytes.numel();

  bool use_nvcomp = false;
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
  use_nvcomp = device_.is_cuda();
#endif

  // Change to tensors
  torch::Tensor processMaskCompressed, prefixMaskCompressed,
      maskLengthCompressed, coeffIntCompressed;
  std::vector<size_t> compressedSizes;

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
  if (use_nvcomp) {
    // processMask/prefixMask are already on GPU (from bitsToBytes).
    std::vector<torch::Tensor> inputs = {
        processMaskBytes.contiguous(), prefixMaskBytes.contiguous(),
        maskLengthBytes.contiguous(), coeffIntBytes.contiguous()};

    auto batchResults = nvcomp_batch_compress(inputs);

    // Release input tensors now that compression is done.
    processMaskBytes = torch::Tensor();
    prefixMaskBytes = torch::Tensor();
    maskLengthBytes = torch::Tensor();
    coeffIntBytes = torch::Tensor();

    processMaskCompressed = std::move(batchResults[0].compressed);
    prefixMaskCompressed = std::move(batchResults[1].compressed);
    maskLengthCompressed = std::move(batchResults[2].compressed);
    coeffIntCompressed = std::move(batchResults[3].compressed);

    compressedSizes = {(size_t)processMaskCompressed.numel(),
                       (size_t)prefixMaskCompressed.numel(),
                       (size_t)maskLengthCompressed.numel(),
                       (size_t)coeffIntCompressed.numel()};
  }
#endif

  if (!use_nvcomp) {
    auto zstd_compress_mt = [&](const torch::Tensor &in_tensor,
                                std::vector<uint8_t> &out, int level,
                                int workers) -> size_t {
      if (in_tensor.numel() == 0) {
        out.clear();
        return 0;
      }

      const uint8_t *data = in_tensor.data_ptr<uint8_t>();
      size_t data_size = (size_t)in_tensor.numel();

// sanity: require zstd >= 1.4.0 for ZSTD_c_nbWorkers
#if !defined(ZSTD_VERSION_NUMBER) || (ZSTD_VERSION_NUMBER < 10400)
      throw std::runtime_error(
          "zstd too old: need >= 1.4.0 for multithread (ZSTD_c_nbWorkers)");
#endif

      ZSTD_CCtx *cctx = ZSTD_createCCtx();
      if (!cctx)
        throw std::runtime_error("ZSTD_createCCtx failed");

      // enable multithread
      size_t s1 = ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers);
      if (ZSTD_isError(s1)) {
        ZSTD_freeCCtx(cctx);
        throw std::runtime_error(std::string("ZSTD_c_nbWorkers set failed: ") +
                                 ZSTD_getErrorName(s1));
      }

      // set compression level
      size_t s2 = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
      if (ZSTD_isError(s2)) {
        ZSTD_freeCCtx(cctx);
        throw std::runtime_error(
            std::string("ZSTD_c_compressionLevel set failed: ") +
            ZSTD_getErrorName(s2));
      }

      // allocate output
      size_t bound = ZSTD_compressBound(data_size);
      out.resize(bound);

      // compress
      size_t compSize =
          ZSTD_compress2(cctx, out.data(), out.size(), data, data_size);

      ZSTD_freeCCtx(cctx);

      if (ZSTD_isError(compSize)) {
        throw std::runtime_error(std::string("zstd compress2 failed: ") +
                                 ZSTD_getErrorName(compSize));
      }

      out.resize(compSize);
      return compSize;
    };

    torch::Tensor pmbCpu = processMaskBytes.cpu().contiguous();
    torch::Tensor pfmbCpu = prefixMaskBytes.cpu().contiguous();
    torch::Tensor mlbCpu = maskLengthBytes.cpu().contiguous();
    torch::Tensor cibCpu = coeffIntBytes.cpu().contiguous();

    processMaskBytes = torch::Tensor();
    prefixMaskBytes = torch::Tensor();
    maskLengthBytes = torch::Tensor();
    coeffIntBytes = torch::Tensor();

    const int workers = get_allocated_cores();

    std::vector<uint8_t> pmc, pfmc, mlc, cic;
    size_t processMaskCompSize =
        zstd_compress_mt(pmbCpu, pmc, compressionLevel, workers);
    size_t prefixMaskCompSize =
        zstd_compress_mt(pfmbCpu, pfmc, compressionLevel, workers);
    size_t maskLengthCompSize =
        zstd_compress_mt(mlbCpu, mlc, compressionLevel, workers);
    size_t coeffIntCompSize =
        zstd_compress_mt(cibCpu, cic, compressionLevel, workers);

    maskLengthBytes = torch::Tensor();
    coeffIntBytes = torch::Tensor();

    // Wrap compressed vectors into tensors for uniform assembly below.
    processMaskCompressed = torch::tensor(pmc, torch::kUInt8);
    prefixMaskCompressed = torch::tensor(pfmc, torch::kUInt8);
    maskLengthCompressed = torch::tensor(mlc, torch::kUInt8);
    coeffIntCompressed = torch::tensor(cic, torch::kUInt8);

    compressedSizes = {processMaskCompSize, prefixMaskCompSize,
                       maskLengthCompSize, coeffIntCompSize};
  }

  size_t comp_process_mask_bytes = (size_t)processMaskCompressed.numel();
  size_t comp_prefix_mask_bytes = (size_t)prefixMaskCompressed.numel();
  size_t comp_mask_length_bytes = (size_t)maskLengthCompressed.numel();
  size_t comp_coeff_int_bytes = (size_t)coeffIntCompressed.numel();

  auto CR = [](size_t rawb, size_t compb) -> double {
    return compb ? (double)rawb / (double)compb : 0.0;
  };

  const size_t totalCompressedBytes =
      comp_process_mask_bytes + comp_prefix_mask_bytes +
      comp_mask_length_bytes + comp_coeff_int_bytes;

  compressedData->data.clear();
  compressedData->data.reserve(4 * sizeof(size_t) + totalCompressedBytes);

  for (size_t sz : compressedSizes) {
    for (int i = 0; i < 8; ++i)
      compressedData->data.push_back((sz >> (i * 8)) & 0xFF);
  }

  auto append_tensor = [&](const torch::Tensor &t) {
    const uint8_t *p = t.data_ptr<uint8_t>();
    compressedData->data.insert(compressedData->data.end(), p, p + t.numel());
  };
  append_tensor(processMaskCompressed);
  append_tensor(prefixMaskCompressed);
  append_tensor(maskLengthCompressed);
  append_tensor(coeffIntCompressed);

  compressedData->coeffIntBytes = raw_coeff_int_bytes;
  totalBytes = compressedData->data.size();
  compressedData->dataBytes = totalBytes;
  return {std::move(compressedData), totalBytes};
}

MainData
PCACompressor::decompressLossless(const MetaData &metaData,
                                  const CompressedData &compressedData) {
  MainData mainData;
  size_t offset = 0;

#define CHECK_ZSTD_BLOCK(name, off, csz)                                       \
  do {                                                                         \
    if ((off) + (csz) > compressedData.data.size())                            \
      throw std::runtime_error(std::string("OOB block: ") + (name));           \
  } while (0)

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
  if (use_nvcomp) {
    size_t processMaskOrigSize = (metaData.nVec + 7) / 8;
    size_t prefixMaskOrigSize = (metaData.prefixLength + 7) / 8;
    size_t coeffIntOrigSize = compressedData.coeffIntBytes;

    // ── Step 1: decompress processMask alone (we need it to compute
    // maskLength's size) ──
    {
      std::vector<const uint8_t *> ptrs = {compressedData.data.data() + offset};
      std::vector<size_t> comp_sizes = {compressedSizes[0]};
      std::vector<size_t> decomp_sizes = {processMaskOrigSize};

      auto res = nvcomp_batch_decompress(ptrs, comp_sizes, decomp_sizes);
      mainData.processMask =
          BitUtils::bytesToBits(torch::from_blob(res[0].data(),
                                                 {(int64_t)res[0].size()},
                                                 torch::kUInt8)
                                    .clone(),
                                metaData.nVec)
              .to(device_);
    }
    offset += compressedSizes[0];

    int64_t numVecsProcessed = torch::sum(mainData.processMask).item<int64_t>();

    // ── Step 2: batch decompress the remaining 3 (prefixMask, maskLength,
    // coeffInt) ──
    {
      std::vector<const uint8_t *> ptrs = {
          compressedData.data.data() + offset,
          compressedData.data.data() + offset + compressedSizes[1],
          compressedData.data.data() + offset + compressedSizes[1] +
              compressedSizes[2]};
      std::vector<size_t> comp_sizes = {compressedSizes[1], compressedSizes[2],
                                        compressedSizes[3]};
      std::vector<size_t> decomp_sizes = {
          prefixMaskOrigSize, (size_t)numVecsProcessed, coeffIntOrigSize};

      auto res = nvcomp_batch_decompress(ptrs, comp_sizes, decomp_sizes);

      // prefixMask
      mainData.prefixMask =
          BitUtils::bytesToBits(torch::from_blob(res[0].data(),
                                                 {(int64_t)res[0].size()},
                                                 torch::kUInt8)
                                    .clone(),
                                metaData.prefixLength)
              .to(device_);

      // maskLength
      mainData.maskLength =
          torch::from_blob(res[1].data(), {numVecsProcessed}, torch::kUInt8)
              .clone()
              .to(device_);

      // coeffInt
      int64_t nUniqueVals = metaData.uniqueVals.size(0);
      torch::ScalarType coeffDtype;
      size_t elementSize;
      if (nUniqueVals < 256) {
        coeffDtype = torch::kUInt8;
        elementSize = 1;
      } else if (nUniqueVals < 32768) {
        coeffDtype = torch::kInt16;
        elementSize = 2;
      } else {
        coeffDtype = torch::kInt32;
        elementSize = 4;
      }

      int64_t numElements = (int64_t)res[2].size() / (int64_t)elementSize;
      mainData.coeffInt = torch::empty({numElements}, coeffDtype);
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
    size_t sz = ZSTD_decompress(processMaskVec.data(), processMaskVec.size(),
                                compressedData.data.data() + offset,
                                compressedSizes[0]);
    if (ZSTD_isError(sz))
      throw std::runtime_error("process_mask decompression failed");
  }
  mainData.processMask =
      BitUtils::bytesToBits(torch::from_blob(processMaskVec.data(),
                                             {(int64_t)processMaskVec.size()},
                                             torch::kUInt8)
                                .clone(),
                            metaData.nVec)
          .to(device_);
  offset += compressedSizes[0];

  // prefixMask
  size_t prefixMaskOrigSize = (metaData.prefixLength + 7) / 8;
  std::vector<uint8_t> prefixMaskVec(prefixMaskOrigSize);
  {
    CHECK_ZSTD_BLOCK("prefix_mask", offset, compressedSizes[1]);
    size_t sz = ZSTD_decompress(prefixMaskVec.data(), prefixMaskVec.size(),
                                compressedData.data.data() + offset,
                                compressedSizes[1]);
    if (ZSTD_isError(sz))
      throw std::runtime_error("prefix_mask decompression failed");
  }
  mainData.prefixMask =
      BitUtils::bytesToBits(torch::from_blob(prefixMaskVec.data(),
                                             {(int64_t)prefixMaskVec.size()},
                                             torch::kUInt8)
                                .clone(),
                            metaData.prefixLength)
          .to(device_);
  offset += compressedSizes[1];

  // maskLength
  int64_t numVecsProcessed = torch::sum(mainData.processMask).item<int64_t>();
  std::vector<uint8_t> maskLengthVec(numVecsProcessed);
  {
    CHECK_ZSTD_BLOCK("mask_length", offset, compressedSizes[2]);
    size_t sz = ZSTD_decompress(maskLengthVec.data(), maskLengthVec.size(),
                                compressedData.data.data() + offset,
                                compressedSizes[2]);
    if (ZSTD_isError(sz))
      throw std::runtime_error("mask_length decompression failed");
  }
  mainData.maskLength =
      torch::from_blob(maskLengthVec.data(), {numVecsProcessed}, torch::kUInt8)
          .clone()
          .to(device_);
  offset += compressedSizes[2];

  // coeffInt
  int64_t nUniqueVals = metaData.uniqueVals.size(0);
  torch::ScalarType coeffDtype;
  size_t elementSize;
  if (nUniqueVals < 256) {
    coeffDtype = torch::kUInt8;
    elementSize = sizeof(uint8_t);
  } else if (nUniqueVals < 32768) {
    coeffDtype = torch::kInt16;
    elementSize = sizeof(int16_t);
  } else {
    coeffDtype = torch::kInt32;
    elementSize = sizeof(int32_t);
  }

  size_t coeffIntOrigSize = compressedData.coeffIntBytes;
  std::vector<uint8_t> coeffIntVec(coeffIntOrigSize);
  {
    CHECK_ZSTD_BLOCK("coeff_int", offset, compressedSizes[3]);
    auto p = compressedData.data.data() + offset;
    auto fsz = ZSTD_getFrameContentSize(p, compressedSizes[3]);

    size_t sz = ZSTD_decompress(coeffIntVec.data(), coeffIntVec.size(),
                                compressedData.data.data() + offset,
                                compressedSizes[3]);
    if (ZSTD_isError(sz))
      throw std::runtime_error("coeff_int decompression failed");
  }

  int64_t numElements = (int64_t)coeffIntVec.size() / (int64_t)elementSize;
  mainData.coeffInt = torch::empty({numElements}, coeffDtype);
  std::memcpy(mainData.coeffInt.data_ptr(), coeffIntVec.data(),
              coeffIntVec.size());
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
