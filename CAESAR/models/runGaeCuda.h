#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <map>
#include <filesystem>
#include <fstream>
#include <utility>
#include <cassert>
#include <cstdint>
#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include <algorithm>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <nvcomp/lz4.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/zstd.h>   
#endif
#include <zstd.h> 
class PCA {
public:
    PCA(int numComponents = -1 , const std::string& device = "cuda");
    PCA& fit(const torch::Tensor& x);
    torch::Tensor components() const { return components_; }
    torch::Tensor mean() const { return mean_; }
private:
    int numComponents_;
    torch::Device device_;
    torch::Tensor components_;
    torch::Tensor mean_;
};

torch::Tensor block2Vector(const torch::Tensor& blockdata ,
    std::pair<int , int> pathSize = { 8,8 });
torch::Tensor vector2Block(const torch::Tensor& vectors ,
    const std::vector<int64_t>& originalShape ,
    std::pair<int , int> patchSize);
std::pair<torch::Tensor , torch::Tensor> indexMaskPrefix(const torch::Tensor& arr2d);
torch::Tensor indexMaskReverse(const torch::Tensor& prefixMask ,
    const torch::Tensor& maskLength ,
    int64_t numCols);

class BitUtils {
public:
    static std::vector<uint8_t> bitsToBytes(const torch::Tensor& bitArray);
    static torch::Tensor bytesToBits(const std::vector<uint8_t>& byteSeq , int64_t numBits = -1);
private:
    static uint8_t packByte(const uint8_t* bits);
    static void unpackByte(uint8_t byte , uint8_t* bits);
};

struct CompressedData {
    std::vector<uint8_t> data;
    int64_t dataBytes;
    size_t coeffIntBytes;
};

struct MetaData {
    bool GAE_correction_occur;
    torch::Tensor pcaBasis;
    torch::Tensor uniqueVals;
    double quanBin;
    int64_t nVec;
    int64_t prefixLength;
    int64_t dataBytes;
};

struct MainData {
    torch::Tensor processMask;
    torch::Tensor prefixMask;
    torch::Tensor maskLength;
    torch::Tensor coeffInt;
};

struct GAECompressionResult {
    MetaData metaData;
    std::unique_ptr<CompressedData> compressedData;
    int64_t dataBytes;
};

class PCACompressor {
public:
    PCACompressor(double nrmse = -1 ,
        double quanFactor = -1 ,
        const std::string& device = "cuda" ,
        const std::string& codecAlgorithm = "Zstd" ,
        std::pair<int , int> patchSize = { 8, 8 });
    ~PCACompressor();

    GAECompressionResult compress(const torch::Tensor& originalData ,
        const torch::Tensor& reconsData);


    torch::Tensor decompress(const torch::Tensor& reconsData ,
        const MetaData& metaData ,
        const CompressedData& compressedData);
    std::pair<std::unique_ptr<CompressedData>, int64_t> compressLossless(
        const MetaData& metaData,
        const MainData& mainData);

    MainData decompressLossless(
        const MetaData& metaData,
        const CompressedData& compressedData);
private:
    double quanBin_;
    torch::Device device_;
    std::string codecAlgorithm_;
    std::pair<int , int> patchSize_;
    int vectorSize_;
    double errorBound_;
    double error_;

//     std::pair<std::unique_ptr<CompressedData> , int64_t> compressLossless(
//         const MetaData& metaData ,
//         const MainData& mainData);


//     MainData decompressLossless(const MetaData& metaData ,
//         const CompressedData& compressedData);

    torch::Tensor toCPUContiguous(const torch::Tensor& tensor);
    std::vector<uint8_t> serializeTensor(const torch::Tensor& tensor);


    torch::Tensor deserializeTensor(const std::vector<uint8_t>& bytes ,
        const std::vector<int64_t>& shape ,
        torch::ScalarType dtype);

    void cleanupGPUMemory();
};
