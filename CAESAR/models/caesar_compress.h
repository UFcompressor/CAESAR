#pragma once

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <vector>
#include <string>
#include "model_cache.h"
#include "array_utils.h"
#include "../dataset/dataset.h"

struct CompressionMetaData {
    std::vector<float> offsets; // local info - corresponding to latent
    std::vector<float> scales; // local info - corresponding to latent
    std::vector<std::vector<int32_t>> indexes; // local info - corresponding to latent
    std::tuple<int32_t , int32_t , std::vector<int32_t>> block_info; // global info
    std::vector<int32_t> data_input_shape; // global info
    std::vector<std::pair<int32_t , float>> filtered_blocks; // global info
    float global_scale; // global info
    float global_offset; // global info
    int64_t pad_T; // global_info
};

struct GaeBatchRecord {
    bool correction_occur = false;

    double quanBin = 0.0;
    int64_t nVec = 0;
    int64_t prefixLength = 0;

    int64_t dataBytes = 0;
    size_t coeffIntBytes = 0;

    std::vector<uint8_t> comp_data;
    std::vector<float> uniqueVals;

    int32_t batch_n = 0;
};

struct GAEMetaData {
    std::vector<std::vector<float>> global_pcaBasis;
};

struct CompressionResult {
    std::vector<std::string> encoded_latents;
    std::vector<std::string> encoded_hyper_latents;

    CompressionMetaData compressionMetaData;

    GAEMetaData gaeMetaData;
    std::vector<GaeBatchRecord> gae_batches;

    int num_samples = 0;
    int num_batches = 0;
};


// **** //

class Compressor {
public:
    explicit Compressor(torch::Device device = torch::Device(torch::kCPU));
    ~Compressor() = default;

    CompressionResult compress(const DatasetConfig& config , int batch_size = 32 , float rel_eb = 0.1);
private:
    torch::Device device_;
    
 
    torch::inductor::AOTIModelPackageLoader* compressor_model_;
    torch::inductor::AOTIModelPackageLoader* hyper_decompressor_model_;
    torch::inductor::AOTIModelPackageLoader* decompressor_model_;
    
    torch::Tensor reshape_batch_2d_3d(const torch::Tensor& batch_data, int64_t batch_size);
    torch::Tensor deblockHW(const torch::Tensor& data, int64_t nH, int64_t nW, const std::vector<int64_t>& padding);
    torch::Tensor recons_data(const torch::Tensor& recons_data, std::vector<int32_t> shape, int64_t pad_T) const;
    
    void load_models();
    void load_probability_tables();
    

    std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
    std::vector<int32_t> vbr_cdf_length_;
    std::vector<int32_t> vbr_offset_;
    std::vector<std::vector<int32_t>> gs_quantized_cdf_;
    std::vector<int32_t> gs_cdf_length_;
    std::vector<int32_t> gs_offset_;
};
