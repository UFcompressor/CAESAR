#pragma once

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <vector>
#include <string>
#include "../dataset/dataset.h"


// ** JL modified ** //
struct GAEMetaData {
    bool GAE_correction_occur;
    std::vector<int> padding_recon_info; // global info before GAE (GAE preparation)
    std::vector<std::vector<float>> pcaBasis; // tensor is converted into vector for adios
    std::vector<float> uniqueVals; // tensor is converted into vector for adios
    double quanBin;
    int64_t nVec;
    int64_t prefixLength;
    int64_t dataBytes;
    size_t coeffIntBytes;
};

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
// **** //

struct CompressionResult {
    std::vector<std::string> encoded_latents;
    std::vector<std::string> encoded_hyper_latents;
    // ** JL modified ** //
    // GAE compressed data
    std::vector<uint8_t> gae_comp_data;
    // record metadata for decompression
    CompressionMetaData compressionMetaData;
    GAEMetaData gaeMetaData;

  
    // **** //
    int num_samples;
    int num_batches;
};

class Compressor {
public:
    explicit Compressor(torch::Device device = torch::Device(torch::kCPU));
    ~Compressor() = default;

    CompressionResult compress(const DatasetConfig& config , int batch_size = 32 , float rel_eb = 0.1);

private:
    torch::Device device_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> compressor_model_;
    // ** JL modified ** //
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> hyper_decompressor_model_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> decompressor_model_;

    torch::Tensor reshape_batch_2d_3d(const torch::Tensor& batch_data , int64_t batch_size);
    torch::Tensor deblockHW(const torch::Tensor& data , int64_t nH , int64_t nW , const std::vector<int64_t>& padding);
    torch::Tensor recons_data(const torch::Tensor& recons_data , std::vector<int32_t> shape , int64_t pad_T) const;
    // **** //

    void load_models();
    void load_probability_tables();

    std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
    std::vector<int32_t> vbr_cdf_length_;
    std::vector<int32_t> vbr_offset_;

    std::vector<std::vector<int32_t>> gs_quantized_cdf_;
    std::vector<int32_t> gs_cdf_length_;
    std::vector<int32_t> gs_offset_;
};
