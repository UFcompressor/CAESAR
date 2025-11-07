#pragma once

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <vector>
#include <string>
#include "../dataset/dataset.h"

struct CompressionResult {
    std::vector<std::string> encoded_latents;
    std::vector<std::string> encoded_hyper_latents;
    // ** JL modified ** //
    // record metadata for decompression
    std::vector<float> offsets; // local info - corresponding to latent
    std::vector<float> scales; // local info - corresponding to latent
    std::vector<std::vector<int32_t>> indexes; // local info - corresponding to latent
    std::tuple<int32_t, int32_t, std::vector<int32_t>> block_info; // global info
    std::vector<int32_t> data_input_shape; // global info
    std::vector<std::pair<int32_t, float>> filtered_blocks; // global info
    // **** //
    int num_samples;
    int num_batches;
};

class Compressor {
public:
    explicit Compressor(torch::Device device = torch::kCPU);
    ~Compressor() = default;

    CompressionResult compress(const DatasetConfig& config , int batch_size = 32);

private:
    torch::Device device_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> compressor_model_;
    // ** JL modified ** //
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> hyper_decompressor_model_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> decompressor_model_;
    
    torch::Tensor reshape_batch_2d_3d(const torch::Tensor& batch_data, int64_t batch_size);
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
