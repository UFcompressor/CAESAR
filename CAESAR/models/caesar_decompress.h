#pragma once

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <vector>
#include <string>

struct DecompressionResult {
    std::vector<torch::Tensor> reconstructed_data;
    int num_samples;
    int num_batches;
};

class Decompressor {
public:
    explicit Decompressor(torch::Device device = torch::kCPU);
    ~Decompressor() = default;

    // Main decompression API
    DecompressionResult decompress(
        const std::vector<std::string>& encoded_latents,
        const std::vector<std::string>& encoded_hyper_latents,
        int batch_size = 32,
        int n_frame = 8
    );

private:
    torch::Device device_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> hyper_decompressor_model_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> decompressor_model_;
    
    // Helper methods
    void load_models();
    void load_probability_tables();
    torch::Tensor reshape_batch_2d_3d(const torch::Tensor& batch_data, int64_t batch_size, int64_t n_frame);
    
    // Probability tables for decoding
    std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
    std::vector<int32_t> vbr_cdf_length_;
    std::vector<int32_t> vbr_offset_;
    
    std::vector<std::vector<int32_t>> gs_quantized_cdf_;
    std::vector<int32_t> gs_cdf_length_;
    std::vector<int32_t> gs_offset_;
};
