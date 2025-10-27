#pragma once

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <vector>
#include <string>
#include "../dataset/dataset.h"

struct CompressionResult {
    std::vector<std::string> encoded_latents;
    std::vector<std::string> encoded_hyper_latents;
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

    void load_models();
    void load_probability_tables();

    std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
    std::vector<int32_t> vbr_cdf_length_;
    std::vector<int32_t> vbr_offset_;

    std::vector<std::vector<int32_t>> gs_quantized_cdf_;
    std::vector<int32_t> gs_cdf_length_;
    std::vector<int32_t> gs_offset_;
};
