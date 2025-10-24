#pragma once
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "../dataset/dataset.h"
#include <string>
#include <vector>
#include <iostream>
#include <memory>


struct CompressionResult {
    std::vector<torch::Tensor> latents;
    std::vector<torch::Tensor> hyper_latents;
    std::vector<torch::Tensor> offsets;
    std::vector<torch::Tensor> scales;
    std::vector<torch::Tensor> indices;
    int64_t num_samples;
    int64_t num_batches;
};

class Compressor {
public:
    explicit Compressor(torch::Device device = torch::kCPU);
    
    std::vector<torch::Tensor> compress_single(const torch::Tensor& input_tensor);
    
    CompressionResult compress(const DatasetConfig& config, int batch_size = 32);
    
private:
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
    torch::Device device_;
    std::string model_path_;
    
    void load_model();
};
