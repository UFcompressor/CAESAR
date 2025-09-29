#pragma once
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <string>
#include <vector>
#include <iostream>

class Compressor {
public:

    Compressor(const std::string& model_path, torch::Device device = torch::kCPU);

    std::vector<torch::Tensor> compress(const torch::Tensor& input_tensor);

private:
    torch::inductor::AOTIModelPackageLoader loader_;
    torch::Device device_;
};

