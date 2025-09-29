#include "compressor.h"

Compressor::Compressor(const std::string& model_path, torch::Device device)
    : loader_(model_path), device_(device) {}

std::vector<torch::Tensor> Compressor::compress(const torch::Tensor& input_tensor) {
    std::vector<torch::Tensor> inputs = { input_tensor.to(device_) };
    return loader_.run(inputs);
}

