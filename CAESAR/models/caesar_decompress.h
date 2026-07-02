#pragma once
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <memory>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include "array_utils.h"
#include "caesar_compress.h"
#include "model_cache.h"
#include "model_utils.h"
#include "range_coder/rans_coder.hpp"
#include "runGaeCuda.h"

struct CompressionResult;
struct DecompressionResult {
  std::vector<torch::Tensor> reconstructed_data;
  int num_samples = 0;
  int num_batches = 0;
};

class Decompressor {
 public:
  explicit Decompressor(torch::Device device = torch::Device(torch::kCPU));
  ~Decompressor() = default;

  torch::Tensor decompress(const unsigned int batch_size,
                           const unsigned int n_frame,
                           const CompressionResult& comp_result);

 private:
  torch::Device device_;

  torch::inductor::AOTIModelPackageLoader* hyper_decompressor_model_;
  torch::inductor::AOTIModelPackageLoader* decompressor_model_;

  void load_models();
  void load_probability_tables();
  void load_text_files();

  torch::Tensor reshape_batch_2d_3d(const torch::Tensor& batch_data,
                                    int64_t batch_size, int64_t n_frame);
  torch::Tensor range_decode_latents(
      const std::vector<std::string>& encoded_latents,
      const std::vector<std::string>& encoded_hyper_latents);
  torch::Tensor apply_inverse_normalization(const torch::Tensor& data,
                                            const std::vector<float>& offsets,
                                            const std::vector<float>& scales);
  torch::Tensor reconstruct_full_tensor(
      const std::vector<torch::Tensor>& block_tensors,
      const std::vector<std::vector<int32_t>>& indexes);

  std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
  std::vector<int32_t> vbr_cdf_length_;
  std::vector<int32_t> vbr_offset_;
  std::vector<std::vector<int32_t>> gs_quantized_cdf_;
  std::vector<int32_t> gs_cdf_length_;
  std::vector<int32_t> gs_offset_;

  std::string model_name_;
  std::string device_type_;
};
