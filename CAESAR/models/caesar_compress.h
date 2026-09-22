#pragma once
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <thread>
#include <utility>

#include "../data_utils.h"
#include "../dataset/dataset.h"
#include "array_utils.h"
#include "correction_method.h"
#include "lbrc.h"
#include "model_cache.h"
#include "model_utils.h"
#include "nglr.h"
#include "range_coder/rans_coder.hpp"
#include "runGaeCuda.h"

// Pass the original [T,H,W], [S,T,H,W], or [V,S,T,H,W] tensor.
// For 5D input only variable 0 is encoded; decoding returns [1,S,T,H,W].
struct CompressionConfig {
  torch::Tensor memory_data;
  // Required. The current compiled architecture supports eight-frame windows.
  int n_frame = 0;
  caesar::CorrectionMethod correction_method = caesar::CorrectionMethod::GAE;
  nglr::NGLRTrainOptions nglr_options;
};

struct GAEMetaData {
  bool GAE_correction_occur = false;
  std::vector<int>
      padding_recon_info; // global info before GAE (GAE preparation)
  std::vector<std::vector<float>>
      pcaBasis;                  // tensor is converted into vector for adios
  std::vector<float> uniqueVals; // tensor is converted into vector for adios
  double quanBin = 0;
  int64_t nVec = 0;
  int64_t prefixLength = 0;
  int64_t dataBytes = 0;
  size_t coeffIntBytes = 0;
};

struct CompressionMetaData {
  std::vector<float> offsets; // local info - corresponding to latent
  std::vector<float> scales;  // local info - corresponding to latent
  std::vector<std::vector<int32_t>>
      indexes; // local info - corresponding to latent
  std::tuple<int32_t, int32_t, std::vector<int32_t>> block_info; // global info
  std::vector<int32_t> data_input_shape;                         // global info
  std::vector<std::pair<int32_t, float>> filtered_blocks;        // global info
  float global_scale = 0;                                        // global info
  float global_offset = 0;                                       // global info
  int64_t pad_T = 0;                                             // global_info
  bool all_filtered = false; // all data is the same
};

struct CompressionResult {
  // Full caller shape, including the original variable count for 5D input.
  std::vector<int64_t> original_shape;
  // Conversion metadata describes the selected variable and restored shape.
  PaddingInfo shape_info{};
  int n_frame = 0;
  std::string model_id;

  std::vector<std::string> encoded_latents;
  std::vector<std::string> encoded_hyper_latents;
  // std::vector<std::vector<uint8_t>> latent_indexes;

  // GAE compressed data
  std::vector<uint8_t> gae_comp_data;

  // LBRC compressed data
  std::vector<LBRCBlock> lbrc_blocks;
  LBRCMetaData lbrcMetaData;

  // record metadata for decompression
  CompressionMetaData compressionMetaData;
  GAEMetaData gaeMetaData;

  // set correction_method here
  caesar::CorrectionMethod correction_method = caesar::CorrectionMethod::GAE;
  nglr::NGLRMetaData nglrMetaData;
  std::vector<uint8_t> nglr_comp_data;
};

class Compressor {
public:
  explicit Compressor(const std::string &required_model_id = "");
  ~Compressor() = default;

  CompressionResult compress(const CompressionConfig &config,
                             float rel_eb = 0.1);

private:
  const std::thread::id owner_thread_ = std::this_thread::get_id();
  torch::Device device_;

  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> compressor_model_;
  std::shared_ptr<torch::inductor::AOTIModelPackageLoader>
      hyper_decompressor_model_;
  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> decompressor_model_;

  torch::Tensor reshape_batch_2d_3d(const torch::Tensor &batch_data,
                                    int64_t batch_size);
  torch::Tensor deblockHW(const torch::Tensor &data, int64_t nH, int64_t nW,
                          const std::vector<int64_t> &padding);
  torch::Tensor recons_data(const torch::Tensor &recons_data,
                            std::vector<int32_t> shape, int64_t pad_T) const;

  void load_models();
  void load_probability_tables();

  std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
  std::vector<int32_t> vbr_cdf_length_;
  std::vector<int32_t> vbr_offset_;
  std::vector<std::vector<int32_t>> gs_quantized_cdf_;
  std::vector<int32_t> gs_cdf_length_;
  std::vector<int32_t> gs_offset_;

  std::string model_name_;
  std::string device_type_;
};
