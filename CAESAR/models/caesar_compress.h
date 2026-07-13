#pragma once
#include "lbrc.h"
#include "nglr_model.h"
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "model_cache.h"
#include "array_utils.h"
#include "../dataset/dataset.h"
#include "range_coder/rans_coder.hpp"
#include "runGaeCuda.h" 
#include "model_utils.h"
#include <fstream>
#include <cmath>
#include <limits>
#include <utility>
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif



struct GAEMetaData {
    bool GAE_correction_occur = false;
    std::vector<int> padding_recon_info; // global info before GAE (GAE preparation)
    std::vector<std::vector<float>> pcaBasis; // tensor is converted into vector for adios
    std::vector<float> uniqueVals; // tensor is converted into vector for adios
    double quanBin = 0.0;
    int64_t nVec = 0;
    int64_t prefixLength = 0;
    int64_t dataBytes = 0;
    size_t coeffIntBytes = 0;
};

struct CompressionMetaData {
    std::vector<float> offsets; // local info - corresponding to latent
    std::vector<float> scales; // local info - corresponding to latent
    std::vector<std::vector<int32_t>> indexes; // local info - corresponding to latent
    std::tuple<int32_t , int32_t , std::vector<int32_t>> block_info; // global info
    std::vector<int32_t> data_input_shape; // global info
    std::vector<std::pair<int32_t , float>> filtered_blocks; // global info
    float global_scale = 1.0f; // global info
    float global_offset = 0.0f; // global info
    int64_t pad_T = 0; // global_info
    bool all_filtered = false; // all data is the same 
};


struct CompressionResult {
    std::vector<std::string> encoded_latents;
    std::vector<std::string> encoded_hyper_latents;
  //  std::vector<std::vector<int32_t>> latent_indexes;

    // GAE compressed data
    std::vector<uint8_t> gae_comp_data;

    // LBRC compressed data
    std::vector<LBRCBlock> lbrc_blocks;
    LBRCMetaData    lbrcMetaData;

    // NGLR compressed data
    caesar::nglr::NGLRMetaData nglrMetaData;
    caesar::nglr::NGLRCompressedData nglrCompressedData;

    // record metadata for decompression
    CompressionMetaData compressionMetaData;
    GAEMetaData gaeMetaData;

    bool use_lbrc = false;
    bool use_nglr = false;
};
 
class Compressor {
public:
    explicit Compressor(torch::Device device = torch::Device(torch::kCPU));
    ~Compressor() = default;

    CompressionResult compress(const DatasetConfig& config,
                               int batch_size = 32,
                               float rel_eb = 0.1,
                               const std::string& correction_method = "gae");
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
    void load_text_files();
    
    std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
    std::vector<int32_t> vbr_cdf_length_;
    std::vector<int32_t> vbr_offset_;
    std::vector<std::vector<int32_t>> gs_quantized_cdf_;
    std::vector<int32_t> gs_cdf_length_;
    std::vector<int32_t> gs_offset_;

    std::string model_name_;
    std::string device_type_;
};
