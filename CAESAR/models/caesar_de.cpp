#include "caesar_decompress.h"
#include "range_coder/rans_coder.hpp"
#include "runGaeCuda.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>

template<typename T>
std::vector<T> load_array_from_bin(const std::string& filename) {
    std::ifstream input_file(filename , std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    input_file.seekg(0 , std::ios::end);
    size_t file_size_in_bytes = input_file.tellg();
    input_file.seekg(0 , std::ios::beg);

    size_t num_elements = file_size_in_bytes / sizeof(T);
    std::vector<T> loaded_data(num_elements);
    input_file.read(reinterpret_cast<char*>(loaded_data.data()) , file_size_in_bytes);
    input_file.close();
    return loaded_data;
}

template<typename T>
std::vector<std::vector<T>> reshape_to_2d(const std::vector<T>& flat_vec , size_t rows , size_t cols) {
    if (flat_vec.size() != rows * cols) {
        throw std::invalid_argument("Invalid dimensions for reshape.");
    }
    std::vector<std::vector<T>> vec_2d;
    vec_2d.reserve(rows);
    auto it = flat_vec.begin();
    for (size_t r = 0; r < rows; ++r) {
        vec_2d.emplace_back(it , it + cols);
        it += cols;
    }
    return vec_2d;
}

template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();
    const T* tensor_data_ptr = cpu_tensor.data_ptr<T>();
    int64_t num_elements = cpu_tensor.numel();
    return std::vector<T>(tensor_data_ptr , tensor_data_ptr + num_elements);
}

torch::Tensor build_indexes_tensor(const std::vector<int32_t>& size) {
    int64_t dims = size.size();
    TORCH_CHECK(dims >= 2 , "Input size must have at least 2 dimensions (N, C, ...)");
    int64_t C = size[1];
    std::vector<int64_t> view_dims = { 1, C };
    view_dims.insert(view_dims.end() , dims - 2 , 1);
    torch::Tensor indexes = torch::arange(C).view(view_dims);
    std::vector<int64_t> size_int64(size.begin() , size.end());
    return indexes.expand(size_int64).to(torch::kInt32);
}

// padding 与 unpadding
std::tuple<torch::Tensor , std::vector<int>> padding(
    const torch::Tensor& data ,
    std::pair<int , int> block_size = {8, 8})
{
    int h_block = block_size.first;
    int w_block = block_size.second;
    auto sizes = data.sizes();
    int ndim = sizes.size();
    int H = sizes[ndim - 2];
    int W = sizes[ndim - 1];

    int H_target = std::ceil((float)H / h_block) * h_block;
    int W_target = std::ceil((float)W / w_block) * w_block;
    int dh = H_target - H;
    int dw = W_target - W;
    int top = dh / 2, down = dh - top;
    int left = dw / 2, right = dw - left;

    auto leading_dims = data.sizes().vec();
    int leading_size = 1;
    for (size_t i = 0; i < leading_dims.size() - 2; ++i)
        leading_size *= leading_dims[i];
    auto data_reshaped = data.view({ leading_size, H, W });

    auto data_padded = torch::nn::functional::pad(
        data_reshaped,
        torch::nn::functional::PadFuncOptions({ left, right, top, down }).mode(torch::kReflect));

    auto new_shape = leading_dims;
    new_shape[new_shape.size() - 2] = data_padded.size(-2);
    new_shape[new_shape.size() - 1] = data_padded.size(-1);
    auto padded_data = data_padded.view(new_shape);

    std::vector<int> padding_info = { top, down, left, right };
    return { padded_data, padding_info };
}

torch::Tensor unpadding(const torch::Tensor& padded_data , const std::vector<int>& padding)
{
    int top = padding[0];
    int down = padding[1];
    int left = padding[2];
    int right = padding[3];
    auto sizes = padded_data.sizes();
    int ndim = sizes.size();
    int H = sizes[ndim - 2];
    int W = sizes[ndim - 1];
    auto unpadded = padded_data.index({
        torch::indexing::Ellipsis,
        torch::indexing::Slice(top, H - down),
        torch::indexing::Slice(left, W - right)
    });
    return unpadded;
}

// ======================================================================
// Decompressor 类实现
// ======================================================================
Decompressor::Decompressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Decompressor::load_models() {
    std::cout << "Loading decompressor models..." << std::endl;
    hyper_decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/adios/Programs/CAESAR_C/exported_model/caesar_hyper_decompressor.pt2");
    decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/adios/Programs/CAESAR_C/exported_model/caesar_decompressor.pt2");
    std::cout << "Models loaded successfully." << std::endl;
}

void Decompressor::load_probability_tables() {
    std::cout << "Loading probability tables..." << std::endl;
    auto vbr_quantized_cdf_1d = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/vbr_quantized_cdf.bin");
    vbr_cdf_length_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/vbr_cdf_length.bin");
    vbr_offset_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/vbr_offset.bin");
    vbr_quantized_cdf_ = reshape_to_2d(vbr_quantized_cdf_1d , 64 , 63);

    auto gs_quantized_cdf_1d = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/gs_quantized_cdf.bin");
    gs_cdf_length_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/gs_cdf_length.bin");
    gs_offset_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/gs_offset.bin");
    gs_quantized_cdf_ = reshape_to_2d(gs_quantized_cdf_1d , 128 , 249);
    std::cout << "Probability tables loaded successfully." << std::endl;
}

torch::Tensor Decompressor::reshape_batch_2d_3d(
    const torch::Tensor& batch_data ,
    int64_t batch_size ,
    int64_t n_frame
) {
    auto sizes = batch_data.sizes();
    TORCH_CHECK(sizes.size() == 4 , "Input tensor must be 4D [B*T, C, H, W]");
    int64_t BT = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];
    int64_t T = BT / batch_size;
    torch::Tensor reshaped_data = batch_data.view({ batch_size, T, C, H, W });
    torch::Tensor permuted_data = reshaped_data.permute({ 0, 2, 1, 3, 4 });
    return permuted_data;
}

// ======================================================================
// GAE 后处理解码
// ======================================================================
torch::Tensor Decompressor::postprocessing_decoding(
    const torch::Tensor& padded_recon_tensor,
    const GaeMetaData& gaeMetaData,
    const std::vector<uint8_t>& gae_comp_data,
    const std::vector<int>& padding_recon_info
) {
    std::cout << "\n========== STARTING GAE DECODING ==========" << std::endl;

    int64_t pca_rows = gaeMetaData.pcaBasis.size();
    int64_t pca_cols = gaeMetaData.pcaBasis[0].size();
    std::vector<float> pca_vec;
    pca_vec.reserve(pca_rows * pca_cols);
    for (const auto& row_vec : gaeMetaData.pcaBasis)
        pca_vec.insert(pca_vec.end(), row_vec.begin(), row_vec.end());

    torch::Tensor pcaBasis = torch::tensor(pca_vec).reshape({ pca_rows, pca_cols }).to(device_);
    torch::Tensor uniqueVals = torch::tensor(gaeMetaData.uniqueVals).to(device_);

    MetaData gae_meta;
    CompressedData gae_comp;
    gae_meta.pcaBasis = pcaBasis;
    gae_meta.uniqueVals = uniqueVals;
    gae_meta.quanBin = gaeMetaData.quanBin;
    gae_meta.nVec = gaeMetaData.nVec;
    gae_meta.prefixLength = gaeMetaData.prefixLength;
    gae_meta.dataBytes = gaeMetaData.dataBytes;

    gae_comp.data = gae_comp_data;
    gae_comp.dataBytes = gaeMetaData.dataBytes;
    gae_comp.coeffIntBytes = gaeMetaData.coeffIntBytes;

    float quan_factor = 2.0;
    float rel_eb = 1e-3;
    std::string codec_alg = "Zstd";
    std::pair<int,int> patch_size = {8,8};

    PCACompressor pca_decompressor(
        rel_eb,
        quan_factor,
        device_.is_cuda() ? "cuda" : "cpu",
        codec_alg,
        patch_size
    );

    torch::Tensor recons_gae = pca_decompressor.decompress(
        padded_recon_tensor,
        gae_meta,
        gae_comp
    );

    std::cout << "[GAE DECODE] recons_gae shape: " << recons_gae.sizes() << std::endl;
    torch::Tensor recons_gae_unpadded = unpadding(recons_gae , padding_recon_info);
    std::cout << "[GAE DECODE] recons_gae_unpadded shape: " << recons_gae_unpadded.sizes() << std::endl;
    std::cout << "========== GAE DECODING COMPLETE ==========" << std::endl;
    return recons_gae_unpadded;
}

// ======================================================================
// 主解压函数
// ======================================================================
DecompressionResult Decompressor::decompress(
    const std::vector<std::string>& encoded_latents,
    const std::vector<std::string>& encoded_hyper_latents,
    const std::vector<torch::Tensor>& offsets,
    const std::vector<torch::Tensor>& scales,
    const std::vector<torch::Tensor>& indexes,
    int batch_size,
    int n_frame,
    const GaeMetaData& gaeMetaData,
    const std::vector<uint8_t>& gae_comp_data,
    const std::vector<int>& padding_recon_info
) {
    c10::InferenceMode guard;
    RansDecoder range_decoder;

    std::cout << "\n========== STARTING DECOMPRESSION ==========" << std::endl;

    // ---------------- Hyper + Latent 解码 ----------------
    torch::Tensor decoded_latents;
    {
        size_t current_batch_size = encoded_latents.size();
        std::vector<int32_t> hyper_size = { (int)current_batch_size, 64, 4, 4 };
        torch::Tensor hyper_index_tensor = build_indexes_tensor(hyper_size);
        torch::Tensor decoded_hyper_latents = torch::zeros({ (int64_t)current_batch_size, 64, 4, 4 }).to(torch::kInt32);

        for (size_t i = 0; i < current_batch_size; i++) {
            std::vector<int32_t> hyper_index_vec = tensor_to_vector<int32_t>(
                hyper_index_tensor.select(0, i).reshape(-1)
            );
            std::vector<int32_t> hyper_decoded = range_decoder.decode_with_indexes(
                encoded_hyper_latents[i],
                hyper_index_vec,
                vbr_quantized_cdf_,
                vbr_cdf_length_,
                vbr_offset_
            );
            torch::Tensor hyper_tensor = torch::tensor(hyper_decoded).reshape({ 64, 4, 4 });
            decoded_hyper_latents.select(0, i).copy_(hyper_tensor);
        }

        std::vector<torch::Tensor> hyper_outputs = hyper_decompressor_model_->run({
            decoded_hyper_latents.to(torch::kFloat32).to(device_)
        });
        torch::Tensor mean = hyper_outputs[0];
        torch::Tensor latent_indexes_recon = hyper_outputs[1];

        torch::Tensor decoded_latents_before_offset = torch::zeros({ (int64_t)current_batch_size, 64, 16, 16 }).to(torch::kInt32);
        for (size_t i = 0; i < current_batch_size; i++) {
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                latent_indexes_recon.select(0, i).reshape(-1)
            );
            std::vector<int32_t> latent_decoded = range_decoder.decode_with_indexes(
                encoded_latents[i],
                latent_index,
                gs_quantized_cdf_,
                gs_cdf_length_,
                gs_offset_
            );
            torch::Tensor latent_tensor = torch::tensor(latent_decoded).reshape({ 64, 16, 16 });
            decoded_latents_before_offset.select(0, i).copy_(latent_tensor);
        }

        decoded_latents = decoded_latents_before_offset.to(device_).to(torch::kFloat32) + mean;
    }

    // ---------------- 主解码器 ----------------
    std::vector<int64_t> new_shape = { -1, 2, 64, 16, 16 };
    torch::Tensor reshaped_latents = decoded_latents.reshape(new_shape);
    std::vector<torch::Tensor> decompressor_outputs = decompressor_model_->run({ reshaped_latents });
    torch::Tensor raw_output = decompressor_outputs[0];
    torch::Tensor output = reshape_batch_2d_3d(raw_output, batch_size / 2, n_frame);

    std::cout << "[DECOMPRESS] Raw output shape: " << output.sizes() << std::endl;

    // ---------------- padding 与 GAE 解码 ----------------
    auto padded_pair = padding(output);
    torch::Tensor padded_recon_tensor = std::get<0>(padded_pair);
    std::vector<int> rec_padding = std::get<1>(padded_pair);

    torch::Tensor final_recon = postprocessing_decoding(
        padded_recon_tensor,
        gaeMetaData,
        gae_comp_data,
        padding_recon_info
    );

    std::cout << "[FINAL] Decompressed reconstruction shape: " << final_recon.sizes() << std::endl;
    std::cout << "========== DECOMPRESSION COMPLETE ==========" << std::endl;

    DecompressionResult result;
    result.reconstructed_data = { final_recon };
    result.num_samples = 1;
    return result;
}
