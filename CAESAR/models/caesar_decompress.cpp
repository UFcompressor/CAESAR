#include "caesar_decompress.h"
#include "range_coder/rans_coder.hpp"
#include <iostream>
#include <fstream>
#include <filesystem> 
#include "runGaeCuda.h"
#include <cmath>
#include <limits>
#include "caesar_compress.h"
#include "model_utils.h"

torch::Tensor deblockHW(const torch::Tensor& data ,
    int64_t nH ,
    int64_t nW ,
    const std::vector<int64_t>& padding);

std::tuple<torch::Tensor , std::vector<int>> padding(
    const torch::Tensor& data ,
    std::pair<int , int> block_size = { 8, 8 });
torch::Tensor recons_data(const torch::Tensor& recons_data ,
    const std::vector<int32_t>& shape ,
    int64_t pad_T)
{
    int64_t stop_t = shape[2] - pad_T;
    return recons_data.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(0, stop_t),
        torch::indexing::Slice(),
        torch::indexing::Slice()
        });
}
torch::Tensor unpadding(const torch::Tensor& padded_data , const std::vector<int>& padding);

template<typename T>
std::vector<T> load_array_from_bin(const std::string& filename) {
    std::ifstream input_file(filename , std::ios::binary);
    if (!input_file.is_open()) throw std::runtime_error("Cannot open file: " + filename);
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
    if (flat_vec.size() != rows * cols) throw std::invalid_argument("Invalid dimensions for reshape.");
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



Decompressor::Decompressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Decompressor::load_models() {

    std::string hyper_decompressor_path;
    std::string decompressor_path;
    std::cout << "Loading decompressor models..." << std::endl;
    hyper_decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        get_model_file("caesar_hyper_decompressor.pt2").string()
    );
    decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        get_model_file("caesar_decompressor.pt2").string()
    );
}

void Decompressor::load_probability_tables() {

    auto vbr_quantized_cdf_1d = load_array_from_bin<int32_t>(get_model_file("vbr_quantized_cdf.bin"));
    vbr_cdf_length_ = load_array_from_bin<int32_t>(get_model_file("vbr_cdf_length.bin"));
    vbr_offset_ = load_array_from_bin<int32_t>(get_model_file("vbr_offset.bin"));
    vbr_quantized_cdf_ = reshape_to_2d(vbr_quantized_cdf_1d , 64 , 63);

    auto gs_quantized_cdf_1d = load_array_from_bin<int32_t>(get_model_file("gs_quantized_cdf.bin"));
    gs_cdf_length_ = load_array_from_bin<int32_t>(get_model_file("gs_cdf_length.bin"));
    gs_offset_ = load_array_from_bin<int32_t>(get_model_file("gs_offset.bin"));
    gs_quantized_cdf_ = reshape_to_2d(gs_quantized_cdf_1d , 128 , 249);

}

torch::Tensor Decompressor::reshape_batch_2d_3d(const torch::Tensor& batch_data , int64_t batch_size , int64_t n_frame) {
    auto sizes = batch_data.sizes();
    TORCH_CHECK(sizes.size() == 4 , "Input tensor must be 4D.");
    int64_t BT = sizes[0] , C = sizes[1] , H = sizes[2] , W = sizes[3];
    int64_t T = BT / batch_size;
    torch::Tensor reshaped_data = batch_data.view({ batch_size, T, C, H, W });
    torch::Tensor permuted_data = reshaped_data.permute({ 0, 2, 1, 3, 4 });
    return permuted_data;
}

torch::Tensor Decompressor::decompress(
    const std::vector<std::string>& encoded_latents ,
    const std::vector<std::string>& encoded_hyper_latents ,
    int batch_size ,
    int n_frame ,
    const CompressionResult& comp_result
) {
    c10::InferenceMode guard;
    std::cout << "\n========== STARTING DECOMPRESSION ==========" << std::endl;
    std::cout << "Device: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;

    DecompressionResult result;
    result.num_samples = 0;
    result.num_batches = 0;

    RansDecoder range_decoder;

    auto& meta = comp_result.compressionMetaData;

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    torch::Tensor offsets_tensor = torch::tensor(meta.offsets , opts);
    torch::Tensor scales_tensor = torch::tensor(meta.scales , opts);

    std::vector<int32_t> flat_indexes;
    flat_indexes.reserve(meta.indexes.size() * meta.indexes[0].size());
    for (const auto& v : meta.indexes)
        flat_indexes.insert(flat_indexes.end() , v.begin() , v.end());

    torch::TensorOptions idx_opts_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor indexes_tensor = torch::from_blob(
        flat_indexes.data() ,
        { (long)meta.indexes.size(), (long)meta.indexes[0].size() } ,
        idx_opts_cpu
    ).clone().to(device_);

    flat_indexes.clear();
    flat_indexes.shrink_to_fit();

    if (indexes_tensor.numel() > 0) {
        std::cout << "[DEBUG] indexes_tensor shape: " << indexes_tensor.sizes() << std::endl;

        int64_t rows_to_print = std::min((int64_t)3 , indexes_tensor.size(0));
        int64_t cols = indexes_tensor.size(1);

        for (int64_t i = 0; i < rows_to_print; ++i) {
            auto row = indexes_tensor[i];
            auto row_cpu = row.to(torch::kCPU);
            auto acc = row_cpu.accessor<int32_t , 1>();
            for (int64_t j = 0; j < cols; ++j) {
                std::cout << acc[j] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::vector<int64_t> input_shape(meta.data_input_shape.begin() , meta.data_input_shape.end());
    torch::Tensor recon_tensor = torch::zeros(input_shape).to(device_);
    std::cout << "\nrecon_tensor shape: " << recon_tensor.sizes() << std::endl;

    for (size_t lat_start = 0; lat_start < encoded_latents.size(); lat_start += (size_t)batch_size * 2) {
        size_t lat_end = std::min(lat_start + (size_t)batch_size * 2 , encoded_latents.size());
        size_t cur_latents = lat_end - lat_start;
        TORCH_CHECK(cur_latents % 2 == 0 , "cur_latents must be even.");
        size_t cur_samples = cur_latents / 2;
        size_t sample_start = lat_start / 2;

        std::vector<int32_t> hyper_size = { (int32_t)cur_latents, 64, 4, 4 };

        torch::Tensor hyper_index_tensor = build_indexes_tensor(hyper_size).contiguous();

        torch::Tensor decoded_hyper_latents = torch::zeros({ (long)cur_latents, 64, 4, 4 }).to(torch::kInt32);

        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> hyper_index_vec = tensor_to_vector<int32_t>(hyper_index_tensor.select(0 , (long)i).reshape(-1));

            std::vector<int32_t> hyper_decoded = range_decoder.decode_with_indexes(
                encoded_hyper_latents[lat_start + i] ,
                hyper_index_vec ,
                vbr_quantized_cdf_ ,
                vbr_cdf_length_ ,
                vbr_offset_
            );
            torch::Tensor hyper_tensor = torch::tensor(hyper_decoded).reshape({ 64, 4, 4 });
            decoded_hyper_latents.select(0 , (long)i).copy_(hyper_tensor);
        }

        std::vector<torch::Tensor> hyper_outputs = hyper_decompressor_model_->run({ decoded_hyper_latents.to(torch::kDouble).to(device_) });
        torch::Tensor mean = hyper_outputs[0].to(torch::kFloat32);

        torch::Tensor latent_indexes_recon = hyper_outputs[1].to(torch::kInt32);

        torch::Tensor latent_indexes_cpu = latent_indexes_recon.cpu().contiguous();

        torch::Tensor decoded_latents_before_offset = torch::zeros({ (long)cur_latents, 64, 16, 16 }).to(torch::kInt32);

        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(latent_indexes_recon.select(0 , (long)i).reshape(-1));

            std::vector<int32_t> latent_decoded = range_decoder.decode_with_indexes(
                encoded_latents[lat_start + i] ,
                latent_index ,
                gs_quantized_cdf_ ,
                gs_cdf_length_ ,
                gs_offset_
            );
            torch::Tensor latent_tensor = torch::tensor(latent_decoded).reshape({ 64, 16, 16 });
            decoded_latents_before_offset.select(0 , (long)i).copy_(latent_tensor);
        }

        torch::Tensor q_latent_with_offset = decoded_latents_before_offset.to(torch::kFloat32).to(device_) + mean;
        auto decoded_latents_sizes = q_latent_with_offset.sizes();
        std::vector<int64_t> new_shape = { -1, 2 };
        new_shape.insert(new_shape.end() , decoded_latents_sizes.begin() + 1 , decoded_latents_sizes.end());
        torch::Tensor reshaped_latents = q_latent_with_offset.reshape(new_shape);

        std::vector<torch::Tensor> decompressor_outputs = decompressor_model_->run({ reshaped_latents.to(torch::kFloat32) });
        torch::Tensor raw_output = decompressor_outputs[0];

        torch::Tensor norm_output = reshape_batch_2d_3d(raw_output , (long)cur_samples , n_frame);

        torch::Tensor batched_offsets = offsets_tensor.narrow(0 , (long)sample_start , (long)cur_samples)
            .view({ -1, 1, 1, 1, 1 });
        torch::Tensor batched_scales = scales_tensor.narrow(0 , (long)sample_start , (long)cur_samples)
            .view({ -1, 1, 1, 1, 1 });
        torch::Tensor denorm_output = norm_output * batched_scales + batched_offsets;

        torch::Tensor indexes_cpu = indexes_tensor.narrow(0 , (long)sample_start , (long)cur_samples).to(torch::kCPU);
        for (int64_t i = 0; i < (int64_t)cur_samples; ++i) {
            torch::Tensor index_row = indexes_cpu.select(0 , i);
            int64_t idx0 = index_row[0].item<int64_t>();
            int64_t idx1 = index_row[1].item<int64_t>();
            int64_t start_t = index_row[2].item<int64_t>();
            int64_t end_t = index_row[3].item<int64_t>();

            torch::Tensor source_slice_3d = denorm_output.select(0 , i).squeeze(0);
            torch::Tensor dest_slice = recon_tensor.select(0 , idx0).select(0 , idx1).slice(0 , start_t , end_t);
            dest_slice.copy_(source_slice_3d);
        }

        result.num_samples += cur_samples;
        result.num_batches++;
    }

    auto [b1_i32 , b2_i32 , pad_i32] = meta.block_info;

    int64_t block_info_1 = b1_i32;
    int64_t block_info_2 = b2_i32;
    std::vector<int64_t> block_info_3(pad_i32.begin() , pad_i32.end());

    for (auto v : block_info_3)
        std::cout << v << " ";
    std::cout << std::endl;

    torch::Tensor recon_tensor_deblock = deblockHW(recon_tensor , block_info_1 , block_info_2 , block_info_3);
    recon_tensor = torch::Tensor();

    if (comp_result.gaeMetaData.GAE_correction_occur) {
        std::tuple<torch::Tensor , std::vector<int>> padding_recon = padding(recon_tensor_deblock);
        recon_tensor_deblock = torch::Tensor();
        torch::Tensor padded_recon_tensor = std::get<0>(padding_recon);
        std::vector<int> padding_recon_info = std::get<1>(padding_recon);

        float global_scale = meta.global_scale;
        float global_offset = meta.global_offset;
        torch::Tensor padded_recon_tensor_norm = (padded_recon_tensor - global_offset) / global_scale;
        double quan_factor = 2.0;

        std::string codec_alg = "Zstd";
        std::pair<int , int> patch_size = { 8, 8 };
        double rel_eb = 1e-3;
        PCACompressor pca_compressor(rel_eb ,
            quan_factor ,
            device_.is_cuda() ? "cuda" : "cpu" ,
            codec_alg ,
            patch_size);

        MetaData gae_record_metaData;
        CompressedData gae_record_compressedData;

        int64_t pca_rows = comp_result.gaeMetaData.pcaBasis.size();
        int64_t pca_cols = comp_result.gaeMetaData.pcaBasis[0].size();

        std::vector<float> pca_vec;
        pca_vec.reserve(pca_rows * pca_cols);

        for (const auto& row_vec : comp_result.gaeMetaData.pcaBasis) {
            pca_vec.insert(pca_vec.end() , row_vec.begin() , row_vec.end());
        }
        torch::Tensor pca_vec_1d = torch::tensor(pca_vec);
        torch::Tensor pcaBasis = pca_vec_1d.reshape({ pca_rows, pca_cols });

        gae_record_metaData.pcaBasis = pcaBasis.to(device_);
        gae_record_metaData.uniqueVals = torch::tensor(comp_result.gaeMetaData.uniqueVals).to(device_);
        gae_record_metaData.quanBin = comp_result.gaeMetaData.quanBin;
        gae_record_metaData.nVec = comp_result.gaeMetaData.nVec;
        gae_record_metaData.prefixLength = comp_result.gaeMetaData.prefixLength;
        gae_record_metaData.dataBytes = comp_result.gaeMetaData.dataBytes;

        gae_record_compressedData.data = comp_result.gae_comp_data;
        gae_record_compressedData.dataBytes = comp_result.gaeMetaData.dataBytes;
        gae_record_compressedData.coeffIntBytes = comp_result.gaeMetaData.coeffIntBytes;

        torch::Tensor recons_gae = pca_compressor.decompress(padded_recon_tensor_norm ,
            gae_record_metaData ,
            gae_record_compressedData);

        torch::Tensor recons_gae_unpadded = unpadding(recons_gae , comp_result.gaeMetaData.padding_recon_info);
        torch::Tensor final_recon_norm = recons_data(recons_gae_unpadded , meta.data_input_shape , meta.pad_T);
        torch::Tensor final_recon = final_recon_norm * meta.global_scale + meta.global_offset;
        return final_recon;
    }
    else {
        torch::Tensor final_recon = recons_data(recon_tensor_deblock , meta.data_input_shape , meta.pad_T);
        return final_recon;
    }
}
