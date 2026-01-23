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
#include "model_cache.h"
#include "array_utils.h"

torch::Tensor deblockHW(const torch::Tensor& data,
                        int64_t nH,
                        int64_t nW,
                        const std::vector<int64_t>& padding);

std::tuple<torch::Tensor, std::vector<int>> padding(
    const torch::Tensor& data,
    std::pair<int, int> block_size = {8, 8});

torch::Tensor unpadding(const torch::Tensor& padded_data, const std::vector<int>& padding);

torch::Tensor recons_data(const torch::Tensor& recons_data,
                          const std::vector<int32_t>& shape,
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

// ========== utils ==========
template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();
    const T* tensor_data_ptr = cpu_tensor.data_ptr<T>();
    int64_t num_elements = cpu_tensor.numel();
    return std::vector<T>(tensor_data_ptr, tensor_data_ptr + num_elements);
}

torch::Tensor build_indexes_tensor(const std::vector<int32_t>& size) {
    int64_t dims = (int64_t)size.size();
    TORCH_CHECK(dims >= 2, "Input size must have at least 2 dimensions (N, C, ...)");
    int64_t C = size[1];
    std::vector<int64_t> view_dims = {1, C};
    view_dims.insert(view_dims.end(), dims - 2, 1);
    torch::Tensor indexes = torch::arange(C).view(view_dims);
    std::vector<int64_t> size_int64(size.begin(), size.end());
    return indexes.expand(size_int64).to(torch::kInt32);
}

// ========== Decompressor ==========
Decompressor::Decompressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Decompressor::load_models() {
    hyper_decompressor_model_ = ModelCache::instance().get_hyper_decompressor_model();
    decompressor_model_ = ModelCache::instance().get_decompressor_model();
}

void Decompressor::load_probability_tables() {
    vbr_quantized_cdf_ = ModelCache::instance().get_vbr_quantized_cdf();
    vbr_cdf_length_ = ModelCache::instance().get_vbr_cdf_length();
    vbr_offset_ = ModelCache::instance().get_vbr_offset();

    gs_quantized_cdf_ = ModelCache::instance().get_gs_quantized_cdf();
    gs_cdf_length_ = ModelCache::instance().get_gs_cdf_length();
    gs_offset_ = ModelCache::instance().get_gs_offset();
}

torch::Tensor Decompressor::reshape_batch_2d_3d(const torch::Tensor& batch_data, int64_t batch_size, int64_t n_frame) {
    auto sizes = batch_data.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D.");
    int64_t BT = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    int64_t T = BT / batch_size;
    torch::Tensor reshaped_data  = batch_data.view({batch_size, T, C, H, W});
    torch::Tensor permuted_data  = reshaped_data.permute({0, 2, 1, 3, 4});
    return permuted_data;
}

// ========== NEW: helper to rebuild PCA basis tensor ==========
static torch::Tensor rebuild_pca_basis_tensor(const std::vector<std::vector<float>>& gb,
                                              torch::Device device)
{
    TORCH_CHECK(!gb.empty(), "global_pcaBasis is empty!");
    int64_t rows = (int64_t)gb.size();
    int64_t cols = (int64_t)gb[0].size();
    TORCH_CHECK(cols > 0, "global_pcaBasis has zero cols!");

    std::vector<float> flat;
    flat.reserve((size_t)rows * (size_t)cols);
    for (const auto& row : gb) {
        TORCH_CHECK((int64_t)row.size() == cols, "global_pcaBasis is ragged!");
        flat.insert(flat.end(), row.begin(), row.end());
    }

    torch::Tensor t = torch::from_blob(
        flat.data(),
        {rows, cols},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    return t.to(device);
}

torch::Tensor Decompressor::decompress(
    const std::vector<std::string>& encoded_latents,
    const std::vector<std::string>& encoded_hyper_latents,
    int batch_size,
    int n_frame,
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
    torch::Tensor offsets_tensor = torch::tensor(meta.offsets, opts);
    torch::Tensor scales_tensor  = torch::tensor(meta.scales,  opts);

    std::vector<int32_t> flat_indexes;
    flat_indexes.reserve(meta.indexes.size() * meta.indexes[0].size());
    for (const auto& v : meta.indexes)
        flat_indexes.insert(flat_indexes.end(), v.begin(), v.end());

    torch::TensorOptions idx_opts_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor indexes_tensor = torch::from_blob(
        flat_indexes.data(),
        {(long)meta.indexes.size(), (long)meta.indexes[0].size()},
        idx_opts_cpu
    ).clone().to(device_);

    flat_indexes.clear();
    flat_indexes.shrink_to_fit();

    std::vector<int64_t> input_shape(meta.data_input_shape.begin(), meta.data_input_shape.end());
    torch::Tensor recon_tensor = torch::zeros(input_shape).to(device_);
    input_shape.clear();
    input_shape.shrink_to_fit();

    torch::Tensor pcaBasis = rebuild_pca_basis_tensor(comp_result.gaeMetaData.global_pcaBasis, device_);

    double quan_factor = 2.0;
    std::string codec_alg = "Zstd";
    std::pair<int,int> patch_size = {8, 8};

    double rel_eb = 1e-3;

    PCACompressor pca_compressor(
        rel_eb,
        quan_factor,
        device_.is_cuda() ? "cuda" : "cpu",
        codec_alg,
        patch_size
    );

    int64_t gae_batch_id = 0;

    for (size_t lat_start = 0; lat_start < encoded_latents.size(); lat_start += (size_t)batch_size * 2) {
        size_t lat_end = std::min(lat_start + (size_t)batch_size * 2, encoded_latents.size());
        size_t cur_latents = lat_end - lat_start;
        TORCH_CHECK(cur_latents % 2 == 0, "cur_latents must be even.");
        size_t cur_samples = cur_latents / 2;
        size_t sample_start = lat_start / 2;

        std::vector<int32_t> hyper_size = {(int32_t)cur_latents, 64, 4, 4};
        torch::Tensor hyper_index_tensor = build_indexes_tensor(hyper_size).contiguous();
        torch::Tensor decoded_hyper_latents = torch::zeros({(long)cur_latents, 64, 4, 4}).to(torch::kInt32);

        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> hyper_index_vec =
                tensor_to_vector<int32_t>(hyper_index_tensor.select(0, (long)i).reshape(-1));

            std::vector<int32_t> hyper_decoded = range_decoder.decode_with_indexes(
                encoded_hyper_latents[lat_start + i],
                hyper_index_vec,
                vbr_quantized_cdf_,
                vbr_cdf_length_,
                vbr_offset_
            );

            torch::Tensor hyper_tensor = torch::tensor(hyper_decoded).reshape({64, 4, 4});
            decoded_hyper_latents.select(0, (long)i).copy_(hyper_tensor);
        }

        std::vector<torch::Tensor> hyper_outputs =
            hyper_decompressor_model_->run({decoded_hyper_latents.to(torch::kDouble).to(device_)});

        torch::Tensor mean = hyper_outputs[0].to(torch::kFloat32);
        torch::Tensor latent_indexes_recon = hyper_outputs[1].to(torch::kInt32);

        torch::Tensor decoded_latents_before_offset = torch::zeros({(long)cur_latents, 64, 16, 16}).to(torch::kInt32);

        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> latent_index =
                tensor_to_vector<int32_t>(latent_indexes_recon.select(0, (long)i).reshape(-1));

            std::vector<int32_t> latent_decoded = range_decoder.decode_with_indexes(
                encoded_latents[lat_start + i],
                latent_index,
                gs_quantized_cdf_,
                gs_cdf_length_,
                gs_offset_
            );

            torch::Tensor latent_tensor = torch::tensor(latent_decoded).reshape({64, 16, 16});
            decoded_latents_before_offset.select(0, (long)i).copy_(latent_tensor);
        }

        torch::Tensor q_latent_with_offset = decoded_latents_before_offset.to(torch::kFloat32).to(device_) + mean;

        // reshape to [-1,2,...] then run decompressor
        auto decoded_latents_sizes = q_latent_with_offset.sizes();
        std::vector<int64_t> new_shape = {-1, 2};
        new_shape.insert(new_shape.end(), decoded_latents_sizes.begin() + 1, decoded_latents_sizes.end());
        torch::Tensor reshaped_latents = q_latent_with_offset.reshape(new_shape);

        std::vector<torch::Tensor> decompressor_outputs = decompressor_model_->run({reshaped_latents.to(torch::kFloat32)});
        torch::Tensor raw_output = decompressor_outputs[0];

        torch::Tensor norm_output = reshape_batch_2d_3d(raw_output, (long)cur_samples, n_frame);

        if (gae_batch_id < (int64_t)comp_result.gae_batches.size()) {
            const auto& rec = comp_result.gae_batches[gae_batch_id];

            if (rec.correction_occur) {
                MetaData gae_md;
                CompressedData gae_cd;

                gae_md.pcaBasis     = pcaBasis; // already on device_
                gae_md.uniqueVals   = torch::tensor(rec.uniqueVals).to(device_);
                gae_md.quanBin      = rec.quanBin;
                gae_md.nVec         = rec.nVec;
                gae_md.prefixLength = rec.prefixLength;
                gae_md.dataBytes    = rec.dataBytes;

                gae_cd.data          = rec.comp_data;      // vector<uint8_t>
                gae_cd.dataBytes     = rec.dataBytes;
                gae_cd.coeffIntBytes = rec.coeffIntBytes;

                norm_output = pca_compressor.decompress(norm_output, gae_md, gae_cd);
            }
        }
        gae_batch_id++;

        torch::Tensor batched_offsets = offsets_tensor.narrow(0, (long)sample_start, (long)cur_samples)
            .view({-1, 1, 1, 1, 1});
        torch::Tensor batched_scales = scales_tensor.narrow(0, (long)sample_start, (long)cur_samples)
            .view({-1, 1, 1, 1, 1});
        torch::Tensor denorm_output = norm_output * batched_scales + batched_offsets;

        torch::Tensor indexes_cpu = indexes_tensor.narrow(0, (long)sample_start, (long)cur_samples).to(torch::kCPU);
        for (int64_t i = 0; i < (int64_t)cur_samples; ++i) {
            torch::Tensor index_row = indexes_cpu.select(0, i);
            int64_t idx0 = index_row[0].item<int64_t>();
            int64_t idx1 = index_row[1].item<int64_t>();
            int64_t start_t = index_row[2].item<int64_t>();
            int64_t end_t   = index_row[3].item<int64_t>();

            torch::Tensor source_slice_3d = denorm_output.select(0, i).squeeze(0);
            torch::Tensor dest_slice = recon_tensor.select(0, idx0).select(0, idx1).slice(0, start_t, end_t);
            dest_slice.copy_(source_slice_3d);
        }

        result.num_samples += cur_samples;
        result.num_batches++;
    }

    offsets_tensor = torch::Tensor();
    scales_tensor  = torch::Tensor();
    indexes_tensor = torch::Tensor();

    auto [b1_i32, b2_i32, pad_i32] = meta.block_info;

    int64_t block_info_1 = b1_i32;
    int64_t block_info_2 = b2_i32;
    std::vector<int64_t> block_info_3(pad_i32.begin(), pad_i32.end());

    torch::Tensor recon_tensor_deblock =
        deblockHW(recon_tensor, block_info_1, block_info_2, block_info_3);
    recon_tensor = torch::Tensor();

    torch::Tensor final_recon =
        recons_data(recon_tensor_deblock, meta.data_input_shape, meta.pad_T);
    return final_recon;

}