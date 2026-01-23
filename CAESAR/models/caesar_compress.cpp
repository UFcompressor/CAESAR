#include "caesar_compress.h"
#include "range_coder/rans_coder.hpp"
#include "runGaeCuda.h" 
#include "model_utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>


template<typename T>
std::vector<std::vector<T>> tensor_to_2d_vector(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 2 , "Input tensor must be 2-dimensional.");

    torch::Tensor cpu_tensor = tensor.is_cpu() ? tensor.contiguous() : tensor.cpu().contiguous();
    const int64_t rows = cpu_tensor.size(0);
    const int64_t cols = cpu_tensor.size(1);
    const T* data_ptr = cpu_tensor.data_ptr<T>();

    std::vector<std::vector<T>> vec_2d;
    vec_2d.reserve(rows);

    for (int64_t r = 0; r < rows; ++r) {
        const T* row_start_ptr = data_ptr + (r * cols);
        std::vector<T> inner_vec(row_start_ptr , row_start_ptr + cols);
        vec_2d.push_back(inner_vec);
    }
    return vec_2d;
}

torch::Tensor Compressor::recons_data(const torch::Tensor& recons_data , std::vector<int32_t> shape , int64_t pad_T) const {
    int64_t stop_t = shape[2] - pad_T;
    return recons_data.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(0, stop_t),
        torch::indexing::Slice(),
        torch::indexing::Slice()
        });
}

torch::Tensor Compressor::reshape_batch_2d_3d(const torch::Tensor& batch_data , int64_t batch_size) {
    auto sizes = batch_data.sizes();
    TORCH_CHECK(sizes.size() == 4 , "Input tensor must be 4-dimensional.");

    int64_t BT = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    int64_t T = BT / batch_size;
    torch::Tensor reshaped_data = batch_data.view({ batch_size, T, C, H, W });
    torch::Tensor permuted_data = reshaped_data.permute({ 0, 2, 1, 3, 4 });

    return permuted_data;
}

std::tuple<torch::Tensor , std::vector<int>> padding(
    const torch::Tensor& data ,
    std::pair<int , int> block_size = { 8, 8 })
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

    int top = dh / 2;
    int down = dh - top;
    int left = dw / 2;
    int right = dw - left;

    auto leading_dims = data.sizes().vec();
    int leading_size = 1;
    for (size_t i = 0; i < leading_dims.size() - 2; ++i)
        leading_size *= leading_dims[i];
    auto data_reshaped = data.view({ leading_size, H, W });
    auto data_padded = torch::nn::functional::pad(
        data_reshaped ,
        torch::nn::functional::PadFuncOptions({ left, right, top, down })
        .mode(torch::kConstant)
        .value(0));
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

template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.is_cpu() ? tensor.contiguous() : tensor.cpu().contiguous();
    const T* tensor_data_ptr = cpu_tensor.data_ptr<T>();
    int64_t num_elements = cpu_tensor.numel();
    return std::vector<T>(tensor_data_ptr , tensor_data_ptr + num_elements);
}

Compressor::Compressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Compressor::load_models() {
    compressor_model_ = ModelCache::instance().get_compressor_model();
    hyper_decompressor_model_ = ModelCache::instance().get_hyper_decompressor_model();
    decompressor_model_ = ModelCache::instance().get_decompressor_model();
}

void Compressor::load_probability_tables() {
    vbr_quantized_cdf_ = ModelCache::instance().get_vbr_quantized_cdf();
    vbr_cdf_length_ = ModelCache::instance().get_vbr_cdf_length();
    vbr_offset_ = ModelCache::instance().get_vbr_offset();
    gs_quantized_cdf_ = ModelCache::instance().get_gs_quantized_cdf();
    gs_cdf_length_ = ModelCache::instance().get_gs_cdf_length();
    gs_offset_ = ModelCache::instance().get_gs_offset();
}

CompressionResult Compressor::compress(const DatasetConfig& config , int batch_size , float rel_eb) {
    c10::InferenceMode guard;

    std::cout << "\n========== STARTING COMPRESSION ==========" << std::endl;
    std::cout << "Device: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    ScientificDataset dataset(config);
    std::cout << "[MEM] dataset loaded " << rss_gb() << " GiB\n";
    
    auto raw = dataset.raw_data();
    std::tuple<torch::Tensor, std::vector<int>> padding_raw = padding(raw);
    raw.reset();
    auto padded_raw = std::get<0>(padding_raw);
    auto stats_raw = torch::stack({
        padded_raw.max(),
        padded_raw.min(),
        padded_raw.mean()
    }).cpu();

    float scale  = stats_raw[0].item<float>() - stats_raw[1].item<float>();
    float offset = stats_raw[2].item<float>();
    if (scale == 0.0f) scale = 1.0f;
    stats_raw = torch::Tensor();
    torch::Tensor padded_raw_norm = (padded_raw - offset) / scale;
    padded_raw = torch::Tensor();
    auto raw_vectors = block2Vector(padded_raw_norm, {8, 8});
    padded_raw_norm = torch::Tensor();
    PCA pca(-1, device_.is_cuda() ? "cuda" : "cpu");
    pca.fit(raw_vectors);
    
    raw_vectors = torch::Tensor();
    auto basis = pca.components();
    
    // std::cout << "[LJ] PCA basis shape = " << basis.sizes() << std::endl;
    
    auto start_inf = get_start_time();
    CompressionResult result;
    result.gaeMetaData.global_pcaBasis = tensor_to_2d_vector<float>(basis);
    
    result.num_samples = 0;
    result.num_batches = 0;

    int64_t pad_T = dataset.get_pad_T();
    result.compressionMetaData.pad_T = pad_T;

    {
        const auto& data_input_shape = dataset.get_data_input().sizes();
        std::vector<int32_t> data_input_shape_i32;
        data_input_shape_i32.reserve(data_input_shape.size());

        for (int64_t dim : data_input_shape) {
            data_input_shape_i32.push_back(static_cast<int32_t>(dim));
        }
        result.compressionMetaData.data_input_shape = data_input_shape_i32;
    }

    {
        const auto& filtered_blocks = dataset.get_filtered_blocks();
        result.compressionMetaData.filtered_blocks.reserve(filtered_blocks.size());
        for (const auto& pair : filtered_blocks) {
            result.compressionMetaData.filtered_blocks.emplace_back(
                static_cast<int32_t>(pair.first) ,
                pair.second
            );
        }
    }

    {
        auto block_info = dataset.get_block_info();
        const auto& padding_vec = std::get<2>(block_info);
        int32_t nH_i32 = static_cast<int32_t>(std::get<0>(block_info));
        int32_t nW_i32 = static_cast<int32_t>(std::get<1>(block_info));
        const std::vector<int64_t>& padding_i64 = std::get<2>(block_info);
        std::vector<int32_t> padding_i32;
        padding_i32.reserve(padding_i64.size());

        for (int64_t pad_val : padding_i64) {
            padding_i32.push_back(static_cast<int32_t>(pad_val));
        }
        result.compressionMetaData.block_info = std::make_tuple(nH_i32 , nW_i32 , padding_i32);
    }

    RansEncoder range_encoder;

    std::vector<torch::Tensor> batch_inputs;
    batch_inputs.reserve(batch_size);

    result.compressionMetaData.offsets.reserve(dataset.size());
    result.compressionMetaData.scales.reserve(dataset.size());
    result.compressionMetaData.indexes.reserve(dataset.size());
    
    double quan_factor = 2.0;
    std::string codec_alg = "Zstd";
    std::pair<int , int> patch_size = { 8, 8 };
    
    PCACompressor pca_compressor(rel_eb ,
        quan_factor ,
        device_.is_cuda() ? "cuda" : "cpu" ,
        codec_alg ,
        patch_size);
    
    for (size_t i = 0; i < dataset.size(); ++i) {

        auto sample = dataset.get_item(i);

        torch::Tensor input_tensor  = sample["input"].to(device_);   // [1, 1, Tblk, H, W] or similar
        torch::Tensor offset_tensor = sample["offset"];              // scalar
        torch::Tensor scale_tensor  = sample["scale"];               // scalar
        torch::Tensor index_tensor  = sample["index"];               // [4] (idx0, idx1, start_t, end_t)

        batch_inputs.push_back(input_tensor);

        // ---- save per-block meta (needed for de-normalization in decompression) ----
        result.compressionMetaData.offsets.push_back(offset_tensor.item<float>());
        result.compressionMetaData.scales.push_back(scale_tensor.item<float>());

        // save index row as int32 vector
        std::vector<int32_t> index_vec;
        index_vec.reserve(index_tensor.numel());
        const int64_t* index_data_ptr = index_tensor.data_ptr<int64_t>();
        for (int j = 0; j < index_tensor.numel(); ++j) {
            index_vec.push_back(static_cast<int32_t>(index_data_ptr[j]));
        }
        result.compressionMetaData.indexes.push_back(std::move(index_vec));


        if (batch_inputs.size() == static_cast<size_t>(batch_size) || i == dataset.size() - 1) {
            const int64_t num_input_samples = static_cast<int64_t>(batch_inputs.size());

            // concat batch: [B, 1, Tblk, H, W] (B = num_input_samples)
            torch::Tensor batched_input = torch::cat(batch_inputs, 0);

            std::vector<torch::Tensor> inputs = { batched_input.to(torch::kFloat16) };
            std::vector<torch::Tensor> outputs = compressor_model_->run(inputs);

            torch::Tensor latent        = outputs[0];
            torch::Tensor q_hyper_latent= outputs[1];
            torch::Tensor hyper_indexes = outputs[2];
            outputs.clear();
            outputs.shrink_to_fit();

            std::vector<torch::Tensor> hyper_outputs =
                hyper_decompressor_model_->run({ q_hyper_latent.to(torch::kDouble) });

            torch::Tensor mean               = hyper_outputs[0].to(torch::kFloat32);
            torch::Tensor latent_indexes_recon = hyper_outputs[1].to(torch::kFloat32);

            hyper_outputs.clear();
            hyper_outputs.shrink_to_fit();

            // quantize latent (int32 symbols)
            torch::Tensor q_latent = (latent - mean).to(torch::kInt32);

            // cast indexes / hyper to int32
            torch::Tensor latent_indexes_int32  = latent_indexes_recon.to(torch::kInt32);
            torch::Tensor q_hyper_latent_int32  = q_hyper_latent.to(torch::kInt32);
            torch::Tensor hyper_indexes_int32   = hyper_indexes.to(torch::kInt32);

            // move to CPU for rANS
            torch::Tensor q_latent_cpu          = q_latent.cpu();
            torch::Tensor latent_indexes_cpu    = latent_indexes_int32.cpu();
            torch::Tensor q_hyper_latent_cpu    = q_hyper_latent_int32.cpu();
            torch::Tensor hyper_indexes_cpu     = hyper_indexes_int32.cpu();

            const int64_t num_latent_codes = q_latent.sizes()[0]; // typically 2*B

            std::vector<int32_t> latent_symbol_buffer;
            std::vector<int32_t> latent_index_buffer;
            std::vector<int32_t> hyper_symbol_buffer;
            std::vector<int32_t> hyper_index_buffer;

            for (int64_t j = 0; j < num_latent_codes; ++j) {
                latent_symbol_buffer = tensor_to_vector<int32_t>(q_latent_cpu.select(0, j).reshape(-1));
                latent_index_buffer  = tensor_to_vector<int32_t>(latent_indexes_cpu.select(0, j).reshape(-1));

                hyper_symbol_buffer  = tensor_to_vector<int32_t>(q_hyper_latent_cpu.select(0, j).reshape(-1));
                hyper_index_buffer   = tensor_to_vector<int32_t>(hyper_indexes_cpu.select(0, j).reshape(-1));

                std::string latent_encoded = range_encoder.encode_with_indexes(
                    latent_symbol_buffer, latent_index_buffer,
                    gs_quantized_cdf_, gs_cdf_length_, gs_offset_
                );

                std::string hyper_encoded = range_encoder.encode_with_indexes(
                    hyper_symbol_buffer, hyper_index_buffer,
                    vbr_quantized_cdf_, vbr_cdf_length_, vbr_offset_
                );

                result.encoded_latents.push_back(std::move(latent_encoded));
                result.encoded_hyper_latents.push_back(std::move(hyper_encoded));
            }

            result.num_samples += num_input_samples;

            torch::Tensor q_latent_with_offset = q_latent.to(torch::kFloat32) + mean;
            auto decoded_latents_sizes = q_latent_with_offset.sizes();

            std::vector<int64_t> new_shape = { -1, 2 };
            new_shape.insert(new_shape.end(), decoded_latents_sizes.begin() + 1, decoded_latents_sizes.end());

            torch::Tensor reshaped_latents = q_latent_with_offset.reshape(new_shape);
            std::vector<torch::Tensor> decompressor_outputs = decompressor_model_->run({ reshaped_latents });

            torch::Tensor raw_output  = decompressor_outputs[0];
            torch::Tensor norm_output = reshape_batch_2d_3d(raw_output, num_input_samples);

            auto gae_res = pca_compressor.compress(batched_input, norm_output, basis);

            GaeBatchRecord rec;
            rec.correction_occur = gae_res.metaData.GAE_correction_occur;
            rec.batch_n = static_cast<int32_t>(num_input_samples);  

            if (rec.correction_occur) {
                rec.quanBin       = gae_res.metaData.quanBin;
                rec.nVec          = gae_res.metaData.nVec;
                rec.prefixLength  = gae_res.metaData.prefixLength;
                rec.dataBytes     = gae_res.metaData.dataBytes;
                rec.coeffIntBytes = gae_res.compressedData->coeffIntBytes;
                rec.comp_data     = gae_res.compressedData->data;
                rec.uniqueVals    = tensor_to_vector<float>(gae_res.metaData.uniqueVals);
            }
            result.gae_batches.push_back(std::move(rec));

            for (auto &t : batch_inputs) {t = torch::Tensor();}
            batch_inputs.clear();
            result.num_batches++;
        }
    }
    return result;
}
