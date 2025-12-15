#include "caesar_compress.h"
#include "range_coder/rans_coder.hpp"
#include "runGaeCuda.h" 
#include "model_utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <unistd.h> // remove soon


// remove this soon
double rss_gb() {
    std::ifstream statm("/proc/self/statm");
    long dummy = 0, rss_pages = 0;
    statm >> dummy >> rss_pages;

    return (double)rss_pages * sysconf(_SC_PAGESIZE)
           / (1024.0 * 1024 * 1024);
}

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
std::vector<std::vector<T>> tensor_to_2d_vector(const torch::Tensor& tensor) {

    TORCH_CHECK(tensor.dim() == 2 , "Input tensor must be 2-dimensional.");

    torch::Tensor cpu_tensor = tensor.cpu().contiguous();

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
        torch::indexing::Slice(),       // 0번 차원 (:)
        torch::indexing::Slice(),       // 1번 차원 (:)
        torch::indexing::Slice(0, stop_t), // 2번 차원 (:stop_t)
        torch::indexing::Slice(),       // 3번 차원 (:)
        torch::indexing::Slice()        // 4번 차원 (:)
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

torch::Tensor Compressor::deblockHW(const torch::Tensor& data ,
    int64_t nH ,
    int64_t nW ,
    const std::vector<int64_t>& padding) {
    if (padding.size() != 4) {
        throw std::invalid_argument("padding must have 4 values: top, down, left, right");
    }

    auto sizes = data.sizes();
    if (sizes.size() != 5) {
        throw std::invalid_argument("Expected 5D input tensor (V, S_blk, T, h_block, w_block)");
    }

    int64_t V = sizes[0];
    int64_t sBlk = sizes[1];
    int64_t T = sizes[2];
    int64_t hBlock = sizes[3];
    int64_t wBlock = sizes[4];

    int64_t top = padding[0];
    int64_t down = padding[1];
    int64_t left = padding[2];
    int64_t right = padding[3];

    if (sBlk % (nH * nW) != 0) {
        throw std::invalid_argument("sBlk must be divisible by nH * nW");
    }
    int64_t sOrig = sBlk / (nH * nW);

    std::vector<int64_t> target_shape = { V, sOrig, nH, nW, T, hBlock, wBlock };
    auto reshaped = data.reshape(target_shape);

    auto permuted = reshaped.permute({ 0, 1, 4, 2, 5, 3, 6 });
    auto merged = permuted.reshape({ V, sOrig, T, nH * hBlock, nW * wBlock });

    int64_t hP = nH * hBlock;
    int64_t wP = nW * wBlock;
    int64_t H = hP - top - down;
    int64_t W = wP - left - right;

    auto result = merged.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(top, top + H),
        torch::indexing::Slice(left, left + W)
        });

    return result;
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
    data_reshaped.cpu(),
    torch::nn::functional::PadFuncOptions({left, right, top, down})
        .mode(torch::kReflect))
    .to(data_reshaped.device());  
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
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();
    const T* tensor_data_ptr = cpu_tensor.data_ptr<T>();
    int64_t num_elements = cpu_tensor.numel();
    return std::vector<T>(tensor_data_ptr , tensor_data_ptr + num_elements);
}

Compressor::Compressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Compressor::load_models() {
    compressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        get_model_file("caesar_compressor.pt2").string()
    );
    hyper_decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        get_model_file("caesar_hyper_decompressor.pt2").string()
    );
    decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        get_model_file("caesar_decompressor.pt2").string()
    );
}

void Compressor::load_probability_tables() {
    auto vbr_quantized_cdf_1d = load_array_from_bin<int32_t>(get_model_file("vbr_quantized_cdf.bin"));
    vbr_cdf_length_ = load_array_from_bin<int32_t>(get_model_file("vbr_cdf_length.bin"));
    vbr_offset_ = load_array_from_bin<int32_t>(get_model_file("vbr_offset.bin"));
    vbr_quantized_cdf_ = reshape_to_2d(vbr_quantized_cdf_1d , 64 , 63);

    auto gs_quantized_cdf_1d = load_array_from_bin<int32_t>(get_model_file("gs_quantized_cdf.bin"));
    gs_cdf_length_ = load_array_from_bin<int32_t>(get_model_file("gs_cdf_length.bin"));
    gs_offset_ = load_array_from_bin<int32_t>(get_model_file("gs_offset.bin"));
    gs_quantized_cdf_ = reshape_to_2d(gs_quantized_cdf_1d , 128 , 249);

}

CompressionResult Compressor::compress(const DatasetConfig& config , int batch_size , float rel_eb) {
    c10::InferenceMode guard;

    std::cout << "\n========== STARTING COMPRESSION ==========" << std::endl;
    std::cout << "Device: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    ScientificDataset dataset(config);

    CompressionResult result;
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
                static_cast<int32_t>(pair.first) , // int -> int32_t
                pair.second                      // float -> float
            );
        }
    }

    {
        auto block_info = dataset.get_block_info();
        const auto& padding_vec = std::get<2>(block_info);
        for (size_t i = 0; i < padding_vec.size(); ++i) {
            std::cout << padding_vec[i];
            if (i < padding_vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        int32_t nH_i32 = static_cast<int32_t>(std::get<0>(block_info));
        int32_t nW_i32 = static_cast<int32_t>(std::get<1>(block_info));
        const std::vector<int64_t>& padding_i64 = std::get<2>(block_info);
        std::vector<int32_t> padding_i32;
        padding_i32.reserve(padding_i64.size());

        for (int64_t pad_val : padding_i64) {
            padding_i32.push_back(static_cast<int32_t>(pad_val)); // int64 -> int32
        }
        result.compressionMetaData.block_info = std::make_tuple(nH_i32 , nW_i32 , padding_i32);
    }

    RansEncoder range_encoder;

    std::vector<torch::Tensor> batch_inputs;
    batch_inputs.reserve(batch_size);

    std::vector<float> batch_offsets_vec;
    batch_offsets_vec.reserve(batch_size);
    std::vector<float> batch_scales_vec;
    batch_scales_vec.reserve(batch_size);
    std::vector<torch::Tensor> batch_indexes;
    batch_indexes.reserve(batch_size);

    result.compressionMetaData.offsets.reserve(dataset.size());
    result.compressionMetaData.scales.reserve(dataset.size());
    result.compressionMetaData.indexes.reserve(dataset.size());

    const std::vector<int32_t>& shape_i32 = result.compressionMetaData.data_input_shape;
    std::vector<int64_t> input_shape(shape_i32.begin() , shape_i32.end());
    torch::Tensor recon_tensor = torch::zeros(input_shape , torch::TensorOptions().device(device_));


    float batch_max = 0.0;
    float batch_min = 1000000.0;

    for (size_t i = 0; i < dataset.size(); i++) {
        auto sample = dataset.get_item(i);
        torch::Tensor input_tensor = sample["input"];

        torch::Tensor offset_tensor = sample["offset"];
        torch::Tensor scale_tensor = sample["scale"];
        torch::Tensor index_tensor = sample["index"];


        batch_inputs.push_back(input_tensor);
        batch_offsets_vec.push_back(offset_tensor.item<float>());
        batch_scales_vec.push_back(scale_tensor.item<float>());
        batch_indexes.push_back(index_tensor.view({ 1, index_tensor.sizes()[0] }));
        result.compressionMetaData.offsets.push_back(offset_tensor.item<float>());
        result.compressionMetaData.scales.push_back(scale_tensor.item<float>());

        std::vector<int32_t> index_vec;
        index_vec.reserve(index_tensor.numel());
        const int64_t* index_data_ptr = index_tensor.data_ptr<int64_t>();

        for (int j = 0; j < index_tensor.numel(); ++j) {
            index_vec.push_back(static_cast<int32_t>(index_data_ptr[j]));
        }
        result.compressionMetaData.indexes.push_back(index_vec);

        float current_max = input_tensor.max().item<float>();
        float current_min = input_tensor.min().item<float>();
        if (current_max > batch_max) {
            batch_max = current_max;
        }
        if (current_min < batch_min) {
            batch_min = current_min;
        }

        if (batch_inputs.size() == static_cast<size_t>(batch_size) || i == dataset.size() - 1) {
            torch::Tensor batched_input = torch::cat(batch_inputs , 0).to(device_);
            torch::Tensor batched_offsets = torch::tensor(batch_offsets_vec).view({ -1, 1, 1, 1, 1 }).to(device_);
            torch::Tensor batched_scales = torch::tensor(batch_scales_vec).view({ -1, 1, 1, 1, 1 }).to(device_);
            torch::Tensor batched_indexes = torch::cat(batch_indexes , 0).to(device_);


            std::vector<torch::Tensor> inputs = { batched_input.to(torch::kDouble) };
         
            std::vector<torch::Tensor> outputs = compressor_model_->run(inputs);
        
            torch::Tensor q_latent = outputs[0].to(torch::kInt32);
            torch::Tensor latent_indexes = outputs[1].to(torch::kInt32);
            torch::Tensor q_hyper_latent = outputs[2].to(torch::kInt32);
            torch::Tensor hyper_indexes = outputs[3].to(torch::kInt32);
            outputs.clear();
            outputs.shrink_to_fit();

            int64_t num_input_samples = batch_inputs.size();
            int64_t num_latent_codes = q_latent.sizes()[0];

            for (int64_t j = 0; j < num_latent_codes; j++) {
                std::vector<int32_t> latent_symbol = tensor_to_vector<int32_t>(
                    q_latent.select(0 , j).reshape(-1)
                );
                std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                    latent_indexes.select(0 , j).reshape(-1)
                );
                std::vector<int32_t> hyper_symbol = tensor_to_vector<int32_t>(
                    q_hyper_latent.select(0 , j).reshape(-1)
                );
                std::vector<int32_t> hyper_index = tensor_to_vector<int32_t>(
                    hyper_indexes.select(0 , j).reshape(-1)
                );

                std::string latent_encoded = range_encoder.encode_with_indexes(
                    latent_symbol , latent_index ,
                    gs_quantized_cdf_ , gs_cdf_length_ , gs_offset_
                );

                std::string hyper_encoded = range_encoder.encode_with_indexes(
                    hyper_symbol , hyper_index ,
                    vbr_quantized_cdf_ , vbr_cdf_length_ , vbr_offset_
                );

                result.encoded_latents.push_back(latent_encoded);
                result.encoded_hyper_latents.push_back(hyper_encoded);

                if (j < 3 || j == num_latent_codes - 1) {
                    std::cout << "  Encoded latent " << j << ": "
                        << latent_encoded.size() << " bytes (latent), "
                        << hyper_encoded.size() << " bytes (hyper)" << std::endl;
                }
            }

            result.num_samples += num_input_samples;
            batch_inputs.clear();
            batch_inputs.shrink_to_fit();

            batch_max = 0.0;
            batch_min = 1000000.0;

            std::vector<torch::Tensor> hyper_outputs = hyper_decompressor_model_->run({ q_hyper_latent.to(torch::kDouble) });


          
            torch::Tensor mean = hyper_outputs[0].to(torch::kFloat32);
            torch::Tensor latent_indexes_recon = hyper_outputs[1].to(torch::kInt32);
            hyper_outputs.clear();
            hyper_outputs.shrink_to_fit();
            torch::Tensor q_latent_with_offset = q_latent.to(torch::kFloat32) + mean;
            auto decoded_latents_sizes = q_latent_with_offset.sizes();

            std::vector<int64_t> new_shape = { -1, 2 };
            new_shape.insert(new_shape.end() , decoded_latents_sizes.begin() + 1 , decoded_latents_sizes.end());


            torch::Tensor reshaped_latents = q_latent_with_offset.reshape(new_shape);
            std::vector<torch::Tensor> decompressor_outputs = decompressor_model_->run({ reshaped_latents });

            torch::Tensor raw_output = decompressor_outputs[0];
            torch::Tensor norm_output = reshape_batch_2d_3d(
                raw_output ,
                num_input_samples
            );

            // why is this to cpu ??????????
            torch::Tensor denorm_output = norm_output * batched_scales + batched_offsets;
            torch::Tensor indexes_cpu = batched_indexes.to(torch::kCPU);
            


         
            for (int64_t i = 0; i < num_input_samples; ++i) {

                torch::Tensor index_row = indexes_cpu.select(0 , i);
                int64_t idx0 = index_row[0].item<int64_t>();
                int64_t idx1 = index_row[1].item<int64_t>();
                int64_t start_t = index_row[2].item<int64_t>();
                int64_t end_t = index_row[3].item<int64_t>();
                
                torch::Tensor source_slice_3d = denorm_output.select(0, i).squeeze(0).cpu();
                torch::Tensor dest_slice = recon_tensor.select(0 , idx0).select(0 , idx1).slice(0 , start_t , end_t);

                dest_slice.copy_(source_slice_3d);
            }

            batch_offsets_vec.clear();
            batch_scales_vec.clear();
            batch_indexes.clear();
            result.num_batches++;
        }
    }
   

    if (!result.compressionMetaData.filtered_blocks.empty()) {
        const int64_t V = static_cast<int64_t>(result.compressionMetaData.data_input_shape[0]);
        const int64_t S = static_cast<int64_t>(result.compressionMetaData.data_input_shape[1]);
        const int64_t T = static_cast<int64_t>(result.compressionMetaData.data_input_shape[2]);

        const int64_t n_frame_filtered = 16;
        const int64_t samples = T / n_frame_filtered;
      
        if (samples == 0 || S == 0) {
            std::cerr << "Error: 'samples' or 'S' is zero, cannot calculate indexes." << std::endl;
        }
        else {
            const int64_t S_times_samples = S * samples;

            for (const auto& pair : result.compressionMetaData.filtered_blocks) {
                const int32_t label = pair.first;
                const float value = pair.second;

                const int64_t v = label / S_times_samples;
                const int64_t remain = label % S_times_samples;
                const int64_t s = remain / samples;
                const int64_t blk_idx = remain % samples;
                const int64_t start_t = blk_idx * n_frame_filtered;
                const int64_t end_t = (blk_idx + 1) * n_frame_filtered;

                
                torch::Tensor dest_slice = recon_tensor
                    .select(0 , v)     // shape: [S, T, H, W]
                    .select(0 , s)     // shape: [T, H, W]
                    .slice(0 , start_t , end_t); // shape: [n_frame_filtered, H, W]

                dest_slice.fill_(value);

            }
        }
    }

    int64_t block_info_1 = static_cast<int64_t>(std::get<0>(result.compressionMetaData.block_info));
    int64_t block_info_2 = static_cast<int64_t>(std::get<1>(result.compressionMetaData.block_info));
    std::vector<int64_t> block_info_3;
    const std::vector<int32_t>& padding_vec_i32 = std::get<2>(result.compressionMetaData.block_info);
    block_info_3.reserve(padding_vec_i32.size());


  
    for (int32_t val : padding_vec_i32) {
        block_info_3.push_back(static_cast<int64_t>(val));
    }
 

    torch::Tensor recon_tensor_deblock = deblockHW(recon_tensor , block_info_1 , block_info_2 , block_info_3);
    std::cout<< "[MEM] before padding "<< rss_gb()  <<" Gib\n";
    std::tuple<torch::Tensor , std::vector<int>> padding_original = padding(dataset.original_data());
    std::cout<< "[MEM] first padded og " << rss_gb() <<" Gib\n";
    std::tuple<torch::Tensor , std::vector<int>> padding_recon = padding(recon_tensor_deblock);
    std::cout << "[MEM] second padded padding_rec"<< rss_gb() <<" GiB\n";
    
    recon_tensor = torch::Tensor();
    recon_tensor_deblock = torch::Tensor();
    std::cout<<"memory after freeing some up "<<rss_gb()<< " GiB\n";
    torch::Tensor padded_original_tensor = std::get<0>(padding_original);
    torch::Tensor padded_recon_tensor = std::get<0>(padding_recon);
    std::vector<int> padding_recon_info = std::get<1>(padding_recon);
    
    result.gaeMetaData.padding_recon_info = padding_recon_info;

    float global_scale = padded_original_tensor.max().item<float>() - padded_original_tensor.min().item<float>();
    float global_offset = padded_original_tensor.mean().item<float>();
    result.compressionMetaData.global_scale = global_scale;
    result.compressionMetaData.global_offset = global_offset;

    torch::Tensor padded_original_tensor_norm = (padded_original_tensor - global_offset) / global_scale;
    torch::Tensor padded_recon_tensor_norm = (padded_recon_tensor - global_offset) / global_scale;
    std::cout<<"Done with it padding "<< rss_gb() <<" Gib\n";
    double quan_factor = 2.0;

    std::string codec_alg = "Zstd";
    std::pair<int , int> patch_size = { 8, 8 };



    std::cout<<"[MEM] before init pca compressor"<<rss_gb() <<" Gib\n";
    PCACompressor pca_compressor(rel_eb ,
        quan_factor ,
        device_.is_cuda() ? "cuda" : "cpu" ,
        codec_alg ,
        patch_size);
    std::cout<<"[MEM] after init pca compressor "<<rss_gb() <<" GiB\n";

    auto gae_compression_result = pca_compressor.compress(padded_original_tensor_norm.to(device_) , padded_recon_tensor_norm.to(device_));
    std::cout << "[MEM] after pca_compress run = " << rss_gb() << " GiB\n";
    result.gaeMetaData.GAE_correction_occur = gae_compression_result.metaData.GAE_correction_occur;

    MetaData gae_record_metaData;
    CompressedData gae_record_compressedData;

    result.gaeMetaData.quanBin = gae_compression_result.metaData.quanBin;
    result.gaeMetaData.nVec = gae_compression_result.metaData.nVec;
    result.gaeMetaData.prefixLength = gae_compression_result.metaData.prefixLength;
    result.gaeMetaData.dataBytes = gae_compression_result.metaData.dataBytes;
    result.gaeMetaData.coeffIntBytes = gae_compression_result.compressedData->coeffIntBytes;
    result.gae_comp_data = gae_compression_result.compressedData->data;

    if (result.gaeMetaData.GAE_correction_occur) {
        result.gaeMetaData.pcaBasis = tensor_to_2d_vector<float>(gae_compression_result.metaData.pcaBasis);
        result.gaeMetaData.uniqueVals = tensor_to_vector<float>(gae_compression_result.metaData.uniqueVals);

        int64_t pca_rows = result.gaeMetaData.pcaBasis.size();
        int64_t pca_cols = result.gaeMetaData.pcaBasis[0].size();

        std::vector<float> pca_vec;
        pca_vec.reserve(pca_rows * pca_cols);

        for (const auto& row_vec : result.gaeMetaData.pcaBasis) {
            pca_vec.insert(pca_vec.end() , row_vec.begin() , row_vec.end());
        }
        torch::Tensor pca_vec_1d = torch::tensor(pca_vec);
        torch::Tensor pcaBasis = pca_vec_1d.reshape({ pca_rows, pca_cols });

        gae_record_metaData.pcaBasis = pcaBasis.to(device_);
        gae_record_metaData.uniqueVals = torch::tensor(result.gaeMetaData.uniqueVals).to(device_);
        gae_record_metaData.quanBin = result.gaeMetaData.quanBin;
        gae_record_metaData.nVec = result.gaeMetaData.nVec;
        gae_record_metaData.prefixLength = result.gaeMetaData.prefixLength;
        gae_record_metaData.dataBytes = result.gaeMetaData.dataBytes;

        gae_record_compressedData.data = result.gae_comp_data;
        gae_record_compressedData.dataBytes = result.gaeMetaData.dataBytes;
        gae_record_compressedData.coeffIntBytes = result.gaeMetaData.coeffIntBytes;

        torch::Tensor recons_gae = pca_compressor.decompress(padded_recon_tensor_norm ,
            gae_record_metaData ,
            gae_record_compressedData);

 
    }

    else {
        std::cout << "[GAE SKIPPED] No data processed by GAE." << std::endl;
        // 'result.gaeMetaData.pcaBasis' and 'uniqueVals' are already emtpy. Skip record these.

    }

    if (!result.compressionMetaData.indexes.empty()) {
        std::cout << ", " << result.compressionMetaData.indexes[0].size(); // 0번째 안쪽 벡터의 크기(4)
    }

    return result;
}
