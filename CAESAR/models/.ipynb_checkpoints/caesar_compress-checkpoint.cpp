 #include "caesar_compress.h"
#include "range_coder/rans_coder.hpp"
#include <iostream>
#include <fstream>

// Helper functions
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

// ** JL modified ** //
torch::Tensor Compressor::reshape_batch_2d_3d(const torch::Tensor& batch_data, int64_t batch_size) {
    
    // 1. BT,C,H,W = batch_data.shape
    auto sizes = batch_data.sizes();

    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4-dimensional.");

    int64_t BT = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    // 2. T = BT//batch_size
    int64_t T = BT / batch_size;

    // 3. batch_data = batch_data.view([batch_size, T, C, H, W])
    torch::Tensor reshaped_data = batch_data.view({batch_size, T, C, H, W});

    // 4. batch_data = batch_data.permute([0,2,1,3,4])
    torch::Tensor permuted_data = reshaped_data.permute({0, 2, 1, 3, 4});

    // 5. return batch_data
    return permuted_data;
}
// **** //

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
    std::cout << "Loading compressor model..." << std::endl;
    compressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/caesar_compressor.pt2"
    );
    // ** JL modified ** //
    std::cout << "Loading decompressor models..." << std::endl;
    hyper_decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/caesar_hyper_decompressor.pt2"
    );
    decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/caesar_decompressor.pt2"
    );
    // **** //
    std::cout << "Model loaded successfully." << std::endl;
}

void Compressor::load_probability_tables() {
    std::cout << "Loading probability tables..." << std::endl;

    // Load VBR tables
    auto vbr_quantized_cdf_1d = load_array_from_bin<int32_t>("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/vbr_quantized_cdf.bin");
    vbr_cdf_length_ = load_array_from_bin<int32_t>("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/vbr_cdf_length.bin");
    vbr_offset_ = load_array_from_bin<int32_t>("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/vbr_offset.bin");
    vbr_quantized_cdf_ = reshape_to_2d(vbr_quantized_cdf_1d , 64 , 63);

    // Load GS tables
    auto gs_quantized_cdf_1d = load_array_from_bin<int32_t>("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/gs_quantized_cdf.bin");
    gs_cdf_length_ = load_array_from_bin<int32_t>("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/gs_cdf_length.bin");
    gs_offset_ = load_array_from_bin<int32_t>("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/exported_model/gs_offset.bin");
    gs_quantized_cdf_ = reshape_to_2d(gs_quantized_cdf_1d , 128 , 249);

    std::cout << "Probability tables loaded successfully." << std::endl;
}

CompressionResult Compressor::compress(const DatasetConfig& config , int batch_size) {
    c10::InferenceMode guard;

    std::cout << "\n========== STARTING COMPRESSION ==========" << std::endl;
    std::cout << "Device: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "N_frame: " << config.n_frame << std::endl;

    // Load dataset
    ScientificDataset dataset(config);
    std::cout << "Dataset loaded. Total samples: " << dataset.size() << std::endl;

    // ** JL modified ** //
    // Load all the necessary data   

    // 1. dataset.original_data()
    torch::Tensor original_data = dataset.original_data();
    std::cout << "[METADATA CHECK] Original data shape: " << original_data.sizes() << std::endl;

    // 2. dataset.input_data()
    torch::Tensor input_data = dataset.input_data();
    std::cout << "[METADATA CHECK] Input data (unpadded) shape: " << input_data.sizes() << std::endl;
    
    // 3. dataset metadata check
    std::cout << "[METADATA CHECK] original data size: " << dataset.original_data().sizes() << std::endl;
    std::cout << "[METADATA CHECK] input data size: " << dataset.input_data().sizes() << std::endl;
    // **** //

    // Initialize result
    CompressionResult result;
    result.num_samples = 0;
    result.num_batches = 0;
    
    // ** JL modified ** // - record metadata
    // int64 -> int32 to reduce size
    // data_input_shape recored
    {
        const auto& data_input_shape = dataset.get_data_input().sizes();
        std::cout << "[METADATA CHECK] Data input shape: " << data_input_shape << std::endl;
        std::vector<int32_t> data_input_shape_i32;
        data_input_shape_i32.reserve(data_input_shape.size());
        
        for (int64_t dim : data_input_shape) {
            data_input_shape_i32.push_back(static_cast<int32_t>(dim));
        }
        result.data_input_shape = data_input_shape_i32;
    }
    
    // filtered_blocks record
    {
        const auto& filtered_blocks = dataset.get_filtered_blocks();
        std::cout << "[METADATA CHECK] Filtered blocks count: " << filtered_blocks.size() << std::endl;
        result.filtered_blocks.reserve(filtered_blocks.size());
        for (const auto& pair : filtered_blocks) {
            result.filtered_blocks.emplace_back(
                static_cast<int32_t>(pair.first), // int -> int32_t
                pair.second                      // float -> float
                );
        }
    }

    // block_info record
    {
        auto block_info = dataset.get_block_info();
        std::cout << "[METADATA CHECK] Block nH: " << std::get<0>(block_info) << std::endl;
        std::cout << "[METADATA CHECK] Block nW: " << std::get<1>(block_info) << std::endl;
        std::cout << "[METADATA CHECK] Padding: [";
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
        result.block_info = std::make_tuple(nH_i32, nW_i32, padding_i32);
    }
    // **** //
        
    // Initialize encoder
    RansEncoder range_encoder;

    // Batch processing
    std::vector<torch::Tensor> batch_inputs;
    batch_inputs.reserve(batch_size);

    // ** JL modified ** //
    std::vector<float> batch_offsets_vec;
    batch_offsets_vec.reserve(batch_size);
    std::vector<float> batch_scales_vec;
    batch_scales_vec.reserve(batch_size);
    std::vector<torch::Tensor> batch_indexes;
    batch_indexes.reserve(batch_size);
    
    result.offsets.reserve(dataset.size());
    result.scales.reserve(dataset.size());
    result.indexes.reserve(dataset.size());

    const std::vector<int32_t>& shape_i32 = result.data_input_shape;
    std::vector<int64_t> input_shape(shape_i32.begin(), shape_i32.end());
    torch::Tensor recon_tensor = torch::zeros(input_shape).to(device_);
    std::cout << "\nrecon_tensor shape: " << recon_tensor.sizes() << std::endl;
    float batch_max = 0.0;
    float batch_min = 1000000.0;
    // **** //
    
    for (size_t i = 0; i < dataset.size(); i++) {
        auto sample = dataset.get_item(i);
        torch::Tensor input_tensor = sample["input"];
        // ** JL modified ** //        
        torch::Tensor offset_tensor = sample["offset"];
        torch::Tensor scale_tensor = sample["scale"];
        torch::Tensor index_tensor = sample["index"];
        // **** //

        if (i == 0) {
            std::cout << "\n*** FIRST SAMPLE CHECK ***" << std::endl;
            std::cout << "Input sample shape: " << input_tensor.sizes() << std::endl;
            // ** JL modified ** //            
            std::cout << "Offset sample shape: " << offset_tensor.sizes() << std::endl;
            std::cout << "Scale sample shape: " << scale_tensor.sizes() << std::endl;
            std::cout << "Index sample shape: " << index_tensor.sizes() << std::endl;            
            // **** //
        }

        batch_inputs.push_back(input_tensor);
        // ** JL modified ** //
        batch_offsets_vec.push_back(offset_tensor.item<float>());
        batch_scales_vec.push_back(scale_tensor.item<float>());
        batch_indexes.push_back(index_tensor.view({ 1, index_tensor.sizes()[0] }));
        // Metadata record
        result.offsets.push_back(offset_tensor.item<float>());
        result.scales.push_back(scale_tensor.item<float>());
        
        std::vector<int32_t> index_vec;
        index_vec.reserve(index_tensor.numel());
        const int64_t* index_data_ptr = index_tensor.data_ptr<int64_t>();
        
        for (int j = 0; j < index_tensor.numel(); ++j) {
            index_vec.push_back(static_cast<int32_t>(index_data_ptr[j]));
        }
        result.indexes.push_back(index_vec);
        
        float current_max = input_tensor.max().item<float>();
        float current_min = input_tensor.min().item<float>();
        if (current_max > batch_max) {
            batch_max = current_max;
        }
        if (current_min < batch_min) {
            batch_min = current_min;
        }
        // **** //

        if (batch_inputs.size() == static_cast<size_t>(batch_size) || i == dataset.size() - 1) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "--- Processing Batch " << result.num_batches << " ---" << std::endl;
            std::cout << "Input samples in batch: " << batch_inputs.size() << std::endl;
            // ** JL modified ** //
            std::cout << "Offset samples in batch: " << batch_offsets_vec.size() << std::endl;
            std::cout << "Scale samples in batch: " << batch_scales_vec.size() << std::endl;
            std::cout << "Index samples in batch: " << batch_indexes.size() << std::endl;
            // **** //
            std::cout << "========================================" << std::endl;

            // Concatenate batch
            torch::Tensor batched_input = torch::cat(batch_inputs , 0).to(device_);
            std::cout << "[COMPRESS] Batched input shape: " << batched_input.sizes() << std::endl;
            // ** JL modified ** //
            torch::Tensor batched_offsets = torch::tensor(batch_offsets_vec).view({-1, 1, 1, 1, 1}).to(device_);
            //torch::Tensor batched_offsets = torch::cat(batch_offsets , 0).to(device_);
            std::cout << "[COMPRESS] Batched offset shape: " << batched_offsets.sizes() << std::endl;
            torch::Tensor batched_scales = torch::tensor(batch_scales_vec).view({-1, 1, 1, 1, 1}).to(device_);
            //torch::Tensor batched_scales = torch::cat(batch_scales , 0).to(device_);
            std::cout << "[COMPRESS] Batched scale shape: " << batched_scales.sizes() << std::endl;
            torch::Tensor batched_indexes = torch::cat(batch_indexes , 0).to(device_);
            std::cout << "[COMPRESS] Batched index shape: " << batched_indexes.sizes() << std::endl;
            // **** //
            
            // Run compression model
            std::cout << "[COMPRESS] Running compressor model..." << std::endl;
            std::vector<torch::Tensor> inputs = { batched_input };
            std::vector<torch::Tensor> outputs = compressor_model_->run(inputs);

            torch::Tensor q_latent = outputs[0];
            torch::Tensor latent_indexes = outputs[1];
            torch::Tensor q_hyper_latent = outputs[2];
            torch::Tensor hyper_indexes = outputs[3];

            std::cout << "[COMPRESS] Compressor outputs:" << std::endl;
            std::cout << "  q_latent shape: " << q_latent.sizes() << std::endl;
            std::cout << "  latent_indexes shape: " << latent_indexes.sizes() << std::endl;
            std::cout << "  q_hyper_latent shape: " << q_hyper_latent.sizes() << std::endl;
            std::cout << "  hyper_indexes shape: " << hyper_indexes.sizes() << std::endl;

            // Check dimension relationship
            int64_t num_input_samples = batch_inputs.size();
            int64_t num_latent_codes = q_latent.sizes()[0];
            std::cout << "\n*** DIMENSION CHECK ***" << std::endl;
            std::cout << "Number of input samples: " << num_input_samples << std::endl;
            std::cout << "Number of latent codes: " << num_latent_codes << std::endl;
            std::cout << "Ratio (latent_codes / input_samples): "
                << static_cast<double>(num_latent_codes) / num_input_samples << std::endl;

            if (num_latent_codes == num_input_samples * 2) {
                std::cout << "✓ CORRECT: Each input sample produces 2 latent codes!" << std::endl;
            }
            else if (num_latent_codes == num_input_samples) {
                std::cout << "✗ WARNING: 1:1 ratio - this may be incorrect!" << std::endl;
            }
            else {
                std::cout << "? UNEXPECTED ratio!" << std::endl;
            }

            // Encode each latent code
            std::cout << "\n[ENCODE] Encoding " << num_latent_codes << " latent codes..." << std::endl;
            for (int64_t j = 0; j < num_latent_codes; j++) {
                // Extract symbols and indexes
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

                // Encode latents
                std::string latent_encoded = range_encoder.encode_with_indexes(
                    latent_symbol , latent_index ,
                    gs_quantized_cdf_ , gs_cdf_length_ , gs_offset_
                );

                // Encode hyper latents
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
            // ** JL modified ** //
            std::cout << "Batch max: " << batch_max << std::endl;
            std::cout << "Batch min: " << batch_min << std::endl;
            batch_max = 0.0;
            batch_min = 1000000.0;
            // **** //

            // ** JL modified ** //
            std::cout << "========================================" << std::endl;
            // Run hyper decompressor
            std::cout << "\n[HYPER DECOMPRESS] Running hyper decompressor..." << std::endl;
            std::vector<torch::Tensor> hyper_outputs = hyper_decompressor_model_->run({q_hyper_latent.to(torch::kFloat32)});
        
            torch::Tensor mean = hyper_outputs[0];
            torch::Tensor latent_indexes_recon = hyper_outputs[1];
            std::cout << "[HYPER DECOMPRESS] Mean shape: " << mean.sizes() << std::endl;
            std::cout << "[HYPER DECOMPRESS] Mean max min: " << mean.max().item<float>() << ", " << mean.min().item<float>() << std::endl;

            torch::Tensor q_latent_with_offset = q_latent.to(torch::kFloat32) + mean;
            std::cout << "[LATENT DECODE] q_latent_with_offsetv shape (before reshape): " << q_latent_with_offset.sizes() << std::endl;
            
            auto decoded_latents_sizes = q_latent_with_offset.sizes();
            std::cout << "[RESHAPE] Original decoded_latents shape: " << decoded_latents_sizes << std::endl;

            std::vector<int64_t> new_shape = { -1, 2 };
            new_shape.insert(new_shape.end() , decoded_latents_sizes.begin() + 1 , decoded_latents_sizes.end());
            std::cout << "[LATENT DECODE - RESHAPE] New shape vector: [";
            for (size_t i = 0; i < new_shape.size(); i++) {
                std::cout << new_shape[i];
                if (i < new_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;

            torch::Tensor reshaped_latents = q_latent_with_offset.reshape(new_shape);
            std::cout << "[LATENT DECODE - RESHAPE] Reshaped latents: " << reshaped_latents.sizes() << std::endl;
            std::cout << "[LATENT DECODE - RESHAPE] This means " << reshaped_latents.sizes()[0]
            << " pairs of latent codes will be processed" << std::endl;
            std::cout << "\n[DECOMPRESS] Running main decompressor..." << std::endl;
            std::vector<torch::Tensor> decompressor_outputs = decompressor_model_->run({reshaped_latents});

            torch::Tensor raw_output = decompressor_outputs[0];
            std::cout << "[DECOMPRESS] Raw output shape: " << raw_output.sizes() << std::endl;

            torch::Tensor norm_output = reshape_batch_2d_3d(
                raw_output ,
                num_input_samples
            );
            std::cout << "[DECOMPRESS] norm_output shape: " << norm_output.sizes() << std::endl;

            torch::Tensor denorm_output = norm_output*batched_scales + batched_offsets;
            std::cout << "[DECOMPRESS] denorm_output max min: " << denorm_output.max().item<float>() << ", " << denorm_output.min().item<float>() << std::endl;
            // Construct recon with input data shape
            torch::Tensor indexes_cpu = batched_indexes.to(torch::kCPU);
            for (int64_t i = 0; i < num_input_samples; ++i) {
                torch::Tensor index_row = indexes_cpu.select(0, i);
                int64_t idx0    = index_row[0].item<int64_t>();
                int64_t idx1    = index_row[1].item<int64_t>();
                int64_t start_t = index_row[2].item<int64_t>();
                int64_t end_t   = index_row[3].item<int64_t>();
                
                torch::Tensor source_slice_3d = denorm_output.select(0, i).squeeze(0);
                
                torch::Tensor source_slice_4d = denorm_output.select(0, i);                
                torch::Tensor dest_slice = recon_tensor.select(0, idx0).select(1, idx1).slice(2, start_t, end_t);
                
                dest_slice.copy_(source_slice);                
            }
            std::cout << "[DECOMPRESS] recon_tensor max min: " << recon_tensor.max().item<float>() << ", " << recon_tensor.min().item<float>() << std::endl;
            
            // **** //
            
            // ** JL modified ** //
            batch_offsets_vec.clear();
            batch_scales_vec.clear();
            batch_indexes.clear();
            // **** //
            
            result.num_batches++;
        }
    }
    std::cout << "[DECOMPRESS OUTPUT CHECK] recon_tensor max min: " << recon_tensor.max().item<float>() << ", " << recon_tensor.min().item<float>() << std::endl;
    
    // ** JL modified ** //    
    // Check
    std::cout << "\n[RECORDED METADATA CHECK] Data input shape: " << result.data_input_shape << std::endl;
    std::cout << "[RECORDED METADATA CHECK] Filtered blocks count: " << result.filtered_blocks.size() << std::endl;
    std::cout << "[RECORDED METADATA CHECK] Block nH: " << std::get<0>(result.block_info) << std::endl;
    std::cout << "[RECORDED METADATA CHECK] Block nW: " << std::get<1>(result.block_info) << std::endl;
    std::cout << "[RECORDED METADATA CHECK] Padding: [";
    {
        const auto& padding_vec = std::get<2>(result.block_info);     
        for (size_t i = 0; i < padding_vec.size(); ++i) {
            std::cout << padding_vec[i];
            if (i < padding_vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "[RECORDED METADATA CHECK] Total offsets collected: " << result.offsets.size() << std::endl;
    std::cout << "[RECORDED METADATA CHECK] Total scales collected: " << result.scales.size() << std::endl;
    std::cout << "[RECORDED METADATA CHECK] Total index vectors collected: " << result.indexes.size();
    if (!result.indexes.empty()) {
        std::cout << ", " << result.indexes[0].size(); // 0번째 안쪽 벡터의 크기(4)
    }
    std::cout << std::endl;
    // **** //

    std::cout << "\n========== COMPRESSION COMPLETE ==========" << std::endl;
    std::cout << "Total input samples processed: " << result.num_samples << std::endl;
    std::cout << "Total latent codes generated: " << result.encoded_latents.size() << std::endl;
    std::cout << "Total batches processed: " << result.num_batches << std::endl;
    // ** JL modified ** //
    std::cout << "Total batched offsets stored (this will be equal to the total batches processed): " << result.offsets.size() << std::endl;
    // **** //
    std::cout << "Ratio (latent_codes / samples): "
        << static_cast<double>(result.encoded_latents.size()) / result.num_samples << std::endl;

    return result;
}