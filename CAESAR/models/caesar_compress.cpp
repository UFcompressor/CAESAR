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

template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();
    const T* tensor_data_ptr = cpu_tensor.data_ptr<T>();
    int64_t num_elements = cpu_tensor.numel();
    return std::vector<T>(tensor_data_ptr , tensor_data_ptr + num_elements);
}

torch::Tensor build_indexes_tensor_C(const std::vector<int32_t>& size) {
    int64_t dims = size.size();
    TORCH_CHECK(dims >= 2 , "Input size must have at least 2 dimensions (N, C, ...)");

    int64_t C = size[1];
    std::vector<int64_t> view_dims = { 1, C };
    view_dims.insert(view_dims.end() , dims - 2 , 1);

    torch::Tensor indexes = torch::arange(C).view(view_dims);
    std::vector<int64_t> size_int64(size.begin() , size.end());
    return indexes.expand(size_int64).to(torch::kInt32);
}

Compressor::Compressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Compressor::load_models() {
    std::cout << "Loading compressor model..." << std::endl;
    compressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/adios/Programs/CAESAR_C/exported_model/caesar_compressor.pt2"
    );
    std::cout << "Model loaded successfully." << std::endl;
}

void Compressor::load_probability_tables() {
    std::cout << "Loading probability tables..." << std::endl;

    // Load VBR tables
    auto vbr_quantized_cdf_1d = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/vbr_quantized_cdf.bin");
    vbr_cdf_length_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/vbr_cdf_length.bin");
    vbr_offset_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/vbr_offset.bin");
    vbr_quantized_cdf_ = reshape_to_2d(vbr_quantized_cdf_1d , 64 , 63);

    // Load GS tables
    auto gs_quantized_cdf_1d = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/gs_quantized_cdf.bin");
    gs_cdf_length_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/gs_cdf_length.bin");
    gs_offset_ = load_array_from_bin<int32_t>("/home/adios/Programs/CAESAR_C/exported_model/gs_offset.bin");
    gs_quantized_cdf_ = reshape_to_2d(gs_quantized_cdf_1d , 128 , 249);

    std::cout << "Probability tables loaded successfully." << std::endl;
}

CompressionResult Compressor::compress(const DatasetConfig& config , int batch_size) {
    c10::InferenceMode guard;

    std::cout << "\n========== STARTING COMPRESSION ==========" << std::endl;
    std::cout << "Device: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    // Load dataset
    ScientificDataset dataset(config);
    std::cout << "Dataset loaded. Total samples: " << dataset.size() << std::endl;

    // Initialize result
    CompressionResult result;
    result.num_samples = 0;
    result.num_batches = 0;

    // Initialize encoder
    RansEncoder range_encoder;

    // Batch processing
    std::vector<torch::Tensor> batch_inputs;
    batch_inputs.reserve(batch_size);

    for (size_t i = 0; i < dataset.size(); i++) {
        auto sample = dataset.get_item(i);
        torch::Tensor input_tensor = sample["input"];
        batch_inputs.push_back(input_tensor);

        if (batch_inputs.size() == static_cast<size_t>(batch_size) || i == dataset.size() - 1) {
            std::cout << "\n--- Processing Batch " << result.num_batches << " ---" << std::endl;
            std::cout << "Samples in batch: " << batch_inputs.size() << std::endl;

            // Concatenate batch
            torch::Tensor batched_input = torch::cat(batch_inputs , 0).to(device_);

            // Run compression model
            std::vector<torch::Tensor> inputs = { batched_input };
            std::vector<torch::Tensor> outputs = compressor_model_->run(inputs);

            torch::Tensor q_latent = outputs[0];
            torch::Tensor latent_indexes = outputs[1];
            torch::Tensor q_hyper_latent = outputs[2];
            torch::Tensor hyper_indexes = outputs[3];

            // Encode each sample in the batch
            for (size_t j = 0; j < batch_inputs.size(); j++) {
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
                result.num_samples++;
            }

            batch_inputs.clear();
            result.num_batches++;
        }
    }

    std::cout << "\n========== COMPRESSION COMPLETE ==========" << std::endl;
    std::cout << "Total samples compressed: " << result.num_samples << std::endl;
    std::cout << "Total batches processed: " << result.num_batches << std::endl;

    return result;
}