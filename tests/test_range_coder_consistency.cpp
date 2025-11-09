#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <filesystem>
#include "../CAESAR/models/caesar_compress.h"
#include "../CAESAR/models/caesar_decompress.h"
#include "../CAESAR/dataset/dataset.h"
#include "../CAESAR/models/range_coder/rans_coder.hpp"

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

bool tensors_equal(const torch::Tensor& a , const torch::Tensor& b , const std::string& name) {
    if (!a.sizes().equals(b.sizes())) {
        std::cerr << "BAD  " << name << " shape mismatch: "
            << a.sizes() << " vs " << b.sizes() << std::endl;
        return false;
    }

    torch::Tensor diff = (a - b).abs();
    float max_diff = diff.max().item<float>();

    if (max_diff > 1e-6) {
        std::cerr << "BAD  " << name << " values differ (max diff: " << max_diff << ")" << std::endl;
        return false;
    }

    std::cout << "GOOD " << name << " match perfectly" << std::endl;
    return true;
}

template<typename T>
bool vectors_equal(const std::vector<T>& a , const std::vector<T>& b , const std::string& name) {
    if (a.size() != b.size()) {
        std::cerr << "BAD  " << name << " size mismatch: "
            << a.size() << " vs " << b.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "BAD  " << name << " differ at index " << i
                << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }

    std::cout << "GOOD " << name << " match perfectly (size: " << a.size() << ")" << std::endl;
    return true;
}

torch::Tensor loadRawBinary(const std::string& bin_path , const std::vector<int64_t>& shape) {
    std::ifstream file(bin_path , std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + bin_path);
    }
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;
    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()) , num_elements * sizeof(float));
    file.close();
    torch::Tensor data = torch::from_blob(buffer.data() , shape , torch::kFloat32).clone();
    return data;
}

int main() {
    try {
        auto device = torch::kCPU;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Using GPU." << std::endl;
            device = torch::kCUDA;
        }
        else {
            std::cout << "Using CPU." << std::endl;
        }

        std::cout << "\n" << std::string(80 , '=') << std::endl;
        std::cout << "RANGE CODER CONSISTENCY TEST" << std::endl;
        std::cout << std::string(80 , '=') << "\n" << std::endl;

        std::cout << "Loading models..." << std::endl;
        auto compressor_model = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            "/home/adios/Programs/CAESAR_C/exported_model/caesar_compressor.pt2"
        );
        auto hyper_decompressor_model = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            "/home/adios/Programs/CAESAR_C/exported_model/caesar_hyper_decompressor.pt2"
        );
        std::cout << "GOOD Models loaded successfully\n" << std::endl;

        std::cout << "Loading probability tables..." << std::endl;
        auto vbr_quantized_cdf_1d = load_array_from_bin<int32_t>(
            "/home/adios/Programs/CAESAR_C/exported_model/vbr_quantized_cdf.bin");
        auto vbr_cdf_length = load_array_from_bin<int32_t>(
            "/home/adios/Programs/CAESAR_C/exported_model/vbr_cdf_length.bin");
        auto vbr_offset = load_array_from_bin<int32_t>(
            "/home/adios/Programs/CAESAR_C/exported_model/vbr_offset.bin");
        auto vbr_quantized_cdf = reshape_to_2d(vbr_quantized_cdf_1d , 64 , 63);

        auto gs_quantized_cdf_1d = load_array_from_bin<int32_t>(
            "/home/adios/Programs/CAESAR_C/exported_model/gs_quantized_cdf.bin");
        auto gs_cdf_length = load_array_from_bin<int32_t>(
            "/home/adios/Programs/CAESAR_C/exported_model/gs_cdf_length.bin");
        auto gs_offset = load_array_from_bin<int32_t>(
            "/home/adios/Programs/CAESAR_C/exported_model/gs_offset.bin");
        auto gs_quantized_cdf = reshape_to_2d(gs_quantized_cdf_1d , 128 , 249);
        std::cout << "GOOD Probability tables loaded\n" << std::endl;

        std::cout << "Loading dataset..." << std::endl;
        torch::Tensor raw_data = loadRawBinary("TCf48.bin.f32" , { 1, 1, 100, 500, 500 });

        DatasetConfig config;
        config.memory_data = raw_data;
        config.variable_idx = 0;
        config.n_frame = 8;
        config.dataset_name = "TCf48 Dataset";
        config.section_range = std::nullopt;
        config.frame_range = std::nullopt;
        config.train_size = 256;
        config.inst_norm = true;
        config.norm_type = "mean_range";
        config.train_mode = false;
        config.n_overlap = 0;
        config.test_size = { 256, 256 };
        config.augment_type = {};

        ScientificDataset dataset(config);
        int batch_size = 32;
        std::cout << "GOOD Dataset created (size: " << dataset.size() << ")\n" << std::endl;

        std::cout << std::string(80 , '-') << std::endl;
        std::cout << "PHASE 1: Initial Compression (get q_latents and q_hyper_latents)" << std::endl;
        std::cout << std::string(80 , '-') << "\n" << std::endl;

        c10::InferenceMode guard;

        std::vector<torch::Tensor> batch_inputs;
        for (int i = 0; i < batch_size && i < dataset.size(); i++) {
            auto sample = dataset.get_item(i);
            batch_inputs.push_back(sample["input"]);
        }

        torch::Tensor batched_input = torch::cat(batch_inputs , 0).to(device);
        std::cout << "Batched input shape: " << batched_input.sizes() << std::endl;

        std::vector<torch::Tensor> comp_outputs = compressor_model->run({ batched_input });
        torch::Tensor q_latent_original = comp_outputs[0];
        torch::Tensor latent_indexes_original = comp_outputs[1];
        torch::Tensor q_hyper_latent_original = comp_outputs[2];
        torch::Tensor hyper_indexes_original = comp_outputs[3];

        std::cout << "GOOD Compressor outputs:" << std::endl;
        std::cout << "  - q_latent shape: " << q_latent_original.sizes() << std::endl;
        std::cout << "  - latent_indexes shape: " << latent_indexes_original.sizes() << std::endl;
        std::cout << "  - q_hyper_latent shape: " << q_hyper_latent_original.sizes() << std::endl;
        std::cout << "  - hyper_indexes shape: " << hyper_indexes_original.sizes() << std::endl;

        std::cout << "\nEncoding with range coder..." << std::endl;
        RansEncoder range_encoder;
        std::vector<std::string> encoded_latents;
        std::vector<std::string> encoded_hyper_latents;

        for (int64_t j = 0; j < q_latent_original.size(0); j++) {
            std::vector<int32_t> latent_symbol = tensor_to_vector<int32_t>(
                q_latent_original.select(0 , j).reshape(-1));
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                latent_indexes_original.select(0 , j).reshape(-1));
            std::vector<int32_t> hyper_symbol = tensor_to_vector<int32_t>(
                q_hyper_latent_original.select(0 , j).reshape(-1));
            std::vector<int32_t> hyper_index = tensor_to_vector<int32_t>(
                hyper_indexes_original.select(0 , j).reshape(-1));

            std::string latent_encoded = range_encoder.encode_with_indexes(
                latent_symbol , latent_index , gs_quantized_cdf , gs_cdf_length , gs_offset);
            std::string hyper_encoded = range_encoder.encode_with_indexes(
                hyper_symbol , hyper_index , vbr_quantized_cdf , vbr_cdf_length , vbr_offset);

            encoded_latents.push_back(latent_encoded);
            encoded_hyper_latents.push_back(hyper_encoded);
        }
        std::cout << "GOOD Encoded " << encoded_latents.size() << " latent streams" << std::endl;

        std::cout << "\nDecoding with range coder..." << std::endl;
        std::cout << "Sample encoded sizes - Latent: " << encoded_latents[0].size()
            << " bytes, Hyper: " << encoded_hyper_latents[0].size() << " bytes" << std::endl;

        RansDecoder range_decoder;
        std::vector<torch::Tensor> decoded_q_latents;
        std::vector<torch::Tensor> decoded_q_hyper_latents;

        auto single_hyper_shape = q_hyper_latent_original.select(0 , 0).sizes().vec();
        auto single_latent_shape = q_latent_original.select(0 , 0).sizes().vec();

        for (size_t j = 0; j < encoded_latents.size(); j++) {
            std::vector<int32_t> hyper_index = tensor_to_vector<int32_t>(
                hyper_indexes_original.select(0 , j).reshape(-1));
            std::vector<int32_t> decoded_hyper = range_decoder.decode_with_indexes(
                encoded_hyper_latents[j] , hyper_index ,
                vbr_quantized_cdf , vbr_cdf_length , vbr_offset);

            torch::Tensor decoded_hyper_tensor = torch::tensor(decoded_hyper , torch::kInt32)
                .reshape(single_hyper_shape);
            decoded_q_hyper_latents.push_back(decoded_hyper_tensor);

            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                latent_indexes_original.select(0 , j).reshape(-1));
            std::vector<int32_t> decoded_latent = range_decoder.decode_with_indexes(
                encoded_latents[j] , latent_index ,
                gs_quantized_cdf , gs_cdf_length , gs_offset);

            torch::Tensor decoded_latent_tensor = torch::tensor(decoded_latent , torch::kInt32)
                .reshape(single_latent_shape);
            decoded_q_latents.push_back(decoded_latent_tensor);
        }
        std::cout << "GOOD Decoded all streams" << std::endl;

        torch::Tensor q_latent_decoded = torch::stack(decoded_q_latents , 0).to(device);
        torch::Tensor q_hyper_latent_decoded = torch::stack(decoded_q_hyper_latents , 0).to(device);

        std::cout << "\n" << std::string(80 , '-') << std::endl;
        std::cout << "PHASE 2: Verify Range Coder Round-trip" << std::endl;
        std::cout << std::string(80 , '-') << "\n" << std::endl;

        bool latents_match = tensors_equal(
            q_latent_original.to(torch::kInt32) ,
            q_latent_decoded.to(torch::kInt32) ,
            "q_latents (encode->decode)"
        );

        bool hyper_latents_match = tensors_equal(
            q_hyper_latent_original.to(torch::kInt32) ,
            q_hyper_latent_decoded.to(torch::kInt32) ,
            "q_hyper_latents (encode->decode)"
        );

        std::cout << "\n" << std::string(80 , '-') << std::endl;
        std::cout << "PHASE 3: Verify Hyper Decompressor Consistency" << std::endl;
        std::cout << std::string(80 , '-') << "\n" << std::endl;

        std::vector<torch::Tensor> hyper_outputs_original = hyper_decompressor_model->run({
            q_hyper_latent_original.to(torch::kFloat32)
            });
        torch::Tensor mean_original = hyper_outputs_original[0];
        torch::Tensor latent_indexes_recon_original = hyper_outputs_original[1];

        std::vector<torch::Tensor> hyper_outputs_decoded = hyper_decompressor_model->run({
            q_hyper_latent_decoded.to(torch::kFloat32)
            });
        torch::Tensor mean_decoded = hyper_outputs_decoded[0];
        torch::Tensor latent_indexes_recon_decoded = hyper_outputs_decoded[1];

        bool indexes_match_original = tensors_equal(
            latent_indexes_original.to(torch::kInt32) ,
            latent_indexes_recon_original.to(torch::kInt32) ,
            "latent_indexes (compressor vs hyper_decompressor on original)"
        );

        bool indexes_match_decoded = tensors_equal(
            latent_indexes_recon_original.to(torch::kInt32) ,
            latent_indexes_recon_decoded.to(torch::kInt32) ,
            "latent_indexes_recon (original vs decoded q_hyper_latent)"
        );

        bool means_match = tensors_equal(
            mean_original ,
            mean_decoded ,
            "mean (original vs decoded q_hyper_latent)"
        );

        std::cout << "\n" << std::string(80 , '=') << std::endl;
        std::cout << "CONSISTENCY TEST SUMMARY" << std::endl;
        std::cout << std::string(80 , '=') << "\n" << std::endl;

        bool all_passed = latents_match && hyper_latents_match &&
            indexes_match_original && indexes_match_decoded && means_match;

        std::cout << "Range Coder Round-trip Tests:" << std::endl;
        std::cout << "  " << (latents_match ? "GOOD" : "BAD ")
            << " q_latents encode->decode consistency" << std::endl;
        std::cout << "  " << (hyper_latents_match ? "GOOD" : "BAD ")
            << " q_hyper_latents encode->decode consistency" << std::endl;

        std::cout << "\nModel Consistency Tests:" << std::endl;
        std::cout << "  " << (indexes_match_original ? "GOOD" : "BAD ")
            << " latent_indexes (compressor == hyper_decompressor)" << std::endl;
        std::cout << "  " << (indexes_match_decoded ? "GOOD" : "BAD ")
            << " hyper_decompressor output stable after range coder" << std::endl;
        std::cout << "  " << (means_match ? "GOOD" : "BAD ")
            << " mean output stable after range coder" << std::endl;

        std::cout << "\n" << std::string(80 , '=') << std::endl;
        if (all_passed) {
            std::cout << "GOOD ALL TESTS PASSED - Range coder is working consistently!" << std::endl;
        }
        else {
            std::cout << "BAD  SOME TESTS FAILED - Range coder has consistency issues!" << std::endl;
        }
        std::cout << std::string(80 , '=') << std::endl;

        return all_passed ? 0 : 1;

    }
    catch (const std::exception& e) {
        std::cerr << "\n ERROR: " << e.what() << std::endl;
        return 1;
    }
}