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

bool tensors_equal(const torch::Tensor& a , const torch::Tensor& b , const std::string& name , float tolerance = 1e-6) {
    if (!a.sizes().equals(b.sizes())) {
        std::cerr << "BAD " << name << " shape mismatch: "
            << a.sizes() << " vs " << b.sizes() << std::endl;
        return false;
    }

    torch::Tensor diff = (a - b).abs();
    float max_diff = diff.max().item<float>();

    if (max_diff > tolerance) {
        std::cerr << "BAD " << name << " values differ (max diff: " << max_diff << ")" << std::endl;
        return false;
    }

    std::cout << "GOOD " << name << " match perfectly (max diff: " << max_diff << ")" << std::endl;
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
        std::cout << "DECOMPRESSION CONSISTENCY TEST" << std::endl;
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
        std::cout << "PHASE 1: Compress and get original latents" << std::endl;
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
        torch::Tensor q_latent_compress = comp_outputs[0];
        torch::Tensor latent_indexes_compress = comp_outputs[1];
        torch::Tensor q_hyper_latent_compress = comp_outputs[2];
        torch::Tensor hyper_indexes_compress = comp_outputs[3];

        std::cout << "GOOD Original compression outputs:" << std::endl;
        std::cout << "  - q_latent shape: " << q_latent_compress.sizes() << std::endl;
        std::cout << "  - latent_indexes shape: " << latent_indexes_compress.sizes() << std::endl;
        std::cout << "  - q_hyper_latent shape: " << q_hyper_latent_compress.sizes() << std::endl;

        std::cout << "\nEncoding with range coder..." << std::endl;
        RansEncoder range_encoder;
        std::vector<std::string> encoded_latents;
        std::vector<std::string> encoded_hyper_latents;

        for (int64_t j = 0; j < q_latent_compress.size(0); j++) {
            std::vector<int32_t> latent_symbol = tensor_to_vector<int32_t>(
                q_latent_compress.select(0 , j).reshape(-1));
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                latent_indexes_compress.select(0 , j).reshape(-1));
            std::vector<int32_t> hyper_symbol = tensor_to_vector<int32_t>(
                q_hyper_latent_compress.select(0 , j).reshape(-1));
            std::vector<int32_t> hyper_index = tensor_to_vector<int32_t>(
                hyper_indexes_compress.select(0 , j).reshape(-1));

            encoded_latents.push_back(range_encoder.encode_with_indexes(
                latent_symbol , latent_index , gs_quantized_cdf , gs_cdf_length , gs_offset));
            encoded_hyper_latents.push_back(range_encoder.encode_with_indexes(
                hyper_symbol , hyper_index , vbr_quantized_cdf , vbr_cdf_length , vbr_offset));
        }
        std::cout << "GOOD Encoded " << encoded_latents.size() << " streams" << std::endl;

        std::cout << "\n" << std::string(80 , '-') << std::endl;
        std::cout << "PHASE 2: Simulate decompression process" << std::endl;
        std::cout << std::string(80 , '-') << "\n" << std::endl;

        size_t lat_start = 0;
        size_t lat_end = std::min(lat_start + (size_t)batch_size * 2 , encoded_latents.size());
        size_t cur_latents = lat_end - lat_start;

        std::cout << "Processing batch with " << cur_latents << " latents" << std::endl;

        std::cout << "\nStep 1: Decode hyper latents..." << std::endl;
        RansDecoder range_decoder;
        std::vector<int32_t> hyper_size = { (int32_t)cur_latents, 64, 4, 4 };
        torch::Tensor hyper_index_tensor = build_indexes_tensor(hyper_size);
        torch::Tensor decoded_hyper_latents = torch::zeros({ (long)cur_latents, 64, 4, 4 }).to(torch::kInt32);

        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> hyper_index_vec = tensor_to_vector<int32_t>(
                hyper_index_tensor.select(0 , (long)i).reshape(-1));
            std::vector<int32_t> hyper_decoded = range_decoder.decode_with_indexes(
                encoded_hyper_latents[lat_start + i] ,
                hyper_index_vec ,
                vbr_quantized_cdf ,
                vbr_cdf_length ,
                vbr_offset
            );
            torch::Tensor hyper_tensor = torch::tensor(hyper_decoded).reshape({ 64, 4, 4 });
            decoded_hyper_latents.select(0 , (long)i).copy_(hyper_tensor);
        }
        std::cout << "GOOD Decoded hyper latents shape: " << decoded_hyper_latents.sizes() << std::endl;

        bool hyper_match = tensors_equal(
            q_hyper_latent_compress.to(torch::kInt32) ,
            decoded_hyper_latents ,
            "Decoded hyper_latents vs original q_hyper_latent"
        );

        std::cout << "\nStep 2: Run hyper decompressor..." << std::endl;
        std::vector<torch::Tensor> hyper_outputs = hyper_decompressor_model->run({
            decoded_hyper_latents.to(torch::kFloat32).to(device)
            });
        torch::Tensor mean_decomp = hyper_outputs[0];
        torch::Tensor latent_indexes_recon = hyper_outputs[1];
        std::cout << "GOOD Hyper decompressor outputs:" << std::endl;
        std::cout << "  - mean shape: " << mean_decomp.sizes() << std::endl;
        std::cout << "  - latent_indexes_recon shape: " << latent_indexes_recon.sizes() << std::endl;

        bool indexes_match = tensors_equal(
            latent_indexes_compress.to(torch::kInt32) ,
            latent_indexes_recon.to(torch::kInt32) ,
            "latent_indexes_recon vs original latent_indexes"
        );

        std::cout << "\nStep 3: Decode latents using latent_indexes_recon..." << std::endl;
        torch::Tensor decoded_latents_before_offset = torch::zeros({ (long)cur_latents, 64, 16, 16 }).to(torch::kInt32);

        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                latent_indexes_recon.select(0 , (long)i).reshape(-1));
            std::vector<int32_t> latent_decoded = range_decoder.decode_with_indexes(
                encoded_latents[lat_start + i] ,
                latent_index ,
                gs_quantized_cdf ,
                gs_cdf_length ,
                gs_offset
            );
            torch::Tensor latent_tensor = torch::tensor(latent_decoded).reshape({ 64, 16, 16 });
            decoded_latents_before_offset.select(0 , (long)i).copy_(latent_tensor);
        }
        std::cout << "GOOD Decoded latents shape: " << decoded_latents_before_offset.sizes() << std::endl;


        bool latents_match = tensors_equal(
            q_latent_compress.to(torch::kInt32) ,
            decoded_latents_before_offset ,
            "Decoded latents vs original q_latent"
        );


        std::cout << "\nStep 4: Add mean offset..." << std::endl;
        torch::Tensor q_latent_with_offset = decoded_latents_before_offset.to(torch::kFloat32).to(device) + mean_decomp;
        std::cout << "GOOD q_latent_with_offset shape: " << q_latent_with_offset.sizes() << std::endl;

        std::vector<torch::Tensor> hyper_outputs_original = hyper_decompressor_model->run({
            q_hyper_latent_compress.to(torch::kFloat32).to(device)
            });
        torch::Tensor mean_original = hyper_outputs_original[0];
        torch::Tensor q_latent_with_offset_original = q_latent_compress.to(torch::kFloat32).to(device) + mean_original;


        bool final_latents_match = tensors_equal(
            q_latent_with_offset_original ,
            q_latent_with_offset ,
            "Final q_latent_with_offset (compression vs decompression path)" ,
            1e-6
        );


        std::cout << "\n" << std::string(80 , '=') << std::endl;
        std::cout << "DECOMPRESSION CONSISTENCY SUMMARY" << std::endl;
        std::cout << std::string(80 , '=') << "\n" << std::endl;

        bool all_passed = hyper_match && indexes_match && latents_match && final_latents_match;

        std::cout << "Decompression Path Tests:" << std::endl;
        std::cout << "  " << (hyper_match ? "GOOD" : "BAD")
            << " Decoded hyper_latents match original q_hyper_latent" << std::endl;
        std::cout << "  " << (indexes_match ? "GOOD" : "BAD")
            << " latent_indexes_recon matches original latent_indexes" << std::endl;
        std::cout << "  " << (latents_match ? "GOOD" : "BAD")
            << " Decoded latents match original q_latent" << std::endl;
        std::cout << "  " << (final_latents_match ? "GOOD" : "BAD")
            << " Final latent representation consistent" << std::endl;

        std::cout << "\n" << std::string(80 , '=') << std::endl;
        if (all_passed) {
            std::cout << "GOOD ALL TESTS PASSED - Decompression path is consistent!" << std::endl;
        }
        else {
            std::cout << "BAD SOME TESTS FAILED - There are inconsistencies in decompression!" << std::endl;
        }
        std::cout << std::string(80 , '=') << std::endl;

        return all_passed ? 0 : 1;

    }
    catch (const std::exception& e) {
        std::cerr << "\n ERROR: " << e.what() << std::endl;
        return 1;
    }
}