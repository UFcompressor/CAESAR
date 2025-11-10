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


bool save_encoded_streams(
    const std::vector<std::string>& streams,
    const std::string& filename
) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
        return false;
    }
    for (const std::string& stream : streams) {
        uint32_t length = static_cast<uint32_t>(stream.size());
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(stream.data(), length);
    }
    file.close();
    return true;
}

std::vector<std::string> load_encoded_streams(const std::string& filename) {
    std::vector<std::string> results;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
        return results;
    }
    uint32_t length;
    while (file.read(reinterpret_cast<char*>(&length), sizeof(length))) {
        std::string data(length, '\0');
        if (length > 0) {
            if (!file.read(&data[0], length)) {
                std::cerr << "Error: Truncated file or read error!" << std::endl;
                break;
            }
        }
        results.push_back(std::move(data));
    }
    file.close();
    return results;
}

static void write_vec_float_with_count(const std::vector<float>& v, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file to write: " + path);
    }
    int32_t n = static_cast<int32_t>(v.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
    if (n > 0) {
        f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(float));
    }
}

static void write_nested_i32_with_count(const std::vector<std::vector<int32_t>>& vv, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file to write: " + path);
    }
    int32_t rows = static_cast<int32_t>(vv.size());
    f.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
    for (const auto& row : vv) {
        int32_t len = static_cast<int32_t>(row.size());
        f.write(reinterpret_cast<const char*>(&len), sizeof(int32_t));
        if (len > 0)
            f.write(reinterpret_cast<const char*>(row.data()), len * sizeof(int32_t));
    }
}

void save_aux_metadata_binary(const CompressionResult& result, const std::string& out_dir) {
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);
    write_vec_float_with_count(result.compressionMetaData.offsets, out_dir + "/offsets.bin");
    write_vec_float_with_count(result.compressionMetaData.scales, out_dir + "/scales.bin");
    write_nested_i32_with_count(result.compressionMetaData.indexes, out_dir + "/indexes.bin");
    std::cout << "GOOD Metadata saved" << std::endl;
}

template<typename T>
std::vector<T> load_array_from_bin(const std::string& filename) {
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    input_file.seekg(0, std::ios::end);
    size_t file_size_in_bytes = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    size_t num_elements = file_size_in_bytes / sizeof(T);
    std::vector<T> loaded_data(num_elements);
    input_file.read(reinterpret_cast<char*>(loaded_data.data()), file_size_in_bytes);
    input_file.close();
    return loaded_data;
}

template<typename T>
std::vector<std::vector<T>> reshape_to_2d(const std::vector<T>& flat_vec, size_t rows, size_t cols) {
    if (flat_vec.size() != rows * cols) {
        throw std::invalid_argument("Invalid dimensions for reshape.");
    }
    std::vector<std::vector<T>> vec_2d;
    vec_2d.reserve(rows);
    auto it = flat_vec.begin();
    for (size_t r = 0; r < rows; ++r) {
        vec_2d.emplace_back(it, it + cols);
        it += cols;
    }
    return vec_2d;
}

template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();
    const T* tensor_data_ptr = cpu_tensor.data_ptr<T>();
    int64_t num_elements = cpu_tensor.numel();
    return std::vector<T>(tensor_data_ptr, tensor_data_ptr + num_elements);
}

torch::Tensor loadRawBinary(const std::string& bin_path, const std::vector<int64_t>& shape) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + bin_path);
    }
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;
    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(float));
    file.close();
    torch::Tensor data = torch::from_blob(buffer.data(), shape, torch::kFloat32).clone();
    return data;
}


bool tensors_equal(const torch::Tensor& a_in, const torch::Tensor& b_in, const std::string& name, float tolerance = 1e-6) {
    // Align both tensors to the same device (GPU)
    auto device = a_in.device();
    torch::Tensor a = a_in.to(device).contiguous();
    torch::Tensor b = b_in.to(device).contiguous();

    if (!a.sizes().equals(b.sizes())) {
        std::cerr << "BAD " << name << " shape mismatch: "
                  << a.sizes() << " vs " << b.sizes() << std::endl;
        return false;
    }

    torch::Tensor diff = (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs();
    float max_diff = diff.max().item<float>();
    if (max_diff > tolerance) {
        std::cerr << "BAD " << name << " values differ (max diff: " << max_diff << ")" << std::endl;
        return false;
    }

    std::cout << "GOOD " << name << " match (max diff: " << max_diff << ")" << std::endl;
    return true;
}


torch::Tensor build_indexes_tensor(const std::vector<int32_t>& size) {
    int64_t dims = size.size();
    TORCH_CHECK(dims >= 2, "Input size must have at least 2 dimensions");
    int64_t C = size[1];
    std::vector<int64_t> view_dims = {1, C};
    view_dims.insert(view_dims.end(), dims - 2, 1);
    torch::Tensor indexes = torch::arange(C).view(view_dims);
    std::vector<int64_t> size_int64(size.begin(), size.end());
    return indexes.expand(size_int64).to(torch::kInt32);
}

int main() {
    try {
        auto device = torch::kCPU;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Using GPU." << std::endl;
            device = torch::kCUDA;
        } else {
            std::cout << "Using CPU." << std::endl;
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "DECOMPRESSION CONSISTENCY TEST" << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;


        std::cout << "Loading models..." << std::endl;
        auto compressor_model = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            "/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/caesar_compressor.pt2"
        );
        auto hyper_decompressor_model = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            "/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/caesar_hyper_decompressor.pt2"
        );
        std::cout << "GOOD Models loaded\n" << std::endl;


        std::cout << "Loading probability tables..." << std::endl;
        auto gs_quantized_cdf = reshape_to_2d(
            load_array_from_bin<int32_t>("/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/gs_quantized_cdf.bin"),
            128, 249);
        auto gs_cdf_length = load_array_from_bin<int32_t>(
            "/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/gs_cdf_length.bin");
        auto gs_offset = load_array_from_bin<int32_t>(
            "/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/gs_offset.bin");
        std::cout << "GOOD Probability tables loaded\n" << std::endl;

        std::cout << std::string(80, '-') << std::endl;
        std::cout << "PHASE 1: COMPRESSION" << std::endl;
        std::cout << std::string(80, '-') << "\n" << std::endl;

 
        Compressor compressor(device);
        torch::Tensor raw_data = loadRawBinary("TCf48.bin.f32", {1, 1, 100, 500, 500});
        
        DatasetConfig config;
        config.memory_data = raw_data;
        config.variable_idx = 0;
        config.n_frame = 8;
        config.dataset_name = "TCf48 Dataset";
        config.train_size = 256;
        config.inst_norm = true;
        config.norm_type = "mean_range";
        config.train_mode = false;
        config.n_overlap = 0;
        config.test_size = {256, 256};

        int batch_size = 32;
        float rel_eb = 0.001f;
        
        std::cout << "Running compression..." << std::endl;
        CompressionResult comp_result = compressor.compress(config, batch_size, rel_eb);
        std::cout << "GOOD Compression complete (" << comp_result.num_samples << " samples, " 
                  << comp_result.num_batches << " batches)\n" << std::endl;

  
        const std::string out_dir = "./test_decompress_output";
        std::filesystem::create_directories(out_dir);
        save_encoded_streams(comp_result.encoded_latents, out_dir + "/encoded_latents.bin");
        save_encoded_streams(comp_result.encoded_hyper_latents, out_dir + "/encoded_hyper_latents.bin");
        save_aux_metadata_binary(comp_result, out_dir);

        std::cout << std::string(80, '-') << std::endl;
        std::cout << "PHASE 2: RE-COMPRESS TO GET FRESH q_latents" << std::endl;
        std::cout << std::string(80, '-') << "\n" << std::endl;

   
        ScientificDataset dataset(config);
        std::cout << "GOOD Dataset re-created (size: " << dataset.size() << ")" << std::endl;

        c10::InferenceMode guard;
        std::vector<torch::Tensor> batch_inputs;
        for (int i = 0; i < batch_size && i < dataset.size(); i++) {
            auto sample = dataset.get_item(i);
            batch_inputs.push_back(sample["input"]);
        }
        
        torch::Tensor batched_input = torch::cat(batch_inputs, 0).to(device);
        std::cout << "Batched input shape: " << batched_input.sizes() << std::endl;

    
        std::vector<torch::Tensor> comp_outputs = compressor_model->run({batched_input});
        torch::Tensor q_latent_fresh = comp_outputs[0];
        torch::Tensor latent_indexes_fresh = comp_outputs[1];
        torch::Tensor q_hyper_latent_fresh = comp_outputs[2];
        
        std::cout << "GOOD Fresh compression outputs:" << std::endl;
        std::cout << "  - q_latent shape: " << q_latent_fresh.sizes() << std::endl;
        std::cout << "  - latent_indexes shape: " << latent_indexes_fresh.sizes() << std::endl;

        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "PHASE 3: LOAD AND DECODE COMPRESSED DATA" << std::endl;
        std::cout << std::string(80, '-') << "\n" << std::endl;

     
        std::vector<std::string> loaded_latents = load_encoded_streams(out_dir + "/encoded_latents.bin");
        std::vector<std::string> loaded_hyper_latents = load_encoded_streams(out_dir + "/encoded_hyper_latents.bin");
        std::cout << "GOOD Loaded " << loaded_latents.size() << " latent streams" << std::endl;

      
        RansDecoder range_decoder;
        size_t cur_latents = std::min((size_t)batch_size * 2, loaded_latents.size());
        
       
        std::cout << "\nDecoding first batch (" << cur_latents << " latents)..." << std::endl;
        std::vector<int32_t> hyper_size = {(int32_t)cur_latents, 64, 4, 4};
        torch::Tensor hyper_index_tensor = build_indexes_tensor(hyper_size);
        torch::Tensor decoded_hyper_latents = torch::zeros({(long)cur_latents, 64, 4, 4}).to(torch::kInt32);

        auto vbr_quantized_cdf = reshape_to_2d(
            load_array_from_bin<int32_t>("/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/vbr_quantized_cdf.bin"),
            64, 63);
        auto vbr_cdf_length = load_array_from_bin<int32_t>(
            "/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/vbr_cdf_length.bin");
        auto vbr_offset = load_array_from_bin<int32_t>(
            "/lustre/blue/ranka/eklasky/CAESAR_C/exported_model/vbr_offset.bin");

        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> hyper_index_vec = tensor_to_vector<int32_t>(
                hyper_index_tensor.select(0, (long)i).reshape(-1));
            std::vector<int32_t> hyper_decoded = range_decoder.decode_with_indexes(
                loaded_hyper_latents[i], hyper_index_vec,
                vbr_quantized_cdf, vbr_cdf_length, vbr_offset);
            torch::Tensor hyper_tensor = torch::tensor(hyper_decoded).reshape({64, 4, 4});
            decoded_hyper_latents.select(0, (long)i).copy_(hyper_tensor);
        }

        std::vector<torch::Tensor> hyper_outputs = hyper_decompressor_model->run({
            decoded_hyper_latents.to(torch::kFloat32).to(device)
        });
        torch::Tensor latent_indexes_recon = hyper_outputs[1];

        torch::Tensor decoded_latents = torch::zeros({(long)cur_latents, 64, 16, 16}).to(torch::kInt32);
        for (size_t i = 0; i < cur_latents; i++) {
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                latent_indexes_recon.select(0, (long)i).reshape(-1));
            std::vector<int32_t> latent_decoded = range_decoder.decode_with_indexes(
                loaded_latents[i], latent_index,
                gs_quantized_cdf, gs_cdf_length, gs_offset);
            torch::Tensor latent_tensor = torch::tensor(latent_decoded).reshape({64, 16, 16});
            decoded_latents.select(0, (long)i).copy_(latent_tensor);
        }

        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "PHASE 4: VERIFY CONSISTENCY" << std::endl;
        std::cout << std::string(80, '-') << "\n" << std::endl;

        bool q_latents_match = tensors_equal(
            q_latent_fresh.to(torch::kInt32),
            decoded_latents,
            "Fresh q_latents vs Decoded latents from file"
        );

        bool indexes_match = tensors_equal(
            latent_indexes_fresh.to(torch::kInt32).to(torch::kCPU),
            latent_indexes_recon.to(torch::kInt32).to(torch::kCPU),
            "Fresh latent_indexes vs latent_indexes_recon"
        );

        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "PHASE 5: RUN DECOMPRESSOR API" << std::endl;
        std::cout << std::string(80, '-') << "\n" << std::endl;

        Decompressor decompressor(device);
        torch::Tensor decompressed_result = decompressor.decompress(
            loaded_latents,
            loaded_hyper_latents,
            batch_size,
            config.n_frame,
            comp_result
        );
        
        std::cout << "GOOD Decompression complete" << std::endl;
        std::cout << "  Decompressed tensor shape: " << decompressed_result.sizes() << std::endl;

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "CONSISTENCY TEST SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << "\n" << std::endl;

        bool all_passed = q_latents_match && indexes_match;

        std::cout << "Critical Checks (what your supervisor asked for):" << std::endl;
        std::cout << "  " << (q_latents_match ? "GOOD" : "BAD") 
                  << " q_latents: Fresh compression == Decoded from file" << std::endl;
        std::cout << "  " << (indexes_match ? "GOOD" : "BAD") 
                  << " latent_indexes: Fresh == Reconstructed from hyper_decompressor" << std::endl;

        std::cout << "\n" << std::string(80, '=') << std::endl;
        if (all_passed) {
            std::cout << "GOOD ALL TESTS PASSED!" << std::endl;
            std::cout << "The range coder is working correctly in the decompression path." << std::endl;
        } else {
            std::cout << "BAD TESTS FAILED!" << std::endl;
            std::cout << "There is an inconsistency - investigate the failed checks above." << std::endl;
        }
        std::cout << std::string(80, '=') << std::endl;

        return all_passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "\n ERROR: " << e.what() << std::endl;
        return 1;
    }
}
