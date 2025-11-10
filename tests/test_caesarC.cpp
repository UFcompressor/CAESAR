#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <filesystem>
#include "../CAESAR/models/caesar_compress.h"
#include "../CAESAR/models/caesar_decompress.h"
#include "../CAESAR/dataset/dataset.h"

bool save_encoded_streams(
    const std::vector<std::string>& streams ,
    const std::string& filename
) {
    std::ofstream file(filename , std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
        return false;
    }
    for (const std::string& stream : streams) {
        uint32_t length = static_cast<uint32_t>(stream.size());
        file.write(reinterpret_cast<const char*>(&length) , sizeof(length));
        file.write(stream.data() , length);
    }
    file.close();
    return true;
}

static void write_vec_float_with_count(const std::vector<float>& v , const std::string& path) {
    std::ofstream f(path , std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file to write: " + path);
    }
    int32_t n = static_cast<int32_t>(v.size());
    f.write(reinterpret_cast<const char*>(&n) , sizeof(int32_t));
    if (n > 0) {
        f.write(reinterpret_cast<const char*>(v.data()) , n * sizeof(float));
    }
}

static void write_nested_i32_with_count(const std::vector<std::vector<int32_t>>& vv , const std::string& path) {
    std::ofstream f(path , std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file to write: " + path);
    }
    int32_t rows = static_cast<int32_t>(vv.size());
    f.write(reinterpret_cast<const char*>(&rows) , sizeof(int32_t));
    for (const auto& row : vv) {
        int32_t len = static_cast<int32_t>(row.size());
        f.write(reinterpret_cast<const char*>(&len) , sizeof(int32_t));
        if (len > 0)
            f.write(reinterpret_cast<const char*>(row.data()) , len * sizeof(int32_t));
    }
}

static std::vector<float> read_vec_float_with_count(const std::string& path) {
    std::ifstream f(path , std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file to read: " + path);
    }
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n) , sizeof(int32_t));
    if (n < 0) throw std::runtime_error("Corrupted float vector count in " + path);
    std::vector<float> v(n);
    if (n > 0) {
        f.read(reinterpret_cast<char*>(v.data()) , n * sizeof(float));
    }
    return v;
}

static std::vector<std::vector<int32_t>> read_nested_i32_with_count(const std::string& path) {
    std::ifstream f(path , std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file to read: " + path);
    }
    int32_t rows = 0;
    f.read(reinterpret_cast<char*>(&rows) , sizeof(int32_t));
    if (rows < 0) throw std::runtime_error("Corrupted rows count in " + path);
    std::vector<std::vector<int32_t>> vv;
    vv.reserve(rows);
    for (int32_t i = 0; i < rows; ++i) {
        int32_t len = 0;
        f.read(reinterpret_cast<char*>(&len) , sizeof(int32_t));
        if (len < 0) throw std::runtime_error("Corrupted row length in " + path);
        std::vector<int32_t> row(len);
        if (len > 0) {
            f.read(reinterpret_cast<char*>(row.data()) , len * sizeof(int32_t));
        }
        vv.push_back(std::move(row));
    }
    return vv;
}

void save_aux_metadata_binary(const CompressionResult& result , const std::string& out_dir) {
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    write_vec_float_with_count(result.compressionMetaData.offsets , out_dir + "/offsets.bin");
    write_vec_float_with_count(result.compressionMetaData.scales , out_dir + "/scales.bin");
    write_nested_i32_with_count(result.compressionMetaData.indexes , out_dir + "/indexes.bin");

    // 可选：根据需要也可以把其它小字段一起存，但本轮我们只存解码必需的三项
    std::cout << "✓ offsets.bin, scales.bin, indexes.bin saved." << std::endl;
}

std::vector<std::string> load_encoded_streams(const std::string& filename) {
    std::vector<std::string> results;
    std::ifstream file(filename , std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
        return results;
    }
    uint32_t length;
    while (file.read(reinterpret_cast<char*>(&length) , sizeof(length))) {
        std::string data(length , '\0');
        if (length > 0) {
            if (!file.read(&data[0] , length)) {
                std::cerr << "Error: Truncated file or read error!" << std::endl;
                break;
            }
        }
        results.push_back(std::move(data));
    }
    file.close();
    return results;
}

void save_tensor_to_bin(const torch::Tensor& tensor , const std::string& filename) {
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
    std::ofstream file(filename , std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    file.write(
        reinterpret_cast<const char*>(cpu_tensor.data_ptr<float>()) ,
        cpu_tensor.numel() * sizeof(float)
    );
    file.close();
    std::cout << "  - Tensor saved to " << filename << std::endl;
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

    torch::Tensor data = torch::from_blob(
        buffer.data() , shape , torch::kFloat32
    ).clone();

    std::cout << "Loaded " << bin_path << " with shape: " << data.sizes() << "\n";
    std::cout << "  Min: " << data.min().item<float>() << ", Max: " << data.max().item<float>() << "\n";
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

        std::cout << "\n" << std::string(60 , '=') << std::endl;
        std::cout << "PHASE 1: COMPRESSION" << std::endl;
        std::cout << std::string(60 , '=') << "\n" << std::endl;

        Compressor compressor(device);

        std::cout << "Loading TCf48.bin.f32 raw data...\n";
        torch::Tensor raw_data = loadRawBinary("TCf48.bin.f32" , { 1, 1, 100, 500, 500 });
        std::cout << "\n";

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

        int batch_size = 32;
        float rel_eb = 0.001f;
        CompressionResult comp_result = compressor.compress(config , batch_size , rel_eb);

        std::cout << "\nSaving compressed data..." << std::endl;
        const std::string out_dir = "/home/adios/Programs/CAESAR_C/build/tests/output";
        std::filesystem::create_directories(out_dir);
        save_encoded_streams(comp_result.encoded_latents , out_dir + "/encoded_latents.bin");
        save_encoded_streams(comp_result.encoded_hyper_latents , out_dir + "/encoded_hyper_latents.bin");
        save_aux_metadata_binary(comp_result , out_dir);
        std::cout << "Compressed data saved to output/" << std::endl;

        size_t total_compressed_bytes = 0;
        for (const auto& s : comp_result.encoded_latents)       total_compressed_bytes += s.size();
        for (const auto& s : comp_result.encoded_hyper_latents) total_compressed_bytes += s.size();

        std::cout << "\nCompression Statistics:" << std::endl;
        std::cout << "  - Total samples: " << comp_result.num_samples << std::endl;
        std::cout << "  - Total batches: " << comp_result.num_batches << std::endl;
        std::cout << "  - Compressed size: " << total_compressed_bytes << " bytes" << std::endl;

        std::cout << "\n" << std::string(60 , '=') << std::endl;
        std::cout << "PHASE 2: DECOMPRESSION" << std::endl;
        std::cout << std::string(60 , '=') << "\n" << std::endl;

        std::cout << "Loading compressed data..." << std::endl;
        std::vector<std::string> loaded_latents = load_encoded_streams(out_dir + "/encoded_latents.bin");
        std::vector<std::string> loaded_hyper_latents = load_encoded_streams(out_dir + "/encoded_hyper_latents.bin");
        std::cout << "Loaded " << loaded_latents.size() << " latent streams and "
            << loaded_hyper_latents.size() << " hyper-latent streams" << std::endl;

        bool data_matches = true;
        if (loaded_latents.size() != comp_result.encoded_latents.size() ||
            loaded_hyper_latents.size() != comp_result.encoded_hyper_latents.size()) {
            std::cerr << "Error: Loaded data size mismatch!" << std::endl;
            data_matches = false;
        }
        else {
            for (size_t i = 0; i < loaded_latents.size(); i++) {
                if (loaded_latents[i] != comp_result.encoded_latents[i] ||
                    loaded_hyper_latents[i] != comp_result.encoded_hyper_latents[i]) {
                    std::cerr << "Error: Loaded data mismatch at sample " << i << std::endl;
                    data_matches = false;
                    break;
                }
            }
        }
        if (!data_matches) {
            std::cerr << "✗ Verification failed: Loaded data does not match!" << std::endl;
            return 1;
        }
        std::cout << "✓ Verification passed: Loaded data matches compressed data" << std::endl;

        std::vector<float> offsets = read_vec_float_with_count(out_dir + "/offsets.bin");
        std::vector<float> scales = read_vec_float_with_count(out_dir + "/scales.bin");
        std::vector<std::vector<int32_t>> indexes = read_nested_i32_with_count(out_dir + "/indexes.bin");

        Decompressor decompressor(device);
        torch::Tensor dec_result = decompressor.decompress(
            loaded_latents ,
            loaded_hyper_latents ,
            batch_size ,
            config.n_frame ,
            comp_result
        );

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n ERROR: " << e.what() << std::endl;
        return 1;
    }
}
