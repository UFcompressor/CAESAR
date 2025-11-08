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
        uint32_t length = stream.length();
        file.write(reinterpret_cast<const char*>(&length) , sizeof(length));
        file.write(stream.data() , length);
    }

    file.close();
    return true;
}

// Helper function to load encoded streams from file
std::vector<std::string> load_encoded_streams(const std::string& filename) {
    std::vector<std::string> results;
    std::ifstream file(filename , std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
        return results;
    }

    uint32_t length;
    while (file.read(reinterpret_cast<char*>(&length) , sizeof(length))) {
        if (length == 0) {
            results.push_back("");
        }
        else {
            std::string data(length , '\0');
            if (!file.read(&data[0] , length)) {
                std::cerr << "Error: Truncated file or read error!" << std::endl;
                break;
            }
            results.push_back(data);
        }
    }

    file.close();
    return results;
}

// Helper function to save tensor to binary file
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

// Function to load raw float32 binary into a tensor with specified shape
torch::Tensor loadRawBinary(const std::string& bin_path , const std::vector<int64_t>& shape) {
    std::ifstream file(bin_path , std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + bin_path);
    }

    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }

    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()) , num_elements * sizeof(float));
    file.close();

    torch::Tensor data = torch::from_blob(
        buffer.data() ,
        shape ,
        torch::kFloat32
    ).clone();

    std::cout << "Loaded " << bin_path << " with shape: " << data.sizes() << "\n";
    std::cout << "  Min: " << data.min().item<float>() << ", Max: " << data.max().item<float>() << "\n";

    return data;
}

int main() {
    try {
        // Determine device
        auto device = torch::kCPU;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Using GPU." << std::endl;
            device = torch::kCUDA;
        }
        else {
            std::cout << "Using CPU." << std::endl;
        }

        // ========== COMPRESSION PHASE ==========
        std::cout << "\n" << std::string(60 , '=') << std::endl;
        std::cout << "PHASE 1: COMPRESSION" << std::endl;
        std::cout << std::string(60 , '=') << "\n" << std::endl;

        // Create compressor
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

        // Compress
        int batch_size = 32;
        float rel_eb = 0.001;
        CompressionResult comp_result = compressor.compress(config , batch_size, rel_eb);

        // Save compressed data
        std::cout << "\nSaving compressed data..." << std::endl;
        save_encoded_streams(comp_result.encoded_latents , "/home/jlx/Projects/CAESAR_ALL/CAESAR_C/build/tests/output/encoded_latents.bin");
        save_encoded_streams(comp_result.encoded_hyper_latents , "/home/jlx/Projects/CAESAR_ALL/CAESAR_C/build/tests/output/encoded_hyper_latents.bin");
        std::cout << "Compressed data saved to output/" << std::endl;

        // Calculate compression statistics
        size_t total_compressed_bytes = 0;
        for (const auto& stream : comp_result.encoded_latents) {
            total_compressed_bytes += stream.size();
        }
        for (const auto& stream : comp_result.encoded_hyper_latents) {
            total_compressed_bytes += stream.size();
        }

        std::cout << "\nCompression Statistics:" << std::endl;
        std::cout << "  - Total samples: " << comp_result.num_samples << std::endl;
        std::cout << "  - Total batches: " << comp_result.num_batches << std::endl;
        std::cout << "  - Compressed size: " << total_compressed_bytes << " bytes" << std::endl;

        // ========== DECOMPRESSION PHASE ==========
        std::cout << "\n" << std::string(60 , '=') << std::endl;
        std::cout << "PHASE 2: DECOMPRESSION" << std::endl;
        std::cout << std::string(60 , '=') << "\n" << std::endl;

        // Load compressed data
        std::cout << "Loading compressed data..." << std::endl;
        std::vector<std::string> loaded_latents = load_encoded_streams("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/build/tests/output/encoded_latents.bin");
        std::vector<std::string> loaded_hyper_latents = load_encoded_streams("/home/jlx/Projects/CAESAR_ALL/CAESAR_C/build/tests/output/encoded_hyper_latents.bin");
        std::cout << "Loaded " << loaded_latents.size() << " latent streams and "
            << loaded_hyper_latents.size() << " hyper-latent streams" << std::endl;

  // Verify loaded data matches original
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

        if (data_matches) {
            std::cout << "✓ Verification passed: Loaded data matches compressed data" << std::endl;
        }
        else {
            std::cerr << "✗ Verification failed: Loaded data does not match!" << std::endl;
            return 1;
        }

        return 0;

    }
        
    catch (const std::exception& e) {
        std::cerr << "\n ERROR: " << e.what() << std::endl;
        return 1;
    }
}