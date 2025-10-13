#include <iostream>
#include <fstream>
#include "../CAESAR/models/compress/compressor.h"
#include "../CAESAR/dataset/dataset.h"

void save_tensor_to_bin(const torch::Tensor& tensor, const std::string& filename) {
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    file.write(
        reinterpret_cast<const char*>(cpu_tensor.data_ptr<float>()),
        cpu_tensor.numel() * sizeof(float)
    );
    file.close();
    std::cout << "  Saved: " << filename << std::endl;
}

int main() {
    // Setup device
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
        device = torch::kCUDA;
    } else {
        std::cout << "Using CPU." << std::endl;
    }


    Compressor compressor(device);


    DatasetConfig config;
    config.binary_path = "tensor_data_1.bin";
    config.n_frame = 8;
    config.dataset_name = "Scientific Inference Dataset";
    config.variable_idx = 0;
    config.section_range = {0, 12};
    config.frame_range = {0, 256};
    config.train_mode = false;
    config.inst_norm = true;
    config.norm_type = "mean_range";
    config.test_size = {256, 256};
    config.n_overlap = 0;

    try {

        CompressionResult result = compressor.compress(config, 32);

        std::cout << "\n========== SAVING RESULTS ==========" << std::endl;
        
        for (size_t i = 0; i < result.latents.size(); i++) {
            std::string latent_filename = "compressed_latent_sample_" + std::to_string(i) + ".bin";
            std::string hyper_filename = "compressed_hyper_latent_sample_" + std::to_string(i) + ".bin";
            std::string offset_filename = "offset_sample_" + std::to_string(i) + ".bin";
            std::string scale_filename = "scale_sample_" + std::to_string(i) + ".bin";
            
            save_tensor_to_bin(result.latents[i], latent_filename);
            save_tensor_to_bin(result.hyper_latents[i], hyper_filename);
            save_tensor_to_bin(result.offsets[i], offset_filename);
            save_tensor_to_bin(result.scales[i], scale_filename);
            
            auto idx = result.indices[i];
            std::cout << "  Sample " << i << " indices: [" 
                      << idx[0].item<int64_t>() << ", "
                      << idx[1].item<int64_t>() << ", "
                      << idx[2].item<int64_t>() << ", "
                      << idx[3].item<int64_t>() << "]" << std::endl;
        }

        std::cout << "\n========== SUMMARY ==========" << std::endl;
        std::cout << "Compressed " << result.num_samples << " samples in " 
                  << result.num_batches << " batches" << std::endl;
        std::cout << "Output files:" << std::endl;
        std::cout << "  - compressed_latent_sample_*.bin (" << result.latents.size() << " files)" << std::endl;
        std::cout << "  - compressed_hyper_latent_sample_*.bin (" << result.hyper_latents.size() << " files)" << std::endl;
        std::cout << "  - offset_sample_*.bin (" << result.offsets.size() << " files)" << std::endl;
        std::cout << "  - scale_sample_*.bin (" << result.scales.size() << " files)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during compression: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
