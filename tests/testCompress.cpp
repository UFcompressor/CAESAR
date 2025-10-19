#include <iostream>
#include <fstream>
#include "../CAESAR/models/compress/compressor.h"
#include "../CAESAR/dataset/dataset.h"

// Struct for tensor metadata
struct TensorMeta {
    std::string name;
    uint64_t offset;
    uint64_t num_bytes;
    std::vector<int64_t> shape;
};

// Append tensor data to the data.bin file and return bytes written
size_t append_tensor(std::ofstream& binfile , const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
    size_t num_bytes = cpu_tensor.numel() * sizeof(float);
    binfile.write(reinterpret_cast<const char*>(cpu_tensor.data_ptr<float>()) , num_bytes);
    return num_bytes;
}

int main() {
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
        device = torch::kCUDA;
    }
    else {
        std::cout << "Using CPU." << std::endl;
    }

    Compressor compressor(device);

    DatasetConfig config;
    config.binary_path = "tensor_data_1.bin";
    config.n_frame = 8;
    config.dataset_name = "Scientific Inference Dataset";
    config.variable_idx = 0;
    config.section_range = { 0, 13};
    config.frame_range = { 0, 256 };
    config.train_mode = false;
    config.inst_norm = true;
    config.norm_type = "mean_range";
    config.test_size = { 256, 256 };
    config.n_overlap = 0;

    try {
        CompressionResult result = compressor.compress(config , 32);

        std::cout << "\n========== SAVING RESULTS ==========" << std::endl;


        std::ofstream data_out("data.bin" , std::ios::binary);
        std::ofstream meta_out("meta.bin" , std::ios::binary);

        if (!data_out.is_open() || !meta_out.is_open()) {
            std::cerr << "Error: could not open output files.\n";
            return 1;
        }

        std::vector<TensorMeta> metadata;
        uint64_t offset = 0;

        // Write tensors into one data.bin file
        for (size_t i = 0; i < result.latents.size(); i++) {
            auto add_tensor = [&](const std::string& name , const torch::Tensor& tensor) {
                size_t bytes_written = append_tensor(data_out , tensor);
                TensorMeta meta{ name, offset, bytes_written, tensor.sizes().vec() };
                metadata.push_back(meta);
                offset += bytes_written;
                };

            add_tensor("compressed_latent_sample_" + std::to_string(i) , result.latents[i]);
            add_tensor("compressed_hyper_latent_sample_" + std::to_string(i) , result.hyper_latents[i]);
            add_tensor("offset_sample_" + std::to_string(i) , result.offsets[i]);
            add_tensor("scale_sample_" + std::to_string(i) , result.scales[i]);

            auto idx = result.indices[i];
            std::cout << "  Sample " << i << " indices: ["
                << idx[0].item<int64_t>() << ", "
                << idx[1].item<int64_t>() << ", "
                << idx[2].item<int64_t>() << ", "
                << idx[3].item<int64_t>() << "]" << std::endl;
        }

        // Write metadata to meta.bin
        uint32_t num_tensors = metadata.size();
        meta_out.write(reinterpret_cast<const char*>(&num_tensors) , sizeof(num_tensors));

        for (const auto& m : metadata) {
            uint32_t name_len = m.name.size();
            meta_out.write(reinterpret_cast<const char*>(&name_len) , sizeof(name_len));
            meta_out.write(m.name.c_str() , name_len);
            meta_out.write(reinterpret_cast<const char*>(&m.offset) , sizeof(m.offset));
            meta_out.write(reinterpret_cast<const char*>(&m.num_bytes) , sizeof(m.num_bytes));

            uint32_t ndim = m.shape.size();
            meta_out.write(reinterpret_cast<const char*>(&ndim) , sizeof(ndim));
            for (auto d : m.shape)
                meta_out.write(reinterpret_cast<const char*>(&d) , sizeof(d));
        }

        data_out.close();
        meta_out.close();

        std::cout << "\n Saved all tensor data to 'data.bin' and metadata to 'meta.bin'\n";

        std::cout << "\n========== SUMMARY ==========" << std::endl;
        std::cout << "Compressed " << result.num_samples << " samples in "
            << result.num_batches << " batches" << std::endl;
        std::cout << "Output files:\n"
            << "  - data.bin (all tensor data)\n"
            << "  - meta.bin (binary metadata)\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Error during compression: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
