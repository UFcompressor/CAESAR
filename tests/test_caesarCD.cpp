#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include <cmath>
#include <torch/torch.h>
#include "../CAESAR/models/caesar_compress.h"
#include "../CAESAR/models/caesar_decompress.h"
#include "../CAESAR/dataset/dataset.h"


bool save_encoded_streams(const std::vector<std::string>& streams , const std::string& filename) {
    std::ofstream file(filename , std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
        return false;
    }
    for (const auto& s : streams) {
        uint64_t len = static_cast<uint64_t>(s.size());
        file.write(reinterpret_cast<const char*>(&len) , sizeof(len));
        if (len) file.write(s.data() , static_cast<std::streamsize>(len));
    }
    file.close();
    return true;
}


std::vector<std::string> load_encoded_streams(const std::string& filename) {
    std::vector<std::string> out;
    std::ifstream file(filename , std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
        return out;
    }
    uint64_t len;
    while (file.read(reinterpret_cast<char*>(&len) , sizeof(len))) {
        std::string s;
        if (len) {
            s.resize(len);
            if (!file.read(&s[0] , static_cast<std::streamsize>(len))) {
                std::cerr << "Error: truncated read while reading " << filename << std::endl;
                break;
            }
        }
        out.push_back(std::move(s));
    }
    file.close();
    return out;
}


torch::Tensor loadRawBinary(const std::string& bin_path , const std::vector<int64_t>& shape) {
    std::ifstream file(bin_path , std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open binary file: " + bin_path);

    size_t num_elems = 1;
    for (auto d : shape) {
        if (d <= 0) throw std::runtime_error("Invalid shape dimension");
        num_elems *= static_cast<size_t>(d);
    }

    std::vector<float> buf(num_elems);
    file.read(reinterpret_cast<char*>(buf.data()) , static_cast<std::streamsize>(num_elems * sizeof(float)));
    if (!file) throw std::runtime_error("Failed to read expected floats from " + bin_path);
    file.close();

    torch::Tensor t = torch::from_blob(buf.data() , torch::IntArrayRef(shape) , torch::kFloat32).clone();
    std::cout << "Loaded " << bin_path << " with shape " << t.sizes() << "\n";
    std::cout << "  Min: " << t.min().item<float>() << ", Max: " << t.max().item<float>() << "\n";
    return t;
}


void save_tensor_to_bin(const torch::Tensor& tensor , const std::string& filename) {
    torch::Tensor cpu = tensor.to(torch::kCPU).contiguous();
    std::ofstream file(filename , std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << " for write\n";
        return;
    }
    file.write(reinterpret_cast<const char*>(cpu.data_ptr<float>()) , static_cast<std::streamsize>(cpu.numel() * sizeof(float)));
    file.close();
    std::cout << "Saved tensor to " << filename << "\n";
}

// ** JL modified ** //
// Compute metadata bytes //
template<typename T>
size_t get_vector_data_size(const std::vector<T>& vec) {
    if (vec.empty()) {
        return 0;
    }
    return vec.size() * sizeof(T);
}

template<typename T>
size_t get_2d_vector_data_size(const std::vector<std::vector<T>>& vec_2d) {
    size_t total_bytes = 0;
    for (const auto& inner_vec : vec_2d) {
        total_bytes += inner_vec.size() * sizeof(T);
    }
    return total_bytes;
}

size_t calculate_metadata_size(const CompressionResult& result) {
    size_t total_bytes = 0;

    // --- 1. CompressionResult except bitstreams ---
    // std::vector<uint8_t> gae_comp_data
    total_bytes += get_vector_data_size(result.gae_comp_data);
    // double final_nrmse
    total_bytes += sizeof(result.final_nrmse);
    // int num_samples
    total_bytes += sizeof(result.num_samples);
    // int num_batches
    total_bytes += sizeof(result.num_batches);

    // --- 2. CompressionMetaData ---
    const auto& meta = result.compressionMetaData;

    // std::vector<float> offsets
    total_bytes += get_vector_data_size(meta.offsets);
    // std::vector<float> scales
    total_bytes += get_vector_data_size(meta.scales);
    // std::vector<std::vector<int32_t>> indexes
    total_bytes += get_2d_vector_data_size(meta.indexes);
    // std::tuple<int32_t, int32_t, std::vector<int32_t>> block_info
    total_bytes += sizeof(std::get<0>(meta.block_info)); // nH (int32_t)
    total_bytes += sizeof(std::get<1>(meta.block_info)); // nW (int32_t)
    total_bytes += get_vector_data_size(std::get<2>(meta.block_info)); // padding (vector<int32_t>)    
    // std::vector<int32_t> data_input_shape
    total_bytes += get_vector_data_size(meta.data_input_shape);
    // std::vector<std::pair<int32_t, float>> filtered_blocks
    total_bytes += get_vector_data_size(meta.filtered_blocks);
    // float global_scale, float global_offset, int64_t pad_T
    total_bytes += sizeof(meta.global_scale);
    total_bytes += sizeof(meta.global_offset);
    total_bytes += sizeof(meta.pad_T);

    // --- 3. GAEMetaData ---
    const auto& gae_meta = result.gaeMetaData;

    // bool GAE_correction_occur
    total_bytes += sizeof(gae_meta.GAE_correction_occur);
    // std::vector<int> padding_recon_info
    total_bytes += get_vector_data_size(gae_meta.padding_recon_info);
    // std::vector<std::vector<float>> pcaBasis
    total_bytes += get_2d_vector_data_size(gae_meta.pcaBasis);
    // std::vector<float> uniqueVals
    total_bytes += get_vector_data_size(gae_meta.uniqueVals);
    // double quanBin, int64_t nVec, int64_t prefixLength, ...
    total_bytes += sizeof(gae_meta.quanBin);
    total_bytes += sizeof(gae_meta.nVec);
    total_bytes += sizeof(gae_meta.prefixLength);
    total_bytes += sizeof(gae_meta.dataBytes);
    total_bytes += sizeof(gae_meta.coeffIntBytes);

    return total_bytes;
}
// **** //

int main() {
    try {
        // device
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available, using GPU\n";
            device = torch::Device(torch::kCUDA);
        }
        else {
            std::cout << "Using CPU\n";
        }
        //device = torch::Device(torch::kCPU);
        //std::cout << "Device: " << device << std::endl;


        const std::vector<int64_t> shape = { 1, 1, 100, 500, 500 };
        const std::string raw_path = "TCf48.bin.f32";
        const std::string out_dir = "/home/adios/Programs/CAESAR_C/build/tests/output/";
        std::filesystem::create_directories(out_dir);
        const int batch_size = 32;
        const int n_frame = 8;

        torch::Tensor raw = loadRawBinary(raw_path , shape);


        std::cout << "\n===== COMPRESSION =====\n";
        Compressor compressor(device);

        DatasetConfig config;
        config.memory_data = raw;
        config.variable_idx = 0;
        config.n_frame = n_frame;
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

        float rel_eb = 0.1f;
        CompressionResult comp = compressor.compress(config , batch_size , rel_eb);


        std::string latents_file = out_dir + "encoded_latents.bin";
        std::string hyper_file = out_dir + "encoded_hyper_latents.bin";
        if (!save_encoded_streams(comp.encoded_latents , latents_file)) {
            std::cerr << "Failed to save encoded_latents\n";
            return 1;
        }
        if (!save_encoded_streams(comp.encoded_hyper_latents , hyper_file)) {
            std::cerr << "Failed to save encoded_hyper_latents\n";
            return 1;
        }
        std::cout << "Compression finished. Encoded streams written to " << out_dir << "\n";


        uint64_t compressed_bytes = 0;
        for (const auto& s : comp.encoded_latents) compressed_bytes += s.size();
        for (const auto& s : comp.encoded_hyper_latents) compressed_bytes += s.size();

        uint64_t num_elements = 1;
        for (auto d : shape) num_elements *= static_cast<uint64_t>(d);
        uint64_t uncompressed_bytes = num_elements * sizeof(float);

        // ** JL modified ** //
        size_t comp_all_meta_size = calculate_metadata_size(comp);

        double CR = (compressed_bytes > 0) ? static_cast<double>(uncompressed_bytes) / (static_cast<double>(compressed_bytes) + static_cast<double>(comp_all_meta_size)) : 0.0;
        // **** //
        std::cout << "\nCompression stats:\n";
        std::cout << "  - Uncompressed bytes: " << uncompressed_bytes << "\n";
        std::cout << "  - Compressed bytes:   " << compressed_bytes << "\n";
        std::cout << "  - Compression Ratio (CR): " << CR << "\n";


        std::cout << "\n===== DECOMPRESSION =====\n";

        std::vector<std::string> loaded_latents = load_encoded_streams(latents_file);
        std::vector<std::string> loaded_hyper = load_encoded_streams(hyper_file);

        std::vector<torch::Tensor> offsets , scales , indexes;
        {
            const auto& meta = comp.compressionMetaData;

            offsets.reserve(meta.offsets.size());
            scales.reserve(meta.scales.size());
            indexes.reserve(meta.indexes.size());

            for (float v : meta.offsets)
                offsets.push_back(torch::tensor({ v } , torch::kFloat32).to(device));

            for (float v : meta.scales)
                scales.push_back(torch::tensor({ v } , torch::kFloat32).to(device));

            for (const auto& idx_vec : meta.indexes) {
                torch::Tensor idx_tensor = torch::from_blob(
                    const_cast<int32_t*>(idx_vec.data()) ,
                    { (int64_t)idx_vec.size() } ,
                    torch::kInt32
                ).clone().to(device);
                indexes.push_back(idx_tensor);
            }
        }


        Decompressor decompressor(device);
        torch::Tensor recon = decompressor.decompress(
            loaded_latents ,
            loaded_hyper ,
            batch_size ,
            config.n_frame ,
            comp
        );

        // Check if the tensor is empty
        if (!recon.defined() || recon.numel() == 0) {
            std::cerr << "Decompression failed: reconstructed tensor is empty.\n";
            return 1;
        }

        std::cout << "Reconstructed tensor shape: " << recon.sizes() << std::endl;

        int full_frames = 100;
        int full_h = 500;
        int full_w = 500;
        int n_patches = recon.size(0);
        int frames_per_patch = recon.size(2);

        torch::Tensor recon_merged = torch::zeros({ 1, 1, full_frames, full_h, full_w } , recon.options());

        int frame_idx = 0;
        for (int i = 0; i < n_patches && frame_idx < full_frames; ++i) {
            torch::Tensor patch = recon[i];
            int frames_to_copy = std::min(frames_per_patch , full_frames - frame_idx);

            torch::Tensor patch_padded = torch::nn::functional::pad(
                patch ,
                torch::nn::functional::PadFuncOptions({ 0, full_w - 256, 0, full_h - 256 })
            );

            recon_merged.index_put_(
                {
                    0, 0,
                    torch::indexing::Slice(frame_idx, frame_idx + frames_to_copy),
                    torch::indexing::Slice(0, full_h),
                    torch::indexing::Slice(0, full_w)
                } ,
                patch_padded.index({
                    0,
                    torch::indexing::Slice(0, frames_to_copy),
                    torch::indexing::Slice(0, full_h),
                    torch::indexing::Slice(0, full_w)
                    })
            );

            frame_idx += frames_to_copy;
        }

        std::cout << "Stitched reconstructed tensor to shape: "
            << recon_merged.sizes() << std::endl;

        torch::Tensor raw_cpu = raw.to(torch::kCPU);
        torch::Tensor recon_cpu = recon_merged.to(torch::kCPU);
        torch::Tensor diff = recon_cpu - raw_cpu;
        double mse = diff.pow(2).mean().item<double>();
        double rmse = std::sqrt(mse);
        double nrmse = rmse / (raw_cpu.max().item<double>() - raw_cpu.min().item<double>());

        std::cout << "=== Quality Metrics ===" << std::endl;
        std::cout << "NRMSE: " << nrmse << std::endl;
        std::cout << "Compression Ratio (CR): " << CR << std::endl;

        std::cout << "Decompression finished. Reconstructed data shape: " << recon_merged.sizes() << "\n";
        return 0;

    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
