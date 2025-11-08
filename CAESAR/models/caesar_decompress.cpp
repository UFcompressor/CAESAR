#include "caesar_decompress.h"
#include "range_coder/rans_coder.hpp"
#include <iostream>
#include <fstream>

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

// Decompressor Implementation
Decompressor::Decompressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Decompressor::load_models() {
    std::cout << "Loading decompressor models..." << std::endl;
    hyper_decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/adios/Programs/CAESAR_C/exported_model/caesar_hyper_decompressor.pt2"
    );
    decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        "/home/adios/Programs/CAESAR_C/exported_model/caesar_decompressor.pt2"
    );
    std::cout << "Models loaded successfully." << std::endl;
}

void Decompressor::load_probability_tables() {
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

torch::Tensor Decompressor::reshape_batch_2d_3d(
    const torch::Tensor& batch_data ,
    int64_t batch_size ,
    int64_t n_frame
) {
    // Input: [B*T, C, H, W] where T = n_frame
    // Output: [B, C, T, H, W]

    auto sizes = batch_data.sizes();
    std::cout << "  [reshape_batch_2d_3d] Input shape: " << sizes << std::endl;
    std::cout << "  [reshape_batch_2d_3d] batch_size: " << batch_size << ", n_frame: " << n_frame << std::endl;

    TORCH_CHECK(sizes.size() == 4 , "Input tensor must be 4-dimensional.");

    int64_t BT = sizes[0];  // B * T
    int64_t C = sizes[1];   // Channel
    int64_t H = sizes[2];   // Height
    int64_t W = sizes[3];   // Width

    int64_t T = BT / batch_size;  // Frames per sample

    std::cout << "  [reshape_batch_2d_3d] Calculated T (frames per sample): " << T << std::endl;
    std::cout << "  [reshape_batch_2d_3d] Expected T (n_frame): " << n_frame << std::endl;

    if (T != n_frame) {
        std::cout << "  [reshape_batch_2d_3d] WARNING: Calculated T != n_frame!" << std::endl;
    }

    // Reshape to [B, T, C, H, W]
    torch::Tensor reshaped_data = batch_data.view({ batch_size, T, C, H, W });
    std::cout << "  [reshape_batch_2d_3d] After view: " << reshaped_data.sizes() << std::endl;

    // Permute to [B, C, T, H, W]
    torch::Tensor permuted_data = reshaped_data.permute({ 0, 2, 1, 3, 4 });
    std::cout << "  [reshape_batch_2d_3d] After permute (output): " << permuted_data.sizes() << std::endl;

    return permuted_data;
}

DecompressionResult Decompressor::decompress(
    const std::vector<std::string>& encoded_latents ,
    const std::vector<std::string>& encoded_hyper_latents ,
    int batch_size ,
    int n_frame
) {
    c10::InferenceMode guard;

    std::cout << "\n========== STARTING DECOMPRESSION ==========" << std::endl;
    std::cout << "Device: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "N_frame: " << n_frame << std::endl;
    std::cout << "Total encoded_latents: " << encoded_latents.size() << std::endl;
    std::cout << "Total encoded_hyper_latents: " << encoded_hyper_latents.size() << std::endl;

    // CRITICAL: Check if the number of latents matches expectations
    std::cout << "\n*** DIMENSION CHECK ***" << std::endl;
    std::cout << "Expected pattern: Each input sample (B, 1, " << n_frame << ", 256, 256)" << std::endl;
    std::cout << "                  should produce (B*2) latent codes" << std::endl;
    std::cout << "If you compressed " << (encoded_latents.size() / 2) << " samples, "
        << "you should have " << encoded_latents.size() << " latent codes" << std::endl;

    DecompressionResult result;
    result.num_samples = 0;
    result.num_batches = 0;

    RansDecoder range_decoder;

    // Process in batches
    for (size_t batch_start = 0; batch_start < encoded_latents.size(); batch_start += batch_size) {
        size_t batch_end = std::min(batch_start + batch_size , encoded_latents.size());
        size_t current_batch_size = batch_end - batch_start;

        std::cout << "\n========================================" << std::endl;
        std::cout << "--- Decompressing Batch " << result.num_batches << " ---" << std::endl;
        std::cout << "Latent codes in this batch: " << current_batch_size << std::endl;
        std::cout << "Expected output samples: " << (current_batch_size / 2) << std::endl;
        std::cout << "========================================" << std::endl;

        // Decode hyper latents
        //  here -----------------------------------------------------------------------------------------------------------------------
        std::vector<int32_t> hyper_size = {
            static_cast<int32_t>(current_batch_size), 64, 4, 4
        };
        std::cout << "\n[HYPER DECODE] Building indexes with size: ["
            << hyper_size[0] << ", " << hyper_size[1] << ", "
            << hyper_size[2] << ", " << hyper_size[3] << "]" << std::endl;

        torch::Tensor hyper_index_tensor = build_indexes_tensor(hyper_size);
        torch::Tensor decoded_hyper_latents = torch::zeros(
            { static_cast<int64_t>(current_batch_size), 64, 4, 4 }
        ).to(torch::kInt32);

        std::cout << "[HYPER DECODE] Decoding " << current_batch_size << " hyper-latent codes..." << std::endl;

        for (size_t i = 0; i < current_batch_size; i++) {
            torch::Tensor hyper_index_slice = hyper_index_tensor.select(0 , i);
            std::vector<int32_t> hyper_index_vec = tensor_to_vector<int32_t>(
                hyper_index_slice.reshape(-1)
            );

            std::vector<int32_t> hyper_decoded = range_decoder.decode_with_indexes(
                encoded_hyper_latents[batch_start + i] ,
                hyper_index_vec ,
                vbr_quantized_cdf_ ,
                vbr_cdf_length_ ,
                vbr_offset_
            );

            torch::Tensor hyper_tensor = torch::tensor(hyper_decoded).reshape({ 64, 4, 4 });
            decoded_hyper_latents.select(0 , i).copy_(hyper_tensor);
        }
        std::cout << "[HYPER DECODE] Decoded hyper_latents shape: " << decoded_hyper_latents.sizes() << std::endl;

        // Run hyper decompressor
        std::cout << "\n[HYPER DECOMPRESS] Running hyper decompressor..." << std::endl;
        std::vector<torch::Tensor> hyper_inputs = {
            decoded_hyper_latents.to(torch::kFloat32).to(device_)
        };
        std::vector<torch::Tensor> hyper_outputs = hyper_decompressor_model_->run(hyper_inputs);

        torch::Tensor mean = hyper_outputs[0];
        torch::Tensor latent_indexes_recon = hyper_outputs[1];
        std::cout << "[HYPER DECOMPRESS] Mean shape: " << mean.sizes() << std::endl;
        std::cout << "[HYPER DECOMPRESS] Latent indexes shape: " << latent_indexes_recon.sizes() << std::endl;
        // --------------------------------------------------------------------------------------------------------------------------------
        // Decode latents
        std::cout << "\n[LATENT DECODE] Decoding " << current_batch_size << " latent codes..." << std::endl;
        torch::Tensor decoded_latents_before_offset = torch::zeros(
            { static_cast<int64_t>(current_batch_size), 64, 16, 16 }
        ).to(torch::kInt32);

        for (size_t i = 0; i < current_batch_size; i++) {
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(
                latent_indexes_recon.select(0 , i).reshape(-1)
            );

            std::vector<int32_t> latent_decoded = range_decoder.decode_with_indexes(
                encoded_latents[batch_start + i] ,
                latent_index ,
                gs_quantized_cdf_ ,
                gs_cdf_length_ ,
                gs_offset_
            );

            torch::Tensor latent_tensor = torch::tensor(latent_decoded).reshape({ 64, 16, 16 });
            decoded_latents_before_offset.select(0 , i).copy_(latent_tensor);
        }

        torch::Tensor decoded_latents =
            decoded_latents_before_offset.to(device_).to(torch::kFloat32) + mean;
        std::cout << "[LATENT DECODE] Decoded latents shape (before reshape): " << decoded_latents.sizes() << std::endl;

        // Reshape for decompressor
        std::cout << "\n[RESHAPE] Preparing latents for decompressor..." << std::endl;
        auto original_sizes = decoded_latents.sizes();
        std::cout << "[RESHAPE] Original decoded_latents shape: " << original_sizes << std::endl;

        std::vector<int64_t> new_shape = { -1, 2 };
        new_shape.insert(new_shape.end() , original_sizes.begin() + 1 , original_sizes.end());

        std::cout << "[RESHAPE] New shape vector: [";
        for (size_t i = 0; i < new_shape.size(); i++) {
            std::cout << new_shape[i];
            if (i < new_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        torch::Tensor reshaped_latents = decoded_latents.reshape(new_shape);
        std::cout << "[RESHAPE] Reshaped latents: " << reshaped_latents.sizes() << std::endl;
        std::cout << "[RESHAPE] This means " << reshaped_latents.sizes()[0]
            << " pairs of latent codes will be processed" << std::endl;

  // Run decompressor
        std::cout << "\n[DECOMPRESS] Running main decompressor..." << std::endl;
        std::vector<torch::Tensor> decompressor_inputs = { reshaped_latents };
        std::vector<torch::Tensor> decompressor_outputs = decompressor_model_->run(decompressor_inputs);

        torch::Tensor raw_output = decompressor_outputs[0];
        std::cout << "[DECOMPRESS] Raw output shape: " << raw_output.sizes() << std::endl;

        // Calculate how many original samples this represents
        int64_t num_pairs = reshaped_latents.sizes()[0];
        int64_t original_num_samples = num_pairs;  // Each pair = 1 original sample

        std::cout << "\n[OUTPUT RESHAPE] Reshaping output..." << std::endl;
        std::cout << "[OUTPUT RESHAPE] Number of original samples: " << original_num_samples << std::endl;
        std::cout << "[OUTPUT RESHAPE] Target shape: [" << original_num_samples
            << ", 1, " << n_frame << ", 256, 256]" << std::endl;

  // Reshape output
        torch::Tensor output = reshape_batch_2d_3d(
            raw_output ,
            original_num_samples ,
            n_frame
        );

        std::cout << "[OUTPUT Check - Before Denormalization] Final output max min: " << output.max().item<float>() << ", " << output.min().item<float>() << std::endl;


        std::cout << "[OUTPUT RESHAPE] Final output shape: " << output.sizes() << std::endl;

        // Split batch into individual samples
        std::cout << "\n[SPLIT] Splitting into individual samples..." << std::endl;
        for (int64_t i = 0; i < original_num_samples; i++) {
            torch::Tensor sample = output.select(0 , i).unsqueeze(0);
            std::cout << "  Sample " << result.num_samples << " shape: " << sample.sizes() << std::endl;
            result.reconstructed_data.push_back(sample.clone());
            result.num_samples++;
        }

        result.num_batches++;
    }

    std::cout << "\n========== DECOMPRESSION COMPLETE ==========" << std::endl;
    std::cout << "Total samples decompressed: " << result.num_samples << std::endl;
    std::cout << "Total batches processed: " << result.num_batches << std::endl;
    return result;
}
