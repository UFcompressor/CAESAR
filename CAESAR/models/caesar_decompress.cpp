#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint> // Include for uint32_t
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <filesystem> 
#include <system_error>
#include "range_coder/rans_coder.hpp" // Assuming this header contains RansEncoder/RansDecoder

/**
 * @brief Creates a flat std::vector<int32_t> containing channel indexes,
 * replicating the behavior of the Python code:
 * torch.arange(C).view(1, C, 1, ...).expand(N, C, H, W, ...).int()
 *
 * @param size A vector describing the desired shape (e.g., {N, C, H, W}).
 * Must contain at least 2 dimensions (N and C).
 * @return A std::vector<int32_t> of size (N*C*H*W*...) filled with channel
 * indices in C-major (row-major) layout.
 */
std::vector<int32_t> build_indexes_vector(const std::vector<int32_t>& size) {
    
    // 1. Check for at least 2 dimensions (N, C)
    if (size.size() < 2) {
        std::cerr << "Error: 'size' vector must have at least 2 dimensions (N, C)." << std::endl;
        return {}; // Return an empty vector on error
    }

    // 2. Get N and C from the input size vector.
    int32_t N = size[0];
    int32_t C = size[1];

    // 3. Calculate the total size of the inner dimensions (H*W*...)
    int64_t inner_size = 1;
    for (size_t i = 2; i < size.size(); ++i) {
        inner_size *= size[i];
    }
    
    // 4. Calculate the total_size and reserve memory for the vector.
    int64_t total_size = N * C * inner_size;
    std::vector<int32_t> indexes;
    
    // reserve() expects a size_t
    indexes.reserve(static_cast<size_t>(total_size)); 

    // 5. Fill the vector by iterating in (N, C, inner_size) order.
    //    The innermost loops just repeat the channel index 'c'.
    for (int32_t n = 0; n < N; ++n) {       // Loop N times
        for (int32_t c = 0; c < C; ++c) {   // Loop C times
            
            // Loop inner_size (H*W*...) times
            for (int64_t i = 0; i < inner_size; ++i) {
                // Push the current channel index 'c'
                indexes.push_back(c);
            }
        }
    }

    return indexes;
}

/**
 * @brief Python's _build_indexes(size)
 */
torch::Tensor build_indexes_tensor(const std::vector<int32_t>& size) {
    // 1. dims = len(size)
    int64_t dims = size.size();

    // 2. C = size[1] (Channel)
    TORCH_CHECK(dims >= 2, "Input size must have at least 2 dimensions (N, C, ...)");
    int64_t C = size[1]; // Use int64_t for arange

    // 3. view_dims = [1, C] + [1] * (dims - 2)
    std::vector<int64_t> view_dims = {1, C};
    view_dims.insert(view_dims.end(), dims - 2, 1);
    
    // 4. indexes = torch.arange(C).view(*view_dims)
    torch::Tensor indexes = torch::arange(C).view(view_dims);
    
    // 5. return indexes.expand(*size).int()
    //    Convert int32_t size vector to int64_t vector for .expand()
    std::vector<int64_t> size_int64(size.begin(), size.end());
    return indexes.expand(size_int64).to(torch::kInt32);
}

torch::Tensor reshape_batch_2d_3d(const torch::Tensor& batch_data, int64_t batch_size) {
    
    // 1. BT,C,H,W = batch_data.shape
    auto sizes = batch_data.sizes(); // 텐서의 shape (at::IntArrayRef)

    // (방어 코드) 입력이 4D 텐서인지 확인
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4-dimensional.");

    int64_t BT = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    // 2. T = BT//batch_size
    int64_t T = BT / batch_size; // C++의 정수 나눗셈은 Python의 //와 동일

    // 3. batch_data = batch_data.view([batch_size, T, C, H, W])
    //    .view()는 C++에서 std::vector<int64_t> 또는 {} 초기화 리스트를 받습니다.
    torch::Tensor reshaped_data = batch_data.view({batch_size, T, C, H, W});

    // 4. batch_data = batch_data.permute([0,2,1,3,4])
    //    .permute()도 동일하게 {0, 2, 1, 3, 4}를 전달합니다.
    torch::Tensor permuted_data = reshaped_data.permute({0, 2, 1, 3, 4});

    // 5. return batch_data
    return permuted_data;
}

// Function to load a tensor from a binary file (.bin)
torch::Tensor load_tensor_from_bin(const std::string& filename, const std::vector<int64_t>& shape) {
    // 1. Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    // Get the file size
    file.seekg(0, std::ios::end); // Move the file pointer to the end
    long long file_size = file.tellg(); // Current position (file size)
    file.seekg(0, std::ios::beg); // Move the file pointer back to the beginning
    
    // Calculate the required size in bytes based on shape and element type (float32)
    long long required_bytes = 1;
    for (auto dim : shape) {
        required_bytes *= dim;
    }
    required_bytes *= sizeof(float);

    // Validate file size
    if (file_size != required_bytes) {
        throw std::runtime_error("Error: File size mismatch for " + filename);
    }

    // 4. Prepare a std::vector of the correct size to hold the data
    std::vector<float> data_vec(file_size / sizeof(float));
    // 5. Read the entire file into the vector's memory at once
    file.read(reinterpret_cast<char*>(data_vec.data()), file_size);
    // 6. Close the file
    file.close();

    // Create a tensor from the blob, clone it to own the memory, and ensure it's float32
    return torch::from_blob(data_vec.data(), shape, torch::kFloat32).clone();
}

// Template function to load an array (vector) of type T from a binary file
template<typename T>
std::vector<T> load_array_from_bin(const std::string& filename) {
    // 1. Open the file in binary mode
    std::ifstream input_file(filename, std::ios::binary);

    // 2. Check if the file was opened successfully
    if (!input_file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        // Return an empty vector on failure
        return std::vector<T>();
    }

    // 3. Get the file size to calculate the number of elements
    input_file.seekg(0, std::ios::end); // Move the file pointer to the end
    size_t file_size_in_bytes = input_file.tellg(); // Current position (file size)
    input_file.seekg(0, std::ios::beg); // Move the file pointer back to the beginning
    
    size_t num_elements = file_size_in_bytes / sizeof(T);

    // 4. Prepare a std::vector of the correct size to hold the data
    std::vector<T> loaded_data(num_elements);

    // 5. Read the entire file into the vector's memory at once
    input_file.read(reinterpret_cast<char*>(loaded_data.data()), file_size_in_bytes);

    // 6. Close the file
    input_file.close();

    // 7. Return the vector filled with data
    return loaded_data;
}

// Template function to reshape a 1D vector into a 2D vector
template<typename T>
std::vector<std::vector<T>> reshape_to_2d(const std::vector<T>& flat_vec, size_t rows, size_t cols) {
    // Size validation
    if (flat_vec.size() != rows * cols) {
        throw std::invalid_argument("Invalid dimensions for reshape.");
    }

    std::vector<std::vector<T>> vec_2d;
    vec_2d.reserve(rows); // Pre-allocate memory for the outer vector

    auto it = flat_vec.begin(); // Iterator for the 1D vector

    for (size_t r = 0; r < rows; ++r) {
        // Create a new row (vector) using the iterator range and add it to the 2D vector
        // std::vector<T> new_row(it, it + cols);
        // vec_2d.push_back(new_row);
        
        // Combine the above two lines into one (can be more efficient)
        vec_2d.emplace_back(it, it + cols);

        // Move the iterator to the beginning of the next row
        it += cols;
    }

    return vec_2d;
}

// Template function to convert a torch::Tensor to a std::vector<T>
template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
    // .contiguous() ensures the memory is contiguous after slicing/reshaping, making .data_ptr() safe.
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();

    // Get data pointer of type T*
    const T* tensor_data_ptr = cpu_tensor.data_ptr<T>();

    int64_t num_elements = cpu_tensor.numel();

    // Create a std::vector<T> from the data pointer
    std::vector<T> symbol(tensor_data_ptr, tensor_data_ptr + num_elements);

    // Return the vector
    return symbol;
}

// Function to save a tensor to a binary file (.bin)
void save_tensor_to_bin(const torch::Tensor& tensor, const std::string& filename) {
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    file.write(
        reinterpret_cast<const char*>(cpu_tensor.data_ptr<float>()), // Data pointer
        cpu_tensor.numel() * sizeof(float) // Total size in bytes
    );
    file.close();
    std::cout << "  - Tensor saved to " << filename << std::endl;
}

// Function to save a bitstream (std::string) to a file
bool save_bitstream_to_file(const std::string& filename, const std::string& bitstream) {
    // 1. Open the file in binary write mode.
    //    std::ios::out is default, but explicit specification with binary is good practice.
    std::ofstream output_file(filename, std::ios::out | std::ios::binary);

    // 2. Check if the file was opened successfully.
    if (!output_file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    // 3. Use the string's .write() function.
    //    .data() returns a (const char*) pointer to the string's data.
    //    .size() returns the exact byte length of the string.
    //    This method correctly writes the entire data even if it contains null characters (\0).
    output_file.write(bitstream.data(), bitstream.size());

    // 4. (Optional) Check for write errors.
    if (output_file.fail()) {
        std::cerr << "Error: Failed to write data to " << filename << std::endl;
        output_file.close(); // It's good practice to close even if an error occurred.
        return false;
    }

    // 5. Close the file.
    //    (Actually, it closes automatically when the output_file object goes out of scope - RAII)
    output_file.close();
    
    return true;
}

// Function to save a vector of encoded streams to a single file
bool save_encoded_streams(const std::vector<std::string>& streams, const std::string& filename) {
    // 1. Open the file in binary write mode.
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
        return false;
    }

    // 2. Iterate through the vector, writing length (uint32_t) and data (char*) sequentially.
    for (const std::string& stream : streams) {
        // 2a. Write the string length as 4 bytes (uint32_t).
        uint32_t length = stream.length();
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));

        // 2b. Write the actual string data.
        file.write(stream.data(), length);
    }

    file.close();
    return true;
}

// Function to load a vector of encoded streams from a single file
std::vector<std::string> load_encoded_streams(const std::string& filename) {
    std::vector<std::string> results;

    // 1. Open the file in binary read mode.
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
        return results; // Return an empty vector
    }

    uint32_t length;

    // 2. Loop reading [length] -> [data] until the end of the file (EOF) is reached.
    //    file.read() returns true on success, false if EOF is reached before reading 4 bytes for length.
    while (file.read(reinterpret_cast<char*>(&length), sizeof(length))) {
        
        if (length == 0) {
            // Handle zero-length strings (empty strings) as well
            results.push_back("");
        } else {
            // 3. Pre-allocate the string with the read length.
            std::string data(length, '\0');

            // 4. Read file data directly into the allocated string's internal buffer.
            if (!file.read(&data[0], length)) {
                // If length was read but data is insufficient (corrupted file)
                std::cerr << "Error: Truncated file or read error!" << std::endl;
                break; // Break the loop
            }
            results.push_back(data);
        }
    }

    file.close();
    return results;
}

// Custom Dataset to load pre-batched tensors from binary files
class CustomDataset : public torch::data::Dataset<CustomDataset> {
public:
    CustomDataset() {
        std::cout << "Loading dataset files..." << std::endl;
        // Load the two pre-batched tensors
        tensors_.push_back(load_tensor_from_bin("./data/TCf48_batch_0.bin", {32, 1, 8, 256, 256}));
        tensors_.push_back(load_tensor_from_bin("./data/TCf48_batch_1.bin", {20, 1, 8, 256, 256}));
        std::cout << "Dataset loaded successfully." << std::endl;
    }

    // Returns the pre-batched tensor at the given index
    torch::data::Example<> get(size_t index) override {
        // Return an empty tensor as the target (label) is not needed
        return {tensors_[index], torch::empty(0)};
    }

    // Returns the total number of pre-batched tensors (which is 2 in this case)
    torch::optional<size_t> size() const override {
        return tensors_.size();
    }

private:
    std::vector<torch::Tensor> tensors_; // Stores the loaded pre-batched tensors
};

// Custom Dataset designed to hold already formed batches
class PreBatchedDataset : public torch::data::Dataset<PreBatchedDataset, torch::data::Example<>> {
public:
    // Constructor: Takes a vector of 'pre-batched' tensors.
    PreBatchedDataset(std::vector<torch::Tensor> pre_batched_tensors)
        : tensors_(std::move(pre_batched_tensors)) {}

    // get() returns one entire 'pre-batched' batch.
    torch::data::Example<> get(size_t index) override {
        // Returns an empty tensor if no target is needed.
        return {tensors_[index], torch::empty(0)};
    }

    // size() is the total number of stored 'pre-batched' batches.
    torch::optional<size_t> size() const override {
        return tensors_.size();
    }

private:
    // This vector stores tensors like [0] = [64, ...], [1] = [40, ...].
    std::vector<torch::Tensor> tensors_;
};


int main(int argc, char* argv[]) {
    c10::InferenceMode mode; // RAII guard for inference mode

    // Create output directory
    std::filesystem::path dir_path = "./output/";

    try {
        bool created = std::filesystem::create_directories(dir_path);

    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error in creating the output directory: " << e.what() << std::endl;
        return 1;
    }

    // Set device (prefer CUDA if available)
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        device = torch::kCUDA;
    } else {
        std::cout << "Running on CPU." << std::endl;
    }
    
    // Load the exported AOTInductor models
    torch::inductor::AOTIModelPackageLoader hyper_decompressor("exported_model/caesar_hyper_decompressor.pt2");
    torch::inductor::AOTIModelPackageLoader decompressor("exported_model/caesar_decompressor.pt2");
    std::cout << "Models loaded successfully." << std::endl;

    // Initialize Range Coder instances
    RansDecoder range_decoder;

    // Load pre-computed CDF tables and related info from binary files
    std::vector<int32_t> vbr_quantized_cdf_1d = load_array_from_bin<int32_t>("exported_model/vbr_quantized_cdf.bin");
    std::vector<int32_t> vbr_cdf_length = load_array_from_bin<int32_t>("exported_model/vbr_cdf_length.bin");
    std::vector<int32_t> vbr_offset = load_array_from_bin<int32_t>("exported_model/vbr_offset.bin");

    std::vector<int32_t> gs_quantized_cdf_1d = load_array_from_bin<int32_t>("exported_model/gs_quantized_cdf.bin");
    std::vector<int32_t> gs_cdf_length = load_array_from_bin<int32_t>("exported_model/gs_cdf_length.bin");
    std::vector<int32_t> gs_offset = load_array_from_bin<int32_t>("exported_model/gs_offset.bin");
    
    // Reshape the loaded 1D CDF arrays into 2D vectors
    std::vector<std::vector<int32_t>> vbr_quantized_cdf = reshape_to_2d(vbr_quantized_cdf_1d, 64, 63); // Assuming 64 channels, length 63
    std::vector<std::vector<int32_t>> gs_quantized_cdf = reshape_to_2d(gs_quantized_cdf_1d, 128, 249); // Assuming 128 channels, length 249
    
    // Free memory of the temporary 1D vectors
    vbr_quantized_cdf_1d.clear();
    vbr_quantized_cdf_1d.shrink_to_fit();
    gs_quantized_cdf_1d.clear();
    gs_quantized_cdf_1d.shrink_to_fit();
    
    // --- Decompression Stage ---

    // Entropy decode hyper latent and latent and prepare for dataset for main decompressor network
    std::vector<torch::Tensor> batches;

    int batch_index = 2;
    for (int b =0; b < batch_index; b++){
        std::vector<std::string> loaded_latents = load_encoded_streams("output/encoded_latents_batch_" + std::to_string(b) + ".bin");
        std::vector<std::string> loaded_hyper_latents = load_encoded_streams("output/encoded_hyper_latents_batch_" + std::to_string(b) + ".bin");
        std::cout << "\nloaded_latents and loaded_hyer_latents size of batch " << b << ": " << loaded_latents.size() << ", " << loaded_hyper_latents.size()  << std::endl;

        std::vector<int32_t> hyper_size = {static_cast<int32_t>(loaded_hyper_latents.size()), 64, 4, 4};
        std::cout << "hyper size: " << hyper_size << std::endl;
        torch::Tensor hyper_index_tensor = build_indexes_tensor(hyper_size);
        
        torch::Tensor decoded_hyper_latents = torch::zeros({static_cast<int64_t>(loaded_hyper_latents.size()),64,4,4}).to(torch::kInt32);
        //torch::Tensor decoded_hyper_latents = torch::zeros(hyper_size).to(torch::kInt32);
        
        for (int i = 0; i < loaded_hyper_latents.size(); i++){
            
            torch::Tensor hyper_index_slice_tensor = hyper_index_tensor.select(0, i);
            std::vector<int32_t> hyper_index_slice_vec = tensor_to_vector<int32_t>(hyper_index_slice_tensor.reshape(-1));

            std::vector<int32_t> hyper_decoded_output = range_decoder.decode_with_indexes(
                loaded_hyper_latents[i],
                hyper_index_slice_vec,
                vbr_quantized_cdf,
                vbr_cdf_length,
                vbr_offset
            );
            
            // Store decoded hyper latents
            torch::Tensor hyper_decoded_tensor = torch::tensor(hyper_decoded_output).reshape({64,4,4});         
            decoded_hyper_latents.select(0, i).copy_(hyper_decoded_tensor);
        }

        std::vector<torch::Tensor> hyper_decoded_inputs = {decoded_hyper_latents.to(torch::kFloat32).to(device)};
        // Run the compressor model
        std::vector<torch::Tensor> hyper_outputs = hyper_decompressor.run(hyper_decoded_inputs);
        
        torch::Tensor mean = hyper_outputs[0];
        torch::Tensor latent_indexes_recon = hyper_outputs[1];

        torch::Tensor decoded_latents_before_offset = torch::zeros({static_cast<int64_t>(loaded_latents.size()),64,16,16}).to(torch::kInt32);
        //torch::Tensor decoded_latents_before_offset = torch::zeros(latent_size).to(torch::kInt32);
        
        for (int i = 0; i < loaded_latents.size(); i++){
            
            std::vector<int32_t> latent_index = tensor_to_vector<int32_t>(latent_indexes_recon.select(0, i).reshape(-1));

            std::vector<int32_t> latent_decoded_output = range_decoder.decode_with_indexes(loaded_latents[i],
            latent_index,
            gs_quantized_cdf,
            gs_cdf_length,
            gs_offset
            );
            
            torch::Tensor latent_tensor = torch::tensor(latent_decoded_output).reshape({64,16,16});            
            decoded_latents_before_offset.select(0, i).copy_(latent_tensor);
        }
        torch::Tensor decoded_latents = decoded_latents_before_offset.to(device) + mean;
        
        batches.push_back(decoded_latents);
    }
    

    // Check if we have the expected number of batches
    if (batches.size() != 2) {
         std::cerr << "Error: Expected 2 batches for decompression, but got " << batches.size() << std::endl;
         return 1;
    }

    // Create CustomDataset using PreBatchedDataset class
    torch::Tensor pre_batch_0 = torch::stack(batches[0].to(torch::kFloat32), 0);
    torch::Tensor pre_batch_1 = torch::stack(batches[1].to(torch::kFloat32), 0);
    
    std::cout << "Stacked pre-batch 0 shape: " << pre_batch_0.sizes() << std::endl;
    std::cout << "Stacked pre-batch 1 shape: " << pre_batch_1.sizes() << std::endl;
    std::cout << "---" << std::endl;

    std::vector<torch::Tensor> dataset_list;
    dataset_list.push_back(pre_batch_0);
    dataset_list.push_back(pre_batch_1);

    auto dataset_latent = std::make_shared<PreBatchedDataset>(std::move(dataset_list));

    // Create DataLoader for the decompression stage (batch_size=1 is essential)
    auto latent_loader = torch::data::make_data_loader(
        *dataset_latent, // Pass the dataset object by dereferencing the shared_ptr
        torch::data::DataLoaderOptions(1)
    );

    std::cout << "\n--- Decompression Stage ---" << std::endl;
    
    batch_index = 0; // Reset batch index
    for (auto& batch : *latent_loader) {
         std::cout << "\n--- Decompressing Batch " << batch_index << " ---" << std::endl;

        // Get the input tensor (decoded latent from previous stage)
        torch::Tensor input_tensor = batch[0].data.select(0, 0).to(device);
        std::cout << "Input shape to decompressor: " << input_tensor.sizes() << std::endl;

        auto original_sizes = input_tensor.sizes();
        
        // 2. 새 shape 벡터를 {-1, 2}로 초기화
        std::vector<int64_t> new_shape = {-1, 2};
        
        // 3. new_shape 벡터의 끝(end())에 original_sizes의 
        //    "두 번째 원소(begin() + 1)"부터 끝(end())까지를 삽입(insert)
        new_shape.insert(new_shape.end(), original_sizes.begin() + 1, original_sizes.end());
        input_tensor = input_tensor.reshape(new_shape);
        
        std::cout << "New shape: " << input_tensor.sizes() << std::endl;
        int64_t batch_size = input_tensor.sizes()[0];
        
        // Prepare input for the decompressor model
        std::vector<torch::Tensor> inputs = {input_tensor};
        // Run the decompressor model
        std::vector<torch::Tensor> outputs = decompressor.run(inputs);

        if (outputs.empty()){
             std::cerr << "Error: Decompressor returned no outputs for batch " << batch_index << std::endl;
             continue;
        }

        // Save the final decompressed output
        std::cout << "Decompressed output shape: " << outputs[0].sizes() << std::endl;

        torch::Tensor output_tensor = reshape_batch_2d_3d(outputs[0], batch_size);
        std::cout << "Reshaped shape: " << output_tensor.sizes() << std::endl;
        
        //save_tensor_to_bin(outputs[0], "output/decompressed_batch_" + std::to_string(batch_index) + ".bin");
        save_tensor_to_bin(output_tensor, "output/decompressed_batch_" + std::to_string(batch_index) + ".bin");
        batch_index++;
    }

    std::cout << "\nDecompression finished." << std::endl;
    
    return 0;
}