#include "../CAESAR/dataset/dataset.h"
#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include <optional>

class TestScientificDataset {
public:
    // Test constructor
    TestScientificDataset() = default;

    // Method to create test data
    torch::Tensor create_test_data() {
        // Create a 5D tensor [Variables=2, Sections=3, Time=10, Height=64, Width=64]
        torch::Tensor data = torch::randn({2, 3, 10, 64, 64}, torch::kFloat32);

        // Add some specific patterns to make verification easier
        for (int v = 0; v < 2; v++) {
            for (int s = 0; s < 3; s++) {
                // Set a unique value pattern for each variable-section combination
                float base_value = v * 10.0f + s;
                data.index({v, s}) = data.index({v, s}) + base_value;
            }
        }

        return data;
    }

    // Write binary file method (simplified version from your code)
    void write_binary_file(const torch::Tensor& tensor, const std::string& file_path) {
        if (tensor.dim() != 5) {
            throw std::runtime_error("Expected 5D tensor for binary file writing");
        }

        // Create directory if it doesn't exist
        std::filesystem::path file_dir = std::filesystem::path(file_path).parent_path();
        if (!file_dir.empty() && !std::filesystem::exists(file_dir)) {
            std::filesystem::create_directories(file_dir);
        }

        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create binary file: " + file_path);
        }

        // Write header: shape (5 int64_t values)
        auto sizes = tensor.sizes();
        std::vector<int64_t> shape(sizes.begin(), sizes.end());
        file.write(reinterpret_cast<const char*>(shape.data()), 5 * sizeof(int64_t));

        // Write data type
        int32_t dtype_int = static_cast<int32_t>(tensor.scalar_type());
        file.write(reinterpret_cast<const char*>(&dtype_int), sizeof(int32_t));

        // Write tensor data
        auto contiguous_tensor = tensor.contiguous();
        size_t data_size = contiguous_tensor.numel() * contiguous_tensor.element_size();
        file.write(reinterpret_cast<const char*>(contiguous_tensor.data_ptr()), data_size);

        if (!file.good()) {
            throw std::runtime_error("Error writing to binary file: " + file_path);
        }

        file.close();
        std::cout << "✓ Written binary file: " << file_path << " with shape: " << tensor.sizes() << std::endl;
    }

    // Read binary file method (simplified version from your code)
    torch::Tensor read_binary_file(const std::string& file_path) {
        std::cout << "Reading data from binary file: " << file_path << std::endl;

        // Check if file exists
        if (!std::filesystem::exists(file_path)) {
            throw std::runtime_error("Binary file does not exist: " + file_path);
        }

        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open binary file: " + file_path);
        }

        // Read header: shape (5 int64_t values) + dtype info
        std::vector<int64_t> shape(5);
        file.read(reinterpret_cast<char*>(shape.data()), 5 * sizeof(int64_t));

        // Read data type (stored as int32_t representing torch::ScalarType)
        int32_t dtype_int;
        file.read(reinterpret_cast<char*>(&dtype_int), sizeof(int32_t));
        torch::ScalarType file_dtype = static_cast<torch::ScalarType>(dtype_int);

        if (!file.good()) {
            throw std::runtime_error("Error reading header from binary file: " + file_path);
        }

        // Validate shape
        for (int64_t dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("Invalid shape in binary file header");
            }
        }

        // Calculate total elements
        int64_t total_elements = 1;
        for (int64_t dim : shape) {
            total_elements *= dim;
        }

        // Read binary data
        size_t dtype_size = torch::elementSize(file_dtype);
        std::vector<uint8_t> buffer(total_elements * dtype_size);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        if (!file.good() || file.gcount() != static_cast<std::streamsize>(buffer.size())) {
            throw std::runtime_error("Error reading data from binary file: " + file_path);
        }

        file.close();

        // Create tensor from binary data
        torch::Tensor data = torch::from_blob(buffer.data(), shape, file_dtype).clone();

        std::cout << "✓ Loaded tensor with shape: " << data.sizes() << " and dtype: " << data.scalar_type() << std::endl;
        return data;
    }

    // Test method to verify data integrity
    bool verify_data_integrity(const torch::Tensor& original, const torch::Tensor& loaded) {
        if (!original.sizes().equals(loaded.sizes())) {
            std::cout << "✗ Shape mismatch: original " << original.sizes()
                      << " vs loaded " << loaded.sizes() << std::endl;
            return false;
        }

        if (original.scalar_type() != loaded.scalar_type()) {
            std::cout << "✗ Data type mismatch: original " << original.scalar_type()
                      << " vs loaded " << loaded.scalar_type() << std::endl;
            return false;
        }

        // Check if tensors are close (allowing for floating point precision)
        if (torch::allclose(original, loaded, 1e-6)) {
            std::cout << "✓ Data integrity verified: tensors are identical" << std::endl;
            return true;
        } else {
            auto diff = torch::abs(original - loaded);
            auto max_diff = torch::max(diff).item<float>();
            std::cout << "✗ Data mismatch: maximum difference = " << max_diff << std::endl;
            return false;
        }
    }

    // Print tensor statistics for verification
    void print_tensor_stats(const torch::Tensor& tensor, const std::string& name) {
        std::cout << name << " statistics:" << std::endl;
        std::cout << "  Shape: " << tensor.sizes() << std::endl;
        std::cout << "  Data type: " << tensor.scalar_type() << std::endl;
        std::cout << "  Min value: " << torch::min(tensor).item<float>() << std::endl;
        std::cout << "  Max value: " << torch::max(tensor).item<float>() << std::endl;
        std::cout << "  Mean value: " << torch::mean(tensor).item<float>() << std::endl;

        // Print a few sample values from different sections
        std::cout << "  Sample values:" << std::endl;
        for (int v = 0; v < std::min(2, static_cast<int>(tensor.size(0))); v++) {
            for (int s = 0; s < std::min(2, static_cast<int>(tensor.size(1))); s++) {
                float sample_val = tensor.index({v, s, 0, 0, 0}).item<float>();
                std::cout << "    tensor[" << v << "," << s << ",0,0,0] = " << sample_val << std::endl;
            }
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "=== Binary File I/O Test ===" << std::endl << std::endl;

    try {
        // Create test instance
        TestScientificDataset test_dataset;

        // Step 1: Create test data
        std::cout << "1. Creating test data..." << std::endl;
        torch::Tensor original_data = test_dataset.create_test_data();
        test_dataset.print_tensor_stats(original_data, "Original data");

        // Step 2: Write to binary file
        std::string test_file_path = "./test_data/scientific_data_test.bin";
        std::cout << "2. Writing data to binary file..." << std::endl;
        test_dataset.write_binary_file(original_data, test_file_path);

        // Check file size
        if (std::filesystem::exists(test_file_path)) {
            auto file_size = std::filesystem::file_size(test_file_path);
            std::cout << "✓ File created successfully. Size: " << file_size << " bytes" << std::endl << std::endl;
        }

        // Step 3: Read from binary file
        std::cout << "3. Reading data from binary file..." << std::endl;
        torch::Tensor loaded_data = test_dataset.read_binary_file(test_file_path);
        test_dataset.print_tensor_stats(loaded_data, "Loaded data");

        // Step 4: Verify data integrity
        std::cout << "4. Verifying data integrity..." << std::endl;
        bool integrity_check = test_dataset.verify_data_integrity(original_data, loaded_data);

        if (integrity_check) {
            std::cout << "✓ All tests passed! Binary I/O working correctly." << std::endl;
        } else {
            std::cout << "✗ Test failed! Data integrity check failed." << std::endl;
            return 1;
        }

        // Step 5: Test with different data types
        std::cout << std::endl << "5. Testing with different data types..." << std::endl;

        // Test with double precision
        torch::Tensor double_data = original_data.to(torch::kDouble);
        std::string double_file_path = "./test_data/scientific_data_double.bin";
        test_dataset.write_binary_file(double_data, double_file_path);
        torch::Tensor loaded_double_data = test_dataset.read_binary_file(double_file_path);
        bool double_check = test_dataset.verify_data_integrity(double_data, loaded_double_data);

        // Test with int32
        torch::Tensor int_data = (original_data * 100).to(torch::kInt32);
        std::string int_file_path = "./test_data/scientific_data_int32.bin";
        test_dataset.write_binary_file(int_data, int_file_path);
        torch::Tensor loaded_int_data = test_dataset.read_binary_file(int_file_path);
        bool int_check = test_dataset.verify_data_integrity(int_data, loaded_int_data);

        if (double_check && int_check) {
            std::cout << "✓ Multiple data type tests passed!" << std::endl;
        } else {
            std::cout << "✗ Some data type tests failed!" << std::endl;
        }

        // Step 6: Performance test with larger data
        std::cout << std::endl << "6. Performance test with larger data..." << std::endl;
        torch::Tensor large_data = torch::randn({5, 10, 50, 128, 128}, torch::kFloat32);
        std::string large_file_path = "./test_data/scientific_data_large.bin";

        auto start_time = std::chrono::high_resolution_clock::now();
        test_dataset.write_binary_file(large_data, large_file_path);
        auto write_time = std::chrono::high_resolution_clock::now();

        torch::Tensor loaded_large_data = test_dataset.read_binary_file(large_file_path);
        auto read_time = std::chrono::high_resolution_clock::now();

        auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(write_time - start_time);
        auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_time - write_time);

        std::cout << "Large data performance:" << std::endl;
        std::cout << "  Data shape: " << large_data.sizes() << std::endl;
        std::cout << "  Write time: " << write_duration.count() << " ms" << std::endl;
        std::cout << "  Read time: " << read_duration.count() << " ms" << std::endl;

        bool large_check = test_dataset.verify_data_integrity(large_data, loaded_large_data);
        if (large_check) {
            std::cout << "✓ Large data test passed!" << std::endl;
        }

        std::cout << std::endl << "=== All tests completed successfully! ===" << std::endl;

        // Cleanup
        std::cout << std::endl << "Cleaning up test files..." << std::endl;
        std::filesystem::remove_all("./test_data");
        std::cout << "✓ Cleanup completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
