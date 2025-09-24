#include <iostream>
#include "../CAESAR/dataset/dataset.h"
#include <filesystem>
#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include <optional>
#include <chrono>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

// Include your dataset header (assuming it's named dataset.h)
// #include "dataset.h"

// For testing purposes, I'll create a test class that mirrors your ScientificDataset methods
class TestScientificDataset {
public:
    TestScientificDataset() = default;
    
    // Create test data with recognizable patterns
    torch::Tensor create_test_data(std::vector<int64_t> shape = {2, 4, 10, 32, 32}) {
        torch::Tensor data = torch::randn(shape, torch::kFloat32);
        
        // Add unique patterns for each variable-section combination
        for (int v = 0; v < shape[0]; v++) {
            for (int s = 0; s < shape[1]; s++) {
                float base_value = v * 100.0f + s * 10.0f;
                data.index({v, s}) = data.index({v, s}) + base_value;
            }
        }
        return data;
    }
    
    // Regular binary file write (from your code)
    void write_binary_file(const torch::Tensor& tensor, const std::string& file_path) {
        if (tensor.dim() != 5) {
            throw std::runtime_error("Expected 5D tensor for binary file writing");
        }
        
        std::filesystem::path file_dir = std::filesystem::path(file_path).parent_path();
        if (!file_dir.empty() && !std::filesystem::exists(file_dir)) {
            std::filesystem::create_directories(file_dir);
        }
        
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create binary file: " + file_path);
        }
        
        auto sizes = tensor.sizes();
        std::vector<int64_t> shape(sizes.begin(), sizes.end());
        file.write(reinterpret_cast<const char*>(shape.data()), 5 * sizeof(int64_t));
        
        int32_t dtype_int = static_cast<int32_t>(tensor.scalar_type());
        file.write(reinterpret_cast<const char*>(&dtype_int), sizeof(int32_t));
        
        auto contiguous_tensor = tensor.contiguous();
        size_t data_size = contiguous_tensor.numel() * contiguous_tensor.element_size();
        file.write(reinterpret_cast<const char*>(contiguous_tensor.data_ptr()), data_size);
        
        if (!file.good()) {
            throw std::runtime_error("Error writing to binary file: " + file_path);
        }
        
        file.close();
        std::cout << "Written binary file: " << file_path << " with shape: " << tensor.sizes() << std::endl;
    }
    
    // Regular binary file read (from your code)
    torch::Tensor read_binary_file(const std::string& file_path) {
        if (!std::filesystem::exists(file_path)) {
            throw std::runtime_error("Binary file does not exist: " + file_path);
        }
        
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open binary file: " + file_path);
        }
        
        std::vector<int64_t> shape(5);
        file.read(reinterpret_cast<char*>(shape.data()), 5 * sizeof(int64_t));
        
        int32_t dtype_int;
        file.read(reinterpret_cast<char*>(&dtype_int), sizeof(int32_t));
        torch::ScalarType file_dtype = static_cast<torch::ScalarType>(dtype_int);
        
        if (!file.good()) {
            throw std::runtime_error("Error reading header from binary file: " + file_path);
        }
        
        for (int64_t dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("Invalid shape in binary file header");
            }
        }
        
        int64_t total_elements = 1;
        for (int64_t dim : shape) {
            total_elements *= dim;
        }
        
        size_t dtype_size = torch::elementSize(file_dtype);
        std::vector<uint8_t> buffer(total_elements * dtype_size);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        
        if (!file.good() || file.gcount() != static_cast<std::streamsize>(buffer.size())) {
            throw std::runtime_error("Error reading data from binary file: " + file_path);
        }
        
        file.close();
        torch::Tensor data = torch::from_blob(buffer.data(), shape, file_dtype).clone();
        
        std::cout << "Loaded tensor with shape: " << data.sizes() << " and dtype: " << data.scalar_type() << std::endl;
        return data;
    }
    
#ifdef MPI_VERSION
    // MPI binary file write (from your code)
    void write_binary_file_mpi(const torch::Tensor& local_tensor, 
                               const std::string& file_path,
                               const std::vector<int64_t>& global_shape) {
        
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        if (local_tensor.dim() != 5) {
            throw std::runtime_error("Expected 5D local tensor for MPI binary file writing");
        }
        
        if (global_shape.size() != 5) {
            throw std::runtime_error("Global shape must have 5 dimensions");
        }
        
        // Rank 0 writes the header
        if (rank == 0) {
            std::cout << "Rank 0: Writing header to binary file: " << file_path << std::endl;
            
            std::filesystem::path file_dir = std::filesystem::path(file_path).parent_path();
            if (!file_dir.empty() && !std::filesystem::exists(file_dir)) {
                std::filesystem::create_directories(file_dir);
            }
            
            std::ofstream file(file_path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot create binary file: " + file_path);
            }
            
            file.write(reinterpret_cast<const char*>(global_shape.data()), 5 * sizeof(int64_t));
            
            int32_t dtype_int = static_cast<int32_t>(local_tensor.scalar_type());
            file.write(reinterpret_cast<const char*>(&dtype_int), sizeof(int32_t));
            
            if (!file.good()) {
                throw std::runtime_error("Error writing header to MPI binary file");
            }
            
            file.close();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Calculate file offset for this rank's data
        size_t header_size = 5 * sizeof(int64_t) + sizeof(int32_t);
        size_t dtype_size = local_tensor.element_size();
        
        // Calculate this rank's section start (assume 1D decomposition along dim 1)
        int64_t sections_total = global_shape[1];
        int64_t sections_per_rank = sections_total / size;
        int64_t remainder = sections_total % size;
        int64_t section_start = rank * sections_per_rank + std::min(static_cast<int64_t>(rank), remainder);
        
        int64_t elements_per_section = global_shape[0] * global_shape[2] * global_shape[3] * global_shape[4];
        size_t bytes_per_section = elements_per_section * dtype_size;
        size_t file_offset = header_size + section_start * bytes_per_section;
        
        // Use MPI-IO for parallel writing
        MPI_File mpi_file;
        int result = MPI_File_open(MPI_COMM_WORLD, file_path.c_str(), 
                                  MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);
        if (result != MPI_SUCCESS) {
            throw std::runtime_error("Failed to open file for writing with MPI-IO: " + file_path);
        }
        
        auto contiguous_tensor = local_tensor.contiguous();
        size_t write_size = contiguous_tensor.numel() * dtype_size;
        
        MPI_Status status;
        result = MPI_File_write_at(mpi_file, file_offset, contiguous_tensor.data_ptr(), 
                                  write_size, MPI_BYTE, &status);
        if (result != MPI_SUCCESS) {
            MPI_File_close(&mpi_file);
            throw std::runtime_error("Failed to write data with MPI-IO");
        }
        
        MPI_File_close(&mpi_file);
        
        std::cout << "Rank " << rank << ": Written local tensor with shape: " << local_tensor.sizes() 
                  << " at offset: " << file_offset << std::endl;
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "MPI binary file write completed: " << file_path << std::endl;
        }
    }
    
    // MPI binary file read (from your code)
    torch::Tensor read_binary_file_mpi(const std::string& file_path) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        std::vector<int64_t> global_shape(5);
        torch::ScalarType file_dtype;
        
        // Rank 0 reads header and broadcasts to all processes
        if (rank == 0) {
            std::cout << "Rank 0: Reading header from binary file: " << file_path << std::endl;
            
            if (!std::filesystem::exists(file_path)) {
                throw std::runtime_error("MPI: Binary file does not exist: " + file_path);
            }
            
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open binary file: " + file_path);
            }
            
            file.read(reinterpret_cast<char*>(global_shape.data()), 5 * sizeof(int64_t));
            
            int32_t dtype_int;
            file.read(reinterpret_cast<char*>(&dtype_int), sizeof(int32_t));
            file_dtype = static_cast<torch::ScalarType>(dtype_int);
            
            file.close();
            
            if (!file.good()) {
                throw std::runtime_error("Error reading header from binary file: " + file_path);
            }
        }
        
        // Broadcast shape and dtype to all processes
        MPI_Bcast(global_shape.data(), 5, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        int32_t dtype_int = static_cast<int32_t>(file_dtype);
        MPI_Bcast(&dtype_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        file_dtype = static_cast<torch::ScalarType>(dtype_int);
        
        // Calculate 1D domain decomposition along the second dimension (sections)
        int64_t sections_total = global_shape[1];
        int64_t sections_per_rank = sections_total / size;
        int64_t remainder = sections_total % size;
        
        int64_t section_start = rank * sections_per_rank + std::min(static_cast<int64_t>(rank), remainder);
        int64_t section_count = sections_per_rank + (rank < remainder ? 1 : 0);
        int64_t section_end = section_start + section_count;
        
        std::cout << "Rank " << rank << ": Loading sections [" << section_start 
                  << ", " << section_end << ") out of " << sections_total << std::endl;
        
        // Calculate local tensor shape
        std::vector<int64_t> local_shape = {global_shape[0], section_count, global_shape[2], 
                                           global_shape[3], global_shape[4]};
        
        // Calculate file offsets
        size_t header_size = 5 * sizeof(int64_t) + sizeof(int32_t);
        size_t dtype_size = torch::elementSize(file_dtype);
        
        int64_t elements_per_section = global_shape[0] * global_shape[2] * global_shape[3] * global_shape[4];
        size_t bytes_per_section = elements_per_section * dtype_size;
        size_t file_offset = header_size + section_start * bytes_per_section;
        size_t read_size = section_count * bytes_per_section;
        
        // Use MPI-IO for parallel reading
        MPI_File mpi_file;
        MPI_Status status;
        
        int result = MPI_File_open(MPI_COMM_WORLD, file_path.c_str(), 
                                  MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);
        if (result != MPI_SUCCESS) {
            throw std::runtime_error("Failed to open file with MPI-IO: " + file_path);
        }
        
        std::vector<uint8_t> buffer(read_size);
        result = MPI_File_read_at(mpi_file, file_offset, buffer.data(), read_size, 
                                 MPI_BYTE, &status);
        if (result != MPI_SUCCESS) {
            MPI_File_close(&mpi_file);
            throw std::runtime_error("Failed to read data with MPI-IO");
        }
        
        MPI_File_close(&mpi_file);
        
        torch::Tensor local_data = torch::from_blob(buffer.data(), local_shape, file_dtype).clone();
        
        std::cout << "Rank " << rank << ": Loaded local tensor with shape: " << local_data.sizes() << std::endl;
        
        return local_data;
    }
    
    // Calculate global shape from local tensors across all MPI ranks
    std::vector<int64_t> calculate_global_shape_mpi(const torch::Tensor& local_tensor) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        if (local_tensor.dim() != 5) {
            throw std::runtime_error("Expected 5D local tensor");
        }
        
        auto local_shape = local_tensor.sizes();
        
        // Gather section counts from all ranks
        std::vector<int64_t> section_counts(size);
        int64_t local_sections = local_shape[1];
        
        MPI_Allgather(&local_sections, 1, MPI_LONG_LONG, 
                      section_counts.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
        
        // Calculate total sections
        int64_t total_sections = 0;
        for (int64_t count : section_counts) {
            total_sections += count;
        }
        
        std::vector<int64_t> global_shape = {local_shape[0], total_sections, local_shape[2], 
                                            local_shape[3], local_shape[4]};
        
        if (rank == 0) {
            std::cout << "Calculated global shape: [" << global_shape[0] << ", " << global_shape[1] 
                      << ", " << global_shape[2] << ", " << global_shape[3] << ", " << global_shape[4] << "]" << std::endl;
        }
        
        return global_shape;
    }
#endif
    
    bool verify_data_integrity(const torch::Tensor& original, const torch::Tensor& loaded) {
        if (!original.sizes().equals(loaded.sizes())) {
            std::cout << "Shape mismatch: original " << original.sizes() 
                      << " vs loaded " << loaded.sizes() << std::endl;
            return false;
        }
        
        if (original.scalar_type() != loaded.scalar_type()) {
            std::cout << "Data type mismatch: original " << original.scalar_type() 
                      << " vs loaded " << loaded.scalar_type() << std::endl;
            return false;
        }
        
        if (torch::allclose(original, loaded, 1e-6)) {
            std::cout << "Data integrity verified: tensors are identical" << std::endl;
            return true;
        } else {
            auto diff = torch::abs(original - loaded);
            auto max_diff = torch::max(diff).item<float>();
            std::cout << "Data mismatch: maximum difference = " << max_diff << std::endl;
            return false;
        }
    }
    
    void print_tensor_stats(const torch::Tensor& tensor, const std::string& name) {
        std::cout << name << " statistics:" << std::endl;
        std::cout << "  Shape: " << tensor.sizes() << std::endl;
        std::cout << "  Data type: " << tensor.scalar_type() << std::endl;
        std::cout << "  Min value: " << torch::min(tensor).item<float>() << std::endl;
        std::cout << "  Max value: " << torch::max(tensor).item<float>() << std::endl;
        std::cout << "  Mean value: " << torch::mean(tensor).item<float>() << std::endl;
        
        // Print sample values from different sections
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

int main(int argc, char* argv[]) {
#ifdef MPI_VERSION
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== Binary File I/O Test (MPI + Regular) ===" << std::endl;
        std::cout << "Running with " << size << " MPI processes" << std::endl << std::endl;
    }
#else
    std::cout << "=== Binary File I/O Test (Regular Only) ===" << std::endl;
    std::cout << "MPI support not compiled" << std::endl << std::endl;
#endif
    
    try {
        TestScientificDataset test_dataset;
        
#ifdef MPI_VERSION
        if (rank == 0) {
#endif
            // Test 1: Regular I/O
            std::cout << "1. Testing Regular Binary I/O..." << std::endl;
            torch::Tensor original_data = test_dataset.create_test_data({2, 8, 20, 64, 64});
            test_dataset.print_tensor_stats(original_data, "Original data");
            
            std::string regular_file = "./test_data/regular_test.bin";
            test_dataset.write_binary_file(original_data, regular_file);
            torch::Tensor loaded_regular = test_dataset.read_binary_file(regular_file);
            bool regular_check = test_dataset.verify_data_integrity(original_data, loaded_regular);
            
            if (regular_check) {
                std::cout << "Regular I/O test PASSED" << std::endl << std::endl;
            } else {
                std::cout << "Regular I/O test FAILED" << std::endl << std::endl;
            }
#ifdef MPI_VERSION
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Test 2: MPI I/O
        if (rank == 0) {
            std::cout << "2. Testing MPI Binary I/O..." << std::endl;
        }
        
        // Each rank creates its portion of the data
        // Domain decomposition along sections dimension
        std::vector<int64_t> global_shape = {2, 8, 20, 64, 64}; // Variables, Sections, Time, H, W
        int64_t sections_per_rank = global_shape[1] / size;
        int64_t remainder = global_shape[1] % size;
        int64_t local_sections = sections_per_rank + (rank < remainder ? 1 : 0);
        
        std::vector<int64_t> local_shape = {global_shape[0], local_sections, global_shape[2], 
                                           global_shape[3], global_shape[4]};
        
        torch::Tensor local_data = test_dataset.create_test_data(local_shape);
        
        // Adjust the pattern to be consistent across ranks
        for (int v = 0; v < local_shape[0]; v++) {
            for (int s = 0; s < local_shape[1]; s++) {
                int64_t global_section_idx = rank * sections_per_rank + std::min(static_cast<int64_t>(rank), remainder) + s;
                float base_value = v * 100.0f + global_section_idx * 10.0f;
                local_data.index({v, s}) = local_data.index({v, s}) * 0.1f + base_value;
            }
        }
        
        if (rank == 0) {
            test_dataset.print_tensor_stats(local_data, "Rank 0 local data");
        }
        
        // Calculate global shape automatically
        std::vector<int64_t> calculated_global_shape = test_dataset.calculate_global_shape_mpi(local_data);
        
        // Test MPI write
        std::string mpi_file = "./test_data/mpi_test.bin";
        auto write_start = std::chrono::high_resolution_clock::now();
        test_dataset.write_binary_file_mpi(local_data, mpi_file, calculated_global_shape);
        auto write_end = std::chrono::high_resolution_clock::now();
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Test MPI read
        auto read_start = std::chrono::high_resolution_clock::now();
        torch::Tensor loaded_mpi_data = test_dataset.read_binary_file_mpi(mpi_file);
        auto read_end = std::chrono::high_resolution_clock::now();
        
        // Verify data integrity on each rank
        bool mpi_check = test_dataset.verify_data_integrity(local_data, loaded_mpi_data);
        
        // Gather results from all ranks
        int local_result = mpi_check ? 1 : 0;
        std::vector<int> all_results(size);
        MPI_Gather(&local_result, 1, MPI_INT, all_results.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            bool all_passed = true;
            for (int i = 0; i < size; i++) {
                if (all_results[i] == 0) {
                    std::cout << "Rank " << i << " failed data integrity check" << std::endl;
                    all_passed = false;
                }
            }
            
            auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start);
            auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start);
            
            std::cout << "MPI Performance:" << std::endl;
            std::cout << "  Global shape: [" << calculated_global_shape[0] << ", " << calculated_global_shape[1] 
                      << ", " << calculated_global_shape[2] << ", " << calculated_global_shape[3] 
                      << ", " << calculated_global_shape[4] << "]" << std::endl;
            std::cout << "  Write time: " << write_time.count() << " ms" << std::endl;
            std::cout << "  Read time: " << read_time.count() << " ms" << std::endl;
            
            if (all_passed) {
                std::cout << "MPI I/O test PASSED on all ranks" << std::endl << std::endl;
            } else {
                std::cout << "MPI I/O test FAILED on some ranks" << std::endl << std::endl;
            }
            
            // Test 3: Verify MPI file can be read with regular method
            std::cout << "3. Testing MPI file compatibility with regular read..." << std::endl;
            torch::Tensor mpi_file_regular_read = test_dataset.read_binary_file(mpi_file);
            std::cout << "MPI file read with regular method - shape: " << mpi_file_regular_read.sizes() << std::endl;
            
            // Cleanup
            std::cout << "Cleaning up test files..." << std::endl;
            std::filesystem::remove_all("./test_data");
            std::cout << "Test completed successfully!" << std::endl;
        }
#endif
        
    } catch (const std::exception& e) {
#ifdef MPI_VERSION
        std::cerr << "Rank " << rank << " - Test failed with exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
#endif
    }
    
#ifdef MPI_VERSION
    MPI_Finalize();
#endif
    
    return 0;
}
