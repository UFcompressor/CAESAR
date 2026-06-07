#pragma once
#include <string>
#include <filesystem>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <chrono>
#include <torch/torch.h>

#ifdef USE_CUDA
    #if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        #include <hip/hip_runtime.h>
    #else
        #include <cuda_runtime.h>
    #endif
#endif
namespace fs = std::filesystem;

/**
 * Get the full path to a model file.
 *
 * Search order:
 * 1. CAESAR_MODEL_DIR environment variable
 * 2. Relative to executable: ../exported_model/
 * 3. CMake-defined install location (if available)
 *
 * @param filename The name of the model file (e.g., "vbr_quantized_cdf.bin")
 * @return Full filesystem path to the model file
 * @throws std::runtime_error if the file cannot be found
 */
fs::path get_model_file(const std::string& filename);


double rss_gb();

#ifdef USE_CUDA
double gpu_used_gb();
#endif

static inline double tensor_gb(const torch::Tensor& t) {
    if (!t.defined()) return 0.0;
    return (double)t.numel() * t.element_size() / (1024.0 * 1024 * 1024);
}

static inline void mem_print(const char* tag) {
#ifdef USE_CUDA
    std::cout << "[MEM] " << tag
              << "  rss="      << rss_gb()      << " GiB"
              << "  gpu_used=" << gpu_used_gb() << " GiB\n";
#else
    std::cout << "[MEM] " << tag
              << "  rss=" << rss_gb() << " GiB\n";
#endif
}

std::chrono::high_resolution_clock::time_point get_start_time();
std::chrono::duration<double> get_time(std::chrono::high_resolution_clock::time_point start);

int get_allocated_cores();

std::string get_model_name();