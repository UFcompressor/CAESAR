#pragma once
#include <string>
#include <filesystem>
#include <cstddef>
#include <fstream>
#include <iostream>
#ifdef USE_CUDA
#include <cuda_runtime.h>
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
double gpu_free_gb();
#endif
