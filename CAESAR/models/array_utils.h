#pragma once

#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

template<typename T>
std::vector<T> load_array_from_bin(const fs::path& filepath) {
    std::ifstream input_file(filepath, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath.string());
    }

    input_file.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(input_file.tellg());
    input_file.seekg(0, std::ios::beg);

    if (file_size % sizeof(T) != 0) {
        throw std::runtime_error("Binary file size mismatch: " + filepath.string());
    }

    const size_t num_elements = file_size / sizeof(T);
    std::vector<T> data(num_elements);

    input_file.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

template<typename T>
std::vector<std::vector<T>> reshape_to_2d(const std::vector<T>& flat_vec, size_t rows, size_t cols) {
    if (flat_vec.size() != rows * cols) {
        throw std::invalid_argument("Invalid dimensions for reshape_to_2d");
    }

    std::vector<std::vector<T>> vec_2d;
    vec_2d.reserve(rows);

    auto it = flat_vec.begin();
    for (size_t r = 0; r < rows; ++r) {
        vec_2d.emplace_back(it, it + cols);
        it += cols;
    }
    return vec_2d;
}

