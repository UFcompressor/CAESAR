#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct CompressionResult;

namespace caesar {
// Versioned, little-endian buffer containing all CompressionResult fields.
// Compiled models and their probability tables remain separate artifacts.
std::vector<uint8_t> serialize(const CompressionResult &result);
CompressionResult deserialize(const void *data, size_t size);
CompressionResult deserialize(const std::vector<uint8_t> &data);
void save(const CompressionResult &result, const std::string &path);
CompressionResult load(const std::string &path);
} // namespace caesar
