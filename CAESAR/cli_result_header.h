#pragma once

#include "models/caesar_compress.h"

// Header for the CLI and file-based test metadata. This is not the ADIOS
// format.
namespace caesar::cli {
template <class T> void write_value(std::ostream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
  if (!out)
    throw std::runtime_error("Failed writing CLI result header");
}
template <class T> T read_value(std::istream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in)
    throw std::runtime_error("Truncated CLI result header");
  return value;
}
inline void write_shape(std::ostream &out, const std::vector<int64_t> &shape) {
  write_value(out, static_cast<uint32_t>(shape.size()));
  for (auto dim : shape)
    write_value(out, dim);
}
inline std::vector<int64_t> read_shape(std::istream &in) {
  auto rank = read_value<uint32_t>(in);
  if (rank < 3 || rank > 5)
    throw std::runtime_error("Invalid rank in CLI result header");
  std::vector<int64_t> shape(rank);
  for (auto &dim : shape) {
    dim = read_value<int64_t>(in);
    if (dim <= 0)
      throw std::runtime_error("Invalid dimension in CLI result header");
  }
  return shape;
}
inline void write_result_header(std::ostream &out,
                                const CompressionResult &result) {
  out.write("CAESAPI1", 8);
  write_value(out, static_cast<int32_t>(result.n_frame));
  write_value(out, static_cast<uint32_t>(result.model_id.size()));
  out.write(result.model_id.data(), result.model_id.size());
  write_shape(out, result.original_shape);
  write_shape(out, result.shape_info.original_shape);
  write_shape(out, result.shape_info.padded_shape);
  write_value(out, result.shape_info.original_length);
}
inline void read_result_header(std::istream &in, CompressionResult &result) {
  char magic[8]{};
  in.read(magic, 8);
  if (!in || std::string(magic, 8) != "CAESAPI1")
    throw std::runtime_error(
        "Unsupported CLI metadata format; recompress with the current CLI");
  result.n_frame = read_value<int32_t>(in);
  auto length = read_value<uint32_t>(in);
  if (length == 0 || length > 4096)
    throw std::runtime_error(
        "Invalid model identity length in CLI result header");
  result.model_id.resize(length);
  in.read(result.model_id.data(), length);
  if (!in)
    throw std::runtime_error("Truncated model identity in CLI result header");
  result.original_shape = read_shape(in);
  result.shape_info.original_shape = read_shape(in);
  result.shape_info.padded_shape = read_shape(in);
  result.shape_info.original_length = read_value<int64_t>(in);
}
} // namespace caesar::cli
