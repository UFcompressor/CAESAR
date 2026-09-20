#include "../CAESAR/models/nglr.h"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_text(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open " + path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

std::vector<int64_t> read_shape(const std::string &path) {
  const auto text = read_text(path);
  std::regex number(R"(\"shape\"\s*:\s*\[([^]]+)\])");
  std::smatch match;
  if (!std::regex_search(text, match, number))
    throw std::runtime_error("shape.json has no shape array");
  std::vector<int64_t> shape;
  std::regex dimension(R"(\d+)");
  for (auto it =
           std::sregex_iterator(match[1].first, match[1].second, dimension);
       it != std::sregex_iterator(); ++it)
    shape.push_back(std::stoll(it->str()));
  if (shape.size() != 5)
    throw std::runtime_error("NGLR comparison requires a 5-D shape");
  return shape;
}

torch::Tensor read_float32(const std::string &path,
                           const std::vector<int64_t> &shape) {
  uint64_t elements = 1;
  for (const auto dimension : shape) {
    if (dimension <= 0 || elements > UINT64_MAX / dimension)
      throw std::runtime_error("Invalid tensor shape");
    elements *= static_cast<uint64_t>(dimension);
  }
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Could not open " + path);
  std::vector<float> values(elements);
  file.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (file.gcount() !=
      static_cast<std::streamsize>(values.size() * sizeof(float)))
    throw std::runtime_error("File has fewer bytes than shape requires: " +
                             path);
  return torch::from_blob(values.data(), shape, torch::kFloat32).clone();
}

size_t metadata_bytes(const nglr::NGLRMetaData &metadata) {
  size_t bytes = sizeof(metadata.schema_version) +
                 sizeof(metadata.correction_occurred) +
                 sizeof(metadata.constant_input) + 5 * sizeof(double) +
                 3 * sizeof(int64_t) + 3 * sizeof(int32_t) + sizeof(uint64_t) +
                 metadata.shape.size() * sizeof(int64_t);
  for (const auto &weight : metadata.weights)
    bytes += sizeof(uint64_t) + weight.name.size() + sizeof(uint64_t) +
             weight.shape.size() * sizeof(int64_t) + sizeof(uint64_t) +
             weight.values.size() * sizeof(float);
  return bytes;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <compare_out_directory>\n";
    return 2;
  }
  try {
    const std::string directory = argv[1];
    const auto shape = read_shape(directory + "/shape.json");
    const auto original = read_float32(directory + "/original.bin", shape);
    const auto recons = read_float32(directory + "/recons.bin", shape);
    const auto original_bytes = original.numel() * sizeof(float);

    for (const double target : {1e-4, 1e-5, 1e-6}) {
      nglr::NGLRTrainOptions options;
      options.epochs = 50;
      options.batch_size = 8;
      options.zstd_level = 3;
      nglr::NGLRMetaData metadata;
      std::vector<uint8_t> correction;
      nglr::compress(original, recons, target, metadata, correction, options);
      const auto decoded = nglr::decompress(recons, metadata, correction);
      const double nrmse =
          ((decoded.to(torch::kFloat64) - original.to(torch::kFloat64)) /
           metadata.quantization.scale)
              .square()
              .mean()
              .sqrt()
              .item<double>();
      const size_t total_bytes = correction.size() + metadata_bytes(metadata);
      const double ratio = total_bytes == 0
                               ? 0.0
                               : static_cast<double>(original_bytes) /
                                     static_cast<double>(total_bytes);
      std::cout << "target=" << target << " achieved_nrmse=" << nrmse
                << " correction_bytes=" << correction.size()
                << " metadata_bytes=" << metadata_bytes(metadata)
                << " total_bytes=" << total_bytes
                << " compression_ratio=" << ratio
                << " status=" << (nrmse <= target ? "PASS" : "MISS") << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "test_nglr_compare: " << error.what() << '\n';
    return 1;
  }
}
