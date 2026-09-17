#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

struct ModelMetadata {
  std::string registration_id;
  std::string name;
  std::string id;
  std::string checkpoint_sha256;
  std::string checkpoint_file;
  std::string architecture;
  std::string device;
  int min_dims;
  int max_dims;
  std::filesystem::path directory;

  void require(const std::string &required_id) const {
    if (required_id != id)
      throw std::runtime_error(
          "Required CAESAR model: " + required_id + ". Installed model: " + id +
          ". Install and compile the required checkpoint, "
          "then set CAESAR_MODEL_DIR to its exported directory.");
  }

  void require_dims(int dims) const {
    if (dims < min_dims || dims > max_dims)
      throw std::runtime_error("CAESAR model " + id + " supports " +
                               std::to_string(min_dims) + "D through " +
                               std::to_string(max_dims) + "D, received " +
                               std::to_string(dims) + "D");
  }
};

// This installation manifest is separate from the compressed-buffer wire
// format.
inline ModelMetadata read_model_metadata(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("Cannot open CAESAR installation metadata: " +
                             path.string() + ". Run compile_model.py first.");
  std::map<std::string, std::string> fields;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    auto separator = line.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == line.size() ||
        !fields.emplace(line.substr(0, separator), line.substr(separator + 1))
             .second)
      throw std::runtime_error("Malformed CAESAR metadata: " + path.string());
  }
  const auto field = [&](const std::string &key) -> std::string {
    auto entry = fields.find(key);
    if (entry == fields.end())
      throw std::runtime_error("Missing CAESAR metadata field: " + key);
    return entry->second;
  };
  if (field("schema_version") != "1" || fields.size() != 10)
    throw std::runtime_error("Unsupported CAESAR installation metadata schema");
  ModelMetadata metadata;
  metadata.registration_id = field("registration_id");
  metadata.name = field("model_name");
  metadata.id = field("model_id");
  metadata.checkpoint_sha256 = field("checkpoint_sha256");
  metadata.checkpoint_file = field("checkpoint_file");
  metadata.architecture = field("architecture");
  metadata.device = field("device");
  const auto valid_name = [](const std::string &s) {
    return !s.empty() &&
           s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"
                               "VWXYZ0123456789_-") == std::string::npos;
  };
  const std::string suffix = ".pt";
  if (!valid_name(metadata.name) || metadata.checkpoint_file.size() <= 3 ||
      metadata.checkpoint_file.substr(metadata.checkpoint_file.size() - 3) !=
          suffix ||
      !valid_name(metadata.checkpoint_file.substr(
          0, metadata.checkpoint_file.size() - 3)) ||
      metadata.checkpoint_sha256.size() != 64 ||
      metadata.checkpoint_sha256.find_first_not_of("0123456789abcdef") !=
          std::string::npos ||
      metadata.registration_id.empty() || metadata.registration_id[0] == '0' ||
      metadata.registration_id.find_first_not_of("0123456789") !=
          std::string::npos ||
      metadata.id != "ufl:" + metadata.registration_id +
                         "@sha256:" + metadata.checkpoint_sha256 ||
      metadata.architecture != "caesar-bcrn-v1")
    throw std::runtime_error("Invalid CAESAR model identity or architecture");
  auto min_dims = field("min_dims"), max_dims = field("max_dims");
  if (min_dims.size() != 1 || max_dims.size() != 1 || min_dims[0] < '2' ||
      max_dims[0] > '5' || min_dims[0] > max_dims[0])
    throw std::runtime_error("Invalid CAESAR supported dimensions");
  metadata.min_dims = min_dims[0] - '0';
  metadata.max_dims = max_dims[0] - '0';
  if (metadata.device != "cpu" && metadata.device != "cuda" &&
      metadata.device != "mps" && metadata.device != "xpu")
    throw std::runtime_error("Invalid CAESAR compiled device: " +
                             metadata.device);
  metadata.directory = std::filesystem::canonical(path).parent_path();
  return metadata;
}
