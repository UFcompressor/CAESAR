#include "../CAESAR/models/model_metadata.h"
#include <chrono>
#include <functional>
#include <iostream>

static void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

static void must_fail(const std::function<void()> &action,
                      const std::string &expected) {
  try {
    action();
  } catch (const std::runtime_error &error) {
    check(std::string(error.what()).find(expected) != std::string::npos,
          error.what());
    return;
  }
  throw std::runtime_error("Expected failure: " + expected);
}

int main() {
  namespace fs = std::filesystem;
  auto root =
      fs::temp_directory_path() /
      ("caesar-metadata-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directory(root);
  auto path = root / "model_metadata.txt";
  const std::string hash(64, 'a');
  const std::string id = "ufl:1@sha256:" + hash;
  const std::string valid =
      "schema_version=1\nregistration_id=1\nmodel_name=caesar_v1\nmodel_id=" +
      id + "\ncheckpoint_sha256=" + hash +
      "\ncheckpoint_file=caesar_v.pt\narchitecture=caesar-bcrn-v1\nmin_dims="
      "2\nmax_dims=5\ndevice=cpu\n";
  auto write = [&](const std::string &data) { std::ofstream(path) << data; };
  try {
    write(valid);
    auto metadata = read_model_metadata(path);
    metadata.require(id);
    metadata.require_dims(2);
    metadata.require_dims(5);
    check(metadata.directory == fs::canonical(root), "Installation directory");
    must_fail([&] { metadata.require("ufl:2@sha256:" + hash); },
              "Required CAESAR model: ufl:2");
    must_fail([&] { metadata.require_dims(6); }, "supports 2D through 5D");
    write(valid + "device=cuda\n");
    must_fail([&] { read_model_metadata(path); }, "Malformed");
    auto malformed = valid;
    malformed.replace(malformed.find("min_dims=2"), 10, "min_dims=9");
    write(malformed);
    must_fail([&] { read_model_metadata(path); }, "dimensions");
    malformed = valid;
    malformed.replace(malformed.find("ufl:1"), 5, "ufl:9");
    write(malformed);
    must_fail([&] { read_model_metadata(path); }, "identity");
    write("schema_version=1\n");
    must_fail([&] { read_model_metadata(path); }, "schema");
    fs::remove(path);
    must_fail([&] { read_model_metadata(path); }, "Cannot open");
    fs::remove(root);
    std::cout << "Model metadata tests passed\n";
  } catch (...) {
    fs::remove_all(root);
    throw;
  }
}
