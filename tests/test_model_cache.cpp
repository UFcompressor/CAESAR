#include "../CAESAR/models/caesar_decompress.h"
#include "../CAESAR/models/model_cache.h"
#include <future>
#include <iostream>

static int run_model_cache_tests(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--missing") {
    try {
      require_model("ufl:2@sha256:" + std::string(64, '0'));
    } catch (const std::runtime_error &error) {
      const std::string message = error.what();
      if (message.find("Required CAESAR model: ufl:2") != std::string::npos &&
          message.find("required file is missing") != std::string::npos) {
        std::cout << "Missing model fails explicitly: " << message << "\n";
        return 0;
      }
      throw;
    }
    throw std::runtime_error("Missing installation unexpectedly accepted");
  }
  const auto &metadata = get_model_metadata();
  auto &cache = ModelCache::instance(metadata.id);
  if (&cache != &ModelCache::instance(metadata.id))
    throw std::runtime_error("Cache was not reused");
  try {
    ModelCache::instance("ufl:999999@sha256:" + std::string(64, '0'));
    throw std::logic_error("Wrong model was accepted");
  } catch (const std::runtime_error &error) {
    if (std::string(error.what()).find("Required CAESAR model") ==
        std::string::npos)
      throw;
  }
  try {
    Decompressor wrong(torch::Device(torch::kCPU),
                       "ufl:999999@sha256:" + std::string(64, '0'));
    throw std::logic_error("Decompressor accepted wrong identity");
  } catch (const std::runtime_error &error) {
    if (std::string(error.what()).find("Required CAESAR model") ==
        std::string::npos)
      throw;
  }
  auto worker = std::async(std::launch::async, [&] {
    auto &other = ModelCache::instance(metadata.id);
    if (&other == &cache || &get_model_metadata() != &metadata)
      throw std::runtime_error(
          "Thread cache or shared metadata lifetime error");
    // Actually load a runner on the second thread to check AOTI lifetime too.
    if (!other.get_hyper_decompressor_model())
      throw std::runtime_error("No model loaded");
  });
  auto model = cache.get_hyper_decompressor_model();
  if (!model || model != cache.get_hyper_decompressor_model())
    throw std::runtime_error("Model handle was not reused");
  worker.get();
  cache.clear();
  if (model.use_count() != 1)
    throw std::runtime_error("Cache clear did not release its ownership");
  if (cache.get_hyper_decompressor_model() == model)
    throw std::runtime_error("Cache clear did not reload the model");
  std::cout << "Cache reuse, thread isolation, and identity checks passed\n";
  return 0;
}

int main(int argc, char **argv) {
  try {
    return run_model_cache_tests(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "Model cache test failed: " << error.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Model cache test failed with an unknown exception"
              << std::endl;
    return 1;
  }
}
