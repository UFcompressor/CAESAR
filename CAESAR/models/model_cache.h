#pragma once
#include "array_utils.h"
#include "model_utils.h"
#include <memory>
#include <mutex>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
class ModelCache {
public:
  ~ModelCache() = default;
  ModelCache(const ModelCache &) = delete;
  ModelCache &operator=(const ModelCache &) = delete;

  // Each thread owns its model runners; no AOTI runner is invoked concurrently.
  // MPI processes naturally have independent registries. One installed bundle
  // is selected per process; a different requested identity is an explicit
  // error.
  static ModelCache &instance(const std::string &required_id = "") {
    if (!required_id.empty())
      require_model(required_id);
    const auto &metadata = get_model_metadata();
#ifdef _WIN32
    // AOTI destruction during Windows teardown is unsafe (existing workaround).
    static thread_local auto *registry = new Registry();
    auto &entries = *registry;
#else
    static thread_local Registry entries;
#endif
    auto &entry = entries[metadata.id];
    if (!entry)
      entry.reset(new ModelCache(metadata));
    return *entry;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    compressor_model_.reset();
    compressor_model_loaded_ = false;

    hyper_decompressor_model_.reset();
    hyper_decompressor_model_loaded_ = false;

    decompressor_model_.reset();
    decompressor_model_loaded_ = false;

    vbr_quantized_cdf_.clear();
    vbr_cdf_length_.clear();
    vbr_offset_.clear();

    gs_quantized_cdf_.clear();
    gs_cdf_length_.clear();
    gs_offset_.clear();

    prob_tables_loaded_ = false;
  }

  const ModelMetadata &metadata() const { return metadata_; }

  std::shared_ptr<torch::inductor::AOTIModelPackageLoader>
  get_compressor_model() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!compressor_model_loaded_) {
      load_compressor_model();
    }
    return compressor_model_;
  }

  std::shared_ptr<torch::inductor::AOTIModelPackageLoader>
  get_hyper_decompressor_model() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hyper_decompressor_model_loaded_) {
      load_hyper_decompressor_model();
    }
    return hyper_decompressor_model_;
  }

  std::shared_ptr<torch::inductor::AOTIModelPackageLoader>
  get_decompressor_model() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!decompressor_model_loaded_) {
      load_decompressor_model();
    }
    return decompressor_model_;
  }

  const std::vector<std::vector<int32_t>> &get_vbr_quantized_cdf() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!prob_tables_loaded_) {
      load_probability_tables();
    }
    return vbr_quantized_cdf_;
  }

  const std::vector<int32_t> &get_vbr_cdf_length() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!prob_tables_loaded_) {
      load_probability_tables();
    }
    return vbr_cdf_length_;
  }

  const std::vector<int32_t> &get_vbr_offset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!prob_tables_loaded_) {
      load_probability_tables();
    }
    return vbr_offset_;
  }

  const std::vector<std::vector<int32_t>> &get_gs_quantized_cdf() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!prob_tables_loaded_) {
      load_probability_tables();
    }
    return gs_quantized_cdf_;
  }

  const std::vector<int32_t> &get_gs_cdf_length() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!prob_tables_loaded_) {
      load_probability_tables();
    }
    return gs_cdf_length_;
  }

  const std::vector<int32_t> &get_gs_offset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!prob_tables_loaded_) {
      load_probability_tables();
    }
    return gs_offset_;
  }

  const std::string &get_model_name() const { return metadata_.name; }
  const std::string &get_model_device() const { return metadata_.device; }

private:
  using Registry = std::map<std::string, std::unique_ptr<ModelCache>>;
  explicit ModelCache(const ModelMetadata &metadata) : metadata_(metadata) {}

  fs::path model_file(const std::string &filename) const {
    auto path = metadata_.directory / filename;
    if (!fs::is_regular_file(path))
      throw std::runtime_error(
          "Required CAESAR model: " + metadata_.id +
          ". Missing installed artifact: " + path.string());
    return path;
  }

  ModelMetadata metadata_;

  void load_compressor_model() {
    auto model_path = model_file("caesar_compressor.pt2");

    compressor_model_ =
        std::make_shared<torch::inductor::AOTIModelPackageLoader>(
            model_path.string());

    compressor_model_loaded_ = true;
  }

  void load_hyper_decompressor_model() {
    auto model_path = model_file("caesar_hyper_decompressor.pt2");

    hyper_decompressor_model_ =
        std::make_shared<torch::inductor::AOTIModelPackageLoader>(
            model_path.string());
    hyper_decompressor_model_loaded_ = true;
  }

  void load_decompressor_model() {
    decompressor_model_ =
        std::make_shared<torch::inductor::AOTIModelPackageLoader>(
            model_file("caesar_decompressor.pt2").string());
    decompressor_model_loaded_ = true;
  }

  void load_probability_tables() {

    // Load VBR tables
    auto vbr_quantized_cdf_1d =
        load_array_from_bin<int32_t>(model_file("vbr_quantized_cdf.bin"));
    vbr_cdf_length_ =
        load_array_from_bin<int32_t>(model_file("vbr_cdf_length.bin"));
    vbr_offset_ = load_array_from_bin<int32_t>(model_file("vbr_offset.bin"));
    vbr_quantized_cdf_ = reshape_to_2d(vbr_quantized_cdf_1d, 64, 63);

    // Load GS tables
    auto gs_quantized_cdf_1d =
        load_array_from_bin<int32_t>(model_file("gs_quantized_cdf.bin"));
    gs_cdf_length_ =
        load_array_from_bin<int32_t>(model_file("gs_cdf_length.bin"));
    gs_offset_ = load_array_from_bin<int32_t>(model_file("gs_offset.bin"));
    gs_quantized_cdf_ = reshape_to_2d(gs_quantized_cdf_1d, 128, 249);

    prob_tables_loaded_ = true;
  }

  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> compressor_model_;
  std::shared_ptr<torch::inductor::AOTIModelPackageLoader>
      hyper_decompressor_model_;
  std::shared_ptr<torch::inductor::AOTIModelPackageLoader> decompressor_model_;

  std::vector<std::vector<int32_t>> vbr_quantized_cdf_;
  std::vector<int32_t> vbr_cdf_length_;
  std::vector<int32_t> vbr_offset_;
  std::vector<std::vector<int32_t>> gs_quantized_cdf_;
  std::vector<int32_t> gs_cdf_length_;
  std::vector<int32_t> gs_offset_;

  bool compressor_model_loaded_ = false;
  bool hyper_decompressor_model_loaded_ = false;
  bool decompressor_model_loaded_ = false;
  bool prob_tables_loaded_ = false;

  std::mutex mutex_;

  /*
  todo add this for
  input_file.read(reinterpret_cast<char*>(data.data()), file_size);
  if (!input_file) {
      throw std::runtime_error("Short read on file: " + filepath.string());
  }
  */
};
