#pragma once
#include <torch/torch.h>
#include <memory>
#include <mutex>
#include <vector>
#include "model_utils.h"
#include "array_utils.h"

class ModelCache {
public:

    ModelCache(const ModelCache&) = delete;
    ModelCache& operator=(const ModelCache&) = delete;


    static ModelCache& instance() {
#ifdef _WIN32
        // AOTIModelPackageLoader's destructor crashes during Windows process
        // teardown. Intentionally leak the singleton so it is never destructed;
        // the OS reclaims everything on process exit.
        static ModelCache* instance_ptr = new ModelCache();
        return *instance_ptr;
#else
        static ModelCache instance;
        return instance;
#endif
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

    torch::inductor::AOTIModelPackageLoader* get_compressor_model() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!compressor_model_loaded_) {
            load_compressor_model();
        }
        return compressor_model_.get();
    }


    torch::inductor::AOTIModelPackageLoader* get_hyper_decompressor_model() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!hyper_decompressor_model_loaded_) {
            load_hyper_decompressor_model();
        }
        return hyper_decompressor_model_.get();
    }


    torch::inductor::AOTIModelPackageLoader* get_decompressor_model() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!decompressor_model_loaded_) {
            load_decompressor_model();
        }
        return decompressor_model_.get();
    }


    const std::vector<std::vector<int32_t>>& get_vbr_quantized_cdf() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!prob_tables_loaded_) {
            load_probability_tables();
        }
        return vbr_quantized_cdf_;
    }

    const std::vector<int32_t>& get_vbr_cdf_length() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!prob_tables_loaded_) {
            load_probability_tables();
        }
        return vbr_cdf_length_;
    }

    const std::vector<int32_t>& get_vbr_offset() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!prob_tables_loaded_) {
            load_probability_tables();
        }
        return vbr_offset_;
    }


    const std::vector<std::vector<int32_t>>& get_gs_quantized_cdf() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!prob_tables_loaded_) {
            load_probability_tables();
        }
        return gs_quantized_cdf_;
    }

    const std::vector<int32_t>& get_gs_cdf_length() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!prob_tables_loaded_) {
            load_probability_tables();
        }
        return gs_cdf_length_;
    }

    const std::vector<int32_t>& get_gs_offset() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!prob_tables_loaded_) {
            load_probability_tables();
        }
        return gs_offset_;
    }

private:
    ModelCache() = default;
    ~ModelCache() = default;

    void load_compressor_model() {
    auto model_path = get_model_file("caesar_compressor.pt2");

    compressor_model_ =
        std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            model_path.string()
        );

    compressor_model_loaded_ = true;
}

    void load_hyper_decompressor_model() {
    auto model_path = get_model_file("caesar_hyper_decompressor.pt2");

    hyper_decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        model_path.string()
    );
    hyper_decompressor_model_loaded_ = true;
}

    void load_decompressor_model() {
        decompressor_model_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            get_model_file("caesar_decompressor.pt2").string()
        );
        decompressor_model_loaded_ = true;
    }

    void load_probability_tables() {
        
        // Load VBR tables
        auto vbr_quantized_cdf_1d = load_array_from_bin<int32_t>(
            get_model_file("vbr_quantized_cdf.bin")
        );
        vbr_cdf_length_ = load_array_from_bin<int32_t>(
            get_model_file("vbr_cdf_length.bin")
        );
        vbr_offset_ = load_array_from_bin<int32_t>(
            get_model_file("vbr_offset.bin")
        );
        vbr_quantized_cdf_ = reshape_to_2d(vbr_quantized_cdf_1d, 64, 63);

        // Load GS tables
        auto gs_quantized_cdf_1d = load_array_from_bin<int32_t>(
            get_model_file("gs_quantized_cdf.bin")
        );
        gs_cdf_length_ = load_array_from_bin<int32_t>(
            get_model_file("gs_cdf_length.bin")
        );
        gs_offset_ = load_array_from_bin<int32_t>(
            get_model_file("gs_offset.bin")
        );
        gs_quantized_cdf_ = reshape_to_2d(gs_quantized_cdf_1d, 128, 249);

        prob_tables_loaded_ = true;
    }

    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> compressor_model_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> hyper_decompressor_model_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> decompressor_model_;


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
};


