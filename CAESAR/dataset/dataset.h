#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>     
#include <algorithm>   
#include <filesystem>
#include <vector>   
#include <string>  
#include <memory> 
#include <iostream>     
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <stdexcept>
#include <random>
#include <variant>
#include <fstream>


torch::Tensor centerCrop(const torch::Tensor& x , std::pair<int64_t , int64_t> tShape);

torch::Tensor deblockHW(const torch::Tensor& x , int64_t nH , int64_t nW ,
    const std::vector<int64_t>& padding);
std::tuple<torch::Tensor , std::tuple<int64_t , int64_t , std::vector<int64_t>>>

blockHW(const torch::Tensor& data ,
    std::pair<int64_t , int64_t> block_size = { 256, 256 });

std::tuple<torch::Tensor , torch::Tensor , torch::Tensor>
normalizeData(const torch::Tensor& data , const std::string& normType , torch::IntArrayRef axis);

std::pair<std::vector<std::pair<int , float>> , std::vector<int>>
dataFiltering(const torch::Tensor& data , int nFrame);

std::unordered_map<int , int> buildReverseIdMap(int visableLen , const std::vector<int>& filteredLables);

std::unordered_map<std::string , int> get_augment_type_arg(
    const std::unordered_map<std::string , torch::Tensor>& args);


struct BinaryFileConfig {
    std::string file_path;
    std::vector<int64_t> dimensions;
    torch::ScalarType data_type;

    void validate() const {
        if (dimensions.size() != 5) {
            throw std::invalid_argument(
                "Binary file dimensions must be 5D [V, S, T, H, W], got " +
                std::to_string(dimensions.size()) + "D");
        }

        if (dimensions[0] < 1) {
            throw std::invalid_argument(
                "First dimension (V) must be at least 1, got " +
                std::to_string(dimensions[0]));
        }

        if (dimensions[2] < 1) {
            throw std::invalid_argument(
                "Third dimension (T/frames) must be at least 1, got " +
                std::to_string(dimensions[2]));
        }

        for (size_t i = 0; i < dimensions.size(); ++i) {
            if (dimensions[i] <= 0) {
                throw std::invalid_argument(
                    "Dimension " + std::to_string(i) + " must be positive, got " +
                    std::to_string(dimensions[i]));
            }
        }
    }
};


struct DatasetConfig {
    std::optional<BinaryFileConfig> binary_config;
    std::optional<std::string> binary_path;
    std::optional<torch::Tensor> memory_data;

    // Required parameters
    std::string dataset_name = "Customized Dataset";
    int n_frame;

    // Data selection parameters
    std::optional<int> variable_idx;
    std::optional<std::pair<int , int>> section_range;
    std::optional<std::pair<int , int>> frame_range;
    std::optional<std::pair<int , int>> resolution;

    // Training parameters
    int train_size = 256;
    bool inst_norm = true;
    std::unordered_map<std::string , int> augment_type;
    std::string norm_type = "mean_range";
    bool train_mode = false;

    // Test parameters
    std::pair<int , int> test_size = { 256, 256 };
    int n_overlap = 0;
    int downsampling = 1;
};

class BaseDataset {
public:
    explicit BaseDataset(const DatasetConfig& config);
    virtual ~BaseDataset() = default;

    torch::Tensor apply_augments(torch::Tensor data);
    torch::Tensor apply_padding_or_crop(torch::Tensor data);
    torch::Tensor apply_inst_norm(torch::Tensor data , bool return_norm = false);
    std::tuple<torch::Tensor , torch::Tensor , torch::Tensor> apply_inst_norm_with_params(torch::Tensor data);

protected:
    std::string dataset_name;
    std::optional<int> variable_idx;
    std::optional<std::pair<int , int>> section_range;
    std::optional<std::pair<int , int>> frame_range;
    int n_frame;
    std::optional<std::pair<int , int>> resolution;
    int train_size;
    bool inst_norm;
    std::unordered_map<std::string , int> augment_type;
    std::string norm_type;
    bool train_mode;
    std::pair<int , int> test_size;
    int n_overlap;
    int downsampling;
    int max_downsample;
    bool enable_ds;

private:
    mutable std::mt19937 rng_;

    torch::Tensor apply_downsampling(torch::Tensor data , int step);
};


class ScientificDataset : public BaseDataset {
public:
    explicit ScientificDataset(const DatasetConfig& config);

    size_t size() const;
    std::unordered_map<std::string , torch::Tensor> get_item(size_t idx);

    torch::Tensor original_data() const;
    torch::Tensor input_data() const;
    torch::Tensor recons_data(const torch::Tensor& recons_data) const;
    torch::Tensor deblocking_hw(const torch::Tensor& data) const;

    std::tuple<int64_t , int64_t , std::vector<int64_t>> get_block_info() const;
    const torch::Tensor& get_data_input() const;
    const std::vector<std::pair<int , float>>& get_filtered_blocks() const;
    const int64_t& get_pad_T() const;
    const std::vector<int64_t>& get_shape_info() const;

private:
    std::vector<int64_t> shape_org;
    std::vector<int64_t> shape;
    int64_t delta_t;
    int64_t t_samples;
    int64_t pad_T;
    int64_t dataset_length;
    int64_t visible_length;

    torch::Tensor data_input;
    torch::ScalarType dtype;

    std::vector<std::pair<int , float>> filtered_blocks;
    std::vector<int> filtered_labels;
    std::unordered_map<int , int> reverse_id_map;

    std::tuple<int64_t , int64_t , std::vector<int64_t>> block_info;

    torch::Tensor var_offset;
    torch::Tensor var_scale;

    // Load from memory (pre-loaded tensor)
    torch::Tensor loadDatasetInMemory(
        const torch::Tensor& memory_data ,
        std::optional<int> variable_idx = std::nullopt ,
        std::optional<std::pair<int , int>> section_range = std::nullopt ,
        std::optional<std::pair<int , int>> frame_range = std::nullopt
    );

    // Load from binary file with explicit configuration
    torch::Tensor
    (
        const BinaryFileConfig& bin_config ,
        std::optional<int> variable_idx = std::nullopt ,
        std::optional<std::pair<int , int>> section_range = std::nullopt ,
        std::optional<std::pair<int , int>> frame_range = std::nullopt
    );

    int64_t update_length();

    std::unordered_map<std::string , torch::Tensor> post_processing(
        const torch::Tensor& data , int var_idx , bool is_training);
};