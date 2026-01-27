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
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <stdexcept>
#include <random>
#include <variant>
#include <fstream>


torch::Tensor centerCrop(const torch::Tensor& x, std::pair<int64_t, int64_t> tShape);

torch::Tensor deblockHW(const torch::Tensor& x, int64_t nH, int64_t nW,
    const std::vector<int64_t>& padding);

std::tuple<torch::Tensor, std::tuple<int64_t, int64_t, std::vector<int64_t>>>
blockHW(const torch::Tensor& data, std::pair<int64_t, int64_t> block_size = {256, 256});

std::pair<std::vector<std::pair<int, float>>, std::vector<int>>
data_filtering(const torch::Tensor& data, int nFrame, const torch::Device& device);

std::unordered_map<int, int> buildReverseIdMap(int visibleLen, const std::vector<int>& filteredLabels);


struct DatasetConfig {
    std::optional<torch::Tensor> memory_data;

    torch::Device device = torch::kCPU;
    
    std::string dataset_name = "Customized Dataset";
    int n_frame;

    std::optional<int> variable_idx;
    std::optional<std::pair<int, int>> section_range;
    std::optional<std::pair<int, int>> frame_range;
    std::optional<std::pair<int, int>> resolution;

    int train_size = 256;
    bool inst_norm = true;
    std::unordered_map<std::string, int> augment_type;
    std::string norm_type = "mean_range";
    bool train_mode = false;

    std::pair<int, int> test_size = {256, 256};
    int n_overlap = 0;
    int downsampling = 1;
};


class BaseDataset {
public:
    explicit BaseDataset(const DatasetConfig& config);
    virtual ~BaseDataset() = default;

    torch::Tensor apply_augments(torch::Tensor data);
    torch::Tensor apply_padding_or_crop(torch::Tensor data);
    torch::Tensor apply_inst_norm(torch::Tensor data, bool return_norm = false);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_inst_norm_with_params(torch::Tensor data);

protected:
    std::string dataset_name;
    std::optional<int> variable_idx;
    std::optional<std::pair<int, int>> section_range;
    std::optional<std::pair<int, int>> frame_range;
    int n_frame;
    std::optional<std::pair<int, int>> resolution;
    int train_size;
    bool inst_norm;
    std::unordered_map<std::string, int> augment_type;
    std::string norm_type;
    bool train_mode;
    std::pair<int, int> test_size;
    int n_overlap;
    int downsampling;
    int max_downsample;
    bool enable_ds;
    
    torch::Device device_;

private:
    mutable std::mt19937 rng_;
    torch::Tensor apply_downsampling(torch::Tensor data, int step);
};


class ScientificDataset : public BaseDataset {
public:
    explicit ScientificDataset(const DatasetConfig& config);
    void clear(); 
    ~ScientificDataset();  

    size_t size() const;
    std::unordered_map<std::string, torch::Tensor> get_item(size_t idx);

    torch::Tensor original_data() const;
    torch::Tensor input_data() const;
    torch::Tensor recons_data(const torch::Tensor& recons_data) const;
    torch::Tensor deblocking_hw(const torch::Tensor& data) const;

    std::tuple<int64_t, int64_t, std::vector<int64_t>> get_block_info() const;
    const torch::Tensor& get_data_input() const;
    const std::vector<std::pair<int, float>>& get_filtered_blocks() const;
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

    std::vector<std::pair<int, float>> filtered_blocks;
    std::vector<int> filtered_labels;
    std::unordered_map<int, int> reverse_id_map;

    std::tuple<int64_t, int64_t, std::vector<int64_t>> block_info;

    torch::Tensor var_offset;
    torch::Tensor var_scale;

    torch::Tensor loadDatasetInMemory(
        const torch::Tensor& memory_data,
        std::optional<int> variable_idx = std::nullopt,
        std::optional<std::pair<int, int>> section_range = std::nullopt,
        std::optional<std::pair<int, int>> frame_range = std::nullopt
    );

    int64_t update_length();
    std::unordered_map<std::string, torch::Tensor> post_processing(
        const torch::Tensor& data, int var_idx, bool is_training);
};
