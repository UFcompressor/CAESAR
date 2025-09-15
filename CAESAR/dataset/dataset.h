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


torch::Tensor centerCrop(const torch::Tensor& x, std::pair<int64_t, int64_t> tShape);
torch::Tensor downSamplingData(const torch::Tensor& data, const std::vector<double>& zoomFactors);

torch::Tensor deblockHW(const torch::Tensor& x, int64_t nH, int64_t nW, 
        const std::vector<int64_t>& padding);
std::tuple<torch::Tensor, std::tuple<int64_t, int64_t, std::vector<int64_t>>>

blockHW(const torch::Tensor& data,
        std::pair<int64_t, int64_t> block_size = {256, 256});

// for debuging i guess not used
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
normalizeData(const torch::Tensor& data, const std::string& normType, torch::IntArrayRef axis);

std::pair<std::vector<std::pair<int, float>>, std::vector<int>>
dataFiltering(const torch::Tensor& data, int nFrame);

std::unordered_map<int, int> buildReverseIdMap(int visableLen, const std::vector<int>& filteredLables);


class BaseDataset {
public:
    explicit BaseDataset(const std::unordered_map<std::string, torch::Tensor>& args);
    virtual ~BaseDataset() = default;

    torch::Tensor apply_augments(torch::Tensor data);
    torch::Tensor apply_padding_or_crop(torch::Tensor data);
    torch::Tensor apply_inst_norm(torch::Tensor data, bool return_norm = false);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_inst_norm_with_params(torch::Tensor data);

protected:
    std::string data_path;
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

private:
    mutable std::mt19937 rng_;
    
    torch::Tensor apply_downsampling(torch::Tensor data, int step);
 template<typename T>
    T get_arg(const std::unordered_map<std::string, torch::Tensor>& args, 
              const std::string& key, const T& default_value);
    
    std::string get_string_arg(const std::unordered_map<std::string, torch::Tensor>& args,
                              const std::string& key, const std::string& default_value);

};
