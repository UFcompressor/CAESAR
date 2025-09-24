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
#include <mpi.h>
#include <fstream>



torch::Tensor centerCrop(const torch::Tensor& x, std::pair<int64_t,int64_t> tShape);
torch::Tensor downSamplingData(const torch::Tensor& data, const std::vector<double>& zoomFactors);
torch::Tensor deblockHW(const torch::Tensor& data, int64_t nH, int64_t nW, const std::vector<int64_t>& padding);
std::tuple<torch::Tensor, std::tuple<int64_t, int64_t, std::vector<int64_t>>> blockHW(const torch::Tensor& data, std::pair<int64_t, int64_t> block_size);
std::pair<std::vector<std::pair<int, float>>, std::vector<int>> data_filtering(const torch::Tensor& data, int nFrame);
std::unordered_map<int, int> buildReverseIdMap(int visibleLength, const std::vector<int>& filteredLabels);

// Helper functions
std::optional<std::pair<int, int>> get_optional_pair_arg(const std::unordered_map<std::string, torch::Tensor>& args, const std::string& key);
bool get_bool_arg(const std::unordered_map<std::string, torch::Tensor>& args, const std::string& key, bool default_value);
std::unordered_map<std::string, int> get_augment_type_arg(const std::unordered_map<std::string, torch::Tensor>& args);

class BaseDataset {
public:
    BaseDataset(const std::unordered_map<std::string, torch::Tensor>& args);
    virtual ~BaseDataset() = default;

    torch::Tensor apply_augments(torch::Tensor data);
    torch::Tensor apply_padding_or_crop(torch::Tensor data);
    torch::Tensor apply_inst_norm(torch::Tensor data, bool return_norm = false);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_inst_norm_with_params(torch::Tensor data);
    torch::Tensor apply_downsampling(torch::Tensor data, int step);

    template<typename T>
    T get_arg(const std::unordered_map<std::string, torch::Tensor>& args, const std::string& key, const T& default_value);
    
    std::string get_string_arg(const std::unordered_map<std::string, torch::Tensor>& args, const std::string& key, const std::string& default_value);

protected:
    std::string data_path;
    std::string dataset_name;
    std::optional<int> variable_idx;
    std::optional<std::pair<int, int>> section_range;
    std::optional<std::pair<int, int>> frame_range;
    int n_frame;
    std::optional<std::pair<int, int>> resolution;
    
    // Training configuration
    int train_size;
    bool inst_norm;
    std::unordered_map<std::string, int> augment_type;
    std::string norm_type;
    bool train_mode;
    
    // Test configuration
    std::pair<int, int> test_size;
    int n_overlap;
    int downsampling;
    int max_downsample;
    bool enable_ds;
    
    // Random number generator
    std::mt19937 rng_;
};

class ScientificDataset : public BaseDataset {
public:
    ScientificDataset(const std::unordered_map<std::string, torch::Tensor>& args);
    
    // Core interface - unified load_dataset method
    torch::Tensor load_dataset(const std::variant<std::string, torch::Tensor>& input,
                              std::optional<int> variable_idx = std::nullopt,
                              std::optional<std::pair<int, int>> section_range = std::nullopt,
                              std::optional<std::pair<int, int>> frame_range = std::nullopt,
                              std::optional<std::string> mpi_mode = std::nullopt);
    
    // Dataset interface methods
    size_t size() const;
    std::unordered_map<std::string, torch::Tensor> get_item(size_t idx);
    
    // Data access methods
    torch::Tensor original_data() const;
    torch::Tensor input_data() const;
    torch::Tensor recons_data(const torch::Tensor& recons_data) const;
    torch::Tensor deblocking_hw(const torch::Tensor& data) const;
    
    // Post-processing
    std::unordered_map<std::string, torch::Tensor> post_processing(const torch::Tensor& data, int var_idx, bool is_training);
    
    // Utility methods for binary file I/O
    static void write_binary_file(const torch::Tensor& tensor, 
                                 const std::string& file_path,
                                 std::optional<std::string> mpi_mode = std::nullopt,
                                 std::optional<std::vector<int64_t>> global_shape = std::nullopt);
    
    // Backward compatibility overload
    static void write_binary_file(const torch::Tensor& tensor, const std::string& file_path);
    
    // MPI-enabled methods
    torch::Tensor load_from_binary_file_mpi(const std::string& file_path,
                                           std::optional<int> variable_idx = std::nullopt,
                                           std::optional<std::pair<int, int>> section_range = std::nullopt,
                                           std::optional<std::pair<int, int>> frame_range = std::nullopt);
    
    static void write_binary_file_mpi(const torch::Tensor& local_tensor, 
                                     const std::string& file_path,
                                     const std::vector<int64_t>& global_shape);
    
    static std::vector<int64_t> calculate_global_shape_mpi(const torch::Tensor& local_tensor);

private:
    // Internal data loading methods
    torch::Tensor load_from_memory(const torch::Tensor& data,
                                  std::optional<int> variable_idx = std::nullopt,
                                  std::optional<std::pair<int, int>> section_range = std::nullopt,
                                  std::optional<std::pair<int, int>> frame_range = std::nullopt);
    
    torch::Tensor load_from_binary_file(const std::string& file_path,
                                       std::optional<int> variable_idx = std::nullopt,
                                       std::optional<std::pair<int, int>> section_range = std::nullopt,
                                       std::optional<std::pair<int, int>> frame_range = std::nullopt);
    
    // Internal dataset management
    int64_t update_length();
    
    // Dataset properties
    std::vector<int64_t> shape_org;
    std::vector<int64_t> shape;
    torch::Tensor data_input;
    torch::ScalarType dtype;
    
    // Temporal processing
    int64_t delta_t;
    int64_t t_samples;
    int64_t pad_T;
    
    // Normalization parameters (for non-instance normalization)
    torch::Tensor var_offset;
    torch::Tensor var_scale;
    
    // Block processing (for test mode)
    std::tuple<int64_t, int64_t, std::vector<int64_t>> block_info;
    
    // Data filtering
    std::vector<std::pair<int, float>> filtered_blocks;
    std::vector<int> filtered_labels;
    std::unordered_map<int, int> reverse_id_map;
    
    // Dataset length management
    int64_t dataset_length;
    size_t visible_length;
};
