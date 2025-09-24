#include "dataset.h"

// ========================================================================================
// UTILITY FUNCTIONS
// ========================================================================================

torch::Tensor centerCrop(const torch::Tensor& x, std::pair<int64_t,int64_t> tShape) {
    auto sizes = x.sizes();
    int64_t dim = sizes.size();

    int64_t H = sizes[dim - 2];
    int64_t W = sizes[dim - 1];

    int64_t target_h = tShape.first;
    int64_t target_w = tShape.second;

    int64_t start_h = (H - target_h) / 2;
    int64_t start_w = (W - target_w) / 2;
    int64_t end_h = start_h + target_h;
    int64_t end_w = start_w + target_w;

    if (dim == 4) {
        return x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(start_h, end_h),
                        torch::indexing::Slice(start_w, end_w)});
    } else if (dim == 5) {
        return x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(start_h, end_h),
                        torch::indexing::Slice(start_w, end_w)});
    } else {
        throw std::runtime_error("center_crop: expected 4D or 5D tensor, got " + std::to_string(dim) + "D");
    }
}

torch::Tensor downSamplingData(const torch::Tensor& data, const std::vector<double>& zoomFactors){
    int dim = data.dim();
    torch::nn::functional::InterpolateFuncOptions opts;

    if (dim == 4) {
        std::vector<int64_t> out_size = {
            static_cast<int64_t>(data.size(2) * zoomFactors[2]),
            static_cast<int64_t>(data.size(3) * zoomFactors[3])
        };
        opts = opts.size(out_size).mode(torch::kBicubic).align_corners(false);
    } else if (dim == 5) {
        std::vector<int64_t> out_size = {
            static_cast<int64_t>(data.size(2) * zoomFactors[2]),
            static_cast<int64_t>(data.size(3) * zoomFactors[3]),
            static_cast<int64_t>(data.size(4) * zoomFactors[4])
        };
        opts = opts.size(out_size).mode(torch::kTrilinear).align_corners(false);
    } else {
        throw std::runtime_error("downsampling_data only supports 4D and 5D tensors.");
    }

    return torch::nn::functional::interpolate(data, opts);
}

torch::Tensor deblockHW(const torch::Tensor& data,
                        int64_t nH,
                        int64_t nW,
                        const std::vector<int64_t>& padding) {
    if (padding.size() != 4) {
        throw std::invalid_argument("padding must have 4 values: top, down, left, right");
    }

    auto sizes = data.sizes();
    if (sizes.size() != 5) {
        throw std::invalid_argument("Expected 5D input tensor (V, S_blk, T, h_block, w_block)");
    }

    int64_t V       = sizes[0];
    int64_t sBlk    = sizes[1];
    int64_t T       = sizes[2];
    int64_t hBlock  = sizes[3];
    int64_t wBlock  = sizes[4];

    int64_t top     = padding[0];
    int64_t down    = padding[1];
    int64_t left    = padding[2];
    int64_t right   = padding[3];

    if (sBlk % (nH * nW) != 0) {
        throw std::invalid_argument("sBlk must be divisible by nH * nW");
    }
    int64_t sOrig = sBlk / (nH * nW);

    std::vector<int64_t> target_shape = {V, sOrig, nH, nW, T, hBlock, wBlock};
    auto reshaped = data.reshape(target_shape);

    auto permuted = reshaped.permute({0, 1, 4, 2, 5, 3, 6});
    auto merged = permuted.reshape({V, sOrig, T, nH * hBlock, nW * wBlock});

    int64_t hP = nH * hBlock;
    int64_t wP = nW * wBlock;
    int64_t H = hP - top - down;
    int64_t W = wP - left - right;

    auto result = merged.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(top, top + H),
        torch::indexing::Slice(left, left + W)
    });

    return result;
}

std::tuple<torch::Tensor, std::tuple<int64_t, int64_t, std::vector<int64_t>>>
blockHW(const torch::Tensor& data, std::pair<int64_t, int64_t> block_size){

    int64_t hBlock = block_size.first;
    int64_t wBlock = block_size.second;

    auto sizes = data.sizes();
    int64_t V = sizes[0];
    int64_t S = sizes[1];
    int64_t T = sizes[2];
    int64_t H = sizes[3];
    int64_t W = sizes[4];

    int64_t H_target = static_cast<int64_t>(std::ceil((double)H / hBlock)) * hBlock;
    int64_t dH = H_target - H;
    int64_t top  = dH / 2;
    int64_t down = dH - top;
    int64_t nH = H_target / hBlock;

    int64_t W_target = static_cast<int64_t>(std::ceil((double)W / wBlock)) * wBlock;
    int64_t dW = W_target - W;
    int64_t left  = dW / 2;
    int64_t right = dW - left;
    int64_t nW = W_target / wBlock;

    auto reshaped = data.view({V * S, T, H, W});

    auto padded = torch::nn::functional::pad(
        reshaped,
        torch::nn::functional::PadFuncOptions({left, right, top, down}).mode(torch::kReflect)
    );

    auto paddedSizes = padded.sizes();
    int64_t H_p = paddedSizes[2];
    int64_t W_p = paddedSizes[3];
    auto restored = padded.view({V, S, T, H_p, W_p});

    auto blocked = restored.reshape({V, S, T, nH, hBlock, nW, wBlock});
    blocked = blocked.permute({0, 1, 3, 5, 2, 4, 6});
    blocked = blocked.reshape({V, S * nH * nW, T, hBlock, wBlock});

    std::vector<int64_t> padding = {top, down, left, right};

    return {blocked, {nH, nW, padding}};
}

std::pair<std::vector<std::pair<int, float>>, std::vector<int>>
data_filtering(const torch::Tensor& data, int nFrame) {
    auto sizes = data.sizes();
    int V = sizes[0];
    int S = sizes[1];
    int T = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    if (T % nFrame != 0){
        throw std::runtime_error("T must be divisible by nFrame");
    }

    int samples = T / nFrame;
    std::vector<std::pair<int, float>> filteredBlocks;
    std::vector<int> filteredLabels;

    for (int v = 0; v < V; v++) {
        for (int s = 0; s < S; s++) {
            for (int blk_idx = 0; blk_idx < samples; blk_idx++) {
                int start = blk_idx * nFrame;
                int end   = (blk_idx + 1) * nFrame;

                torch::Tensor block = data.index({v, s, torch::indexing::Slice(start, end)});

                torch::Tensor flat = block.reshape({-1});
                float first_val = flat[0].item<float>();

                if (torch::allclose(flat, torch::full_like(flat, first_val))) {
                    int label = v * (S * samples) + s * samples + blk_idx;
                    filteredBlocks.push_back({label, first_val});
                    filteredLabels.push_back(label);
                }
            }
        }
    }

    return {filteredBlocks, filteredLabels};
}

std::unordered_map<int, int> buildReverseIdMap(
    int visibleLength,
    const std::vector<int>& filteredLabels) {

    std::unordered_set<int> filteredSet(filteredLabels.begin(), filteredLabels.end());
    std::vector<int> validIds;
    validIds.reserve(visibleLength);

    for (int i = 0; i < visibleLength; ++i){
        if (filteredSet.find(i) == filteredSet.end()){
            validIds.push_back(i);
        }
    }
    
    std::unordered_map<int, int> reverseMap;
    for(size_t i = 0; i < validIds.size(); ++i){
        reverseMap[static_cast<int>(i)] = validIds[i];
    }

    return reverseMap;
}

// Helper functions for argument parsing
std::optional<std::pair<int, int>> get_optional_pair_arg(
    const std::unordered_map<std::string, torch::Tensor>& args,
    const std::string& key) {

    auto it = args.find(key);
    if (it != args.end()) {
        auto tensor = it->second;
        if (tensor.numel() >= 2) {
            return std::make_pair(tensor[0].item<int>(), tensor[1].item<int>());
        }
    }
    return std::nullopt;
}

bool get_bool_arg(const std::unordered_map<std::string, torch::Tensor>& args,
                  const std::string& key, bool default_value) {
    auto it = args.find(key);
    if (it != args.end()) {
        return it->second.item<bool>();
    }
    return default_value;
}

std::unordered_map<std::string, int> get_augment_type_arg(
    const std::unordered_map<std::string, torch::Tensor>& args) {

    std::unordered_map<std::string, int> result;

    auto downsample_it = args.find("augment_downsample");
    if (downsample_it != args.end()) {
        result["downsample"] = downsample_it->second.item<int>();
    }

    auto randsample_it = args.find("augment_randsample");
    if (randsample_it != args.end()) {
        result["randsample"] = randsample_it->second.item<int>();
    }

    return result;
}

// ========================================================================================
// BASE DATASET CLASS IMPLEMENTATION
// ========================================================================================

BaseDataset::BaseDataset(const std::unordered_map<std::string, torch::Tensor>& args) : rng_(std::random_device{}()) {
    if (args.find("data_path") == args.end()) {
        throw std::invalid_argument("data_path is required");
    }
    data_path = get_string_arg(args, "data_path", "");

    dataset_name = get_string_arg(args, "name", "Customized Dataset");

    if (args.find("variable_idx") != args.end()) {
        variable_idx = args.at("variable_idx").item<int>();
    }

    section_range = get_optional_pair_arg(args, "section_range");
    frame_range = get_optional_pair_arg(args, "frame_range");

    if (args.find("n_frame") == args.end()) {
        throw std::invalid_argument("n_frame is required");
    }
    n_frame = args.at("n_frame").item<int>();

    resolution = get_optional_pair_arg(args, "resolution");

    train_size = get_arg<int>(args, "train_size", 256);
    inst_norm = get_bool_arg(args, "inst_norm", true);
    augment_type = get_augment_type_arg(args);
    norm_type = get_string_arg(args, "norm_type", "mean_range");
    train_mode = get_bool_arg(args, "train", false);

    test_size = {get_arg<int>(args, "test_size_h", 256), get_arg<int>(args, "test_size_w", 256)};
    n_overlap = get_arg<int>(args, "n_overlap", 0);
    downsampling = get_arg<int>(args, "downsampling", 1);

    if (augment_type.find("downsample") != augment_type.end()) {
        max_downsample = augment_type["downsample"];
    } else if (augment_type.find("randsample") != augment_type.end()) {
        max_downsample = augment_type["randsample"];
    } else {
        max_downsample = 1;
    }

    enable_ds = true;
}

torch::Tensor BaseDataset::apply_augments(torch::Tensor data) {
    if (augment_type.find("downsample") != augment_type.end() && enable_ds) {
        data = apply_downsampling(data, augment_type["downsample"]);
    } else if (augment_type.find("randsample") != augment_type.end() && enable_ds) {
        std::uniform_int_distribution<> dis(1, augment_type["randsample"]);
        int step = dis(rng_);
        data = apply_downsampling(data, step);
    }
    return data;
}

torch::Tensor BaseDataset::apply_padding_or_crop(torch::Tensor data) {
    int cur_size = data.size(-1);

    if (train_size > cur_size) {
        int pad_size = train_size - cur_size;
        int pad_left = pad_size / 2;
        int pad_right = pad_size - pad_left;

        data = torch::nn::functional::pad(data.unsqueeze(0),
            torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_left, pad_right})
                .mode(torch::kReflect)).squeeze(0);
    } else if (train_size < cur_size) {
        int start_h = std::uniform_int_distribution<>(0, data.size(-2) - train_size)(rng_);
        int start_w = std::uniform_int_distribution<>(0, data.size(-1) - train_size)(rng_);

        data = data.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(start_h, start_h + train_size),
            torch::indexing::Slice(start_w, start_w + train_size)
        });
    }

    return data;
}

torch::Tensor BaseDataset::apply_inst_norm(torch::Tensor data, bool return_norm) {
    if (return_norm) {
        auto result = apply_inst_norm_with_params(data);
        return std::get<0>(result);
    }

    torch::Tensor offset, scale;

    if (norm_type == "mean_range") {
        offset = torch::mean(data).view({1, 1, 1});
        scale = data.max() - data.min();

        if (scale.item<float>() == 0) {
            throw std::runtime_error("Scale is zero.");
        }

        data = (data - offset) / scale;

    } else if (norm_type == "min_max") {
        auto dmin = data.min();
        auto dmax = data.max();
        offset = (dmax + dmin) / 2;
        scale = (dmax - dmin) / 2;

        if (scale.item<float>() == 0) {
            throw std::runtime_error("Scale is zero.");
        }

        data = (data - offset) / scale;

    } else if (norm_type == "mean_range_hw") {
        offset = torch::mean(data, {-2, -1}, true);
        scale = torch::amax(data, {-2, -1}, true) - torch::amin(data, {-2, -1}, true);

        if (torch::any(scale == 0).item<bool>()) {
            throw std::runtime_error("Scale contains zero values.");
        }

        data = (data - offset) / scale;

    } else {
        throw std::runtime_error("Normalization type " + norm_type + " not implemented.");
    }

    return data;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BaseDataset::apply_inst_norm_with_params(torch::Tensor data) {
    torch::Tensor offset, scale;

    if (norm_type == "mean_range") {
        offset = torch::mean(data).view({1, 1, 1});
        scale = data.max() - data.min();

        if (scale.item<float>() == 0) {
            throw std::runtime_error("Scale is zero.");
        }

        data = (data - offset) / scale;
        offset = offset.view({1, 1, 1});
        scale = scale.view({1, 1, 1});

    } else if (norm_type == "min_max") {
        auto dmin = data.min();
        auto dmax = data.max();
        offset = (dmax + dmin) / 2;
        scale = (dmax - dmin) / 2;

        if (scale.item<float>() == 0) {
            throw std::runtime_error("Scale is zero.");
        }

        data = (data - offset) / scale;
        offset = offset.view({1, 1, 1});
        scale = scale.view({1, 1, 1});

    } else if (norm_type == "mean_range_hw") {
        offset = torch::mean(data, {-2, -1}, true);
        scale = torch::amax(data, {-2, -1}, true) - torch::amin(data, {-2, -1}, true);

        if (torch::any(scale == 0).item<bool>()) {
            throw std::runtime_error("Scale contains zero values.");
        }

        data = (data - offset) / scale;

    } else {
        throw std::runtime_error("Normalization type " + norm_type + " not implemented.");
    }

    return std::make_tuple(data, offset, scale);
}

torch::Tensor BaseDataset::apply_downsampling(torch::Tensor data, int step) {
    if (step <= 0) {
        throw std::invalid_argument("Downsampling step must be positive");
    }
    
    // Apply downsampling to the last dimension
    auto indices = torch::arange(0, data.size(-1), step, data.device());
    return data.index_select(-1, indices);
}

template<typename T>
T BaseDataset::get_arg(const std::unordered_map<std::string, torch::Tensor>& args,
                      const std::string& key, const T& default_value) {
    auto it = args.find(key);
    if (it != args.end()) {
        return it->second.item<T>();
    }
    return default_value;
}

std::string BaseDataset::get_string_arg(const std::unordered_map<std::string, torch::Tensor>& args,
                                       const std::string& key, const std::string& default_value) {
    // Note: In a real implementation, you'd need a way to store and retrieve strings
    // This is a placeholder implementation
    return default_value;
}

// Explicit template instantiations
template int BaseDataset::get_arg<int>(const std::unordered_map<std::string, torch::Tensor>&, const std::string&, const int&);
template float BaseDataset::get_arg<float>(const std::unordered_map<std::string, torch::Tensor>&, const std::string&, const float&);
template bool BaseDataset::get_arg<bool>(const std::unordered_map<std::string, torch::Tensor>&, const std::string&, const bool&);

// ========================================================================================
// SCIENTIFIC DATASET CLASS IMPLEMENTATION
// ========================================================================================

ScientificDataset::ScientificDataset(const std::unordered_map<std::string, torch::Tensor>& args)
    : BaseDataset(args), dataset_length(0), visible_length(0) {

    std::cout << "*************** Loading " << dataset_name << " ***************\n";

    auto data = load_dataset(data_path, variable_idx, section_range, frame_range);

    auto sizes = data.sizes();
    shape_org = std::vector<int64_t>(sizes.begin(), sizes.end());

    delta_t = n_frame - n_overlap;
    int64_t T = data.size(2);
    t_samples = static_cast<int64_t>(std::ceil(static_cast<double>(T - n_frame) / delta_t)) + 1;

    pad_T = (t_samples - 1) * delta_t + n_frame - T;

    if (pad_T > 0) {
        auto tail_frames = data.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(T - pad_T, T)
        });

        tail_frames = torch::flip(tail_frames, {2});
        data = torch::cat({data, tail_frames}, 2);
    }

    if (!inst_norm) {
        if (norm_type == "mean_range_hw") {
            throw std::runtime_error("mean_range_hw normalization requires inst_norm=true");
        }

        torch::Tensor offset, scale;
        if (norm_type == "mean_range") {
            offset = torch::mean(data, {1, 2, 3, 4}, true);
            auto data_max = torch::amax(data, {1, 2, 3, 4}, true);
            auto data_min = torch::amin(data, {1, 2, 3, 4}, true);
            scale = data_max - data_min;
            data = (data - offset) / scale;
        } else if (norm_type == "min_max") {
            auto data_min = torch::amin(data, {1, 2, 3, 4}, true);
            auto data_max = torch::amax(data, {1, 2, 3, 4}, true);
            offset = (data_max + data_min) / 2;
            scale = (data_max - data_min) / 2;
            data = (data - offset) / scale;
        } else if (norm_type == "std") {
            offset = torch::mean(data, {1, 2, 3, 4}, true);
            scale = torch::std(data, {1, 2, 3, 4}, true);
            data = (data - offset) / scale;
        }

        var_offset = offset.to(torch::kFloat);
        var_scale = scale.to(torch::kFloat);
    }

    data = data.to(torch::kFloat);

    if (!train_mode) {
        auto block_result = blockHW(data, test_size);
        data = std::get<0>(block_result);
        block_info = std::get<1>(block_result);
    }

    auto filter_result = data_filtering(data, delta_t);
    filtered_blocks = filter_result.first;
    filtered_labels = filter_result.second;

    data_input = data;
    auto final_sizes = data.sizes();
    shape = std::vector<int64_t>(final_sizes.begin(), final_sizes.end());

    visible_length = update_length();
    reverse_id_map = buildReverseIdMap(visible_length, filtered_labels);
}

// ========================================================================================
// CORE UNIFIED LOAD_DATASET METHOD
// ========================================================================================

torch::Tensor ScientificDataset::load_dataset(const std::variant<std::string, torch::Tensor>& input,
                                             std::optional<int> variable_idx,
                                             std::optional<std::pair<int, int>> section_range,
                                             std::optional<std::pair<int, int>> frame_range,
                                             std::optional<std::string> mpi_mode) {
    
    // Check if MPI mode is requested
    if (mpi_mode.has_value() && mpi_mode.value() == "mpi") {
        // MPI mode - only works with file paths
        if (std::holds_alternative<std::string>(input)) {
            const std::string& file_path = std::get<std::string>(input);
            return load_from_binary_file_mpi(file_path, variable_idx, section_range, frame_range);
        } else {
            throw std::runtime_error("MPI mode only supports file path input, not tensor input");
        }
    }
    
    // Regular (non-MPI) mode
    if (std::holds_alternative<std::string>(input)) {
        // Input is a file path - load from binary file
        const std::string& file_path = std::get<std::string>(input);
        return load_from_binary_file(file_path, variable_idx, section_range, frame_range);
    } 
    else if (std::holds_alternative<torch::Tensor>(input)) {
        // Input is a tensor - load from memory
        const torch::Tensor& data = std::get<torch::Tensor>(input);
        return load_from_memory(data, variable_idx, section_range, frame_range);
    }
    else {
        throw std::runtime_error("Invalid input type for load_dataset");
    }
}

// ========================================================================================
// PRIVATE HELPER METHODS
// ========================================================================================

torch::Tensor ScientificDataset::load_from_memory(const torch::Tensor& data,
                                                 std::optional<int> variable_idx,
                                                 std::optional<std::pair<int, int>> section_range,
                                                 std::optional<std::pair<int, int>> frame_range) {
    
    torch::Tensor processed_data = data.clone();
    
    // Validate input is 5D
    if (processed_data.dim() != 5) {
        throw std::runtime_error("Expected 5D tensor with shape [Variables, Sections, Time, Height, Width], got " + 
                               std::to_string(processed_data.dim()) + "D tensor");
    }
    
    std::cout << "Loading data from memory tensor with shape: " << processed_data.sizes() << std::endl;
    
    // Apply variable selection
    if (variable_idx.has_value()) {
        int var_idx = variable_idx.value();
        if (var_idx < 0 || var_idx >= processed_data.size(0)) {
            throw std::out_of_range("Variable index out of range");
        }
        processed_data = processed_data.index({var_idx});
        processed_data = processed_data.unsqueeze(0);
    }
    
    // Apply section range
    if (section_range.has_value()) {
        auto range = section_range.value();
        if (range.first < 0 || range.second > processed_data.size(1) || range.first >= range.second) {
            throw std::out_of_range("Invalid section range");
        }
        processed_data = processed_data.index({torch::indexing::Slice(), 
                                             torch::indexing::Slice(range.first, range.second)});
    }
    
    // Apply frame range
    if (frame_range.has_value()) {
        auto range = frame_range.value();
        if (range.first < 0 || range.second > processed_data.size(2) || range.first >= range.second) {
            throw std::out_of_range("Invalid frame range");
        }
        processed_data = processed_data.index({torch::indexing::Slice(), 
                                             torch::indexing::Slice(),
                                             torch::indexing::Slice(range.first, range.second)});
    }
    
    // Apply resolution cropping
    if (resolution.has_value()) {
        auto res = resolution.value();
        if (res.first > processed_data.size(3) || res.second > processed_data.size(4)) {
            throw std::invalid_argument("Resolution larger than data dimensions");
        }
        processed_data = centerCrop(processed_data, resolution.value());
    }
    
    dtype = processed_data.scalar_type();
    processed_data = processed_data.to(torch::kFloat);
    
    return processed_data;
}

torch::Tensor ScientificDataset::load_from_binary_file(const std::string& file_path,
                                                      std::optional<int> variable_idx,
                                                      std::optional<std::pair<int, int>> section_range,
                                                      std::optional<std::pair<int, int>> frame_range) {
    
    std::cout << "Loading data from binary file: " << file_path << std::endl;
    
    // Check if file exists
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("Binary file does not exist: " + file_path);
    }
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + file_path);
    }
    
    // Read header: shape (5 int64_t values) + dtype info
    std::vector<int64_t> shape(5);
    file.read(reinterpret_cast<char*>(shape.data()), 5 * sizeof(int64_t));
    
    // Read data type (stored as int32_t representing torch::ScalarType)
    int32_t dtype_int;
    file.read(reinterpret_cast<char*>(&dtype_int), sizeof(int32_t));
    torch::ScalarType file_dtype = static_cast<torch::ScalarType>(dtype_int);
    
    if (!file.good()) {
        throw std::runtime_error("Error reading header from binary file: " + file_path);
    }
    
    // Validate shape
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid shape in binary file header");
        }
    }
    
    // Calculate total elements
    int64_t total_elements = 1;
    for (int64_t dim : shape) {
        total_elements *= dim;
    }
    
    if (total_elements > INT64_MAX / 16) { // Safety check for memory allocation
        throw std::runtime_error("File too large to load into memory");
    }
    
    // Read binary data
    size_t dtype_size = torch::elementSize(file_dtype);
    std::vector<uint8_t> buffer(total_elements * dtype_size);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    
    if (!file.good() || file.gcount() != static_cast<std::streamsize>(buffer.size())) {
        throw std::runtime_error("Error reading data from binary file: " + file_path);
    }
    
    file.close();
    
    // Create tensor from binary data
    torch::Tensor data = torch::from_blob(buffer.data(), shape, file_dtype).clone();
    
    std::cout << "Loaded tensor with shape: " << data.sizes() << " and dtype: " << data.scalar_type() << std::endl;
    
    // Apply the same processing as memory method
    return load_from_memory(data, variable_idx, section_range, frame_range);
}

// ========================================================================================
// BINARY FILE WRITING METHODS
// ========================================================================================

void ScientificDataset::write_binary_file(const torch::Tensor& tensor, 
                                         const std::string& file_path,
                                         std::optional<std::string> mpi_mode,
                                         std::optional<std::vector<int64_t>> global_shape) {
    
    // Check if MPI mode is requested
    if (mpi_mode.has_value() && mpi_mode.value() == "mpi") {
        // MPI mode
        if (!global_shape.has_value()) {
            // Calculate global shape automatically
            auto calculated_global_shape = calculate_global_shape_mpi(tensor);
            write_binary_file_mpi(tensor, file_path, calculated_global_shape);
        } else {
            // Use provided global shape
            write_binary_file_mpi(tensor, file_path, global_shape.value());
        }
        return;
    }
    
    // Regular (non-MPI) mode
    if (tensor.dim() != 5) {
        throw std::runtime_error("Expected 5D tensor for binary file writing");
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path file_dir = std::filesystem::path(file_path).parent_path();
    if (!file_dir.empty() && !std::filesystem::exists(file_dir)) {
        std::filesystem::create_directories(file_dir);
    }
    
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create binary file: " + file_path);
    }
    
    // Write header: shape (5 int64_t values)
    auto sizes = tensor.sizes();
    std::vector<int64_t> shape(sizes.begin(), sizes.end());
    file.write(reinterpret_cast<const char*>(shape.data()), 5 * sizeof(int64_t));
    
    // Write data type
    int32_t dtype_int = static_cast<int32_t>(tensor.scalar_type());
    file.write(reinterpret_cast<const char*>(&dtype_int), sizeof(int32_t));
    
    // Write tensor data
    auto contiguous_tensor = tensor.contiguous();
    size_t data_size = contiguous_tensor.numel() * contiguous_tensor.element_size();
    file.write(reinterpret_cast<const char*>(contiguous_tensor.data_ptr()), data_size);
    
    if (!file.good()) {
        throw std::runtime_error("Error writing to binary file: " + file_path);
    }
    
    file.close();
    std::cout << "Written binary file: " << file_path << " with shape: " << tensor.sizes() << std::endl;
}

// Backward compatibility: keep the old signature
void ScientificDataset::write_binary_file(const torch::Tensor& tensor, const std::string& file_path) {
    write_binary_file(tensor, file_path, std::nullopt, std::nullopt);
}

// ========================================================================================
// DATASET INTERFACE METHODS
// ========================================================================================

int64_t ScientificDataset::update_length() {
    dataset_length = shape[0] * shape[1] * t_samples;
    return dataset_length;
}

size_t ScientificDataset::size() const {
    return visible_length - filtered_blocks.size();
}

torch::Tensor ScientificDataset::original_data() const {
    torch::Tensor data = data_input.clone();

    if (!train_mode) {
        data = deblockHW(data, std::get<0>(block_info),
                        std::get<1>(block_info), std::get<2>(block_info));
    }

    if (!inst_norm) {
        data = data * var_scale + var_offset;
    }

    return data;
}

torch::Tensor ScientificDataset::input_data() const {
    auto data = original_data();
    data = data.index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(0, shape[2] - pad_T)});
    return data;
}

torch::Tensor ScientificDataset::recons_data(const torch::Tensor& recons_data) const {
    // Remove padding from temporal dimension
    return recons_data.index({torch::indexing::Slice(),
                             torch::indexing::Slice(),
                             torch::indexing::Slice(0, shape[2] - pad_T)});
}

torch::Tensor ScientificDataset::deblocking_hw(const torch::Tensor& data) const {
    if (!train_mode) {
        return deblockHW(data, std::get<0>(block_info),
                        std::get<1>(block_info), std::get<2>(block_info));
    }
    return data; // In training mode, no deblocking needed
}

std::unordered_map<std::string, torch::Tensor> ScientificDataset::post_processing(
    const torch::Tensor& data, int var_idx, bool is_training) {

    torch::Tensor processed_data = data.clone();

    if (is_training) {
        processed_data = apply_augments(processed_data);
        processed_data = apply_padding_or_crop(processed_data);
    }

    torch::Tensor offset, scale;

    if (inst_norm) {
        auto norm_result = apply_inst_norm_with_params(processed_data);
        processed_data = std::get<0>(norm_result);
        offset = std::get<1>(norm_result);
        scale = std::get<2>(norm_result);
    } else {
        if (var_idx >= var_offset.size(0)) {
            throw std::out_of_range("Variable index out of range for normalization parameters");
        }
        offset = var_offset.index({var_idx}).view({1, 1, 1});
        scale = var_scale.index({var_idx}).view({1, 1, 1});
    }

    std::unordered_map<std::string, torch::Tensor> data_dict;
    data_dict["input"] = processed_data.unsqueeze(0);
    data_dict["offset"] = offset.unsqueeze(0);
    data_dict["scale"] = scale.unsqueeze(0);

    return data_dict;
}

// ========================================================================================
// MPI METHODS (conditional compilation)
// ========================================================================================

#ifdef MPI_VERSION

torch::Tensor ScientificDataset::load_from_binary_file_mpi(const std::string& file_path,
                                                           std::optional<int> variable_idx,
                                                           std::optional<std::pair<int, int>> section_range,
                                                           std::optional<std::pair<int, int>> frame_range) {
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<int64_t> global_shape(5);
    torch::ScalarType file_dtype;
    
    // Rank 0 reads header and broadcasts to all processes
    if (rank == 0) {
        std::cout << "Rank 0: Reading header from binary file: " << file_path << std::endl;
        
        if (!std::filesystem::exists(file_path)) {
            throw std::runtime_error("MPI: Binary file does not exist: " + file_path);
        }
        
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open binary file: " + file_path);
        }
        
        // Read header
        file.read(reinterpret_cast<char*>(global_shape.data()), 5 * sizeof(int64_t));
        
        int32_t dtype_int;
        file.read(reinterpret_cast<char*>(&dtype_int), sizeof(int32_t));
        file_dtype = static_cast<torch::ScalarType>(dtype_int);
        
        file.close();
        
        if (!file.good()) {
            throw std::runtime_error("Error reading header from binary file: " + file_path);
        }
        
        // Validate shape
        for (int64_t dim : global_shape) {
            if (dim <= 0) {
                throw std::runtime_error("Invalid shape in MPI binary file header");
            }
        }
        
        std::cout << "Global tensor shape: [" << global_shape[0] << ", " << global_shape[1] 
                  << ", " << global_shape[2] << ", " << global_shape[3] << ", " << global_shape[4] << "]" << std::endl;
    }
    
    // Broadcast shape and dtype to all processes
    MPI_Bcast(global_shape.data(), 5, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    int32_t dtype_int = static_cast<int32_t>(file_dtype);
    MPI_Bcast(&dtype_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    file_dtype = static_cast<torch::ScalarType>(dtype_int);
    
    // Calculate 1D domain decomposition along the second dimension (sections)
    int64_t sections_total = global_shape[1];  // Total sections
    int64_t sections_per_rank = sections_total / size;
    int64_t remainder = sections_total % size;
    
    // Calculate this rank's section range
    int64_t section_start = rank * sections_per_rank + std::min(static_cast<int64_t>(rank), remainder);
    int64_t section_count = sections_per_rank + (rank < remainder ? 1 : 0);
    int64_t section_end = section_start + section_count;
    
    std::cout << "Rank " << rank << ": Loading sections [" << section_start 
              << ", " << section_end << ") out of " << sections_total << std::endl;
    
    // Calculate local tensor shape
    std::vector<int64_t> local_shape = {global_shape[0], section_count, global_shape[2], 
                                       global_shape[3], global_shape[4]};
    
    // Calculate file offsets
    size_t header_size = 5 * sizeof(int64_t) + sizeof(int32_t);
    size_t dtype_size = torch::elementSize(file_dtype);
    
    // Elements per section (Variables * Time * Height * Width)
    int64_t elements_per_section = global_shape[0] * global_shape[2] * global_shape[3] * global_shape[4];
    size_t bytes_per_section = elements_per_section * dtype_size;
    
    // File offset for this rank's data
    size_t file_offset = header_size + section_start * bytes_per_section;
    size_t read_size = section_count * bytes_per_section;
    
    // Use MPI-IO for parallel reading
    MPI_File mpi_file;
    MPI_Status status;
    
    int result = MPI_File_open(MPI_COMM_WORLD, file_path.c_str(), 
                              MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);
    if (result != MPI_SUCCESS) {
        throw std::runtime_error("Failed to open file with MPI-IO: " + file_path);
    }
    
    // Allocate buffer for local data
    std::vector<uint8_t> buffer(read_size);
    
    // Read this rank's portion
    result = MPI_File_read_at(mpi_file, file_offset, buffer.data(), read_size, 
                             MPI_BYTE, &status);
    if (result != MPI_SUCCESS) {
        MPI_File_close(&mpi_file);
        throw std::runtime_error("Failed to read data with MPI-IO");
    }
    
    MPI_File_close(&mpi_file);
    
    // Create local tensor
    torch::Tensor local_data = torch::from_blob(buffer.data(), local_shape, file_dtype).clone();
    
    std::cout << "Rank " << rank << ": Loaded local tensor with shape: " << local_data.sizes() << std::endl;
    
    // Apply processing (variable selection, frame range, etc.) to local data
    // Note: section_range is ignored in MPI mode as it's handled by domain decomposition
    return load_from_memory(local_data, variable_idx, std::nullopt, frame_range);
}

void ScientificDataset::write_binary_file_mpi(const torch::Tensor& local_tensor, 
                                             const std::string& file_path,
                                             const std::vector<int64_t>& global_shape) {
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (local_tensor.dim() != 5) {
        throw std::runtime_error("Expected 5D local tensor for MPI binary file writing");
    }
    
    if (global_shape.size() != 5) {
        throw std::runtime_error("Global shape must have 5 dimensions");
    }
    
    // Validate global shape
    for (int64_t dim : global_shape) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid global shape for MPI write");
        }
    }
    
    // Rank 0 writes the header
    if (rank == 0) {
        std::cout << "Rank 0: Writing header to binary file: " << file_path << std::endl;
        
        // Create directory if it doesn't exist
        std::filesystem::path file_dir = std::filesystem::path(file_path).parent_path();
        if (!file_dir.empty() && !std::filesystem::exists(file_dir)) {
            std::filesystem::create_directories(file_dir);
        }
        
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create binary file: " + file_path);
        }
        
        // Write global shape
        file.write(reinterpret_cast<const char*>(global_shape.data()), 5 * sizeof(int64_t));
        
        // Write data type
        int32_t dtype_int = static_cast<int32_t>(local_tensor.scalar_type());
        file.write(reinterpret_cast<const char*>(&dtype_int), sizeof(int32_t));
        
        if (!file.good()) {
            throw std::runtime_error("Error writing header to MPI binary file");
        }
        
        file.close();
        
        std::cout << "Global tensor shape: [" << global_shape[0] << ", " << global_shape[1] 
                  << ", " << global_shape[2] << ", " << global_shape[3] << ", " << global_shape[4] << "]" << std::endl;
    }
    
    // Wait for rank 0 to finish writing header
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Calculate file offset for this rank's data
    size_t header_size = 5 * sizeof(int64_t) + sizeof(int32_t);
    size_t dtype_size = local_tensor.element_size();
    
    // Calculate this rank's section start (assume 1D decomposition along dim 1)
    int64_t sections_total = global_shape[1];
    int64_t sections_per_rank = sections_total / size;
    int64_t remainder = sections_total % size;
    int64_t section_start = rank * sections_per_rank + std::min(static_cast<int64_t>(rank), remainder);
    
    // Elements per section
    int64_t elements_per_section = global_shape[0] * global_shape[2] * global_shape[3] * global_shape[4];
    size_t bytes_per_section = elements_per_section * dtype_size;
    size_t file_offset = header_size + section_start * bytes_per_section;
    
    // Use MPI-IO for parallel writing
    MPI_File mpi_file;
    int result = MPI_File_open(MPI_COMM_WORLD, file_path.c_str(), 
                              MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);
    if (result != MPI_SUCCESS) {
        throw std::runtime_error("Failed to open file for writing with MPI-IO: " + file_path);
    }
    
    // Write this rank's data
    auto contiguous_tensor = local_tensor.contiguous();
    size_t write_size = contiguous_tensor.numel() * dtype_size;
    
    MPI_Status status;
    result = MPI_File_write_at(mpi_file, file_offset, contiguous_tensor.data_ptr(), 
                              write_size, MPI_BYTE, &status);
    if (result != MPI_SUCCESS) {
        MPI_File_close(&mpi_file);
        throw std::runtime_error("Failed to write data with MPI-IO");
    }
    
    MPI_File_close(&mpi_file);
    
    std::cout << "Rank " << rank << ": Written local tensor with shape: " << local_tensor.sizes() 
              << " at offset: " << file_offset << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "MPI binary file write completed: " << file_path << std::endl;
    }
}

std::vector<int64_t> ScientificDataset::calculate_global_shape_mpi(const torch::Tensor& local_tensor) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (local_tensor.dim() != 5) {
        throw std::runtime_error("Expected 5D local tensor");
    }
    
    auto local_shape = local_tensor.sizes();
    
    // Gather section counts from all ranks
    std::vector<int64_t> section_counts(size);
    int64_t local_sections = local_shape[1];
    
    MPI_Allgather(&local_sections, 1, MPI_LONG_LONG, 
                  section_counts.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
    
    // Calculate total sections
    int64_t total_sections = 0;
    for (int64_t count : section_counts) {
        total_sections += count;
    }
    
    // Global shape: same as local except for sections dimension
    std::vector<int64_t> global_shape = {local_shape[0], total_sections, local_shape[2], 
                                        local_shape[3], local_shape[4]};
    
    if (rank == 0) {
        std::cout << "Calculated global shape: [" << global_shape[0] << ", " << global_shape[1] 
                  << ", " << global_shape[2] << ", " << global_shape[3] << ", " << global_shape[4] << "]" << std::endl;
    }
    
    return global_shape;
}

#else
// Non-MPI versions - throw errors
torch::Tensor ScientificDataset::load_from_binary_file_mpi(const std::string& file_path,
                                                           std::optional<int> variable_idx,
                                                           std::optional<std::pair<int, int>> section_range,
                                                           std::optional<std::pair<int, int>> frame_range) {
    throw std::runtime_error("MPI support not compiled. Please compile with MPI to use MPI methods.");
}

void ScientificDataset::write_binary_file_mpi(const torch::Tensor& local_tensor, 
                                             const std::string& file_path,
                                             const std::vector<int64_t>& global_shape) {
    throw std::runtime_error("MPI support not compiled. Please compile with MPI to use MPI methods.");
}

std::vector<int64_t> ScientificDataset::calculate_global_shape_mpi(const torch::Tensor& local_tensor) {
    throw std::runtime_error("MPI support not compiled. Please compile with MPI to use MPI methods.");
}

#endif


std::unordered_map<std::string, torch::Tensor> ScientificDataset::get_item(size_t idx) {
    idx = idx % dataset_length;

    if (!filtered_labels.empty()) {
        auto it = reverse_id_map.find(static_cast<int>(idx));
        if (it != reverse_id_map.end()) {
            idx = it->second;
        }
    }

    int64_t idx0 = idx / (shape[1] * t_samples);
    int64_t idx1 = (idx / t_samples) % shape[1];
    int64_t idx2 = idx % t_samples;

    int64_t start_t = idx2 * delta_t;
    int64_t end_t   = start_t + n_frame;

    torch::Tensor data = data_input.index({
        static_cast<int64_t>(idx0),
        static_cast<int64_t>(idx1),
        torch::indexing::Slice(start_t, end_t)
    });

    auto data_dict = post_processing(data, static_cast<int>(idx0), train_mode);

    torch::Tensor index_tensor = torch::tensor({idx0, idx1, start_t, end_t}, torch::kLong);
    data_dict["index"] = index_tensor;

    return data_dict;
}

