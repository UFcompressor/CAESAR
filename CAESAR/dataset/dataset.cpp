#include "dataset.h"
#include <omp.h>

torch::Tensor centerCrop(const torch::Tensor& x , std::pair<int64_t , int64_t> tShape) {
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
        return x.index({ torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(start_h, end_h),
                        torch::indexing::Slice(start_w, end_w) });
    }
    else if (dim == 5) {
        return x.index({ torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(start_h, end_h),
                        torch::indexing::Slice(start_w, end_w) });
    }
    else {
        throw std::runtime_error("center_crop: expected 4D or 5D tensor, got " + std::to_string(dim) + "D");
    }
}

torch::Tensor downSamplingData(const torch::Tensor& data , const std::vector<double>& zoomFactors) {
    int dim = data.dim();
    torch::nn::functional::InterpolateFuncOptions opts;

    if (dim == 4) {
        std::vector<int64_t> out_size = {
            static_cast<int64_t>(data.size(2) * zoomFactors[2]),
            static_cast<int64_t>(data.size(3) * zoomFactors[3])
        };
        opts = opts.size(out_size).mode(torch::kBicubic).align_corners(false);
    }
    else if (dim == 5) {
        std::vector<int64_t> out_size = {
            static_cast<int64_t>(data.size(2) * zoomFactors[2]),
            static_cast<int64_t>(data.size(3) * zoomFactors[3]),
            static_cast<int64_t>(data.size(4) * zoomFactors[4])
        };
        opts = opts.size(out_size).mode(torch::kTrilinear).align_corners(false);
    }
    else {
        throw std::runtime_error("downsampling_data only supports 4D and 5D tensors.");
    }

    return torch::nn::functional::interpolate(data , opts);
}

torch::Tensor deblockHW(const torch::Tensor& data ,
    int64_t nH ,
    int64_t nW ,
    const std::vector<int64_t>& padding) {
    if (padding.size() != 4) {
        throw std::invalid_argument("padding must have 4 values: top, down, left, right");
    }

    auto sizes = data.sizes();
    if (sizes.size() != 5) {
        throw std::invalid_argument("Expected 5D input tensor (V, S_blk, T, h_block, w_block)");
    }

    int64_t V = sizes[0];
    int64_t sBlk = sizes[1];
    int64_t T = sizes[2];
    int64_t hBlock = sizes[3];
    int64_t wBlock = sizes[4];

    int64_t top = padding[0];
    int64_t down = padding[1];
    int64_t left = padding[2];
    int64_t right = padding[3];

    if (sBlk % (nH * nW) != 0) {
        throw std::invalid_argument("sBlk must be divisible by nH * nW");
    }
    int64_t sOrig = sBlk / (nH * nW);

    std::vector<int64_t> target_shape = { V, sOrig, nH, nW, T, hBlock, wBlock };
    auto reshaped = data.reshape(target_shape);

    auto permuted = reshaped.permute({ 0, 1, 4, 2, 5, 3, 6 });
    auto merged = permuted.reshape({ V, sOrig, T, nH * hBlock, nW * wBlock });

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

std::tuple<torch::Tensor , std::tuple<int64_t , int64_t , std::vector<int64_t>>>
blockHW(const torch::Tensor& data ,
    std::pair<int64_t , int64_t> block_size) {

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
    int64_t top = dH / 2;
    int64_t down = dH - top;
    int64_t nH = H_target / hBlock;

    int64_t W_target = static_cast<int64_t>(std::ceil((double)W / wBlock)) * wBlock;
    int64_t dW = W_target - W;
    int64_t left = dW / 2;
    int64_t right = dW - left;
    int64_t nW = W_target / wBlock;

    auto reshaped = data.view({ V * S, T, H, W });

    auto padded = torch::nn::functional::pad(
        reshaped ,
        torch::nn::functional::PadFuncOptions({ left, right, top, down }).mode(torch::kReflect)
    );

    auto paddedSizes = padded.sizes();
    int64_t H_p = paddedSizes[2];
    int64_t W_p = paddedSizes[3];
    auto restored = padded.view({ V, S, T, H_p, W_p });

    auto blocked = restored.reshape({ V, S, T, nH, hBlock, nW, wBlock });
    blocked = blocked.permute({ 0, 1, 3, 5, 2, 4, 6 });
    blocked = blocked.reshape({ V, S * nH * nW, T, hBlock, wBlock });

    std::vector<int64_t> padding = { top, down, left, right };

    return { blocked, {nH, nW, padding} };
}

std::pair<std::vector<std::pair<int , float>> , std::vector<int>>
data_filtering(const torch::Tensor& data , int nFrame) {
    auto sizes = data.sizes();
    int V = sizes[0];
    int S = sizes[1];
    int T = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    if (T % nFrame != 0) {
        throw std::runtime_error("T must be divisible by nFrame");
    }

    int samples = T / nFrame;
    // Thread-safe containers
    std::vector<std::pair<int , float>> filteredBlocks;
    std::vector<int> filteredLabels;

    // Total number of blocks to check
    int64_t total_blocks = (int64_t)V * S * samples;

    // Use OpenMP to parallelize the loop
#pragma omp parallel
    {
        // Each thread collects its own results to avoid locking overhead
        std::vector<std::pair<int , float>> local_blocks;
        std::vector<int> local_labels;

#pragma omp for nowait
        for (int64_t i = 0; i < total_blocks; ++i) {
            // Decode flat index 'i' back to v, s, blk_idx
            int64_t temp = i;
            int blk_idx = temp % samples;
            temp /= samples;
            int s = temp % S;
            int v = temp / S;

            int start = blk_idx * nFrame;
            int end = (blk_idx + 1) * nFrame;

            // Avoid creating new Tensor object overhead if possible, or just slice
            torch::Tensor block = data.index({ v, s, torch::indexing::Slice(start, end) });

            // Fast check: check min and max. If min == max, all values are equal.
            // This is often faster on CPU than creating a comparison tensor.
            float min_val = block.min().item<float>();
            float max_val = block.max().item<float>();

            if (min_val == max_val) {
                int label = v * (S * samples) + s * samples + blk_idx;
                local_blocks.push_back({ label, min_val });
                local_labels.push_back(label);
            }
        }

        // Merge local results into global results within a critical section
#pragma omp critical
        {
            filteredBlocks.insert(filteredBlocks.end() , local_blocks.begin() , local_blocks.end());
            filteredLabels.insert(filteredLabels.end() , local_labels.begin() , local_labels.end());
        }
    }

    // Sort to maintain deterministic order (OpenMP doesn't guarantee order)
    std::sort(filteredLabels.begin() , filteredLabels.end());
    std::sort(filteredBlocks.begin() , filteredBlocks.end() ,
        [](const auto& a , const auto& b) { return a.first < b.first; });

    return { filteredBlocks, filteredLabels };
}

std::unordered_map<int , int> buildReverseIdMap(
    int visibleLength ,
    const std::vector<int>& filteredLabels) {

    std::unordered_set<int> filteredSet(filteredLabels.begin() , filteredLabels.end());

    std::vector<int> validIds;
    validIds.reserve(visibleLength);

    for (int i = 0; i < visibleLength; ++i) {
        if (filteredSet.find(i) == filteredSet.end()) {
            validIds.push_back(i);
        }
    }
    std::unordered_map<int , int> reverseMap;

    for (size_t i = 0; i < validIds.size(); ++i) {
        reverseMap[static_cast<int>(i)] = validIds[i];
    }

    return reverseMap;
}

BaseDataset::BaseDataset(const DatasetConfig& config) : rng_(std::random_device{}()) {
    dataset_name = config.dataset_name;
    variable_idx = config.variable_idx;
    section_range = config.section_range;
    frame_range = config.frame_range;
    n_frame = config.n_frame;
    resolution = config.resolution;
    train_size = config.train_size;
    inst_norm = config.inst_norm;
    augment_type = config.augment_type;
    norm_type = config.norm_type;
    train_mode = config.train_mode;
    test_size = config.test_size;
    n_overlap = config.n_overlap;
    downsampling = config.downsampling;

    if (augment_type.find("downsample") != augment_type.end()) {
        max_downsample = augment_type["downsample"];
    }
    else if (augment_type.find("randsample") != augment_type.end()) {
        max_downsample = augment_type["randsample"];
    }
    else {
        max_downsample = 1;
    }

    enable_ds = true;
}

torch::Tensor BaseDataset::apply_augments(torch::Tensor data) {
    if (augment_type.find("downsample") != augment_type.end() && enable_ds) {
        data = apply_downsampling(data , augment_type["downsample"]);
    }
    else if (augment_type.find("randsample") != augment_type.end() && enable_ds) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1 , augment_type["randsample"]);
        int step = dis(gen);
        data = apply_downsampling(data , step);
    }
    return data;
}

// BUG HERE ______________________________________________________________________________________
torch::Tensor BaseDataset::apply_padding_or_crop(torch::Tensor data) {
    int cur_size = data.size(-1);

    if (train_size > cur_size) {
        int pad_size = train_size - cur_size;
        int pad_left = pad_size / 2;
        int pad_right = pad_size - pad_left;

        data = torch::nn::functional::pad(data.unsqueeze(0) ,
            torch::nn::functional::PadFuncOptions({ pad_left, pad_right, pad_left, pad_right })
            .mode(torch::kReflect)).squeeze(0);
    }
    else if (train_size < cur_size) {
        int start_h = std::uniform_int_distribution<>(0 , data.size(-2) - train_size)(rng_);
        int start_w = std::uniform_int_distribution<>(0 , data.size(-1) - train_size)(rng_);

        data = data.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(start_h, start_h + train_size),
            torch::indexing::Slice(start_w, start_w + train_size)
            });
    }

    return data;
}
// ______________________________________________________________________________________________

torch::Tensor BaseDataset::apply_inst_norm(torch::Tensor data , bool return_norm) {
    if (return_norm) {
        auto result = apply_inst_norm_with_params(data);
        return std::get<0>(result);
    }

    torch::Tensor offset , scale;

    if (norm_type == "mean_range") {
        offset = torch::mean(data);


        float data_max = data.max().item<float>();
        float data_min = data.min().item<float>();
        float scale_val = data_max - data_min;

        if (scale_val == 0.0f) {
            throw std::runtime_error("Scale is zero.");
        }

        scale = torch::tensor(scale_val);
        data = (data - offset) / scale;

        offset = offset.view({ 1, 1, 1 });
        scale = scale.view({ 1, 1, 1 });
    }
    else if (norm_type == "min_max") {
        auto dmin = data.min();
        auto dmax = data.max();
        offset = (dmax + dmin) / 2;
        scale = (dmax - dmin) / 2;

        if (scale.item<float>() == 0) {
            throw std::runtime_error("Scale is zero.");
        }

        data = (data - offset) / scale;

    }
    else if (norm_type == "mean_range_hw") {
        offset = torch::mean(data , { -2, -1 } , true);
        scale = torch::amax(data , { -2, -1 } , true) - torch::amin(data , { -2, -1 } , true);

        if (torch::any(scale == 0).item<bool>()) {
            throw std::runtime_error("Scale contains zero values.");
        }

        data = (data - offset) / scale;

    }
    else {
        throw std::runtime_error("Normalization type " + norm_type + " not implemented.");
    }

    return data;
}

std::tuple<torch::Tensor , torch::Tensor , torch::Tensor> BaseDataset::apply_inst_norm_with_params(torch::Tensor data) {
    torch::Tensor offset , scale;

    if (norm_type == "mean_range") {
        offset = torch::mean(data).view({ 1, 1, 1 });
        scale = data.max() - data.min();

        if (scale.item<float>() == 0) {
            throw std::runtime_error("Scale is zero.");
        }

        data = (data - offset) / scale;
        offset = offset.view({ 1, 1, 1 });
        scale = scale.view({ 1, 1, 1 });

    }
    else if (norm_type == "min_max") {
        auto dmin = data.min();
        auto dmax = data.max();
        offset = (dmax + dmin) / 2;
        scale = (dmax - dmin) / 2;

        if (scale.item<float>() == 0) {
            throw std::runtime_error("Scale is zero.");
        }

        data = (data - offset) / scale;
        offset = offset.view({ 1, 1, 1 });
        scale = scale.view({ 1, 1, 1 });

    }
    else if (norm_type == "mean_range_hw") {
        offset = torch::mean(data , { -2, -1 } , true);
        scale = torch::amax(data , { -2, -1 } , true) - torch::amin(data , { -2, -1 } , true);

        if (torch::any(scale == 0).item<bool>()) {
            throw std::runtime_error("Scale contains zero values.");
        }

        data = (data - offset) / scale;

    }
    else {
        throw std::runtime_error("Normalization type " + norm_type + " not implemented.");
    }

    return std::make_tuple(data , offset , scale);
}

torch::Tensor BaseDataset::apply_downsampling(torch::Tensor data , int step) {
    auto indices = torch::arange(0 , data.size(-1) , step , data.device());
    return data.index_select(-1 , indices);
}

ScientificDataset::ScientificDataset(const DatasetConfig& config)
    : BaseDataset(config)
{
    std::cout << "*************** Loading dataset ***************\n";

    torch::Tensor data;

    // Decide where to load data from
    if (config.memory_data.has_value()) {
        data = loadDatasetInMemory(
            config.memory_data.value() ,
            config.variable_idx ,
            config.section_range ,
            config.frame_range
        );
    }
    else if (config.binary_path.has_value()) {
        data = loadDatasetFromBinary(
            config.binary_path.value() ,
            config.variable_idx ,
            config.section_range ,
            config.frame_range
        );
    }
    else {
        throw std::runtime_error("No data source provided (memory_data or binary_path required).");
    }


    auto sizes = data.sizes();
    shape_org = std::vector<int64_t>(sizes.begin() , sizes.end());

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
        tail_frames = torch::flip(tail_frames , { 2 });
        data = torch::cat({ data, tail_frames } , 2);
    }

    // Normalization comment it out ???????? ======================================================================
    if (!inst_norm) {
        if (norm_type == "mean_range_hw") throw std::runtime_error("mean_range_hw requires inst_norm=true");

        torch::Tensor offset , scale;
        if (norm_type == "mean_range") {
            offset = torch::mean(data , { 1,2,3,4 } , true);
            scale = torch::amax(data , { 1,2,3,4 } , true) - torch::amin(data , { 1,2,3,4 } , true);
            data = (data - offset) / scale;
        }
        else if (norm_type == "min_max") {
            auto data_min = torch::amin(data , { 1,2,3,4 } , true);
            auto data_max = torch::amax(data , { 1,2,3,4 } , true);
            offset = (data_max + data_min) / 2;
            scale = (data_max - data_min) / 2;
            data = (data - offset) / scale;
        }
        else if (norm_type == "std") {
            offset = torch::mean(data , { 1,2,3,4 } , true);
            scale = torch::std(data , { 1,2,3,4 } , true);
            data = (data - offset) / scale;
        }

        var_offset = offset.to(torch::kFloat);
        var_scale = scale.to(torch::kFloat);
    }
            // =====================================================================================================

    data = data.to(torch::kFloat);

    if (!train_mode) {
        auto block_result = blockHW(data , test_size);
        data = std::get<0>(block_result);
        block_info = std::get<1>(block_result);
    }

    auto filter_result = data_filtering(data , delta_t);
    filtered_blocks = filter_result.first;
    filtered_labels = filter_result.second;

    data_input = data;
    auto final_sizes = data.sizes();
    shape = std::vector<int64_t>(final_sizes.begin() , final_sizes.end());

    visible_length = update_length();
    reverse_id_map = buildReverseIdMap(visible_length , filtered_labels);
}

// ------------------- Load from memory -------------------
torch::Tensor ScientificDataset::loadDatasetInMemory(
    const torch::Tensor& memory_data ,
    std::optional<int> variable_idx ,
    std::optional<std::pair<int , int>> section_range ,
    std::optional<std::pair<int , int>> frame_range)
{
    torch::Tensor data = memory_data;

    if (variable_idx.has_value()) {
        data = data.index({ variable_idx.value() });
        data = data.unsqueeze(0);
    }

    if (section_range.has_value()) {
        auto r = section_range.value();
        data = data.index({ torch::indexing::Slice(), torch::indexing::Slice(r.first, r.second) });
    }

    if (frame_range.has_value()) {
        auto r = frame_range.value();
        data = data.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(r.first, r.second) });
    }

    return data.to(torch::kFloat);
}

// NOTE: this function may be a bit buggy since it is hard to use when reading in binary files since they are just a blob of data it is better to load it in memory and then to use load from memory
// ------------------- Load from binary -------------------
torch::Tensor ScientificDataset::loadDatasetFromBinary(
    const std::string& bin_path ,
    std::optional<int> variable_idx ,
    std::optional<std::pair<int , int>> section_range ,
    std::optional<std::pair<int , int>> frame_range)
{
    std::ifstream file(bin_path , std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open binary file: " + bin_path);

    int64_t shape[5];
    file.read(reinterpret_cast<char*>(shape) , 5 * sizeof(int64_t));

    size_t num_elements = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()) , num_elements * sizeof(float));
    file.close();

    torch::Tensor data = torch::from_blob(
        buffer.data() ,
        { shape[0], shape[1], shape[2], shape[3], shape[4] } ,
        torch::kFloat32
    ).clone();

    if (variable_idx.has_value()) {
        data = data.index({ variable_idx.value() }).unsqueeze(0);
    }

    if (section_range.has_value()) {
        auto r = section_range.value();
        data = data.index({ torch::indexing::Slice(), torch::indexing::Slice(r.first, r.second) });
    }

    if (frame_range.has_value()) {
        auto r = frame_range.value();
        data = data.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(r.first, r.second) });
    }

    return data;
}

int64_t ScientificDataset::update_length() {
    dataset_length = shape[0] * shape[1] * t_samples;
    return dataset_length;
}

size_t ScientificDataset::size() const {
    return visible_length - filtered_blocks.size();
}

torch::Tensor ScientificDataset::original_data() const {
    torch::Tensor data = data_input;


    if (!train_mode) {
        data = deblockHW(data , std::get<0>(block_info) ,
            std::get<1>(block_info) , std::get<2>(block_info));
    }


    if (!inst_norm) {
        data = data * var_scale + var_offset;
    }

    return data;
}

torch::Tensor ScientificDataset::input_data() const {
    auto data = original_data();
    data = data.index({ torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(0, shape[2] - pad_T) });
    return data;
}

torch::Tensor ScientificDataset::recons_data(const torch::Tensor& recons_data) const {
    return recons_data.index({ torch::indexing::Slice(),
                             torch::indexing::Slice(),
                             torch::indexing::Slice(0, shape[2] - pad_T) });
}

torch::Tensor ScientificDataset::deblocking_hw(const torch::Tensor& data) const {
    return deblockHW(data , std::get<0>(block_info) ,
        std::get<1>(block_info) , std::get<2>(block_info));
}

std::unordered_map<std::string , torch::Tensor> ScientificDataset::post_processing(
    const torch::Tensor& data , int var_idx , bool is_training) {
    torch::Tensor processed_data = data; // changing to shallow copy note this data is const
    if (is_training) {
        processed_data = apply_augments(processed_data);
        processed_data = apply_padding_or_crop(processed_data);
    }
    torch::Tensor offset , scale;
    if (inst_norm) {
        auto norm_result = apply_inst_norm_with_params(processed_data);
        processed_data = std::get<0>(norm_result);
        offset = std::get<1>(norm_result);
        scale = std::get<2>(norm_result);
    }
    else {
        offset = var_offset.index({ var_idx }).view({ 1, 1, 1 });
        scale = var_scale.index({ var_idx }).view({ 1, 1, 1 });
    }
    std::unordered_map<std::string , torch::Tensor> data_dict;
    data_dict["input"] = processed_data.unsqueeze(0);
    data_dict["offset"] = offset.unsqueeze(0);
    data_dict["scale"] = scale.unsqueeze(0);
    return data_dict;
}

std::unordered_map<std::string , torch::Tensor> ScientificDataset::get_item(size_t idx) {
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
    int64_t end_t = start_t + n_frame;

    torch::Tensor data = data_input.index({
        static_cast<int64_t>(idx0),
        static_cast<int64_t>(idx1),
        torch::indexing::Slice(start_t, end_t)
        });


    data = data.unsqueeze(0);

    auto data_dict = post_processing(data , static_cast<int>(idx0) , train_mode);
    data = torch::Tensor();
    torch::Tensor index_tensor = torch::tensor({ idx0, idx1, start_t, end_t } , torch::kLong);
    data_dict["index"] = index_tensor;
    index_tensor = torch::Tensor();
    return data_dict;
}

std::tuple<int64_t , int64_t , std::vector<int64_t>> ScientificDataset::get_block_info() const {
    return block_info;
}

const torch::Tensor& ScientificDataset::get_data_input() const {
    return data_input;
}

const std::vector<std::pair<int , float>>& ScientificDataset::get_filtered_blocks() const {
    return filtered_blocks;
}

const int64_t& ScientificDataset::get_pad_T() const {
    return pad_T;
}

const std::vector<int64_t>& ScientificDataset::get_shape_info() const {
    return shape;
}

void ScientificDataset::clear() {
    data_input = torch::Tensor();
    var_offset = torch::Tensor();
    var_scale = torch::Tensor();

    filtered_blocks.clear();
    filtered_blocks.shrink_to_fit();

    filtered_labels.clear();
    filtered_labels.shrink_to_fit();

    reverse_id_map.clear();

    shape_org.clear();
    shape_org.shrink_to_fit();

    shape.clear();
    shape.shrink_to_fit();

}

ScientificDataset::~ScientificDataset() {
    clear();
}
