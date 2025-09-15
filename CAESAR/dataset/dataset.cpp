#include "dataset.h"

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
blockHW(const torch::Tensor& data,
        std::pair<int64_t, int64_t> block_size){

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

    for (int i =0; i < visibleLength; ++i){
        if (filteredSet.find(i) == filteredSet.end()){
            validIds.push_back(i);
        }
    }
    std::unordered_map<int, int> reverseMap;

    for(size_t i = 0; i < validIds.size(); ++i){
        reverseMap[static_cast<int>(i)]  = validIds[i];
    }

    return reverseMap;
}


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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, augment_type["randsample"]);
        int step = dis(gen);
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

// Note: this is not defined in the  code so i think there was a bug becasue it is used my impl of it
// 
torch::Tensor BaseDataset::apply_downsampling(torch::Tensor data, int step) {
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
    return default_value;
}

template int BaseDataset::get_arg<int>(const std::unordered_map<std::string, torch::Tensor>&, const std::string&, const int&);
template float BaseDataset::get_arg<float>(const std::unordered_map<std::string, torch::Tensor>&, const std::string&, const float&);
template bool BaseDataset::get_arg<bool>(const std::unordered_map<std::string, torch::Tensor>&, const std::string&, const bool&);
