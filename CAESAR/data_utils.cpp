#include "data_utils.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

std::pair<torch::Tensor , PaddingInfo> to_5d_and_pad(
    const torch::Tensor& arr ,
    int64_t H ,
    int64_t W
) {
    std::vector<int64_t> original_shape;
    for (int64_t i = 0; i < arr.dim(); ++i) {
        original_shape.push_back(arr.size(i));
    }

    int64_t N = arr.numel();
    int64_t patch_area = H * W;

    int64_t D = (N + patch_area - 1) / patch_area; 

    if (N % patch_area == 0) {
        torch::Tensor padded_5d = arr.view({ 1, 1, D, H, W });

        PaddingInfo info;
        info.original_shape = original_shape;
        info.original_length = N;
        info.padded_shape = { 1, 1, D, H, W };
        info.H = H;
        info.W = W;
        
        return { padded_5d, info };
    }
    
    int64_t padded_length = D * H * W;
    torch::Tensor padded = torch::zeros({ padded_length } , arr.options());

    torch::Tensor flat = arr.flatten();
    padded.index_put_({ torch::indexing::Slice(0, N) } , flat);

    torch::Tensor padded_5d = padded.reshape({ 1, 1, D, H, W });

    // Create metadata
    PaddingInfo info;
    info.original_shape = original_shape;
    info.original_length = N;
    info.padded_shape = { 1, 1, D, H, W };
    info.H = H;
    info.W = W;

    std::cout << "Converted tensor from shape [";
    for (size_t i = 0; i < original_shape.size(); ++i) {
        std::cout << original_shape[i];
        if (i < original_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "] to 5D padded shape [1, 1, " << D << ", " << H << ", " << W << "]\n";
    std::cout << "  Original elements: " << N << ", Padded elements: " << padded_length << "\n";

    return { padded_5d, info };
}

torch::Tensor restore_from_5d(
    const torch::Tensor& padded_5d ,
    const PaddingInfo& info
) {
    torch::Tensor flat = padded_5d.flatten();
    torch::Tensor trimmed = flat.index({ torch::indexing::Slice(0, info.original_length) });

    torch::Tensor restored = trimmed.reshape(torch::IntArrayRef(info.original_shape));
    return restored;
}
