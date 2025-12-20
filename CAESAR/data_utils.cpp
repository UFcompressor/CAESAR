#include "data_utils.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

std::pair<torch::Tensor, PaddingInfo> to_5d_and_pad(
    const torch::Tensor& arr,
    int64_t H,
    int64_t W,
    bool force_padding
) {
    std::vector<int64_t> original_shape;
    for (int64_t i = 0; i < arr.dim(); ++i) {
        original_shape.push_back(arr.size(i));
    }

    int64_t num_dims = arr.dim();
    int64_t N = arr.numel();

    
    // Conditions: 3D or 4D input, meets dimension thresholds, and not forcing padding
    if (!force_padding && (num_dims == 3 || num_dims == 4)) {
    
        int64_t check_d, check_h, check_w;
        
        if (num_dims == 3) {
   
            check_d = arr.size(0);
            check_h = arr.size(1);
            check_w = arr.size(2);
        } else { // num_dims == 4
  
            check_d = arr.size(1);
            check_h = arr.size(2);
            check_w = arr.size(3);
        }

        // Check if dimensions meet thresholds: shape[2] >= 8 && shape[3] >= 128 && shape[4] >= 128
        // For 3D: arr.size(0) >= 8 && arr.size(1) >= 128 && arr.size(2) >= 128
        // For 4D: arr.size(1) >= 8 && arr.size(2) >= 128 && arr.size(3) >= 128
        if (check_d >= 8 && check_h >= 128 && check_w >= 128) {
            torch::Tensor result_5d;
            
            if (num_dims == 3) {
                result_5d = arr.unsqueeze(0).unsqueeze(0);
            } else {
                result_5d = arr.unsqueeze(0);
            }

            PaddingInfo info;
            info.original_shape = original_shape;
            info.original_length = N;
            info.padded_shape = {result_5d.size(0), result_5d.size(1), result_5d.size(2), 
                                 result_5d.size(3), result_5d.size(4)};
            info.H = check_h;
            info.W = check_w;
            info.was_padded = false; 

            std::cout << "Fast path: Converted tensor from shape [";
            for (size_t i = 0; i < original_shape.size(); ++i) {
                std::cout << original_shape[i];
                if (i < original_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "] to 5D shape [";
            for (int i = 0; i < 5; ++i) {
                std::cout << result_5d.size(i);
                if (i < 4) std::cout << ", ";
            }
            std::cout << "] (no padding - dimensions meet threshold)\n";

            return {result_5d, info};
        }
    }

    std::cout << "Padding path: Input is " << num_dims << "D";
    if (force_padding) {
        std::cout << " (forced padding)";
    } else if (num_dims <= 2) {
        std::cout << " (too few dimensions)";
    } else if (num_dims >= 5) {
        std::cout << " (already 5D or greater)";
    } else {
        std::cout << " (dimensions below threshold)";
    }
    std::cout << "\n";

    int64_t patch_area = H * W;
    int64_t D = (N + patch_area - 1) / patch_area;

    if (N % patch_area == 0) {
      
        torch::Tensor padded_5d = arr.view({1, 1, D, H, W});

        PaddingInfo info;
        info.original_shape = original_shape;
        info.original_length = N;
        info.padded_shape = {1, 1, D, H, W};
        info.H = H;
        info.W = W;
        info.was_padded = true; 
        
        return {padded_5d, info};
    }
    

    int64_t padded_length = D * H * W;
    torch::Tensor padded = torch::zeros({padded_length}, arr.options());

    torch::Tensor flat = arr.flatten();
    padded.index_put_({torch::indexing::Slice(0, N)}, flat);

    torch::Tensor padded_5d = padded.reshape({1, 1, D, H, W});

    PaddingInfo info;
    info.original_shape = original_shape;
    info.original_length = N;
    info.padded_shape = {1, 1, D, H, W};
    info.H = H;
    info.W = W;
    info.was_padded = true;

    std::cout << "Converted tensor from shape [";
    for (size_t i = 0; i < original_shape.size(); ++i) {
        std::cout << original_shape[i];
        if (i < original_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "] to 5D padded shape [1, 1, " << D << ", " << H << ", " << W << "]\n";
    std::cout << "  Original elements: " << N << ", Padded elements: " << padded_length << "\n";

    return {padded_5d, info};
}

torch::Tensor restore_from_5d(
    const torch::Tensor& padded_5d,
    const PaddingInfo& info
) {
    if (!info.was_padded) {
        torch::Tensor restored = padded_5d.reshape(torch::IntArrayRef(info.original_shape));
        
        std::cout << "Fast restore: Reshaped from [";
        for (int i = 0; i < padded_5d.dim(); ++i) {
            std::cout << padded_5d.size(i);
            if (i < padded_5d.dim() - 1) std::cout << ", ";
        }
        std::cout << "] back to [";
        for (size_t i = 0; i < info.original_shape.size(); ++i) {
            std::cout << info.original_shape[i];
            if (i < info.original_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "] (no unpadding needed)\n";
        
        return restored;
    }

    torch::Tensor flat = padded_5d.flatten();
    torch::Tensor trimmed = flat.index({torch::indexing::Slice(0, info.original_length)});
    torch::Tensor restored = trimmed.reshape(torch::IntArrayRef(info.original_shape));
    
    std::cout << "Slow restore: Unpadded and reshaped back to original shape\n";
    
    return restored;
}
