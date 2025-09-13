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







