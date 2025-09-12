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
        // [N, C, H, W]
        return x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(start_h, end_h),
                        torch::indexing::Slice(start_w, end_w)});
    } else if (dim == 5) {
        // [N, C, D, H, W]
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

