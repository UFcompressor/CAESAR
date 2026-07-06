#include "data_utils.h"

std::pair<torch::Tensor, PaddingInfo> to_5d_and_pad(torch::Tensor& arr,
                                                    int64_t H, int64_t W,
                                                    bool force_padding) {
  std::vector<int64_t> original_shape;
  for (int64_t i = 0; i < arr.dim(); ++i) {
    original_shape.push_back(arr.size(i));
  }
  int64_t num_dims = arr.dim();
  int64_t N = arr.numel();

  if (!force_padding && (num_dims >= 3 && num_dims <= 5)) {
    int64_t check_d, check_h, check_w;
    if (num_dims == 3) {
      check_d = arr.size(0);
      check_h = arr.size(1);
      check_w = arr.size(2);
    } else if (num_dims == 4) {
      check_d = arr.size(1);
      check_h = arr.size(2);
      check_w = arr.size(3);
    } else {  // num_dims == 5
      check_d = arr.size(2);
      check_h = arr.size(3);
      check_w = arr.size(4);
    }

    if (check_d >= 8 && check_h >= 2 && check_w >= 2) {
      torch::Tensor result_5d;
      if (num_dims == 3) {
        result_5d = arr.reshape({1, 1, arr.size(0), arr.size(1), arr.size(2)});
      } else if (num_dims == 4) {
        result_5d = arr.reshape(
            {1, arr.size(0), arr.size(1), arr.size(2), arr.size(3)});
      } else {  // num_dims == 5
        result_5d = arr.reshape(
            {arr.size(0), arr.size(1), arr.size(2), arr.size(3), arr.size(4)});
      }
      arr = torch::Tensor();
      PaddingInfo info;
      info.original_shape = original_shape;
      info.original_length = N;
      info.padded_shape = {result_5d.size(0), result_5d.size(1),
                           result_5d.size(2), result_5d.size(3),
                           result_5d.size(4)};
      info.H = check_h;
      info.W = check_w;
      info.was_padded = false;
      return {result_5d, info};
    }
  }


  int64_t patch_area = H * W;
  int64_t D = (N + patch_area - 1) / patch_area;
  if (N % patch_area == 0) {
    torch::Tensor padded_5d = arr.reshape({1, 1, D, H, W});
    arr = torch::Tensor();
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
  padded.index_put_({torch::indexing::Slice(0, N)}, arr.flatten());
  arr = torch::Tensor();
  torch::Tensor padded_5d = padded.reshape({1, 1, D, H, W});
  padded = torch::Tensor();
  PaddingInfo info;
  info.original_shape = original_shape;
  info.original_length = N;
  info.padded_shape = {1, 1, D, H, W};
  info.H = H;
  info.W = W;
  info.was_padded = true;
  return {padded_5d, info};
}
torch::Tensor restore_from_5d(torch::Tensor& padded_5d,
                              const PaddingInfo& info) {
  torch::Device device = select_model_device();
  torch::Tensor flat = padded_5d.flatten().contiguous().to(device);

  padded_5d = torch::Tensor();

  torch::Tensor trimmed =
      flat.index({torch::indexing::Slice(0, info.original_length)});

  torch::Tensor restored =
      trimmed.reshape(torch::IntArrayRef(info.original_shape));

  restored = restored.contiguous();

  flat = torch::Tensor();
  trimmed = torch::Tensor();

  return restored;
}

