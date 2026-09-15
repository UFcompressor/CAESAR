#include "data_utils.h"

std::pair<torch::Tensor, PaddingInfo> to_5d(torch::Tensor &arr) {
  std::vector<int64_t> original_shape;
  for (int64_t i = 0; i < arr.dim(); ++i) {
    original_shape.push_back(arr.size(i));
  }
  const int64_t num_dims = arr.dim();
  const int64_t N = arr.numel();

  if (num_dims < 2 || num_dims > 5) {
    throw std::invalid_argument("to_5d: expected a 2D-5D tensor, got " +
                                std::to_string(num_dims) + " dims");
  }

  std::vector<int64_t> shape5d(5, 1);
  for (int64_t i = 0; i < num_dims; ++i) {
    shape5d[5 - num_dims + i] = arr.size(i);
  }

  torch::Tensor result_5d = arr.reshape(shape5d);
  arr = torch::Tensor();

  PaddingInfo info;
  info.original_shape = original_shape;
  info.original_length = N;
  info.padded_shape = shape5d;
  info.H = 0;
  info.W = 0;
  info.was_padded = false;

  return {result_5d, info};
}

torch::Tensor restore_from_5d(torch::Tensor &padded_5d,
                              const PaddingInfo &info) {
  if (padded_5d.numel() != info.original_length) {
    throw std::runtime_error(
        "restore_from_5d: element count mismatch -- expected " +
        std::to_string(info.original_length) + " but decompressed tensor has " +
        std::to_string(padded_5d.numel()) +
        ". This means the compress/decompress path lost or duplicated data "
        "upstream of restore_from_5d, not a padding bug.");
  }

  torch::Tensor restored =
      padded_5d.reshape(torch::IntArrayRef(info.original_shape)).contiguous();
  padded_5d = torch::Tensor();
  return restored;
}
