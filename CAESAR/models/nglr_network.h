#pragma once

#include <torch/torch.h>

namespace nglr {

torch::Tensor recons_features(const torch::Tensor &x);

struct ResBlock3DImpl : torch::nn::Module {
  torch::nn::Conv3d c1{nullptr}, c2{nullptr};
  torch::nn::GroupNorm n1{nullptr}, n2{nullptr};
  explicit ResBlock3DImpl(int64_t channels);
  torch::Tensor forward(const torch::Tensor &x);
};
TORCH_MODULE(ResBlock3D);

struct CausalNeuralLorenzoNetImpl : torch::nn::Module {
  torch::nn::Sequential recons_in{nullptr}, recons_blocks{nullptr};
  torch::nn::Sequential q_branch{nullptr}, fusion{nullptr};
  CausalNeuralLorenzoNetImpl(int64_t hidden = 32, int64_t q_hidden = 16,
                             int64_t blocks = 4);
  torch::Tensor encode_recons(const torch::Tensor &r);
  torch::Tensor forward_from_feature(const torch::Tensor &features,
                                     const torch::Tensor &context);
  torch::Tensor forward(const torch::Tensor &r, const torch::Tensor &context);
};
TORCH_MODULE(CausalNeuralLorenzoNet);

} // namespace nglr
