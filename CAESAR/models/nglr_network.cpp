#include "nglr_network.h"

namespace nglr {
torch::Tensor recons_features(const torch::Tensor &x) {
  torch::Tensor dt = torch::zeros_like(x);
  torch::Tensor dh = torch::zeros_like(x);
  torch::Tensor dw = torch::zeros_like(x);

  dt.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(),
       torch::indexing::Slice(1, torch::indexing::None),
       torch::indexing::Slice(), torch::indexing::Slice()},
      x.index({torch::indexing::Slice(), torch::indexing::Slice(),
               torch::indexing::Slice(1, torch::indexing::None),
               torch::indexing::Slice(), torch::indexing::Slice()}) -
          x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                   torch::indexing::Slice(0, -1), torch::indexing::Slice(),
                   torch::indexing::Slice()}));

  dh.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(),
       torch::indexing::Slice(),
       torch::indexing::Slice(1, torch::indexing::None),
       torch::indexing::Slice()},
      x.index({torch::indexing::Slice(), torch::indexing::Slice(),
               torch::indexing::Slice(),
               torch::indexing::Slice(1, torch::indexing::None),
               torch::indexing::Slice()}) -
          x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                   torch::indexing::Slice(), torch::indexing::Slice(0, -1),
                   torch::indexing::Slice()}));

  dw.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(1, torch::indexing::None)},
                x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(1, torch::indexing::None)}) -
                    x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                             torch::indexing::Slice(), torch::indexing::Slice(),
                             torch::indexing::Slice(0, -1)}));

  return torch::cat({x, dt, dh, dw, dt.abs(), dh.abs(), dw.abs()}, 1);
}

ResBlock3DImpl::ResBlock3DImpl(int64_t ch) {
  c1 = register_module(
      "c1", torch::nn::Conv3d(torch::nn::Conv3dOptions(ch, ch, 3).padding(1)));

  c2 = register_module(
      "c2", torch::nn::Conv3d(torch::nn::Conv3dOptions(ch, ch, 3).padding(1)));

  const int64_t groups = std::min<int64_t>(4, ch);

  n1 = register_module(
      "n1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, ch)));

  n2 = register_module(
      "n2", torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, ch)));
}

torch::Tensor ResBlock3DImpl::forward(const torch::Tensor &x) {
  torch::Tensor y = torch::gelu(n1(c1(x)));

  y = n2(c2(y));

  return torch::gelu(x + y);
}

CausalNeuralLorenzoNetImpl::CausalNeuralLorenzoNetImpl(int64_t hidden,
                                                       int64_t q_hidden,
                                                       int64_t blocks) {
  const int64_t groups = std::min<int64_t>(4, hidden);

  recons_in = register_module(
      "recons_in",
      torch::nn::Sequential(
          torch::nn::Conv3d(torch::nn::Conv3dOptions(7, hidden, 3).padding(1)),
          torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, hidden)),
          torch::nn::GELU()));

  recons_blocks = register_module("recons_blocks", torch::nn::Sequential());

  for (int64_t i = 0; i < blocks; ++i) {
    recons_blocks->push_back(ResBlock3D(hidden));
  }

  q_branch = register_module(
      "q_branch",
      torch::nn::Sequential(
          torch::nn::Conv3d(torch::nn::Conv3dOptions(8, q_hidden, 1)),
          torch::nn::GELU(),
          torch::nn::Conv3d(torch::nn::Conv3dOptions(q_hidden, q_hidden, 1)),
          torch::nn::GELU()));

  fusion = register_module(
      "fusion",
      torch::nn::Sequential(
          torch::nn::Conv3d(
              torch::nn::Conv3dOptions(hidden + q_hidden, hidden, 1)),
          torch::nn::GELU(),
          torch::nn::Conv3d(torch::nn::Conv3dOptions(hidden, hidden, 1)),
          torch::nn::GELU(),
          torch::nn::Conv3d(torch::nn::Conv3dOptions(hidden, 1, 1))));

  {
    torch::NoGradGuard no_grad;
    auto &final_conv = fusion->at<torch::nn::Conv3dImpl>(4);
    final_conv.weight.zero_();
    if (final_conv.bias.defined()) {
      final_conv.bias.zero_();
    }
  }
}

torch::Tensor
CausalNeuralLorenzoNetImpl::encode_recons(const torch::Tensor &r) {
  auto features = recons_in->forward(recons_features(r));
  return recons_blocks->is_empty() ? features
                                   : recons_blocks->forward(features);
}

torch::Tensor
CausalNeuralLorenzoNetImpl::forward_from_feature(const torch::Tensor &rf,
                                                 const torch::Tensor &qctx) {
  torch::Tensor qf = q_branch->forward(qctx);

  return fusion->forward(torch::cat({rf, qf}, 1));
}

torch::Tensor CausalNeuralLorenzoNetImpl::forward(const torch::Tensor &r,
                                                  const torch::Tensor &qctx) {
  return forward_from_feature(encode_recons(r), qctx);
}

} // namespace nglr
