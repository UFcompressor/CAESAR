#pragma once

#include <torch/torch.h>
#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include "utils.h"
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cassert>

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf, int precision = 16);

torch::Tensor pmfToQuantizedCDFTensor(const torch::Tensor& pmf, int precision = 16);

struct ResidualImpl : torch::nn::Module {
    torch::nn::AnyModule fn;

    ResidualImpl(torch::nn::AnyModule fn_module) {
        fn = std::move(fn_module);
        register_module("fn", fn.ptr());
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = fn.forward(x);
        return out + x;
    }
};

TORCH_MODULE(Residual);

