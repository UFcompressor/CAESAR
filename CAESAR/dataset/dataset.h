#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>                  // math (like Python's math)
#include <algorithm>              // bisect equivalents (lower_bound, upper_bound)
#include <thread>                 // threading
#include <filesystem>             // os + glob equivalent
#include <vector>                 // numpy-like arrays
#include <string>                 // string handling
#include <memory>                 // smart pointers
#include <iostream>               // printing/debugging


torch::Tensor centerCrop(const torch::Tensor& x, std::pair<int64_t, int64_t> tShape);
torch::Tensor downSamplingData(const torch::Tensor& data, const std::vector<double>& zoomFactors);

