#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>     
#include <algorithm>   
#include <thread>     
#include <filesystem>
#include <vector>   
#include <string>  
#include <memory> 
#include <iostream>     


torch::Tensor centerCrop(const torch::Tensor& x, std::pair<int64_t, int64_t> tShape);
torch::Tensor downSamplingData(const torch::Tensor& data, const std::vector<double>& zoomFactors);

