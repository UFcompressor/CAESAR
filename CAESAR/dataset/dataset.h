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
#include <cmath>
#include <tuple>

torch::Tensor centerCrop(const torch::Tensor& x, std::pair<int64_t, int64_t> tShape);
torch::Tensor downSamplingData(const torch::Tensor& data, const std::vector<double>& zoomFactors);

torch::Tensor deblockHW(const torch::Tensor& x, int64_t nH, int64_t nW, 
        const std::vector<int64_t>& padding);
std::tuple<torch::Tensor, std::tuple<int64_t, int64_t, std::vector<int64_t>>>

blockHW(const torch::Tensor& data,
        std::pair<int64_t, int64_t> block_size = {256, 256});

// for debuging i guess not used
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
normalizeData(const torch::Tensor& data, const std::string& normType, torch::IntArrayRef axis);

std::pair<std::vector<std::pair<int, float>>, std::vector<int>>
dataFiltering(const torch::Tensor& data, int nFrame);
