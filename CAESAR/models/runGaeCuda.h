#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <memory>
#include <cstdlib>
#include <map>
#include <filesystem>
#include <fstream>

#include <torch/torch.h>
#include <torch/script.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <zstd.h> 

#include <nvcomp/lz4.h>
#include <nvcomp/cascaded.h>

// CUDA FOR NOW WILL ADD AMD HOPEFULLY IN THE FUTURE
//

class PCA {
public:
    PCA(int numComponents = -1, const std::string& device = "cuda");

    PCA& fit(const torch::Tensor& x);

    torch::Tensor components() const { return components_; }
    torch::Tensor mean() const { return mean_; }

private:
    int numComponents_;
    torch::Device device_;
    torch::Tensor components_;
    torch::Tensor mean_;
};

torch::Tensor block2Vector(const torch::Tensor& blockdata,
        std::pair<int, int> pathSize = {8,8});

torch::Tensor vector2Block(const torch::Tensor& vectors,
                           const std::vector<int64_t>& originalShape,
                           std::pair<int, int> patchSize);
