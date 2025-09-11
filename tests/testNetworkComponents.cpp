#include "../CAESAR/models/networkComponents.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <list>
#include <deque>
#include <array>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>

void test_pmf_to_quantized_cdf_tensor() {
    std::cout << "Starting to test pmf_to_quantized_cdf_tensor..." << std::endl;

    // Example PMF tensor
    torch::Tensor pmf = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}, torch::kFloat32);

    try {
        torch::Tensor cdf = pmfToQuantizedCDFTensor(pmf, 16);

        std::cout << "Input PMF: " << pmf << std::endl;
        std::cout << "Output CDF: " << cdf << std::endl;

        // Optional simple check: last element should be 2^16
        if (cdf[-1].item<int>() == (1 << 16)) {
            std::cout << "Test passed!" << std::endl;
        } else {
            std::cout << "Test failed: last element != 2^16" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
    }

    std::cout << "Done testing pmf_to_quantized_cdf_tensor" << std::endl;
}

void testResidual() {
    std::cout << "Starting testResidual..." << std::endl;

    // Define a simple linear layer
    auto linear = torch::nn::Linear(5, 5);

    // Wrap inside Residual using AnyModule (use {} to avoid vexing-parse!)
    Residual residual{torch::nn::AnyModule(linear)};

    // Input tensor
    auto x = torch::randn({2, 5});

    // Forward pass
    auto y = residual->forward(x);

    std::cout << "Input: " << x << std::endl;
    std::cout << "Output: " << y << std::endl;

    std::cout << "Done testing testResidual." << std::endl;
}


int main() {
    std::cout<<"Starting to test Network Components"<<std::endl;
    test_pmf_to_quantized_cdf_tensor();
    testResidual();
    testResidual();
    std::cout<<"Done testing Network Compents"<<std::endl;

}
