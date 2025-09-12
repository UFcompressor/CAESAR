#include "../CAESAR/dataset/dataset.h"
#include <iostream>

int main() {
    std::cout << "Starting to test dataset" << std::endl;

    // Create a 4D tensor [N, C, H, W]
    auto x = torch::rand({1, 1, 64, 64});

    // ---- Test center_crop ----
    std::cout << "Starting to test center_crop" << std::endl;
    auto cropped = centerCrop(x, {32, 32});
    std::cout << "Center crop shape: " << cropped.sizes() << std::endl;
    std::cout << "Done testing center_crop" << std::endl;

    // ---- Test downsampling_data ----
    std::cout << "Starting to test downsampling_data" << std::endl;
    auto down = downSamplingData(x, {1.0, 1.0, 0.5, 0.5});
    std::cout << "Downsampled shape: " << down.sizes() << std::endl;
    std::cout << "Done testing downsampling_data" << std::endl;

// ---- Test with 5D tensor ----
std::cout << "Starting 5D tensor tests" << std::endl;
auto x5d = torch::rand({1, 1, 8, 64, 64});  // [N, C, D, H, W]

// Center crop in 5D
auto cropped5d = centerCrop(x5d, {32, 32});
std::cout << "5D Center crop shape: " << cropped5d.sizes() << std::endl;

// Downsampling in 5D
auto down5d = downSamplingData(x5d, {1.0, 1.0, 0.5, 0.5, 0.5});
std::cout << "5D Downsampled shape: " << down5d.sizes() << std::endl;

std::cout << "Done with 5D tests" << std::endl;



    std::cout << "All dataset tests complete" << std::endl;
    return 0;
}

