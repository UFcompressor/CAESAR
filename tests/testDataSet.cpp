#include "../CAESAR/dataset/dataset.h"
#include <iostream>

void roundTrip(){
 std::cout << "Starting to test dataset" << std::endl;

    // ==== 5D Input Tensor ====
    auto x5d = torch::rand({1, 1, 8, 32, 32});
    std::cout << "5D Center crop shape: " << centerCrop(x5d, {32, 32}).sizes() << std::endl;
    std::cout << "5D Downsampled shape: " << downSamplingData(x5d, {1.0, 1.0, 0.5, 1.0, 1.0}).sizes() << std::endl;
    std::cout << "Done with 5D tests" << std::endl;

    // ==== deblockHW test ====
    auto dblock_in = torch::rand({1, 4, 2, 4, 4}); // small dummy
    auto deblocked = deblockHW(dblock_in, 2, 2, {0,0,0,0});
    std::cout << "Deblocked shape: " << deblocked.sizes() << std::endl;
    std::cout << "Done deblockHW test" << std::endl;

    // ==== blockHW test ====
    std::cout << "Starting to test blockHW" << std::endl;
    auto x = torch::rand({1, 2, 3, 50, 70}); // [V, S, T, H, W]
    auto block_result = blockHW(x, {16, 16});
    auto blocked = std::get<0>(block_result);
    auto meta = std::get<1>(block_result);

    int64_t nH = std::get<0>(meta);
    int64_t nW = std::get<1>(meta);
    auto padding = std::get<2>(meta);

    std::cout << "Blocked tensor shape: " << blocked.sizes() << std::endl;
    std::cout << "nH: " << nH << ", nW: " << nW << std::endl;
    std::cout << "Padding: top=" << padding[0]
              << ", down=" << padding[1]
              << ", left=" << padding[2]
              << ", right=" << padding[3] << std::endl;
    std::cout << "Done testing blockHW" << std::endl;

    // ==== Round-trip test: blockHW -> deblockHW ====
    std::cout << "Starting blockHW -> deblockHW round-trip test" << std::endl;

    auto recovered = deblockHW(blocked, nH, nW, padding);
    std::cout << "Recovered tensor shape: " << recovered.sizes() << std::endl;

    // Compare original H, W vs recovered
    auto orig_sizes = x.sizes();
    auto rec_sizes = recovered.sizes();
    std::cout << "Original shape (before block): " << orig_sizes << std::endl;
    std::cout << "Recovered shape (after deblock): " << rec_sizes << std::endl;

    std::cout << "Done round-trip test" << std::endl;

}

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
    std::cout << "Starting deblock_hw test" << std::endl;

    // Example: (V=1, S_blk=4, T=1, h_block=2, w_block=2)
    auto data = torch::rand({1, 4, 1, 2, 2});
    int64_t n_h = 2, n_w = 2;
    std::vector<int64_t> padding = {0, 0, 0, 0};

    auto deblocked = deblockHW(data, n_h, n_w, padding);
    std::cout << "Deblocked shape: " << deblocked.sizes() << std::endl;

    std::cout << "Done deblock_hw test" << std::endl;

    std::cout << "Starting to test blockHW" << std::endl;

    // Create a small dummy tensor
    auto S = torch::rand({1, 2, 3, 50, 70}); // [V, S, T, H, W]

    // Call blockHW
    auto result = blockHW(S, {16, 16});
    auto blocked = std::get<0>(result);
    auto meta = std::get<1>(result);

    int64_t nH = std::get<0>(meta);
    int64_t nW = std::get<1>(meta);
    auto pad = std::get<2>(meta);

    std::cout << "Blocked tensor shape: " << blocked.sizes() << std::endl;
    std::cout << "nH: " << nH << ", nW: " << nW << std::endl;
    std::cout << "Padding: top=" << pad[0]
              << ", down=" << pad[1]
              << ", left=" << pad[2]
              << ", right=" << pad[3] << std::endl;

    std::cout << "Done testing blockHW" << std::endl;

    roundTrip();
    

    std::cout << "All dataset tests complete" << std::endl;

    return 0;
}

