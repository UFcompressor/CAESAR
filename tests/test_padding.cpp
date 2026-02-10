#include <iostream>
#include "../CAESAR/data_utils.h"
#include <cassert>

void test_1d_padding() {
    std::cout << "\n=== Test 1D Padding ===" << std::endl;
    torch::Tensor data = torch::randn({1000});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);
    
    std::cout << "Original shape: [1000]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    assert(info.was_padded == true);
    assert(padded.dim() == 5);
    
    // Restore and verify
    torch::Tensor restored = restore_from_5d(padded, info);
    std::cout << "Restored shape: " << restored.sizes() << std::endl;
    assert(restored.numel() == 1000);
}

void test_2d_padding() {
    std::cout << "\n=== Test 2D Padding ===" << std::endl;
    torch::Tensor data = torch::randn({50, 50});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);
    
    std::cout << "Original shape: [50, 50]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    assert(info.was_padded == true);
    assert(padded.dim() == 5);
    
    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.sizes()[0] == 50 && restored.sizes()[1] == 50);
}

void test_3d_small_padding() {
    std::cout << "\n=== Test 3D Small (needs padding) ===" << std::endl;
    torch::Tensor data = torch::randn({5, 100, 100});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);
    
    std::cout << "Original shape: [5, 100, 100]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    assert(info.was_padded == true);
    assert(padded.dim() == 5);
    
    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.sizes()[0] == 5);
}

void test_3d_large_no_padding() {
    std::cout << "\n=== Test 3D Large (no padding needed) ===" << std::endl;
    torch::Tensor data = torch::randn({10, 256, 256});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);
    
    std::cout << "Original shape: [10, 256, 256]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    assert(info.was_padded == false);
    assert(padded.dim() == 5);
    assert(padded.sizes()[2] == 10);
    assert(padded.sizes()[3] == 256);
    assert(padded.sizes()[4] == 256);
    
    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.sizes()[0] == 10);
}

void test_4d_small_padding() {
    std::cout << "\n=== Test 4D Small (needs padding) ===" << std::endl;
    torch::Tensor data = torch::randn({2, 5, 100, 100});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);
    
    std::cout << "Original shape: [2, 5, 100, 100]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    assert(info.was_padded == true);
    assert(padded.dim() == 5);
    
    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.sizes()[0] == 2 && restored.sizes()[1] == 5);
}

void test_4d_large_no_padding() {
    std::cout << "\n=== Test 4D Large (no padding needed) ===" << std::endl;
    torch::Tensor data = torch::randn({3, 10, 256, 256});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);

    std::cout << "Original shape: [3, 10, 256, 256]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;

    assert(info.was_padded == false);
    assert(padded.dim() == 5);
    assert(padded.sizes()[1] == 3);  
    assert(padded.sizes()[2] == 10);

    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.sizes()[0] == 3 && restored.sizes()[1] == 10);
}

void test_5d_padding() {
    std::cout << "\n=== Test 5D (always needs padding) ===" << std::endl;
    torch::Tensor data = torch::randn({2, 3, 10, 256, 256});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);
    
    std::cout << "Original shape: [2, 3, 10, 256, 256]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    assert(info.was_padded == true);
    assert(padded.dim() == 5);
    
    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.dim() == 5);
}

void test_force_padding() {
    std::cout << "\n=== Test Force Padding ===" << std::endl;
    torch::Tensor data = torch::randn({10, 256, 256});
    auto [padded, info] = to_5d_and_pad(data, 128, 128, true);
    
    std::cout << "Original shape: [10, 256, 256]" << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    assert(info.was_padded == true);
    
    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.sizes()[0] == 10);
}

void test_exact_fit_no_extra_padding() {
    std::cout << "\n=== Test Exact Fit (256*256 elements) ===" << std::endl;
    torch::Tensor data = torch::randn({256 * 256});
    auto [padded, info] = to_5d_and_pad(data, 256, 256, false);
    
    std::cout << "Original elements: " << data.numel() << std::endl;
    std::cout << "Padded shape: " << padded.sizes() << std::endl;
    std::cout << "Padded elements: " << padded.numel() << std::endl;
    std::cout << "Was padded: " << info.was_padded << std::endl;
    
    // Should not need extra zero padding since it fits exactly
    assert(padded.sizes()[2] == 1);
    assert(padded.sizes()[3] == 256);
    assert(padded.sizes()[4] == 256);
    
    torch::Tensor restored = restore_from_5d(padded, info);
    assert(restored.numel() == 256 * 256);
}

int main() {
    try {
        std::cout << "Starting data_utils tests..." << std::endl;
        
        test_1d_padding();
        test_2d_padding();
        test_3d_small_padding();
        test_3d_large_no_padding();
        test_4d_small_padding();
        test_4d_large_no_padding();
        test_5d_padding();
        test_force_padding();
        test_exact_fit_no_extra_padding();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
