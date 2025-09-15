#include "../CAESAR/dataset/dataset.h"
#include <iostream>

void test_scientific_dataset() {
    try {
        // Create test arguments
        std::unordered_map<std::string, torch::Tensor> args;
        
        // Required arguments
        args["data_path"] = torch::zeros({1}); // Dummy, not used in memory version
        args["n_frame"] = torch::tensor(16);
        args["train"] = torch::tensor(true);
        
        // Optional arguments
        args["variable_idx"] = torch::tensor(0);
        args["train_size"] = torch::tensor(32);
        args["inst_norm"] = torch::tensor(true);
        args["norm_type"] = torch::zeros({1}); // Will use default "mean_range"
        args["n_overlap"] = torch::tensor(0);
        
        std::cout << "Creating ScientificDataset...\n";
        ScientificDataset dataset(args);
        
        std::cout << "Dataset size: " << dataset.size() << "\n";
        
        // Test getting an item
        std::cout << "Testing get_item(0)...\n";
        auto data_item = dataset.get_item(0);
        
        std::cout << "Input shape: " << data_item["input"].sizes() << "\n";
        std::cout << "Offset shape: " << data_item["offset"].sizes() << "\n";
        std::cout << "Scale shape: " << data_item["scale"].sizes() << "\n";
        std::cout << "Index: " << data_item["index"] << "\n";
        
        // Test original data access
        std::cout << "Testing original_data()...\n";
        auto orig_data = dataset.original_data();
        std::cout << "Original data shape: " << orig_data.sizes() << "\n";
        
        // Test input data access
        std::cout << "Testing input_data()...\n";
        auto input_data = dataset.input_data();
        std::cout << "Input data shape: " << input_data.sizes() << "\n";
        
        std::cout << "All tests passed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << "\n";
    }
}

void test_constructor() {
    std::cout << "=== Testing BaseDataset::BaseDataset() ===" << std::endl;

    std::unordered_map<std::string, torch::Tensor> args;
    args["data_path"] = torch::tensor(0); // Dummy value - we need this for constructor
    args["n_frame"] = torch::tensor(16);
    args["train_size"] = torch::tensor(128);
    args["inst_norm"] = torch::tensor(true);
    args["train"] = torch::tensor(true);
    args["n_overlap"] = torch::tensor(4);

    try {
        BaseDataset dataset(args);
        std::cout << " Constructor test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Constructor test failed: " << e.what() << std::endl;
    }
}

void test_apply_inst_norm() {
    std::cout << "=== Testing BaseDataset::apply_inst_norm() ===" << std::endl;

    std::unordered_map<std::string, torch::Tensor> args;
    args["data_path"] = torch::tensor(0); // Dummy value
    args["n_frame"] = torch::tensor(16);

    try {
        BaseDataset dataset(args);

        // Create test data with known statistics
        torch::Tensor test_data = torch::randn({3, 64, 64}) * 10 + 5;

        auto normalized = dataset.apply_inst_norm(test_data);

        std::cout << "Original - Mean: " << torch::mean(test_data).item<float>()
                  << ", Min: " << test_data.min().item<float>()
                  << ", Max: " << test_data.max().item<float>() << std::endl;

        std::cout << "Normalized - Mean: " << torch::mean(normalized).item<float>()
                  << ", Min: " << normalized.min().item<float>()
                  << ", Max: " << normalized.max().item<float>() << std::endl;

        std::cout << " apply_inst_norm test completed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << " apply_inst_norm test failed: " << e.what() << std::endl;
    }
}

void test_apply_padding_or_crop() {
    std::cout << "=== Testing BaseDataset::apply_padding_or_crop() ===" << std::endl;

    std::unordered_map<std::string, torch::Tensor> args;
    args["data_path"] = torch::tensor(0); // Dummy value
    args["n_frame"] = torch::tensor(16);
    args["train_size"] = torch::tensor(128);

    try {
        BaseDataset dataset(args);

        // Test padding (input smaller than train_size)
        torch::Tensor small_data = torch::randn({3, 64, 64});
        auto padded = dataset.apply_padding_or_crop(small_data);

        std::cout << "Padding test - Original: " << small_data.sizes()
                  << " -> Padded: " << padded.sizes() << std::endl;

        assert(padded.size(-1) == 128 && padded.size(-2) == 128);

        // Test cropping (input larger than train_size)
        torch::Tensor large_data = torch::randn({3, 256, 256});
        auto cropped = dataset.apply_padding_or_crop(large_data);

        std::cout << "Cropping test - Original: " << large_data.sizes()
                  << " -> Cropped: " << cropped.sizes() << std::endl;

        assert(cropped.size(-1) == 128 && cropped.size(-2) == 128);

        std::cout << " apply_padding_or_crop test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << " apply_padding_or_crop test failed: " << e.what() << std::endl;
    }
}

void test_apply_augments_downsample() {
    std::cout << "=== Testing BaseDataset::apply_augments() with downsample ===" << std::endl;

    std::unordered_map<std::string, torch::Tensor> args;
    args["data_path"] = torch::tensor(0); // Dummy value
    args["n_frame"] = torch::tensor(16);
    args["augment_downsample"] = torch::tensor(2); // Fixed: use augment_downsample key

    try {
        BaseDataset dataset(args);

        // Create test data [C, H, W] - the apply_downsampling operates on last dimension
        torch::Tensor test_data = torch::randn({3, 64, 32});

        auto augmented = dataset.apply_augments(test_data);

        std::cout << "Downsample test - Original last dim: " << test_data.size(-1)
                  << " -> Augmented last dim: " << augmented.size(-1) << std::endl;

        // Should be roughly half the size (32 -> 16)
        assert(augmented.size(-1) == 16);

        std::cout << " apply_augments downsample test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << " apply_augments downsample test failed: " << e.what() << std::endl;
    }
}

void test_apply_augments_randsample() {
    std::cout << "=== Testing BaseDataset::apply_augments() with randsample ===" << std::endl;

    std::unordered_map<std::string, torch::Tensor> args;
    args["data_path"] = torch::tensor(0); // Dummy value
    args["n_frame"] = torch::tensor(16);
    args["augment_randsample"] = torch::tensor(3); // Fixed: use augment_randsample key

    try {
        BaseDataset dataset(args);

        torch::Tensor test_data = torch::randn({3, 64, 30});

        auto augmented = dataset.apply_augments(test_data);

        std::cout << "Randsample test - Original last dim: " << test_data.size(-1)
                  << " -> Augmented last dim: " << augmented.size(-1) << std::endl;

        // Should be smaller due to random downsampling (step 1-3)
        assert(augmented.size(-1) <= test_data.size(-1));

        std::cout << " apply_augments randsample test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << " apply_augments randsample test failed: " << e.what() << std::endl;
    }
}

void test_apply_downsampling_directly() {
    std::cout << "=== Testing BaseDataset::apply_downsampling() directly ===" << std::endl;

    std::unordered_map<std::string, torch::Tensor> args;
    args["data_path"] = torch::tensor(0); // Dummy value
    args["n_frame"] = torch::tensor(16);

    try {
        BaseDataset dataset(args);

        torch::Tensor test_data = torch::randn({3, 64, 20});

        // Test direct downsampling with step=2
        // Note: apply_downsampling is private, so we test through apply_augments
        args["augment_downsample"] = torch::tensor(2); // Fixed: use correct key
        BaseDataset ds_dataset(args);

        auto downsampled = ds_dataset.apply_augments(test_data);

        std::cout << "Direct downsample test - Original: " << test_data.size(-1)
                  << " -> Downsampled: " << downsampled.size(-1) << std::endl;

        assert(downsampled.size(-1) == 10); // 20 -> 10 with step=2

        std::cout << " apply_downsampling direct test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cout << " apply_downsampling direct test failed: " << e.what() << std::endl;
    }
}

void test_normalization_with_params() {
    std::cout << "=== Testing BaseDataset::apply_inst_norm_with_params() ===" << std::endl;

    std::unordered_map<std::string, torch::Tensor> args;
    args["data_path"] = torch::tensor(0); // Dummy value
    args["n_frame"] = torch::tensor(16);

    try {
        BaseDataset dataset(args);

        torch::Tensor test_data = torch::randn({3, 64, 64}) * 5 + 10;

        auto [normalized, offset, scale] = dataset.apply_inst_norm_with_params(test_data);

        std::cout << "Normalization params - Offset: " << offset.item<float>()
                  << ", Scale: " << scale.item<float>() << std::endl;
        std::cout << "Normalized mean: " << torch::mean(normalized).item<float>() << std::endl;

        // Verify reconstruction
        auto reconstructed = normalized * scale + offset;
        auto diff = torch::mean(torch::abs(reconstructed - test_data));

        std::cout << "Reconstruction error: " << diff.item<float>() << std::endl;

        if (diff.item<float>() < 1e-5) {
            std::cout << " apply_inst_norm_with_params test passed" << std::endl;
        } else {
            std::cout << " apply_inst_norm_with_params test failed - reconstruction error too large" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << " apply_inst_norm_with_params test failed: " << e.what() << std::endl;
    }
}



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

void filtered_labels(){
    int visible_length = 6;
    std::cout<<"Starting to test build`ReverseIdMap"<<std::endl;
    std::vector<int> filtered_labels = {1, 3, 5};

    auto reverse_map = buildReverseIdMap(visible_length, filtered_labels);

    for (const auto& [key, val] : reverse_map) {
        std::cout << key << " -> " << val << "\n";
    }
    std::cout<<"Done testing buildReverseIdMap"<<std::endl;

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
    

 std::cout << "Running BaseDataset tests..." << std::endl;
    std::cout << "============================================" << std::endl;
    
    test_constructor();
    std::cout << std::endl;
    
    test_apply_inst_norm();
    std::cout << std::endl;
    
    test_apply_padding_or_crop();
    std::cout << std::endl;
    
    test_apply_augments_downsample();
    std::cout << std::endl;
    
    test_apply_augments_randsample();
    std::cout << std::endl;
    
    test_apply_downsampling_directly();
    std::cout << std::endl;
    
    test_normalization_with_params();
    std::cout << std::endl;
    
    std::cout << "============================================" << std::endl;
    std::cout << "All BaseDataset tests completed!" << std::endl;


    test_scientific_dataset();
    std::cout << "All dataset tests complete" << std::endl;

    return 0;
}

