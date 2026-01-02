#include <iostream>
#include <fstream>
#include <vector>
#include <torch/torch.h>
#include <iomanip>
#include <cassert>
#include "../CAESAR/dataset/dataset.h"


// Simple test framework
int tests_run = 0;
int tests_passed = 0;

#define TEST(name) void test_##name() { \
    std::cout << "\nTesting " #name "..." << std::endl; \
    tests_run++;

#define ASSERT_TRUE(condition, msg) \
    if (condition) { \
        std::cout << "  ✓ " << msg << std::endl; \
    } else { \
        std::cout << "  ✗ " << msg << std::endl; \
        return; \
    } \

#define ASSERT_EQ(a, b, msg) \
    if ((a) == (b)) { \
        std::cout << "  ✓ " << msg << std::endl; \
    } else { \
        std::cout << "  ✗ " << msg << " (got " << (a) << ", expected " << (b) << ")" << std::endl; \
        return; \
    }

#define TEST_PASS() tests_passed++; std::cout << "  PASSED" << std::endl;

// Helper: create test data
torch::Tensor create_test_tensor() {
    torch::manual_seed(42);
    return torch::randn({ 2, 3, 24, 64, 64 } , torch::kFloat);
}

// Helper: save binary data
void save_binary(const torch::Tensor& data , const std::string& path) {
    std::ofstream file(path , std::ios::binary);
    auto sizes = data.sizes();
    std::vector<int64_t> shape(sizes.begin() , sizes.end());
    file.write(reinterpret_cast<const char*>(shape.data()) , shape.size() * sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(data.data_ptr<float>()) , data.numel() * sizeof(float));
    file.close();
}

TEST(memory_loading)
auto data = create_test_tensor();

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.train_size = 64; // Match input size to avoid padding issues

ScientificDataset dataset(config);

ASSERT_TRUE(dataset.size() > 0 , "Dataset created from memory");
ASSERT_TRUE(dataset.original_data().defined() , "Original data accessible");
ASSERT_TRUE(dataset.input_data().defined() , "Input data accessible");

auto item = dataset.get_item(0);
ASSERT_TRUE(item.find("input") != item.end() , "Item has input");
ASSERT_TRUE(item.find("offset") != item.end() , "Item has offset");
ASSERT_TRUE(item.find("scale") != item.end() , "Item has scale");
ASSERT_TRUE(item.find("index") != item.end() , "Item has index");

TEST_PASS();
}


TEST(variable_selection)
auto data = create_test_tensor(); // [2, 3, 24, 64, 64]

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.variable_idx = 1; // Select second variable
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.train_size = 64; // Match input size

ScientificDataset dataset(config);

auto original = dataset.original_data();
ASSERT_EQ(original.size(0) , 1 , "Selected 1 variable");
ASSERT_EQ(original.size(1) , 3 , "Kept all sections");

TEST_PASS();
}

TEST(section_range_selection)
auto data = create_test_tensor(); // [2, 3, 24, 64, 64]

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.section_range = { 1, 3 }; // Select sections 1-2
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.train_size = 64; // Match input size

ScientificDataset dataset(config);

auto original = dataset.original_data();
ASSERT_EQ(original.size(0) , 2 , "Kept all variables");
ASSERT_EQ(original.size(1) , 2 , "Selected 2 sections");

TEST_PASS();
}

TEST(frame_range_selection)
auto data = create_test_tensor(); // [2, 3, 24, 64, 64]

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.frame_range = { 8, 20 }; // Select frames 8-19 (12 frames)
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.train_size = 64; // Match input size

ScientificDataset dataset(config);

auto input = dataset.input_data();
ASSERT_EQ(input.size(2) , 12 , "Selected 12 frames");

TEST_PASS();
}

TEST(padding_logic)
    // Test the padding logic specifically
    auto data = torch::randn({ 1, 1, 8, 32, 32 }); // Smaller than train_size

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.train_size = 48; // Larger than input (32) but reasonable

try {
    ScientificDataset dataset(config);
    if (dataset.size() > 0) {
        auto item = dataset.get_item(0);
        auto input = item.at("input");
        ASSERT_EQ(input.size(-1) , 48 , "Input padded to train_size");
        ASSERT_EQ(input.size(-2) , 48 , "Input padded to train_size");
    }
    TEST_PASS();
}
catch (const std::exception& e) {
    std::cout << "  ✗ Padding failed: " << e.what() << std::endl;
    return;
}
}

TEST(cropping_logic)
    // Test the cropping logic
    auto data = torch::randn({ 1, 1, 8, 80, 80 }); // Larger than train_size

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.train_size = 64; // Smaller than input (80)

ScientificDataset dataset(config);

if (dataset.size() > 0) {
    auto item = dataset.get_item(0);
    auto input = item.at("input");
    ASSERT_EQ(input.size(-1) , 64 , "Input cropped to train_size");
    ASSERT_EQ(input.size(-2) , 64 , "Input cropped to train_size");
}

TEST_PASS();
}

TEST(mean_range_normalization)
auto data = torch::ones({ 1, 1, 8, 14, 14 }) * 10.0f; // All values = 10

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;

ScientificDataset dataset(config);

if (dataset.size() > 0) {
    auto item = dataset.get_item(0);
    auto input = item.at("input");

    // With constant data, mean_range should center at 0
    float mean_val = torch::mean(input).item<float>();
    ASSERT_TRUE(std::abs(mean_val) < 1e-5 , "Mean range normalization centers data");
}

TEST_PASS();
}

TEST(min_max_normalization)
auto data = torch::arange(0 , 64).view({ 1, 1, 8, 32, 32 }).to(torch::kFloat); // Values 0-63

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "min_max";
config.train_mode = true;

ScientificDataset dataset(config);

if (dataset.size() > 0) {
    auto item = dataset.get_item(0);
    auto input = item.at("input");

    float min_val = torch::min(input).item<float>();
    float max_val = torch::max(input).item<float>();

    ASSERT_TRUE(min_val >= -1.1f && min_val <= -0.9f , "Min-max normalization min ~= -1");
    ASSERT_TRUE(max_val >= 0.9f && max_val <= 1.1f , "Min-max normalization max ~= 1");
}

TEST_PASS();
}

TEST(temporal_overlap)
auto data = create_test_tensor(); // [2, 3, 24, 64, 64]

DatasetConfig config;
config.n_frame = 8;
config.n_overlap = 4; // 50% overlap
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;

ScientificDataset dataset(config);

if (dataset.size() >= 2) {
    auto item1 = dataset.get_item(0);
    auto item2 = dataset.get_item(1);

    auto index1 = item1.at("index");
    auto index2 = item2.at("index");

    int64_t start1 = index1[2].item<int64_t>();
    int64_t start2 = index2[2].item<int64_t>();

    // Delta should be n_frame - n_overlap = 8 - 4 = 4
    ASSERT_EQ(start2 - start1 , 4 , "Overlap delta is correct");
}

TEST_PASS();
}

TEST(test_mode_blocking)
auto data = create_test_tensor(); // [2, 3, 24, 64, 64]

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = false; // Test mode
config.test_size = { 32, 32 };

ScientificDataset dataset(config);

ASSERT_TRUE(dataset.size() > 0 , "Test mode creates valid dataset");

if (dataset.size() > 0) {
    auto item = dataset.get_item(0);
    auto input = item.at("input");
    ASSERT_TRUE(input.defined() , "Test mode produces valid samples");
}

auto original = dataset.original_data();
ASSERT_TRUE(original.defined() , "Original data available in test mode");

TEST_PASS();
}

TEST(data_reconstruction)
auto data = create_test_tensor();

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;

ScientificDataset dataset(config);

auto original = dataset.original_data();
auto reconstructed = dataset.recons_data(original);

ASSERT_TRUE(reconstructed.defined() , "Reconstruction method works");
ASSERT_TRUE(reconstructed.sizes() == dataset.input_data().sizes() , "Reconstruction has correct shape");

TEST_PASS();
}

TEST(index_bounds)
auto data = create_test_tensor();

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;

ScientificDataset dataset(config);

size_t size = dataset.size();
ASSERT_TRUE(size > 0 , "Dataset has valid size");

// Test first item
auto item_first = dataset.get_item(0);
ASSERT_TRUE(item_first.at("input").defined() , "First item valid");

// Test last item
auto item_last = dataset.get_item(size - 1);
ASSERT_TRUE(item_last.at("input").defined() , "Last item valid");

// Test wraparound (should work due to modulo in get_item)
auto item_wrap = dataset.get_item(size);
ASSERT_TRUE(item_wrap.at("input").defined() , "Index wraparound works");

TEST_PASS();
}

TEST(consistent_output)
auto data = create_test_tensor();

DatasetConfig config;
config.n_frame = 8;
config.memory_data = data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.augment_type.clear(); // No augmentations

ScientificDataset dataset(config);

if (dataset.size() > 0) {
    auto item1 = dataset.get_item(0);
    auto item2 = dataset.get_item(0);

    bool inputs_equal = torch::allclose(item1.at("input") , item2.at("input") , 1e-6 , 1e-7);
    ASSERT_TRUE(inputs_equal , "Same index gives consistent results");

    bool indices_equal = torch::equal(item1.at("index") , item2.at("index"));
    ASSERT_TRUE(indices_equal , "Same index gives same indices");
}

TEST_PASS();
}

TEST(empty_data_handling)
    // Test with minimal valid data
    auto small_data = torch::randn({ 1, 1, 8, 4, 4 });

DatasetConfig config;
config.n_frame = 8;
config.memory_data = small_data;
config.inst_norm = true;
config.norm_type = "mean_range";
config.train_mode = true;
config.train_size = 4;

ScientificDataset dataset(config);

ASSERT_TRUE(dataset.size() > 0 , "Minimal data creates valid dataset");

if (dataset.size() > 0) {
    auto item = dataset.get_item(0);
    ASSERT_TRUE(item.at("input").defined() , "Minimal data produces valid items");
}

TEST_PASS();
}

int main() {
    std::cout << "Dataset Functionality Tests" << std::endl;
    std::cout << "============================" << std::endl;

    try {
        test_memory_loading();
        test_variable_selection();
        test_section_range_selection();
        test_frame_range_selection();
        test_padding_logic();
        test_cropping_logic();
        test_mean_range_normalization();
        test_min_max_normalization();
        test_temporal_overlap();
        test_test_mode_blocking();
        test_data_reconstruction();
        test_index_bounds();
        test_consistent_output();
        test_empty_data_handling();

        std::cout << "\n============================" << std::endl;
        std::cout << "Results: " << tests_passed << "/" << tests_run << " tests passed" << std::endl;

        if (tests_passed == tests_run) {
            std::cout << "All tests PASSED!" << std::endl;
            return 0;
        }
        else {
            std::cout << (tests_run - tests_passed) << " tests FAILED!" << std::endl;
            return 1;
        }

    }
    catch (const std::exception& e) {
        std::cout << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
