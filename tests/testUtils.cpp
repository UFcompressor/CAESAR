#include "../CAESAR/models/utils.h"
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



// Mock DataLoader class to simulate PyTorch DataLoader behavior
template<typename T>
class DataLoader {
private:
    std::vector<T> data;
    size_t batch_size;
    bool shuffle;
    bool pin_memory;

public:
    // Required type definitions for the cycle template
    using value_type = T;
    using const_iterator = typename std::vector<T>::const_iterator;
    using iterator = typename std::vector<T>::iterator;

    DataLoader(const std::vector<T>& dataset, size_t bs = 1, bool shuf = false, bool pin = false) 
        : data(dataset), batch_size(bs), shuffle(shuf), pin_memory(pin) {
        if (shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(data.begin(), data.end(), g);
        }
    }
    
    // Required iterator interface for cycle template
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }
    iterator begin() { return data.begin(); }
    iterator end() { return data.end(); }
    
    // Required for cycle template
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
    
    // Your existing functionality
    std::vector<T> get_batch(size_t start_idx) const {
        std::vector<T> batch;
        for (size_t i = start_idx; i < std::min(start_idx + batch_size, data.size()); ++i) {
            batch.push_back(data[i]);
        }
        return batch;
    }
};

// Test helper functions
void test_basic_functionality() {
    std::cout << "=== BASIC FUNCTIONALITY TESTS ===\n";

    // Test 1: Simple integer vector
    std::vector<int> ints = {1, 2, 3};
    auto int_cycler = cycle(ints);

    std::cout << "Test 1 - Integer cycling: ";
    for (int i = 0; i < 9; ++i) {
        std::cout << int_cycler() << " ";
    }
    std::cout << "\n";

    // Test 2: String vector
    std::vector<std::string> strings = {"hello", "world", "test"};
    auto string_cycler = cycle(strings);

    std::cout << "Test 2 - String cycling: ";
    for (int i = 0; i < 7; ++i) {
        std::cout << string_cycler() << " ";
    }
    std::cout << "\n";

    // Test 3: Single element
    std::vector<int> single = {42};
    auto single_cycler = cycle(single);

    std::cout << "Test 3 - Single element: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << single_cycler() << " ";
    }
    std::cout << "\n\n";
}

void test_different_containers() {
    std::cout << "=== DIFFERENT CONTAINER TESTS ===\n";

    // Test with std::list
    std::list<int> int_list = {10, 20, 30, 40};
    auto list_cycler = cycle(int_list);

    std::cout << "Test 1 - std::list: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << list_cycler() << " ";
    }
    std::cout << "\n";

    // Test with std::deque
    std::deque<char> char_deque = {'A', 'B', 'C'};
    auto deque_cycler = cycle(char_deque);

    std::cout << "Test 2 - std::deque: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << deque_cycler() << " ";
    }
    std::cout << "\n";

    // Test with std::array
    std::array<float, 4> float_array = {1.1f, 2.2f, 3.3f, 4.4f};
    auto array_cycler = cycle(float_array);

    std::cout << "Test 3 - std::array: ";
    for (int i = 0; i < 9; ++i) {
        std::cout << array_cycler() << " ";
    }
    std::cout << "\n\n";
}

void test_dataloader_simulation() {
    std::cout << "=== DATALOADER SIMULATION TESTS ===\n";

    // Simulate dataset
    std::vector<int> dataset = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Test 1: Basic DataLoader
    DataLoader<int> dl1(dataset, 3, false, false);  // batch_size=3, no shuffle
    auto dl_cycler1 = cycle(dl1);

    std::cout << "Test 1 - DataLoader basic (batch_size=3): ";
    for (int i = 0; i < 15; ++i) {
        std::cout << dl_cycler1() << " ";
    }
    std::cout << "\n";

    // Test 2: Shuffled DataLoader
    DataLoader<int> dl2(dataset, 2, true, true);  // batch_size=2, shuffle=true, pin_memory=true
    auto dl_cycler2 = cycle(dl2);

    std::cout << "Test 2 - DataLoader shuffled (batch_size=2): ";
    for (int i = 0; i < 15; ++i) {
        std::cout << dl_cycler2() << " ";
    }
    std::cout << "\n";

    // Test 3: String DataLoader (simulating image paths or labels)
    std::vector<std::string> image_paths = {"img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"};
    DataLoader<std::string> image_dl(image_paths, 2, true, false);
    auto image_cycler = cycle(image_dl);

    std::cout << "Test 3 - Image path DataLoader: ";
    for (int i = 0; i < 12; ++i) {
        std::cout << image_cycler() << " ";
    }
    std::cout << "\n\n";
}

void test_complex_data_structures() {
    std::cout << "=== COMPLEX DATA STRUCTURE TESTS ===\n";

    // Test with pairs (simulating data/label pairs)
    std::vector<std::pair<int, std::string>> data_label_pairs = {
        {1, "cat"}, {2, "dog"}, {3, "bird"}, {4, "fish"}
    };
    auto pair_cycler = cycle(data_label_pairs);

    std::cout << "Test 1 - Data/Label pairs: ";
    for (int i = 0; i < 10; ++i) {
        auto pair = pair_cycler();
        std::cout << "(" << pair.first << "," << pair.second << ") ";
    }
    std::cout << "\n";

    // Test with vectors of vectors (simulating batches)
    std::vector<std::vector<int>> batches = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
    };
    auto batch_cycler = cycle(batches);

    std::cout << "Test 2 - Batch cycling: ";
    for (int i = 0; i < 7; ++i) {
        auto batch = batch_cycler();
        std::cout << "[";
        for (size_t j = 0; j < batch.size(); ++j) {
            std::cout << batch[j];
            if (j < batch.size() - 1) std::cout << ",";
        }
        std::cout << "] ";
    }
    std::cout << "\n";

    // Test with shared_ptr (simulating tensor/matrix objects)
    std::vector<std::shared_ptr<std::vector<float>>> tensor_ptrs = {
        std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f}),
        std::make_shared<std::vector<float>>(std::vector<float>{3.0f, 4.0f}),
        std::make_shared<std::vector<float>>(std::vector<float>{5.0f, 6.0f})
    };
    auto tensor_cycler = cycle(tensor_ptrs);

    std::cout << "Test 3 - Tensor pointers: ";
    for (int i = 0; i < 8; ++i) {
        auto tensor = tensor_cycler();
        std::cout << "[" << (*tensor)[0] << "," << (*tensor)[1] << "] ";
    }
    std::cout << "\n\n";
}

void test_edge_cases() {
    std::cout << "=== EDGE CASE TESTS ===\n";

    // Test 1: Very large dataset (performance test)
    std::vector<int> large_dataset;
    for (int i = 0; i < 10000; ++i) {
        large_dataset.push_back(i);
    }

    auto large_cycler = cycle(large_dataset);
    auto start = std::chrono::high_resolution_clock::now();

    // Cycle through 50000 elements
    for (int i = 0; i < 50000; ++i) {
        volatile int val = large_cycler();  // volatile to prevent optimization
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Test 1 - Large dataset (10k elements, 50k cycles): " << duration.count() << "ms\n";

    // Test 2: Multiple cyclers from same container
    std::vector<int> shared_data = {100, 200, 300};
    auto cycler_a = cycle(shared_data);
    auto cycler_b = cycle(shared_data);

    std::cout << "Test 2 - Multiple cyclers: ";
    for (int i = 0; i < 6; ++i) {
        std::cout << "A:" << cycler_a() << " B:" << cycler_b() << " ";
    }
    std::cout << "\n";

    // Test 3: Cycling after container modification (dangerous but test behavior)
    std::vector<int> modifiable = {1, 2, 3};
    auto mod_cycler = cycle(modifiable);

    std::cout << "Test 3 - Before modification: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << mod_cycler() << " ";
    }

    // Note: In real usage, don't modify container after creating cycler!
    // This is just to test current behavior
    std::cout << "\n";

    // Test 4: Very frequent cycling
    std::vector<char> frequent = {'X', 'Y'};
    auto freq_cycler = cycle(frequent);

    std::cout << "Test 4 - Frequent cycling: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << freq_cycler();
    }
    std::cout << "\n\n";
}

void test_error_handling() {
    std::cout << "=== ERROR HANDLING TESTS ===\n";

    // Test 1: Empty container
    std::vector<int> empty_vec;
    auto empty_cycler = cycle(empty_vec);

    std::cout << "Test 1 - Empty container: ";
    try {
        empty_cycler();
        std::cout << "ERROR: Should have thrown exception!\n";
    } catch (const std::runtime_error& e) {
        std::cout << "Correctly threw exception: " << e.what() << "\n";
    }

    std::cout << "\n";
}

void test_training_simulation() {
    std::cout << "=== TRAINING SIMULATION TEST ===\n";

    // Simulate a training scenario like the original Python code
    struct TrainingBatch {
        std::vector<float> data;
        int label;

        TrainingBatch(std::vector<float> d, int l) : data(std::move(d)), label(l) {}
    };

    // Create dataset
    std::vector<TrainingBatch> training_dataset;
    for (int i = 0; i < 100; ++i) {
        std::vector<float> sample_data = {
            static_cast<float>(i),
            static_cast<float>(i * 2),
            static_cast<float>(i * 3)
        };
        training_dataset.emplace_back(std::move(sample_data), i % 10);
    }

    // Create DataLoader equivalent
    DataLoader<TrainingBatch> train_loader(training_dataset, 8, true, true);  // batch_size=8, shuffle=true

    // Create cycler (equivalent to: self.dl = cycle(DataLoader(...)))
    auto train_cycler = cycle(train_loader);

    std::cout << "Simulating training epochs with cycling DataLoader:\n";

    // Simulate multiple epochs
    for (int epoch = 0; epoch < 3; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << ": ";

        // Process some batches (in real training, this would be all batches)
        for (int batch_idx = 0; batch_idx < 10; ++batch_idx) {
            auto batch = train_cycler();  // Get next training batch
            std::cout << "L" << batch.label << " ";  // Print label for verification
        }
        std::cout << "\n";
    }

    std::cout << "\nTraining simulation complete!\n\n";
}






int main() {
  using std::optional;
    using std::string;

    std::cout << "Running defaultVaule tests...\n";
   
    {
        optional<int> x = 42;
        int result = defaultVaule(x, 99);
        assert(result == 42 && "Test 1 failed: expected 42");
    }

    {
        optional<int> x = std::nullopt;
        int result = defaultVaule(x, 99);
        assert(result == 99 && "Test 2 failed: expected 99");
    }
    
    {
        optional<int> x = std::nullopt;
        int result = defaultVaule(x, [](){ return 123; });
        assert(result == 123 && "Test 3 failed: expected 123");
    }

    {
        optional<double> y = 5.5;
        double result = defaultVaule(y, 7.7);
        assert(result == 5.5 && "Test 4 failed: expected 5.5");
    }

    {
        optional<string> s = string("hello");
        string result = defaultVaule(s, string("world"));
        assert(result == "hello" && "Test 5 failed: expected hello");
    }

    {
        optional<string> s = std::nullopt;
        string result = defaultVaule(s, string("world"));
        assert(result == "world" && "Test 6 failed: expected world");
    }

    {
        optional<string> s = std::nullopt;
        string result = defaultVaule(s, [](){ return string("computed"); });
        assert(result == "computed" && "Test 7 failed: expected computed");
    }

    {
        auto tensor = torch::tensor({1, 2, 3}, torch::kInt32);
        optional<torch::Tensor> t = tensor;
        auto result = defaultVaule(t, torch::zeros({3}, torch::kInt32));
        assert(torch::equal(result, tensor) && "Test 8 failed: tensor mismatch");
    }


    {
        optional<torch::Tensor> t = std::nullopt;
        auto result = defaultVaule(t, torch::zeros({3}, torch::kInt32));
        assert(torch::equal(result, torch::zeros({3}, torch::kInt32)) &&
               "Test 9 failed: expected zeros tensor");
    }

    {
        int a = 42;
        optional<int*> ptr = &a;
        int* result = defaultVaule(ptr, nullptr);
        assert(result == &a && *result == 42 && "Test 10 failed: pointer mismatch");
    }
    std::cout<<"All defaultVaule tests passed!\n";

    std::cout << "COMPREHENSIVE CYCLE FUNCTION TESTS\n";
    std::cout << "==================================\n\n";
    
    test_basic_functionality();
    test_different_containers();
    test_dataloader_simulation();
    test_complex_data_structures();
    test_edge_cases();
    test_error_handling();
    test_training_simulation();
    
    std::cout << "=== ALL TESTS COMPLETED ===\n";



    return 0;
}

