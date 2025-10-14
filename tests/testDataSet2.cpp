#include "../CAESAR/dataset/dataset.h"

int main() {

    {
        DatasetConfig config;
        config.binary_path = "tensor_data_1.bin";  // String path to binary file
        config.n_frame = 16;
        config.dataset_name = "My Scientific Dataset";
        config.variable_idx = 0;  // Select first variable
        config.section_range = { 0, 100 };  // Use sections 0-99
        config.frame_range = { 0, 1000 };   // Use frames 0-999
        config.train_mode = true;
        config.train_size = 256;
        config.inst_norm = true;
        config.norm_type = "mean_range";

        try {
            ScientificDataset dataset(config);
            std::cout << "Dataset loaded successfully!" << std::endl;
            std::cout << "Dataset size: " << dataset.size() << std::endl;

            // Get a sample
            auto sample = dataset.get_item(0);
            std::cout << "Sample input shape: " << sample["input"].sizes() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }


    {
        // Assume you have a pre-loaded tensor (e.g., from another source)
        torch::Tensor preloaded_data = torch::randn({ 2, 10, 100, 64, 64 }); // [V, S, T, H, W]

        DatasetConfig config;
        config.memory_data = preloaded_data;  // Pass the tensor directly
        config.n_frame = 20;
        config.dataset_name = "Memory Dataset";
        config.train_mode = false;  // Test mode
        config.test_size = { 128, 128 };
        config.inst_norm = false;
        config.norm_type = "min_max";

        try {
            ScientificDataset dataset(config);
            std::cout << "Memory dataset loaded successfully!" << std::endl;
            std::cout << "Dataset size: " << dataset.size() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    // Example 3: With augmentations
    {
        DatasetConfig config;
        config.binary_path = "tensor_data_2.bin";
        config.n_frame = 8;
        config.train_mode = true;
        config.train_size = 512;

        // Add augmentations
        config.augment_type["downsample"] = 4;  // Max downsample factor of 4
        // Or use random sampling: config.augment_type["randsample"] = 3;

        try {
            ScientificDataset dataset(config);
            std::cout << "Dataset with augmentations loaded!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}

// Helper function to create binary dataset file (for testing)
void create_test_binary_file(const std::string& filename) {
    // Create test data: 2 variables, 5 sections, 50 time steps, 32x32 spatial
    int64_t shape[5] = { 2, 5, 50, 32, 32 };
    size_t num_elements = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];

    std::vector<float> data(num_elements);

    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f , 1.0f);

    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = dis(gen);
    }

    // Write to binary file
    std::ofstream file(filename , std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create binary file: " + filename);
    }

    // Write shape first
    file.write(reinterpret_cast<const char*>(shape) , 5 * sizeof(int64_t));

    // Write data
    file.write(reinterpret_cast<const char*>(data.data()) , num_elements * sizeof(float));

    file.close();
    std::cout << "Created test binary file: " << filename << std::endl;
}
