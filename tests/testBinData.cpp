#include "dataset.h"
#include <torch/torch.h>

int main() {
    {
        BinaryFileConfig bin_config;
        bin_config.file_path = "test_data_small.bin";
        bin_config.dimensions = { 1, 1, 8, 256, 256 };
        bin_config.data_type = torch::kFloat32;

        DatasetConfig config;
        config.binary_config = bin_config;
        config.n_frame = 8;
        config.train_mode = true;
        config.inst_norm = true;

        try {
            ScientificDataset dataset(config);
            std::cout << "Dataset loaded successfully!" << std::endl;
            std::cout << "Dataset size: " << dataset.size() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }


    {
        BinaryFileConfig bin_config;
        bin_config.file_path = "test_data_large.bin";
        bin_config.dimensions = { 1, 10, 30, 300, 300 };
        bin_config.data_type = torch::kFloat64;

        DatasetConfig config;
        config.binary_config = bin_config;
        config.n_frame = 10;
        config.train_mode = false;
        config.test_size = { 256, 256 };

        try {
            ScientificDataset dataset(config);
            std::cout << "Large dataset loaded successfully!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    {
        BinaryFileConfig bin_config;
        bin_config.file_path = "test_data_multivar.bin";
        bin_config.dimensions = { 5, 100, 100, 100, 100 };
        bin_config.data_type = torch::kFloat32;

        DatasetConfig config;
        config.binary_config = bin_config;
        config.variable_idx = 2;
        config.n_frame = 8;
        config.train_mode = true;

        try {
            ScientificDataset dataset(config);
            std::cout << "Multi-variable dataset loaded with selection!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }


    {
        torch::Tensor memory_data = torch::randn({ 1, 1, 16, 256, 256 });

        DatasetConfig config;
        config.memory_data = memory_data;
        config.n_frame = 8;
        config.train_mode = true;

        try {
            ScientificDataset dataset(config);
            std::cout << "Memory dataset loaded successfully!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }


    return 0;
}

