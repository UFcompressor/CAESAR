#include "../CAESAR/dataset/dataset.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

// Function to load raw float32 binary into a tensor with specified shape
torch::Tensor loadRawBinary(const std::string& bin_path , const std::vector<int64_t>& shape) {
    std::ifstream file(bin_path , std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + bin_path);
    }

    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }

    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()) , num_elements * sizeof(float));
    file.close();

    torch::Tensor data = torch::from_blob(
        buffer.data() ,
        shape ,
        torch::kFloat32
    ).clone();

    std::cout << "Loaded " << bin_path << " with shape: " << data.sizes() << "\n";
    std::cout << "  Min: " << data.min().item<float>() << ", Max: " << data.max().item<float>() << "\n";

    return data;
}


double computeNRMSE(const torch::Tensor& pred , const torch::Tensor& target) {
    auto diff = pred - target;
    auto mse = torch::mean(diff * diff).item<double>();
    auto rmse = std::sqrt(mse);

    auto target_min = target.min().item<double>();
    auto target_max = target.max().item<double>();
    auto range = target_max - target_min;

    double nrmse = rmse / range;

    return nrmse;
}

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "   Testing C++ Dataset NRMSE\n";
        std::cout << "========================================\n\n";


        std::cout << "Loading TCf48.bin.f32 raw data...\n";
        torch::Tensor raw_data = loadRawBinary("TCf48.bin.f32" , { 1, 1, 100, 500, 500 });
        std::cout << "\n";

        DatasetConfig config;
        config.memory_data = raw_data;
        config.variable_idx = 0;
        config.n_frame = 8;
        config.dataset_name = "TCf48 Dataset";
        config.section_range = std::nullopt;
        config.frame_range = std::nullopt;
        config.train_size = 256;
        config.inst_norm = true;
        config.norm_type = "mean_range";
        config.train_mode = false;
        config.n_overlap = 0;
        config.test_size = { 256, 256 };
        config.augment_type = {};

        std::cout << "Creating dataset with configuration:\n";
        std::cout << "  - Data source: memory\n";
        std::cout << "  - variable_idx: " << config.variable_idx.value() << "\n";
        std::cout << "  - n_frame: " << config.n_frame << "\n";
        std::cout << "  - train_size: " << config.train_size << "\n";
        std::cout << "  - inst_norm: " << (config.inst_norm ? "true" : "false") << "\n";
        std::cout << "  - norm_type: " << config.norm_type << "\n";
        std::cout << "  - train_mode: " << (config.train_mode ? "true" : "false") << "\n\n";

        // Create dataset
        ScientificDataset dataset(config);

        std::cout << "Dataset created successfully!\n";
        std::cout << "Dataset size: " << dataset.size() << " samples\n\n";


        std::cout << "Loading reference batches from Python...\n";
        torch::Tensor batch_0_ref = loadRawBinary("TCf48_batch_0.bin" , { 32, 1, 8, 256, 256 });
        torch::Tensor batch_1_ref = loadRawBinary("TCf48_batch_1.bin" , { 20, 1, 8, 256, 256 });

        std::cout << "\nBatch 0 reference - shape: " << batch_0_ref.sizes()
            << ", min: " << batch_0_ref.min().item<float>()
            << ", max: " << batch_0_ref.max().item<float>() << "\n";
        std::cout << "Batch 1 reference - shape: " << batch_1_ref.sizes()
            << ", min: " << batch_1_ref.min().item<float>()
            << ", max: " << batch_1_ref.max().item<float>() << "\n\n";


        std::cout << "Creating batches from C++ dataset...\n";


        std::vector<torch::Tensor> batch_0_samples;
        for (size_t i = 0; i < 32; ++i) {
            auto data_dict = dataset.get_item(i);
            batch_0_samples.push_back(data_dict["input"]);
        }
        torch::Tensor batch_0_cpp = torch::cat(batch_0_samples , 0);

        std::cout << "Batch 0 C++ - shape: " << batch_0_cpp.sizes()
            << ", min: " << batch_0_cpp.min().item<float>()
            << ", max: " << batch_0_cpp.max().item<float>() << "\n";


        std::vector<torch::Tensor> batch_1_samples;
        for (size_t i = 32; i < 52; ++i) {
            auto data_dict = dataset.get_item(i);
            batch_1_samples.push_back(data_dict["input"]);
        }
        torch::Tensor batch_1_cpp = torch::cat(batch_1_samples , 0);

        std::cout << "Batch 1 C++ - shape: " << batch_1_cpp.sizes()
            << ", min: " << batch_1_cpp.min().item<float>()
            << ", max: " << batch_1_cpp.max().item<float>() << "\n\n";

        std::cout << "========================================\n";
        std::cout << "   Computing NRMSE\n";
        std::cout << "========================================\n\n";

        double nrmse_batch_0 = computeNRMSE(batch_0_cpp , batch_0_ref);
        double nrmse_batch_1 = computeNRMSE(batch_1_cpp , batch_1_ref);

        std::cout << std::scientific << std::setprecision(10);
        std::cout << "Batch 0 NRMSE: " << nrmse_batch_0 << "\n";
        std::cout << "Batch 1 NRMSE: " << nrmse_batch_1 << "\n\n";


        double threshold = 1e-6;
        bool batch_0_pass = nrmse_batch_0 < threshold;
        bool batch_1_pass = nrmse_batch_1 < threshold;

        std::cout << "========================================\n";
        std::cout << "   Results\n";
        std::cout << "========================================\n\n";
        std::cout << "Threshold: " << threshold << "\n";
        std::cout << "Batch 0: " << (batch_0_pass ? "PASS " : "FAIL ") << "\n";
        std::cout << "Batch 1: " << (batch_1_pass ? "PASS " : "FAIL ") << "\n\n";

        if (batch_0_pass && batch_1_pass) {
            std::cout << "SUCCESS: All batches are within acceptable NRMSE!\n";
            std::cout << "C++ and Python dataloaders produce equivalent results.\n";
            return 0;
        }
        else {
            std::cout << "FAILURE: Some batches exceed NRMSE threshold.\n";
            std::cout << "There may be differences between C++ and Python implementations.\n";
            return 1;
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}