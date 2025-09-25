#include "../CAESAR/dataset/dataset.h"
#include <torch/torch.h>
#include <iostream>
#include <unordered_map>

int main() {
    // 1️⃣ Create numeric tensor data
    torch::Tensor memory_data = torch::rand({2,3,10,64,64});

    // 2️⃣ Set up args for ScientificDataset
    std::unordered_map<std::string, torch::Tensor> args;
    args["memory_data"] = memory_data;
    args["variable_idx"] = torch::tensor(0);
    args["section_start"] = torch::tensor(0);
    args["section_end"] = torch::tensor(3);
    args["frame_start"] = torch::tensor(0);
    args["frame_end"] = torch::tensor(10);

    // 3️⃣ Construct dataset (constructor handles everything)
    ScientificDataset dataset(args);

    // 4️⃣ Just print shape from a tensor we passed in
    auto shape = memory_data.sizes(); // Use the original tensor shape
    std::cout << "Dataset shape: ";
    for (auto dim : shape) std::cout << dim << " ";
    std::cout << "\n";

    return 0;
}

