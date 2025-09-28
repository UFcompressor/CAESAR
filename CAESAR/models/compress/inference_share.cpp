#include <iostream>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>


void save_tensor_to_bin(const torch::Tensor& tensor, const std::string& filename) {
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    file.write(
        reinterpret_cast<const char*>(cpu_tensor.data_ptr<float>()), 
        cpu_tensor.numel() * sizeof(float) 
    );

    file.close();
    std::cout << "Tensor saved to " << filename << std::endl;
}


int main() {
    c10::InferenceMode mode;
    
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        device = torch::kCUDA;
    } else {
        std::cout << "Running on CPU." << std::endl;
    }
    
    const std::string in_filename = "../toy_test/RHO_example.f32"; 
    const std::vector<int64_t> target_shape = {1, 1, 8, 256, 256};
    std::ifstream infile(in_filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << in_filename << std::endl;
        return 1;
    }
    infile.seekg(0, std::ios::end);
    long long file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    long long num_floats = file_size / sizeof(float);
    long long required_elements = 1;
    for (auto dim : target_shape) { required_elements *= dim; }
    if (num_floats != required_elements) {
        std::cerr << "Error: Input file size mismatch." << std::endl;
        return 1;
    }
    std::vector<float> data_vec(num_floats);
    infile.read(reinterpret_cast<char*>(data_vec.data()), file_size);
    infile.close();
    torch::Tensor input_tensor = torch::from_blob(data_vec.data(), {num_floats}, torch::kFloat32)
                                     .clone()
                                     .view(target_shape)
                                     .to(device);
    std::cout << "Tensor loaded successfully. Shape: " << input_tensor.sizes() << std::endl;

    torch::inductor::AOTIModelPackageLoader loader("/home/eklasky/model.pt2");
    std::vector<torch::Tensor> inputs = {input_tensor};
    std::vector<torch::Tensor> outputs = loader.run(inputs);
    
    
    if (outputs.size() != 2) {
        std::cerr << "Error: Expected 2 output tensors, but got " << outputs.size() << std::endl;
        return 1;
    }

    
    torch::Tensor output1 = outputs[0];
    torch::Tensor output2 = outputs[1];

    std::cout << "Inference successful. Received 2 output tensors." << std::endl;
    std::cout << "Output 1 shape: " << output1.sizes() << std::endl;
    std::cout << "Output 2 shape: " << output2.sizes() << std::endl;

    
    save_tensor_to_bin(output1, "output1.bin");
    save_tensor_to_bin(output2, "output2.bin");

    return 0;
}
