#include "compressor.h"
#include <filesystem>
// NOTE: I am hard coding the path this needs to be changed I rather the user not have to give an extra argument
Compressor::Compressor(torch::Device device)
    : device_(device) , model_path_("/home/adios/Programs/CAESAR_C/models/compress_models/model.pt2") {
    load_model();
}

void Compressor::load_model() {
    if (!std::filesystem::exists(model_path_)) {
        throw std::runtime_error("Model file not found at: " + model_path_);
    }

    std::cout << "Loading model from: " << model_path_ << std::endl;
    loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_path_);
    std::cout << "Model loaded successfully!" << std::endl;
}

std::vector<torch::Tensor> Compressor::compress_single(const torch::Tensor& input_tensor) {
    std::vector<torch::Tensor> inputs = { input_tensor.to(device_) };
    return loader_->run(inputs);
}

CompressionResult Compressor::compress(const DatasetConfig& config , int batch_size) {
    c10::InferenceMode guard;

    std::cout << "\n========== STARTING COMPRESSION ==========" << std::endl;
    std::cout << "Device: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    ScientificDataset dataset(config);

    CompressionResult result;
    result.num_samples = dataset.size();
    result.num_batches = (dataset.size() + batch_size - 1) / batch_size;


    std::vector<torch::Tensor> batch_inputs;
    std::vector<torch::Tensor> batch_offsets;
    std::vector<torch::Tensor> batch_scales;
    std::vector<torch::Tensor> batch_indices;

    batch_inputs.reserve(batch_size);
    batch_offsets.reserve(batch_size);
    batch_scales.reserve(batch_size);
    batch_indices.reserve(batch_size);

    std::vector<torch::Tensor> all_latents;
    std::vector<torch::Tensor> all_hyper_latents;
    std::vector<torch::Tensor> all_offsets;
    std::vector<torch::Tensor> all_scales;
    std::vector<torch::Tensor> all_indices;

    int batch_count = 0;
    //TODO thread the batches
    for (size_t i = 0; i < dataset.size(); i++) {
        auto sample = dataset.get_item(i);

        batch_inputs.push_back(sample["input"]);
        batch_offsets.push_back(sample["offset"]);
        batch_scales.push_back(sample["scale"]);
        batch_indices.push_back(sample["index"]);

        if (batch_inputs.size() == static_cast<size_t>(batch_size) || i == dataset.size() - 1) {


            torch::Tensor batched_input = torch::cat(batch_inputs , 0).to(device_);


            auto outputs = compress_single(batched_input);

            if (outputs.size() != 2) {
                throw std::runtime_error("Expected 2 outputs (latent, hyper_latent), got " +
                    std::to_string(outputs.size()));
            }

            torch::Tensor latent = outputs[0];
            torch::Tensor hyper_latent = outputs[1];

            int64_t output_batch_size = latent.size(0);
            int64_t samples_in_batch = batch_inputs.size();

            for (size_t j = 0; j < samples_in_batch; ++j) {
                int64_t outputs_per_sample = output_batch_size / samples_in_batch;
                int64_t start_idx = j * outputs_per_sample;
                int64_t end_idx = (j + 1) * outputs_per_sample;

                all_latents.push_back(latent.slice(0 , start_idx , end_idx).cpu());
                all_hyper_latents.push_back(hyper_latent.slice(0 , start_idx , end_idx).cpu());
                all_offsets.push_back(batch_offsets[j].cpu());
                all_scales.push_back(batch_scales[j].cpu());
                all_indices.push_back(batch_indices[j].cpu());
            }


            batch_inputs.clear();
            batch_offsets.clear();
            batch_scales.clear();
            batch_indices.clear();

            batch_count++;
            if (batch_count % 10 == 0 || i == dataset.size() - 1) {
                std::cout << "  Processed " << (i + 1) << "/" << dataset.size()
                    << " samples (" << batch_count << " batches)" << std::endl;
            }
        }
    }


    result.latents = all_latents;
    result.hyper_latents = all_hyper_latents;
    result.offsets = all_offsets;
    result.scales = all_scales;
    result.indices = all_indices;


    return result;
}
