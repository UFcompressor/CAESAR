#pragma once
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include  <map>
#include <utility>


struct CompressionResult {
  // A bit of a guess for right now for datatypes mainly first 2
    std::vector<float> latent;
    std::vector<uint8_t> postProcess;
    std::map<std::string, int> meta_data;
    std::vector<int> shape;
    std::vector<int> padding;
    std::vector<int> filteredBlocks;

};

struct MyDataset : torch::data::datasets::Dataset<MyDataset> {
    torch::Tensor data;

    MyDataset(torch::Tensor d) : data(d) {}

    // test methods i am not sure
    torch::data::Example<> get(size_t index) override {
        // just return the whole tensor as a single sample
        return {data, data};
    }

    // return the entier dataset
    torch::optional<size_t> size() const override {
        return 1;
    }
};


// for compression method
using CompressReturn = std::pair<CompressionResult, int>;

class Compressor {
public:
    Compressor(const std::string& modelPath,
               bool useDiffusion = true,
               const std::string& device = "cuda",
               int nFrames = 16,
               int interpolationRate = 3,
               int diffusionSteps = 32);
    
   std::map<std::string, torch::Tensor> removeModulePrefix(const std::map<std::string, torch::Tensor>& stateDict);

   // what the hell is a dataloader look into it more this also needs to be saved
 CompressReturn compress(MyDataset& data, double errorBound);






private:

   void loadModels();
 
   //both here
   // py def compress(self, dataloader, eb = 1e-3):
   

   // CAESAR V methods ------------------------
   void loadCaesarVCompressor();






   // CAESAR D methods down ---------------------
   void loadCaesarDCompressor(); // later to add in 

    std::string pretrainedPath;
    bool useDiffusion;
    std::string device;
    int nFrames;
    int interpolationRate;
    int diffusionSteps;

    torch::Tensor condIdx;
    torch::Tensor predIdx;

    torch::jit::script::Module model;

};



