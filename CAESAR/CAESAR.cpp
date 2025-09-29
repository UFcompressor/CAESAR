/*
 * I am wright now just converting the files stright into libtoch nothing fancy
 * only CAESAR_V
 * 
 *
 */

#include "CAESAR.h"
#include <iostream>


// TODO: I need to convert a PCA class from this file path .ipynb_checkpoints/
// run_gae_cuda-checkpoint.py
//i and this file as well     
////   this is for model D    from .models.video_diffusion_interpo import Unet3D, GaussianDiffusion
// convert this class as  well    from .models import compress_modules3d_mid_SR as compress_modules
//
//

Compressor::Compressor(const std::string& modelPath,
                       bool useDiffusion,
                       const std::string& device,
                       int nFrames,
                       int interpolationRate,
                       int diffusionSteps)
    : pretrainedPath(modelPath),
      useDiffusion(useDiffusion),
      device(device),
      nFrames(nFrames),
      interpolationRate(interpolationRate),
      diffusionSteps(diffusionSteps)
{
   loadModels();

    condIdx = torch::arange(0, nFrames, interpolationRate);


    auto allIdx = torch::arange(nFrames);
    auto mask = torch::zeros({nFrames}, torch::kBool);
    for (int i = 0; i < condIdx.size(0); i++) {
        mask.index_put_({condIdx[i].item<int>()}, true);
    }
    predIdx = ~mask;
    

    // this should be change for later for serial parallel cpu and amd/invida gpu backend
    torch::globalContext().setDeterministicCuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(false);

}


std::map<std::string, torch::Tensor>
Compressor::removeModulePrefix(const std::map<std::string, torch::Tensor>& stateDict)
{
    std::map<std::string, torch::Tensor> newStateDict;
    for (const auto& kv : stateDict) {
        std::string newKey = kv.first;
        const std::string prefix = "module.";
        if (newKey.rfind(prefix, 0) == 0) {
            newKey = newKey.substr(prefix.size());
        }
        newStateDict[newKey] = kv.second;
    }
    return newStateDict;
}

void Compressor::loadModels(){
    if (useDiffusion == false)
       loadCaesarVCompressor();
    else
       loadCaesarDCompressor();

}



void Compressor::loadCaesarVCompressor(){
    std::cout<<"Loading CAESAR-V"<<std::endl;
    // I have to think about this because I read it in, in main convert my logic to this
    /*here is the python logic bellow
     *        from .models import compress_modules3d_mid_SR as compress_modules
        print("Loading CAESAE-V")
        model = compress_modules.CompressorMix(
            dim=16,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1,
            d3=True,
            sr_dim=16
        )
       note I think this is what needs to be changed with the load to my load -?????????????????????????????
        state_dict = self.remove_module_prefix(torch.load(self.pretrained_path, map_location=self.device))
        model.load_state_dict(state_dict)
        self.compressor_v = model.to(self.device).eval()
   
     */

}


void Compressor::loadCaesarDCompressor(){
    std::cout<<"Loading CAESAR-D"<<std::endl;
    std::cout<<"NOT YET ADDED IN USE MODEL V!"<<std::endl;
}


// CompressReturn compress(MyDataset& data, double errorBound){
// 
// }






