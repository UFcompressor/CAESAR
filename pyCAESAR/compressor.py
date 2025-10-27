import torch
from collections import OrderedDict
from .models.run_gae_cuda import PCACompressor
import math
import torch.nn.functional as F
import numpy as np
import time
def normalize_latent(x):
    x_min = torch.amin(x, dim=(1, 2, 3, 4), keepdim=True)
    x_max = torch.amax(x, dim=(1, 2, 3, 4), keepdim=True)

    scale = (x_max - x_min + 1e-8) / 2
    offset = x_min + scale

    x_norm = (x - offset) / scale  # result in [-1, 1]
    return x_norm, offset, scale


class CAESAR:
    def __init__(self, 
                 model_path, 
                 use_diffusion=True, 
                 device='cuda',  n_frame = 16, interpo_rate=3, diffusion_steps = 32
                 ):
        self.pretrained_path = model_path
        self.use_diffusion = use_diffusion
        self.device = device
        self.n_frame = n_frame
        self.diffusion_steps = diffusion_steps

        self._load_models()
        
        self.interpo_rate = interpo_rate
        self.cond_idx = torch.arange(0,n_frame,interpo_rate)
        self.pred_idx = ~torch.isin(torch.arange(n_frame), self.cond_idx)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    def remove_module_prefix(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    def _load_models(self):
        if not self.use_diffusion:
            self._load_caesar_v_compressor()
        else:
            self._load_caesar_d_compressor()

    def _load_caesar_v_compressor(self):
        from .models import compress_modules3d_mid_SR as compress_modules
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

        state_dict = self.remove_module_prefix(torch.load(self.pretrained_path, map_location=self.device))
        model.load_state_dict(state_dict)
        self.compressor_v = model.to(self.device).eval()

    def _load_caesar_d_compressor(self):
        print("Loading CAESAE-D")
        from .models import keyframe_compressor as compress_modules
        
        pretrained_models = torch.load(self.pretrained_path, map_location=self.device)
       
        model = compress_modules.ResnetCompressor(
            dim=16,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2, 1],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1
        )

        state_dict = self.remove_module_prefix(pretrained_models["vae"])
        model.load_state_dict(state_dict)
        self.keyframe_model = model.to(self.device).eval()
        
        
        from .models.video_diffusion_interpo import Unet3D, GaussianDiffusion
        model = Unet3D(
            dim=64,
            out_dim=64,
            channels=64,
            dim_mults=(1, 2, 4, 8),
            use_bert_text_cond=False
        )

        diffusion = GaussianDiffusion(
            model,
            image_size=16,
            num_frames=10,
            channels=64,
            timesteps=self.diffusion_steps,
            loss_type='l2'
        )
        
        state_dict = self.remove_module_prefix(pretrained_models["diffusion"])
        diffusion.load_state_dict(state_dict)

        self.diffusion_model = diffusion.to(self.device).eval()
        
        
    def forward(self, dataloader):
        
        #dataset_org = dataloader.dataset
        #self.transform_shape = dataset_org.deblocking_hw
        
        compressed_latent, latent_bytes = self.compress_caesar_v(dataloader)
  
        #original_data = dataset_org.original_data()
        #print("original_data.shape after compress", original_data.shape, recons_data.shape)
        #original_data, org_padding = self.padding(original_data)
        #recons_data, rec_padding= self.padding(recons_data)
        
        #meta_data, compressed_gae = self.postprocessing_encoding(original_data, recons_data, eb)
        return compressed_latent
    

    def compress_caesar_v(self, dataloader):

        total_bits = 0
        all_compressed_latent = []

        with torch.no_grad():
            for data in dataloader:
                outputs = self.compressor_v.compress(data[0].to(self.device))
                total_bits += torch.sum(outputs["bpf_real"])
                
                compressed_latent = outputs["compressed"]
                all_compressed_latent.append(compressed_latent)
                
        return all_compressed_latent, total_bits/8
                
                
    def padding(self, data, block_size=(8, 8)):
        *leading_dims, H, W = data.shape
        h_block, w_block = block_size

        H_target = math.ceil(H / h_block) * h_block
        W_target = math.ceil(W / w_block) * w_block
        dh = H_target - H
        dw = W_target - W
        top, down = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        data_reshaped = data.view(-1, H, W)
        data_padded = F.pad(data_reshaped, (left, right, top, down), mode='reflect')
        padded_data = data_padded.view(*leading_dims, *data_padded.shape[-2:])
        padding = (top, down, left, right)
        return padded_data, padding
    
    def unpadding(self, padded_data, padding):
        top, down, left, right = padding
        *leading_dims, H, W = padded_data.shape
        unpadded_data = padded_data[..., top:H-down, left:W-right]
        return unpadded_data
    