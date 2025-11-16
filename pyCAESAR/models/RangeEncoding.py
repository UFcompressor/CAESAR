import math
import torch
import torch.nn as nn
import numpy as np

# import compressai.entropy_models.entropy_models as entropy_models_lib
# import compressai.entropy_models.entropy_models_vbr as entropy_models_vbr_lib

#from compressai.ops.bound_ops import LowerBoundFunction  # modify
class PatchedLowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.tensor([float(bound)]))
    
    # @torch.jit.unused
    # def lower_bound(self, x):
    #     return LowerBoundFunction.apply(x, self.bound)
    
    def forward(self, x):
        
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        
        return self.lower_bound(x)
        
from compressai.ops import bound_ops
bound_ops.LowerBound = PatchedLowerBound
# entropy_models_lib.LowerBound = PatchedLowerBound
# entropy_models_vbr_lib.LowerBound = PatchedLowerBound

from compressai.entropy_models import GaussianConditional

class PatchedGaussianConditional(GaussianConditional):
    def build_indexes(self, scales: torch.Tensor) -> torch.Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), self.scale_table.size(0) - 1).int()
        for s in self.scale_table[:-1]:
            indexes = indexes - (scales <= s).int()
        return indexes

from compressai.entropy_models.entropy_models_vbr import EntropyModelVbr

'''
def _build_indexes(size):
    dims = len(size)
    N = size[0]
    C = size[1]

    view_dims = np.ones((dims,), dtype=np.int64)
    view_dims[1] = -1
    indexes = torch.arange(C).view(*view_dims)
    indexes = indexes.int()

    return indexes.repeat(N, 1, *size[2:])

'''
def _build_indexes(size):
    dims = len(size)
    N = size[0]
    C = size[1]

    view_dims = [1, C] + [1] * (dims - 2)
    indexes = torch.arange(C).view(*view_dims)
    
    return indexes.expand(*size).int()


def _extend_ndims(tensor, n):
    return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    
#class RangeCoder:
class RangeCoder(nn.Module):
    def __init__(self, scale_bound=0.1, max_scale=20, scale_steps=128, 
                 _quantized_cdf = None, _cdf_length= None, _offset= None, medians = None,
                 device="cpu"):
        
        super().__init__()
        
        self.device = device
        self.gaussian = PatchedGaussianConditional(None, scale_bound=scale_bound)
        #self.gaussian.lower_bound_scale = PatchedLowerBound(scale_bound) # modify
        
        lower = self.gaussian.lower_bound_scale.bound.item()
        scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=scale_steps))
        self.gaussian.update_scale_table(scale_table)
        self.gaussian.update()
        
        self.entropy_model = EntropyModelVbr()
        
        self.entropy_model._quantized_cdf = _quantized_cdf
        self.entropy_model._cdf_length = _cdf_length
        self.entropy_model._offset = _offset
        
        #self.medians = medians
        self.register_buffer("medians", medians)
        

    def compress(self, latent, mean, scale):
        latent, mean, scale = latent.to(self.device), mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        strings = self.gaussian.compress(latent, indexes, means=mean)
        return strings

    def compress_return_para(self, latent, mean, scale):
        latent, mean, scale = latent.to(self.device), mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        q_latent = self.gaussian.quantize(latent, "symbols", mean)
        return q_latent, indexes

    def compress_return_index(self, scale):
        scale = scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        return indexes
    
    def compress_hyperlatent(self, latent, qs= None):
        indexes = _build_indexes(latent.size())
        spatial_dims = len(latent.size()) - 2
        medians = _extend_ndims(self.medians, spatial_dims)
        medians = medians.expand(latent.size(0), *([-1] * (spatial_dims + 1)))
        strings = self.entropy_model.compress(latent, indexes, medians, qs)
        return strings

    def compress_hyperlatent_return_para(self, latent, qs= None):
        indexes = _build_indexes(latent.size())
        spatial_dims = len(latent.size()) - 2
        medians = _extend_ndims(self.medians, spatial_dims)
        medians = medians.expand(latent.size(0), *([-1] * (spatial_dims + 1)))
        latent, indexes, medians = latent.to(self.device), indexes.to(self.device), medians.to(self.device)
        #print('latent indexes medians devices: ', latent.device, indexes.device, medians.device)
        q_latent = self.entropy_model.quantize(latent, "symbols", medians)
        #return q_latent, indexes, medians
        return q_latent, indexes
    
    def decompress_hyperlatent(self, strings, size, qs= None):
        size = size[-2:]
        output_size = (len(strings), self.entropy_model._quantized_cdf.size(0), *size[-2:])
        indexes = _build_indexes(output_size).to(self.entropy_model._quantized_cdf.device)
        medians = _extend_ndims(self.medians, len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return self.entropy_model.decompress(strings, indexes, medians.dtype, medians, qs)

    def decompress_hyperlatent_return_para(self, strings, size, qs= None):
        size = size[-2:]
        output_size = (len(strings), self.entropy_model._quantized_cdf.size(0), *size[-2:])
        indexes = _build_indexes(output_size).to(self.entropy_model._quantized_cdf.device)
        medians = _extend_ndims(self.medians, len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return indexes, medians
    

    def decompress(self, strings, mean, scale):
        mean, scale = mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        decoded_latent = self.gaussian.decompress(strings, indexes, means=mean)
        return decoded_latent
    
