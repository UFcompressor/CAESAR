"""Training-only CAESAR compressor model.

NOTE: Training and AOTI inference compilation are intentionally kept separate.
This file contains the original trainable network. Do not use or modify
``compress_modules3d_mid_SR.py`` for training; that file belongs to the
three-part AOTI inference/compilation path.
"""

import torch.nn as nn
from .network_components_4_train import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution
import time
import yaml
from .BCRN.bcrn_model import BluePrintConvNeXt_SR
import torch
import torch.nn.init as init


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def super_resolution_model(
    img_size=64, in_chans=32, out_chans=1, sr_dim="HAT", pretrain=False, sr_type="BCRN"
):

    if sr_type == "HAT":
        yaml_file = load_yaml("./configs/HAT-S_SRx4.yml")
        network_g = yaml_file["network_g"]
        network_g["img_size"] = img_size
        network_g["in_chans"] = in_chans
        network_g["out_chans"] = out_chans
        sr_model = HAT(**network_g)
        loaded_params, not_loaded_params = sr_model.load_part_model(
            "./pretrain/HAT-S_SRx4.pth"
        )
        print("Loading HAT model")
        return sr_model, loaded_params, not_loaded_params

    if sr_type == "BCRN":
        sr_model = BluePrintConvNeXt_SR(in_chans, 1, 4, sr_dim)
        if pretrain:
            loaded_params, not_loaded_params = sr_model.load_part_model(
                "./pretrain/BCRN_SRx4.pth"
            )
        else:
            loaded_params, not_loaded_params = [], sr_model.parameters()
        print("Loading BCRN model")

        return sr_model, loaded_params, not_loaded_params


def reshape_batch_2d_3d(batch_data, batch_size):
    BT, C, H, W = batch_data.shape
    T = BT // batch_size
    batch_data = batch_data.view([batch_size, T, C, H, W])
    batch_data = batch_data.permute([0, 2, 1, 3, 4])
    return batch_data


def reshape_batch_3d_2d(batch_data):
    B, C, T, H, W = batch_data.shape
    batch_data = batch_data.permute([0, 2, 1, 3, 4]).reshape([B * T, C, H, W])
    return batch_data


class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        d3=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))

        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(
            zip(self.reversed_dims[:-1], self.reversed_dims[1:])
        )

        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1])

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, input):

        self.t_dim = input.shape[2]

        for i, (resnet, down) in enumerate(self.enc):  # [b, 1, t, 256, 256]
            if i == 0:
                input = input.permute(0, 2, 1, 3, 4)
                input = input.reshape(-1, *input.shape[2:])  # [b*t, 1, 256, 256]
            if i == 2:
                input = input.reshape(-1, self.t_dim, *input.shape[1:])
                input = input.permute(0, 2, 1, 3, 4)  # [b, c, t, h, w]

            input = resnet(input)
            input = down(input)

        input = input.permute(0, 2, 1, 3, 4)
        input = input.reshape(-1, *input.shape[2:])

        latent = input
        for i, (conv, act) in enumerate(self.hyper_enc):
            input = conv(input)
            input = act(input)

        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp, mean

    def decode(self, input):  # [n*t, c, h,w ] [8, 256, 16, 16]
        # output = []

        for i, (resnet, up) in enumerate(self.dec):

            if i == 0:
                input = input.reshape(-1, self.t_dim // 4, *input.shape[1:])
                input = input.permute(0, 2, 1, 3, 4)  # [b, c, t, h, w]

            if i == 2:
                input = input.permute(0, 2, 1, 3, 4)
                input = input.reshape(-1, *input.shape[2:])  # [b*t, 1, 256, 256]

            input = resnet(input)
            input = up(input)

        return input

    def bpp(self, shape, state4bpp):
        B, H, W = shape[0], shape[-2], shape[-1]
        n_pixels = shape[-3] * shape[-2] * shape[-1]

        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()

        frame_bit = hyper_rate.reshape(B, -1).sum(dim=-1) + cond_rate.reshape(
            B, -1
        ).sum(dim=-1)
        bpp = (
            hyper_rate.reshape(B, -1).sum(dim=-1) + cond_rate.reshape(B, -1).sum(dim=-1)
        ) / n_pixels

        return frame_bit, bpp

    def forward(self, input, return_time=False):

        result = {}

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        q_latent, q_hyper_latent, state4bpp, mean = self.encode(input)

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            result["encoding_time"] = time.time() - start_time

        frame_bit, bpp = self.bpp(input.shape, state4bpp)

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        output = self.decode(q_latent)

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            result["decoding_time"] = time.time() - start_time

        result.update(
            {
                "output": output,
                "bpp": bpp,
                "frame_bit": frame_bit,
                "mean": mean,
                "q_latent": q_latent,
                "q_hyper_latent": q_hyper_latent,
            }
        )

        return result


class ResnetCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        d3=False,
    ):
        super().__init__(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels,
            d3,
        )
        self.d3 = d3
        self.conv_layer = nn.Conv3d if d3 else nn.Conv2d
        self.deconv_layer = nn.ConvTranspose3d if d3 else nn.ConvTranspose2d

        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            d3 = self.d3 if ind >= 2 else False
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in, dim_out, None, True if ind == 0 else False, d3=d3
                        ),
                        Downsample(dim_out, d3=d3),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            d3 = self.d3 if ind < 2 else False

            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in, d3=d3),
                        (
                            Upsample(dim_out if not is_last else dim_in, dim_out, d3=d3)
                            if d3
                            else nn.Identity()
                        ),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        (
                            nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                            if ind == 0
                            else nn.Conv2d(dim_in, dim_out, 5, 2, 2)
                        ),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        (
                            nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                            if is_last
                            else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1)
                        ),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )


#         for module in self.modules():
#             if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
#                 init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#                 if module.bias is not None:
#                     init.zeros_(module.bias)

#             elif isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     init.zeros_(module.bias)


class CompressorMix(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        d3=False,
        sr_dim=16,
    ):
        super().__init__()  # Initialize the nn.Module parent class

        self.entropy_model = ResnetCompressor(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels,
            d3,
        )

        # Update channels for sr_model based on entropy_model's output
        channels = dim * reverse_dim_mults[-1]

        # Initialize super-resolution model
        self.sr_model, self.loaded_params, self.not_loaded_params = (
            super_resolution_model(
                img_size=64,
                in_chans=channels,
                out_chans=out_channels,
                sr_type="BCRN",
                sr_dim=sr_dim,
            )
        )

    def forward(self, inputs, return_time=False):
        B = inputs.shape[0]

        results = self.entropy_model(inputs, return_time)
        outputs = results["output"]

        # Apply super-resolution model
        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        outputs = self.sr_model(outputs)  # Use self.sr_model instead of sr_model

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            results["decoding_time"] += time.time() - start_time

        # Reshape if needed
        outputs = reshape_batch_2d_3d(outputs, B)
        results["output"] = outputs

        return results
