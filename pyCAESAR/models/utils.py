import torch
from inspect import isfunction
from torch.autograd import Function
import numpy as np
import os
import json
import yaml


def build_dataset(dataset_args, syn_length=False):
    """Build scientific datasets from the mappings produced by ``convert_args``."""
    # Imported lazily because this module is also used by the model components.
    from pyCAESAR.dataset import ScientificDataset

    datasets = [ScientificDataset(dataset_args[name]) for name in dataset_args]
    if syn_length:
        max_length = np.max([len(dataset) for dataset in datasets])
        for dataset in datasets:
            dataset.visble_length = max_length
    return datasets


def convert_args(args, train=True):
    """Read the original CAESAR dataset configuration format."""
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    dataset_names = (args.train_set if train else args.test_set).split(",")
    converted = {}
    for dataset_name in dataset_names:
        dataset_name = dataset_name.strip()
        dataset_config = config[dataset_name]
        options = {"name": dataset_name, "data_path": dataset_config["data_path"]}
        options.update(dataset_config["train_subset" if train else "test_subset"])
        options.update(config["train_config" if train else "test_config"])
        converted[dataset_name] = options
    return converted


def save_json(json_pth, data, mode="update"):
    if os.path.exists(json_pth):
        with open(json_pth, "r") as json_file:
            existing_data = json.load(json_file)
        if mode == "update":
            existing_data.update(data)
            data = existing_data
        elif mode == "cat":
            for key in data:
                if key in existing_data:
                    existing_data[key] += data[key]
                else:
                    existing_data[key] = data[key]
            data = existing_data
    with open(json_pth, "w") as json_file:
        json.dump(data, json_file, indent=4)


import torch


def relative_rmse_error_ornl(original, reconstructed, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original = torch.as_tensor(original, dtype=torch.float64, device=device)
    reconstructed = torch.as_tensor(reconstructed, dtype=torch.float64, device=device)

    rmse = torch.sqrt(torch.mean((original - reconstructed) ** 2))
    data_range = torch.max(original) - torch.min(original)

    relative_rmse = torch.where(
        data_range != 0, rmse / data_range, torch.tensor(0.0, device=device)
    )
    return relative_rmse


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_tensor(a, t, place_holder=None):
    return a[t, torch.arange(len(t))]


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def noise(input, scale):
    return input + scale * (torch.rand_like(input) - 0.5)


def round_w_offset(input, loc):
    diff = STERound.apply(input - loc)
    return diff + loc


def quantize(x, mode="noise", offset=None):
    if mode == "noise":
        return noise(x, 1)
    elif mode == "round":
        return STERound.apply(x)
    elif mode == "dequantize":
        return round_w_offset(x, offset)
    else:
        raise NotImplementedError


class STERound(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class UpperBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.min(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs <= b
        pass_through_2 = grad_output > 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class NormalDistribution:
    """
    A normal distribution
    """

    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc.detach()

    def std_cdf(self, inputs):
        half = 0.5
        const = -(2**-0.5)
        return half * torch.erfc(const * inputs)

    def sample(self):
        return self.scale * torch.randn_like(self.scale) + self.loc

    def likelihood(self, x, min=1e-9):
        x = torch.abs(x - self.loc)
        upper = self.std_cdf((0.5 - x) / self.scale)
        lower = self.std_cdf((-0.5 - x) / self.scale)
        return LowerBound.apply(upper - lower, min)

    def scaled_likelihood(self, x, s=1, min=1e-9):
        x = torch.abs(x - self.loc)
        s = s * 0.5
        upper = self.std_cdf((s - x) / self.scale)
        lower = self.std_cdf((-s - x) / self.scale)
        return LowerBound.apply(upper - lower, min)
