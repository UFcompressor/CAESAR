import torch
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Save synthetic tensors to binary files.")
parser.add_argument("output_dir", type=str, help="Path to the directory where .bin files will be saved")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

def save_tensor_binary(tensor, filename):
    contig = tensor.contiguous()
    shape = np.array(tensor.shape, dtype=np.int64)
    with open(filename, "wb") as f:
        f.write(shape.tobytes())
        f.write(contig.numpy().astype(np.float64).tobytes())
    print(f"Saved tensor with shape {tuple(tensor.shape)} to {filename}")


# Shape: (batch, channels, time, height, width)
shape = (2, 12, 256, 256, 256)

# Create grid for spatial coordinates
x = torch.linspace(0, 1+ 2 * np.pi, shape[3])
y = torch.linspace(0, 1- 2 * np.pi, shape[4])
t = torch.linspace(0,2 - 4 * np.pi, shape[2])  # time dimension
X, Y, T = torch.meshgrid(x, y, t, indexing="ij")  # shape (H, W, T)

# Example synthetic "scientific" fields
# Standing wave (H, W, T)
wave = torch.sin(X) * torch.cos(Y)

# Propagating wave (H, W, T)
prop_wave = torch.sin(X - T)

# Permute to (T, H, W) to match dataset convention
wave = wave.permute(2, 0, 1).contiguous()
prop_wave = prop_wave.permute(2, 0, 1).contiguous()

# Fill tensor (batch, channels, time, height, width)
data = torch.zeros(shape, dtype=torch.float64)
for c in range(shape[1]):
    if c % 2 == 0:
        data[0, c] = wave
    else:
        data[0, c] = prop_wave

# Second batch is just a scaled version
data[1] = data[0] * 0.5

# Save a few samples
for i in range(1, 4):
    filename = os.path.join(output_dir, f"tensor_data_{i}.bin")
    save_tensor_binary(data, filename)

