import torch
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Generate dummy test data for CAESAR compression testing.")
parser.add_argument("output_dir", type=str, help="Path to the directory where the binary file will be saved")
parser.add_argument("--filename", type=str, default="TCf48.bin.f32", help="Output filename (default: TCf48.bin.f32)")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


shape = (1, 1, 100, 500, 500)

print(f"Generating dummy data with shape: {shape}")


x = torch.linspace(0, 2 * np.pi, shape[4])  
y = torch.linspace(0, 2 * np.pi, shape[3])  
t = torch.linspace(0, 4 * np.pi, shape[2])  


X, Y, T = torch.meshgrid(y, x, t, indexing="ij")  

data = torch.sin(X) * torch.cos(Y) * torch.cos(T * 0.1)
data = data + 0.3 * torch.sin(2 * X + T * 0.05)
data = data.permute(2, 0, 1).contiguous()  
data = data.unsqueeze(0).unsqueeze(0)  
data = data.to(torch.float32)

output_path = os.path.join(output_dir, args.filename)
with open(output_path, "wb") as f:
    f.write(data.numpy().tobytes())

print(f"  Saved {data.numel()} float32 values to {output_path}")
print(f"  Shape: {tuple(data.shape)}")
print(f"  Data range: [{data.min().item():.4f}, {data.max().item():.4f}]")
print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")