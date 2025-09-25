import torch
import numpy as np
import os

output_dir = "/lustre/blue/ranka/eklasky/CAESAR_C/build/tests"
os.makedirs(output_dir, exist_ok=True)

def save_tensor_binary(tensor, filename):
    contig = tensor.contiguous()
    shape = np.array(tensor.shape, dtype=np.int64)  # 5D shape as int64
    with open(filename, "wb") as f:
        f.write(shape.tobytes())                                   # write header
        f.write(contig.numpy().astype(np.float32).tobytes())       # write data
    print(f"Saved tensor with shape {tuple(tensor.shape)} to {filename}")

# Use bigger H, W so reflection pad works (>= 256)
shape = (2, 10, 100, 256, 256)

for i in range(1, 4):
    preloaded_data = torch.randn(shape, dtype=torch.float32)
    filename = os.path.join(output_dir, f"tensor_data_{i}.bin")
    save_tensor_binary(preloaded_data, filename)

