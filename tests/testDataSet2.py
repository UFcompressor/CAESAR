import torch
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description="Save random tensors to binary files.")
parser.add_argument("output_dir", type=str, help="Path to the directory where .bin files will be saved")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

def save_tensor_binary(tensor, filename):
    contig = tensor.contiguous()
    shape = np.array(tensor.shape, dtype=np.int64)  
    with open(filename, "wb") as f:
        f.write(shape.tobytes())                   
        f.write(contig.numpy().astype(np.float32).tobytes())      
    print(f"Saved tensor with shape {tuple(tensor.shape)} to {filename}")


shape = (2, 12, 256, 256, 256)

for i in range(1, 4):
    preloaded_data = torch.randn(shape, dtype=torch.float64)
    filename = os.path.join(output_dir, f"tensor_data_{i}.bin")
    save_tensor_binary(preloaded_data, filename)

