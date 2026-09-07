"""Single-process checkpoint evaluation: ADIOS/NPZ -> model -> CR and NRMSE."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pyCAESAR.data_io import read_array, scientific_layout
from pyCAESAR.train_gx import normalized_batch, restart_files


def input_files(paths):
    files = []
    for value in paths:
        path = Path(value)
        if path.is_dir() and path.suffix != '.bp':
            files.extend(restart_files(path))
        elif path.exists():
            files.append(str(path.resolve()))
        else:
            raise FileNotFoundError(path)
    return list(dict.fromkeys(files))


@torch.inference_mode()
def evaluate(model, files, device, batch_size):
    # Global SSE, values, estimated bits, sum of per-volume NRMSE, valid volumes.
    sums = torch.zeros(5, dtype=torch.float64, device=device)
    minimum = torch.tensor(float('inf'), device=device)
    maximum = torch.tensor(float('-inf'), device=device)
    worst = torch.zeros((), dtype=torch.float64, device=device)
    volume_count = 0
    for file_index, path in enumerate(files):
        array = scientific_layout(read_array(path))
        volumes = torch.from_numpy(np.ascontiguousarray(array)).to(device, dtype=torch.float32)
        volumes = volumes.reshape(-1, *volumes.shape[-3:])
        if not volumes.numel() or not torch.isfinite(volumes).all().item():
            raise ValueError(f'Empty or nonfinite input: {path}')
        volume_count += len(volumes)
        for start in range(0, len(volumes), batch_size):
            inputs, raw, offset, scale = normalized_batch(volumes[start:start + batch_size], device)
            result = model(inputs)
            if result['output'].shape != inputs.shape:
                raise RuntimeError('Model reconstruction shape differs from input')
            restored = result['output'] * scale + offset
            error = (restored - raw).double().square()
            sums[0] += error.sum()
            sums[1] += raw.numel()
            sums[2] += result['frame_bit'].double().sum()
            minimum = torch.minimum(minimum, raw.min())
            maximum = torch.maximum(maximum, raw.max())
            ranges = raw.flatten(1).amax(1) - raw.flatten(1).amin(1)
            valid = ranges > 0
            per_volume = error.flatten(1).mean(1).sqrt() / ranges.clamp_min(torch.finfo(raw.dtype).tiny)
            scores = torch.where(valid, per_volume, torch.zeros_like(per_volume))
            sums[3] += scores.sum()
            sums[4] += valid.sum()
            worst = torch.maximum(worst, scores.max())
        del volumes, array
        print(f'Evaluated file {file_index + 1}/{len(files)}: {path}', flush=True)
    sse, values, bits, score_sum, valid_count = sums.tolist()
    rmse = (sse / values) ** 0.5
    data_range = (maximum - minimum).item()
    return {
        'files': len(files), 'volumes': volume_count,
        'rmse': rmse,
        'nrmse': rmse / data_range if data_range else None,
        'mean_volume_nrmse': score_sum / valid_count if valid_count else None,
        'worst_volume_nrmse': worst.item() if valid_count else None,
        'constant_volumes_excluded_from_volume_nrmse': volume_count - int(valid_count),
        'bpp': bits / values,
        'compression_ratio': 32 * values / bits if bits > 0 else None,
        'cr_is_entropy_estimate': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', nargs='+', required=True,
                        help='Restart BP/NPZ paths or experiment directories (all restarts)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-dim', type=int, default=None)
    parser.add_argument('--sr-dim', type=int, default=None)
    parser.add_argument('--output', type=Path, help='Optional metrics JSON file')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    files = input_files(args.data)
    config = {}
    manifest = args.checkpoint.parent / 'split.json'
    if manifest.exists():
        config = json.loads(manifest.read_text()).get('arguments', {})
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    from pyCAESAR.models.compress_4_train_modules3d_mid_SR import CompressorMix

    model = CompressorMix(
        dim=args.model_dim if args.model_dim is not None else config.get('model_dim', 16),
        dim_mults=[1, 2, 3, 4], reverse_dim_mults=[4, 3, 2],
        hyper_dims_mults=[4, 4, 4], channels=1, out_channels=1, d3=True,
        sr_dim=args.sr_dim if args.sr_dim is not None else config.get('sr_dim', 16),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.to(device).eval()
    metrics = evaluate(model, files, device, args.batch_size)
    metrics.update(checkpoint=str(args.checkpoint), inputs=files)
    report = json.dumps(metrics, indent=2, allow_nan=False)
    print(report)
    if args.output:
        args.output.write_text(report + '\n')


if __name__ == '__main__':
    main()
