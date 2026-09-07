"""GX training from scratch: serial rank-zero input, CPU scatter, GPU DDP."""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import re

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pyCAESAR.data_io import read_array, scientific_layout

DATA_ROOT = Path('/lustre/blue2/ranka/shared-eklasky/GX/data')


def restart_files(directory):
    files = list(Path(directory).glob('gx.restart.nc.Time_*.bp'))

    def timestamp(path):
        match = re.fullmatch(r'gx\.restart\.nc\.Time_([0-9]+(?:\.[0-9]+)?)\.bp', path.name)
        if match is None:
            raise ValueError(f'Invalid restart filename: {path}')
        return float(match.group(1))

    files.sort(key=timestamp)
    if not files:
        raise ValueError(f'No gx.restart.nc.Time_*.bp files in {directory}')
    return [str(path.resolve()) for path in files]


def experiment_split(first, second):
    full = restart_files(first)
    alternate = restart_files(second)
    train, test = full + alternate[::2], alternate[1::2]
    if not test:
        raise ValueError('The second experiment needs at least two restart files')
    if set(train) & set(test) or len(set(train)) != len(train):
        raise ValueError('Experiments must be distinct with disjoint train/test files')
    return train, test


def scatter_file(path, group, seed=None):
    """Read one file on rank zero; distribute volume shards through Gloo.

    Two ranks divide the 256 GX volumes exactly. Other sample counts are padded
    for equal training steps; evaluation excludes padding using the valid count.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    message, chunks = [None], None
    if rank == 0:
        try:
            array = scientific_layout(read_array(path))
            volumes = np.ascontiguousarray(array.reshape(-1, *array.shape[-3:]), dtype=np.float32)
            if not len(volumes) or not np.isfinite(volumes).all():
                raise ValueError('Input is empty or contains NaN/Inf')
            data = torch.from_numpy(volumes)
            count = len(data)
            if seed is not None:
                order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
                data = data[order]
            per_rank = math.ceil(count / world)
            padding = per_rank * world - count
            if padding:
                data = torch.cat((data, data[torch.arange(padding) % count]))
            chunks = list(data.split(per_rank))
            message[0] = {'shape': list(chunks[0].shape), 'count': count}
        except Exception as exc:
            message[0] = {'error': f'{path}: {exc}'}
    dist.broadcast_object_list(message, src=0, group=group)
    metadata = message[0]
    if 'error' in metadata:
        raise RuntimeError(metadata['error'])
    shard = torch.empty(metadata['shape'], dtype=torch.float32)
    dist.scatter(shard, scatter_list=chunks, src=0, group=group)
    valid = max(0, min(len(shard), metadata['count'] - rank * len(shard)))
    return shard, valid


def normalized_batch(volumes, device):
    raw = volumes.unsqueeze(1).to(device)
    axes = (2, 3, 4)
    offset = raw.mean(dim=axes, keepdim=True)
    scale = raw.amax(dim=axes, keepdim=True) - raw.amin(dim=axes, keepdim=True)
    # Constant GX components remain valid samples without division by zero.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (raw - offset) / scale, raw, offset, scale


@torch.no_grad()
def evaluate(model, paths, group, device, batch_size, smoke_test=False):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    minimum = torch.tensor(float('inf'), device=device)
    maximum = torch.tensor(float('-inf'), device=device)
    for path in paths:
        shard, valid = scatter_file(path, group)
        if smoke_test:
            valid = min(valid, 2 * batch_size)
        for start in range(0, valid, batch_size):
            inputs, raw, offset, scale = normalized_batch(shard[start:min(start + batch_size, valid)], device)
            result = model(inputs)
            restored = result['output'] * scale + offset
            totals[0] += (restored - raw).double().square().sum()
            totals[1] += raw.numel()
            totals[2] += result['frame_bit'].double().sum()
            minimum = torch.minimum(minimum, raw.min())
            maximum = torch.maximum(maximum, raw.max())
    dist.all_reduce(totals)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    rmse = (totals[0] / totals[1]).sqrt().item()
    data_range = (maximum - minimum).item()
    bpp = (totals[2] / totals[1]).item()
    return {'rmse': rmse, 'nrmse': rmse / data_range if data_range else None,
            'bpp': bpp, 'compression_ratio': 32 / bpp if bpp > 0 else None}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-experiment', type=Path,
                        default=DATA_ROOT / 'exp-n125-fp0.7-0.7-tp3.0-3.0')
    parser.add_argument('--second-experiment', type=Path,
                        default=DATA_ROOT / 'exp-n125-fp1.0-1.0-tp1.0-5.0')
    parser.add_argument('--save-path', type=Path, default=Path('snapshots/gx-scratch'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1, help='Full 3D volumes per GPU')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--model-dim', type=int, default=16)
    parser.add_argument('--sr-dim', type=int, default=16)
    parser.add_argument('--init-beta', type=float, default=0.00001)
    parser.add_argument('--end-beta', type=float, default=0.00003)
    parser.add_argument('--beta-start', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-test', action='store_true',
                        help='One epoch, one train/test file, two batches per GPU each')
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.sr_dim < 1:
        parser.error('batch-size, epochs and sr-dim must be positive')
    if args.smoke_test:
        args.epochs = 1
    return args


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('GX training requires CUDA GPUs; launch with torchrun')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(hours=1))
    try:
        run(args, device)
    finally:
        dist.destroy_process_group()


def run(args, device):
    group = dist.new_group(backend='gloo', timeout=timedelta(hours=1))
    rank = dist.get_rank()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = [None]
    if rank == 0:
        try:
            train, test = experiment_split(args.first_experiment, args.second_experiment)
            if args.smoke_test:
                train, test = train[:1], test[:1]
            args.save_path.mkdir(parents=True, exist_ok=True)
            with (args.save_path / 'split.json').open('x') as stream:
                json.dump({'train': train, 'test': test, 'arguments': vars(args)},
                          stream, indent=2, default=str)
            manifest[0] = {'train': train, 'test': test}
        except Exception as exc:
            manifest[0] = {'error': str(exc)}
    dist.broadcast_object_list(manifest, src=0, group=group)
    if 'error' in manifest[0]:
        raise RuntimeError(manifest[0]['error'])
    train, test = manifest[0]['train'], manifest[0]['test']

    from pyCAESAR.models.compress_4_train_modules3d_mid_SR import CompressorMix

    base_model = CompressorMix(
        dim=args.model_dim, dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2], hyper_dims_mults=[4, 4, 4],
        channels=1, out_channels=1, d3=True, sr_dim=args.sr_dim,
    ).to(device)
    model = DistributedDataParallel(base_model, device_ids=[device.index],
                                    find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sorted(set(max(1, int(args.epochs * i / 5)) for i in range(1, 5))),
        gamma=0.5,
    )
    best = float('inf')
    if rank == 0:
        print(f'Training from scratch: {len(train)} train files, {len(test)} test files, '
              f'{dist.get_world_size()} GPUs, batch size {args.batch_size} per GPU', flush=True)
    for epoch in range(args.epochs):
        model.train()
        paths = train.copy()
        random.Random(args.seed + epoch).shuffle(paths)
        beta = args.init_beta if epoch < args.epochs * args.beta_start else args.end_beta
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        for file_index, path in enumerate(paths):
            shard, _ = scatter_file(path, group, seed=args.seed + epoch * len(paths) + file_index)
            if args.smoke_test:
                shard = shard[:2 * args.batch_size]
            for start in range(0, len(shard), args.batch_size):
                inputs, _, _, _ = normalized_batch(shard[start:start + args.batch_size], device)
                optimizer.zero_grad(set_to_none=True)
                result = model(inputs)
                mse = torch.nn.functional.mse_loss(result['output'], inputs)
                rate = result['bpp'].mean()
                loss = mse + beta * rate
                loss.backward()
                optimizer.step()
                totals[0] += mse.detach() * len(inputs)
                totals[1] += rate.detach() * len(inputs)
                totals[2] += len(inputs)
            if rank == 0:
                print(f'Epoch {epoch + 1}/{args.epochs}: train file {file_index + 1}/{len(paths)}', flush=True)
        dist.all_reduce(totals)
        scheduler.step()
        # Unwrapped evaluation permits uneven valid shard lengths without DDP collectives.
        metrics = evaluate(base_model, test, group, device, args.batch_size, args.smoke_test)
        if rank == 0:
            metrics.update(epoch=epoch + 1, train_mse=(totals[0] / totals[2]).item(),
                           train_bpp=(totals[1] / totals[2]).item(), beta=beta)
            with (args.save_path / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
            torch.save(base_model.state_dict(), args.save_path / 'model_final.pt')
            if metrics['rmse'] < best:
                best = metrics['rmse']
                torch.save(base_model.state_dict(), args.save_path / 'model_best.pt')
            print(json.dumps(metrics), flush=True)
        dist.barrier()


if __name__ == '__main__':
    main()
