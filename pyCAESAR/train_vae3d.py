"""Train CAESAR with the training-only model implementation.

NOTE: Training and AOTI inference compilation are separate paths. This script
uses the ``*_4_train`` model files and does not use or modify the three-part
AOTI inference model.
"""

import os
from contextlib import ExitStack, nullcontext
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import math
import shutil
import time
from collections import OrderedDict
from itertools import islice
from pyCAESAR.models.utils import (
    build_dataset,
    convert_args,
    relative_rmse_error_ornl,
    save_json,
)


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict


def check_finite(name, tensor, iteration, data_dict):
    """Stop at the first nonfinite tensor and report the affected batch."""
    with torch.no_grad():
        values = tensor.detach()
        finite_mask = torch.isfinite(values)
        if finite_mask.all().item():
            return
        finite = values[finite_mask]
        finite_range = (
            (finite.min().item(), finite.max().item()) if finite.numel() else None
        )
        print(
            f"[NONFINITE] iteration={iteration}, tensor={name}, "
            f"shape={tuple(values.shape)}, dtype={values.dtype}, "
            f"device={values.device}, finite_range={finite_range}, "
            f"nan={torch.isnan(values).sum().item()}, "
            f"inf={torch.isinf(values).sum().item()}",
            flush=True,
        )
        print("Dataset indices:", data_dict["index"], flush=True)
        print("Normalization offsets:", data_dict["offset"], flush=True)
        print("Normalization scales:", data_dict["scale"], flush=True)
        raise RuntimeError(f"Nonfinite values in {name} at iteration {iteration}")


def debug_backward_failure(
    model,
    inputs,
    data_dict,
    criterion,
    loss_beta,
    iteration,
    rng_state,
    cuda_rng_state,
    buffers,
):
    """Save a reproducible batch and isolate each loss without updating weights."""
    path = os.path.join(args.save_path, f"debug_iteration_{iteration}.pt")
    torch.save(
        {
            "iteration": iteration,
            "model": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "buffers_before_forward": buffers,
            "batch": {
                name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for name, value in data_dict.items()
            },
            "rng_state": rng_state,
            "cuda_rng_state": cuda_rng_state,
            "loss_beta": loss_beta,
            "arguments": vars(args),
        },
        path,
    )
    print(f"[DEBUG] Saved failing batch and pre-update weights: {path}", flush=True)
    for branch in ("MSE", "entropy"):
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                result = model(inputs)
                branch_loss = (
                    criterion(result["output"], inputs)
                    if branch == "MSE"
                    else result["bpp"].mean() * loss_beta
                )
                check_finite(f"{branch} replay loss", branch_loss, iteration, data_dict)
                branch_loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        check_finite(
                            f"{branch} replay gradient: {name}",
                            parameter.grad,
                            iteration,
                            data_dict,
                        )
            print(f"[DEBUG] {branch} backward: finite", flush=True)
        except RuntimeError as error:
            print(f"[DEBUG] {branch} backward FAILED: {error}", flush=True)
        finally:
            model.zero_grad(set_to_none=True)


def train_epoch_vae(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    loss_beta,
    device,
    iteration=0,
    debug=False,
    max_iterations=None,
    norm_type=None,
):
    """Train the model for one epoch."""
    model.train()
    running_loss1 = torch.zeros((), device=device)
    running_loss2 = torch.zeros((), device=device)

    sample_count = 0

    def check(name, tensor, current_iteration, batch):
        if debug:
            check_finite(name, tensor, current_iteration, batch)

    replay = debug and not isinstance(model, DistributedDataParallel)
    for data_dict in loader:
        if max_iterations is not None and iteration >= max_iterations:
            break
        t0 = time.time()
        # inputs = data_dict["input"].to(device, non_blocking=True)
        inputs = (
            data_dict["input"]
            .to(device, non_blocking=True)
            .to(memory_format=torch.channels_last_3d)
        )
        if norm_type is not None:
            from pyCAESAR.training_data import normalize_batch

            inputs, offset, scale = normalize_batch(inputs, norm_type)
            data_dict = {**data_dict, "offset": offset, "scale": scale}
        targets = inputs

        check("input", inputs, iteration + 1, data_dict)
        check("normalization offset", data_dict["offset"], iteration + 1, data_dict)
        check("normalization scale", data_dict["scale"], iteration + 1, data_dict)

        optimizer.zero_grad(set_to_none=True)

        if replay:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all()
            buffers = {
                name: buffer.detach().cpu().clone()
                for name, buffer in model.named_buffers()
            }
        try:
            with (
                torch.autograd.detect_anomaly(check_nan=True)
                if debug
                else nullcontext()
            ):
                results = model(inputs)
                outputs = results["output"]
                check("model output", outputs, iteration + 1, data_dict)
                check("bpp", results["bpp"], iteration + 1, data_dict)

                loss_mse = criterion(outputs, targets)
                loss_bpp = results["bpp"].mean() * loss_beta
                loss = (loss_mse + loss_bpp) if loss_beta > 0.0 else loss_mse

                check("MSE loss", loss_mse, iteration + 1, data_dict)
                check("entropy loss", loss_bpp, iteration + 1, data_dict)

                loss.backward()
                for name, parameter in (model.named_parameters() if debug else ()):
                    if parameter.grad is not None:
                        check(
                            f"gradient: {name}",
                            parameter.grad,
                            iteration + 1,
                            data_dict,
                        )
        except RuntimeError:
            if replay:
                try:
                    debug_backward_failure(
                        model,
                        inputs,
                        data_dict,
                        criterion,
                        loss_beta,
                        iteration + 1,
                        rng_state,
                        cuda_rng_state,
                        buffers,
                    )
                except Exception as debug_error:
                    print(
                        f"[DEBUG] Failure diagnostics could not complete: {debug_error}",
                        flush=True,
                    )
            raise
        optimizer.step()
        for name, parameter in (model.named_parameters() if debug else ()):
            check(
                f"parameter after update: {name}", parameter, iteration + 1, data_dict
            )
        scheduler.step()

        running_loss1 += loss_mse.detach() * inputs.size(0)
        running_loss2 += loss_bpp.detach() * inputs.size(0)

        sample_count += inputs.size(0)
        iteration += 1
        if debug and iteration < 30:
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"[TIMING] iter {iteration}: {time.time() - t0:.3f}s", flush=True)

    epoch_loss1 = (running_loss1 / sample_count).item()
    epoch_loss2 = (running_loss2 / sample_count).item()

    return epoch_loss1, epoch_loss2, iteration


def test_epoch_vae(model, loader, criterion, device):
    """Test the model and compute the reconstruction results."""
    model.eval()
    all_data = []

    recons_data = torch.zeros_like(loader.dataset.data_input)

    bit_count = 0

    with torch.no_grad():
        for data_dict in loader:
            inputs = data_dict["input"].to(device)

            results = model(inputs)
            outputs = results["output"].detach() * data_dict["scale"].to(
                device
            ) + data_dict["offset"].to(device)
            bit_count += torch.sum(results["frame_bit"].detach()).item()

            idx0, idx1, start_t, end_t = data_dict["index"]
            idx0, idx1, start_t, end_t = (
                idx0.tolist(),
                idx1.tolist(),
                start_t.tolist(),
                end_t.tolist(),
            )
            for i in range(len(inputs)):
                recons_data[idx0[i], idx1[i], start_t[i] : end_t[i]] = outputs[i]

    return recons_data, bit_count


class Info:
    def __init__(self, data_name, bpp=32, model_path=None, json_path=None):
        self.json_path = json_path
        self.model_path = model_path

        self.data_name = data_name
        self.bpp = bpp
        self.best_nrmse = 1e10
        self.best_nrmse_cr = 0
        self.best_epoch = -1
        self.all_eval_nrmse = []
        self.all_eval_bpp = []
        self.all_eval_cr = []

    def save_json(self):
        save_json(
            self.json_path,
            {
                self.data_name: {
                    "NRMSE": self.all_eval_nrmse,
                    "best_nrmse": self.best_nrmse,
                    "best_nrmse_cr": self.best_nrmse_cr,
                    "best_index": self.best_epoch,
                    "bpp": self.all_eval_bpp,
                    "cr": self.all_eval_cr,
                }
            },
        )

    def save_last_model(self, model):
        torch.save(model.state_dict(), self.model_path.replace(".pt", "_final.pt"))

    def update(self, model, epoch, nrmse, bpp, dname):
        assert self.data_name == dname
        self.all_eval_nrmse.append(nrmse)
        self.all_eval_bpp.append(bpp)
        self.all_eval_cr.append(self.bpp / bpp)

        if nrmse <= self.best_nrmse:
            torch.save(model.state_dict(), self.model_path)
            self.best_nrmse = nrmse
            self.best_nrmse_cr = self.bpp / bpp
            self.best_epoch = epoch

        self.save_json()


def get_argument():
    parser = argparse.ArgumentParser(
        description="Train a UNet with Channel Attention model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./snapshots/E3SM/E3SM_VAE",
        help="Path to save model and results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=400,
        help="Thousands of rank-iterations, rounded down to whole synchronized steps",
    )

    parser.add_argument(
        "--plan_only",
        action="store_true",
        help="Inspect distributed shards without loading payloads",
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=0,
        help="Distributed validation interval in steps (0 validates at completion)",
    )
    parser.add_argument(
        "--loader_workers", type=int, help="Training loader workers per rank"
    )

    parser.add_argument(
        "--sr_dim", type=int, default=16, help="Number of epochs for training"
    )

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_gamma", type=float, default=0.5, help="Learning rate gamma"
    )

    parser.add_argument("--init_beta", type=float, default=1e-5, help="loss beta")
    parser.add_argument("--end_beta", type=float, default=2e-5, help="loss beta")

    parser.add_argument("--beta_start", type=float, default=0.5, help="loss beta")
    parser.add_argument("--model_dim", type=int, default=16, help="loss beta")
    parser.add_argument("--pretrain", type=str, default="", help="pretrain path")

    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable finite checks, anomaly detection, failure snapshots and timing (default: off)",
    )

    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt into torch.compile training (default: eager, for backward compatibility)",
    )

    # Dataset
    parser.add_argument("--train_set", type=str, default="S3D")
    parser.add_argument("--test_set", type=str, default="E3SM_test")
    parser.add_argument("--config", type=str, default="./configs/config_vae3d.yaml")

    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("The iteration budget must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.validate_every < 0 or (
        args.loader_workers is not None and args.loader_workers < 0
    ):
        parser.error("Validation interval and loader workers must be nonnegative")

    return args


def resolve_budget(args, world_size):
    if args.batch_size % world_size:
        raise ValueError(
            f"Global batch size {args.batch_size} must divide across {world_size} ranks"
        )
    steps = args.iterations * 1000 // world_size
    if steps == 0:
        raise ValueError(
            "The iteration budget must cover at least one synchronized step"
        )
    return args.batch_size // world_size, steps


def run_training(args, rank, world_size, device, staging):
    distributed = world_size > 1
    local_batch_size, steps = resolve_budget(args, world_size)
    unused_iterations = args.iterations * 1000 - steps * world_size
    if rank == 0 and unused_iterations:
        print(
            f"Unused rank-iterations after rounding down: {unused_iterations}",
            flush=True,
        )
    budget_label = f"{args.iterations}k"
    args.iterations = steps
    print(
        f"rank={rank}/{world_size} device={device} global_batch={args.batch_size} "
        f"local_batch={local_batch_size} synchronized_steps={steps}",
        flush=True,
    )

    train_args = convert_args(args, train=True)
    test_args = convert_args(args, train=False)
    norm_type = None
    test_loaders = []
    startup = time.monotonic()
    if distributed or args.plan_only:
        from pyCAESAR.training_data import ShardDataset, source_for, stage_npz

        if distributed and not args.plan_only:
            staging = stage_npz(
                (*train_args.values(), *test_args.values()),
                staging,
                int(os.environ["LOCAL_RANK"]),
            )
        norms = {
            options.get("norm_type", "mean_range") for options in train_args.values()
        }
        if len(norms) != 1:
            raise ValueError("Distributed training requires a common norm_type")
        norm_type = norms.pop()
        train_datasets = []
        for options in train_args.values():
            source, describe = source_for(options, staging, args.plan_only)
            train_datasets.append(
                ShardDataset(
                    options, rank, world_size, source, describe, args.plan_only
                )
            )
        if args.plan_only:
            if sum(map(len, train_datasets)) < local_batch_size:
                raise ValueError(
                    f"Rank {rank} has fewer windows than one complete batch"
                )
            return
        # Match the single-process weighting by repeating smaller datasets locally.
        largest = max(map(len, train_datasets))
        for dataset in train_datasets:
            dataset.visible_length = largest
    else:
        train_datasets = build_dataset(train_args, syn_length=True)
        for dataset in build_dataset(test_args, syn_length=False):
            test_loaders.append(
                DataLoader(
                    dataset,
                    batch_size=local_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=device.type == "cuda",
                    persistent_workers=True,
                )
            )

    workers = (
        args.loader_workers
        if args.loader_workers is not None
        else (0 if distributed else 4)
    )
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=local_batch_size,
        shuffle=True,
        drop_last=distributed,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    epoch_batches = len(train_loader)
    if distributed:
        common_batches = torch.tensor(epoch_batches, device=device)
        dist.all_reduce(common_batches, op=dist.ReduceOp.MIN)
        epoch_batches = int(common_batches)
        if not epoch_batches:
            raise ValueError(
                "At least one rank has fewer windows than a complete batch"
            )
        print(
            f"rank={rank} local_batches={len(train_loader)} common_batches={epoch_batches} "
            f"load_seconds={time.monotonic() - startup:.2f}",
            flush=True,
        )

    save_path = args.save_path
    model_path = os.path.join(
        save_path, f"model_bs{args.batch_size}_ep{budget_label}.pt"
    )
    json_path = os.path.join(
        save_path, f"model_bs{args.batch_size}_ep{budget_label}.json"
    )
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        shutil.copy(args.config, os.path.join(save_path, "config_vae3d.yaml"))
        save_json(
            json_path,
            {
                "argument": vars(args),
                "world_size": world_size,
                "local_batch_size": local_batch_size,
            },
        )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # main(args)
    if args.sr_dim > 0:

        # Training counterpart of compress_modules3d_mid_SR, whose deployment
        # implementation is split into three AOTI-compiled model components.
        from pyCAESAR.models import (
            compress_4_train_modules3d_mid_SR as compress_modules,
        )

        model = compress_modules.CompressorMix(
            dim=args.model_dim,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1,
            d3=True,
            sr_dim=args.sr_dim,
        )
        print("Load Model with SR")

    else:
        # The same training module exposes the compressor without its SR wrapper.
        from pyCAESAR.models import (
            compress_4_train_modules3d_mid_SR as compress_modules,
        )

        model = compress_modules.ResnetCompressor(
            dim=args.model_dim,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2, 1],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1,
            d3=True,
        )
        print("Load Model without SR")

    if args.pretrain != "":
        print("Load pretrain model:", args.pretrain)
        state_dict = torch.load(args.pretrain, map_location="cpu")
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)

    model = model.to(device)
    checkpoint_model = model
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=True
        )
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.compile:
        model = torch.compile(model)
    print(f"Training execution: {'compiled' if args.compile else 'eager'}")
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(i / 5 * args.iterations) for i in range(1, 5)],
        gamma=args.lr_gamma,
    )

    test_names = test_args if rank == 0 else ()
    loggers = {name: Info(name, 32, model_path, json_path) for name in test_names}

    cur_iters = 0
    evaluated_progress = set()

    print(
        f"Learning rate milestones: {[int(i/5*args.iterations) for i in range(1, 5)]}"
    )

    while cur_iters < args.iterations:

        beta = (
            args.init_beta
            if cur_iters < (args.iterations * args.beta_start)
            else args.end_beta
        )

        limit = args.iterations
        if distributed:
            beta_transition = math.ceil(args.iterations * args.beta_start)
            if cur_iters < beta_transition:
                limit = min(limit, beta_transition)
            if args.validate_every:
                next_validation = (
                    cur_iters // args.validate_every + 1
                ) * args.validate_every
                limit = min(limit, next_validation)
        batches = min(epoch_batches, limit - cur_iters)
        training_start = time.monotonic()
        mse_loss, bbp_loss, cur_iters = train_epoch_vae(
            model,
            islice(train_loader, batches),
            optimizer,
            scheduler,
            criterion,
            beta,
            device,
            cur_iters,
            debug=args.debug,
            max_iterations=limit,
            norm_type=norm_type,
        )
        train_loss = mse_loss + bbp_loss

        eval_index = min(100, cur_iters * 100 // args.iterations)
        if distributed:
            loss = torch.tensor([mse_loss, bbp_loss], device=device)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            train_loss = float(loss.sum()) / world_size
            if rank == 0:
                print(
                    f"step={cur_iters}/{args.iterations} loss={train_loss:.6f} "
                    f"training_seconds={time.monotonic() - training_start:.2f}",
                    flush=True,
                )
                interval = args.validate_every
                if cur_iters == args.iterations or (
                    interval and cur_iters % interval == 0
                ):
                    from pyCAESAR.training_data import source_for, validate_volumes

                    validation_start = time.monotonic()
                    for name, options in test_args.items():
                        source, describe = source_for(options, staging)
                        nrmse, bpp = validate_volumes(
                            checkpoint_model, options, device, source, describe
                        )
                        loggers[name].update(
                            checkpoint_model, cur_iters, nrmse, bpp, name
                        )
                        loggers[name].save_last_model(checkpoint_model)
                        print(
                            f"{name}: step={cur_iters} NRMSE={nrmse:.6g} BPP={bpp:.6g} "
                            f"validation_seconds={time.monotonic() - validation_start:.2f}",
                            flush=True,
                        )
            dist.barrier()
        elif eval_index not in evaluated_progress:
            evaluated_progress.add(eval_index)

            for test_loader in test_loaders:
                cur_dataset = test_loader.dataset
                dname = cur_dataset.dataset_name
                original_data = cur_dataset.original_data()

                recons_data, bit_count = test_epoch_vae(
                    model, test_loader, criterion, device
                )
                recons_data = cur_dataset.deblocking_hw(recons_data)

                bpp = float(bit_count / recons_data.numel())

                nrmse = relative_rmse_error_ornl(original_data, recons_data)
                nrmse = float(nrmse)

                loggers[dname].update(checkpoint_model, cur_iters, nrmse, bpp, dname)
                loggers[dname].save_last_model(checkpoint_model)

                print(
                    dname,
                    f"Progress: {eval_index}/100 ,  Iter {cur_iters}, Train Loss: {train_loss:.6f} ({mse_loss:.6f} + {bbp_loss:.6f})",
                    "NRMSE:",
                    nrmse,
                    f"BPP: {bpp:.6f} CR: {32/bpp:.6f}",
                )
                del original_data, recons_data

            print()

    print("Training complete.")


def main():
    args = get_argument()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise ValueError("Invalid torchrun rank topology")
    resolve_budget(args, world_size)

    if args.plan_only:
        run_training(args, rank, world_size, torch.device("cpu"), None)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        if (
            not torch.cuda.is_available()
            or local_world_size > torch.cuda.device_count()
        ):
            raise RuntimeError(
                "Distributed training requires one visible CUDA GPU per local rank"
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        print(
            f"rank={rank} PyTorch={torch.__version__} CUDA={torch.version.cuda} "
            f"NCCL={torch.cuda.nccl.version()}",
            flush=True,
        )

    try:
        if world_size > 1:
            dist.init_process_group("nccl", timeout=timedelta(hours=2))
        with ExitStack() as stack:
            if world_size > 1:
                from pyCAESAR.training_data import npz_staging

            staging = stack.enter_context(npz_staging()) if world_size > 1 else None
            run_training(args, rank, world_size, device, staging)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
