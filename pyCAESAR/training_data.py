"""Plan scientific training windows before reading their payloads."""

import math
import os
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import torch
from torch.utils.data import Dataset

from pyCAESAR.dataset import BaseDataset, block_hw, deblock_hw


def selected_indices(length, bounds):
    return range(length)[slice(*bounds) if bounds is not None else slice(None)]


def dimensions(value):
    return tuple(int(part) for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Volume:
    variable: str
    steps: tuple
    shape: tuple
    frame_start: int
    axis: int | None = None
    section: int | None = None
    block: int | None = None
    boundaries: tuple = ()

    def read(self, reader, frame, length):
        start = [self.frame_start + frame, 0, 0]
        count = [length, *self.shape[1:]]
        if self.axis is not None:
            start.insert(self.axis, self.section)
            count.insert(self.axis, 1)
        if len(self.steps) > 1:
            start, count = start[1:], count[1:]
        options = {
            "step_selection": [
                self.steps[0]
                + (self.frame_start + frame if len(self.steps) > 1 else 0),
                length if len(self.steps) > 1 else 1,
            ]
        }
        if self.block is None:
            values = reader.read(self.variable, start=start, count=count, **options)
        else:
            values = reader.read(self.variable, block_id=self.block, **options)
            values = values.reshape(-1, *self.shape[1:])
            values = values[start[0] : start[0] + length]
        values = np.asarray(values, dtype=np.float32).reshape(length, *self.shape[1:])
        return values if values.flags.writeable else values.copy()


def describe_bp(reader, options):
    metadata = reader.available_variables()
    names = [
        name
        for name in sorted(metadata)
        if metadata[name]["Type"] in ("float", "double")
    ]
    indices = options.get("variable_idx")
    if isinstance(indices, int):
        indices = [indices]
    if indices is not None and (
        not indices or min(indices) < 0 or max(indices) >= len(names)
    ):
        raise ValueError(f"Invalid variable_idx {indices!r} for {names}")

    volumes = []
    for index, name in enumerate(names):
        if indices is not None and index not in indices:
            continue
        info = metadata[name]
        steps = tuple(
            selected_indices(
                int(info["AvailableStepsCount"]), options.get("section_range")
            )
        )
        if not steps:
            raise ValueError(f"{name}: empty step selection")
        variable = reader.inquire_variable(name)
        shape = tuple(variable.shape(steps[0]))
        block_info = [reader.engine.blocks_info(name, step) for step in steps]
        if any(tuple(variable.shape(step)) != shape for step in steps):
            raise ValueError(f"{name}: shape changes across selected steps")
        print(
            f"{options['name']} {name}: shape={shape}, "
            f"stored blocks per selected step={[len(b) for b in block_info]}",
            flush=True,
        )

        def add(
            step_group, spatial, axis=None, section=None, block=None, boundaries=()
        ):
            frames = selected_indices(spatial[0], options.get("frame_range"))
            if not frames:
                raise ValueError(f"{name}: empty frame selection")
            volumes.append(
                Volume(
                    name,
                    step_group,
                    (len(frames), *spatial[1:]),
                    frames.start,
                    axis,
                    section,
                    block,
                    boundaries,
                )
            )

        if not shape:
            for step, blocks in zip(steps, block_info):
                for block in blocks:
                    spatial = dimensions(block["Count"])
                    if len(spatial) != 3:
                        raise ValueError(
                            f"{name}: expected 3D local arrays, got {spatial}"
                        )
                    add((step,), spatial, block=int(block["BlockID"]))
        elif len(shape) == 2:
            if len(steps) == 1:
                raise ValueError(f"{name}: a 2D field needs multiple selected steps")
            add(steps, (len(steps), *shape))
        elif len(shape) == 3:
            for step, blocks in zip(steps, block_info):
                starts = {dimensions(block["Start"]) for block in blocks}
                add(
                    (step,),
                    shape,
                    boundaries=tuple(sorted({start[0] for start in starts if start})),
                )
        elif len(shape) == 4:
            axis = options.get("independent_dim")
            if isinstance(axis, dict):
                axis = axis.get(name)
            if axis is None:
                if len(steps) > 1:
                    raise ValueError(
                        f"{name}: set independent_dim for a 4D field with multiple steps"
                    )
                axis = 0
                print(f"{name}: using independent_dim=0", flush=True)
            if axis not in range(4):
                raise ValueError(f"{name}: independent_dim must be in [0, 3]")
            spatial = shape[:axis] + shape[axis + 1 :]
            for step in steps:
                for section in range(shape[axis]):
                    add((step,), spatial, axis=axis, section=section)
        else:
            raise ValueError(f"{name}: unsupported ADIOS shape {shape}")
    if not volumes:
        raise ValueError(f"No volumes selected from {options['data_path']}")
    return volumes


def window_count(frames, width, stride):
    return 1 + max(0, math.ceil((frames - width) / stride))


def plan_windows(volumes, width, stride, rank, world_size):
    """Assign contiguous ranges of window starts, including halos in reads."""
    ends = list(accumulate(window_count(v.shape[0], width, stride) for v in volumes))
    total = ends[-1]
    candidates = set(ends[:-1])
    for volume, base, end in zip(volumes, [0, *ends], ends):
        for edge in volume.boundaries:
            cut = base + math.ceil((edge - volume.frame_start) / stride)
            if base < cut < end:
                candidates.add(cut)
    margin = total // (world_size * 20)
    cuts = [0]
    for owner in range(1, world_size):
        target = total * owner // world_size
        # Prefer nearby writer boundaries while keeping rank sizes balanced.
        nearby = [cut for cut in candidates if abs(cut - target) <= margin]
        cuts.append(
            min(nearby, key=lambda cut: (abs(cut - target), cut)) if nearby else target
        )
    cuts.append(total)

    for volume, base, end in zip(volumes, [0, *ends], ends):
        first = max(base, cuts[rank])
        stop = min(end, cuts[rank + 1])
        if first < stop:
            start = (first - base) * stride
            count = stop - first
            length = min(volume.shape[0] - start, (count - 1) * stride + width)
            # Include frames needed to reproduce reflection at the global tail.
            tail = (count - 1) * stride + width - length
            halo = max(0, tail - length)
            yield volume, start - halo, length + halo, count, halo, tail


def open_bp(options):
    from adios2 import FileReader

    return FileReader(options["data_path"])


class ShardDataset(BaseDataset):
    def __init__(
        self, options, rank, world_size, open_source, describe, plan_only=False
    ):
        super().__init__(options)
        if not self.inst_norm:
            raise ValueError("Distributed training currently requires inst_norm: true")
        self.stride = self.n_frame - self.n_overlap
        if not 0 < self.stride <= self.n_frame:
            raise ValueError("Require 0 <= n_overlap < n_frame")

        self.shards = []
        self.ends = []
        self.length = 0
        with open_source(options) as reader:
            volumes = describe(reader, options)
            for volume, start, length, count, halo, tail in plan_windows(
                volumes, self.n_frame, self.stride, rank, world_size
            ):
                retained = (length + tail) * math.prod(volume.shape[1:]) * 4
                print(
                    f"rank={rank} {self.dataset_name} {volume.variable} "
                    f"steps={volume.steps} section={volume.section} block={volume.block} "
                    f"frame={volume.frame_start + start} read_frames={length} "
                    f"windows={count} host_bytes={retained}",
                    flush=True,
                )
                if not plan_only:
                    try:
                        raw = volume.read(reader, start, length)
                        if tail:
                            raw = pad_frames(raw, length + tail)
                        self.shards.append((torch.from_numpy(raw), halo))
                    except Exception as exc:
                        raise RuntimeError(
                            f"{os.uname().nodename}: rank {rank} reading {self.data_path} "
                            f"{volume.variable}, retaining {retained} bytes"
                        ) from exc
                self.length += count
                self.ends.append(self.length)
        if not self.length:
            raise ValueError(f"Rank {rank} has no windows for {self.dataset_name}")
        self.visible_length = self.length

    def __len__(self):
        return self.visible_length

    def __getitem__(self, index):
        from bisect import bisect_right

        index %= self.length
        shard = bisect_right(self.ends, index)
        base = self.ends[shard - 1] if shard else 0
        raw, halo = self.shards[shard]
        start = (index - base) * self.stride + halo
        crop = self.apply_padding_or_crop(
            self.apply_augments(raw[start : start + self.n_frame])
        )
        return {"input": crop[None], "index": (shard, start - halo)}


def pad_frames(raw, width):
    if len(raw) == width:
        return raw
    return np.pad(raw, ((0, width - len(raw)), (0, 0), (0, 0)), mode="symmetric")


def normalize_batch(inputs, norm_type):
    spatial_axes = (-3, -2, -1)
    axes = (-2, -1) if norm_type == "mean_range_hw" else spatial_axes
    low = inputs.amin(dim=axes, keepdim=True)
    high = inputs.amax(dim=axes, keepdim=True)
    scale = high - low
    if norm_type == "min_max":
        offset = (low + high) * 0.5
        scale = scale * 0.5
    elif norm_type in ("mean_range", "mean_range_hw"):
        offset = inputs.mean(dim=axes, keepdim=True)
    else:
        raise ValueError(f"Unsupported instance normalization {norm_type!r}")
    scale = torch.where(scale == 0, 1, scale)
    return (inputs - offset) / scale, offset, scale


@torch.inference_mode()
def validate_volumes(model, options, device, open_source, describe):
    """Evaluate all real cells with bounded host and GPU working sets."""
    if not options.get("inst_norm", True) or options.get("n_overlap", 0):
        raise ValueError(
            "Distributed validation requires instance normalization and n_overlap: 0"
        )
    width = options["n_frame"]
    tile = options.get("test_size", (256, 256))
    norm = options.get("norm_type", "mean_range")
    sse = torch.zeros((), device=device, dtype=torch.float64)
    bits = torch.zeros_like(sse)
    low = torch.full_like(sse, float("inf"))
    high = torch.full_like(sse, -float("inf"))
    cells = 0
    model.eval()

    with open_source(options) as reader:
        for volume in describe(reader, options):
            for start in range(0, volume.shape[0], width):
                count = min(width, volume.shape[0] - start)
                read_start = max(0, start - max(0, width - 2 * count))
                raw = volume.read(reader, read_start, start + count - read_start)
                raw = pad_frames(raw, len(raw) + width - count)[start - read_start :]
                actual = torch.from_numpy(raw[:count]).to(device, copy=True)[None, None]
                original = torch.from_numpy(raw)[None, None].to(device)
                blocks, layout = block_hw(original, tile)
                for block in blocks.unbind(dim=1):
                    normalized, offset, scale = normalize_batch(block[:, None], norm)
                    result = model(normalized)
                    block.copy_((result["output"] * scale + offset)[:, 0])
                    bits += result["frame_bit"].sum()
                recon = deblock_hw(blocks, *layout)[:, :, :count]
                sse += (actual - recon).square().sum(dtype=torch.float64)
                low = torch.minimum(low, actual.amin())
                high = torch.maximum(high, actual.amax())
                cells += actual.numel()

    dynamic_range = float(high - low)
    rmse = math.sqrt(float(sse) / cells)
    return (
        rmse / dynamic_range if dynamic_range else (float("inf") if rmse else 0.0),
        float(bits) / cells,
    )


class NpzReader:
    """Expose NPZ selections with the same read contract as the BP reader."""

    def __init__(self, shape, data):
        self.shape = shape
        self.data = data

    def read(self, variable, start, count, step_selection):
        selection = tuple(
            slice(offset, offset + size) for offset, size in zip(start, count)
        )
        index = (int(variable),)
        if len(self.shape) == 5 and len(start) == 3:
            index += (step_selection[0],)
        return self.data[index + selection]


def describe_npz(reader, options):
    shape = reader.shape
    if len(shape) not in (4, 5):
        raise ValueError(
            "NPZ data must have a leading variable axis and 3D or 4D fields"
        )
    indices = options.get("variable_idx")
    if indices is None:
        indices = range(shape[0])
    elif isinstance(indices, int):
        indices = [indices]
    volumes = []
    for index in indices:
        if not 0 <= index < shape[0]:
            raise ValueError(f"Invalid NPZ variable index {index}")
        axis = options.get("independent_dim")
        if isinstance(axis, dict):
            axis = axis.get(index, axis.get(str(index)))
        if axis is not None and (len(shape) != 5 or axis not in range(4)):
            raise ValueError(
                "NPZ independent_dim requires a 4D field and an axis in [0, 3]"
            )
        source_shape = shape[1:]
        if axis is None:
            spatial = shape[-3:]
            sections = selected_indices(
                shape[1] if len(shape) == 5 else 1, options.get("section_range")
            )
        else:
            spatial = source_shape[:axis] + source_shape[axis + 1 :]
            sections = selected_indices(
                source_shape[axis], options.get("section_range")
            )
        frames = selected_indices(spatial[0], options.get("frame_range"))
        if not frames or not sections:
            raise ValueError(f"Empty NPZ selection for variable {index}")
        for section in sections:
            volumes.append(
                Volume(
                    str(index),
                    (section,),
                    (len(frames), *spatial[1:]),
                    frames.start,
                    axis,
                    section,
                )
            )
    if not volumes:
        raise ValueError(f"No volumes selected from {options['data_path']}")
    return volumes


def is_bp(path):
    return path.rstrip("/").endswith((".bp", ".bp3", ".bp4", ".bp5"))


def npz_staging():
    from tempfile import TemporaryDirectory

    return TemporaryDirectory(
        prefix="caesar-training-", dir=os.environ.get("SLURM_TMPDIR")
    )


def stage_npz(datasets, staging, local_rank):
    """Extract each NPY member once per node and share it through a read-only mmap."""
    import shutil
    from zipfile import ZipFile

    import torch.distributed as dist

    paths = sorted(
        {
            options["data_path"]
            for options in datasets
            if not is_bp(options["data_path"])
        }
    )
    if not paths:
        return {}
    local_files = {}
    if local_rank == 0:
        for index, path in enumerate(paths):
            resolved = os.path.join(staging, f"{index}.npy")
            with ZipFile(path) as archive, archive.open("data.npy") as source:
                with open(resolved, "wb") as destination:
                    shutil.copyfileobj(source, destination)
            local_files[path] = resolved
    node_files = [None] * dist.get_world_size()
    dist.all_gather_object(node_files, local_files)
    return node_files[dist.get_rank() - local_rank]


def source_for(options, staging, plan_only=False):
    from contextlib import contextmanager
    from zipfile import ZipFile

    if is_bp(options["data_path"]):
        return open_bp, describe_bp

    @contextmanager
    def open_npz(options):
        path = options["data_path"]
        if plan_only:
            with ZipFile(path) as archive, archive.open("data.npy") as member:
                version = np.lib.format.read_magic(member)
                shape, _, _ = np.lib.format._read_array_header(member, version)
            yield NpzReader(shape, None)
        else:
            data = np.load(staging[path], mmap_mode="r")
            yield NpzReader(data.shape, data)

    return open_npz, describe_npz
