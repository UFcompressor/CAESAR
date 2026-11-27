"""Plan scientific training windows before reading their payloads."""

import logging
import math
import os
import time
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import accumulate, product

import numpy as np
import torch


from pyCAESAR.dataset import BaseDataset, block_hw, deblock_hw

logger = logging.getLogger("pyCAESAR.training")


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
    writer_blocks: tuple = ()

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
        logger.debug(
            f"{options['name']} {name}: shape={shape}, "
            f"stored blocks per selected step={[len(b) for b in block_info]}",
        )

        def add(
            step_group,
            spatial,
            axis=None,
            section=None,
            block=None,
            boundaries=(),
            writer_blocks=(),
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
                    writer_blocks,
                )
            )

        # Preserve the complete native extent. Multiple 4D sections can refer
        # to the same physical writer block; read it once for all its sections.
        layouts = {}
        if shape:
            for step, blocks in zip(steps, block_info):
                extents = tuple(
                    (
                        step,
                        dimensions(b["Start"]),
                        dimensions(b["Count"]),
                        int(b["BlockID"]),
                    )
                    for b in blocks
                )
                layouts[step] = extents if len(extents) > 1 else ()
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
            add(
                steps,
                (len(steps), *shape),
                writer_blocks=(
                    tuple(extent for step in steps for extent in layouts[step])
                    if all(layouts.values())
                    else ()
                ),
            )
        elif len(shape) == 3:
            for step, blocks in zip(steps, block_info):
                starts = {dimensions(block["Start"]) for block in blocks}
                add(
                    (step,),
                    shape,
                    writer_blocks=layouts[step],
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
                logger.debug(f"{name}: using independent_dim=0")
            if axis not in range(4):
                raise ValueError(f"{name}: independent_dim must be in [0, 3]")
            spatial = shape[:axis] + shape[axis + 1 :]
            for step in steps:
                for section in range(shape[axis]):
                    add(
                        (step,),
                        spatial,
                        axis=axis,
                        section=section,
                        writer_blocks=layouts[step],
                    )
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
            halo = min(start, max(0, tail - length))
            yield volume, start - halo, length + halo, count, halo, tail


def open_bp(options):
    from adios2 import FileReader

    return FileReader(options["data_path"])


def block_intersection(volume, extent, frame, length):
    """Map the intersection of a native writer block and a 3D shard to both arrays."""
    step, origin, count, _ = extent
    lower = [volume.frame_start + frame, 0, 0]
    upper = [lower[0] + length, *volume.shape[1:]]
    if len(origin) == 2:
        origin = (step - volume.steps[0], *origin)
        count = (1, *count)
    if volume.axis is not None:
        lower.insert(volume.axis, volume.section)
        upper.insert(volume.axis, volume.section + 1)
    start = [max(a, b) for a, b in zip(lower, origin)]
    end = [min(a, b + c) for a, b, c in zip(upper, origin, count)]
    shape = [b - a for a, b in zip(start, end)]
    if min(shape) <= 0:
        return None
    source = [slice(a - o, b - o) for a, b, o in zip(start, end, origin)]
    dest = [slice(a - o, b - o) for a, b, o in zip(start, end, lower)]
    if volume.axis is not None:
        source[volume.axis] = source[volume.axis].start
        del dest[volume.axis]
        del shape[volume.axis]
    elif len(extent[1]) == 2:
        source = source[1:]
    return tuple(source), tuple(dest), tuple(shape)


def load_writer_blocks(reader, plans, volumes, rank, group, plan_only):
    """One disk owner per block. Forward only intersecting boundary pieces."""
    import torch.distributed as dist
    from adios2 import Mode

    requests = defaultdict(list)
    arrays = {}
    filled = defaultdict(int)
    for owner, entries in enumerate(plans):
        for vi, (volume, start, length, *_) in entries.items():
            if not volume.writer_blocks:
                continue
            if owner == rank and not plan_only:
                arrays[vi] = np.empty((length, *volume.shape[1:]), dtype=np.float32)
            for extent in volume.writer_blocks:
                part = block_intersection(volume, extent, start, length)
                if part is not None:
                    requests[volume.variable, extent].append((owner, vi, *part))

    get_seconds = 0.0

    def distribute(batch):
        nonlocal get_seconds
        if not batch:
            return
        if any(owner == rank for owner, _, _, _ in batch):
            start = time.monotonic()
            reader.engine.perform_gets()
            get_seconds += time.monotonic() - start
        for owner, values, parts, _ in batch:
            for target, vi, source, dest, shape in parts:
                if rank == owner:
                    piece = values[source].reshape(shape)
                    if target == rank:
                        arrays[vi][dest] = piece
                    else:
                        data = np.asarray(piece, dtype=np.float32, order="C")
                        dist.send(torch.from_numpy(data), target, group=group)
                elif rank == target:
                    piece = torch.empty(shape, dtype=torch.float32)
                    dist.recv(piece, owner, group=group)
                    arrays[vi][dest] = piece.numpy()
                if rank == target:
                    filled[vi] += math.prod(shape)

    batch = []
    pending_bytes = 0
    logical_bytes = 0
    block_count = 0
    assigned_bytes = [0] * len(plans)
    variables = {
        name: reader.inquire_variable(name) for name in {key[0] for key in requests}
    }
    for name, extent in sorted(requests):
        step, _, count, block_id = extent
        parts = requests[name, extent]
        sizes = defaultdict(int)
        for target, _, _, _, shape in parts:
            sizes[target] += math.prod(shape)
        owner = max(
            sizes, key=lambda candidate: (sizes[candidate], -assigned_bytes[candidate])
        )
        var = variables[name]
        dtype = np.float64 if var.type() == "double" else np.float32
        size = math.prod(count) * np.dtype(dtype).itemsize
        assigned_bytes[owner] += size
        if owner == rank:
            logical_bytes += size
            block_count += 1
        if plan_only:
            continue
        if pending_bytes + size > 128 * 1024**2 or len(batch) >= 64:
            distribute(batch)
            batch = []
            pending_bytes = 0
        values = None
        if owner == rank:
            values = np.empty(count, dtype=dtype)
            var.set_step_selection([step, 1])
            var.set_block_selection(block_id)
            reader.engine.get(var, values, Mode.Deferred)
        batch.append((owner, values, parts, size))
        pending_bytes += size
    distribute(batch)
    for vi, array in arrays.items():
        if filled[vi] != array.size:
            raise ValueError(f"Incomplete block coverage for {volumes[vi].variable}")
    if requests:
        logger.debug(
            f"rank={rank} writer_blocks={block_count} selected_bytes={logical_bytes} deferred_get_seconds={get_seconds:.3f}",
        )
    return arrays


class ShardDataset(BaseDataset):
    def __init__(
        self,
        options,
        rank,
        world_size,
        open_source,
        describe,
        plan_only=False,
        group=None,
        use_writer_blocks=True,
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
            indices = {id(v): i for i, v in enumerate(volumes)}
            plans = [
                {
                    indices[id(entry[0])]: entry
                    for entry in plan_windows(
                        volumes, self.n_frame, self.stride, owner, world_size
                    )
                }
                for owner in range(world_size)
            ]
            arrays = (
                load_writer_blocks(reader, plans, volumes, rank, group, plan_only)
                if use_writer_blocks and any(v.writer_blocks for v in volumes)
                else {}
            )
            for vi, (volume, start, length, count, halo, tail) in plans[rank].items():
                retained = length * math.prod(volume.shape[1:]) * 4
                logger.debug(
                    f"rank={rank} {self.dataset_name} {volume.variable} "
                    f"steps={volume.steps} section={volume.section} block={volume.block} "
                    f"frame={volume.frame_start + start} read_frames={length} "
                    f"windows={count} host_bytes={retained}",
                )
                if not plan_only:
                    try:
                        raw = (
                            arrays.pop(vi)
                            if vi in arrays
                            else volume.read(reader, start, length)
                        )
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
        index %= self.length
        shard = bisect_right(self.ends, index)
        base = self.ends[shard - 1] if shard else 0
        start = (index - base) * self.stride + self.shards[shard][1]
        return {"dataset": self.dataset_name, "index": (shard, start)}


class DeviceBatch:
    """Cache bounded 3D tiles on the rank device and assemble crops there."""

    def __init__(self, datasets, device, cache_bytes=512 * 1024**2):
        self.datasets = {d.dataset_name: d for d in datasets}
        self.device = device
        self.cache = OrderedDict()
        self.bytes = 0
        if cache_bytes < 4:
            raise ValueError("Tile cache must fit at least one float32 value")
        self.limit = cache_bytes
        self.profile = (
            device.type == "cuda" and os.environ.get("CAESAR_PROFILE_BATCH") == "1"
        )
        self.copy_time = 0.0
        self.batch_events = []
        self.volume_bytes = 0

    def tile(self, name, shard, raw, origin, tile_shape):
        key = (name, shard, *origin)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        selection = tuple(slice(o, o + n) for o, n in zip(origin, tile_shape))
        src = raw[selection]
        size = src.numel() * src.element_size()
        while self.cache and self.bytes + size > self.limit:
            old = self.cache.popitem(last=False)[1]
            self.bytes -= old.numel() * old.element_size()
        before = time.monotonic() if self.profile else 0.0
        tile = src.to(self.device, memory_format=torch.contiguous_format)
        if self.profile:
            self.copy_time += time.monotonic() - before
            self.volume_bytes += size
        self.cache[key] = tile
        self.bytes += size
        return tile

    def sample(self, name, shard, frame):
        ds = self.datasets[name]
        raw = ds.shards[shard][0]
        aug = ds.augment_type
        stride = aug.get("downsample", 1)
        if "randsample" in aug and "downsample" not in aug:
            stride = int(torch.randint(1, aug["randsample"] + 1, ()))
        if stride < 1:
            raise ValueError("Downsampling stride must be positive")
        # Schedule only tiles touched by the random crop. Tensor slicing,
        # striding, boundary assembly, and padding operate on the GPU.
        depth = len(raw)
        tail = max(0, frame + ds.n_frame - depth)
        first = max(0, min(frame, depth - tail))
        start = [first]
        shape = [min(depth, frame + ds.n_frame) - first]
        for extent in raw.shape[1:]:
            available = (extent + stride - 1) // stride
            length = min(ds.train_size, available)
            start.append(int(torch.randint(available - length + 1, ())) * stride)
            shape.append(length)
        result = torch.empty(shape, device=self.device)

        tile_shape = [
            min(d, k)
            for d, k in zip(raw.shape, (ds.n_frame * 2, ds.train_size, ds.train_size))
        ]
        while math.prod(tile_shape) * raw.element_size() > self.limit:
            longest = max(range(len(tile_shape)), key=tile_shape.__getitem__)
            tile_shape[longest] = (tile_shape[longest] + 1) // 2
        chunks = []
        for offset, length, step, tile_size in zip(
            start, shape, (1, stride, stride), tile_shape
        ):
            axis = []
            pos = 0
            while pos < length:
                coord = offset + pos * step
                origin = coord // tile_size * tile_size
                n = min(length - pos, (origin + tile_size - coord + step - 1) // step)
                axis.append(
                    (
                        origin,
                        slice(coord - origin, coord - origin + n * step, step),
                        slice(pos, pos + n),
                    )
                )
                pos += n
            chunks.append(axis)
        for parts in product(*chunks):
            origin, src, dest = zip(*parts)
            result[dest].copy_(self.tile(name, shard, raw, origin, tile_shape)[src])
        if tail:
            positions = (torch.arange(ds.n_frame, device=self.device) + frame) % (
                2 * depth
            )
            positions = torch.minimum(positions, 2 * depth - 1 - positions)
            result = result.index_select(0, positions - first)
        h, w = shape[1:]
        dh, dw = ds.train_size - h, ds.train_size - w
        if dh or dw:
            result = torch.nn.functional.pad(
                result[None],
                (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2),
                mode="reflect",
            )[0]
        return result[None]

    def __call__(self, batch):
        shards, starts = batch["index"]
        if self.profile:
            begin = torch.cuda.Event(enable_timing=True)
            begin.record()
        result = {
            **batch,
            "input": torch.stack(
                [
                    self.sample(name, shard, start)
                    for name, shard, start in zip(
                        batch["dataset"], shards.tolist(), starts.tolist()
                    )
                ]
            ),
        }
        if self.profile:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.batch_events.append((begin, end))
        return result

    def report_profile(self, rank):
        if self.profile:
            # Called after epoch losses have been copied to the host.
            elapsed = sum(start.elapsed_time(end) for start, end in self.batch_events)
            print(
                f"rank={rank} staged_bytes={self.volume_bytes} "
                f"blocking_transfer_seconds={self.copy_time:.3f} "
                f"batch_assembly_including_transfer_seconds={elapsed/1000:.3f}",
                flush=True,
            )
            self.batch_events.clear()
            self.copy_time = 0.0
            self.volume_bytes = 0


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
