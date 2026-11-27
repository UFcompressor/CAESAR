"""Exercise real multi-block BP reads, boundary windows and an optimizer step."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

try:
    import adios2
except ImportError:
    adios2 = None

from pyCAESAR.training_data import DeviceBatch, ShardDataset, source_for


def exercise_rank(rank, size, directory, options, reference):
    if size > 1:
        dist.init_process_group(
            "gloo",
            init_method=Path(directory, "store").as_uri(),
            rank=rank,
            world_size=size,
        )
    try:
        source, describe = source_for(options, None)
        ds = ShardDataset(options, rank, size, source, describe)
        for raw, _ in ds.shards:
            # The monotone fixture uniquely identifies each slab's first plane.
            start = int(raw[0, 0, 0]) // reference[0].size
            np.testing.assert_array_equal(raw, reference[start : start + len(raw)])
        builder = DeviceBatch([ds], torch.device("cpu"), cache_bytes=256)
        batch = next(iter(DataLoader(ds, batch_size=2)))
        values = builder(batch)["input"]
        for sample in values:
            position = int(sample.flatten()[0])
            z, y, x = np.unravel_index(position, reference.shape)
            np.testing.assert_array_equal(
                sample[0], reference[z : z + 4, y : y + 6, x : x + 6]
            )
        layer = torch.nn.Conv3d(1, 1, 1)
        model = DistributedDataParallel(layer) if size > 1 else layer
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
        model(values).square().mean().backward()
        optimizer.step()
        if size > 1:
            other = layer.weight.detach().clone()
            dist.broadcast(other, 0)
            torch.testing.assert_close(other, layer.weight)
    finally:
        if size > 1:
            dist.destroy_process_group()


@unittest.skipIf(adios2 is None, "ADIOS2 bindings required")
class WriterBlockTests(unittest.TestCase):
    def test_native_axes_and_step_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "axes.bp5"))
            reference = np.arange(8 * 7 * 6 * 3, dtype=np.float32).reshape(8, 7, 6, 3)
            with adios2.Stream(path, "w") as writer:
                for step in writer.steps(3):
                    t = step.current_step()
                    for z, end in ((0, 3), (3, 8)):
                        field = (reference[z:end] + t).copy()
                        writer.write(
                            "field4d",
                            field,
                            list(reference.shape),
                            [z, 0, 0, 0],
                            list(field.shape),
                        )
                        plane = field[:, :, 0, 0].copy()
                        writer.write("plane", plane, [8, 7], [z, 0], list(plane.shape))
            options = dict(
                name="fixture",
                data_path=path,
                train_size=4,
                train=True,
                n_frame=4,
                n_overlap=3,
                independent_dim=3,
            )
            source, describe = source_for(options, None)
            four = ShardDataset({**options, "variable_idx": 0}, 0, 1, source, describe)
            for idx, (raw, _) in enumerate(four.shards):
                np.testing.assert_array_equal(raw, reference[..., idx % 3] + idx // 3)
            planes = ShardDataset(
                {**options, "variable_idx": 1}, 0, 1, source, describe
            )
            expected = np.stack([reference[:, :, 0, 0] + t for t in range(3)])
            np.testing.assert_array_equal(planes.shards[0][0], expected)
            padded = DeviceBatch([planes], torch.device("cpu")).sample("fixture", 0, 0)[
                0
            ]
            torch.testing.assert_close(padded[2], padded[3])

    def test_writer_blocks_and_cross_boundary_crops(self):
        reference = np.arange(19 * 9 * 10, dtype=np.float64).reshape(19, 9, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "uneven.bp5"))
            with adios2.Stream(path, "w") as writer:
                for _ in writer.steps(1):
                    for z, end in ((0, 9), (9, 19)):
                        for y, stop in ((0, 3), (3, 9)):
                            piece = reference[z:end, y:stop].copy()
                            writer.write(
                                "field",
                                piece,
                                list(reference.shape),
                                [z, y, 0],
                                list(piece.shape),
                            )
            options = dict(
                data_path=path,
                name="fixture",
                n_frame=4,
                n_overlap=3,
                train_size=6,
                train=True,
            )
            exercise_rank(0, 1, directory, options, reference)
            torch.multiprocessing.spawn(
                exercise_rank, args=(2, directory, options, reference), nprocs=2
            )
