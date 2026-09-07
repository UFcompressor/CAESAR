# GX training on two GPUs

Activate your CAESAR GPU Python environment on a compute node with two GPUs.
Install the additional reader dependency if ADIOS2 is not already available:

```bash
python3 -m pip install -r requirements-gx.txt
```

From the repository root, first run:

```bash
bash train.sh --smoke-test 2>&1 | tee gx-smoke.log
```

This reads one training restart and one held-out restart, runs two batches per
GPU for each, and writes metrics and model weights. It checks the actual full
volume shape; it does not shrink the volumes. The smoke-test metrics cover only
these batches and are not the full test-set results.

For the full run:

```bash
bash train.sh 2>&1 | tee gx-train.log
```

Defaults: 100 epochs, one full `(96, 83, 42)` volume per GPU per batch, two GPU
processes, freshly initialized weights. No checkpoint is loaded. Override epochs
or per-GPU batch size with `--epochs 20 --batch-size 2`.

The configured root is `/lustre/blue2/ranka/shared-eklasky/GX/data`:

- All restarts from `exp-n125-fp0.7-0.7-tp3.0-3.0` train.
- Numerically time-sorted restarts from `exp-n125-fp1.0-1.0-tp1.0-5.0` alternate:
  positions 1, 3, 5, ... train; positions 2, 4, 6, ... test.
- Only `gx.restart.nc.Time_*.bp` paths are selected, including BP directories.
  `gx.big.bp` and `gx.out.bp` are excluded.

Rank zero reads `G` using ADIOS2 FileReader without MPI. If reading fails, the
same path is tried as NPZ (`data` or `G` key), then an error reports both failures.
The shared reader also supports the existing scientific NPZ 5D layout.
Seven-dimensional GX data must have shape `(2, 16, 4, 96, 83, 42, 2)`; axes are
permuted before reshaping into 256 independent volumes. The final real/imaginary
axis becomes part of the variable index. Each volume is normalized by its own
mean and range; constant volumes are supported. No spatial cropping is applied.

One file at a time is scattered on CPU with Gloo, giving each GPU process 128
volumes. File and volume order shuffle each epoch. DDP synchronizes gradients
with NCCL. Each full epoch includes every selected training volume; evaluation
includes every held-out volume and reduces metrics across ranks.

Each launch creates `snapshots/gx-scratch-<timestamp>/` containing:

- `split.json`: exact file lists and run arguments.
- `metrics.jsonl`: one record per epoch; NRMSE uses the global held-out range.
  BPP and compression ratio are entropy estimates, excluding container overhead.
- `model_final.pt`: latest weights; `model_best.pt`: lowest held-out RMSE weights.

An existing `split.json` causes an error to prevent overwriting an earlier run.
Use a fresh `--save-path` if specifying your own output directory. GPU training
and ADIOS reads must be verified on the cluster; they were not run locally.

Implementation references: [ADIOS2 Python API](https://adios2.readthedocs.io/en/master/api_python/api_python.html)
and [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
