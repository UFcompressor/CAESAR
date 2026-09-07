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

Defaults: 100 epochs, eight full `(96, 83, 42)` volumes per GPU per batch, two GPU
processes, freshly initialized weights. No checkpoint is loaded. Override epochs
or per-GPU batch size with `--epochs 20 --batch-size 4`. The default global batch
size is 16; if the smoke test runs out of GPU memory, retry with `--batch-size 4`.

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

Rank zero reads one file on CPU and transfers it once to its GPU. Casting,
axis permutation, finite-value checks, and volume shuffling run on GPU. NCCL
scatters 128 volumes to each GPU, where the shard stays for all of its batches.
Gloo carries only small metadata and input-error messages. File order shuffles
on CPU each epoch. DDP synchronizes gradients with NCCL. A float32 GX file is
about 327 MiB; each GPU retains about 163 MiB of input, with additional temporary
preprocessing memory on rank zero. Each full epoch includes every selected training volume; evaluation
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

## Standalone evaluation on unseen data

Run from the repository root in the same Python environment. Only one GPU is
needed; do not use torchrun. Replace the checkpoint and unseen experiment paths:

```bash
python3 -m pyCAESAR.eval_gx \
    --checkpoint snapshots/YOUR_RUN/model_best.pt \
    --data /lustre/blue2/ranka/shared-eklasky/GX/data/YOUR_UNSEEN_EXPERIMENT \
    --batch-size 8 \
    --output unseen-metrics.json
```

An experiment directory evaluates all its restart BP files. To evaluate one
file, pass its full `.bp` path instead (BP directories are supported). You may
also pass multiple files after `--data`, including NPZ files. This command reads
exactly the supplied inputs; it does not apply the training alternating split.
Keep the unseen experiment separate from both training and validation data.

The evaluator loads weights strictly, switches to evaluation mode, and disables
gradients. Model dimensions are read from the checkpoint's sibling `split.json`
when available, otherwise they default to 16; explicit `--model-dim` and
`--sr-dim` options override them. It uses the same volume normalization and
reconstruction as training validation. It does not update or save model weights.

- `nrmse`: global RMSE divided by the global range, matching training validation.
- `mean_volume_nrmse`: arithmetic mean of each volume's RMSE divided by its own range.
- `worst_volume_nrmse`: largest such per-volume score.
- Constant volumes have undefined per-volume range normalization and are excluded
  from those two scores, with their count reported. They remain in global metrics.
- `compression_ratio`: original float32 bits divided by estimated entropy bits.
  This is not measured encoded-file CR and excludes headers and normalization metadata.

This evaluator has not been run locally; verify it on your cluster checkpoint.

## What C++ deployment would require

These are follow-up changes, not implemented by the Python evaluation command:

1. Make the three Python export scripts load the GX checkpoint instead of their
   hard-coded `pretrained/caesar_v.pt`. Preserve the training model's shape-aware
   decoding, final output crop, and BCRN pooling behavior in deployment.
2. Export the compressor for `(batch, 1, 96, 83, 42)` and derive the corresponding
   hyper-decoder and decoder input shapes. Current export examples use
   `(batch, 1, 8, 256, 256)` with only the batch dimension dynamic.
3. Replace hard-coded latent/hyperlatent sizes in
   `CAESAR/models/caesar_decompress.cpp` (including `64x16x16`, `64x4x4`, and
   the temporal grouping of 2) with the GX shapes or serialized shape metadata.
   Align any matching compressor assumptions as well.
4. Match the full-volume preprocessing in C++: 256 independent volumes, per-volume
   mean/range normalization, constant-volume handling, and restoration to the
   original layout. Bypass spatial blocking that would change the model input.
   To read BP directly, add an ADIOS2 reader and the GX axis permutation; callers
   that already supply correctly arranged tensors can perform this step externally.
5. Regenerate all three AOTI packages and entropy CDF tables from the same GX
   checkpoint, then validate Python/C++ reconstruction parity and a real encoded
   round trip. Measure actual CR from the resulting bytes, including metadata.

DDP and the two-GPU training setup are not required for C++ inference.
