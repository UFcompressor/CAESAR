# Distributed training: current status and next work

## Current status

The six-node Riker run (job 221030) completed with 12 GPU ranks, global batch 120, and 8,333 synchronized optimizer updates. It needed `NCCL_P2P_DISABLE=1`; the one-node, two-node, and six-node communication probes then passed. The run trained on the configured TGV ADIOS data and validated `ux`: final loss 0.000057, NRMSE 0.001399918, BPP 0.126289. Model-only best and final files, plus metrics/configuration, were written to `snapshots/tgv-riker-221030/`.

The distributed ADIOS path uses stored writer-block metadata and reads assigned blocks directly. It retains block Start, Count, and BlockID, stages data for GPU processing, and transfers only needed boundary pieces between ranks. This path is intended to preserve the input file's I/O decomposition instead of making every rank read a large slab. ADIOS2 BP-family input is the primary target for multi-node training.

The project owner committed the AMD and LBRC work and verified that LBRC runs on an AMD GPU.

Release mode is the default. Rank 0 reports loading, training progress, saved weights, validation and completion. Use `--debug` for per-rank device, block and timing details. The dedicated shard plan, NCCL check and I/O benchmark also report their diagnostics.

## Prioritized work

1. **Add recoverable training checkpoints.** The existing trainer's `Info` logger writes a model `state_dict` to the standard `.pt` path when validation improves, writes `_final.pt` at validation points, and records validation metrics in the JSON file. Preserve those model-only files and filenames for existing evaluation/export workflows. Add a separate resumable checkpoint, written by rank 0 every configurable number of synchronized optimizer steps and at clean shutdown. It should contain model, optimizer, scheduler, current step, and the state needed to continue the run consistently. Write to a temporary file and atomically rename only after the save completes. On restart, resume from the latest complete checkpoint and continue at its next step. Save a final model-only checkpoint immediately after the last optimizer update, before full-field validation, so a validation failure cannot lose the trained weights.
2. **Remove the training compile option.** Training should use the ordinary eager PyTorch path. Remove the unused `--compile`/`torch.compile` training branch and update the example documentation and CLI help accordingly. This does not change the separate AOTI deployment build or its compiled model components.
3. **Document and keep ADIOS as the main distributed data path.** ADIOS BP3/BP4/BP5 reading should use the original stored block extents and IDs where available, distribute block reads across ranks, keep window ownership explicit, and stage bounded data for GPU batch preparation. Preserve correctness for uneven blocks and windows crossing block boundaries. The existing TGV run demonstrates end-to-end training; remaining work is targeted correctness/performance review of I/O and GPU preparation, not a redesign around a synthetic slab decomposition.
4. **Verify distributed NPZ staging.** In `training_data.py`, `stage_npz` already extracts the archive's `data.npy` member once per node. Local ranks open the extracted file as a read-only memory map and consume assigned windows. Check node-local scratch capacity for the uncompressed member and verify window coverage across ranks on Riker.
5. **Measure strong scaling.** Hold the model, dataset/selection, global batch, and synchronized optimizer-step count fixed as GPU count changes. Record startup I/O, steady-state seconds per update and samples per second, validation, and checkpoint time separately. The completed 12-GPU run is a baseline; choose compatible GPU counts and global batch sizes for the comparison. Weak scaling can be a later, separate experiment.

## Checkpoint compatibility and restart behavior

- Model-only best and final checkpoints remain `state_dict` files compatible with the current load/export path. Keep validation metrics and run arguments in the JSON metadata.
- Resumable checkpoints are training-state files and should use a distinct filename/extension so they cannot be mistaken for model-only weights.
- A restart must restore optimizer and scheduler state and the synchronized step, then use the same training configuration and data selection. Save any random-number-generator state needed for a consistent continuation.
- Only rank 0 writes shared checkpoint and metrics files. Other ranks must follow the same save/continue schedule and receive a clear failure if rank 0 cannot complete a checkpoint.
- Retain at least the latest complete recovery checkpoint and avoid leaving a partial file looking valid after interruption.

## Data and execution notes

- For ADIOS global arrays with multiple stored blocks, inspect per-step block metadata and use each block's full-dimensional Start, Count, and BlockID for the payload read. Balance ownership while accounting for block size and training-window count. A single stored block may require logical subregions as a fallback.
- Keep file I/O and metadata handling on CPUs. Move numeric window assembly, cropping, augmentation, normalization, and final batch shaping to the assigned GPU where practical. Transfer bounded groups of blocks and include only the neighboring values needed for boundary windows.
- For NPZ, use node-level staging rather than ADIOS-style block ownership. Keep the source array shared within a node where practical; do not load a full copy independently for every local GPU process.
- Keep configuration and shape handling generic. Derive selections from YAML and input metadata for every dataset.
- Normal runs should print concise progress. Debug-only output should be opt-in. Keep startup I/O, training, validation, and checkpoint timings distinguishable.

## One-GPU review and Riker rerun

With one visible GPU and `WORLD_SIZE=1`, both formats use `ScientificDataset`. It loads training and validation selections into host RAM, prepares crops on the CPU, and sends batches to the GPU. The six-node run exercised a different reader and validation loop.

- **NPZ.** Use the documented five-dimensional `data` array and a list for `variable_idx`, including when selecting one variable.
- **ADIOS.** The reader discovers float/double fields, sorts their names and applies the configured variable indices. For multi-step inputs it expects a 3D field per selected step. It reads each selected field and step before slicing the frame range.
- **Validation.** Choose a selection that fits the available memory. The current one-GPU metric holds the original and reconstructed selections on the GPU as float64.
- **Manual tests.** The project owner will test one visible GPU with NPZ and ADIOS, including checkpoint writes and restart. Launch the module directly for these runs. The Riker example starts two GPU workers per node. The owner will then repeat the six-node ADIOS run with `NCCL_P2P_DISABLE=1` and compare with job 221030 before merging.

## Testing constraints and acceptance

Local CPU fixtures can check checkpoint serialization/resume logic, NPZ coverage, ADIOS metadata planning, and window values. They cannot establish multi-GPU memory use, multi-node communication, or storage performance. Use Riker for GPU and scaling claims.

Before considering the follow-up ready, verify that an interrupted run can restart from a periodic checkpoint; that model-only best/final artifacts remain loadable; that rank 0 reports checkpoint completion; that debug mode adds diagnostics while release mode stays concise; and that both ADIOS and the documented NPZ path cover every intended window without rank-count-dependent omissions. Then run the fixed-work strong-scaling comparison on Riker.
