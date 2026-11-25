# Distributed training implementation plan

The first prototype covers the Python foundation-model trainer and will be handed off for manual testing. Start with a fixed number of nodes and the same number of GPUs per node.

## Launch and training semantics

- Run one `torchrun` launcher per node with one process per GPU. Use `LOCAL_RANK` for CUDA device selection, `RANK` for shard ownership, and `WORLD_SIZE` for the total number of GPU processes. Use NCCL for DDP. Each GPU holds a full model replica and DDP synchronizes gradients.
- Treat `--batch_size` as global. Two nodes with two GPUs each and a global batch of 64 give each rank 16 samples per step. Require the batch size to divide evenly by the world size.
- Treat `--iterations` as thousands of rank-iterations across the run. With `--iterations 100`, one GPU performs 100,000 optimizer updates. Four GPUs perform 25,000 local iterations each and 25,000 synchronized optimizer updates. Round the budget down to whole synchronized steps and log any unused rank-iterations. Base the learning-rate schedule and validation cadence on synchronized optimizer steps. Compare equal optimizer step counts when measuring strong scaling at a fixed global batch size.

## Initial load and data decomposition

- **ADIOS2.** Inspect the selected variables and steps for global shape, block count, and stored block extents. Check the installed bindings before choosing the metadata calls. The [Python API reviewed for this plan](https://adios2.readthedocs.io/en/v2.13.0-rc1/api_python/api_python.html) is ADIOS2 2.13.0-rc1. Each training rank opens the BP file independently without an MPI communicator. The original MPI writer layout is reflected in the file metadata. The training ranks can use a different decomposition.
- **Global arrays.** Prefer stored block boundaries when they yield balanced assignments. Balance by estimated training-window count and use assigned bytes to check for load and memory imbalance. If a global array has a single large stored block, compute logical subdomains and request them using `start` and `count`. For example, a `2 × 2 × 1` grid divides `4098 × 4098 × 4098` into four `2049 × 2049 × 4098` regions. This is a reader-side decomposition. For local arrays, use block IDs and report when the available blocks cannot supply balanced shards.
- **NPZ.** Load the selected arrays into host memory at startup, aiming for one load per node. Apply the same logical decomposition and give local ranks access to their assigned regions through node-local shared memory. Account for peak host memory during loading and staging.
- **Window ownership.** Form 3D training inputs using the layout rules below, then divide those inputs into smaller blocks for the ranks. Give each window one owner and include any boundary overlap needed to construct it. Preserve the existing dataset weighting and shuffle locally. Ensure every rank can produce the same number of complete batches and fail early if any rank lacks usable windows.
- **Training after load.** Keep each shard in host memory and form batches from it. Storage format affects startup. The steady-state loader uses the same in-memory path for both formats. Recommend training on original simulation values or a lossless copy in the release training guide.

## Input axes and 3D training blocks

ADIOS steps are an outer sequence. Axis numbers for a native array refer to the dimensions inside that variable, in file order.

| Input | Form the 3D training inputs |
| --- | --- |
| 2D field per ADIOS step | Read the selected 2D fields and stack them as `[T, H, W]`, with the step axis at position 0. Divide that 3D array into smaller training blocks. |
| Native 3D array | Use its three axes directly. If ADIOS contains multiple steps of 3D fields, treat each selected step as a separate 3D input. |
| Native 4D array | Accept a per-variable YAML option `independent_dim`, using zero-based indices `0`, `1`, `2`, or `3`. Fix one index at a time along that axis to produce a 3D input, then divide it into smaller blocks. |

An explicit `independent_dim` takes priority for native 4D arrays. Otherwise default to axis 0 and log the choice. For `[T, Z, Y, X]`, axis 0 gives one spatial volume per time index. For `[Z, Y, X, T]`, axis 3 gives the same result. If preserving a separate ADIOS step dimension alongside a native 4D field would create a 5D layout, require an explicit layout before reading the payload.

Apply the same native-array rules to NPZ after selecting the variable and resolving the configured input layout. For ADIOS global arrays, translate each rank's owned 3D blocks back to selections in file coordinates before loading payloads. Keep block ownership and any window overlap explicit so each training window has one owner.

## Output and manual verification

- Have global rank 0 write the resolved YAML snapshot, metrics, and checkpoints. Save the underlying model's state dict in the existing format for export. Keep writes out of the per-step path. Run whole-field validation on rank 0 at agreed step boundaries and coordinate the other ranks before training resumes.
- Log the file block layout, rank assignments, expected window counts, global and local batch sizes, and synchronized step budget at startup. Keep startup load and staging time distinct from steady-state step time and validation or checkpoint time.
- During manual testing, compare the single-GPU path and a two-node, four-GPU launch. Check shard coverage, boundary windows, step counts, and the files produced by rank 0. If scaling stalls, look at startup storage I/O and host RAM first for slow loads. For slow steps, check CPU batch preparation, host-to-GPU transfer, GPU compute, and NCCL wait time.

## Versions and runtime record

| Component | Version reference | What to establish before manual GPU testing |
| --- | --- | --- |
| PyTorch and CUDA | `requirements.txt` pins `torch==2.2.0+cu118`. This identifies the repository requirement and its CUDA 11.8 build, not the cluster environment. | Record the imported PyTorch version, its CUDA build, the NVIDIA driver, and NCCL on both nodes. Use the same environment on all ranks. Check eager DDP first. |
| ADIOS2 Python | The API reference reviewed is **2.13.0-rc1**. The training requirements do not pin ADIOS2, and the workspace Python did not report an installed ADIOS2 package. | Record the imported ADIOS2 version and BP engine. Confirm the installed reader exposes the needed block metadata and selection APIs before pinning a training version. |

## Suggested implementation order

1. **Resolve the launch layout.** Keep launcher parsing and validation together. Derive a checked local device, global rank, local worker count, local batch size, and synchronized step budget before opening training files. Keep the direct single-GPU invocation available for quick iteration.
2. **Plan before reading payloads.** Keep metadata inspection and shard planning separate from bulk reads. The planner takes shapes, stored extents, selected steps and frame ranges, and rank topology. It returns read selections plus owned windows. A plan-only invocation should print the assignments and estimated bytes without loading arrays, initializing a model, or allocating GPU memory. Validate per-step shapes and fail early on layouts the prototype cannot interpret.
3. **Load and stage.** Keep an ADIOS2 reader and an NPZ staging path behind the same shard contract. The current ADIOS2 reader uses `FileReader` for BP-family files and applies some slicing after the read. Push shard and frame selections into the reader where possible. For NPZ, choose and document a node-local sharing mechanism with explicit lifetime and cleanup. Estimate peak memory including the full array during its initial load, shard buffers, and prefetching. A `4098 × 4098 × 4098` float32 field is about 256 GiB before staging copies, so use small fixtures for bring-up.
4. **Keep tensor math on the GPU.** CPU work covers file reads, metadata and index bookkeeping, host-memory staging, and batch assembly. After the transfer, run numeric training preprocessing such as normalization, as well as forward, loss, backward, and optimizer updates on the assigned GPU. Keep the per-step path free of host scalar extraction where practical. Batch GPU preprocessing and preserve the existing augmentation and normalization order and per-window semantics. If raw windows are too large or vary in shape, settle the transfer and batching layout explicitly before implementation. Validation follows the same device principle for tensor math.
5. **Wire the trainer and artifacts.** Keep ownership, read selection, and GPU batch preparation in focused helpers with explicit inputs and outputs. Reuse the current model and export-compatible checkpoint path. Coordinate startup failures and rank-0 validation so peers do not wait indefinitely, and clean up shared memory and the process group on exit. Check that all ranks take the same number of backward passes, including when a shard runs short.

## Open decisions and quick manual checks

- **Axis and window semantics.** Verify the configured or default axis mapping against representative inputs and choose how to handle uneven edge windows. If the configured window cannot fit a shard, resolve ownership and overlap before training.
- **Fair sampling across unequal shards.** Decide when to subdivide blocks further and when to resample. Check that the resulting per-rank queues preserve the intended dataset mix. Log any deliberate repetition or dropped tail windows.
- **Memory and worker budget.** Pick a practical host-memory ceiling and a per-node loader worker budget. Check available node-local shared-memory capacity before choosing NPZ staging. Make an initialization failure identify the node, file, variable, and requested allocation.
- **GPU preprocessing parity.** Compare a small fixed batch with the existing path, including offsets and scales used for reconstruction. Check that moving normalization does not introduce per-batch device synchronization. Preserve the scope of normalization statistics when those statistics cover data spanning multiple shards.
- **Validation and checkpoint policy.** Confirm which validation dataset selects the best checkpoint when multiple datasets are configured, and choose a write cadence suitable for shared storage. Confirm whether resuming optimizer and scheduler state is needed for the prototype.
- **Fast manual progression.** Inspect the shard plan for a tiny uneven NPZ and BP input, including a BP file with one global block and one with several. Then try a few eager steps on one GPU, follow with two GPUs on one node, and use the two-node run after basic launch and checkpoint behavior are understood. Use a small valid iteration budget or a dedicated short-run override so bring-up takes seconds or minutes. Compare the union of owned windows, check that each process binds the right GPU, and watch for validation or input failures that leave peers waiting. Automated tests can be added later if needed.

## ADIOS2 compatibility target

Target the [ADIOS2 2.10.2 Python API](https://adios2.readthedocs.io/en/v2.10.2/api_python/api_python.html) for the initial reader. Its documentation includes random-access `FileReader`, block metadata, and spatial, block-ID, and step selections. These APIs are also documented in [2.11.0](https://adios2.readthedocs.io/en/v2.11.0/api_python/api_python.html) and [2.12.0](https://adios2.readthedocs.io/en/v2.12.0/api_python/api_python.html). Use this common API surface. Prefer per-step block metadata where supported when only a small subset of steps is selected. The release-candidate docs above explain the design, and the prototype can use the older API.

2.10.2 is the earliest version checked for this plan, with compatibility based on documentation. Before advertising a minimum version, manually verify metadata and selection reads against the cluster's installed bindings and representative BP input. Record the original writer version and BP engine when known, especially if a newer writer produced files for an older reader.

The first useful details from the training environment are the installed ADIOS2 version, the BP engine used by a representative file, and a representative input to verify the layout rules. If the cluster requires a version older than 2.10.2, check its API before picking the compatibility approach.
