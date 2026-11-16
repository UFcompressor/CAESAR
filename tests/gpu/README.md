# GPU tests on a Slurm cluster

Only the performance suite uses the full `256,256,256,256` float32 dataset
(17,179,869,184 bytes = 16 GiB). The input must contain at least that many
bytes; if larger, C/D read the first 16 GiB without changing the file. It runs `test_caesarC 0.1` followed by
`test_caesarD` five times, requiring the fastest compression to reach **1.3
GB/s** and the fastest decompression to reach **3.5 GB/s** (decimal GB/s).
Every decompression must also pass its error-bound verification. The timing
covers the compression/decompression call with GPU synchronization, excluding
file I/O, quality checks, and construction of the compressor/decompressor.
Each run starts a fresh executable; there is no additional warm-up run.
These are absolute hardware-dependent thresholds, not comparisons with a baseline.

Correctness runs the existing model cache, metadata, padding, serialization,
rANS, GAE, LBRC, and NGLR tests, a normal-sized `test_caesarC`/`test_caesarD`
pair, `test_caesarCD`, and GAE/LBRC CLI round trips. Foundation-model round trips
use a small `1,1,20,256,256` prefix copied from the input; the original is never
modified. NGLR also runs a **synthetic `1,1,100,500,500` workload**, with
`8,256,256` blocks and one training epoch. It has no throughput threshold and
may be slow because correction coding is autoregressive. This suite is not the
separate nine-dataset/five-error-bound campaign.

## Submit jobs

Commit the changes, check out the exact commit you want to test on the cluster,
and prepare/export its GPU models using your usual environment. No script
installs PyTorch or downloads a dataset. An existing GPU-enabled PyTorch must
already be in the selected Python environment.

From the repository root, submit these scripts (no initial build is required):

```bash
# Optional, if you normally need this environment setup:
export CAESAR_ENV_FILE=/absolute/path/to/set_env_ufl_caesar.sh

sbatch tests/gpu/gpu-performance.slurm
sbatch tests/gpu/gpu-correctness.slurm
```

Both scripts configure CMake, run **`make -j6`**, then run their suite under
`srun` with six CPUs and one GPU. Defaults match the supplied B200 job:
`hpg-b200`, `--gres=gpu:1`, 135G RAM. Override scheduler options on the command
line for your site; the large NGLR case may need a longer walltime:

```bash
sbatch --time=08:00:00 tests/gpu/gpu-correctness.slurm
```

Default paths are `build/`, `exported_model/`, and
`build/tests/TCf48.bin.f32`. Optional absolute-path overrides are
`CAESAR_BUILD_DIR`, `CAESAR_MODEL_DIR`, and `CAESAR_GPU_DATA`.
The Slurm entry points are also copied into `build/tests/` during CMake
configuration, so subsequent submissions can be:

```bash
cd build/tests
sbatch gpu-performance.slurm
sbatch gpu-correctness.slurm
```

Do not run both jobs against the same build directory concurrently: each
reconfigures/builds it. Submit sequentially, or use distinct `CAESAR_BUILD_DIR`
values. Every suite gets a separate results directory and output files.

For an existing interactive allocation:

```bash
srun --ntasks=1 --cpus-per-task=6 bash tests/gpu/run.sh performance
srun --ntasks=1 --cpus-per-task=6 bash tests/gpu/run.sh correctness
```

For AMD/Frontier use your site's normal Slurm allocation options and existing
ROCm build environment. The B200 partition/resource defaults are not Frontier
settings. The runner detects ROCm, exercises the portable tests, and explicitly
skips NVIDIA-only CUDA rANS and nvCOMP tests. NVIDIA rANS is mandatory on NVIDIA;
nvCOMP is optional and its absence is recorded in the report.

## GitHub waiting checks

For a major PR, add the **`gpu-e2e`** label. The status-only workflow creates:

- `gpu-tests/nvidia/correctness`
- `gpu-tests/nvidia/performance`

They say **Waiting for manually submitted Slurm GPU tests**. New commits on a
labeled PR get new pending checks. The workflow does not execute PR code or
hold a GitHub runner while Slurm is queued. Alternatively, use **Actions →
Request external GPU tests → Run workflow**, selecting the branch/tag and
NVIDIA or AMD backend. AMD creates corresponding `gpu-tests/amd/...` checks.
Manual dispatch requires the workflow on the default branch.

After a suite finishes, its Slurm log prints the path to `result.json` and the
report command. From a network-connected login node/workstation with `gh`
already authenticated for that repository, run:

```bash
python3 tests/gpu/report.py /path/to/result.json --repo OWNER/REPO --dry-run
python3 tests/gpu/report.py /path/to/result.json --repo OWNER/REPO
```

Report both result files. This updates only the exact commit recorded at test
time; it never marks a newer commit successful. Tokens are not needed on GPU
compute nodes. Reports contain logs, executable hashes, GPU/PyTorch details,
all five timings, thresholds, and pass/fail results. Keep the results directory
for review. The reporter trusts maintainer-provided files; it is not a signed
attestation. Build failures/timeouts leave the check pending: fix/retry rather
than manually marking success.

Pending statuses block merging only when the repository's rules require those
checks. Configure that separately; no branch protection is changed by this PR.
Requiring them globally will require GPU results on every PR, including ones
without the label. For selective major-change testing, require a maintainer to
review these checks before merging labeled PRs, or use an appropriate ruleset.
