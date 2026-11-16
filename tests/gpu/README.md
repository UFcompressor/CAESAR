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
and prepare/export its GPU models using your usual environment. Before Python
runs, both jobs source these three files in order:

```bash
source ~/.bashrc
source /lustre/blue2/ranka/eklasky/data/test/set_env_ufl_caesar.sh
source /lustre/blue2/ranka/eklasky/caesar_venv/bin/activate
```

No script installs PyTorch or downloads a dataset. User-site Python packages
are disabled so `~/.local` cannot override the selected virtual environment.
On another cluster, set `CAESAR_ENV_FILE` and `CAESAR_VENV` to your own paths.

From the repository root, submit these scripts (no initial build is required):

```bash
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
skips NVIDIA-only CUDA rANS and nvCOMP tests. Set `CAESAR_GPU_BACKEND=amd` for AMD jobs so even setup failures are reported
to the AMD check (the backend is otherwise detected after Torch loads).
NVIDIA rANS is mandatory on NVIDIA;
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

After a suite finishes, the job automatically runs `report.py` for
`UFcompressor/CAESAR`, including when tests fail. Setup/configuration/build
failures also generate a failed report. Authenticate `gh` once using `gh auth
login` on the cluster; credentials must be available to the batch job.
Override the repository with `CAESAR_GPU_REPO=OWNER/REPO` if needed.

If the compute node cannot reach GitHub or authentication fails, the batch job
exits nonzero and prints the saved `result.json` path and an exact retry command.
From a network-connected login node/workstation, run:

```bash
python3 tests/gpu/report.py /path/to/result.json --repo OWNER/REPO --dry-run
python3 tests/gpu/report.py /path/to/result.json --repo OWNER/REPO
```

Both jobs report their own result files. This updates only the exact commit recorded at test
time; it never marks a newer commit successful. Automatic reporting uses your existing `gh` authentication on the compute
node; no token is embedded in a script. Reports contain logs, executable hashes, GPU/PyTorch details,
all five timings, thresholds, and pass/fail results. Keep the results directory
for review. The reporter trusts maintainer-provided files; it is not a signed
attestation. Hard kills/timeouts that prevent the exit handler from finishing can leave
the check pending: retry the job or report an existing saved result.

Pending statuses block merging only when the repository's rules require those
checks. Configure that separately; no branch protection is changed by this PR.
Requiring them globally will require GPU results on every PR, including ones
without the label. For selective major-change testing, require a maintainer to
review these checks before merging labeled PRs, or use an appropriate ruleset.
