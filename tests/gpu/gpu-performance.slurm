#!/usr/bin/env bash
#SBATCH --job-name=caesar-gpu-performance
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=135G
#SBATCH --time=01:00:00
#SBATCH --output=caesar-performance-%j.out
#SBATCH --error=caesar-performance-%j.err

set -euo pipefail
ROOT=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
srun --ntasks=1 --cpus-per-task=6 bash "$ROOT/tests/gpu/run.sh" performance
