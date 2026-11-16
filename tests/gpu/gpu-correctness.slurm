#!/usr/bin/env bash
#SBATCH --job-name=caesar-gpu-correctness
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=135G
#SBATCH --time=02:00:00
#SBATCH --output=caesar-correctness-%j.out
#SBATCH --error=caesar-correctness-%j.err

set -euo pipefail
ROOT=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
srun --ntasks=1 --cpus-per-task=6 bash "$ROOT/tests/gpu/run.sh" correctness
