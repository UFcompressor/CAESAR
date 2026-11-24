#!/bin/bash
#SBATCH --job-name=caesar-tgv
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=caesar-tgv-%j.out
#SBATCH --error=caesar-tgv-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the CAESAR repository root}"
source ~/set_env_riker.sh
source ../caesar_venv/bin/activate
set -euo pipefail

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Requires the distributed trainer and a Riker allocation with at least 40 GB per GPU.
node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${node_list%%$'\n'*}
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

srun --kill-on-bad-exit=1 bash -c '
    exec torchrun \
        --nnodes="$SLURM_JOB_NUM_NODES" \
        --nproc-per-node=2 \
        --node-rank="$SLURM_NODEID" \
        --master-addr="$MASTER_ADDR" \
        --master-port="$MASTER_PORT" \
        --module pyCAESAR.train_vae3d \
        --config examples/config_riker_tgv.yaml \
        --train_set tgv \
        --test_set tgv \
        --save_path "snapshots/tgv-riker-${SLURM_JOB_ID}" \
        --batch_size 120 \
        --iterations 100
'
