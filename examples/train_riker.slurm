#!/bin/bash
#SBATCH --job-name=caesar-tgv
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=caesar-tgv-%j.out
#SBATCH --error=caesar-tgv-%j.err
#SBATCH --account=CSC143
#SBATCH --cpus-per-gpu=16

source ~/.bashrc
source ~/set_env_riker.sh
source ../caesar_venv/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Riker's six-node NCCL check passed with direct GPU peer access disabled.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# Requires the distributed trainer and a Riker allocation with at least 40 GB per GPU.
node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${node_list%%$'\n'*}
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

if [[ ${CAESAR_NCCL_CHECK_ONLY:-0} == 1 ]]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
fi

# The trainer defaults to release mode when submitted without arguments.
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --gres=gpu:2 --kill-on-bad-exit=1 bash -c '
    exec python3 -m torch.distributed.run \
        --nnodes="$SLURM_JOB_NUM_NODES" \
        --nproc-per-node=2 \
        --node-rank="$SLURM_NODEID" \
        --master-addr="$MASTER_ADDR" \
        --master-port="$MASTER_PORT" \
        --module pyCAESAR.train_vae3d \
        --config examples/config_riker_tgv.yaml \
        --train_set tgv \
        --test_set tgv \
        --pretrain pretrained/caesar_v.pt \
        --save_path "snapshots/tgv-riker-${SLURM_JOB_ID}" \
        --batch_size "${CAESAR_BATCH_SIZE:-120}" \
        --iterations "${CAESAR_ITERATIONS:-100}" "$@"
' bash "$@"
