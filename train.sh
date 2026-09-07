#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Run in a single-node allocation with two GPUs and your Python environment active.
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m pyCAESAR.train_gx \
    --save-path "snapshots/gx-scratch-$(date +%Y%m%d-%H%M%S)" \
    --batch-size 1 \
    --epochs 100 \
    "$@"
