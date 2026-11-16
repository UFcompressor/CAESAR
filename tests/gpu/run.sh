#!/usr/bin/env bash
# Run inside a GPU allocation, or let the staged Slurm entry points call this.
set -eo pipefail
ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
if [[ -n "${CAESAR_ENV_FILE:-}" ]]; then
    source "$CAESAR_ENV_FILE"
fi
set -u
unset UCX_NET_DEVICES UCX_TLS
MODE=${1:?Usage: run.sh performance|correctness}
[[ "$MODE" == performance || "$MODE" == correctness ]]
BUILD_DIR=${CAESAR_BUILD_DIR:-$ROOT/build}
export CAESAR_MODEL_DIR=${CAESAR_MODEL_DIR:-$ROOT/exported_model}
export CAESAR_GPU_DATA=${CAESAR_GPU_DATA:-$BUILD_DIR/tests/TCf48.bin.f32}
export CAESAR_REQUIRE_GPU=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export CAESAR_GPU_SHA=$(git -C "$ROOT" rev-parse HEAD)
# The environment must already contain the user's GPU PyTorch. Never install it.
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S "$ROOT" -B "$BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release -DCAESAR_TEST_ASSERTIONS=ON
make -C "$BUILD_DIR" -j6
python3 "$ROOT/tests/gpu/run_suite.py" "$MODE" --build "$BUILD_DIR"
