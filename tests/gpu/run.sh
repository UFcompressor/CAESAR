#!/usr/bin/env bash
# Run inside a GPU allocation, or let the staged Slurm entry points call this.
set -eo pipefail
ROOT=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
MODE=${1:?Usage: run.sh performance|correctness}
[[ "$MODE" == performance || "$MODE" == correctness ]]
BUILD_DIR=${CAESAR_BUILD_DIR:-$ROOT/build}
export CAESAR_GPU_SHA=$(git -C "$ROOT" rev-parse HEAD)
export CAESAR_GPU_BACKEND=${CAESAR_GPU_BACKEND:-nvidia}
export CAESAR_GPU_REPO=${CAESAR_GPU_REPO:-UFcompressor/CAESAR}
mkdir -p "$BUILD_DIR/tests/gpu-results"
RESULT_DIR=$(mktemp -d "$BUILD_DIR/tests/gpu-results/${CAESAR_GPU_SHA:0:12}-$MODE-${SLURM_JOB_ID:-manual}-XXXXXX")
export CAESAR_GPU_RESULT="$RESULT_DIR/result.json"

finish() {
    local test_status=$?
    trap - EXIT
    set +e
    # Setup/configuration/build errors must also produce a failed result.
    if [[ ! -f "$CAESAR_GPU_RESULT" ]]; then
        python3 - "$CAESAR_GPU_RESULT" "$MODE" "$test_status" <<'PY'
import json
import os
from pathlib import Path
import sys

path, mode, status = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema_version": 1,
    "sha": os.environ["CAESAR_GPU_SHA"],
    "mode": mode,
    "backend": os.environ["CAESAR_GPU_BACKEND"],
    "passed": False,
    "tests": [],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "error": f"Setup, build, or suite failed before writing a result (exit {status}); see Slurm logs",
}, indent=2) + "\n")
PY
        # A missing result must never be treated as a successful suite.
        [[ "$test_status" -ne 0 ]] || test_status=1
    fi
    echo "Reporting GPU result to $CAESAR_GPU_REPO: $CAESAR_GPU_RESULT"
    python3 "$ROOT/tests/gpu/report.py" "$CAESAR_GPU_RESULT" --repo "$CAESAR_GPU_REPO"
    local report_status=$?
    if [[ "$report_status" -ne 0 ]]; then
        echo "ERROR: Automatic GitHub reporting failed. Retry on a network-connected login node:" >&2
        printf 'python3 %q %q --repo %q\n' "$ROOT/tests/gpu/report.py" "$CAESAR_GPU_RESULT" "$CAESAR_GPU_REPO" >&2
        [[ "$test_status" -ne 0 ]] || test_status=$report_status
    fi
    exit "$test_status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Source all three in order, before using Python or importing Torch.
# Keep nounset disabled while sourcing site/user shell initialization scripts.
source "$ROOT/tests/gpu/setup_env.sh"
set -u
unset UCX_NET_DEVICES UCX_TLS
export CAESAR_MODEL_DIR=${CAESAR_MODEL_DIR:-$ROOT/exported_model}
export CAESAR_GPU_DATA=${CAESAR_GPU_DATA:-$BUILD_DIR/tests/TCf48.bin.f32}
export CAESAR_REQUIRE_GPU=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
# The environment must already contain the user's GPU PyTorch. Never install it.
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
export CAESAR_GPU_BACKEND=$(python3 -c 'import torch; print("amd" if torch.version.hip else "nvidia")')
cmake -S "$ROOT" -B "$BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release -DCAESAR_TEST_ASSERTIONS=ON
make -C "$BUILD_DIR" -j6
python3 "$ROOT/tests/gpu/run_suite.py" "$MODE" --build "$BUILD_DIR"
