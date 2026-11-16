#!/usr/bin/env bash
# Run inside a GPU allocation, or let the staged Slurm entry points call this.
set -Eeo pipefail
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

export CAESAR_GPU_STAGE="starting GPU job"
record_error() {
    local status=$1 file=$2 line=$3
    export CAESAR_GPU_FAILURE_DETAIL="${CAESAR_GPU_STAGE:-unknown stage}: $file:$line (exit $status)"
    echo "ERROR: $CAESAR_GPU_FAILURE_DETAIL" >&2
}
trap 'record_error "$?" "${BASH_SOURCE[0]}" "$LINENO"' ERR

stage() {
    export CAESAR_GPU_STAGE="$*"
    echo "[GPU] $CAESAR_GPU_STAGE"
}

finish() {
    local test_status=$?
    trap - EXIT ERR
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
    "error": os.environ.get("CAESAR_GPU_FAILURE_DETAIL",
        f"{os.environ.get('CAESAR_GPU_STAGE', 'unknown stage')} failed (exit {status}); see Slurm logs"),
}, indent=2) + "\n")
PY
        # A missing result must never be treated as a successful suite.
        [[ "$test_status" -ne 0 ]] || test_status=1
    fi
    echo "GPU job exit status: $test_status; last stage: ${CAESAR_GPU_STAGE:-unknown}"
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
stage "Checking Python and loading PyTorch"
command -v python3
python3 -c 'import sys; print("Python executable:", sys.executable)'
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
CAESAR_GPU_BACKEND=$(python3 -c 'import torch; print("amd" if torch.version.hip else "nvidia")')
export CAESAR_GPU_BACKEND
stage "Configuring CMake"
cmake -S "$ROOT" -B "$BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release -DCAESAR_TEST_ASSERTIONS=ON
stage "Building with make -j6"
make -C "$BUILD_DIR" -j6
stage "Running $MODE suite"
python3 "$ROOT/tests/gpu/run_suite.py" "$MODE" --build "$BUILD_DIR"
