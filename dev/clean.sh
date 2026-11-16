#!/usr/bin/env bash
# Remove GPU Slurm stdout/stderr logs from the repository root.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
shopt -s nullglob
logs=()
for path in "$ROOT"/caesar-{correctness,performance}-*.{out,err}; do
    if [[ "${path##*/}" =~ ^caesar-(correctness|performance)-[0-9]+\.(out|err)$ ]]; then
        logs+=("$path")
    fi
done

if (( ${#logs[@]} )); then
    rm -v -- "${logs[@]}"
else
    echo "No GPU Slurm logs to remove."
fi
