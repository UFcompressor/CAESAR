#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <device>"
    echo
    echo "Available devices:"
    echo "  cpu  - CPU"
    echo "  cuda - NVIDIA GPU or AMD GPU using ROCm"
    echo "  mps  - Apple Silicon GPU"
    echo "  xpu  - Intel GPU"
    exit 1
fi

device="$1"

case "$device" in
    cpu|cuda|mps|xpu)
        ;;
    *)
        echo "Error: unsupported device '$device'."
        echo "Allowed devices: cpu, cuda, mps, xpu"
        exit 1
        ;;
esac

scripts=(
    "CAESAR_compressor.py"
    "CAESAR_hyper_decompressor.py"
    "CAESAR_decompressor.py"
)

for script in "${scripts[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "Error: cannot find $script"
        exit 1
    fi

    echo
    echo "=================================================="
    echo "Running $script on device: $device"
    echo "=================================================="

    python3 "$script" "$device"

    echo "Finished $script"
done

echo
echo "=================================================="
echo "All three CAESAR models were compiled successfully."
echo "The exported models are now ready for CMake and the"
echo "C++ build."
echo "=================================================="
