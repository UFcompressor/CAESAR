#!/bin/bash
set -e

MODEL_DIR="pretrained"
mkdir -p "$MODEL_DIR"

MODEL_URL="https://github.com/UFcompressor/UFL_MODELS/raw/main/model_bs64_ep100k.pt"
MODEL_NAME="caesar_v.pt"

echo "========================================="
echo "      CAESAR Model Downloader"
echo "========================================="
echo
echo "Downloading pretrained model..."

if curl -L --fail -o "${MODEL_DIR}/${MODEL_NAME}" "${MODEL_URL}"; then
    echo
    echo "  Download complete!"
    echo "  Saved as: ${MODEL_DIR}/${MODEL_NAME}"
    echo "  CAESAR is ready to use!"
else
    echo
    echo "  Download failed."
    exit 1
fi
