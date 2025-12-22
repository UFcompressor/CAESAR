#!/bin/bash
set -e

MODEL_DIR="pretrained"
mkdir -p "$MODEL_DIR"

GDRIVE_FILE_ID="1sVmxgdg0EdyRK2PhihVamToR2gdRu1nz"
HF_URL="https://huggingface.co/E53klasky/UFL_CAESAR_MODELS/resolve/main/caesar_v.pt"
MODEL_NAME="caesar_v.pt"

download_from_gdrive() {
    FILE_ID="$1"
    OUTPUT="$2"

    echo "Trying to download $OUTPUT from Google Drive..."
    set +e
    curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${FILE_ID}" -L -o /tmp/intermezzo.html
    CONFIRM=$(awk '/download/ {print $NF}' /tmp/intermezzo.html | sed 's/.*confirm=\(.*\)&id=.*/\1/')
    curl -Lb /tmp/cookies "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "${MODEL_DIR}/${OUTPUT}"
    CURL_EXIT=$?
    set -e
    return $CURL_EXIT
}

download_from_huggingface() {
    URL="$1"
    OUTPUT="$2"
    echo "Downloading $OUTPUT from Hugging Face..."
    curl -L -o "${MODEL_DIR}/${OUTPUT}" "$URL"
}


if ! download_from_gdrive "$GDRIVE_FILE_ID" "$MODEL_NAME"; then
    echo "Google Drive download failed, falling back to Hugging Face."
    download_from_huggingface "$HF_URL" "$MODEL_NAME"
else
    echo "Downloaded $MODEL_NAME from Google Drive successfully."
fi

echo "All models are in ${MODEL_DIR}/"

