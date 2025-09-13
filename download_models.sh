#!/bin/bash
set -e


MODEL_DIR="pretrained"
mkdir -p "$MODEL_DIR"

download_from_gdrive() {
    FILE_ID="$1"
    OUTPUT="$2"

    echo "Downloading $OUTPUT ..."
    curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${FILE_ID}" -L -o /tmp/intermezzo.html
    CONFIRM=$(awk '/download/ {print $NF}' /tmp/intermezzo.html | sed 's/.*confirm=\(.*\)&id=.*/\1/')
    curl -Lb /tmp/cookies "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "${MODEL_DIR}/${OUTPUT}"
    echo "Saved to ${MODEL_DIR}/${OUTPUT}"
}


download_from_gdrive "1sVmxgdg0EdyRK2PhihVamToR2gdRu1nz" "caesar_v.pt"


echo " All models downloaded into ${MODEL_DIR}/"

echo "All done rember to convert the files to a readable format for libtorch with extract_tensors.py /"
