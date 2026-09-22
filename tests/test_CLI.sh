#!/bin/bash

# Quick CAESAR Test - Tests GAE and LBRC compress/decompress cycles.
# NGLR is intentionally excluded because its per-compression training makes it
# too slow for this quick CLI test.

set -e

echo "========================================"
echo "CAESAR Quick Test"
echo "========================================"

DATA="TCf48.bin.f32"
SHAPE="1,1,20,256,256"

# Check files exist
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found!"
    exit 1
fi

if [ ! -f "./caesar" ]; then
    echo "ERROR: caesar executable not found!"
    exit 1
fi

run_round_trip() {
    METHOD="$1"
    OUTPUT_BASE="quick_test_${METHOD}.cae"
    OUTPUT_DATA="quick_test_${METHOD}_output.bin"

    rm -f "${OUTPUT_BASE}"* "$OUTPUT_DATA"

    echo ""
    echo "Compressing $DATA with $METHOD..."
    echo "---"
    ./caesar compress "$DATA" \
        -s "$SHAPE" \
        -o "$OUTPUT_BASE" \
        -e 0.001 \
        --correction "$METHOD" \
        -t \
        --metadata

    echo ""
    echo "Checking $METHOD compressed files..."
    echo "---"
    for SUFFIX in latents hyper meta; do
        if [ ! -f "${OUTPUT_BASE}.${SUFFIX}" ]; then
            echo "ERROR: ${OUTPUT_BASE}.${SUFFIX} not created!"
            exit 1
        fi
        echo "✓ ${OUTPUT_BASE}.${SUFFIX} created"
    done

    echo ""
    echo "$METHOD compressed file sizes:"
    ls -lh "${OUTPUT_BASE}"*

    echo ""
    echo "Decompressing $METHOD result..."
    echo "---"
    ./caesar decompress "$OUTPUT_BASE" \
        -o "$OUTPUT_DATA" \
        -s "$SHAPE" \
        -t \
        --verify \
        --original "$DATA"

    if [ ! -f "$OUTPUT_DATA" ]; then
        echo "ERROR: $OUTPUT_DATA not created!"
        exit 1
    fi
    echo "✓ $OUTPUT_DATA created"

    ORIG_SIZE=$(stat -c%s "$DATA")
    DECOMP_SIZE=$(stat -c%s "$OUTPUT_DATA")

    if [ "$ORIG_SIZE" -ne "$DECOMP_SIZE" ]; then
        echo "ERROR: File sizes don't match for $METHOD!"
        exit 1
    fi
    echo "✓ $METHOD file sizes match ($ORIG_SIZE bytes)"
}

run_round_trip gae
run_round_trip lbrc

echo ""
echo "========================================"
echo "✓✓✓ QUICK TEST PASSED ✓✓✓"
echo "========================================"
echo ""
echo "CAESAR GAE and LBRC compress/decompress cycles work correctly!"
echo ""
echo "Cleanup: Run 'rm quick_test*' to remove test files"
