#!/bin/bash

# Quick CAESAR Test - Tests GAE and LBRC compress/decompress cycles.
# NGLR is intentionally excluded because its per-compression training makes it
# too slow for this quick CLI test.

set -euo pipefail

echo "========================================"
echo "CAESAR Quick Test"
echo "========================================"

DATA="${CAESAR_TEST_DATA:-TCf48.bin.f32}"
SHAPE="${CAESAR_TEST_SHAPE:-1,1,20,256,256}"
ERROR_BOUND="${CAESAR_TEST_ERROR_BOUND:-0.001}"
CAESAR_EXE="${CAESAR_TEST_EXE:-./caesar}"

# Check files exist
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found!"
    exit 1
fi

if [ ! -f "$CAESAR_EXE" ]; then
    echo "ERROR: $CAESAR_EXE executable not found!"
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
    "$CAESAR_EXE" compress "$DATA" \
        -s "$SHAPE" \
        -o "$OUTPUT_BASE" \
        -f 8 \
        -e "$ERROR_BOUND" \
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
    DECOMP_LOG=$("$CAESAR_EXE" decompress "$OUTPUT_BASE" \
        -o "$OUTPUT_DATA" \
        -t \
        --verify \
        --original "$DATA")
    printf '%s\n' "$DECOMP_LOG"
    NRMSE=$(printf '%s\n' "$DECOMP_LOG" | awk '/NRMSE:/ {print $2}')
    if ! awk -v value="$NRMSE" -v target="$ERROR_BOUND" 'BEGIN {
        if (value !~ /^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$/ || value + 0 > target + 0)
            exit 1
    }'; then
        echo "ERROR: $METHOD NRMSE $NRMSE exceeds target $ERROR_BOUND or is invalid"
        exit 1
    fi
    echo "✓ $METHOD NRMSE $NRMSE meets target $ERROR_BOUND"

    if [ ! -f "$OUTPUT_DATA" ]; then
        echo "ERROR: $OUTPUT_DATA not created!"
        exit 1
    fi
    echo "✓ $OUTPUT_DATA created"

    ORIG_SIZE=$(wc -c < "$DATA")
    DECOMP_SIZE=$(wc -c < "$OUTPUT_DATA")

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
