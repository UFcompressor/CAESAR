#!/bin/bash

# Quick CAESAR Test - Tests basic compress/decompress cycle

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

# Clean up any previous test files
rm -f quick_test.cae*
rm -f quick_test_output.bin

echo ""
echo "Step 1: Compressing $DATA..."
echo "---"
./caesar compress "$DATA" \
    -s "$SHAPE" \
    -o quick_test.cae \
    -e 0.001 \
    -t \
    --metadata

echo ""
echo "Step 2: Checking compressed files were created..."
echo "---"
if [ ! -f "quick_test.cae.latents" ]; then
    echo "ERROR: quick_test.cae.latents not created!"
    exit 1
fi
echo "✓ quick_test.cae.latents created"

if [ ! -f "quick_test.cae.hyper" ]; then
    echo "ERROR: quick_test.cae.hyper not created!"
    exit 1
fi
echo "✓ quick_test.cae.hyper created"

if [ ! -f "quick_test.cae.meta" ]; then
    echo "ERROR: quick_test.cae.meta not created!"
    exit 1
fi
echo "✓ quick_test.cae.meta created"

echo ""
echo "Compressed file sizes:"
ls -lh quick_test.cae*

echo ""
echo "Step 3: Decompressing..."
echo "---"
./caesar decompress quick_test.cae \
    -o quick_test_output.bin \
    -s "$SHAPE" \
    -t \
    --verify \
    --original "$DATA"

echo ""
echo "Step 4: Checking decompressed file..."
echo "---"
if [ ! -f "quick_test_output.bin" ]; then
    echo "ERROR: Decompressed file not created!"
    exit 1
fi
echo "✓ quick_test_output.bin created"

# Compare file sizes
ORIG_SIZE=$(stat -c%s "$DATA")
DECOMP_SIZE=$(stat -c%s "quick_test_output.bin")

echo ""
echo "File size comparison:"
echo "  Original:      $ORIG_SIZE bytes"
echo "  Decompressed:  $DECOMP_SIZE bytes"

if [ "$ORIG_SIZE" -eq "$DECOMP_SIZE" ]; then
    echo "✓ File sizes match!"
else
    echo "ERROR: File sizes don't match!"
    exit 1
fi

echo ""
echo "========================================"
echo "✓✓✓ QUICK TEST PASSED ✓✓✓"
echo "========================================"
echo ""
echo "CAESAR compress/decompress cycle works correctly!"
echo ""
echo "Cleanup: Run 'rm quick_test*' to remove test files"
