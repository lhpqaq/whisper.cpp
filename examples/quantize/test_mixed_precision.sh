#!/bin/bash

# Test script for mixed precision quantization
# This script demonstrates how to use the mixed precision quantization feature

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUANTIZE_BIN="$WHISPER_ROOT/build/bin/quantize"
MODELS_DIR="$WHISPER_ROOT/models"
TMP_DIR="/tmp/whisper_quant_test"

echo "=== Mixed Precision Quantization Test ==="
echo ""

# Check if quantize binary exists
if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "Error: quantize binary not found at $QUANTIZE_BIN"
    echo "Please build the project first with: make quantize"
    exit 1
fi

# Create temp directory
mkdir -p "$TMP_DIR"

# Use the tiny test model
TEST_MODEL="$MODELS_DIR/for-tests-ggml-tiny.bin"

if [ ! -f "$TEST_MODEL" ]; then
    echo "Error: test model not found at $TEST_MODEL"
    exit 1
fi

echo "Using test model: $TEST_MODEL"
echo ""

# Test 1: Standard quantization (baseline)
echo "=== Test 1: Standard Q4_0 quantization (baseline) ==="
OUTPUT_STANDARD="$TMP_DIR/model-q4_0.bin"
"$QUANTIZE_BIN" "$TEST_MODEL" "$OUTPUT_STANDARD" q4_0
echo ""
if [ -f "$OUTPUT_STANDARD" ]; then
    SIZE_STANDARD=$(stat -c%s "$OUTPUT_STANDARD" 2>/dev/null || stat -f%z "$OUTPUT_STANDARD" 2>/dev/null || echo "unknown")
    echo "Standard quantization complete: $OUTPUT_STANDARD"
    echo "Size: $SIZE_STANDARD bytes"
else
    echo "Error: standard quantization failed"
    exit 1
fi
echo ""

# Test 2: Mixed precision - encoder Q8_0, decoder Q4_0
echo "=== Test 2: Mixed precision - encoder Q8_0, decoder Q4_0 ==="
OUTPUT_MIXED1="$TMP_DIR/model-mixed-enc8-dec4.bin"
"$QUANTIZE_BIN" \
    --tensor-type 'encoder\..*\.weight'=q8_0 \
    --tensor-type 'decoder\..*\.weight'=q4_0 \
    "$TEST_MODEL" "$OUTPUT_MIXED1" q4_0
echo ""
if [ -f "$OUTPUT_MIXED1" ]; then
    SIZE_MIXED1=$(stat -c%s "$OUTPUT_MIXED1" 2>/dev/null || stat -f%z "$OUTPUT_MIXED1" 2>/dev/null || echo "unknown")
    echo "Mixed precision quantization complete: $OUTPUT_MIXED1"
    echo "Size: $SIZE_MIXED1 bytes"
else
    echo "Error: mixed precision quantization failed"
    exit 1
fi
echo ""

# Test 3: Mixed precision - attention layers Q8_0, rest Q4_0
echo "=== Test 3: Mixed precision - attention layers Q8_0, rest Q4_0 ==="
OUTPUT_MIXED2="$TMP_DIR/model-mixed-attn8.bin"
"$QUANTIZE_BIN" \
    --tensor-type '.*attn.*\.weight'=q8_0 \
    "$TEST_MODEL" "$OUTPUT_MIXED2" q4_0
echo ""
if [ -f "$OUTPUT_MIXED2" ]; then
    SIZE_MIXED2=$(stat -c%s "$OUTPUT_MIXED2" 2>/dev/null || stat -f%z "$OUTPUT_MIXED2" 2>/dev/null || echo "unknown")
    echo "Mixed precision quantization complete: $OUTPUT_MIXED2"
    echo "Size: $SIZE_MIXED2 bytes"
else
    echo "Error: mixed precision quantization failed"
    exit 1
fi
echo ""

# Summary
echo "=== Summary ==="
echo "All tests passed successfully!"
echo ""
echo "Output files:"
echo "  1. Standard Q4_0:                     $OUTPUT_STANDARD ($SIZE_STANDARD bytes)"
echo "  2. Mixed (encoder Q8_0, decoder Q4_0): $OUTPUT_MIXED1 ($SIZE_MIXED1 bytes)"
echo "  3. Mixed (attention Q8_0, rest Q4_0):  $OUTPUT_MIXED2 ($SIZE_MIXED2 bytes)"
echo ""
echo "Note: Mixed precision models are slightly larger than standard Q4_0"
echo "      due to higher precision quantization on selected layers."
echo ""
echo "Test files are in: $TMP_DIR"
echo "You can test these models with the main whisper binary."
echo ""
