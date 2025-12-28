#!/bin/bash

# Visual demonstration of mixed precision quantization feature
# This script shows the new CLI interface and capabilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================================================="
echo "  Mixed Precision Quantization for whisper.cpp"
echo "=================================================================="
echo ""
echo "This feature allows you to quantize different layers with"
echo "different quantization methods in the same model."
echo ""
echo "=================================================================="
echo ""

# Show the help message
echo "📖 Help Message:"
echo "----------------------------------------------------------------"
"$WHISPER_ROOT/build/bin/quantize" --help 2>&1 | head -10
echo "----------------------------------------------------------------"
echo ""

# Show usage examples
echo "💡 Usage Examples:"
echo "=================================================================="
echo ""

echo "Example 1: High-quality encoder, compact decoder"
echo "----------------------------------------------------------------"
cat << 'EOF'
./quantize \
  --tensor-type 'encoder\..*\.weight'=q8_0 \
  --tensor-type 'decoder\..*\.weight'=q4_0 \
  model-f32.bin model-mixed.bin q4_k
EOF
echo ""
echo "This quantizes:"
echo "  • Encoder layers: Q8_0 (8-bit, higher quality)"
echo "  • Decoder layers: Q4_0 (4-bit, smaller size)"
echo "  • Other tensors: Q4_K (default)"
echo ""

echo "Example 2: Preserve attention quality"
echo "----------------------------------------------------------------"
cat << 'EOF'
./quantize \
  --tensor-type '.*attn.*'=q8_0 \
  model-f32.bin model-mixed.bin q4_0
EOF
echo ""
echo "This quantizes:"
echo "  • Attention layers: Q8_0 (8-bit, high quality)"
echo "  • All other layers: Q4_0 (4-bit, compact)"
echo ""

echo "Example 3: Layer-by-layer control"
echo "----------------------------------------------------------------"
cat << 'EOF'
./quantize \
  --tensor-type 'encoder\.blocks\.0\..*'=q8_0 \
  --tensor-type 'encoder\.blocks\.[1-3]\..*'=q5_0 \
  --tensor-type 'encoder\.blocks\.[4-9]\..*'=q4_0 \
  model-f32.bin model-mixed.bin q4_k
EOF
echo ""
echo "This applies progressive quantization:"
echo "  • First encoder block: Q8_0 (highest quality)"
echo "  • Blocks 1-3: Q5_0 (medium quality)"
echo "  • Blocks 4-9: Q4_0 (compact)"
echo "  • Other tensors: Q4_K (default)"
echo ""

echo "=================================================================="
echo "📊 Quantization Types Comparison"
echo "=================================================================="
echo ""
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Type" "Bits" "Size" "Quality" "Speed"
echo "---------|-------|--------|------------|------------"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "F32" "32" "100%" "Best" "Slow"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "F16" "16" "50%" "Excellent" "Slow"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q8_0" "8" "25%" "Very Good" "Medium"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q6_K" "6" "19%" "Good" "Medium"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q5_K" "5.5" "17%" "Good" "Fast"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q4_K" "4.5" "14%" "Good" "Fast"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q5_0" "5" "16%" "Good" "Fast"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q4_0" "4" "13%" "Fair" "Fast"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q3_K" "3.5" "11%" "Fair" "Very Fast"
printf "%-8s | %-5s | %-6s | %-10s | %-10s\n" "Q2_K" "2.6" "8%" "Poor" "Very Fast"
echo ""

echo "=================================================================="
echo "🎯 Best Practices"
echo "=================================================================="
echo ""
echo "1. Start with uniform quantization to establish a baseline"
echo "2. Preserve critical layers (attention, early encoder) at higher precision"
echo "3. Apply aggressive quantization to MLP/FFN layers"
echo "4. Test different strategies to find the best balance"
echo "5. Monitor output quality for your specific use case"
echo ""

echo "=================================================================="
echo "📖 Documentation"
echo "=================================================================="
echo ""
echo "For detailed documentation, see:"
echo "  • examples/quantize/README.md"
echo "  • examples/quantize/README_MIXED_PRECISION.md"
echo "  • MIXED_PRECISION_SUMMARY.md"
echo ""

echo "To run tests:"
echo "  • bash examples/quantize/test_mixed_precision.sh"
echo "  • python3 examples/quantize/test_mixed_precision.py"
echo ""

echo "=================================================================="
echo "✨ Feature Ready for Use!"
echo "=================================================================="
echo ""
