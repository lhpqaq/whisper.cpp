# Mixed Precision Quantization

This document explains how to use mixed precision quantization in whisper.cpp, which allows you to quantize different layers or tensors with different quantization methods.

## Overview

Mixed precision quantization enables you to specify different quantization types for different tensors in the model. This can help you balance model size, inference speed, and accuracy by applying heavier quantization to less sensitive layers while preserving quality in more important layers.

## Usage

The basic syntax is:

```bash
./quantize [--tensor-type PATTERN=TYPE ...] model-f32.bin model-quant.bin default_type
```

### Parameters

- `--tensor-type PATTERN=TYPE`: Specify quantization type for tensors matching PATTERN (can be used multiple times)
  - `PATTERN`: A regex pattern to match tensor names
  - `TYPE`: A quantization type (q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k, f16, f32)
- `model-f32.bin`: Input model in F32 format
- `model-quant.bin`: Output quantized model
- `default_type`: Default quantization type for tensors not matching any pattern

## Examples

### Example 1: Quantize encoder with Q8_0 and decoder with Q4_0

```bash
./quantize \
  --tensor-type 'encoder\..*\.weight'=q8_0 \
  --tensor-type 'decoder\..*\.weight'=q4_0 \
  model-f32.bin model-mixed.bin q4_k
```

This will:
- Quantize all encoder weight tensors to Q8_0 (higher quality)
- Quantize all decoder weight tensors to Q4_0 (smaller size)
- Use Q4_K as the default for all other tensors

### Example 2: Keep attention layers at higher precision

```bash
./quantize \
  --tensor-type '.*attn.*\.weight'=q8_0 \
  --tensor-type '.*mlp.*\.weight'=q4_0 \
  model-f32.bin model-mixed.bin q4_k
```

This will:
- Quantize all attention layer weights to Q8_0 (higher quality)
- Quantize all MLP layer weights to Q4_0 (smaller size)
- Use Q4_K as the default for all other tensors

### Example 3: Layer-specific quantization

```bash
./quantize \
  --tensor-type 'encoder\.blocks\.0\..*\.weight'=q8_0 \
  --tensor-type 'encoder\.blocks\.[1-3]\..*\.weight'=q5_0 \
  model-f32.bin model-mixed.bin q4_0
```

This will:
- Quantize layer 0 of encoder blocks to Q8_0 (highest quality)
- Quantize layers 1-3 of encoder blocks to Q5_0 (medium quality)
- Use Q4_0 as the default for all other tensors

## Quantization Type Guide

| Type  | Bits per weight | Size | Quality | Speed |
|-------|----------------|------|---------|-------|
| f32   | 32             | 100% | Best    | Slow  |
| f16   | 16             | 50%  | Excellent | Slow |
| q8_0  | 8              | 25%  | Very Good | Medium |
| q6_k  | 6              | 19%  | Good | Medium |
| q5_k  | 5.5            | 17%  | Good | Fast |
| q4_k  | 4.5            | 14%  | Good | Fast |
| q5_0  | 5              | 16%  | Good | Fast |
| q4_0  | 4              | 13%  | Fair | Fast |
| q3_k  | 3.5            | 11%  | Fair | Very Fast |
| q2_k  | 2.6            | 8%   | Poor | Very Fast |

## Pattern Matching

Patterns use standard regex syntax. Some common patterns:

- `.*`: Match any characters
- `encoder\..*`: Match all tensors starting with "encoder."
- `.*weight`: Match all tensors ending with "weight"
- `encoder\.blocks\.[0-5]\..*`: Match encoder blocks 0-5
- `(encoder|decoder)\..*`: Match tensors starting with "encoder." or "decoder."

## Tips for Mixed Precision Quantization

1. **Start with uniform quantization**: Test a single quantization level first to establish a baseline.

2. **Preserve critical layers**: Keep attention mechanisms and early encoder layers at higher precision.

3. **Experiment**: Different models may benefit from different strategies. Test various combinations to find the best balance for your use case.

4. **Monitor output quality**: Always compare the output quality against your requirements when using mixed precision.

5. **Layer sensitivity**: Generally:
   - Encoder layers are more sensitive to quantization
   - Attention layers benefit from higher precision
   - MLP/FFN layers can tolerate more aggressive quantization

## Implementation Reference

This implementation is inspired by llama.cpp's mixed precision quantization feature. The key advantage is the ability to fine-tune the model size and quality trade-off based on your specific requirements and hardware constraints.
