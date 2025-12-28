# Bug Fix: Tensor Size Calculation

## Issue
When using mixed precision quantization on models that contain tensors with various types (not just F32/F16), the quantization would fail or produce corrupted models. The error manifested as:

```
whisper_model_load: tensor 'decoder.token_embedding.weight' has wrong size in model file: got 7468632, expected 358490880
```

## Root Cause
The code that handles non-quantized tensors (tensors that should be copied as-is) incorrectly calculated the byte size:

```cpp
// INCORRECT - assumes only F32 or F16
const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);
data_u8.resize(nelements * bpe);
```

This fails when:
1. The input model already contains quantized tensors (Q4_0, Q8_0, etc.)
2. Tensors have types other than F32/F16
3. You're requantizing or selectively quantizing an already-quantized model

## Solution
Use the proper GGML function to calculate row size based on the actual tensor type:

```cpp
// CORRECT - handles all tensor types
const size_t row_size = ggml_row_size((ggml_type) ttype, ne[0]);
const size_t data_size = row_size * (nelements / ne[0]);
data_u8.resize(data_size);
```

This ensures that tensors not being quantized (embeddings, biases, or tensors excluded by skip patterns) are copied with their correct size, regardless of type.

## Testing
The fix was applied to both the original `ggml_common_quantize_0()` function and the new mixed precision overload. 

To test:
```bash
# This should now work without errors
./quantize \
  --tensor-type 'encoder\..*\.weight'=q8_0 \
  --tensor-type 'decoder\..*\.weight'=q4_0 \
  model.bin model-mixed.bin q4_k
```

## Impact
- Fixes corruption when quantizing models with diverse tensor types
- Allows requantization of already-quantized models
- Ensures embeddings and other non-2D tensors are handled correctly
- Maintains backward compatibility with existing workflows
