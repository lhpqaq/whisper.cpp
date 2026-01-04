# Tensor Quantization Guidelines

## Tensors That Should NOT Be Quantized

When using mixed precision quantization, certain tensors should be excluded from quantization to maintain model quality. These tensors are automatically added to the skip list:

### 1. Embeddings
- **`decoder.token_embedding.weight`** - Token embeddings for the decoder
  - Critical for vocabulary representation
  - Quantizing this severely degrades output quality
  - Size: Usually the largest single tensor in the model
  - Dimensions: `[vocab_size, hidden_dim]`

- **`encoder.positional_embedding`** - Positional embeddings for encoder
  - Encodes position information in audio features
  - Important for temporal understanding

- **`decoder.positional_embedding`** - Positional embeddings for decoder
  - Encodes position information in text generation
  - Important for sequence coherence

### 2. Biases
- **`encoder.conv1.bias`** - First convolutional layer bias
- **`encoder.conv2.bias`** - Second convolutional layer bias
- These are typically small and already in F32

## Why Skip These Tensors?

### Token Embeddings
Token embeddings are the lookup table that maps token IDs to vector representations. They are:
- **High sensitivity**: Small changes in embedding values can significantly impact output
- **Large vocabulary**: With ~50k tokens, this is one of the largest tensors
- **Direct impact**: Every token goes through this layer, so errors propagate to all outputs

### Positional Embeddings
Positional embeddings encode sequence position information:
- **Critical for attention**: Used to provide positional context in transformer layers
- **Relatively small**: Not a major contributor to model size
- **High impact**: Quantization here affects all sequence positions

### Biases
Bias terms are typically:
- **Already small**: Usually in F32 and relatively tiny
- **Simple operations**: Added directly to activations, so precision matters
- **Minimal size benefit**: Quantizing them saves negligible space

## Quantization Priority

When using mixed precision quantization, consider this priority:

### Safe to Quantize (Lowest Quality Impact)
1. **MLP/FFN layers** - Feed-forward network weights
   - `*.mlp.*.weight` patterns
   - Can tolerate aggressive quantization (Q4_0, Q4_K)
   - Large in size, good compression candidates

2. **Later encoder/decoder blocks** - Higher layer numbers
   - `encoder.blocks.[4-9].*.weight`
   - Less critical than early layers
   - Can use Q4_0 to Q5_0

### Moderate Quality Impact
3. **Attention projection layers** - Q, K, V projections
   - `*.attn_q.weight`, `*.attn_k.weight`, `*.attn_v.weight`
   - Use Q5_0 to Q8_0 for better quality
   - Important for attention mechanism

4. **Early encoder blocks** - First few layers
   - `encoder.blocks.[0-3].*.weight`
   - Process raw audio features
   - Use Q6_K to Q8_0

### Critical - Do Not Quantize
5. **Embeddings** (as listed above)
6. **Layer norms** - Usually F32 and very small
7. **Biases** - Minimal size benefit

## Example: Balanced Mixed Precision

```bash
./quantize \
  --tensor-type '.*mlp.*\.weight'=q4_0 \           # MLP layers - aggressive
  --tensor-type '.*attn.*\.weight'=q6_k \          # Attention - moderate
  --tensor-type 'encoder\.blocks\.[0-2]\..*'=q8_0 \  # Early encoder - conservative
  model-f32.bin model-mixed.bin q5_0                # Default for others
```

## Output Indicators

When running quantization, look for these indicators:

- **`[skipped]`** - Tensor was excluded from quantization (auto-skip list)
- **`matched pattern -> TYPE`** - Tensor matched a custom pattern
- **`size = X MB -> Y MB`** - Quantization applied, showing size reduction
- **`size = X MB`** - Tensor was not quantized (kept as-is)

## Common Mistakes to Avoid

### ❌ Wrong: Quantizing Everything
```bash
# This will quantize token embeddings and hurt quality!
./quantize --tensor-type '.*'=q4_0 model.bin output.bin q4_0
```

### ✅ Correct: Respecting Skip List
```bash
# Skip list automatically protects critical tensors
./quantize \
  --tensor-type '.*\.weight'=q4_0 \
  model.bin output.bin q5_0
# decoder.token_embedding.weight will be [skipped]
```

### ❌ Wrong: Overly Specific Patterns That Include Embeddings
```bash
# This pattern catches token_embedding!
./quantize --tensor-type 'decoder\..*'=q4_0 model.bin output.bin q4_0
```

### ✅ Correct: Exclude Embeddings Explicitly
The skip list now handles this automatically, but you can also be more specific:
```bash
./quantize \
  --tensor-type 'decoder\.blocks\..*\.weight'=q4_0 \  # Only block weights
  model.bin output.bin q5_0
```

## Summary

- **Always check for `[skipped]` in the output** to verify critical tensors are protected
- **Token embeddings are the most critical** - never quantize them
- **Start conservative** (Q6_K to Q8_0) and gradually increase compression
- **Test model quality** after quantization with your specific use case
- **The skip list is your friend** - it protects critical tensors automatically
