# @whisper-cpp-node/win32-x64

Native binary for Windows x64 with full GPU acceleration.

This package is automatically installed by `whisper-cpp-node` on compatible platforms.

**Do not install this package directly.** Install `whisper-cpp-node` instead.

## Requirements

- Windows 10/11 x64
- Node.js 18+

## Acceleration Features

This binary includes:

| Feature | Description |
|---------|-------------|
| **Vulkan** | GPU acceleration for decoder (Intel/AMD/NVIDIA) |
| **OpenVINO** | Intel CPU/GPU encoder acceleration |
| **AVX2/FMA** | Optimized CPU instructions |
| **Flash Attention** | Faster attention computation |

### Usage

```javascript
const ctx = createWhisperContext({
  model: "./models/ggml-large-v3-turbo-q5_0.bin",
  use_gpu: true,           // Vulkan decoder
  flash_attn: true,        // Flash attention
  use_openvino: true,      // OpenVINO encoder (optional)
  openvino_device: "GPU",  // or "CPU"
});
```

## License

MIT
