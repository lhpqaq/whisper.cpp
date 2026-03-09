# Quant Speed Benchmark

This example benchmarks the per-quantization matvec speed that Chapter 3 uses as the hardware-aware latency term.

- CPU runs `ggml_mul_mat` on the CPU backend with a single-vector input, so the hot path stays on the quantized dot-product / GEMV family used by deployment.
- CUDA runs the same `ggml_mul_mat` batch-1 matvec on the CUDA backend, which is shaped to prefer the quantized matvec path on supported GPUs.
- The program reports per-shape timings and geometric-mean latency scales relative to `f16`, which can be pasted into the Chapter 3 scoring scripts.

Example:

```bash
./bin/whisper-quant-speed --backend both --shape 512x512 --shape 1280x5120 --threads 8
```

The final summary includes ready-to-use flags like:

```text
--latency-scale-8 <value> --latency-scale-5 <value> --latency-scale-4 <value> --latency-scale-2 <value>
```
