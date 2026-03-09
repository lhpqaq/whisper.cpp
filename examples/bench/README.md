# whisper.cpp/examples/bench

A benchmarking tool for measuring whisper.cpp inference performance on your device. The default mode now runs a larger,
configurable synthetic end-to-end workload: encoder passes over synthetic mel frames plus controlled prompt, batched,
and autoregressive decoder workloads. This makes it easier to collect deployment-style latency and throughput data when
the old fixed benchmark was too small.

Benchmark results are tracked in the following Github issue: https://github.com/ggml-org/whisper.cpp/issues/89

```bash
# run the end-to-end benchmark on a larger synthetic audio window
$ ./build/bin/whisper-bench -m ./models/ggml-small.en.bin -t 4 \
    --audio-ms 120000 --prompt-tokens 256 --prompt-runs 32 \
    --batch-tokens 16 --batch-runs 128 --gen-steps 512

# customize the ggml_mul_mat matrix sizes too
$ ./build/bin/whisper-bench -w 2 --mul-mat-sizes 512,1024,2048,4096,8192

# force a specific mel length directly
$ ./build/bin/whisper-bench --mel-frames 30000 --gen-steps 1024

whisper_model_load: loading model from './models/ggml-small.en.bin'
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 768
whisper_model_load: n_audio_head  = 12
whisper_model_load: n_audio_layer = 12
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 768
whisper_model_load: n_text_head   = 12
whisper_model_load: n_text_layer  = 12
whisper_model_load: n_mels        = 80
whisper_model_load: f16           = 1
whisper_model_load: type          = 3
whisper_model_load: mem_required  = 1048.00 MB
whisper_model_load: adding 1607 extra tokens
whisper_model_load: ggml ctx size = 533.05 MB
whisper_model_load: memory size =    68.48 MB 
whisper_model_load: model size  =   464.44 MB

benchmark_summary:
  encode_total       =  8432.10 ms |   140.53 ms/run |  4268.12 frames/s
  prompt_total       =  2107.22 ms |  3885.44 tok/s
  batch_total        =  1543.18 ms | 13271.07 tok/s
  generation_total   =   911.44 ms |   561.75 tok/s
  decode_total       =  4561.84 ms |  4695.25 tok/s
  total_end_to_end   = 12998.67 ms | RTF = 0.1083

whisper_print_timings:     load time =   240.82 ms
whisper_print_timings:    encode time =  8430.00 ms /    60 runs (  140.50 ms per run)
whisper_print_timings:    decode time =  4560.00 ms /   512 runs (    8.91 ms per run)
whisper_print_timings:    total time = 12998.67 ms

system_info: n_threads = 4 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | 

If you wish, you can submit these results here:

  https://github.com/ggml-org/whisper.cpp/issues/89

Please include the following information:

  - CPU model
  - Operating system
  - Compiler

```

Key flags for `--what 0`:

- `--audio-ms`: requested synthetic audio duration, converted to mel frames.
- `--mel-frames`: direct control over mel length; useful when you want exact encoder workload size.
- `--prompt-tokens` / `--prompt-runs`: prompt-processing decoder workload.
- `--batch-tokens` / `--batch-runs`: batched decoder workload.
- `--gen-steps`: autoregressive one-token decode steps.

Key flags for `--what 2`:

- `--mul-mat-size`: add one matrix size.
- `--mul-mat-sizes`: provide a comma-separated list of matrix sizes.
- `--mul-mat-min-ms`: minimum measurement time per type/size pair.
- `--mul-mat-max-runs`: cap the number of measured runs.
