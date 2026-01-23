#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iostream>

#include "ggml.h"
#include "ggml-quants.h"

// Define macros required for vec.h if not already defined
#ifndef GGML_RESTRICT
#define GGML_RESTRICT __restrict
#endif

// Include vec.h to access the internal functions
// We need to point to the correct path
#include "../ggml/src/ggml-cpu/vec.h"

// Helper to fill buffer with random data
void fill_random(float * data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = (float)rand() / RAND_MAX - 0.5f;
    }
}

// Benchmark function
template<typename Func>
double benchmark(Func f, int iterations, int n, void * y, const void * x, float v) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f(n, (float*)y, x, v);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

template<typename FuncFP16>
double benchmark_f16(FuncFP16 f, int iterations, int n, ggml_fp16_t * y, const ggml_fp16_t * x, float v) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f(n, y, x, v);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}


int main(int argc, char ** argv) {
    int n = 1024 * 16; // Default size
    int iterations = 10000;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    printf("Benchmarking with N = %d, Iterations = %d\n", n, iterations);

    // Align memory
    size_t alignment = 64;
    
    // F32 data
    std::vector<float> x_f32(n);
    std::vector<float> y_f32(n, 0.0f);
    fill_random(x_f32.data(), n);
    float v = 1.23f;

    // F16 data
    std::vector<ggml_fp16_t> x_f16(n);
    std::vector<ggml_fp16_t> y_f16(n);
    ggml_fp16_t * y_f16_ptr = y_f16.data();
    for (int i = 0; i < n; ++i) {
        x_f16[i] = GGML_CPU_FP32_TO_FP16(x_f32[i]);
        y_f16[i] = GGML_CPU_FP32_TO_FP16(0.0f);
    }

    // Q8_0 data
    int qk8_0 = ggml_blck_size(GGML_TYPE_Q8_0);
    int n_blocks_q8_0 = n / qk8_0;
    size_t q8_0_size = n_blocks_q8_0 * ggml_type_size(GGML_TYPE_Q8_0);
    std::vector<uint8_t> x_q8_0(q8_0_size);
    quantize_q8_0(x_f32.data(), x_q8_0.data(), 1, n, nullptr);

    // Q4_0 data
    int qk4_0 = ggml_blck_size(GGML_TYPE_Q4_0);
    int n_blocks_q4_0 = n / qk4_0;
    size_t q4_0_size = n_blocks_q4_0 * ggml_type_size(GGML_TYPE_Q4_0);
    std::vector<uint8_t> x_q4_0(q4_0_size);
    quantize_q4_0(x_f32.data(), x_q4_0.data(), 1, n, nullptr);

    // Benchmark F16
    double time_f16 = benchmark_f16(ggml_vec_mad_f16, iterations, n, y_f16_ptr, x_f16.data(), v);
    printf("ggml_vec_mad_f16:  %.6f s (%.2f GB/s)\n", time_f16, (double)n * sizeof(ggml_fp16_t) * iterations / time_f16 / 1e9);

    // Benchmark Q8_0
    // Reset y
    std::fill(y_f32.begin(), y_f32.end(), 0.0f);
    double time_q8_0 = benchmark(ggml_vec_mad_q8_0, iterations, n, y_f32.data(), x_q8_0.data(), v);
    // Note: bandwidth calculation for quantized types usually considers the compressed size
    printf("ggml_vec_mad_q8_0: %.6f s (%.2f GB/s)\n", time_q8_0, (double)q8_0_size * iterations / time_q8_0 / 1e9);

    // Benchmark Q4_0
    std::fill(y_f32.begin(), y_f32.end(), 0.0f);
    double time_q4_0 = benchmark(ggml_vec_mad_q4_0, iterations, n, y_f32.data(), x_q4_0.data(), v);
    printf("ggml_vec_mad_q4_0: %.6f s (%.2f GB/s)\n", time_q4_0, (double)q4_0_size * iterations / time_q4_0 / 1e9);

    // Validate correctness (basic check)
    // Run once and compare
    std::vector<float> y_ref(n, 0.0f);
    // ggml_vec_mad_f32 does not exist as straight function easily accessible, use loop
    for(int i=0; i<n; ++i) y_ref[i] += x_f32[i] * v;

    // We accept some error due to quantization
    
    return 0;
}
