#include "whisper.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct whisper_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t what = 0; // 0 - whisper end-to-end, 1 - memcpy, 2 - ggml_mul_mat

    int32_t warmup = 1;

    int32_t audio_ms = 0;
    int32_t mel_frames = 0;

    int32_t prompt_tokens = 256;
    int32_t prompt_runs   = 16;
    int32_t batch_tokens  = 5;
    int32_t batch_runs    = 64;
    int32_t gen_steps     = 256;

    int32_t mul_mat_max_runs = 128;
    double  mul_mat_min_ms   = 1000.0;

    std::string model = "models/ggml-base.en.bin";

    bool use_gpu    = true;
    bool flash_attn = true;
    int32_t gpu_device = 0;

    std::vector<int32_t> mul_mat_sizes = { 64, 128, 256, 512, 1024, 2048, 4096 };
};

static void whisper_print_usage(char ** argv, const whisper_params & params);

static std::string join_sizes(const std::vector<int32_t> & values) {
    std::ostringstream oss;

    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }

    return oss.str();
}

static bool parse_int_list(const std::string & arg, std::vector<int32_t> & out) {
    std::stringstream ss(arg);
    std::string item;

    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());

        if (item.empty()) {
            continue;
        }

        try {
            const int32_t value = std::stoi(item);
            if (value <= 0) {
                return false;
            }
            out.push_back(value);
        } catch (...) {
            return false;
        }
    }

    return !out.empty();
}

static void normalize_sizes(std::vector<int32_t> & values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

static std::vector<float> make_synthetic_mel(int n_mels, int n_frames) {
    std::vector<float> mel((size_t) n_mels * (size_t) n_frames);

    for (int j = 0; j < n_mels; ++j) {
        for (int i = 0; i < n_frames; ++i) {
            const float x = 0.30f * sinf(0.011f * (float) (i + 1) * (float) (j + 1))
                          + 0.15f * cosf(0.003f * (float) (i + 1) * (float) (j + 7));
            mel[(size_t) j * (size_t) n_frames + (size_t) i] = x;
        }
    }

    return mel;
}

static std::vector<whisper_token> make_synthetic_tokens(struct whisper_context * ctx, int n_tokens) {
    const int n_vocab = whisper_model_n_vocab(ctx);
    std::vector<whisper_token> tokens((size_t) std::max(1, n_tokens));

    tokens[0] = whisper_token_sot(ctx);
    for (int i = 1; i < n_tokens; ++i) {
        tokens[(size_t) i] = (whisper_token) ((i * 9973) % n_vocab);
    }

    return tokens;
}

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    bool custom_mul_mat_sizes = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argv, params);
            std::exit(0);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-w" || arg == "--what") {
            params.what = std::atoi(argv[++i]);
        } else if (arg == "-ng" || arg == "--no-gpu") {
            params.use_gpu = false;
        } else if (arg == "-fa" || arg == "--flash-attn") {
            params.flash_attn = true;
        } else if (arg == "-nfa" || arg == "--no-flash-attn") {
            params.flash_attn = false;
        } else if (arg == "-dev" || arg == "--device") {
            params.gpu_device = std::stoi(argv[++i]);
        } else if (arg == "--warmup") {
            params.warmup = std::stoi(argv[++i]);
        } else if (arg == "--audio-ms") {
            params.audio_ms = std::stoi(argv[++i]);
        } else if (arg == "--mel-frames") {
            params.mel_frames = std::stoi(argv[++i]);
        } else if (arg == "--prompt-tokens") {
            params.prompt_tokens = std::stoi(argv[++i]);
        } else if (arg == "--prompt-runs") {
            params.prompt_runs = std::stoi(argv[++i]);
        } else if (arg == "--batch-tokens") {
            params.batch_tokens = std::stoi(argv[++i]);
        } else if (arg == "--batch-runs") {
            params.batch_runs = std::stoi(argv[++i]);
        } else if (arg == "--gen-steps") {
            params.gen_steps = std::stoi(argv[++i]);
        } else if (arg == "--mul-mat-size") {
            if (!custom_mul_mat_sizes) {
                params.mul_mat_sizes.clear();
                custom_mul_mat_sizes = true;
            }
            params.mul_mat_sizes.push_back(std::stoi(argv[++i]));
        } else if (arg == "--mul-mat-sizes") {
            std::vector<int32_t> parsed;
            if (!parse_int_list(argv[++i], parsed)) {
                fprintf(stderr, "error: failed to parse --mul-mat-sizes\n");
                return false;
            }
            if (!custom_mul_mat_sizes) {
                params.mul_mat_sizes.clear();
                custom_mul_mat_sizes = true;
            }
            params.mul_mat_sizes.insert(params.mul_mat_sizes.end(), parsed.begin(), parsed.end());
        } else if (arg == "--mul-mat-min-ms") {
            params.mul_mat_min_ms = std::atof(argv[++i]);
        } else if (arg == "--mul-mat-max-runs") {
            params.mul_mat_max_runs = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argv, params);
            std::exit(0);
        }
    }

    params.n_threads = std::max(1, params.n_threads);
    params.warmup = std::max(0, params.warmup);
    params.audio_ms = std::max(0, params.audio_ms);
    params.mel_frames = std::max(0, params.mel_frames);
    params.prompt_tokens = std::max(0, params.prompt_tokens);
    params.prompt_runs = std::max(0, params.prompt_runs);
    params.batch_tokens = std::max(0, params.batch_tokens);
    params.batch_runs = std::max(0, params.batch_runs);
    params.gen_steps = std::max(0, params.gen_steps);
    params.mul_mat_max_runs = std::max(1, params.mul_mat_max_runs);
    params.mul_mat_min_ms = std::max(0.0, params.mul_mat_min_ms);

    if (params.mul_mat_sizes.empty()) {
        fprintf(stderr, "error: no valid mul_mat sizes configured\n");
        return false;
    }

    for (int32_t size : params.mul_mat_sizes) {
        if (size <= 0) {
            fprintf(stderr, "error: mul_mat sizes must be positive\n");
            return false;
        }
    }

    normalize_sizes(params.mul_mat_sizes);

    return true;
}

static void whisper_print_usage(char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help                 show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N            [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME          [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -w N,     --what N               [%-7d] what to benchmark:\n", params.what);
    fprintf(stderr, "                                    %-7s  0 - whisper end-to-end (synthetic mel + controlled decode)\n", "");
    fprintf(stderr, "                                    %-7s  1 - memcpy\n", "");
    fprintf(stderr, "                                    %-7s  2 - ggml_mul_mat\n", "");
    fprintf(stderr, "  -ng,      --no-gpu               [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -dev N,   --device N             [%-7d] GPU device index\n", params.gpu_device);
    fprintf(stderr, "  -fa,      --flash-attn           [%-7s] enable flash attention\n", params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,     --no-flash-attn        [%-7s] disable flash attention\n", params.flash_attn ? "false" : "true");
    fprintf(stderr, "\n");
    fprintf(stderr, "whisper end-to-end benchmark controls (used with --what 0):\n");
    fprintf(stderr, "            --warmup N             [%-7d] warmup passes for each stage\n", params.warmup);
    fprintf(stderr, "            --audio-ms N           [%-7d] requested synthetic audio length in ms (converted to mel frames)\n", params.audio_ms);
    fprintf(stderr, "            --mel-frames N         [%-7d] requested synthetic mel length (overrides default when > 0)\n", params.mel_frames);
    fprintf(stderr, "            --prompt-tokens N      [%-7d] tokens per prompt-processing decode\n", params.prompt_tokens);
    fprintf(stderr, "            --prompt-runs N        [%-7d] number of prompt-processing decodes\n", params.prompt_runs);
    fprintf(stderr, "            --batch-tokens N       [%-7d] tokens per batched decode\n", params.batch_tokens);
    fprintf(stderr, "            --batch-runs N         [%-7d] number of batched decodes\n", params.batch_runs);
    fprintf(stderr, "            --gen-steps N          [%-7d] number of autoregressive one-token decode steps\n", params.gen_steps);
    fprintf(stderr, "\n");
    fprintf(stderr, "ggml_mul_mat benchmark controls (used with --what 2):\n");
    fprintf(stderr, "            --mul-mat-size N       add a matrix size N (repeatable)\n");
    fprintf(stderr, "            --mul-mat-sizes LIST   [%-7s] comma-separated matrix sizes\n", join_sizes(params.mul_mat_sizes).c_str());
    fprintf(stderr, "            --mul-mat-min-ms N     [%-7.2f] minimum measurement time per type and size\n", params.mul_mat_min_ms);
    fprintf(stderr, "            --mul-mat-max-runs N   [%-7d] maximum runs per type and size\n", params.mul_mat_max_runs);
    fprintf(stderr, "\n");
}

template<typename Fn>
static double measure_ms(Fn && fn) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int whisper_bench_full(const whisper_params & params) {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.gpu_device = params.gpu_device;

    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            params.n_threads,
            std::thread::hardware_concurrency(),
            whisper_print_system_info());

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    const int n_mels      = whisper_model_n_mels(ctx);
    const int n_audio_ctx = whisper_model_n_audio_ctx(ctx);
    const int n_text_ctx  = whisper_model_n_text_ctx(ctx);

    int mel_frames = params.mel_frames;
    if (params.audio_ms > 0) {
        mel_frames = std::max(mel_frames, (params.audio_ms + 9) / 10);
    }
    if (mel_frames <= 0) {
        mel_frames = 2 * n_audio_ctx;
    }

    const int encode_window = 2 * n_audio_ctx;
    const int encode_runs   = std::max(1, (mel_frames + encode_window - 1) / encode_window);
    const int effective_mel_frames = encode_runs * encode_window;

    const int prompt_tokens = std::min(std::max(0, params.prompt_tokens), n_text_ctx);
    const int batch_tokens  = std::min(std::max(0, params.batch_tokens),  n_text_ctx);
    const int gen_steps     = std::min(std::max(0, params.gen_steps), std::max(0, n_text_ctx - 1));

    std::vector<float> mel = make_synthetic_mel(n_mels, mel_frames);
    if (int ret = whisper_set_mel(ctx, mel.data(), mel_frames, n_mels)) {
        fprintf(stderr, "error: failed to set mel: %d\n", ret);
        whisper_free(ctx);
        return 3;
    }

    const std::vector<whisper_token> prompt = make_synthetic_tokens(ctx, std::max(1, prompt_tokens));
    const std::vector<whisper_token> batch  = make_synthetic_tokens(ctx, std::max(1, batch_tokens));
    const std::vector<whisper_token> single = make_synthetic_tokens(ctx, 1);

    for (int w = 0; w < params.warmup; ++w) {
        for (int offset = 0; offset < mel_frames; offset += encode_window) {
            if (int ret = whisper_encode(ctx, offset, params.n_threads)) {
                fprintf(stderr, "error: failed to warm up encoder: %d\n", ret);
                whisper_free(ctx);
                return 4;
            }
        }

        if (prompt_tokens > 0 && prompt_tokens <= n_text_ctx) {
            if (int ret = whisper_decode(ctx, prompt.data(), prompt_tokens, 0, params.n_threads)) {
                fprintf(stderr, "error: failed to warm up prompt decode: %d\n", ret);
                whisper_free(ctx);
                return 4;
            }
        }

        if (batch_tokens > 0 && batch_tokens <= n_text_ctx) {
            if (int ret = whisper_decode(ctx, batch.data(), batch_tokens, 0, params.n_threads)) {
                fprintf(stderr, "error: failed to warm up batched decode: %d\n", ret);
                whisper_free(ctx);
                return 4;
            }
        }

        const int warmup_gen_steps = std::min(gen_steps, 16);
        for (int i = 0; i < warmup_gen_steps; ++i) {
            if (int ret = whisper_decode(ctx, single.data(), 1, i, params.n_threads)) {
                fprintf(stderr, "error: failed to warm up autoregressive decode: %d\n", ret);
                whisper_free(ctx);
                return 4;
            }
        }
    }

    whisper_reset_timings(ctx);

    double encode_ms = 0.0;
    double gen_ms    = 0.0;
    double batch_ms  = 0.0;
    double prompt_ms = 0.0;
    const double total_ms = measure_ms([&]() {
        for (int offset = 0; offset < mel_frames; offset += encode_window) {
            encode_ms += measure_ms([&]() {
                if (int ret = whisper_encode(ctx, offset, params.n_threads)) {
                    fprintf(stderr, "error: failed to encode: %d\n", ret);
                    std::exit(4);
                }
            });
        }

        for (int i = 0; i < gen_steps; ++i) {
            gen_ms += measure_ms([&]() {
                if (int ret = whisper_decode(ctx, single.data(), 1, i, params.n_threads)) {
                    fprintf(stderr, "error: failed to decode: %d\n", ret);
                    std::exit(4);
                }
            });
        }

        for (int i = 0; i < params.batch_runs; ++i) {
            if (batch_tokens == 0) {
                break;
            }

            batch_ms += measure_ms([&]() {
                if (int ret = whisper_decode(ctx, batch.data(), batch_tokens, 0, params.n_threads)) {
                    fprintf(stderr, "error: failed to decode: %d\n", ret);
                    std::exit(4);
                }
            });
        }

        for (int i = 0; i < params.prompt_runs; ++i) {
            if (prompt_tokens == 0) {
                break;
            }

            prompt_ms += measure_ms([&]() {
                if (int ret = whisper_decode(ctx, prompt.data(), prompt_tokens, 0, params.n_threads)) {
                    fprintf(stderr, "error: failed to decode: %d\n", ret);
                    std::exit(4);
                }
            });
        }
    });

    const int64_t total_decode_tokens = (int64_t) gen_steps
                                      + (int64_t) params.batch_runs * batch_tokens
                                      + (int64_t) params.prompt_runs * prompt_tokens;
    const double effective_audio_s = 0.01 * effective_mel_frames;

    fprintf(stderr, "\nbenchmark_config:\n");
    fprintf(stderr, "  model              = %s\n", params.model.c_str());
    fprintf(stderr, "  requested_audio_ms = %d\n", params.audio_ms > 0 ? params.audio_ms : mel_frames * 10);
    fprintf(stderr, "  requested_mel      = %d frames\n", mel_frames);
    fprintf(stderr, "  audio_ctx          = %d frames\n", n_audio_ctx);
    fprintf(stderr, "  encode_window      = %d frames per encode\n", encode_window);
    fprintf(stderr, "  encode_runs        = %d\n", encode_runs);
    fprintf(stderr, "  prompt             = %d tokens x %d runs\n", prompt_tokens, params.prompt_runs);
    fprintf(stderr, "  batch              = %d tokens x %d runs\n", batch_tokens, params.batch_runs);
    fprintf(stderr, "  autoregressive     = %d steps\n", gen_steps);

    fprintf(stderr, "\nbenchmark_summary:\n");
    fprintf(stderr, "  encode_total       = %8.2f ms | %8.2f ms/run | %8.2f frames/s\n",
            encode_ms,
            encode_runs > 0 ? encode_ms / encode_runs : 0.0,
            encode_ms > 0.0 ? 1000.0 * effective_mel_frames / encode_ms : 0.0);
    fprintf(stderr, "  prompt_total       = %8.2f ms | %8.2f tok/s\n",
            prompt_ms,
            prompt_ms > 0.0 ? 1000.0 * (double) prompt_tokens * params.prompt_runs / prompt_ms : 0.0);
    fprintf(stderr, "  batch_total        = %8.2f ms | %8.2f tok/s\n",
            batch_ms,
            batch_ms > 0.0 ? 1000.0 * (double) batch_tokens * params.batch_runs / batch_ms : 0.0);
    fprintf(stderr, "  generation_total   = %8.2f ms | %8.2f tok/s\n",
            gen_ms,
            gen_ms > 0.0 ? 1000.0 * (double) gen_steps / gen_ms : 0.0);
    fprintf(stderr, "  decode_total       = %8.2f ms | %8.2f tok/s\n",
            prompt_ms + batch_ms + gen_ms,
            (prompt_ms + batch_ms + gen_ms) > 0.0 ? 1000.0 * (double) total_decode_tokens / (prompt_ms + batch_ms + gen_ms) : 0.0);
    fprintf(stderr, "  total_end_to_end   = %8.2f ms | RTF = %.4f\n",
            total_ms,
            effective_audio_s > 0.0 ? (total_ms / 1000.0) / effective_audio_s : 0.0);

    whisper_print_timings(ctx);
    whisper_free(ctx);

    fprintf(stderr, "\n");
    fprintf(stderr, "If you wish, you can submit these results here:\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  https://github.com/ggerganov/whisper.cpp/issues/89\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Please include the following information:\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  - CPU model\n");
    fprintf(stderr, "  - Operating system\n");
    fprintf(stderr, "  - Compiler\n");
    fprintf(stderr, "\n");

    return 0;
}

static int whisper_bench_mul_mat(const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            params.n_threads,
            std::thread::hardware_concurrency(),
            whisper_print_system_info());
    fprintf(stderr, "mul_mat_sizes: %s\n", join_sizes(params.mul_mat_sizes).c_str());
    fprintf(stderr, "mul_mat_min_ms: %.2f | mul_mat_max_runs: %d\n", params.mul_mat_min_ms, params.mul_mat_max_runs);

    struct type_desc {
        ggml_type type;
        const char * name;
    };

    const std::vector<type_desc> types = {
        { GGML_TYPE_Q4_0, "Q4_0" },
        { GGML_TYPE_Q4_1, "Q4_1" },
        { GGML_TYPE_Q5_0, "Q5_0" },
        { GGML_TYPE_Q5_1, "Q5_1" },
        { GGML_TYPE_Q8_0, "Q8_0" },
        { GGML_TYPE_F16,  "F16"  },
        { GGML_TYPE_F32,  "F32"  },
    };

    for (int32_t N : params.mul_mat_sizes) {
        const size_t tensor_bytes = 3llu * (size_t) N * (size_t) N * sizeof(float)
                                  + 3 * ggml_tensor_overhead()
                                  + ggml_graph_overhead();
        std::vector<uint8_t> buf(tensor_bytes);

        for (size_t i = 0; i < buf.size(); ++i) {
            buf[i] = (uint8_t) i;
        }

        fprintf(stderr, "\n%6d x %6d\n", N, N);
        fprintf(stderr, "  %-6s %-14s %-10s\n", "type", "gflops", "runs");

        for (const auto & desc : types) {
            struct ggml_init_params gparams = {
                /*.mem_size   =*/ buf.size(),
                /*.mem_buffer =*/ buf.data(),
                /*.no_alloc   =*/ false,
            };

            struct ggml_context * ctx0 = ggml_init(gparams);
            struct ggml_tensor * a = ggml_new_tensor_2d(ctx0, desc.type, N, N);
            struct ggml_tensor * b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, N);
            struct ggml_tensor * c = ggml_mul_mat(ctx0, a, b);
            struct ggml_cgraph * gf = ggml_new_graph(ctx0);
            ggml_build_forward_expand(gf, c);

            ggml_graph_compute_with_ctx(ctx0, gf, params.n_threads);

            double total_s = 0.0;
            int runs = 0;
            while (runs < params.mul_mat_max_runs) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                ggml_graph_compute_with_ctx(ctx0, gf, params.n_threads);
                const auto t1 = std::chrono::high_resolution_clock::now();

                total_s += std::chrono::duration<double>(t1 - t0).count();
                ++runs;

                if (runs >= 3 && total_s * 1000.0 >= params.mul_mat_min_ms) {
                    break;
                }
            }

            const double gflops = total_s > 0.0 ? (2.0 * (double) N * (double) N * (double) N * runs) / total_s / 1e9 : 0.0;
            fprintf(stderr, "  %-6s %10.2f GFLOPS %4d\n", desc.name, gflops, runs);

            ggml_free(ctx0);
        }
    }

    return 0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;
    if (!whisper_params_parse(argc, argv, params)) {
        return 1;
    }

    int ret = -1;

    switch (params.what) {
        case 0: ret = whisper_bench_full(params);                break;
        case 1: ret = whisper_bench_memcpy(params.n_threads);    break;
        case 2: ret = whisper_bench_mul_mat(params);             break;
        default:
            fprintf(stderr, "error: unknown benchmark: %d\n", params.what);
            break;
    }

    return ret;
}
