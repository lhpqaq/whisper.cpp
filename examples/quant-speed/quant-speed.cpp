#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct shape_spec {
    int64_t rows;
    int64_t cols;

    std::string to_string() const {
        return std::to_string(rows) + "x" + std::to_string(cols);
    }
};

struct quant_speed_params {
    int32_t threads = std::max<int32_t>(1, std::min<int32_t>(4, (int32_t) std::thread::hardware_concurrency()));
    int32_t warmup = 2;
    int32_t min_iter = 5;
    int32_t max_iter = 200;
    int32_t cuda_device = 0;
    double min_seconds = 0.25;
    uint32_t seed = 1337;
    std::string backend = "both";
    std::string csv_path;
    std::vector<shape_spec> shapes;
    std::vector<ggml_type> types;
};

struct backend_handle {
    ggml_backend_t backend = nullptr;
    std::string key;
    std::string name;
    std::string description;
};

struct benchmark_result {
    std::string backend;
    std::string device;
    std::string implementation;
    shape_spec shape = { 0, 0 };
    ggml_type type = GGML_TYPE_COUNT;
    int iterations = 0;
    double total_seconds = 0.0;
    double avg_ms = 0.0;
    double gops = 0.0;
    double scale_vs_f16 = std::numeric_limits<double>::quiet_NaN();
    double checksum = 0.0;
    bool skipped = false;
    std::string message;
};

struct shape_inputs {
    std::vector<float> weights;
    std::vector<float> input;
};

static std::vector<shape_spec> default_shapes() {
    return {
        {  512,  512 },
        {  512, 2048 },
        { 1280, 1280 },
        { 1280, 5120 },
    };
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
    });
    return s;
}

static std::string join_strings(const std::vector<std::string> & values, const std::string & sep) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << sep;
        }
        oss << values[i];
    }
    return oss.str();
}

static std::string cli_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q5_0: return "q5_0";
        case GGML_TYPE_Q5_1: return "q5_1";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_1: return "q4_1";
        case GGML_TYPE_Q2_K: return "q2_k";
        case GGML_TYPE_Q3_K: return "q3_k";
        case GGML_TYPE_Q4_K: return "q4_k";
        case GGML_TYPE_Q5_K: return "q5_k";
        case GGML_TYPE_Q6_K: return "q6_k";
        default:             return ggml_type_name(type);
    }
}

static std::string cpu_kernel_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "ggml_vec_dot_f32";
        case GGML_TYPE_F16:  return "ggml_vec_dot_f16";
        case GGML_TYPE_Q8_0: return "ggml_vec_dot_q8_0_q8_0";
        case GGML_TYPE_Q5_0: return "ggml_vec_dot_q5_0_q8_0";
        case GGML_TYPE_Q5_1: return "ggml_vec_dot_q5_1_q8_1";
        case GGML_TYPE_Q4_0: return "ggml_vec_dot_q4_0_q8_0";
        case GGML_TYPE_Q4_1: return "ggml_vec_dot_q4_1_q8_1";
        case GGML_TYPE_Q2_K: return "ggml_vec_dot_q2_K_q8_K";
        case GGML_TYPE_Q3_K: return "ggml_vec_dot_q3_K_q8_K";
        case GGML_TYPE_Q4_K: return "ggml_vec_dot_q4_K_q8_K";
        case GGML_TYPE_Q5_K: return "ggml_vec_dot_q5_K_q8_K";
        case GGML_TYPE_Q6_K: return "ggml_vec_dot_q6_K_q8_K";
        default:             return "ggml_mul_mat";
    }
}

static bool try_parse_type(const std::string & token, ggml_type & type) {
    const std::string key = lower_copy(token);

    if (key == "f32")  { type = GGML_TYPE_F32;  return true; }
    if (key == "f16")  { type = GGML_TYPE_F16;  return true; }
    if (key == "q8_0") { type = GGML_TYPE_Q8_0; return true; }
    if (key == "q5_0") { type = GGML_TYPE_Q5_0; return true; }
    if (key == "q5_1") { type = GGML_TYPE_Q5_1; return true; }
    if (key == "q4_0") { type = GGML_TYPE_Q4_0; return true; }
    if (key == "q4_1") { type = GGML_TYPE_Q4_1; return true; }
    if (key == "q2_k") { type = GGML_TYPE_Q2_K; return true; }
    if (key == "q3_k") { type = GGML_TYPE_Q3_K; return true; }
    if (key == "q4_k") { type = GGML_TYPE_Q4_K; return true; }
    if (key == "q5_k") { type = GGML_TYPE_Q5_K; return true; }
    if (key == "q6_k") { type = GGML_TYPE_Q6_K; return true; }

    return false;
}

static bool parse_shape(const std::string & value, shape_spec & shape) {
    const size_t pos = value.find('x');
    const size_t pos_upper = value.find('X');
    const size_t split = pos != std::string::npos ? pos : pos_upper;

    if (split == std::string::npos || split == 0 || split + 1 >= value.size()) {
        return false;
    }

    try {
        shape.rows = std::stoll(value.substr(0, split));
        shape.cols = std::stoll(value.substr(split + 1));
    } catch (...) {
        return false;
    }

    return shape.rows > 0 && shape.cols > 0;
}

static std::vector<std::string> split_csv(const std::string & value) {
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }

    return parts;
}

static std::string default_shapes_string() {
    std::vector<std::string> parts;
    for (const auto & shape : default_shapes()) {
        parts.push_back(shape.to_string());
    }
    return join_strings(parts, ", ");
}

static std::string default_types_string() {
    return "f16,q8_0,q5_0,q4_0,q2_k";
}

static void print_usage(const char * argv0, const quant_speed_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv0);
    fprintf(stderr, "\n");
    fprintf(stderr, "Compute per-quantization matvec speed terms for Chapter 3.\n");
    fprintf(stderr, "The benchmark uses ggml_mul_mat with a batch-1 input vector so the hot path stays on\n");
    fprintf(stderr, "backend-native quantized matvec kernels.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");
    fprintf(stderr, "  -b, --backend STR          [%-7s] cpu, cuda, or both\n", params.backend.c_str());
    fprintf(stderr, "  -t, --threads N            [%-7d] CPU threads for the CPU backend\n", params.threads);
    fprintf(stderr, "      --cuda-device N        [%-7d] CUDA device index when --backend includes cuda\n", params.cuda_device);
    fprintf(stderr, "  -s, --shape ROWSxCOLS      add a shape to benchmark (repeatable)\n");
    fprintf(stderr, "                           default shapes: %s\n", default_shapes_string().c_str());
    fprintf(stderr, "      --types LIST           [%-7s] comma-separated types\n", default_types_string().c_str());
    fprintf(stderr, "      --warmup N             [%-7d] warmup iterations per run\n", params.warmup);
    fprintf(stderr, "      --min-iter N           [%-7d] minimum measured iterations per run\n", params.min_iter);
    fprintf(stderr, "      --max-iter N           [%-7d] maximum measured iterations per run\n", params.max_iter);
    fprintf(stderr, "      --min-seconds SEC      [%-7.2f] minimum measured seconds per run\n", params.min_seconds);
    fprintf(stderr, "      --csv PATH             save per-run results as CSV\n");
    fprintf(stderr, "      --seed N               [%-7u] random seed\n", params.seed);
    fprintf(stderr, "\n");
    fprintf(stderr, "example:\n");
    fprintf(stderr, "  %s --backend both --shape 512x512 --shape 1280x5120 --threads 8\n", argv0);
    fprintf(stderr, "\n");
}

static bool parse_params(int argc, char ** argv, quant_speed_params & params) {
    params.shapes = default_shapes();
    params.types  = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0, GGML_TYPE_Q2_K };

    bool custom_shapes = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0], params);
            std::exit(0);
        } else if (arg == "-b" || arg == "--backend") {
            if (++i >= argc) {
                return false;
            }
            params.backend = lower_copy(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.threads = std::max(1, std::atoi(argv[i]));
        } else if (arg == "--cuda-device") {
            if (++i >= argc) {
                return false;
            }
            params.cuda_device = std::max(0, std::atoi(argv[i]));
        } else if (arg == "-s" || arg == "--shape") {
            if (++i >= argc) {
                return false;
            }
            shape_spec shape = { 0, 0 };
            if (!parse_shape(argv[i], shape)) {
                fprintf(stderr, "error: invalid shape: %s\n", argv[i]);
                return false;
            }
            if (!custom_shapes) {
                params.shapes.clear();
                custom_shapes = true;
            }
            params.shapes.push_back(shape);
        } else if (arg == "--types") {
            if (++i >= argc) {
                return false;
            }
            params.types.clear();
            for (const auto & token : split_csv(argv[i])) {
                ggml_type type = GGML_TYPE_COUNT;
                if (!try_parse_type(token, type)) {
                    fprintf(stderr, "error: unsupported type: %s\n", token.c_str());
                    return false;
                }
                params.types.push_back(type);
            }
        } else if (arg == "--warmup") {
            if (++i >= argc) {
                return false;
            }
            params.warmup = std::max(0, std::atoi(argv[i]));
        } else if (arg == "--min-iter") {
            if (++i >= argc) {
                return false;
            }
            params.min_iter = std::max(1, std::atoi(argv[i]));
        } else if (arg == "--max-iter") {
            if (++i >= argc) {
                return false;
            }
            params.max_iter = std::max(1, std::atoi(argv[i]));
        } else if (arg == "--min-seconds") {
            if (++i >= argc) {
                return false;
            }
            params.min_seconds = std::max(0.0, std::atof(argv[i]));
        } else if (arg == "--csv") {
            if (++i >= argc) {
                return false;
            }
            params.csv_path = argv[i];
        } else if (arg == "--seed") {
            if (++i >= argc) {
                return false;
            }
            params.seed = (uint32_t) std::strtoul(argv[i], nullptr, 10);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (params.backend != "cpu" && params.backend != "cuda" && params.backend != "both") {
        fprintf(stderr, "error: --backend must be cpu, cuda, or both\n");
        return false;
    }

    if (params.types.empty()) {
        fprintf(stderr, "error: --types produced an empty type set\n");
        return false;
    }

    if (std::find(params.types.begin(), params.types.end(), GGML_TYPE_F16) == params.types.end()) {
        params.types.insert(params.types.begin(), GGML_TYPE_F16);
    }

    std::vector<ggml_type> deduped;
    for (ggml_type type : params.types) {
        if (std::find(deduped.begin(), deduped.end(), type) == deduped.end()) {
            deduped.push_back(type);
        }
    }
    params.types = deduped;

    if (params.min_iter > params.max_iter) {
        std::swap(params.min_iter, params.max_iter);
    }

    return true;
}

static uint32_t mix_seed(uint32_t seed, const shape_spec & shape, uint32_t salt) {
    uint64_t mixed = seed;
    mixed ^= (uint64_t) shape.rows * 0x9E3779B185EBCA87ull;
    mixed ^= (uint64_t) shape.cols * 0xC2B2AE3D27D4EB4Full;
    mixed ^= (uint64_t) salt * 0x165667B19E3779F9ull;
    return (uint32_t) (mixed ^ (mixed >> 32));
}

static shape_inputs make_shape_inputs(const shape_spec & shape, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    shape_inputs data;
    data.weights.resize((size_t) shape.rows * (size_t) shape.cols);
    data.input.resize((size_t) shape.cols);

    for (float & value : data.weights) {
        value = dist(rng);
    }

    for (float & value : data.input) {
        value = dist(rng);
    }

    return data;
}

static std::vector<uint8_t> encode_weight_matrix(ggml_type type, const shape_spec & shape, const std::vector<float> & weights) {
    const ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(type);
    if (traits == nullptr || traits->from_float == nullptr) {
        throw std::runtime_error("missing from_float for type " + cli_type_name(type));
    }

    const size_t row_size = ggml_row_size(type, shape.cols);
    std::vector<uint8_t> encoded(row_size * (size_t) shape.rows);

    for (int64_t row = 0; row < shape.rows; ++row) {
        traits->from_float(weights.data() + row * shape.cols, encoded.data() + row * row_size, shape.cols);
    }

    return encoded;
}

static backend_handle make_cpu_backend(int threads) {
    backend_handle handle;
    handle.key = "cpu";

    handle.backend = ggml_backend_cpu_init();
    if (handle.backend == nullptr) {
        handle.description = "failed to initialize CPU backend";
        return handle;
    }

    ggml_backend_cpu_set_n_threads(handle.backend, threads);

    handle.name = ggml_backend_name(handle.backend);
    ggml_backend_dev_t device = ggml_backend_get_device(handle.backend);
    handle.description = device ? ggml_backend_dev_description(device) : "CPU";

    return handle;
}

static backend_handle make_cuda_backend(int device_index) {
    backend_handle handle;
    handle.key = "cuda";

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (reg == nullptr) {
        handle.description = "CUDA backend is not available; rebuild with -DGGML_CUDA=ON";
        return handle;
    }

    const size_t count = ggml_backend_reg_dev_count(reg);
    if (count == 0) {
        handle.description = "CUDA backend is built, but no CUDA devices were found";
        return handle;
    }

    if (device_index < 0 || (size_t) device_index >= count) {
        std::ostringstream oss;
        oss << "CUDA device index " << device_index << " is out of range (0-" << (count - 1) << ")";
        handle.description = oss.str();
        return handle;
    }

    ggml_backend_dev_t device = ggml_backend_reg_dev_get(reg, (size_t) device_index);
    handle.backend = ggml_backend_dev_init(device, nullptr);
    if (handle.backend == nullptr) {
        handle.description = "failed to initialize CUDA backend";
        return handle;
    }

    handle.name = ggml_backend_name(handle.backend);
    handle.description = ggml_backend_dev_description(device);

    return handle;
}

static void free_backend(backend_handle & handle) {
    if (handle.backend != nullptr) {
        ggml_backend_free(handle.backend);
        handle.backend = nullptr;
    }
}

static benchmark_result benchmark_backend_matvec(
        const backend_handle & handle,
        const quant_speed_params & params,
        const shape_spec & shape,
        ggml_type type,
        const shape_inputs & inputs) {
    benchmark_result result;
    result.backend = handle.key;
    result.device = handle.description.empty() ? handle.name : handle.description;
    result.implementation = handle.key == "cpu" ? cpu_kernel_name(type) : "ggml_mul_mat";
    result.shape = shape;
    result.type = type;

    if (handle.backend == nullptr) {
        result.skipped = true;
        result.message = handle.description;
        return result;
    }

    const int64_t block = ggml_blck_size(type);
    if (shape.cols % block != 0) {
        result.skipped = true;
        std::ostringstream oss;
        oss << "cols=" << shape.cols << " is not divisible by block size " << block;
        result.message = oss.str();
        return result;
    }

    std::vector<uint8_t> encoded_weights;
    try {
        encoded_weights = encode_weight_matrix(type, shape, inputs.weights);
    } catch (const std::exception & e) {
        result.skipped = true;
        result.message = e.what();
        return result;
    }

    ggml_init_params gparams = {
        /*.mem_size   =*/ 4*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(gparams);
    if (ctx == nullptr) {
        result.skipped = true;
        result.message = "ggml_init failed";
        return result;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, shape.cols, shape.rows);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.cols, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(handle.backend));
    if (buffer == nullptr) {
        ggml_free(ctx);
        result.skipped = true;
        result.message = "failed to allocate backend tensor buffer";
        return result;
    }

    ggml_backend_tensor_set(weight, encoded_weights.data(), 0, encoded_weights.size());
    ggml_backend_tensor_set(input, inputs.input.data(), 0, inputs.input.size() * sizeof(float));

    for (int i = 0; i < params.warmup; ++i) {
        const ggml_status status = ggml_backend_graph_compute(handle.backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            result.skipped = true;
            result.message = "warmup failed with ggml status " + std::to_string((int) status);
            return result;
        }
    }

    const auto bench_begin = std::chrono::steady_clock::now();

    while (result.iterations < params.max_iter) {
        const auto t0 = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(handle.backend, graph);
        const auto t1 = std::chrono::steady_clock::now();

        if (status != GGML_STATUS_SUCCESS) {
            result.skipped = true;
            result.message = "benchmark failed with ggml status " + std::to_string((int) status);
            break;
        }

        result.total_seconds += std::chrono::duration<double>(t1 - t0).count();
        result.iterations += 1;

        if (result.iterations >= params.min_iter && result.total_seconds >= params.min_seconds) {
            break;
        }
    }

    std::vector<float> output_host((size_t) shape.rows);
    if (!result.skipped) {
        ggml_backend_tensor_get(output, output_host.data(), 0, output_host.size() * sizeof(float));
        result.checksum = std::accumulate(output_host.begin(), output_host.end(), 0.0);
        result.avg_ms = 1000.0 * result.total_seconds / std::max(1, result.iterations);
        result.gops = (2.0 * (double) shape.rows * (double) shape.cols * (double) result.iterations) /
                      std::max(result.total_seconds, 1e-12) / 1e9;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    if (!result.skipped && result.iterations == 0) {
        result.skipped = true;
        result.message = "no iterations executed";
    }

    const auto bench_end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(bench_end - bench_begin).count();
    GGML_UNUSED(wall_seconds);

    return result;
}

static std::string csv_escape(const std::string & value) {
    if (value.find_first_of(",\"") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '\"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += "\"";
    return escaped;
}

static void assign_latency_scales(std::vector<benchmark_result> & results) {
    std::map<std::pair<std::string, std::string>, double> baseline_ms;

    for (const auto & result : results) {
        if (!result.skipped && result.type == GGML_TYPE_F16) {
            baseline_ms[{ result.backend, result.shape.to_string() }] = result.avg_ms;
        }
    }

    for (auto & result : results) {
        if (result.skipped) {
            continue;
        }

        const auto key = std::make_pair(result.backend, result.shape.to_string());
        const auto it = baseline_ms.find(key);
        if (it != baseline_ms.end() && it->second > 0.0) {
            result.scale_vs_f16 = result.avg_ms / it->second;
        }
    }
}

static double geometric_mean(const std::vector<double> & values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double sum_log = 0.0;
    for (double value : values) {
        sum_log += std::log(value);
    }
    return std::exp(sum_log / values.size());
}

static std::vector<const benchmark_result *> collect_backend_results(
        const std::vector<benchmark_result> & results,
        const std::string & backend) {
    std::vector<const benchmark_result *> filtered;
    for (const auto & result : results) {
        if (result.backend == backend) {
            filtered.push_back(&result);
        }
    }
    return filtered;
}

static void print_backend_summary(
        const std::string & backend,
        const quant_speed_params & params,
        const std::vector<ggml_type> & ordered_types,
        const std::vector<benchmark_result> & results) {
    const auto filtered = collect_backend_results(results, backend);
    if (filtered.empty()) {
        return;
    }

    std::string device = "unknown";
    for (const auto * result : filtered) {
        if (!result->device.empty()) {
            device = result->device;
            break;
        }
    }

    fprintf(stdout, "\n[%s] %s\n", backend.c_str(), device.c_str());
    if (backend == "cpu") {
        fprintf(stdout, "  threads: %d\n", params.threads);
        fprintf(stdout, "  path:    ggml_mul_mat matvec on CPU backend (dispatches to quantized dot/GEMV kernels)\n");
    } else if (backend == "cuda") {
        fprintf(stdout, "  path:    ggml_mul_mat batch-1 matvec on CUDA backend (structured to prefer quantized matvec kernels)\n");
    }

    for (const auto & shape : params.shapes) {
        fprintf(stdout, "\n  shape %s\n", shape.to_string().c_str());
        fprintf(stdout, "  %-10s %-12s %-12s %-13s %-9s %s\n", "type", "avg_ms", "gops", "scale_vs_f16", "iters", "impl");

        for (ggml_type type : ordered_types) {
            const benchmark_result * match = nullptr;
            for (const auto * result : filtered) {
                if (result->shape.rows == shape.rows && result->shape.cols == shape.cols && result->type == type) {
                    match = result;
                    break;
                }
            }

            if (match == nullptr) {
                continue;
            }

            if (match->skipped) {
                fprintf(stdout, "  %-10s %-12s %-12s %-13s %-9s %s\n",
                        cli_type_name(match->type).c_str(),
                        "skip",
                        "-",
                        "-",
                        "-",
                        match->message.c_str());
            } else {
                char scale_buf[32] = {0};
                if (std::isfinite(match->scale_vs_f16)) {
                    snprintf(scale_buf, sizeof(scale_buf), "%.4f", match->scale_vs_f16);
                } else {
                    snprintf(scale_buf, sizeof(scale_buf), "n/a");
                }

                fprintf(stdout, "  %-10s %-12.4f %-12.2f %-13s %-9d %s\n",
                        cli_type_name(match->type).c_str(),
                        match->avg_ms,
                        match->gops,
                        scale_buf,
                        match->iterations,
                        match->implementation.c_str());
            }
        }
    }

    fprintf(stdout, "\n  geometric-mean latency scales vs f16\n");

    std::map<ggml_type, double> gmeans;
    for (ggml_type type : ordered_types) {
        std::vector<double> scales;
        for (const auto * result : filtered) {
            if (!result->skipped && result->type == type && std::isfinite(result->scale_vs_f16) && result->scale_vs_f16 > 0.0) {
                scales.push_back(result->scale_vs_f16);
            }
        }

        if (!scales.empty()) {
            const double gmean = geometric_mean(scales);
            gmeans[type] = gmean;
            fprintf(stdout, "  %-10s %.4f\n", cli_type_name(type).c_str(), gmean);
        }
    }

    const auto it8 = gmeans.find(GGML_TYPE_Q8_0);
    const auto it5 = gmeans.find(GGML_TYPE_Q5_0);
    const auto it4 = gmeans.find(GGML_TYPE_Q4_0);
    const auto it2 = gmeans.find(GGML_TYPE_Q2_K);

    if (it8 != gmeans.end() && it5 != gmeans.end() && it4 != gmeans.end() && it2 != gmeans.end()) {
        fprintf(stdout, "\n  chapter3 score.py flags\n");
        fprintf(stdout, "  --latency-scale-8 %.4f --latency-scale-5 %.4f --latency-scale-4 %.4f --latency-scale-2 %.4f\n",
                it8->second,
                it5->second,
                it4->second,
                it2->second);
    }
}

static void maybe_write_csv(const std::string & path, const std::vector<benchmark_result> & results) {
    if (path.empty()) {
        return;
    }

    std::ofstream out(path);
    if (!out) {
        fprintf(stderr, "warning: failed to open CSV output: %s\n", path.c_str());
        return;
    }

    out << "backend,device,shape,rows,cols,type,avg_ms,gops,scale_vs_f16,iterations,checksum,implementation,status,message\n";
    out << std::fixed << std::setprecision(6);

    for (const auto & result : results) {
        out << csv_escape(result.backend) << ','
            << csv_escape(result.device) << ','
            << csv_escape(result.shape.to_string()) << ','
            << result.shape.rows << ','
            << result.shape.cols << ','
            << cli_type_name(result.type) << ',';

        if (result.skipped) {
            out << ",,,";
        } else {
            out << result.avg_ms << ','
                << result.gops << ',';
            if (std::isfinite(result.scale_vs_f16)) {
                out << result.scale_vs_f16;
            }
            out << ',';
        }

        out << result.iterations << ','
            << result.checksum << ','
            << csv_escape(result.implementation) << ','
            << (result.skipped ? "skipped" : "ok") << ','
            << csv_escape(result.message) << '\n';
    }
}

int main(int argc, char ** argv) {
    ggml_time_init();
    ggml_backend_load_all();

    quant_speed_params params;
    if (!parse_params(argc, argv, params)) {
        print_usage(argv[0], params);
        return 1;
    }

    fprintf(stdout, "quant-speed: hardware-aware matvec benchmark for Chapter 3\n");
    fprintf(stdout, "backend:     %s\n", params.backend.c_str());
    fprintf(stdout, "types:       ");
    for (size_t i = 0; i < params.types.size(); ++i) {
        if (i > 0) {
            fprintf(stdout, ",");
        }
        fprintf(stdout, "%s", cli_type_name(params.types[i]).c_str());
    }
    fprintf(stdout, "\n");
    fprintf(stdout, "shapes:      ");
    for (size_t i = 0; i < params.shapes.size(); ++i) {
        if (i > 0) {
            fprintf(stdout, ", ");
        }
        fprintf(stdout, "%s", params.shapes[i].to_string().c_str());
    }
    fprintf(stdout, "\n");
    fprintf(stdout, "warmup:      %d\n", params.warmup);
    fprintf(stdout, "min_iter:    %d\n", params.min_iter);
    fprintf(stdout, "max_iter:    %d\n", params.max_iter);
    fprintf(stdout, "min_seconds: %.2f\n", params.min_seconds);

    backend_handle cpu_backend;
    backend_handle cuda_backend;

    if (params.backend == "cpu" || params.backend == "both") {
        cpu_backend = make_cpu_backend(params.threads);
        if (cpu_backend.backend == nullptr) {
            fprintf(stderr, "warning: %s\n", cpu_backend.description.c_str());
        }
    }

    if (params.backend == "cuda" || params.backend == "both") {
        cuda_backend = make_cuda_backend(params.cuda_device);
        if (cuda_backend.backend == nullptr) {
            fprintf(stderr, "warning: %s\n", cuda_backend.description.c_str());
        }
    }

    if (params.backend == "cpu" && cpu_backend.backend == nullptr) {
        return 2;
    }
    if (params.backend == "cuda" && cuda_backend.backend == nullptr) {
        return 2;
    }

    std::vector<benchmark_result> results;

    for (const auto & shape : params.shapes) {
        const shape_inputs inputs = make_shape_inputs(shape, mix_seed(params.seed, shape, 0));

        for (ggml_type type : params.types) {
            if (cpu_backend.backend != nullptr) {
                results.push_back(benchmark_backend_matvec(cpu_backend, params, shape, type, inputs));
            }
            if (cuda_backend.backend != nullptr) {
                results.push_back(benchmark_backend_matvec(cuda_backend, params, shape, type, inputs));
            }
        }
    }

    assign_latency_scales(results);

    if (cpu_backend.backend != nullptr) {
        print_backend_summary("cpu", params, params.types, results);
    }
    if (cuda_backend.backend != nullptr) {
        print_backend_summary("cuda", params, params.types, results);
    }

    maybe_write_csv(params.csv_path, results);
    if (!params.csv_path.empty()) {
        fprintf(stdout, "\ncsv: %s\n", params.csv_path.c_str());
    }

    free_backend(cpu_backend);
    free_backend(cuda_backend);

    return 0;
}
