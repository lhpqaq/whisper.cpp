#include "common-whisper.h"
#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct options {
    std::string dataset_root;
    std::string split = "test";
    std::string transcript_path;
    std::string manifest_path;
    std::string model_path;
    std::string summary_file;
    std::string detail_file;
    std::string language = "zh";
    std::string prompt = u8"以下是普通话的简体中文转写。";
    int threads = std::max(1u, std::thread::hardware_concurrency());
    int beam_size = 5;
    int subset_size = 0;
};

struct utterance {
    std::string utt_id;
    std::string wav_path;
};

struct metrics {
    uint64_t utterances = 0;
    uint64_t failed_utterances = 0;
    uint64_t total_ref_chars = 0;
    uint64_t total_char_errors = 0;
};

static std::string now_local_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static void ensure_parent_dir(const std::string & path) {
    if (path.empty()) {
        return;
    }

    const fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
}

static void write_text_atomic(const std::string & path, const std::string & content) {
    ensure_parent_dir(path);
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("failed to open temporary file for writing: " + tmp_path);
        }
        out << content;
    }
    fs::rename(tmp_path, path);
}

static std::string sanitize_field(std::string text) {
    for (char & ch : text) {
        if (ch == '\t' || ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    return text;
}

static bool decode_utf8_one(const std::string & text, size_t & i, char32_t & out) {
    if (i >= text.size()) {
        return false;
    }

    const unsigned char c0 = static_cast<unsigned char>(text[i]);
    if (c0 < 0x80) {
        out = c0;
        ++i;
        return true;
    }

    auto invalid = [&]() {
        out = 0xfffd;
        ++i;
        return true;
    };

    if ((c0 >> 5) == 0x6) {
        if (i + 1 >= text.size()) {
            return invalid();
        }
        const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        if ((c1 & 0xc0) != 0x80) {
            return invalid();
        }
        out = ((c0 & 0x1f) << 6) | (c1 & 0x3f);
        i += 2;
        return true;
    }

    if ((c0 >> 4) == 0xe) {
        if (i + 2 >= text.size()) {
            return invalid();
        }
        const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
        if ((c1 & 0xc0) != 0x80 || (c2 & 0xc0) != 0x80) {
            return invalid();
        }
        out = ((c0 & 0x0f) << 12) | ((c1 & 0x3f) << 6) | (c2 & 0x3f);
        i += 3;
        return true;
    }

    if ((c0 >> 3) == 0x1e) {
        if (i + 3 >= text.size()) {
            return invalid();
        }
        const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
        const unsigned char c3 = static_cast<unsigned char>(text[i + 3]);
        if ((c1 & 0xc0) != 0x80 || (c2 & 0xc0) != 0x80 || (c3 & 0xc0) != 0x80) {
            return invalid();
        }
        out = ((c0 & 0x07) << 18) | ((c1 & 0x3f) << 12) | ((c2 & 0x3f) << 6) | (c3 & 0x3f);
        i += 4;
        return true;
    }

    return invalid();
}

static std::vector<char32_t> utf8_to_codepoints(const std::string & text) {
    std::vector<char32_t> out;
    size_t i = 0;
    while (i < text.size()) {
        char32_t cp = 0;
        decode_utf8_one(text, i, cp);
        out.push_back(cp);
    }
    return out;
}

static std::string codepoints_to_utf8(const std::vector<char32_t> & cps) {
    std::string out;
    for (char32_t cp : cps) {
        if (cp <= 0x7f) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | ((cp >> 6) & 0x1f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | ((cp >> 12) & 0x0f)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | ((cp >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }
    return out;
}

static bool is_unicode_space(char32_t cp) {
    switch (cp) {
        case 0x0009:
        case 0x000a:
        case 0x000b:
        case 0x000c:
        case 0x000d:
        case 0x0020:
        case 0x0085:
        case 0x00a0:
        case 0x1680:
        case 0x2000:
        case 0x2001:
        case 0x2002:
        case 0x2003:
        case 0x2004:
        case 0x2005:
        case 0x2006:
        case 0x2007:
        case 0x2008:
        case 0x2009:
        case 0x200a:
        case 0x2028:
        case 0x2029:
        case 0x202f:
        case 0x205f:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

static bool is_extra_punctuation(char32_t cp) {
    switch (cp) {
        case 0x00b7:
        case 0x2010:
        case 0x2013:
        case 0x2014:
        case 0x2018:
        case 0x2019:
        case 0x201c:
        case 0x201d:
        case 0x2026:
        case 0x3001:
        case 0x3002:
        case 0x3008:
        case 0x3009:
        case 0x300a:
        case 0x300b:
        case 0x300c:
        case 0x300d:
        case 0x300e:
        case 0x300f:
        case 0x3010:
        case 0x3011:
        case 0x3014:
        case 0x3015:
        case 0x3016:
        case 0x3017:
        case 0x3018:
        case 0x3019:
        case 0x301a:
        case 0x301b:
        case 0xff01:
        case 0xff02:
        case 0xff03:
        case 0xff04:
        case 0xff05:
        case 0xff06:
        case 0xff07:
        case 0xff08:
        case 0xff09:
        case 0xff0a:
        case 0xff0b:
        case 0xff0c:
        case 0xff0d:
        case 0xff0e:
        case 0xff0f:
        case 0xff1a:
        case 0xff1b:
        case 0xff1f:
        case 0xff20:
        case 0xff3b:
        case 0xff3c:
        case 0xff3d:
        case 0xff3f:
        case 0xff5b:
        case 0xff5d:
        case 0xff5e:
        case 0xff61:
        case 0xff62:
        case 0xff63:
        case 0xff64:
        case 0xff65:
            return true;
        default:
            return false;
    }
}

static char32_t fold_fullwidth_ascii(char32_t cp) {
    if (cp >= 0xff01 && cp <= 0xff5e) {
        return cp - 0xfee0;
    }
    return cp;
}

static std::vector<char32_t> normalize_to_chars(const std::string & text) {
    std::vector<char32_t> out;
    for (char32_t cp : utf8_to_codepoints(text)) {
        cp = fold_fullwidth_ascii(cp);

        if (cp < 128) {
            if (std::isspace(static_cast<unsigned char>(cp)) || std::ispunct(static_cast<unsigned char>(cp))) {
                continue;
            }
            if (std::isupper(static_cast<unsigned char>(cp))) {
                cp = static_cast<char32_t>(std::tolower(static_cast<unsigned char>(cp)));
            }
        }

        if (is_unicode_space(cp) || is_extra_punctuation(cp) || cp == 0xfffd) {
            continue;
        }

        out.push_back(cp);
    }
    return out;
}

static std::string normalize_text(const std::string & text) {
    return codepoints_to_utf8(normalize_to_chars(text));
}

static size_t edit_distance(const std::vector<char32_t> & ref, const std::vector<char32_t> & hyp) {
    std::vector<size_t> prev(hyp.size() + 1);
    std::vector<size_t> cur(hyp.size() + 1);

    for (size_t j = 0; j <= hyp.size(); ++j) {
        prev[j] = j;
    }

    for (size_t i = 1; i <= ref.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= hyp.size(); ++j) {
            const size_t sub_cost = ref[i - 1] == hyp[j - 1] ? 0 : 1;
            cur[j] = std::min({
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + sub_cost,
            });
        }
        std::swap(prev, cur);
    }

    return prev[hyp.size()];
}

static void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model MODEL --dataset-root ROOT --summary-file FILE --detail-file FILE [options]\n"
        << "Options:\n"
        << "  --split NAME          Dataset split under wav/ (default: test)\n"
        << "  --transcript FILE     Transcript file (default: auto-detect)\n"
        << "  --manifest FILE       TSV manifest: utt_id<TAB>wav_path\n"
        << "  --threads N           Decoder threads (default: hardware concurrency)\n"
        << "  --beam-size N         Beam size (default: 5)\n"
        << "  --subset-size N       If scanning directly, keep first N sorted wavs (0 = all)\n"
        << "  --language LANG       Whisper language code (default: zh)\n"
        << "  --prompt TEXT         Initial prompt to bias output style\n";
}

static options parse_args(int argc, char ** argv) {
    options opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--dataset-root") {
            opts.dataset_root = next("--dataset-root");
        } else if (arg == "--split") {
            opts.split = next("--split");
        } else if (arg == "--transcript") {
            opts.transcript_path = next("--transcript");
        } else if (arg == "--manifest") {
            opts.manifest_path = next("--manifest");
        } else if (arg == "--model") {
            opts.model_path = next("--model");
        } else if (arg == "--summary-file") {
            opts.summary_file = next("--summary-file");
        } else if (arg == "--detail-file") {
            opts.detail_file = next("--detail-file");
        } else if (arg == "--threads") {
            opts.threads = std::max(1, std::stoi(next("--threads")));
        } else if (arg == "--beam-size") {
            opts.beam_size = std::max(1, std::stoi(next("--beam-size")));
        } else if (arg == "--subset-size") {
            opts.subset_size = std::max(0, std::stoi(next("--subset-size")));
        } else if (arg == "--language") {
            opts.language = next("--language");
        } else if (arg == "--prompt") {
            opts.prompt = next("--prompt");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.model_path.empty()) {
        throw std::runtime_error("--model is required");
    }
    if (opts.dataset_root.empty()) {
        throw std::runtime_error("--dataset-root is required");
    }
    if (opts.summary_file.empty()) {
        throw std::runtime_error("--summary-file is required");
    }
    if (opts.detail_file.empty()) {
        throw std::runtime_error("--detail-file is required");
    }

    if (opts.transcript_path.empty()) {
        const fs::path root(opts.dataset_root);
        const fs::path data_text = root / "transcript" / "data.text";
        const fs::path legacy_text = root / "transcript" / "aishell_transcript_v0.8.text";
        if (fs::exists(data_text)) {
            opts.transcript_path = data_text.string();
        } else if (fs::exists(legacy_text)) {
            opts.transcript_path = legacy_text.string();
        } else {
            throw std::runtime_error("failed to auto-detect transcript file under dataset root");
        }
    }

    return opts;
}

static std::unordered_map<std::string, std::string> load_transcripts(const std::string & path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open transcript file: " + path);
    }

    std::unordered_map<std::string, std::string> refs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const size_t split = line.find_first_of(" \t");
        if (split == std::string::npos) {
            continue;
        }

        const std::string utt_id = line.substr(0, split);
        size_t text_start = line.find_first_not_of(" \t", split);
        if (text_start == std::string::npos) {
            text_start = line.size();
        }
        refs[utt_id] = normalize_text(line.substr(text_start));
    }

    return refs;
}

static std::vector<utterance> load_manifest_entries(const std::string & manifest_path) {
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open manifest file: " + manifest_path);
    }

    std::vector<utterance> entries;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        const size_t split = line.find('\t');
        if (split != std::string::npos) {
            entries.push_back({line.substr(0, split), line.substr(split + 1)});
            continue;
        }

        const fs::path wav_path(line);
        entries.push_back({wav_path.stem().string(), wav_path.string()});
    }

    return entries;
}

static std::vector<utterance> scan_dataset_entries(const options & opts) {
    const fs::path split_root = fs::path(opts.dataset_root) / "wav" / opts.split;
    if (!fs::exists(split_root)) {
        throw std::runtime_error("dataset split directory does not exist: " + split_root.string());
    }

    std::vector<utterance> entries;
    for (const auto & item : fs::recursive_directory_iterator(split_root)) {
        if (!item.is_regular_file()) {
            continue;
        }
        if (item.path().extension() != ".wav") {
            continue;
        }
        entries.push_back({item.path().stem().string(), item.path().string()});
    }

    std::sort(entries.begin(), entries.end(), [](const utterance & a, const utterance & b) {
        return a.utt_id < b.utt_id;
    });

    if (opts.subset_size > 0 && static_cast<int>(entries.size()) > opts.subset_size) {
        entries.resize(opts.subset_size);
    }

    return entries;
}

static std::vector<utterance> collect_entries(const options & opts) {
    if (!opts.manifest_path.empty()) {
        return load_manifest_entries(opts.manifest_path);
    }
    return scan_dataset_entries(opts);
}

static std::string join_segments(whisper_context * ctx) {
    std::string out;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        out += whisper_full_get_segment_text(ctx, i);
    }
    return out;
}

static std::string build_summary(
        const options & opts,
        const metrics & m,
        const size_t manifest_size,
        const std::string & started_at,
        const std::string & finished_at,
        const std::string & status,
        const std::string & message) {
    const double cer = m.total_ref_chars == 0 ? 0.0 : 100.0 * static_cast<double>(m.total_char_errors) / static_cast<double>(m.total_ref_chars);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "status: " << status << "\n";
    oss << "message: " << message << "\n";
    oss << "model: " << opts.model_path << "\n";
    oss << "dataset_root: " << opts.dataset_root << "\n";
    oss << "split: " << opts.split << "\n";
    oss << "transcript: " << opts.transcript_path << "\n";
    if (!opts.manifest_path.empty()) {
        oss << "manifest: " << opts.manifest_path << "\n";
    }
    oss << "manifest_entries: " << manifest_size << "\n";
    oss << "utterances: " << m.utterances << "\n";
    oss << "failed_utterances: " << m.failed_utterances << "\n";
    oss << "total_ref_chars: " << m.total_ref_chars << "\n";
    oss << "total_char_errors: " << m.total_char_errors << "\n";
    oss << "cer: " << cer << "%\n";
    oss << "threads: " << opts.threads << "\n";
    oss << "beam_size: " << opts.beam_size << "\n";
    oss << "language: " << opts.language << "\n";
    oss << "prompt: " << opts.prompt << "\n";
    oss << "started_at: " << started_at << "\n";
    oss << "finished_at: " << finished_at << "\n";
    return oss.str();
}

int main(int argc, char ** argv) {
    options opts;
    metrics m;
    std::string started_at = now_local_string();

    try {
        opts = parse_args(argc, argv);
        const auto refs = load_transcripts(opts.transcript_path);
        const auto entries = collect_entries(opts);

        if (entries.empty()) {
            throw std::runtime_error("no wav files found for evaluation");
        }

        ensure_parent_dir(opts.detail_file);
        const std::string detail_tmp = opts.detail_file + ".tmp";
        std::ofstream detail_out(detail_tmp, std::ios::binary);
        if (!detail_out.is_open()) {
            throw std::runtime_error("failed to open detail file for writing: " + detail_tmp);
        }

        detail_out << "utt_id\twav_path\tref_norm\thyp_norm\tref_chars\tchar_errors\tok\n";

        whisper_context_params cparams = whisper_context_default_params();
        whisper_context * ctx = whisper_init_from_file_with_params(opts.model_path.c_str(), cparams);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to load model: " + opts.model_path);
        }

        if (!whisper_is_multilingual(ctx) && opts.language != "en") {
            whisper_free(ctx);
            throw std::runtime_error("selected model is not multilingual; cannot force Chinese output");
        }

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
        wparams.print_progress   = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.translate        = false;
        wparams.no_context       = true;
        wparams.no_timestamps    = true;
        wparams.n_threads        = opts.threads;
        wparams.language         = opts.language.c_str();
        wparams.detect_language  = false;
        wparams.initial_prompt   = opts.prompt.empty() ? nullptr : opts.prompt.c_str();
        wparams.carry_initial_prompt = false;
        wparams.beam_search.beam_size = opts.beam_size;
        wparams.greedy.best_of = 1;

        std::cerr << "Loaded model once: " << opts.model_path << "\n";
        std::cerr << "Evaluating " << entries.size() << " utterances from split '" << opts.split << "'\n";

        for (size_t index = 0; index < entries.size(); ++index) {
            const auto & entry = entries[index];
            const auto it = refs.find(entry.utt_id);
            if (it == refs.end()) {
                whisper_free(ctx);
                throw std::runtime_error("missing transcript for utterance: " + entry.utt_id);
            }

            std::vector<float> pcmf32;
            std::vector<std::vector<float>> pcmf32s;
            std::string hyp_norm;
            bool ok = true;

            if (!read_audio_data(entry.wav_path, pcmf32, pcmf32s, false)) {
                ok = false;
                ++m.failed_utterances;
            } else if (whisper_full(ctx, wparams, pcmf32.data(), static_cast<int>(pcmf32.size())) != 0) {
                ok = false;
                ++m.failed_utterances;
            } else {
                hyp_norm = normalize_text(join_segments(ctx));
            }

            const std::vector<char32_t> ref_chars = normalize_to_chars(it->second);
            const std::vector<char32_t> hyp_chars = normalize_to_chars(hyp_norm);
            const size_t char_errors = ok ? edit_distance(ref_chars, hyp_chars) : ref_chars.size();

            ++m.utterances;
            m.total_ref_chars += ref_chars.size();
            m.total_char_errors += char_errors;

            detail_out
                << entry.utt_id << '\t'
                << entry.wav_path << '\t'
                << sanitize_field(it->second) << '\t'
                << sanitize_field(hyp_norm) << '\t'
                << ref_chars.size() << '\t'
                << char_errors << '\t'
                << (ok ? 1 : 0) << '\n';
            detail_out.flush();

            if ((index + 1) % 50 == 0 || index + 1 == entries.size()) {
                const double running_cer = m.total_ref_chars == 0 ? 0.0 : 100.0 * static_cast<double>(m.total_char_errors) / static_cast<double>(m.total_ref_chars);
                std::cerr
                    << "[" << (index + 1) << "/" << entries.size() << "] "
                    << entry.utt_id << " running CER=" << std::fixed << std::setprecision(4) << running_cer << "%\n";
            }
        }

        detail_out.close();
        whisper_free(ctx);
        fs::rename(detail_tmp, opts.detail_file);

        const std::string finished_at = now_local_string();
        write_text_atomic(opts.summary_file, build_summary(opts, m, entries.size(), started_at, finished_at, "ok", "completed"));
        return 0;
    } catch (const std::exception & ex) {
        const std::string finished_at = now_local_string();
        try {
            write_text_atomic(opts.summary_file.empty() ? "aishell1-cer.error.txt" : opts.summary_file,
                    build_summary(opts, m, 0, started_at, finished_at, "error", ex.what()));
        } catch (...) {
        }
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
