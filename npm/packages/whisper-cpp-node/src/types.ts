/**
 * Options for creating a WhisperContext
 */
export interface WhisperContextOptions {
  /** Path to the GGML model file */
  model: string;
  /** Enable GPU acceleration (default: true) */
  use_gpu?: boolean;
  /** Enable Flash Attention (default: false) */
  flash_attn?: boolean;
  /** GPU device index (default: 0) */
  gpu_device?: number;
  /** Enable Core ML acceleration on macOS (default: false) */
  use_coreml?: boolean;
  /**
   * Enable OpenVINO encoder acceleration (Intel CPUs/GPUs, default: false)
   * Requires build with -DADDON_OPENVINO=ON and OpenVINO runtime installed.
   * The OpenVINO encoder model must exist alongside the GGML model
   * (e.g., ggml-base.en-encoder-openvino.xml for ggml-base.en.bin)
   */
  use_openvino?: boolean;
  /**
   * Path to OpenVINO encoder model (optional)
   * If not specified, derived from the GGML model path with "-encoder-openvino.xml" suffix
   */
  openvino_model_path?: string;
  /**
   * OpenVINO device to run encoder inference on (default: "CPU")
   * Options: "CPU", "GPU", "NPU", etc.
   */
  openvino_device?: string;
  /**
   * OpenVINO cache directory for compiled models (optional)
   * Can speed up init time, especially for GPU, by caching compiled 'blobs'
   */
  openvino_cache_dir?: string;
  /**
   * DTW alignment preset for word-level timestamps.
   *
   * Named presets: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
   *   'medium', 'medium.en', 'large.v1', 'large.v2', 'large.v3', 'large.v3.turbo'
   *
   * Dynamic presets:
   *   'top-N'      - Use all heads from top N text layers (plain averaging)
   *   'top-N-norm' - Use top N layers + L2 norm filtering (keeps dtw_norm_top_k best heads)
   *
   * @example
   * dtw: 'large.v3.turbo'   // named preset
   * dtw: 'top-4'            // top 4 layers, all heads averaged
   * dtw: 'top-4-norm'       // top 4 layers, L2 norm selects best heads
   */
  dtw?: string;
  /**
   * Number of attention heads to keep after L2 norm filtering (default: 10).
   * Only used when dtw is set to a 'top-N-norm' preset.
   */
  dtw_norm_top_k?: number;
  /** Suppress whisper.cpp log output (default: false) */
  no_prints?: boolean;
}

/**
 * Base transcription options (shared between file and buffer input)
 */
export interface TranscribeOptionsBase {
  // === Language ===
  /** Language code (e.g., 'en', 'zh', 'auto') */
  language?: string;
  /** Translate to English */
  translate?: boolean;
  /** Detect language automatically */
  detect_language?: boolean;

  // === Threading ===
  /** Number of threads to use */
  n_threads?: number;
  /** Number of processors for parallel processing */
  n_processors?: number;

  // === Audio Processing ===
  /** Start offset in milliseconds */
  offset_ms?: number;
  /** Duration to process in milliseconds (0 = all) */
  duration_ms?: number;
  /** Audio context size */
  audio_ctx?: number;

  // === Output Control ===
  /** Disable timestamps in output */
  no_timestamps?: boolean;
  /** Single segment mode */
  single_segment?: boolean;
  /** Maximum segment length in characters (0 = no limit) */
  max_len?: number;
  /** Maximum tokens per segment (0 = no limit) */
  max_tokens?: number;
  /** Maximum context size (-1 = default) */
  max_context?: number;
  /** Split segments on word boundaries */
  split_on_word?: boolean;
  /** Include token-level timestamps */
  token_timestamps?: boolean;
  /** Word timestamp threshold */
  word_thold?: number;
  /** Use comma in timestamp format (default: true) */
  comma_in_time?: boolean;

  // === Sampling ===
  /** Temperature for sampling (0.0 = greedy) */
  temperature?: number;
  /** Temperature increment for fallback */
  temperature_inc?: number;
  /** Best of N sampling candidates */
  best_of?: number;
  /** Beam size for beam search (-1 = greedy) */
  beam_size?: number;
  /** Disable temperature fallback */
  no_fallback?: boolean;

  // === Thresholds ===
  /** Entropy threshold for fallback */
  entropy_thold?: number;
  /** Log probability threshold */
  logprob_thold?: number;
  /** No speech probability threshold */
  no_speech_thold?: number;

  // === Context ===
  /** Initial prompt text for context */
  prompt?: string;
  /** Don't use previous context */
  no_context?: boolean;
  /** Suppress blank outputs */
  suppress_blank?: boolean;
  /** Suppress non-speech tokens */
  suppress_nst?: boolean;

  // === Diarization ===
  /** Enable speaker diarization */
  diarize?: boolean;
  /** Enable tinydiarize for speaker turn detection */
  tinydiarize?: boolean;

  // === Debug/Print ===
  /** Print special tokens */
  print_special?: boolean;
  /** Print progress */
  print_progress?: boolean;
  /** Print realtime output */
  print_realtime?: boolean;
  /** Print timestamps */
  print_timestamps?: boolean;

  // === VAD (Voice Activity Detection) ===
  /** Enable VAD preprocessing */
  vad?: boolean;
  /** Path to VAD model */
  vad_model?: string;
  /** VAD speech detection threshold (0.0-1.0) */
  vad_threshold?: number;
  /** Minimum speech duration in milliseconds */
  vad_min_speech_duration_ms?: number;
  /** Minimum silence duration in milliseconds */
  vad_min_silence_duration_ms?: number;
  /** Maximum speech duration in seconds */
  vad_max_speech_duration_s?: number;
  /** Speech padding in milliseconds */
  vad_speech_pad_ms?: number;
  /** VAD samples overlap ratio */
  vad_samples_overlap?: number;

  // === Callbacks ===
  /** Progress callback function (progress: 0-100) */
  progress_callback?: (progress: number) => void;

  /**
   * Streaming callback - called for each new segment during transcription.
   * Enables real-time output as audio is processed.
   * The final result callback will still receive all segments at completion.
   */
  on_new_segment?: (segment: StreamingSegment) => void;
}

/**
 * Transcription options with file input
 */
export interface TranscribeOptionsFile extends TranscribeOptionsBase {
  /** Path to the audio file */
  fname_inp: string;
  pcmf32?: never;
}

/**
 * Transcription options with PCM buffer input
 */
export interface TranscribeOptionsBuffer extends TranscribeOptionsBase {
  /** Raw PCM audio samples (16kHz, mono, float32, values -1.0 to 1.0) */
  pcmf32: Float32Array;
  fname_inp?: never;
}

/**
 * Options for transcription - either file path or PCM buffer
 */
export type TranscribeOptions = TranscribeOptionsFile | TranscribeOptionsBuffer;

/**
 * Transcription result segment
 */
export interface TranscriptSegment {
  /** Start time in format "HH:MM:SS,mmm" */
  start: string;
  /** End time in format "HH:MM:SS,mmm" */
  end: string;
  /** Transcribed text */
  text: string;
  /** Token-level data (only present when token_timestamps is enabled) */
  tokens?: StreamingToken[];
}

/**
 * Token information for streaming callbacks
 */
export interface StreamingToken {
  /** Token text */
  text: string;
  /** Token probability (0.0 to 1.0) */
  probability: number;
  /** Token timestamp start (in centiseconds from audio start) */
  t0: number;
  /** Token timestamp end (in centiseconds from audio start) */
  t1: number;
  /**
   * DTW-aligned timestamp (in centiseconds from audio start).
   * Only meaningful when the context was created with a DTW preset.
   * Typically more accurate than t0/t1 for word-level alignment.
   */
  t_dtw: number;
}

/**
 * Segment data passed to streaming callback
 */
export interface StreamingSegment {
  /** Start time in format "HH:MM:SS,mmm" */
  start: string;
  /** End time in format "HH:MM:SS,mmm" */
  end: string;
  /** Transcribed text for this segment */
  text: string;
  /** Segment index (0-based) */
  segment_index: number;
  /** Whether this is a partial segment (may be updated) */
  is_partial: boolean;
  /** Token-level information (only if token_timestamps enabled) */
  tokens?: StreamingToken[];
}

/**
 * Transcription result
 */
export interface TranscribeResult {
  /** Array of transcript segments */
  segments: TranscriptSegment[];
  /** Detected language (when detect_language is true) */
  language?: string;
}

/**
 * Options for creating a VadContext
 */
export interface VadContextOptions {
  /** Path to the Silero VAD model file */
  model: string;
  /** Speech detection threshold (default: 0.5) */
  threshold?: number;
  /** Number of threads (default: 1) */
  n_threads?: number;
  /** Suppress model loading prints */
  no_prints?: boolean;
}

/**
 * WhisperContext class for persistent model context
 */
export interface WhisperContext {
  /** Get whisper.cpp system info string */
  getSystemInfo(): string;
  /** Check if model is multilingual */
  isMultilingual(): boolean;
  /** Free the context and release resources */
  free(): void;
}

/**
 * WhisperContext constructor type
 */
export interface WhisperContextConstructor {
  new (options: WhisperContextOptions): WhisperContext;
}

/**
 * VadContext class for voice activity detection
 */
export interface VadContext {
  /** Get the required window size in samples */
  getWindowSamples(): number;
  /** Get the expected sample rate (16000 Hz) */
  getSampleRate(): number;
  /** Process audio samples and return speech probability [0, 1] */
  process(samples: Float32Array): number;
  /** Reset the internal LSTM state */
  reset(): void;
  /** Free the context and release resources */
  free(): void;
}

/**
 * VadContext constructor type
 */
export interface VadContextConstructor {
  new (options: VadContextOptions): VadContext;
}

/**
 * Transcribe callback function signature
 */
export type TranscribeCallback = (
  error: Error | null,
  result?: TranscribeResult
) => void;

/**
 * Native addon interface
 */
export interface WhisperAddon {
  WhisperContext: WhisperContextConstructor;
  VadContext: VadContextConstructor;
  transcribe: (
    context: WhisperContext,
    options: TranscribeOptions,
    callback: TranscribeCallback
  ) => void;
  whisper: Record<string, unknown>;
}
