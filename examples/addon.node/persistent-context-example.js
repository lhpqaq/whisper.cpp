const path = require("path");
const { WhisperContext, transcribe, whisper } = require(path.join(
  __dirname,
  "../../build/Release/addon.node"
));
const { promisify } = require("util");

const transcribeAsync = promisify(transcribe);

// ============================================================================
// Example: Using persistent WhisperContext for efficient batch transcription
// ============================================================================

async function runPersistentContextExample() {
  console.log("=== Whisper.cpp Persistent Context Example ===\n");

  // ---------------------------------------------------------------------------
  // Step 1: Create a persistent context (model loaded once)
  // ---------------------------------------------------------------------------
  console.log("Loading model into persistent context...");

  const ctx = new WhisperContext({
    model: path.join(__dirname, "../../models/ggml-base.en.bin"),
    use_gpu: true,
    flash_attn: false,
    no_prints: false,
    // Optional: DTW alignment preset (for word-level timestamps)
    // dtw: "base.en"
  });

  console.log("Model loaded successfully!");
  console.log(`System info: ${ctx.getSystemInfo()}`);
  console.log(`Is multilingual: ${ctx.isMultilingual()}\n`);

  // ---------------------------------------------------------------------------
  // Step 2: Transcribe multiple files using the same context
  // ---------------------------------------------------------------------------

  const audioFiles = [
    path.join(__dirname, "../../samples/jfk.wav"),
    // Add more files here for batch processing
    // path.join(__dirname, "../../samples/sample2.wav"),
  ];

  for (const audioFile of audioFiles) {
    console.log(`\nTranscribing: ${audioFile}`);
    console.log("-".repeat(50));

    try {
      const result = await transcribeAsync(ctx, {
        fname_inp: audioFile,
        language: "en",

        // Threading options
        n_threads: 4,
        n_processors: 1,

        // Transcription options
        translate: false,
        no_timestamps: false,
        detect_language: false,

        // Token/segment options
        max_len: 0,           // 0 = no limit
        max_tokens: 0,        // 0 = no limit
        max_context: -1,      // -1 = use default
        single_segment: false,

        // Sampling options
        temperature: 0.0,
        temperature_inc: 0.2,
        best_of: 2,
        beam_size: -1,        // -1 = greedy search

        // Quality thresholds
        entropy_thold: 2.4,
        logprob_thold: -1.0,
        no_speech_thold: 0.6,

        // Output format
        comma_in_time: true,
        token_timestamps: false,
        split_on_word: false,

        // Suppression
        suppress_blank: true,
        suppress_nst: false,

        // Progress callback
        progress_callback: (progress) => {
          process.stdout.write(`\rProgress: ${progress}%`);
        }
      });

      console.log("\n\nTranscription result:");
      for (const [start, end, text] of result.segments) {
        console.log(`[${start} --> ${end}] ${text}`);
      }

      if (result.language) {
        console.log(`\nDetected language: ${result.language}`);
      }
    } catch (error) {
      console.error(`Error transcribing ${audioFile}:`, error);
    }
  }

  // ---------------------------------------------------------------------------
  // Step 3: Transcribe with VAD enabled (optional)
  // ---------------------------------------------------------------------------

  const vadModelPath = path.join(__dirname, "../../models/ggml-silero-v6.2.0.bin");
  const fs = require("fs");

  if (fs.existsSync(vadModelPath)) {
    console.log("\n\n=== VAD-Enabled Transcription ===");
    console.log("-".repeat(50));

    try {
      const vadResult = await transcribeAsync(ctx, {
        fname_inp: audioFiles[0],
        language: "en",

        // Enable VAD
        vad: true,
        vad_model: vadModelPath,
        vad_threshold: 0.5,
        vad_min_speech_duration_ms: 250,
        vad_min_silence_duration_ms: 100,
        vad_max_speech_duration_s: 30.0,
        vad_speech_pad_ms: 30,
        vad_samples_overlap: 0.1,

        progress_callback: (progress) => {
          process.stdout.write(`\rVAD Progress: ${progress}%`);
        }
      });

      console.log("\n\nVAD transcription result:");
      for (const [start, end, text] of vadResult.segments) {
        console.log(`[${start} --> ${end}] ${text}`);
      }
    } catch (error) {
      console.error("Error with VAD transcription:", error);
    }
  } else {
    console.log("\n\nSkipping VAD example (model not found)");
    console.log("Download VAD model: ./models/download-vad-model.sh silero-v6.2.0");
  }

  // ---------------------------------------------------------------------------
  // Step 4: Clean up - free the context when done
  // ---------------------------------------------------------------------------

  console.log("\n\nFreeing context and printing timings...");
  ctx.free();
  console.log("Context freed. Done!");
}

// ============================================================================
// Example: Passing PCM audio buffer directly
// ============================================================================

async function transcribeFromBuffer(ctx, audioBuffer) {
  // audioBuffer should be a Float32Array of PCM samples at 16kHz mono
  const result = await transcribeAsync(ctx, {
    pcmf32: audioBuffer,  // Pass audio buffer directly
    language: "en",
  });
  return result;
}

// ============================================================================
// Example: Language detection
// ============================================================================

async function detectLanguage(ctx, audioFile) {
  const result = await transcribeAsync(ctx, {
    fname_inp: audioFile,
    detect_language: true,
    language: "auto",
  });
  return result.language;
}

// ============================================================================
// Example: Using with promisified legacy API (backwards compatible)
// ============================================================================

async function legacyApiExample() {
  const whisperAsync = promisify(whisper);

  // This creates a new context for each call (legacy behavior)
  const result = await whisperAsync({
    language: "en",
    model: path.join(__dirname, "../../models/ggml-base.en.bin"),
    fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
    use_gpu: true,
    no_prints: true,
    progress_callback: (progress) => {
      console.log(`Legacy progress: ${progress}%`);
    }
  });

  console.log("Legacy API result:", result);
}

// Run the example
if (require.main === module) {
  runPersistentContextExample().catch(console.error);
}

module.exports = {
  runPersistentContextExample,
  transcribeFromBuffer,
  detectLanguage,
  legacyApiExample,
};
