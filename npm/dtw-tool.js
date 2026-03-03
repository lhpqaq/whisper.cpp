#!/usr/bin/env node

const path = require("path");
const fs = require("fs");
const whisper = require("./packages/whisper-cpp-node/dist/index.js");

// Parse args
const args = process.argv.slice(2);

// Extract flags
let model = path.join(__dirname, "../models/ggml-large-v3-turbo-q5_0.bin");
let language = "auto";
let audioFile = null;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--model" && args[i + 1]) {
    model = args[++i];
  } else if (args[i] === "--language" && args[i + 1]) {
    language = args[++i];
  } else if (!args[i].startsWith("--")) {
    audioFile = args[i];
  }
}

if (!audioFile) {
  console.error("Usage: node dtw-tool.js <audio.wav> [--model <path>] [--language <code>]");
  console.error("");
  console.error("Examples:");
  console.error("  node dtw-tool.js ../samples/test.wav");
  console.error("  node dtw-tool.js ../samples/jfk.wav --model ../models/ggml-base.en.bin --language en");
  process.exit(1);
}

// Resolve audio path
audioFile = path.resolve(audioFile);
model = path.resolve(model);

if (!fs.existsSync(audioFile)) {
  console.error(`Error: Audio file not found: ${audioFile}`);
  process.exit(1);
}
if (!fs.existsSync(model)) {
  console.error(`Error: Model file not found: ${model}`);
  process.exit(1);
}

async function main() {
  console.error(`Model:    ${path.basename(model)}`);
  console.error(`Audio:    ${path.basename(audioFile)}`);
  console.error(`Language: ${language}`);
  console.error(`DTW:      top-2-norm (top_k=10)`);
  console.error("");
  console.error("Loading model...");

  const ctx = whisper.createWhisperContext({
    model: model,
    use_gpu: true,
    dtw: "top-2-norm",
    dtw_norm_top_k: 10,
    no_prints: true,
  });

  console.error("Transcribing...\n");

  const result = await whisper.transcribeAsync(ctx, {
    fname_inp: audioFile,
    language: language,
    token_timestamps: true,
    n_threads: 4,
  });

  // Print token-level DTW timestamps to stdout
  for (const segment of result.segments) {
    if (segment.tokens) {
      for (const token of segment.tokens) {
        console.log(`${token.text} @ ${token.t_dtw * 10}ms`);
      }
    }
  }

  console.error(`\n--- ${result.segments.length} segments, ${result.segments.reduce((sum, s) => sum + (s.tokens ? s.tokens.length : 0), 0)} tokens ---`);

  ctx.free();
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
