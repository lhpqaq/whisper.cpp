#!/usr/bin/env node
/**
 * Test Core ML transcription performance
 *
 * Usage:
 *   node test-coreml-transcribe.js [audio.wav] [model.bin]
 *
 * Examples:
 *   node test-coreml-transcribe.js
 *   node test-coreml-transcribe.js ../../samples/jfk.wav
 *   node test-coreml-transcribe.js myaudio.wav ../../models/ggml-large-v3-turbo-q4_0.bin
 */

const path = require("path");
const { promisify } = require("util");

// Load the addon
let addon;
try {
  addon = require(path.join(__dirname, "build/Release/addon.node.node"));
} catch (e) {
  addon = require(path.join(__dirname, "../../build/Release/addon.node.node"));
}

const { WhisperContext, transcribe } = addon;
const transcribeAsync = promisify(transcribe);

// Parse arguments
const args = process.argv.slice(2);
const audioArg = args[0] || path.join(__dirname, "../../samples/jfk.wav");
const modelArg =
  args[1] || path.join(__dirname, "../../models/ggml-large-v3-turbo-q4_0.bin");

// Resolve relative paths from current working directory
const audioPath = path.isAbsolute(audioArg)
  ? audioArg
  : path.resolve(process.cwd(), audioArg);
const modelPath = path.isAbsolute(modelArg)
  ? modelArg
  : path.resolve(process.cwd(), modelArg);

async function main() {
  console.log("=== Core ML Transcription Test ===\n");
  console.log(`Audio: ${audioPath}`);
  console.log(`Model: ${modelPath}\n`);

  // Load model with Core ML
  console.log("Loading model WITH Core ML...\n");
  const startLoad = Date.now();
  const ctx = new WhisperContext({
    model: modelPath,
    use_coreml: true,
    use_gpu: true,
    no_prints: false,
  });
  console.log(`\nModel loaded in ${Date.now() - startLoad}ms`);
  console.log(`System info: ${ctx.getSystemInfo()}\n`);

  // Transcribe
  console.log("Transcribing with Core ML encoder...\n");
  const startTranscribe = Date.now();

  const result = await transcribeAsync(ctx, {
    fname_inp: audioPath,
    language: "en",
    n_threads: 4,
    no_timestamps: false,
  });

  const transcribeTime = Date.now() - startTranscribe;

  console.log("\n=== Results ===");
  console.log("Output:", JSON.stringify(result, null, 2));
  console.log(`\nTranscription time: ${transcribeTime}ms`);

  ctx.free();
  console.log("\nDone!");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
