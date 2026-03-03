#!/usr/bin/env node
/**
 * Test script to verify whisper-cpp-node package works correctly
 */

const path = require("path");

// Simulate how the package will be loaded
process.env.DEBUG_LOADER = "1";

async function main() {
  console.log("=== Testing whisper-cpp-node ===\n");

  // Test 1: Load the package
  console.log("1. Loading package...");
  let whisper;
  try {
    whisper = require("./packages/whisper-cpp-node/dist/index.js");
    console.log("   ✅ Package loaded successfully\n");
  } catch (err) {
    console.error("   ❌ Failed to load package:", err.message);
    process.exit(1);
  }

  // Test 2: Check exports
  console.log("2. Checking exports...");
  const expectedExports = [
    "WhisperContextClass",
    "VadContextClass",
    "transcribe",
    "transcribeAsync",
    "createWhisperContext",
    "createVadContext",
  ];

  for (const exp of expectedExports) {
    if (whisper[exp]) {
      console.log(`   ✅ ${exp}: ${typeof whisper[exp]}`);
    } else {
      console.log(`   ❌ ${exp}: missing`);
    }
  }
  console.log();

  // Test 3: Create a context with a model (if available)
  const modelPath = path.join(
    __dirname,
    "../models/ggml-large-v3-turbo-q4_0.bin"
  );
  const audioPath = path.join(__dirname, "../samples/jfk.wav");

  console.log("3. Testing WhisperContext creation...");
  console.log(`   Model: ${modelPath}`);

  try {
    const fs = require("fs");
    if (!fs.existsSync(modelPath)) {
      console.log("   ⚠️  Model not found, skipping context test\n");
    } else {
      const ctx = whisper.createWhisperContext({
        model: modelPath,
        use_gpu: true,
        use_coreml: true,
      });
      console.log("   ✅ WhisperContext created successfully");
      console.log(`   System info: ${ctx.getSystemInfo()}\n`);

      // Test 4: Transcribe if audio exists
      if (fs.existsSync(audioPath)) {
        console.log("4. Testing transcription...");
        console.log(`   Audio: ${audioPath}`);

        const result = await whisper.transcribeAsync(ctx, {
          fname_inp: audioPath,
          language: "en",
          n_threads: 4,
        });

        console.log("   ✅ Transcription completed");
        console.log(`   Segments: ${result.segments.length}`);
        for (const [start, end, text] of result.segments) {
          console.log(`   [${start} - ${end}]: ${text}`);
        }
        console.log();
      } else {
        console.log(
          "4. ⚠️  Audio file not found, skipping transcription test\n"
        );
      }

      ctx.free();
      console.log("   ✅ Context freed\n");
    }
  } catch (err) {
    console.error("   ❌ Error:", err.message);
    console.error(err.stack);
  }

  console.log("=== All tests completed ===");
}

main().catch(console.error);
