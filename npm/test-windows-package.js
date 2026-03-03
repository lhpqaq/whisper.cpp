#!/usr/bin/env node
/**
 * Test script to verify whisper-cpp-node package works on Windows
 * Tests Vulkan GPU acceleration and optionally OpenVINO encoder
 */

const path = require("path");
const fs = require("fs");

// Simulate how the package will be loaded
process.env.DEBUG_LOADER = "1";

async function main() {
  console.log("=== Testing whisper-cpp-node (Windows) ===\n");

  // Test 1: Load the package
  console.log("1. Loading package...");
  let whisper;
  try {
    whisper = require("./packages/whisper-cpp-node/dist/index.js");
    console.log("   ✅ Package loaded successfully\n");
  } catch (err) {
    console.error("   ❌ Failed to load package:", err.message);
    
    // Try loading the binary directly for debugging
    console.log("\n   Attempting direct binary load for diagnostics...");
    try {
      const binaryPath = path.join(__dirname, "packages/win32-x64/whisper.node");
      if (fs.existsSync(binaryPath)) {
        console.log(`   Binary exists at: ${binaryPath}`);
        const addon = require(binaryPath);
        console.log("   ✅ Direct binary load succeeded");
        console.log("   Exports:", Object.keys(addon));
      } else {
        console.log(`   ❌ Binary not found at: ${binaryPath}`);
      }
    } catch (directErr) {
      console.error("   ❌ Direct load failed:", directErr.message);
    }
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

  let allExportsPresent = true;
  for (const exp of expectedExports) {
    if (whisper[exp]) {
      console.log(`   ✅ ${exp}: ${typeof whisper[exp]}`);
    } else {
      console.log(`   ❌ ${exp}: missing`);
      allExportsPresent = false;
    }
  }
  console.log();

  if (!allExportsPresent) {
    console.log("   ⚠️  Some exports are missing, continuing with available functionality\n");
  }

  // Test 3: Create a context with a model (if available)
  const modelPath = path.join(
    __dirname,
    "../models/ggml-large-v3-turbo-q5_0.bin"
  );
  const audioPath = path.join(__dirname, "../samples/test.wav");
  
  // OpenVINO model paths (optional)
  const openvinoModelPath = path.join(
    __dirname,
    "../models/ggml-large-v3-turbo-encoder-openvino.xml"
  );
  const hasOpenVINO = fs.existsSync(openvinoModelPath);

  console.log("3. Testing WhisperContext creation (Vulkan GPU)...");
  console.log(`   Model: ${modelPath}`);
  if (hasOpenVINO) {
    console.log(`   OpenVINO encoder available: ${openvinoModelPath}`);
  }

  try {
    if (!fs.existsSync(modelPath)) {
      console.log("   ⚠️  Model not found, skipping context test\n");
    } else {
      // Create context with Vulkan GPU acceleration
      const contextOptions = {
        model: modelPath,
        use_gpu: true,      // Enable Vulkan GPU
        flash_attn: true,   // Enable flash attention for better performance
        no_prints: false,   // Show whisper.cpp logs for debugging
      };

      // Add OpenVINO options if available
      if (hasOpenVINO) {
        contextOptions.use_openvino = true;
        contextOptions.openvino_model_path = openvinoModelPath;
        contextOptions.openvino_device = "GPU";  // or "CPU", "NPU"
        contextOptions.openvino_cache_dir = path.join(__dirname, "openvino_cache");
        console.log("   Using OpenVINO encoder acceleration");
      }

      console.log("\n   Creating context...");
      const loadStart = performance.now();
      
      const ctx = whisper.createWhisperContext(contextOptions);
      
      const loadTime = ((performance.now() - loadStart) / 1000).toFixed(2);
      console.log(`   ✅ WhisperContext created successfully (${loadTime}s)`);
      console.log(`   System info: ${ctx.getSystemInfo()}`);
      console.log(`   Multilingual: ${ctx.isMultilingual()}\n`);

      // Test 4: Transcribe if audio exists
      if (fs.existsSync(audioPath)) {
        console.log("4. Testing transcription...");
        console.log(`   Audio: ${audioPath}`);

        const transcribeStart = performance.now();
        
        const result = await whisper.transcribeAsync(ctx, {
          fname_inp: audioPath,
          language: "en",
          n_threads: 4,
          progress_callback: (progress) => {
            process.stdout.write(`\r   Progress: ${progress}%`);
          },
        });

        const transcribeTime = ((performance.now() - transcribeStart) / 1000).toFixed(2);
        console.log(`\r   ✅ Transcription completed (${transcribeTime}s)`);
        console.log(`   Segments: ${result.segments.length}`);
        
        for (const [start, end, text] of result.segments) {
          console.log(`   [${start} --> ${end}]${text}`);
        }
        console.log();
        
        // Performance summary
        console.log("5. Performance Summary:");
        console.log(`   Model load time:    ${loadTime}s`);
        console.log(`   Transcription time: ${transcribeTime}s`);
        console.log(`   Total time:         ${(parseFloat(loadTime) + parseFloat(transcribeTime)).toFixed(2)}s`);
        console.log();
      } else {
        console.log("4. ⚠️  Audio file not found, skipping transcription test\n");
      }

      ctx.free();
      console.log("   ✅ Context freed\n");
    }
  } catch (err) {
    console.error("   ❌ Error:", err.message);
    console.error(err.stack);
  }

  // Test 5: Test VAD if model available
  const vadModelPath = path.join(__dirname, "../models/ggml-silero-v6.2.0.bin");
  if (fs.existsSync(vadModelPath) && whisper.createVadContext) {
    console.log("6. Testing VAD (Voice Activity Detection)...");
    console.log(`   VAD Model: ${vadModelPath}`);
    
    try {
      const vad = whisper.createVadContext({
        model: vadModelPath,
        threshold: 0.5,
        no_prints: true,
      });
      
      console.log(`   ✅ VadContext created`);
      console.log(`   Window samples: ${vad.getWindowSamples()}`);
      console.log(`   Sample rate: ${vad.getSampleRate()}`);
      
      // Test with silent audio
      const samples = new Float32Array(vad.getWindowSamples()).fill(0);
      const probability = vad.process(samples);
      console.log(`   Speech probability (silence): ${probability.toFixed(4)}`);
      
      vad.free();
      console.log("   ✅ VadContext freed\n");
    } catch (err) {
      console.error("   ❌ VAD Error:", err.message);
    }
  } else {
    console.log("6. ⚠️  VAD model not found or VadContext unavailable, skipping VAD test\n");
  }

  console.log("=== All Windows tests completed ===");
}

main().catch(console.error);
