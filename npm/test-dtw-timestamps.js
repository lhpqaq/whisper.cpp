#!/usr/bin/env node

const path = require("path");
const fs = require("fs");

const whisper = require("./packages/whisper-cpp-node/dist/index.js");

const TURBO_MODEL = path.join(__dirname, "../models/ggml-large-v3-turbo-q5_0.bin");
const BASE_EN_MODEL = path.join(__dirname, "../models/ggml-base.en.bin");
const AUDIO = path.join(__dirname, "../samples/test.wav");
const JFK_AUDIO = path.join(__dirname, "../samples/jfk.wav");
const AUDIO_DURATION_CS = 6000;

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`  ✅ ${message}`);
    passed++;
  } else {
    console.error(`  ❌ ${message}`);
    failed++;
  }
}

async function testTurboDTW() {
  console.log("\n=== Test 1: Turbo model with top-2-norm DTW ===\n");

  if (!fs.existsSync(TURBO_MODEL)) {
    console.log("  ⚠️  Turbo model not found, skipping");
    return;
  }

  const ctx = whisper.createWhisperContext({
    model: TURBO_MODEL,
    use_gpu: true,
    dtw: "top-2-norm",
    dtw_norm_top_k: 10,
    no_prints: true,
  });

  const streamingSegments = [];

  console.log("  Transcribing test.wav with DTW (this may take 30-60s)...");
  const result = await new Promise((resolve, reject) => {
    whisper.transcribe(ctx, {
      fname_inp: AUDIO,
      language: "en",
      token_timestamps: true,
      n_threads: 4,
      on_new_segment: (seg) => {
        streamingSegments.push(seg);
      },
    }, (err, res) => {
      if (err) return reject(err);
      // TSFN streaming callbacks arrive after OnOK; wait for event loop to drain
      setTimeout(() => resolve(res), 2000);
    });
  });

  console.log("\n  --- Main result checks ---");
  assert(result.segments && result.segments.length > 0, "Result has segments");

  for (let i = 0; i < result.segments.length; i++) {
    const seg = result.segments[i];
    assert(typeof seg.start === "string", `Segment ${i}: start is string ("${seg.start}")`);
    assert(typeof seg.end === "string", `Segment ${i}: end is string ("${seg.end}")`);
    assert(typeof seg.text === "string", `Segment ${i}: text is string`);
    assert(Array.isArray(seg.tokens) && seg.tokens.length > 0, `Segment ${i}: has tokens array (${seg.tokens ? seg.tokens.length : 0} tokens)`);

    if (seg.tokens && seg.tokens.length > 0) {
      let prevDtw = -1;
      let hasNonZeroDtw = false;

      for (let j = 0; j < seg.tokens.length; j++) {
        const tok = seg.tokens[j];
        assert(typeof tok.text === "string", `Segment ${i} Token ${j}: has text`);
        assert(typeof tok.probability === "number", `Segment ${i} Token ${j}: has probability`);
        assert(typeof tok.t0 === "number", `Segment ${i} Token ${j}: has t0`);
        assert(typeof tok.t1 === "number", `Segment ${i} Token ${j}: has t1`);
        assert(typeof tok.t_dtw === "number", `Segment ${i} Token ${j}: has t_dtw (${tok.t_dtw})`);

        if (tok.t_dtw !== 0) hasNonZeroDtw = true;

        assert(tok.t_dtw >= prevDtw, `Segment ${i} Token ${j}: t_dtw monotonically non-decreasing (${tok.t_dtw} >= ${prevDtw})`);
        prevDtw = tok.t_dtw;

        assert(tok.t_dtw >= 0 && tok.t_dtw <= AUDIO_DURATION_CS, `Segment ${i} Token ${j}: t_dtw in range [0, ${AUDIO_DURATION_CS}] (got ${tok.t_dtw})`);
      }

      assert(hasNonZeroDtw, `Segment ${i}: at least one non-zero t_dtw (DTW was computed)`);
    }
  }

  console.log("\n  --- Streaming callback checks ---");
  // Known limitation: whisper.cpp's new_segment_callback may not fire when DTW is enabled.
  // We still verify the data shape if any streaming segments were received.
  if (streamingSegments.length > 0) {
    console.log(`  ✅ Streaming callback fired (${streamingSegments.length} segments)`);
    passed++;

    for (let i = 0; i < streamingSegments.length; i++) {
      const seg = streamingSegments[i];
      if (seg.tokens && seg.tokens.length > 0) {
        const tok = seg.tokens[0];
        assert(typeof tok.t_dtw === "number", `Streaming segment ${i}: tokens have t_dtw field`);
      }
    }

    if (result.segments.length > 0) {
      const streamTokens = streamingSegments[0].tokens;
      const resultTokens = result.segments[0].tokens;
      if (streamTokens && resultTokens && streamTokens.length > 0 && resultTokens.length > 0) {
        assert(
          streamTokens[0].t_dtw === resultTokens[0].t_dtw,
          `Streaming t_dtw matches final result t_dtw (${streamTokens[0].t_dtw} === ${resultTokens[0].t_dtw})`
        );
      }
    }
  } else {
    console.log("  ⚠️  Streaming callback did not fire with DTW (known whisper.cpp limitation)");
  }

  ctx.free();
}

async function testBaseEnDTW() {
  console.log("\n=== Test 2: base.en model with named DTW preset ===\n");

  if (!fs.existsSync(BASE_EN_MODEL)) {
    console.log("  ⚠️  base.en model not found, skipping");
    return;
  }

  if (!fs.existsSync(JFK_AUDIO)) {
    console.log("  ⚠️  jfk.wav not found, skipping");
    return;
  }

  const ctx = whisper.createWhisperContext({
    model: BASE_EN_MODEL,
    use_gpu: true,
    dtw: "base.en",
    no_prints: true,
  });

  console.log("  Transcribing jfk.wav with base.en DTW preset...");
  const result = await whisper.transcribeAsync(ctx, {
    fname_inp: JFK_AUDIO,
    language: "en",
    token_timestamps: true,
    n_threads: 4,
  });

  assert(result.segments && result.segments.length > 0, "Result has segments");

  for (let i = 0; i < result.segments.length; i++) {
    const seg = result.segments[i];
    assert(Array.isArray(seg.tokens) && seg.tokens.length > 0, `Segment ${i}: has tokens`);

    if (seg.tokens && seg.tokens.length > 0) {
      let hasNonZeroDtw = false;
      for (const tok of seg.tokens) {
        assert(typeof tok.t_dtw === "number", `Segment ${i}: token has t_dtw`);
        if (tok.t_dtw !== 0) hasNonZeroDtw = true;
      }
      assert(hasNonZeroDtw, `Segment ${i}: at least one non-zero t_dtw`);
    }
  }

  ctx.free();
}

async function testStreamingWithoutDTW() {
  console.log("\n=== Test 3: Streaming callback t_dtw field (without DTW) ===\n");

  if (!fs.existsSync(BASE_EN_MODEL)) {
    console.log("  ⚠️  base.en model not found, skipping");
    return;
  }

  if (!fs.existsSync(JFK_AUDIO)) {
    console.log("  ⚠️  jfk.wav not found, skipping");
    return;
  }

  const ctx = whisper.createWhisperContext({
    model: BASE_EN_MODEL,
    use_gpu: true,
    no_prints: true,
  });

  const streamingSegments = [];

  console.log("  Transcribing jfk.wav with streaming + token_timestamps (no DTW)...");
  await new Promise((resolve, reject) => {
    whisper.transcribe(ctx, {
      fname_inp: JFK_AUDIO,
      language: "en",
      token_timestamps: true,
      n_threads: 4,
      on_new_segment: (seg) => {
        streamingSegments.push(seg);
      },
    }, (err, res) => {
      if (err) return reject(err);
      setTimeout(() => resolve(res), 2000);
    });
  });

  assert(streamingSegments.length > 0, `Streaming callback fired (${streamingSegments.length} segments)`);

  if (streamingSegments.length > 0 && streamingSegments[0].tokens) {
    const tok = streamingSegments[0].tokens[0];
    assert(typeof tok.t_dtw === "number", `Streaming token has t_dtw field (value: ${tok.t_dtw})`);
    assert(typeof tok.t0 === "number", `Streaming token has t0 field`);
    assert(typeof tok.t1 === "number", `Streaming token has t1 field`);
  }

  ctx.free();
}

async function testWithoutDTW() {
  console.log("\n=== Test 4: Transcription without DTW (segments are objects) ===\n");

  if (!fs.existsSync(BASE_EN_MODEL)) {
    console.log("  ⚠️  base.en model not found, skipping");
    return;
  }

  if (!fs.existsSync(JFK_AUDIO)) {
    console.log("  ⚠️  jfk.wav not found, skipping");
    return;
  }

  const ctx = whisper.createWhisperContext({
    model: BASE_EN_MODEL,
    use_gpu: true,
    no_prints: true,
  });

  console.log("  Transcribing jfk.wav without DTW or token_timestamps...");
  const result = await whisper.transcribeAsync(ctx, {
    fname_inp: JFK_AUDIO,
    language: "en",
    n_threads: 4,
  });

  assert(result.segments && result.segments.length > 0, "Result has segments");
  const seg = result.segments[0];
  assert(typeof seg.start === "string", "Segment is object with start string");
  assert(typeof seg.end === "string", "Segment is object with end string");
  assert(typeof seg.text === "string", "Segment is object with text string");
  assert(seg.tokens === undefined, "No tokens when token_timestamps not enabled");

  ctx.free();
}

async function main() {
  console.log("=== DTW Word-Level Timestamp Tests ===");

  await testTurboDTW();
  await testBaseEnDTW();
  await testStreamingWithoutDTW();
  await testWithoutDTW();

  console.log(`\n${"=".repeat(40)}`);
  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
