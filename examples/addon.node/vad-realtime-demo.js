#!/usr/bin/env node
/**
 * Real-time VAD Demo with Synchronized Audio Playback
 *
 * Streams audio through both the VAD and speaker simultaneously,
 * ensuring perfect synchronization between speech detection and playback.
 *
 * Usage:
 *   node vad-realtime-demo.js [audio.wav]
 *
 * Prerequisites:
 *   - npm install speaker
 *   - Download VAD model: ./models/download-vad-model.sh silero-v6.2.0
 */

const path = require("path");
const fs = require("fs");
const Speaker = require("speaker");

// Load the addon
let addon;
try {
    addon = require(path.join(__dirname, "build/Release/addon.node.node"));
} catch (e) {
    addon = require(path.join(__dirname, "../../build/Release/addon.node.node"));
}
const { VadContext } = addon;

// ANSI color codes
const COLORS = {
    reset: "\x1b[0m",
    red: "\x1b[31m",
    green: "\x1b[32m",
    yellow: "\x1b[33m",
    cyan: "\x1b[36m",
    bgRed: "\x1b[41m",
    bgGreen: "\x1b[42m",
    bgYellow: "\x1b[43m",
    bold: "\x1b[1m",
    dim: "\x1b[2m",
};

const DOTS = {
    filled: "●",
    empty: "○",
    block: "█",
};

/**
 * Parse WAV file header and return audio info
 */
function parseWavHeader(buffer) {
    if (buffer.toString("ascii", 0, 4) !== "RIFF") {
        throw new Error("Not a valid WAV file");
    }

    // Find fmt chunk
    let offset = 12;
    let audioFormat, numChannels, sampleRate, bitsPerSample;

    while (offset < buffer.length) {
        const chunkId = buffer.toString("ascii", offset, offset + 4);
        const chunkSize = buffer.readUInt32LE(offset + 4);

        if (chunkId === "fmt ") {
            audioFormat = buffer.readUInt16LE(offset + 8);
            numChannels = buffer.readUInt16LE(offset + 10);
            sampleRate = buffer.readUInt32LE(offset + 12);
            bitsPerSample = buffer.readUInt16LE(offset + 22);
        }

        if (chunkId === "data") {
            return {
                audioFormat,
                numChannels,
                sampleRate,
                bitsPerSample,
                dataOffset: offset + 8,
                dataSize: chunkSize,
            };
        }

        offset += 8 + chunkSize;
    }

    throw new Error("Could not parse WAV file");
}

/**
 * Convert Int16 samples to Float32 for VAD processing
 */
function int16ToFloat32(int16Array) {
    const float32 = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32[i] = int16Array[i] / 32768.0;
    }
    return float32;
}

/**
 * Get intensity bar based on probability
 */
function getIntensityBar(prob, width = 25) {
    const filled = Math.round(prob * width);
    const empty = width - filled;

    let color;
    if (prob >= 0.5) {
        color = COLORS.green;
    } else if (prob >= 0.3) {
        color = COLORS.yellow;
    } else {
        color = COLORS.red;
    }

    return color + DOTS.block.repeat(filled) + COLORS.dim + "░".repeat(empty) + COLORS.reset;
}

/**
 * Format time as MM:SS.mmm
 */
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toFixed(2).padStart(5, "0")}`;
}

/**
 * Clear current line
 */
function clearLine() {
    process.stdout.write("\r\x1b[K");
}

/**
 * Main demo function
 */
async function main() {
    const args = process.argv.slice(2);
    const audioPath = args[0] || path.join(__dirname, "../../samples/jfk.wav");
    const vadModelPath = path.join(__dirname, "../../models/ggml-silero-v6.2.0.bin");

    // Check files exist
    if (!fs.existsSync(vadModelPath)) {
        console.error(`VAD model not found: ${vadModelPath}`);
        console.error("Download with: ./models/download-vad-model.sh silero-v6.2.0");
        process.exit(1);
    }

    if (!fs.existsSync(audioPath)) {
        console.error(`Audio file not found: ${audioPath}`);
        process.exit(1);
    }

    console.log(`${COLORS.cyan}${COLORS.bold}=== Real-time Streaming VAD Demo ===${COLORS.reset}\n`);

    // Load VAD model
    console.log(`${COLORS.dim}Loading VAD model...${COLORS.reset}`);
    const vad = new VadContext({
        model: vadModelPath,
        threshold: 0.5,
        n_threads: 1,
        no_prints: true,
    });

    const windowSamples = vad.getWindowSamples();
    const vadSampleRate = vad.getSampleRate();
    console.log(`${COLORS.dim}VAD window: ${windowSamples} samples (${(windowSamples / vadSampleRate * 1000).toFixed(1)}ms)${COLORS.reset}`);

    // Load and parse audio file
    console.log(`${COLORS.dim}Loading audio: ${path.basename(audioPath)}${COLORS.reset}`);
    const fileBuffer = fs.readFileSync(audioPath);
    const wavInfo = parseWavHeader(fileBuffer);

    console.log(`${COLORS.dim}Format: ${wavInfo.sampleRate}Hz, ${wavInfo.numChannels}ch, ${wavInfo.bitsPerSample}-bit${COLORS.reset}`);

    const duration = wavInfo.dataSize / (wavInfo.sampleRate * wavInfo.numChannels * (wavInfo.bitsPerSample / 8));
    console.log(`${COLORS.dim}Duration: ${duration.toFixed(2)}s${COLORS.reset}\n`);

    // Legend
    console.log(`${COLORS.bold}Legend:${COLORS.reset}`);
    console.log(`  ${COLORS.green}${DOTS.filled}${COLORS.reset} Speech detected (>50%)`);
    console.log(`  ${COLORS.yellow}${DOTS.filled}${COLORS.reset} Maybe speech (30-50%)`);
    console.log(`  ${COLORS.red + COLORS.dim}${DOTS.filled}${COLORS.reset} Silence (<30%)\n`);

    console.log(`${COLORS.bold}Streaming audio with real-time VAD...${COLORS.reset}\n`);

    // Create speaker for audio output
    const speaker = new Speaker({
        channels: wavInfo.numChannels,
        bitDepth: wavInfo.bitsPerSample,
        sampleRate: wavInfo.sampleRate,
    });

    // Hide cursor
    process.stdout.write("\x1b[?25l");

    // Calculate chunk size for speaker that aligns with VAD window
    // VAD needs 512 samples at 16kHz, but audio might be different sample rate
    const vadChunkBytes = windowSamples * wavInfo.numChannels * (wavInfo.bitsPerSample / 8);

    // If audio sample rate differs from VAD rate, we need to resample
    const needsResample = wavInfo.sampleRate !== vadSampleRate;
    if (needsResample) {
        console.log(`${COLORS.yellow}Note: Audio is ${wavInfo.sampleRate}Hz, VAD expects ${vadSampleRate}Hz - resampling${COLORS.reset}\n`);
    }

    // Process audio in chunks
    let offset = wavInfo.dataOffset;
    let currentTime = 0;
    const bytesPerSample = wavInfo.bitsPerSample / 8;
    const bytesPerFrame = bytesPerSample * wavInfo.numChannels;

    // Use small chunks (1 VAD window = 32ms) for tight sync between display and audio
    // Smaller chunks = less buffering = better sync, but more overhead
    const playbackChunkSamples = windowSamples; // 1 VAD window = 32ms per chunk
    const playbackChunkBytes = playbackChunkSamples * bytesPerFrame;

    // Accumulator for VAD processing
    let vadBuffer = new Float32Array(0);
    let lastProb = 0;

    /**
     * Simple linear resampling
     */
    function resample(samples, fromRate, toRate) {
        if (fromRate === toRate) return samples;

        const ratio = fromRate / toRate;
        const newLength = Math.floor(samples.length / ratio);
        const resampled = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const srcIdx = i * ratio;
            const srcIdxFloor = Math.floor(srcIdx);
            const srcIdxCeil = Math.min(srcIdxFloor + 1, samples.length - 1);
            const frac = srcIdx - srcIdxFloor;
            resampled[i] = samples[srcIdxFloor] * (1 - frac) + samples[srcIdxCeil] * frac;
        }

        return resampled;
    }

    /**
     * Process a chunk of audio data
     */
    function processChunk(audioChunk) {
        // Convert to Int16Array for sample access
        const int16Samples = new Int16Array(
            audioChunk.buffer,
            audioChunk.byteOffset,
            audioChunk.length / 2
        );

        // Convert to mono Float32 for VAD
        let monoSamples;
        if (wavInfo.numChannels === 2) {
            // Average stereo to mono
            monoSamples = new Float32Array(int16Samples.length / 2);
            for (let i = 0; i < monoSamples.length; i++) {
                monoSamples[i] = (int16Samples[i * 2] + int16Samples[i * 2 + 1]) / 2 / 32768.0;
            }
        } else {
            monoSamples = int16ToFloat32(int16Samples);
        }

        // Resample if needed
        if (needsResample) {
            monoSamples = resample(monoSamples, wavInfo.sampleRate, vadSampleRate);
        }

        // Add to VAD buffer
        const newBuffer = new Float32Array(vadBuffer.length + monoSamples.length);
        newBuffer.set(vadBuffer);
        newBuffer.set(monoSamples, vadBuffer.length);
        vadBuffer = newBuffer;

        // Process complete VAD windows
        while (vadBuffer.length >= windowSamples) {
            const chunk = vadBuffer.slice(0, windowSamples);
            vadBuffer = vadBuffer.slice(windowSamples);

            lastProb = vad.process(chunk);
        }

        return lastProb;
    }

    // Stream and process
    let chunkCount = 0;
    const totalChunks = Math.ceil((fileBuffer.length - wavInfo.dataOffset) / playbackChunkBytes);
    let interrupted = false;

    // Queue of {chunk, prob, time} waiting to be displayed when played
    const pendingDisplay = [];

    function displayResult(prob, time) {
        const timeStr = formatTime(time);
        const totalTimeStr = formatTime(duration);
        const probPercent = (prob * 100).toFixed(1).padStart(5);
        const bar = getIntensityBar(prob, 30);

        const dot = prob >= 0.5
            ? `${COLORS.bgGreen}${COLORS.bold} ${DOTS.filled} ${COLORS.reset}`
            : prob >= 0.3
                ? `${COLORS.bgYellow}${COLORS.bold} ${DOTS.filled} ${COLORS.reset}`
                : `${COLORS.dim} ${DOTS.empty} ${COLORS.reset}`;

        const label = prob >= 0.5
            ? `${COLORS.green}${COLORS.bold}SPEECH${COLORS.reset}`
            : prob >= 0.3
                ? `${COLORS.yellow}MAYBE ${COLORS.reset}`
                : `${COLORS.dim}silence${COLORS.reset}`;

        clearLine();
        process.stdout.write(
            `  ${dot} ${COLORS.cyan}${timeStr}${COLORS.reset}/${totalTimeStr}  ` +
            `${bar}  ${probPercent}%  ${label}`
        );
    }

    function streamNextChunk() {
        if (interrupted) return;
        if (offset >= wavInfo.dataOffset + wavInfo.dataSize) {
            // End of file
            speaker.end();
            return;
        }

        const remainingBytes = (wavInfo.dataOffset + wavInfo.dataSize) - offset;
        const chunkSize = Math.min(playbackChunkBytes, remainingBytes);
        const chunk = fileBuffer.slice(offset, offset + chunkSize);

        // Process through VAD
        const prob = processChunk(chunk);

        // Calculate time for this chunk
        const samplesProcessed = (offset - wavInfo.dataOffset) / bytesPerFrame;
        currentTime = samplesProcessed / wavInfo.sampleRate;

        // Display the result NOW - as we're about to play this chunk
        // The speaker.write() is non-blocking and sends to audio buffer
        // so we show the result right as the audio enters the pipeline
        displayResult(prob, currentTime);

        // Write to speaker
        const canContinue = speaker.write(chunk);

        offset += chunkSize;
        chunkCount++;

        // Calculate delay based on chunk duration to sync display with audio
        const chunkDurationMs = (chunkSize / bytesPerFrame) / wavInfo.sampleRate * 1000;

        if (canContinue) {
            // Use setTimeout matching chunk duration for proper pacing
            setTimeout(streamNextChunk, chunkDurationMs);
        } else {
            // Wait for speaker to drain before sending more
            speaker.once('drain', streamNextChunk);
        }
    }

    // Cleanup function
    function cleanup(message, color, exitCode) {
        if (interrupted) return; // Prevent double cleanup
        interrupted = true;

        clearLine();
        console.log(`\n${color}${COLORS.bold}${message}${COLORS.reset}\n`);
        process.stdout.write("\x1b[?25h"); // Show cursor

        try {
            speaker.removeAllListeners();
            speaker.end();
        } catch (e) {
            // Ignore speaker errors during cleanup
        }

        try {
            vad.free();
        } catch (e) {
            // Ignore VAD errors during cleanup
        }

        process.exit(exitCode);
    }

    // Handle completion
    speaker.on('close', () => {
        cleanup("Playback complete!", COLORS.green, 0);
    });

    speaker.on('error', (err) => {
        if (!interrupted) {
            console.error(`\nSpeaker error: ${err.message}`);
            cleanup("Playback error", COLORS.red, 1);
        }
    });

    // Handle Ctrl+C
    process.on("SIGINT", () => {
        cleanup("Interrupted", COLORS.yellow, 0);
    });

    // Start streaming
    streamNextChunk();
}

main().catch((err) => {
    process.stdout.write("\x1b[?25h"); // Show cursor
    console.error(`Error: ${err.message}`);
    process.exit(1);
});
