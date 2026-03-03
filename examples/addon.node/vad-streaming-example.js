/**
 * VAD Streaming Example
 *
 * This example demonstrates how to use the VadContext class for
 * real-time voice activity detection, similar to the onnxruntime-node
 * based VadProcessor but using whisper.cpp's built-in Silero VAD.
 *
 * Usage:
 *   node vad-streaming-example.js
 *
 * Prerequisites:
 *   - Download VAD model: ./models/download-vad-model.sh silero-v6.2.0
 *   - Have a test WAV file (16kHz mono)
 */

const path = require("path");
const fs = require("fs");

// Load the addon - try standalone build first, then integrated build
let addon;
try {
    addon = require(path.join(__dirname, "build/Release/addon.node.node"));
} catch (e) {
    addon = require(path.join(__dirname, "../../build/Release/addon.node.node"));
}
const { VadContext } = addon;

// Constants matching Silero VAD requirements
const SAMPLE_RATE = 16000;
const WINDOW_SIZE = 512; // samples per frame at 16kHz = 32ms

/**
 * Simple WAV file reader for PCM audio
 * Returns Float32Array of samples normalized to [-1, 1]
 */
function readWavFile(filePath) {
    const buffer = fs.readFileSync(filePath);

    // Parse WAV header
    const riff = buffer.toString('ascii', 0, 4);
    if (riff !== 'RIFF') {
        throw new Error('Not a valid WAV file');
    }

    const format = buffer.toString('ascii', 8, 12);
    if (format !== 'WAVE') {
        throw new Error('Not a valid WAV file');
    }

    // Find data chunk
    let offset = 12;
    let dataOffset = 0;
    let dataSize = 0;

    while (offset < buffer.length) {
        const chunkId = buffer.toString('ascii', offset, offset + 4);
        const chunkSize = buffer.readUInt32LE(offset + 4);

        if (chunkId === 'data') {
            dataOffset = offset + 8;
            dataSize = chunkSize;
            break;
        }

        offset += 8 + chunkSize;
    }

    if (dataOffset === 0) {
        throw new Error('Could not find data chunk in WAV file');
    }

    // Read audio format info
    const audioFormat = buffer.readUInt16LE(20);
    const numChannels = buffer.readUInt16LE(22);
    const sampleRate = buffer.readUInt32LE(24);
    const bitsPerSample = buffer.readUInt16LE(34);

    console.log(`WAV Info: ${sampleRate}Hz, ${numChannels} channel(s), ${bitsPerSample}-bit`);

    if (sampleRate !== SAMPLE_RATE) {
        console.warn(`Warning: Sample rate is ${sampleRate}Hz, expected ${SAMPLE_RATE}Hz`);
    }

    // Convert to Float32Array
    const bytesPerSample = bitsPerSample / 8;
    const numSamples = dataSize / bytesPerSample / numChannels;
    const samples = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
        const sampleOffset = dataOffset + i * bytesPerSample * numChannels;

        if (bitsPerSample === 16) {
            // 16-bit signed integer
            const sample = buffer.readInt16LE(sampleOffset);
            samples[i] = sample / 32768.0;
        } else if (bitsPerSample === 32 && audioFormat === 3) {
            // 32-bit float
            samples[i] = buffer.readFloatLE(sampleOffset);
        } else {
            throw new Error(`Unsupported audio format: ${bitsPerSample}-bit, format ${audioFormat}`);
        }
    }

    return samples;
}

/**
 * VadProcessor class that mirrors the onnxruntime-node based implementation
 * but uses whisper.cpp's built-in Silero VAD
 */
class VadProcessor {
    constructor(options = {}) {
        this.threshold = options.threshold ?? 0.5;
        this.minSpeechDurationMs = options.minSpeechDurationMs ?? 250;
        this.minSilenceDurationMs = options.minSilenceDurationMs ?? 100;

        // Initialize VAD context
        this.vad = new VadContext({
            model: options.modelPath,
            threshold: this.threshold,
            n_threads: options.nThreads ?? 1,
            use_gpu: options.useGpu ?? false,
            no_prints: options.noPrints ?? true,
        });

        // Get window configuration
        this.windowSamples = this.vad.getWindowSamples();
        this.sampleRate = this.vad.getSampleRate();

        // Calculate durations in samples
        this.minSpeechSamples = Math.floor(this.minSpeechDurationMs * this.sampleRate / 1000);
        this.minSilenceSamples = Math.floor(this.minSilenceDurationMs * this.sampleRate / 1000);

        // State tracking
        this.reset();
    }

    reset() {
        this.isSpeaking = false;
        this.speechStartSample = null;
        this.silenceStartSample = null;
        this.currentSample = 0;
        this.segments = [];

        // Reset the VAD context internal state
        this.vad.reset();
    }

    /**
     * Process a chunk of audio and return speech probability
     * Uses the streaming API that preserves LSTM state across calls
     * @param {Float32Array} samples - Audio samples (should be windowSamples in length)
     * @returns {number} Speech probability [0, 1]
     */
    processChunk(samples) {
        // The streaming API returns a single probability value
        // and preserves LSTM state across calls
        return this.vad.process(samples);
    }

    /**
     * Process audio and detect speech segments
     * @param {Float32Array} samples - Full audio buffer
     * @returns {Array} Array of {start, end} segment objects (in seconds)
     */
    detectSpeech(samples) {
        this.reset();

        const segments = [];
        let speechStart = null;
        let silenceCounter = 0;
        let speechCounter = 0;

        // Process in window-sized chunks
        for (let i = 0; i + this.windowSamples <= samples.length; i += this.windowSamples) {
            const chunk = samples.slice(i, i + this.windowSamples);
            const prob = this.processChunk(chunk);
            const isSpeech = prob >= this.threshold;

            if (isSpeech) {
                silenceCounter = 0;
                speechCounter += this.windowSamples;

                if (!this.isSpeaking && speechCounter >= this.minSpeechSamples) {
                    // Speech started
                    this.isSpeaking = true;
                    speechStart = Math.max(0, i - speechCounter + this.windowSamples);
                }
            } else {
                speechCounter = 0;

                if (this.isSpeaking) {
                    silenceCounter += this.windowSamples;

                    if (silenceCounter >= this.minSilenceSamples) {
                        // Speech ended
                        this.isSpeaking = false;
                        const speechEnd = i - silenceCounter + this.windowSamples;

                        if (speechStart !== null) {
                            segments.push({
                                start: speechStart / this.sampleRate,
                                end: speechEnd / this.sampleRate,
                            });
                        }
                        speechStart = null;
                    }
                }
            }
        }

        // Handle trailing speech
        if (this.isSpeaking && speechStart !== null) {
            segments.push({
                start: speechStart / this.sampleRate,
                end: samples.length / this.sampleRate,
            });
        }

        return segments;
    }

    /**
     * Process audio in real-time streaming mode
     * @param {Float32Array} samples - Audio chunk (should be exactly windowSamples)
     * @returns {Object} {probability: number, isSpeech: boolean}
     */
    processStreaming(samples) {
        const prob = this.processChunk(samples);
        return {
            probability: prob,
            isSpeech: prob >= this.threshold,
        };
    }

    free() {
        if (this.vad) {
            this.vad.free();
            this.vad = null;
        }
    }
}

// ============================================================================
// Main example
// ============================================================================

async function main() {
    const vadModelPath = path.join(__dirname, "../../models/ggml-silero-v6.2.0.bin");
    const testAudioPath = path.join(__dirname, "../../samples/jfk.wav");

    // Check if VAD model exists
    if (!fs.existsSync(vadModelPath)) {
        console.error(`VAD model not found at: ${vadModelPath}`);
        console.error("Please download it with: ./models/download-vad-model.sh silero-v6.2.0");
        process.exit(1);
    }

    // Check if test audio exists
    if (!fs.existsSync(testAudioPath)) {
        console.error(`Test audio not found at: ${testAudioPath}`);
        process.exit(1);
    }

    console.log("=== VadContext Streaming Example ===\n");

    // Create VAD processor
    console.log("Loading VAD model...");
    const vad = new VadProcessor({
        modelPath: vadModelPath,
        threshold: 0.5,
        minSpeechDurationMs: 250,
        minSilenceDurationMs: 100,
        noPrints: true,
    });

    console.log(`Window size: ${vad.windowSamples} samples (${vad.windowSamples / vad.sampleRate * 1000}ms)`);
    console.log(`Sample rate: ${vad.sampleRate}Hz`);
    console.log();

    // Load test audio
    console.log(`Loading audio: ${testAudioPath}`);
    const samples = readWavFile(testAudioPath);
    console.log(`Loaded ${samples.length} samples (${(samples.length / SAMPLE_RATE).toFixed(2)}s)\n`);

    // Detect speech segments
    console.log("Detecting speech segments...");
    const startTime = Date.now();
    const segments = vad.detectSpeech(samples);
    const elapsed = Date.now() - startTime;

    console.log(`\nDetected ${segments.length} speech segment(s) in ${elapsed}ms:\n`);

    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const duration = (seg.end - seg.start).toFixed(2);
        console.log(`  [${i + 1}] ${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s (${duration}s)`);
    }

    // Demonstrate streaming mode
    console.log("\n=== Streaming Mode Demo ===\n");

    vad.reset();
    const windowSamples = vad.windowSamples;

    console.log("Processing first 10 windows in streaming mode:\n");

    for (let i = 0; i < 10 && i * windowSamples + windowSamples <= samples.length; i++) {
        const chunk = samples.slice(i * windowSamples, (i + 1) * windowSamples);
        const result = vad.processStreaming(chunk);

        const timeMs = (i * windowSamples / SAMPLE_RATE * 1000).toFixed(0);
        const bar = '█'.repeat(Math.floor(result.probability * 20));
        const empty = '░'.repeat(20 - Math.floor(result.probability * 20));
        const label = result.isSpeech ? 'SPEECH' : 'silence';

        console.log(`  ${timeMs.padStart(4)}ms: [${bar}${empty}] ${(result.probability * 100).toFixed(1).padStart(5)}% ${label}`);
    }

    // Cleanup
    console.log("\nFreeing VAD context...");
    vad.free();

    console.log("Done!");
}

main().catch(console.error);
