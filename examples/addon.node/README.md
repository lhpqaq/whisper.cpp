# whisper.cpp Node.js Addon

A high-performance Node.js native addon for whisper.cpp, providing speech-to-text transcription and real-time Voice Activity Detection (VAD).

## Features

- **Persistent WhisperContext**: Load model once, transcribe multiple times
- **Streaming VAD**: Real-time voice activity detection with LSTM state preservation
- **Async Processing**: Non-blocking transcription with progress callbacks
- **Full Parameter Control**: Access to all whisper.cpp options
- **Self-contained Build**: Standalone builds without external dependencies

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [WhisperContext](#whispercontext)
  - [VadContext](#vadcontext)
  - [transcribe()](#transcribe)
  - [whisper() (Legacy)](#whisper-legacy)
- [Examples](#examples)
- [Building](#building)

---

## Installation

### Prerequisites

- Node.js 16+
- CMake 3.14+
- C++11 compatible compiler
- For standalone builds: cmake-js (`npm install -g cmake-js`)

**Windows-specific:**
- Visual Studio 2019+ with "Desktop development with C++" workload
- Vulkan SDK (optional, for GPU acceleration)
- OpenVINO Runtime (optional, for Intel CPU/GPU encoder acceleration)

### Install Dependencies

```bash
cd examples/addon.node
npm install
```

### Build the Addon

**Standalone build** (self-contained, recommended for distribution):

```bash
# Basic build (CPU only)
npx cmake-js compile

# Windows with Vulkan GPU support (auto-enabled on Windows)
npx cmake-js compile

# With OpenVINO support (Intel encoder acceleration)
npx cmake-js compile --CDADDON_OPENVINO=ON

# With both Vulkan and OpenVINO
npx cmake-js compile --CDADDON_VULKAN=ON --CDADDON_OPENVINO=ON
```

**Integrated build** (uses parent whisper.cpp build):

```bash
# From whisper.cpp root
cmake -B build
cmake --build build
```

### GPU Backend Defaults

| Platform | Default GPU Backend | Flag |
|----------|-------------------|------|
| macOS | Metal | `use_gpu: true` |
| Windows | Vulkan | `use_gpu: true` |
| Linux | Vulkan (if enabled) | `use_gpu: true` |

**Note:** On Windows, Vulkan is auto-enabled during build. If you don't have the Vulkan SDK installed, the build will fall back to CPU-only.

### Download Models

```bash
# Whisper model
./models/download-ggml-model.sh base.en

# VAD model (for streaming VAD)
./models/download-vad-model.sh silero-v6.2.0
```

### OpenVINO Model Setup

For OpenVINO acceleration, you need to convert your Whisper model to OpenVINO IR format:

```bash
# Install dependencies
pip install openvino openvino-dev

# Convert the encoder to OpenVINO format (from whisper.cpp root)
python models/convert-whisper-to-openvino.py --model base.en
```

This creates `ggml-base.en-encoder-openvino.xml` and `.bin` files alongside your model.

---

## Quick Start

### Basic Transcription

```javascript
const path = require('path');
const addon = require('./build/Release/addon.node.node');

// Create persistent context (load model once)
const ctx = new addon.WhisperContext({
    model: './models/ggml-base.en.bin',
    use_gpu: true
});

// Transcribe audio file
addon.transcribe(ctx, {
    fname_inp: './samples/jfk.wav',
    language: 'en'
}, (err, result) => {
    if (err) {
        console.error('Error:', err);
        return;
    }

    // Result: { segments: [[start, end, text], ...] }
    for (const [start, end, text] of result.segments) {
        console.log(`[${start} --> ${end}] ${text}`);
    }

    // Free when done
    ctx.free();
});
```

### Real-time VAD

```javascript
const addon = require('./build/Release/addon.node.node');

// Create VAD context
const vad = new addon.VadContext({
    model: './models/ggml-silero-v6.2.0.bin',
    threshold: 0.5,
    no_prints: true
});

const windowSamples = vad.getWindowSamples(); // 512 samples at 16kHz

// Process audio in 32ms chunks (streaming)
function processAudioChunk(samples) {
    // samples: Float32Array of windowSamples length
    const probability = vad.process(samples);

    if (probability >= 0.5) {
        console.log('Speech detected!', probability);
    }
}

// Reset LSTM state when starting new audio stream
vad.reset();

// Free when done
vad.free();
```

---

## API Reference

### WhisperContext

A persistent wrapper around the whisper.cpp context. Load the model once and use it for multiple transcriptions.

#### Constructor

```javascript
new WhisperContext(options)
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | *required* | Path to the GGML model file |
| `use_gpu` | boolean | `true` | Enable GPU acceleration (Metal on macOS, Vulkan on Windows) |
| `flash_attn` | boolean | `false` | Enable Flash Attention |
| `gpu_device` | number | `0` | GPU device index |
| `use_coreml` | boolean | `false` | Enable Core ML acceleration (macOS only) |
| `use_openvino` | boolean | `false` | Enable OpenVINO encoder acceleration (Intel, requires build with `-DADDON_OPENVINO=ON`) |
| `openvino_device` | string | `'CPU'` | OpenVINO device: 'CPU', 'GPU', 'NPU' |
| `openvino_model_path` | string | *auto* | Path to OpenVINO encoder model (auto-derived if not set) |
| `openvino_cache_dir` | string | `undefined` | Cache directory for compiled OpenVINO models |
| `dtw` | string | `undefined` | DTW alignment preset (e.g., 'base.en', 'small', 'large.v3') |
| `no_prints` | boolean | `false` | Suppress whisper.cpp log output |

**Example:**

```javascript
const ctx = new addon.WhisperContext({
    model: './models/ggml-base.en.bin',
    use_gpu: true,
    flash_attn: true,
    no_prints: true
});
```

#### OpenVINO Example (Intel)

```javascript
// Requires build with: npx cmake-js compile --CDADDON_OPENVINO=ON
const ctx = new addon.WhisperContext({
    model: './models/ggml-base.en.bin',
    use_openvino: true,
    openvino_device: 'CPU',  // or 'GPU' for Intel iGPU
    openvino_cache_dir: './openvino_cache',  // speeds up subsequent loads
    no_prints: true
});
```

#### Core ML Acceleration (macOS)

On macOS, you can enable Core ML to use the Apple Neural Engine (ANE) for encoder inference, which can provide significant speedups (3x or more on Apple Silicon).

**Prerequisites:**
1. Generate a Core ML model for your Whisper model:
   ```bash
   pip install ane_transformers openai-whisper coremltools
   ./models/generate-coreml-model.sh base.en
   ```

2. The Core ML model must be at the same location as the GGML model with `-encoder.mlmodelc` suffix:
   ```
   models/ggml-base.en.bin            # GGML model
   models/ggml-base.en-encoder.mlmodelc/  # Core ML model (folder)
   ```

**Usage:**
```javascript
const ctx = new addon.WhisperContext({
    model: './models/ggml-base.en.bin',
    use_coreml: true,  // Enable Core ML
    use_gpu: true
});
```

**Notes:**
- The first run may be slow as the ANE compiles the Core ML model
- If the Core ML model is not found, it will fall back to regular inference
- Core ML is only used for the encoder; the decoder runs normally

#### Methods

##### `free()`

Releases the context and frees memory. Prints timing information before freeing.

```javascript
ctx.free();
```

##### `isMultilingual()`

Returns `true` if the loaded model supports multiple languages.

```javascript
if (ctx.isMultilingual()) {
    console.log('Model supports multiple languages');
}
```

##### `getSystemInfo()`

Returns whisper.cpp system information string.

```javascript
console.log(ctx.getSystemInfo());
// Output: "AVX = 1 | AVX2 = 1 | AVX512 = 0 | ..."
```

---

### VadContext

A streaming Voice Activity Detection context using Silero VAD. Designed for real-time audio processing with LSTM state preservation across calls.

#### Constructor

```javascript
new VadContext(options)
```

**Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | *required* | Path to Silero VAD model (ggml-silero-v6.2.0.bin) |
| `threshold` | number | `0.5` | Speech probability threshold (0.0-1.0) |
| `n_threads` | number | `4` | Number of CPU threads |
| `use_gpu` | boolean | `false` | Enable GPU acceleration |
| `gpu_device` | number | `0` | GPU device index |
| `no_prints` | boolean | `false` | Suppress log output |

**Example:**

```javascript
const vad = new addon.VadContext({
    model: './models/ggml-silero-v6.2.0.bin',
    threshold: 0.5,
    n_threads: 1,
    no_prints: true
});
```

#### Methods

##### `process(samples)`

Process a single audio frame and return speech probability. **LSTM state is preserved across calls**, enabling true streaming VAD.

**Parameters:**
- `samples`: `Float32Array` - Audio samples (should be exactly `getWindowSamples()` length)

**Returns:** `number` - Speech probability (0.0-1.0)

```javascript
const windowSamples = vad.getWindowSamples(); // 512
const audioChunk = new Float32Array(windowSamples);
// ... fill audioChunk with audio data ...

const probability = vad.process(audioChunk);
console.log(`Speech probability: ${(probability * 100).toFixed(1)}%`);
```

##### `reset()`

Reset the internal LSTM hidden/cell states. Call this when starting a new audio stream or after a long pause.

```javascript
// Starting new audio stream
vad.reset();

// Process new audio...
```

##### `getWindowSamples()`

Returns the number of samples per VAD window (512 for Silero VAD at 16kHz = 32ms).

```javascript
const windowSize = vad.getWindowSamples(); // 512
const windowDurationMs = windowSize / vad.getSampleRate() * 1000; // 32ms
```

##### `getSampleRate()`

Returns the expected sample rate (16000 Hz).

```javascript
const sampleRate = vad.getSampleRate(); // 16000
```

##### `free()`

Releases the VAD context and frees memory.

```javascript
vad.free();
```

---

### transcribe()

Async function to transcribe audio using a persistent WhisperContext.

```javascript
addon.transcribe(context, options, callback)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `context` | WhisperContext | Initialized WhisperContext instance |
| `options` | object | Transcription options (see below) |
| `callback` | function | Callback `(err, result) => {}` |

#### Options

**Audio Input** (one required):

| Parameter | Type | Description |
|-----------|------|-------------|
| `fname_inp` | string | Path to audio file (WAV, MP3, etc.) |
| `pcmf32` | Float32Array | Raw PCM audio samples (16kHz, mono, float32) |

**Basic Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `'en'` | Language code or `'auto'` for detection |
| `translate` | boolean | `false` | Translate to English |
| `detect_language` | boolean | `false` | Auto-detect language |

**Threading:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_threads` | number | `4` | CPU threads for processing |
| `n_processors` | number | `1` | Parallel processors |

**Audio Processing:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset_ms` | number | `0` | Start offset in milliseconds |
| `duration_ms` | number | `0` | Duration to process (0 = all) |

**Output Control:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `no_timestamps` | boolean | `false` | Omit timestamps from output |
| `max_len` | number | `0` | Max segment length in characters |
| `max_tokens` | number | `0` | Max tokens per segment |
| `split_on_word` | boolean | `false` | Split on word boundaries |
| `token_timestamps` | boolean | `false` | Include token-level timestamps |

**Sampling:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | number | `0.0` | Sampling temperature |
| `temperature_inc` | number | `0.2` | Temperature increment on fallback |
| `best_of` | number | `2` | Best-of sampling candidates |
| `beam_size` | number | `-1` | Beam search size (-1 = greedy) |

**Thresholds:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_thold` | number | `2.4` | Entropy threshold for fallback |
| `logprob_thold` | number | `-1.0` | Log probability threshold |
| `no_speech_thold` | number | `0.6` | No-speech probability threshold |

**VAD Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vad` | boolean | `false` | Enable VAD preprocessing |
| `vad_model` | string | `''` | Path to VAD model |
| `vad_threshold` | number | `0.5` | VAD speech threshold |
| `vad_min_speech_duration_ms` | number | `250` | Min speech segment duration |
| `vad_min_silence_duration_ms` | number | `100` | Min silence duration |
| `vad_speech_pad_ms` | number | `30` | Padding around speech segments |

**Callbacks:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `progress_callback` | function | `null` | Progress callback `(progress) => {}` |
| `on_new_segment` | function | `null` | Streaming callback `(segment) => {}` - called for each new segment during transcription |

**Other:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | `''` | Initial prompt for context |
| `suppress_blank` | boolean | `true` | Suppress blank outputs |
| `single_segment` | boolean | `false` | Force single segment output |
| `no_context` | boolean | `true` | Don't use previous context |
| `comma_in_time` | boolean | `true` | Use comma in timestamp format |

#### Result Object

```javascript
{
    segments: [
        ['00:00:00,000', '00:00:02,500', ' Hello world'],
        ['00:00:02,500', '00:00:05,000', ' This is a test'],
        // ...
    ],
    language: 'en'  // Only if detect_language is true
}
```

#### Streaming Segment Object

When using `on_new_segment`, each segment passed to the callback has this structure:

```javascript
{
    start: '00:00:00,000',      // Start timestamp
    end: '00:00:02,500',        // End timestamp
    text: ' Hello world',       // Transcribed text
    segment_index: 0,           // 0-based segment index
    is_partial: false,          // Always false (reserved for future use)
    tokens: [                   // Only present if token_timestamps: true
        { text: ' Hello', probability: 0.95, t0: 0, t1: 120 },
        { text: ' world', probability: 0.92, t0: 120, t1: 250 },
    ]
}
```

#### Streaming Example

Real-time output as audio is being processed:

```javascript
const ctx = new addon.WhisperContext({
    model: './models/ggml-base.en.bin',
    no_prints: true
});

addon.transcribe(ctx, {
    fname_inp: './long-audio.wav',
    language: 'en',

    // Streaming: called for each segment as it's generated
    on_new_segment: (segment) => {
        console.log(`[${segment.start} --> ${segment.end}]${segment.text}`);
    },

    progress_callback: (progress) => {
        // Progress still works alongside streaming
    }
}, (err, result) => {
    // Final callback still receives all segments
    console.log(`\nComplete! Total segments: ${result.segments.length}`);
    ctx.free();
});
```

#### Example with All Options

```javascript
addon.transcribe(ctx, {
    // Audio input
    fname_inp: './audio.wav',

    // Language
    language: 'auto',
    detect_language: true,
    translate: false,

    // Threading
    n_threads: 4,
    n_processors: 1,

    // Output
    max_len: 50,
    split_on_word: true,
    token_timestamps: false,

    // Sampling
    temperature: 0.0,
    beam_size: 5,

    // VAD preprocessing
    vad: true,
    vad_model: './models/ggml-silero-v6.2.0.bin',
    vad_threshold: 0.5,

    // Progress
    progress_callback: (progress) => {
        process.stdout.write(`\rProgress: ${progress}%`);
    }
}, (err, result) => {
    if (err) {
        console.error(err);
        return;
    }

    console.log(`\nDetected language: ${result.language}`);
    for (const [start, end, text] of result.segments) {
        console.log(`[${start} --> ${end}]${text}`);
    }
});
```

---

### whisper() (Legacy)

Legacy function that creates a new context for each transcription. Use `WhisperContext` + `transcribe()` for better performance.

```javascript
addon.whisper(options, callback)
```

**Options:** Similar to `transcribe()`, but includes `model` path directly.

```javascript
addon.whisper({
    model: './models/ggml-base.en.bin',
    fname_inp: './samples/jfk.wav',
    language: 'en',
    no_prints: true
}, (err, result) => {
    // result.transcription instead of result.segments
    for (const [start, end, text] of result.transcription) {
        console.log(`[${start} --> ${end}]${text}`);
    }
});
```

---

## Examples

### Complete VAD Streaming Example

```javascript
const path = require('path');
const fs = require('fs');
const addon = require('./build/Release/addon.node.node');

/**
 * VadProcessor - A streaming VAD processor class
 */
class VadProcessor {
    constructor(modelPath, options = {}) {
        this.threshold = options.threshold ?? 0.5;
        this.minSpeechDurationMs = options.minSpeechDurationMs ?? 250;
        this.minSilenceDurationMs = options.minSilenceDurationMs ?? 100;

        this.vad = new addon.VadContext({
            model: modelPath,
            threshold: this.threshold,
            n_threads: options.nThreads ?? 1,
            no_prints: true
        });

        this.windowSamples = this.vad.getWindowSamples();
        this.sampleRate = this.vad.getSampleRate();
        this.minSpeechSamples = Math.floor(this.minSpeechDurationMs * this.sampleRate / 1000);
        this.minSilenceSamples = Math.floor(this.minSilenceDurationMs * this.sampleRate / 1000);

        this.reset();
    }

    reset() {
        this.isSpeaking = false;
        this.speechCounter = 0;
        this.silenceCounter = 0;
        this.speechStart = null;
        this.vad.reset();
    }

    /**
     * Process a single audio chunk
     * @param {Float32Array} samples - Audio samples (windowSamples length)
     * @returns {object} {probability, isSpeech, event}
     */
    processChunk(samples) {
        const prob = this.vad.process(samples);
        const isSpeech = prob >= this.threshold;
        let event = null;

        if (isSpeech) {
            this.silenceCounter = 0;
            this.speechCounter += this.windowSamples;

            if (!this.isSpeaking && this.speechCounter >= this.minSpeechSamples) {
                this.isSpeaking = true;
                this.speechStart = Date.now();
                event = 'speech_start';
            }
        } else {
            this.speechCounter = 0;

            if (this.isSpeaking) {
                this.silenceCounter += this.windowSamples;

                if (this.silenceCounter >= this.minSilenceSamples) {
                    this.isSpeaking = false;
                    event = 'speech_end';
                }
            }
        }

        return { probability: prob, isSpeech, event };
    }

    free() {
        this.vad.free();
    }
}

// Usage
const processor = new VadProcessor('./models/ggml-silero-v6.2.0.bin', {
    threshold: 0.5,
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100
});

// Simulate streaming audio
const audioBuffer = loadAudioSomehow(); // Float32Array at 16kHz
const windowSize = processor.windowSamples;

for (let i = 0; i + windowSize <= audioBuffer.length; i += windowSize) {
    const chunk = audioBuffer.slice(i, i + windowSize);
    const result = processor.processChunk(chunk);

    if (result.event === 'speech_start') {
        console.log('Speech started!');
    } else if (result.event === 'speech_end') {
        console.log('Speech ended!');
    }
}

processor.free();
```

### Transcribe with Progress

```javascript
const addon = require('./build/Release/addon.node.node');

const ctx = new addon.WhisperContext({
    model: './models/ggml-base.en.bin',
    no_prints: true
});

console.log('Transcribing...');

addon.transcribe(ctx, {
    fname_inp: './samples/jfk.wav',
    language: 'en',
    progress_callback: (progress) => {
        const bar = '█'.repeat(progress / 5) + '░'.repeat(20 - progress / 5);
        process.stdout.write(`\r[${bar}] ${progress}%`);
    }
}, (err, result) => {
    console.log('\n');

    if (err) {
        console.error('Error:', err);
        ctx.free();
        return;
    }

    let fullText = '';
    for (const [start, end, text] of result.segments) {
        console.log(`[${start} --> ${end}]${text}`);
        fullText += text;
    }

    console.log('\nFull transcription:', fullText.trim());
    ctx.free();
});
```

### Transcribe PCM Buffer

```javascript
const addon = require('./build/Release/addon.node.node');
const fs = require('fs');

// Read raw PCM file or get from microphone
function readPcmFromWav(filePath) {
    const buffer = fs.readFileSync(filePath);

    // Find data chunk
    let offset = 12;
    while (offset < buffer.length) {
        const chunkId = buffer.toString('ascii', offset, offset + 4);
        const chunkSize = buffer.readUInt32LE(offset + 4);

        if (chunkId === 'data') {
            const dataStart = offset + 8;
            const samples = new Float32Array(chunkSize / 2);

            for (let i = 0; i < samples.length; i++) {
                samples[i] = buffer.readInt16LE(dataStart + i * 2) / 32768.0;
            }

            return samples;
        }
        offset += 8 + chunkSize;
    }

    throw new Error('No data chunk found');
}

const ctx = new addon.WhisperContext({
    model: './models/ggml-base.en.bin'
});

const pcmData = readPcmFromWav('./samples/jfk.wav');

addon.transcribe(ctx, {
    pcmf32: pcmData,  // Pass Float32Array directly
    language: 'en'
}, (err, result) => {
    if (!err) {
        console.log(result.segments.map(s => s[2]).join(''));
    }
    ctx.free();
});
```

### Real-time Microphone VAD

```javascript
// Requires: npm install naudiodon
const portAudio = require('naudiodon');
const addon = require('./build/Release/addon.node.node');

const vad = new addon.VadContext({
    model: './models/ggml-silero-v6.2.0.bin',
    threshold: 0.5,
    no_prints: true
});

const windowSamples = vad.getWindowSamples(); // 512
const sampleRate = vad.getSampleRate(); // 16000

// Audio input stream
const ai = new portAudio.AudioIO({
    inOptions: {
        channelCount: 1,
        sampleFormat: portAudio.SampleFormat16Bit,
        sampleRate: sampleRate,
        deviceId: -1,  // Default device
        closeOnError: true
    }
});

let buffer = new Float32Array(0);

ai.on('data', (chunk) => {
    // Convert Int16 to Float32
    const int16 = new Int16Array(chunk.buffer, chunk.byteOffset, chunk.length / 2);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768.0;
    }

    // Accumulate samples
    const newBuffer = new Float32Array(buffer.length + float32.length);
    newBuffer.set(buffer);
    newBuffer.set(float32, buffer.length);
    buffer = newBuffer;

    // Process complete windows
    while (buffer.length >= windowSamples) {
        const windowData = buffer.slice(0, windowSamples);
        buffer = buffer.slice(windowSamples);

        const prob = vad.process(windowData);
        const status = prob >= 0.5 ? 'SPEECH' : 'silence';
        const bar = '█'.repeat(Math.round(prob * 20));

        process.stdout.write(`\r[${bar.padEnd(20)}] ${(prob * 100).toFixed(1).padStart(5)}% ${status}  `);
    }
});

console.log('Listening for speech... (Ctrl+C to stop)');
ai.start();

process.on('SIGINT', () => {
    ai.quit();
    vad.free();
    console.log('\nStopped.');
    process.exit(0);
});
```

---

## Building

### Standalone Build (Recommended)

The standalone build creates a self-contained addon with all dependencies statically linked:

```bash
cd examples/addon.node
npm install
npx cmake-js compile
```

Output: `build/Release/addon.node.node`

### Integrated Build

When building as part of the main whisper.cpp project:

```bash
# From whisper.cpp root
cmake -B build -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build
```

Output: `build/examples/addon.node/addon.node.node`

### Build Options

Configure via cmake-js:

```bash
# Debug build
npx cmake-js compile --debug

# With GPU support
npx cmake-js compile -DGGML_CUDA=ON

# Specific architecture
npx cmake-js compile -DCMAKE_OSX_ARCHITECTURES=arm64
```

### macOS Deployment Target

The addon is configured for macOS 13.3+ by default. To change:

```cmake
# In CMakeLists.txt
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
```

---

## Troubleshooting

### "Model not found" Error

Ensure you've downloaded the models:

```bash
# From whisper.cpp root
./models/download-ggml-model.sh base.en
./models/download-vad-model.sh silero-v6.2.0
```

### "Context has been freed" Error

Don't use a WhisperContext or VadContext after calling `.free()`:

```javascript
const ctx = new addon.WhisperContext({...});
ctx.free();
// ERROR: ctx.isMultilingual() will throw
```

### Audio Format Issues

Ensure audio is:
- Sample rate: 16000 Hz
- Channels: Mono (1 channel)
- Format: Float32 (-1.0 to 1.0) for `pcmf32`, or valid audio file for `fname_inp`

### VAD Sync Issues

For real-time VAD display with audio playback:
1. Use small chunks (1 VAD window = 32ms)
2. Pace display updates with `setTimeout` matching chunk duration
3. Reset VAD state when starting new streams

---

## License

MIT License - See whisper.cpp repository for full license.
