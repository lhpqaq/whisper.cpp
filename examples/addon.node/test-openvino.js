/**
 * Transcription with OpenVINO encoder + Vulkan decoder acceleration
 * 
 * Usage:
 *   node test-openvino.js [audio_path] [device]
 * 
 * Examples:
 *   node test-openvino.js                           # Default audio, GPU device
 *   node test-openvino.js ../../samples/jfk.wav     # Specific audio
 *   node test-openvino.js ../../samples/long.wav GPU
 *   node test-openvino.js - CPU                     # Default audio, CPU device
 */

const path = require('path');
const fs = require('fs');

// Load the addon
const addon = require('./build/Release/addon.node.node');

// Configuration
const CONFIG = {
    // Model paths
    ggmlModel: path.join(__dirname, '../../models/ggml-large-v3-turbo-q5_0.bin'),
    openvinoModel: path.join(__dirname, '../../models/ggml-large-v3-turbo-encoder-openvino.xml'),
    
    // Default audio
    defaultAudio: path.join(__dirname, '../../samples/test.wav'),
    
    // OpenVINO cache (speeds up subsequent loads)
    cacheDir: path.join(__dirname, 'openvino_cache'),
};

// Parse arguments
const arg = (idx, defaultVal) => {
    const val = process.argv[idx];
    return (val && val !== '-') ? val : defaultVal;
};

const audioPath = arg(2, CONFIG.defaultAudio);
const openvinoDevice = arg(3, 'GPU');  // 'CPU', 'GPU', or 'NPU'

// Validate files
if (!fs.existsSync(CONFIG.ggmlModel)) {
    console.error(`Error: GGML model not found: ${CONFIG.ggmlModel}`);
    process.exit(1);
}
if (!fs.existsSync(CONFIG.openvinoModel)) {
    console.error(`Error: OpenVINO model not found: ${CONFIG.openvinoModel}`);
    console.error('Run: python models/quantize-openvino-encoder.py --model large-v3-turbo');
    process.exit(1);
}
if (!fs.existsSync(audioPath)) {
    console.error(`Error: Audio file not found: ${audioPath}`);
    process.exit(1);
}

console.log('='.repeat(60));
console.log('Whisper Transcription (OpenVINO + Vulkan)');
console.log('='.repeat(60));
console.log(`Audio:    ${audioPath}`);
console.log(`Model:    ${path.basename(CONFIG.ggmlModel)}`);
console.log(`Encoder:  OpenVINO (${openvinoDevice})`);
console.log(`Decoder:  Vulkan GPU`);
console.log('');

/**
 * Transcribe audio file
 */
async function transcribe(audioFile) {
    return new Promise((resolve, reject) => {
        console.log('Loading model...');
        const loadStart = performance.now();
        
        const ctx = new addon.WhisperContext({
            model: CONFIG.ggmlModel,
            use_gpu: true,           // Vulkan for decoder
            flash_attn: true,        // Flash attention for faster decoder
            use_openvino: true,
            openvino_model_path: CONFIG.openvinoModel,
            openvino_device: openvinoDevice,
            openvino_cache_dir: CONFIG.cacheDir,
            no_prints: false
        });
        
        const loadTime = ((performance.now() - loadStart) / 1000).toFixed(2);
        console.log(`Model loaded in ${loadTime}s`);
        console.log(`System: ${ctx.getSystemInfo()}`);
        console.log('');
        
        console.log('Transcribing...');
        const transcribeStart = performance.now();
        
        addon.transcribe(ctx, {
            fname_inp: audioFile,
            language: 'ko',
            n_threads: 4,
            // Progress indicator
            progress_callback: (progress) => {
                process.stdout.write(`\rProgress: ${progress}%`);
            }
        }, (err, result) => {
            const transcribeTime = ((performance.now() - transcribeStart) / 1000).toFixed(2);
            console.log(`\rTranscription completed in ${transcribeTime}s`);
            console.log('');
            
            ctx.free();
            
            if (err) {
                reject(err);
                return;
            }
            
            resolve({
                segments: result.segments,
                loadTime: parseFloat(loadTime),
                transcribeTime: parseFloat(transcribeTime)
            });
        });
    });
}

async function main() {
    try {
        const result = await transcribe(audioPath);
        
        console.log('='.repeat(60));
        console.log('Transcription Result');
        console.log('='.repeat(60));
        
        let fullText = '';
        for (const [start, end, text] of result.segments) {
            console.log(`[${start} --> ${end}]${text}`);
            fullText += text;
        }
        
        console.log('');
        console.log('='.repeat(60));
        console.log('Summary');
        console.log('='.repeat(60));
        console.log(`Load time:       ${result.loadTime}s`);
        console.log(`Transcribe time: ${result.transcribeTime}s`);
        console.log(`Total time:      ${(result.loadTime + result.transcribeTime).toFixed(2)}s`);
        console.log(`Segments:        ${result.segments.length}`);
        console.log(`Characters:      ${fullText.trim().length}`);
        
    } catch (err) {
        console.error('Error:', err.message);
        process.exit(1);
    }
}

main();
