const path = require('path');
const os = require('os');

const isWindows = os.platform() === 'win32';
const buildPath = isWindows ? "./build/Release/addon.node.node" : "./build/Release/addon.node.node";

const addon = require(path.join(__dirname, buildPath));

const modelPath = path.join(__dirname, "../../models/ggml-base.en.bin");
const audioPath = path.join(__dirname, "../../samples/jfk.wav");

const args = process.argv.slice(2);
const params = {};
for (const arg of args) {
    if (arg.startsWith("--")) {
        const [key, value] = arg.slice(2).split("=");
        params[key] = value;
    }
}

const model = params.model || modelPath;
const audio = params.audio || audioPath;
const includeTokens = params.tokens === "true";

console.log("=== Streaming Transcription Example ===\n");
console.log(`Model: ${model}`);
console.log(`Audio: ${audio}`);
console.log(`Include tokens: ${includeTokens}\n`);

const ctx = new addon.WhisperContext({
    model: model,
    use_gpu: true,
    no_prints: true
});

console.log("Transcribing with streaming output...\n");
console.log("--- Segments (real-time) ---");

let segmentCount = 0;
const startTime = Date.now();

addon.transcribe(ctx, {
    fname_inp: audio,
    language: "en",
    token_timestamps: includeTokens,

    on_new_segment: (segment) => {
        segmentCount++;
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
        
        console.log(`[${elapsed}s] Segment ${segment.segment_index}: [${segment.start} --> ${segment.end}]`);
        console.log(`         ${segment.text}`);
        
        if (segment.tokens && segment.tokens.length > 0) {
            const tokenSummary = segment.tokens.slice(0, 3).map(t => 
                `"${t.text.trim()}" (${(t.probability * 100).toFixed(0)}%)`
            ).join(", ");
            console.log(`         Tokens: ${tokenSummary}${segment.tokens.length > 3 ? '...' : ''}`);
        }
        console.log();
    },

    progress_callback: (progress) => {
        process.stdout.write(`\r[Progress: ${progress}%]`);
    }
}, (err, result) => {
    if (err) {
        console.error("\nError:", err);
        ctx.free();
        process.exit(1);
    }

    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
    
    console.log("\n--- Summary ---");
    console.log(`Total segments: ${result.segments.length}`);
    console.log(`Streaming segments received: ${segmentCount}`);
    console.log(`Total time: ${totalTime}s`);
    
    if (result.language) {
        console.log(`Detected language: ${result.language}`);
    }

    console.log("\n--- Full Transcript ---");
    const fullText = result.segments.map(s => s[2]).join('');
    console.log(fullText.trim());

    ctx.free();
});
