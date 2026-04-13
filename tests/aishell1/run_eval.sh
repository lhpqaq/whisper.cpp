#!/usr/bin/env bash

set -euo pipefail

MODELS=(
    "small"
    # "base"
    # "base-mixed-a0p8-b0p2"
    # "small-mixed-a0p8-b0p2"
    # "medium"
    # "medium-mixed-a0p8-b0p2"
    # "large-v3-turbo-mixed-a0p8-b0p2"
    # "large-v3-turbo"
    # "base-mixed-a0p2-b0p8"
    # "small-mixed-a0p2-b0p8"
    # "medium-mixed-a0p2-b0p8"
    # "large-v3-turbo-mixed-a0p2-b0p8"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/home/aim/speech_asr_aishell1_testsets/speech_asr_aishell1_testsets}"
SPLIT="${AISHELL_SPLIT:-test}"
SUBSET_SIZE="${AISHELL_SUBSET_SIZE:-1000}"
THREADS="${WHISPER_THREADS:-4}"
BEAM_SIZE="${WHISPER_BEAM_SIZE:-5}"
PROMPT="${WHISPER_INITIAL_PROMPT:-以下是普通话的简体中文转写。}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
SKIP_BUILD="${SKIP_BUILD:-0}"

MODEL_PATH_PREFIX="$REPO_ROOT/models/ggml-"
MODEL_EXTENSION=".bin"
EVAL_BIN="$REPO_ROOT/build/bin/aishell1-cer"

RESULT_ROOT="$REPO_ROOT/result/aishell1/cer"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
SUMMARY_DIR="$RUN_DIR/summary"
DETAIL_DIR="$RUN_DIR/details"
LOG_DIR="$RUN_DIR/log"
MANIFEST_FILE="$RUN_DIR/manifest.tsv"
RESULT_TABLE="$RUN_DIR/results.tsv"
MANIFEST_META="$RUN_DIR/manifest.meta.txt"

mkdir -p "$SUMMARY_DIR" "$DETAIL_DIR" "$LOG_DIR"

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Error: dataset root does not exist: $DATASET_ROOT" >&2
    exit 1
fi

TRANSCRIPT_FILE="$DATASET_ROOT/transcript/data.text"
if [[ ! -f "$TRANSCRIPT_FILE" ]]; then
    TRANSCRIPT_FILE="$DATASET_ROOT/transcript/aishell_transcript_v0.8.text"
fi
if [[ ! -f "$TRANSCRIPT_FILE" ]]; then
    echo "Error: transcript file not found under $DATASET_ROOT/transcript" >&2
    exit 1
fi

echo "Checking model files..."
MISSING_MODELS=()
for model_name in "${MODELS[@]}"; do
    full_path="${MODEL_PATH_PREFIX}${model_name}${MODEL_EXTENSION}"
    if [[ ! -f "$full_path" ]]; then
        MISSING_MODELS+=("$model_name ($full_path)")
    fi
done

if [[ ${#MISSING_MODELS[@]} -ne 0 ]]; then
    echo "Error: missing model files:" >&2
    for item in "${MISSING_MODELS[@]}"; do
        echo "  - $item" >&2
    done
    exit 1
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
    echo "Building aishell1-cer target..."
    cmake --build "$REPO_ROOT/build" --target aishell1-cer -j "$BUILD_JOBS"
else
    echo "Skipping build because SKIP_BUILD=1"
fi

if [[ ! -x "$EVAL_BIN" ]]; then
    echo "Error: evaluator binary not found after build: $EVAL_BIN" >&2
    exit 1
fi

if [[ ! -f "$MANIFEST_FILE" ]]; then
    echo "Creating fixed manifest: $MANIFEST_FILE"
    python - "$DATASET_ROOT" "$SPLIT" "$SUBSET_SIZE" "$MANIFEST_FILE" "$MANIFEST_META" <<'PY'
from pathlib import Path
import sys

dataset_root = Path(sys.argv[1])
split = sys.argv[2]
subset_size = int(sys.argv[3])
manifest_path = Path(sys.argv[4])
meta_path = Path(sys.argv[5])

split_root = dataset_root / "wav" / split
if not split_root.is_dir():
    raise SystemExit(f"split directory not found: {split_root}")

wav_files = sorted(split_root.rglob("*.wav"), key=lambda p: p.stem)
selected = wav_files if subset_size <= 0 else wav_files[:subset_size]

manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
manifest_tmp.parent.mkdir(parents=True, exist_ok=True)
with manifest_tmp.open("w", encoding="utf-8") as fp:
    for wav_path in selected:
        fp.write(f"{wav_path.stem}\t{wav_path}\n")
manifest_tmp.replace(manifest_path)

meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
with meta_tmp.open("w", encoding="utf-8") as fp:
    fp.write(f"dataset_root: {dataset_root}\n")
    fp.write(f"split: {split}\n")
    fp.write(f"subset_size: {subset_size}\n")
    fp.write(f"selected_utterances: {len(selected)}\n")
    fp.write(f"full_split_utterances: {len(wav_files)}\n")
meta_tmp.replace(meta_path)

print(f"Selected {len(selected)} / {len(wav_files)} utterances")
PY
else
    echo "Reusing existing manifest: $MANIFEST_FILE"
fi

refresh_result_table() {
    python - "$SUMMARY_DIR" "$RESULT_TABLE" <<'PY'
from pathlib import Path
import sys

summary_dir = Path(sys.argv[1])
result_table = Path(sys.argv[2])

rows = []
for path in sorted(summary_dir.glob("*.txt")):
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            data[key] = value
    rows.append((
        path.stem,
        data.get("status", ""),
        data.get("cer", ""),
        data.get("utterances", ""),
        data.get("failed_utterances", ""),
        data.get("total_ref_chars", ""),
    ))

tmp_path = result_table.with_suffix(result_table.suffix + ".tmp")
with tmp_path.open("w", encoding="utf-8") as fp:
    fp.write("model\tstatus\tcer\tutterances\tfailed_utterances\ttotal_ref_chars\n")
    for row in rows:
        fp.write("\t".join(row) + "\n")
tmp_path.replace(result_table)
PY
}

refresh_result_table

echo "Run directory: $RUN_DIR"
echo "Transcript file: $TRANSCRIPT_FILE"
echo "Manifest file: $MANIFEST_FILE"
echo "Starting evaluation..."

for model_name in "${MODELS[@]}"; do
    echo "-----------------------------------"
    echo "Processing model: $model_name"

    summary_file="$SUMMARY_DIR/${model_name}.txt"
    detail_file="$DETAIL_DIR/${model_name}.tsv"
    log_file="$LOG_DIR/${model_name}.log"
    model_path="${MODEL_PATH_PREFIX}${model_name}${MODEL_EXTENSION}"

    if [[ -s "$summary_file" ]] && grep -q '^status: ok$' "$summary_file"; then
        echo "Skipping completed model: $model_name"
        continue
    fi

    if "$EVAL_BIN" \
        --model "$model_path" \
        --dataset-root "$DATASET_ROOT" \
        --split "$SPLIT" \
        --transcript "$TRANSCRIPT_FILE" \
        --manifest "$MANIFEST_FILE" \
        --summary-file "$summary_file" \
        --detail-file "$detail_file" \
        --threads "$THREADS" \
        --beam-size "$BEAM_SIZE" \
        --language zh \
        --prompt "$PROMPT" \
        > "$log_file" 2>&1; then
        echo "Finished model: $model_name"
        refresh_result_table
    else
        echo "Error: evaluation failed for $model_name; see $log_file" >&2
        refresh_result_table
    fi
done

echo "-----------------------------------"
echo "Done. Results saved in: $RUN_DIR"
echo "Quick summary: $RESULT_TABLE"
