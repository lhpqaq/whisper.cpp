#!/bin/bash

# 模型列表
# MODELS=("l3t-a9b1" "l3t-a1b9" "large-v3-turbo" "large-v3-turbo-q40" "large-v3-turbo-q80" "large-v3-turbo-q50" "large-v3-turbo-q2k")
# MODELS=("medium-q80" "medium-q40" "medium-q50" "medium-q2k" "medium-mixed-a0p2-b0p8" "medium-mixed-a0p8-b0p2" "medium" )
# MODELS=("base" "base-q80" "base-q40" "base-q50" "base-q2k" "base-mixed-a0p2-b0p8" "base-mixed-a0p8-b0p2" "small-q50" "small-mixed-a0p2-b0p8" "small-mixed-a0p8-b0p2")
# MODELS=("base-mixed-a0p2-b0p8")
MODELS=("small-round_robin_8_5_4_2" "small-random_seed111" "small-pure_mse_quartiles_8_5_4_2" "small-uniform_2bit" "small-uniform_8bit" "small-uniform_4bit" "small-uniform_5bit" "small-random_seed222" "small-random_seed333")
MODEL_PATH_PREFIX="../../models/ggml-"
MODEL_EXTENSION=".bin" # 假设模型后缀是 .bin，如果没有后缀请设为空 ""
# 结果存储目录
RESULT_DIR="../../result/mix/e4"

# 确保结果目录存在
mkdir -p "$RESULT_DIR"
# --- 预处理：检查模型是否存在 ---
echo "Checking if all models exist..."
MISSING_MODELS=()

for model_name in "${MODELS[@]}"; do
    # 拼接完整路径进行检查
    FULL_PATH="${MODEL_PATH_PREFIX}${model_name}${MODEL_EXTENSION}"
    if [ ! -f "$FULL_PATH" ]; then
        MISSING_MODELS+=("$model_name ($FULL_PATH)")
    fi
done

if [ ${#MISSING_MODELS[@]} -ne 0 ]; then
    echo "Error: The following model files are missing:"
    for missing in "${MISSING_MODELS[@]}"; do
        echo "  - $missing"
    done
    echo "Aborting script."
    exit 1
fi

echo "All models verified. Starting processing..."
# ------------------------------

# 确保结果目录和日志目录存在
mkdir -p "$RESULT_DIR"
mkdir -p "log"
# 遍历模型列表
for model_name in "${MODELS[@]}"; do
    echo "Processing model: $model_name"

    # 1. 执行 make clean
    make clean
    # 2. 修改 eval.conf 中的 WHISPER_MODEL
    sed -i "s/^WHISPER_MODEL = .*/WHISPER_MODEL = $model_name/" eval.conf

    # 3. 执行 makes
    make > log/librispeech_eval_$model_name.log 2>&1

    # 4. 复制结果文件到目标目录
    cp "$model_name.txt" "$RESULT_DIR/"
done

echo "All models processed. Results are in $RESULT_DIR."
