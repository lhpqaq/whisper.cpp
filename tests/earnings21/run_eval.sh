#!/bin/bash

# 模型列表
# MODELS=("medium-mixed-a0p8-b0p2")
MODELS=("base" "base-mixed-a0p2-b0p8" "base-mixed-a0p8-b0p2" "small-mixed-a0p8-b0p2" "small-mixed-a0p2-b0p8" "medium" "medium-mixed-a0p8-b0p2" "medium-mixed-a0p2-b0p8" "large-v3-turbo-mixed-a0p2-b0p8" "large-v3-turbo-mixed-a0p8-b0p2" "large-v3-turbo")

# 假设模型文件所在的目录（请根据实际情况修改此路径）
# 如果模型就在当前目录下，保持 "." 即可
MODEL_PATH_PREFIX="../../models/ggml-"
MODEL_EXTENSION=".bin" # 假设模型后缀是 .bin，如果没有后缀请设为空 ""

# 结果存储目录
RESULT_DIR="../../result/earnings/mix"
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
    echo "-----------------------------------"
    echo "Processing model: $model_name"

    # 1. 执行 make clean
    make clean

    # 2. 修改 eval.conf 中的 WHISPER_MODEL
    # 使用 ^WHISPER_MODEL 来精确匹配行首
    sed -i "s/^WHISPER_MODEL = .*/WHISPER_MODEL = $model_name/" eval.conf

    # 3. 执行 make 并记录日志
    # 增加了一个执行成功的判断
    if make > "log/librispeech_eval_$model_name.log" 2>&1; then
        echo "Make successful for $model_name"
        
        # 4. 复制结果文件到目标目录
        if [ -f "$model_name.txt" ]; then
            cp "$model_name.txt" "$RESULT_DIR/"
            echo "Result saved to $RESULT_DIR"
        else
            echo "Warning: Result file $model_name.txt not found!"
        fi
    else
        echo "Error: Make failed for $model_name. Check log/librispeech_eval_$model_name.log"
    fi
done

echo "-----------------------------------"
echo "All models processed. Results are in $RESULT_DIR."