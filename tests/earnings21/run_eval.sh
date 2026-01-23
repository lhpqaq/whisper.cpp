#!/bin/bash

# 模型列表
MODELS=("small-q80" "small-q40" "small-best" "small-6" "small" "small-1" "small-2" "small-3" "small-4" "small-5" "small-7" "small-r1" "small-r2" "small-r3" "small-r4" "small-r5" "small-r6" "small-r7")
# MODELS=("largev3-q40" "largev3-q2k" "large-v3" "largev3-q2k-r1" "largev3-q2k-r2" "largev3-q2k-r3" "largev3-q2k-r4" "largev3-q2k-r5" "largev3-q2k-r6" "largev3-q2k-r7" "largev3-q2k-1" "largev3-q2k-2" "largev3-q2k-3" "largev3-q2k-4" "largev3-q2k-5" "largev3-q2k-6" "largev3-q2k-7" "largev3-q80")

# 结果存储目录
RESULT_DIR="../../result/earnings/largev3"

# 确保结果目录存在
mkdir -p "$RESULT_DIR"

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
