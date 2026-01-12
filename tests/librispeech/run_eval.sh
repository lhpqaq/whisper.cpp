#!/bin/bash

# 模型列表
MODELS=("small-q50" "small-q6k" "small-q2k-1" "small-q2k-2" "small-q2k-3" "small-q2k-4" "small-q2k-5" "small-q2k-6" "small-q2k-7" "small-q2k-r1" "small-q2k-r2" "small-q2k-r3" "small-q2k-r4" "small-q2k-r5" "small-q2k-r6" "small-q2k-r7")

# 结果存储目录
RESULT_DIR="../../result/b8t2"

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
