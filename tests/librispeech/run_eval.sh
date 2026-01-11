#!/bin/bash

# 模型列表
MODELS=("small-1" "small-2" "small-3" "small-4" "small-5" "small-6" "small-7")

# 结果存储目录
RESULT_DIR="../../result/b8t4"

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
