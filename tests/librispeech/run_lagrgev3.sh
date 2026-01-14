#!/bin/bash

# 模型列表
MODELS=("largev3-q40" "large-v3" "largev3-q2k-r1" "largev3-q2k-r2" "largev3-q2k-r3" "largev3-q2k-r4" "largev3-q2k-r5" "largev3-q2k-r6" "largev3-q2k-r7" "largev3-q2k-1" "largev3-q2k-2" "largev3-q2k-3" "largev3-q2k-4" "largev3-q2k-5" "largev3-q2k-6" "largev3-q2k-7" "largev3-q80" "largev3-q2k")

# 结果存储目录
RESULT_DIR="../../result/largev3"

# 确保结果目录存在
mkdir -p "$RESULT_DIR"

# 时间记录文件
TIME_LOG="$RESULT_DIR/execution_times.txt"

# 清空或创建时间记录文件
echo "Model Execution Times" > "$TIME_LOG"
echo "=====================" >> "$TIME_LOG"
echo "" >> "$TIME_LOG"

# 记录总开始时间
TOTAL_START=$(date +%s)
echo "Total execution started at: $(date)" >> "$TIME_LOG"
echo "" >> "$TIME_LOG"

# 遍历模型列表
for model_name in "${MODELS[@]}"; do
    echo "Processing model: $model_name"
    
    # 记录模型开始时间
    MODEL_START=$(date +%s)
    echo "Model: $model_name" >> "$TIME_LOG"
    echo "  Start time: $(date)" >> "$TIME_LOG"

    # 1. 执行 make clean
    make clean

    # 2. 修改 eval.conf 中的 WHISPER_MODEL
    sed -i "s/^WHISPER_MODEL = .*/WHISPER_MODEL = $model_name/" eval.conf

    # 3. 执行 makes
    make > log/librispeech_eval_$model_name.log 2>&1

    # 4. 复制结果文件到目标目录
    cp "$model_name.txt" "$RESULT_DIR/"
    
    # 记录模型结束时间和耗时
    MODEL_END=$(date +%s)
    MODEL_DURATION=$((MODEL_END - MODEL_START))
    echo "  End time: $(date)" >> "$TIME_LOG"
    echo "  Duration: $MODEL_DURATION seconds ($(date -u -d @${MODEL_DURATION} +"%H:%M:%S"))" >> "$TIME_LOG"
    echo "" >> "$TIME_LOG"
done

# 记录总结束时间
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "=====================" >> "$TIME_LOG"
echo "Total execution ended at: $(date)" >> "$TIME_LOG"
echo "Total duration: $TOTAL_DURATION seconds ($(date -u -d @${TOTAL_DURATION} +"%H:%M:%S"))" >> "$TIME_LOG"

echo "All models processed. Results are in $RESULT_DIR."
echo "Execution times saved to $TIME_LOG"