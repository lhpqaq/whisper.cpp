#!/bin/bash
./build/bin/whisper-bench -m ./models/ggml-largev3-q80.bin &
PROGRAM_NAME="./build/bin/whisper-bench -m ./models/ggml-largev3-q80.bin"
PID=$(pgrep -n "./build/bin/whisper-bench")  # 取最新启动的同名进程
if [ -z "$PID" ]; then
    echo "Program not running"
    exit 1
fi

echo "Time,RSS(KB),GPU_Mem(MiB)" > resource_usage.csv

while kill -0 $PID 2>/dev/null; do
    # 内存
    RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
    
    # 显存（NVIDIA）
    GPU_MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -v p=$PID '$1 == p {print $2}' || echo 0)
    
    echo "$(date +%s),$RSS,$GPU_MEM" >> resource_usage.csv
    sleep 0.5
done

echo "Monitoring finished. Data saved to resource_usage.csv"