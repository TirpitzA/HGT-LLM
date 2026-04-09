#!/bin/bash

# 定义要测试的模型列表和数据集列表
MODELS=("Ours" "WDCNN" "TCNN" "QCNN")
DATASETS=("XJTU-SY")

#CWRU DIRG HIT IMS JNU JUST MFPT PU XJTU
#联网搜索这些数据集各自所测试的机械结构由哪些部件组成，有哪些故障类型，是否都由内外圈故障组成

# 创建日志文件夹
mkdir -p logs
export PYTHONPATH="/root/autodl-tmp/XJTU_Multimodal_LLM:$PYTHONPATH"
echo "Starting One-Click Benchmark Suite..."

for dataset in "${DATASETS[@]}"; do
    echo "======================================"
    echo "Evaluating on Dataset: $dataset"
    echo "======================================"
    
    for model in "${MODELS[@]}"; do
        log_file="logs/run_${model}_${dataset}.log"
        echo "Training $model ... (Log saved to $log_file)"
        
        # 运行 Python 脚本，并将输出重定向到日志文件
        python benchmark_train.py \
            --model "$model" \
            --dataset "$dataset" \
            --epochs 30 \
            > "$log_file" 2>&1
            
        echo "Finished $model."
    done
done

echo "All experiments completed!"