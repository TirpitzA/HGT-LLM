#!/bin/bash
# 自动在后台运行 DIRG 评估，并确保 session 断开后不断连 (nohup)

LOG_FILE="results/dirg_full_log_v2.txt"

echo "--------------------------------------------------" >> $LOG_FILE
echo "🚀 [$(date)] 重新启动评估 (Background Mode)" >> $LOG_FILE
echo "--------------------------------------------------" >> $LOG_FILE

nohup /root/miniconda3/envs/mypaper/bin/python -u src/evaluate.py \
  --config configs/dirg_config.yaml \
  --checkpoint bearllm_weights/checkpoints_dirg/best_checkpoint \
  >> $LOG_FILE 2>&1 &

echo "✅ 评测已在后台启动 (PID: $!)"
echo "🔎 正在实时输出至: $LOG_FILE"
