#!/bin/bash

# 开启严格模式：任何一步指令报错返回非 0 状态码，脚本就会立刻停止，不再执行后续步骤
set -e

# 设置陷阱（Trap）：无论脚本是正常执行完毕，还是中途报错退出，都会触发关机指令
trap 'echo "任务结束或中途报错，准备关机..."; sleep 5; shutdown -h now' EXIT

echo "========== [1/4] 开始训练物理骨干网 =========="
python src/train_physics.py --config configs/dirg_config.yaml --epochs 40 --batch_size 128

echo "========== [2/4] 开始生成多模态对齐数据集 =========="
python src/generate_dataset.py --config configs/dirg_config.yaml

echo "========== [3/4] 开始评估 HGT-LLM =========="
python src/evaluate.py --config configs/dirg_config.yaml

echo "========== [4/4] 开始评估 BearLLM =========="
python src/evaluate_bearllm.py --config configs/dirg_config.yaml

echo "========== 所有任务顺利执行完毕！ =========="
