#!/bin/bash

# =================================================================
# 1. 初始化 Conda 环境
# 注意：在脚本中直接用 'conda activate' 可能会失效
# 需要先 source conda 的初始化脚本，通常路径为 /root/miniconda3
# =================================================================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mypaper

# =================================================================
# 2. 定义任务执行函数
# 这样做是为了方便捕获整个过程的状态
# =================================================================
run_tasks() {
    echo "--- [$(date)] 开始训练 Qwen2-1.5B LoRA ---"
    python src/train.py --config configs/cwru_config.yaml
    
    echo "--- [$(date)] 开始评估 HGT-LLM (Token Replacement) ---"
    python src/evaluate.py --config configs/cwru_config.yaml
    
    echo "--- [$(date)] 开始评估 BearLLM ---"
    python src/evaluate_bearllm.py --config configs/cwru_config.yaml
}

# 执行任务
run_tasks

# =================================================================
# 3. 关机逻辑
# 无论上面的 run_tasks 是正常结束还是中途 crash 报错，
# 脚本都会继续向下运行执行 shutdown
# =================================================================
echo "--- [$(date)] 所有任务已尝试执行完毕，准备关机 ---"

# AutoDL 专用关机指令
shutdown
