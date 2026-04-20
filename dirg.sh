# 1. 训练物理骨干网络 (监督学习，产出纯正的领域知识特征权重)
python src/train_physics.py --config configs/cwru_config.yaml

# 2. 知识注入：根据上一步获取的最佳权重，构建 7:2:1 的多模态问答指令集
python src/generate_dataset.py --config configs/cwru_config.yaml

# 3. 联合微调：训练 Alignment Layer 和 LLM LoRA
python src/train.py --config configs/cwru_config.yaml

# 4. 严谨评估：开启无干预推断模式，输出准确率(Acc)和语言模型困惑度(PPL)
python src/evaluate.py --config configs/cwru_config.yaml
