# HGT-LLM

本项目是基于轴承数据集的多模态故障诊断大语言模型。它通过将底层的物理特征（由分层解释性网络提取）与高层的领域专家知识（由大语言模型处理）相结合，实现了具有解释性的自动故障诊断。

## 项目结构 (参考 BearLLM 整理)

```text
XJTU_Multimodal_LLM/
├── data/                    # 数据集文件 (.json, .pt)
├── models/                  # 模型定义 (Physics Model & Multimodal Wrapper)
├── src/                     # 核心执行代码 (生成、训练、评估)
├── utils/                   # 工具类与物理拓扑、专家知识逻辑
├── checkpoints/             # 训练输出的检查点与适配器
├── scripts/                 # 快捷运行脚本
└── README.md                # 本文件
```

## 核心流程

1. **数据集生成**: 从预训练的物理网络提取深层语义特征，并结合 DIG 生成专家推理链。
   ```bash
   conda run -n llama-factory python src/generate_dataset.py
   ```

2. **多模态联合微调**: 使用 LoRA 对 Qwen2.5 进行微调，同时训练 AlignmentLayer 实现特征对齐。
   ```bash
   conda run -n llama-factory python src/train.py
   ```

3. **多模态评估**: 验证模型在未见过的振动信号上的诊断结果与推理能力。
   ```bash
   bash scripts/run_eval.sh
   ```

cwru数据集运行方法：
执行流程：
1.先运行 models/cwru_physics_pipeline.py 完成物理骨干网的训练，生成 best_model.pth 和 7:2:1 划分的 .npy 原始文件。

2.生成 JSON：然后运行此 src/generate_dataset.py 脚本，它会自动读取 CWRU 专用的 10 分类图谱和物理模型，生成用于大模型微调的 JSON 文件和对应的特征向量 .pt 文件。

3.模型微调：最后直接进入 src/train.py 开始 Qwen 大模型的联合微调。

## 环境要求
- Python 3.9+
- PyTorch 2.1+
- Transformers, PEFT
- vLLM (可选，用于加速推理)

## DIRG数据集下载方法：wget -O VariableSpeedAndLoad.zip "https://zenodo.org/records/3559553/files/VariableSpeedAndLoad.zip?download=1"

（apt-get update && apt-get install aria2 -y
   aria2c -s 16 -x 16 -k 1M -c -o VariableSpeedAndLoad.zip "https://zenodo.org/records/3559553/files/VariableSpeedAndLoad.zip?download=1"//多线程接管下载
）
PS：下载之前记得开梯子


DIRG说明文档：wget "https://zenodo.org/records/3559553/files/Description%20and%20analysis%20of%20open%20access%20data.pdf?download=1"


## XJTU-SY数据集下载方法:gdown --folder --id 1_ycmG46PARiykt82ShfnFfyQsaXv3_VK (需要先安装gdown:pip install gdown)
ps:开代理

