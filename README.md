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

## 核心流程与运行方法 (基于 YAML 配置)

本项目采用 YAML 配置文件驱动，支持 CWRU、DIRG 和 XJTU 等多种数据集。

### 执行流程

1. **生成多模态数据集**: 结合预训练物理骨干网提取特征并生成专家推理链 JSON。
   ```bash
   python src/generate_dataset.py --config configs/cwru_config.yaml
   ```

2. **多模态联合微调**: 训练 AlignmentLayer 并使用 LoRA 微调大语言模型。
   ```bash
   python src/train.py --config configs/cwru_config.yaml
   ```

3. **模型性能评估**: 验证诊断准确率与推理能力。
   ```bash
   python src/evaluate.py --config configs/cwru_config.yaml
   ```

### 核心参数说明

配置文件位于 `configs/` 目录下（如 `cwru_config.yaml`），关键参数如下：

| 参数 | 意义 | 设置方法 |
| :--- | :--- | :--- |
| `dataset_name` | 数据集标识 | 修改为 "CWRU", "DIRG" 或 "XJTU" |
| `data_dir` | 数据存储路径 | 指向包含 `.npy` 原始文件的目录 |
| `checkpoint_path` | 物理模型权重 | 指向预训练的 `.pth` 物理骨干网文件 |
| `model_params` | 物理网络结构参数 | 包含 `in_channels`, `num_classes`, `slice_length` 等 |
| `labels` | 故障标签映射 | 定义数字分类 ID 到自然语言描述的映射 |
| `reasoning_chains` | 推理逻辑模板 | 根据物理拓扑定义的专家诊断推理链路 |

### 如何切换数据集
只需在运行指令时指定对应的配置文件即可：
- **CWRU**: `--config configs/cwru_config.yaml`
- **DIRG**: `--config configs/dirg_config.yaml`
- **XJTU**: `--config configs/xjtu_config.yaml`

## 环境要求
- Python 3.9+
- PyTorch 2.1+
- Transformers, PEFT
- vLLM (可选，用于加速推理)

## DIRG数据集下载方法：
wget -O VariableSpeedAndLoad.zip "https://zenodo.org/records/3559553/files/VariableSpeedAndLoad.zip?download=1"

（apt-get update && apt-get install aria2 -y
   aria2c -s 16 -x 16 -k 1M -c -o VariableSpeedAndLoad.zip "https://zenodo.org/records/3559553/files/VariableSpeedAndLoad.zip?download=1"//多线程接管下载
）
PS：下载之前记得开梯子


DIRG说明文档：
wget "https://zenodo.org/records/3559553/files/Description%20and%20analysis%20of%20open%20access%20data.pdf?download=1"


## XJTU-SY数据集下载方法:
gdown --folder --id 1_ycmG46PARiykt82ShfnFfyQsaXv3_VK (需要先安装gdown:pip install gdown)
ps:开代理

## JUST数据集下载方法:
```bash
# Condition 1
wget -c "https://data.mendeley.com/public-api/zip/hwg8v5j8t6/download/1" -O JUST_condition1.zip
# Condition 2
wget -c "https://data.mendeley.com/public-api/zip/rcxgmdxhbr/download/1" -O JUST_condition2.zip
```

## IMS数据集下载方法:
```bash
wget -c "https://data.nasa.gov/docs/legacy/IMS.zip" -O IMS_Bearings.zip
```


