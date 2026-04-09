<div align="center">
<a href="https://github.com/SIA-IDE/BearLLM">
<img src="https://raw.githubusercontent.com/SIA-IDE/BearLLM/refs/heads/main/docs/images/logo.svg" width="200" alt="logo"/>
</a>
<h1>BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation</h1>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/Pytorch-latest-orange"></a>
<a href="https://arxiv.org/abs/2408.11281"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-B31B1B"></a>
<a href="https://huggingface.co/datasets/SIA-IDE/MBHM"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-🤗-FFFDF5"></a>
<a href="https://github.com/SIA-IDE/BearLLM"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SIA-IDE/BearLLM"></a>
</div>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/SIA-IDE/BearLLM/blob/main/docs/README_zh.md">简体中文</a>
    </p>
</h4>

## 🔥 NEWS
- **[2025-04-11]** 🎉 The [AAAI-25 Proceedings](https://aaai.org/proceeding/aaai-39-2025/) are now officially published! Our [conference paper](https://ojs.aaai.org/index.php/AAAI/article/view/34188) is included. We welcome you to read and cite it!
- **[2025-03-06]** 🌟 The complete dataset and code are now officially open source!
- **[2024-12-11]** ⏫ We are now working on making the code of BearLLM public. Stay tuned!
- **[2024-12-10]** 🎉 The BearLLM paper is accepted by the Thirty-Ninth AAAI Conference on Artificial Intelligence ([AAAI-25](https://aaai.org/conference/aaai/aaai-25/)).
- **[2024-08-21]** 📝 The preprint of the BearLLM paper is available on arXiv. Check the [paper page](https://arxiv.org/abs/2408.11281) for more details.

## 📅 TODO
- [ ] Improve related comments and documentation.
- [x] Upload the complete BearLLM demo code.
- [x] Upload the health management corpus of the MBHM dataset.
- [x] Collect the codes for pre-training and fine-tuning BearLLM.
- [x] Collect the codes of BearLLM's classification network and other comparison models.
- [x] Upload the vibration signal portion of the MBHM dataset.

## 📚 Introduction
The [MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM) dataset is the first multimodal dataset designed for the study of bearing health management. It is divided into two parts: vibration signals and health management corpus. The vibration signals and condition information are derived from 9 publicly available datasets, and are still under continuous updating and improvement. The thousands of working conditions pose more difficult challenges for the identification model and better represent real-world usage scenarios.

[BearLLM](https://github.com/SIA-IDE/BearLLM) is a prior knowledge-enhanced bearing health management framework with a unified vibration signal representation. This framework transforms the signal to be tested into the frequency domain, enabling effective identification of spectral differences compared to the vibration signal under fault-free conditions. By aligning the vibration signal with the fault semantic embedding, we achieve a unified natural language response for various health management tasks through a fine-tuned language model with low computational overhead. Experiments demonstrate that this framework achieves leading performance under thousands of working conditions.

## 💻 Requirements

The code is implemented in Python 3.12. The required packages are listed in the `requirements.txt` file. You can install the required packages by running the following command:

```bash
conda create --name bearllm python=3.12
conda activate bearllm
pip install -r requirements.txt
```


## 🚀 Quick Start

### 1. Download Demo Data / Use Your Own Data

First, you need to download the `demo_data.json` from the [MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM/tree/main) dataset.
For users in mainland China, you can use the [mirror link](https://hf-mirror.com/datasets/SIA-IDE/MBHM/tree/main) to speed up the download:

Or, you can also build your own test data in the same format:
`instruction`: Text instruction for health management task.
`vib_data`: Vibration signal data to be identified, with a required duration of 1 second.
`ref_data`: Reference vibration signal data without faults, with a required duration of 1 second.

```json
{
    "instruction": "xxx.",
    "vib_data": [1.0, 0.0, 1.0, ...],
    "ref_data": [1.0, 0.0, 1.0, ...],
}
```

### 2. Download Weights

You can download the pre-trained weights of [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main) from Hugging Face.

Additionally, you need to download the weights of [BearLLM](https://huggingface.co/SIA-IDE/BearLLM/tree/main).

### 3. Organize Files

It is recommended to organize the weights and test data as follows:

```
BearLLM/
├── qwen_weights/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   └── other files...
├── bearllm_weights/
│   ├── vibration_adapter.pth
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── mbhm_dataset/
    └── demo_data.json 
```

### 4. Run Code
First, copy the `.env.example` file to `.env` and modify the data paths inside.
Then, you can run the code using the following command:

```bash
python run_demo.py
```

## ⚙️ Development

### 1. Download Dataset

First, you need to download the following files from the [MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM/tree/main) dataset. For users in mainland China, you can use the [mirror link](https://hf-mirror.com/datasets/SIA-IDE/MBHM/tree/main) to speed up the download:

- `data.hdf5`: Contains the vibration signal data.
- `corpus.json`: Contains the health management corpus.
- `metadata.sqlite`: Contains metadata information of the dataset.

### 2. Download Weights

You can download the pre-trained weights of [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main) from Hugging Face.

### 3. Modify Environment Variables

Copy the `.env.example` file to `.env` and modify the data paths inside.

### 4. Pre-train and Fine-tune Model

Pre-train according to `src/pre_training.py`.
Fine-tune according to `src/fine_tuning.py`.

## 📖 Citation
Please cite the following paper if you use this study in your research:

```
@article{pengBearLLMPriorKnowledgeEnhanced2025,
  title = {{{BearLLM}}: {{A Prior Knowledge-Enhanced Bearing Health Management Framework}} with {{Unified Vibration Signal Representation}}},
  author = {Peng, Haotian and Liu, Jiawei and Du, Jinsong and Gao, Jie and Wang, Wei},
  year = {2025},
  month = apr,
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {39},
  number = {19},
  pages = {19866--19874},
  issn = {2374-3468},
  doi = {10.1609/aaai.v39i19.34188},
  urldate = {2025-04-11},
}
```
强制清空旧特征空间与权重
这一步是“回到同一起跑线”的关键，彻底删除之前所有的 CWRU 缓存数据、FCN 预训练权重以及 LoRA 微调权重：

Bash
rm -rf /root/autodl-tmp/BearLLM/data/processed/cwru*
rm -rf /root/autodl-tmp/BearLLM/bearllm_weights/fcn/*
rm -rf /root/autodl-tmp/BearLLM/bearllm_weights/lora/*
2. 重新运行预训练（生成对齐缓存）
因为这一步会触发 cwru.py 中的 process_and_cache_cwru()，将对比项目的 X_train.npy 提取并平铺到 24000 长度。由于我们在代码里做过内存和效率优化，它很快就能跑完，所以建议直接在前台运行，以便你第一时间看到“数据对齐完毕”的成功提示：

Bash
python -m src.pre_training
3. 后台挂起微调任务
确认预训练顺利跑完并保存了 FCN 权重后，使用 nohup 将耗时最长的微调任务挂载到后台，压榨 4090 的算力：

Bash
nohup python -u -m src.fine_tuning > finetune.log 2>&1 &
4. 后台挂起自动测试与关机脚本
最后，为了给你省下不必要的服务器租用费，顺手把我们之前写的监控脚本也挂到后台。它会在微调结束后自动跑完测试并关机：

Bash
nohup /root/autodl-tmp/BearLLM/auto_eval_shutdown.sh > /dev/null 2>&1 &
💡 运行小贴士：
执行完第 3 步和第 4 步后，终端会分别返回两个进程号（例如 [1] 12345 和 [2] 12346）。
此时你可以输入以下命令来“偷窥”一下微调是不是正常开始了，顺便看看速度有没有大幅飙升：

Bash
tail -f finetune.log
确认进度条跑起来之后，按 Ctrl+C 退出日志查看，直接关掉电脑去休息就可以了。祝你一觉醒来拿到完美的 CWRU 测试成绩！