import os
import scipy.io as sio
import numpy as np
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import dotenv_values
from .dcn import dcn

env = dotenv_values()
# 根目录
cwru_dir = env.get('CWRU_DATASET', '/root/autodl-tmp/XJTU_Multimodal_LLM/data/CWRU')
# 【修正1】：指向 raw 子目录
raw_dir = os.path.join(cwru_dir, 'raw') 
processed_dir = env.get('CWRU_PROCESSED', './data/processed')

os.makedirs(processed_dir, exist_ok=True)
cache_path = os.path.join(processed_dir, 'cwru_dataset.json')
data_cache_path = os.path.join(processed_dir, 'cwru_signals.npy')
corpus_path = os.path.join(processed_dir, 'cwru_corpus.json')

# 【修正2】：严格匹配你的实际文件名，取消下划线，修改 Normal 关键字
CWRU_CLASSES = {
    "Time_Normal": 0,       # 对应 Time_Normal_1_098.mat
    "IR007": 1,             # 对应 IR007_1_110.mat
    "IR014": 2,             # 对应 IR014_1_175.mat
    "IR021": 3,             # 对应 IR021_1_214.mat
    "B007": 4,              # 对应 B007_1_123.mat
    "B014": 5,              # 对应 B014_1_190.mat
    "B021": 6,              # 对应 B021_1_227.mat
    "OR007": 7,             # 对应 OR007_6_1_136.mat
    "OR014": 8,             # 对应 OR014_6_1_202.mat
    "OR021": 9              # 对应 OR021_6_1_239.mat
}

DESCRIPTION_TEXT = [
    "Fault-Free", "Minor Inner Ring Fault", "Moderate Inner Ring Fault", "Severe Inner Ring Fault",
    "Minor Ball Fault", "Moderate Ball Fault", "Severe Ball Fault",
    "Minor Outer Ring Fault", "Moderate Outer Ring Fault", "Severe Outer Ring Fault"
]

def process_and_cache_cwru():
    """解析 CWRU raw 目录下的 .mat 文件，按 BearLLM 要求的 24000 长度滑窗切分并缓存"""
    if os.path.exists(data_cache_path) and os.path.exists(cache_path) and os.path.exists(corpus_path):
        return

    print("正在处理 CWRU 原始数据并生成缓存，这可能需要几分钟...")
    signals = []
    labels = []
    window_size = 24000
    stride = 12000 # 50% 重叠率

    # 获取所有 Normal 数据作为 Reference 池
    ref_indices = []

    # 遍历 raw 文件夹
    for file_name in os.listdir(raw_dir):
        if not file_name.endswith('.mat'): continue
        
        # 解析标签
        label = -1
        for key, val in CWRU_CLASSES.items():
            if key in file_name:
                label = val
                break
        if label == -1: 
            print(f"警告: 无法识别文件 {file_name} 的分类，已跳过。")
            continue 

        mat_data = sio.loadmat(os.path.join(raw_dir, file_name))
        
        # 提取驱动端 (DE) 振动信号
        # CWRU 的键名通常形如 'X105_DE_time' 或 'X098_DE_time'
        de_key = [k for k in mat_data.keys() if 'DE_time' in k]
        
        if not de_key:
            print(f"警告: 文件 {file_name} 中未找到 DE_time 数据，已跳过。")
            continue
            
        raw_signal = mat_data[de_key[0]].flatten()
        
        # 长度检查：如果原始信号不足 24000，原项目的 dcn 函数中的 pad_or_cut 会自动补零
        # 这里进行滑窗切分
        for start in range(0, max(1, len(raw_signal) - window_size + 1), stride):
            segment = raw_signal[start:start+window_size]
            
            # 使用原项目的 DCN 处理标准化和 DCT，且会自动 pad 到 24000
            processed_segment = dcn(segment, length=window_size) 
            signals.append(processed_segment)
            labels.append(label)
            
            if label == 0:
                ref_indices.append(len(signals) - 1)

    signals = np.array(signals, dtype=np.float32)
    np.save(data_cache_path, signals)

    # 划分训练/验证/测试集，并构建指令集 (Corpus)
    dataset_info = {'train': [], 'val': [], 'test': []}
    corpus_data = []
    
    for idx, label in enumerate(labels):
        # 随机分配一个 Normal 信号作为 reference
        ref_idx = random.choice(ref_indices) if ref_indices else idx 
        
        rand_val = random.random()
        if rand_val < 0.7: subset = 'train'
        elif rand_val < 0.9: subset = 'val'
        else: subset = 'test'
        
        dataset_info[subset].append([idx, ref_idx, label])
        
        # 生成 LLM 指令数据
        if subset == 'train':
            state_desc = DESCRIPTION_TEXT[label]
            corpus_data.append({
                "id": len(corpus_data),
                "label_id": label,
                "vib_id": idx,
                "ref_id": ref_idx,
                "instruction": f"The dynamic sensor captured this vibration signal. Can you analyze the bearing status based on it? #state_place_holder#",
                "response": f"Based on the unified vibration signal representation, the bearing exhibits a {state_desc}."
            })

    with open(cache_path, 'w') as f:
        json.dump(dataset_info, f)
    with open(corpus_path, 'w') as f:
        json.dump(corpus_data, f)
    print(f"CWRU 数据处理完成！共生成 {len(signals)} 个样本，特征维度为 24000。")

class VibDataset(Dataset):
    def __init__(self, subset_info, signals):
        self.subset_info = subset_info
        self.signals = signals

    def __len__(self):
        return len(self.subset_info)

    def __getitem__(self, idx):
        file_id, ref_id, label = self.subset_info[idx]
        data = self.signals[file_id]
        ref = self.signals[ref_id]
        combined_data = np.array([data, ref]) 
        return combined_data, label

def get_loaders(batch_size, num_workers):
    process_and_cache_cwru()
    with open(cache_path, 'r') as f:
        dataset_info = json.load(f)
    signals = np.load(data_cache_path, mmap_mode='r') 
    
    train_set = VibDataset(dataset_info['train'], signals)
    val_set = VibDataset(dataset_info['val'], signals)
    test_set = VibDataset(dataset_info['test'], signals)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)
    return train_loader, val_loader, test_loader

class CorpusDataset:
    def __init__(self):
        process_and_cache_cwru()
        self.signals = np.load(data_cache_path, mmap_mode='r')
        self.corpus = json.load(open(corpus_path, 'r'))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        corpus_data = self.corpus[idx]
        sample_id = corpus_data['id']
        instruction = corpus_data['instruction']
        response = corpus_data['response']
        ref_id = corpus_data['ref_id']
        vib_id = corpus_data['vib_id']
        
        vib_data = self.signals[vib_id]
        ref_data = self.signals[ref_id]
        vib = np.array([vib_data, ref_data])
        label_id = corpus_data['label_id']
        
        return sample_id, label_id, vib, instruction, response