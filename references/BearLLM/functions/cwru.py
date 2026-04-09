import os
import numpy as np
import json
import random
import torch
import gc
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dotenv import dotenv_values
from .dcn import dcn

env = dotenv_values('.env') 
data_dir = env.get('DATA_DIR', '/root/autodl-tmp/BearLLM')

# 【核心修改点】直接指向 XJTU_Multimodal_LLM 已经处理好的数据目录
cwru_processed_source = '/root/autodl-tmp/XJTU_Multimodal_LLM/data'
processed_dir = env.get('CWRU_PROCESSED', os.path.join(data_dir, 'data', 'processed'))

os.makedirs(processed_dir, exist_ok=True)
cache_path = os.path.join(processed_dir, 'cwru_dataset.json')
data_cache_path = os.path.join(processed_dir, 'cwru_signals.npy')
corpus_path = os.path.join(processed_dir, 'cwru_corpus.json')

# CWRU 的 10 分类描述文本，严格对应 Label 0-9
DESCRIPTION_TEXT = [
    "Fault-Free", 
    "Minor Inner Ring Fault", 
    "Moderate Inner Ring Fault", 
    "Severe Inner Ring Fault",
    "Minor Ball Fault", 
    "Moderate Ball Fault", 
    "Severe Ball Fault",
    "Minor Outer Ring Fault", 
    "Moderate Outer Ring Fault", 
    "Severe Outer Ring Fault"
]

def process_and_cache_cwru():
    if os.path.exists(data_cache_path) and os.path.exists(cache_path) and os.path.exists(corpus_path):
        return

    print("[INFO] 正在同步 XJTU_Multimodal_LLM 的 CWRU 数据集划分 (7:2:1)...")
    
    # 1. 强行加载对比项目的 .npy 划分文件，保证样本与对比方法 100% 一模一样
    splits = ['train', 'val', 'test']
    raw_data = {}
    for split in splits:
        x_path = os.path.join(cwru_processed_source, f'X_{split}.npy')
        y_path = os.path.join(cwru_processed_source, f'y_{split}.npy')
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"[ERROR] 找不到文件: {x_path}\n请确保在 XJTU_Multimodal_LLM 中已经生成了该文件。")
        raw_data[split] = (np.load(x_path), np.load(y_path))

    signals = []
    split_data = {'train': [], 'val': [], 'test': []}
    ref_indices = {'train': [], 'val': [], 'test': []}
    
    # BearLLM 特征编码器需要的标准输入长度
    window_size = 24000
    
    print(f"[INFO] 正在将样本平铺重采样并转换为 BearLLM 兼容特征...")
    
    for subset in splits:
        X_raw, y_raw = raw_data[subset]
        for i in tqdm(range(len(X_raw)), desc=f"处理 {subset} 集"):
            # 展平特征，应对可能的 shape 差异
            segment = X_raw[i].flatten()
            
            # 【核心对齐逻辑】将较短的信号平铺(Tile)至 24000 长度
            # 在不改变原始频率特性的前提下，完美适配 BearLLM 的 FCN 输入层
            if len(segment) < window_size:
                repeats = (window_size // len(segment)) + 1
                segment = np.tile(segment, repeats)[:window_size]
            elif len(segment) > window_size:
                segment = segment[:window_size]
                
            processed_segment = dcn(segment, length=window_size) 
            
            # 兼容 One-Hot 编码 或 标量数字 label
            label = int(np.argmax(y_raw[i])) if y_raw[i].ndim > 0 and len(y_raw[i]) > 1 else int(y_raw[i])
            
            global_idx = len(signals)
            signals.append(processed_segment)
            split_data[subset].append((global_idx, label))
            
            # 记录健康样本(Fault-Free)作为提示词 Reference
            if label == 0:  
                ref_indices[subset].append(global_idx)

    print(f"\n[INFO] 共转换了 {len(signals)} 个样本，正在写入缓存...")
    signals = np.array(signals, dtype=np.float32)
    np.save(data_cache_path, signals)
    del signals
    gc.collect()

    dataset_info = {'train': [], 'val': [], 'test': []}
    corpus_data = []
    
    for subset in ['train', 'val', 'test']:
        for global_idx, label in split_data[subset]:
            if ref_indices[subset]:
                ref_idx = random.choice(ref_indices[subset])
            else:
                ref_idx = global_idx # Fallback
                
            dataset_info[subset].append([global_idx, ref_idx, label])
            
            if subset == 'train':
                # 防越界保护
                state_desc = DESCRIPTION_TEXT[label] if label < len(DESCRIPTION_TEXT) else f"Unknown State {label}"
                corpus_data.append({
                    "id": len(corpus_data), 
                    "label_id": label,
                    "vib_id": global_idx,
                    "ref_id": ref_idx,
                    "instruction": "The dynamic sensor captured this vibration signal. Can you analyze the bearing status based on it? #state_place_holder#",
                    "response": f"Based on the unified vibration signal representation, the bearing exhibits a {state_desc}."
                })

    with open(cache_path, 'w') as f:
        json.dump(dataset_info, f)
    with open(corpus_path, 'w') as f:
        json.dump(corpus_data, f)
        
    print(f"[SUCCESS] 数据对齐完毕！完全复用了 XJTU_Multimodal_LLM 的 CWRU 数据集。")
    print(f" - Train 样本: {len(dataset_info['train'])}")
    print(f" - Val   样本: {len(dataset_info['val'])}")
    print(f" - Test  样本: {len(dataset_info['test'])}")

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