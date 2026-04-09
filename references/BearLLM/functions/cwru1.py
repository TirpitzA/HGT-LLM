import os
import scipy.io as sio
import numpy as np
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import dotenv_values
from .dcn import dcn

env = dotenv_values('.env') # 显式指定读取根目录下的 .env
cwru_dir = env.get('CWRU_DATASET', '/root/autodl-tmp/XJTU_Multimodal_LLM/data/CWRU')
raw_dir = os.path.join(cwru_dir, 'raw') 
processed_dir = env.get('CWRU_PROCESSED', '/root/autodl-tmp/BearLLM/data/processed')

os.makedirs(processed_dir, exist_ok=True)
cache_path = os.path.join(processed_dir, 'cwru_dataset.json')
data_cache_path = os.path.join(processed_dir, 'cwru_signals.npy')
corpus_path = os.path.join(processed_dir, 'cwru_corpus.json')

CWRU_CLASSES = {
    "Time_Normal": 0, "IR007": 1, "IR014": 2, "IR021": 3,
    "B007": 4, "B014": 5, "B021": 6,
    "OR007": 7, "OR014": 8, "OR021": 9
}

DESCRIPTION_TEXT = [
    "Fault-Free", "Minor Inner Ring Fault", "Moderate Inner Ring Fault", "Severe Inner Ring Fault",
    "Minor Ball Fault", "Moderate Ball Fault", "Severe Ball Fault",
    "Minor Outer Ring Fault", "Moderate Outer Ring Fault", "Severe Outer Ring Fault"
]

def process_and_cache_cwru():
    """采用严格的 Chronological Split 防止时序泄露与特征泄露"""
    if os.path.exists(data_cache_path) and os.path.exists(cache_path) and os.path.exists(corpus_path):
        return

    print("[INFO] 正在执行严格的时序切分 (Chronological Split)，彻底隔离训练与测试数据...")
    signals = []
    
    # 记录每个子集的样本全局索引及其对应的真实标签
    split_data = {'train': [], 'val': [], 'test': []}
    # 记录每个子集内的 Normal (Fault-Free) 样本，防止 Reference 跨界泄露
    ref_indices = {'train': [], 'val': [], 'test': []}
    
    window_size = 24000
    stride = 12000 # 统一滑窗步长

    for file_name in os.listdir(raw_dir):
        if not file_name.endswith('.mat'): continue
        
        label = -1
        for key, val in CWRU_CLASSES.items():
            if key in file_name:
                label = val
                break
        if label == -1: continue 

        mat_data = sio.loadmat(os.path.join(raw_dir, file_name))
        de_key = [k for k in mat_data.keys() if 'DE_time' in k]
        if not de_key: continue
            
        raw_signal = mat_data[de_key[0]].flatten()
        
        # 核心：在滑窗之前，先进行物理切断！
        total_len = len(raw_signal)
        n_train = int(total_len * 0.7)
        n_val = int(total_len * 0.1)
        
        splits = {
            'train': raw_signal[:n_train],
            'val': raw_signal[n_train : n_train+n_val],
            'test': raw_signal[n_train+n_val :]
        }
        
        # 对切断后的三段独立数据分别进行滑窗提取
        for subset, sig_part in splits.items():
            for start in range(0, max(1, len(sig_part) - window_size + 1), stride):
                segment = sig_part[start : start+window_size]
                processed_segment = dcn(segment, length=window_size) 
                
                global_idx = len(signals)
                signals.append(processed_segment)
                split_data[subset].append((global_idx, label))
                
                if label == 0:
                    ref_indices[subset].append(global_idx)

    signals = np.array(signals, dtype=np.float32)
    np.save(data_cache_path, signals)

    # 构建并封装给 DataLoader 和 LLM 的格式
    dataset_info = {'train': [], 'val': [], 'test': []}
    corpus_data = []
    
    for subset in ['train', 'val', 'test']:
        for global_idx, label in split_data[subset]:
            
            # 从当前 subset 中随机抽取 Normal 样本作为参照，杜绝泄露
            if ref_indices[subset]:
                ref_idx = random.choice(ref_indices[subset])
            else:
                ref_idx = global_idx # Fallback 保护机制
                
            dataset_info[subset].append([global_idx, ref_idx, label])
            
            if subset == 'train':
                state_desc = DESCRIPTION_TEXT[label]
                corpus_data.append({
                    "id": len(corpus_data), # 保证微调 ID 严格连续
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
        
    print(f"[SUCCESS] 无泄露数据重构完毕！总计生成 {len(signals)} 个窗口特征。")
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