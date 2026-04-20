"""
prepare_dirg_data.py
====================
专门用于处理 DIRG (Politecnico di Torino) 数据集的脚本。
1. 从 data/DIRG/VariableSpeedAndLoad/*.mat 中提取 6 通道信号。
2. 进行切片处理 (3 slices * 1024 length = 3072 total)。
3. 为每个文件分配标签 (C0-C6)。
4. **按文件顺序 7:2:1 切分** (时序物理隔离，与参考代码对齐)。
5. 保存为 npy 文件到 data/DIRG_processed/。
"""
import os
import glob
import scipy.io
import numpy as np
from tqdm import tqdm

# 配置
DATA_ROOT = "data/DIRG/VariableSpeedAndLoad"
OUTPUT_DIR = "data/DIRG_processed"
SLICE_LEN = 1024
NUM_SLICES = 3
TOTAL_LEN = SLICE_LEN * NUM_SLICES
CHANNELS = 6

def process_dirg():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 只匹配以 C 开头的 .mat 文件 (C0 ~ C6)
    mat_files = sorted(glob.glob(os.path.join(DATA_ROOT, "C*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"未找到 C*.mat 文件，请检查路径：{DATA_ROOT}")
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    print(f"[*] 发现 {len(mat_files)} 个 MAT 文件，开始处理...")
    
    # 按文件（每个工况）单独切分，保证同一工况下的时序物理隔离
    for fpath in tqdm(mat_files, desc="处理 .mat 文件"):
        fname = os.path.basename(fpath)
        
        # 获取标签: C0A_... -> 0, C1A_... -> 1, ..., C6A_... -> 6
        label = int(fname.split('C')[1][0])
        
        # 加载数据
        data = scipy.io.loadmat(fpath)
        # 自动寻找真实的信号数据数组（忽略 __header__ 等元数据）
        data_key = [k for k in data.keys() if not k.startswith('__')][0]
        signal = data[data_key]  # (Points, 6)
        
        # 丢弃末尾无法整除的数据，进行切片
        valid_len = (len(signal) // TOTAL_LEN) * TOTAL_LEN
        if valid_len == 0:
            continue
            
        signal = signal[:valid_len]
        # 变形为 (样本数, 3个切片, 1024个点, 6个通道)
        X_c = signal.reshape(-1, NUM_SLICES, SLICE_LEN, CHANNELS).astype(np.float32)
        y_c = np.full(len(X_c), label, dtype=np.int64)
        
        # 7:2:1 顺序切分 (时序物理隔离)
        n_total = len(X_c)
        n_train = int(n_total * 0.7)
        n_val   = int(n_total * 0.2)
        
        X_train_list.append(X_c[:n_train])
        y_train_list.append(y_c[:n_train])
        X_val_list.append(X_c[n_train:n_train+n_val])
        y_val_list.append(y_c[n_train:n_train+n_val])
        X_test_list.append(X_c[n_train+n_val:])
        y_test_list.append(y_c[n_train+n_val:])
    
    # 合并所有文件的数据
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    print(f"\n[*] 数据划分完成 (按文件顺序时序隔离):")
    print(f"    Train: {X_train.shape}")
    print(f"    Val:   {X_val.shape}")
    print(f"    Test:  {X_test.shape}")
    
    # 保存
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    
    print(f"\n[OK] DIRG 数据预处理完毕 (时序物理隔离)，保存在 {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_dirg()
