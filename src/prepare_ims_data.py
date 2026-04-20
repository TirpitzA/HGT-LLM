import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置
DATA_ROOT = "data/IMS/IMS_Bearings_Data/raw"
OUTPUT_DIR = "data/IMS_processed"
SLICE_LEN = 1024
NUM_SLICES = 3
TOTAL_LEN = SLICE_LEN * NUM_SLICES
CHANNELS = 1 # 我们只取一个主要通道

# IMS 标注策略
# Label 0: Healthy (Test 2 first 500 files, Ch 1)
# Label 1: Inner (Test 1 last 100 files, Ch 5)
# Label 2: Outer (Test 2 last 100 files, Ch 1)
# Label 3: Ball (Test 1 last 100 files, Ch 7)

def get_file_list(test_dir):
    files = sorted(os.listdir(test_dir))
    return [os.path.join(test_dir, f) for f in files]

def process_ims():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tasks = [
        {"name": "Healthy", "dir": "2nd/2nd_test", "files_slice": slice(0, 500), "channel": 0, "label": 0},
        {"name": "Inner",   "dir": "1st/1st_test", "files_slice": slice(-100, None), "channel": 4, "label": 1},
        {"name": "Outer",   "dir": "2nd/2nd_test", "files_slice": slice(-100, None), "channel": 0, "label": 2},
        {"name": "Ball",    "dir": "1st/1st_test", "files_slice": slice(-100, None), "channel": 6, "label": 3},
    ]
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    for task in tasks:
        test_path = os.path.join(DATA_ROOT, task["dir"])
        if not os.path.exists(test_path):
            print(f"[WARN] Path not found: {test_path}")
            continue
            
        all_files = get_file_list(test_path)
        target_files = all_files[task["files_slice"]]
        
        print(f"[*] Processing {task['name']} from {len(target_files)} files...")
        
        X_task = []
        for fpath in tqdm(target_files, desc=task["name"]):
            try:
                # IMS 文件是制表符或空格分隔的 ASCII
                df = pd.read_csv(fpath, sep='\t', header=None)
                if df.empty or len(df.columns) <= task["channel"]:
                    # 尝试空格分隔
                    df = pd.read_csv(fpath, sep='\s+', header=None)
                
                signal = df.iloc[:, task["channel"]].values
                
                # 切片处理
                valid_len = (len(signal) // TOTAL_LEN) * TOTAL_LEN
                if valid_len == 0: continue
                
                signal = signal[:valid_len]
                X_f = signal.reshape(-1, NUM_SLICES, SLICE_LEN, 1).astype(np.float32)
                X_task.append(X_f)
            except Exception as e:
                print(f"[ERROR] Failed to process {fpath}: {e}")
        
        if not X_task: continue
        X_task = np.concatenate(X_task, axis=0)
        y_task = np.full(len(X_task), task["label"], dtype=np.int64)
        
        # 7:2:1 划分
        n = len(X_task)
        n_train = int(n * 0.7)
        n_val   = int(n * 0.2)
        
        X_train_list.append(X_task[:n_train])
        y_train_list.append(y_task[:n_train])
        X_val_list.append(X_task[n_train:n_train+n_val])
        y_val_list.append(y_task[n_train:n_train+n_val])
        X_test_list.append(X_task[n_train+n_val:])
        y_test_list.append(y_task[n_train+n_val:])
        
    # 合并并保存
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), np.concatenate(X_train_list, axis=0))
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), np.concatenate(y_train_list, axis=0))
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   np.concatenate(X_val_list, axis=0))
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   np.concatenate(y_val_list, axis=0))
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  np.concatenate(X_test_list, axis=0))
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  np.concatenate(y_test_list, axis=0))
    
    print(f"\n[OK] IMS 数据预处理完毕，保存在 {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_ims()
