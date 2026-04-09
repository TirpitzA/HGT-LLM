"""
CWRU Physics-Based Fault Diagnosis Pipeline
Includes: CWRU .npz Parsing, Temporal Grouping (num_slices=3), and 7:2:1 Sequential Split.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
if MODELS_DIR not in sys.path: sys.path.insert(0, MODELS_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CWRU_DIR = os.path.join(DATA_DIR, "CWRU")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "bearllm_weights")

from xjtu_model import HierarchicalExplainableBearingNet
# CWRU 专用的图谱关系
from utils.cwru_physics_graph import get_bearing_physics_adjacency
# 10 分类标签映射
CWRU_LABEL_MAP = {
    'Normal': 0,
    'IR_007': 1, 'IR_014': 2, 'IR_021': 3,
    'Ball_007': 4, 'Ball_014': 5, 'Ball_021': 6,
    'OR_007': 7, 'OR_014': 8, 'OR_021': 9
}

class CWRUPreprocessor:
    def __init__(self, npz_path: str, num_slices: int = 3):
        self.npz_path = npz_path
        self.num_slices = num_slices
        self.slice_length = 1024 # 32 * 32

    def run(self):
        print(f"[PREPROCESS] 正在加载 CWRU 数据集: {self.npz_path}")
        data = np.load(self.npz_path, allow_pickle=True)
        X_raw = data['data']     # (4600, 32, 32)
        y_raw = data['labels']   # (4600,)

        # 展平为 (4600, 1024, 1) 单通道格式
        X_flat = X_raw.reshape(-1, self.slice_length, 1)
        y_mapped = np.array([CWRU_LABEL_MAP[label] for label in y_raw])

        # 初始化存储列表
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []

        # 按类别进行时序分组和 7:2:1 顺序切分，防止时间序列泄露
        for class_idx in range(10):
            # 取出当前类的所有样本
            class_mask = (y_mapped == class_idx)
            X_c = X_flat[class_mask]
            y_c = y_mapped[class_mask]

            # 丢弃余数，使其能被 num_slices (3) 整除
            valid_len = (len(X_c) // self.num_slices) * self.num_slices
            X_c = X_c[:valid_len]
            y_c = y_c[:valid_len]

            # 重塑为 (N_samples, 3, 1024, 1)
            X_grouped = X_c.reshape(-1, self.num_slices, self.slice_length, 1)
            y_grouped = y_c[::self.num_slices] # 每 3 个切片共用 1 个标签

            n_total = len(X_grouped)
            n_train = int(n_total * 0.7)
            n_val   = int(n_total * 0.2)

            # 顺序切分
            X_train_list.append(X_grouped[:n_train])
            y_train_list.append(y_grouped[:n_train])
            
            X_val_list.append(X_grouped[n_train:n_train+n_val])
            y_val_list.append(y_grouped[n_train:n_train+n_val])
            
            X_test_list.append(X_grouped[n_train+n_val:])
            y_test_list.append(y_grouped[n_train+n_val:])

        def merge(X_list, y_list):
            return np.concatenate(X_list, axis=0).astype(np.float32), np.concatenate(y_list, axis=0).astype(np.longlong)

        X_train, y_train = merge(X_train_list, y_train_list)
        X_val, y_val = merge(X_val_list, y_val_list)
        X_test, y_test = merge(X_test_list, y_test_list)

        os.makedirs(DATA_DIR, exist_ok=True)
        for name, arr in [("X_train", X_train), ("y_train", y_train), 
                          ("X_val", X_val), ("y_val", y_val), 
                          ("X_test", X_test), ("y_test", y_test)]:
            np.save(os.path.join(DATA_DIR, f"{name}.npy"), arr)

        print(f"[PREPROCESS] CWRU 预处理完成！时序切片格式 (N, {self.num_slices}, {self.slice_length}, 1)")
        print(f"  -> Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")


class CWRUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X, return_attention=False)
            loss = criterion(output['logits'], y)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(output['logits'], dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1_macro

def train_pipeline(epochs=40, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    
    train_loader = DataLoader(CWRUDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CWRUDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    _, edge_index, _ = get_bearing_physics_adjacency()
    
    # 【核心适配】: in_channels=1, num_classes=10
    model = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device), num_nodes=4, 
        in_channels=1, num_classes=10,
        slice_length=X_train.shape[2], num_slices=X_train.shape[1],
        cnn_hidden=32, gat_hidden=32, transformer_dim=64,
        transformer_heads=2, transformer_layers=2, dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_f1 = 0
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_model_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X)['logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f" ---> 发现更优模型并保存至 {best_model_path} (F1: {best_f1:.4f})")
            
    print(f"第一阶段训练全部完成！最高验证集 F1: {best_f1:.4f}")

def load_pretrained_physics_net(checkpoint_path, device):
    from utils.cwru_physics_graph import get_bearing_physics_adjacency
    _, edge_index, _ = get_bearing_physics_adjacency()
    model = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device), num_nodes=4, 
        in_channels=1, num_classes=10,
        slice_length=1024, num_slices=3,
        cnn_hidden=32, gat_hidden=32, transformer_dim=64,
        transformer_heads=2, transformer_layers=2, dropout=0.0
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, edge_index.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["preprocess", "train", "all"])
    parser.add_argument("--npz_path", type=str, default="data/CWRU/CWRU_48k_load_1_CNN_data.npz")
    args = parser.parse_args()
    
    if args.mode in ["preprocess", "all"]:
        preprocessor = CWRUPreprocessor(npz_path=args.npz_path)
        preprocessor.run()
    if args.mode in ["train", "all"]:
        train_pipeline()