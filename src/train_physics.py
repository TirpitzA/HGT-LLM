"""
train_physics.py
================
统一的 物理骨干网(Physics-Based Hierarchical Fault Diagnosis Model) 训练脚本。
支持通过 YAML 动态实例化网络结构并训练，产出最佳模型权重用于后续多模态推理。
"""
import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# ==========================================
# 初始化路径与模型导入
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.hgt_net import HierarchicalExplainableBearingNet
from utils.physics_graph import get_bearing_physics_adjacency

class BearingPhysicsDataset(Dataset):
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
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(X, return_attention=False)
            loss = criterion(output['logits'], y)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(output['logits'], dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1_macro

def train_physics(config_path: str, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3):
    # 1. 解析配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_name = config.get("dataset_name", "Unknown")
    data_dir = config.get('data_dir', 'data/')
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
        
    m_params = config['model_params']
    checkpoint_path = config.get('checkpoint_path', '')
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 [TRAIN PHYSICS] 准备训练 {dataset_name} | 设备: {device}")
    
    # 2. 加载纯净数据 (此脚本仅负责训练，假定 7:2:1 的 .npy 已预处理好存在 data_dir 下)
    try:
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    except FileNotFoundError as e:
        print(f"[ERROR] 找不到预处理数据。请确保执行了有效的数据生成/预处理。 {e}")
        sys.exit(1)

    print(f"  -> Train: {X_train.shape} | Val: {X_val.shape}")

    train_loader = DataLoader(BearingPhysicsDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(BearingPhysicsDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. 初始化通用物理网络
    _, edge_index = get_bearing_physics_adjacency()
    model = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device),
        num_nodes=m_params['num_nodes'],
        in_channels=m_params['in_channels'],
        slice_length=m_params['slice_length'],
        num_slices=m_params['num_slices'],
        cnn_hidden=m_params['cnn_hidden'],
        gat_hidden=m_params['gat_hidden'],
        transformer_dim=m_params['transformer_dim'],
        transformer_heads=m_params['transformer_heads'],
        transformer_layers=m_params['transformer_layers'],
        num_classes=m_params['num_classes'],
        dropout=m_params['dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_f1 = 0.0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}", leave=False)
        for X, y in pbar:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            loss = criterion(model(X)['logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:02d}/{epochs} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            
    print(f"\n✅ 物理骨干网络预训练完成！最高验证集 F1: {best_f1:.4f}")
    print(f"模型权重已保存至: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="物理骨干网统一训练脚本")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    args = parser.parse_args()
    
    train_physics(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
