"""
run_ablation.py
===============
消融实验自动化脚本。
对比: HGT-LLM (GAT+Transformer) vs 消融版 (无GAT, 仅Transformer)
在指定数据集上训练两个版本并评估，输出对比结果。

用法:
    python ablation_study/run_ablation.py --config configs/cwru_config.yaml --data_dir data/CWRU_processed
    python ablation_study/run_ablation.py --config configs/xjtu_config.yaml --data_dir data/XJTU_processed
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入原始模型和消融模型
from network.hgt_net import HierarchicalExplainableBearingNet
from ablation_study.model_no_gat import HierarchicalBearingNet_NoGAT
from utils.physics_graph import get_bearing_physics_adjacency


class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X, return_attention=False)
            logits = output['logits']
            loss = criterion(logits, y)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1_macro


def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       epochs=40, lr=1e-3, model_name="Model"):
    """统一训练+评估流程，返回最终指标和收敛速度"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_f1 = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1:02d}/{epochs}", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X)['logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_f1)

        print(f"  [{model_name}] Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 加载最优权重并在测试集上评估
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, device, criterion)

    return {
        'test_acc': test_acc,
        'test_f1': test_f1,
        'best_val_f1': best_f1,
        'convergence_epoch': best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="消融实验: GAT+Transformer vs Transformer-only")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--data_dir", type=str, required=True, help="预处理数据目录 (含 X_train.npy 等)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    m = config['model_params']
    dataset_name = config.get('dataset_name', 'Unknown')

    # 加载数据
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    train_loader = DataLoader(BearingDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(BearingDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(BearingDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  消融实验: {dataset_name} | 设备: {device}")
    print(f"  超参: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}")

    # --- 模型 A: 完整 HGT (GAT + Transformer) ---
    _, edge_index = get_bearing_physics_adjacency()
    model_full = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device),
        num_nodes=m['num_nodes'], in_channels=m['in_channels'],
        slice_length=m['slice_length'], num_slices=m['num_slices'],
        cnn_hidden=m['cnn_hidden'], gat_hidden=m['gat_hidden'],
        transformer_dim=m['transformer_dim'], transformer_heads=m['transformer_heads'],
        transformer_layers=m['transformer_layers'],
        num_classes=m['num_classes'], dropout=m['dropout']
    ).to(device)

    print("\n[A] 完整模型 (CNN + GAT + Transformer)")
    results_full = train_and_evaluate(
        model_full, train_loader, val_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr, model_name="Full-HGT"
    )

    # --- 模型 B: 消融版 (无 GAT, 仅 Transformer) ---
    model_no_gat = HierarchicalBearingNet_NoGAT(
        num_nodes=m['num_nodes'], in_channels=m['in_channels'],
        slice_length=m['slice_length'], num_slices=m['num_slices'],
        cnn_hidden=m['cnn_hidden'],
        transformer_dim=m['transformer_dim'], transformer_heads=m['transformer_heads'],
        transformer_layers=m['transformer_layers'],
        num_classes=m['num_classes'], dropout=m['dropout']
    ).to(device)

    print("\n[B] 消融模型 (CNN + Transformer, 无 GAT)")
    results_no_gat = train_and_evaluate(
        model_no_gat, train_loader, val_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr, model_name="No-GAT"
    )

    # --- 汇总对比 ---
    print(f"\n{'='*60}")
    print(f"  消融实验结果 ({dataset_name})")
    print(f"{'='*60}")
    print(f"  {'指标':<25} {'Full-HGT':>12} {'No-GAT':>12} {'GAT 贡献':>12}")
    print(f"  {'-'*61}")

    for metric_name, key in [("Test Accuracy (%)", 'test_acc'), ("Test Macro F1 (%)", 'test_f1')]:
        v_full = results_full[key] * 100
        v_ablation = results_no_gat[key] * 100
        delta = v_full - v_ablation
        print(f"  {metric_name:<25} {v_full:>11.2f}% {v_ablation:>11.2f}% {delta:>+11.2f}%")

    print(f"  {'收敛速度 (Epoch)':<25} {results_full['convergence_epoch']:>12d} {results_no_gat['convergence_epoch']:>12d}")
    print(f"{'='*60}")

    # 保存结果
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"ablation_{dataset_name}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Full-HGT: Acc={results_full['test_acc']:.4f}, F1={results_full['test_f1']:.4f}, Converge@{results_full['convergence_epoch']}\n")
        f.write(f"No-GAT:   Acc={results_no_gat['test_acc']:.4f}, F1={results_no_gat['test_f1']:.4f}, Converge@{results_no_gat['convergence_epoch']}\n")
    print(f"\n  结果已保存至 {result_file}")


if __name__ == "__main__":
    main()
