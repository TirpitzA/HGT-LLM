"""
benchmark_master.py (v3 - Full Pipeline)
=======================================
HGT-LLM 严谨评估框架方案 (v2) 的核心驱动脚本。
支持全流程自动化：骨干网训练 -> SFT数据生成 -> 联合微调 -> 全量评测。

流程:
1. Backbone (HGT): 训练物理骨干网 (200 Epochs + Early Stopping)。
2. Generation: 生成多模态 SFT 指令数据集。
3. Joint SFT: 联合微调 Alignment Layer + Qwen LoRA。
4. Evaluation: 评估 HGT-LLM (准确率, PPL, 鲁棒性)。
5. Baselines: 评测基线模型 (WDCNN, TCNN, QCNN, BearingFM, MagNet)。

用法:
    python src/benchmark_master.py --config configs/cwru_config.yaml --data_dir data/CWRU_processed --full_pipeline
"""

import os
import sys
import json
import math
import yaml
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.hgt_net import HierarchicalExplainableBearingNet
from utils.physics_graph import get_bearing_physics_adjacency
from models.baselines.WDCNN import WDCNN
from models.baselines.TCNN import TCNN
from models.baselines.QCNN import QCNN
from models.baselines.BearingFM import BearingFM
from models.baselines.MagNet import MagNet

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TMP_DIR = "/tmp/benchmark_weights"
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "bearllm_weights")

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数与数据集
# ═══════════════════════════════════════════════════════════════════════════════

class HGTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class BaselineDataset(Dataset):
    def __init__(self, X, y, config):
        N, slices, length, channels = X.shape[0], config['model_params']['num_slices'], config['model_params']['slice_length'], config['model_params']['in_channels']
        X = X.transpose(0, 3, 1, 2).reshape(N, channels, slices * length)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def inject_awgn_noise(feature_tensor, snr_db):
    signal_power = torch.mean(feature_tensor ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(feature_tensor) * math.sqrt(noise_power)
    return feature_tensor + noise

def run_cmd(cmd):
    """运行子进程命令并实时打印输出"""
    print(f"\n[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + ":" + env.get("PYTHONPATH", "")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"[ERROR] 命令执行失败 (Code {process.returncode})")
        sys.exit(process.returncode)

# ═══════════════════════════════════════════════════════════════════════════════
# 阶段一：Backbone 优化训练
# ═══════════════════════════════════════════════════════════════════════════════

def train_hgt_backbone(config, train_loader, val_loader, device, epochs, lr, dataset_name):
    best_path = os.path.join(WEIGHTS_DIR, f"best_backbone_{dataset_name.lower()}.pth")
    if os.path.exists(best_path):
        print(f"\n✅ [PHASE 1] Backbone weight found at {best_path}, skipping training.")
        return best_path, 0
    print(f"\n🚀 [PHASE 1] 训练 HGT 物理骨干网 (Max Epochs: {epochs})")
    m = config['model_params']
    _, edge_index = get_bearing_physics_adjacency()
    model = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device),
        num_nodes=m['num_nodes'], in_channels=m['in_channels'],
        slice_length=m['slice_length'], num_slices=m['num_slices'],
        cnn_hidden=m['cnn_hidden'], gat_hidden=m['gat_hidden'],
        transformer_dim=m['transformer_dim'], transformer_heads=m['transformer_heads'],
        transformer_layers=m['transformer_layers'],
        num_classes=m['num_classes'], dropout=m['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_path = os.path.join(WEIGHTS_DIR, f"best_backbone_{dataset_name.lower()}.pth")
    
    best_f1 = 0.0
    patience_counter = 0
    max_patience = 10 # 早停阈值 (与参考代码对齐)
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"E{epoch+1:03d} Train", leave=False):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(X)['logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, non_blocking=True)
                logits = model(X)['logits']
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(y.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)[2]
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  E{epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= max_patience:
            print(f"  [EARLY STOP] 验证集 F1 在 {max_patience} 轮内未提升，停止训练。")
            break

    print(f"  ✅ Backbone 训练完成! Best Val F1: {best_f1:.4f} at Epoch {best_epoch}")
    # 更新配置中的 checkpoint_path 供后续阶段使用
    config['checkpoint_path'] = best_path
    return best_path, best_epoch

# ═══════════════════════════════════════════════════════════════════════════════
# 基线模型训练 (保持 v2 优化版)
# ═══════════════════════════════════════════════════════════════════════════════

def train_baseline(model_cls, model_name, config, train_loader, val_loader, device, epochs, lr, dataset_name):
    print(f"\n🚀 [BASELINE] 训练 {model_name}")
    m = config['model_params']
    model = model_cls(in_channels=m['in_channels'], num_classes=m['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    os.makedirs(TMP_DIR, exist_ok=True)
    best_path = os.path.join(TMP_DIR, f"best_{model_name}_{dataset_name.lower()}.pth")
    best_f1, best_epoch = 0.0, 0
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                outputs = model(X)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(y.numpy())
        
        val_f1 = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)[2]
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_epoch

# ═══════════════════════════════════════════════════════════════════════════════
# 综合评测流程
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(config_path, data_dir, epochs=200, batch_size=32, lr=1e-3, full_pipeline=False, smoke_test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    dataset_name = config.get('dataset_name', 'Unknown')

    # 0. 自动数据准备
    if not os.path.exists(os.path.join(data_dir, "X_train.npy")):
        print(f"[*] Data not found in {data_dir}, triggering preparation for {dataset_name}...")
        
        # 统一使用 prepare_data_rigorous.py 执行严格按时序的切分
        prep_cmd = [sys.executable, "src/prepare_data_rigorous.py", "--dataset", dataset_name.lower()]
        
        # 特殊情况兼容
        if dataset_name.lower() == 'ims':
            prep_cmd = [sys.executable, "src/prepare_ims_data.py"]
            
        print(f"[RUN] {' '.join(prep_cmd)}")
        run_cmd(prep_cmd)

    # 1. 加载数据
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    if smoke_test:
        X_train, y_train = X_train[:128], y_train[:128]
        X_val, y_val = X_val[:64], y_val[:64]
        X_test, y_test = X_test[:64], y_test[:64]
        epochs = 2
        print(f"  🔬 Smoke Test Mode")

    hgt_train_loader = DataLoader(HGTDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4)
    hgt_val_loader   = DataLoader(HGTDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4)
    hgt_test_loader  = DataLoader(HGTDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=4)

    results = {}

    # ─── 阶段一：Backbone ───
    backbone_path, hgt_conv = train_hgt_backbone(config, hgt_train_loader, hgt_val_loader, device, epochs, lr, dataset_name)
    
    # 获取 Backbone 在测试集上的性能 (直接分类)
    from network.hgt_net import HierarchicalExplainableBearingNet
    m = config['model_params']
    _, edge_index = get_bearing_physics_adjacency()
    model_bb = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device), num_nodes=m['num_nodes'], in_channels=m['in_channels'],
        slice_length=m['slice_length'], num_slices=m['num_slices'],
        cnn_hidden=m['cnn_hidden'], gat_hidden=m['gat_hidden'],
        transformer_dim=m['transformer_dim'], transformer_heads=m['transformer_heads'],
        transformer_layers=m['transformer_layers'], num_classes=m['num_classes']
    ).to(device)
    model_bb.load_state_dict(torch.load(backbone_path, map_location=device))
    
    from thop import profile
    dummy = torch.randn(1, m['num_slices'], m['slice_length'], m['in_channels']).to(device)
    flops, params_m = profile(model_bb, inputs=(dummy,), verbose=False)
    
    model_bb.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in hgt_test_loader:
            logits = model_bb(X.to(device))['logits']
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
    
    bb_acc = accuracy_score(all_labels, all_preds)
    bb_f1 = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[2]
    
    results['HGT_Backbone'] = {
        'Accuracy': bb_acc, 'F1': bb_f1, 'GFLOPS': flops/1e9, 
        'Params_M': params_m/1e6, 'Convergence_Epoch': hgt_conv
    }
    
    # 混淆矩阵
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{dataset_name.lower()}_backbone.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues')
    plt.savefig(cm_path)
    plt.close()

    # ─── 阶段二 & 三：Full HGT-LLM (如果启用) ───
    if full_pipeline:
        print(f"\n🚀 [PHASE 2] 生成多模态 SFT 数据集...")
        # 需要修改配置中的 checkpoint_path 和 data_dir 写入文件，供 generate_dataset.py 读取
        temp_config_path = f"/tmp/config_{dataset_name.lower()}.yaml"
        config['checkpoint_path'] = backbone_path
        config['data_dir'] = data_dir
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        run_cmd([sys.executable, "src/generate_dataset.py", "--config", temp_config_path])
        
        print(f"\n🚀 [PHASE 3] 联合微调 Alignment + Qwen LoRA...")
        # 设置更强的微调参数：8 Epochs, 1e-4 LR
        sft_env = os.environ.copy()
        # 我们可以在这里直接修改 train.py 的超参数，或者通过命令行传递（如果 train.py 支持）
        # 目前 train.py 是硬编码的，我们先直接修改 benchmark_master。
        run_cmd([sys.executable, "src/train.py", "--config", temp_config_path])
        
        print(f"\n🚀 [PHASE 4] 全量多模态评估 (HGT-LLM)...")
        checkpoint_dir = os.path.join(WEIGHTS_DIR, f"checkpoints_{dataset_name.lower()}", "best_checkpoint")
        run_cmd([sys.executable, "src/evaluate.py", "--config", temp_config_path, "--checkpoint", checkpoint_dir])
        
        # 解析 evaluate.py 产生的结果 JSON 并并入 results
        hgt_llm_result_path = os.path.join(RESULTS_DIR, f"benchmark_{config['dataset_name']}.json")
        if os.path.exists(hgt_llm_result_path):
            with open(hgt_llm_result_path, 'r') as f:
                hgt_llm_data = json.load(f)
                if 'HGT-LLM' in hgt_llm_data:
                    results['HGT-LLM'] = hgt_llm_data['HGT-LLM']
                else:
                    # 如果 evaluate.py 没按照 key 存，尝试直接取
                    results['HGT-LLM'] = hgt_llm_data
    
    # ─── 基线模型 ───
    bl_train_loader = DataLoader(BaselineDataset(X_train, y_train, config), batch_size=batch_size, shuffle=True)
    bl_val_loader   = DataLoader(BaselineDataset(X_val, y_val, config), batch_size=batch_size, shuffle=False)
    bl_test_loader  = DataLoader(BaselineDataset(X_test, y_test, config), batch_size=batch_size, shuffle=False)

    baselines = {'WDCNN': WDCNN, 'TCNN': TCNN, 'QCNN': QCNN, 'BearingFM': BearingFM, 'MagNet': MagNet}
    for name, cls in baselines.items():
        model, conv = train_baseline(cls, name, config, bl_train_loader, bl_val_loader, device, 100 if not smoke_test else 2, lr, dataset_name)
        
        model.eval()
        p, l = [], []
        with torch.no_grad():
            for X, y in bl_test_loader:
                out = model(X.to(device))
                logits = out[0] if isinstance(out, tuple) else out
                p.extend(torch.argmax(logits, dim=1).cpu().numpy())
                l.extend(y.numpy())
        
        acc = accuracy_score(l, p)
        f1 = precision_recall_fscore_support(l, p, average='macro', zero_division=0)[2]
        dummy_bl = torch.randn(1, m['in_channels'], m['num_slices']*m['slice_length']).to(device)
        f, pm = profile(cls(in_channels=m['in_channels'], num_classes=m['num_classes']).to(device), inputs=(dummy_bl,), verbose=False)
        
        results[name] = {'Accuracy': acc, 'F1': f1, 'GFLOPS': f/1e9, 'Params_M': pm/1e6, 'Convergence_Epoch': conv}
        print(f"  ✅ {name}: Acc={acc:.4f}, F1={f1:.4f}")

    # 保存结果
    with open(os.path.join(RESULTS_DIR, f"benchmark_{dataset_name.lower()}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 评测完成！结果已保存至 results/benchmark_{dataset_name.lower()}.json")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--full_pipeline", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        full_pipeline=args.full_pipeline,
        smoke_test=args.smoke_test
    )

if __name__ == "__main__":
    main()
