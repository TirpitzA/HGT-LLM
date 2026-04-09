import os
import sys
import math
import torch
import yaml
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from thop import profile
from tqdm import tqdm

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.baselines.WDCNN import WDCNN
from models.baselines.TCNN import TCNN
from models.baselines.QCNN import QCNN
from models.baselines.BearingFM import BearingFM
from models.baselines.MagNet import MagNet

def inject_awgn_noise(feature_tensor: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    在特征张量中注入受控信噪比 (SNR) 的高斯白噪声。
    确保与 HGT-LLM 的噪声注入逻辑 100% 物理对齐。
    """
    signal_power = torch.mean(feature_tensor ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(feature_tensor) * math.sqrt(noise_power)
    return feature_tensor + noise

class UniversalBaselineDataset(Dataset):
    def __init__(self, config, split='train'):
        data_dir = config.get('data_dir', 'data/')
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(PROJECT_ROOT, data_dir)
            
        x_path = os.path.join(data_dir, f"X_{split}.npy")
        y_path = os.path.join(data_dir, f"y_{split}.npy")
        
        X_raw = np.load(x_path)
        y_raw = np.load(y_path)
        
        # 维度自适应适配
        # 预想形状: (N, Slices, Length, Channels) -> (N, Channels, Slices * Length)
        N = X_raw.shape[0]
        slices = config['model_params']['num_slices']
        length = config['model_params']['slice_length']
        channels = config['model_params']['in_channels']
        
        if X_raw.ndim == 4:
            # 严格按照用户要求的 Transpose 陷阱修复逻辑
            # 第一步：把 Channels 换到第二维 -> (N, Channels, Slices, Length)
            X_raw = X_raw.transpose(0, 3, 1, 2)
            # 第二步：融合后两个维度 -> (N, Channels, Slices * Length)
            self.X = X_raw.reshape(N, channels, slices * length)
        elif X_raw.ndim == 3:
            # 处理 (N, Slices * Length, Channels) 情况
            if X_raw.shape[2] == channels:
                self.X = X_raw.transpose(0, 2, 1)
            else:
                self.X = X_raw # 假设已经是 (N, C, L)
        elif X_raw.ndim == 2:
            # 处理 (N, Length) 情况
            self.X = np.expand_dims(X_raw, axis=1)
        else:
            self.X = X_raw
            
        self.y = y_raw.astype(np.int64)

    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]]).squeeze()

def train_and_eval(model_class, name, config, device, snr=None):
    in_channels = config['model_params']['in_channels']
    num_classes = config['model_params']['num_classes']
    total_length = config['model_params']['num_slices'] * config['model_params']['slice_length']
    
    # 实例化模型（支持默认参数向后兼容）
    model = model_class(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # 算力测试 (thop) - 确保设备对齐
    dummy_input = torch.randn(1, in_channels, total_length).to(device)
    # MagNet 返回 tuple，thop 默认处理单个输出，需要特殊处理
    if name == "MagNet":
        def model_forward_wrapper(x):
            y, d = model(x)
            return y
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    else:
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    g_flops = flops / 1e9
    m_params = params / 1e6
    
    train_dataset = UniversalBaselineDataset(config, 'train')
    val_dataset = UniversalBaselineDataset(config, 'val')
    test_dataset = UniversalBaselineDataset(config, 'test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 统一 AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_f1 = 0.0
    best_model_path = f"/tmp/best_{name}_{config.get('dataset_name', 'dataset')}.pth"

    print(f"\n🚀 开始训练 {name} | 数据集: {config.get('dataset_name', 'Unknown')} | GFLOPs: {g_flops:.4f} | Params: {m_params:.2f}M")
    
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                pred = torch.argmax(logits, dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            
        scheduler.step(val_f1)
    
    # ================= 最终测试 (支持 SNR 鲁棒性注入) =================
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_labels = [], []
    
    snr_msg = f" (SNR={snr}dB)" if snr is not None else " (Clean)"
    print(f"   [Test] 开始评估集测试{snr_msg}...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # 动态噪声注入
            if snr is not None:
                images = inject_awgn_noise(images, snr)
                
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            pred = torch.argmax(logits, dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    print(f"🎯 {name} 最终结果 -> Acc: {test_acc*100:.2f}% | F1: {test_f1:.4f} | GFLOPs: {g_flops:.4f} | Params: {m_params:.2f}M")
    return {"acc": test_acc, "f1": test_f1, "flops": g_flops, "params": m_params}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="跨数据集统一基线评估管线")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--model", type=str, default="all", help="基线模型名称 (WDCNN, TCNN, QCNN, BearingFM, MagNet) 或 all")
    # 新增 SNR 参数
    parser.add_argument("--snr", type=float, default=None, help="注入的 AWGN 噪声强度(dB)。例如: 5, 0, -5。不传则为纯净测试。")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    snr_display = f"SNR={args.snr}dB" if args.snr is not None else "Clean"
    print(f"[INFO] 使用设备: {device} | 数据集: {config.get('dataset_name', 'Unknown')} | 测试环境: {snr_display}")

    all_models = {
        "WDCNN": WDCNN, 
        "TCNN": TCNN, 
        "QCNN": QCNN, 
        "BearingFM": BearingFM, 
        "MagNet": MagNet
    }

    results = {}
    if args.model == "all":
        target_models = all_models
    else:
        if args.model in all_models:
            target_models = {args.model: all_models[args.model]}
        else:
            print(f"[ERROR] 未知模型: {args.model}")
            sys.exit(1)

    for name, m_class in target_models.items():
        results[name] = train_and_eval(m_class, name, config, device, snr=args.snr)

    print("\n" + "="*80)
    print(f"{'Model':<15} | {'Acc (%)':<10} | {'F1-Macro':<10} | {'GFLOPs':<10} | {'Params (M)':<10}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<15} | {res['acc']*100:<10.2f} | {res['f1']:<10.4f} | {res['flops']:<10.4f} | {res['params']:<10.2f}")
    print("=" * 80)