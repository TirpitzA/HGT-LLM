import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.baselines.WDCNN import WDCNN
from models.baselines.TCNN import TCNN
from models.baselines.QCNN import QCNN
from models.baselines.BearingFM import BearingFM
from models.baselines.MagNet import MagNet

class BaselineCWRUDataset(Dataset):
    def __init__(self, split='train'):
        # 复用 generate_dataset.py 划分的 7:2:1 阵列
        X_raw = np.load(os.path.join(PROJECT_ROOT, f"data/X_{split}.npy")) 
        y_raw = np.load(os.path.join(PROJECT_ROOT, f"data/y_{split}.npy"))
        
        # 【核心修复】：处理异常维度，确保输入基准模型的形状严格为 (N, 1, L)
        # 1. 挤压掉大小为 1 的多余维度 (例如将 (N, 3, 1024, 1) 变为 (N, 3, 1024))
        X_raw = np.squeeze(X_raw) 
        
        # 2. 规范化为 (N, 1, L)
        if X_raw.ndim == 2:
            # 如果是 (N, L) -> 扩展为 (N, 1, L)
            self.X = np.expand_dims(X_raw, axis=1)
        elif X_raw.ndim == 3:
            # 如果是 (N, C, L) 或 (N, L, C)
            if X_raw.shape[1] > X_raw.shape[2]:
                # 若 L 在中间位置 (N, L, C)，进行转置变为 (N, C, L)
                X_raw = X_raw.transpose(0, 2, 1)
            # 强制取第一个通道，以适配基准模型 in_channels=1
            self.X = X_raw[:, 0:1, :]
        else:
            raise ValueError(f"数据加载失败，无法处理的维度形状: {X_raw.shape}")
            
        self.y = y_raw.astype(np.int64)

    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]]).squeeze()

def train_and_eval(model_class, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    
    train_dataset = BaselineCWRUDataset('train')
    val_dataset = BaselineCWRUDataset('val')
    test_dataset = BaselineCWRUDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 【核心适配】：CWRU 为 10 分类任务
    num_classes = 10
    counts = np.bincount(train_dataset.y, minlength=num_classes)
    weights = len(train_dataset.y) / (float(num_classes) * (counts + 1e-6))
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = 0.0
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    best_model_path = os.path.join(PROJECT_ROOT, f"data/best_{name}_cwru.pth")

    print(f"\n{'='*50}")
    print(f"🚀 开始训练基准模型: {name} (CWRU 10分类)")
    print(f"{'='*50}")
    
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

    # 最终测试
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            pred = torch.argmax(logits, dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    print(f"🎯 {name} 测试集最终结果 -> 准确率 (Acc): {test_acc*100:.2f}% | F1 分数: {test_f1:.4f}")
    return test_acc

if __name__ == "__main__":
    models = {
        "WDCNN": WDCNN, 
        "TCNN": TCNN, 
        "QCNN": QCNN, 
        "BearingFM": BearingFM, 
        "MagNet": MagNet
    }
    for name, m_class in models.items():
        train_and_eval(m_class, name)