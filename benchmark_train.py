import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from models.model_factory import create_model

# 复用你原来的 Dataset 和 evaluate 函数
from models.physics_pipeline import XJTUDataset

def main():
    parser = argparse.ArgumentParser(description="One-Click Benchmark Training")
    parser.add_argument("--model", type=str, required=True, choices=["Ours", "WDCNN", "TCNN", "QCNN"])
    parser.add_argument("--dataset", type=str, default="XJTU-SY", choices=["XJTU-SY", "CWRU"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"========== Running {args.model} on {args.dataset} ==========")

    # 1. 加载数据 (此处以你原来的 npy 路径为例)
    # 如果是其他数据集，可以通过 if args.dataset == "CWRU": 切换路径
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_val = np.load("data/X_val.npy")
    y_val = np.load("data/y_val.npy")

    train_loader = DataLoader(XJTUDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(XJTUDataset(X_val, y_val), batch_size=32, shuffle=False)

    # 2. 调用工厂初始化模型
    model = create_model(args.model, args.dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3. 统一训练循环
    for epoch in range(args.epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # 【核心差异处理】：适配维度
            # X 原始 shape: [Batch, num_slices(3), slice_length(4096), channels(2)]
            if args.model == "Ours":
                # 你的模型直接接受原始 shape
                output = model(X, return_attention=False)['logits']
            else:
                # Baseline 模型需要 [Batch, Channels, Total_Length]
                batch_size, slices, slice_len, channels = X.shape
                # 展平时序切片 -> [Batch, Channels, slices * slice_len]
                X_flat = X.view(batch_size, slices * slice_len, channels).permute(0, 2, 1)
                output = model(X_flat)
                
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{args.epochs} completed.")
        
    # 保存该模型的专属权重
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/{args.model}_{args.dataset}_best.pth")
    print(f"Saved {args.model} weights.")

if __name__ == "__main__":
    main()