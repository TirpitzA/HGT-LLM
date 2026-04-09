import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# 导入你原始代码中的必要组件
# 假设你的原始代码文件名为 cwru_physics_pipeline.py，请根据实际文件名修改下方的导入
try:
    from cwru_physics_pipeline import CWRUDataset, load_pretrained_physics_net, CWRU_LABEL_MAP, DATA_DIR, WEIGHTS_DIR
except ImportError:
    print("请确保将 'cwru_physics_pipeline' 替换为你原始代码的实际 Python 文件名（不含 .py）")
    sys.exit(1)

def test_model():
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用计算设备: {device}")

    # 2. 检查并加载测试数据
    x_test_path = os.path.join(DATA_DIR, "X_test.npy")
    y_test_path = os.path.join(DATA_DIR, "y_test.npy")
    
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print("[-] 找不到测试数据，请确保已运行预处理 pipeline (--mode preprocess)。")
        return

    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    print(f"[*] 成功加载测试集: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    # 3. 构建 DataLoader
    test_loader = DataLoader(CWRUDataset(X_test, y_test), batch_size=32, shuffle=False)

    # 4. 加载训练好的模型
    model_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"[-] 找不到模型权重文件: {model_path}，请确保已经完成训练。")
        return
    
    print(f"[*] 正在加载预训练模型: {model_path}")
    model, _ = load_pretrained_physics_net(model_path, device)
    model.eval()

    # 5. 执行推理与评估
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []

    print("[*] 开始在测试集上进行评估...")
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X, return_attention=False) # 如果你的 forward 需要这个参数
            
            # 兼容模型输出字典的情况
            logits = output['logits'] if isinstance(output, dict) else output
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 6. 计算评估指标
    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "="*50)
    print(" " * 15 + "TEST RESULTS")
    print("="*50)
    print(f"Test Loss:  {avg_loss:.4f}")
    print(f"Accuracy:   {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro F1:   {f1_macro:.4f} ({f1_macro * 100:.2f}%)")
    print("="*50)

    # 7. 打印详细的分类报告
    # 反转标签字典以便在报告中显示具体故障类型名称
    target_names = [name for name, idx in sorted(CWRU_LABEL_MAP.items(), key=lambda item: item[1])]
    
    print("\n[Detailed Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

if __name__ == "__main__":
    test_model()