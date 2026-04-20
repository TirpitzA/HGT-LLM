import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from network.hgt_net import HierarchicalExplainableBearingNet
from models.multimodal_qwen import BearingMultimodalQwen
from src.zero_shot_adapter import ZeroShotChannelAdapter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Baselines
from models.baselines.WDCNN import WDCNN
from models.baselines.TCNN import TCNN
from models.baselines.QCNN import QCNN
from models.baselines.BearingFM import BearingFM
from models.baselines.MagNet import MagNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class IMSZeroShotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def run_zero_shot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # configs
    with open('configs/dirg_config.yaml', 'r') as f:
        src_config = yaml.safe_load(f)
    with open('configs/ims_config.yaml', 'r') as f:
        tgt_config = yaml.safe_load(f)
        
    # Load IMS Test Data
    data_dir = 'data/IMS_processed'
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # 标注映射 (DIRG 7类, IMS 4类)
    # DIRG: 0:H, 1,2,3:I, 4,5,6:O/B
    # IMS:  0:H, 1:I, 2:O, 3:B
    # 为了简化测试，我们将 IMS 的 1,2,3 映射到 DIRG 的相应大类进行比对
    # 或者直接看物理网络对 IMS 的分类表现
    
    loader = DataLoader(IMSZeroShotDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    results = {}
    
    # 1. HGT-LLM (Physics only for zero-shot accuracy check)
    print("\n[*] Evaluating DIRG-trained HGT Backbone on IMS...")
    m = src_config['model_params']
    from utils.physics_graph import get_bearing_physics_adjacency
    _, edge_index = get_bearing_physics_adjacency()
    hgt = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device), num_nodes=4, in_channels=6,
        slice_length=1024, num_slices=3, cnn_hidden=32, gat_hidden=32,
        transformer_dim=64, transformer_heads=2, transformer_layers=2, num_classes=7
    ).to(device)
    hgt.load_state_dict(torch.load('bearllm_weights/best_backbone_dirg.pth', map_location=device))
    hgt.eval()
    
    adapter = ZeroShotChannelAdapter(1, 6).to(device)
    
    hgt_preds = []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="HGT-ZeroShot"):
            X = adapter(X.to(device)) # [B, S, L, 1] -> [B, S, L, 6]
            logits = hgt(X)['logits']
            pred = logits.argmax(dim=-1).cpu().numpy()
            # Map DIRG labels back to IMS major classes for ACC calc
            # DIRG 1,2,3 -> Inner(1), 4,5,6 -> Outer/Ball(2/3)
            mapped_pred = []
            for p in pred:
                if p == 0: mapped_pred.append(0)
                elif p in [1, 2, 3]: mapped_pred.append(1)
                else: mapped_pred.append(2) # 简化：由于 DIRG 将 Ball 和 Outer 合并在 4,5,6，这里统一记为 2
            hgt_preds.extend(mapped_pred)
            
    # Adjust y_test for Comparison (IMS 3(Ball) -> 2)
    y_test_adj = [y if y < 3 else 2 for y in y_test]
    acc = accuracy_score(y_test_adj, hgt_preds)
    print(f"HGT Zero-shot Accuracy (DIRG -> IMS): {acc:.4f}")
    results['HGT_LLM_ZeroShot'] = acc
    
    # 2. Baselines
    baselines = {
        'WDCNN': WDCNN, 'TCNN': TCNN, 'QCNN': QCNN, 'BearingFM': BearingFM, 'MagNet': MagNet
    }
    
    # Mapping for Baselines (They were trained on DIRG 6-channel)
    for name, cls in baselines.items():
        print(f"\n[*] Evaluating DIRG-trained {name} on IMS...")
        try:
            model = cls(in_channels=6, num_classes=7).to(device)
            path = f'/tmp/benchmark_weights/best_{name}_dirg.pth'
            if not os.path.exists(path):
                print(f"Weights not found: {path}")
                continue
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            
            p_list = []
            with torch.no_grad():
                for X, y in tqdm(loader, desc=f"{name}-ZS"):
                    # Baseline input needs [B, C, L]
                    # IMS X is [B, S, L, 1] -> reshape to [B, 1, S*L]
                    X_res = X.permute(0, 3, 1, 2).reshape(X.shape[0], 1, -1).to(device)
                    # Pad to 6 channels
                    X_pad = torch.cat([X_res, torch.zeros(X_res.shape[0], 5, X_res.shape[2], device=device)], dim=1)
                    out = model(X_pad)
                    logits = out[0] if isinstance(out, tuple) else out
                    pred = logits.argmax(dim=-1).cpu().numpy()
                    mapped_p = [p if p == 0 else (1 if p in [1,2,3] else 2) for p in pred]
                    p_list.extend(mapped_p)
            
            acc_bl = accuracy_score(y_test_adj, p_list)
            print(f"{name} Zero-shot Accuracy: {acc_bl:.4f}")
            results[name] = acc_bl
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # Output
    out_dir = 'results/zero_shot_ims'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'zero_shot_results.yaml'), 'w') as f:
        yaml.dump(results, f)
    print(f"\n✅ Zero-shot results saved to {out_dir}/")

if __name__ == "__main__":
    run_zero_shot()
