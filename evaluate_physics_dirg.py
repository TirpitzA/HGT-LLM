import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.hgt_net import HierarchicalExplainableBearingNet
from utils.physics_graph import get_bearing_physics_adjacency

def evaluate_physics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = "configs/dirg_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    weights_path = "bearllm_weights/best_backbone_dirg.pth"
    data_dir = config['data_dir']
    
    print(f"[*] Loading data from {data_dir}...")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    _, edge_index = get_bearing_physics_adjacency()
    
    model = HierarchicalExplainableBearingNet(
        edge_index=edge_index.to(device),
        num_nodes=config['model_params']['num_nodes'],
        in_channels=config['model_params']['in_channels'],
        num_classes=config['model_params']['num_classes'],
        slice_length=config['model_params']['slice_length'],
        num_slices=config['model_params']['num_slices'],
        cnn_hidden=config['model_params']['cnn_hidden'],
        gat_hidden=config['model_params']['gat_hidden'],
        transformer_dim=config['model_params']['transformer_dim'],
        transformer_heads=config['model_params']['transformer_heads'],
        transformer_layers=config['model_params']['transformer_layers']
    ).to(device)
    
    print(f"[*] Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    X_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_torch = torch.tensor(y_test, dtype=torch.long).to(device)
    
    correct = 0
    total = len(y_test)
    
    per_class_correct = {i: 0 for i in range(7)}
    per_class_total = {i: 0 for i in range(7)}
    
    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 128
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_x = X_torch[i:end]
            batch_y = y_torch[i:end]
            
            output = model(batch_x)
            preds = torch.argmax(output['logits'], dim=1)
            
            correct += (preds == batch_y).sum().item()
            
            for j in range(len(batch_y)):
                ty = batch_y[j].item()
                py = preds[j].item()
                per_class_total[ty] += 1
                if ty == py:
                    per_class_correct[ty] += 1
                    
    print(f"\n[RESULTS] Physics Backbone Accuracy: {correct/total*100:.2f}%")
    for i in range(7):
        n = per_class_total[i]
        c = per_class_correct[i]
        acc = c/n*100 if n > 0 else 0
        print(f"Class {i}: {c}/{n} ({acc:.2f}%)")

if __name__ == "__main__":
    evaluate_physics()
