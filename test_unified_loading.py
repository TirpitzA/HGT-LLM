import yaml
import torch
import numpy as np

from network.hgt_net import HierarchicalExplainableBearingNet
from utils.physics_graph import get_bearing_physics_adjacency
from utils.dig_construction import BearingDynamicInstanceGraph

def test_loading(config_path):
    print(f"--- Testing {config_path} ---")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config for {config['dataset_name']}")

    A, edge_index = get_bearing_physics_adjacency()
    print("Physics Graph Adjacency created.")

    model_params = config['model_params']
    
    # Initialize network
    model = HierarchicalExplainableBearingNet(
        edge_index=edge_index,
        num_nodes=model_params['num_nodes'],
        in_channels=model_params['in_channels'],
        slice_length=model_params['slice_length'],
        num_slices=model_params['num_slices'],
        cnn_hidden=model_params['cnn_hidden'],
        gat_hidden=model_params['gat_hidden'],
        transformer_dim=model_params['transformer_dim'],
        transformer_heads=model_params['transformer_heads'],
        transformer_layers=model_params['transformer_layers'],
        num_classes=model_params['num_classes'],
        dropout=model_params['dropout']
    )
    print("Network instantiated successfully.")

    # Initialize DIG Construction
    dig = BearingDynamicInstanceGraph(A, config)
    
    # Simple simulated input for DIG logic
    pred = 1 if 1 in config['labels'] else 0
    res = dig.process_sample(
        prediction=pred, 
        confidence=98.5,
        edge_attention=np.zeros((3, 10)),
        slice_attention=np.array([0.2, 0.7, 0.1]),
        lang='zh'
    )
    print("DIG Generation Test successful.")
    # To reduce output log size, we don't print the full explanation here unless debugging
    # print(res['explanation'])
    print("-" * 50)

if __name__ == "__main__":
    try:
        test_loading("configs/cwru_config.yaml")
        test_loading("configs/xjtu_config.yaml")
        test_loading("configs/dirg_config.yaml")
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
