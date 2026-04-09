import os
import torch
import yaml
from network.hgt_net import HierarchicalExplainableBearingNet
from utils.physics_graph import get_bearing_physics_adjacency

def load_pretrained_physics_net(config_path, device):
    """
    通用物理骨干网络加载器。
    从提供的 config 文件自动获取模型参数并加载对应的预训练权重。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_params = config['model_params']
    checkpoint_path = config.get('checkpoint_path', '')
    if not os.path.isabs(checkpoint_path):
        # 假设当前执行路径的根目录或者根据 utils 相对于根目录的位置
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    if not os.path.exists(checkpoint_path):
         # 我们抛出明确异常而不是直接return none，方便上层捕捉
         raise FileNotFoundError(f"[ERROR] 找不到模型权重: {checkpoint_path}")

    # 获取物理邻接矩阵
    _, edge_index = get_bearing_physics_adjacency()
    edge_index = edge_index.to(device)

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
        dropout=0.0  # 推理时强制 dropout 0
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return model, edge_index, config
