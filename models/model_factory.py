import torch
import torch.nn as nn
#这个文件的作用是集中管理所有模型。通过传入一个字符串名字，它就能自动返回初始化好的模型，便于进行实验。
# 导入你的模型
from models.xjtu_model import HierarchicalExplainableBearingNet
from utils.xjtu_physics_graph import get_bearing_physics_adjacency

# 导入 BearLLM 提供的 Baseline 模型 (假设你把它们放在了 models/baselines/ 下)
from models.baselines.WDCNN import WDCNN
from models.baselines.TCNN import TCNN
from models.baselines.QCNN import QCNN

def create_model(model_name: str, dataset_name: str, device: torch.device):
    """
    模型工厂：根据名字动态实例化模型，并根据数据集自动适配参数。
    """
    # 1. 动态获取数据集对应的输入通道数和分类数
    in_channels = 2 if dataset_name == "XJTU-SY" else 1
    num_classes = 5 if dataset_name == "XJTU-SY" else 10

    if model_name == "Ours":
        # 加载你自己的模型与物理图谱
        if dataset_name == "XJTU-SY":
            _, edge_index, _ = get_bearing_physics_adjacency()
        else:
            # 这里预留给你未来写的 CWRU 图谱
            # _, edge_index, _ = get_cwru_physics_adjacency()
            edge_index = torch.zeros((2, 0), dtype=torch.long) # 占位
            
        model = HierarchicalExplainableBearingNet(
            edge_index=edge_index.to(device),
            num_nodes=4,
            in_channels=in_channels,
            slice_length=4096,
            num_slices=3,
            num_classes=num_classes
        ).to(device)
        
    elif model_name == "WDCNN":
        model = WDCNN().to(device)
        # 注意：Baseline 模型的底层通常写死了分类数为 10，通道数为 1
        # 如果要在 XJTU (5分类，2通道) 上跑，你需要动态修改其第一层和最后一层
        if in_channels != 1:
            model.cnn.Conv1D_1 = nn.Conv1d(in_channels, 16, 64, 8, 28).to(device)
        if num_classes != 10:
            model.fc2 = nn.Linear(100, num_classes).to(device)
            
    elif model_name == "TCNN":
        model = TCNN().to(device)
        if in_channels != 1:
            model.b1[0] = nn.Conv1d(in_channels, 32, 64, stride=16, padding=28).to(device)
        if num_classes != 10:
            model.b6[-1] = nn.Linear(100, num_classes).to(device)
            
    # elif model_name == "QCNN": ... 依此类推
    else:
        raise ValueError(f"Model {model_name} not supported!")
        
    return model