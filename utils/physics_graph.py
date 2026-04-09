"""
Generalized Physics Graph Definition for Bearing Datasets
"""
import torch
import numpy as np

def get_bearing_physics_adjacency():
    """
    构建统一的轴承物理空间拓扑关系图。
    节点顺序 (基于配置约定): 0-内圈(Inner Race), 1-外圈(Outer Race), 2-滚珠(Ball), 3-保持架(Cage)
    注意：节点的名称字典将直接从相应的数据集 YAML config 中加载。
    
    Returns:
        A: (4, 4) 邻接矩阵 numpy array
        edge_index: (2, num_edges) PyTorch Geometric edge index
    """
    num_nodes = 4
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # 物理接触关系定义
    edges = [
        (0, 2), (2, 0),  # 内圈 <-> 滚珠
        (1, 2), (2, 1),  # 外圈 <-> 滚珠
        (3, 2), (2, 3),  # 保持架 <-> 滚珠
    ]
    
    # 加入自环 (Self-loops，允许节点保留自身特征)
    for i in range(num_nodes):
        edges.append((i, i))
        
    for src, dst in edges:
        A[src, dst] = 1.0
        
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    
    return A, edge_index

if __name__ == "__main__":
    A, edge_index = get_bearing_physics_adjacency()
    print("=== 通用物理邻接矩阵 A ===")
    print(A)
    print("\n=== Edge Index ===")
    print(edge_index)
