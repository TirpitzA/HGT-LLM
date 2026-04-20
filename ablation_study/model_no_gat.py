"""
model_no_gat.py
===============
消融实验：去除 PhysicsConstrainedGATLayer 的模型变体。
架构: Dual-Stream CNN -> Linear Projection -> Temporal Transformer -> Classification
用于验证 GAT 层对最终性能的贡献。

注意：此文件完全独立于 network/hgt_net.py，不做任何修改。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.hgt_net import CNNFeatureExtractor, TemporalTransformer, PositionalEncoding


class HierarchicalBearingNet_NoGAT(nn.Module):
    """
    消融版本：去掉 GAT 层，直接将 CNN 提取的节点特征展平后投影到 Transformer。
    其他结构 (CNN, Transformer, Attention, FC) 完全保持一致。
    """
    def __init__(self,
                 num_nodes: int = 4,
                 in_channels: int = 2,
                 slice_length: int = 4096,
                 num_slices: int = 3,
                 cnn_hidden: int = 64,
                 transformer_dim: int = 256,
                 transformer_heads: int = 4,
                 transformer_layers: int = 2,
                 num_classes: int = 5,
                 dropout: float = 0.3):
        super(HierarchicalBearingNet_NoGAT, self).__init__()

        self.num_nodes = num_nodes
        self.slice_length = slice_length
        self.num_slices = num_slices
        self.num_classes = num_classes

        # 与原始模型相同的 CNN
        self.cnn = CNNFeatureExtractor(in_channels, cnn_hidden, num_nodes)

        # 直接投影：跳过 GAT，将 CNN 输出 (num_nodes * cnn_hidden) 投影到 transformer_dim
        self.feat_proj = nn.Linear(num_nodes * cnn_hidden, transformer_dim)

        self.pos_encoder = PositionalEncoding(transformer_dim, dropout, max_len=num_slices)
        self.transformer = TemporalTransformer(transformer_dim, transformer_heads, transformer_layers, dropout)
        self.temporal_attention = nn.Linear(transformer_dim, 1)

        self.fc1 = nn.Linear(transformer_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t, x_f=None, return_attention=False):
        batch_size = x_t.size(0)
        x_t = x_t.permute(0, 1, 3, 2)
        if x_f is not None:
            x_f = x_f.permute(0, 1, 3, 2)

        slice_features_list = []

        for slice_idx in range(self.num_slices):
            slice_data_t = x_t[:, slice_idx, :, :]
            slice_data_f = x_f[:, slice_idx, :, :] if x_f is not None else None

            node_features = self.cnn(slice_data_t, slice_data_f)  # (B, num_nodes, cnn_hidden)

            # 【消融关键】直接展平 CNN 特征，跳过 GAT
            cnn_flat = node_features.view(batch_size, -1)  # (B, num_nodes * cnn_hidden)
            slice_repr = self.feat_proj(cnn_flat)           # (B, transformer_dim)
            slice_features_list.append(slice_repr)

        slice_features = torch.stack(slice_features_list, dim=1)  # (B, num_slices, transformer_dim)
        slice_features = self.pos_encoder(slice_features)

        temporal_output = self.transformer(slice_features)
        temporal_attn_scores = self.temporal_attention(temporal_output)
        temporal_attn_weights = F.softmax(temporal_attn_scores.squeeze(-1), dim=1)
        context = torch.bmm(temporal_attn_weights.unsqueeze(1), temporal_output).squeeze(1)

        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        deep_feature = out
        logits = self.fc3(deep_feature)

        output = {'logits': logits}
        if return_attention:
            output['slice_weights'] = temporal_attn_weights
            output['deep_feature'] = deep_feature
        return output
