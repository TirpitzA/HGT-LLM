"""
Physics-Based Hierarchical Fault Diagnosis Model
Architecture: Dual-Stream CNN (Time + Freq) -> GAT -> Temporal Transformer -> Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math

class CNNFeatureExtractor(nn.Module):
    """
    恢复你原本的骄傲：双流 1D CNN。
    时域分支捕捉瞬态冲击，频域分支捕捉共振峰，完美融合馈送给后续物理网络！
    """
    def __init__(self, in_channels: int = 2, hidden_dim: int = 64, num_nodes: int = 4):
        super(CNNFeatureExtractor, self).__init__()
        branch_dim = (hidden_dim * num_nodes) // 2
        
        # Time Branch
        self.conv1_t = nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7)
        self.conv2_t = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.conv3_t = nn.Conv1d(64, branch_dim, kernel_size=3, stride=1, padding=1)
        self.bn1_t = nn.BatchNorm1d(32)
        self.bn2_t = nn.BatchNorm1d(64)
        self.bn3_t = nn.BatchNorm1d(branch_dim)
        
        # Frequency Branch
        self.conv1_f = nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7)
        self.conv2_f = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.conv3_f = nn.Conv1d(64, branch_dim, kernel_size=3, stride=1, padding=1)
        self.bn1_f = nn.BatchNorm1d(32)
        self.bn2_f = nn.BatchNorm1d(64)
        self.bn3_f = nn.BatchNorm1d(branch_dim)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
    def forward(self, x_t, x_f=None):
        # Time Branch Forward
        t = F.relu(self.bn1_t(self.conv1_t(x_t)))
        t = F.relu(self.bn2_t(self.conv2_t(t)))
        t = F.relu(self.bn3_t(self.conv3_t(t)))
        t = self.pool(t).squeeze(-1) 
        
        # Frequency Branch Forward (动态执行FFT保留切片间差异)
        if x_f is None:
            fft_complex = torch.fft.rfft(x_t, dim=2)
            x_f = torch.abs(fft_complex) / x_t.size(2)
            
        f = F.relu(self.bn1_f(self.conv1_f(x_f)))
        f = F.relu(self.bn2_f(self.conv2_f(f)))
        f = F.relu(self.bn3_f(self.conv3_f(f)))
        f = self.pool(f).squeeze(-1) 
        
        feat = torch.cat([t, f], dim=1) 
        feat = feat.view(-1, self.num_nodes, self.hidden_dim) 
        return feat

class PhysicsConstrainedGATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super(PhysicsConstrainedGATLayer, self).__init__()
        self.gat = GATConv(in_features, out_features, heads=num_heads, dropout=dropout, concat=True)
        self.num_heads = num_heads
        self.out_features = out_features
        
    def forward(self, x, edge_index):
        x, attention_weights = self.gat(x, edge_index, return_attention_weights=True)
        return x, attention_weights

class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
        
    def forward(self, x):
        return self.transformer(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class HierarchicalExplainableBearingNet(nn.Module):
    def __init__(self, 
                 edge_index: torch.Tensor,
                 num_nodes: int = 4,
                 in_channels: int = 2,
                 slice_length: int = 4096,
                 num_slices: int = 3,
                 cnn_hidden: int = 64,
                 gat_hidden: int = 64,
                 gat_heads: int = 4,
                 transformer_dim: int = 256,
                 transformer_heads: int = 4,
                 transformer_layers: int = 2,
                 num_classes: int = 5,
                 dropout: float = 0.3):
        super(HierarchicalExplainableBearingNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.slice_length = slice_length
        self.num_slices = num_slices
        self.gat_heads = gat_heads
        self.gat_hidden = gat_hidden
        self.num_classes = num_classes
        
        self.cnn = CNNFeatureExtractor(in_channels, cnn_hidden, num_nodes)
        self.gat = PhysicsConstrainedGATLayer(cnn_hidden, gat_hidden, gat_heads, dropout)
        self.feat_proj = nn.Linear(num_nodes * gat_hidden * gat_heads, transformer_dim)
        
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout, max_len=num_slices)
        self.transformer = TemporalTransformer(transformer_dim, transformer_heads, transformer_layers, dropout)
        self.temporal_attention = nn.Linear(transformer_dim, 1)
        
        self.fc1 = nn.Linear(transformer_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('edge_index', edge_index)
        
    def forward(self, x_t, x_f=None, return_attention=False):
        batch_size = x_t.size(0)
        x_t = x_t.permute(0, 1, 3, 2)
        if x_f is not None: x_f = x_f.permute(0, 1, 3, 2)
        
        sensor_attention_weights = []
        slice_features_list = []
        
        for slice_idx in range(self.num_slices):
            slice_data_t = x_t[:, slice_idx, :, :] 
            slice_data_f = x_f[:, slice_idx, :, :] if x_f is not None else None
            
            node_features = self.cnn(slice_data_t, slice_data_f) 
            
            gat_outputs = []
            sensor_attn_list = []
            
            for b_idx in range(batch_size):
                b_node_features = node_features[b_idx] 
                gat_out, attn_weights = self.gat(b_node_features, self.edge_index) 
                gat_outputs.append(gat_out)
                
                if return_attention:
                    if isinstance(attn_weights, tuple):
                        _, attn_vals = attn_weights
                    else:
                        attn_vals = attn_weights
                    edge_attn_mean = attn_vals.mean(dim=1)
                    sensor_attn_list.append(edge_attn_mean)
            
            gat_output = torch.stack(gat_outputs, dim=0) 
            if return_attention:
                 sensor_attention_weights.append(torch.stack(sensor_attn_list, dim=0))
                 
            gat_flat = gat_output.view(batch_size, -1)
            slice_repr = self.feat_proj(gat_flat) 
            slice_features_list.append(slice_repr)
            
        slice_features = torch.stack(slice_features_list, dim=1)
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
            output['edge_weights'] = torch.stack(sensor_attention_weights, dim=1)  
            output['slice_weights'] = temporal_attn_weights
            output['deep_feature'] = deep_feature  
        return output