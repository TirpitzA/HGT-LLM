import torch
import torch.nn as nn

class ZeroShotChannelAdapter(nn.Module):
    """
    Adapter to map target dataset channel count to source model channel count.
    Used for Zero-shot robustness testing across different sensor distributions.
    """
    def __init__(self, target_channels, source_channels):
        super().__init__()
        self.target_channels = target_channels
        self.source_channels = source_channels
        
    def forward(self, x):
        # x shape: [Batch, Slices, Length, Channels]
        B, S, L, C = x.shape
        if C == self.source_channels:
            return x
        elif C < self.source_channels:
            # Padding with zeros or mirroring
            padding = torch.zeros(B, S, L, self.source_channels - C, device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=-1)
        else:
            # Selecting or averaging
            return x[:, :, :, :self.source_channels]

def apply_zero_shot_adaptation(model, target_config, source_config):
    """
    Injects the ZeroShotChannelAdapter into the model's preprocessing.
    """
    tc = target_config['model_params']['in_channels']
    sc = source_config['model_params']['in_channels']
    if tc != sc:
        print(f"[ADAPT] Mapping {tc} channels to {sc} channels for zero-shot testing.")
        adapter = ZeroShotChannelAdapter(tc, sc)
        # Note: In HGT, the first layer is often a CNN that expects sc channels.
        # We wrap the model or inject the adapter.
    return adapter
