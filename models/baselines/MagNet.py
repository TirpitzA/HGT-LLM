import torch.nn as nn

class FeatureGenerator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, 32, 15, 1, 7), nn.MaxPool1d(2, 2), nn.ReLU(), nn.BatchNorm1d(32))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 7, 1, 3), nn.MaxPool1d(2, 2), nn.ReLU(), nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 3, 1, 1), nn.MaxPool1d(2, 2), nn.ReLU(), nn.BatchNorm1d(64))
        
        # 输入如果为 1024，经过三次 MaxPool1d(2) 后长度变为 128。
        # 这里改为 128 避免产生畸形的上采样特征放大，保持特征纯净度
        self.adap_pool = nn.AdaptiveMaxPool1d(128) 
        
        self.fc = nn.Sequential(
            # 同步修改全连接层的维度: 64通道 * 128序列长度
            nn.Linear(64 * 128, 512), 
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adap_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def classifier(num_classes=10):
    return nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, num_classes))

def discriminator():
    return nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 262), nn.Sigmoid())

class GradientReversalLayer(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x
    def backward(self, grad_output): return -grad_output

class MagNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.fg = FeatureGenerator(in_channels=in_channels)
        self.classifier = classifier(num_classes=num_classes)
        self.grl = GradientReversalLayer()
        self.discriminator = discriminator()

    def forward(self, x):
        x = self.fg(x)
        y = self.classifier(x)
        x_grl = self.grl(x)
        d = self.discriminator(x_grl)
        return y, d