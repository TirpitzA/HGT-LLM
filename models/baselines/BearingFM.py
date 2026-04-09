import torch.nn as nn

class BearingFM(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # 加入 padding（补零），防止长度为 1024 的短序列经过大 stride 卷积后长度变为负数或过小
        self.conv0 = nn.Sequential(nn.Conv1d(in_channels, 128, 64, stride=32, padding=32), nn.ReLU(), nn.BatchNorm1d(128))
        self.conv1 = nn.Sequential(nn.Conv1d(128, 128, 8, stride=3, padding=4), nn.ReLU(), nn.BatchNorm1d(128))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 256, 5, stride=3, padding=2), nn.ReLU(), nn.BatchNorm1d(256))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 3, stride=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128))
        
        # 自带自适应池化，天然不受序列长度影响
        self.gp = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x