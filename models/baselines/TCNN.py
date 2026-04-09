import torch.nn as nn

class TCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.name = 'TCNN'
        self.b1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, 64, stride=16, padding=28),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.2),
        )
        self.b2 = nn.Sequential(nn.Conv1d(32, 32, 3, stride=1, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(0.2))
        self.b3 = nn.Sequential(nn.Conv1d(32, 64, 3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(0.2))
        self.b4 = nn.Sequential(nn.Conv1d(64, 64, 3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(0.2))
        self.b5 = nn.Sequential(nn.Conv1d(64, 64, 3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2), nn.Dropout(0.2))
        
        self.adap_pool = nn.AdaptiveMaxPool1d(4) # 强制 64 * 4 = 256
        
        self.b6 = nn.Sequential(
            nn.Linear(256, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.adap_pool(x)
        x = x.view(x.size(0), -1)
        x = self.b6(x)
        return x