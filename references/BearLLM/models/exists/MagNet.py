import torch.nn as nn


class FeatureGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 15, 1, 7),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 300, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def classifier():
    return nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def discriminator():
    return nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, 262),
        nn.Sigmoid(),
    )


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return -grad_output


class MagNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fg = FeatureGenerator()
        self.classifier = classifier()
        self.grl = GradientReversalLayer()
        self.discriminator = discriminator()

    def forward(self, x):
        x = self.fg(x)
        y = self.classifier(x)
        x = self.grl(x)
        d = self.discriminator(x)
        return y, d


if __name__ == "__main__":
    import torch
    from math import log
    test_input = torch.randn(5, 1, 2400)
    magnet = MagNet()
    _y, _d = magnet(test_input)
    print(_y.size(), _d.size())