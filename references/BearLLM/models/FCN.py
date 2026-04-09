import os
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ConvWide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=8):
        super(ConvWide, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMultiScale, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError('out_channels should be divisible by 4')
        out_channels = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 4, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_channels * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x
        x = torch.cat([x1, x], dim=1)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.conv_query = ConvWide(1, 60, 8, 8)
        self.conv_ref = ConvWide(1, 8, 8, 8)
        self.conv_res = ConvWide(1, 60, 8, 8)
        self.conv = nn.Sequential(
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128)
        )

    def forward(self, x):
        query = x[:, :1, :]
        ref = x[:, 1:, :]
        res = query - ref
        query = self.conv_query(query)
        ref = self.conv_ref(ref)
        res = self.conv_res(res)
        x = torch.cat([query, ref, res], dim=1)
        x = self.conv(x)
        return x

    def save_weights(self, weights_dir):
        torch.save(self.state_dict(), weights_dir + '/feature_encoder.pth')

    def load_weights(self, weights_dir):
        self.load_state_dict(torch.load(weights_dir + '/feature_encoder.pth', map_location='cpu'))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(128 * 47, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

    def save_weights(self, weights_dir):
        torch.save(self.state_dict(), weights_dir + '/classifier.pth')

    def load_weights(self, weights_dir):
        self.load_state_dict(torch.load(weights_dir + '/classifier.pth', map_location='cpu'))


class FaultClassificationNetwork(nn.Module):
    def __init__(self):
        super(FaultClassificationNetwork, self).__init__()
        self.encoder = FeatureEncoder()
        self.classifier = Classifier()

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

    def save_weights(self, weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        self.encoder.save_weights(weights_dir)
        self.classifier.save_weights(weights_dir)

    def load_weights(self, weights_dir):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f'{weights_dir} not found')
        self.encoder.load_weights(weights_dir)
        self.classifier.load_weights(weights_dir)


if __name__ == '__main__':
    def test():
        test_signal = torch.randn(32, 3, 24000)
        model = FaultClassificationNetwork()
        model(test_signal)
        model.save_weights('test')
        model.load_weights('test')
        os.remove('test/feature_encoder.pth')
        os.remove('test/classifier.pth')
        os.rmdir('test')

    test()
