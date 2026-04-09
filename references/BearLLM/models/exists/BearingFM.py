import os
import numpy as np
from dotenv import dotenv_values
from torch.utils.data import Dataset
from scipy.signal import resample
from torch import nn
from BearLLM.models.FCN import ConvWide, ConvMultiScale


def enhance_data(data):
    if np.random.rand() < 0:
        return data
    raw_length = 12000
    rands = np.random.randn(4)
    horizontal_stretch = rands[0] * 0.01 + 1
    vertical_stretch = rands[1] * 0.01 + 1
    vertical_shift = rands[2] * 0.01
    horizontal_shift = int(rands[3] * 240)
    data = data * vertical_stretch + vertical_shift
    data = np.roll(data, horizontal_shift)
    new_len = int(raw_length * horizontal_stretch)
    if new_len != raw_length:
        data = resample(data, new_len)
        if new_len > raw_length:
            data = data[:raw_length]
        else:
            data = np.pad(data, (0, raw_length - new_len))
    return data


class BearingFMDataset(Dataset):
    def __init__(self, subset, dataset='all'):
        self.subset = subset
        self.data_dir = f'{dotenv_values()['MBHM_DATA_DIR']}{subset}/'
        self.files = os.listdir(self.data_dir)
        if dataset != 'all':
            self.files = [f for f in self.files if dataset in f]
        self.file_num = len(self.files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(f'{self.data_dir}{file}')
        label = int(file.split('_')[1][1:])
        if self.subset == 'train':
            data = enhance_data(data)
        return data.reshape(1, -1), label





class BearingFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv1d(1, 128, 64, stride=32),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 8, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 3, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.gp = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gp(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BearingFMPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvWide(1, 128, 8, 8),
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128),
            ConvMultiScale(128, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3072, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    import torch
    test_input = torch.randn(1024, 1, 12000, dtype=torch.float32).to('cuda')
    model = BearingFM()
    model.to('cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    test_output = model(test_input)
    loss = torch.nn.CrossEntropyLoss()(test_output, torch.randint(0, 10, (1024,)).to('cuda'))
    loss.backward()