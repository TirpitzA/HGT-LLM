import torch
import torch.nn as nn

class ConvQuadraticOperation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvQuadraticOperation, self).__init__()
        # 线性项卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # 二次项卷积
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # 结合一次特征和二次特征：W1*x + W2*(x^2)
        return self.conv1(x) + self.conv2(torch.pow(x, 2))


class QCNN(nn.Module):
    def __init__(self):
        super(QCNN, self).__init__()
        self.name = 'QCNN'
        self.cnn = nn.Sequential()
        self.cnn.add_module('Conv1D_1', ConvQuadraticOperation(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))
        self.__make_layerq(16, 32, 1, 2)
        self.__make_layerq(32, 64, 1, 3)
        self.__make_layerq(64, 64, 1, 4)
        self.__make_layerq(64, 64, 1, 5)
        self.__make_layerq(64, 64, 0, 6)

        self.adap_pool = nn.AdaptiveMaxPool1d(3)

        self.fc1 = nn.Linear(192, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)

    def __make_layerq(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), ConvQuadraticOperation(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def forward(self, x):
        out1 = self.cnn(x)
        out1 = self.adap_pool(out1)
        out = self.fc1(out1.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return out