import math
import torch
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()


class ConvQuadraticOperation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 bias: bool = True):
        super(ConvQuadraticOperation, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_r = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_g = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))
        self.weight_b = Parameter(torch.empty(
            (out_channels, in_channels, kernel_size)))

        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))

        self.__reset_bias()

    def forward(self, x):
        out = F.conv1d(x, self.weight_r, self.bias_r, self.stride, self.padding, 1, 1) \
              * F.conv1d(x, self.weight_g, self.bias_g, self.stride, self.padding, 1, 1) \
              + F.conv1d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 1, 1)
        return out


class ConvTransQuadraticOperation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 bias: bool = True):
        super(ConvTransQuadraticOperation, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_r = Parameter(torch.empty(
            (in_channels, out_channels, kernel_size)))
        self.weight_g = Parameter(torch.empty(
            (in_channels, out_channels, kernel_size)))
        self.weight_b = Parameter(torch.empty(
            (in_channels, out_channels, kernel_size)))

        if bias:
            self.bias_r = Parameter(torch.empty(out_channels))
            self.bias_g = Parameter(torch.empty(out_channels))
            self.bias_b = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        self.reset_parameters()

    def __reset_bias(self):
        if self.bias_r is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_r, -bound, bound)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        self.__reset_bias()

    def forward(self, x):
        out = F.conv_transpose1d(x, self.weight_r, self.bias_r, self.stride, self.padding, 0, 1) \
              * F.conv_transpose1d(x, self.weight_g, self.bias_g, self.stride, self.padding, 0, 1) \
              + F.conv_transpose1d(torch.pow(x, 2), self.weight_b, self.bias_b, self.stride, self.padding, 0, 1)
        return out


class QCNN(nn.Module):
    """
    QCNN builder
    """

    def __init__(self):
        super(QCNN, self).__init__()
        self.name = 'QCNN'
        self.cnn = nn.Sequential()
        # self.cnn1 = nn.Sequential()
        self.cnn.add_module('Conv1D_1', ConvQuadraticOperation(1, 16, 64, 8, 28))
        self.cnn.add_module('BN_1', nn.BatchNorm1d(16))
        self.cnn.add_module('Relu_1', nn.ReLU())
        self.cnn.add_module('MAXPool_1', nn.MaxPool1d(2, 2))
        self.__make_layerq(16, 32, 1, 2)
        self.__make_layerq(32, 64, 1, 3)
        self.__make_layerq(64, 64, 1, 4)
        self.__make_layerq(64, 64, 1, 5)
        self.__make_layerq(64, 64, 0, 6)

        self.fc1 = nn.Linear(192, 100)
        self.relu1 = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)

    def __make_layerq(self, in_channels, out_channels, padding, nb_patch):
        self.cnn.add_module('Conv1D_%d' % (nb_patch), ConvQuadraticOperation(in_channels, out_channels, 3, 1, padding))
        self.cnn.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def __make_layerc(self, in_channels, out_channels, padding, nb_patch):
        self.cnn1.add_module('Conv1D_%d' % (nb_patch), nn.Conv1d(in_channels, out_channels, 3, 1, padding))
        self.cnn1.add_module('BN_%d' % (nb_patch), nn.BatchNorm1d(out_channels))
        # self.cnn.add_module('DP_%d' %(nb_patch), nn.Dropout(0.5))
        self.cnn1.add_module('ReLu_%d' % (nb_patch), nn.ReLU())
        self.cnn1.add_module('MAXPool_%d' % (nb_patch), nn.MaxPool1d(2, 2))

    def forward(self, x):
        out1 = self.cnn(x)
        out = self.fc1(out1.view(x.size(0), -1))
        out = self.relu1(out)
        out = self.dp(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    X = torch.rand(32, 1, 2048).cuda()
    m = QCNN()
    m = m.cuda()
    print(m(X).shape)
