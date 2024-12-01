import torch
import torch.nn as nn

# Define a Quantised Depthwise Separable Convolution layer
class DepthwiseSeparableConv_q(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv_q, self).__init__()

        # Depthwise convolution: in_channels == groups
        self.depthwise = nn.quantized.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)

        # Pointwise convolution: 1x1 convolution to combine the depthwise outputs
        self.pointwise = nn.quantized.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Define the Separable Convolutional Network based on LeNet
class SeparableConv_LeNet_q(nn.Module):
    def __init__(self, num_classes=1, scale=None, zero_point=None):
        super(SeparableConv_LeNet_q, self).__init__()
        self.num_classes = num_classes

        self.scale = scale
        self.zero_point = zero_point

        self.quant = nn.quantized.Quantize(scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8)
        self.dequant = nn.quantized.DeQuantize()

        # Replace quantised Conv2d layers with DepthwiseSeparableConv_q layers
        self.conv1 = DepthwiseSeparableConv_q(in_channels=2, out_channels=6, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = DepthwiseSeparableConv_q(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = DepthwiseSeparableConv_q(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu3 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.quantized.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.quantized.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.quant(x)

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant(x)

        return x
