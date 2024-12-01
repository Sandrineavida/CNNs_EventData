import torch
import torch.nn as nn

# Define a Depthwise Separable Convolution layer
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution: in_channels == groups
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)

        # Pointwise convolution: 1x1 convolution to combine the depthwise outputs
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Define the Separable Convolutional Network based on LeNet
class SeparableConv_LeNet(nn.Module):
    def __init__(self, num_classes=1, quantised=False):
        super(SeparableConv_LeNet, self).__init__()
        self.num_classes = num_classes

        self.quantised = quantised
        self.quant = torch.quantization.QuantStub() if quantised else None
        self.dequant = torch.quantization.DeQuantStub() if quantised else None

        # Replace Conv2d with DepthwiseSeparableConv layers
        self.conv1 = DepthwiseSeparableConv(in_channels=2, out_channels=6, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = DepthwiseSeparableConv(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = DepthwiseSeparableConv(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu3 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=120, out_features=84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes, bias=False)

    def forward(self, x):
        if self.quantised:
            x = self.quant(x)

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        if self.quantised:
            x = self.dequant(x)

        return x
