import torch
import torch.nn as nn


class CNNLeNet_q(nn.Module):
    def __init__(self, num_classes=1, scale=None, zero_point=None):
        super(CNNLeNet_q, self).__init__()
        self.num_classes = num_classes

        self.scale = scale
        self.zero_point = zero_point

        self.quant = nn.quantized.Quantize(scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8)
        self.dequant = nn.quantized.DeQuantize()

        self.conv1 = nn.quantized.Conv2d(2, 6, 5, bias=False, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.quantized.Conv2d(6, 16, 5, bias=False, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.quantized.Conv2d(16, 120, 5, bias=False, stride=1, padding=0)
        self.relu3 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.quantized.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.quantized.Linear(84, num_classes)

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



