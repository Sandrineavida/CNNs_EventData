import torch.nn as nn
import torch.quantization as quantization

class CNNLeNet(nn.Module):
    def __init__(self, num_classes = 1, quantised = False):
        super(CNNLeNet, self).__init__()
        self.num_classes = num_classes

        self.quantised = quantised
        self.quant = quantization.QuantStub() if quantised else None
        self.dequant = quantization.DeQuantStub() if quantised else None

        self.conv1 = nn.Conv2d(2, 6, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2,2)
        self.conv3 = nn.Conv2d(16, 120, 5, bias=False)
        self.relu3 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes, bias=False)

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

class CNNLeNetWithSkip(nn.Module):
    def __init__(self, num_classes=1, quantised=False):
        super(CNNLeNetWithSkip, self).__init__()
        self.num_classes = num_classes

        self.quantised = quantised
        self.quant = quantization.QuantStub() if quantised else None
        self.dequant = quantization.DeQuantStub() if quantised else None

        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 6, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5, bias=False)
        self.relu3 = nn.ReLU()

        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes, bias=False)

    def forward(self, x):
        if self.quantised:
            x = self.quant(x)

        # First block
        x1 = self.conv1(x)  # Output of conv1
        x = self.pool1(self.relu1(x1))

        # Second block with skip connection
        x2 = self.conv2(x)
        x = self.pool2(self.relu2(x2 + x1))  # Add the skip connection from x1

        # Third block
        x = self.relu3(self.conv3(x))

        # Fully connected layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        if self.quantised:
            x = self.dequant(x)

        return x
