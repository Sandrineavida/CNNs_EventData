import torch
import torch.nn as nn

# 定义一个 AdaptiveAvgPool2d 层
pool = nn.AdaptiveAvgPool2d(1)

# 输入张量 (Batch size=2, Channels=3, Height=4, Width=4)
x = torch.randn(2, 3, 4, 4)

# 应用池化
output = pool(x)

print("Input shape:", x.shape)  # (2, 3, 4, 4)
print("Output shape:", output.shape)  # (2, 3, 1, 1)

# print("x: ", x)
# print("output: ", output)

print(output.view(output.size(0), -1).shape) # (2, 3), 目的是将输出展平为 (Batch size, Channels)
print(output.size(0)) # 2
print(output.size(1)) # 3
print(output.size(2)) # 1




