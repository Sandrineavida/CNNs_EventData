import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
import psutil
import os
from models.cnn_lenet import CNNLeNet, CNNwithSkip
from models.separable_convolution import SeparableConv_LeNet

model = CNNLeNet(num_classes=1, quantised=False)
inputs = torch.randn(1, 2, 32, 32)
flop_analysis = FlopCountAnalysis(model, inputs)
print("FLOP for CNNLeNet with 1 input sample of size (32,32) on a Binary Classification task: ")
print(flop_count_table(flop_analysis))

# # 检查当前进程的内存使用情况
# process = psutil.Process(os.getpid())
# start_mem = process.memory_info().rss  # 物理内存占用（RSS）
# output = model(inputs) # 推理
# end_mem = process.memory_info().rss
# print(f"Memory usage during inference: {(end_mem - start_mem) :.2f} B")


print("\n")
model = CNNwithSkip(num_classes=1, quantised=True)
inputs = torch.randn(1, 2, 32, 32)
flop_analysis = FlopCountAnalysis(model, inputs)
print("FLOP for CNNwithSkip with 1 input sample of size (32,32) on a Binary Classification task: ")
print(flop_count_table(flop_analysis))

print("\n")
model = SeparableConv_LeNet(num_classes=1, quantised=False)
inputs = torch.randn(1, 2, 32, 32)
flop_analysis = FlopCountAnalysis(model, inputs)
print("FLOP for SeparableConv_LeNet with 1 input sample of size (32,32) on a Binary Classification task: ")
print(flop_count_table(flop_analysis))




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




