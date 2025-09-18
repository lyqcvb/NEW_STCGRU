import torch
import torch.nn as nn
import torch.nn.functional as F

class PSDNet(nn.Module):
    def __init__(self, input_channels):
        super(PSDNet, self).__init__()
        # 使用动态输入通道数
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Input: (1, channels, 159)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 注意：此时展平后的维度 = 16 * (channels // 2) * (159 // 2)
        # 为了动态计算，FC 层将在 forward 中延后定义（lazy init）
        self.fc1 = None  # 占位
        self.fc2 = nn.Linear(in_features=300, out_features=2)  # 分类层

    def forward(self, x):
        x = self.conv1(x)    # -> (batch, 16, ch, 159)
        x = self.relu(x)
        x = self.pool(x)     # -> (batch, 16, ch//2, 79)

        x = x.view(x.size(0), -1)  # 展平，例如 (batch, 11376) if input was 19x159

        # 动态构造全连接层（只有第一次 forward 会触发）
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 300)
            self.fc1.to(x.device)

        x = self.fc1(x)      # -> (batch, 300)
        x = self.fc2(x)      # -> (batch, 2)
        return x
    
# 例子：输入有 8 个样本，每个样本为 2 个通道、159 时间点
input_tensor = torch.randn(8, 1, 19, 159)   # 或者换成 8,1,19,159

model = PSDNet(input_channels=2)   # 动态传入通道数
output = model(input_tensor)
print(output.shape)  # (8, 2)

