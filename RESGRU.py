import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个残差块（ResNet模块）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 卷积层2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 用于匹配输入和输出的通道数
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        # 原始输入通过shortcut跳过
        shortcut = self.shortcut(x)
        
        # 卷积操作
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # 将输入和输出相加，形成残差连接
        x += shortcut
        return F.relu(x)


class EEG_ResNet_GRU(nn.Module):
    def __init__(self, num_channels, num_classes=2):
        super(EEG_ResNet_GRU, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        
        # 通过几个残差块提取空间特征
        self.res_block1 = ResidualBlock(4, 8)
        self.res_block2 = ResidualBlock(8, 16)

        # 自适应池化，固定输出尺寸
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # GRU层，用于建模时序依赖
        self.gru = nn.GRU(input_size=16, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)

        # 分类输出层
        self.fc = nn.Linear(16 * 2, num_classes)  # 128 * 2 because of bidirectional GRU

    def forward(self, x):
        # 第一层卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # 通过残差块提取空间特征
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 自适应池化，输出固定尺寸
        x = self.pool(x)
        
        # 扁平化（Flatten）
        x = x.view(x.size(0), -1)
        
        # 增加一个时间维度，适应GRU输入
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, features)
        
        # GRU处理时序依赖
        gru_out, _ = self.gru(x)
        
        # 获取GRU的最后一个时间步的输出
        gru_out_last = gru_out[:, -1, :]  # Shape: (batch_size, hidden_size * num_directions)

        # 通过全连接层进行分类
        output = self.fc(gru_out_last)
        
        return output

# 变体1: 只使用一个Residual块
class AblationModel_SingleResBlock(nn.Module):
    """
    消融实验变体1:
    - 移除了第二个残差块 (res_block2).
    - 只保留一个残差块 (res_block1).
    - 调整了后续GRU和FC层的输入维度以确保兼容性.
    """
    def __init__(self, num_classes=2):
        super(AblationModel_SingleResBlock, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        
        # 只使用一个残差块
        self.res_block1 = ResidualBlock(4, 16)

        # 自适应池化，固定输出尺寸
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # GRU层
        self.gru = nn.GRU(input_size=16, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)

        # 分类输出层 (输入维度与GRU的hidden_size*2保持一致)
        self.fc = nn.Linear(16 * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x) # 只通过一个残差块
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        gru_out_last = gru_out[:, -1, :]
        output = self.fc(gru_out_last)
        return output


# 变体2: 不使用GRU层，仅残差模块
class AblationModel_FullResidual(nn.Module):
    """
    消融实验变体3:
    - 保留了完整的ResNet部分.
    - 移除了GRU层.
    - ResNet提取的特征经池化和扁平化后直接送入FC层进行分类.
    """
    def __init__(self, num_classes=2):
        super(AblationModel_FullResidual, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        
        # 完整的残差块
        self.res_block1 = ResidualBlock(4, 8)
        self.res_block2 = ResidualBlock(8, 16)

        # 自适应池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # (无GRU层)

        # 分类输出层，输入维度为ResNet部分的最终输出通道数16
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # 扁平化，shape: (batch_size, 16)
        # (无GRU层)
        output = self.fc(x)
        return output

# 变体3: 只使用GRU模块 
class AblationModel_PreprocessedGRU(nn.Module):
    """
    消融实验变体:
    - 在原始的OnlyGRU模型基础上，增加了与其他模型一致的预处理模块。
    - 预处理模块: 一个Conv2d层 + 一个BatchNorm2d层。
    - 目的是测试一个浅层卷积预处理对GRU性能的影响，并与ResNet的深度特征提取进行对比。
    """
    def __init__(self,num_channels, num_classes=2):
        super(AblationModel_PreprocessedGRU, self).__init__()
        
        # --- 新增的预处理模块 ---
        # 与主模型 EEG_ResNet_GRU 的第一层保持一致 
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        
        # --- 调整GRU层以适应预处理后的数据 ---
        # 预处理后，特征维度变为 4 * num_channels (conv1的输出通道数 * 原始EEG通道数)
        self.gru = nn.GRU(input_size=4*num_channels , hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)

        # 分类输出层保持不变
        self.fc = nn.Linear(16 * 2, num_classes)

    def forward(self, x):
        # 原始输入维度: (batch, 1, channels, time_points)
        
        # --- 1. 通过新增的预处理模块 ---
        x = F.relu(self.bn1(self.conv1(x)))
        # 此处x的维度: (batch, 4, channels, time_points)

        # --- 2. 调整维度以匹配GRU输入 (batch, seq_len, features) ---
        # (batch, 4, channels, time_points) -> (batch, time_points, 4, channels)
        x = x.permute(0, 3, 1, 2)
        
        # 将最后两个维度展平，作为GRU的特征维度
        # (batch, time_points, 4, channels) -> (batch, time_points, 4 * channels)
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.reshape(batch_size, seq_len, -1)
        
        # --- 3. GRU处理时序依赖 ---
        gru_out, _ = self.gru(x)
        
        # 获取最后一个时间步的输出
        gru_out_last = gru_out[:, -1, :]
        
        # --- 4. 分类 ---
        output = self.fc(gru_out_last)
        return output
# ---------------------------------------------------------------------------
# 示例使用: 验证模型结构和输出维度
# ---------------------------------------------------------------------------

# print("--- 验证消融实验模型 ---")

# # 定义一个模拟的输入数据
# # batch_size=64, input_channels=1, num_eeg_channels=2, time_points=1000
# # 注意: num_eeg_channels可以根据您脑区的实际通道数调整
# input_tensor = torch.randn(64, 1, 2, 1000) 
# print(f"输入数据维度: {input_tensor.shape}\n")

# # 变体1: 单个残差块
# model_v1 = AblationModel_SingleResBlock(num_classes=2)
# output_v1 = model_v1(input_tensor)
# print(f"变体1 (单个ResBlock) 输出维度: {output_v1.shape}") # 期望输出: torch.Size([64, 2])

# # 变体2: 无残差块
# model_v2 = AblationModel_OnlyGRU(num_channels=2,num_classes=2)
# output_v2 = model_v2(input_tensor)
# print(f"变体2 (GRU) 输出维度: {output_v2.shape}") # 期望输出: torch.Size([64, 2])

# # 变体3: 无GRU
# model_v3 = AblationModel_NoGRU(num_classes=2)
# output_v3 = model_v3(input_tensor)
# print(f"变体3 (无GRU) 输出维度: {output_v3.shape}") # 期望输出: torch.Size([64, 2])

# # 示例使用

# model = EEG_ResNet_GRU(num_channels=2, num_classes=2)
# # def model(num_channels=2, num_classes=2):
# #     return EEG_ResNet_GRU(num_channels=2, num_classes=2)
# input_data = torch.randn(64, 1, 2, 1000)  # batch_size=64, channels=19, time_series=1000
# output = model(input_data)
# print(output.shape)  # Expected output: torch.Size([64, 2]) for 2 classes
