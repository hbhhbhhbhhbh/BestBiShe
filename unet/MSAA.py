import torch
import torch.nn as nn
import torch.nn.functional as F

class MSAA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(MSAA, self).__init__()
        self.in_channels = in_channels
        
        # 多尺度卷积层
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),  # 升维
            nn.Sigmoid()  # 生成通道注意力权重
        )
        
        # 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),  # 空间特征提取
            nn.Sigmoid()  # 生成空间注意力权重
        )
        
    def forward(self, x):
        # 多尺度特征提取
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.conv7x7(x)
        # 多尺度特征融合（相加）
        x_fused = x1 + x2 + x3
        
        # 通道注意力
        channel_weights = self.channel_attention(x_fused)
        x_channel = x_fused * channel_weights
        
        # 空间注意力
        spatial_weights = self.spatial_attention(x_channel)
        x_spatial = x_channel * spatial_weights
        
        # 最终输出
        return x_spatial