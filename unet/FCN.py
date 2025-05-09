import torch
import torch.nn.functional as F
import numpy as np

from torchvision import models
from torch import nn
class FCN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.name = "FCN"
        self.n_classes = n_classes
        
        # 加载预训练VGG16
        pretrained_net = models.vgg16_bn(pretrained=True)
        
        # 划分阶段（注意：VGG16的MaxPool2d会下采样2倍）
        self.stage1 = pretrained_net.features[:7]    # 输出尺寸: 128x128 (256/2)
        self.stage2 = pretrained_net.features[7:14]  # 输出尺寸: 64x64   (256/4)
        self.stage3 = pretrained_net.features[14:24] # 输出尺寸: 32x32   (256/8)
        self.stage4 = pretrained_net.features[24:34] # 输出尺寸: 16x16   (256/16)
        self.stage5 = pretrained_net.features[34:]   # 输出尺寸: 8x8     (256/32)

        # 1x1卷积调整通道数
        self.scores1 = nn.Conv2d(512, n_classes, 1)  # 用于stage5输出 (8x8)
        self.scores2 = nn.Conv2d(512, n_classes, 1)  # 用于stage4输出 (16x16)
        self.scores3 = nn.Conv2d(256, n_classes, 1)  # 用于stage3输出 (32x32)

        # 初始化权重为零
        for layer in [self.scores1, self.scores2, self.scores3]:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # 反卷积层（适配256x256输入）
        self.upsample_2x_1 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_2x_2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_8x = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16, stride=8, padding=4, bias=False)

        # 双线性插值初始化
        self.upsample_2x_1.weight.data = self.bilinear_kernel(n_classes, n_classes, 4)
        self.upsample_2x_2.weight.data = self.bilinear_kernel(n_classes, n_classes, 4)
        self.upsample_8x.weight.data = self.bilinear_kernel(n_classes, n_classes, 16)

    def forward(self, x):
        s1 = self.stage1(x)    # [B, 64, 128, 128]
        s2 = self.stage2(s1)   # [B, 128, 64, 64]
        s3 = self.stage3(s2)   # [B, 256, 32, 32]
        s4 = self.stage4(s3)   # [B, 512, 16, 16]
        s5 = self.stage5(s4)   # [B, 512, 8, 8]

        # FCN-32s分支
        scores1 = self.scores1(s5)            # [B, n_classes, 8, 8]
        s5_up = self.upsample_2x_1(scores1)  # [B, n_classes, 16, 16]

        # FCN-16s分支
        s4_scores = self.scores2(s4)         # [B, n_classes, 16, 16]
        add1 = s5_up + s4_scores             # 跳跃连接1
        add1_up = self.upsample_2x_2(add1)   # [B, n_classes, 32, 32]

        # FCN-8s分支
        s3_scores = self.scores3(s3)         # [B, n_classes, 32, 32]
        add2 = add1_up + s3_scores           # 跳跃连接2
        output = self.upsample_8x(add2)      # [B, n_classes, 256, 256]

        return output


    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        """生成双线性插值核"""
        factor = (kernel_size + 1) // 2
        center = factor - 0.5 if kernel_size % 2 == 0 else factor - 1
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)