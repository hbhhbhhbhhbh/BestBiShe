import torch
import torch.nn.functional as F
import numpy as np

from torchvision import models
from torch import nn

 
class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.name="FCN"
        self.n_classes=num_classes
        pretrained_net = models.vgg16_bn(pretrained=True)
        self.stage1 = pretrained_net.features[:7]
        self.stage2 = pretrained_net.features[7:14]
        self.stage3 = pretrained_net.features[14:24]
        self.stage4 = pretrained_net.features[24:34]
        self.stage5 = pretrained_net.features[34:]
        
        #1x1卷积用于调整通道数
        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, num_classes, 1)
        
        #转置卷积用于上采样
        # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **args)
        # 8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        #使用双线性插值对反卷积核进行初始化
        self.upsample_8x.weight.data =  self.bilinear_kernel(num_classes, num_classes, 16)
        #2倍上采样
        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
         #使用双线性插值对反卷积核进行初始化
        self.upsample_2x_1.weight.data =  self.bilinear_kernel(512, 512, 4)
		#2倍上采样
        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        #使用双线性插值对反卷积核进行初始化
        self.upsample_2x_2.weight.data =  self.bilinear_kernel(256, 256, 4)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        s5 = self.upsample_2x_1(s5)
        add1 = s5 + s4

        add1 = self.conv_trans1(add1)
        add1 = self.upsample_2x_2(add1)
        add2 = add1 + s3

        output = self.conv_trans2(add2)
        output = self.upsample_8x(output)
        return output



    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
        weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
        return torch.from_numpy(weight)
