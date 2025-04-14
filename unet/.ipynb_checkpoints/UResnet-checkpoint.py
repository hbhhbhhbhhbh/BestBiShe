import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .CBAM import *
from .MSAA import *
from .ASPP import *
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)  # 添加 Dropout
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)  # 添加 Dropout
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)
        return out

    
class Up(nn.Module):  # 将x1上采样，然后调整为x2的大小
    """Upscaling then double conv"""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x1, x2):
        x1 = self.up(x1) # 将传入数据上采样，
        
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 填充为x2相同的大小
        return x1   
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),  # 添加 Dropout
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

   
    
class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        )
        self.conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv(x)
        return x
class UResnet(nn.Module):
    def __init__(self,block,layers,num_classes, input_channels=3, deep_supervision=False):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]
        self.n_channels=3
        self.n_classes=num_classes
        self.name="UR"
        self.in_channels = nb_filter[0]
        self.relu = nn.ReLU()
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Up = Up()

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block,nb_filter[1],layers[0],1)
        self.conv2_0 = self._make_layer(block,nb_filter[2],layers[1],1)
        self.conv3_0 = self._make_layer(block,nb_filter[3],layers[2],1)
        self.conv4_0 = self._make_layer(block,nb_filter[4],layers[3],1)

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock((nb_filter[1] +nb_filter[2]) * block.expansion, nb_filter[1], nb_filter[1] * block.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] +nb_filter[3]) * block.expansion, nb_filter[2], nb_filter[2] * block.expansion)
        self.conv3_1 = VGGBlock((nb_filter[3] +nb_filter[4]) * block.expansion, nb_filter[3], nb_filter[3] * block.expansion)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock((nb_filter[1]*2+nb_filter[2]) * block.expansion, nb_filter[1], nb_filter[1] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2]*2+nb_filter[3]) * block.expansion, nb_filter[2], nb_filter[2] * block.expansion)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock((nb_filter[1]*3+nb_filter[2]) * block.expansion, nb_filter[1], nb_filter[1] * block.expansion)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])
        self.cbam= CBAM(64)
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)
        self.ASPP = ASPP(64,64)
        self.ASPP1 = ASPP(128,128)
        self.ASPP2 = ASPP(256,256)
        self.ASPP3 = ASPP(512,512)
        self.ASPP4=ASPP(1024,1024)
        self.MSAA0=MSAA(64)
        self.MSAA=MSAA(128)
        self.MSAA1 = MSAA(256)
        self.MSAA2 = MSAA(512)
        self.MSAA3 = MSAA(1024 )
        self.MSAA4 = MSAA(2048)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self,block, middle_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, middle_channels, stride))
            self.in_channels = middle_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
    # 编码器部分
        x0_0 = self.conv0_0(input)
        x0_0 = self.MSAA0(x0_0)
        x0_0 = self.ASPP(x0_0)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.MSAA(x1_0)
        x1_0 = self.ASPP1(x1_0)
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.MSAA1(x2_0)
        x2_0 = self.ASPP2(x2_0)
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.MSAA2(x3_0)
        x3_0 = self.ASPP3(x3_0)
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print("x4: ",x4_0.shape)
        x4_0 = self.cbam4(x4_0)
        
        
        
        # 解码器部分
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0, x0_0)], 1))
        x0_1=self.cbam(x0_1)
        # print("x01: ",x0_1.shape)

        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0, x1_0)], 1))
        # print("x11: ",x1_1.shape)
        x1_1=self.cbam1(x1_1)


        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1, x0_0)], 1))
        # print("x02: ",x0_2.shape)
        x0_2=self.cbam(x0_2)


        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0, x2_0)], 1))
        # print("x21: ",x2_1.shape)
        x2_1=self.cbam2(x2_1)
  

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1, x1_0)], 1))
        # print("x12: ",x1_2.shape)
        x1_2=self.cbam1(x1_2)


        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2, x0_0)], 1))
        # print("x03: ",x0_3.shape)
        x0_3=self.cbam(x0_3)


        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0, x3_0)], 1))
        x3_1 = self.cbam3(x3_1)
        

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1, x2_0)], 1))
        x2_2 = self.cbam2(x2_2)
        

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2, x1_0)], 1))
        x1_3 = self.cbam1(x1_3)
        

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3, x0_0)], 1))
        x0_4 = self.cbam(x0_4)
        

        # 输出部分
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:

      
            output = self.final(x0_4)
            return output