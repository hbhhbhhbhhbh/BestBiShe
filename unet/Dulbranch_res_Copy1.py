import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .MSAA import *

from .unet_parts import *
from .CBAM import *
from .resnet50 import resnet50
from .edge_detection import EdgeDetectionModule
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),	
        )
        self.branch3 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),	
        )
        self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),	
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),		
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class DualBranchUNetCBAMResnet1(nn.Module):
    def __init__(self, n_channels, n_classes, writer,bilinear=False):
        super(DualBranchUNetCBAMResnet1, self).__init__()
        self.name="DBUCR1"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.writer=writer
        self.num=0
        factor = 2 if bilinear else 1
        self.feature_fusion = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        # 使用 ResNet50 作为特征提取器
        self.resnet = resnet50()

        # UNet 的下采样模块
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # 主体分割分支
        self.up=Up(2048,1024,bilinear)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.outM=OutConv(256,n_classes)
        
        # self.cbam1 = CBAM(128)
        self.MSAA=MSAA(64)
        self.MSAA1 = MSAA(256)
        self.MSAA2 = MSAA(512)
        self.MSAA3 = MSAA(1024 // factor)
        self.MSAA4 = MSAA(2048 // factor)
        
        self.ASPP=ASPP(64,64)
        self.ASPP1 = ASPP(256,256)
        self.ASPP2 = ASPP(512,512)
        self.ASPP3 = ASPP(1024,1024)
        self.ASPP4 = ASPP(2048 ,2048)
        # # CBAM 注意力机制
        self.cbam= CBAM(64)
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024 // factor)
        self.cbam5=CBAM(2048)
        # 边缘检测模块
        self.edge_detector = EdgeDetectionModule()

    def forward(self, x):
        # x0;   [16,64,128,128]
        # x1；  torch.Size([16, 256, 128, 128])
        # x2；  torch.Size([16, 512, 64, 64])
        # x3；  torch.Size([16, 1024, 32, 32])
        # x4；  torch.Size([16, 2048, 16, 16])
        self.num=self.num+1
        # self.printImage(x,"origin",self.num)
        
        #[3,128,128]
        x0=self.inc(x)
        # print("x0: ",x0.shape)
        # x0=self.MSAA(x0)
        x0=self.ASPP(x0)
        #[64,128,128]
        x1=self.resnet.layer1(x0)
        x1=self.MSAA1(x1)
        x1=self.ASPP1(x1)
        # x1=self.cbam2(x1)
        # print("x1； ",x1.shape)
        #[128,64,64]
        x2=self.resnet.layer2(x1)
        x2=self.MSAA2(x2)
        x2=self.ASPP2(x2)
        # x2=self.cbam3(x2)
        
        # print("x2； ",x2.shape)
        
        #[256,32,32]
        x3=self.resnet.layer3(x2)
        x3=self.MSAA3(x3)
        x3=self.ASPP3(x3)
        # x3=self.cbam4(x3)
        
        # print("x3； ",x3.shape)
        
        #[512,16,16]
        x4=self.resnet.layer4(x3)
        # x4=self.MSAA4(x4)
        # x4=self.ASPP4(x4)
        x4=self.cbam5(x4)
        x3=self.up(x4,(x3))
        # x3=self.MSAA3(x3)
        # x3=self.ASPP3(x3)
        x3=self.cbam4(x3)
        x2=self.up1(x3,(x2))
        # x2=self.MSAA2(x2)
        # x2=self.ASPP2(x2)
        x2=self.cbam3(x2)
        x1=self.up2(x2,(x1))
        # x1=self.MSAA1(x1)
        # x1=self.ASPP1(x1)
        x1=self.cbam2(x1)
        # print("x3； ",x3.shape)
        # print("x2； ",x2.shape)
        # print("x1； ",x1.shape)
        
        # print("x: ",x.shape)
        #[1024,8,8]
        logits=self.outM(x1)
        # print("logits: ",logits.shape)
        # 特征融合
        # edge_features = self.edge_detector(logits)
        # print("edge: ",edge_features.shape)
        fused_logits = logits
        # self.printImage(fused_logits,"out",self.num)
        
        return fused_logits
    def use_checkpointing(self):
        # self.resnet = torch.utils.checkpoint(self.resnet)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.edge_up1 = torch.utils.checkpoint(self.edge_up1)
        self.edge_up2 = torch.utils.checkpoint(self.edge_up2)
        self.edge_up3 = torch.utils.checkpoint(self.edge_up3)
        self.edge_up4 = torch.utils.checkpoint(self.edge_up4)
        self.edge_outc = torch.utils.checkpoint(self.edge_outc)
    def printImage(self,inputImage,name,num):
        self.writer.add_image('Train-dual-Res/'+name,inputImage[0,0].unsqueeze(0), num)  # Shape: [1, 128, 128]
        