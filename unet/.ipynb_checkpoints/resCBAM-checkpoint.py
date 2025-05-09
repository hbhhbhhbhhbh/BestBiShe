import torch
import torch.nn as nn
from unet.CBAM import CBAM
from unet.resnet import resnet50
from unet.unet_parts import Up, OutConv
class unetUpWithCBAM(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUpWithCBAM, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
#         self.conv1  = DOConv2d(in_size, out_size, kernel_size = 3, padding = 1)
#         self.conv2  = DOConv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_size)  # 添加 CBAM 注意力机制

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # print("outputs: ",outputs.shape)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.cbam(outputs)  # 使用 CBAM 增强特征
        return outputs
class UnetWithCBAM(nn.Module):
    def __init__(self, n_classes=2, pretrained=False, backbone='vgg',n_channels=3):
        super(UnetWithCBAM, self).__init__()
        self.n_classes = n_classes
        self.resnet = resnet50(pretrained=False)
        in_filters = [192, 384, 768, 1536]
        self.name="UCR"
        out_filters = [64, 128, 256, 512]

        # 替换为带 CBAM 的上采样模块
        self.up_concat4 = unetUpWithCBAM(in_filters[3], out_filters[3])
        self.up_concat3 = unetUpWithCBAM(in_filters[2], out_filters[2])
        self.up_concat2 = unetUpWithCBAM(in_filters[1], out_filters[1])
        self.up_concat1 = unetUpWithCBAM(in_filters[0], out_filters[0])

        
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2), 
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
#                 DOConv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),

            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
#                 DOConv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
        )
        

        self.final = nn.Conv2d(out_filters[0], n_classes, kernel_size=3, stride=2, padding=1)

#         self.final = DOConv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        # print("inputs: ",inputs.shape)
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        # print("out: ",final.shape)
        return final