import torch
import torch.nn as nn

# 定义3x3卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 定义1x1卷积
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 调整 conv1 的 stride 和 padding
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 调整 maxpool 的 stride
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)  # [5, 64, 128, 128]

        x = self.maxpool(feat1)
        feat2 = self.layer1(x)  # [5, 256, 64, 64]

        feat3 = self.layer2(feat2)  # [5, 512, 32, 32]
        feat4 = self.layer3(feat3)  # [5, 1024, 16, 16]
        feat5 = self.layer4(feat4)  # [5, 2048, 8, 8]

        # 调整通道数以匹配目标尺寸
        feat2 = conv1x1(256, 128).to(x.device)(feat2)  # [5, 128, 64, 64]
        feat3 = conv1x1(512, 256).to(x.device)(feat3)  # [5, 256, 32, 32]
        feat4 = conv1x1(1024, 512).to(x.device)(feat4)  # [5, 512, 16, 16]
        feat5 = conv1x1(2048, 1024).to(x.device)(feat5)  # [5, 1024, 8, 8]
        print("feat1: ",feat1.shape)
        print("feat2: ",feat2.shape)
        print("feat3: ",feat3.shape)
        print("feat4: ",feat4.shape)
        print("feat5: ",feat5.shape)
        
        return [feat1, feat2, feat3, feat5]

# 定义ResNet50
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # 加载预训练权重
        load_partial_weights(model, '/root/autodl-tmp/BiShe/resnet50-19c8e357.pth')
    return model

# 加载权重部分
def load_partial_weights(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    # 过滤掉第一层卷积和全连接层的权重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['conv1.weight', 'conv1.bias', 'fc.weight', 'fc.bias']}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

