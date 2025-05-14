import numpy as np
import torch
from torch import nn
from torch.nn import init
 #我是让科研变的更简单的叫叫兽！国奖，多篇SCI，深耕目标检测领域，多项竞赛经历，拥有软件著作权，核心期刊等成果。实战派up主，只做干货！让你不走弯路，直冲成果输出！！

# 大家关注我的B站：Ai学术叫叫兽
# 链接在这：https://space.bilibili.com/3546623938398505
# 科研不痛苦，跟着叫兽走！！！
# 更多捷径——B站干货见！！！

# 本环境和资料纯粉丝福利！！！
# 必须让我叫叫兽的粉丝有牌面！！！冲吧，青年们，遥遥领先！！！
 
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        #self.act=nn.SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()    
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)    
        self.act = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out    



class CBAM(nn.Module):
    def __init__(self, c1):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    #详细改进流程和操作，请关注B站博主：AI学术叫叫兽 
    