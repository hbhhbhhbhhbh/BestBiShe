import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import torch.nn as nn
class edge_loss(nn.Module):
    # 使用 Sobel 滤波器提取边缘（对预测和标签）
    def __init__(self):
        super(edge_loss, self).__init__()
    def sobel_edge_map(self,tensor):
        # 假设输入为 [B, 1, H, W]，单通道灰度图（可对多通道做平均）
        # if tensor.dtype != torch.float32:
        #     tensor = tensor.float()  # 确保输入是 float
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)

        edge_x = F.conv2d(tensor, sobel_x, padding=1)
        edge_y = F.conv2d(tensor, sobel_y, padding=1)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge
    def forward(self,pred, target, alpha=0.8):
        """
        pred: 模型输出 (B, 1, H, W) - Sigmoid 激活后
        target: Ground truth 标签 (B, 1, H, W)
        alpha: 原始分割 loss 与边缘 loss 的权重因子
        """
        bce_loss = nn.CrossEntropyLoss()(pred, target)
        
        target = target.float().unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        # 二分类任务：pred 是 [B, 2, H, W]，取前景概率（sigmoid 或 softmax）
        if pred.shape[1] == 2:
            pred_prob = torch.softmax(pred, dim=1)[:, 1:2, :, :]  # 取类别 1 的概率 [B, 1, H, W]
        else:
            pred_prob = torch.sigmoid(pred)  # 如果 pred 是 [B, 1, H, W]，直接 sigmoid
        # 常规分割损失
        pred_prob = pred_prob.float()  # [B, 1, H, W]
        print("pred ",pred_prob.shape)
        print("target ",target.shape)
        # 边缘提取
        pred_edge = self.sobel_edge_map(pred_prob)
        target_edge = self.sobel_edge_map(target)

        # 边缘损失
        edge_loss = F.l1_loss(pred_edge, target_edge)

        # 总损失
        total_loss = alpha * bce_loss + (1 - alpha) * edge_loss
        return total_loss