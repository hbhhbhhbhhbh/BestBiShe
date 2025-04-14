# train-dualbranch.py
import cv2
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import wandb
import numpy as np
import matplotlib.pyplot as plt
from evaluate import evaluate
from unet.unet_model import UNet
from unet.Dulbranch_res import DualBranchUNetCBAMResnet
from unet.Dulbranch_res_Copy1 import DualBranchUNetCBAMResnet1

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from torch.utils.tensorboard import SummaryWriter
from utils.distance_transform import one_hot2dist, SurfaceLoss
# 1. 设置设备和参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = False  # 启用混合精度

# 2. 加载测试数据集
dir_img_test = Path(f'./data1_resized/imgs/test/')  # 测试集图像路径
dir_mask_test = Path(f'./data1_resized/masks/test/')  # 测试集掩码路径
img_scale = 0.5  # 图像缩放比例

# 创建测试数据集
try:
    test_dataset = CarvanaDataset(dir_img_test, dir_mask_test, img_scale)
except (AssertionError, RuntimeError, IndexError):
    test_dataset = BasicDataset(dir_img_test, dir_mask_test, img_scale)

# 创建测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# 3. 加载训练好的模型
model = DualBranchUNetCBAMResnet1(n_channels=3, n_classes=2,writer=None)  # 实例化模型，替换为您的模型类

state_dict = torch.load("checkpoints-dual-data1_resized/checkpoint_epoch50.pth")
del state_dict['mask_values']
model.load_state_dict(state_dict)
model = model.to(device)

# 4. 运行测试
dice_score, pixel_accuracy, iou, f1, recall, precision = evaluate(
    net=model,
    dataloader=test_loader,
    device=device,
    amp=amp
)

# 5. 打印测试结果
print(f"Test Dice Score: {dice_score}")
print(f"Test Pixel Accuracy: {pixel_accuracy}")
print(f"Test IoU: {iou}")
print(f"Test F1 Score: {f1}")
print(f"Test Recall: {recall}")
print(f"Test Precision: {precision}")