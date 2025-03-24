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
from unet.Dulbranch_res_Copy1 import DualBranchUNetCBAMResnet1

from unet.Dulbranch_res import DualBranchUNetCBAMResnet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from torch.utils.tensorboard import SummaryWriter
from utils.distance_transform import one_hot2dist, SurfaceLoss
def process_images(images):
    """
    处理图像：RGB → HSV → 处理 V 分量 → 转换回 RGB
    :param images: 输入图像，形状为 [batch_size, channels, height, width]
    :return: 处理后的图像，形状为 [batch_size, channels, height, width]
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    images_np = images.cpu().numpy()
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # 调整为 [batch_size, height, width, channels]
    images_np = (images_np * 255).astype(np.uint8)  # 将数据范围从 [0, 1] 转换为 [0, 255]

    # 对每张图像进行处理
    processed_images = []
    for img in images_np:
        # RGB → HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 分解 HSV 图像
        h, s, v = cv2.split(hsv_image)
        
        # 对 V 分量进行自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_processed = clahe.apply(v)
        
        # 融合处理后的 V 分量
        hsv_processed = cv2.merge([h, s, v_processed])
        
        # HSV → RGB
        rgb_processed = cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2RGB)
        processed_images.append(rgb_processed)

    # 将 NumPy 数组转换回 PyTorch 张量
    processed_images = np.stack(processed_images)  # 形状为 [batch_size, height, width, channels]
    processed_images = torch.from_numpy(processed_images).permute(0, 3, 1, 2).float() / 255.0  # 调整为 [batch_size, channels, height, width]
    return processed_images.to(images.device)  # 移动到 GPU
# train-dualbranch.py
class CombinedLoss(nn.Module):
    def __init__(self, idc, surface_loss_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.idc = idc
        self.surface_loss_weight = surface_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.surface_loss = SurfaceLoss(idc=idc)

    def forward(self, logits, edge_logits, targets, dist_maps,writer,global_step):
        ce_loss = self.ce_loss(logits, targets)
        
        # print("dist_maps: ",dist_maps)
        edge_logits = torch.sigmoid(edge_logits)
        # print("edge: ",edge_logits)
        surface_loss = self.surface_loss(edge_logits, dist_maps)
        # print("ce_loss :",ce_loss)
        # print("surface_loss: ",surface_loss)
        writer.add_scalar('surface_loss/surface_loss', surface_loss, global_step)
        total_loss = ce_loss +surface_loss*self.surface_loss_weight
        writer.add_scalar('surface_loss/surface_loss_weight', self.surface_loss_weight, global_step)
        return total_loss
dir_img = Path('./data-after/imgs/train/')
dir_mask = Path('./data-after/masks/train/')
dir_checkpoint = Path('./checkpoints-dual-update/')

def overlay_two_masks(groundtruth_mask, pred_mask, alpha=0.5, pred_alpha=0.5):
    """
    将 groundtruth mask 和 prediction mask 叠加在一起，每个 mask 保持透明度。
    :param groundtruth_mask: 真实标签的 mask [H, W] (0 或 1)
    :param pred_mask: 预测的 mask [H, W] (0 或 1)
    :param alpha: groundtruth mask 的透明度
    :param pred_alpha: prediction mask 的透明度
    :return: 合成的图像
    """
    # 将 groundtruth_mask 和 pred_mask 转为 [H, W] 格式（确保是二进制）
    groundtruth_mask = groundtruth_mask.squeeze(0).cpu().numpy()
    pred_mask = pred_mask.squeeze(0).cpu().numpy()

    # 创建一个全黑的背景
    overlay_image = np.zeros((groundtruth_mask.shape[0], groundtruth_mask.shape[1], 3))

    # 将 groundtruth mask 叠加为绿色 (使用 alpha 混合透明度)
    overlay_image[groundtruth_mask == 1] = [0, 1, 0]  # 绿色
    overlay_image = overlay_image * (1 - alpha)  # 调整透明度

    # 将 prediction mask 叠加为红色 (使用 pred_alpha 混合透明度)
    pred_overlay = np.zeros_like(overlay_image)  # 创建一个空白图层
    pred_overlay[pred_mask == 1] = [1, 0, 0]  # 红色
    overlay_image = overlay_image + pred_overlay * pred_alpha  # 混合透明度

    return overlay_image

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    将二进制 mask（0 或 1）叠加到原图上，并调整透明度。
    :param image: 原始图像 [C, H, W] (C: 通道数, H: 高度, W: 宽度)
    :param mask: 预测的二进制 mask [H, W] (0 或 1)
    :param alpha: mask 的透明度
    :return: 合成的图像
    """
    # 将图像转换为 [H, W, C] 格式（从 [C, H, W] 转换）
    image = image.permute(1, 2, 0).cpu().numpy()

    # 将 mask 转为 [H, W] 格式（确保是二进制）
    mask = mask.squeeze(0).cpu().numpy()  # 假设 mask 是 [1, H, W]，去掉通道维度

    # 创建一个白色的 mask 图像
    mask_overlay = np.zeros_like(image)  # 初始化为黑色
    mask_overlay[mask == 1] = [1, 1, 1]  # 只在 mask == 1 的地方设置为白色

    # 将原始图像和 mask 合成
    overlay = image * (1 - alpha) + mask_overlay * alpha  # 调整透明度，创建合成图像

    return overlay

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Initialize TensorBoard writer
    # log_dir = f'/root/tf-logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}/{model.name}'
    # writer = SummaryWriter(log_dir=log_dir)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = CombinedLoss(idc=[1], surface_loss_weight=1)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 生成距离图
                dist_maps_single = torch.tensor(one_hot2dist(true_masks.cpu().numpy()), device=device)
                dist_maps = torch.cat([dist_maps_single, dist_maps_single], dim=1)  # 复制第一个通道创建第二个通道

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 处理输入图像
                    p_images = process_images(images)

                    # 模型推理
                    masks_pred= model(images)

                    # 计算损失
                    loss = criterion(masks_pred, true_masks)  # 主分割任务损失

                    # 添加 Dice 损失
                    loss = loss + dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks.squeeze(1).to(torch.long), model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                    # print("masks_pred: ",masks_pred.shape)
                    # total_loss = loss+dice_loss(
                    #         F.softmax(masks_pred, dim=1).float(),
                    #         F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    #         multiclass=True
                    #     )
                    # print("loss: ",total_loss)
                    total_loss=loss
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(total_loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += total_loss.item()
                writer.add_scalar('Loss/Train-dual-res', total_loss.item(), global_step)
                writer.add_scalar('Learning Rate-dual-res', optimizer.param_groups[0]['lr'], global_step)
                pbar.set_postfix(**{'loss (batch)': total_loss.item()})
                # if epoch >epoch*0.8:
                #     criterion.surface_loss_weight=1.5
                if global_step % 50 == 0:  # Log every 100 steps
                    # writer.add_image('Train/Image', images[0], global_step)
                    # writer.add_image('Train/Mask', true_masks[0].unsqueeze(0), global_step)  # Add channel dimension
                    # writer.add_image('Train/Prediction', masks_pred.argmax(dim=1)[0], global_step)
                   # Log images, true masks, and predictions to TensorBoard
                    overlay_image = overlay_two_masks(
                        true_masks[0],  # Ground truth mask
                        masks_pred.argmax(dim=1)[0],  # Prediction mask
                        alpha=0.5,  # Ground truth mask 的透明度
                        pred_alpha=0.7  # Prediction mask 的透明度
                    )

                    # 将合成图像记录到 TensorBoard
                    writer.add_image('Train-dual-Res/MaskOverlay', torch.tensor(overlay_image).permute(2, 0, 1), global_step)
                    overlay_image = overlay_mask_on_image(images[0], masks_pred.argmax(dim=1)[0])

                    # 将合成图像记录到 TensorBoard
                    writer.add_image('Train-dual-Res/Overlay', torch.tensor(overlay_image).permute(2, 0, 1), global_step)
                    writer.add_image('Train-dual-Res/Image', images[0], global_step)  # Shape: [3, 128, 128]
                    # writer.add_image('Train-dual-Res/p_Image', p_images[0], global_step)  # Shape: [3, 128, 128]
                    
                    writer.add_image('Train-dual-Res/Mask', true_masks[0].unsqueeze(0), global_step)  # Shape: [1, 128, 128]
                    writer.add_image('Train-dual-Res/Prediction', masks_pred.argmax(dim=1)[0].unsqueeze(0), global_step)  # Shape: [1, 128, 128]
                    

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        
                        #avg_dice_score, avg_pixel_accuracy, avg_iou, avg_f1, avg_recall, avg_precision
                        val_score, acc, iou,f1,recall,precision = evaluate(model, val_loader, device, amp)
                        writer.add_scalar('Dice/Val-dual-res', val_score, global_step)
                        writer.add_scalar('Accuracy/Val-dual-res', acc, global_step)
                        writer.add_scalar('IoU/Val-dual-res', iou, global_step)
                        writer.add_scalar('f1/Val-dual-res', f1, global_step)
                        writer.add_scalar('recall/Val-dual-res', recall, global_step)
                        writer.add_scalar('precision/Val-dual-res', precision, global_step)
                        scheduler.step(val_score)
                        # if scheduler.num_bad_epochs > 3:  # 如果学习率调度器触发了 bad epochs
                        #     criterion.surface_loss_weight = min(2.0, criterion.surface_loss_weight + 0.2)
                        # logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'acc': acc,
                                'iou': iou,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        # Save checkpoints
        if save_checkpoint:
            if epoch==epochs:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    # Close TensorBoard writer
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # Initialize TensorBoard writer
    log_dir = f'/root/tf-logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    writer = SummaryWriter(log_dir=log_dir)
    model = DualBranchUNetCBAMResnet1(n_classes=args.classes, n_channels=3, writer=writer)
    model = model.to(memory_format=torch.channels_last)
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )