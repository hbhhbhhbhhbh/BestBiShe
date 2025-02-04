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

import wandb
from evaluate import evaluate
from unet import UNet,UNetCBAM,UNetCBAMResnet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from torch.utils.tensorboard import SummaryWriter

dir_img = Path('./data-pre/imgs/train/')
dir_mask = Path('./data-pre/masks/train/')
dir_checkpoint = Path('./checkpoints-res-pre/')
import matplotlib.pyplot as plt

import torch
import numpy as np
import time
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
    log_dir = f'/root/tf-logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('Loss/Train-CBAM-res', loss.item(), global_step)
                writer.add_scalar('Learning Rate-CBAM-res', optimizer.param_groups[0]['lr'], global_step)
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Log to TensorBoard
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
                    writer.add_image('Train-CBAM-Res/MaskOverlay', torch.tensor(overlay_image).permute(2, 0, 1), global_step)
                    overlay_image = overlay_mask_on_image(images[0], masks_pred.argmax(dim=1)[0])

                    # 将合成图像记录到 TensorBoard
                    writer.add_image('Train-CBAM-Res/Overlay', torch.tensor(overlay_image).permute(2, 0, 1), global_step)
                    writer.add_image('Train-CBAM-Res/Image', images[0], global_step)  # Shape: [3, 128, 128]
                    writer.add_image('Train-CBAM-Res/Mask', true_masks[0].unsqueeze(0), global_step)  # Shape: [1, 128, 128]
                    writer.add_image('Train-CBAM-Res/Prediction', masks_pred.argmax(dim=1)[0].unsqueeze(0), global_step)  # Shape: [1, 128, 128]

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, acc, iou = evaluate(model, val_loader, device, amp)
                        writer.add_scalar('Dice/Val-CBAM-res', val_score, global_step)
                        writer.add_scalar('Accuracy/Val-CBAM-res', acc, global_step)
                        writer.add_scalar('IoU/Val-CBAM-res', iou, global_step)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
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
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
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

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNetCBAMResnet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

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
#我是让科研变的更简单的叫叫兽！国奖，多篇SCI，深耕目标检测领域，多项竞赛经历，拥有软件著作权，核心期刊等成果。实战派up主，只做干货！让你不走弯路，直冲成果输出！！

# 大家关注我的B站：Ai学术叫叫兽
# 链接在这：https://space.bilibili.com/3546623938398505
# 科研不痛苦，跟着叫兽走！！！
# 更多捷径——B站干货见！！！

# 本环境和资料纯粉丝福利！！！
# 必须让我叫叫兽的粉丝有牌面！！！冲吧，青年们，遥遥领先！！！