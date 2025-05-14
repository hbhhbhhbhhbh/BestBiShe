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
from unet.UResnet import UResnet
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
from unet.Deeplab import DeepLab
from unet.SegNet import SegNet
import wandb
from unet.FCN import FCN
import numpy as np
import matplotlib.pyplot as plt
from evaluate import evaluate
from unet.unet_model import UNet
from unet.SegNet import SegNet
from unet.Dulbranch_res_Copy1 import DualBranchUNetCBAMResnet1
from unet import UNet,UNetCBAM,UNetCBAMResnet
from unet.Dulbranch_res import DualBranchUNetCBAMResnet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from torch.utils.tensorboard import SummaryWriter
from utils.distance_transform import one_hot2dist, SurfaceLoss
from utils.Loss import edge_loss

def overlay_two_masks(groundtruth_mask, pred_mask, alpha=0.5, pred_alpha=0.5):
    """
    Overlay groundtruth mask and prediction mask with transparency
    :param groundtruth_mask: [H, W] (0 or 1)
    :param pred_mask: [H, W] (0 or 1)
    :param alpha: groundtruth transparency
    :param pred_alpha: prediction transparency
    :return: overlay image [H, W, 3]
    """
    groundtruth_mask = groundtruth_mask.squeeze(0).cpu().numpy()
    pred_mask = pred_mask.squeeze(0).cpu().numpy()

    overlay_image = np.zeros((groundtruth_mask.shape[0], groundtruth_mask.shape[1], 3))
    overlay_image[groundtruth_mask == 1] = [0, 1, 0]  # Green for groundtruth
    overlay_image = overlay_image * (1 - alpha)

    pred_overlay = np.zeros_like(overlay_image)
    pred_overlay[pred_mask == 1] = [1, 0, 0]  # Red for prediction
    overlay_image = overlay_image + pred_overlay * pred_alpha

    return overlay_image

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay binary mask on original image with transparency
    :param image: [C, H, W]
    :param mask: [H, W] (0 or 1)
    :param alpha: mask transparency
    :return: overlay image [H, W, 3]
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()

    mask_overlay = np.zeros_like(image)
    mask_overlay[mask == 1] = [1, 1, 1]  # White mask
    overlay = image * (1 - alpha) + mask_overlay * alpha

    return overlay

def save_overlay_images(images, true_masks, pred_masks, output_dir, idx):
    """
    Save overlay images to disk
    :param images: batch of images [B, C, H, W]
    :param true_masks: batch of groundtruth masks [B, 1, H, W]
    :param pred_masks: batch of predicted masks [B, 1, H, W]
    :param output_dir: directory to save images
    :param idx: starting index for filenames
    """
    for i in range(images.shape[0]):
        # Create overlays
        mask_overlay = overlay_mask_on_image(images[i], pred_masks[i])
        combined_overlay = overlay_two_masks(true_masks[i], pred_masks[i])
        
        # Convert to 0-255 range
        mask_overlay = (mask_overlay * 255).astype(np.uint8)
        combined_overlay = (combined_overlay * 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, f"mask_overlay_{idx+i}.png"), cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"combined_overlay_{idx+i}.png"), cv2.cvtColor(combined_overlay, cv2.COLOR_RGB2BGR))

def test_model(
        model_path,
        dataset_dir,
        output_dir='./test_results',
        batch_size=4,
        img_scale=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Test the model and save overlay images
    :param model_path: path to trained model checkpoint
    :param dataset_dir: directory containing test images and masks
    :param output_dir: directory to save results
    :param batch_size: batch size for testing
    :param img_scale: image scaling factor
    :param device: device to run on
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    log_dir = f'/root/tf-logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}/dual1'
    writer = SummaryWriter(log_dir=log_dir)
    # Load model
    model =DualBranchUNetCBAMResnet1(n_classes=2, n_channels=3,writer=writer)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    test_img_dir = Path(dataset_dir) / 'imgs/test/'
    test_mask_dir = Path(dataset_dir) / 'masks/test/'
    
    try:
        test_set = CarvanaDataset(test_img_dir, test_mask_dir, img_scale)
    except:
        test_set = BasicDataset(test_img_dir, test_mask_dir, img_scale)
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Run testing
    idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = batch['mask'].to(device, dtype=torch.long)
            
            # Get predictions
            pred_masks = model(images)
            pred_masks = pred_masks.argmax(dim=1).unsqueeze(1)
            
            # Save overlay images
            save_overlay_images(images, true_masks, pred_masks, output_dir, idx)
            idx += images.shape[0]

if __name__ == '__main__':
    # Example usage
    test_model(
        model_path='./checkpoints-dual-data-enhanced/data-enhanced-dual.pth',
        dataset_dir='./data-enhanced',
        output_dir='./test/MUCAR_output',
        batch_size=4
    )