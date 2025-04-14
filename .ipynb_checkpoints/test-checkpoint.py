import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet.UResnet import BasicBlock,UResnet
from utils.data_loading import BasicDataset
from unet import UNet, UNetCBAM, UNetCBAMResnet
from utils.utils import plot_img_and_mask
from unet.Dulbranch_res import DualBranchUNetCBAMResnet
from unet.Dulbranch_res_Copy1 import DualBranchUNetCBAMResnet1
from unet.UResnet import UResnet
def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output= net(img)
        output=output.cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints-dual-data1_resized/checkpoint_epoch50.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', required=True,
                        help='Folder containing input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', required=True,
                        help='Folder to save output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def mask_to_red(mask: np.ndarray):
    """
    Convert mask to a red overlay.
    """
    # Create an empty RGB image
    red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Set the red channel where the mask is 1
    red_mask[mask == 1] = [255, 0, 0]  # Red color
    return red_mask


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Get all image files in the input directory
    in_files = glob(os.path.join(args.input, '*.*'))  # Supports all image formats
    logging.info(f'Found {len(in_files)} images in {args.input}')

    # Load the model
    net= DualBranchUNetCBAMResnet1(n_classes=args.classes, n_channels=3,writer=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)

    del state_dict['mask_values']

    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    # Process each image
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        # Predict the mask
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # Convert mask to red overlay
        red_mask = mask_to_red(mask)

        # Blend the original image with the red mask
        img_array = np.array(img)
        blended = Image.fromarray(np.where(red_mask != 0, red_mask, img_array))

        # Save the result
        if not args.no_save:
            out_filename = os.path.join(args.output, os.path.basename(filename))
            blended.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        # Visualize the result
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)