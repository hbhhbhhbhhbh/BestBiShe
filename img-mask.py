from PIL import Image
from pathlib import Path
import os

# 目标尺寸
target_size = (128, 128)

# 数据集文件夹结构
data_folders = {
    'train': {'imgs': './data2/imgs/train/', 'masks': './data2/masks/train/'},
    'val': {'imgs': './data2/imgs/val/', 'masks': './data2/masks/val/'},
    'test': {'imgs': './data2/imgs/test/', 'masks': './data2/masks/test/'}
}

# 输出文件夹结构
output_folders = {
    'train': {'imgs': './data2_resized/imgs/train/', 'masks': './data2_resized/masks/train/'},
    'val': {'imgs': './data2_resized/imgs/val/', 'masks': './data2_resized/masks/val/'},
    'test': {'imgs': './data2_resized/imgs/test/', 'masks': './data2_resized/masks/test/'}
}

# 调整图像和掩码尺寸
def resize_images_and_masks(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, target_size):
    # 获取所有图像文件（过滤掉目录和非图像文件）
    img_files = sorted([f for f in input_img_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.png', '.jpeg']])
    # 获取所有掩码文件（过滤掉目录和非图像文件）
    mask_files = sorted([f for f in input_mask_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.bmp', '.jpg', '.png', '.jpeg']])

    # 确保图像和掩码文件数量一致
    if len(img_files) != len(mask_files):
        print(f"Warning: Number of images ({len(img_files)}) and masks ({len(mask_files)}) do not match in {input_img_dir}!")
        return

    # 遍历每对图像和掩码
    for img_path, mask_path in zip(img_files, mask_files):
        # 打开图像和掩码
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # 调整尺寸
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        mask_resized = mask.resize(target_size, Image.Resampling.NEAREST)

        # 保存调整后的图像和掩码
        img_output_path = os.path.join(output_img_dir, img_path.name)
        mask_output_path = os.path.join(output_mask_dir, mask_path.name)

        img_resized.save(img_output_path)
        mask_resized.save(mask_output_path)

        print(f"Resized and saved: {img_output_path}, {mask_output_path}")

# 处理所有数据集
for dataset_type, folders in data_folders.items():
    input_img_dir = Path(folders['imgs'])
    input_mask_dir = Path(folders['masks'])
    output_img_dir = Path(output_folders[dataset_type]['imgs'])
    output_mask_dir = Path(output_folders[dataset_type]['masks'])

    # 创建输出文件夹
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 调整图像和掩码尺寸
    print(f"Processing {dataset_type} dataset...")
    resize_images_and_masks(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, target_size)
    print(f"Finished processing {dataset_type} dataset.\n")