from PIL import Image
import os
import random
import numpy as np
import shutil

# 设置随机种子以确保可重复性
random.seed(42)

# 输入文件夹路径
input_dir = "Dataset2"  # 包含 img 和 mask 的文件夹
output_dir = "data2"  # 输出文件夹

# 创建输出文件夹结构
os.makedirs(os.path.join(output_dir, "img", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "img", "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "img", "test"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "mask", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "mask", "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "mask", "test"), exist_ok=True)

# 将 img 从 .bmp 转换为 .jpg
def convert_img_to_jpg(img_path, output_path):
    img = Image.open(img_path)
    img = img.convert("RGB")  # 确保图像是 RGB 格式
    img.save(output_path, "JPEG")

# 将 mask 转换为二值图像（黑色为 0，其他为 1）
def convert_mask_to_binary(mask_path, output_path):
    # 打开 mask 图像并转换为灰度模式
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)

    # 将黑色像素（值为 0）设为 0，其他像素设为 1
    binary_mask_array = np.where(mask_array == 0, 0, 1)

    # 将二值数组转换为图像并保存为 PNG
    binary_mask = Image.fromarray(binary_mask_array.astype(np.uint8))  # 直接保存为 0 和 1
    binary_mask.save(output_path)

# 获取所有 img 和 mask 文件
img_files = [f for f in os.listdir(input_dir) if f.endswith(".bmp")]
mask_files = [f.replace(".bmp", ".png") for f in img_files]  # 假设 mask 文件是 .png 格式

# 检查 img 和 mask 文件是否一一对应
for img, mask in zip(img_files, mask_files):
    if mask not in os.listdir(input_dir):
        raise FileNotFoundError(f"Mask file {mask} not found for {img}")

# 随机打乱文件列表
combined = list(zip(img_files, mask_files))
random.shuffle(combined)
img_files, mask_files = zip(*combined)

# 划分数据集
total = len(img_files)
train_size = int(0.7 * total)
val_size = int(0.2 * total)
test_size = total - train_size - val_size

train_img = img_files[:train_size]
train_mask = mask_files[:train_size]

val_img = img_files[train_size:train_size + val_size]
val_mask = mask_files[train_size:train_size + val_size]

test_img = img_files[train_size + val_size:]
test_mask = mask_files[train_size + val_size:]

# 处理并保存文件
def process_and_save_files(img_files, mask_files, img_output_dir, mask_output_dir):
    for img_file, mask_file in zip(img_files, mask_files):
        # 处理 img
        img_path = os.path.join(input_dir, img_file)
        img_output_path = os.path.join(img_output_dir, img_file.replace(".bmp", ".jpg"))
        convert_img_to_jpg(img_path, img_output_path)

        # 处理 mask
        mask_path = os.path.join(input_dir, mask_file)
        mask_output_path = os.path.join(mask_output_dir, mask_file)
        convert_mask_to_binary(mask_path, mask_output_path)

# 处理并保存 train 数据
process_and_save_files(train_img, train_mask, os.path.join(output_dir, "img", "train"), os.path.join(output_dir, "mask", "train"))

# 处理并保存 val 数据
process_and_save_files(val_img, val_mask, os.path.join(output_dir, "img", "val"), os.path.join(output_dir, "mask", "val"))

# 处理并保存 test 数据
process_and_save_files(test_img, test_mask, os.path.join(output_dir, "img", "test"), os.path.join(output_dir, "mask", "test"))

print("数据集处理完成！")