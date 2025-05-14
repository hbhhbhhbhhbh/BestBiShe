import os
from PIL import Image

def resize_masks(root_dir, target_size=(256, 256)):
    """
    调整masks文件夹下所有PNG图片到指定尺寸
    
    参数:
        root_dir: 数据集根目录路径
        target_size: 目标尺寸 (width, height)
    """
    masks_dir = os.path.join(root_dir, "masks")
    
    for subset in ['train', 'val', 'test']:
        subset_dir = os.path.join(masks_dir, subset)
        
        if not os.path.exists(subset_dir):
            print(f"警告: {subset_dir} 不存在，跳过")
            continue
            
        print(f"正在处理: {subset_dir}")
        
        for filename in os.listdir(subset_dir):
            if filename.lower().endswith('.png'):
                filepath = os.path.join(subset_dir, filename)
                
                try:
                    # 打开图片
                    with Image.open(filepath) as img:
                        # 调整尺寸 (使用LANCZOS高质量下采样)
                        img_resized = img.resize(target_size, Image.LANCZOS)
                        
                        # 保存覆盖原文件
                        img_resized.save(filepath)
                        print(f"已调整: {filename} -> {target_size}")
                        
                except Exception as e:
                    print(f"处理 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 使用示例 - 替换为你的数据集根目录路径
    dataset_root = "data-GANenhanced"  # 替换为实际路径
    resize_masks(dataset_root, target_size=(256, 256))
    
    print("所有mask图片已调整完毕！")