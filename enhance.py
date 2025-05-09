import os
import cv2
import numpy as np
import shutil
from skimage import exposure, filters
from skimage.util import img_as_ubyte
from tqdm import tqdm

class CellImageEnhancer:
    def __init__(self):
        self.gamma = 1.2
        self.clip_limit = 3.0
        self.tile_size = (8, 8)
        
    def apply_clahe(self, image):
        """对比度受限的自适应直方图均衡化"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)
            return clahe.apply(image)
    def hsv_color_normalization(self,img):
        """针对H&E染色图像的专用颜色归一化"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = cv2.normalize(hsv[:, :, 0], None, 0, 180, cv2.NORM_MINMAX)  # 色调归一化
        hsv[:, :, 1] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(hsv[:, :, 1])  # 饱和度增强
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def gamma_correction(self, image):
        """伽马校正"""
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def adaptive_denoising(self, image):
        """自适应去噪"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def edge_enhancement(self, image):
        """边缘增强（修复了值范围问题）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = filters.sobel(gray)
        
        # 修复：确保值范围在[-1,1]之间
        edges = exposure.rescale_intensity(edges, in_range='image', out_range=(-1, 1))
        edges = img_as_ubyte(edges)  # 现在可以安全转换为ubyte
        
        # 确保边缘图像是单通道的
        if len(image.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges

    def enhance_image(self, image):
        """完整的增强流程"""
        # enhanced = self.gamma_correction(image)
        enhanced = self.apply_clahe(image)
        # enhanced = self.adaptive_denoising(enhanced)
        # enhanced=image
        edge_enhanced = self.hsv_color_normalization(enhanced)
        edge_enhanced=self.edge_enhancement(edge_enhanced)
        # 修复：确保两个图像尺寸和通道数匹配
        if len(enhanced.shape) != len(edge_enhanced.shape):
            if len(enhanced.shape) == 3:
                edge_enhanced = cv2.cvtColor(edge_enhanced, cv2.COLOR_GRAY2BGR)
            else:
                edge_enhanced = cv2.cvtColor(edge_enhanced, cv2.COLOR_BGR2GRAY)
        
        final = cv2.addWeighted(enhanced, 0.7, edge_enhanced, 0.3, 0)
        return final

def process_dataset(input_root, output_root):
    """处理整个数据集"""
    enhancer = CellImageEnhancer()
    
    # 创建输出目录结构
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'masks'), exist_ok=True)
    
    # 处理所有图像子集
    for subset in ['train', 'val', 'test']:
        print(f"\nProcessing {subset} images...")
        img_dir = os.path.join(input_root, 'imgs', subset)
        output_img_dir = os.path.join(output_root, 'imgs', subset)
        os.makedirs(output_img_dir, exist_ok=True)
        
        # 复制对应的mask目录结构
        mask_src = os.path.join(input_root, 'masks', subset)
        mask_dst = os.path.join(output_root, 'masks', subset)
        if os.path.exists(mask_src):
            if not os.path.exists(mask_dst):
                shutil.copytree(mask_src, mask_dst)
        
        # 处理图像
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        for img_file in tqdm(img_files, desc=f"Enhancing {subset} images"):
            img_path = os.path.join(img_dir, img_file)
            output_path = os.path.join(output_img_dir, img_file)
            
            try:
                # 读取并处理图像
                img = cv2.imread(img_path)
                if img is not None:
                    enhanced_img = enhancer.enhance_image(img)
                    cv2.imwrite(output_path, enhanced_img)
                else:
                    print(f"\nWarning: Could not read image {img_path}")
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue

if __name__ == "__main__":
    input_dir = "data-pre"
    output_dir = "data-enhanced4"
    
    # 处理整个数据集
    process_dataset(input_dir, output_dir)
    print(f"\nAll images processed and saved to {output_dir}")