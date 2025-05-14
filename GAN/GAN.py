import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from utils.data_loading import BasicDataset, CarvanaDataset

# 配置参数
batch_size = 8
latent_dim = 256
img_size = 256
num_epochs = 100
lr = 0.0002
save_dir = "./gan_generated_pairs"
os.makedirs(save_dir, exist_ok=True)

# 改进的生成器（同时生成图像和掩码）
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2)
        )
        
        # 图像解码器
        self.img_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )
        
        # 掩码解码器
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z):
        # 将噪声向量转换为特征图
        z = z.view(z.size(0), -1, 1, 1)
        z = z.repeat(1, 1, 16, 16)  # 扩展到初始特征尺寸
        
        features = self.encoder(z)
        img = self.img_decoder(features)
        mask = self.mask_decoder(features)
        return img, mask

# 判别器（同时判别图像和掩码）
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )
        
    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.model(x)

# 可视化掩码并保存结果
def save_visual_results(images, masks, epoch, n_samples=4):
    os.makedirs(f"{save_dir}/epoch_{epoch}", exist_ok=True)
    
    for i in range(n_samples):
        # 处理图像
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img + 1) * 127.5  # [-1,1] -> [0,255]
        img = img.astype(np.uint8)
        
        # 处理掩码（可视化）
        mask = masks[i].squeeze().cpu().numpy()
        mask_viz = (mask * 255).astype(np.uint8)
        colored_mask = cv2.applyColorMap(mask_viz, cv2.COLORMAP_JET)
        
        # 合成可视化图像
        overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
        
        # 保存结果
        cv2.imwrite(f"{save_dir}/epoch_{epoch}/sample_{i}_img.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{save_dir}/epoch_{epoch}/sample_{i}_mask.png", mask_viz)
        cv2.imwrite(f"{save_dir}/epoch_{epoch}/sample_{i}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# 训练函数
def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 数据加载
    dataset_name = 'data-enhanced'
    dir_img = Path(f'./{dataset_name}/imgs/train/')
    dir_mask = Path(f'./{dataset_name}/masks/train/')
    
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale=1.0)
    except:
        dataset = BasicDataset(dir_img, dir_mask, img_scale=1.0)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            real_imgs = batch['image'].to(device)
            real_masks = batch['mask'].unsqueeze(1).float().to(device)
            
            # 真实和假标签
            valid = torch.ones(real_imgs.size(0), 1, 1, 1).to(device)
            fake = torch.zeros(real_imgs.size(0), 1, 1, 1).to(device)
            
            # 训练判别器
            optimizer_D.zero_grad()
            
            # 真实样本损失
            real_loss = criterion(discriminator(real_imgs, real_masks), valid)
            
            # 生成假样本
            z = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs, fake_masks = generator(z)
            
            # 假样本损失
            fake_loss = criterion(discriminator(fake_imgs.detach(), fake_masks.detach()), fake)
            
            # 总判别器损失
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # 训练生成器
            optimizer_G.zero_grad()
            
            # 对抗损失
            g_loss_adv = criterion(discriminator(fake_imgs, fake_masks), valid)
            
            # 添加L1损失使生成更稳定
            g_loss_l1 = nn.L1Loss()(fake_masks, real_masks) * 10
            
            # 总生成器损失
            g_loss = g_loss_adv + g_loss_l1
            g_loss.backward()
            optimizer_G.step()
        
        # 每个epoch保存结果
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                test_z = torch.randn(4, latent_dim).to(device)
                gen_imgs, gen_masks = generator(test_z)
                save_visual_results(gen_imgs, gen_masks, epoch+1)
            
            # 保存模型
            torch.save(generator.state_dict(), f"{save_dir}/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_gan()