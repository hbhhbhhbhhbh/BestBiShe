import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.data_loading import BasicDataset, CarvanaDataset

# 参数设置
batch_size = 32
latent_dim = 256
image_size = 128
num_epochs = 100
lr = 0.0002
output_dir = "./generated_pairs"

# 创建输出目录
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks", exist_ok=True)

# 数据转换 - 确保输出为float32
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # 自动转换为float32并归一化到[0,1]
])

# 加载数据集
dataset_name = 'data-enhanced'
dir_img = Path(f'./{dataset_name}/imgs/train/')
dir_mask = Path(f'./{dataset_name}/masks/train/')

try:
    dataset = CarvanaDataset(dir_img, dir_mask, 0.5)
except:
    dataset = BasicDataset(dir_img, dir_mask, 0.5)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 1. 双分支生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # 共享的编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        
        # 图像生成分支 (输出3通道)
        self.img_decoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * image_size * image_size),
            nn.Tanh()
        )
        
        # 掩码生成分支 (输出单通道)
        self.mask_decoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, image_size * image_size),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        features = self.encoder(z)
        img = self.img_decoder(features).view(-1, 3, image_size, image_size)
        mask = self.mask_decoder(features).view(-1, 1, image_size, image_size)
        return img, mask

# 2. 判别器
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 确保输入是4D张量且为float32
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        x = x.float()  # 确保输入为float32
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator_img = Discriminator(input_channels=3).to(device)  # 处理RGB图像
discriminator_mask = Discriminator(input_channels=1).to(device)  # 处理掩码

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_img = optim.Adam(discriminator_img.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_mask = optim.Adam(discriminator_mask.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练循环
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # 准备真实数据并确保类型正确
        real_imgs = batch['image'].to(device).float()  # [batch, 3, 128, 128]
        real_masks = batch['mask'].to(device).float().unsqueeze(1)  # [batch, 1, 128, 128]
        
        batch_size = real_imgs.size(0)
        
        # 真实和假标签
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # ===================== 训练判别器 =====================
        # 训练图像判别器
        optimizer_D_img.zero_grad()
        real_img_output = discriminator_img(real_imgs)
        loss_real_img = criterion(real_img_output, real_labels)
        
        # 训练掩码判别器
        optimizer_D_mask.zero_grad()
        real_mask_output = discriminator_mask(real_masks)
        loss_real_mask = criterion(real_mask_output, real_labels)
        
        # 生成假数据
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs, fake_masks = generator(z)
        
        # 假图像损失
        fake_img_output = discriminator_img(fake_imgs.detach())
        loss_fake_img = criterion(fake_img_output, fake_labels)
        
        # 假掩码损失
        fake_mask_output = discriminator_mask(fake_masks.detach())
        loss_fake_mask = criterion(fake_mask_output, fake_labels)
        
        # 总判别器损失
        loss_D_img = (loss_real_img + loss_fake_img) / 2
        loss_D_mask = (loss_real_mask + loss_fake_mask) / 2
        
        loss_D_img.backward()
        loss_D_mask.backward()
        optimizer_D_img.step()
        optimizer_D_mask.step()
        
        # ===================== 训练生成器 =====================
        optimizer_G.zero_grad()
        
        # 生成器希望生成的假数据被判别为真
        gen_img_output = discriminator_img(fake_imgs)
        gen_mask_output = discriminator_mask(fake_masks)
        
        loss_G_img = criterion(gen_img_output, real_labels)
        loss_G_mask = criterion(gen_mask_output, real_labels)
        
        # 组合损失
        loss_G = (loss_G_img + loss_G_mask) / 2
        
        # 添加L1损失使生成的掩码更准确
        if epoch > 20:
            l1_loss = nn.L1Loss()(fake_masks, real_masks) * 0.1
            loss_G += l1_loss
        
        loss_G.backward()
        optimizer_G.step()
        
        # 打印训练进度
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"D_img_loss: {loss_D_img.item():.4f} D_mask_loss: {loss_D_mask.item():.4f} "
                  f"G_loss: {loss_G.item():.4f}")

# 保存生成样本
def save_samples(num_samples=16):
    z = torch.randn(num_samples, latent_dim, device=device)
    with torch.no_grad():
        gen_imgs, gen_masks = generator(z)
    
    gen_imgs = gen_imgs.cpu().numpy().transpose(0, 2, 3, 1)  # [batch, H, W, C]
    gen_masks = gen_masks.cpu().numpy()
    
    for i in range(num_samples):
        # 处理图像 (-1到1 -> 0到255)
        img = ((gen_imgs[i] + 1) * 127.5).astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/images/gen_{i}.png")
        
        # 处理掩码 (0到1 -> 0到255)
        mask = (gen_masks[i].squeeze() * 255).astype(np.uint8)
        Image.fromarray(mask).save(f"{output_dir}/masks/gen_{i}_gray.png")
        
        # 创建彩色掩码 (红色表示前景)
        colored_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        colored_mask[..., 0] = mask  # 红色通道
        Image.fromarray(colored_mask).save(f"{output_dir}/masks/gen_{i}_color.png")

# 生成并保存样本
save_samples(32)
print("生成完成！样本已保存到", output_dir)