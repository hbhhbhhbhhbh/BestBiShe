import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
 
# 设定参数
batch_size = 128
latent_dim = 100
image_size = 28
num_epochs = 50
lr = 0.0002
 
# 载入 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
# **1. 生成器**
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, image_size * image_size), nn.Tanh()
        )
 
    def forward(self, z):
        img = self.model(z).view(-1, 1, image_size, image_size)
        return img
 
# **2. 判别器**
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
 
    def forward(self, img):
        img = img.view(img.size(0), -1)
        return self.model(img)
 
# **训练 DCGAN**
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
 
for epoch in range(num_epochs):
    for imgs, _ in dataloader:
        real_imgs = imgs.to(torch.float32)
        real_labels = torch.ones(imgs.size(0), 1)
        fake_labels = torch.zeros(imgs.size(0), 1)
 
        # **训练判别器**
        optimizer_D.zero_grad()
        loss_real = criterion(discriminator(real_imgs), real_labels)
        z = torch.randn(imgs.size(0), latent_dim)
        fake_imgs = generator(z)
        loss_fake = criterion(discriminator(fake_imgs.detach()), fake_labels)
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()
 
        # **训练生成器**
        optimizer_G.zero_grad()
        loss_G = criterion(discriminator(fake_imgs), real_labels)
        loss_G.backward()
        optimizer_G.step()
 
# **生成增强数据**
z = torch.randn(10, latent_dim)
generated_images = generator(z).detach().numpy().squeeze()
 
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, img in enumerate(generated_images):
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off')
plt.show()