import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, n_heads, emb_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(emb_dim, 3*emb_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=out_proj_bias)
        self.d_heads = emb_dim // n_heads
        
    def forward(self, x, causal_mask = False): 
        # x: (batch_size, seq_len, dim)
        
        batch_size, seq_len, d_emed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        
        # (batch_size, seq_len, dim) -> 3 * (batch_size, seq_len, d_embed)
        
        q, k, v =self.in_proj(x).chunk(3, dim=-1)
        
        # change the shape of q, k and v to match the interim shape
        q = q.view(interim_shape)
        k = k.view(interim_shape)
        v = v.view(interim_shape)
        
        # swap the elements within matrix using transpose
        # take n_heads before seq_len, like that: (batch_size, n_heads, seq_len, d_embe
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        



        weight = q@k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, float('-inf'))
            
            
        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)
        out = weight@v
        out = out.transpose(1, 2)
        out = out.reshape((batch_size, seq_len, d_emed))
        return self.out_proj(out)
        
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        # x: (batch_size, channels, h, w)
        residual = x.clone()

        # (batch_size, channels, h, w) -> (batch_size, channels, h, w)
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # (batch_size, channels, h, w) -> (batch_size, channels, h * w)
        x = x.view((n, c, h * w))

        # (batch_size, channels, h * w) -> (batch_size, h * w, channels)
        x = x.transpose(-1, -2)

        # perform self-attention without mask
        # (batch_size, h * w, channels) -> (batch_size, h * w, channels)
        x = self.attention(x)

        # (batch_size, h * w, channels) -> (batch_size, channels, h * w)
        x = x.transpose(-1, -2)

        # (batch_size, channels, h * w) -> (batch_size, channels, h, w)
        x = x.view((n, c, h, w))

        # (batch_size, channels, h, w) -> (batch_size, channels, h, w)
        x += residual

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1= nn.GroupNorm(32, in_channels)
        self. conv1= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.groupnorm2= nn.GroupNorm(32, out_channels)
        self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu= nn.ReLU()
        
        if in_channels == out_channels:
            self.residual_layer= nn.Identity()
        else:
            self.residual_layer= nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        residual = x.clone()
        x = self.groupnorm1(x)
        x = F.selu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.conv2(x)
        
        return x + self.residual_layer(residual)
    
    
class Encoder(nn.Sequential):
    def  __init__(self):
        super().__init__(
            # (batch_size, channel, h, w) -> (batch_size, 128, h, w)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            ResidualBlock(128, 128),

            # (batch_size, 128, h, w) -> (batch_size, 128, h / 2, w / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, h / 2, w / 2) -> (batch_size, 256, h / 2, w / 2)
            ResidualBlock(128, 256),

            # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h / 2, w / 2)
            ResidualBlock(256, 256),

            # (batch_size, 256, h / 2, w / 2) -> (batch_size, 256, h / 4, w / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            ResidualBlock(256, 512),

            # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 4, w / 4)
            ResidualBlock(512, 512),

            # (batch_size, 512, h / 4, w / 4) -> (batch_size, 512, h / 8, w / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            ResidualBlock(512, 512),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            ResidualBlock(512, 512),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            ResidualBlock(512, 512),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            AttentionBlock(512),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            ResidualBlock(512, 512),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            nn.GroupNorm(32, 512),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 512, h / 8, w / 8)
            nn.SiLU(),

            # (batch_size, 512, h / 8, w / 8) -> (batch_size, 8, h / 8, w / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, h / 8, w / 8) -> (batch_size, 8, h / 8, w / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x):
        # x: (batch_size, channel, h, w)

        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # (left, right, top, bottom)
            x = module(x)

        # (batch_size, 8, h / 8, w / 8) -> two tensors of shape (batch_size, 4, h / 8, w / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp log variance between -30 and 20
        log_variance = torch.clamp(log_variance, -30, 20)

        # Reparameterization trick
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        x = mean + eps * std

        # Scale the latent representation
        x *= 0.18215

        return x
    
class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, 4, 32, 32) -> (batch_size, 512, 32, 32)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),

            # (batch_Size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            AttentionBlock(512),

            # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),

            # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),

            # (batch_size, 512, 32, 32) -> (batch_size, 512, 32, 32)
            ResidualBlock(512, 512),

            # (batch_size, 512, 32, 32) -> (batch_size, 512, 64, 64)
            nn.Upsample(scale_factor=2),

            # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            ResidualBlock(512, 512),

            # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            ResidualBlock(512, 512),

            # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            ResidualBlock(512, 512),

            # (batch_size, 512, 64, 64) -> (batch_size, 512, 128, 128)
            nn.Upsample(scale_factor=2),

            # (batch_size, 512, 128, 128) -> (batch_size, 512, 128, 128)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (batch_size, 512, 128, 128) -> (batch_size, 256, 128, 128)
            ResidualBlock(512, 256),

            # (batch_size, 256, 128, 128) -> (batch_size, 256, 128, 128)
            ResidualBlock(256, 256),

            # (batch_size, 256, 128, 128) -> (batch_size, 256, 128, 128)
            ResidualBlock(256, 256),

            # (batch_size, 256, 128, 128) -> (batch_size, 256, 256, 256)
            nn.Upsample(scale_factor=2),

            # (batch_size, 256, 256, 256) -> (batch_size, 256, 256, 256)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # (batch_size, 256, 256, 256) -> (batch_size, 128, 256, 256)
            ResidualBlock(256, 128),

            # (batch_size, 128, 256, 256) -> (batch_size, 128, 256, 256)
            ResidualBlock(128, 128),

            # (batch_size, 128, 256, 256) -> (batch_size, 128, 256, 256)
            ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (batch_size, 128, 256, 256) -> (batch_size, 3, 256, 256)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
    def forward(self, x):
        # x: (batch_size, 4, h / 8, w / 8)

        # remove the scaling adding by the encoder
        x /= 0.18215

        for module in self:
            x = module(x)

        # (batch_size, 3, h, w)
        return x
    
# !gdown 1KXRTB_q4uub_XOHecpsQjE4Kmv76sZbV
# !unzip all-dogs.zip


def split_dataset(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

    print(f"Dataset split complete. {len(train_files)} training images, {len(test_files)} test images.")

source_dir = "./all-dogs"
train_dir = "./data/train/dogs"
test_dir = "./data/test/dogs"

split_dataset(source_dir, train_dir, test_dir)
     
     
# Model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
# from model import Encoder, Decoder

# Device configuration
device = torch.device('mps')

# Hyperparameters
num_epochs = 1
learning_rate = 1e-4
beta = 0.00025  # KL divergence weight

# Data loading
transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 4
dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add these hyperparameters
accumulation_steps = 1  # Adjust as needed
effective_batch_size = batch_size * accumulation_steps

train_losses = []

# training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        reconstructed, encoded = model(images)

        # Compute loss
        recon_loss = nn.MSELoss()(reconstructed, images)

        # Extract mean and log_variance from encoded
        mean, log_variance = torch.chunk(encoded, 2, dim=1)

        kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
        loss = recon_loss + beta * kl_div

        # Normalize the loss to account for accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
              f'Loss: {loss.item()*accumulation_steps:.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}')



        with torch.no_grad():
            # Take the first image from the batch
            sample_image = images[0].unsqueeze(0)
            sample_reconstructed = model(sample_image)[0]

            sample_image = (sample_image * 0.5) + 0.5
            sample_reconstructed = (sample_reconstructed * 0.5) + 0.5

            torchvision.utils.save_image(sample_reconstructed, 'reconstructed.png')

    train_losses.append(train_loss / len(dataloader))
  # Save the model checkpoint
    torch.save(model.state_dict(), f'vae_model_epoch_{epoch+1}.pth')

print('Training finished!')