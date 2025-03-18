import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

dataset_path = 'datasets'

device = torch.device("mps")

batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 200 
lr = 0.001
epochs = 30



mnist_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = MNIST(root= dataset_path, train=True, transform=mnist_transform, download=True) 
test_dataset = MNIST(root= dataset_path, train=False, transform=mnist_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcmu = nn.Linear(hidden_dim, latent_dim)
        self.fcvar = nn.Linear(hidden_dim, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.training = True
        
    def forward(self, x):
        h_ = self.leaky_relu(self.fc1(x))
        h_ = self.leaky_relu(self.fc2(h_))
        mu = self.fcmu(h_)
        logvar = self.fcvar(h_)
        
        return mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.leaky_relu(self.fc1(x))
        h = self.leaky_relu(self.fc2(h))
        x_hat = torch.sigmoid(self.fcout(h))
        return x_hat
        
        
        
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder
    def reparameterize(self, mu, var):
        eps = torch.randn_like(var).to(device)
        return mu + eps * var
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, torch.exp(0.5 * logvar))
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


encoder = Encoder(x_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, hidden_dim, x_dim)
model = VAE(encoder, decoder).to(device)


loss_fn = nn.BCELoss()

def loss_function(x, recon_x, mu, logvar):
    reproduction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=1e-3)



print("Starting training...")
model.train()

for epoch in range(epochs):
    total_loss = 0.0
    for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss = loss_function(x, x_hat, mu, logvar)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", total_loss / (batch_idx*batch_size))
