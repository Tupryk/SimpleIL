import os
import json
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Encoder with ResNet-like architecture
class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=10):
        super(Encoder, self).__init__()
        self.layer1 = ResidualBlock(in_channels, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        # Flatten and dense layers for mean and log variance
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=10, out_channels=1):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, 4, 4)
        
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = torch.sigmoid(self.deconv3(out))  # Sigmoid for outputting pixel values
        return out

# VAE Model combining Encoder and Decoder
class ResNetVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=10):
        super(ResNetVAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)
        self.path = "."

        self.model_params = {
            'latent_dimension': latent_dim,
        }

    def generate_log_data_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.path = f"./logs/models/resnet_{current_time}"
        if not os.path.exists(f"{self.path}/pth"):
            os.makedirs(f"{self.path}/pth")

    def log_model(self):
        self.generate_log_data_path()
        with open(f'{self.path}/model_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.model_params, f)

    def save(self, epoch: int):
        file_name = f"{self.path}/pth/epoch_{epoch}.pth"
        torch.save(self.state_dict(), file_name)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def forward_clean(self, original_images):
        # This should be done with the dataloader stuff maybe
        pred_images = self.forward(original_images)
        pred_images = pred_images.detach().cpu().numpy()
        pred_images = np.maximum(pred_images, 0)
        pred_images = np.minimum(pred_images, 1)
        pred_images = np.transpose(pred_images, (0, 2, 3, 1))
        return pred_images
