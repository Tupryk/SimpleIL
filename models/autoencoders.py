import os
import json
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime


class AE(nn.Module):
    def __init__(self,
                 latent_dim: int = 6,
                 input_dim: int = 256,
                 dropout_prob: float = 0.5,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 activation: str = 'silu',
                 layers_encoder: list[tuple[int, int]] = [(3, 16), (16, 32), (32, 64), (64, 128)]):
        
        super(AE, self).__init__()

        # Initialize encoder layers
        layers = []
        current_dim = input_dim
        for i, (in_channels, out_channels) in enumerate(layers_encoder):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            current_dim = (current_dim - kernel_size + 2 * padding) // stride + 1 
            
            if i != len(layers_encoder)-1:
                # Add activation function
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'silu':
                    layers.append(nn.SiLU())
                else:
                    raise ValueError(f"Error: Activation function '{activation}' not implemented")
                
                # Add dropout layer
                layers.append(nn.Dropout(dropout_prob))

        self.encoder = nn.Sequential(*layers)

        # Flatten to a vector
        a = layers_encoder[-1][1]
        b = layers_encoder[0][1]
        self.flattened_size = out_channels * current_dim * current_dim
        self.latent_dim = latent_dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, self.latent_dim)

        # Decoder
        self.fc2 = nn.Linear(self.latent_dim, self.flattened_size)
        self.unflatten = nn.Unflatten(1, (a, input_dim // b, input_dim // b))

        layers = []
        for i, (in_channels, out_channels) in enumerate(layers_encoder[::-1]):
            layers.append(
                nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1)
            )
            
            if i != len(layers_encoder)-1:
                # Add activation function
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'silu':
                    layers.append(nn.SiLU())
                else:
                    raise ValueError(f"Error: Activation function '{activation}' not implemented")
                
                # Add dropout layer
                layers.append(nn.Dropout(dropout_prob))

        self.decoder = nn.Sequential(*layers)

        self.model_params = {
            'image_size': input_dim,
            'latent_dimension': latent_dim,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'encoder_layers': layers_encoder,
            'activation': activation,
            'dropout_prob': dropout_prob
        }

        self.path = "."

    def generate_log_data_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.path = f"./logs/models/autoencoder_{current_time}"
        if not os.path.exists(f"{self.path}/pth"):
            os.makedirs(f"{self.path}/pth")

    def log_model(self):
        self.generate_log_data_path()
        with open(f'{self.path}/model_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.model_params, f)
    
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = self.flatten(encoded)
        encoded = self.fc1(encoded)
        return encoded
    
    def decode(self, x):
        decoded = self.fc2(x)
        decoded = self.unflatten(decoded)
        decoded = self.decoder(decoded)
        return decoded
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def forward_clean(self, original_images):
        # This should be done with the dataloader stuff maybe
        pred_images = self.forward(original_images)
        pred_images = pred_images.detach().cpu().numpy()
        pred_images = np.maximum(pred_images, 0)
        pred_images = np.minimum(pred_images, 1)
        pred_images = np.transpose(pred_images, (0, 2, 3, 1))
        return pred_images
    
    def save(self, epoch: int):
        file_name = f"{self.path}/pth/epoch_{epoch}.pth"
        torch.save(self.state_dict(), file_name)
    

class VAE(AE):
    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)
        self.fc_log_var = nn.Linear(self.flattened_size, self.latent_dim)

    def generate_log_data_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.path = f"./logs/models/VAE_{current_time}"
        if not os.path.exists(f"{self.path}/pth"):
            os.makedirs(f"{self.path}/pth")

    def encode(self, x):
        encoded = self.encoder(x)
        encoded = self.flatten(encoded)
        mu = self.fc1(encoded)  # fc_mu
        log_var = self.fc_log_var(encoded)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        decoded = self.fc2(z)  # fc_decoder
        decoded = self.unflatten(decoded)
        decoded = self.decoder(decoded)
        return decoded
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return decoded, mu, log_var
    
    def forward_clean(self, original_images):
        # This should be done with the dataloader stuff maybe
        pred_images, _, _ = self.forward(original_images)
        pred_images = pred_images.detach().cpu().numpy()
        pred_images = np.maximum(pred_images, 0)
        pred_images = np.minimum(pred_images, 1)
        pred_images = np.transpose(pred_images, (0, 2, 3, 1))
        return pred_images


class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(UNetAutoencoder, self).__init__()

        # Encoder path
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer for encoding
        self.fc_encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, latent_dim),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer for decoding
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 8 * 8),
            nn.ReLU(inplace=True)
        )

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def encode(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Flatten and encode to latent vector
        encoded = self.fc_encode(enc5)
        return encoded

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Flatten and encode to latent vector
        encoded = self.fc_encode(enc5)

        # Decode latent vector back to spatial features
        decoded = self.fc_decode(encoded)
        decoded = decoded.view(-1, 1024, 8, 8)  # Reshape back to feature map size

        # Decoder
        dec4 = self.upconv4(decoded)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # Output
        return self.out_conv(dec1)
    
    def forward_clean(self, original_images):
        # This should be done with the dataloader stuff maybe
        pred_images = self.forward(original_images)
        pred_images = pred_images.detach().cpu().numpy()
        pred_images = np.maximum(pred_images, 0)
        pred_images = np.minimum(pred_images, 1)
        pred_images = np.transpose(pred_images, (0, 2, 3, 1))
        return pred_images
