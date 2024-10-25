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
