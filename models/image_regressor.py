import os
import json
import torch
import torch.nn as nn
from datetime import datetime


class ImageRegressor(nn.Module):
    def __init__(self,
                 image_dim: int,
                 output_size: int,
                 dropout_prob: float = 0.2,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 activation: str = 'silu',
                 cnn_layers: list[tuple[int, int]] = [(3, 16), (16, 32), (32, 64), (64, 128)],
                 hidden_linear_layers: list[int] = [128, 64]):
        
        super(ImageRegressor, self).__init__()

        # CNN layers
        layers = []
        for i, (in_channels, out_channels) in enumerate(cnn_layers):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            
            if i != len(cnn_layers)-1:
                # Add activation function
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'silu':
                    layers.append(nn.SiLU())
                else:
                    raise ValueError(f"Error: Activation function '{activation}' not implemented")
                
                # Add dropout layer
                layers.append(nn.Dropout(dropout_prob))

        # Flatten to a vector
        a = cnn_layers[-1][1]
        b = cnn_layers[0][1]
        self.flattened_size = a * (image_dim // b) * (image_dim // b)
        layers.append(nn.Flatten())

        # Add linear layers
        last_dim = self.flattened_size
        for size in hidden_linear_layers:
            layers.append(nn.Linear(last_dim, size))
            last_dim = size
            
            # Add activation function
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'silu':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Error: Activation function '{activation}' not implemented")
            
            # Add dropout layer
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(last_dim, output_size))

        self.layers = nn.Sequential(*layers)

        self.model_params = {
            'image_dim': image_dim,
            'output_size': output_size,
            'hidden_linear_layers': hidden_linear_layers,
            'activation': activation,
            'dropout_prob': dropout_prob,
            'kernel_size': kernel_size,
            'padding': padding,
            'stride': stride,
            'cnn_layers': cnn_layers
        }

        self.path = "."

    def generate_log_data_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.path = f"./logs/models/image-regressor_{current_time}"
        if not os.path.exists(f"{self.path}/pth"):
            os.makedirs(f"{self.path}/pth")

    def log_model(self):
        self.generate_log_data_path()
        with open(f'{self.path}/model_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.model_params, f)
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def save(self, epoch: int):
        file_name = f"{self.path}/pth/epoch_{epoch}.pth"
        torch.save(self.state_dict(), file_name)
    
