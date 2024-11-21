import os
import json
import torch
from torch import nn
from datetime import datetime


class MLPd(nn.Module):
  
    def __init__(self, input_dim: int, hidden_layers: list[int]=[256, 256, 256]):
        super(MLPd, self).__init__()
        layers = []
        self.input_dim = input_dim
        last_dim = input_dim + 1
        for size in hidden_layers:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size

        layers.append(nn.Linear(last_dim, input_dim))

        self.layers = nn.Sequential(*layers)

        self.model_params = {
            'input_size': input_dim,
        }

        self.path = "."

    def generate_log_data_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.path = f"./logs/models/diffuser-mlp_{current_time}"
        if not os.path.exists(f"{self.path}/pth"):
            os.makedirs(f"{self.path}/pth")

    def log_model(self):
        self.generate_log_data_path()
        with open(f'{self.path}/model_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.model_params, f)
    
    def save(self, epoch: int):
        file_name = f"{self.path}/pth/epoch_{epoch}.pth"
        torch.save(self.state_dict(), file_name)

    def forward(self, x, t):
        x = torch.concat([x, t], axis=-1)
        x = self.layers(x)
        return x
    
    def sample(self, device: str, n_samples: int = 50, n_steps: int=100):
        x_t = torch.randn((n_samples, self.input_dim)).to(device)
        
        for i in range(n_steps):
            x_t += torch.randn((n_samples, self.input_dim)).to(device) * .001

            # t = torch.zeros((n_samples, 1)).to(device) + i / n_steps
            t = torch.ones((n_samples, 1)).to(device)

            noise_prediction = self(x_t, t)
            # noise_prediction /= n_steps - i
            noise_prediction /= n_steps
            x_t -= noise_prediction

        return x_t
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, token_dim, cond_dim, num_heads):
        """
        Multi-Head Cross-Attention for tokens and conditioning vector.
        Args:
            token_dim (int): Dimension of input token embeddings.
            cond_dim (int): Dimension of conditioning vector as sequence embeddings.
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadCrossAttention, self).__init__()
        self.token_dim = token_dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads

        # Projection layers for query, key, and value
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(cond_dim, token_dim)
        self.v_proj = nn.Linear(cond_dim, token_dim)

        # Output projection
        self.out_proj = nn.Linear(token_dim, token_dim)

        # Number of heads in multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)

    def forward(self, tokens, conditioning_vector):
        """
        Forward pass for multi-head cross-attention.
        Args:
            tokens (Tensor): Input tokens of shape (batch_size, seq_len, token_dim).
            conditioning_vector (Tensor): Conditioning vector of shape (batch_size, cond_len, cond_dim).
        Returns:
            Tensor: Attention-updated tokens of shape (batch_size, seq_len, token_dim).
        """
        # Compute queries, keys, and values
        queries = self.q_proj(tokens)  # Shape: (batch_size, seq_len, token_dim)
        keys = self.k_proj(conditioning_vector)  # Shape: (batch_size, cond_len, token_dim)
        values = self.v_proj(conditioning_vector)  # Shape: (batch_size, cond_len, token_dim)

        # Apply multi-head attention
        attn_output, _ = self.attention(queries, keys, values)  # Shape: (batch_size, seq_len, token_dim)

        # Output projection
        output = self.out_proj(attn_output + tokens)  # Residual connection
        return output

    
class ConditionalTransformerDiffuser(nn.Module):
  
    def __init__(self, cond_dim: int, token_dim: int=9, num_heads: int=9, max_sequence_len: int=6, layer_count: int=4):
        super(ConditionalTransformerDiffuser, self).__init__()

        self.token_dim = token_dim
        self.max_sequence_len = max_sequence_len

        self.cross_attention_layers = []
        for _ in range(layer_count):
            cross_attention = MultiHeadCrossAttention(token_dim, cond_dim+1, num_heads)
            self.cross_attention_layers.append(cross_attention)

        self.model_params = {
            "token_dim": token_dim,
            "cond_dim": cond_dim,
            "num_heads": num_heads,
            "max_sequence_len": max_sequence_len,
            "layer_count": layer_count,
        }

        self.path = "."

    def generate_log_data_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.path = f"./logs/models/conditional-transformer-diffuser_{current_time}"
        if not os.path.exists(f"{self.path}/pth"):
            os.makedirs(f"{self.path}/pth")

    def log_model(self):
        self.generate_log_data_path()
        with open(f'{self.path}/model_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.model_params, f)
    
    def save(self, epoch: int):
        file_name = f"{self.path}/pth/epoch_{epoch}.pth"
        torch.save(self.state_dict(), file_name)

    def forward(self, x, c, t):
        c = torch.concat([c, t], axis=-1)
        for i in range(len(self.cross_attention_layers)):
            x = self.cross_attention_layers[i](x, c)
        return x
    
    def sample(self, conditioning, device: str, n_samples: int=50, n_steps: int=100, renoising_scale: float=.001):
        
        conditioning = conditioning.to(device)
        x_t = torch.randn((n_samples, self.max_sequence_len, self.token_dim)).to(device)
        
        for i in range(n_steps):
            x_t += torch.randn((n_samples, self.max_sequence_len, self.token_dim)).to(device) * renoising_scale
            t = torch.ones((n_samples, 1)).to(device)

            noise_prediction = self(x_t, conditioning, t)
            noise_prediction /= n_steps
            x_t -= noise_prediction

        return x_t
  