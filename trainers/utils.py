import torch
from torch import nn


def vae_loss(reconstructed, original, mu, log_var, beta):
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_div
