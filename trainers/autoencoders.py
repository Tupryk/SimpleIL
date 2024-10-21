import json
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from models.autoencoders import AE, VAE
from torch.utils.data import DataLoader

from trainers.utils import vae_loss


def train_vae(model: VAE,
              device: str,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 10,
              checkpoint_every: int = 1000,
              lr: float = 1e-3,
              beta: float=1.):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    train_losses = []
    val_losses = []

    train_info = {
        "val_size": len(val_loader) * val_loader.batch_size,
        "train_size": len(train_loader) * train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "train_batch_size": train_loader.batch_size,
        "n_epochs": n_epochs,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "loss_fn": "MSELoss",
        "VAE_beta": beta
    }

    for epoch in range(n_epochs):

        # Train step
        model.train()
        train_loss = 0
        for [data] in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            reconstructed, mu, log_var = model(data)
            loss = vae_loss(reconstructed, data, mu, log_var, beta)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for [data] in val_loader:
                data = data.to(device)
                reconstructed, mu, log_var = model(data)                
                val_loss += vae_loss(reconstructed, data, mu, log_var, beta).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print losses for this epoch
        print(f"Epoch {epoch + 1},\t Train Loss: {np.mean(train_losses):.6f},\t Val Loss: {val_loss:.6f}")
        train_losses = []

        if (epoch+1) % checkpoint_every == 0:
            train_info[f"epoch_{epoch+1}_loss_val: "] = val_loss
            train_info[f"epoch_{epoch+1}_loss_train: "] = train_loss
            with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
                json.dump(train_info, f)
            
            model.save(epoch+1)

    model.save(n_epochs)
    with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
        json.dump(train_info, f)
    return model


def train_ae(model: AE,
             device: str,
             train_loader: DataLoader,
             val_loader: DataLoader,
             n_epochs: int = 10,
             checkpoint_every: int = 1000,
             lr: float = 1e-3):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    train_losses = []
    val_losses = []
    loss_fn = nn.MSELoss()

    train_info = {
        "val_size": len(val_loader) * val_loader.batch_size,
        "train_size": len(train_loader) * train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "train_batch_size": train_loader.batch_size,
        "n_epochs": n_epochs,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "loss_fn": "MSELoss"
    }

    for epoch in range(n_epochs):
        
        # Train step
        model.train()
        train_loss = 0
        for [data] in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            loss = loss_fn(data, out)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for [data] in val_loader:
                data = data.to(device)
                out = model(data)
                val_loss += loss_fn(data, out)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print losses for this epoch
        print(f"Epoch {epoch + 1},\t Train Loss: {np.mean(train_losses):.6f},\t Val Loss: {val_loss:.6f}")
        train_losses = []

        if (epoch+1) % checkpoint_every == 0:
            train_info[f"epoch_{epoch+1}_loss_val: "] = float(val_loss.cpu().numpy())
            train_info[f"epoch_{epoch+1}_loss_train: "] = train_loss
            with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
                json.dump(train_info, f)
            
            model.save(epoch+1)

    model.save(n_epochs)
    with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
        json.dump(train_info, f)
    return model
