import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.autoencoders import AE


def train(model: nn.Module,
          device: str,
          train_loader: DataLoader,
          val_loader: DataLoader,
          n_epochs: int = 10,
          checkpoint_every: int = 1000,
          lr: float = 1e-3):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.log_model()
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
        "optimizer": "Adam",
        "loss_fn": "MSELoss",
    }

    for epoch in range(n_epochs):

        # Train step
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(X)
            loss = loss_fn(out, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()  
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = loss_fn(out, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print losses for this epoch
        print(f"Epoch {epoch + 1},\t Train Loss: {train_loss:.6f},\t Val Loss: {val_loss:.6f}")

        if (epoch+1) % checkpoint_every == 0:
            with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
                json.dump(train_info, f)
            
            model.save(epoch+1)

    model.save(n_epochs)
    with open(f'{model.path}/train_info.json', 'w', encoding='utf-8') as f:
        json.dump(train_info, f)
    return model


def train_with_encoder(
        model: nn.Module,
        encoder: AE,
        device: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 10,
        lr: float = 1e-3
    ):
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(encoder.parameters()),
        lr=lr
    )
    
    model.to(device)
    encoder.to(device)
    train_losses = []
    val_losses = []
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        
        # Train step
        encoder.train()
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            X, _ = encoder.encode(X)  # Encode input
            out = model(X)  # Forward pass through model
            loss = loss_fn(out, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        encoder.eval()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                X, _ = encoder.encode(X)
                out = model(X)
                loss = loss_fn(out, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print losses for this epoch
        print(f"Epoch {epoch + 1},\t Train Loss: {train_loss:.6f},\t Val Loss: {val_loss:.6f}")

    return model, encoder

