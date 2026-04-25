# src/lbp_kikuchi/training/engine.py

import torch

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = ((pred - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        total_loss += loss.item()

    return total_loss / len(loader)