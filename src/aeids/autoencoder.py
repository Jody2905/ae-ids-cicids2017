"""Dense autoencoder in PyTorch, trained on benign flows only."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: list[int]):
        super().__init__()
        dims = [n_features] + hidden
        enc, dec = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        enc = enc[:-1]  # linear bottleneck
        rdims = dims[::-1]
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        dec = dec[:-1]  # linear output (inputs are standardised, not in [0, 1])
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, hidden: list[int], epochs: int,
                      batch_size: int, lr: float, patience: int, seed: int):
    torch.manual_seed(seed)
    model = AutoEncoder(X_train.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_train)
    Xv = torch.from_numpy(X_val)
    gen = torch.Generator().manual_seed(seed)

    history, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            xb = Xt[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(Xt)
        val_loss = float(reconstruction_error(model, X_val).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": round(time.time() - t0, 2)})
        print(f"  epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  ({time.time() - t0:.1f}s)", flush=True)
        if val_loss < best - 1e-6:
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop: no val improvement for {patience} epochs")
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def feature_errors(model: AutoEncoder, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Squared reconstruction error per sample and feature, shape (n, d)."""
    model.eval()
    out = np.empty_like(X)
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        out[i:i + batch_size] = ((model(xb) - xb) ** 2).numpy()
    return out


def reconstruction_error(model: AutoEncoder, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Mean squared reconstruction error per sample (the anomaly score)."""
    scores = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch_size):
        scores[i:i + batch_size] = feature_errors(model, X[i:i + batch_size], batch_size).mean(axis=1)
    return scores
