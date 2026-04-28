"""Training loop and evaluation helpers."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config


def train_one_stage(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    *,
    num_epochs: int = config.PRETRAIN_EPOCHS,
    lr: float = config.LR,
    patience: int = config.PATIENCE,
    pos_weight: torch.Tensor | None = None,
    label: str = "",
) -> dict:
    """
    Train the model for up to num_epochs with early stopping.

    Returns the best model state_dict (by training loss).
    """
    pw        = pos_weight.to(device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss      = float("inf")
    best_state     = None
    patience_count = 0
    prefix         = f"[{label}] " if label else ""

    pbar = tqdm(range(num_epochs), desc=f"{prefix}training", unit="ep", dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", best=f"{best_loss:.4f}", patience=patience_count)

        if avg_loss < best_loss:
            best_loss      = avg_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                tqdm.write(f"{prefix}Early stopping at epoch {epoch + 1}  best loss: {best_loss:.4f}")
                break

    model.load_state_dict(best_state)
    tqdm.write(f"{prefix}Restored best model (loss {best_loss:.4f})")
    return {"state_dict": best_state, "best_loss": best_loss}


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    *,
    label: str = "",
) -> dict:
    """Print accuracy, classification report, confusion matrix; return metrics dict."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = (torch.sigmoid(model(X_batch.to(device))) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    prefix     = f"[{label}] " if label else ""

    acc = accuracy_score(all_labels, all_preds)
    p, r, f, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0,
    )

    print(f"\n{prefix}Accuracy: {acc:.3f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=["HC", "PD"], zero_division=0))
    print(f"{prefix}Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(
        confusion_matrix(all_labels, all_preds),
        index=["True HC", "True PD"],
        columns=["Pred HC", "Pred PD"],
    ))

    return {
        "accuracy":    acc,
        "hc_precision": p[0], "hc_recall": r[0], "hc_f1": f[0],
        "pd_precision": p[1], "pd_recall": r[1], "pd_f1": f[1],
    }
