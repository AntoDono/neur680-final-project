"""PyTorch Dataset, DataLoader, and balanced-split helpers."""

import numpy as np
import pandas as pd
import torch
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src import config


class BrainDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Split, scale, undersample the majority class, and return DataLoaders.

    Parameters
    ----------
    X      : (n_samples, n_features) float array
    y      : (n_samples,) int array  (0 = HC, 1 = PD)
    scaler : optional pre-fit StandardScaler (used when fine-tuning to
             keep the same scale as pretraining is NOT desired — each stage
             fits its own scaler by default).

    Returns
    -------
    train_loader, test_loader, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    if scaler is None:
        scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_pd = int(y_train.sum())
    n_hc = int((1 - y_train).sum())
    print(f"  Train before balance: HC={n_hc} PD={n_pd} | Test: {len(y_test)}")

    # Undersample majority class down to minority class count
    rus = RandomUnderSampler(random_state=config.RANDOM_SEED)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

    n_pd_b = int(y_train_bal.sum())
    n_hc_b = int((1 - y_train_bal).sum())
    print(f"  Train after balance:  HC={n_hc_b} PD={n_pd_b}")

    train_loader = DataLoader(
        BrainDataset(X_train_bal, y_train_bal),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        BrainDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader, scaler
