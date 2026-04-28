"""PyTorch Dataset, DataLoader, and balanced-split helpers."""

import numpy as np
import torch
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
) -> tuple[DataLoader, DataLoader, StandardScaler, np.ndarray]:
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
    train_loader, test_loader, scaler, test_indices, pos_weight
    pos_weight is a scalar tensor = n_hc / n_pd (original training split),
    passed to BCEWithLogitsLoss to down-weight the majority PD class.
    """
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, _, test_indices = train_test_split(
        X, y, indices,
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
    print(f"  Train: HC={n_hc} PD={n_pd} | Test: {len(y_test)}")

    # full inverse ratio — aggressively up-weights HC to counteract imbalance
    pos_weight = torch.tensor([n_hc / n_pd], dtype=torch.float32)

    train_loader = DataLoader(
        BrainDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        BrainDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader, scaler, test_indices, pos_weight
