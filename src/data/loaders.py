"""PyTorch Dataset, DataLoader, and split/harmonization helpers.

ComBat harmonization is applied here, *after* the train/test split, so that
batch-effect correction parameters are estimated exclusively from training
subjects and then applied (frozen) to the held-out test subjects.  This
prevents any leakage of test-set statistics into the preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import torch
from neuroCombat import neuroCombat
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
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[DataLoader, DataLoader, StandardScaler, dict | None, np.ndarray, torch.Tensor]:
    """
    Split, harmonize (ComBat on train only), scale, and return DataLoaders.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame as returned by load_combined().
        Must contain 'label', 'source', and all columns in feature_cols.
    feature_cols : list[str]
        Feature columns to use as model input (ASEG + Age + Sex).
    scaler : optional pre-fit StandardScaler.

    Returns
    -------
    train_loader, test_loader, scaler, combat_estimates, test_indices, pos_weight

    combat_estimates : dict or None
        The fitted ComBat γ*/δ* estimates (from neuroCombat).  Save these
        alongside the model checkpoint so the same correction can be applied
        at inference time via apply_combat_estimates().  None when ComBat is
        disabled or only one site is present.
    """
    y       = df["label"].values
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    _print_split_summary(train_df, test_df)

    # ComBat: fit on train partition, apply frozen estimates to test
    combat_estimates: dict | None = None
    if (
        config.USE_COMBAT
        and "source" in df.columns
        and df["source"].nunique() >= 2
    ):
        train_df, test_df, combat_estimates = _combat_fit_apply(
            train_df, test_df, feature_cols
        )

    X_train = train_df[feature_cols].values.astype("float32")
    X_test  = test_df[feature_cols].values.astype("float32")
    y_train = train_df["label"].values.astype("float32")
    y_test  = test_df["label"].values.astype("float32")

    if scaler is None:
        scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_pd       = int(y_train.sum())
    n_hc       = int((1 - y_train).sum())
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
    return train_loader, test_loader, scaler, combat_estimates, test_idx, pos_weight


# ── Helpers ────────────────────────────────────────────────────────────────────

def _combat_fit_apply(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit ComBat on train_df, apply frozen estimates to test_df, return estimates.

    Only imaging features (everything except Age and Sex) are harmonized;
    Age and Sex are passed as biological covariates to protect them.

    Returns
    -------
    train_df, test_df : harmonized DataFrames
    estimates : dict
        Raw ComBat estimates (γ*, δ*, pooled variance, etc.) suitable for
        saving in the model checkpoint and re-applying at inference time via
        apply_combat_estimates().
    """
    imaging_feats = [f for f in feature_cols if f not in ("Age", "Sex")]

    covars_train = train_df[["source", "label", "Age", "Sex"]].copy()
    covars_train["Sex"]   = covars_train["Sex"].astype(int)
    covars_train["label"] = covars_train["label"].astype(int)

    result = neuroCombat(
        dat=train_df[imaging_feats].values.T,
        covars=covars_train,
        batch_col="source",
        categorical_cols=["label", "Sex"],
        continuous_cols=["Age"],
    )
    estimates = result["estimates"]

    train_df = train_df.copy()
    train_df[imaging_feats] = result["data"].T

    # Apply frozen estimates to test (no re-estimation)
    test_df = test_df.copy()
    test_df[imaging_feats] = apply_combat_estimates(
        dat=test_df[imaging_feats].values.T,
        batch=test_df["source"].values,
        estimates=estimates,
    ).T

    site_list = " / ".join(sorted(covars_train["source"].unique()))
    print(
        f"[ComBat] Fit on train ({covars_train['source'].nunique()} sites: {site_list})"
        f" → applied frozen estimates to test"
    )
    return train_df, test_df, estimates


def apply_combat_estimates(
    dat: np.ndarray,
    batch: np.ndarray,
    estimates: dict,
) -> np.ndarray:
    """
    Apply pre-fitted ComBat estimates to new data (NumPy 2.0-compatible).

    Replicates neuroCombatFromTraining() from the neuroCombat package but
    fixes the ``int(np.where(...)[0])`` call that raises TypeError on NumPy
    ≥ 2.0 (1-D arrays can no longer be converted to scalars directly).

    Parameters
    ----------
    dat       : (n_features, n_subjects) array of imaging data to harmonize.
    batch     : (n_subjects,) array of site/batch labels (strings).
    estimates : the ``estimates`` dict returned by ``neuroCombat()``.

    Returns
    -------
    harmonized : (n_features, n_subjects) array.
    """
    batch      = np.array(batch, dtype="str")
    old_levels = np.array(estimates["batches"], dtype="str")

    missing = np.setdiff1d(np.unique(batch), old_levels)
    if missing.size:
        raise ValueError(
            f"[ComBat] Test batches not seen during training: {missing.tolist()}"
        )

    # Index of each test subject's batch in the training batch list.
    # Use [0][0] (not [0]) to stay compatible with NumPy ≥ 2.0.
    wh = [int(np.where(old_levels == x)[0][0]) for x in batch]

    var_pooled = estimates["var.pooled"]
    stand_mean = estimates["stand.mean"][:, 0] + estimates["mod.mean"].mean(axis=1)
    gamma_star = estimates["gamma.star"]
    delta_star = estimates["delta.star"]

    n_subjects = dat.shape[1]
    stand_mean = np.transpose([stand_mean] * n_subjects)

    harmonized  = np.subtract(dat, stand_mean) / np.sqrt(var_pooled)
    gamma       = np.transpose(gamma_star[wh, :])
    delta       = np.transpose(delta_star[wh, :])
    harmonized  = np.subtract(harmonized, gamma) / np.sqrt(delta)
    harmonized  = harmonized * np.sqrt(var_pooled) + stand_mean
    return harmonized


def _print_split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    def _counts(df: pd.DataFrame) -> tuple[int, int]:
        n_pd = int(df["label"].sum())
        n_hc = int((1 - df["label"]).sum())
        return n_hc, n_pd

    tr_hc, tr_pd = _counts(train_df)
    te_hc, te_pd = _counts(test_df)
    print(f"  Train : HC={tr_hc:>4}  PD={tr_pd:>4}  (n={len(train_df)})")
    print(f"  Eval  : HC={te_hc:>4}  PD={te_pd:>4}  (n={len(test_df)})")
