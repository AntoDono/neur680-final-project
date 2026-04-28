"""Train a single TabularTransformer on the combined PPMI + Lab + OASIS-3 dataset.

Data preparation (runs automatically on each invocation)
---------------------------------------------------------
Each source loader processes its raw data and writes a per-source CSV to
processed_data/ before merging:

    processed_data/ppmi.csv    ← raw_data/ppmi/
    processed_data/lab.csv     ← raw_data/lab/
    processed_data/oasis3.csv  ← raw_data/oasis3/healthy/*/aseg.stats
                                  (empty if not yet downloaded; see
                                   raw_data/oasis3/aseg_lh_rh_download_oasis_freesurfer.sh)

The three sources are then merged on their common canonical ASEG features,
ComBat-harmonized across sites, and fed to the TabularTransformer.

Usage
-----
    python train.py
"""

import os
import torch

from src import config
from src.data.combined import load_combined
from src.data.loaders import make_loaders
from src.model import TabularTransformer
from src.trainer import train_one_stage, evaluate
from src.attention import plot_attention_maps, print_attention_ranking

CHECKPOINT = "checkpoints/combined.pt"
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_combined() -> None:
    print("\n" + "=" * 60)
    print("Training on combined PPMI + Lab + OASIS-3 dataset")
    print("=" * 60)

    df, feature_cols = load_combined()

    X = df[feature_cols].values.astype("float32")
    y = df["label"].values.astype("float32")

    train_loader, test_loader, scaler, test_indices, pos_weight = make_loaders(X, y)
    test_subject_ids = df.index[test_indices].tolist()

    model = TabularTransformer(num_features=len(feature_cols)).to(device)
    print(f"\nFeatures: {len(feature_cols)} | "
          f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    result  = train_one_stage(model, train_loader, device,
                              num_epochs=config.PRETRAIN_EPOCHS,
                              pos_weight=pos_weight, label="Combined")
    metrics = evaluate(model, test_loader, device, label="Combined")
    metrics["best_loss"] = result["best_loss"]

    torch.save({"state_dict": model.state_dict(), "feature_cols": feature_cols},
               CHECKPOINT)
    print(f"\nCheckpoint saved → {CHECKPOINT}")

    plot_attention_maps(model, scaler, feature_cols, device,
                        test_subject_ids=test_subject_ids, metrics=metrics)
    print_attention_ranking(model, scaler, feature_cols, device,
                            test_subject_ids=test_subject_ids)


if __name__ == "__main__":
    train_combined()
