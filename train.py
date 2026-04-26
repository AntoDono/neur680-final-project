"""Entry point for two-stage training.

Usage
-----
    python train.py --stage pretrain   # PPMI → checkpoints/ppmi_pretrained.pt
    python train.py --stage finetune   # load checkpoint, fine-tune on lab
    python train.py --stage both       # default: pretrain then finetune
"""

import argparse
import os
import torch

from src import config
from src.data.lab import load_lab
from src.data.ppmi import load_ppmi
from src.data.normalize import aseg_common_features
from src.data.loaders import make_loaders
from src.model import TabularTransformer
from src.trainer import train_one_stage, evaluate
from src.attention import plot_attention_maps

CHECKPOINT = "checkpoints/ppmi_pretrained.pt"
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_on_ppmi() -> None:
    print("\n" + "=" * 60)
    print("STAGE 1: Pretrain on PPMI (ASEG volumes)")
    print("=" * 60)

    df, aseg_cols = load_ppmi()
    feature_cols  = aseg_cols + ["Age", "Sex"]

    X = df[feature_cols].values.astype("float32")
    y = df["label"].values.astype("float32")

    train_loader, test_loader, scaler = make_loaders(X, y)

    model = TabularTransformer(num_features=len(feature_cols)).to(device)
    print(f"\nFeatures: {len(feature_cols)} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_one_stage(model, train_loader, device, label="PPMI pretrain")
    evaluate(model, test_loader, device, label="PPMI pretrain")

    torch.save(
        {"state_dict": model.state_dict(), "feature_cols": feature_cols},
        CHECKPOINT,
    )
    print(f"\nCheckpoint saved → {CHECKPOINT}")


def finetune_on_lab() -> None:
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tune on Lab (ASEG + cortical thickness)")
    print("=" * 60)

    df, aseg_cols, thickness_cols = load_lab()
    feature_cols = aseg_cols + thickness_cols + ["Age", "Sex"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].dropna(axis=1).values.astype("float32")
    feature_cols = [c for c in feature_cols if c in df.dropna(axis=1).columns]
    y = df["label"].values.astype("float32")

    train_loader, test_loader, scaler = make_loaders(X, y)

    model = TabularTransformer(num_features=len(feature_cols)).to(device)
    print(f"\nFeatures: {len(feature_cols)} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights if checkpoint exists
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded pretrained weights from {CHECKPOINT}")
        ft_lr = config.LR * 0.3   # smaller LR for fine-tuning
    else:
        print("No pretrain checkpoint found — training from scratch.")
        ft_lr = config.LR

    train_one_stage(model, train_loader, device, lr=ft_lr, label="Lab finetune")
    evaluate(model, test_loader, device, label="Lab finetune")
    plot_attention_maps(model, scaler, feature_cols, device)


def main() -> None:
    ap = argparse.ArgumentParser(description="PD vs HC transformer training")
    ap.add_argument(
        "--stage",
        choices=["pretrain", "finetune", "both"],
        default="both",
        help="Which training stage to run (default: both)",
    )
    args = ap.parse_args()

    if args.stage in ("pretrain", "both"):
        pretrain_on_ppmi()
    if args.stage in ("finetune", "both"):
        finetune_on_lab()


if __name__ == "__main__":
    main()
