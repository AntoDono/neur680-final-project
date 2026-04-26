"""Attention map extraction and visualization for TabularTransformer."""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src import config


def extract_attention_matrix(
    model,
    x_scaled_1d: np.ndarray,
    device: torch.device,
) -> list[np.ndarray]:
    """
    Return the full square attention matrix for each transformer layer.

    Parameters
    ----------
    model       : trained TabularTransformer
    x_scaled_1d : np.ndarray, shape (n_features,)
    device      : torch.device

    Returns
    -------
    list of np.ndarray, shape (n_tokens, n_tokens) per layer,
    where n_tokens = 1 (CLS) + n_features.
    attn[i, j] = weight token-i places on token-j.
    """
    model.eval()
    x = torch.tensor(x_scaled_1d[np.newaxis, :], dtype=torch.float32).to(device)
    attn_matrices = []

    with torch.no_grad():
        B, F = x.shape
        h   = model.feature_proj(x.unsqueeze(-1))   # (1, F, d_model)
        cls = model.cls_token.expand(B, -1, -1)      # (1, 1, d_model)
        h   = torch.cat([cls, h], dim=1)             # (1, F+1, d_model)

        for layer in model.transformer.layers:
            h2, attn_w = layer.self_attn(
                h, h, h,
                need_weights=True,
                average_attn_weights=True,   # (1, F+1, F+1)
            )
            attn_matrices.append(attn_w[0].cpu().numpy())   # (F+1, F+1)

            # Replicate TransformerEncoderLayer forward (post-norm)
            h = h + layer.dropout1(h2)
            h = layer.norm1(h)
            h2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = h + layer.dropout2(h2)
            h = layer.norm2(h)

    return attn_matrices


def plot_attention_maps(
    model,
    scaler: StandardScaler,
    feature_cols: list[str],
    device: torch.device,
    raw_dir: str = "raw_data/lab",
    out_path: str = "attention_maps.png",
) -> None:
    """
    Plot full (N_tokens × N_tokens) attention heatmaps for 2 HC and 2 PD subjects.
    Reads directly from the lab processed CSV so it works independently of training split.
    """
    raw    = pd.read_csv("processed_data/patient_data.csv", index_col=0)
    groups = raw["Group"].copy()

    raw2 = raw.drop(columns=[c for c in ["Code", "Group", "BrainSegVolNotVent_y",
                                          "eTIV_x", "eTIV_y"]
                              if c in raw.columns]).copy()
    raw2["Sex"] = (raw2["Sex"] == "M").astype(int)

    # Build a DataFrame with exactly the columns the scaler was fit on,
    # filling any missing columns with 0 so the shape matches.
    aligned = pd.DataFrame(0.0, index=raw2.index, columns=feature_cols)
    for c in feature_cols:
        if c in raw2.columns:
            aligned[c] = raw2[c].values
    X_all   = scaler.transform(aligned.values.astype("float32"))
    available = feature_cols          # all tokens present after alignment
    idx_map = {sid: i for i, sid in enumerate(raw.index)}

    hc_ids    = groups[groups == "HC"].index[:2].tolist()
    pd_ids    = groups[groups == "PD"].index[:2].tolist()
    subjects  = hc_ids + pd_ids
    grp_names = ["HC", "HC", "PD", "PD"]

    def shorten(name: str) -> str:
        return (name
                .replace("_thickness", "")
                .replace("lh_", "L-")
                .replace("rh_", "R-")
                .replace("Left_", "L-")
                .replace("Right_", "R-"))

    tok_labels = ["CLS"] + [shorten(f) for f in available]
    n_tokens   = len(tok_labels)

    fig, axes = plt.subplots(4, config.NUM_LAYERS, figsize=(14, 24))

    for row_i, (subj, grp) in enumerate(zip(subjects, grp_names)):
        attn_mats = extract_attention_matrix(model, X_all[idx_map[subj]], device)

        for col_i, attn in enumerate(attn_mats):
            ax = axes[row_i, col_i]

            im = ax.imshow(attn, aspect="auto", cmap="viridis", vmin=0, vmax=attn.max())
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor("#c62828" if grp == "PD" else "#1565c0")

            ticks = list(range(0, n_tokens, 20))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([tok_labels[t] for t in ticks], rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels([tok_labels[t] for t in ticks], fontsize=6)
            ax.set_xlabel("Key token (attended to)", fontsize=7)
            ax.set_ylabel("Query token (attending)", fontsize=7)
            ax.set_title(
                f"{grp}  ·  {subj} — Layer {col_i + 1}",
                fontsize=10, fontweight="bold",
                color="#c62828" if grp == "PD" else "#1565c0",
            )
            ax.axhline(0.5, color="white", linewidth=1.2, alpha=0.6)
            ax.axvline(0.5, color="white", linewidth=1.2, alpha=0.6)

    fig.suptitle(
        f"Attention Maps ({n_tokens}×{n_tokens}) per Patient and Layer\n"
        "Rows = query tokens (what attends), Columns = key tokens (what is attended to)\n"
        f"Token 0 = CLS, Tokens 1–{n_tokens - 1} = brain features",
        fontsize=12, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Attention matrix shape: {n_tokens}×{n_tokens}  |  Saved → {out_path}")
