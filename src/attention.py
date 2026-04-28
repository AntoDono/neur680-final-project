"""Attention map extraction and visualization for TabularTransformer."""

import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    list of np.ndarray, shape (n_features, n_features) per layer.
    attn[i, j] = weight token-i places on token-j.
    """
    model.eval()
    x = torch.tensor(x_scaled_1d[np.newaxis, :], dtype=torch.float32).to(device)
    attn_matrices = []

    with torch.no_grad():
        B, F = x.shape
        h = model.feature_proj(x.unsqueeze(-1))   # (1, F, d_model)

        for layer in model.transformer.layers:
            h2, attn_w = layer.self_attn(
                h, h, h,
                need_weights=True,
                average_attn_weights=True,   # (1, F, F)
            )
            attn_matrices.append(attn_w[0].cpu().numpy())   # (F, F)

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
    test_subject_ids: list[str] | None = None,
    metrics: dict | None = None,
) -> None:
    """
    Plot full (N_tokens × N_tokens) attention heatmaps for 2 HC and 2 PD subjects
    drawn from the test split.
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

    # Restrict candidate pool to test-set subjects when provided
    if test_subject_ids is not None:
        test_set = set(test_subject_ids)
        test_groups = groups[groups.index.isin(test_set)]
    else:
        test_groups = groups

    hc_id = test_groups[test_groups == "HC"].index[0]
    pd_id = test_groups[test_groups == "PD"].index[0]
    subjects  = [hc_id, pd_id]
    grp_names = ["HC", "PD"]
    colors    = {"HC": "#1565c0", "PD": "#c62828"}

    def shorten(name: str) -> str:
        return (name
                .replace("_thickness", "")
                .replace("lh_", "L-")
                .replace("rh_", "R-")
                .replace("Left_", "L-")
                .replace("Right_", "R-"))

    tok_labels = [shorten(f) for f in available]
    n_tokens   = len(tok_labels)

    # Numeric tick labels on the heatmap axes (every 20 tokens)
    ticks      = list(range(0, n_tokens, 20))
    num_labels = [str(t) for t in ticks]

    # Legend layout: n_tokens entries arranged in legend_cols columns
    legend_cols = 6
    legend_rows = -(-n_tokens // legend_cols)           # ceiling division
    legend_h    = max(3.5, legend_rows * 0.145 + 0.9)

    # rows = layers, cols = subjects (HC | PD)
    cell_size   = 4
    metrics_h   = 1.1      # thin strip: title + eval metrics
    header_h    = 4.0      # hyperparameter panel
    total_h     = metrics_h + header_h + cell_size * config.NUM_LAYERS + legend_h

    fig = plt.figure(figsize=(cell_size * 2 + 2, total_h))

    # GridSpec layout:
    #   row 0        → metrics / title strip
    #   row 1        → hyperparameter banner
    #   rows 2..N+1  → attention heatmaps
    #   row  N+2     → feature legend
    gs = gridspec.GridSpec(
        config.NUM_LAYERS + 3, 2,
        figure=fig,
        height_ratios=(
            [metrics_h / cell_size, header_h / cell_size]
            + [1] * config.NUM_LAYERS
            + [legend_h / cell_size]
        ),
        hspace=0.3,
        wspace=0.35,
    )

    # ── Metrics / title strip (row 0, spans both columns) ─────────────────
    ax_met = fig.add_subplot(gs[0, :])
    ax_met.set_axis_off()
    ax_met.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_met.transAxes,
        facecolor="white", edgecolor="#bbbbbb", linewidth=1.2,
        clip_on=False,
    ))

    # Main title
    ax_met.text(
        0.5, 0.92,
        f"Attention Maps ({n_tokens} feature tokens) — All {config.NUM_LAYERS} Layers"
        "        Left: HC  ·  Right: PD",
        transform=ax_met.transAxes,
        ha="center", va="top", fontsize=11, fontweight="bold", color="#1a1a2e",
    )

    # Eval metrics row (shown only if metrics were passed in)
    if metrics:
        acc  = metrics.get("accuracy", float("nan"))
        loss = metrics.get("best_loss", float("nan"))
        hc_p = metrics.get("hc_precision", float("nan"))
        hc_r = metrics.get("hc_recall",    float("nan"))
        hc_f = metrics.get("hc_f1",        float("nan"))
        pd_p = metrics.get("pd_precision", float("nan"))
        pd_r = metrics.get("pd_recall",    float("nan"))
        pd_f = metrics.get("pd_f1",        float("nan"))

        metric_str = (
            f"Accuracy {acc:.3f}    Loss {loss:.4f}"
            f"        HC  P {hc_p:.2f}  R {hc_r:.2f}  F1 {hc_f:.2f}"
            f"        PD  P {pd_p:.2f}  R {pd_r:.2f}  F1 {pd_f:.2f}"
        )
        ax_met.text(
            0.5, 0.30,
            metric_str,
            transform=ax_met.transAxes,
            ha="center", va="center", fontsize=8.5, color="#333333",
            fontfamily="monospace",
        )

    # ── Hyperparameter header (row 1, spans both columns) ─────────────────
    ax_hdr = fig.add_subplot(gs[1, :])
    ax_hdr.set_axis_off()

    # Background rectangle
    ax_hdr.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_hdr.transAxes,
        facecolor="white", edgecolor="#bbbbbb", linewidth=1.5,
        clip_on=False,
    ))

    # Title line — sits just inside the top edge
    ax_hdr.text(
        0.5, 0.93,
        "TabularTransformer — Model Hyperparameters",
        transform=ax_hdr.transAxes,
        ha="center", va="top",
        fontsize=13, fontweight="bold", color="#1a1a2e",
        fontfamily="monospace",
    )

    # Divider — tight under the title
    ax_hdr.axhline(0.80, color="#1565c0", linewidth=1.2, xmin=0.02, xmax=0.98)

    # Hyperparameter entries — two columns of key/value pairs
    model_params = [
        ("d_model",     config.D_MODEL),
        ("n_heads",     config.NHEAD),
        ("n_layers",    config.NUM_LAYERS),
        ("dropout",     config.DROPOUT),
    ]
    train_params = [
        ("batch_size",  config.BATCH_SIZE),
        ("pretrain_ep", config.PRETRAIN_EPOCHS),
        ("finetune_ep", config.FINETUNE_EPOCHS),
        ("lr",          config.LR),
        ("test_size",   config.TEST_SIZE),
    ]

    def _fmt_val(v):
        if isinstance(v, float) and v < 0.01:
            return f"{v:.0e}"
        return str(v)

    row_step = 0.13

    # Left column — architecture
    ax_hdr.text(
        0.05, 0.72, "Architecture",
        transform=ax_hdr.transAxes,
        ha="left", va="top", fontsize=9.5, color="#1565c0",
        fontweight="bold", fontfamily="monospace",
    )
    for i, (k, v) in enumerate(model_params):
        ax_hdr.text(
            0.05, 0.58 - i * row_step,
            f"{k:<14}  {_fmt_val(v)}",
            transform=ax_hdr.transAxes,
            ha="left", va="top", fontsize=9, color="#222222",
            fontfamily="monospace",
        )

    # Right column — training
    ax_hdr.text(
        0.55, 0.72, "Training",
        transform=ax_hdr.transAxes,
        ha="left", va="top", fontsize=9.5, color="#1565c0",
        fontweight="bold", fontfamily="monospace",
    )
    for i, (k, v) in enumerate(train_params):
        ax_hdr.text(
            0.55, 0.58 - i * row_step,
            f"{k:<14}  {_fmt_val(v)}",
            transform=ax_hdr.transAxes,
            ha="left", va="top", fontsize=9, color="#222222",
            fontfamily="monospace",
        )

    # Timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ax_hdr.text(
        0.02, 0.04,
        f"Generated  {ts}",
        transform=ax_hdr.transAxes,
        ha="left", va="bottom", fontsize=7.5, color="#777799",
        fontfamily="monospace", style="italic",
    )

    # Vertical divider between the two param columns
    ax_hdr.axvline(0.5, color="#ccccdd", linewidth=0.8, ymin=0.08, ymax=0.78)

    # ── Attention heatmaps ────────────────────────────────────────────────
    attn_by_subj = {
        subj: extract_attention_matrix(model, X_all[idx_map[subj]], device)
        for subj in subjects
    }

    axes = np.empty((config.NUM_LAYERS, 2), dtype=object)
    for layer_i in range(config.NUM_LAYERS):
        for col_i in range(2):
            axes[layer_i, col_i] = fig.add_subplot(gs[layer_i + 2, col_i])

    for layer_i in range(config.NUM_LAYERS):
        for col_i, (subj, grp) in enumerate(zip(subjects, grp_names)):
            ax    = axes[layer_i, col_i]
            attn  = attn_by_subj[subj][layer_i]
            color = colors[grp]

            im = ax.imshow(attn, aspect="auto", cmap="viridis", vmin=0, vmax=attn.max())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor(color)

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(num_labels, rotation=0, ha="center", fontsize=6)
            ax.set_yticklabels(num_labels, fontsize=6)

            if layer_i == config.NUM_LAYERS - 1:
                ax.set_xlabel("Key token (attended to)", fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"Layer {layer_i + 1}\nQuery token", fontsize=8)

            ax.set_title(
                f"{grp}  ·  {subj} — Layer {layer_i + 1}",
                fontsize=9, fontweight="bold", color=color,
            )
            ax.axhline(0.5, color="white", linewidth=1, alpha=0.6)
            ax.axvline(0.5, color="white", linewidth=1, alpha=0.6)

    # ── Feature index legend (spans both columns) ─────────────────────────
    ax_leg = fig.add_subplot(gs[config.NUM_LAYERS + 2, :])
    ax_leg.set_axis_off()

    ax_leg.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_leg.transAxes,
        facecolor="white", edgecolor="#bbbbbb", linewidth=1.2,
        clip_on=False,
    ))

    ax_leg.text(
        0.5, 0.97,
        "Feature Index Legend",
        transform=ax_leg.transAxes,
        ha="center", va="top",
        fontsize=10, fontweight="bold", color="#1a1a2e",
        fontfamily="monospace",
    )
    ax_leg.axhline(0.90, color="#1565c0", linewidth=1.0, xmin=0.01, xmax=0.99)

    # Lay out entries left-to-right, top-to-bottom across legend_cols columns
    col_w   = 1.0 / legend_cols
    row_h   = 0.86 / legend_rows          # fraction of axes height per row
    pad_top = 0.88                         # starting y position

    for idx, label in enumerate(tok_labels):
        col_i = idx % legend_cols
        row_i = idx // legend_cols
        x = col_i * col_w + 0.01
        y = pad_top - row_i * row_h

        # Index number in accent colour
        ax_leg.text(
            x, y, f"{idx:>3}",
            transform=ax_leg.transAxes,
            ha="left", va="top", fontsize=5.2, color="#1565c0",
            fontfamily="monospace",
        )
        # Feature name in dark colour
        ax_leg.text(
            x + col_w * 0.22, y, label,
            transform=ax_leg.transAxes,
            ha="left", va="top", fontsize=5.2, color="#333333",
            fontfamily="monospace",
        )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Attention matrix shape: {n_tokens}×{n_tokens}  |  Saved → {out_path}")


def print_attention_ranking(
    model,
    scaler: StandardScaler,
    feature_cols: list[str],
    device: torch.device,
    test_subject_ids: list[str] | None = None,
) -> None:
    """
    Print features ranked by mean attention received, averaged across all
    transformer layers and all test subjects (most → least attended).

    attn[i, j] = weight query-i places on key-j, so column-j sum gives
    total attention token-j receives from all other tokens.
    """
    raw = pd.read_csv("processed_data/patient_data.csv", index_col=0)
    raw2 = raw.drop(columns=[c for c in ["Code", "Group", "BrainSegVolNotVent_y",
                                          "eTIV_x", "eTIV_y"]
                              if c in raw.columns]).copy()
    raw2["Sex"] = (raw2["Sex"] == "M").astype(int)

    aligned = pd.DataFrame(0.0, index=raw2.index, columns=feature_cols)
    for c in feature_cols:
        if c in raw2.columns:
            aligned[c] = raw2[c].values
    X_all = scaler.transform(aligned.values.astype("float32"))
    idx_map = {sid: i for i, sid in enumerate(raw.index)}

    subject_ids = test_subject_ids if test_subject_ids is not None else list(raw.index)
    subject_ids = [s for s in subject_ids if s in idx_map]

    # Accumulate column-sum attention per feature across all subjects × layers
    # feature tokens are indices 1..n_features (index 0 is CLS)
    n_features = len(feature_cols)
    total_attention = np.zeros(n_features)
    count = 0

    for sid in subject_ids:
        attn_matrices = extract_attention_matrix(model, X_all[idx_map[sid]], device)
        for attn in attn_matrices:                  # attn: (F, F)
            total_attention += attn.sum(axis=0)          # sum over all query rows
            count += 1

    mean_attention = total_attention / count

    ranked = sorted(
        zip(feature_cols, mean_attention),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n" + "=" * 60)
    print(f"Feature attention ranking  ({len(subject_ids)} subjects, "
          f"{count // len(subject_ids)} layers each)")
    print(f"{'Rank':<6}{'Feature':<45}{'Mean Attention':>14}")
    print("-" * 65)
    for rank, (feat, score) in enumerate(ranked, 1):
        print(f"{rank:<6}{feat:<45}{score:>14.6f}")
    print("=" * 60)
