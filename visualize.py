"""Brain visualization of transformer attention scores using nilearn.

Produces two images in ``out_dir``:

  attention_brain.png    — glass brain (4 views) with ASEG regions as
                           bubbles; size and colour (plasma scale) encode
                           mean attention score.  Coordinates for the main
                           subcortical structures are derived from the AAL
                           atlas (SPM12) centroid of each region; structures
                           not covered by AAL (ventricles, corpus callosum,
                           WM hypointensities, …) fall back to published MNI
                           literature values.

  attention_ranking.png  — horizontal bar chart of the top-N features ranked
                           by mean attention score.

Called automatically from train.py after the attention ranking step.
"""

import os
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


# ── AAL → ASEG name mapping ───────────────────────────────────────────────────
# Maps each ASEG canonical feature name to the corresponding AAL region label.
# Cerebellum uses multiple sub-regions that are averaged together.
_ASEG_TO_AAL: dict[str, str | list[str]] = {
    "Left_Hippocampus":              "Hippocampus_L",
    "Right_Hippocampus":             "Hippocampus_R",
    "Left_Amygdala":                 "Amygdala_L",
    "Right_Amygdala":                "Amygdala_R",
    "Left_Caudate":                  "Caudate_L",
    "Right_Caudate":                 "Caudate_R",
    "Left_Putamen":                  "Putamen_L",
    "Right_Putamen":                 "Putamen_R",
    "Left_Pallidum":                 "Pallidum_L",
    "Right_Pallidum":                "Pallidum_R",
    "Left_Thalamus":                 "Thalamus_L",
    "Right_Thalamus":                "Thalamus_R",
    "Left_Cerebellum_Cortex": [
        "Cerebelum_Crus1_L", "Cerebelum_Crus2_L", "Cerebelum_3_L",
        "Cerebelum_4_5_L", "Cerebelum_6_L", "Cerebelum_7b_L",
        "Cerebelum_8_L", "Cerebelum_9_L", "Cerebelum_10_L",
    ],
    "Right_Cerebellum_Cortex": [
        "Cerebelum_Crus1_R", "Cerebelum_Crus2_R", "Cerebelum_3_R",
        "Cerebelum_4_5_R", "Cerebelum_6_R", "Cerebelum_7b_R",
        "Cerebelum_8_R", "Cerebelum_9_R", "Cerebelum_10_R",
    ],
    "Left_Cerebellum_White_Matter": [
        "Cerebelum_4_5_L", "Cerebelum_6_L", "Cerebelum_3_L",
    ],
    "Right_Cerebellum_White_Matter": [
        "Cerebelum_4_5_R", "Cerebelum_6_R", "Cerebelum_3_R",
    ],
    "lhCortexVol":  ["Frontal_Sup_L", "Parietal_Sup_L", "Temporal_Sup_L",
                     "Occipital_Sup_L"],
    "rhCortexVol":  ["Frontal_Sup_R", "Parietal_Sup_R", "Temporal_Sup_R",
                     "Occipital_Sup_R"],
    "lhCerebralWhiteMatterVol": ["Frontal_Mid_L", "Parietal_Inf_L"],
    "rhCerebralWhiteMatterVol": ["Frontal_Mid_R", "Parietal_Inf_R"],
}

# Fallback MNI coordinates for structures not covered by AAL
# (ventricles, corpus callosum, WM lesion volumes, optic chiasm, …)
_FALLBACK_COORDS: dict[str, tuple[int, int, int]] = {
    "Left_Lateral_Ventricle":        (-20,   5,  15),
    "Right_Lateral_Ventricle":       ( 20,   5,  15),
    "Left_Inf_Lat_Vent":             (-16,   0,  10),
    "Right_Inf_Lat_Vent":            ( 16,   0,  10),
    "3rd_Ventricle":                 (  0, -10,   0),
    "4th_Ventricle":                 (  0, -36, -32),
    "5th_Ventricle":                 (  0, -10,  -4),
    "Brain_Stem":                    (  0, -28, -32),
    "Left_VentralDC":                (-12, -24,  -6),
    "Right_VentralDC":               ( 12, -24,  -6),
    "Left_Accumbens_area":           (-10,  10,  -4),
    "Right_Accumbens_area":          ( 10,  10,  -4),
    "Left_choroid_plexus":           (-22,  -4,  14),
    "Right_choroid_plexus":          ( 22,  -4,  14),
    "Left_vessel":                   (-18,  -6,  10),
    "Right_vessel":                  ( 18,  -6,  10),
    "Left_WM_hypointensities":       (-25,   0,  20),
    "Right_WM_hypointensities":      ( 25,   0,  20),
    "Left_non_WM_hypointensities":   (-22,   2,  18),
    "Right_non_WM_hypointensities":  ( 22,   2,  18),
    "WM_hypointensities":            (  0,   0,  20),
    "non_WM_hypointensities":        (  0,   2,  18),
    "Optic_Chiasm":                  (  0,   2,  -6),
    "CC_Anterior":                   (  0,  20,  20),
    "CC_Mid_Anterior":               (  0,  14,  22),
    "CC_Central":                    (  0,   5,  20),
    "CC_Mid_Posterior":              (  0,  -5,  20),
    "CC_Posterior":                  (  0, -20,  18),
    "lhSurfaceHoles":                (-35,   0,  15),
    "rhSurfaceHoles":                ( 35,   0,  15),
    "SurfaceHoles":                  (  0,   0,  15),
}


# ── Public API ────────────────────────────────────────────────────────────────

def visualize_attention(
    ranked_features: list[tuple[str, float]],
    out_dir: str = "attention_analysis",
    top_n_bar: int = 30,
) -> None:
    """
    Produce brain visualization and ranking bar chart from attention scores.

    Parameters
    ----------
    ranked_features : list of (feature_name, mean_attention_score)
        Sorted highest-first, as returned by print_attention_ranking().
    out_dir : str
        Directory where output PNGs are written (created if absent).
    top_n_bar : int
        How many top features to include in the bar chart.
    """
    os.makedirs(out_dir, exist_ok=True)
    scores   = {feat: score for feat, score in ranked_features}
    coords   = _build_mni_coords()
    has_aal  = _aal_available()

    _plot_glass_brain(scores, coords, out_dir, used_aal=has_aal)
    _plot_bar_chart(ranked_features, coords, top_n_bar, out_dir)
    print(f"[Visualize] Brain maps saved → {out_dir}/")


# ── Atlas coordinate loading ──────────────────────────────────────────────────

def _aal_available() -> bool:
    try:
        _aal_centroids()
        return True
    except Exception:
        return False


def _aal_centroids() -> dict[str, tuple[int, int, int]]:
    """Return {aal_label: (mx, my, mz)} from the cached AAL atlas."""
    import nibabel as nib
    import requests
    import urllib3
    from nilearn import datasets, image

    urllib3.disable_warnings()
    _orig = requests.Session.send
    requests.Session.send = lambda s, *a, **kw: _orig(s, *a, **{**kw, "verify": False})

    aal  = datasets.fetch_atlas_aal(version="SPM12")
    img  = nib.load(aal.maps)
    data = img.get_fdata()
    aff  = img.affine

    out: dict[str, tuple[int, int, int]] = {}
    for idx_str, label in zip(aal.indices, aal.labels):
        vox = np.argwhere(data == int(idx_str))
        if len(vox) == 0:
            continue
        cx, cy, cz = vox.mean(axis=0)
        mx, my, mz = image.coord_transform(cx, cy, cz, aff)
        out[label] = (int(round(mx)), int(round(my)), int(round(mz)))
    return out


def _build_mni_coords() -> dict[str, tuple[int, int, int]]:
    """
    Build the full ASEG → MNI coordinate map.

    Priority:
    1. AAL atlas centroids (computed from the downloaded atlas image).
    2. Hardcoded fallback values for structures AAL doesn't cover.
    """
    try:
        aal = _aal_centroids()
    except Exception as exc:
        warnings.warn(f"[Visualize] Could not load AAL atlas ({exc}); "
                      "using fallback coordinates.", stacklevel=2)
        return dict(_FALLBACK_COORDS)

    coords: dict[str, tuple[int, int, int]] = {}

    for aseg_name, aal_ref in _ASEG_TO_AAL.items():
        if isinstance(aal_ref, str):
            if aal_ref in aal:
                coords[aseg_name] = aal[aal_ref]
        else:
            pts = [aal[r] for r in aal_ref if r in aal]
            if pts:
                arr = np.array(pts)
                coords[aseg_name] = tuple(int(round(v)) for v in arr.mean(axis=0))

    # Merge fallback (don't overwrite AAL values)
    for name, coord in _FALLBACK_COORDS.items():
        coords.setdefault(name, coord)

    return coords


# ── Glass brain ───────────────────────────────────────────────────────────────

def _plot_glass_brain(
    scores: dict[str, float],
    coords: dict[str, tuple[int, int, int]],
    out_dir: str,
    used_aal: bool,
) -> None:
    from nilearn import plotting

    localizable = [
        (feat, coord)
        for feat, coord in coords.items()
        if feat in scores
    ]
    if not localizable:
        print("[Visualize] No localizable ASEG features — skipping glass brain.")
        return

    feats, marker_coords = zip(*localizable)
    raw_scores = np.array([scores[f] for f in feats])
    s_min, s_max = raw_scores.min(), raw_scores.max()
    norm_scores  = (raw_scores - s_min) / max(s_max - s_min, 1e-10)

    sizes  = 15 + 285 * norm_scores
    cmap   = colormaps["plasma"]
    colors = [cmap(s) for s in norm_scores]

    source_note = "AAL SPM12 atlas centroids" if used_aal else "literature MNI values"
    fig = plt.figure(figsize=(16, 5))
    display = plotting.plot_glass_brain(
        None,
        display_mode="lyrz",
        title=(
            "Mean Transformer Attention per Brain Region — ASEG features\n"
            f"(bubble size & colour ∝ attention · coordinates from {source_note})"
        ),
        figure=fig,
        axes=fig.add_axes([0, 0.05, 0.90, 0.88]),
    )
    display.add_markers(
        marker_coords=list(marker_coords),
        marker_color=colors,
        marker_size=sizes.tolist(),
    )

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    norm    = mcolors.Normalize(vmin=s_min, vmax=s_max)
    cb      = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="plasma"), cax=cbar_ax,
    )
    cb.set_label("Mean attention score", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out = os.path.join(out_dir, "attention_brain.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Visualize] Glass brain    → {out}  ({len(localizable)} regions)")


# ── Bar chart ─────────────────────────────────────────────────────────────────

def _plot_bar_chart(
    ranked_features: list[tuple[str, float]],
    coords: dict[str, tuple[int, int, int]],
    top_n: int,
    out_dir: str,
) -> None:
    top    = ranked_features[:top_n]
    labels = [_shorten(f) for f, _ in reversed(top)]
    vals   = [s            for _, s in reversed(top)]
    orig   = [f            for f, _ in reversed(top)]

    bar_colors = ["#c62828" if f in coords else "#1565c0" for f in orig]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.33)))
    bars = ax.barh(labels, vals, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + max(vals) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=7, color="#333333",
        )

    ax.set_xlabel("Mean attention score", fontsize=10)
    ax.set_title(
        f"Top-{top_n} Features by Mean Transformer Attention\n"
        "  ■ red = mapped to glass brain   ■ blue = global / demographic feature",
        fontsize=10, loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlim(right=max(vals) * 1.12)
    fig.tight_layout()

    out = os.path.join(out_dir, "attention_ranking.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Bar chart      → {out}")


def _shorten(name: str) -> str:
    return (
        name
        .replace("_thickness", "")
        .replace("lh_", "L·")
        .replace("rh_", "R·")
        .replace("Left_", "L·")
        .replace("Right_", "R·")
        .replace("Cerebellum_", "Cb·")
        .replace("EstimatedTotalIntraCranialVol", "eTIV")
    )
