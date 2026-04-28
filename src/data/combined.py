"""Merge PPMI, Lab, and OASIS-3 datasets on shared ASEG features for training.

Each loader (load_ppmi, load_lab, load_oasis3) processes its raw source and
writes a per-source CSV to processed_data/.  This module reads those results,
aligns them on the common canonical ASEG feature set, and imputes missing
Age/Sex for OASIS-3 rows (not present in that export).

ComBat harmonization is NOT applied here.  It is applied inside make_loaders()
(src/data/loaders.py) *after* the train/test split so that batch-effect
correction parameters are estimated from training subjects only and applied
as frozen transforms to the held-out test set.

OASIS-3 degrades gracefully: if no valid data has been downloaded yet,
load_oasis3() returns an empty DataFrame and training continues on PPMI + Lab.
"""

import pandas as pd

from src import config
from src.data.lab import load_lab
from src.data.oasis3 import load_oasis3
from src.data.ppmi import load_ppmi


def load_combined() -> tuple[pd.DataFrame, list[str]]:
    """
    Load and merge PPMI, Lab, and OASIS-3 on their common ASEG features.

    Processing steps
    ----------------
    1. Call each source loader (which also writes processed_data/<source>.csv).
    2. Find the canonical ASEG columns present in *all non-empty* sources.
    3. Impute Age / Sex NaN for OASIS-3 rows using the median from PPMI + Lab.
    4. Concatenate and drop any remaining NaN rows in feature columns.
    5. Apply ComBat harmonization if config.USE_COMBAT is True.

    Returns
    -------
    df : pd.DataFrame
        Index = subject_id.
        Columns: 'label', 'source', <common_aseg_features>, 'Age', 'Sex'
    feature_cols : list[str]
        Ordered list of feature column names (common ASEG + Age + Sex).
    """
    ppmi_df,  ppmi_aseg      = load_ppmi()
    lab_df,   lab_aseg, _    = load_lab()
    oasis_df, oasis_aseg     = load_oasis3()

    has_oasis = len(oasis_df) > 0

    # ── Common ASEG features across all present sources ───────────────────
    active_aseg_sets = [set(ppmi_aseg), set(lab_aseg)]
    if has_oasis:
        active_aseg_sets.append(set(oasis_aseg))

    common_aseg  = sorted(set.intersection(*active_aseg_sets))
    feature_cols = [f for f in common_aseg + ["Age", "Sex"]
                    if f not in config.BLACKLIST_FEATURES]
    keep         = ["label"] + feature_cols

    ppmi_df = ppmi_df[keep].copy()
    ppmi_df["source"] = "ppmi"

    lab_df = lab_df[keep].copy()
    lab_df["source"] = "lab"

    frames = [ppmi_df, lab_df]

    if has_oasis:
        oasis_df = oasis_df[[c for c in keep if c in oasis_df.columns]].copy()
        oasis_df["source"] = "oasis3"

        # Impute Age and Sex with medians from PPMI + Lab (computed on labelled data)
        ref = pd.concat([ppmi_df, lab_df], axis=0)
        if "Age" in feature_cols:
            age_median = ref["Age"].median()
            oasis_df["Age"] = oasis_df["Age"].fillna(age_median)
        if "Sex" in feature_cols:
            sex_median = ref["Sex"].median()
            oasis_df["Sex"] = oasis_df["Sex"].fillna(sex_median)

        frames.append(oasis_df)

    df = pd.concat(frames, axis=0)
    df = df.dropna(subset=feature_cols)

    # ComBat is applied later in make_loaders(), after the train/test split.

    n_ppmi  = int((df["source"] == "ppmi").sum())
    n_lab   = int((df["source"] == "lab").sum())
    n_oasis = int((df["source"] == "oasis3").sum()) if has_oasis else 0
    n_pd    = int(df["label"].sum())
    n_hc    = int((1 - df["label"]).sum())

    oasis_str = f" | OASIS3: {n_oasis}" if has_oasis else " | OASIS3: (not loaded)"
    print(
        f"[Combined] {len(df)} subjects | "
        f"PPMI: {n_ppmi} | Lab: {n_lab}{oasis_str} | "
        f"PD: {n_pd} | HC: {n_hc} | "
        f"features: {len(feature_cols)}"
    )

    return df, feature_cols
