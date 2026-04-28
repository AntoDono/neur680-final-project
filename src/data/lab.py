"""Load and preprocess the lab (in-house FreeSurfer) dataset."""

import os

import pandas as pd

from src.data.normalize import canonicalize_lab_aseg, aseg_common_features

RAW_DIR       = "raw_data/lab"
PROCESSED_DIR = "processed_data"


def load_lab(
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Load the lab dataset and return a processed DataFrame.

    Returns
    -------
    df : pd.DataFrame
        Index = subject_id.
        Columns: 'label', 'Age', 'Sex', <aseg_features>, <thickness_features>
    aseg_cols : list[str]
        Canonical ASEG column names present in df.
    thickness_cols : list[str]
        Cortical thickness column names present in df.
    """
    aseg  = pd.read_csv(f"{raw_dir}/aseg_volumes.txt",    sep="\t", index_col=0)
    demo  = pd.read_csv(f"{raw_dir}/demographic_clean.txt", sep="\t", index_col=0)
    lh    = pd.read_csv(f"{raw_dir}/lh_thickness.txt",    sep="\t", index_col=0)
    rh    = pd.read_csv(f"{raw_dir}/rh_thickness.txt",    sep="\t", index_col=0)

    for frame in (aseg, demo, lh, rh):
        frame.index.name = "subject_id"

    # Merge everything (copy to defragment before any assignments)
    df = demo.join(aseg, how="inner").join(lh, how="inner", rsuffix="_lh").join(rh, how="inner", rsuffix="_rh").copy()

    # Labels and demographics
    df["label"] = (df["Group"] == "PD").astype(int)
    df["Sex"]   = (df["Sex"]   == "M").astype(int)
    df = df.drop(columns=["Code", "Group"])

    # Canonical ASEG columns
    aseg_df   = canonicalize_lab_aseg(df[[c for c in df.columns if c in aseg.columns or c.replace("-", "_") in aseg_common_features()]])
    aseg_cols = aseg_df.columns.tolist()

    # Cortical thickness columns (lh_ and rh_ thickness, strip the _thickness suffix for consistency)
    thickness_raw = [c for c in df.columns if c.endswith("_thickness")]
    thickness_df  = df[thickness_raw].copy()
    # Normalise: drop duplicated BrainSegVolNotVent / eTIV carried in the thickness files
    thickness_df  = thickness_df.loc[:, ~thickness_df.columns.duplicated()]
    thickness_cols = thickness_df.columns.tolist()

    # Assemble final df, drop any columns that are all-NaN
    meta = df[["label", "Age", "Sex"]]
    result = pd.concat([meta, aseg_df, thickness_df], axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    result = result.dropna(axis=1, how="all")

    _save_processed(result, source="lab", processed_dir=processed_dir)
    _print_summary("Lab", result)
    return result, aseg_cols, thickness_cols


def _save_processed(df: pd.DataFrame, source: str, processed_dir: str) -> None:
    out = df.copy()
    out.insert(0, "source", source)
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, f"{source}.csv")
    out.to_csv(path)
    print(f"[Lab] Processed data saved → {path}")


def _print_summary(name: str, df: pd.DataFrame) -> None:
    n_pd  = int(df["label"].sum())
    n_hc  = int((1 - df["label"]).sum())
    print(f"[{name}] {len(df)} subjects | PD: {n_pd} | HC: {n_hc} | features: {df.shape[1] - 3}")
