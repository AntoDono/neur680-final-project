"""Load and preprocess the OASIS-3 healthy control dataset.

OASIS-3 subjects are all cognitively healthy (CDR = 0 throughout the study).
The data lives in a single consolidated CSV:

    raw_data/oasis3/healthy/aseg_volumes.csv

Each row is one FreeSurfer session; rows are matched against the healthy
subject manifest (healthy_oasis3.csv) by freesurfer_id so that only the
sessions designated as HC baselines are included.

Column name differences vs. the canonical ASEG set are resolved by the
_canonicalize_columns() helper below:
  - Structure volumes carry a ``vol_`` prefix that is stripped.
  - FreeSurfer 5.3 used ``Left/Right_Thalamus_Proper``; canonical is
    ``Left/Right_Thalamus``.
  - Several global-measure names differ (ICV → EstimatedTotalIntraCranialVol,
    Cortical → Cerebral for white matter volumes).

All subjects are labelled HC (label = 0).  Age and Sex are not present in
this export; they are left as NaN and imputed from the other datasets inside
load_combined() before training.
"""

import os

import pandas as pd

from src.data.normalize import aseg_common_features

RAW_DIR       = "raw_data/oasis3"
PROCESSED_DIR = "processed_data"

# Explicit column renames for global measures that differ from canonical names.
_GLOBAL_RENAME: dict[str, str] = {
    "ICV":                      "EstimatedTotalIntraCranialVol",
    "lhCorticalWhiteMatterVol": "lhCerebralWhiteMatterVol",
    "rhCorticalWhiteMatterVol": "rhCerebralWhiteMatterVol",
    "CorticalWhiteMatterVol":   "CerebralWhiteMatterVol",
}


def load_oasis3(
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the OASIS-3 healthy control dataset from the consolidated ASEG CSV.

    Parameters
    ----------
    raw_dir : str
        Root of the OASIS-3 raw data tree (contains healthy_oasis3.csv and
        the healthy/ subdirectory with aseg_volumes.csv).
    processed_dir : str
        Directory where processed_data/oasis3.csv will be written.

    Returns
    -------
    df : pd.DataFrame
        Index = freesurfer_id.
        Columns: 'label', 'source', 'Age', 'Sex', <aseg_features>
        label = 0 (all HC), Age / Sex = NaN (not in export).
    aseg_cols : list[str]
        Canonical ASEG column names present in df.
    """
    volumes_path  = os.path.join(raw_dir, "healthy", "aseg_volumes.csv")
    manifest_path = os.path.join(raw_dir, "healthy_oasis3.csv")

    volumes  = pd.read_csv(volumes_path)
    manifest = pd.read_csv(manifest_path)
    healthy_ids: set[str] = set(manifest["freesurfer_id"].tolist())

    # Filter to sessions listed in the healthy manifest
    volumes = volumes[volumes["freesurfer_id"].isin(healthy_ids)].copy()
    volumes = volumes.set_index("freesurfer_id")
    volumes.index.name = "subject_id"
    volumes = volumes.drop(columns=["subject_id"], errors="ignore")

    # Canonicalize column names
    volumes = _canonicalize_columns(volumes)

    # Keep only canonical ASEG columns present in this export
    canonical = aseg_common_features()
    aseg_cols = [c for c in canonical if c in volumes.columns]
    aseg_df   = volumes[aseg_cols].copy()

    # Metadata: all HC, no demographics in this export
    meta = pd.DataFrame(
        {
            "label":  0,
            "source": "oasis3",
            "Age":    float("nan"),
            "Sex":    float("nan"),
        },
        index=aseg_df.index,
    )

    result = pd.concat([meta, aseg_df], axis=1)
    result.index.name = "subject_id"

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "oasis3.csv")
    result.to_csv(out_path)

    n_hc = len(result)
    print(
        f"[OASIS3] {n_hc} subjects | HC: {n_hc} | PD: 0 | "
        f"features: {len(aseg_cols)} | saved → {out_path}"
    )
    return result, aseg_cols


# ── Column canonicalization ────────────────────────────────────────────────────

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename OASIS-3 aseg_volumes.csv columns to match the canonical ASEG names.

    Rules applied in order:
    1. Explicit global-measure renames (_GLOBAL_RENAME dict).
    2. Strip the ``vol_`` prefix from structure-volume columns.
    3. Strip the ``_Proper`` suffix (FreeSurfer 5.3 thalamus label).
    """
    rename: dict[str, str] = {}

    for col in df.columns:
        new = col

        # Step 1: explicit global overrides
        if new in _GLOBAL_RENAME:
            new = _GLOBAL_RENAME[new]
            rename[col] = new
            continue

        # Step 2: strip vol_ prefix
        if new.startswith("vol_"):
            new = new[4:]

        # Step 3: strip _Proper suffix (FreeSurfer 5.3 thalamus naming)
        if new.endswith("_Proper"):
            new = new[: -len("_Proper")]

        if new != col:
            rename[col] = new

    return df.rename(columns=rename)
