"""Load and preprocess the PPMI dataset.

Labels come from the PPMI Curated Data Cut xlsx (subgroup column):
  PD  = confirmed PD diagnosis (Sporadic PD, LRRK2, GBA, PRKN, PINK1, SNCA, …)
  HC  = Healthy Control / Normosmic
  Excluded = prodromal-only (Hyposmia, RBD, SWEDD) and uncertain groups
"""

import glob
import os

import pandas as pd

from src.data.normalize import canonicalize_ppmi_aseg

RAW_DIR = "raw_data/ppmi"

# Subgroups treated as confirmed PD
_PD_SUBGROUPS = {
    "Sporadic PD", "LRRK2", "GBA", "PRKN", "PINK1", "SNCA",
    "PARK7", "VPS35", "Other GV",
    "LRRK2 + GBA", "LRRK2 + VPS35", "LRRK2 + PINK1",
    "LRRK2 + GBA + SNCA + PRKN + Normosmic",
    "LRRK2 + GBA + Normosmic", "GBA + Normosmic", "GBA + Other GV",
    "PARK7 + RBD",   # genetic PD with prodromal feature → call PD
}

# Subgroups treated as healthy control
_HC_SUBGROUPS = {"Healthy Control", "Normosmic"}

# Everything else (Hyposmia, RBD, SWEDD, PRKN + RBD, …) is excluded


def load_ppmi(raw_dir: str = RAW_DIR) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the PPMI baseline dataset.

    Returns
    -------
    df : pd.DataFrame
        Index = PATNO (str, prefixed 'ppmi-').
        Columns: 'label', 'Age', 'Sex', <aseg_features>
    aseg_cols : list[str]
        Canonical ASEG column names present in df.
    """
    aseg_file    = _find_file(raw_dir, "FS7_ASEG_VOL*.csv")
    curated_file = _find_file(raw_dir, "PPMI_Curated_Data_Cut*.xlsx")

    aseg    = pd.read_csv(aseg_file)
    curated = pd.read_excel(curated_file, sheet_name=0)

    # Keep only baseline scans in both tables
    aseg    = aseg[aseg["EVENT_ID"] == "BL"].copy()
    curated = curated[curated["EVENT_ID"] == "BL"].drop_duplicates("PATNO")

    # ── Labels from curated subgroup ──────────────────────────────────────
    subgroup_map = {}
    for _, row in curated[["PATNO", "subgroup"]].iterrows():
        sg = row["subgroup"]
        if sg in _PD_SUBGROUPS:
            subgroup_map[row["PATNO"]] = 1
        elif sg in _HC_SUBGROUPS:
            subgroup_map[row["PATNO"]] = 0
        # else: excluded (prodromal / uncertain)

    aseg["label"] = aseg["PATNO"].map(subgroup_map)
    aseg = aseg.dropna(subset=["label"])
    aseg["label"] = aseg["label"].astype(int)

    # ── Age and Sex from curated file (pre-computed, no NaN at BL) ────────
    demo = curated[["PATNO", "age", "SEX"]].set_index("PATNO")
    aseg = aseg.join(demo, on="PATNO", how="left")
    aseg["Age"] = aseg["age"]
    aseg["Sex"] = aseg["SEX"].fillna(0).astype(int)

    # ── ASEG features ─────────────────────────────────────────────────────
    aseg_df   = canonicalize_ppmi_aseg(aseg)
    aseg_cols = aseg_df.columns.tolist()

    # ── Assemble ──────────────────────────────────────────────────────────
    meta   = aseg[["label", "Age", "Sex"]].copy()
    result = pd.concat([meta.reset_index(drop=True),
                        aseg_df.reset_index(drop=True)], axis=1)
    result.index = "ppmi-" + aseg["PATNO"].astype(str).values
    result.index.name = "subject_id"
    result = result.dropna()

    _print_summary("PPMI", result)
    return result, aseg_cols


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_file(directory: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory!r}")
    return sorted(matches)[-1]


def _print_summary(name: str, df: pd.DataFrame) -> None:
    n_pd = int(df["label"].sum())
    n_hc = int((1 - df["label"]).sum())
    print(f"[{name}] {len(df)} subjects | PD: {n_pd} | HC: {n_hc} | features: {df.shape[1] - 3}")
