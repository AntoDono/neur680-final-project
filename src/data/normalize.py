"""
Column-name canonicalization across the lab and PPMI datasets.

The lab (FreeSurfer) ASEG files use hyphens as separators
(e.g. 'Left-Lateral-Ventricle'), while the PPMI CSV export uses underscores
('Left_Lateral_Ventricle').  Both come from FreeSurfer 7, so the underlying
measurements are directly comparable once names are aligned.

Canonical form  →  underscore-separated (matches PPMI export).
"""

# Full set of ASEG columns present in both datasets (64 features).
_ASEG_CANONICAL = [
    "3rd_Ventricle", "4th_Ventricle", "5th_Ventricle",
    "Brain_Stem", "BrainSegVol", "BrainSegVol_to_eTIV", "BrainSegVolNotVent",
    "CC_Anterior", "CC_Central", "CC_Mid_Anterior", "CC_Mid_Posterior", "CC_Posterior",
    "CSF", "CerebralWhiteMatterVol", "CortexVol", "EstimatedTotalIntraCranialVol",
    "Left_Accumbens_area", "Left_Amygdala", "Left_Caudate",
    "Left_Cerebellum_Cortex", "Left_Cerebellum_White_Matter",
    "Left_Hippocampus", "Left_Inf_Lat_Vent", "Left_Lateral_Ventricle",
    "Left_Pallidum", "Left_Putamen", "Left_Thalamus", "Left_VentralDC",
    "Left_WM_hypointensities", "Left_choroid_plexus",
    "Left_non_WM_hypointensities", "Left_vessel",
    "MaskVol", "MaskVol_to_eTIV", "Optic_Chiasm",
    "Right_Accumbens_area", "Right_Amygdala", "Right_Caudate",
    "Right_Cerebellum_Cortex", "Right_Cerebellum_White_Matter",
    "Right_Hippocampus", "Right_Inf_Lat_Vent", "Right_Lateral_Ventricle",
    "Right_Pallidum", "Right_Putamen", "Right_Thalamus", "Right_VentralDC",
    "Right_WM_hypointensities", "Right_choroid_plexus",
    "Right_non_WM_hypointensities", "Right_vessel",
    "SubCortGrayVol", "SupraTentorialVol", "SupraTentorialVolNotVent",
    "SurfaceHoles", "TotalGrayVol", "WM_hypointensities",
    "lhCerebralWhiteMatterVol", "lhCortexVol", "lhSurfaceHoles",
    "non_WM_hypointensities", "rhCerebralWhiteMatterVol", "rhCortexVol",
    "rhSurfaceHoles",
]


def aseg_common_features() -> list[str]:
    """Return the list of canonical ASEG feature names shared by both datasets."""
    return list(_ASEG_CANONICAL)


def canonicalize_lab_aseg(df):
    """
    Rename lab ASEG DataFrame columns to canonical (underscore) form.
    Also drops duplicate/redundant columns produced by the lab merge pipeline.
    """
    drop = {"BrainSegVolNotVent_x", "BrainSegVolNotVent_y", "eTIV_x", "eTIV_y"}
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # Lab uses hyphens; replace with underscores and fix the one 'to' separator.
    rename = {c: c.replace("-", "_") for c in df.columns}
    df = df.rename(columns=rename)

    # Keep only canonical ASEG columns that are present.
    keep = [c for c in _ASEG_CANONICAL if c in df.columns]
    return df[keep]


def canonicalize_ppmi_aseg(df):
    """
    Select only canonical ASEG columns from the PPMI ASEG DataFrame.
    PPMI already uses underscores, so no renaming is needed.
    """
    keep = [c for c in _ASEG_CANONICAL if c in df.columns]
    return df[keep]
