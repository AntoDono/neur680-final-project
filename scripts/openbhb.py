from huggingface_hub import hf_hub_download

repo = "benoit-dufumier/openBHB"
local_dir = "/home/d1/workspace/neur680-final-project/raw_data/openBHB"

files = [
    "participants.tsv",
    "qc.tsv",
    "resource/freesurfer_channels.txt",
    "resource/freesurfer_atlas-desikan_labels.txt",
    "resource/freesurfer_atlas-destrieux_labels.txt",
    "train/derivatives/freesurfer_roi/desikan_roi_features.csv",
    "train/derivatives/freesurfer_roi/destrieux_roi_features.csv",
    "val/derivatives/freesurfer_roi/desikan_roi_features.csv",
    "val/derivatives/freesurfer_roi/destrieux_roi_features.csv",
]

for f in files:
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=f,
        local_dir=local_dir,
    )
    print(f"✓ {f}")
print("Done.")