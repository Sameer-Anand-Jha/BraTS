# main.py

import csv
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from unet_model import UNet
from data_loader import load_case, extract_slices
from evaluate_feature_maps import evaluate_similarity

# ------------------------ USER CONFIGURATION ------------------------

# 1) Path to the folder containing BraTS case subfolders
BASE_DIR = "/home/hiranmoy/Downloads/Sameer/Brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# 2) Limit number of cases to process (None = all)
MAX_CASES = None

# 3) Device (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4) Path to output CSV
CSV_PATH = "checkpoints/unet_results.csv"

# 5) Path to the trained U-Net weights (with skip connections)
TRAINED_CHECKPOINT = "checkpoints/unet_trained.pth"
# ---------------------------------------------------------------------

def main():
    # 1) Find all case directories
    case_dirs = sorted([
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("BraTS20_Training_")
    ])

    if not case_dirs:
        print(f"No case folders found in {BASE_DIR}. Exiting.")
        return

    print(f"Found {len(case_dirs)} cases in: {BASE_DIR}\n")

    # 2) Open CSV for writing (overwrite if exists)
    with open(CSV_PATH, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # CSV header
        writer.writerow([
            "CaseID",
            "UniqueLabels",
            "Corr_TrainedWithSkips",
            "Corr_RandomNoSkips"
        ])

        # 3) Loop over each case folder
        for idx, case_path in enumerate(case_dirs):
            if (MAX_CASES is not None) and (idx >= MAX_CASES):
                break

            case_id = os.path.basename(case_path)
            print(f"=== Processing {case_id} ({idx+1}/{len(case_dirs)}) ===")

            # Paths to flair and segmentation .nii files
            flair_path = os.path.join(case_path, f"{case_id}_flair.nii")
            seg_path   = os.path.join(case_path, f"{case_id}_seg.nii")

            # Check existence
            if not os.path.isfile(flair_path):
                print(f"  [Error] Missing FLAIR file: {flair_path}")
                continue
            if not os.path.isfile(seg_path):
                print(f"  [Error] Missing SEG file:   {seg_path}")
                continue

            # 4) Load volumes and normalize flair
            flair_vol, seg_vol = load_case(flair_path, seg_path)
            print(f"  FLAIR shape: {flair_vol.shape}, SEG shape: {seg_vol.shape}")
            unique_labels = np.unique(seg_vol)
            print(f"  Unique SEG labels: {unique_labels}")

            # 5) Extract 2D slices with tumor present
            X_slices, _ = extract_slices(flair_vol, seg_vol)
            if X_slices.shape[0] == 0:
                print(f"  [Warning] No tumor‐containing slice. Skipping.")
                continue

            # 6) Take the first slice and convert to tensor (1,1,240,240)
            first_slice = X_slices[0]  # shape (240,240,1)
            input_tensor = torch.from_numpy(
                first_slice.transpose((2, 0, 1))
            ).unsqueeze(0).float().to(DEVICE)

            # 7) Instantiate two U-Nets:
            #    a) Trained with skips
            model_with_skips = UNet(n_channels=1, n_classes=1, use_skip=True).to(DEVICE)
            #    Load trained weights
            try:
                model_with_skips.load_state_dict(torch.load(TRAINED_CHECKPOINT, map_location=DEVICE))
                model_with_skips.eval()
            except Exception as e:
                print(f"  [Error] Could not load trained checkpoint '{TRAINED_CHECKPOINT}': {e}")
                return

            #    b) Random no-skips (for comparison)
            model_without_skips = UNet(n_channels=1, n_classes=1, use_skip=False).to(DEVICE)
            model_without_skips.eval()

            # 8) Compute correlations
            try:
                corr_trained   = evaluate_similarity(model_with_skips,    input_tensor)
                corr_random_ns = evaluate_similarity(model_without_skips, input_tensor)
            except Exception as e:
                print(f"  [Error] Feature evaluation failed: {e}\n")
                continue

            print(f"  Corr (trained w/ skips)    = {corr_trained:.6f}")
            print(f"  Corr (random no skips)     = {corr_random_ns:.6f}\n")

            # 9) Write row to CSV
            writer.writerow([
                case_id,
                ";".join(map(str, unique_labels)),
                f"{corr_trained:.6f}",
                f"{corr_random_ns:.6f}"
            ])

    print(f"\nAll done. Results saved to '{CSV_PATH}'.")


if __name__ == "__main__":
    main()
