# apply_unet.py

import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from unet_model import UNet

# ------------------- CONFIGURATION -------------------

# Path to a single BraTS case folder (adjust as needed)
CASE_ID   = "BraTS20_Training_001"
BASE_DIR  = "/home/hiranmoy/Downloads/Sameer/Brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
CASE_DIR  = os.path.join(BASE_DIR, CASE_ID)

# Filenames for FLAIR and SEG (SEG not used for inference here)
FLAIR_PATH = os.path.join(CASE_DIR, f"{CASE_ID}_flair.nii")
SEG_PATH   = os.path.join(CASE_DIR, f"{CASE_ID}_seg.nii")

# Which axial slice index to process (0 … 154)
SLICE_IDX = 80

# Output paths
OUT_PROB         = f"{CASE_ID}_pred_prob.png"   # Raw probability heatmap
OUT_MASK         = f"{CASE_ID}_pred_mask.png"   # Binary (>0.5) mask
TRAINED_CHECKPT  = "checkpoints/unet_trained.pth"           # Path to trained weights

# Device ("cuda" or "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------

def load_flair_slice(flair_path, slice_idx):
    """
    Load the 3D FLAIR, extract one axial slice, and normalize it.
    Returns a NumPy array of shape (240, 240).
    """
    if not os.path.isfile(flair_path):
        raise FileNotFoundError(f"Cannot find FLAIR file: {flair_path}")
    flair_nii = nib.load(flair_path)
    flair_vol = flair_nii.get_fdata().astype(np.float32)  # (240,240,155)

    if slice_idx < 0 or slice_idx >= flair_vol.shape[2]:
        raise ValueError(f"slice_idx {slice_idx} out of range [0, {flair_vol.shape[2]-1}]")

    slice_img = flair_vol[:, :, slice_idx]
    # Min-max normalization to [0,1]
    slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-6)
    return slice_norm

def main():
    # 1) Load & normalize FLAIR slice
    flair_slice = load_flair_slice(FLAIR_PATH, SLICE_IDX)
    H, W = flair_slice.shape  # should be (240, 240)

    # 2) Convert to PyTorch tensor of shape (1, 1, H, W)
    input_tensor = torch.from_numpy(flair_slice).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    # Now input_tensor.shape == (1,1,240,240)

    # 3) Instantiate U-Net and load trained weights
    model = UNet(n_channels=1, n_classes=1, use_skip=True).to(DEVICE)
    try:
        model.load_state_dict(torch.load(TRAINED_CHECKPT, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"[Error] Could not load trained checkpoint '{TRAINED_CHECKPT}': {e}")
        return

    # 4) Forward pass
    with torch.no_grad():
        output_logits, _bottleneck = model(input_tensor)  # shape (1,1,240,240)
        prob_map = torch.sigmoid(output_logits)[0, 0]     # shape (240,240), values in [0,1]

    # 5) Convert to NumPy
    prob_np = prob_map.cpu().numpy()
    mask_np = (prob_np > 0.5).astype(np.uint8)  # binary mask

    # 6) Save probability heatmap
    plt.imsave(OUT_PROB, prob_np, cmap='hot')
    print(f"Saved probability map to {OUT_PROB}")

    # 7) Save binary mask
    plt.imsave(OUT_MASK, mask_np, cmap='gray')
    print(f"Saved binary mask to {OUT_MASK}")

if __name__ == "__main__":
    main()
