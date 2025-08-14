import nibabel as nib
import numpy as np
import os

def load_case(flair_path, seg_path):
    flair_nii = nib.load(flair_path)
    seg_nii = nib.load(seg_path)
    flair = flair_nii.get_fdata().astype(np.float32)
    seg = seg_nii.get_fdata().astype(np.int32)
    brain_mask = flair > 0
    mean, std = flair[brain_mask].mean(), flair[brain_mask].std()
    flair_norm = (flair - mean) / (std + 1e-6)
    return flair_norm, seg

def extract_slices(flair, seg):
    X_slices, y_slices = [], []
    for z in range(flair.shape[2]):
        slice_img = flair[:, :, z]
        slice_seg = seg[:, :, z]
        if np.any(slice_seg):
            X_slices.append(slice_img[..., np.newaxis])
            y_slices.append((slice_seg > 0).astype(np.int32))
    return np.stack(X_slices), np.stack(y_slices)