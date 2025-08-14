import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Paths to one BraTS case
flair_path = "/home/hiranmoy/Downloads/Sameer/Brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"
seg_path = "/home/hiranmoy/Downloads/Sameer/Brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii"

# Load FLAIR and segmentation mask
flair_nifti = nib.load(flair_path)
seg_nifti = nib.load(seg_path)

flair = flair_nifti.get_fdata()
seg = seg_nifti.get_fdata()

# Select the central axial slice (can change axis or slice index)
slice_index = flair.shape[2] // 2
flair_slice = flair[:, :, slice_index]
seg_slice = seg[:, :, slice_index]

# Normalize image for visualization
flair_slice = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min())

# Plot with segmentation mask overlaid
plt.figure(figsize=(10, 5))

plt.imshow(flair_slice, cmap='gray')
# Transparent overlay of segmentation mask
plt.imshow(seg_slice, cmap='Reds', alpha=0.4)

plt.title("FLAIR with Segmentation Overlay")
plt.axis('off')
plt.colorbar(label="Segmentation Label")
plt.savefig("/home/hiranmoy/Downloads/Sameer/unet_brats_analysis/Visuals/flair_seg_overlay_001.png", bbox_inches='tight')

