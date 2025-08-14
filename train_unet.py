# train_unet.py

import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from unet_model import UNet
from data_loader import load_case, extract_slices

# -------------------- USER CONFIGURATION --------------------

# Path to the root of BraTS cases (each subfolder named BraTS20_Training_XXX)
BASE_DIR = "/home/hiranmoy/Downloads/Sameer/Brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Hyperparameters
BATCH_SIZE = 16
LR         = 1e-4
NUM_EPOCHS = 10
VAL_SPLIT  = 0.2      # 20% of data for validation
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where to save the best model
CHECKPOINT_PATH = "checkpoints/unet_trained.pth"
# ----------------------------------------------------------------


def gather_slices(base_dir):
    """
    Iterates over each BraTS case folder, loads the FLAIR and seg volumes,
    extracts all 2D slices that contain tumor, and returns two NumPy arrays:
      - X: shape (N, 1, 240, 240), dtype=float32, values ∈ [0,1]
      - y: shape (N, 240, 240),       dtype=uint8, values in {0,1}
    """
    X_list = []
    y_list = []

    case_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("BraTS20_Training_")
    ])

    print(f"Found {len(case_dirs)} cases. Gathering slices...")

    for case_path in case_dirs:
        case_id = os.path.basename(case_path)
        flair_path = os.path.join(case_path, f"{case_id}_flair.nii")
        seg_path   = os.path.join(case_path, f"{case_id}_seg.nii")

        if not os.path.isfile(flair_path) or not os.path.isfile(seg_path):
            print(f"  [Warning] Missing files for {case_id}. Skipping.")
            continue

        # Load and normalize FLAIR; load seg (int labels)
        flair_vol, seg_vol = load_case(flair_path, seg_path)
        # flair_vol: float32 (240,240,155); seg_vol: int32 (240,240,155)

        # Extract only slices containing tumor
        X_slices, y_slices = extract_slices(flair_vol, seg_vol)
        # X_slices: (n_slices,240,240,1), float32 normalized
        # y_slices: (n_slices,240,240),   int32 (binary mask)

        if X_slices.shape[0] == 0:
            # No tumor slices in this case
            continue

        # Move channel axis to front: (n,1,240,240)
        X_slices = np.transpose(X_slices, (0, 3, 1, 2))  # (n,1,240,240)
        # Convert y to uint8
        y_slices = y_slices.astype(np.uint8)             # (n,240,240)

        # Append to lists
        X_list.append(X_slices)
        y_list.append(y_slices)

    # Concatenate all cases
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"Collected {X_all.shape[0]} tumor‐containing slices total.")

    return X_all, y_all


def main():
    # 1) Gather all slices from BraTS
    X_np, y_np = gather_slices(BASE_DIR)
    # X_np.shape = (N,1,240,240), float32 ∈ [0,1]
    # y_np.shape = (N,240,240),    uint8 {0,1}

    # 2) Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_np).float()  # (N,1,240,240)
    y_tensor = torch.from_numpy(y_np).long()   # (N,240,240)

    # 3) Wrap in TensorDataset (we'll treat y as a float for BCEWithLogitsLoss)
    #    BCEWithLogitsLoss expects target as float in {0,1}
    dataset = TensorDataset(X_tensor, y_tensor.float())

    # 4) Split into train/validation
    num_samples = len(dataset)
    num_val     = int(num_samples * VAL_SPLIT)
    num_train   = num_samples - num_val
    train_ds, val_ds = random_split(dataset, [num_train, num_val])
    print(f"Training slices: {num_train}, Validation slices: {num_val}")

    # 5) Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 6) Instantiate U-Net
    model = UNet(n_channels=1, n_classes=1, use_skip=True).to(DEVICE)

    # 7) Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + binary cross‐entropy
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    # 8) Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)         # (B,1,240,240)
            y_batch = y_batch.unsqueeze(1).to(DEVICE)  # (B,1,240,240) for BCE

            optimizer.zero_grad()
            outputs, _ = model(X_batch)         # outputs: (B,1,240,240)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_loss / num_train

        # 9) Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.unsqueeze(1).to(DEVICE)

                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        epoch_val_loss = val_loss / num_val

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]  "
              f"Train Loss: {epoch_train_loss:.4f}  "
              f"Val Loss: {epoch_val_loss:.4f}")

        # 10) Save best model by validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  → Saved new best model (val_loss: {best_val_loss:.4f})")

    print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
    print(f"Trained model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
