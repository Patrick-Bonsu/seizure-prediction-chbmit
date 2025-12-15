import torch
from torch.utils.data import DataLoader

from src.datasets.eeg_dataset import EEGWindowsDataset, make_stratified_filewise_split_indices


NPZ_PATH = "data/processed/chb_subset_windows.npz"

def main():
    full_ds = EEGWindowsDataset(NPZ_PATH)
    print("[DATASET] Full stats:", full_ds.stats())

    train_idx, val_idx, test_idx = make_stratified_filewise_split_indices(NPZ_PATH, seed=123)

    train_ds = EEGWindowsDataset(NPZ_PATH, indices=train_idx)
    val_ds = EEGWindowsDataset(NPZ_PATH, indices=val_idx)

    print("[DATASET] Train stats:", train_ds.stats())
    print("[DATASET] Val stats:", val_ds.stats())

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

    xb, yb = next(iter(loader))
    print("[DATALOADER] xb shape:", xb.shape)  # (B, C, L)
    print("[DATALOADER] yb shape:", yb.shape)  # (B,)
    print("[DATALOADER] y batch:", yb.tolist())

    assert xb.ndim == 3 and xb.shape[1] == 23 and xb.shape[2] == 1280
    assert yb.ndim == 1

if __name__ == "__main__":
    main()
