import torch
from torch.utils.data import DataLoader

from src.datasets.eeg_dataset import EEGWindowsDataset, make_stratified_filewise_split_indices
from src.models.cnn_1d import CNN1D

NPZ_PATH = "data/processed/chb_subset_windows.npz"

def main():
    # dataset + split (consistent with Day 3)
    train_idx, val_idx, test_idx = make_stratified_filewise_split_indices(NPZ_PATH, seed=123)
    train_ds = EEGWindowsDataset(NPZ_PATH, indices=train_idx)

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    xb, yb = next(iter(loader))

    print("[SANITY] Batch xb:", xb.shape, xb.dtype)  # expected: (8, 23, 1280)
    print("[SANITY] Batch yb:", yb.shape, yb.dtype)

    model = CNN1D(in_channels=xb.shape[1], num_classes=2)
    logits = model(xb)

    print("[SANITY] Logits shape:", logits.shape)    # expected: (8, 2)
    print("[SANITY] Logits dtype:", logits.dtype)

    assert logits.shape == (xb.shape[0], 2)

if __name__ == "__main__":
    main()
