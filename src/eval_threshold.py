import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.eeg_dataset import EEGWindowsDataset, make_stratified_filewise_split_indices
from src.models.cnn_1d import CNN1D

NPZ_PATH = "data/processed/chb_subset_windows.npz"
CKPT_PATH = "reports/best_model.pt"
SEED = 123
BATCH_SIZE = 128

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, val_idx, test_idx = make_stratified_filewise_split_indices(NPZ_PATH, seed=SEED)
    test_ds = EEGWindowsDataset(NPZ_PATH, indices=test_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CNN1D(in_channels=test_ds.stats()["num_channels"], num_classes=2).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    probs_all = []
    y_all = []

    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_all.append(probs)
        y_all.append(yb.numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)

    # sweep thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("[INFO] Threshold sweep on TEST")
    for th in thresholds:
        preds = (probs_all >= th).astype(int)

        tp = int(np.sum((preds == 1) & (y_all == 1)))
        tn = int(np.sum((preds == 0) & (y_all == 0)))
        fp = int(np.sum((preds == 1) & (y_all == 0)))
        fn = int(np.sum((preds == 0) & (y_all == 1)))

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

        print(f"th={th:.1f} | tp={tp} fp={fp} fn={fn} | precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

if __name__ == "__main__":
    main()
