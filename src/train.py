import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.eeg_dataset import EEGWindowsDataset, make_stratified_filewise_split_indices
from src.models.cnn_1d import CNN1D


# --------------------
# Config (Day 4 baseline)
# --------------------
NPZ_PATH = "data/processed/chb_subset_windows.npz"
SEED = 123

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
WEIGHT_DECAY = 1e-4

REPORTS_DIR = Path("reports")
BEST_MODEL_PATH = REPORTS_DIR / "best_model.pt"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics_from_logits(logits: torch.Tensor, y_true: torch.Tensor):
    """
    Returns accuracy, precision, recall, f1, and average precision (AUPRC).
    We compute AUPRC using probabilities for class 1 (seizure).
    """
    # Move to CPU numpy
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    y = y_true.detach().cpu().numpy()

    # confusion counts
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

    # AUPRC (Average Precision)
    # Implemented without sklearn to keep deps simple inside training script
    # (You can swap to sklearn.metrics.average_precision_score later if you want.)
    # Compute precision-recall curve by sorting by probability.
    order = np.argsort(-probs)
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    prec_curve = cum_tp / np.maximum(1, (cum_tp + cum_fp))
    rec_curve = cum_tp / max(1, np.sum(y_sorted == 1))

    # Average precision approximation
    # sum over changes in recall * precision
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(prec_curve, rec_curve):
        if r > prev_recall:
            ap += p * (r - prev_recall)
            prev_recall = r

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": float(ap),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    all_logits = []
    all_y = []
    total_loss = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)

        all_logits.append(logits)
        all_y.append(yb)

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    metrics = compute_metrics_from_logits(all_logits, all_y)
    metrics["loss"] = total_loss / max(1, n)
    return metrics


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # Leakage-safe split (consistent with Day 3)
    train_idx, val_idx, test_idx = make_stratified_filewise_split_indices(NPZ_PATH, seed=SEED)
    train_ds = EEGWindowsDataset(NPZ_PATH, indices=train_idx)
    val_ds = EEGWindowsDataset(NPZ_PATH, indices=val_idx)
    test_ds = EEGWindowsDataset(NPZ_PATH, indices=test_idx)

    print("[INFO] Train:", train_ds.stats())
    print("[INFO] Val:", val_ds.stats())
    print("[INFO] Test:", test_ds.stats())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Weighted loss for imbalance
    train_counts = train_ds.stats()["label_counts"]
    n0 = train_counts.get(0, 0)
    n1 = train_counts.get(1, 0)
    if n1 == 0:
        raise ValueError("No seizure windows in train split. Adjust split/patients.")
    w0 = 1.0
    w1 = n0 / max(1, n1)  # upweight seizures
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"[INFO] Class weights: {class_weights.tolist()} (0=non,1=seizure)")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = CNN1D(in_channels=train_ds.stats()["num_channels"], num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auprc = -1.0
    history = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = running_loss / max(1, n)

        # Evaluate
        train_metrics = evaluate(model, train_loader, device, loss_fn)
        val_metrics = evaluate(model, val_loader, device, loss_fn)

        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"[EPOCH {epoch:02d}] "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_auprc={val_metrics['auprc']:.4f} val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        # Save best by val AUPRC
        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "npz_path": NPZ_PATH,
                        "seed": SEED,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "epochs": EPOCHS,
                        "class_weights": [w0, w1],
                    },
                },
                BEST_MODEL_PATH,
            )
            print(f"[INFO] Saved best model to: {BEST_MODEL_PATH} (val_auprc={best_val_auprc:.4f})")

    # Final test evaluation with best model loaded
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, loss_fn)

    out = {
        "best_val_auprc": best_val_auprc,
        "history": history,
        "test": test_metrics,
        "config": {
            "npz_path": NPZ_PATH,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
        },
    }

    METRICS_PATH.write_text(json.dumps(out, indent=2))
    print(f"[INFO] Saved metrics to: {METRICS_PATH}")
    print("[TEST] Metrics:", test_metrics)


if __name__ == "__main__":
    main()
