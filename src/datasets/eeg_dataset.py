from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetMeta:
    fs: float
    window_sec: float
    channels: list[str]
    patients: list[str]


class EEGWindowsDataset(Dataset):
    """
    Loads windowed EEG data saved as .npz with arrays:
      X: (N, C, L) float
      y: (N,) int (0/1)
    """

    def __init__(
        self,
        npz_path: str,
        indices: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.npz_path = npz_path
        self._npz = np.load(npz_path, allow_pickle=True)

        X = self._npz["X"]
        y = self._npz["y"]

        if indices is None:
            indices = np.arange(X.shape[0], dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        self.indices = indices
        self.dtype = dtype

        # Metadata
        self.meta = DatasetMeta(
            fs=float(self._npz["fs"]),
            window_sec=float(self._npz["window_sec"]),
            channels=[str(c) for c in self._npz["channels"].tolist()],
            patients=[str(p) for p in self._npz["patients"].tolist()],
        )

        # Keep references for shape checks
        self._X = X
        self._y = y

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])

        x = self._X[real_idx]  # (C, L)
        y = self._y[real_idx]  # scalar

        # Convert to torch tensors
        x_t = torch.tensor(x, dtype=self.dtype)          # (C, L)
        y_t = torch.tensor(int(y), dtype=torch.long)     # ()

        return x_t, y_t

    def stats(self) -> dict:
        y_subset = self._y[self.indices]
        unique, counts = np.unique(y_subset, return_counts=True)
        return {
            "num_windows": len(self),
            "num_channels": int(self._X.shape[1]),
            "window_len": int(self._X.shape[2]),
            "label_counts": dict(zip(unique.tolist(), counts.tolist())),
            "fs": self.meta.fs,
            "window_sec": self.meta.window_sec,
            "patients": self.meta.patients,
        }


def make_filewise_split_indices(
    npz_path: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """
    Leakage-safe split by EDF file.
    All windows from the same EDF file go to the same split.

    Returns:
        train_idx, val_idx, test_idx (arrays of window indices)
    """
    d = np.load(npz_path, allow_pickle=True)
    source_files = d["source_files"]  # (N,) strings like 'chb01_03.edf'
    n = source_files.shape[0]

    # Unique files
    files = np.unique(source_files)

    rng = np.random.default_rng(seed)
    files = rng.permutation(files)

    n_files = len(files)
    n_train = int(train_frac * n_files)
    n_val = int(val_frac * n_files)

    train_files = set(files[:n_train].tolist())
    val_files = set(files[n_train:n_train + n_val].tolist())
    test_files = set(files[n_train + n_val:].tolist())

    idx = np.arange(n, dtype=np.int64)

    train_idx = idx[np.isin(source_files, list(train_files))]
    val_idx = idx[np.isin(source_files, list(val_files))]
    test_idx = idx[np.isin(source_files, list(test_files))]

    return train_idx, val_idx, test_idx

def make_stratified_filewise_split_indices(
    npz_path: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    ensure_seizure_in_val_test: bool = True,
):
    """
    Leakage-safe + stratified by seizure presence at the FILE level.

    If ensure_seizure_in_val_test is True, we try to guarantee at least one
    seizure-file in val and one in test (when enough seizure-files exist).
    """
    d = np.load(npz_path, allow_pickle=True)
    source_files = d["source_files"]  # (N,)
    y = d["y"]                        # (N,)

    files = np.unique(source_files)
    seizure_files = []
    nonseizure_files = []

    for f in files:
        f_mask = (source_files == f)
        if np.any(y[f_mask] == 1):
            seizure_files.append(f)
        else:
            nonseizure_files.append(f)

    rng = np.random.default_rng(seed)
    seizure_files = rng.permutation(np.array(seizure_files, dtype=object))
    nonseizure_files = rng.permutation(np.array(nonseizure_files, dtype=object))

    # --- Split non-seizure files normally ---
    def split_files(file_arr):
        n = len(file_arr)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        train = set(file_arr[:n_train].tolist())
        val = set(file_arr[n_train:n_train + n_val].tolist())
        test = set(file_arr[n_train + n_val:].tolist())
        return train, val, test

    tr_n, va_n, te_n = split_files(nonseizure_files)

    # --- Split seizure files with optional guarantees ---
    s = seizure_files.tolist()
    tr_s, va_s, te_s = set(), set(), set()

    if len(s) == 0:
        # No seizure files at all (unexpected)
        tr_s, va_s, te_s = set(), set(), set()
    elif not ensure_seizure_in_val_test or len(s) < 3:
        # Too few seizure files to guarantee both val and test
        # Fall back to normal split
        tr_s, va_s, te_s = split_files(np.array(s, dtype=object))
    else:
        # Guarantee: 1 seizure file to val, 1 to test, rest to train
        va_s.add(s[0])
        te_s.add(s[1])
        for f in s[2:]:
            tr_s.add(f)

    train_files = tr_s.union(tr_n)
    val_files = va_s.union(va_n)
    test_files = te_s.union(te_n)

    idx = np.arange(source_files.shape[0], dtype=np.int64)
    train_idx = idx[np.isin(source_files, list(train_files))]
    val_idx = idx[np.isin(source_files, list(val_files))]
    test_idx = idx[np.isin(source_files, list(test_files))]

    return train_idx, val_idx, test_idx
