import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/eda.ipynb")

def make_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }

def main():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb["cells"].append(make_cell("markdown", "# CHB-MIT EDA (Day 3)\n"
        "This notebook loads the processed window dataset and performs sanity checks:\n"
        "- dataset stats\n"
        "- leakage-safe split (file-wise)\n"
        "- plot one seizure vs one non-seizure window\n"
        "- alignment check for seizure-labeled windows\n"
    ))

    nb["cells"].append(make_cell("code", [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from src.datasets.eeg_dataset import EEGWindowsDataset, make_stratified_filewise_split_indices\n",
        "\n",
        "NPZ_PATH = 'data/processed/chb_subset_windows.npz'\n",
        "d = np.load(NPZ_PATH, allow_pickle=True)\n",
        "print('keys:', list(d.keys()))\n",
        "X = d['X']\n",
        "y = d['y']\n",
        "source_files = d['source_files']\n",
        "patient_ids = d['patient_ids']\n",
        "fs = float(d['fs'])\n",
        "window_sec = float(d['window_sec'])\n",
        "channels = d['channels'].tolist()\n",
        "print('X:', X.shape, 'y:', y.shape)\n",
        "print('label counts:', dict(zip(*np.unique(y, return_counts=True))))\n",
        "print('fs:', fs, 'window_sec:', window_sec)\n",
        "print('channels:', len(channels))\n",
        "print('patients list in file:', d['patients'].tolist())\n",
    ]))

    nb["cells"].append(make_cell("code", [
        "# Train/Val/Test split (file-wise, with seizure guaranteed in val/test when possible)\n",
        "train_idx, val_idx, test_idx = make_stratified_filewise_split_indices(NPZ_PATH, seed=123)\n",
        "train_ds = EEGWindowsDataset(NPZ_PATH, indices=train_idx)\n",
        "val_ds = EEGWindowsDataset(NPZ_PATH, indices=val_idx)\n",
        "test_ds = EEGWindowsDataset(NPZ_PATH, indices=test_idx)\n",
        "print('Train:', train_ds.stats())\n",
        "print('Val:', val_ds.stats())\n",
        "print('Test:', test_ds.stats())\n",
    ]))

    nb["cells"].append(make_cell("code", [
        "# Pick one non-seizure and one seizure window (from full y)\n",
        "idx_non = int(np.where(y == 0)[0][0])\n",
        "idx_seiz = int(np.where(y == 1)[0][0])\n",
        "print('Example non-seizure idx:', idx_non)\n",
        "print('Example seizure idx:', idx_seiz)\n",
        "print('Seizure window source file:', source_files[idx_seiz])\n",
        "\n",
        "x0 = X[idx_non]\n",
        "x1 = X[idx_seiz]\n",
        "\n",
        "# Plot a single channel for readability (channel 0)\n",
        "t = np.arange(x0.shape[1]) / fs\n",
        "\n",
        "plt.figure()\n",
        "plt.plot(t, x0[0])\n",
        "plt.title('Non-seizure window (channel 0)')\n",
        "plt.xlabel('Time (s)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.show()\n",
        "\n",
        "plt.figure()\n",
        "plt.plot(t, x1[0])\n",
        "plt.title('Seizure-labeled window (channel 0)')\n",
        "plt.xlabel('Time (s)')\n",
        "plt.ylabel('Amplitude')\n",
        "plt.show()\n",
    ]))

    nb["cells"].append(make_cell("code", [
        "# Alignment check: for the chosen seizure window, compute its approximate time range inside its EDF\n",
        "# We can only do this reliably if we know the window's position within that EDF.\n",
        "# Here we'll estimate by counting how many windows came earlier in the same EDF.\n",
        "\n",
        "seiz_file = source_files[idx_seiz]\n",
        "mask_same_file = (source_files == seiz_file)\n",
        "idxs_in_file = np.where(mask_same_file)[0]\n",
        "\n",
        "# position of idx_seiz among windows of that file\n",
        "pos = int(np.where(idxs_in_file == idx_seiz)[0][0])\n",
        "win_start_sec = pos * window_sec\n",
        "win_end_sec = win_start_sec + window_sec\n",
        "\n",
        "print('Seizure window belongs to:', seiz_file)\n",
        "print('Window position within that file:', pos)\n",
        "print('Estimated window time range (sec):', (win_start_sec, win_end_sec))\n",
        "\n",
        "print('\\nNext: compare this time range against seizure intervals from the summary file for that EDF (manual check).')\n",
    ]))

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2))
    print(f"[OK] Wrote notebook: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    main()
