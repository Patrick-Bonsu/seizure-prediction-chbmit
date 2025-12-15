import numpy as np

NPZ_PATH = "data/processed/chb_subset_windows.npz"

def main():
    d = np.load(NPZ_PATH, allow_pickle=True)

    X = d["X"]
    y = d["y"]

    fs = float(d["fs"])
    window_sec = float(d["window_sec"])
    channels = d["channels"]
    patients = d["patients"]

    print("[SANITY] Loaded:", NPZ_PATH)
    print("[SANITY] X shape:", X.shape)  # (num_windows, num_channels, window_len)
    print("[SANITY] y shape:", y.shape)

    # Basic checks
    assert X.ndim == 3, "X should be 3D: (N, C, L)"
    assert y.ndim == 1, "y should be 1D: (N,)"
    assert X.shape[0] == y.shape[0], "X and y must have same number of windows"

    # Stats
    unique, counts = np.unique(y, return_counts=True)
    print("[SANITY] Label counts:", dict(zip(unique.tolist(), counts.tolist())))
    print("[SANITY] Sampling rate (fs):", fs)
    print("[SANITY] Window seconds:", window_sec)
    print("[SANITY] Window length samples:", X.shape[2])
    print("[SANITY] Num channels:", X.shape[1])
    print("[SANITY] First 5 channels:", channels[:5].tolist() if hasattr(channels, "tolist") else channels[:5])
    print("[SANITY] Patients:", patients.tolist() if hasattr(patients, "tolist") else patients)

    # Find one seizure and one non-seizure index
    idx0 = int(np.where(y == 0)[0][0])
    idx1 = int(np.where(y == 1)[0][0]) if np.any(y == 1) else None

    print("[SANITY] Example non-seizure index:", idx0)
    if idx1 is None:
        print("[SANITY] No seizure windows found in y (unexpected).")
    else:
        print("[SANITY] Example seizure index:", idx1)

if __name__ == "__main__":
    main()
