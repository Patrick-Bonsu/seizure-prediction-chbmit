import os
from typing import Dict, List, Tuple

import mne
import numpy as np

# ------------------
# CONFIG
# ------------------

RAW_DATA_DIR = "data/raw/chb-mit"   # root folder containing chb01, chb02, ...
PATIENT_ID = "chb01"                # we work with this patient for now
EDF_FILENAME = "chb01_03.edf"       # one EDF file inside chb01 folder

#For the full dataset build: 
PATIENT_IDS =["chb01"]
WINDOW_SEC = 5.0                    # will be used later
OUTPUT_PATH = "data/processed/chb_subset_windows.npz"  # where to save processed data
# ------------------
# CHUNK 2: LOAD ONE EDF
# ------------------

def load_one_edf():
    """Load a single EDF file for testing purposes."""
    edf_path = os.path.join(RAW_DATA_DIR, PATIENT_ID, EDF_FILENAME)
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"EDF file not found at {edf_path}")
    
    print(f"[INFO] Loading EDF file from {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    fs = raw.info["sfreq"]
    ch_names = raw.ch_names
    n_channels = len(ch_names)
    n_samples = raw.n_times
    duration_sec = n_samples / fs

    print(f"[INFO] Sampling rate: {fs} Hz")
    print(f"[INFO] Channels ({n_channels}): {ch_names}")
    print(f"[INFO] Duration: {duration_sec:.2f} seconds")

    return raw


# ------------------
# CHUNK 3: PARSE SUMMARY FILE FOR ONE PATIENT
# ------------------

def parse_summary_for_patient(
    raw_data_dir: str,
    patient_id: str,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse chbXX-summary.txt for a single patient folder (e.g., chb01).

    Returns:
        annotations: dict mapping edf_filename (e.g., 'chb01_03.edf')
                     -> list of (start_sec, end_sec) tuples
    """

    summary_path = os.path.join(raw_data_dir, patient_id, f"{patient_id}-summary.txt")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found at: {summary_path}")

    print(f"[INFO] Parsing summary file: {summary_path}")

    annotations: Dict[str, List[Tuple[float, float]]] = {}

    current_file = None
    current_start = None

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()

            # Example line: "File Name: chb01_03.edf"
            if line.startswith("File Name:"):
                parts = line.split(":")
                if len(parts) >= 2:
                    current_file = parts[1].strip()
                    if current_file not in annotations:
                        annotations[current_file] = []

            # Example line: "Seizure 1 Start Time: 2996 seconds"
            elif "Seizure" in line and "Start Time" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    time_str = parts[1].strip().split()[0]
                    current_start = float(time_str)

            # Example line: "Seizure 1 End Time: 3036 seconds"
            elif "Seizure" in line and "End Time" in line and current_file is not None:
                parts = line.split(":")
                if len(parts) >= 2 and current_start is not None:
                    time_str = parts[1].strip().split()[0]
                    end_time = float(time_str)
                    annotations[current_file].append((current_start, end_time))
                    current_start = None

    print(f"[INFO] Found seizure annotations for {len(annotations)} EDF files.")
    return annotations


def compute_window_labels_for_file(
    seizure_intervals: List[Tuple[float, float]],
    n_samples: int,
    fs: float,
    window_sec: float,
):
    """
    Create non-overlapping windows and label them based on seizure intervals.

    Args:
        seizure_intervals: list of (start_sec, end_sec) for this EDF file
        n_samples: total number of samples in the signal
        fs: sampling frequency (Hz)
        window_sec: window length in seconds

    Returns:
        windows: list of (start_idx, end_idx) sample indices
        labels: numpy array of shape (num_windows,) with 1 (seizure) or 0 (non-seizure)
    """
    window_len = int(window_sec * fs)
    windows = []
    labels = []

    if window_len <= 0:
        raise ValueError("window_len must be > 0")

    # non-overlapping windows across the entire recording
    for start in range(0, n_samples - window_len + 1, window_len):
        end = start + window_len

        # convert sample indices to seconds
        w_start_sec = start / fs
        w_end_sec = end / fs

        # default: non-seizure
        label = 0

        # check if this window overlaps ANY seizure interval
        for (sz_start, sz_end) in seizure_intervals:
            # Overlap condition:
            # window_start < seizure_end AND window_end > seizure_start
            if w_start_sec < sz_end and w_end_sec > sz_start:
                label = 1
                break

        windows.append((start, end))
        labels.append(label)

    return windows, np.array(labels, dtype=np.int64)


def create_windows_for_raw(
    raw: mne.io.BaseRaw,
    seizure_intervals: List[Tuple[float, float]],
    window_sec: float,
):
    """
    From a raw MNE object and seizure intervals, create windowed data and labels.

    Returns:
        X: (num_windows, num_channels, window_length)
        y: (num_windows,)
        fs: sampling frequency
        ch_names: list of channel names used
    """
    fs = raw.info["sfreq"]
    ch_names = raw.ch_names

    # Drop dummy channels named '-' if any
    picks = [i for i, ch in enumerate(ch_names) if ch != "-"]
    if len(picks) < len(ch_names):
        print(f"[INFO] Dropping {len(ch_names) - len(picks)} dummy channels ('-').")

    data = raw.get_data(picks=picks)  # shape: (num_channels, n_samples)
    ch_names = [ch_names[i] for i in picks]
    n_channels, n_samples = data.shape

    # Compute windows + labels
    windows, labels = compute_window_labels_for_file(
        seizure_intervals=seizure_intervals,
        n_samples=n_samples,
        fs=fs,
        window_sec=window_sec,
    )

    if len(windows) == 0:
        print("[WARN] No windows created.")
        return np.empty((0, n_channels, 0)), np.empty((0,)), fs, ch_names

    window_len = int(window_sec * fs)
    X = np.zeros((len(windows), n_channels, window_len), dtype=np.float32)

    for i, (start, end) in enumerate(windows):
        X[i] = data[:, start:end]

    return X, labels, fs, ch_names


# ------------------
# MAIN (for now: test load + summary parsing)
# ------------------

def main():
    # -------------------------
    # 1. Optional sanity check: load single EDF
    # -------------------------
    print("[INFO] Running single-file sanity check...")
    raw_debug = load_one_edf()

    # Parse annotations only for the debug patient (PATIENT_ID)
    annotations_debug = parse_summary_for_patient(RAW_DATA_DIR, PATIENT_ID)
    seizure_intervals_debug = annotations_debug.get(EDF_FILENAME, [])
    print(f"[INFO] Seizure intervals for debug file {EDF_FILENAME}: {seizure_intervals_debug}")

    X_debug, y_debug, fs_debug, ch_debug = create_windows_for_raw(
        raw_debug,
        seizure_intervals=seizure_intervals_debug,
        window_sec=WINDOW_SEC,
    )
    print("[INFO] Debug windowing result:")
    print(f"  X_debug shape: {X_debug.shape}")
    print(f"  y_debug shape: {y_debug.shape}")
    if y_debug.size > 0:
        print(f"  Debug label counts (0,1): {np.bincount(y_debug)}")
    print()

    # -------------------------
    # 2. Full dataset build over PATIENT_IDS
    # -------------------------
    all_X = []
    all_y = []
    all_source_files = []
    all_patient_ids = []
    channels_ref = None
    fs_ref = None


    print("[INFO] Building full windowed dataset...")
    for pid in PATIENT_IDS:
        print(f"\n[INFO] Processing patient: {pid}")
        patient_dir = os.path.join(RAW_DATA_DIR, pid)

        if not os.path.exists(patient_dir):
            print(f"[WARN] Patient folder not found: {patient_dir}, skipping.")
            continue

        # Parse summary file for this patient
        annotations = parse_summary_for_patient(RAW_DATA_DIR, pid)

        # Loop over EDF files in this patient's folder
        for fname in sorted(os.listdir(patient_dir)):
            if not fname.endswith(".edf"):
                continue

            edf_path = os.path.join(patient_dir, fname)
            seizure_intervals = annotations.get(fname, [])

            print(f"  [INFO] File: {fname} | #seizures: {len(seizure_intervals)}")

            # Load EDF
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

            # Create windows + labels
            X_file, y_file, fs, ch_names = create_windows_for_raw(
                raw,
                seizure_intervals=seizure_intervals,
                window_sec=WINDOW_SEC,
            )

            if X_file.shape[0] == 0:
                print(f"  [WARN] No windows created for {fname}, skipping.")
                continue

            # Ensure consistent channels and sampling rate
            if channels_ref is None:
                channels_ref = ch_names
                fs_ref = fs
            else:
                if ch_names != channels_ref:
                    print(f"  [WARN] Channel mismatch in {fname}, skipping this file for now.")
                    continue
                if fs != fs_ref:
                    print(f"  [WARN] Sampling rate mismatch in {fname}, skipping this file.")
                    continue

            all_X.append(X_file)
            all_y.append(y_file)

            all_source_files.extend([fname] * X_file.shape[0])
            all_patient_ids.extend([pid] * X_file.shape[0])


    if len(all_X) == 0:
        print("[ERROR] No data collected. Check RAW_DATA_DIR, PATIENT_IDS, and file structure.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print("\n[INFO] Final dataset summary:")
    print(f"  X shape: {X.shape}  (num_windows, num_channels, window_length)")
    print(f"  y shape: {y.shape}")
    print(f"  Label counts (0=non-seizure, 1=seizure): {np.bincount(y)}")
    print(f"  Sampling frequency: {fs_ref} Hz")
    print(f"  Channels ({len(channels_ref)}): {channels_ref}")

    # -------------------------
    # 3. Save to .npz
    # -------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    assert len(all_source_files) == X.shape[0], f"source_files length {len(all_source_files)} != num_windows {X.shape[0]}"
    assert len(all_patient_ids) == X.shape[0], f"patient_ids length {len(all_patient_ids)} != num_windows {X.shape[0]}"

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        fs=fs_ref,
        window_sec=WINDOW_SEC,
        channels=np.array(channels_ref),
        patients=np.array(PATIENT_IDS),
        source_files=np.array(all_source_files),
        patient_ids=np.array(all_patient_ids),
    )

    print(f"\n[INFO] Saved processed dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
