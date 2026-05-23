"""
STEW Dataset Loader
====================
Loads raw EEG data from the STEW (Simultaneous Task EEG Workload) dataset.

Dataset: 48 subjects, EMOTIV EPOC, 14 channels, 128 Hz, 2.5 min per condition.
Files: {subject_id}_lo.txt (rest, label=0), {subject_id}_hi.txt (workload, label=1)
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# EMOTIV EPOC channel order
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
            'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

FS = 128  # Sampling frequency (Hz)
WINDOW_SECONDS = 4
WINDOW_SAMPLES = WINDOW_SECONDS * FS  # 512
OVERLAP_RATIO = 0.5
STRIDE = int(WINDOW_SAMPLES * (1 - OVERLAP_RATIO))  # 256


def load_stew_file(filepath: Path) -> np.ndarray:
    """
    Load a single STEW EEG text file.
    
    Returns:
        np.ndarray: Shape (14, N_samples) — channels × time
    """
    data = np.loadtxt(filepath)
    # STEW files are (N_samples, 14) — transpose to (14, N_samples)
    if data.shape[1] == 14:
        return data.T
    elif data.shape[0] == 14:
        return data
    else:
        raise ValueError(f"Unexpected shape {data.shape} in {filepath}")


def segment_into_windows(
    data: np.ndarray,
    window_size: int = WINDOW_SAMPLES,
    stride: int = STRIDE
) -> np.ndarray:
    """
    Segment continuous EEG data into overlapping windows.
    
    Args:
        data: Shape (C, T) — channels × total_samples
        window_size: Number of samples per window
        stride: Step size between consecutive windows
    
    Returns:
        np.ndarray: Shape (N_windows, C, window_size)
    """
    C, T = data.shape
    n_windows = (T - window_size) // stride + 1
    
    if n_windows <= 0:
        raise ValueError(f"Data too short ({T} samples) for window_size={window_size}")
    
    windows = np.zeros((n_windows, C, window_size))
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[:, start:end]
    
    return windows


def load_stew_dataset(
    data_dir: str,
    subjects: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full STEW dataset.
    
    Args:
        data_dir: Path to directory containing STEW .txt files
        subjects: Optional list of subject IDs to load (default: all found)
    
    Returns:
        X: np.ndarray of shape (N_total_windows, 14, 512)
        y: np.ndarray of shape (N_total_windows,) — 0=rest, 1=workload
        subject_ids: np.ndarray of shape (N_total_windows,) — subject index
    """
    data_path = Path(data_dir)
    
    all_windows = []
    all_labels = []
    all_subjects = []
    
    # Discover subject files
    lo_files = sorted(data_path.glob("*_lo.txt"))
    
    for lo_file in lo_files:
        # Extract subject ID from filename
        stem = lo_file.stem  # e.g., "sub01_lo" or "1_lo"
        subject_str = stem.replace("_lo", "")
        # Strip common prefixes: "sub01" → "01", "s01" → "01"
        for prefix in ["sub", "SUB", "Sub", "s", "S"]:
            if subject_str.lower().startswith(prefix.lower()):
                subject_str = subject_str[len(prefix):]
                break
        try:
            subject_id = int(subject_str)
        except ValueError:
            continue
        
        if subjects is not None and subject_id not in subjects:
            continue
        
        hi_file = lo_file.parent / lo_file.name.replace("_lo", "_hi")
        
        if not hi_file.exists():
            print(f"[WARN] Missing hi file for subject {subject_id}: {hi_file}")
            continue
        
        # Load rest condition (label=0)
        try:
            lo_data = load_stew_file(lo_file)
            lo_windows = segment_into_windows(lo_data)
            all_windows.append(lo_windows)
            all_labels.append(np.zeros(lo_windows.shape[0], dtype=int))
            all_subjects.append(np.full(lo_windows.shape[0], subject_id, dtype=int))
        except Exception as e:
            print(f"[WARN] Error loading {lo_file}: {e}")
            continue
        
        # Load workload condition (label=1)
        try:
            hi_data = load_stew_file(hi_file)
            hi_windows = segment_into_windows(hi_data)
            all_windows.append(hi_windows)
            all_labels.append(np.ones(hi_windows.shape[0], dtype=int))
            all_subjects.append(np.full(hi_windows.shape[0], subject_id, dtype=int))
        except Exception as e:
            print(f"[WARN] Error loading {hi_file}: {e}")
            continue
    
    if not all_windows:
        raise FileNotFoundError(f"No STEW data found in {data_dir}")
    
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subject_ids = np.concatenate(all_subjects, axis=0)
    
    print(f"[STEW] Loaded {X.shape[0]} windows from {len(set(subject_ids))} subjects")
    print(f"       Class 0 (rest): {(y==0).sum()}, Class 1 (workload): {(y==1).sum()}")
    
    return X, y, subject_ids
