"""
Mendeley EEG Dataset Loader
============================
Loads raw EEG data from the Mendeley "Fusion relaxation and concentration moods" dataset.

Dataset: 30 subjects, EMOTIV EPOC+, 14 channels, 250 Hz (→ downsample to 128 Hz).
Protocol: 3 minutes per session × 4 sessions per subject.
  - Minute 0 (eyes-open rest)   → Label 0
  - Minute 1 (concentration)    → Label 1  
  - Minute 2 (eyes-closed rest) → DISCARDED (Alpha contamination)

Naming: S001E03 = Subject 1, Session 3
"""

import numpy as np
from scipy.signal import resample_poly
from pathlib import Path
from typing import List, Tuple, Optional
from math import gcd

# EMOTIV EPOC+ channel order (same as STEW)
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
            'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

FS_ORIGINAL = 250    # Original sampling frequency
FS_TARGET = 128      # Target frequency (matches STEW)
WINDOW_SECONDS = 4
WINDOW_SAMPLES = WINDOW_SECONDS * FS_TARGET  # 512
OVERLAP_RATIO = 0.5
STRIDE = int(WINDOW_SAMPLES * (1 - OVERLAP_RATIO))  # 256

# Duration definitions (in samples at ORIGINAL fs)
MINUTE_SAMPLES_ORIG = 60 * FS_ORIGINAL  # 15000 samples per minute at 250 Hz


def _downsample_to_128(data: np.ndarray, fs_orig: int = FS_ORIGINAL) -> np.ndarray:
    """
    Anti-alias downsample from fs_orig to 128 Hz using polyphase resampling.
    
    Args:
        data: Shape (C, T) at original sampling rate
        fs_orig: Original sampling frequency
    
    Returns:
        np.ndarray: Shape (C, T_new) at 128 Hz
    """
    # Find up/down factors: 128/250 = 64/125
    g = gcd(FS_TARGET, fs_orig)
    up = FS_TARGET // g
    down = fs_orig // g
    
    # resample_poly applies anti-aliasing filter internally
    resampled = resample_poly(data, up, down, axis=1)
    return resampled


def _parse_filename(filename: str) -> Tuple[int, int]:
    """
    Parse Mendeley filename format: S001E03 → (subject=1, session=3)
    """
    stem = Path(filename).stem.upper()
    # Extract subject number
    s_idx = stem.index('S') + 1
    e_idx = stem.index('E')
    subject_id = int(stem[s_idx:e_idx])
    session_id = int(stem[e_idx + 1:])
    return subject_id, session_id


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
        return np.zeros((0, C, window_size))
    
    windows = np.zeros((n_windows, C, window_size))
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[:, start:end]
    
    return windows


def load_mendeley_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single Mendeley EEG file, downsample, and extract Minute 0 + Minute 1.
    
    Returns:
        rest_data: Shape (14, T_rest) at 128 Hz — Minute 0 (label=0)
        conc_data: Shape (14, T_conc) at 128 Hz — Minute 1 (label=1)
    """
    # Load raw data (format varies: CSV or space-delimited)
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except ValueError:
        data = np.loadtxt(filepath)
    
    # Ensure shape is (14, N_samples)
    if data.shape[0] != 14 and data.shape[1] == 14:
        data = data.T
    
    C, T = data.shape
    assert C == 14, f"Expected 14 channels, got {C} in {filepath}"
    
    # Extract minutes at ORIGINAL fs (250 Hz)
    min0_end = min(MINUTE_SAMPLES_ORIG, T)
    min1_start = MINUTE_SAMPLES_ORIG
    min1_end = min(2 * MINUTE_SAMPLES_ORIG, T)
    # Minute 2 (eyes-closed) → DISCARDED
    
    if min1_start >= T:
        raise ValueError(f"File {filepath} too short ({T} samples) for 2 minutes at {FS_ORIGINAL} Hz")
    
    rest_orig = data[:, :min0_end]       # Minute 0: eyes-open rest
    conc_orig = data[:, min1_start:min1_end]  # Minute 1: concentration
    
    # Downsample 250 Hz → 128 Hz
    rest_128 = _downsample_to_128(rest_orig)
    conc_128 = _downsample_to_128(conc_orig)
    
    return rest_128, conc_128


def load_mendeley_dataset(
    data_dir: str,
    subjects: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full Mendeley dataset.
    
    Args:
        data_dir: Path to directory containing Mendeley EEG files
        subjects: Optional list of subject IDs to include
    
    Returns:
        X: np.ndarray of shape (N_total_windows, 14, 512)
        y: np.ndarray of shape (N_total_windows,) — 0=rest, 1=concentration
        subject_ids: np.ndarray of shape (N_total_windows,)
    """
    data_path = Path(data_dir)
    
    all_windows = []
    all_labels = []
    all_subjects = []
    
    # Discover files — common patterns: S001E01.txt, S001E01.csv, etc.
    files = sorted(list(data_path.glob("S*E*.*")))
    if not files:
        files = sorted(list(data_path.glob("s*e*.*")))
    
    for fpath in files:
        if fpath.suffix.lower() not in ['.txt', '.csv', '.dat']:
            continue
        
        try:
            subject_id, session_id = _parse_filename(fpath.name)
        except (ValueError, IndexError):
            continue
        
        if subjects is not None and subject_id not in subjects:
            continue
        
        try:
            rest_data, conc_data = load_mendeley_file(fpath)
            
            # Window the rest period (label=0)
            rest_windows = segment_into_windows(rest_data)
            if rest_windows.shape[0] > 0:
                all_windows.append(rest_windows)
                all_labels.append(np.zeros(rest_windows.shape[0], dtype=int))
                all_subjects.append(np.full(rest_windows.shape[0], subject_id + 1000, dtype=int))
                # +1000 offset to avoid collision with STEW subject IDs
            
            # Window the concentration period (label=1)
            conc_windows = segment_into_windows(conc_data)
            if conc_windows.shape[0] > 0:
                all_windows.append(conc_windows)
                all_labels.append(np.ones(conc_windows.shape[0], dtype=int))
                all_subjects.append(np.full(conc_windows.shape[0], subject_id + 1000, dtype=int))
                
        except Exception as e:
            print(f"[WARN] Error loading {fpath}: {e}")
            continue
    
    if not all_windows:
        raise FileNotFoundError(f"No Mendeley data found in {data_dir}")
    
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subject_ids = np.concatenate(all_subjects, axis=0)
    
    print(f"[Mendeley] Loaded {X.shape[0]} windows from {len(set(subject_ids))} subjects")
    print(f"           Class 0 (rest): {(y==0).sum()}, Class 1 (concentration): {(y==1).sum()}")
    
    return X, y, subject_ids
