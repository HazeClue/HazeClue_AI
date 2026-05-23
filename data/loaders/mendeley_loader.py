"""
Mendeley EEG Dataset Loader
============================
Loads raw EEG data from the Mendeley "Fusion relaxation and concentration moods" dataset.

Dataset: 30 subjects, EMOTIV EPOC+, 14 channels, 256 Hz (→ downsample to 128 Hz).
Format: EDF files organized as S{subject}/S{subject}E{session}.edf
Protocol: 3 minutes per session × 4 sessions per subject.
  - Minute 0 (eyes-open rest)   → Label 0
  - Minute 1 (concentration)    → Label 1  
  - Minute 2 (eyes-closed rest) → DISCARDED (Alpha contamination)
"""

import numpy as np
from scipy.signal import resample_poly
from pathlib import Path
from typing import List, Tuple, Optional
from math import gcd
import mne

# EMOTIV EPOC+ channel order
CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
            'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

FS_ORIGINAL = 256    # Actual sampling frequency from EDF
FS_TARGET = 128      # Target frequency (matches STEW)
WINDOW_SECONDS = 4
WINDOW_SAMPLES = WINDOW_SECONDS * FS_TARGET  # 512
OVERLAP_RATIO = 0.5
STRIDE = int(WINDOW_SAMPLES * (1 - OVERLAP_RATIO))  # 256

# Duration definitions (in samples at ORIGINAL fs)
MINUTE_SAMPLES_ORIG = 60 * FS_ORIGINAL  # 15360 samples per minute at 256 Hz


def _downsample_to_128(data: np.ndarray, fs_orig: int = FS_ORIGINAL) -> np.ndarray:
    """
    Anti-alias downsample from fs_orig to 128 Hz using polyphase resampling.
    256 Hz → 128 Hz = factor of 2 (simple and clean).
    """
    g = gcd(FS_TARGET, fs_orig)
    up = FS_TARGET // g
    down = fs_orig // g
    resampled = resample_poly(data, up, down, axis=1)
    return resampled


def segment_into_windows(
    data: np.ndarray,
    window_size: int = WINDOW_SAMPLES,
    stride: int = STRIDE
) -> np.ndarray:
    """
    Segment continuous EEG data into overlapping windows.
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


def load_mendeley_edf(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single Mendeley EDF file, downsample, and extract Minute 0 + Minute 1.
    
    Returns:
        rest_data: Shape (14, T_rest) at 128 Hz — Minute 0 (label=0)
        conc_data: Shape (14, T_conc) at 128 Hz — Minute 1 (label=1)
    """
    # Read EDF with MNE (suppress verbose output)
    raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
    data = raw.get_data()  # (C, T) in Volts
    
    # Convert to µV for consistency with STEW
    data = data * 1e6
    
    C, T = data.shape
    assert C == 14, f"Expected 14 channels, got {C} in {filepath}"
    
    fs_actual = int(raw.info['sfreq'])
    minute_samples = 60 * fs_actual
    
    # Extract minutes at ORIGINAL fs
    min0_end = min(minute_samples, T)
    min1_start = minute_samples
    min1_end = min(2 * minute_samples, T)
    # Minute 2 (eyes-closed) → DISCARDED
    
    if min1_start >= T:
        raise ValueError(f"File {filepath} too short ({T} samples) for 2 minutes at {fs_actual} Hz")
    
    rest_orig = data[:, :min0_end]            # Minute 0: eyes-open rest
    conc_orig = data[:, min1_start:min1_end]  # Minute 1: concentration
    
    # Downsample to 128 Hz
    rest_128 = _downsample_to_128(rest_orig, fs_actual)
    conc_128 = _downsample_to_128(conc_orig, fs_actual)
    
    return rest_128, conc_128


def load_mendeley_dataset(
    data_dir: str,
    subjects: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full Mendeley dataset from EDF files.
    
    Directory structure expected:
        data_dir/
        └── Emotiv 30s EDF/    (or directly subject folders)
            ├── S001/
            │   ├── S001E01.edf
            │   ├── S001E02.edf
            │   ├── S001E03.edf
            │   └── S001E04.edf
            ├── S002/
            ...
    
    Returns:
        X: np.ndarray of shape (N_total_windows, 14, 512)
        y: np.ndarray of shape (N_total_windows,) — 0=rest, 1=concentration
        subject_ids: np.ndarray of shape (N_total_windows,)
    """
    data_path = Path(data_dir)
    
    # Auto-detect subfolder structure
    emotiv_subdir = data_path / "Emotiv 30s EDF"
    if emotiv_subdir.exists():
        data_path = emotiv_subdir
    
    all_windows = []
    all_labels = []
    all_subjects = []
    
    # Discover subject folders
    subject_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('S')])
    
    for subj_dir in subject_dirs:
        # Extract subject ID: S001 → 1
        try:
            subject_id = int(subj_dir.name[1:])
        except ValueError:
            continue
        
        if subjects is not None and subject_id not in subjects:
            continue
        
        # Find EDF files in this subject's folder
        edf_files = sorted(subj_dir.glob("*.edf"))
        
        for edf_file in edf_files:
            try:
                rest_data, conc_data = load_mendeley_edf(edf_file)
                
                # Window the rest period (label=0)
                rest_windows = segment_into_windows(rest_data)
                if rest_windows.shape[0] > 0:
                    all_windows.append(rest_windows)
                    all_labels.append(np.zeros(rest_windows.shape[0], dtype=int))
                    all_subjects.append(np.full(rest_windows.shape[0], subject_id + 1000, dtype=int))
                
                # Window the concentration period (label=1)
                conc_windows = segment_into_windows(conc_data)
                if conc_windows.shape[0] > 0:
                    all_windows.append(conc_windows)
                    all_labels.append(np.ones(conc_windows.shape[0], dtype=int))
                    all_subjects.append(np.full(conc_windows.shape[0], subject_id + 1000, dtype=int))
                    
            except Exception as e:
                print(f"[WARN] Error loading {edf_file}: {e}")
                continue
    
    if not all_windows:
        raise FileNotFoundError(f"No Mendeley EDF data found in {data_dir}")
    
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subject_ids = np.concatenate(all_subjects, axis=0)
    
    print(f"[Mendeley] Loaded {X.shape[0]} windows from {len(set(subject_ids))} subjects")
    print(f"           Class 0 (rest): {(y==0).sum()}, Class 1 (concentration): {(y==1).sum()}")
    
    return X, y, subject_ids
