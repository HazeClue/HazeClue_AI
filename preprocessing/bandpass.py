"""
Bandpass Filtering
===================
Zero-phase Butterworth bandpass filter (1–40 Hz).
Preserves temporal structure of oscillatory EEG components.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    data: np.ndarray,
    low: float = 1.0,
    high: float = 40.0,
    fs: int = 128,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.
    
    Args:
        data: Shape (C, T) or (N, C, T)
        low: Lower cutoff frequency (Hz) — suppresses DC drift
        high: Upper cutoff frequency (Hz) — suppresses EMG/noise
        fs: Sampling frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered data (same shape as input)
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    
    if data.ndim == 2:
        # Single window: (C, T)
        return filtfilt(b, a, data, axis=1)
    elif data.ndim == 3:
        # Batch of windows: (N, C, T)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i], axis=1)
        return filtered
    else:
        raise ValueError(f"Expected 2D or 3D input, got {data.ndim}D")


def notch_filter(
    data: np.ndarray,
    freq: float = 50.0,
    fs: int = 128,
    quality: float = 30.0
) -> np.ndarray:
    """
    Optional notch filter for powerline noise (50/60 Hz).
    
    Note: EMOTIV devices include hardware-level filtering, so this
    is typically not needed. Enable only if deployment environment
    diagnostics indicate residual powerline contamination.
    
    Args:
        data: Shape (C, T) or (N, C, T)
        freq: Frequency to notch out (Hz)
        fs: Sampling frequency
        quality: Quality factor (higher = narrower notch)
    """
    b, a = iirnotch(freq, quality, fs)
    
    if data.ndim == 2:
        return filtfilt(b, a, data, axis=1)
    elif data.ndim == 3:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i], axis=1)
        return filtered
    else:
        raise ValueError(f"Expected 2D or 3D input, got {data.ndim}D")


def preprocess_batch(
    X: np.ndarray,
    fs: int = 128,
    apply_notch: bool = False,
    notch_freq: float = 50.0
) -> np.ndarray:
    """
    Apply full preprocessing chain to a batch of EEG windows.
    
    Args:
        X: Shape (N, 14, 512) — batch of windows
        fs: Sampling frequency
        apply_notch: Whether to apply powerline notch filter
        notch_freq: Powerline frequency (50 Hz Europe, 60 Hz US)
    
    Returns:
        Preprocessed batch (N, 14, 512)
    """
    X_filtered = bandpass_filter(X, low=1.0, high=40.0, fs=fs)
    
    if apply_notch:
        X_filtered = notch_filter(X_filtered, freq=notch_freq, fs=fs)
    
    return X_filtered
