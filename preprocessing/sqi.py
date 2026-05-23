"""
Signal Quality Index (SQI)
===========================
Per-channel reliability estimation for EEG signals.
Provides continuous weighting (not binary rejection) to preserve
covariance topology stability.

Each channel receives: σᵢ ∈ [0.1, 1.0]
Lower bound is intentionally non-zero to prevent singular covariance.
"""

import numpy as np
from typing import Tuple


# Thresholds (empirically tuned for EMOTIV dry electrodes)
VARIANCE_LOW = 0.1       # µV² — near-zero → flatline/detachment
VARIANCE_HIGH = 5000.0   # µV² — excessive → motion artifact
FLATLINE_THRESHOLD = 0.01
SATURATION_THRESHOLD = 0.95  # fraction of max ADC range

SQI_MIN = 0.1
SQI_MAX = 1.0


def compute_channel_sqi(channel: np.ndarray) -> float:
    """
    Compute Signal Quality Index for a single EEG channel.
    
    Integrates:
      1. Variance stability
      2. Flatline detection
      3. Saturation detection
      4. Spectral instability
    
    Args:
        channel: 1D array of shape (T,) — single channel, single window
    
    Returns:
        float: SQI score in [0.1, 1.0]
    """
    T = len(channel)
    score = 1.0
    
    # --- 1. Variance Stability ---
    var = np.var(channel)
    if var < VARIANCE_LOW:
        # Near-zero variance → electrode detachment or flatline
        score *= 0.2
    elif var > VARIANCE_HIGH:
        # Excessive variance → motion artifact
        score *= 0.3
    
    # --- 2. Flatline Detection ---
    # Penalize if temporal derivative energy ≈ 0
    diff_energy = np.mean(np.abs(np.diff(channel)))
    if diff_energy < FLATLINE_THRESHOLD:
        score *= 0.15
    
    # --- 3. Saturation Detection ---
    # Penalize if signal stays near hardware clipping limits
    channel_range = np.ptp(channel)  # peak-to-peak
    if channel_range > 0:
        max_val = np.max(np.abs(channel))
        # Estimate ADC range from data (EMOTIV typically ±4000 µV)
        adc_estimate = max(max_val * 1.1, 4000.0)
        saturation_ratio = max_val / adc_estimate
        if saturation_ratio > SATURATION_THRESHOLD:
            score *= 0.25
    
    # --- 4. Spectral Instability ---
    # Detect broadband energy spikes (motion contamination)
    if T > 10:
        fft_mag = np.abs(np.fft.rfft(channel))
        spectral_std = np.std(fft_mag)
        spectral_mean = np.mean(fft_mag) + 1e-10
        spectral_cv = spectral_std / spectral_mean
        
        if spectral_cv > 3.0:
            # Abnormally spiky spectrum → contamination
            score *= 0.4
    
    return np.clip(score, SQI_MIN, SQI_MAX)


def compute_sqi(window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SQI for all channels in an EEG window.
    
    Args:
        window: Shape (C, T) — single EEG window
    
    Returns:
        sigma: 1D array of shape (C,) — per-channel SQI scores
        Sigma_matrix: 2D diagonal matrix of shape (C, C) — weight matrix
    """
    C, T = window.shape
    sigma = np.array([compute_channel_sqi(window[i]) for i in range(C)])
    Sigma_matrix = np.diag(sigma)
    return sigma, Sigma_matrix


def compute_sqi_batch(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SQI for a batch of EEG windows.
    
    Args:
        X: Shape (N, C, T)
    
    Returns:
        sigmas: Shape (N, C) — per-channel SQI for each window
        global_sqi: Shape (N,) — mean SQI per window
    """
    N, C, T = X.shape
    sigmas = np.zeros((N, C))
    
    for i in range(N):
        sigmas[i], _ = compute_sqi(X[i])
    
    global_sqi = np.mean(sigmas, axis=1)
    return sigmas, global_sqi
