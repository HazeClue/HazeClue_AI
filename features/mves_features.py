"""
MVES Feature Extraction — Statistical Path
=============================================
Robust Euclidean features extracted when Riemannian geometry is unreliable.
Features are SQI-weighted to attenuate unreliable channels.

Feature categories:
  1. Bandpower (Welch PSD) — Delta, Theta, Alpha, Beta: 14×4 = 56
  2. Hjorth parameters — Activity, Mobility, Complexity: 14×3 = 42
  3. Correlation features — Upper-tri pairwise Pearson: 14×13/2 = 91
  4. Spectral slope — Per channel: 14

Total: ~203 features
"""

import numpy as np
from scipy.signal import welch
from typing import Tuple

# Canonical EEG frequency bands (Hz)
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}

FS = 128  # Default sampling frequency


def compute_bandpower(
    window: np.ndarray,
    sigma: np.ndarray,
    fs: int = FS
) -> np.ndarray:
    """
    Compute SQI-weighted bandpower using Welch PSD estimation.
    
    For each channel i and band b:
      P_b = Σ_{f ∈ b} PSD(f)
      P_b_weighted = σᵢ × P_b
    
    Args:
        window: Shape (C, T)
        sigma: Shape (C,) — SQI weights
        fs: Sampling frequency
    
    Returns:
        features: Shape (C × 4,) = (56,) for 14 channels
    """
    C, T = window.shape
    n_bands = len(BANDS)
    features = np.zeros(C * n_bands)
    
    for ch in range(C):
        freqs, psd = welch(window[ch], fs=fs, nperseg=min(256, T))
        
        for b_idx, (band_name, (f_low, f_high)) in enumerate(BANDS.items()):
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            band_power = np.sum(psd[band_mask])
            # SQI weighting
            features[ch * n_bands + b_idx] = sigma[ch] * band_power
    
    return features


def compute_hjorth(window: np.ndarray) -> np.ndarray:
    """
    Compute Hjorth parameters per channel.
    
    Activity: variance of the signal
    Mobility: sqrt(var(dx) / var(x))
    Complexity: mobility(dx) / mobility(x)
    
    Args:
        window: Shape (C, T)
    
    Returns:
        features: Shape (C × 3,) = (42,) for 14 channels
    """
    C, T = window.shape
    features = np.zeros(C * 3)
    
    for ch in range(C):
        x = window[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)
        
        var_x = np.var(x) + 1e-10
        var_dx = np.var(dx) + 1e-10
        var_ddx = np.var(ddx) + 1e-10
        
        activity = var_x
        mobility = np.sqrt(var_dx / var_x)
        complexity = np.sqrt(var_ddx / var_dx) / mobility
        
        features[ch * 3] = activity
        features[ch * 3 + 1] = mobility
        features[ch * 3 + 2] = complexity
    
    return features


def compute_correlation(window: np.ndarray) -> np.ndarray:
    """
    Extract pairwise Pearson correlation coefficients (upper triangle).
    
    Provides partial spatial information even when SPD geometry is unstable.
    
    Args:
        window: Shape (C, T)
    
    Returns:
        features: Shape (C*(C-1)/2,) = (91,) for 14 channels
    """
    corr_matrix = np.corrcoef(window)
    # Extract upper triangle (excluding diagonal)
    upper_idx = np.triu_indices(window.shape[0], k=1)
    features = corr_matrix[upper_idx]
    # Handle NaN from constant channels
    features = np.nan_to_num(features, nan=0.0)
    return features


def compute_spectral_slope(
    window: np.ndarray,
    fs: int = FS
) -> np.ndarray:
    """
    Estimate spectral slope in log-power space per channel.
    
    S(f) = a × log(f) + b
    Abnormally flat spectra → severe contamination.
    
    Args:
        window: Shape (C, T)
        fs: Sampling frequency
    
    Returns:
        features: Shape (C,) = (14,) — slope values
    """
    C, T = window.shape
    slopes = np.zeros(C)
    
    for ch in range(C):
        freqs, psd = welch(window[ch], fs=fs, nperseg=min(256, T))
        
        # Avoid log(0)
        valid = (freqs > 0) & (psd > 0)
        if np.sum(valid) < 2:
            slopes[ch] = 0.0
            continue
        
        log_freqs = np.log(freqs[valid])
        log_psd = np.log(psd[valid])
        
        # Linear regression in log space
        coeffs = np.polyfit(log_freqs, log_psd, 1)
        slopes[ch] = coeffs[0]  # slope
    
    return slopes


def extract_mves_features(
    window: np.ndarray,
    sigma: np.ndarray,
    fs: int = FS
) -> np.ndarray:
    """
    Full MVES feature extraction pipeline.
    
    Concatenates: bandpower(56) + hjorth(42) + correlation(91) + spectral_slope(14) = 203
    
    Args:
        window: Shape (C, T)
        sigma: Shape (C,) — SQI weights
        fs: Sampling frequency
    
    Returns:
        features: Shape (~203,) feature vector
    """
    bp = compute_bandpower(window, sigma, fs)
    hj = compute_hjorth(window)
    cr = compute_correlation(window)
    ss = compute_spectral_slope(window, fs)
    
    features = np.concatenate([bp, hj, cr, ss])
    return features


def extract_mves_features_batch(
    windows: np.ndarray,
    sigmas: np.ndarray,
    fs: int = FS
) -> np.ndarray:
    """
    Extract MVES features for a batch of windows.
    
    Args:
        windows: Shape (N, C, T)
        sigmas: Shape (N, C)
        fs: Sampling frequency
    
    Returns:
        features: Shape (N, ~203)
    """
    N = windows.shape[0]
    
    # Compute first to get feature dimension
    first = extract_mves_features(windows[0], sigmas[0], fs)
    n_features = len(first)
    
    features = np.zeros((N, n_features))
    features[0] = first
    
    for i in range(1, N):
        features[i] = extract_mves_features(windows[i], sigmas[i], fs)
    
    return features
