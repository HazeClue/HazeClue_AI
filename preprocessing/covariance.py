"""
Covariance Estimation & SPD Enforcement
=========================================
Computes stabilized, SQI-weighted covariance matrices suitable for
Riemannian manifold operations.

Pipeline:
  1. Raw covariance estimation
  2. SQI-weighted construction (attenuate, don't remove channels)
  3. Ledoit-Wolf shrinkage regularization
  4. SPD enforcement via eigenvalue flooring
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from typing import Tuple

EPSILON = 1e-6  # Minimum eigenvalue for SPD enforcement


def raw_covariance(window: np.ndarray) -> np.ndarray:
    """
    Compute centered sample covariance matrix.
    
    P_raw = (1/(T-1)) * X @ X^T
    
    Args:
        window: Shape (C, T) — single EEG window, mean-removed
    
    Returns:
        P_raw: Shape (C, C) — symmetric covariance matrix
    """
    C, T = window.shape
    # Mean removal per channel
    X_centered = window - window.mean(axis=1, keepdims=True)
    P_raw = (X_centered @ X_centered.T) / (T - 1)
    return P_raw


def sqi_weighted_covariance(P_raw: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Inject channel reliability weights into covariance.
    
    P_weighted = Σ^(1/2) @ P_raw @ Σ^(1/2)
    
    This preserves symmetry and positive definiteness while
    attenuating unreliable channels. Channels are NEVER fully removed
    to maintain stable covariance topology.
    
    Args:
        P_raw: Shape (C, C) — raw covariance
        sigma: Shape (C,) — per-channel SQI scores ∈ [0.1, 1.0]
    
    Returns:
        P_weighted: Shape (C, C)
    """
    sqrt_sigma = np.sqrt(sigma)
    Sigma_sqrt = np.diag(sqrt_sigma)
    P_weighted = Sigma_sqrt @ P_raw @ Sigma_sqrt
    return P_weighted


def ledoit_wolf_shrinkage(P_weighted: np.ndarray, window: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply automatic Ledoit-Wolf shrinkage regularization.
    
    P_reg = (1 - λ_LW) * P_weighted + λ_LW * I
    
    This stabilizes eigendecomposition, prevents covariance collapse,
    and reduces overfitting under short EEG windows.
    
    Args:
        P_weighted: Shape (C, C) — SQI-weighted covariance
        window: Shape (C, T) — original window (used for LW estimation)
    
    Returns:
        P_reg: Shape (C, C) — regularized covariance
        lambda_lw: float — estimated shrinkage coefficient
    """
    C = P_weighted.shape[0]
    
    # Estimate optimal shrinkage from data
    try:
        lw = LedoitWolf()
        # LedoitWolf expects (n_samples, n_features) = (T, C)
        lw.fit(window.T)
        lambda_lw = lw.shrinkage_
    except Exception:
        # Fallback to fixed shrinkage if estimation fails
        lambda_lw = 0.1
    
    I = np.eye(C)
    P_reg = (1 - lambda_lw) * P_weighted + lambda_lw * I
    
    return P_reg, lambda_lw


def enforce_spd(P: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """
    Enforce Symmetric Positive Definite (SPD) constraint.
    
    Eigenvalue flooring: λᵢ ← max(λᵢ, ε)
    
    This guarantees P ∈ SPD for all subsequent geometric operations:
    - Matrix logarithm
    - Geodesic distance
    - Tangent-space projection
    - Fréchet mean
    
    Args:
        P: Shape (C, C) — covariance matrix
        epsilon: Minimum eigenvalue
    
    Returns:
        P_spd: Shape (C, C) — guaranteed SPD
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    
    # Floor eigenvalues
    eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Reconstruct
    P_spd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure perfect symmetry (numerical precision)
    P_spd = (P_spd + P_spd.T) / 2.0
    
    return P_spd


def condition_number(P: np.ndarray) -> float:
    """
    Compute condition number κ(P) = λ_max / λ_min.
    
    Used for mode routing decisions:
    - κ < 50: stable geometry → RARD path
    - κ ≥ 50: unstable geometry → MVES path
    
    Args:
        P: Shape (C, C) — covariance matrix
    
    Returns:
        float: Condition number
    """
    eigenvalues = np.linalg.eigvalsh(P)
    lambda_min = np.min(np.abs(eigenvalues))
    lambda_max = np.max(np.abs(eigenvalues))
    
    if lambda_min < 1e-12:
        return float('inf')
    
    return lambda_max / lambda_min


def compute_stabilized_covariance(
    window: np.ndarray,
    sigma: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Full covariance stabilization pipeline for a single window.
    
    Args:
        window: Shape (C, T)
        sigma: Shape (C,) — SQI weights
    
    Returns:
        P_spd: Stabilized SPD covariance matrix (C, C)
        kappa: Condition number
        lambda_lw: Shrinkage coefficient used
    """
    P_raw = raw_covariance(window)
    P_weighted = sqi_weighted_covariance(P_raw, sigma)
    P_reg, lambda_lw = ledoit_wolf_shrinkage(P_weighted, window)
    P_spd = enforce_spd(P_reg)
    kappa = condition_number(P_spd)
    
    return P_spd, kappa, lambda_lw
