"""
Euclidean Alignment (EA) for Cross-Subject Transfer
=====================================================
Aligns each subject's data by whitening with their mean covariance,
reducing inter-subject variability on the SPD manifold.

Reference: He & Wu (2020) "Transfer Learning for Brain-Computer Interfaces"

R̄_s = (1/N_s) Σᵢ Xᵢ Xᵢᵀ / T        (mean covariance for subject s)
X̃ᵢ = R̄_s^(-1/2) · Xᵢ                 (aligned data)
"""

import numpy as np
from scipy.linalg import sqrtm, inv


def compute_subject_mean_cov(X_subject: np.ndarray) -> np.ndarray:
    """
    Compute mean covariance matrix for a subject's windows.
    
    Args:
        X_subject: Shape (N_windows, C, T) — all windows for one subject
    
    Returns:
        R_mean: Shape (C, C) — mean covariance matrix
    """
    N, C, T = X_subject.shape
    covs = np.zeros((C, C))
    
    for i in range(N):
        covs += X_subject[i] @ X_subject[i].T / T
    
    R_mean = covs / N
    
    # Regularize to ensure positive definiteness
    R_mean += np.eye(C) * 1e-6
    
    return R_mean


def align_subject(X_subject: np.ndarray, R_mean: np.ndarray = None) -> np.ndarray:
    """
    Apply Euclidean Alignment to a single subject's data.
    
    Args:
        X_subject: Shape (N_windows, C, T)
        R_mean: Shape (C, C) — precomputed mean covariance (optional)
    
    Returns:
        X_aligned: Shape (N_windows, C, T) — aligned data
    """
    if R_mean is None:
        R_mean = compute_subject_mean_cov(X_subject)
    
    # Compute R^(-1/2)
    R_invsqrt = np.real(inv(sqrtm(R_mean)))
    
    # Apply alignment to all windows
    N = X_subject.shape[0]
    X_aligned = np.zeros_like(X_subject)
    
    for i in range(N):
        X_aligned[i] = R_invsqrt @ X_subject[i]
    
    return X_aligned


def euclidean_alignment(
    X: np.ndarray,
    subject_ids: np.ndarray
) -> np.ndarray:
    """
    Apply Euclidean Alignment to the entire dataset, per-subject.
    
    Each subject's data is whitened by their own mean covariance,
    so all subjects' data is centered around the identity matrix
    in the SPD manifold.
    
    Args:
        X: Shape (N_total, C, T) — all windows
        subject_ids: Shape (N_total,) — subject ID per window
    
    Returns:
        X_aligned: Shape (N_total, C, T) — aligned windows
    """
    X_aligned = np.zeros_like(X)
    unique_subjects = np.unique(subject_ids)
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        X_subj = X[mask]
        X_aligned[mask] = align_subject(X_subj)
    
    print(f"[EA] Aligned {len(unique_subjects)} subjects")
    return X_aligned
