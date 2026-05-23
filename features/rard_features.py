"""
RARD Feature Extraction — Riemannian Tangent Space
====================================================
Projects SPD covariance matrices onto Euclidean tangent space
around a reference point, producing 105 spatial connectivity features.

V = vec(log(P_ref^(-1/2) · P · P_ref^(-1/2)))

For 14 channels: N(N+1)/2 = 14×15/2 = 105 unique features.
"""

import numpy as np
from scipy.linalg import logm, sqrtm, inv


def matrix_sqrt(P: np.ndarray) -> np.ndarray:
    """Compute matrix square root of SPD matrix."""
    return np.real(sqrtm(P))


def matrix_invsqrt(P: np.ndarray) -> np.ndarray:
    """Compute inverse matrix square root of SPD matrix."""
    return np.real(inv(sqrtm(P)))


def matrix_log(P: np.ndarray) -> np.ndarray:
    """Compute matrix logarithm of SPD matrix."""
    return np.real(logm(P))


def tangent_space_projection(
    P: np.ndarray,
    P_ref: np.ndarray
) -> np.ndarray:
    """
    Project an SPD matrix onto the tangent space at P_ref.
    
    T_{P_ref}(P) = log(P_ref^(-1/2) · P · P_ref^(-1/2))
    
    Args:
        P: Shape (C, C) — SPD covariance matrix
        P_ref: Shape (C, C) — reference point on manifold
    
    Returns:
        T: Shape (C, C) — tangent space representation (symmetric)
    """
    P_ref_invsqrt = matrix_invsqrt(P_ref)
    inner = P_ref_invsqrt @ P @ P_ref_invsqrt
    T = matrix_log(inner)
    return T


def vectorize_tangent(T: np.ndarray) -> np.ndarray:
    """
    Vectorize symmetric tangent matrix, keeping only upper triangle.
    
    Off-diagonal elements are scaled by √2 to preserve Frobenius norm.
    For 14×14 matrix: 14×15/2 = 105 features.
    
    Args:
        T: Shape (C, C) — symmetric tangent matrix
    
    Returns:
        v: Shape (C*(C+1)/2,) — flattened feature vector
    """
    C = T.shape[0]
    n_features = C * (C + 1) // 2
    v = np.zeros(n_features)
    
    idx = 0
    for i in range(C):
        for j in range(i, C):
            if i == j:
                v[idx] = T[i, j]
            else:
                v[idx] = T[i, j] * np.sqrt(2)
            idx += 1
    
    return v


def extract_rard_features(
    P: np.ndarray,
    P_ref: np.ndarray
) -> np.ndarray:
    """
    Full RARD feature extraction: SPD → Tangent Space → 105-d vector.
    
    Args:
        P: Shape (C, C) — stabilized SPD covariance
        P_ref: Shape (C, C) — reference point (from calibration)
    
    Returns:
        features: Shape (105,) for 14-channel EEG
    """
    T = tangent_space_projection(P, P_ref)
    features = vectorize_tangent(T)
    return features


def extract_rard_features_batch(
    covariances: np.ndarray,
    P_ref: np.ndarray
) -> np.ndarray:
    """
    Extract RARD features for a batch of covariance matrices.
    
    Args:
        covariances: Shape (N, C, C) — batch of SPD matrices
        P_ref: Shape (C, C) — reference point
    
    Returns:
        features: Shape (N, 105)
    """
    N = covariances.shape[0]
    C = covariances.shape[1]
    n_features = C * (C + 1) // 2
    
    features = np.zeros((N, n_features))
    for i in range(N):
        features[i] = extract_rard_features(covariances[i], P_ref)
    
    return features


def compute_frechet_mean(
    covariances: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Compute Fréchet mean of SPD matrices using iterative algorithm.
    
    P_base = argmin_P Σᵢ d_R(P, Pᵢ)²
    
    Uses the geometric mean algorithm (Karcher/Fréchet mean on SPD manifold).
    
    Args:
        covariances: Shape (N, C, C) — collection of SPD matrices
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        P_mean: Shape (C, C) — Fréchet mean (SPD)
    """
    from scipy.linalg import expm
    
    N, C, _ = covariances.shape
    
    # Initialize with arithmetic mean (rough approximation)
    P_mean = np.mean(covariances, axis=0)
    # Ensure SPD
    eigenvalues, eigenvectors = np.linalg.eigh(P_mean)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    P_mean = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    for iteration in range(max_iter):
        # Project all matrices to tangent space at current mean
        P_invsqrt = matrix_invsqrt(P_mean)
        P_sqrt = matrix_sqrt(P_mean)
        
        tangent_sum = np.zeros((C, C))
        for i in range(N):
            inner = P_invsqrt @ covariances[i] @ P_invsqrt
            tangent_sum += matrix_log(inner)
        
        tangent_mean = tangent_sum / N
        
        # Check convergence
        norm = np.linalg.norm(tangent_mean, 'fro')
        if norm < tol:
            break
        
        # Update mean using exponential map
        P_mean = P_sqrt @ np.real(expm(tangent_mean)) @ P_sqrt
        
        # Ensure symmetry
        P_mean = (P_mean + P_mean.T) / 2.0
    
    return P_mean
