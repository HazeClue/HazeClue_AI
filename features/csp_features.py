"""
Filter Bank Common Spatial Patterns (FBCSP)
=============================================
CSP finds spatial filters that maximize variance for one class
while minimizing it for the other. Filter Bank applies CSP
across multiple frequency sub-bands for richer features.

Reference: Ang et al. (2008) "Filter Bank Common Spatial Pattern"
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh


FS = 128  # Sampling frequency

# Filter bank: sub-bands covering relevant EEG frequencies
FILTER_BANK = [
    (4, 8),    # Theta
    (8, 12),   # Alpha
    (12, 16),  # Low Beta
    (16, 24),  # Mid Beta
    (24, 36),  # High Beta / Low Gamma
]


def bandpass_filter(data: np.ndarray, low: float, high: float, fs: int = FS, order: int = 4) -> np.ndarray:
    """Apply zero-phase bandpass filter."""
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=-1)


class CSP:
    """
    Common Spatial Patterns for binary EEG classification.
    
    Finds spatial filters W that maximize the ratio of
    class-conditional variances.
    """
    
    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self.W = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSP':
        """
        Fit CSP filters from training data.
        
        Args:
            X: Shape (N, C, T) — EEG windows
            y: Shape (N,) — binary labels (0 or 1)
        """
        # Compute normalized class covariances
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        C0 = self._compute_mean_cov(X0)
        C1 = self._compute_mean_cov(X1)
        
        # Composite covariance
        Cc = C0 + C1
        
        # Solve generalized eigenvalue problem: C1 v = λ (C0 + C1) v
        eigenvalues, eigenvectors = eigh(C1, Cc)
        
        # Sort by eigenvalue (most discriminative at extremes)
        idx = np.argsort(eigenvalues)
        
        # Take top and bottom n_components/2 filters
        n_half = self.n_components // 2
        selected = np.concatenate([idx[:n_half], idx[-n_half:]])
        
        self.W = eigenvectors[:, selected].T  # Shape: (n_components, C)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP spatial filters and extract log-variance features.
        
        Args:
            X: Shape (N, C, T)
        
        Returns:
            features: Shape (N, n_components)
        """
        N = X.shape[0]
        features = np.zeros((N, self.n_components))
        
        for i in range(N):
            Z = self.W @ X[i]  # Apply spatial filter
            # Log-variance features (normalized)
            var_z = np.var(Z, axis=1)
            var_z = var_z / np.sum(var_z)  # Normalize
            features[i] = np.log(var_z + 1e-10)
        
        return features
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
    
    def _compute_mean_cov(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized mean covariance."""
        N, C, T = X.shape
        covs = np.zeros((C, C))
        for i in range(N):
            cov = X[i] @ X[i].T / T
            cov /= np.trace(cov)  # Trace-normalize
            covs += cov
        covs /= N
        # Regularize
        covs += np.eye(C) * 1e-6
        return covs


class FilterBankCSP:
    """
    Filter Bank CSP — applies CSP on multiple frequency sub-bands
    and concatenates features for richer representation.
    
    Total features: n_components × len(FILTER_BANK)
    Default: 6 × 5 = 30 features
    """
    
    def __init__(self, n_components: int = 6, filter_bank: list = None, fs: int = FS):
        self.n_components = n_components
        self.filter_bank = filter_bank or FILTER_BANK
        self.fs = fs
        self.csps = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FilterBankCSP':
        """
        Fit CSP for each frequency sub-band.
        
        Args:
            X: Shape (N, C, T) — raw/filtered EEG
            y: Shape (N,) — binary labels
        """
        self.csps = []
        
        for low, high in self.filter_bank:
            # Filter data to sub-band
            X_band = np.zeros_like(X)
            for i in range(len(X)):
                X_band[i] = bandpass_filter(X[i], low, high, self.fs)
            
            # Fit CSP on this band
            csp = CSP(n_components=self.n_components)
            try:
                csp.fit(X_band, y)
                self.csps.append((low, high, csp))
            except Exception as e:
                print(f"[FBCSP] Warning: CSP failed for band {low}-{high} Hz: {e}")
                continue
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract FBCSP features.
        
        Returns:
            features: Shape (N, n_components × n_bands)
        """
        all_features = []
        
        for low, high, csp in self.csps:
            X_band = np.zeros_like(X)
            for i in range(len(X)):
                X_band[i] = bandpass_filter(X[i], low, high, self.fs)
            
            feat = csp.transform(X_band)
            all_features.append(feat)
        
        return np.hstack(all_features)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
    
    @property
    def n_features(self) -> int:
        return self.n_components * len(self.csps)
