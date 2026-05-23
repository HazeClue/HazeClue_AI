"""
Adaptive Personalization
=========================
Implements dual-timescale personalization for EEG baseline drift.

Fast Learner (On-Device):
  Euclidean EMA in tangent space: V_fast = 0.95 V_prev + 0.05 V_task

Slow Learner (On-Server):
  Periodic Fréchet mean re-centering with manifold retraction.

Safety Constraints:
  Adaptation is FROZEN when SQI < threshold, SAFE mode active,
  or uncertainty exceeds confidence limit.
"""

import numpy as np
from typing import Optional
from scipy.linalg import expm

from features.rard_features import matrix_log, matrix_sqrt, matrix_invsqrt
from routing.mode_router import ExecutionMode


class PersonalizationManager:
    """
    Manages adaptive baseline personalization.
    
    Prevents catastrophic drift by only updating from high-confidence windows.
    """
    
    # Adaptation coefficient (deliberately small)
    BETA = 0.05
    
    # Fast learner coefficient
    ALPHA_FAST = 0.05
    
    # Safety thresholds
    MIN_SQI_FOR_UPDATE = 0.65
    MAX_KAPPA_FOR_UPDATE = 50.0
    
    def __init__(self, n_channels: int = 14):
        self.n_channels = n_channels
        self.P_baseline = np.eye(n_channels)  # SPD baseline center
        self.V_fast = np.zeros((n_channels, n_channels))  # Tangent vector (fast learner)
        self.update_count = 0
        self.frozen_count = 0
        self.is_initialized = False
    
    def initialize(self, P_baseline: np.ndarray):
        """Set initial baseline from calibration phase."""
        self.P_baseline = P_baseline.copy()
        self.V_fast = np.zeros_like(P_baseline)
        self.is_initialized = True
        self.update_count = 0
        print(f"[Personalization] Initialized with baseline shape {P_baseline.shape}")
    
    def can_update(
        self,
        sqi_mean: float,
        kappa: float,
        mode: ExecutionMode
    ) -> bool:
        """
        Check if adaptation update is permitted.
        
        Updates are DISABLED when:
          - SQI is below threshold
          - SAFE mode is active
          - Covariance condition number is unstable
        """
        if mode == ExecutionMode.SAFE:
            return False
        if sqi_mean < self.MIN_SQI_FOR_UPDATE:
            return False
        if kappa > self.MAX_KAPPA_FOR_UPDATE:
            return False
        return True
    
    def update_fast(
        self,
        P_new: np.ndarray,
        sqi_mean: float,
        kappa: float,
        mode: ExecutionMode
    ):
        """
        Fast learner update (on-device Euclidean EMA).
        
        V_fast = (1 - α) V_prev + α V_task
        
        Only updates if safety constraints are satisfied.
        """
        if not self.can_update(sqi_mean, kappa, mode):
            self.frozen_count += 1
            return
        
        try:
            # Project new covariance to tangent space at baseline
            P_ref_invsqrt = matrix_invsqrt(self.P_baseline)
            inner = P_ref_invsqrt @ P_new @ P_ref_invsqrt
            V_new = matrix_log(inner)
            
            if np.all(np.isfinite(V_new)):
                self.V_fast = (1 - self.ALPHA_FAST) * self.V_fast + self.ALPHA_FAST * V_new
                self.update_count += 1
        except Exception:
            self.frozen_count += 1
    
    def update_slow(self, P_new: np.ndarray):
        """
        Slow learner update (on-server manifold retraction).
        
        P_t = β P_new + (1 - β) P_{t-1}
        
        This is called periodically by the cloud server after
        geometric validation.
        """
        try:
            # Update baseline using exponential interpolation
            P_ref_sqrt = matrix_sqrt(self.P_baseline)
            P_ref_invsqrt = matrix_invsqrt(self.P_baseline)
            
            inner = P_ref_invsqrt @ P_new @ P_ref_invsqrt
            V = matrix_log(inner)
            
            # Scale by beta
            V_scaled = self.BETA * V
            
            # Retract back to manifold
            self.P_baseline = P_ref_sqrt @ np.real(expm(V_scaled)) @ P_ref_sqrt
            
            # Ensure symmetry
            self.P_baseline = (self.P_baseline + self.P_baseline.T) / 2.0
            
        except Exception as e:
            print(f"[Personalization] Slow update failed: {e}")
    
    def get_current_reference(self) -> np.ndarray:
        """
        Get the current reference point for tangent space projection.
        
        Combines baseline with fast learner drift.
        """
        if not self.is_initialized:
            return np.eye(self.n_channels)
        
        try:
            # Apply fast learner correction to baseline
            P_sqrt = matrix_sqrt(self.P_baseline)
            P_adjusted = P_sqrt @ np.real(expm(self.V_fast)) @ P_sqrt
            P_adjusted = (P_adjusted + P_adjusted.T) / 2.0
            return P_adjusted
        except Exception:
            return self.P_baseline
    
    def get_stats(self) -> dict:
        """Return personalization statistics."""
        return {
            'is_initialized': self.is_initialized,
            'update_count': self.update_count,
            'frozen_count': self.frozen_count,
            'update_ratio': self.update_count / max(self.update_count + self.frozen_count, 1),
        }
