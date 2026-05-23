"""
Execution Mode Router
======================
Dynamically switches between RARD, MVES, and SAFE processing modes
based on quantitative signal quality thresholds.

Routing Logic:
  RARD: SQI_mean > 0.65 AND κ(P) < 50   → Full Riemannian path
  MVES: 0.35 < SQI_mean ≤ 0.65 OR κ ≥ 50 → Statistical fallback
  SAFE: N_bad/14 > 0.70                   → Catastrophic rejection
"""

import numpy as np
from enum import Enum
from typing import Tuple
from dataclasses import dataclass


class ExecutionMode(Enum):
    RARD = "rard"    # Riemannian Artifact-Robust Decoding
    MVES = "mves"    # Multivariate Euclidean Statistical
    SAFE = "safe"    # Catastrophic corruption — reject/coarse output


# Routing thresholds (from methodology v2.2)
SQI_RARD_THRESHOLD = 0.65       # Minimum mean SQI for RARD
SQI_MVES_LOWER = 0.35           # Below this → check SAFE
KAPPA_THRESHOLD = 50.0           # Maximum condition number for RARD
BAD_CHANNEL_RATIO = 0.70         # SAFE mode activation
BAD_CHANNEL_SQI = 0.25           # SQI below this = "bad" channel


@dataclass
class RoutingDecision:
    """Result of mode routing decision."""
    mode: ExecutionMode
    sqi_mean: float
    kappa: float
    n_bad_channels: int
    confidence: float  # 0-1, higher = more reliable decision


def route_window(
    sigma: np.ndarray,
    kappa: float,
    n_channels: int = 14
) -> RoutingDecision:
    """
    Determine execution mode for a single EEG window.
    
    Args:
        sigma: Shape (C,) — per-channel SQI scores
        kappa: Condition number of the regularized covariance
        n_channels: Number of EEG channels
    
    Returns:
        RoutingDecision with mode, metrics, and confidence
    """
    sqi_mean = np.mean(sigma)
    n_bad = int(np.sum(sigma < BAD_CHANNEL_SQI))
    bad_ratio = n_bad / n_channels
    
    # --- SAFE Mode Check (highest priority) ---
    if bad_ratio > BAD_CHANNEL_RATIO:
        return RoutingDecision(
            mode=ExecutionMode.SAFE,
            sqi_mean=sqi_mean,
            kappa=kappa,
            n_bad_channels=n_bad,
            confidence=0.1
        )
    
    # --- RARD Mode Check ---
    if sqi_mean > SQI_RARD_THRESHOLD and kappa < KAPPA_THRESHOLD:
        # High-quality signal + stable geometry
        confidence = min(sqi_mean, 1.0) * min(1.0, KAPPA_THRESHOLD / max(kappa, 1.0))
        return RoutingDecision(
            mode=ExecutionMode.RARD,
            sqi_mean=sqi_mean,
            kappa=kappa,
            n_bad_channels=n_bad,
            confidence=np.clip(confidence, 0.0, 1.0)
        )
    
    # --- MVES Mode (default fallback) ---
    confidence = max(0.2, sqi_mean * 0.8)
    return RoutingDecision(
        mode=ExecutionMode.MVES,
        sqi_mean=sqi_mean,
        kappa=kappa,
        n_bad_channels=n_bad,
        confidence=np.clip(confidence, 0.0, 1.0)
    )


def route_batch(
    sigmas: np.ndarray,
    kappas: np.ndarray
) -> list:
    """
    Route a batch of windows.
    
    Args:
        sigmas: Shape (N, C) — SQI scores per window
        kappas: Shape (N,) — condition numbers
    
    Returns:
        List of RoutingDecision objects
    """
    decisions = []
    for i in range(len(sigmas)):
        decision = route_window(sigmas[i], kappas[i])
        decisions.append(decision)
    return decisions


def get_mode_statistics(decisions: list) -> dict:
    """
    Compute statistics about mode distribution.
    
    Returns:
        dict with counts and percentages per mode
    """
    modes = [d.mode for d in decisions]
    total = len(modes)
    stats = {
        'total': total,
        'rard_count': sum(1 for m in modes if m == ExecutionMode.RARD),
        'mves_count': sum(1 for m in modes if m == ExecutionMode.MVES),
        'safe_count': sum(1 for m in modes if m == ExecutionMode.SAFE),
    }
    stats['rard_pct'] = 100 * stats['rard_count'] / max(total, 1)
    stats['mves_pct'] = 100 * stats['mves_count'] / max(total, 1)
    stats['safe_pct'] = 100 * stats['safe_count'] / max(total, 1)
    return stats
