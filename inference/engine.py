"""
Real-Time Inference Engine
============================
Full pipeline: Raw EEG → Preprocess → Route → Extract → Classify → Smooth

Features:
  - Mode-switching between RARD and MVES in real-time
  - Temporal prediction smoothing (EMA γ=0.7)
  - Window acceptance/rejection policy
  - Calibration phase support
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from preprocessing.bandpass import bandpass_filter
from preprocessing.sqi import compute_sqi
from preprocessing.covariance import compute_stabilized_covariance
from routing.mode_router import route_window, ExecutionMode, RoutingDecision
from features.rard_features import extract_rard_features
from features.mves_features import extract_mves_features


@dataclass
class InferenceResult:
    """Result of single-window inference."""
    prediction: int          # 0 or 1 (binary class)
    probability: float       # Confidence [0, 1]
    smoothed_output: float   # Temporally smoothed prediction
    mode: ExecutionMode      # Which path was used
    sqi_mean: float          # Mean signal quality
    kappa: float             # Covariance condition number
    accepted: bool           # Whether window was accepted
    raw_score: float         # Raw classifier output


class HazeClueInferenceEngine:
    """
    Real-time EEG inference engine implementing RARD–MVES v2.2.
    
    Usage:
        engine = HazeClueInferenceEngine()
        engine.load_models('trained_models/')
        engine.calibrate(resting_eeg_data)  # 60s resting state
        
        for window in stream:
            result = engine.infer(window)
            print(f"Focus: {result.smoothed_output:.2f}, Mode: {result.mode}")
    """
    
    # Temporal smoothing coefficient
    GAMMA = 0.7
    
    # Window rejection thresholds
    MIN_GLOBAL_SQI = 0.2
    MAX_ENTROPY = 10.0
    
    def __init__(self):
        self.clf_rard = None
        self.clf_mves = None
        self.scaler_rard = None
        self.scaler_mves = None
        self.P_ref = None  # Riemannian reference point
        self.smoothed_prediction = 0.5  # Initial neutral state
        self.is_calibrated = False
        self.window_count = 0
        self.rejection_count = 0
    
    def load_models(self, model_dir: str):
        """Load trained classifiers and scalers from disk."""
        model_path = Path(model_dir)
        
        rard_path = model_path / 'rard_classifier.joblib'
        mves_path = model_path / 'mves_classifier.joblib'
        rard_scaler_path = model_path / 'rard_scaler.joblib'
        mves_scaler_path = model_path / 'mves_scaler.joblib'
        ref_path = model_path / 'P_ref.npy'
        
        if rard_path.exists():
            self.clf_rard = joblib.load(rard_path)
            print("[Engine] Loaded RARD classifier")
        if mves_path.exists():
            self.clf_mves = joblib.load(mves_path)
            print("[Engine] Loaded MVES classifier")
        if rard_scaler_path.exists():
            self.scaler_rard = joblib.load(rard_scaler_path)
        if mves_scaler_path.exists():
            self.scaler_mves = joblib.load(mves_scaler_path)
        if ref_path.exists():
            self.P_ref = np.load(ref_path)
            self.is_calibrated = True
            print("[Engine] Loaded reference point")
    
    def calibrate(self, resting_data: np.ndarray, fs: int = 128):
        """
        Perform initial calibration from resting-state EEG.
        
        60 seconds eyes-open resting state → Fréchet baseline.
        
        Args:
            resting_data: Shape (14, T) — continuous resting EEG
            fs: Sampling frequency
        """
        from features.rard_features import compute_frechet_mean
        from data.loaders.stew_loader import segment_into_windows
        
        # Filter
        resting_filtered = bandpass_filter(resting_data)
        
        # Segment into windows
        windows = segment_into_windows(resting_filtered)
        
        # Compute stable covariances
        covs = []
        for window in windows:
            sigma, _ = compute_sqi(window)
            P_spd, kappa, _ = compute_stabilized_covariance(window, sigma)
            if kappa < 100:
                covs.append(P_spd)
        
        if len(covs) < 5:
            print("[Engine] WARNING: Too few stable calibration windows, using identity")
            self.P_ref = np.eye(14)
        else:
            covs_array = np.array(covs)
            self.P_ref = compute_frechet_mean(covs_array, max_iter=30)
        
        self.is_calibrated = True
        self.smoothed_prediction = 0.5
        print(f"[Engine] Calibrated with {len(covs)} stable windows")
    
    def _check_window_validity(self, sigma: np.ndarray) -> bool:
        """
        Window acceptance policy.
        
        Reject if:
          - Global SQI collapse
          - Too many bad channels
        """
        sqi_mean = np.mean(sigma)
        if sqi_mean < self.MIN_GLOBAL_SQI:
            return False
        return True
    
    def infer(self, window: np.ndarray) -> InferenceResult:
        """
        Run inference on a single EEG window.
        
        Args:
            window: Shape (14, 512) — single EEG window (raw or filtered)
        
        Returns:
            InferenceResult with prediction, confidence, and diagnostics
        """
        self.window_count += 1
        
        # Step 1: Bandpass filter
        window_filtered = bandpass_filter(window)
        
        # Step 2: SQI computation
        sigma, Sigma = compute_sqi(window_filtered)
        sqi_mean = np.mean(sigma)
        
        # Step 3: Window acceptance check
        if not self._check_window_validity(sigma):
            self.rejection_count += 1
            return InferenceResult(
                prediction=-1,
                probability=0.0,
                smoothed_output=self.smoothed_prediction,
                mode=ExecutionMode.SAFE,
                sqi_mean=sqi_mean,
                kappa=float('inf'),
                accepted=False,
                raw_score=0.0
            )
        
        # Step 4: Covariance + SPD
        P_spd, kappa, _ = compute_stabilized_covariance(window_filtered, sigma)
        
        # Step 5: Mode routing
        decision = route_window(sigma, kappa)
        
        # Step 6: Feature extraction + Classification
        raw_score = 0.5
        probability = 0.5
        prediction = 0
        
        if decision.mode == ExecutionMode.RARD and self.clf_rard is not None and self.P_ref is not None:
            try:
                features = extract_rard_features(P_spd, self.P_ref)
                if np.all(np.isfinite(features)):
                    if self.scaler_rard is not None:
                        features = self.scaler_rard.transform(features.reshape(1, -1))
                    else:
                        features = features.reshape(1, -1)
                    
                    prediction = int(self.clf_rard.predict(features)[0])
                    if hasattr(self.clf_rard, 'predict_proba'):
                        probability = float(self.clf_rard.predict_proba(features)[0, 1])
                    raw_score = probability
                else:
                    decision = RoutingDecision(
                        mode=ExecutionMode.MVES,
                        sqi_mean=sqi_mean, kappa=kappa,
                        n_bad_channels=0, confidence=0.5
                    )
            except Exception:
                decision = RoutingDecision(
                    mode=ExecutionMode.MVES,
                    sqi_mean=sqi_mean, kappa=kappa,
                    n_bad_channels=0, confidence=0.5
                )
        
        if decision.mode == ExecutionMode.MVES and self.clf_mves is not None:
            features = extract_mves_features(window_filtered, sigma)
            if np.all(np.isfinite(features)):
                if self.scaler_mves is not None:
                    features = self.scaler_mves.transform(features.reshape(1, -1))
                else:
                    features = features.reshape(1, -1)
                
                prediction = int(self.clf_mves.predict(features)[0])
                if hasattr(self.clf_mves, 'predict_proba'):
                    probability = float(self.clf_mves.predict_proba(features)[0, 1])
                raw_score = probability
        
        # Step 7: Temporal smoothing
        # ŷ_t = γ × y_t + (1 - γ) × ŷ_{t-1}
        self.smoothed_prediction = (
            self.GAMMA * raw_score +
            (1 - self.GAMMA) * self.smoothed_prediction
        )
        
        return InferenceResult(
            prediction=prediction,
            probability=probability,
            smoothed_output=self.smoothed_prediction,
            mode=decision.mode,
            sqi_mean=sqi_mean,
            kappa=kappa,
            accepted=True,
            raw_score=raw_score
        )
    
    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            'total_windows': self.window_count,
            'rejected_windows': self.rejection_count,
            'acceptance_rate': (self.window_count - self.rejection_count) / max(self.window_count, 1),
            'is_calibrated': self.is_calibrated,
            'current_smoothed': self.smoothed_prediction,
        }
