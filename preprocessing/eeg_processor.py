import numpy as np
from scipy.signal import butter, sosfiltfilt

class EEGProcessor:
    """
    Handles preprocessing of raw EEG signals for inference.
    Applies the same protocol used during training:
    Bandpass -> CAR -> MAD Clip -> Z-Score.
    """
    def __init__(self, sfreq=128):
        self.sfreq = sfreq

    def bandpass_filter(self, sig, lo=1.0, hi=40.0, order=4):
        nyq = self.sfreq / 2
        sos = butter(order, [lo/nyq, hi/nyq], btype="band", output="sos")
        return sosfiltfilt(sos, sig, axis=0).astype(np.float32)

    def common_average_reference(self, sig):
        return (sig - sig.mean(axis=1, keepdims=True)).astype(np.float32)

    def mad_clip(self, sig, k=5.0):
        med = np.median(sig, axis=0)
        mad = np.median(np.abs(sig - med), axis=0)
        # Avoid division by zero if mad is 0
        mad[mad == 0] = 1e-6
        return np.clip(sig, med - k*mad, med + k*mad).astype(np.float32)

    def zscore_normalize(self, sig):
        mu = sig.mean(axis=0)
        std = sig.std(axis=0)
        std[std == 0] = 1.0
        return ((sig - mu) / std).astype(np.float32)

    def preprocess_window(self, sig):
        """
        Process a single (N_samples, N_channels) array.
        For example, (256, 14) for 2 seconds at 128Hz.
        """
        sig = self.bandpass_filter(sig)
        sig = self.common_average_reference(sig)
        sig = self.mad_clip(sig)
        sig = self.zscore_normalize(sig)
        return sig
