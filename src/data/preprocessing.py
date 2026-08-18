"""
ECG signal preprocessing module.
Includes filtering, normalization, and segmentation.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


class ECGPreprocessor:
    """Preprocess ECG signals."""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.scaler = None
    
    def remove_baseline_wander(self, ecg_signal, fs=None):
        if fs is None:
            fs = self.config.SAMPLING_RATE
        nyq = 0.5 * fs
        cutoff = self.config.BASELINE_FILTER_CUTOFF / nyq
        b, a = butter(self.config.FILTER_ORDER, cutoff, btype='high')
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def bandpass_filter(self, ecg_signal, fs=None):
        if fs is None:
            fs = self.config.SAMPLING_RATE
        nyq = 0.5 * fs
        low = self.config.FILTER_LOWCUT / nyq
        high = self.config.FILTER_HIGHCUT / nyq
        b, a = butter(self.config.FILTER_ORDER, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def normalize(self, ecg_signal, method=None):
        if method is None:
            method = self.config.NORMALIZATION_METHOD
        ecg_signal = ecg_signal.reshape(-1, 1)
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized = self.scaler.fit_transform(ecg_signal)
            else:
                normalized = self.scaler.transform(ecg_signal)
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                normalized = self.scaler.fit_transform(ecg_signal)
            else:
                normalized = self.scaler.transform(ecg_signal)
        else:
            normalized = ecg_signal
        return normalized.flatten()
    
    def pad_or_truncate(self, ecg_signal, target_length=None):
        if target_length is None:
            target_length = self.config.SIGNAL_LENGTH
        current_length = len(ecg_signal)
        if current_length < target_length:
            padding = target_length - current_length
            ecg_signal = np.pad(ecg_signal, (0, padding), mode='constant')
        elif current_length > target_length:
            ecg_signal = ecg_signal[:target_length]
        return ecg_signal
    
    def preprocess_pipeline(self, ecg_signal, apply_filter=True, 
                           apply_normalization=True, target_length=None):
        if apply_filter:
            ecg_signal = self.remove_baseline_wander(ecg_signal)
            ecg_signal = self.bandpass_filter(ecg_signal)
        if target_length:
            ecg_signal = self.pad_or_truncate(ecg_signal, target_length)
        if apply_normalization:
            ecg_signal = self.normalize(ecg_signal)
        return ecg_signal
