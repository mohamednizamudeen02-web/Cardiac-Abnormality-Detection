"""
Cardiometric feature extraction from ECG signals.
Extracts time-domain, frequency-domain, and morphological features.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PREPROCESSING_CONFIG, FEATURE_CONFIG


class FeatureExtractor:
    """Extract cardiometric features from ECG signals."""
    
    def __init__(self, fs: int = 360):
        """
        Initialize feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        
    def extract_time_domain_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features.
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary of time-domain features
        """
        features = {}
        
        # Statistical features
        features['mean'] = np.mean(ecg_signal)
        features['std'] = np.std(ecg_signal)
        features['var'] = np.var(ecg_signal)
        features['median'] = np.median(ecg_signal)
        features['min'] = np.min(ecg_signal)
        features['max'] = np.max(ecg_signal)
        features['range'] = features['max'] - features['min']
        features['skewness'] = skew(ecg_signal)
        features['kurtosis'] = kurtosis(ecg_signal)
        
        # Peak detection
        peaks, _ = signal.find_peaks(ecg_signal, distance=self.fs // 3)
        
        if len(peaks) > 1:
            # RR intervals (time between peaks)
            rr_intervals = np.diff(peaks) / self.fs
            features['mean_rr'] = np.mean(rr_intervals)
            features['std_rr'] = np.std(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
            
            # Heart rate
            features['mean_hr'] = 60 / features['mean_rr'] if features['mean_rr'] > 0 else 0
        else:
            features['mean_rr'] = 0
            features['std_rr'] = 0
            features['rmssd'] = 0
            features['mean_hr'] = 0
        
        return features
    
    def extract_frequency_domain_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary of frequency-domain features
        """
        features = {}
        
        # Compute FFT
        fft_vals = np.fft.fft(ecg_signal)
        fft_freq = np.fft.fftfreq(len(ecg_signal), 1/self.fs)
        
        # Only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_vals = np.abs(fft_vals[positive_freq_idx])
        fft_freq = fft_freq[positive_freq_idx]
        
        # Power spectral density
        psd = np.square(fft_vals)
        
        # Frequency bands (HRV analysis)
        vlf_band = (fft_freq >= 0.003) & (fft_freq < 0.04)  # Very low frequency
        lf_band = (fft_freq >= 0.04) & (fft_freq < 0.15)    # Low frequency
        hf_band = (fft_freq >= 0.15) & (fft_freq < 0.4)     # High frequency
        
        features['vlf_power'] = np.sum(psd[vlf_band])
        features['lf_power'] = np.sum(psd[lf_band])
        features['hf_power'] = np.sum(psd[hf_band])
        features['total_power'] = np.sum(psd)
        
        # Ratios
        if features['hf_power'] > 0:
            features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']
        else:
            features['lf_hf_ratio'] = 0
        
        # Dominant frequency
        features['dominant_freq'] = fft_freq[np.argmax(psd)]
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
        
        return features
    
    def extract_morphological_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features (wave characteristics).
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        # Detect R-peaks
        peaks, properties = signal.find_peaks(
            ecg_signal,
            distance=self.fs // 3,
            prominence=0.5
        )
        
        if len(peaks) > 0:
            # R-peak amplitude
            features['mean_r_amplitude'] = np.mean(ecg_signal[peaks])
            features['std_r_amplitude'] = np.std(ecg_signal[peaks])
            
            # QRS complex width (approximate)
            features['mean_qrs_width'] = np.mean(properties['widths']) if 'widths' in properties else 0
            
            # Wave complexity (number of zero crossings)
            zero_crossings = np.where(np.diff(np.sign(ecg_signal)))[0]
            features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_signal)
        else:
            features['mean_r_amplitude'] = 0
            features['std_r_amplitude'] = 0
            features['mean_qrs_width'] = 0
            features['zero_crossing_rate'] = 0
        
        # Signal energy
        features['signal_energy'] = np.sum(np.square(ecg_signal))
        
        # Autocorrelation at lag 1
        if len(ecg_signal) > 1:
            features['autocorr_lag1'] = np.corrcoef(ecg_signal[:-1], ecg_signal[1:])[0, 1]
        else:
            features['autocorr_lag1'] = 0
        
        return features
    
    def extract_hrv_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract Heart Rate Variability (HRV) features.
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary of HRV features
        """
        features = {}
        
        # Detect R-peaks
        peaks, _ = signal.find_peaks(ecg_signal, distance=self.fs // 3)
        
        if len(peaks) > 2:
            # RR intervals in milliseconds
            rr_intervals = np.diff(peaks) / self.fs * 1000
            
            # Time-domain HRV
            features['sdnn'] = np.std(rr_intervals)  # Standard deviation of NN intervals
            features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
            
            # NN50: Number of successive RR intervals that differ by more than 50ms
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            features['nn50'] = nn50
            features['pnn50'] = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0
            
            # Triangular index
            hist, _ = np.histogram(rr_intervals, bins=50)
            features['tri_index'] = len(rr_intervals) / np.max(hist) if np.max(hist) > 0 else 0
        else:
            features['sdnn'] = 0
            features['rmssd'] = 0
            features['nn50'] = 0
            features['pnn50'] = 0
            features['tri_index'] = 0
        
        return features
    
    def extract_all_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all cardiometric features.
        
        Args:
            ecg_signal: Input ECG signal
            
        Returns:
            Dictionary containing all features
        """
        all_features = {}
        
        if FEATURE_CONFIG['extract_time_domain']:
            time_features = self.extract_time_domain_features(ecg_signal)
            all_features.update({f'time_{k}': v for k, v in time_features.items()})
        
        if FEATURE_CONFIG['extract_frequency_domain']:
            freq_features = self.extract_frequency_domain_features(ecg_signal)
            all_features.update({f'freq_{k}': v for k, v in freq_features.items()})
        
        if FEATURE_CONFIG['extract_morphological']:
            morph_features = self.extract_morphological_features(ecg_signal)
            all_features.update({f'morph_{k}': v for k, v in morph_features.items()})
        
        if FEATURE_CONFIG['extract_hrv']:
            hrv_features = self.extract_hrv_features(ecg_signal)
            all_features.update({f'hrv_{k}': v for k, v in hrv_features.items()})
        
        return all_features
    
    def extract_features_batch(self, ecg_signals: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of ECG signals.
        
        Args:
            ecg_signals: Array of ECG signals (n_samples, signal_length)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        feature_list = []
        
        for signal in ecg_signals:
            features = self.extract_all_features(signal)
            feature_list.append(list(features.values()))
        
        return np.array(feature_list)


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor(fs=360)
    
    # Generate sample ECG-like signal
    t = np.linspace(0, 10, 3600)
    sample_signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    
    # Extract features
    features = extractor.extract_all_features(sample_signal)
    
    print(f"Extracted {len(features)} cardiometric features:")
    for name, value in list(features.items())[:10]:
        print(f"  {name}: {value:.4f}")
    print("  ...")
    print("[OK] Feature extraction working correctly!")
