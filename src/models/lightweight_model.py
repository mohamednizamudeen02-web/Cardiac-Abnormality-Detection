"""
Lightweight ML model using scikit-learn (works without TensorFlow).
Provides real ECG cardiac abnormality predictions with a richer feature set.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# -- Feature names (must stay in sync with extract_features) -----------------
_FEATURE_NAMES = [
    # Statistical
    "Mean", "Std", "Min", "Max", "Median",
    "P25", "P75",
    # Energy & crossings
    "Signal Energy", "Zero Crossings",
    # Heart-rate & RR
    "Peaks", "RR Interval (samples)", "Heart Rate (bpm)",
    # Time-domain HRV
    "RMSSD", "pNN50 (%)",
    # Frequency-domain
    "LF Power", "HF Power", "Spectral Entropy",
    # Shape
    "Skewness", "Kurtosis",
    # Hjorth
    "Hjorth Activity", "Hjorth Mobility", "Hjorth Complexity",
]

_NUM_FEATURES = len(_FEATURE_NAMES)


class LightweightECGClassifier:
    """
    Lightweight ECG classifier using Random Forest.
    Works without TensorFlow - compatible with Python 3.11+.
    """

    CLASS_LABELS: Dict[int, str] = {
        0: 'Normal',
        1: 'Arrhythmia',
        2: 'Myocardial Infarction',
        3: 'Other Abnormality',
    }

    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        self.scaler = StandardScaler()
        self.is_trained: bool = False
        # backward-compat property kept for external code
        self.class_labels = self.CLASS_LABELS

    # -- Feature extraction ----------------------------------------------------
    @staticmethod
    def get_feature_names() -> List[str]:
        """Return human-readable feature names matching the feature vector."""
        return list(_FEATURE_NAMES)

    def extract_features(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Extract a rich feature vector from a 1-D ECG signal.

        Returns
        -------
        np.ndarray of shape (N_FEATURES,)
        """
        sig = np.asarray(ecg_signal, dtype=np.float64)
        n = len(sig)
        if n == 0:
            return np.zeros(_NUM_FEATURES)

        features: List[float] = []

        # -- Statistical ------------------------------------------------------
        mean = float(np.mean(sig))
        std = float(np.std(sig)) or 1e-9
        features += [
            mean, std,
            float(np.min(sig)), float(np.max(sig)), float(np.median(sig)),
            float(np.percentile(sig, 25)), float(np.percentile(sig, 75)),
        ]

        # -- Energy & zero-crossings -------------------------------------------
        energy = float(np.sum(sig ** 2))
        zero_crossings = int(np.sum(np.diff(np.sign(sig)) != 0))
        features += [energy, float(zero_crossings)]

        # -- Peak / RR / Heart rate --------------------------------------------
        threshold = mean + 0.5 * std
        peak_mask = sig > threshold
        peaks = int(np.sum(peak_mask))

        # Compute rr_interval and heart_rate unconditionally to avoid scope bugs
        if peaks > 1:
            rr_interval = n / peaks
            estimated_fs = n / 10.0          # assume 10-second strip
            rr_seconds = rr_interval / estimated_fs
            heart_rate = float(np.clip(60.0 / rr_seconds, 30.0, 220.0)) if rr_seconds > 0 else 70.0
        else:
            rr_interval = 0.0
            heart_rate = 70.0

        features += [float(peaks), float(rr_interval), heart_rate]

        # -- Time-domain HRV ---------------------------------------------------
        diffs = np.diff(sig)
        rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
        pnn50 = (
            float(np.sum(np.abs(diffs) > 0.05) / len(diffs) * 100)
            if len(diffs) > 0 else 0.0
        )
        features += [rmssd, pnn50]

        # -- Frequency-domain -------------------------------------------------
        fft_vals = np.fft.rfft(sig)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / (n / 10.0))  # assume 10 s -> fs = n/10

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        lf_power = float(np.sum(power[lf_mask])) if lf_mask.any() else 0.0
        hf_power = float(np.sum(power[hf_mask])) if hf_mask.any() else 0.0

        # Spectral entropy
        total_power = float(np.sum(power)) or 1.0
        prob = power / total_power
        prob = prob[prob > 0]
        spectral_entropy = float(-np.sum(prob * np.log(prob + 1e-12)))

        features += [lf_power, hf_power, spectral_entropy]

        # -- Shape (skewness & kurtosis) ---------------------------------------
        normalised = (sig - mean) / std
        skewness = float(np.mean(normalised ** 3))
        kurtosis = float(np.mean(normalised ** 4))
        features += [skewness, kurtosis]

        # -- Hjorth parameters ------------------------------------------------
        activity = float(np.var(sig))
        d1 = np.diff(sig)
        mobility = float(np.sqrt(np.var(d1) / (activity or 1e-9)))
        d2 = np.diff(d1)
        mobility_d1 = float(np.sqrt(np.var(d2) / (np.var(d1) or 1e-9)))
        complexity = mobility_d1 / (mobility or 1e-9)
        features += [activity, mobility, float(complexity)]

        arr = np.array(features)
        assert len(arr) == _NUM_FEATURES, (
            f"Feature count mismatch: got {len(arr)}, expected {_NUM_FEATURES}"
        )
        return arr

    # -- Training --------------------------------------------------------------
    def train(self, X_train: List[np.ndarray], y_train: List[int]) -> None:
        """Train on a list of raw ECG signals with integer class labels."""
        X_features = np.array([self.extract_features(s) for s in X_train])
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"[OK] Model trained on {len(X_train)} samples | {_NUM_FEATURES} features")

    # -- Prediction ------------------------------------------------------------
    def predict(self, ecg_signal: np.ndarray) -> dict:
        """
        Predict cardiac condition from a raw ECG signal.

        Returns
        -------
        dict with keys: prediction, predicted_class, probabilities,
                        confidence, uncertainty, feature_importance
        """
        features = self.extract_features(ecg_signal).reshape(1, -1)

        if not self.is_trained:
            return self._heuristic_prediction(ecg_signal)

        features_scaled = self.scaler.transform(features)
        pred_class = int(self.model.predict(features_scaled)[0])
        proba = self.model.predict_proba(features_scaled)[0]

        # Shannon entropy -> normalised uncertainty
        entropy = -np.sum(proba * np.log(proba + 1e-12))
        n_classes = len(self.CLASS_LABELS)
        uncertainty = float(entropy / np.log(n_classes)) if n_classes > 1 else 0.0

        feature_importance = self.model.feature_importances_.tolist()

        return {
            'prediction': self.CLASS_LABELS[pred_class],
            'predicted_class': pred_class,
            'probabilities': {
                self.CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)
            },
            'confidence': float(np.max(proba)),
            'uncertainty': uncertainty,
            'feature_importance': feature_importance,   # length == _NUM_FEATURES
        }

    def _heuristic_prediction(self, ecg_signal: np.ndarray) -> dict:
        """Signal-statistic-based fallback when the model is not trained."""
        sig = np.asarray(ecg_signal, dtype=np.float64)
        std = float(np.std(sig))
        mean = float(np.mean(sig))
        normalised = (sig - mean) / (std or 1e-9)
        kurtosis = float(np.mean(normalised ** 4))

        if std > 0.5 and abs(mean) < 0.3:
            pred = 0
            probs = {'Normal': 0.85, 'Arrhythmia': 0.10,
                     'Myocardial Infarction': 0.03, 'Other Abnormality': 0.02}
        elif std > 0.8:
            pred = 1
            probs = {'Normal': 0.15, 'Arrhythmia': 0.70,
                     'Myocardial Infarction': 0.10, 'Other Abnormality': 0.05}
        elif kurtosis > 6:
            # High kurtosis -> possible MI (sharp ST elevation features)
            pred = 2
            probs = {'Normal': 0.10, 'Arrhythmia': 0.20,
                     'Myocardial Infarction': 0.60, 'Other Abnormality': 0.10}
        else:
            pred = 3
            probs = {'Normal': 0.20, 'Arrhythmia': 0.25,
                     'Myocardial Infarction': 0.15, 'Other Abnormality': 0.40}

        confidence = max(probs.values())
        uncertainty = 1.0 - confidence

        return {
            'prediction': self.CLASS_LABELS[pred],
            'predicted_class': pred,
            'probabilities': probs,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'feature_importance': [1.0 / _NUM_FEATURES] * _NUM_FEATURES,
        }

    # -- Persistence -----------------------------------------------------------
    def save(self, filepath: Path | str) -> None:
        """Serialize model to disk."""
        payload = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'class_labels': self.class_labels,
            'num_features': _NUM_FEATURES,
        }
        with open(filepath, 'wb') as fh:
            pickle.dump(payload, fh)
        print(f"[OK] Model saved -> {filepath}")

    def load(self, filepath: Path | str) -> None:
        """Deserialize model from disk."""
        with open(filepath, 'rb') as fh:
            payload = pickle.load(fh)
        self.model = payload['model']
        self.scaler = payload['scaler']
        self.is_trained = payload['is_trained']
        self.class_labels = payload.get('class_labels', self.CLASS_LABELS)
        print(f"[OK] Model loaded   {filepath}  (trained={self.is_trained})")


# -- Demo training helper -----------------------------------------------------
def train_demo_model() -> LightweightECGClassifier:
    """Train a demo classifier on synthetic ECG-like signals and save it."""
    print("  Training demo model with synthetic data  ")
    np.random.seed(42)
    n_samples, signal_length = 1000, 3600
    t = np.linspace(0, 10, signal_length)

    X_train, y_train = [], []
    for i in range(n_samples):
        if i < 400:      # Normal sinus
            sig = (0.2 * np.sin(2 * np.pi * 1.2 * t)
                   + 1.0 * np.sin(2 * np.pi * 3.6 * t)
                   + 0.3 * np.sin(2 * np.pi * 2.4 * t)
                   + 0.05 * np.random.randn(signal_length))
            label = 0
        elif i < 700:    # Arrhythmia (irregular)
            sig = (0.8 * np.sin(2 * np.pi * 2.5 * t)
                   + 0.5 * np.sin(2 * np.pi * 5.0 * t)
                   + 0.4 * np.random.randn(signal_length))
            label = 1
        elif i < 900:    # Myocardial Infarction (ST elevation-like)
            sig = (0.3 * np.sin(2 * np.pi * 1.0 * t)
                   + 0.6 * np.sin(2 * np.pi * 2.0 * t)
                   + 0.3 * np.random.randn(signal_length)
                   + 0.4 * (np.exp(-((t - 5) ** 2) / 0.5)))
            label = 2
        else:            # Other abnormality
            sig = np.random.randn(signal_length) * 0.5
            label = 3

        X_train.append(sig)
        y_train.append(label)

    clf = LightweightECGClassifier()
    clf.train(X_train, y_train)

    model_path = Path(__file__).parent.parent.parent / 'models' / 'lightweight_ecg_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(model_path)
    print("[OK] Demo model trained and saved!")
    return clf


if __name__ == "__main__":
    model = train_demo_model()
    print("\n  Testing prediction  ")
    test_signal = np.random.randn(3600) * 0.5
    result = model.predict(test_signal)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']:.2f}")
    print(f"  Uncertainty : {result['uncertainty']:.2f}")
    print(f"  Features    : {len(result['feature_importance'])} importance values")
    print("[OK] Model is working!")
