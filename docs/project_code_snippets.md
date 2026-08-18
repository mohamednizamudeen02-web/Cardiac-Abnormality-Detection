# Cardiac Detection Project - Core Code Snippets

## 1. ECG Signal Feature Extraction
```python
# File: src/models/lightweight_model.py
    def extract_features(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Extract a rich feature vector from a 1-D ECG signal."""
        sig = np.asarray(ecg_signal, dtype=np.float64)
        n = len(sig)
        if n == 0:
            return np.zeros(_NUM_FEATURES)

        features: List[float] = []

        # Statistical
        mean = float(np.mean(sig))
        std = float(np.std(sig)) or 1e-9
        features += [mean, std, float(np.min(sig)), float(np.max(sig)), float(np.median(sig)), float(np.percentile(sig, 25)), float(np.percentile(sig, 75))]

        # Time-domain HRV (RMSSD, pNN50)
        diffs = np.diff(sig)
        rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
        pnn50 = float(np.sum(np.abs(diffs) > 0.05) / len(diffs) * 100) if len(diffs) > 0 else 0.0
        features += [rmssd, pnn50]

        # Shape (skewness & kurtosis)
        normalised = (sig - mean) / std
        skewness = float(np.mean(normalised ** 3))
        kurtosis = float(np.mean(normalised ** 4))
        features += [skewness, kurtosis]

        return np.array(features)
```

## 2. Cardiac Condition Prediction using Random Forest
```python
# File: src/models/lightweight_model.py
    def predict(self, ecg_signal: np.ndarray) -> dict:
        """Predict cardiac condition from a raw ECG signal."""
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

        return {
            'prediction': self.CLASS_LABELS[pred_class],
            'predicted_class': pred_class,
            'probabilities': {self.CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)},
            'confidence': float(np.max(proba)),
            'uncertainty': uncertainty,
            'feature_importance': self.model.feature_importances_.tolist(),
        }
```

## 3. Automated Emergency Threshold Evaluation
```python
# File: src/alerts/alert_engine.py
    def _check_thresholds(self, ecg_pred, ecg_conf, risk_level, ucrs, vitals):
        """Returns (severity, trigger_description) or (None, None) if no alert."""
        t = self.thresholds

        # UCRS critical
        if ucrs is not None and ucrs >= t.get('ucrs_critical', 70):
            return 'Critical', f'Unified Risk Score {ucrs:.1f}/100 exceeds critical threshold'

        # ECG Critical classes
        if ecg_pred in t.get('ecg_classes_critical', []) and ecg_conf >= t.get('ecg_confidence_min', 0.5):
            return 'Critical', f'ECG classified as {ecg_pred} (confidence {ecg_conf*100:.0f}%)'

        # Vitals
        if vitals:
            spo2 = vitals.get('spo2')
            sbp  = vitals.get('systolic')
            hr   = vitals.get('heart_rate')
            if spo2 is not None and spo2 < t.get('spo2_critical', 90):
                return 'Critical', f'SpO2 critically low at {spo2:.0f}%'
            if sbp is not None and (sbp >= t.get('sbp_high_crit', 180) or sbp <= t.get('sbp_low_crit', 80)):
                return 'Critical', f'Blood pressure critical: {sbp:.0f} mmHg systolic'
            if hr is not None and (hr >= t.get('hr_high_crit', 150) or hr <= t.get('hr_low_crit', 40)):
                return 'Critical', f'Heart rate critical: {hr:.0f} bpm'

        return None, None
```

## 4. Multi-Channel Alert Dispatch (SMS/WhatsApp)
```python
# File: src/alerts/alert_engine.py
    def _send_sms(self, phone: str, event: AlertEvent) -> bool:
        try:
            sid   = os.environ.get('TWILIO_ACCOUNT_SID', '')
            token = os.environ.get('TWILIO_AUTH_TOKEN', '')
            from_ = os.environ.get('TWILIO_FROM_NUMBER', '')
            if not all([sid, token, from_, phone]):
                print(f"[ALERT] SMS skipped (Twilio not configured) → {phone}")
                return False
            
            from twilio.rest import Client
            body = f"[{event.severity}] Cardiac Alert – Pt {event.patient_id}: {event.trigger}"
            Client(sid, token).messages.create(body=body, from_=from_, to=phone)
            print(f"[ALERT] ✅ SMS sent → {phone}")
            return True
        except Exception as exc:
            print(f"[ALERT] ❌ SMS failed → {phone}: {exc}")
            return False
```

## 5. MongoDB Database Connection & Seeding
```python
# File: app/mongodb_database.py
def init_database() -> None:
    """Create indexes and seed sample patients if collection is empty."""
    try:
        col = patients_col()

        # Indexes
        col.create_index([('patient_id', ASCENDING)], unique=True, name='idx_patient_id')
        analyses_col().create_index([('patient_id', ASCENDING)], name='idx_analysis_patient')

        # Seed sample patients (upsert by patient_id)
        seeded = 0
        for p in _SAMPLE_PATIENTS:
            result = col.update_one(
                {'patient_id': p['patient_id']},
                {'$setOnInsert': p},
                upsert=True,
            )
            if result.upserted_id:
                seeded += 1

        if seeded:
            print(f'[OK] Seeded {seeded} sample patient(s) into MongoDB')
            
    except Exception as exc:
        print(f'[ERROR] MongoDB init error: {exc}')
        raise
```
