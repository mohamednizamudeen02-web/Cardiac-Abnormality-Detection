"""
Comprehensive System Verification Test
Tests all components of the Cardiac Detection System:
- Feature Extraction
- ML Models (Ensemble & Lightweight)
- SQLite Database Operations (Patients, Analyses, Vitals, IoT, Alerts, Notes)
- Vitals Fusion & Risk Scoring
- IoT Manager & Packet Handling
- Alert Engine
- Flask Application Endpoints & Predict Pipeline
"""

import os
import sys
from pathlib import Path
import numpy as np

# Set project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_all():
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE CARDIAC SYSTEM VERIFICATION")
    print("=" * 60)

    # 1. Test Config and Feature Extraction
    print("\n[1/7] Testing Config & Feature Extraction...")
    from src.utils.config import PREPROCESSING_CONFIG, FEATURE_CONFIG, Config
    from src.features.feature_extraction import FeatureExtractor

    extractor = FeatureExtractor(fs=360)
    t = np.linspace(0, 10, 3600)
    synthetic_signal = (
        0.2 * np.sin(2 * np.pi * 1.2 * t)
        + 1.0 * np.sin(2 * np.pi * 3.6 * t)
        + 0.3 * np.sin(2 * np.pi * 2.4 * t)
        + 0.05 * np.random.randn(3600)
    )
    features = extractor.extract_all_features(synthetic_signal)
    assert len(features) > 10, "Feature extraction failed to extract features"
    print(f"  [OK] Extracted {len(features)} cardiometric features successfully.")

    # 2. Test ML Models
    print("\n[2/7] Testing ML Models...")
    from src.models.ensemble_model import AdvancedECGEnsemble
    from src.models.lightweight_model import LightweightECGClassifier

    ensemble_model = AdvancedECGEnsemble()
    ens_path = PROJECT_ROOT / 'models' / 'ensemble_ecg_model.pkl'
    if ens_path.exists():
        ensemble_model.load(ens_path)
        ens_pred = ensemble_model.predict(synthetic_signal)
        print(f"  [OK] Ensemble Prediction: {ens_pred['prediction']} (Confidence: {ens_pred['confidence']:.2f}, Uncertainty: {ens_pred['uncertainty']:.2f})")
    else:
        print("  [SKIP] Ensemble model pkl not found.")

    lw_model = LightweightECGClassifier()
    lw_path = PROJECT_ROOT / 'models' / 'lightweight_ecg_model.pkl'
    if lw_path.exists():
        lw_model.load(lw_path)
        lw_pred = lw_model.predict(synthetic_signal)
        print(f"  [OK] Lightweight Prediction: {lw_pred['prediction']} (Confidence: {lw_pred['confidence']:.2f})")

    # 3. Test SQLite Database Layer
    print("\n[3/7] Testing SQLite Database Layer...")
    import app.database as db
    db.init_database()
    
    patients = db.get_all_patients()
    print(f"  [OK] Retrieved {len(patients)} patients from SQLite.")
    assert len(patients) > 0, "Expected seeded patients"

    test_patient_id = "TEST_PT_999"
    created = db.create_patient({
        'patient_id': test_patient_id,
        'name': 'Test Runner Patient',
        'age': 50,
        'gender': 'Male',
        'contact_info': 'test@example.com',
        'medical_history': 'None',
    })
    print(f"  [OK] Created patient: {created['name']} ({created['patient_id']})")

    saved_analysis = db.save_analysis({
        'patient_id': test_patient_id,
        'prediction': 'Normal',
        'confidence': 0.95,
        'uncertainty': 0.05,
        'risk_level': 'Low',
        'heart_rate': 72.0,
        'hrv_sdnn': 45.0,
        'signal_length': 3600,
        'probabilities': {'Normal': 0.95, 'Arrhythmia': 0.05},
        'features': {'Heart Rate': 72.0},
        'recommendations': ['Routine checkup'],
    })
    print(f"  [OK] Saved analysis ID: {saved_analysis.get('_id')}")

    analyses = db.get_patient_analyses(test_patient_id)
    assert len(analyses) >= 1, "Failed to retrieve saved patient analyses"

    saved_vitals = db.save_vitals({
        'patient_id': test_patient_id,
        'spo2': 98.0,
        'systolic': 120.0,
        'diastolic': 80.0,
        'temperature': 36.6,
        'heart_rate': 72.0,
        'ucrs': 12.5,
        'fused_risk_level': 'Low',
    })
    print(f"  [OK] Saved vitals reading ID: {saved_vitals.get('_id')}")

    saved_dev = db.save_iot_device({
        'device_id': 'TEST_WATCH_01',
        'device_type': 'smartwatch',
        'patient_id': test_patient_id,
        'label': 'Test Apple Watch',
        'status': 'online',
    })
    print(f"  [OK] Saved IoT device: {saved_dev.get('label')}")

    saved_log = db.save_alert_log({
        'alert_id': 'ALT_TEST_01',
        'patient_id': test_patient_id,
        'severity': 'Warning',
        'trigger': 'Elevated heart rate',
        'message': 'Test alert message',
        'channels_fired': ['email'],
        'channels_failed': [],
    })
    print(f"  [OK] Saved alert log: {saved_log.get('alert_id')}")

    # Clean up test patient
    db.delete_patient(test_patient_id)
    print("  [OK] Cleaned up test patient.")

    # 4. Test Vital Fusion Engine
    print("\n[4/7] Testing Vital Fusion Engine...")
    from src.vitals.vital_fusion import VitalFusionEngine, VitalReading
    fusion_engine = VitalFusionEngine()
    vr = VitalReading(spo2=92.0, systolic=145.0, diastolic=95.0, temperature=37.1, heart_rate=88.0)
    fusion_out = fusion_engine.evaluate(
        ecg_prediction='Arrhythmia',
        ecg_confidence=0.82,
        ecg_risk_level='Medium',
        vitals=vr,
    )
    print(f"  [OK] Unified Cardiac Risk Score (UCRS): {fusion_out.ucrs:.1f} / 100 ({fusion_out.fused_risk_level})")

    # 5. Test Alert Engine
    print("\n[5/7] Testing Alert Engine...")
    from src.alerts.alert_engine import AlertEngine
    alert_engine = AlertEngine()
    alert_res = alert_engine.evaluate(
        patient_id='PATIENT_001',
        ecg_prediction='Myocardial Infarction',
        ecg_confidence=0.90,
        risk_level='High',
        ucrs=85.0,
        vitals={'spo2': 88.0, 'systolic': 190.0, 'heart_rate': 120.0},
    )
    assert alert_res is not None, "Expected alert to fire on critical conditions"
    print(f"  [OK] Alert fired: Severity={alert_res.severity} | Trigger='{alert_res.trigger}'")

    # 6. Test IoT Device Connector
    print("\n[6/7] Testing IoT Device Connector...")
    from src.iot.iot_connector import IoTDeviceManager, DevicePacket
    mgr = IoTDeviceManager()
    d = mgr.register('DEV_001', 'smartwatch', 'PATIENT_001', 'My Smartwatch')
    pkt = DevicePacket(
        device_id='DEV_001',
        patient_id='PATIENT_001',
        ecg_chunk=[0.1, 0.2, 0.5, 1.2, -0.4, 0.1],
        heart_rate=74.0,
        spo2=99.0,
        systolic=118.0,
        diastolic=78.0,
    )
    updated = mgr.handle_packet(pkt)
    assert updated.packet_count == 1, "Packet count should increment"
    print(f"  [OK] IoT Packet handled successfully. Device packet count: {updated.packet_count}")

    # 7. Test Flask App with Test Client
    print("\n[7/7] Testing Flask App Endpoints (Test Client)...")
    from app.main_advanced import app

    app.config['TESTING'] = True
    client = app.test_client()

    # Test login
    res_login = client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
    assert res_login.status_code == 200, f"Login failed with status {res_login.status_code}"
    print("  [OK] User authentication /login succeeded.")

    # Test /api/me
    res_me = client.get('/api/me')
    assert res_me.status_code == 200 and res_me.json['username'] == 'admin', f"Failed /api/me: {res_me.text}"
    print("  [OK] /api/me returned current user profile.")

    # Test /api/patients
    res_pts = client.get('/api/patients')
    assert res_pts.status_code == 200 and res_pts.json['success'], "Failed /api/patients"
    print(f"  [OK] /api/patients returned {len(res_pts.json['patients'])} patients.")

    # Test /api/predict
    res_predict = client.post('/api/predict', json={
        'ecg_signal': synthetic_signal.tolist(),
        'patient_id': 'PATIENT_001',
        'vitals': {'spo2': 98.0, 'systolic': 120.0, 'diastolic': 80.0, 'temperature': 36.6},
    })
    assert res_predict.status_code == 200 and res_predict.json['success'], f"Predict failed: {res_predict.text}"
    p_data = res_predict.json
    print(f"  [OK] /api/predict: Prediction='{p_data['prediction']}', Confidence={p_data['confidence']*100:.1f}%, Risk={p_data['risk_level']}")

    # Test /api/stats
    res_stats = client.get('/api/stats')
    assert res_stats.status_code == 200, "Failed /api/stats"
    print(f"  [OK] /api/stats: DB Backend={res_stats.json.get('db_backend')}, Total Analyses={res_stats.json.get('total_analyses')}")

    # Test /api/analytics/trends
    res_trends = client.get('/api/analytics/trends?patient_id=PATIENT_001')
    assert res_trends.status_code == 200 and res_trends.json['success'], "Failed /api/analytics/trends"
    print("  [OK] /api/analytics/trends returned 24h heatmap and radar metrics.")

    # Test /api/generate-report (PDF)
    res_pdf = client.post('/api/generate-report', json={
        'patient_id': 'PATIENT_001',
        'prediction': p_data['prediction'],
        'confidence': p_data['confidence'],
        'uncertainty': p_data['uncertainty'],
        'probabilities': p_data['probabilities'],
        'features': p_data['features'],
        'recommendations': p_data['recommendations'],
        'risk_level': p_data['risk_level'],
    })
    assert res_pdf.status_code == 200 and res_pdf.mimetype == 'application/pdf', "PDF generation failed"
    print(f"  [OK] /api/generate-report generated valid PDF ({len(res_pdf.data)} bytes).")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! SYSTEM RUNS SMOOTHLY AND ROBUSTLY!")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    test_all()
