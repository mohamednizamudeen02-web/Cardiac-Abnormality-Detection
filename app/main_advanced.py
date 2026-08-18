import os
import sys
import warnings

warnings.filterwarnings('ignore')

# --- Windows Stability Fixes ---
# 1. Prevent Access Violation (0xC0000005) from conflicting OpenMP runtimes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 2. Prevent threading conflicts on Windows with heavy ML libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import json
import io
import traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

load_dotenv()

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
from functools import wraps

sys.path.append(str(Path(__file__).parent.parent))

# -- New Real-World Integration modules ---------------------------------------
from src.vitals.vital_fusion import VitalFusionEngine, VitalReading
from src.iot.iot_connector import IoTDeviceManager, DevicePacket, DeviceSimulator, device_manager
from src.alerts.alert_engine import AlertEngine, alert_engine

_vital_fusion = VitalFusionEngine()
_simulators: dict = {}   # patient_id -> DeviceSimulator (for demo streaming)

# Auto-detect MongoDB: enable if MONGO_URI is set in the environment
_USE_MONGO = bool(os.environ.get('MONGO_URI'))
if _USE_MONGO:
    import app.mongodb_database as db_layer
else:
    import app.database as db_layer

# Unified database API accessors
init_database = db_layer.init_database
get_all_patients = db_layer.get_all_patients
get_patient = db_layer.get_patient
create_patient = db_layer.create_patient
update_patient = db_layer.update_patient
delete_patient = db_layer.delete_patient
get_patient_analyses = db_layer.get_patient_analyses
save_analysis = db_layer.save_analysis
get_stats = db_layer.get_stats
save_vitals = db_layer.save_vitals
get_patient_vitals = db_layer.get_patient_vitals
save_iot_device = db_layer.save_iot_device
get_all_iot_devices = db_layer.get_all_iot_devices
delete_iot_device = db_layer.delete_iot_device
save_alert_log = db_layer.save_alert_log
get_recent_alert_logs = db_layer.get_recent_alert_logs
save_note = getattr(db_layer, 'save_note', None)

# -- ML model loading (deferred to avoid conflicts on Windows) ---------------
DEMO_MODE = True
ml_model = None

def load_ml_models():
    """Load ML models lazily to avoid library conflicts during startup."""
    global DEMO_MODE, ml_model
    try:
        from src.models.ensemble_model import AdvancedECGEnsemble

        ml_model = AdvancedECGEnsemble()
        model_path = Path(__file__).parent.parent / 'models' / 'ensemble_ecg_model.pkl'

        if model_path.exists():
            ml_model.load(model_path)
            DEMO_MODE = False
            print("[OK] Advanced Ensemble model loaded!")
        else:
            raise FileNotFoundError("Ensemble model file not found")

    except Exception:
        # Fallback: lightweight scikit-learn model
        try:
            from src.models.lightweight_model import LightweightECGClassifier

            ml_model = LightweightECGClassifier()
            lw_path = Path(__file__).parent.parent / 'models' / 'lightweight_ecg_model.pkl'

            if lw_path.exists():
                ml_model.load(lw_path)
                DEMO_MODE = False
                print("[OK] Lightweight ML model loaded!")
            else:
                print("[WARNING] No model file found - using heuristic predictions")
        except Exception as exc:
            print(f"[WARNING] Could not load any ML model: {exc}")
            ml_model = None


# -- Flask / SocketIO setup ----------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('CARDIAC_SECRET_KEY', 'cardiac-monitor-secret-2024')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -- Users (demo credentials - replace with DB-backed auth in production) ------
# access_level: 'admin' = full edit/create/delete  |  'viewer' = read-only
USERS = {
    'admin':   {'password': 'admin123',   'role': 'Administrator', 'name': 'Admin User',    'access_level': 'admin'},
    'doctor':  {'password': 'doctor123',  'role': 'Cardiologist',  'name': 'Dr. Smith',     'access_level': 'admin'},
    'analyst': {'password': 'analyst123', 'role': 'ECG Analyst',   'name': 'Jane Analyst',  'access_level': 'admin'},
    'viewer':  {'password': 'viewer123',  'role': 'Viewer',        'name': 'View Only',     'access_level': 'viewer'},
}

def login_required(f):
    """Decorator that redirects unauthenticated requests to /login."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash('Please sign in to access the dashboard.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """Decorator: reject with 403 JSON if the current user is not an admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        if session.get('access_level') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied - admin role required'}), 403
        return f(*args, **kwargs)
    return decorated

print("=" * 60)
print("INTELLIGENT CARDIAC ABNORMALITY IDENTIFICATION SYSTEM")
print("=" * 60)
print("Features: Real-time WebSocket | Explainable AI | PDF Reports")
if DEMO_MODE:
    print("\n[WARNING] Heuristic mode (ML model not loaded)")
else:
    print("\n[OK] ML mode - Ensemble / scikit-learn model active")
print("=" * 60 + "\n")

# -- Initialising database & models --------------------------------------------
_mongo_init_error = None  # Stores any MongoDB init error for diagnostics
try:
    print("[STATS] Initialising ML models...")
    load_ml_models()

    print("[STATS] Initialising database...")
    init_database()

except Exception as init_err:
    if _USE_MONGO:
        import traceback
        _mongo_init_error = f"{type(init_err).__name__}: {init_err}"
        print(f"[ERROR] MongoDB initialization failed:")
        traceback.print_exc()
        print("[SYSTEM] Falling back to local SQLite database...")
        _USE_MONGO = False
        import app.database as db_layer
        init_database = db_layer.init_database
        get_all_patients = db_layer.get_all_patients
        get_patient = db_layer.get_patient
        create_patient = db_layer.create_patient
        update_patient = db_layer.update_patient
        delete_patient = db_layer.delete_patient
        get_patient_analyses = db_layer.get_patient_analyses
        save_analysis = db_layer.save_analysis
        get_stats = db_layer.get_stats
        save_vitals = db_layer.save_vitals
        get_patient_vitals = db_layer.get_patient_vitals
        save_iot_device = db_layer.save_iot_device
        get_all_iot_devices = db_layer.get_all_iot_devices
        delete_iot_device = db_layer.delete_iot_device
        save_alert_log = db_layer.save_alert_log
        get_recent_alert_logs = db_layer.get_recent_alert_logs
        save_note = getattr(db_layer, 'save_note', None)
        init_database()
    else:
        print(f"  Database initialization failed: {init_err}")
        raise


# -- Load persisted users from MongoDB ----------------------------------------
# -- Local User Persistence (Security Fallback) --------------------------------
_USERS_FILE = Path(__file__).parent.parent / 'data' / 'local_users.json'

def _save_local_users():
    """Save the USERS dictionary to a local JSON file as a fallback."""
    try:
        _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_USERS_FILE, 'w') as f:
            json.dump(USERS, f, indent=4)
    except Exception as e:
        print(f"[WARNING]   Could not save local user fallback: {e}")

def _load_registered_users():
    """On startup, read users from MongoDB or local JSON fallback."""
    # 1. Load from local JSON first (ensures speed and baseline stability)
    if _USERS_FILE.exists():
        try:
            with open(_USERS_FILE, 'r') as f:
                data = json.load(f)
                USERS.update(data)
                print(f"[SYSTEM] Loaded {len(data)} user(s) from local cache")
        except Exception as e:
            print(f"[WARNING] Could not load local users: {e}")

    # 2. Try loading from MongoDB
    if not _USE_MONGO:
        return
    try:
        _db = get_database()
        cursor = _db.users.find({}, {'_id': 0})
        count = 0
        for doc in cursor:
            username = doc.get('username')
            if username:
                USERS[username] = {
                    'password':     doc.get('password', ''),
                    'role':         doc.get('role', 'User'),
                    'name':         doc.get('name', username),
                    'access_level': doc.get('access_level', 'viewer'),
                }
                count += 1
        
        # Ensure index on users
        _db.users.create_index([('username', 1)], unique=True)
        
        if count:
            print(f"[SYSTEM] Sync'd {count} user(s) from MongoDB Atlas")
    except Exception as e:
        print(f"[WARNING]   Could not sync users from MongoDB: {e}")

def _sync_users_to_mongo():
    """Upsert all in-memory USERS into MongoDB so they persist across restarts."""
    if not _USE_MONGO:
        return
    try:
        synced = 0
        for username, user in USERS.items():
            users_col().update_one(
                {'username': username},
                {'$setOnInsert': {**user, 'username': username}},
                upsert=True,
            )
            synced += 1
        if synced:
            print(f"[SYSTEM] Synced {synced} user(s) to MongoDB Atlas")
    except Exception as e:
        print(f"[WARNING]   Could not sync users to MongoDB: {e}")

_load_registered_users()
_sync_users_to_mongo()

active_sessions: dict = {}

# -- Class labels --------------------------------------------------------------
CLASS_LABELS = {0: 'Normal', 1: 'Arrhythmia', 2: 'Myocardial Infarction', 3: 'Other Abnormality'}
MIN_SIGNAL_LENGTH = 100


# -- Helpers -------------------------------------------------------------------
def _compute_fft_band_power(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Return integrated power in a frequency band using the FFT periodogram."""
    n = len(signal)
    if n < 2:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.sum(power[mask]))


def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    """Apply a simple box-car smoothing window."""
    if len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def generate_gradcam_simulation(ecg_signal: np.ndarray) -> np.ndarray:
    """
    Simulate Grad-CAM importance scores aligned with signal peaks.
    Smoothed with a Gaussian-like window for realistic visualisation.
    """
    importance = np.abs(ecg_signal)
    # Gaussian smoothing via convolution
    window = min(51, len(importance) // 4 * 2 + 1)
    if window > 2:
        x = np.linspace(-3, 3, window)
        kernel = np.exp(-0.5 * x ** 2)
        kernel /= kernel.sum()
        importance = np.convolve(importance, kernel, mode='same')
    rng = importance.max() - importance.min()
    if rng > 0:
        importance = (importance - importance.min()) / rng
    return importance


def _compute_risk_level(predicted_class: int, confidence: float) -> str:
    """Determine risk level from predicted class and model confidence."""
    if predicted_class == 0:
        return 'Low'
    if predicted_class == 2:           # MI -> always High
        return 'High'
    # Arrhythmia or Other
    if confidence >= 0.75:
        return 'High'
    elif confidence >= 0.50:
        return 'Medium'
    return 'Low'


def generate_recommendations(predicted_class: int, features: dict) -> list[str]:
    """Generate clinical recommendations based on prediction and features."""
    hr = features.get('Heart Rate', 70)

    if predicted_class == 0:  # Normal
        recs = ["Continue regular monitoring", "Maintain healthy lifestyle"]
        if hr > 100:
            recs.append("Heart rate slightly elevated - consider stress evaluation")
        elif hr < 50:
            recs.append("Heart rate is low - consult a physician if symptomatic")
        recs.append("Schedule routine checkup in 6 months")
        return recs

    if predicted_class == 1:  # Arrhythmia
        return [
            "Consult a cardiologist promptly",
            "Consider Holter monitoring for 24 48 hours",
            "Avoid strenuous physical activity until assessed",
            "Monitor heart rate and rhythm regularly",
            "Review current medications with your physician",
        ]

    if predicted_class == 2:  # MI
        return [
            "[WARNING]  Seek emergency medical attention immediately",
            "Call emergency services (112/911) if chest pain is present",
            "Do NOT drive yourself - wait for medical assistance",
            "Avoid all physical exertion",
            "Administer aspirin if not contraindicated",
        ]

    # Other
    return [
        "Consult a cardiologist for further evaluation",
        "Perform additional diagnostic tests (echo, stress test)",
        "Consider hospitalisation if symptoms worsen",
    ]


def _extract_key_features(ecg_signal: np.ndarray) -> dict:
    """
    Derive key display features directly from the ECG signal using FFT
    (no random noise injected).
    """
    n = len(ecg_signal)
    estimated_fs = n / 10.0       # assume 10-second strip

    # -- R-peak detection (local maxima above adaptive threshold) ----------
    # A true R-peak must:
    #   1. Exceed mean + 0.6*std (adaptive threshold)
    #   2. Be a local maximum in a  min_sep window
    mean = float(np.mean(ecg_signal))
    std  = float(np.std(ecg_signal))
    min_sep = max(1, int(estimated_fs * 0.33))   # min 330 ms between peaks (  180 bpm max)
    threshold = mean + 0.6 * std

    candidate_idx = np.where(ecg_signal > threshold)[0]
    r_peaks = []
    for idx in candidate_idx:
        lo = max(0, idx - min_sep)
        hi = min(n, idx + min_sep + 1)
        if ecg_signal[idx] == np.max(ecg_signal[lo:hi]):
            if not r_peaks or (idx - r_peaks[-1]) >= min_sep:
                r_peaks.append(idx)

    num_peaks = len(r_peaks)
    if num_peaks >= 2:
        # Mean RR interval across all detected peaks
        rr_intervals_samples = np.diff(r_peaks)
        mean_rr_samples = float(np.mean(rr_intervals_samples))
        mean_rr_seconds = mean_rr_samples / estimated_fs
        heart_rate = float(np.clip(60.0 / mean_rr_seconds, 30.0, 200.0))
        mean_rr = mean_rr_seconds
    elif num_peaks == 1:
        # Only one peak found - estimate from signal length
        heart_rate = float(np.clip(num_peaks / 10.0 * 60.0, 30.0, 200.0))
        mean_rr = 60.0 / heart_rate
    else:
        heart_rate = 70.0    # safe fallback
        mean_rr = 60.0 / 70.0

    # HRV time-domain
    diffs = np.diff(ecg_signal)
    sdnn = float(np.std(ecg_signal) * 1000)            # proxy in ms
    rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000) if len(diffs) > 0 else 0.0
    pnn50 = (float(np.sum(np.abs(diffs) > 0.05) / len(diffs) * 100)
             if len(diffs) > 0 else 0.0)

    # Frequency-domain (real FFT)
    lf_power = _compute_fft_band_power(ecg_signal, estimated_fs, 0.04, 0.15)
    hf_power = _compute_fft_band_power(ecg_signal, estimated_fs, 0.15, 0.40)
    lf_hf_ratio = (lf_power / hf_power) if hf_power > 0 else 1.0

    return {
        'Heart Rate': heart_rate,
        'Mean RR Interval': mean_rr,
        'SDNN (HRV)': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'LF Power': lf_power,
        'HF Power': hf_power,
        'LF/HF Ratio': float(lf_hf_ratio),
        'Signal Energy': float(np.sum(ecg_signal ** 2)),
        'QRS Width': float(0.08 + std * 0.01),
    }


# -- Routes --------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if 'username' in session:
        return redirect(url_for('index'))   # already logged in

    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        user = USERS.get(username)
        # -- Fallback: check MongoDB directly if not in memory (sync for Render workers) --
        if not user and _USE_MONGO:
            try:
                doc = users_col().find_one({'username': username})
                if doc:
                    user = {
                        'password':     doc.get('password', ''),
                        'role':         doc.get('role', 'User'),
                        'name':         doc.get('name', username),
                        'access_level': doc.get('access_level', 'viewer'),
                    }
                    USERS[username] = user  # cache it
            except Exception as e:
                print(f"[WARNING]   Login DB lookup failed: {e}")

        if user and user['password'] == password:
            session['username']     = username
            session['role']         = user['role']
            session['name']         = user['name']
            session['access_level'] = user['access_level']
            if remember:
                app.permanent_session_lifetime = __import__('datetime').timedelta(days=7)
                session.permanent = True
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clear session and redirect to login."""
    name = session.get('name', 'User')
    session.clear()
    flash(f'Goodbye, {name}! You have been signed out.', 'success')
    return redirect(url_for('login'))


# Role -> display title and access level
_ROLE_MAP = {
    'administrator':  ('Administrator',  'admin'),
    'cardiologist':   ('Cardiologist',   'admin'),
    'medical_staff':  ('Medical Staff',  'admin'),
    'viewer':         ('Viewer',         'viewer'),
}

@app.route('/register', methods=['POST'])
def register():
    """Handle new user sign-up."""
    first     = request.form.get('first_name', '').strip()
    last      = request.form.get('last_name',  '').strip()
    role_raw  = request.form.get('role',       '').strip().lower()
    username  = request.form.get('username',   '').strip().lower()
    password  = request.form.get('password',   '')
    confirm   = request.form.get('confirm',    '')

    # Validate
    if not all([first, last, role_raw, username, password, confirm]):
        flash('All fields are required.', 'error')
        return redirect(url_for('login') + '?tab=signup')
    if password != confirm:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('login') + '?tab=signup')
    if len(password) < 6:
        flash('Password must be at least 6 characters.', 'error')
        return redirect(url_for('login') + '?tab=signup')
    if username in USERS:
        flash(f'Username "{username}" is already taken. Please choose another.', 'error')
        return redirect(url_for('login') + '?tab=signup')

    role_title, access_level = _ROLE_MAP.get(role_raw, ('Viewer', 'viewer'))
    full_name = f'{first} {last}'
    if role_title == 'Cardiologist':
        full_name = f'Dr. {last}'

    new_user = {
        'password':     password,
        'role':         role_title,
        'name':         full_name,
        'access_level': access_level,
    }
    USERS[username] = new_user
    _save_local_users()

    # Persist to MongoDB so the account survives restarts
    if _USE_MONGO:
        try:
            users_col().update_one(
                {'username': username},
                {'$set': {**new_user, 'username': username}},
                upsert=True,
            )
        except Exception as e:
            print(f'[WARNING]   Could not persist user to MongoDB: {e}')

    flash(
        f'Account successfully created. You can now sign in with your username: <strong>{username}</strong>.',
        'success'
    )
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """Render main dashboard."""
    return render_template('index_advanced.html')


@app.route('/api/me')
@login_required
def me():
    """Return current user info including access level."""
    return jsonify({
        'username':     session.get('username'),
        'role':         session.get('role'),
        'name':         session.get('name'),
        'access_level': session.get('access_level', 'viewer'),
    })


@app.route('/api/predict', methods=['POST'])
@login_required
@admin_required
def predict():
    """
    Advanced ECG prediction with explainability features.
    """
    try:
        data = request.get_json(silent=True) or {}

        if 'ecg_signal' not in data:
            return jsonify({'error': 'No ECG signal provided'}), 400

        ecg_signal = np.asarray(data['ecg_signal'], dtype=np.float64)

        if ecg_signal.ndim != 1:
            return jsonify({'error': 'ECG signal must be a 1-D array'}), 400
        if len(ecg_signal) < MIN_SIGNAL_LENGTH:
            return jsonify({
                'error': f'Signal too short - minimum {MIN_SIGNAL_LENGTH} samples required'
            }), 400

        # -- ML prediction --------------------------------------------------
        if ml_model is not None and not DEMO_MODE:
            ml_result = ml_model.predict(ecg_signal)
            predicted_class = ml_result['predicted_class']
            probabilities = ml_result['probabilities']
            confidence = ml_result['confidence']
            uncertainty = ml_result['uncertainty']
            feature_importance = ml_result['feature_importance']
        else:
            # Heuristic fallback
            std = float(np.std(ecg_signal))
            mean = float(np.mean(ecg_signal))
            if std > 0.5 and abs(mean) < 0.3:
                predicted_class, confidence = 0, 0.85
                probabilities = {'Normal': 0.85, 'Arrhythmia': 0.10,
                                 'Myocardial Infarction': 0.03, 'Other Abnormality': 0.02}
            else:
                predicted_class, confidence = 1, 0.70
                probabilities = {'Normal': 0.15, 'Arrhythmia': 0.70,
                                 'Myocardial Infarction': 0.10, 'Other Abnormality': 0.05}
            uncertainty = 1.0 - confidence
            feature_importance = [0.1] * 10

        # -- Derived features ------------------------------------------------
        key_features = _extract_key_features(ecg_signal)

        # -- XAI ------------------------------------------------------------
        gradcam = generate_gradcam_simulation(ecg_signal)

        feature_names = ['Heart Rate', 'HRV Metrics', 'QRS Morphology',
                         'Frequency Features', 'Signal Energy', 'Other']
        total_imp = sum(feature_importance[:6]) or 1.0
        shap_values = {
            name: float(feature_importance[i] / total_imp)
            for i, name in enumerate(feature_names)
        }

        risk_level = _compute_risk_level(predicted_class, confidence)
        recommendations = generate_recommendations(predicted_class, key_features)

        # -- Vital Fusion (optional) ----------------------------------------
        vitals_data = data.get('vitals', {})
        fusion_result = None
        if vitals_data:
            try:
                vital_reading = VitalReading(
                    spo2=vitals_data.get('spo2'),
                    systolic=vitals_data.get('systolic'),
                    diastolic=vitals_data.get('diastolic'),
                    temperature=vitals_data.get('temperature'),
                    heart_rate=key_features.get('Heart Rate'),
                )
                fusion_result = _vital_fusion.evaluate(
                    ecg_prediction=CLASS_LABELS.get(predicted_class, 'Unknown'),
                    ecg_confidence=float(confidence),
                    ecg_risk_level=risk_level,
                    vitals=vital_reading,
                )
                # Save vitals
                try:
                    save_vitals({
                        'patient_id': data.get('patient_id', 'PATIENT_001'),
                        **vital_reading.to_dict(),
                        'ucrs': fusion_result.ucrs,
                        'fused_risk_level': fusion_result.fused_risk_level,
                    })
                except Exception as ve:
                    print(f'[WARNING] Vitals save error: {ve}')
            except Exception as vf_err:
                print(f'[WARNING] Vital fusion error: {vf_err}')

        response = {
            'success': True,
            'prediction': CLASS_LABELS.get(predicted_class, 'Unknown'),
            'predicted_class': predicted_class,
            'probabilities': probabilities,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'features': key_features,
            'gradcam_heatmap': gradcam.tolist(),
            'shap_values': shap_values,
            'processed_signal': ecg_signal.tolist()[:1000],
            'demo_mode': DEMO_MODE,
            'ml_model_active': ml_model is not None and not DEMO_MODE,
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'fusion': fusion_result.to_dict() if fusion_result else None,
        }

        # -- Persist analysis --------------------------------------------
        analysis_patient_id = data.get('patient_id', 'PATIENT_001')
        try:
            doc = save_analysis({
                'patient_id':   analysis_patient_id,
                'prediction':   response['prediction'],
                'confidence':   float(confidence),
                'uncertainty':  float(uncertainty),
                'risk_level':   risk_level,
                'heart_rate':   key_features.get('Heart Rate'),
                'hrv_sdnn':     key_features.get('SDNN (HRV)'),
                'signal_length': len(ecg_signal),
                'probabilities': probabilities,
                'features':     key_features,
                'recommendations': recommendations,
            })
            response['analysis_id'] = str(doc.get('_id', doc.get('id', 'local')))
        except Exception as db_err:
            print(f'[WARNING] Analysis save error: {db_err}')

        # -- Fire alerts if thresholds breached ----------------------------
        try:
            alert_vitals = vitals_data if vitals_data else None
            alert_event = alert_engine.evaluate(
                patient_id   = data.get('patient_id', 'PATIENT_001'),
                ecg_prediction = response['prediction'],
                ecg_confidence = float(confidence),
                risk_level   = risk_level,
                ucrs         = fusion_result.ucrs if fusion_result else None,
                vitals       = alert_vitals,
            )
            if alert_event:
                response['alert_fired'] = alert_event.to_dict()
                try:
                    save_alert_log(alert_event.to_dict())
                except Exception as ale:
                    print(f'[WARNING] Alert log save error: {ale}')
        except Exception as ae:
            print(f'[WARNING] Alert engine error: {ae}')

        return jsonify(response)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/digitize-ecg', methods=['POST'])
@login_required
@admin_required
def digitize_ecg():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = file.filename.lower()
    try:
        file_bytes = file.read()
        from src.utils.digitizer import extract_signal_from_image, extract_signal_from_pdf
        
        if filename.endswith('.pdf'):
            signal_arr = extract_signal_from_pdf(file_bytes)
        elif filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            signal_arr = extract_signal_from_image(file_bytes)
        else:
            return jsonify({'error': 'Unsupported file format for digitization. Use PNG, JPEG, or PDF.'}), 400
            
        return jsonify({'success': True, 'ecg_signal': signal_arr.tolist()})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-report', methods=['POST'])
@login_required
@admin_required
def analyze_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    mime_type = file.content_type
    if mime_type not in ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg']:
        return jsonify({'error': 'Unsupported file format for report. Use PDF, PNG, or JPEG.'}), 400

    try:
        file_bytes = file.read()
        import google.generativeai as genai
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        
        if not api_key:
            return jsonify({'error': 'Gemini API Key is not configured on the server.'}), 500
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = "Analyze this medical report. Provide a clear summary of the findings in HTML-compatible markdown, identify any abnormal values, and suggest recommended measures or lifestyle changes. Emphasize that this is AI-generated and the user should consult a primary care physician."
        
        response = model.generate_content([
            {'mime_type': mime_type, 'data': file_bytes},
            prompt
        ])
        
        return jsonify({'success': True, 'analysis': response.text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# -- WebSocket ------------------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'session_id': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    active_sessions.pop(request.sid, None)


@socketio.on('start_monitoring')
def handle_start_monitoring(data):
    patient_id = data.get('patient_id', 'PATIENT_001')
    active_sessions[request.sid] = {
        'patient_id': patient_id,
        'start_time': datetime.now().isoformat(),
        'status': 'active',
    }
    emit('monitoring_started', {
        'patient_id': patient_id,
        'status': 'active',
        'message': 'Real-time monitoring initiated',
    })


@socketio.on('stream_ecg')
def handle_ecg_stream(data):
    """Process an ECG chunk and return a live prediction."""
    try:
        ecg_chunk = np.asarray(data['ecg_data'], dtype=np.float64)
        patient_id = data.get('patient_id', 'PATIENT_001')

        # Use the ML model when available
        if ml_model is not None and not DEMO_MODE and len(ecg_chunk) >= MIN_SIGNAL_LENGTH:
            result_ml = ml_model.predict(ecg_chunk)
            prediction = result_ml['prediction']
            confidence = float(result_ml['confidence'])
        else:
            chunk_std = float(np.std(ecg_chunk))
            chunk_mean = float(np.mean(ecg_chunk))
            if chunk_std > 0.5 and abs(chunk_mean) < 0.3:
                prediction, confidence = 'Normal', 0.85
            else:
                prediction, confidence = 'Abnormal', 0.70

        n = len(ecg_chunk)
        estimated_fs = max(n / 10.0, 1.0)
        threshold = np.mean(ecg_chunk) + 0.5 * np.std(ecg_chunk)
        peaks = int(np.sum(ecg_chunk > threshold))
        if peaks > 1:
            rr_seconds = (n / peaks) / estimated_fs
            heart_rate = float(np.clip(60.0 / rr_seconds, 30.0, 220.0))
        else:
            heart_rate = 70.0

        alert_level = 'normal' if prediction == 'Normal' else 'warning'

        result = {
            'patient_id': patient_id,
            'prediction': prediction,
            'confidence': confidence,
            'heart_rate': heart_rate,
            'alert_level': alert_level,
            'timestamp': datetime.now().isoformat(),
            'ecg_chunk': ecg_chunk.tolist(),
        }

        emit('prediction_update', result)

        if confidence < 0.4 or alert_level == 'critical':
            emit('critical_alert', {
                'patient_id': patient_id,
                'message': f'Critical condition detected for {patient_id}',
                'timestamp': datetime.now().isoformat(),
            })

    except Exception as exc:
        emit('error', {'message': str(exc)})



# -- Report generation ----------------------------------------------------------
@app.route('/api/generate-report', methods=['POST'])
@login_required
def generate_report():
    """Generate a comprehensive PDF cardiac analysis report."""
    try:
        data = request.get_json(silent=True) or {}
        patient_id = data.get('patient_id', 'PATIENT_001')
        prediction = data.get('prediction', 'Unknown')
        confidence = float(data.get('confidence', 0))
        uncertainty = float(data.get('uncertainty', 0))
        probabilities: dict = data.get('probabilities', {})
        features: dict = data.get('features', {})
        recommendations: list = data.get('recommendations', [])
        risk_level = data.get('risk_level', 'Unknown')

        buf = io.BytesIO()
        p = canvas.Canvas(buf, pagesize=letter)
        width, height = letter

        # Header
        p.setFillColorRGB(0.05, 0.15, 0.45)
        p.rect(0, height - 1.6 * inch, width, 1.6 * inch, fill=True, stroke=False)
        p.setFillColorRGB(0, 0.83, 1)
        p.setFont("Helvetica-Bold", 22)
        p.drawString(1 * inch, height - 0.85 * inch, "CARDIAC ANALYSIS REPORT")
        p.setFillColorRGB(1, 1, 1)
        p.setFont("Helvetica", 10)
        p.drawString(1 * inch, height - 1.15 * inch,
                     f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        p.drawString(1 * inch, height - 1.35 * inch,
                     "Advanced Cardiac Abnormality Detection System v2.0")

        def section_title(title: str, y: float) -> None:
            p.setFillColorRGB(0.05, 0.15, 0.45)
            p.setFont("Helvetica-Bold", 13)
            p.drawString(1 * inch, y, title)
            p.setStrokeColorRGB(0, 0.83, 1)
            p.setLineWidth(1)
            p.line(1 * inch, y - 4, width - 1 * inch, y - 4)
            p.setFillColorRGB(0, 0, 0)
            p.setFont("Helvetica", 11)

        # Patient info
        y = height - 2.1 * inch
        section_title("Patient Information", y)
        y -= 0.35 * inch
        p.drawString(1.2 * inch, y, f"Patient ID : {patient_id}")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Report Date: {datetime.now().strftime('%B %d, %Y')}")

        # Diagnosis
        y -= 0.45 * inch
        section_title("Diagnosis", y)
        y -= 0.35 * inch
        p.drawString(1.2 * inch, y, f"Condition   : {prediction}")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Risk Level  : {risk_level}")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Confidence  : {confidence * 100:.1f}%")
        y -= 0.22 * inch
        p.drawString(1.2 * inch, y, f"Uncertainty : {uncertainty * 100:.1f}%")

        # Probabilities
        if probabilities:
            y -= 0.45 * inch
            section_title("Class Probabilities", y)
            y -= 0.32 * inch
            p.setFont("Helvetica", 10)
            for cls_name, prob in probabilities.items():
                p.drawString(1.2 * inch, y, f"{cls_name:<30} {prob * 100:.1f}%")
                y -= 0.20 * inch

        # Features table
        if features:
            y -= 0.35 * inch
            section_title("Cardiometric Features", y)
            y -= 0.32 * inch
            p.setFont("Helvetica", 9)
            for feature, value in list(features.items())[:10]:
                p.drawString(1.2 * inch, y, f"{feature:<30} {value:.3f}")
                y -= 0.18 * inch

        # Recommendations
        if recommendations and y > 1.8 * inch:
            y -= 0.35 * inch
            section_title("Clinical Recommendations", y)
            y -= 0.32 * inch
            p.setFont("Helvetica", 10)
            for i, rec in enumerate(recommendations[:6], 1):
                if y < 1.5 * inch:
                    break
                p.drawString(1.2 * inch, y, f"{i}. {rec}")
                y -= 0.24 * inch

        # Footer
        p.setStrokeColorRGB(0.8, 0.8, 0.8)
        p.line(1 * inch, 1.2 * inch, width - 1 * inch, 1.2 * inch)
        p.setFont("Helvetica-Oblique", 8)
        p.setFillColorRGB(0.4, 0.4, 0.4)
        p.drawString(
            1 * inch, 1 * inch,
            "[WARNING] This report is AI-generated and must be reviewed by a qualified medical professional."
        )

        p.showPage()
        p.save()
        buf.seek(0)

        fname = f'cardiac_report_{patient_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        return send_file(buf, as_attachment=True, download_name=fname, mimetype='application/pdf')

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# -- Database API --------------------------------------------------------------------
@app.route('/api/patients', methods=['GET', 'POST'])
@login_required
def manage_patients():
    """List all patients or create a new one."""
    if request.method == 'POST' and session.get('access_level') != 'admin':
        return jsonify({'success': False, 'error': 'Access denied - admin role required'}), 403
    if request.method == 'GET':
        return jsonify({'success': True, 'patients': get_all_patients()})
    data = request.get_json(silent=True) or {}
    missing = [f for f in ('patient_id', 'name') if not data.get(f)]
    if missing:
        return jsonify({'success': False, 'error': f'Missing: {missing}'}), 400
    try:
        patient = create_patient({
            'patient_id':      data['patient_id'],
            'name':            data['name'],
            'age':             data.get('age'),
            'gender':          data.get('gender'),
            'contact_info':    data.get('contact_info'),
            'medical_history': data.get('medical_history', ''),
        })
        return jsonify({'success': True, 'patient': patient}), 201
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/patients/<patient_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def manage_patient(patient_id: str):
    """Get, update, or delete a specific patient."""
    if request.method in ('PUT', 'DELETE') and session.get('access_level') != 'admin':
        return jsonify({'success': False, 'error': 'Access denied - admin role required'}), 403
    patient = get_patient(patient_id)
    if not patient:
        return jsonify({'success': False, 'error': 'Patient not found'}), 404
    if request.method == 'GET':
        return jsonify({'success': True, 'patient': patient})
    if request.method == 'PUT':
        data = request.get_json(silent=True) or {}
        updates = {k: v for k, v in data.items() if k in ('name','age','gender','contact_info','medical_history')}
        return jsonify({'success': True, 'patient': update_patient(patient_id, updates)})
    delete_patient(patient_id)
    return jsonify({'success': True, 'message': f'Patient {patient_id} deleted'})


@app.route('/api/analyses/<patient_id>', methods=['GET'])
@login_required
def get_patient_analyses_route(patient_id: str):
    """Return all analyses for a given patient, newest first."""
    try:
        analyses = get_patient_analyses(patient_id)
        return jsonify({'success': True, 'analyses': analyses})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/patient-ecg-history/<patient_id>', methods=['GET'])
@login_required
def get_patient_ecg_history(patient_id: str):
    """
    Return a patient's full ECG history with summary stats and risk trend.
    Used by the Patient ECG Tracker panel.
    """
    try:
        patient = get_patient(patient_id)
        analyses = get_patient_analyses(patient_id, limit=100)

        if not patient:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404

        # Build risk trend list (newest first → reverse for chart)
        risk_trend = []
        for a in reversed(analyses):
            ts = a.get('analysis_timestamp', a.get('timestamp', ''))
            risk_trend.append({
                'date':       ts[:10] if ts else 'N/A',
                'timestamp':  ts,
                'risk':       a.get('risk_level', 'Low'),
                'confidence': round(float(a.get('confidence', 0)) * 100, 1),
                'prediction': a.get('prediction', 'Unknown'),
                'heart_rate': a.get('heart_rate') or a.get('features', {}).get('Heart Rate'),
            })

        # Summary
        total = len(analyses)
        last  = analyses[0] if analyses else {}
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        for a in analyses:
            rl = a.get('risk_level', 'Low')
            risk_counts[rl] = risk_counts.get(rl, 0) + 1

        return jsonify({
            'success':   True,
            'patient_id': patient_id,
            'patient':   patient,
            'analyses':  analyses,
            'summary': {
                'total_sessions':  total,
                'last_diagnosis':  last.get('prediction', 'N/A') if last else 'N/A',
                'last_risk':       last.get('risk_level', 'N/A') if last else 'N/A',
                'last_confidence': round(float(last.get('confidence', 0)) * 100, 1) if last else 0,
                'risk_counts':     risk_counts,
                'risk_trend':      risk_trend,
            },
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats_route():
    """Overall system statistics."""
    try:
        stats = get_stats()
        stats['ml_model_active'] = ml_model is not None and not DEMO_MODE
        stats['demo_mode']       = DEMO_MODE
        stats['success']         = True
        stats['db_backend']      = 'MongoDB' if _USE_MONGO else 'SQLite'
        return jsonify(stats)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health-check endpoint."""
    import sys
    has_eventlet = 'eventlet' in sys.modules
    return jsonify({
        'status': 'healthy',
        'model_loaded': ml_model is not None and not DEMO_MODE,
        'demo_mode': DEMO_MODE,
        'db_backend': 'MongoDB' if _USE_MONGO else 'SQLite',
        '_USE_MONGO': _USE_MONGO,
        'eventlet_loaded': has_eventlet,
        'pid': os.getpid(),
        'features': ['Real-time monitoring', 'Explainable AI',
                     'Advanced analytics', 'Report generation'],
    })


# -- AI Chat -------------------------------------------------------------------
@app.route('/api/chat', methods=['POST'])
@app.route('/api/ai/chat', methods=['POST'])
@login_required
def ai_chat():
    """
    Real AI chat endpoint powered by Google Gemini.
    Falls back gracefully to the static knowledge base if the API key is
    not configured or the call fails.
    """
    CARDIAC_SYSTEM_PROMPT = """You are CardiacAI, an expert medical AI assistant specialised in:
- Electrocardiography (ECG/EKG) analysis and interpretation
- Cardiac arrhythmias, myocardial infarction, heart failure, and other conditions
- Cardiometric features: Heart Rate, HRV (SDNN, RMSSD, pNN50), QRS duration, QT interval
- Explainable AI methods: SHAP, Grad-CAM in cardiac diagnosis
- Clinical guidelines and when to seek emergency care
- This CardiacMonitor Pro dashboard: ECG upload, real-time monitoring, PDF report generation, patient history

Rules:
- Always be concise, accurate, and compassionate.
- ALWAYS recommend consulting a qualified cardiologist for personal medical decisions.
- Never give a definitive personal medical diagnosis.
- Use markdown-like formatting (bold with <b> tags) in your answer.
- Keep answers under 120 words unless the question requires more depth.
- If the question is completely unrelated to cardiology or this dashboard, politely redirect."""

    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Try Google Gemini
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=CARDIAC_SYSTEM_PROMPT,
            )
            response = model.generate_content(user_message)
            return jsonify({'reply': response.text, 'source': 'gemini'})
        except ImportError:
            pass  # google-generativeai not installed → use fallback
        except Exception as exc:
            print(f'[AI Chat] Gemini error: {exc}')
            # fall through to static KB

    # Static knowledge-base fallback
    q = user_message.lower()
    KB = [
        (['arrhythmia', 'irregular', 'rhythm', 'afib', 'atrial'], '<b>Arrhythmia</b> is an irregular heart rhythm detected via R-peak interval analysis. Symptoms include palpitations, dizziness, or dyspnoea. Always consult a cardiologist — some arrhythmias require urgent treatment.'),
        (['myocardial infarction', 'heart attack', 'mi', 'st elevation', 'infarct'], '<b>Myocardial Infarction (MI)</b> is caused by blocked coronary arteries. Detected via ST-segment elevation in ECG. 🚨 If suspected, call emergency services immediately — time is critical.'),
        (['bradycardia', 'slow heart', 'low bpm'], '<b>Bradycardia</b> is a resting heart rate below 60 bpm. It may be normal in athletes but can cause fainting if severe. An ECG analysis and Holter monitoring help confirm it.'),
        (['tachycardia', 'fast heart', 'high bpm', 'fast rate'], '<b>Tachycardia</b> is a resting heart rate above 100 bpm. Causes range from stress and dehydration to serious arrhythmias. ECG analysis helps distinguish the type.'),
        (['spo2', 'oxygen', 'blood oxygen'], 'Normal <b>SpO₂</b> is 95–100%. Below 90% is critical hypoxaemia requiring immediate attention. The Vital Signs Monitor on this dashboard lets you log SpO₂ alongside your ECG analysis.'),
        (['blood pressure', 'systolic', 'diastolic', 'hypertension', 'bp'], 'Normal BP is <120/80 mmHg. Above 180/120 is a <b>hypertensive crisis</b> — seek emergency care. This dashboard tracks BP as part of the Unified Cardiac Risk Score (UCRS).'),
        (['hrv', 'heart rate variability', 'sdnn', 'rmssd'], '<b>HRV (Heart Rate Variability)</b> measures the variation between beats. <b>SDNN</b> reflects overall autonomic function; <b>RMSSD</b> reflects parasympathetic activity. Reduced HRV is linked to higher cardiac risk.'),
        (['ucrs', 'unified', 'fusion', 'vital fusion'], 'The <b>Unified Cardiac Risk Score (UCRS)</b> is unique to this platform — it fuses ECG predictions with SpO₂, BP, temperature, and heart rate using weighted clinical thresholds into a single 0–100 risk score.'),
        (['iot', 'device', 'wearable', 'simulate', 'sensor'], 'The <b>IoT Device Hub</b> allows real wearable devices to stream ECG and vital data to this dashboard in real-time via the WebSocket API. Use "Simulate Device" to see a live demo.'),
        (['alert', 'emergency', 'notification', 'sms', 'email'], 'The <b>Emergency Alert System</b> monitors UCRS and vital thresholds. When exceeded, it automatically sends email/SMS/WhatsApp alerts to configured emergency contacts.'),
        (['pdf', 'report', 'download'], 'After running an ECG analysis, click <b>"PDF Report"</b> to download a clinically formatted PDF including diagnosis, confidence, probabilities, features, and recommendations.'),
        (['shap', 'explain', 'feature importance', 'xai'], '<b>SHAP values</b> show which ECG features most influenced the AI prediction — making the model transparent and clinically interpretable rather than a "black box".'),
        (['gradcam', 'heatmap', 'grad-cam'], '<b>Grad-CAM</b> highlights which regions of the ECG signal the model focused on when predicting. Brighter = higher influence on the prediction.'),
        (['normal', 'healthy', 'no disease'], 'A <b>Normal Sinus Rhythm</b> result means no detectable cardiac abnormalities at analysis time. Continue routine monitoring every 6–12 months for preventive care.'),
        (['hello', 'hi', 'help', 'what can you do', 'hey'], 'Hello! 👋 I\'m <b>CardiacAI</b>, your real-time cardiac assistant. Ask me about:<br>• ECG conditions (arrhythmia, MI, HRV)<br>• Vital signs interpretation<br>• UCRS, IoT sync, alerts<br>• SHAP/Grad-CAM explanations<br>• How to use this dashboard'),
    ]
    for keys, answer in KB:
        if any(k in q for k in keys):
            return jsonify({'reply': answer, 'source': 'kb'})

    fallback = "🤔 I'm not certain about that query. I'm specialised in cardiac topics — try asking about <b>arrhythmia</b>, <b>blood pressure</b>, <b>UCRS</b>, <b>HRV</b>, or <b>ECG interpretation</b>. For personal medical advice, always consult your cardiologist."
    return jsonify({'reply': fallback, 'source': 'kb'})


@app.route('/api/debug-mongo', methods=['GET'])
def debug_mongo():
    """Debug endpoint to test MongoDB connectivity step by step."""
    import sys
    results = {
        'pid': os.getpid(),
        '_USE_MONGO': _USE_MONGO,
        'MONGO_URI_set': bool(os.environ.get('MONGO_URI')),
        'eventlet_in_modules': 'eventlet' in sys.modules,
        'mongo_init_error': _mongo_init_error,
        'steps': [],
    }
    
    if not _USE_MONGO:
        results['steps'].append('SKIPPED: _USE_MONGO is False, MongoDB is disabled')
        return jsonify(results)
    
    try:
        from app.mongodb_database import get_client, get_database, users_col
        results['steps'].append('OK: imports succeeded')
    except Exception as e:
        results['steps'].append(f'FAIL: import error: {e}')
        return jsonify(results)
    
    try:
        client = get_client()
        results['steps'].append(f'OK: got MongoClient')
    except Exception as e:
        results['steps'].append(f'FAIL: get_client error: {e}')
        return jsonify(results)
    
    try:
        db = get_database()
        results['steps'].append(f'OK: got database "{db.name}"')
    except Exception as e:
        results['steps'].append(f'FAIL: get_database error: {e}')
        return jsonify(results)
    
    try:
        col = users_col()
        results['steps'].append(f'OK: got users collection')
    except Exception as e:
        results['steps'].append(f'FAIL: users_col error: {e}')
        return jsonify(results)
    
    try:
        count = col.count_documents({})
        results['steps'].append(f'OK: count_documents returned {count}')
        results['user_count'] = count
    except Exception as e:
        results['steps'].append(f'FAIL: count_documents error: {e}')
        return jsonify(results)
    
    try:
        docs = list(col.find({}, {'_id': 0, 'password': 0}).limit(10))
        results['steps'].append(f'OK: found {len(docs)} users')
        results['users'] = [d.get('username', '?') for d in docs]
    except Exception as e:
        results['steps'].append(f'FAIL: find error: {e}')
    
    return jsonify(results)


# =============================================================================
# REAL-WORLD INTEGRATION — NEW ENDPOINTS
# =============================================================================

# -- Multi-Vital Fusion -------------------------------------------------------

@app.route('/api/vitals/submit', methods=['POST'])
@login_required
def submit_vitals():
    """Submit a standalone vitals reading for a patient and compute UCRS."""
    try:
        data = request.get_json(silent=True) or {}
        patient_id = data.get('patient_id', 'PATIENT_001')

        vital_reading = VitalReading(
            spo2=data.get('spo2'),
            systolic=data.get('systolic'),
            diastolic=data.get('diastolic'),
            temperature=data.get('temperature'),
            heart_rate=data.get('heart_rate'),
        )

        # Evaluate fusion with no ECG (Normal baseline)
        fusion = _vital_fusion.evaluate(
            ecg_prediction='Normal',
            ecg_confidence=1.0,
            ecg_risk_level='Low',
            vitals=vital_reading,
        )

        result = {
            'success': True,
            'patient_id': patient_id,
            'vitals': vital_reading.to_dict(),
            'fusion': fusion.to_dict(),
        }

        # Persist vitals
        try:
            save_vitals({'patient_id': patient_id, **vital_reading.to_dict(),
                         'ucrs': fusion.ucrs, 'fused_risk_level': fusion.fused_risk_level})
        except Exception as ve:
            print(f'[WARNING] Vitals save error: {ve}')

        # Evaluate alerts
        alert_event = alert_engine.evaluate(
            patient_id=patient_id,
            ecg_prediction='Normal',
            ecg_confidence=1.0,
            risk_level=fusion.fused_risk_level,
            ucrs=fusion.ucrs,
            vitals=data,
        )
        if alert_event:
            result['alert_fired'] = alert_event.to_dict()

        # Broadcast via WebSocket so dashboard updates live
        socketio.emit('vitals_update', result, broadcast=True)

        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/vitals/<patient_id>', methods=['GET'])
@login_required
def get_vitals(patient_id: str):
    """Return last 50 vitals readings for a patient."""
    try:
        readings = get_patient_vitals(patient_id)
        return jsonify({'success': True, 'patient_id': patient_id, 'vitals': readings})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# -- IoT Device Hub -----------------------------------------------------------

@app.route('/api/iot/devices', methods=['GET'])
@login_required
def list_iot_devices():
    """List all registered IoT devices."""
    try:
        devices = device_manager.list_devices()
        return jsonify({'success': True, 'devices': devices})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/iot/register', methods=['POST'])
@login_required
@admin_required
def register_iot_device():
    """Register a new IoT device."""
    try:
        data = request.get_json(silent=True) or {}
        device_id   = data.get('device_id', '').strip()
        device_type = data.get('device_type', 'custom')
        patient_id  = data.get('patient_id', 'PATIENT_001')
        label       = data.get('label', '')
        if not device_id:
            return jsonify({'success': False, 'error': 'device_id is required'}), 400
        dev = device_manager.register(device_id, device_type, patient_id, label)
        try:
            save_iot_device(dev.to_dict())
        except Exception:
            pass
        socketio.emit('device_registered', dev.to_dict(), broadcast=True)
        return jsonify({'success': True, 'device': dev.to_dict()}), 201
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/iot/devices/<device_id>', methods=['DELETE'])
@login_required
@admin_required
def deregister_iot_device(device_id: str):
    """Deregister (remove) an IoT device."""
    try:
        ok = device_manager.deregister(device_id)
        try:
            delete_iot_device(device_id)
        except Exception:
            pass
        socketio.emit('device_removed', {'device_id': device_id}, broadcast=True)
        return jsonify({'success': ok})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/iot/stream', methods=['POST'])
@login_required
def iot_stream():
    """
    IoT device data ingestion endpoint.
    Accepts a data packet from any registered device, runs lightweight
    analysis, and broadcasts the result to all dashboard WebSocket clients.
    """
    try:
        data = request.get_json(silent=True) or {}
        packet = DevicePacket.from_dict(data)

        # Update device state
        dev = device_manager.handle_packet(packet)
        if dev is None:
            # Auto-register unknown devices
            dev = device_manager.register(packet.device_id, 'custom', packet.patient_id)

        # Quick ECG heuristic on incoming chunk
        ecg = np.asarray(packet.ecg_chunk, dtype=np.float64) if packet.ecg_chunk else np.array([])
        if len(ecg) >= MIN_SIGNAL_LENGTH:
            key_feats  = _extract_key_features(ecg)
            std = float(np.std(ecg))
            mean_ = float(np.mean(ecg))
            if std > 0.5 and abs(mean_) < 0.3:
                ecg_pred, ecg_conf = 'Normal', 0.85
            else:
                ecg_pred, ecg_conf = 'Arrhythmia', 0.68
        else:
            ecg_pred, ecg_conf = 'Unknown', 0.5
            key_feats = {}

        # Vital fusion
        vital_reading = VitalReading(
            spo2=packet.spo2,
            systolic=packet.systolic,
            diastolic=packet.diastolic,
            temperature=packet.temperature,
            heart_rate=packet.heart_rate or key_feats.get('Heart Rate'),
        )
        fusion = _vital_fusion.evaluate(
            ecg_prediction=ecg_pred,
            ecg_confidence=ecg_conf,
            ecg_risk_level='Low' if ecg_pred == 'Normal' else 'Medium',
            vitals=vital_reading,
        )

        broadcast_payload = {
            'device_id':   packet.device_id,
            'patient_id':  packet.patient_id,
            'ecg_prediction': ecg_pred,
            'ecg_confidence': ecg_conf,
            'heart_rate':  packet.heart_rate or key_feats.get('Heart Rate', 70),
            'spo2':        packet.spo2,
            'systolic':    packet.systolic,
            'diastolic':   packet.diastolic,
            'temperature': packet.temperature,
            'fusion':      fusion.to_dict(),
            'ecg_chunk':   packet.ecg_chunk[:100] if packet.ecg_chunk else [],
            'timestamp':   packet.timestamp,
            'packet_count': dev.packet_count,
        }

        socketio.emit('iot_data_update', broadcast_payload, broadcast=True)

        # Alert check
        alert_engine.evaluate(
            patient_id=packet.patient_id,
            ecg_prediction=ecg_pred,
            ecg_confidence=ecg_conf,
            risk_level=fusion.fused_risk_level,
            ucrs=fusion.ucrs,
            vitals={'spo2': packet.spo2, 'systolic': packet.systolic,
                    'heart_rate': packet.heart_rate},
        )

        return jsonify({'success': True, 'fusion': fusion.to_dict()})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/iot/simulate', methods=['POST'])
@login_required
def iot_simulate():
    """
    Start/stop a built-in device simulator that POSTs packets
    to /api/iot/stream every second for demo purposes.
    Runs in a daemon thread — stops when 'stop' action is sent.
    """
    import threading, time
    data = request.get_json(silent=True) or {}
    action     = data.get('action', 'start')
    patient_id = data.get('patient_id', 'PATIENT_001')
    device_id  = f'SIM-{patient_id}'

    if action == 'stop':
        _simulators.pop(patient_id, None)
        device_manager.mark_offline(device_id)
        socketio.emit('simulator_stopped', {'device_id': device_id}, broadcast=True)
        return jsonify({'success': True, 'message': 'Simulator stopped'})

    if patient_id in _simulators:
        return jsonify({'success': True, 'message': 'Simulator already running', 'device_id': device_id})

    sim = DeviceSimulator(device_id=device_id, patient_id=patient_id)
    _simulators[patient_id] = sim
    device_manager.register(device_id, 'mobile_app', patient_id, f'Simulator ({patient_id})')
    socketio.emit('simulator_started', {'device_id': device_id, 'patient_id': patient_id}, broadcast=True)

    def _run():
        while patient_id in _simulators:
            try:
                packet = sim.next_packet(samples=250)
                # POST internally via SocketIO broadcast instead of HTTP to avoid re-login
                ecg = np.asarray(packet.ecg_chunk, dtype=np.float64)
                if len(ecg) >= MIN_SIGNAL_LENGTH:
                    std_ = float(np.std(ecg))
                    mean_ = float(np.mean(ecg))
                    ecg_pred = 'Normal' if (std_ > 0.5 and abs(mean_) < 0.3) else 'Arrhythmia'
                    ecg_conf = 0.85 if ecg_pred == 'Normal' else 0.68
                else:
                    ecg_pred, ecg_conf = 'Unknown', 0.5
                vital_r = VitalReading(spo2=packet.spo2, systolic=packet.systolic,
                                       diastolic=packet.diastolic, temperature=packet.temperature,
                                       heart_rate=packet.heart_rate)
                fusion = _vital_fusion.evaluate(ecg_pred, ecg_conf,
                                                'Low' if ecg_pred == 'Normal' else 'Medium', vital_r)
                dev_ref = device_manager.handle_packet(packet)
                payload = {
                    'device_id': device_id, 'patient_id': patient_id,
                    'ecg_prediction': ecg_pred, 'ecg_confidence': ecg_conf,
                    'heart_rate': packet.heart_rate, 'spo2': packet.spo2,
                    'systolic': packet.systolic, 'diastolic': packet.diastolic,
                    'temperature': packet.temperature, 'fusion': fusion.to_dict(),
                    'ecg_chunk': packet.ecg_chunk[:100], 'timestamp': packet.timestamp,
                    'packet_count': dev_ref.packet_count if dev_ref else 0,
                }
                socketio.emit('iot_data_update', payload, broadcast=True)
                alert_engine.evaluate(patient_id=patient_id, ecg_prediction=ecg_pred,
                                      ecg_confidence=ecg_conf, risk_level=fusion.fused_risk_level,
                                      ucrs=fusion.ucrs,
                                      vitals={'spo2': packet.spo2, 'systolic': packet.systolic,
                                              'heart_rate': packet.heart_rate})
            except Exception as e:
                print(f'[SIM] Error: {e}')
            time.sleep(1)
        device_manager.mark_offline(device_id)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'success': True, 'device_id': device_id, 'patient_id': patient_id})


# -- Emergency Alert System ---------------------------------------------------

@app.route('/api/alerts/config', methods=['GET', 'POST'])
@login_required
@admin_required
def alerts_config():
    """Get or update alert thresholds, contacts, and channel toggles."""
    if request.method == 'GET':
        return jsonify({'success': True, 'config': alert_engine.get_config()})
    data = request.get_json(silent=True) or {}
    alert_engine.update_config(
        thresholds=data.get('thresholds'),
        contacts=data.get('contacts'),
        channels=data.get('channels'),
    )
    return jsonify({'success': True, 'config': alert_engine.get_config()})


@app.route('/api/alerts/logs', methods=['GET'])
@login_required
def alerts_logs():
    """Return recent alert history (in-memory + MongoDB)."""
    try:
        limit = int(request.args.get('limit', 50))
        # Merge: in-memory first, then MongoDB for persistence
        mem_logs = alert_engine.get_recent_alerts(limit)
        if _USE_MONGO and len(mem_logs) < limit:
            try:
                from app.mongodb_database import get_recent_alert_logs
                db_logs = get_recent_alert_logs(limit)
                # De-duplicate by alert_id
                seen = {l.get('alert_id') for l in mem_logs}
                for log in db_logs:
                    if log.get('alert_id') not in seen:
                        mem_logs.append(log)
                        seen.add(log.get('alert_id'))
            except Exception:
                pass
        mem_logs = mem_logs[:limit]
        return jsonify({'success': True, 'alerts': mem_logs, 'total': len(mem_logs)})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/alerts/test', methods=['POST'])
@login_required
@admin_required
def alerts_test():
    """Fire a test alert to verify channel configuration."""
    import uuid
    from src.alerts.alert_engine import AlertEvent
    test_event = AlertEvent(
        alert_id='TEST-' + str(uuid.uuid4())[:6].upper(),
        patient_id='TEST-PATIENT',
        severity='Warning',
        trigger='Manual test alert fired by admin',
        message='🧪 TEST ALERT\nThis is a test alert from CardiacMonitor Pro.\nIf you received this, your alert channels are working correctly.',
        channels_fired=[],
        channels_failed=[],
    )
    import threading
    t = threading.Thread(target=alert_engine._dispatch, args=(test_event,), daemon=True)
    t.start()
    return jsonify({'success': True, 'test_event': test_event.to_dict()})




# -- Clinical Analytics -------------------------------------------------------

@app.route('/api/analytics/trends', methods=['GET'])
@login_required
def analytics_trends():
    """
    Generate synthetic 24-hour trend data for clinical analytics.
    In a real system, this would aggregate historical DB records.
    """
    try:
        patient_id = request.args.get('patient_id', 'PATIENT_001')
        
        # Hours labels
        hours = [f"{h:02d}:00" for h in range(24)]
        
        # Synthetic Vitals Heatmap Data (SpO2, HR, SysBP, DiaBP)
        heatmap_vitals = [
            # SpO2 (mostly 94-99)
            [98, 97, 98, 99, 98, 97, 96, 95, 94, 95, 96, 97, 98, 98, 99, 98, 97, 96, 95, 96, 97, 98, 98, 99],
            # HR (fluctuates during day)
            [62, 60, 58, 65, 70, 75, 80, 85, 90, 88, 85, 82, 80, 85, 95, 105, 100, 90, 85, 80, 75, 70, 68, 65],
            # SysBP (daytime peak)
            [115, 110, 112, 118, 122, 125, 130, 135, 140, 138, 135, 132, 130, 135, 140, 145, 142, 135, 130, 128, 125, 122, 120, 118],
            # DiaBP
            [75, 72, 74, 78, 80, 82, 85, 88, 90, 88, 85, 82, 80, 82, 85, 90, 88, 85, 82, 80, 78, 76, 75, 74]
        ]
        
        # Radar Data: Arrhythmia events by type over last 24h
        radar_categories = ['PVC', 'PAC', 'Bradycardia', 'Tachycardia', 'AFom', 'Baseline Offset']
        radar_values = [5, 12, 2, 8, 3, 1] if "001" in patient_id else [2, 6, 0, 4, 1, 0]

        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'hours': hours,
            'heatmap_vitals': heatmap_vitals,
            'vitals_labels': ['SpO2 (%)', 'HR (bpm)', 'Sys BP (mmHg)', 'Dia BP (mmHg)'],
            'radar_categories': radar_categories,
            'radar_values': radar_values
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# -- Collaborative Notes -----------------------------------------------------

@app.route('/api/notes/submit', methods=['POST'])
@login_required
@admin_required
def submit_note():
    """Save a clinical note for a specific patient/reading and broadcast it."""
    try:
        data = request.get_json(silent=True) or {}
        patient_id = data.get('patient_id', 'PATIENT_001')
        note_text  = data.get('note', '').strip()
        reading_id = data.get('reading_id', 'latest')
        
        if not note_text:
            return jsonify({'success': False, 'error': 'Note text is required'}), 400
            
        note_data = {
            'note_id': str(__import__('uuid').uuid4())[:8].upper(),
            'patient_id': patient_id,
            'reading_id': reading_id,
            'author': session.get('name', 'Doctor'),
            'note': note_text,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Persist note
        try:
            if save_note:
                save_note(note_data)
        except Exception:
            pass
            
        # Broadcast to all clinicians
        socketio.emit('new_clinical_note', note_data, broadcast=True)
        
        return jsonify({'success': True, 'note': note_data})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# -- Entry point ---------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 60)
    print("[START] Starting Advanced Cardiac Monitoring System")
    print(f"Navigate to: http://localhost:{port}")
    print("=" * 60 + "\n")
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except OSError as e:
        if "address already in use" in str(e).lower() or getattr(e, 'winerror', 0) == 10048:
            alt_port = 5050
            print(f"[WARNING] Port {port} is busy, falling back to http://localhost:{alt_port}")
            socketio.run(app, debug=False, host='0.0.0.0', port=alt_port, allow_unsafe_werkzeug=True)
        else:
            raise
