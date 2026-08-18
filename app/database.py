"""
Database module for Cardiac Detection System.
SQLite database with SQLAlchemy ORM for patient and analysis management.
Compatible with SQLAlchemy 2.x.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import json

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, ForeignKey, event
)
from sqlalchemy.orm import (
    DeclarativeBase, sessionmaker, relationship, Session
)

# -- Database path ------------------------------------------------------------
DB_DIR = Path(__file__).parent.parent / 'data'
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / 'cardiac_data.db'

# Enable WAL mode for better concurrent read performance
engine = create_engine(
    f'sqlite:///{DB_PATH}',
    echo=False,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# -- ORM Base -----------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# -- Models -------------------------------------------------------------------
class Patient(Base):
    """Patient information table."""

    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    contact_info = Column(String(200))          # NEW: email / phone
    medical_history = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    analyses = relationship(
        "ECGAnalysis",
        back_populates="patient",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'contact_info': self.contact_info,
            'medical_history': self.medical_history,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'total_analyses': len(self.analyses),
        }


class ECGAnalysis(Base):
    """ECG analysis results table."""

    __tablename__ = 'ecg_analyses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        String(50), ForeignKey('patients.patient_id', ondelete='CASCADE'),
        nullable=False, index=True
    )
    prediction = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    heart_rate = Column(Float)
    hrv_sdnn = Column(Float)
    signal_length = Column(Integer)             # NEW: number of ECG samples
    probabilities_json = Column(Text)
    features_json = Column(Text)
    recommendations_json = Column(Text)
    analysis_timestamp = Column(DateTime, default=datetime.now, index=True)
    ecg_data_path = Column(String(500))

    patient = relationship("Patient", back_populates="analyses")
    reports = relationship(
        "GeneratedReport",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'risk_level': self.risk_level,
            'heart_rate': self.heart_rate,
            'hrv_sdnn': self.hrv_sdnn,
            'signal_length': self.signal_length,
            'probabilities': json.loads(self.probabilities_json) if self.probabilities_json else {},
            'features': json.loads(self.features_json) if self.features_json else {},
            'recommendations': json.loads(self.recommendations_json) if self.recommendations_json else [],
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'ecg_data_path': self.ecg_data_path,
        }


class GeneratedReport(Base):
    """Generated PDF reports table."""

    __tablename__ = 'generated_reports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(
        Integer, ForeignKey('ecg_analyses.id', ondelete='CASCADE'),
        nullable=False, index=True
    )
    report_path = Column(String(500), nullable=False)
    generated_at = Column(DateTime, default=datetime.now)

    analysis = relationship("ECGAnalysis", back_populates="reports")

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'report_path': self.report_path,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
        }


class VitalRecord(Base):
    """Auxiliary vitals records table."""
    __tablename__ = 'vitals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), nullable=False, index=True)
    spo2 = Column(Float)
    systolic = Column(Float)
    diastolic = Column(Float)
    temperature = Column(Float)
    heart_rate = Column(Float)
    ucrs = Column(Float)
    fused_risk_level = Column(String(50))
    timestamp = Column(DateTime, default=datetime.now, index=True)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            '_id': str(self.id),
            'patient_id': self.patient_id,
            'spo2': self.spo2,
            'systolic': self.systolic,
            'diastolic': self.diastolic,
            'temperature': self.temperature,
            'heart_rate': self.heart_rate,
            'ucrs': self.ucrs,
            'fused_risk_level': self.fused_risk_level,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class IoTDeviceRecord(Base):
    """Registered IoT devices table."""
    __tablename__ = 'iot_devices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(100), unique=True, nullable=False, index=True)
    device_type = Column(String(50), nullable=False)
    patient_id = Column(String(50), nullable=False)
    label = Column(String(200))
    status = Column(String(50), default='offline')
    last_seen = Column(String(100))
    registered_at = Column(DateTime, default=datetime.now)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            '_id': str(self.id),
            'device_id': self.device_id,
            'device_type': self.device_type,
            'patient_id': self.patient_id,
            'label': self.label or self.device_id,
            'status': self.status,
            'last_seen': self.last_seen,
            'registered_at': self.registered_at.isoformat() if self.registered_at else None,
        }


class AlertLogRecord(Base):
    """Alert logs table."""
    __tablename__ = 'alert_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(100), index=True)
    patient_id = Column(String(50), index=True)
    severity = Column(String(50))
    trigger = Column(Text)
    message = Column(Text)
    channels_fired_json = Column(Text)
    channels_failed_json = Column(Text)
    timestamp = Column(DateTime, default=datetime.now, index=True)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            '_id': str(self.id),
            'alert_id': self.alert_id,
            'patient_id': self.patient_id,
            'severity': self.severity,
            'trigger': self.trigger,
            'message': self.message,
            'channels_fired': json.loads(self.channels_fired_json) if self.channels_fired_json else [],
            'channels_failed': json.loads(self.channels_failed_json) if self.channels_failed_json else [],
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class ClinicalNoteRecord(Base):
    """Clinical collaborative notes table."""
    __tablename__ = 'clinical_notes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    note_id = Column(String(100), index=True)
    patient_id = Column(String(50), index=True)
    reading_id = Column(String(100))
    author = Column(String(100))
    note = Column(Text)
    timestamp = Column(DateTime, default=datetime.now, index=True)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            '_id': str(self.id),
            'note_id': self.note_id,
            'patient_id': self.patient_id,
            'reading_id': self.reading_id,
            'author': self.author,
            'note': self.note,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


# -- Session helpers -----------------------------------------------------------
@contextmanager
def get_db() -> Session:
    """Context-manager database session (preferred)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Return a raw session (caller is responsible for close/rollback)."""
    return SessionLocal()


# -- Sample Patients -----------------------------------------------------------
_SAMPLE_PATIENTS = [
    {
        'patient_id': 'PATIENT_001',
        'name': 'John Smith',
        'age': 58,
        'gender': 'Male',
        'contact_info': 'john.smith@example.com',
        'medical_history': 'Hypertension, Type 2 Diabetes',
    },
    {
        'patient_id': 'PATIENT_002',
        'name': 'Sarah Johnson',
        'age': 45,
        'gender': 'Female',
        'contact_info': 'sarah.j@example.com',
        'medical_history': 'Previous MI, Stent placement 2022',
    },
    {
        'patient_id': 'PATIENT_003',
        'name': 'Mike Chen',
        'age': 67,
        'gender': 'Male',
        'contact_info': 'mchen@example.com',
        'medical_history': 'Atrial Fibrillation, Warfarin therapy',
    },
    {
        'patient_id': 'PATIENT_004',
        'name': 'Emma Wilson',
        'age': 34,
        'gender': 'Female',
        'contact_info': 'ewilson@example.com',
        'medical_history': 'No significant history',
    },
    {
        'patient_id': 'P001',
        'name': 'John Doe',
        'age': 45,
        'gender': 'Male',
        'contact_info': 'john.doe@example.com',
        'medical_history': 'Hypertension, controlled with medication',
    },
]


def _auto_migrate(conn_engine) -> None:
    """Safely add any missing columns to existing SQLite tables."""
    with conn_engine.connect() as conn:
        # Check patients table
        try:
            result = conn.exec_driver_sql("PRAGMA table_info(patients)").fetchall()
            existing_cols = {row[1] for row in result}
            if existing_cols:
                if 'contact_info' not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN contact_info VARCHAR(200)")
                if 'medical_history' not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN medical_history TEXT")
                if 'updated_at' not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE patients ADD COLUMN updated_at DATETIME")
            conn.commit()
        except Exception as e:
            print(f"[DB-MIGRATE] patients migration notice: {e}")

        # Check ecg_analyses table
        try:
            result = conn.exec_driver_sql("PRAGMA table_info(ecg_analyses)").fetchall()
            existing_cols = {row[1] for row in result}
            if existing_cols:
                if 'signal_length' not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE ecg_analyses ADD COLUMN signal_length INTEGER")
                if 'ecg_data_path' not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE ecg_analyses ADD COLUMN ecg_data_path VARCHAR(500)")
                if 'uncertainty' not in existing_cols:
                    conn.exec_driver_sql("ALTER TABLE ecg_analyses ADD COLUMN uncertainty FLOAT DEFAULT 0.0")
            conn.commit()
        except Exception as e:
            print(f"[DB-MIGRATE] analyses migration notice: {e}")


def init_database() -> None:
    """Create all tables and seed sample patients if the table is empty."""
    Base.metadata.create_all(engine)
    _auto_migrate(engine)
    print(f"[OK] Database initialized at {DB_PATH}")

    with get_db() as session:
        existing_ids = {
            row[0] for row in session.query(Patient.patient_id).all()
        }
        new_patients = [
            Patient(**p)
            for p in _SAMPLE_PATIENTS
            if p['patient_id'] not in existing_ids
        ]
        if new_patients:
            session.add_all(new_patients)
            print(f"[OK] Seeded {len(new_patients)} sample patient(s)")


# -- Patient CRUD --------------------------------------------------------------
def get_all_patients() -> list[dict]:
    with get_db() as session:
        patients = session.query(Patient).order_by(Patient.name.asc()).all()
        return [p.to_dict() for p in patients]


def get_patient(patient_id: str) -> dict | None:
    with get_db() as session:
        p = session.query(Patient).filter_by(patient_id=patient_id).first()
        return p.to_dict() if p else None


def create_patient(data: dict) -> dict:
    with get_db() as session:
        p = Patient(
            patient_id=data['patient_id'],
            name=data['name'],
            age=data.get('age'),
            gender=data.get('gender'),
            contact_info=data.get('contact_info'),
            medical_history=data.get('medical_history', ''),
        )
        session.add(p)
        session.flush()
        return p.to_dict()


def update_patient(patient_id: str, data: dict) -> dict | None:
    with get_db() as session:
        p = session.query(Patient).filter_by(patient_id=patient_id).first()
        if not p:
            return None
        for key, value in data.items():
            if hasattr(p, key) and key not in ('id', 'patient_id', 'created_at'):
                setattr(p, key, value)
        p.updated_at = datetime.now()
        session.flush()
        return p.to_dict()


def delete_patient(patient_id: str) -> bool:
    with get_db() as session:
        p = session.query(Patient).filter_by(patient_id=patient_id).first()
        if p:
            session.delete(p)
            return True
        return False


# -- ECG Analysis CRUD ---------------------------------------------------------
def save_analysis(data: dict) -> dict:
    """Insert a new ECG analysis record and return it serialised."""
    with get_db() as session:
        # Ensure patient exists or create lightweight record
        patient_id = data.get('patient_id', 'PATIENT_001')
        existing = session.query(Patient).filter_by(patient_id=patient_id).first()
        if not existing:
            new_p = Patient(patient_id=patient_id, name=f"Patient {patient_id}")
            session.add(new_p)
            session.flush()

        analysis = ECGAnalysis(
            patient_id=patient_id,
            prediction=data.get('prediction', 'Normal'),
            confidence=float(data.get('confidence', 0.0)),
            uncertainty=float(data.get('uncertainty', 0.0)),
            risk_level=data.get('risk_level', 'Low'),
            heart_rate=data.get('heart_rate'),
            hrv_sdnn=data.get('hrv_sdnn'),
            signal_length=data.get('signal_length'),
            probabilities_json=json.dumps(data.get('probabilities', {})),
            features_json=json.dumps(data.get('features', {})),
            recommendations_json=json.dumps(data.get('recommendations', [])),
            ecg_data_path=data.get('ecg_data_path'),
            analysis_timestamp=datetime.now(),
        )
        session.add(analysis)
        session.flush()
        doc = analysis.to_dict()
        doc['_id'] = str(analysis.id)
        return doc


def get_patient_analyses(patient_id: str, limit: int = 50) -> list[dict]:
    with get_db() as session:
        analyses = (
            session.query(ECGAnalysis)
            .filter_by(patient_id=patient_id)
            .order_by(ECGAnalysis.analysis_timestamp.desc())
            .limit(limit)
            .all()
        )
        results = []
        for a in analyses:
            d = a.to_dict()
            d['_id'] = str(a.id)
            results.append(d)
        return results


# -- Stats ---------------------------------------------------------------------
def get_stats() -> dict:
    with get_db() as session:
        total_patients = session.query(Patient).count()
        total_analyses = session.query(ECGAnalysis).count()
        low_count = session.query(ECGAnalysis).filter_by(risk_level='Low').count()
        med_count = session.query(ECGAnalysis).filter_by(risk_level='Medium').count()
        high_count = session.query(ECGAnalysis).filter_by(risk_level='High').count()
        return {
            'total_patients': total_patients,
            'total_analyses': total_analyses,
            'risk_breakdown': {
                'Low': low_count,
                'Medium': med_count,
                'High': high_count,
            },
        }


# -- Vitals CRUD ---------------------------------------------------------------
def save_vitals(data: dict) -> dict:
    with get_db() as session:
        v = VitalRecord(
            patient_id=data.get('patient_id', 'PATIENT_001'),
            spo2=data.get('spo2'),
            systolic=data.get('systolic'),
            diastolic=data.get('diastolic'),
            temperature=data.get('temperature'),
            heart_rate=data.get('heart_rate'),
            ucrs=data.get('ucrs'),
            fused_risk_level=data.get('fused_risk_level'),
            timestamp=datetime.now(),
        )
        session.add(v)
        session.flush()
        return v.to_dict()


def get_patient_vitals(patient_id: str, limit: int = 50) -> list[dict]:
    with get_db() as session:
        records = (
            session.query(VitalRecord)
            .filter_by(patient_id=patient_id)
            .order_by(VitalRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in records]


# -- IoT Device CRUD -----------------------------------------------------------
def save_iot_device(data: dict) -> dict:
    device_id = data.get('device_id', 'unknown')
    with get_db() as session:
        dev = session.query(IoTDeviceRecord).filter_by(device_id=device_id).first()
        if not dev:
            dev = IoTDeviceRecord(
                device_id=device_id,
                device_type=data.get('device_type', 'smartwatch'),
                patient_id=data.get('patient_id', 'PATIENT_001'),
                label=data.get('label', device_id),
                status=data.get('status', 'online'),
                last_seen=data.get('last_seen', datetime.now().isoformat()),
            )
            session.add(dev)
        else:
            dev.device_type = data.get('device_type', dev.device_type)
            dev.patient_id = data.get('patient_id', dev.patient_id)
            dev.label = data.get('label', dev.label)
            dev.status = data.get('status', dev.status)
            dev.last_seen = data.get('last_seen', datetime.now().isoformat())
        session.flush()
        return dev.to_dict()


def get_all_iot_devices() -> list[dict]:
    with get_db() as session:
        devices = session.query(IoTDeviceRecord).all()
        return [d.to_dict() for d in devices]


def delete_iot_device(device_id: str) -> bool:
    with get_db() as session:
        dev = session.query(IoTDeviceRecord).filter_by(device_id=device_id).first()
        if dev:
            session.delete(dev)
            return True
        return False


# -- Alert Logs CRUD -----------------------------------------------------------
def save_alert_log(data: dict) -> dict:
    with get_db() as session:
        log = AlertLogRecord(
            alert_id=data.get('alert_id', 'ALERT'),
            patient_id=data.get('patient_id', 'PATIENT_001'),
            severity=data.get('severity', 'Warning'),
            trigger=data.get('trigger', ''),
            message=data.get('message', ''),
            channels_fired_json=json.dumps(data.get('channels_fired', [])),
            channels_failed_json=json.dumps(data.get('channels_failed', [])),
            timestamp=datetime.now(),
        )
        session.add(log)
        session.flush()
        return log.to_dict()


def get_recent_alert_logs(limit: int = 100) -> list[dict]:
    with get_db() as session:
        logs = (
            session.query(AlertLogRecord)
            .order_by(AlertLogRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [l.to_dict() for l in logs]


# -- Clinical Notes CRUD -------------------------------------------------------
def save_note(data: dict) -> dict:
    with get_db() as session:
        note = ClinicalNoteRecord(
            note_id=data.get('note_id', 'NOTE'),
            patient_id=data.get('patient_id', 'PATIENT_001'),
            reading_id=data.get('reading_id', 'latest'),
            author=data.get('author', 'Doctor'),
            note=data.get('note', ''),
            timestamp=datetime.now(),
        )
        session.add(note)
        session.flush()
        return note.to_dict()


if __name__ == "__main__":
    init_database()

