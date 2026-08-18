"""
MongoDB database layer for Cardiac Monitor Pro.
Replaces the SQLite/SQLAlchemy layer with PyMongo.

Collections:
  - patients       -> Patient records
  - ecg_analyses   -> ECG analysis results
  - reports        -> Generated report metadata
"""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
import json
from datetime import datetime
from typing import Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from bson import ObjectId
from bson.errors import InvalidId

# -- Connection config ---------------------------------------------------------
MONGO_URI = os.environ.get('MONGO_URI')
DB_NAME   = os.environ.get('MONGO_DB', 'cardiac_monitor')

# Cache the resolved URI (string only - safe to share across threads/forks)
_resolved_uri: Optional[str] = None

import dns.resolver
import certifi

def _resolve_srv_to_standard(uri: str) -> str:
    """
    Workaround for Python 3.14 + Windows access violations in SRV processing.
    Converts mongodb+srv://... to a standard multi-shard mongodb://... string.
    Only applies on Windows - on Linux (Render), PyMongo handles SRV natively.
    """
    import platform
    if platform.system() != 'Windows':
        print("[DB] Non-Windows platform detected, using native SRV resolution")
        return uri
    if not uri.startswith('mongodb+srv://'):
        return uri
        
    try:
        # 1. Parse auth and cluster
        parts = uri.replace('mongodb+srv://', '').split('/', 1)
        auth_and_cluster = parts[0]
        params = parts[1] if len(parts) > 1 else ''
        
        if '@' not in auth_and_cluster:
            return uri # No auth, handle as-is
            
        auth, cluster = auth_and_cluster.rsplit('@', 1)
        
        # 2. Resolve SRV record
        print(f"[DB] Resolving SRV workaround for {cluster}...")
        try:
            answers = dns.resolver.resolve(f'_mongodb._tcp.{cluster}', 'SRV')
            print("[DB] SRV resolution successful.")
        except Exception as resolver_err:
            print(f"[ERROR] Resolver failed: {resolver_err}")
            return uri
            
        shards = [f"{rdata.target.to_text().rstrip('.')}:{rdata.port}" for rdata in answers]
        print(f"[DB] Found {len(shards)} shards.")
        
        if not shards:
            return uri
            
        shard_str = ",".join(shards)
        print(f"[DB] Shard string: {shard_str[:60]}...")
        
        # 3. Construct standard URI
        query_parts = []
        if '?' in params:
            base_params = params.split('?', 1)[1]
            query_parts = [p for p in base_params.split('&') if not p.startswith('appName=')]
            
        if 'ssl=true' not in query_parts and 'tls=true' not in query_parts:
            query_parts.append('ssl=true')
        if 'authSource=' not in "".join(query_parts):
            query_parts.append('authSource=admin')
            
        new_uri = f"mongodb://{auth}@{shard_str}/?{'&'.join(query_parts)}"
        print(f"[DB] Using expanded standard URI (bypass SRV crash)")
        return new_uri
        
    except Exception as e:
        print(f"[WARNING] SRV resolution workaround failed: {e}")
        return uri


def _get_resolved_uri() -> Optional[str]:
    """Resolve and cache the MongoDB URI (safe to share - it's just a string)."""
    global _resolved_uri
    if _resolved_uri is not None:
        return _resolved_uri
    
    uri = MONGO_URI
    if not uri:
        return None
        
    uri = _resolve_srv_to_standard(uri)
    
    # URL-encode special characters in the password (e.g. @ # etc.)
    if uri and '://' in uri and '@' in uri:
        try:
            prefix, rest = uri.split('://', 1)
            # Use rsplit to handle @ in the password correctly
            auth_part, host_part = rest.rsplit('@', 1)
            if ':' in auth_part:
                user, pwd = auth_part.split(':', 1)
                if '%' not in pwd:
                    pwd = urllib.parse.quote_plus(pwd)
                uri = f"{prefix}://{user}:{pwd}@{host_part}"
        except Exception:
            pass
    
    _resolved_uri = uri
    return _resolved_uri


def get_client() -> MongoClient:
    """Create a fresh MongoClient each time to avoid stale lock issues
    with gunicorn/eventlet/threading on Render."""
    uri = _get_resolved_uri()
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=20000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
        tlsCAFile=certifi.where(),
        connect=False,
    )


# --- Connection Health Heartbeat ---
_db_healthy: bool = True
_last_check: float = 0
_CHECK_INTERVAL = 30 # seconds

def is_db_reachable() -> bool:
    """Check if MongoDB is actually reachable within a strict 2s timeout."""
    global _db_healthy, _last_check
    import time
    now = time.time()
    if now - _last_check < _CHECK_INTERVAL:
        return _db_healthy
    
    _last_check = now
    try:
        client = get_client()
        # The 'hello' command is the lightweight way to check connectivity
        client.admin.command('hello', serverSelectionTimeoutMS=2000)
        _db_healthy = True
        return True
    except Exception as e:
        print(f"[DB] ⚠️ MongoDB Unreachable: {e}. Falling back to sample data for stability.")
        _db_healthy = False
        return False

def get_database() -> Database:
    return get_client()[DB_NAME]


# -- Collection accessors ------------------------------------------------------
def patients_col() -> Collection:
    return get_database()['patients']


def analyses_col() -> Collection:
    return get_database()['ecg_analyses']


def reports_col() -> Collection:
    return get_database()['reports']


def users_col() -> Collection:
    return get_database()['users']


def vitals_col() -> Collection:
    return get_database()['vitals']


def iot_devices_col() -> Collection:
    return get_database()['iot_devices']


def alert_logs_col() -> Collection:
    return get_database()['alert_logs']



# -- ID helpers ----------------------------------------------------------------
def _oid(val) -> Optional[ObjectId]:
    """Safely convert a string to ObjectId, returning None on failure."""
    try:
        return ObjectId(str(val)) if val else None
    except (InvalidId, Exception):
        return None


# -- Serialisers ---------------------------------------------------------------
def _patient_to_dict(doc: dict) -> dict:
    if not doc:
        return {}
    doc = dict(doc)
    doc['_id'] = str(doc['_id'])
    if isinstance(doc.get('created_at'), datetime):
        doc['created_at'] = doc['created_at'].isoformat()
    if isinstance(doc.get('updated_at'), datetime):
        doc['updated_at'] = doc['updated_at'].isoformat()
    # Count analyses inline
    doc['total_analyses'] = analyses_col().count_documents(
        {'patient_id': doc.get('patient_id', '')}
    )
    return doc


def _analysis_to_dict(doc: dict) -> dict:
    if not doc:
        return {}
    doc = dict(doc)
    doc['_id'] = str(doc['_id'])
    if isinstance(doc.get('analysis_timestamp'), datetime):
        doc['analysis_timestamp'] = doc['analysis_timestamp'].isoformat()
    return doc


# -- Initialise DB (indexes + seed data) --------------------------------------
_SAMPLE_PATIENTS = [
    {
        'patient_id': 'PATIENT_001',
        'name':       'John Smith',
        'age':        58,
        'gender':     'Male',
        'contact_info':    'john.smith@example.com',
        'medical_history': 'Hypertension, Type 2 Diabetes',
        'created_at':      datetime.now(),
        'updated_at':      datetime.now(),
    },
    {
        'patient_id': 'PATIENT_002',
        'name':       'Sarah Johnson',
        'age':        45,
        'gender':     'Female',
        'contact_info':    'sarah.j@example.com',
        'medical_history': 'Previous MI, Stent placement 2022',
        'created_at':      datetime.now(),
        'updated_at':      datetime.now(),
    },
    {
        'patient_id': 'PATIENT_003',
        'name':       'Mike Chen',
        'age':        67,
        'gender':     'Male',
        'contact_info':    'mchen@example.com',
        'medical_history': 'Atrial Fibrillation, Warfarin therapy',
        'created_at':      datetime.now(),
        'updated_at':      datetime.now(),
    },
    {
        'patient_id': 'PATIENT_004',
        'name':       'Emma Wilson',
        'age':        34,
        'gender':     'Female',
        'contact_info':    'ewilson@example.com',
        'medical_history': 'No significant history',
        'created_at':      datetime.now(),
        'updated_at':      datetime.now(),
    },
]


def init_database() -> None:
    """Create indexes and seed sample patients if collection is empty."""
    try:
        col = patients_col()

        # Indexes
        col.create_index([('patient_id', ASCENDING)], unique=True, name='idx_patient_id')
        analyses_col().create_index([('patient_id', ASCENDING)], name='idx_analysis_patient')
        analyses_col().create_index([('analysis_timestamp', DESCENDING)], name='idx_analysis_ts')

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
        print(f'[OK] MongoDB connected - database: "{DB_NAME}"')

        # -- Indexes for new collections -----------------------------------
        vitals_col().create_index([('patient_id', ASCENDING)], name='idx_vitals_patient')
        vitals_col().create_index([('timestamp', DESCENDING)], name='idx_vitals_ts')
        iot_devices_col().create_index([('device_id', ASCENDING)], unique=True, name='idx_iot_device_id')
        alert_logs_col().create_index([('patient_id', ASCENDING)], name='idx_alert_patient')
        alert_logs_col().create_index([('timestamp', DESCENDING)], name='idx_alert_ts')

    except Exception as exc:
        print(f'[ERROR] MongoDB init error: {exc}')
        raise


# -- Patient CRUD --------------------------------------------------------------
def get_all_patients() -> list[dict]:
    if not is_db_reachable():
        print("[DB] Returning sample patients (DB offline)")
        return [_patient_to_dict(p) for p in _SAMPLE_PATIENTS]
        
    try:
        docs = list(patients_col().find({}).sort('name', ASCENDING))
        return [_patient_to_dict(d) for d in docs]
    except Exception:
        return [_patient_to_dict(p) for p in _SAMPLE_PATIENTS]


def get_patient(patient_id: str) -> Optional[dict]:
    if not is_db_reachable():
        doc = next((p for p in _SAMPLE_PATIENTS if p['patient_id'] == patient_id), None)
        return _patient_to_dict(doc) if doc else None
        
    try:
        doc = patients_col().find_one({'patient_id': patient_id})
        return _patient_to_dict(doc) if doc else None
    except Exception:
        doc = next((p for p in _SAMPLE_PATIENTS if p['patient_id'] == patient_id), None)
        return _patient_to_dict(doc) if doc else None


def create_patient(data: dict) -> dict:
    data['created_at'] = datetime.now()
    data['updated_at'] = datetime.now()
    result = patients_col().insert_one(data)
    return _patient_to_dict(patients_col().find_one({'_id': result.inserted_id}))


def update_patient(patient_id: str, data: dict) -> Optional[dict]:
    data['updated_at'] = datetime.now()
    patients_col().update_one({'patient_id': patient_id}, {'$set': data})
    return get_patient(patient_id)


def delete_patient(patient_id: str) -> bool:
    result = patients_col().delete_one({'patient_id': patient_id})
    # Cascade-delete related analyses
    analyses_col().delete_many({'patient_id': patient_id})
    return result.deleted_count > 0


# -- ECG Analysis CRUD ---------------------------------------------------------
def save_analysis(data: dict) -> dict:
    """Insert a new ECG analysis document and return it serialised."""
    if not is_db_reachable():
        data['_id'] = "local_" + datetime.now().strftime("%Y%m%d%H%M%S")
        data['analysis_timestamp'] = datetime.now()
        return _analysis_to_dict(data)
        
    try:
        data['analysis_timestamp'] = datetime.now()
        result = analyses_col().insert_one(data)
        doc = analyses_col().find_one({'_id': result.inserted_id})
        return _analysis_to_dict(doc)
    except Exception:
        data['_id'] = "local_" + datetime.now().strftime("%Y%m%d%H%M%S")
        return _analysis_to_dict(data)


def get_patient_analyses(patient_id: str, limit: int = 50) -> list[dict]:
    if not is_db_reachable():
        return []
    try:
        docs = list(
            analyses_col()
            .find({'patient_id': patient_id})
            .sort('analysis_timestamp', DESCENDING)
            .limit(limit)
        )
        return [_analysis_to_dict(d) for d in docs]
    except Exception:
        return []


# -- Stats ---------------------------------------------------------------------
def get_stats() -> dict:
    if not is_db_reachable():
        return {'total_patients': len(_SAMPLE_PATIENTS), 'total_analyses': 0, 'risk_breakdown': {'Low': 0, 'Medium': 0, 'High':0}}
    try:
        total_patients = patients_col().count_documents({})
        total_analyses = analyses_col().count_documents({})
        risk_breakdown = {
            'Low':    analyses_col().count_documents({'risk_level': 'Low'}),
            'Medium': analyses_col().count_documents({'risk_level': 'Medium'}),
            'High':   analyses_col().count_documents({'risk_level': 'High'}),
        }
        return {
            'total_patients':  total_patients,
            'total_analyses':  total_analyses,
            'risk_breakdown':  risk_breakdown,
        }
    except Exception:
        return {'total_patients': len(_SAMPLE_PATIENTS), 'total_analyses': 0, 'risk_breakdown': {'Low': 0, 'Medium': 0, 'High':0}}


# -- Vitals CRUD ---------------------------------------------------------------
def save_vitals(data: dict) -> dict:
    """Insert a vitals reading; returns serialised doc."""
    if not is_db_reachable():
        data['timestamp'] = datetime.now()
        data['_id'] = "local_v_" + datetime.now().strftime("%Y%m%d%H%M%S")
        return data
        
    try:
        data['timestamp'] = datetime.now()
        result = vitals_col().insert_one(data)
        doc = vitals_col().find_one({'_id': result.inserted_id})
        if doc:
            doc = dict(doc)
            doc['_id'] = str(doc['_id'])
            if isinstance(doc.get('timestamp'), datetime):
                doc['timestamp'] = doc['timestamp'].isoformat()
            return doc
        return data
    except Exception:
        data['timestamp'] = datetime.now()
        data['_id'] = "local_v_" + datetime.now().strftime("%Y%m%d%H%M%S")
        return data


def get_patient_vitals(patient_id: str, limit: int = 50) -> list[dict]:
    if not is_db_reachable():
        return []
        
    try:
        docs = list(
            vitals_col()
            .find({'patient_id': patient_id})
            .sort('timestamp', DESCENDING)
            .limit(limit)
        )
        result = []
        for d in docs:
            d = dict(d)
            d['_id'] = str(d['_id'])
            if isinstance(d.get('timestamp'), datetime):
                d['timestamp'] = d['timestamp'].isoformat()
            result.append(d)
        return result
    except Exception:
        return []


# -- IoT Device CRUD -----------------------------------------------------------
def save_iot_device(data: dict) -> dict:
    """Upsert an IoT device record; returns serialised doc."""
    if not is_db_reachable():
        data['updated_at'] = datetime.now()
        data['_id'] = "local_dev_" + data.get('device_id', 'unknown')
        return data
        
    try:
        data['updated_at'] = datetime.now()
        iot_devices_col().update_one(
            {'device_id': data['device_id']},
            {'$set': data, '$setOnInsert': {'registered_at': datetime.now()}},
            upsert=True,
        )
        doc = dict(iot_devices_col().find_one({'device_id': data['device_id']}))
        doc['_id'] = str(doc['_id'])
        return doc
    except Exception:
        return data


def get_all_iot_devices() -> list[dict]:
    if not is_db_reachable():
        return []
        
    try:
        docs = list(iot_devices_col().find({}))
        result = []
        for d in docs:
            d = dict(d)
            d['_id'] = str(d['_id'])
            result.append(d)
        return result
    except Exception:
        return []


def delete_iot_device(device_id: str) -> bool:
    if not is_db_reachable():
        return True # Optimistic local delete
        
    try:
        r = iot_devices_col().delete_one({'device_id': device_id})
        return r.deleted_count > 0
    except Exception:
        return True


# -- Alert Logs CRUD -----------------------------------------------------------
def save_alert_log(data: dict) -> dict:
    """Insert an alert log entry."""
    if not is_db_reachable():
        data['timestamp'] = datetime.now()
        data['_id'] = "local_alert_" + datetime.now().strftime("%Y%m%d%H%M%S")
        return data
        
    try:
        data['timestamp'] = datetime.now()
        result = alert_logs_col().insert_one(data)
        doc = dict(alert_logs_col().find_one({'_id': result.inserted_id}))
        doc['_id'] = str(doc['_id'])
        if isinstance(doc.get('timestamp'), datetime):
            doc['timestamp'] = doc['timestamp'].isoformat()
        return doc
    except Exception:
        return data


def get_recent_alert_logs(limit: int = 100) -> list[dict]:
    if not is_db_reachable():
        return []
        
    try:
        docs = list(
            alert_logs_col()
            .find({})
            .sort('timestamp', DESCENDING)
            .limit(limit)
        )
        result = []
        for d in docs:
            d = dict(d)
            d['_id'] = str(d['_id'])
            if isinstance(d.get('timestamp'), datetime):
                d['timestamp'] = d['timestamp'].isoformat()
            result.append(d)
        return result
    except Exception:
        return []
