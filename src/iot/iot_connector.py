"""
IoT Device Connector — Wearable / IoT Device Sync
=================================================
Manages registration and real-time data streaming from IoT
wearable devices (smartwatches, ECG patches, bedside monitors).

Architecture:
  - Devices POST data to  /api/iot/stream
  - IoTDeviceManager validates packet, runs lightweight analysis
  - Result is broadcast via SocketIO to all dashboard clients
  - A built-in DeviceSimulator generates realistic demo data
"""

from __future__ import annotations

import math
import random
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Device types
# ---------------------------------------------------------------------------
DEVICE_TYPES = {
    'smartwatch':    'Smartwatch (BLE)',
    'ecg_patch':     'ECG Patch Sensor',
    'bedside_monitor': 'Bedside Monitor',
    'mobile_app':    'Mobile ECG App',
    'custom':        'Custom IoT Device',
}

STATUS_ONLINE  = 'online'
STATUS_OFFLINE = 'offline'
STATUS_STREAMING = 'streaming'


@dataclass
class IoTDevice:
    device_id:   str
    device_type: str
    patient_id:  str
    label:       str = ''
    status:      str = STATUS_OFFLINE
    last_seen:   Optional[str] = None
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    packet_count: int = 0

    def to_dict(self) -> dict:
        return {
            'device_id':    self.device_id,
            'device_type':  self.device_type,
            'patient_id':   self.patient_id,
            'label':        self.label or self.device_id,
            'status':       self.status,
            'last_seen':    self.last_seen,
            'registered_at': self.registered_at,
            'packet_count': self.packet_count,
        }


@dataclass
class DevicePacket:
    device_id:   str
    patient_id:  str
    ecg_chunk:   list          # list of floats — ECG samples
    heart_rate:  Optional[float] = None
    spo2:        Optional[float] = None
    systolic:    Optional[float] = None
    diastolic:   Optional[float] = None
    temperature: Optional[float] = None
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_dict(cls, data: dict) -> 'DevicePacket':
        return cls(
            device_id   = data.get('device_id', 'UNKNOWN'),
            patient_id  = data.get('patient_id', 'PATIENT_001'),
            ecg_chunk   = data.get('ecg_chunk', []),
            heart_rate  = data.get('heart_rate'),
            spo2        = data.get('spo2'),
            systolic    = data.get('systolic'),
            diastolic   = data.get('diastolic'),
            temperature = data.get('temperature'),
            timestamp   = data.get('timestamp', datetime.now().isoformat()),
        )

    def to_dict(self) -> dict:
        return {
            'device_id':   self.device_id,
            'patient_id':  self.patient_id,
            'ecg_chunk':   self.ecg_chunk[:200],   # truncate for wire
            'heart_rate':  self.heart_rate,
            'spo2':        self.spo2,
            'systolic':    self.systolic,
            'diastolic':   self.diastolic,
            'temperature': self.temperature,
            'timestamp':   self.timestamp,
        }


class IoTDeviceManager:
    """Thread-safe in-memory device registry."""

    def __init__(self) -> None:
        self._devices: dict[str, IoTDevice] = {}
        self._lock = threading.Lock()

    # -- Registration -----------------------------------------------------
    def register(self, device_id: str, device_type: str, patient_id: str,
                 label: str = '') -> IoTDevice:
        with self._lock:
            dev = IoTDevice(
                device_id=device_id,
                device_type=device_type,
                patient_id=patient_id,
                label=label or device_id,
                status=STATUS_ONLINE,
                last_seen=datetime.now().isoformat(),
            )
            self._devices[device_id] = dev
            return dev

    def deregister(self, device_id: str) -> bool:
        with self._lock:
            return bool(self._devices.pop(device_id, None))

    def get_device(self, device_id: str) -> Optional[IoTDevice]:
        return self._devices.get(device_id)

    def list_devices(self) -> list[dict]:
        with self._lock:
            return [d.to_dict() for d in self._devices.values()]

    # -- Packet handling --------------------------------------------------
    def handle_packet(self, packet: DevicePacket) -> Optional[IoTDevice]:
        """Update device status from an incoming data packet."""
        with self._lock:
            dev = self._devices.get(packet.device_id)
            if dev:
                dev.status       = STATUS_STREAMING
                dev.last_seen    = packet.timestamp
                dev.packet_count += 1
            return dev

    def mark_offline(self, device_id: str) -> None:
        with self._lock:
            dev = self._devices.get(device_id)
            if dev:
                dev.status = STATUS_OFFLINE


# ---------------------------------------------------------------------------
# Realistic ECG + Vital Signal Simulator
# ---------------------------------------------------------------------------

class DeviceSimulator:
    """
    Generates realistic synthetic ECG chunks and accompanying vitals
    for demo/testing purposes.  Mimics a Holter patch sending 250-sample
    packets at 250 Hz (1-second windows).
    """

    PATIENT_PROFILES = {
        'PATIENT_001': {'condition': 'normal',    'hr_base': 72,  'spo2': 98.0, 'sbp': 120, 'dbp': 80, 'temp': 36.7},
        'PATIENT_002': {'condition': 'arrhythmia','hr_base': 95,  'spo2': 96.0, 'sbp': 135, 'dbp': 88, 'temp': 37.1},
        'PATIENT_003': {'condition': 'mi',        'hr_base': 110, 'spo2': 92.0, 'sbp': 170, 'dbp': 100,'temp': 37.8},
        'PATIENT_004': {'condition': 'normal',    'hr_base': 65,  'spo2': 99.0, 'sbp': 115, 'dbp': 75, 'temp': 36.5},
    }

    def __init__(self, device_id: str, patient_id: str, fs: int = 250) -> None:
        self.device_id  = device_id
        self.patient_id = patient_id
        self.fs         = fs
        self._t         = 0.0           # running time (seconds)
        profile = self.PATIENT_PROFILES.get(patient_id, self.PATIENT_PROFILES['PATIENT_001'])
        self._condition  = profile['condition']
        self._hr_base    = profile['hr_base']
        self._spo2_base  = profile['spo2']
        self._sbp_base   = profile['sbp']
        self._dbp_base   = profile['dbp']
        self._temp_base  = profile['temp']

    def _ecg_sample(self, t: float) -> float:
        """Synthesise one ECG sample at time t (seconds)."""
        hr    = self._hr_base
        period = 60.0 / hr
        phase = (t % period) / period   # 0..1 within one heartbeat

        # PQRST morphology approximation
        p  =  0.15 * math.exp(-((phase - 0.15) ** 2) / 0.002)
        q  = -0.05 * math.exp(-((phase - 0.35) ** 2) / 0.0003)
        r  =  1.00 * math.exp(-((phase - 0.40) ** 2) / 0.0002)
        s  = -0.15 * math.exp(-((phase - 0.45) ** 2) / 0.0003)
        t_ =  0.30 * math.exp(-((phase - 0.65) ** 2) / 0.004)

        ecg = p + q + r + s + t_

        # Condition-specific artefacts
        if self._condition == 'arrhythmia' and random.random() < 0.05:
            ecg += random.uniform(-0.4, 0.4)   # ectopic beat
        elif self._condition == 'mi':
            if 0.38 < phase < 0.55:
                ecg += 0.25                    # ST elevation

        ecg += random.gauss(0, 0.015)          # baseline noise
        return round(ecg, 4)

    def next_packet(self, samples: int = 250) -> DevicePacket:
        """Generate next packet of `samples` ECG samples + current vitals."""
        chunk = []
        dt    = 1.0 / self.fs
        for _ in range(samples):
            chunk.append(self._ecg_sample(self._t))
            self._t += dt

        # Vitals with small random drift
        spo2  = round(max(85.0, self._spo2_base + random.gauss(0, 0.3)), 1)
        sbp   = round(self._sbp_base + random.gauss(0, 2), 0)
        dbp   = round(self._dbp_base + random.gauss(0, 1.5), 0)
        temp  = round(self._temp_base + random.gauss(0, 0.05), 1)
        hr    = round(self._hr_base + random.gauss(0, 2), 0)

        return DevicePacket(
            device_id=self.device_id,
            patient_id=self.patient_id,
            ecg_chunk=chunk,
            heart_rate=hr,
            spo2=spo2,
            systolic=sbp,
            diastolic=dbp,
            temperature=temp,
        )


# Module-level singleton device manager
device_manager = IoTDeviceManager()
