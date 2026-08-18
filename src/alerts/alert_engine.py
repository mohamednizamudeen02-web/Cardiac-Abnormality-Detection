"""
Alert Engine — Automated Emergency Alert System
===============================================
Monitors cardiac analysis results against configurable thresholds
and fires multi-channel alerts:

  • Email   — via SMTP (Gmail / any provider)
  • SMS     — via Twilio REST API (optional)
  • WhatsApp — via Twilio WhatsApp Sandbox (optional)

All alerts are logged regardless of whether Twilio is configured.
SMS/WhatsApp silently log-only if credentials are absent.
"""

from __future__ import annotations

import os
import smtplib
import threading
import json
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Alert channels
# ---------------------------------------------------------------------------
class AlertChannel(str, Enum):
    EMAIL     = 'email'
    SMS       = 'sms'
    WHATSAPP  = 'whatsapp'


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    # ECG
    'ecg_classes_critical': ['Myocardial Infarction'],
    'ecg_classes_warning':  ['Arrhythmia'],
    'ecg_confidence_min':   0.55,   # only alert if model is confident enough
    # UCRS
    'ucrs_critical': 70.0,
    'ucrs_warning':  45.0,
    # Vitals
    'spo2_critical':  90.0,
    'spo2_warning':   94.0,
    'sbp_high_crit':  180,
    'sbp_low_crit':   80,
    'hr_high_crit':   150,
    'hr_low_crit':    40,
}


@dataclass
class AlertEvent:
    alert_id:    str
    patient_id:  str
    severity:    str          # Warning / Critical
    trigger:     str          # what caused the alert
    message:     str
    channels_fired: list
    channels_failed: list
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'alert_id':       self.alert_id,
            'patient_id':     self.patient_id,
            'severity':       self.severity,
            'trigger':        self.trigger,
            'message':        self.message,
            'channels_fired': self.channels_fired,
            'channels_failed': self.channels_failed,
            'timestamp':      self.timestamp,
        }


class AlertEngine:
    """
    Evaluates cardiac analysis results and fires configured alerts.
    Thread-safe; shares a single in-memory config dict.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.thresholds: dict = dict(DEFAULT_THRESHOLDS)
        # contacts: list of {'name', 'email', 'phone', 'channels': [...]}
        self.contacts: list = []
        # channel toggles
        self.channels_enabled: dict = {
            AlertChannel.EMAIL:    True,
            AlertChannel.SMS:      False,    # requires Twilio
            AlertChannel.WHATSAPP: False,    # requires Twilio
        }
        # in-memory recent alerts (last 200)
        self._recent_alerts: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        patient_id:     str,
        ecg_prediction: str,
        ecg_confidence: float,
        risk_level:     str,
        ucrs:           Optional[float] = None,
        vitals:         Optional[dict]  = None,
    ) -> Optional[AlertEvent]:
        """
        Run threshold checks.  Returns AlertEvent if alert fired, else None.
        Does NOT block — dispatches alert delivery in a background thread.
        """
        severity, trigger = self._check_thresholds(
            ecg_prediction, ecg_confidence, risk_level, ucrs, vitals
        )
        if not severity:
            return None

        message = self._build_message(patient_id, ecg_prediction,
                                      ecg_confidence, risk_level,
                                      ucrs, vitals, trigger)

        import uuid
        event = AlertEvent(
            alert_id=str(uuid.uuid4())[:8].upper(),
            patient_id=patient_id,
            severity=severity,
            trigger=trigger,
            message=message,
            channels_fired=[],
            channels_failed=[],
        )

        # Fire in background so the HTTP response is not delayed
        t = threading.Thread(target=self._dispatch, args=(event,), daemon=True)
        t.start()

        with self._lock:
            self._recent_alerts.insert(0, event.to_dict())
            if len(self._recent_alerts) > 200:
                self._recent_alerts = self._recent_alerts[:200]

        return event

    def get_recent_alerts(self, limit: int = 50) -> list:
        with self._lock:
            return self._recent_alerts[:limit]

    def update_config(self, thresholds: Optional[dict] = None,
                      contacts: Optional[list] = None,
                      channels: Optional[dict] = None) -> None:
        with self._lock:
            if thresholds:
                self.thresholds.update(thresholds)
            if contacts is not None:
                self.contacts = contacts
            if channels:
                for ch, enabled in channels.items():
                    try:
                        self.channels_enabled[AlertChannel(ch)] = bool(enabled)
                    except ValueError:
                        pass

    def get_config(self) -> dict:
        with self._lock:
            return {
                'thresholds': dict(self.thresholds),
                'contacts':   list(self.contacts),
                'channels': {ch.value: v for ch, v in self.channels_enabled.items()},
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_thresholds(self, ecg_pred, ecg_conf, risk_level, ucrs, vitals):
        """
        Returns (severity, trigger_description) or (None, None) if no alert.
        """
        t = self.thresholds

        # UCRS critical
        if ucrs is not None and ucrs >= t.get('ucrs_critical', 70):
            return 'Critical', f'Unified Risk Score {ucrs:.1f}/100 exceeds critical threshold'

        # ECG Critical classes
        if ecg_pred in t.get('ecg_classes_critical', []) and ecg_conf >= t.get('ecg_confidence_min', 0.5):
            return 'Critical', f'ECG classified as {ecg_pred} (confidence {ecg_conf*100:.0f}%)'

        # UCRS warning
        if ucrs is not None and ucrs >= t.get('ucrs_warning', 45):
            return 'Warning', f'Unified Risk Score {ucrs:.1f}/100 exceeds warning threshold'

        # ECG Warning classes
        if ecg_pred in t.get('ecg_classes_warning', []) and ecg_conf >= t.get('ecg_confidence_min', 0.5):
            return 'Warning', f'ECG classified as {ecg_pred} (confidence {ecg_conf*100:.0f}%)'

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
            if spo2 is not None and spo2 < t.get('spo2_warning', 94):
                return 'Warning', f'SpO2 below normal at {spo2:.0f}%'

        return None, None

    @staticmethod
    def _build_message(patient_id, ecg_pred, ecg_conf, risk_level, ucrs, vitals, trigger):
        lines = [
            f"🚨 CARDIAC ALERT — Patient {patient_id}",
            f"Trigger : {trigger}",
            f"ECG     : {ecg_pred} ({ecg_conf*100:.0f}% confidence)",
            f"Risk    : {risk_level}",
        ]
        if ucrs is not None:
            lines.append(f"UCRS    : {ucrs:.1f}/100")
        if vitals:
            if vitals.get('spo2'):
                lines.append(f"SpO2    : {vitals['spo2']:.0f}%")
            if vitals.get('systolic'):
                lines.append(f"BP      : {vitals['systolic']:.0f}/{vitals.get('diastolic',0):.0f} mmHg")
            if vitals.get('heart_rate'):
                lines.append(f"HR      : {vitals['heart_rate']:.0f} bpm")
        lines.append(f"Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return '\n'.join(lines)

    def _dispatch(self, event: AlertEvent) -> None:
        """Attempt to deliver alert on all enabled channels to all contacts."""
        for contact in self.contacts:
            channels = contact.get('channels', [AlertChannel.EMAIL.value])
            email = contact.get('email', '')
            phone = contact.get('phone', '')

            if AlertChannel.EMAIL.value in channels and self.channels_enabled.get(AlertChannel.EMAIL):
                if email:
                    ok = self._send_email(email, contact.get('name', ''), event)
                    (event.channels_fired if ok else event.channels_failed).append(
                        f'email:{email}')

            if AlertChannel.SMS.value in channels and self.channels_enabled.get(AlertChannel.SMS):
                ok = self._send_sms(phone, event)
                (event.channels_fired if ok else event.channels_failed).append(
                    f'sms:{phone}')

            if AlertChannel.WHATSAPP.value in channels and self.channels_enabled.get(AlertChannel.WHATSAPP):
                ok = self._send_whatsapp(phone, event)
                (event.channels_fired if ok else event.channels_failed).append(
                    f'whatsapp:{phone}')

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    def _send_email(self, to_email: str, to_name: str, event: AlertEvent) -> bool:
        try:
            smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
            smtp_port = int(os.environ.get('SMTP_PORT', 587))
            smtp_user = os.environ.get('SMTP_USER', '')
            smtp_pass = os.environ.get('SMTP_PASS', '')
            from_addr = os.environ.get('ALERT_FROM_EMAIL', smtp_user)

            if not smtp_user or not smtp_pass:
                print(f"[ALERT] Email skipped (no SMTP credentials configured) → {to_email}")
                # Log the alert message even without sending
                print(f"[ALERT-LOG] {event.message}")
                return False

            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 [{event.severity}] Cardiac Alert — Patient {event.patient_id}"
            msg['From']    = f"CardiacMonitor Pro <{from_addr}>"
            msg['To']      = to_email

            html_body = f"""
            <html><body style="font-family:Arial,sans-serif;background:#0a0a1a;color:#e0e0e0;padding:20px;">
            <div style="max-width:600px;margin:auto;background:#1a1a2e;border-radius:12px;overflow:hidden;">
              <div style="background:{'#c0392b' if event.severity=='Critical' else '#e67e22'};padding:20px;text-align:center;">
                <h1 style="color:white;margin:0;">{'🚨' if event.severity=='Critical' else '⚠️'} {event.severity} Alert</h1>
                <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;">Cardiac Abnormality Detection System</p>
              </div>
              <div style="padding:24px;">
                <p><strong>Patient:</strong> {event.patient_id}</p>
                <p><strong>Alert ID:</strong> {event.alert_id}</p>
                <p><strong>Trigger:</strong> {event.trigger}</p>
                <pre style="background:#0d1117;padding:16px;border-radius:8px;color:#00d4ff;font-size:13px;white-space:pre-wrap;">{event.message}</pre>
                <p style="color:#888;font-size:12px;margin-top:20px;">
                  This is an automated alert from CardiacMonitor Pro.<br>
                  <em>For medical emergencies, call 112 / 911 immediately.</em>
                </p>
              </div>
            </div>
            </body></html>"""

            msg.attach(MIMEText(event.message, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(from_addr, to_email, msg.as_string())

            print(f"[ALERT] ✅ Email sent → {to_email}")
            return True

        except Exception as exc:
            print(f"[ALERT] ❌ Email failed → {to_email}: {exc}")
            return False

    def _send_sms(self, phone: str, event: AlertEvent) -> bool:
        try:
            sid   = os.environ.get('TWILIO_ACCOUNT_SID', '')
            token = os.environ.get('TWILIO_AUTH_TOKEN', '')
            from_ = os.environ.get('TWILIO_FROM_NUMBER', '')
            if not all([sid, token, from_, phone]):
                print(f"[ALERT] SMS skipped (Twilio not configured) → {phone}")
                print(f"[ALERT-LOG] SMS would send: {event.trigger}")
                return False
            from twilio.rest import Client
            body = f"[{event.severity}] Cardiac Alert – Pt {event.patient_id}: {event.trigger}"
            Client(sid, token).messages.create(body=body, from_=from_, to=phone)
            print(f"[ALERT] ✅ SMS sent → {phone}")
            return True
        except Exception as exc:
            print(f"[ALERT] ❌ SMS failed → {phone}: {exc}")
            return False

    def _send_whatsapp(self, phone: str, event: AlertEvent) -> bool:
        try:
            sid   = os.environ.get('TWILIO_ACCOUNT_SID', '')
            token = os.environ.get('TWILIO_AUTH_TOKEN', '')
            wa_from = os.environ.get('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
            if not all([sid, token, phone]):
                print(f"[ALERT] WhatsApp skipped (Twilio not configured)")
                return False
            from twilio.rest import Client
            body = f"*{event.severity} Cardiac Alert*\nPatient: {event.patient_id}\n{event.trigger}"
            Client(sid, token).messages.create(
                body=body, from_=wa_from, to=f'whatsapp:{phone}')
            print(f"[ALERT] ✅ WhatsApp sent → {phone}")
            return True
        except Exception as exc:
            print(f"[ALERT] ❌ WhatsApp failed → {phone}: {exc}")
            return False


# Module-level singleton
alert_engine = AlertEngine()
