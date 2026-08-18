"""
Vital Fusion Engine — Multi-Vital Cardiac Risk Scoring
======================================================
Combines ECG classification results with real-world vitals:
  • SpO2         (peripheral oxygen saturation, %)
  • Blood Pressure (systolic / diastolic, mmHg)
  • Temperature   (°C)

Outputs a Unified Cardiac Risk Score (UCRS) 0–100 and per-vital status.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Vital thresholds (clinical reference ranges)
# ---------------------------------------------------------------------------

SPO2_CRITICAL  = 90.0   # % — below this → Critical
SPO2_WARNING   = 94.0   # % — below this → Warning

SBP_HIGH_CRIT  = 180    # mmHg systolic — hypertensive crisis
SBP_LOW_CRIT   = 80     # mmHg systolic — shock
SBP_HIGH_WARN  = 140    # mmHg systolic — Stage 2 hypertension
SBP_LOW_WARN   = 90     # mmHg systolic — hypotension

DBP_HIGH_CRIT  = 120    # mmHg diastolic — hypertensive crisis
DBP_LOW_CRIT   = 50     # mmHg diastolic
DBP_HIGH_WARN  = 90     # mmHg diastolic — Stage 2 hypertension
DBP_LOW_WARN   = 60     # mmHg diastolic — hypotension

TEMP_HIGH_CRIT = 40.0   # °C — hyperpyrexia
TEMP_LOW_CRIT  = 35.0   # °C — hypothermia
TEMP_HIGH_WARN = 38.5   # °C — high fever
TEMP_LOW_WARN  = 36.0   # °C — low-normal

HR_HIGH_CRIT   = 150    # bpm — severe tachycardia
HR_LOW_CRIT    = 40     # bpm — severe bradycardia
HR_HIGH_WARN   = 120    # bpm — tachycardia
HR_LOW_WARN    = 50     # bpm — bradycardia


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_NORMAL   = "Normal"
STATUS_WARNING  = "Warning"
STATUS_CRITICAL = "Critical"
STATUS_UNKNOWN  = "Unknown"


@dataclass
class VitalReading:
    spo2:        Optional[float] = None   # %
    systolic:    Optional[float] = None   # mmHg
    diastolic:   Optional[float] = None   # mmHg
    temperature: Optional[float] = None   # °C
    heart_rate:  Optional[float] = None   # bpm
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'spo2':        self.spo2,
            'systolic':    self.systolic,
            'diastolic':   self.diastolic,
            'temperature': self.temperature,
            'heart_rate':  self.heart_rate,
            'timestamp':   self.timestamp,
        }


@dataclass
class VitalFusionResult:
    ucrs:              float   # Unified Cardiac Risk Score 0–100
    fused_risk_level:  str     # Low / Medium / High / Critical
    vitals_status:     dict    # per-vital status dict
    commentary:        list    # clinical commentary strings
    vital_scores:      dict    # intermediate per-vital penalty scores
    timestamp:         str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'ucrs':             round(self.ucrs, 1),
            'fused_risk_level': self.fused_risk_level,
            'vitals_status':    self.vitals_status,
            'commentary':       self.commentary,
            'vital_scores':     self.vital_scores,
            'timestamp':        self.timestamp,
        }


class VitalFusionEngine:
    """
    Fuses ECG classification + auxiliary vitals into a
    Unified Cardiac Risk Score (UCRS).

    Weights:
        ECG risk   50%
        SpO2       20%
        BP         20%
        Temp        5%
        HR          5%
    """

    # ECG class → base ECG penalty (0–100)
    _ECG_PENALTY = {
        'Normal':               0,
        'Arrhythmia':          50,
        'Myocardial Infarction': 90,
        'Other Abnormality':   40,
        'Unknown':             30,
    }

    _ECG_WEIGHT   = 0.50
    _SPO2_WEIGHT  = 0.20
    _BP_WEIGHT    = 0.20
    _TEMP_WEIGHT  = 0.05
    _HR_WEIGHT    = 0.05

    # ------------------------------------------------------------------
    def evaluate(
        self,
        ecg_prediction:  str,
        ecg_confidence:  float,
        ecg_risk_level:  str,
        vitals:          VitalReading,
    ) -> VitalFusionResult:
        """
        Main entry point.  Returns a VitalFusionResult with the UCRS and details.
        """
        status     = {}
        commentary = []
        scores     = {}

        # -- ECG component -------------------------------------------
        base_ecg = self._ECG_PENALTY.get(ecg_prediction, 30)
        # Scale by confidence (low confidence → partial penalty)
        ecg_score = base_ecg * max(0.5, float(ecg_confidence))
        scores['ecg'] = round(ecg_score, 1)

        # -- SpO2 component ------------------------------------------
        spo2_score, spo2_status, spo2_notes = self._eval_spo2(vitals.spo2)
        scores['spo2']  = round(spo2_score, 1)
        status['spo2']  = spo2_status
        commentary.extend(spo2_notes)

        # -- Blood Pressure component --------------------------------
        bp_score, bp_status, bp_notes = self._eval_bp(vitals.systolic, vitals.diastolic)
        scores['bp']   = round(bp_score, 1)
        status['bp']   = bp_status
        commentary.extend(bp_notes)

        # -- Temperature component -----------------------------------
        temp_score, temp_status, temp_notes = self._eval_temperature(vitals.temperature)
        scores['temperature'] = round(temp_score, 1)
        status['temperature'] = temp_status
        commentary.extend(temp_notes)

        # -- Heart Rate component ------------------------------------
        hr_score, hr_status, hr_notes = self._eval_heart_rate(vitals.heart_rate)
        scores['heart_rate'] = round(hr_score, 1)
        status['heart_rate'] = hr_status
        commentary.extend(hr_notes)

        # -- UCRS weighted sum ---------------------------------------
        ucrs = (
            ecg_score   * self._ECG_WEIGHT  +
            spo2_score  * self._SPO2_WEIGHT +
            bp_score    * self._BP_WEIGHT   +
            temp_score  * self._TEMP_WEIGHT +
            hr_score    * self._HR_WEIGHT
        )
        ucrs = round(min(100.0, max(0.0, ucrs)), 1)

        # -- Fused risk level ----------------------------------------
        fused_risk = self._classify_ucrs(ucrs)

        # Add overall commentary
        if not commentary:
            commentary.append("All vitals within acceptable ranges")

        return VitalFusionResult(
            ucrs=ucrs,
            fused_risk_level=fused_risk,
            vitals_status=status,
            commentary=commentary,
            vital_scores=scores,
        )

    # ------------------------------------------------------------------
    # Per-vital evaluators (return penalty 0–100, status str, notes list)
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_spo2(spo2: Optional[float]):
        if spo2 is None:
            return 0.0, STATUS_UNKNOWN, []
        notes = []
        if spo2 < SPO2_CRITICAL:
            notes.append(f"⚠️ Critical: SpO2 {spo2:.0f}% — immediate oxygen therapy required")
            return 100.0, STATUS_CRITICAL, notes
        if spo2 < SPO2_WARNING:
            notes.append(f"⚡ Warning: SpO2 {spo2:.0f}% — below normal, monitor closely")
            return 60.0, STATUS_WARNING, notes
        return 0.0, STATUS_NORMAL, notes

    @staticmethod
    def _eval_bp(systolic: Optional[float], diastolic: Optional[float]):
        if systolic is None and diastolic is None:
            return 0.0, STATUS_UNKNOWN, []
        notes = []
        sbp = systolic or 120.0
        dbp = diastolic or 80.0
        score = 0.0
        if sbp >= SBP_HIGH_CRIT or dbp >= DBP_HIGH_CRIT:
            notes.append(f"🚨 Critical: BP {sbp:.0f}/{dbp:.0f} mmHg — hypertensive crisis")
            return 100.0, STATUS_CRITICAL, notes
        if sbp <= SBP_LOW_CRIT or dbp <= DBP_LOW_CRIT:
            notes.append(f"🚨 Critical: BP {sbp:.0f}/{dbp:.0f} mmHg — shock-level hypotension")
            return 100.0, STATUS_CRITICAL, notes
        if sbp >= SBP_HIGH_WARN or dbp >= DBP_HIGH_WARN:
            notes.append(f"⚡ Warning: BP {sbp:.0f}/{dbp:.0f} mmHg — Stage 2 hypertension")
            score = 55.0
        elif sbp <= SBP_LOW_WARN or dbp <= DBP_LOW_WARN:
            notes.append(f"⚡ Warning: BP {sbp:.0f}/{dbp:.0f} mmHg — hypotension")
            score = 50.0
        status = STATUS_WARNING if score > 0 else STATUS_NORMAL
        return score, status, notes

    @staticmethod
    def _eval_temperature(temp: Optional[float]):
        if temp is None:
            return 0.0, STATUS_UNKNOWN, []
        notes = []
        if temp >= TEMP_HIGH_CRIT or temp <= TEMP_LOW_CRIT:
            notes.append(f"🚨 Critical: Temp {temp:.1f}°C — extreme temperature deviation")
            return 90.0, STATUS_CRITICAL, notes
        if temp >= TEMP_HIGH_WARN:
            notes.append(f"⚡ Warning: Temp {temp:.1f}°C — high fever, cardiac demand increased")
            return 40.0, STATUS_WARNING, notes
        if temp <= TEMP_LOW_WARN:
            notes.append(f"⚡ Warning: Temp {temp:.1f}°C — low temperature, monitor for hypothermia")
            return 30.0, STATUS_WARNING, notes
        return 0.0, STATUS_NORMAL, notes

    @staticmethod
    def _eval_heart_rate(hr: Optional[float]):
        if hr is None:
            return 0.0, STATUS_UNKNOWN, []
        notes = []
        if hr >= HR_HIGH_CRIT or hr <= HR_LOW_CRIT:
            notes.append(f"🚨 Critical: HR {hr:.0f} bpm — severe rate abnormality")
            return 90.0, STATUS_CRITICAL, notes
        if hr >= HR_HIGH_WARN:
            notes.append(f"⚡ Warning: HR {hr:.0f} bpm — tachycardia")
            return 45.0, STATUS_WARNING, notes
        if hr <= HR_LOW_WARN:
            notes.append(f"⚡ Warning: HR {hr:.0f} bpm — bradycardia")
            return 40.0, STATUS_WARNING, notes
        return 0.0, STATUS_NORMAL, notes

    # ------------------------------------------------------------------
    @staticmethod
    def _classify_ucrs(ucrs: float) -> str:
        if ucrs >= 70:
            return "Critical"
        if ucrs >= 45:
            return "High"
        if ucrs >= 20:
            return "Medium"
        return "Low"
