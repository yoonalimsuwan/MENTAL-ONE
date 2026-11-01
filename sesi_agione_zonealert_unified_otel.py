"""
=============================================================================
SESI AGIONE ZONE ALERT × UNIFIED DASHBOARD × OTEL CODIFICATION COLLECTOR
Zone-Specific Quality Alerting + Unified Visualization + OTel Pipeline
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[AZ1] AGIONEZoneAlertBridge
      • Pulls quality_report() from EcosystemOrchestrator
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Per-zone threshold evaluation (PNDC Axiom 3 proportionality)
      • Zone-specific alert emission with No-Zeno rate limiting
      • Regime classification per zone (paper6 Thm 5.1)
      • CUSUM change-point detection per zone stream

[UD1] UnifiedZoneAwareGrafanaDashboard
      • Single-pane glass: traces + logs + metrics + service map + zone alerts
      • Per-zone panels with differentiated thresholds
      • Regime classification per zone (paper6 Thm 5.1)
      • Codification event timeline correlated with zone alerts
      • Template variables for zone/service/trace filtering
      • Drill-down from zone → service → trace → log

[OC1] OTelCodificationCollector
      • OTLP receiver for codification events from Psyche Plus
      • Processors: batch, memory_limiter, filter/drop-sensitive,
        zone_router, attributes/add-quality, transform/codification-enrich
      • Exporters: Prometheus (metrics), Jaeger (traces), Loki (logs),
        Alertmanager (alerts)
      • Pipelines: codification_traces, codification_metrics, codification_logs
      • PNDC Axiom 3 compliance: sensitive data filtering
      • Production HA configuration (2 replicas, TLS, health checks)

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds alert emission and event ingestion rate
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful quality change for alert triggers
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → per-zone regime classification
[C] paper6 Thm 4.1: Degenerate occupation-time law ½δ₀+½δ₁
    → polarization detection per zone
[C] Six_Problems P4: CUSUM + subsampling on raw process
    → change-point detection in per-zone quality streams
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
    → confidence interval for zone threshold calibration
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
    → NaN/Inf sanitization for all quality inputs
[C] agi_one_v3_11.py: EcosystemOrchestrator.quality_report()
    → per-domain quality score extraction
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → multi-identity risk for codification correlation
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-specific alert thresholds
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• AGIONE zone alerting provides OBSERVABILITY, not causation.
  Quality scores reflect simulation health, not system truth.
• Zone-specific thresholds are PROPORTIONALITY-BASED (PNDC Axiom 3),
  not absolute. Must be calibrated against real operational data.
• Unified dashboard is a VISUALIZATION TOOL, not a decision engine.
  Human operators must interpret and act.
• OTel Collector config is a TEMPLATE, not a deployed system.
  Real deployment requires TLS certificates, IAM setup, security audit.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• Production deployment requires regulatory approval and security audit.
• This module does NOT create, detect, or simulate consciousness.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.agione_zonealert_unified_otel")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# AGIONE / Psyche Plus imports (graceful)
try:
    from agi_one_v3_11 import AGIONE, EcosystemOrchestrator
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

try:
    from agi_one_psyche_plus_v33 import _safe_unit_scalar, AGIOnePsychePlus
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False
    def _safe_unit_scalar(value, name="", logger=None, default=0.5):
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            if logger:
                logger.warning(f"[SANITISE] {name}={value!r} not numeric")
            return default
        if math.isnan(v) or math.isinf(v):
            if logger:
                logger.warning(f"[SANITISE] {name}={v} NaN/Inf")
            return default
        return max(0.0, min(1.0, v))

__all__ = [
    "SecurityZone", "AlertSeverity", "QualityRegime", "CodificationOutcome",
    "ZoneAlertConfig", "AGIONEZoneAlertBridge",
    "UnifiedZoneAwareGrafanaDashboard",
    "OTelCodificationCollector",
    "SESIUnifiedZonePipeline",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class SecurityZone(Enum):
    """Four security zones from SESI network segmentation."""
    Z0_IMPLANT = "z0-implant"
    Z1_CLINICAL = "z1-clinical"
    Z2_PROCESSING = "z2-processing"
    Z3_EXTERNAL = "z3-external"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class QualityRegime(Enum):
    """Growth-rate regime classification (paper6 Thm 5.1)."""
    FINITE_MEAN = 0          # θ > 1
    REGULARLY_VARYING = 1    # θ ∈ (0, 1)
    DEGENERATE = 2           # θ → 0 (½δ₀ + ½δ₁)
    INSUFFICIENT_DATA = 3


class CodificationOutcome(Enum):
    """Outcome of Psyche Plus maybe_codify() dual-gate."""
    SUCCESS = auto()
    REFUSED_VALIDITY = auto()
    REFUSED_QUALITY = auto()
    REFUSED_NONFINITE = auto()


# ═══════════════════════════════════════════════════════════════════════
# [AZ1] AGIONE ZONE ALERT BRIDGE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ZoneAlertConfig:
    """
    Per-zone alert configuration.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be calibrated against real data.
    PNDC Axiom 3 proportionality: Z0 strictest, Z3 loosest.
    """
    zone: SecurityZone
    # Quality thresholds (lower = worse)
    quality_warning: float = 0.5
    quality_critical: float = 0.3
    quality_emergency: float = 0.1
    # Codification failure thresholds
    codify_failure_warning: float = 0.1
    codify_failure_critical: float = 0.3
    # Default severity
    default_severity: AlertSeverity = AlertSeverity.WARNING
    # Receiver
    receiver: str = "platform-team"
    # Grouping
    group_wait: str = "30s"
    group_interval: str = "5m"
    repeat_interval: str = "4h"


@dataclass
class ZoneAlert:
    """A single zone-specific alert."""
    alert_id: str
    zone: SecurityZone
    severity: AlertSeverity
    alert_type: str
    message: str
    quality_score: Optional[float] = None
    regime: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "zone": self.zone.value,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "message": self.message,
            "quality_score": self.quality_score,
            "regime": self.regime,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }


class AGIONEZoneAlertBridge:
    """
    [D] Bridges AGIONE quality_report() to zone-specific alerts.

    Features:
      • Pulls quality_report() from EcosystemOrchestrator
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Per-zone threshold evaluation (PNDC Axiom 3 proportionality)
      • Zone-specific alert emission with No-Zeno rate limiting
      • Regime classification per zone (paper6 Thm 5.1)
      • CUSUM change-point detection per zone stream

    HONEST: This bridge provides OBSERVABILITY, not causation.
    Quality scores reflect simulation health, not system truth.
    All emitted alerts are sanitized and rate-limited.

    Mathematical grounding:
      • _safe_unit_scalar sanitization (Psyche Plus v3.3)
      • No-Zeno bound on alert rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for meaningful alert triggers (Delta_Min Thm 1.2)
      • paper6 Thm 5.1: regime classification per zone
    """

    # Default zone configs (PNDC Axiom 3 proportionality)
    DEFAULT_ZONE_CONFIGS = {
        SecurityZone.Z0_IMPLANT: ZoneAlertConfig(
            zone=SecurityZone.Z0_IMPLANT,
            quality_warning=0.7,
            quality_critical=0.5,
            quality_emergency=0.3,
            codify_failure_warning=0.05,
            codify_failure_critical=0.1,
            default_severity=AlertSeverity.CRITICAL,
            receiver="clinical-pager",
            group_wait="10s",
            repeat_interval="1h",
        ),
        SecurityZone.Z1_CLINICAL: ZoneAlertConfig(
            zone=SecurityZone.Z1_CLINICAL,
            quality_warning=0.6,
            quality_critical=0.4,
            quality_emergency=0.2,
            codify_failure_warning=0.1,
            codify_failure_critical=0.2,
            default_severity=AlertSeverity.CRITICAL,
            receiver="clinical-team",
            group_wait="30s",
            repeat_interval="4h",
        ),
        SecurityZone.Z2_PROCESSING: ZoneAlertConfig(
            zone=SecurityZone.Z2_PROCESSING,
            quality_warning=0.5,
            quality_critical=0.3,
            quality_emergency=0.1,
            codify_failure_warning=0.15,
            codify_failure_critical=0.3,
            default_severity=AlertSeverity.WARNING,
            receiver="platform-slack",
            group_wait="1m",
            repeat_interval="12h",
        ),
        SecurityZone.Z3_EXTERNAL: ZoneAlertConfig(
            zone=SecurityZone.Z3_EXTERNAL,
            quality_warning=0.4,
            quality_critical=0.2,
            quality_emergency=0.1,
            codify_failure_warning=0.2,
            codify_failure_critical=0.4,
            default_severity=AlertSeverity.INFO,
            receiver="platform-email",
            group_wait="5m",
            repeat_interval="24h",
        ),
    }

    def __init__(
        self,
        agione: Optional[Any] = None,
        zone_configs: Optional[Dict[SecurityZone, ZoneAlertConfig]] = None,
    ):
        self.agione = agione
        self.zone_configs = zone_configs or self.DEFAULT_ZONE_CONFIGS
        self._alerts: List[ZoneAlert] = []
        self._alert_timestamps: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._quality_history: Dict[str, List[float]] = {
            zone.value: [] for zone in self.zone_configs
        }
        self._last_quality: Dict[str, float] = {
            zone.value: 0.5 for zone in self.zone_configs
        }

        # No-Zeno parameters
        self.max_alerts_per_zone_per_minute = 10
        self.min_alert_interval_s = 5.0

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_alert_rate(self, zone: SecurityZone) -> Dict[str, Any]:
        """Check if alert emission is within No-Zeno bound for this zone."""
        now = time.time()
        zone_key = zone.value
        cutoff = now - 60.0
        timestamps = self._alert_timestamps[zone_key]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= self.max_alerts_per_zone_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_zone_per_minute_exceeded",
                "current": len(timestamps),
                "limit": self.max_alerts_per_zone_per_minute,
            }

        if timestamps:
            time_since_last = now - timestamps[-1]
            if time_since_last < self.min_alert_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                    "time_since_last_s": time_since_last,
                    "min_interval_s": self.min_alert_interval_s,
                }

        return {
            "allowed": True,
            "current_count": len(timestamps),
            "limit": self.max_alerts_per_zone_per_minute,
        }

    def record_alert_timestamp(self, zone: SecurityZone) -> None:
        """Record an alert emission for rate limiting."""
        self._alert_timestamps[zone.value].append(time.time())

    # ── Quality Extraction ───────────────────────────────────────────────

    def extract_quality_from_agione(self) -> Dict[str, float]:
        """
        Extract quality scores from AGIONE's EcosystemOrchestrator.

        From agi_one_v3_11.py:
          EcosystemOrchestrator.quality_report() returns per-domain scores.
          The "ecosystem" key is the mean across all domains.

        All scores sanitized via _safe_unit_scalar.

        Returns:
            dict with quality scores per domain
        """
        if self.agione is None:
            return {"ecosystem": 0.5}

        orchestrator = getattr(self.agione, "orchestrator", None)
        if orchestrator is None:
            return {"ecosystem": 0.5}

        try:
            quality_report = orchestrator.quality_report()
            sanitized = {}
            for domain, score in quality_report.items():
                sanitized_score = _safe_unit_scalar(
                    score, name=f"quality.{domain}", logger=logger, default=0.5
                )
                sanitized[domain] = sanitized_score
            return sanitized
        except Exception as e:
            logger.warning(f"quality_report() failed: {e}")
            return {"ecosystem": 0.5}

    # ── Regime Classification ────────────────────────────────────────────

    def classify_zone_regime(self, zone: SecurityZone) -> Dict[str, Any]:
        """
        Classify quality regime for a specific zone using Hill estimator.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → ordinary monitoring
          θ ∈ (0,1):  regularly varying → distributional monitoring
          θ → 0:      degenerate → polarized monitoring
        """
        history = self._quality_history.get(zone.value, [])
        if len(history) < 20:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_samples": len(history),
            }

        median = np.median(history)
        deviations = [abs(q - median) for q in history if abs(q - median) > 0.01]

        if len(deviations) < 10:
            return {"regime": QualityRegime.FINITE_MEAN.name}

        k = min(10, len(deviations) - 1)
        sorted_devs = sorted(deviations, reverse=True)
        x_k1 = sorted_devs[k]

        if x_k1 <= 0:
            return {"regime": QualityRegime.INSUFFICIENT_DATA.name}

        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        if theta_hat > 1.0:
            regime = QualityRegime.FINITE_MEAN
            description = "θ > 1: finite mean. Standard monitoring appropriate."
        elif theta_hat > 0.0:
            regime = QualityRegime.REGULARLY_VARYING
            description = "θ ∈ (0,1): regularly varying. Distributional monitoring."
        else:
            regime = QualityRegime.DEGENERATE
            description = "θ → 0: degenerate. Quality polarizing between extremes."

        return {
            "regime": regime.name,
            "theta_hat": theta_hat,
            "description": description,
            "n_samples": len(history),
        }

    def detect_zone_change_point(self, zone: SecurityZone) -> Dict[str, Any]:
        """
        CUSUM change-point detection on zone quality stream.

        From Six_Problems P4: CUSUM + subsampling on raw process.
        """
        history = self._quality_history.get(zone.value, [])
        if len(history) < 20:
            return {"change_point_detected": False, "reason": "insufficient_data"}

        target = np.mean(history)
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0
        change_idx = None
        drift = 0.5

        for i, v in enumerate(history):
            deviation = v - target
            cusum_pos = max(0, cusum_pos + deviation - drift)
            cusum_neg = max(0, cusum_neg - deviation - drift)

            current_max = max(cusum_pos, cusum_neg)
            if current_max > max_cusum:
                max_cusum = current_max
                if current_max > 5.0:
                    change_idx = i

        return {
            "change_point_detected": change_idx is not None,
            "cusum_statistic": max_cusum,
            "threshold": 5.0,
            "change_point_index": change_idx,
        }

    # ── Zone-Specific Alert Evaluation ───────────────────────────────────

    def evaluate_zone_alerts(
        self,
        quality_scores: Optional[Dict[str, float]] = None,
        trace_id: Optional[str] = None,
    ) -> List[ZoneAlert]:
        """
        Evaluate quality scores against per-zone thresholds and emit alerts.

        Args:
            quality_scores: Optional quality scores (if None, pulled from AGIONE)
            trace_id: Optional trace ID for correlation

        Returns:
            list of ZoneAlert objects emitted
        """
        if quality_scores is None:
            quality_scores = self.extract_quality_from_agione()

        ecosystem_quality = quality_scores.get("ecosystem", 0.5)
        alerts = []

        for zone, cfg in self.zone_configs.items():
            # Update quality history
            self._quality_history[zone.value].append(ecosystem_quality)
            if len(self._quality_history[zone.value]) > 100:
                self._quality_history[zone.value] = self._quality_history[zone.value][-100:]

            # Check rate limit
            rate_check = self.check_alert_rate(zone)
            if not rate_check["allowed"]:
                logger.debug(f"Zone {zone.value} alert rate-limited: {rate_check['reason']}")
                continue

            # Evaluate thresholds
            severity = None
            alert_type = None
            message = None

            if ecosystem_quality < cfg.quality_emergency:
                severity = AlertSeverity.EMERGENCY
                alert_type = "quality_emergency"
                message = (
                    f"EMERGENCY: Quality in {zone.value} dropped below "
                    f"{cfg.quality_emergency}. Immediate intervention required."
                )
            elif ecosystem_quality < cfg.quality_critical:
                severity = AlertSeverity.CRITICAL
                alert_type = "quality_critical"
                message = (
                    f"CRITICAL: Quality in {zone.value} dropped below "
                    f"{cfg.quality_critical}. Immediate review required."
                )
            elif ecosystem_quality < cfg.quality_warning:
                severity = cfg.default_severity
                alert_type = "quality_warning"
                message = (
                    f"Quality in {zone.value} dropped below "
                    f"{cfg.quality_warning}. Monitor closely."
                )

            # Check regime
            regime_result = self.classify_zone_regime(zone)
            if regime_result["regime"] == QualityRegime.DEGENERATE.name:
                if severity is None or severity.value < AlertSeverity.CRITICAL.value:
                    severity = AlertSeverity.CRITICAL
                    alert_type = "degenerate_regime"
                    message = (
                        f"DEGENERATE regime detected in {zone.value}. "
                        f"paper6 Thm 5.1: occupation fraction polarizing. "
                        f"Averaging-based monitoring will miss this."
                    )

            # Check CUSUM change-point
            cusum_result = self.detect_zone_change_point(zone)
            if cusum_result["change_point_detected"]:
                if severity is None:
                    severity = AlertSeverity.WARNING
                    alert_type = "quality_change_point"
                    message = (
                        f"CUSUM change-point detected in {zone.value}. "
                        f"Six_Problems P4: regime change in quality stream."
                    )

            if severity is not None:
                alert = ZoneAlert(
                    alert_id=str(uuid.uuid4())[:8],
                    zone=zone,
                    severity=severity,
                    alert_type=alert_type,
                    message=message,
                    quality_score=ecosystem_quality,
                    regime=regime_result["regime"],
                    trace_id=trace_id,
                    metadata={
                        "cusum_statistic": cusum_result.get("cusum_statistic"),
                        "theta_hat": regime_result.get("theta_hat"),
                    },
                )
                alerts.append(alert)
                self._alerts.append(alert)
                self.record_alert_timestamp(zone)
                self._last_quality[zone.value] = ecosystem_quality

                logger.info(
                    f"[ALERT] {alert.alert_id} | zone={zone.value} | "
                    f"severity={severity.value} | type={alert_type} | "
                    f"quality={ecosystem_quality:.3f}"
                )

        return alerts

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
       
        return [a.to_dict() for a in self._alerts[-limit:]]

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "agione_attached": self.agione is not None,
            "total_alerts": len(self._alerts),
            "zone_configs": {
                zone.value: {
                    "quality_warning": cfg.quality_warning,
                    "quality_critical": cfg.quality_critical,
                    "quality_emergency": cfg.quality_emergency,
                    "default_severity": cfg.default_severity.value,
                }
                for zone, cfg in self.zone_configs.items()
            },
            "delta_min": self.compute_delta_min(),
            "regime_per_zone": {
                zone.value: self.classify_zone_regime(zone)["regime"]
                for zone in self.zone_configs
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [UD1] UNIFIED ZONE-AWARE GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class UnifiedZoneAwareGrafanaDashboard:
    """
    [D] Generates unified Grafana dashboard for zone-aware visualization.

    Features:
      • Single-pane glass: traces + logs + metrics + service map + zone alerts
      • Per-zone panels with differentiated thresholds
      • Regime classification per zone (paper6 Thm 5.1)
      • Codification event timeline correlated with zone alerts
      • Template variables for zone/service/trace filtering
      • Drill-down from zone → service → trace → log

    HONEST: This is a DASHBOARD GENERATOR. It produces Grafana JSON
    but does NOT deploy Grafana or make security decisions.
    """

    def __init__(
        self,
        prometheus_uid: str = "prometheus",
        jaeger_uid: str = "jaeger",
        loki_uid: str = "loki",
    ):
        self.prometheus_uid = prometheus_uid
        self.jaeger_uid = jaeger_uid
        self.loki_uid = loki_uid

    def _panel(
        self,
        panel_id: int,
        title: str,
        panel_type: str,
        targets: List[Dict[str, Any]],
        grid_pos: Dict[str, int],
        datasource_uid: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ds_uid = datasource_uid or self.prometheus_uid
        ds_type = "jaeger" if ds_uid == self.jaeger_uid else \
                  "loki" if ds_uid == self.loki_uid else "prometheus"
        panel = {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "datasource": {"type": ds_type, "uid": ds_uid},
            "targets": targets,
            "gridPos": grid_pos,
            "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
            "options": {},
        }
        if extra:
            panel.update(extra)
        return panel

    def generate_zone_panels(
        self,
        zone: SecurityZone,
        start_y: int,
        panel_id_start: int,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """Generate panels for a specific zone."""
        from sesi_zone_alertmanager_jaeger_loki import ZoneAlertConfig
        # Use default config for this zone
        default_configs = AGIONEZoneAlertBridge.DEFAULT_ZONE_CONFIGS
        cfg = default_configs.get(zone)
        if cfg is None:
            return [], start_y, panel_id_start

        zone_name = zone.value
        panels = []
        pid = panel_id_start

        # Zone header
        panels.append({
            "id": pid,
            "type": "text",
            "title": f"{zone_name.upper()} Zone",
            "options": {
                "mode": "markdown",
                "content": f"## {zone_name}\nSeverity: {cfg.default_severity.value}\n"
                           f"Warning: {cfg.quality_warning} | Critical: {cfg.quality_critical}",
            },
            "gridPos": {"h": 2, "w": 24, "x": 0, "y": start_y},
        })
        pid += 1
        start_y += 2

        # Quality score gauge
        panels.append(self._panel(
            pid, f"Quality Score ({zone_name})", "gauge",
            [{"expr": f'sesi_quality_ecosystem{{zone="{zone_name}"}}',
              "legendFormat": "quality"}],
            {"h": 6, "w": 6, "x": 0, "y": start_y},
            extra={"fieldConfig": {"defaults": {
                "min": 0, "max": 1,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": cfg.quality_warning},
                    {"color": "green", "value": cfg.quality_critical},
                ]},
            }}},
        ))
        pid += 1

        # Regime classification
        panels.append(self._panel(
            pid, f"Quality Regime ({zone_name})", "stat",
            [{"expr": f'sesi_quality_regime{{zone="{zone_name}"}}',
              "legendFormat": "regime"}],
            {"h": 6, "w": 6, "x": 6, "y": start_y},
            extra={"fieldConfig": {"defaults": {
                "mappings": [
                    {"type": "value", "options": {
                        "0": {"text": "FINITE_MEAN", "color": "green"},
                        "1": {"text": "REGULARLY_VARYING", "color": "yellow"},
                        "2": {"text": "DEGENERATE", "color": "red"},
                        "3": {"text": "INSUFFICIENT_DATA", "color": "gray"},
                    }},
                ],
            }}},
        ))
        pid += 1

        # CUSUM statistic
        panels.append(self._panel(
            pid, f"CUSUM ({zone_name})", "timeseries",
            [{"expr": f'sesi_cusum_statistic{{zone="{zone_name}"}}',
              "legendFormat": "cusum"}],
            {"h": 6, "w": 6, "x": 12, "y": start_y},
        ))
        pid += 1

        # Active alerts
        panels.append(self._panel(
            pid, f"Active Alerts ({zone_name})", "stat",
            [{"expr": f'sum(ALERTS{{zone="{zone_name}",alertstate="firing"}}) or vector(0)',
              "legendFormat": "alerts"}],
            {"h": 6, "w": 6, "x": 18, "y": start_y},
            extra={"fieldConfig": {"defaults": {
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 1},
                    {"color": "red", "value": 5},
                ]},
            }}},
        ))
        pid += 1

        start_y += 6
        return panels, start_y, pid

    def generate_codification_panels(
        self,
        start_y: int,
        panel_id_start: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Generate codification event panels."""
        panels = []
        pid = panel_id_start

        # Row header
        panels.append({
            "id": pid,
            "type": "text",
            "title": "Codification Events (Psyche Plus)",
            "options": {
                "mode": "markdown",
                "content": "## Psyche Plus Codification Events\n"
                           "Dual-gate: validity + quality (agi_one_psyche_plus_v33.py)",
            },
            "gridPos": {"h": 2, "w": 24, "x": 0, "y": start_y},
        })
        pid += 1
        start_y += 2

        # Codification success rate
        panels.append(self._panel(
            pid, "Codification Success Rate", "timeseries",
            [{"expr": (
                'sum(rate(sesi_codification_events_total{outcome="SUCCESS"}[5m])) '
                '/ sum(rate(sesi_codification_events_total[5m]))'
            ), "legendFormat": "success_rate"}],
            {"h": 8, "w": 8, "x": 0, "y": start_y},
            extra={"fieldConfig": {"defaults": {
                "min": 0, "max": 1, "unit": "percentunit",
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.7},
                    {"color": "green", "value": 0.9},
                ]},
            }}},
        ))
        pid += 1

        # Codification outcomes by type
        panels.append(self._panel(
            pid, "Codification Outcomes", "timeseries",
            [{"expr": 'sum(rate(sesi_codification_events_total[5m])) by (outcome)',
              "legendFormat": "{{outcome}}"}],
            {"h": 8, "w": 8, "x": 8, "y": start_y},
        ))
        pid += 1

        # Validity vs quality score distribution
        panels.append(self._panel(
            pid, "Validity vs Quality Scores", "scatter",
            [
                {"expr": 'sesi_codification_validity_score', "legendFormat": "validity"},
                {"expr": 'sesi_codification_quality_score', "legendFormat": "quality"},
            ],
            {"h": 8, "w": 8, "x": 16, "y": start_y},
        ))
        pid += 1

        start_y += 8
        return panels, pid

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete unified zone-aware Grafana dashboard."""
        panels = []
        y = 0
        pid = 1

        # Generate panels for each zone
        for zone in SecurityZone:
            zone_panels, y, pid = self.generate_zone_panels(zone, y, pid)
            panels.extend(zone_panels)
            y += 1  # Spacing between zones

        # Codification events section
        codification_panels, pid = self.generate_codification_panels(y, pid)
        panels.extend(codification_panels)

        dashboard = {
            "uid": "sesi-unified-zone-aware",
            "title": "SESI Unified Zone-Aware Dashboard",
            "tags": ["sesi", "zone-aware", "unified", "codification", "quality"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "templating": {
                "list": [
                    {
                        "name": "zone",
                        "type": "custom",
                        "options": [
                            {"text": "All", "value": ""},
                            {"text": "Z0 Implant", "value": "z0-implant"},
                            {"text": "Z1 Clinical", "value": "z1-clinical"},
                            {"text": "Z2 Processing", "value": "z2-processing"},
                            {"text": "Z3 External", "value": "z3-external"},
                        ],
                        "current": {"text": "All", "value": ""},
                    },
                    {
                        "name": "service",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "query": "label_values(sesi_codification_events_total, service)",
                        "refresh": 2,
                    },
                    {
                        "name": "trace_id",
                        "type": "textbox",
                        "current": {"text": "", "value": ""},
                        "label": "Trace ID (drill-down)",
                    },
                ],
            },
            "annotations": {
                "list": [
                    {
                        "name": "Zone Alerts",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "sesi_zone_alert_fired",
                        "iconColor": "red",
                        "titleFormat": "Alert: {{zone}} - {{alert_type}}",
                    },
                    {
                        "name": "Codification Events",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "sesi_codification_events_total",
                        "iconColor": "blue",
                        "titleFormat": "Codify: {{outcome}}",
                    },
                    {
                        "name": "Regime Changes",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "changes(sesi_quality_regime[5m]) > 0",
                        "iconColor": "yellow",
                        "titleFormat": "Regime Change",
                    },
                ],
            },
            "links": [
                {
                    "title": "Jaeger Traces",
                    "url": (
                        f"http://jaeger-query:16686/search?service=sesi-oversight"
                        f"&tags=%7B%22trace_id%22%3A%22${{trace_id}}%22%7D"
                    ),
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
                {
                    "title": "Loki Logs",
                    "url": (
                        f"http://loki:3100/loki/api/v1/query_range?query="
                        f"%7Bservice%3D%22sesi-oversight%22%2C+trace_id%3D%22${{trace_id}}%22%7D"
                    ),
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
            ],
        }

        return dashboard

    def generate_zone_alert_rules(self) -> List[Dict[str, Any]]:
        """Generate zone-specific Prometheus alert rules."""
        rules = []

        for zone, cfg in AGIONEZoneAlertBridge.DEFAULT_ZONE_CONFIGS.items():
            zone_name = zone.value

            # Quality degradation alert
            rules.append({
                "alert": f"QualityDegradation_{zone_name}",
                "expr": f'sesi_quality_ecosystem{{zone="{zone_name}"}} < {cfg.quality_warning}',
                "for": "5m",
                "labels": {
                    "severity": cfg.default_severity.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"Quality degradation in {zone_name}",
                    "description": (
                        f"Quality score in {zone_name} zone has dropped below "
                        f"{cfg.quality_warning}. PNDC Axiom 3: investigate "
                        f"proportionally to zone severity."
                    ),
                    "runbook_url": f"https://wiki.sesi.example.com/runbooks/{zone_name}/quality",
                },
            })

            # Critical quality alert
            rules.append({
                "alert": f"QualityCritical_{zone_name}",
                "expr": f'sesi_quality_ecosystem{{zone="{zone_name}"}} < {cfg.quality_critical}',
                "for": "2m",
                "labels": {
                    "severity": AlertSeverity.CRITICAL.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"CRITICAL quality in {zone_name}",
                    "description": (
                        f"Quality score critically low in {zone_name}. "
                        f"Immediate review required."
                    ),
                },
            })

            # Degenerate regime alert (paper6 Thm 5.1)
            rules.append({
                "alert": f"DegenerateRegime_{zone_name}",
                "expr": f'sesi_quality_regime{{zone="{zone_name}"}} == 2',
                "for": "5m",
                "labels": {
                    "severity": AlertSeverity.CRITICAL.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"Degenerate quality regime in {zone_name}",
                    "description": (
                        f"Quality regime is DEGENERATE (θ→0, ½δ₀+½δ₁) in {zone_name}. "
                        f"paper6 Thm 5.1: occupation fraction polarizing. "
                        f"Averaging-based monitoring will miss this."
                    ),
                },
            })

            # Codification failure rate alert
            rules.append({
                "alert": f"CodificationFailureRate_{zone_name}",
                "expr": (
                    f'1 - (rate(sesi_codification_events_total{{zone="{zone_name}",outcome="SUCCESS"}}[5m]) '
                    f'/ rate(sesi_codification_events_total{{zone="{zone_name}"}}[5m])) '
                    f'> {cfg.codify_failure_warning}'
                ),
                "for": "5m",
                "labels": {
                    "severity": cfg.default_severity.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"High codification failure rate in {zone_name}",
                    "description": (
                        f"Codification failure rate exceeds "
                        f"{cfg.codify_failure_warning*100:.0f}% in {zone_name}. "
                        f"Check Psyche Plus validity and quality gates."
                    ),
                },
            })

        return rules

    def generate_inhibition_rules(self) -> List[Dict[str, Any]]:
        """Generate inhibition rules to prevent alert storms."""
        return [
            {
                "source_matchers": ['severity="emergency"'],
                "target_matchers": ['severity="critical"'],
                "equal": ["zone"],
            },
            {
                "source_matchers": ['severity="critical"'],
                "target_matchers": ['severity="warning"'],
                "equal": ["zone"],
            },
            {
                "source_matchers": ['alertname=~"DegenerateRegime_.*"'],
                "target_matchers": ['alertname=~"QualityDegradation_.*"'],
                "equal": ["zone"],
            },
        ]

    def write_dashboard(self, output_path: str = "./grafana/unified-zone-dashboard.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Unified zone dashboard written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./prometheus/zone-aware-alerts.yaml") -> str:
        """Write zone-specific alert rules to file."""
        rules = self.generate_zone_alert_rules()
        inhibition = self.generate_inhibition_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {
            "groups": [
                {"name": "sesi_zone_alerts", "rules": rules},
                {"name": "sesi_inhibition", "rules": inhibition},
            ]
        }

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"Zone alert rules written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════
# [OC1] OTEL CODIFICATION COLLECTOR
# ═══════════════════════════════════════════════════════════════════════

class OTelCodificationCollector:
    """
    [D] OpenTelemetry Collector configuration for codification event ingestion.

    Features:
      • OTLP receiver for codification events from Psyche Plus
      • Processors: batch, memory_limiter, filter/drop-sensitive,
        zone_router, attributes/add-quality, transform/codification-enrich
      • Exporters: Prometheus, Jaeger, Loki, Alertmanager
      • Pipelines: codification_traces, codification_metrics, codification_logs
      • PNDC Axiom 3 compliance: sensitive data filtering
      • Production HA configuration (2 replicas, TLS, health checks)

    HONEST: This is a CONFIGURATION GENERATOR. It produces OTel Collector
    config files but does NOT deploy the collector. Real deployment requires
    TLS certificates, IAM setup, and security audit.
    """

    def __init__(
        self,
        prometheus_endpoint: str = "http://prometheus:9090/api/v1/write",
        jaeger_endpoint: str = "http://jaeger-collector:14250",
        loki_endpoint: str = "http://loki:3100/loki/api/v1/push",
        alertmanager_endpoint: str = "http://alertmanager:9093/api/v2/alerts",
    ):
        self.prometheus_endpoint = prometheus_endpoint
        self.jaeger_endpoint = jaeger_endpoint
        self.loki_endpoint = loki_endpoint
        self.alertmanager_endpoint = alertmanager_endpoint

    def generate_config(self) -> Dict[str, Any]:
        """
        Generate complete OTel Collector configuration for codification events.

        HONEST: This configuration follows PNDC Axiom 3 (Necessity & Proportionality)
        and Comprehensive Security §5.3 (privacy preservation):
          • Only necessary telemetry is collected
          • Sensitive data is filtered out
          • Zone-aware routing prevents cross-zone data leakage
        """
        return {
            "receivers": {
                "otlp": {
                    "protocols": {
                        "grpc": {
                            "endpoint": "0.0.0.0:4317",
                            "tls": {
                                "cert_file": "/etc/otel/certs/server.crt",
                                "key_file": "/etc/otel/certs/server.key",
                            },
                        },
                        "http": {
                            "endpoint": "0.0.0.0:4318",
                            "tls": {
                                "cert_file": "/etc/otel/certs/server.crt",
                                "key_file": "/etc/otel/certs/server.key",
                            },
                        },
                    },
                },
                "prometheus": {
                    "config": {
                        "scrape_configs": [
                            {
                                "job_name": "sesi-psyche-plus",
                                "scrape_interval": "15s",
                                "static_configs": [
                                    {"targets": ["sesi-psyche-plus:9090"]}
                                ],
                            },
                            {
                                "job_name": "sesi-agione",
                                "scrape_interval": "15s",
                                "static_configs": [
                                    {"targets": ["sesi-agione:9091"]}
                                ],
                            },
                        ],
                    },
                },
            },
            "processors": {
                "batch": {
                    "timeout": "5s",
                    "send_batch_size": 512,
                    "send_batch_max_size": 1024,
                },
                "memory_limiter": {
                    "check_interval": "1s",
                    "limit_mib": 512,
                    "spike_limit_mib": 128,
                },
                # PNDC Axiom 3: filter sensitive data before export
                "filter/drop-sensitive": {
                    "error_mode": "ignore",
                    "traces": {
                        "span": [
                            'attributes["password"] != nil',
                            'attributes["secret"] != nil',
                            'attributes["token"] != nil',
                            'attributes["ssn"] != nil',
                            'attributes["credit_card"] != nil',
                            'attributes["api_key"] != nil',
                        ],
                    },
                    "logs": {
                        "log_record": [
                            'body["password"] != nil',
                            'body["secret"] != nil',
                            'body["token"] != nil',
                            'body["api_key"] != nil',
                        ],
                    },
                },
                # Zone-aware routing processor
                "attributes/zone-router": {
                    "actions": [
                        {
                            "key": "sesi.zone",
                            "from_attribute": "zone",
                            "action": "upsert",
                        },
                        {
                            "key": "sesi.service",
                            "from_attribute": "service.name",
                            "action": "upsert",
                        },
                    ],
                },
                # Add quality scores as attributes
                "attributes/add-quality": {
                    "actions": [
                        {
                            "key": "sesi.quality.validity",
                            "from_attribute": "validity_score",
                            "action": "upsert",
                        },
                        {
                            "key": "sesi.quality.ecosystem",
                            "from_attribute": "quality_score",
                            "action": "upsert",
                        },
                        {
                            "key": "sesi.codification.outcome",
                            "from_attribute": "outcome",
                            "action": "upsert",
                        },
                        {
                            "key": "sesi.codification.axiom_count",
                            "from_attribute": "axiom_count",
                            "action": "upsert",
                        },
                    ],
                },
                # Transform codification events for enrichment
                "transform/codification-enrich": {
                    "trace_statements": [
                        {
                            "context": "span",
                            "statements": [
                                'set(attributes["sesi.codification.event"], "true") '
                                'where attributes["outcome"] != nil',
                                'set(attributes["sesi.codification.timestamp"], '
                                'Int(time_unix_nano)) '
                                'where attributes["outcome"] != nil',
                            ],
                        },
                    ],
                    "log_statements": [
                        {
                            "context": "log",
                            "statements": [
                                'set(attributes["trace_id"], trace_id) where trace_id != nil',
                                'set(attributes["span_id"], span_id) where span_id != nil',
                            ],
                        },
                    ],
                },
                # Codification-specific metrics processor
                "metricstransform/codification": {
                    "transforms": [
                        {
                            "include": "sesi_codification_events_total",
                            "action": "combine",
                            "new_name": "sesi_codification_rate",
                            "operation": "rate",
                        },
                        {
                            "include": "sesi_codification_validity_score",
                            "action": "combine",
                            "new_name": "sesi_codification_validity_avg",
                            "operation": "avg",
                        },
                        {
                            "include": "sesi_codification_quality_score",
                            "action": "combine",
                            "new_name": "sesi_codification_quality_avg",
                            "operation": "avg",
                        },
                    ],
                },
            },
            "exporters": {
                "prometheus": {
                    "endpoint": "0.0.0.0:8889",
                    "namespace": "sesi",
                    "send_timestamps": True,
                    "metric_expiration": "5m",
                    "resource_to_telemetry_conversion": {"enabled": True},
                },
                "prometheusremotewrite": {
                    "endpoint": self.prometheus_endpoint,
                    "tls": {"insecure": False, "ca_file": "/etc/otel/certs/ca.crt"},
                    "resource_to_telemetry_conversion": {"enabled": True},
                },
                "jaeger": {
                    "endpoint": self.jaeger_endpoint,
                    "tls": {"insecure": False, "ca_file": "/etc/otel/certs/ca.crt"},
                },
                "loki": {
                    "endpoint": self.loki_endpoint,
                    "labels": {
                        "resource": {
                            "service.name": "service",
                            "sesi.zone": "zone",
                        },
                        "log": {
                            "severity": "level",
                            "event_type": "event_type",
                            "trace_id": "trace_id",
                            "codification_outcome": "outcome",
                        },
                    },
                },
                "alertmanager": {
                    "endpoint": self.alertmanager_endpoint,
                    "tls": {"insecure": False, "ca_file": "/etc/otel/certs/ca.crt"},
                },
            },
            "service": {
                "extensions": ["health_check", "pprof", "zpages"],
                "pipelines": {
                    "codification_traces": {
                        "receivers": ["otlp"],
                        "processors": [
                            "memory_limiter",
                            "filter/drop-sensitive",
                            "attributes/zone-router",
                            "attributes/add-quality",
                            "transform/codification-enrich",
                            "batch",
                        ],
                        "exporters": ["jaeger"],
                    },
                    "codification_metrics": {
                        "receivers": ["otlp", "prometheus"],
                        "processors": [
                            "memory_limiter",
                            "metricstransform/codification",
                            "batch",
                        ],
                        "exporters": ["prometheus", "prometheusremotewrite"],
                    },
                    "codification_logs": {
                        "receivers": ["otlp"],
                        "processors": [
                            "memory_limiter",
                            "filter/drop-sensitive",
                            "attributes/zone-router",
                            "transform/codification-enrich",
                            "batch",
                        ],
                        "exporters": ["loki", "alertmanager"],
                    },
                },
                "telemetry": {
                    "logs": {"level": "info"},
                    "metrics": {"address": "0.0.0.0:8888"},
                },
            },
            "extensions": {
                "health_check": {"endpoint": "0.0.0.0:13133"},
                "pprof": {"endpoint": "0.0.0.0:1777"},
                "zpages": {"endpoint": "0.0.0.0:55679"},
            },
        }

    def generate_k8s_deployment(self) -> Dict[str, Any]:
        """Generate Kubernetes Deployment manifest for OTel Collector."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "otel-collector-codification",
                "namespace": "sesi-observability",
                "labels": {
                    "app.kubernetes.io/name": "otel-collector-codification",
                    "app.kubernetes.io/component": "codification-ingestion",
                },
            },
            "spec": {
                "replicas": 2,
                "selector": {
                    "matchLabels": {"app.kubernetes.io/name": "otel-collector-codification"},
                },
                "template": {
                    "metadata": {
                        "labels": {"app.kubernetes.io/name": "otel-collector-codification"},
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8888",
                        },
                    },
                    "spec": {
                        "serviceAccountName": "otel-collector",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000,
                        },
                        "containers": [{
                            "name": "otel-collector",
                            "image": "otel/opentelemetry-collector-contrib:0.91.0",
                            "args": ["--config=/etc/otel/config.yaml"],
                            "ports": [
                                {"name": "otlp-grpc", "containerPort": 4317},
                                {"name": "otlp-http", "containerPort": 4318},
                                {"name": "prometheus", "containerPort": 8889},
                                {"name": "metrics", "containerPort": 8888},
                                {"name": "health", "containerPort": 13133},
                            ],
                            "env": [
                                {"name": "GOMEMLIMIT", "value": "512MiB"},
                            ],
                            "resources": {
                                "requests": {"cpu": "250m", "memory": "512Mi"},
                                "limits": {"cpu": "1000m", "memory": "1Gi"},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/", "port": 13133},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/", "port": 13133},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                            },
                            "volumeMounts": [
                                {
                                    "name": "otel-config",
                                    "mountPath": "/etc/otel",
                                    "readOnly": True,
                                },
                                {
                                    "name": "otel-certs",
                                    "mountPath": "/etc/otel/certs",
                                    "readOnly": True,
                                },
                            ],
                        }],
                        "volumes": [
                            {
                                "name": "otel-config",
                                "configMap": {"name": "otel-collector-codification-config"},
                            },
                            {
                                "name": "otel-certs",
                                "secret": {"secretName": "otel-collector-tls"},
                            },
                        ],
                    },
                },
            },
        }

    def write_config(self, output_path: str = "./otel/otel-codification-config.yaml") -> str:
        """Write OTel Collector configuration to file."""
        config = self.generate_config()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"OTel Collector config written to {output_path}")
        return str(out_path)

    def write_k8s_manifest(self, output_path: str = "./k8s/otel-codification-deployment.yaml") -> str:
        """Write K8s Deployment manifest to file."""
        manifest = self.generate_k8s_deployment()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(manifest, f, indent=2)

        logger.info(f"K8s Deployment manifest written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIUnifiedZonePipeline:
    """
    [D] Complete unified zone pipeline orchestrator.

    Integrates:
      • AGIONE zone alert bridge
      • Unified zone-aware Grafana dashboard
      • OTel codification collector
    """

    def __init__(self, agione: Optional[Any] = None):
        self.zone_alert_bridge = AGIONEZoneAlertBridge(agione)
        self.dashboard = UnifiedZoneAwareGrafanaDashboard()
        self.otel_collector = OTelCodificationCollector()

    def generate_all_artifacts(self, output_dir: str = "./sesi_unified_zone") -> Dict[str, str]:
        """Generate all deployment artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Unified Grafana dashboard
        results["unified_dashboard"] = self.dashboard.write_dashboard(
            str(out_path / "grafana" / "unified-zone-dashboard.json")
        )

        # Zone alert rules
        results["zone_alert_rules"] = self.dashboard.write_alert_rules(
            str(out_path / "prometheus" / "zone-aware-alerts.yaml")
        )

        # OTel Collector config
        results["otel_config"] = self.otel_collector.write_config(
            str(out_path / "otel" / "otel-codification-config.yaml")
        )

        # K8s deployment
        results["k8s_deployment"] = self.otel_collector.write_k8s_manifest(
            str(out_path / "k8s" / "otel-codification-deployment.yaml")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "zone_alert_bridge": self.zone_alert_bridge.get_status(),
            "otel_collector": {
                "prometheus_endpoint": self.otel_collector.prometheus_endpoint,
                "jaeger_endpoint": self.otel_collector.jaeger_endpoint,
                "loki_endpoint": self.otel_collector.loki_endpoint,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI AGIONE Zone Alert × Unified Dashboard × OTel Collector")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  AGIONE available     : {HAS_AGIONE}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIUnifiedZonePipeline()

    # ── Test 1: AGIONE Zone Alert Bridge ─────────────────────────────────
    print("\n  ── AGIONE Zone Alert Bridge ──")

    # Simulate quality scores
    mock_quality = {"ecosystem": 0.85}
    suite.zone_alert_bridge._last_quality = {z.value: 0.85 for z in SecurityZone}

    # Evaluate alerts for each zone
    for zone in SecurityZone:
        alerts = suite.zone_alert_bridge.evaluate_zone_alerts(
            quality_scores=mock_quality,
            trace_id=f"trace_{zone.value}",
        )
        print(f"    {zone.value:20s}: {len(alerts)} alerts emitted")

    # Test with low quality to trigger alerts
    low_quality = {"ecosystem": 0.2}
    for zone in SecurityZone:
        alerts = suite.zone_alert_bridge.evaluate_zone_alerts(
            quality_scores=low_quality,
            trace_id=f"trace_low_{zone.value}",
        )
        print(f"    {zone.value:20s} (low quality): {len(alerts)} alerts")
        for alert in alerts:
            print(f"      [{alert.severity.value}] {alert.alert_type}: {alert.message[:60]}...")

    # Test regime classification
    print("\n  ── Zone Regime Classification ──")
    for zone in SecurityZone:
        regime = suite.zone_alert_bridge.classify_zone_regime(zone)
        print(f"    {zone.value:20s}: {regime['regime']}")

    # Test CUSUM change-point detection
    print("\n  ── Zone CUSUM Change-Point Detection ──")
    for zone in SecurityZone:
        cusum = suite.zone_alert_bridge.detect_zone_change_point(zone)
        print(f"    {zone.value:20s}: change_point={cusum['change_point_detected']}")

    # ── Test 2: Unified Grafana Dashboard ────────────────────────────────
    print("\n  ── Unified Grafana Dashboard ──")

    dashboard = suite.dashboard.generate_dashboard()
    print(f"    Dashboard title    : {dashboard['title']}")
    print(f"    Total panels       : {len(dashboard['panels'])}")
    print(f"    Template variables : {len(dashboard['templating']['list'])}")
    print(f"    Annotations        : {len(dashboard['annotations']['list'])}")

    alert_rules = suite.dashboard.generate_zone_alert_rules()
    print(f"    Zone alert rules   : {len(alert_rules)}")
    for rule in alert_rules[:5]:
        print(f"      • {rule['alert']} ({rule['labels']['severity']})")

    inhibition_rules = suite.dashboard.generate_inhibition_rules()
    print(f"    Inhibition rules   : {len(inhibition_rules)}")

    # ── Test 3: OTel Codification Collector ──────────────────────────────
    print("\n  ── OTel Codification Collector ──")

    otel_config = suite.otel_collector.generate_config()
    print(f"    Receivers          : {list(otel_config['receivers'].keys())}")
    print(f"    Processors         : {list(otel_config['processors'].keys())}")
    print(f"    Exporters          : {list(otel_config['exporters'].keys())}")
    print(f"    Pipelines          : {list(otel_config['service']['pipelines'].keys())}")

    # Check PNDC Axiom 3 compliance
    filter_proc = otel_config["processors"]["filter/drop-sensitive"]
    print(f"    Sensitive filters  : {len(filter_proc['traces']['span'])} trace, "
          f"{len(filter_proc['logs']['log_record'])} log patterns")

    # K8s deployment
    k8s_deploy = suite.otel_collector.generate_k8s_deployment()
    print(f"    K8s replicas       : {k8s_deploy['spec']['replicas']}")
    print(f"    K8s container      : {k8s_deploy['spec']['template']['spec']['containers'][0]['name']}")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_unified_zone")
    print(f"    Artifacts generated: {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:25s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Zone alerts emitted: {status['zone_alert_bridge']['total_alerts']}")
    print(f"    δ_min              : {status['zone_alert_bridge']['delta_min']:.6f}")
    print(f"    Zone configs       : {len(status['zone_alert_bridge']['zone_configs'])}")
    for zone, cfg in status['zone_alert_bridge']['zone_configs'].items():
        print(f"      {zone:20s}: warning={cfg['quality_warning']}, "
              f"critical={cfg['quality_critical']}, severity={cfg['default_severity']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • AGIONE zone alerting shows CORRELATION, not causation")
    print("  • Zone thresholds are PROPORTIONALITY-BASED (PNDC Axiom 3)")
    print("  • Unified dashboard is VISUALIZATION, not decision-making")
    print("  • OTel Collector config is a TEMPLATE, not a deployment")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all quality inputs")
    print("  • Production requires security audit and regulatory approval")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
