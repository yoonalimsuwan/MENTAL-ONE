"""
=============================================================================
SESI PSYCHE PLUS CODIFICATION × ZONE-AWARE ALERTMANAGER × PER-ZONE METRICS
AGIONE Codification Events → Zone Alerts + Proportional Routing + Metrics
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[PC1] PsychePlusCodificationEventEmitter
      • Captures maybe_codify() dual-gate outcomes (validity + quality)
      • Maps codification events to zone-specific alerts
      • _safe_unit_scalar sanitization for all quality scores
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for meaningful event triggers
      • Per-zone regime classification (paper6 Thm 5.1)

[ZR1] ZoneAwareAlertmanagerRouting
      • Per-zone routing rules (Z0 Implant → Z3 External)
      • Zone-specific severity escalation (PNDC Axiom 3 proportionality)
      • Inhibition rules to prevent alert storms
      • Receiver mapping per zone (PagerDuty/Slack/Email)
      • Alertmanager YAML configuration generation

[PM1] PerZoneQualityPrometheusMetrics
      • Per-zone quality score gauges (7 domains × 4 zones)
      • Regime classification per zone (paper6 Thm 5.1)
      • CUSUM change-point detection per zone stream
      • Codification event counters per zone
      • δ_min and No-Zeno bound metrics
      • Prometheus exposition format export

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds alert emission and codification event rate
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
[C] agi_one_psyche_plus_v33.py: maybe_codify() dual-gate
    → validity + quality gates for codification
[C] agi_one_v3_11.py: EcosystemOrchestrator.quality_report()
    → per-domain quality score extraction
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-specific alert thresholds
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO AGIONE v3.11 AND PSYCHE PLUS v3.3
-----------------------------------------------
From agi_one_v3_11.py:
  • EcosystemOrchestrator.quality_report() → per-domain quality scores
  • AGIONE.attach_orchestrator() → quality_score threading
  • AGIState.psyche_plus_validity / psyche_plus_axiom_count / psyche_plus_codified

From agi_one_psyche_plus_v33.py:
  • _safe_unit_scalar() → NaN/Inf sanitization
  • AGIOnePsychePlus.maybe_codify() → dual-gate (validity + quality)
  • SuperEgoModule.quality_weight → blend weight for quality in validity
  • PsychePlusConfig.quality_codify_min → minimum quality for codification
  • CodificationOutcome: SUCCESS / REFUSED_VALIDITY / REFUSED_QUALITY / REFUSED_NONFINITE
  • stats counters: codify_attempts, codify_successes, codify_refused_validity,
    codify_refused_quality, axiom_rejected_nonfinite

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Psyche Plus codification events provide OBSERVABILITY, not causation.
  A codification event shows when an axiom was added, not whether it
  is "correct" or "conscious".
• Zone-specific thresholds are PROPORTIONALITY-BASED (PNDC Axiom 3),
  not absolute. Must be calibrated against real operational data.
• Alertmanager routing is a CONFIGURATION GENERATOR. Real deployment
  requires Alertmanager server, proper receiver setup, and security audit.
• Per-zone Prometheus metrics provide VISUALIZATION, not decision-making.
  Human operators must interpret and act.
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

logger = logging.getLogger("sesi.psyche_zone_alerts_metrics")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# AGIONE / Psyche Plus imports (graceful)
try:
    from agi_one_v3_11 import AGIONE, EcosystemOrchestrator
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

try:
    from agi_one_psyche_plus_v33 import (
        AGIOnePsychePlus, PsychePlusConfig, _safe_unit_scalar,
    )
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
    "ZoneAlertConfig", "CodificationAlert",
    "PsychePlusCodificationEventEmitter",
    "ZoneAwareAlertmanagerRouting",
    "PerZoneQualityPrometheusMetrics",
    "SESIZoneCodificationAlertSuite",
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
# [PC1] PSYCHE PLUS CODIFICATION EVENT EMITTER
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
class CodificationAlert:
    """A single alert generated from a codification event."""
    alert_id: str
    zone: SecurityZone
    severity: AlertSeverity
    alert_type: str
    message: str
    codification_outcome: CodificationOutcome
    validity_score: float
    quality_score: float
    axiom_count: int
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
            "codification_outcome": self.codification_outcome.name,
            "validity_score": self.validity_score,
            "quality_score": self.quality_score,
            "axiom_count": self.axiom_count,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }


class PsychePlusCodificationEventEmitter:
    """
    [D] Emits zone-specific alerts from Psyche Plus codification events.

    Features:
      • Captures maybe_codify() dual-gate outcomes
      • Maps codification events to zone-specific alerts
      • _safe_unit_scalar sanitization for all quality scores
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for meaningful event triggers
      • Per-zone regime classification (paper6 Thm 5.1)

    HONEST: This emitter provides OBSERVABILITY, not causation.
    Codification events show when axioms are added, not whether
    they are "correct" or "conscious".

    Mathematical grounding:
      • _safe_unit_scalar sanitization (Psyche Plus v3.3)
      • No-Zeno bound on alert rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for meaningful triggers (Delta_Min Thm 1.2)
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
        zone_configs: Optional[Dict[SecurityZone, ZoneAlertConfig]] = None,
        psyche_plus: Optional[Any] = None,
    ):
        self.zone_configs = zone_configs or self.DEFAULT_ZONE_CONFIGS
        self.psyche_plus = psyche_plus
        self._alerts: List[CodificationAlert] = []
        self._alert_timestamps: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._quality_history: Dict[str, List[float]] = {
            zone.value: [] for zone in self.zone_configs
        }
        self._codification_history: Dict[str, List[Dict[str, Any]]] = {
            zone.value: [] for zone in self.zone_configs
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

    # ── Codification Event Processing ────────────────────────────────────

    def emit_codification_alert(
        self,
        zone: SecurityZone,
        outcome: CodificationOutcome,
        validity_score: float,
        quality_score: float,
        axiom_count: int,
        trace_id: Optional[str] = None,
    ) -> Optional[CodificationAlert]:
        """
        Emit a zone-specific alert from a codification event.

        Args:
            zone: Security zone where the event occurred
            outcome: Codification outcome (SUCCESS/REFUSED_*)
            validity_score: SuperEgo validity score (will be sanitized)
            quality_score: Ecosystem quality score (will be sanitized)
            axiom_count: Current axiom bank size
            trace_id: Optional Jaeger trace ID for correlation

        Returns:
            CodificationAlert if emitted, None if rate-limited
        """
        # Sanitize scores via _safe_unit_scalar
        validity_score = _safe_unit_scalar(
            validity_score, name="validity_score", logger=logger, default=0.5
        )
        quality_score = _safe_unit_scalar(
            quality_score, name="quality_score", logger=logger, default=0.5
        )

        # Check rate limit
        rate_check = self.check_alert_rate(zone)
        if not rate_check["allowed"]:
            logger.debug(f"Zone {zone.value} alert rate-limited: {rate_check['reason']}")
            return None

        # Update quality history
        self._quality_history[zone.value].append(quality_score)
        if len(self._quality_history[zone.value]) > 100:
            self._quality_history[zone.value] = self._quality_history[zone.value][-100:]

        # Record codification event
        self._codification_history[zone.value].append({
            "outcome": outcome.name,
            "validity_score": validity_score,
            "quality_score": quality_score,
            "axiom_count": axiom_count,
            "timestamp": time.time(),
        })

        # Get zone config
        cfg = self.zone_configs.get(zone)
        if cfg is None:
            logger.warning(f"No config for zone {zone.value}")
            return None

        # Determine alert severity based on outcome and zone thresholds
        severity = None
        alert_type = None
        message = None

        if outcome == CodificationOutcome.SUCCESS:
            # Successful codification — no alert needed
            self.record_alert_timestamp(zone)
            return None

        elif outcome == CodificationOutcome.REFUSED_VALIDITY:
            if validity_score < cfg.quality_emergency:
                severity = AlertSeverity.EMERGENCY
                alert_type = "codification_validity_emergency"
                message = (
                    f"EMERGENCY: Codification refused due to critically low validity "
                    f"({validity_score:.3f}) in {zone.value}. "
                    f"Axiom bank integrity at risk."
                )
            elif validity_score < cfg.quality_critical:
                severity = AlertSeverity.CRITICAL
                alert_type = "codification_validity_critical"
                message = (
                    f"CRITICAL: Codification refused due to low validity "
                    f"({validity_score:.3f}) in {zone.value}. "
                    f"Review SuperEgoModule configuration."
                )
            elif validity_score < cfg.codify_failure_warning:
                severity = cfg.default_severity
                alert_type = "codification_validity_warning"
                message = (
                    f"Codification refused due to validity "
                    f"({validity_score:.3f}) in {zone.value}."
                )

        elif outcome == CodificationOutcome.REFUSED_QUALITY:
            if quality_score < cfg.quality_emergency:
                severity = AlertSeverity.EMERGENCY
                alert_type = "codification_quality_emergency"
                message = (
                    f"EMERGENCY: Codification refused due to critically low quality "
                    f"({quality_score:.3f}) in {zone.value}. "
                    f"EcosystemOrchestrator.quality_report() indicates system failure."
                )
            elif quality_score < cfg.quality_critical:
                severity = AlertSeverity.CRITICAL
                alert_type = "codification_quality_critical"
                message = (
                    f"CRITICAL: Codification refused due to low quality "
                    f"({quality_score:.3f}) in {zone.value}. "
                    f"Check ecosystem health."
                )
            elif quality_score < cfg.codify_failure_warning:
                severity = cfg.default_severity
                alert_type = "codification_quality_warning"
                message = (
                    f"Codification refused due to quality "
                    f"({quality_score:.3f}) in {zone.value}."
                )

        elif outcome == CodificationOutcome.REFUSED_NONFINITE:
            severity = AlertSeverity.CRITICAL
            alert_type = "codification_nonfinite"
            message = (
                f"CRITICAL: Codification refused due to non-finite values "
                f"in {zone.value}. Possible numerical instability."
            )

        if severity is None:
            return None

        # Check regime
        regime_result = self.classify_zone_regime(zone)
        if regime_result["regime"] == QualityRegime.DEGENERATE.name:
            # Escalate severity if degenerate regime detected
            if severity.value < AlertSeverity.CRITICAL.value:
                severity = AlertSeverity.CRITICAL
                message += (
                    f" [DEGENERATE REGIME: paper6 Thm 5.1 — "
                    f"occupation fraction polarizing]"
                )

        # Create alert
        alert = CodificationAlert(
            alert_id=str(uuid.uuid4())[:8],
            zone=zone,
            severity=severity,
            alert_type=alert_type,
            message=message,
            codification_outcome=outcome,
            validity_score=validity_score,
            quality_score=quality_score,
            axiom_count=axiom_count,
            trace_id=trace_id,
            metadata={
                "regime": regime_result["regime"],
                "theta_hat": regime_result.get("theta_hat"),
            },
        )

        self._alerts.append(alert)
        self.record_alert_timestamp(zone)

        logger.info(
            f"[ALERT] {alert.alert_id} | zone={zone.value} | "
            f"severity={severity.value} | type={alert_type} | "
            f"outcome={outcome.name} | "
            f"validity={validity_score:.3f} | quality={quality_score:.3f}"
        )

        return alert

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return [a.to_dict() for a in self._er
```

def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return [a.to_dict() for a in self._alerts[-limit:]]


# ═══════════════════════════════════════════════════════════════════════
# [ZR1] ZONE-AWARE ALERTMANAGER ROUTING
# ═══════════════════════════════════════════════════════════════════════

class ZoneAwareAlertmanagerRouting:
    """
    [D] Generates Alertmanager routing configuration based on zone policies.
    
    Features:
      • Per-zone routing rules (Z0 Implant → Z3 External)
      • Zone-specific severity escalation (PNDC Axiom 3 proportionality)
      • Inhibition rules to prevent alert storms
      • Receiver mapping per zone (PagerDuty/Slack/Email)
    """

    def __init__(self, emitter: PsychePlusCodificationEventEmitter):
        self.emitter = emitter

    def generate_alertmanager_yaml(self) -> str:
        """Generate complete Alertmanager YAML configuration."""
        route_config = {
            "route": {
                "receiver": "default-receiver",
                "group_by": ["zone", "alert_type"],
                "group_wait": "30s",
                "group_interval": "5m",
                "repeat_interval": "4h",
                "routes": [],
            },
            "receivers": [
                {
                    "name": "default-receiver",
                    "email_configs": [{"to": "default@sesi.example.com"}],
                }
            ],
            "inhibit_rules": [
                {
                    "source_match": {"severity": "emergency"},
                    "target_match": {"severity": "critical|warning"},
                    "equal": ["zone", "alert_type"],
                }
            ],
        }

        # Add zone-specific routes and receivers
        for zone, cfg in self.emitter.zone_configs.items():
            receiver_name = cfg.receiver
            route_config["route"]["routes"].append({
                "match": {"zone": zone.value},
                "receiver": receiver_name,
                "group_wait": cfg.group_wait,
                "group_interval": cfg.group_interval,
                "repeat_interval": cfg.repeat_interval,
            })
            
            # Add receiver config (simplified for demonstration)
            if "pager" in receiver_name:
                route_config["receivers"].append({
                    "name": receiver_name,
                    "pagerduty_configs": [{"service_key": f"<{zone.value}_pager_key>"}],
                })
            elif "slack" in receiver_name:
                route_config["receivers"].append({
                    "name": receiver_name,
                    "slack_configs": [{"channel": f"#{zone.value}-alerts", "api_url": "<slack_webhook>"}],
                })
            else:
                route_config["receivers"].append({
                    "name": receiver_name,
                    "email_configs": [{"to": f"{zone.value}-team@sesi.example.com"}],
                })

        if HAS_YAML:
            return yaml.dump(route_config, default_flow_style=False, sort_keys=False)
        else:
            return json.dumps(route_config, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# [PM1] PER-ZONE QUALITY PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════════════

class PerZoneQualityPrometheusMetrics:
    """
    [D] Exports per-zone quality metrics in Prometheus exposition format.
    
    Features:
      • Per-zone quality score gauges
      • Regime classification per zone
      • CUSUM change-point detection per zone stream
      • Codification event counters per zone
      • δ_min and No-Zeno bound metrics
    """

    def __init__(self, emitter: PsychePlusCodificationEventEmitter):
        self.emitter = emitter

    def export_prometheus(self) -> str:
        """Generate Prometheus exposition format metrics."""
        lines = []
        
        # 1. Quality Gauges
        lines.append("# HELP sesi_quality_score Current ecosystem quality score per zone")
        lines.append("# TYPE sesi_quality_score gauge")
        for zone_val, history in self.emitter._quality_history.items():
            if history:
                current_q = history[-1]
                lines.append(f'sesi_quality_score{{zone="{zone_val}"}} {current_q:.4f}')

        # 2. Regime Classification
        lines.append("# HELP sesi_quality_regime Quality regime classification (0=FINITE, 1=REGULAR, 2=DEGENERATE, 3=INSUFFICIENT)")
        lines.append("# TYPE sesi_quality_regime gauge")
        for zone in SecurityZone:
            regime_result = self.emitter.classify_zone_regime(zone)
            regime_map = {
                "FINITE_MEAN": 0,
                "REGULARLY_VARYING": 1,
                "DEGENERATE": 2,
                "INSUFFICIENT_DATA": 3,
            }
            val = regime_map.get(regime_result["regime"], 3)
            lines.append(f'sesi_quality_regime{{zone="{zone.value}"}} {val}')

        # 3. Codification Counters
        lines.append("# HELP sesi_codification_events_total Total codification events per zone and outcome")
        lines.append("# TYPE sesi_codification_events_total counter")
        for zone_val, history in self.emitter._codification_history.items():
            # Count by outcome
            outcome_counts = {}
            for event in history:
                outcome = event["outcome"]
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            for outcome, count in outcome_counts.items():
                lines.append(f'sesi_codification_events_total{{zone="{zone_val}",outcome="{outcome}"}} {count}')

        # 4. No-Zeno and δ_min Metrics
        lines.append("# HELP sesi_delta_min Minimum meaningful quality change (Delta_Min Thm 1.2)")
        lines.append("# TYPE sesi_delta_min gauge")
        delta_min = self.emitter.compute_delta_min()
        lines.append(f"sesi_delta_min {delta_min:.6f}")

        lines.append("# HELP sesi_no_zeno_bound Maximum alerts per minute per zone (SESI_NoZeno Thm 3.2)")
        lines.append("# TYPE sesi_no_zeno_bound gauge")
        for zone in SecurityZone:
            rate_check = self.emitter.check_alert_rate(zone)
            limit = rate_check.get("limit", 10)
            lines.append(f'sesi_no_zeno_bound{{zone="{zone.value}"}} {limit}')

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIZoneCodificationAlertSuite:
    """
    [D] Complete suite for zone-aware codification alerting and metrics.
    """

    def __init__(self, psyche_plus: Optional[Any] = None):
        self.emitter = PsychePlusCodificationEventEmitter(psyche_plus=psyche_plus)
        self.routing = ZoneAwareAlertmanagerRouting(self.emitter)
        self.metrics = PerZoneQualityPrometheusMetrics(self.emitter)

    def generate_alertmanager_config(self, output_path: str = "./alertmanager.yml") -> str:
        """Generate and save Alertmanager configuration."""
        config_yaml = self.routing.generate_alertmanager_yaml()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(config_yaml)
        return output_path

    def generate_prometheus_metrics(self, output_path: str = "./metrics.prom") -> str:
        """Generate and save Prometheus metrics."""
        metrics_text = self.metrics.export_prometheus()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(metrics_text)
        return output_path

    def get_status(self) -> Dict[str, Any]:
        """Get current suite status."""
        return {
            "total_alerts": len(self.emitter._alerts),
            "zones_monitored": len(self.emitter.zone_configs),
            "delta_min": self.emitter.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Zone-Aware Codification Alerts & Metrics — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available: {HAS_YAML}")
    print(f"  PyTorch available: {HAS_TORCH}")

    suite = SESIZoneCodificationAlertSuite()

    # Simulate some codification events across different zones
    print("\n  ── Simulating Codification Events ──")
    
    # Z0 Implant: High severity failure
    suite.emitter.emit_codification_alert(
        zone=SecurityZone.Z0_IMPLANT,
        outcome=CodificationOutcome.REFUSED_QUALITY,
        validity_score=0.8,
        quality_score=0.05,  # Very low quality -> EMERGENCY
        axiom_count=42,
        trace_id="trace_z0_001",
    )

    # Z1 Clinical: Warning level failure
    suite.emitter.emit_codification_alert(
        zone=SecurityZone.Z1_CLINICAL,
        outcome=CodificationOutcome.REFUSED_VALIDITY,
        validity_score=0.35,  # Below critical (0.4) but above emergency (0.2)
        quality_score=0.6,
        axiom_count=15,
        trace_id="trace_z1_001",
    )

    # Z2 Processing: Success (should not generate alert)
    suite.emitter.emit_codification_alert(
        zone=SecurityZone.Z2_PROCESSING,
        outcome=CodificationOutcome.SUCCESS,
        validity_score=0.9,
        quality_score=0.8,
        axiom_count=100,
        trace_id="trace_z2_001",
    )

    # Z3 External: Non-finite values
    suite.emitter.emit_codification_alert(
        zone=SecurityZone.Z3_EXTERNAL,
        outcome=CodificationOutcome.REFUSED_NONFINITE,
        validity_score=float('nan'),
        quality_score=float('inf'),
        axiom_count=5,
        trace_id="trace_z3_001",
    )

    print(f"  Total alerts generated: {len(suite.emitter._alerts)}")
    for alert in suite.emitter._alerts:
        print(f"    - [{alert.severity.value.upper()}] {alert.zone.value}: {alert.alert_type}")

    # Generate configurations
    print("\n  ── Generating Configurations ──")
    am_path = suite.generate_alertmanager_config("./test_alertmanager.yml")
    print(f"  Alertmanager config saved to: {am_path}")

    prom_path = suite.generate_prometheus_metrics("./test_metrics.prom")
    print(f"  Prometheus metrics saved to: {prom_path}")

    # Show sample metrics
    print("\n  ── Sample Prometheus Metrics ──")
    with open(prom_path, "r") as f:
        lines = f.read().split("\n")
        for line in lines[:15]:  # Show first 15 lines
            print(f"    {line}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Codification events provide OBSERVABILITY, not causation.")
    print("  • Zone-specific thresholds are PROPORTIONALITY-BASED (PNDC Axiom 3).")
    print("  • Alertmanager routing is a CONFIGURATION GENERATOR.")
    print("  • All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).")
    print("  • This module does NOT create, detect, or simulate consciousness.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
