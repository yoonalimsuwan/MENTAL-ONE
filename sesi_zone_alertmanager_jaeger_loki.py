"""
=============================================================================
SESI ZONE-SPECIFIC ALERTMANAGER × ZONE-AWARE JAEGER × LOKI CODIFICATION LOGS
Per-Zone Alert Routing + Topology Visualization + Axiom Codification Events
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[ZR1] ZoneSpecificAlertmanagerRouter
      • Per-zone alert routing (Z0 Implant → Z3 External)
      • Zone-specific severity escalation (PNDC Axiom 3 proportionality)
      • Inhibition rules to prevent alert storms
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for alert threshold calibration
      • Alertmanager YAML configuration generation

[JM1] ZoneAwareJaegerServiceMap
      • Zone-colored service topology graph
      • Per-edge regime classification (paper6 Thm 5.1)
      • CUSUM change-point detection on call streams (Six_Problems P4)
      • Cross-zone flow highlighting
      • Grafana nodeGraph panel generation
      • Jaeger-compatible JSON export

[CL1] LokiCodificationEventAggregator
      • Structured JSON logs for Psyche Plus codification events
      • Event types: SUCCESS, REFUSED_VALIDITY, REFUSED_QUALITY, REFUSED_NONFINITE
      • _safe_unit_scalar sanitization for all quality scores
      • Trace-log correlation via trace_id/span_id
      • LogQL query templates for codification analysis
      • Regime classification on codification streams
      • Multi-identity risk correlation (Comprehensive Security §5.2)

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
[C] paper6 Thm 4.1: Degenerate occupation-time law ½δ₀+½δ₁
[C] Six_Problems P4: CUSUM + subsampling on raw process
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar() + maybe_codify() dual-gate
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
[C] PNDC Axiom 3: Necessity and Proportionality
[C] Unified_Ordinal Hyp 7.9: Free design parameters

RELATION TO AGIONE v3.11 AND PSYCHE PLUS v3.3
-----------------------------------------------
From agi_one_v3_11.py:
  • AGIONE.psyche_plus → speculative-axiom evolution layer
  • Step 7-B: psyche_plus(ws, quality_score=...)
  • EcosystemOrchestrator.quality_report() → per-domain quality scores

From agi_one_psyche_plus_v33.py:
  • _safe_unit_scalar() → NaN/Inf sanitization
  • AGIOnePsychePlus.maybe_codify() → dual-gate (validity + quality)
  • CodificationOutcome: SUCCESS / REFUSED_VALIDITY / REFUSED_QUALITY / REFUSED_NONFINITE
  • stats counters: codify_attempts, codify_successes, codify_refused_validity,
    codify_refused_quality, axiom_rejected_nonfinite

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Alertmanager routing is a CONFIGURATION GENERATOR, not a deployed system.
• Jaeger service map is a VISUALIZATION of topology, not a security guarantee.
• Loki codification logs are STRUCTURED RECORDS, not omniscience.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• This module does NOT create, detect, or simulate consciousness.
• Production deployment requires security audit and regulatory approval.
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

logger = logging.getLogger("sesi.zone_alertmanager_jaeger_loki")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# AGIONE / Psyche Plus imports (graceful)
try:
    from agi_one_psyche_plus_v33 import _safe_unit_scalar, CodificationOutcome
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False
    class CodificationOutcome(Enum):
        SUCCESS = auto()
        REFUSED_VALIDITY = auto()
        REFUSED_QUALITY = auto()
        REFUSED_NONFINITE = auto()

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
    "SecurityZone", "AlertSeverity", "QualityRegime",
    "ZoneSpecificAlertmanagerRouter", "ZoneAlertConfig",
    "ZoneAwareJaegerServiceMap", "ServiceNode", "ServiceEdge",
    "LokiCodificationEventAggregator", "CodificationLogEntry",
    "SESIZoneObservabilitySuite",
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


# ═══════════════════════════════════════════════════════════════════════
# [ZR1] ZONE-SPECIFIC ALERTMANAGER ROUTER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ZoneAlertConfig:
    """
    Per-zone alert configuration.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be calibrated against real data.
    """
    zone: SecurityZone
    # Quality thresholds (PNDC Axiom 3 proportionality)
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


class ZoneSpecificAlertmanagerRouter:
    """
    [D] Generates zone-specific Alertmanager routing configuration.

    Features:
      • Per-zone alert rules with differentiated thresholds
      • Zone-specific severity escalation
      • Inhibition rules to prevent alert storms
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for threshold calibration
      • Alertmanager YAML generation

    HONEST: This is a CONFIGURATION GENERATOR. Real deployment requires:
      • Alertmanager server running
      • Receiver configuration (PagerDuty, Slack, Email)
      • TLS certificates
      • Security audit

    Mathematical grounding:
      • PNDC Axiom 3: Necessity and Proportionality
      • No-Zeno bound on alert rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate (Delta_Min Thm 1.2)
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
    ):
        self.zone_configs = zone_configs or self.DEFAULT_ZONE_CONFIGS
        self._alert_timestamps: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_alert_rate(self, alert_key: str) -> Dict[str, Any]:
        """Check if alert emission is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        timestamps = self._alert_timestamps[alert_key]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        delta_min = self.compute_delta_min()
        c_v = 1.0
        T = 60.0
        bound = 1 + (2 * c_v * T) / max(delta_min, 1e-8)

        if len(timestamps) >= bound:
            return {
                "allowed": False,
                "reason": "no_zeno_bound_exceeded",
                "current": len(timestamps),
                "bound": bound,
            }

        timestamps.append(now)
        return {
            "allowed": True,
            "current": len(timestamps),
            "bound": bound,
        }

    # ── Zone-Specific Alert Rules ────────────────────────────────────────

    def generate_zone_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Generate Prometheus alert rules per zone.

        HONEST: Thresholds are FREE DESIGN PARAMETERS.
        Must be calibrated against real operational data.
        """
        rules = []

        for zone, cfg in self.zone_configs.items():
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

            # Codification failure rate alert
            rules.append({
                "alert": f"CodificationFailureRate_{zone_name}",
                "expr": (
                    f'1 - (rate(sesi_codification_events_total{{zone="{zone_name}",outcome="success"}}[5m]) '
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

        return rules

    def generate_inhibition_rules(self) -> List[Dict[str, Any]]:
        """
        Generate inhibition rules to prevent alert storms.

        From PNDC Axiom 3: alerts should be necessary and proportionate.
        """
        return [
            {
                "source_matchers": ['severity="emergency"'],
                "target_matchers": ['severity="critical"'],
                "equal": ["zone", "service"],
            },
            {
                "source_matchers": ['severity="critical"'],
                "target_matchers": ['severity="warning"'],
                "equal": ["zone", "service"],
            },
            {
                "source_matchers": ['alertname=~"DegenerateRegime_.*"'],
                "target_matchers": ['alertname=~"QualityDegradation_.*"'],
                "equal": ["zone"],
            },
            {
                "source_matchers": ['alertname=~"QualityCritical_.*"'],
                "target_matchers": ['alertname=~"QualityDegradation_.*"'],
                "equal": ["zone"],
            },
        ]

    def generate_alertmanager_config(self) -> Dict[str, Any]:
        """
        Generate complete Alertmanager configuration.

        HONEST: This is a TEMPLATE. Production requires:
          • Proper receiver configuration
          • TLS certificates
          • Security audit
        """
        routes = []
        for zone, cfg in self.zone_configs.items():
            routes.append({
                "match": {"zone": zone.value},
                "receiver": cfg.receiver,
                "group_by": ["alertname", "zone"],
                "group_wait": cfg.group_wait,
                "group_interval": cfg.group_interval,
                "repeat_interval": cfg.repeat_interval,
            })

        config = {
            "global": {
                "resolve_timeout": "5m",
            },
            "route": {
                "receiver": "default-receiver",
                "group_by": ["alertname", "zone", "severity"],
                "group_wait": "30s",
                "group_interval": "5m",
                "repeat_interval": "4h",
                "routes": routes,
            },
            "receivers": [
                {
                    "name": "default-receiver",
                    "email_configs": [{"to": "platform-team@sesi.example.com"}],
                },
                {
                    "name": "clinical-pager",
                    "pagerduty_configs": [{"service_key": "${PAGERDUTY_KEY}"}],
                },
                {
                    "name": "clinical-team",
                    "slack_configs": [{
                        "api_url": "${SLACK_WEBHOOK}",
                        "channel": "#clinical-alerts",
                    }],
                },
                {
                    "name": "platform-slack",
                    "slack_configs": [{
                        "api_url": "${SLACK_WEBHOOK}",
                        "channel": "#platform-alerts",
                    }],
                },
                {
                    "name": "platform-email",
                    "email_configs": [{"to": "platform@sesi.example.com"}],
                },
            ],
            "inhibit_rules": self.generate_inhibition_rules(),
        }

        return config

    def write_config(self, output_path: str = "./alertmanager/alertmanager.yml") -> str:
        """Write Alertmanager configuration to file."""
        config = self.generate_alertmanager_config()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"Alertmanager config written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./prometheus/zone-alerts.yaml") -> str:
        """Write zone-specific alert rules to file."""
        rules = self.generate_zone_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {"groups": [{"name": "sesi_zone_alerts", "rules": rules}]}

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"Zone alert rules written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "zones_configured": len(self.zone_configs),
            "alert_rules": len(self.generate_zone_alert_rules()),
            "inhibition_rules": len(self.generate_inhibition_rules()),
            "delta_min": self.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [JM1] ZONE-AWARE JAEGER SERVICE MAP
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ServiceNode:
    """A node in the service map."""
    name: str
    zone: SecurityZone
    service_type: str
    n_calls_in: int = 0
    n_calls_out: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    regime: QualityRegime = QualityRegime.FINITE_MEAN


@dataclass
class ServiceEdge:
    """An edge in the service map."""
    source: str
    target: str
    call_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    error_rate: float = 0.0
    zone_crossing: bool = False
    regime: QualityRegime = QualityRegime.FINITE_MEAN
    latency_history: List[float] = field(default_factory=list)


@dataclass
class ServiceMapConfig:
    """Configuration for service map generation."""
    min_calls_to_display: int = 10
    latency_history_size: int = 100
    zone_colors: Dict[str, str] = field(default_factory=lambda: {
        "z0-implant": "#ff6b6b",        # red — most restricted
        "z1-clinical": "#ffd93d",        # yellow — clinician access
        "z2-processing": "#6bcf7f",      # green — AI inference
        "z3-external": "#4d96ff",        # blue — public API
    })


class ZoneAwareJaegerServiceMap:
    """
    [D] Zone-aware Jaeger service map with regime classification.

    Features:
      • Zone-colored service topology graph
      • Per-edge regime classification (paper6 Thm 5.1)
      • CUSUM change-point detection on call streams (Six_Problems P4)
      • Cross-zone flow highlighting
      • Grafana nodeGraph panel generation
      • Jaeger-compatible JSON export

    HONEST: This is a VISUALIZATION TOOL. It shows call patterns,
    not whether those calls are safe or correct.

    Mathematical grounding:
      • paper6 Thm 5.1: Growth-rate dichotomy per edge
      • Six_Problems P4: CUSUM change-point detection
    """

    def __init__(self, cfg: Optional[ServiceMapConfig] = None):
        self.cfg = cfg or ServiceMapConfig()
        self.nodes: Dict[str, ServiceNode] = {}
        self.edges: Dict[Tuple[str, str], ServiceEdge] = {}

    def register_service(
        self,
        name: str,
        zone: SecurityZone,
        service_type: str,
    ) -> None:
        """Register a service in the map."""
        if name not in self.nodes:
            self.nodes[name] = ServiceNode(
                name=name, zone=zone, service_type=service_type,
            )
            logger.info(f"Registered service: {name} (zone={zone.value})")

    def record_call(
        self,
        source: str,
        target: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a service-to-service call."""
        # Ensure nodes exist
        if source not in self.nodes:
            self.register_service(source, SecurityZone.Z2_PROCESSING, "unknown")
        if target not in self.nodes:
            self.register_service(target, SecurityZone.Z2_PROCESSING, "unknown")

        # Update edge
        edge_key = (source, target)
        if edge_key not in self.edges:
            self.edges[edge_key] = ServiceEdge(
                source=source,
                target=target,
                zone_crossing=(self.nodes[source].zone != self.nodes[target].zone),
            )

        edge = self.edges[edge_key]
        edge.call_count += 1

        # Update latency history (bounded)
        edge.latency_history.append(latency_ms)
        if len(edge.latency_history) > self.cfg.latency_history_size:
            edge.latency_history = edge.latency_history[-self.cfg.latency_history_size:]

        # Running averages
        edge.avg_latency_ms = (
            (edge.avg_latency_ms * (edge.call_count - 1) + latency_ms)
            / edge.call_count
        )

        # Percentiles from history
        if edge.latency_history:
            sorted_lat = sorted(edge.latency_history)
            edge.p50_latency_ms = sorted_lat[len(sorted_lat) // 2]
            edge.p95_latency_ms = sorted_lat[int(len(sorted_lat) * 0.95)]

        # Error rate
        if not success:
            edge.error_rate = (
                (edge.error_rate * (edge.call_count - 1) + 1.0)
                / edge.call_count
            )

        # Regime classification for this edge
        edge.regime = self._classify_edge_regime(edge)

        # Update node stats
        self.nodes[source].n_calls_out += 1
        self.nodes[target].n_calls_in += 1

    def _classify_edge_regime(self, edge: ServiceEdge) -> QualityRegime:
        """
        Classify edge regime based on latency pattern.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → ordinary monitoring
          θ ∈ (0,1):  regularly varying → distributional monitoring
          θ → 0:      degenerate → polarized monitoring
        """
        if len(edge.latency_history) < 20:
            return QualityRegime.INSUFFICIENT_DATA

        # Use Hill estimator on latency deviations
        latencies = edge.latency_history
        median = np.median(latencies)
        deviations = [abs(l - median) for l in latencies if abs(l - median) > 0.01]

        if len(deviations) < 10:
            return QualityRegime.FINITE_MEAN

        # Simple Hill estimator
        k = min(10, len(deviations) - 1)
        sorted_devs = sorted(deviations, reverse=True)
        x_k1 = sorted_devs[k]

        if x_k1 <= 0:
            return QualityRegime.INSUFFICIENT_DATA

        log
        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        if theta_hat > 1.0:
            return QualityRegime.FINITE_MEAN
        elif theta_hat > 0.0:
            return QualityRegime.REGULARLY_VARYING
        else:
            return QualityRegime.DEGENERATE

    def _cusum_detect(self, edge: ServiceEdge) -> Dict[str, Any]:
        """
        CUSUM change-point detection on edge latency stream.

        From Six_Problems P4: CUSUM + subsampling on raw process.
        """
        if len(edge.latency_history) < 20:
            return {"change_point_detected": False, "reason": "insufficient_data"}

        values = edge.latency_history
        target = np.mean(values)
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0
        change_idx = None
        drift = 0.5  # CUSUM drift parameter

        for i, v in enumerate(values):
            deviation = v - target
            cusum_pos = max(0, cusum_pos + deviation - drift)
            cusum_neg = max(0, cusum_neg - deviation - drift)

            current_max = max(cusum_pos, cusum_neg)
            if current_max > max_cusum:
                max_cusum = current_max
                if current_max > 5.0:  # threshold
                    change_idx = i

        return {
            "change_point_detected": change_idx is not None,
            "cusum_statistic": max_cusum,
            "change_point_index": change_idx,
        }

    def generate_jaeger_service_map(self) -> Dict[str, Any]:
        """Generate Jaeger-compatible service map JSON."""
        # Filter edges by PNDC Axiom 3 (only necessary edges displayed)
        filtered_edges = {
            k: v for k, v in self.edges.items()
            if v.call_count >= self.cfg.min_calls_to_display
        }

        # Build nodes list
        nodes = []
        for name, node in self.nodes.items():
            nodes.append({
                "id": name,
                "name": name,
                "zone": node.zone.value,
                "service_type": node.service_type,
                "color": self.cfg.zone_colors.get(node.zone.value, "#888"),
                "n_calls_in": node.n_calls_in,
                "n_calls_out": node.n_calls_out,
                "avg_latency_ms": node.avg_latency_ms,
                "error_rate": node.error_rate,
                "regime": node.regime.name,
            })

        # Build edges list
        edges = []
        for (src, tgt), edge in filtered_edges.items():
            cusum = self._cusum_detect(edge)
            edges.append({
                "source": src,
                "target": tgt,
                "call_count": edge.call_count,
                "avg_latency_ms": edge.avg_latency_ms,
                "p50_latency_ms": edge.p50_latency_ms,
                "p95_latency_ms": edge.p95_latency_ms,
                "error_rate": edge.error_rate,
                "zone_crossing": edge.zone_crossing,
                "regime": edge.regime.name,
                "cusum_change_point": cusum["change_point_detected"],
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "generated_at": time.time(),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

    def generate_grafana_panel(self) -> Dict[str, Any]:
        """Generate Grafana nodeGraph panel for service map."""
        return {
            "id": 100,
            "type": "nodeGraph",
            "title": "SESI Zone-Aware Service Map",
            "datasource": {"type": "jaeger", "uid": "jaeger"},
            "options": {
                "nodeGrid": {"x": 0, "y": 0, "columns": 4, "rows": 1},
                "edges": [
                    {"field": "call_count", "color": "orange"},
                    {"field": "error_rate", "color": "red"},
                ],
                "nodes": {
                    "main": {"color": "zone"},
                    "arc": "regime",
                },
            },
            "gridPos": {"h": 16, "w": 24, "x": 0, "y": 40},
        }

    def write_service_map(self, output_path: str = "./jaeger/service-map.json") -> str:
        """Write service map to file."""
        service_map = self.generate_jaeger_service_map()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(service_map, f, indent=2)
        logger.info(f"Service map written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get service map status."""
        filtered = {
            k: v for k, v in self.edges.items()
            if v.call_count >= self.cfg.min_calls_to_display
        }
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "filtered_edges": len(filtered),
            "zone_crossing_edges": sum(1 for e in self.edges.values() if e.zone_crossing),
            "regime_distribution": {
                regime.name: sum(1 for e in self.edges.values() if e.regime == regime)
                for regime in QualityRegime
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [CL1] LOKI CODIFICATION EVENT AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CodificationLogEntry:
    """
    [D] A single codification event log entry.

    HONEST: All quality scores are sanitized via _safe_unit_scalar
    (Psyche Plus v3.3 pattern). NaN/Inf → 0.5 with WARNING log.
    """
    event_id: str
    timestamp: float
    outcome: CodificationOutcome
    validity_score: float
    quality_score: float
    axiom_count: int
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    zone: SecurityZone = SecurityZone.Z2_PROCESSING
    service: str = "sesi-psyche-plus"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON for Loki ingestion."""
        entry = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "outcome": self.outcome.name,
            "validity_score": self.validity_score,
            "quality_score": self.quality_score,
            "axiom_count": self.axiom_count,
            "zone": self.zone.value,
            "service": self.service,
        }
        if self.trace_id:
            entry["trace_id"] = self.trace_id
        if self.span_id:
            entry["span_id"] = self.span_id
        if self.metadata:
            entry.update(self.metadata)
        return json.dumps(entry)


class LokiCodificationEventAggregator:
    """
    [D] Aggregates Psyche Plus codification events for Loki.

    Features:
      • Structured JSON logs for codification events
      • _safe_unit_scalar sanitization for all quality scores
      • Trace-log correlation via trace_id/span_id
      • LogQL query templates for codification analysis
      • Regime classification on codification streams
      • Multi-identity risk correlation (Comprehensive Security §5.2)
      • No-Zeno rate limiting on log emission

    HONEST: This is a LOGGING FRAMEWORK. It does NOT:
      • Guarantee that all codification events are captured
      • Replace security audit
      • Provide causation, only correlation

    Mathematical grounding:
      • _safe_unit_scalar sanitization (Psyche Plus v3.3)
      • paper6 Thm 5.1: Regime classification
      • No-Zeno: Log emission rate limiting
    """

    # Standard LogQL queries for codification analysis
    STANDARD_QUERIES = [
        {
            "name": "codification_success_rate",
            "query": (
                'sum(rate({service="sesi-psyche-plus"} | json | outcome="SUCCESS"[5m])) '
                '/ sum(rate({service="sesi-psyche-plus"} | json[5m]))'
            ),
            "description": "Overall codification success rate",
        },
        {
            "name": "codification_by_zone",
            "query": (
                'sum(rate({service="sesi-psyche-plus"} | json[5m])) by (zone, outcome)'
            ),
            "description": "Codification events per zone and outcome",
        },
        {
            "name": "validity_score_distribution",
            "query": (
                '{service="sesi-psyche-plus"} | json | '
                'line_format "{{.validity_score}}"'
            ),
            "description": "Distribution of validity scores",
        },
        {
            "name": "quality_score_distribution",
            "query": (
                '{service="sesi-psyche-plus"} | json | '
                'line_format "{{.quality_score}}"'
            ),
            "description": "Distribution of quality scores (sanitized)",
        },
        {
            "name": "refused_validity",
            "query": (
                '{service="sesi-psyche-plus"} | json | outcome="REFUSED_VALIDITY"'
            ),
            "description": "All codification refusals due to low validity",
        },
        {
            "name": "refused_quality",
            "query": (
                '{service="sesi-psyche-plus"} | json | outcome="REFUSED_QUALITY"'
            ),
            "description": "All codification refusals due to low quality",
        },
        {
            "name": "trace_correlation",
            "query": (
                '{service="sesi-psyche-plus"} | json | trace_id!="" | '
                'line_format "{{.trace_id}} {{.outcome}} validity={{.validity_score}}"'
            ),
            "description": "Codification events with trace correlation",
        },
    ]

    def __init__(
        self,
        service_name: str = "sesi-psyche-plus",
        max_buffer_size: int = 10000,
        max_logs_per_minute: int = 10000,
    ):
        self.service_name = service_name
        self.log_buffer: deque = deque(maxlen=max_buffer_size)
        self.max_logs_per_minute = max_logs_per_minute
        self._log_timestamps: deque = deque(maxlen=max_logs_per_minute * 2)
        self._codification_history: List[CodificationLogEntry] = []

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def check_log_rate(self) -> Dict[str, Any]:
        """Check if log emission is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        while self._log_timestamps and self._log_timestamps[0] < cutoff:
            self._log_timestamps.popleft()

        if len(self._log_timestamps) >= self.max_logs_per_minute:
            return {
                "allowed": False,
                "reason": "rate_limited",
                "current": len(self._log_timestamps),
                "limit": self.max_logs_per_minute,
            }

        return {"allowed": True}

    # ── Codification Event Logging ───────────────────────────────────────

    def log_codification_event(
        self,
        outcome: CodificationOutcome,
        validity_score: float,
        quality_score: float,
        axiom_count: int,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        zone: SecurityZone = SecurityZone.Z2_PROCESSING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[CodificationLogEntry]:
        """
        Log a codification event.

        Applies:
          1. _safe_unit_scalar sanitization for quality scores
          2. No-Zeno rate limiting
          3. Structured JSON formatting

        Args:
            outcome: Codification outcome (SUCCESS/REFUSED_*)
            validity_score: SuperEgo validity score (will be sanitized)
            quality_score: Ecosystem quality score (will be sanitized)
            axiom_count: Current axiom bank size
            trace_id: Optional Jaeger trace ID
            span_id: Optional Jaeger span ID
            zone: Security zone
            metadata: Optional additional metadata

        Returns:
            CodificationLogEntry if logged, None if rate-limited
        """
        # Check rate limit
        rate_check = self.check_log_rate()
        if not rate_check["allowed"]:
            logger.debug(f"Codification log rate-limited: {rate_check['reason']}")
            return None

        # Sanitize quality scores via _safe_unit_scalar
        validity_score = _safe_unit_scalar(
            validity_score, name="validity_score", logger=logger, default=0.5
        )
        quality_score = _safe_unit_scalar(
            quality_score, name="quality_score", logger=logger, default=0.5
        )

        # Data minimization (PNDC Axiom 3 + Comp. Security §5.3)
        safe_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if k.lower() not in ("password", "secret", "token", "ssn", "credit_card"):
                    safe_metadata[k] = v

        entry = CodificationLogEntry(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            outcome=outcome,
            validity_score=validity_score,
            quality_score=quality_score,
            axiom_count=axiom_count,
            trace_id=trace_id,
            span_id=span_id,
            zone=zone,
            service=self.service_name,
            metadata=safe_metadata,
        )

        self.log_buffer.append(entry)
        self._codification_history.append(entry)
        self._log_timestamps.append(time.time())

        logger.info(
            f"Codification event logged: {entry.event_id} | "
            f"outcome={outcome.name} | "
            f"validity={validity_score:.3f} | "
            f"quality={quality_score:.3f} | "
            f"axioms={axiom_count}"
        )

        return entry

    # ── LogQL Query Generation ───────────────────────────────────────────

    def generate_logql_queries(self) -> List[Dict[str, Any]]:
        """Generate LogQL query templates."""
        return self.STANDARD_QUERIES.copy()

    # ── Regime Classification ────────────────────────────────────────────

    def classify_codification_regime(self) -> Dict[str, Any]:
        """
        Classify codification event stream regime using Hill estimator.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → ordinary LLN
          θ ∈ (0,1):  regularly varying → Beta(θ,θ) arcsine
          θ → 0:      degenerate → ½δ₀ + ½δ₁
        """
        if len(self._codification_history) < 20:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_events": len(self._codification_history),
            }

        # Use validity scores for regime classification
        scores = [e.validity_score for e in self._codification_history[-100:]]
        median = np.median(scores)
        deviations = [abs(s - median) for s in scores if abs(s - median) > 0.01]

        if len(deviations) < 10:
            return {"regime": QualityRegime.FINITE_MEAN.name}

        # Hill estimator
        k = min(10, len(deviations) - 1)
        sorted_devs = sorted(deviations, reverse=True)
        x_k1 = sorted_devs[k]

        if x_k1 <= 0:
            return {"regime": QualityRegime.INSUFFICIENT_DATA.name}

        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        if theta_hat > 1.0:
            regime = QualityRegime.FINITE_MEAN
            description = "θ > 1: finite mean. Stable codification pattern."
        elif theta_hat > 0.0:
            regime = QualityRegime.REGULARLY_VARYING
            description = "θ ∈ (0,1): heavy-tailed codification pattern."
        else:
            regime = QualityRegime.DEGENERATE
            description = "θ → 0: polarized codification pattern."

        return {
            "regime": regime.name,
            "theta_hat": theta_hat,
            "description": description,
            "n_events": len(scores),
        }

    # ── Multi-Identity Risk Correlation ──────────────────────────────────

    def correlate_multi_identity(self, trace_ids: List[str]) -> Dict[str, Any]:
        """
        Correlate codification events across traces for multi-identity risk.

        From Comprehensive Security §5.2:
          R_id(u_i) = f(B_i, T_i, C_i, V_i)

        HONEST: This is a HEURISTIC risk assessment.
        """
        # Group events by trace_id
        trace_groups: Dict[str, List[CodificationLogEntry]] = defaultdict(list)
        for entry in self._codification_history:
            if entry.trace_id in trace_ids:
                trace_groups[entry.trace_id].append(entry)

        # Compute risk indicators
        refused_count = sum(
            1 for e in self._codification_history
            if e.outcome != CodificationOutcome.SUCCESS
        )
        total = len(self._codification_history)
        refusal_rate = refused_count / max(total, 1)

        # Temporal pattern analysis
        if total > 1:
            times = sorted(e.timestamp for e in self._codification_history)
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            rapid_events = sum(1 for iv in intervals if iv < 1.0)
            temporal_risk = rapid_events / max(len(intervals), 1)
        else:
            temporal_risk = 0.0

        # Composite risk
        risk = 0.5 * refusal_rate + 0.5 * temporal_risk

        return {
            "total_events": total,
            "refusal_rate": refusal_rate,
            "temporal_risk": temporal_risk,
            "composite_risk": risk,
            "risk_level": (
                "HIGH" if risk > 0.7
                else "MEDIUM" if risk > 0.4
                else "LOW"
            ),
            "note": (
                "Multi-identity risk per Comprehensive Security §5.2. "
                "This is a HEURISTIC score, not proof of identity fraud."
            ),
        }

    # ── Log Export ───────────────────────────────────────────────────────

    def write_logs_to_file(self, output_path: str = "./logs/codification.jsonl") -> str:
        """Write logs to JSONL file for Loki ingestion."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for entry in self.log_buffer:
                f.write(entry.to_json() + "\n")

        logger.info(f"Wrote {len(self.log_buffer)} codification logs to {output_path}")
        return str(out_path)

    def write_loki_config(self, output_path: str = "./loki/loki-config.yaml") -> str:
        """Generate Loki configuration file."""
        config = {
            "auth_enabled": False,
            "server": {"http_listen_port": 3100},
            "ingester": {
                "wal": {"enabled": True, "dir": "/tmp/wal"},
                "lifecycler": {
                    "address": "127.0.0.1",
                    "ring": {
                        "kvstore": {"store": "inmemory"},
                        "replication_factor": 1,
                    },
                },
            },
            "schema_config": {
                "configs": [{
                    "from": "2020-10-24",
                    "store": "boltdb-shipper",
                    "object_store": "filesystem",
                    "schema": "v11",
                    "index": {"prefix": "index_", "period": "24h"},
                }],
            },
            "storage_config": {
                "boltdb_shipper": {
                    "active_index_directory": "/tmp/loki/boltdb-shipper-active",
                    "cache_location": "/tmp/loki/boltdb-shipper-cache",
                    "shared_store": "filesystem",
                },
                "filesystem": {"directory": "/tmp/loki/chunks"},
            },
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"Loki config written to {output_path}")
        return str(out_path)

    def write_logql_queries(self, output_path: str = "./loki/codification-queries.json") -> str:
        """Write LogQL queries to file."""
        queries = self.generate_logql_queries()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(queries, f, indent=2)
        logger.info(f"LogQL queries written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get aggregator status."""
        return {
            "service_name": self.service_name,
            "buffer_size": len(self.log_buffer),
            "total_events": len(self._codification_history),
            "regime": self.classify_codification_regime(),
            "recent_log_rate_per_min": len([
                t for t in self._log_timestamps
                if t > time.time() - 60.0
            ]),
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIZoneObservabilitySuite:
    """
    [D] Complete zone observability suite.

    Integrates:
      • Zone-specific Alertmanager routing
      • Zone-aware Jaeger service map
      • Loki codification event aggregation
    """

    def __init__(self):
        self.alertmanager = ZoneSpecificAlertmanagerRouter()
        self.service_map = ZoneAwareJaegerServiceMap()
        self.codification_logs = LokiCodificationEventAggregator()

        # Register default services
        self._register_default_services()

    def _register_default_services(self) -> None:
        """Register default SESI services in the service map."""
        services = [
            ("sesi-implant-gateway", SecurityZone.Z0_IMPLANT, "implant"),
            ("sesi-clinical-gateway", SecurityZone.Z1_CLINICAL, "clinical"),
            ("sesi-processing", SecurityZone.Z2_PROCESSING, "processing"),
            ("sesi-external-api", SecurityZone.Z3_EXTERNAL, "external"),
            ("sesi-vault", SecurityZone.Z1_CLINICAL, "security"),
            ("sesi-psyche-plus", SecurityZone.Z2_PROCESSING, "agi"),
            ("sesi-agione", SecurityZone.Z2_PROCESSING, "agi"),
        ]
        for name, zone, stype in services:
            self.service_map.register_service(name, zone, stype)

    def log_codification(
        self,
        outcome: CodificationOutcome,
        validity_score: float,
        quality_score: float,
        axiom_count: int,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        zone: SecurityZone = SecurityZone.Z2_PROCESSING,
    ) -> Optional[CodificationLogEntry]:
        """Log a codification event across all observability layers."""
        return self.codification_logs.log_codification_event(
            outcome=outcome,
            validity_score=validity_score,
            quality_score=quality_score,
            axiom_count=axiom_count,
            trace_id=trace_id,
            span_id=span_id,
            zone=zone,
        )

    def record_service_call(
        self,
        source: str,
        target: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a service-to-service call."""
        self.service_map.record_call(source, target, latency_ms, success)

    def generate_all_artifacts(self, output_dir: str = "./sesi_zone_obs") -> Dict[str, str]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Alertmanager config
        results["alertmanager_config"] = self.alertmanager.write_config(
            str(out_path / "alertmanager" / "alertmanager.yml")
        )
        results["alert_rules"] = self.alertmanager.write_alert_rules(
            str(out_path / "prometheus" / "zone-alerts.yaml")
        )

        # Service map
        results["service_map"] = self.service_map.write_service_map(
            str(out_path / "jaeger" / "service-map.json")
        )

        # Loki config
        results["loki_config"] = self.codification_logs.write_loki_config(
            str(out_path / "loki" / "loki-config.yaml")
        )
        results["logql_queries"] = self.codification_logs.write_logql_queries(
            str(out_path / "loki" / "codification-queries.json")
        )
        results["codification_logs"] = self.codification_logs.write_logs_to_file(
            str(out_path / "logs" / "codification.jsonl")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "alertmanager": self.alertmanager.get_status(),
            "service_map": self.service_map.get_status(),
            "codification_logs": self.codification_logs.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Zone Alertmanager × Jaeger × Loki Codification — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIZoneObservabilitySuite()

    # ── Test 1: Zone-Specific Alertmanager ───────────────────────────────
    print("\n  ── Zone-Specific Alertmanager Router ──")
    am_status = suite.alertmanager.get_status()
    print(f"    Zones configured     : {am_status['zones_configured']}")
    print(f"    Alert rules          : {am_status['alert_rules']}")
    print(f"    Inhibition rules     : {len(suite.alertmanager.generate_inhibition_rules())}")
    print(f"    δ_min                : {am_status['delta_min']:.6f}")

    # Generate alert rules
    alert_rules = suite.alertmanager.generate_zone_alert_rules()
    print(f"    Total alert rules    : {len(alert_rules)}")
    for rule in alert_rules[:3]:
        print(f"      • {rule['alert']} ({rule['labels']['severity']})")

    # Test alert rate limiting
    rate_check = suite.alertmanager.check_alert_rate("test_alert")
    print(f"    Alert rate check     : {rate_check['allowed']}")

    # ── Test 2: Zone-Aware Jaeger Service Map ────────────────────────────
    print("\n  ── Zone-Aware Jaeger Service Map ──")

    # Record service calls
    service_calls = [
        ("sesi-external-api", "sesi-processing", 50.0, True),
        ("sesi-processing", "sesi-clinical-gateway", 30.0, True),
        ("sesi-clinical-gateway", "sesi-implant-gateway", 100.0, True),
        ("sesi-clinical-gateway", "sesi-vault", 20.0, True),
        ("sesi-processing", "sesi-psyche-plus", 80.0, True),
        ("sesi-psyche-plus", "sesi-agione", 40.0, True),
        ("sesi-external-api", "sesi-vault", 200.0, False),
    ]

    for _ in range(25):
        for src, tgt, latency, success in service_calls:
            suite.record_service_call(src, tgt, latency, success)

    # Generate service map
    service_map = suite.service_map.generate_jaeger_service_map()
    print(f"    Nodes                : {service_map['n_nodes']}")
    print(f"    Edges (filtered)     : {service_map['n_edges']}")

    # Service map status
    sm_status = suite.service_map.get_status()
    print(f"    Zone crossings       : {sm_status['zone_crossing_edges']}")
    print(f"    Regime distribution  : {sm_status['regime_distribution']}")

    # ── Test 3: Loki Codification Event Aggregation ──────────────────────
    print("\n  ── Loki Codification Event Aggregation ──")

    # Log codification events
    events = [
        (CodificationOutcome.SUCCESS, 0.85, 0.9, 10, "trace_001", "span_001"),
        (CodificationOutcome.REFUSED_VALIDITY, 0.3, 0.8, 10, "trace_002", "span_002"),
        (CodificationOutcome.REFUSED_QUALITY, 0.9, 0.2, 10, "trace_003", "span_003"),
        (CodificationOutcome.SUCCESS, 0.75, 0.85, 11, "trace_004", "span_004"),
        (CodificationOutcome.REFUSED_NONFINITE, float('nan'), 0.5, 11, "trace_005", "span_005"),
    ]

    for outcome, validity, quality, axioms, tid, sid in events:
        entry = suite.log_codification(
            outcome=outcome,
            validity_score=validity,
            quality_score=quality,
            axiom_count=axioms,
            trace_id=tid,
            span_id=sid,
        )
        if entry:
            print(f"    Logged               : {entry.event_id} | {outcome.name} | "
                  f"v={entry.validity_score:.3f} q={entry.quality_score:.3f}")

    # Regime classification
    regime = suite.codification_logs.classify_codification_regime()
    print(f"    Codification regime  : {regime['regime']}")
    if 'theta_hat' in regime:
        print(f"    θ̂ (Hill)            : {regime['theta_hat']:.4f}")

    # Multi-identity correlation
    trace_ids = [f"trace_{i:03d}" for i in range(1, 6)]
    mi_risk = suite.codification_logs.correlate_multi_identity(trace_ids)
    print(f"    Multi-identity risk  : {mi_risk['composite_risk']:.3f} ({mi_risk['risk_level']})")

    # Generate LogQL queries
    queries = suite.codification_logs.generate_logql_queries()
    print(f"    LogQL queries        : {len(queries)}")
    for q in queries[:3]:
        print(f"      • {q['name']}: {q['description'][:50]}...")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_zone_obs")
    print(f"    Artifacts generated  : {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:25s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Alertmanager rules   : {status['alertmanager']['alert_rules']}")
    print(f"    Service map nodes    : {status['service_map']['n_nodes']}")
    print(f"    Service map edges    : {status['service_map']['n_edges']}")
    print(f"    Codification events  : {status['codification_logs']['total_events']}")
    print(f"    Codification regime  : {status['codification_logs']['regime']['regime']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Alertmanager is a CONFIG GENERATOR, not a deployed system")
    print("  • Jaeger service map is TOPOLOGY, not security guarantee")
    print("  • Loki codification logs are STRUCTURED RECORDS, not omniscience")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all quality inputs")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
