"""
=============================================================================
SESI AGI: TRACE-BASET ALERTMANAGER + UNIFIED GRAFANA + LOKI LIGHTWEIGHT
Span-Level Alerting + Cross-Component Dashboard + Lightweight Log Aggregation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[TBA1] TraceBasedAlertmanager
      • Span-level alert rules from Alloy/Tempo traces
      • Per-zone thresholds (PNDC Axiom 3 proportionality)
      • Alert types: latency P95, error rate, duration anomaly, trace regime
      • Inhibition rules to prevent alert storms
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for threshold calibration
      • Alertmanager YAML + Prometheus rule generation

[UGD1] UnifiedGrafanaDashboard
      • Single-pane glass across Alloy/Tempo/OpenSearch
      • Row 1: System Overview (all components health)
      • Row 2: Alloy Pipeline Metrics (per-zone throughput)
      • Row 3: Tempo Trace Metrics (latency, error, regime)
      • Row 4: OpenSearch Log Metrics (search rate, indexing)
      • Row 5: Loki Lightweight Logs (per-module log volume)
      • Row 6: Cross-Component Correlation (trace_id linking)
      • Row 7: AGI Module Health (10 modules × metrics)
      • Template variables for zone/module/trace_id filtering

[LL1] LokiLightweightAggregator
      • Structured JSON logging for AGI modules
      • Label-based indexing (module, zone, level, trace_id)
      • LogQL query templates for AGI-specific analysis
      • Zone-based retention (PNDC Axiom 3)
      • _safe_unit_scalar sanitization on all values
      • No-Zeno rate limiting on log emission
      • K8s Deployment + Service manifests

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
[C] Six_Problems P4: CUSUM + subsampling for change-point detection
[C] PNDC Axiom 3: Necessity and Proportionality
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
[C] Unified_Ordinal Hyp 7.9: Free design parameters

RELATION TO AGI UNIFIED FRAMEWORK (from knowledge base)
-------------------------------------------------------
From AGI Unified Framework Single File.py (10 modules):
  • CoreReasoningEngine → span: "agi.reasoning"
  • EpisodicSemanticMemory → span: "agi.memory"
  • PerceptionMultimodal → span: "agi.perception"
  • ActionActuationController → span: "agi.actuation"
  • EpistemicCuriosityDrive → span: "agi.curiosity"
  • ConceptSynthesisEngine → span: "agi.synthesis"
  • SelfEvalReflection → span: "agi.reflection"
  • SafetyAlignmentGuard → span: "agi.safety"
  • DynamicSubstrateAllocator → span: "agi.substrate"
  • MetaLearningPipeline → span: "agi.meta_learning"

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Trace-based alerts are CONFIGURATION; they do not replace human review.
• Unified dashboard is VISUALIZATION; it does not make decisions.
• Loki is LIGHTWEIGHT; it does not replace OpenSearch for full-text search.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
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

logger = logging.getLogger("sesi.alertmanager_unified_loki")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from agi_one_psyche_plus_v33 import _safe_unit_scalar
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
    "SecurityZone", "AlertSeverity", "QualityRegime", "AGIModule",
    "TraceBasedAlertmanager", "ZoneAlertConfig", "TraceAlert",
    "UnifiedGrafanaDashboard",
    "LokiLightweightAggregator", "LokiConfig", "LokiLogEntry",
    "SESIUnifiedObservabilityV6",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

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
    FINITE_MEAN = 0
    REGULARLY_VARYING = 1
    DEGENERATE = 2
    INSUFFICIENT_DATA = 3


class AGIModule(Enum):
    """The 10 AGI modules from the unified framework."""
    REASONING = "reasoning"
    MEMORY = "memory"
    PERCEPTION = "perception"
    ACTUATION = "actuation"
    CURIOSITY = "curiosity"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"
    SAFETY = "safety"
    SUBSTRATE = "substrate"
    META_LEARNING = "meta_learning"


# ═══════════════════════════════════════════════════════════════════════════
# [TBA1] TRACE-BASED ALERTMANAGER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneAlertConfig:
    """
    Per-zone trace-based alert configuration.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be calibrated against real data.
    PNDC Axiom 3 proportionality: Z0 strictest, Z3 loosest.
    """
    zone: SecurityZone
    # Trace latency thresholds (seconds)
    latency_warning_s: float = 1.0
    latency_critical_s: float = 5.0
    latency_emergency_s: float = 30.0
    # Trace error rate thresholds
    error_rate_warning: float = 0.01    # 1%
    error_rate_critical: float = 0.05   # 5%
    error_rate_emergency: float = 0.20  # 20%
    # Trace duration anomaly (multiplier vs baseline)
    duration_anomaly_warning: float = 2.0
    duration_anomaly_critical: float = 5.0
    # Default severity
    default_severity: AlertSeverity = AlertSeverity.WARNING
    # Receiver
    receiver: str = "platform-team"
    # Grouping
    group_wait: str = "30s"
    group_interval: str = "5m"
    repeat_interval: str = "4h"


@dataclass
class TraceAlert:
    """A single trace-based alert."""
    alert_id: str
    zone: SecurityZone
    module: AGIModule
    severity: AlertSeverity
    alert_type: str
    message: str
    trace_id: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "zone": self.zone.value,
            "module": self.module.value,
            "severity": self.severity.value,
            "alert_type": self.alert_type,
            "message": self.message,
            "trace_id": self.trace_id,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class TraceBasedAlertmanager:
    """
    [D] Trace-based alerting from Alloy/Tempo span data.

    Features:
      • Span-level alert rules (latency, error rate, duration anomaly)
      • Per-zone thresholds (PNDC Axiom 3 proportionality)
      • Inhibition rules to prevent alert storms
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for threshold calibration
      • Alertmanager YAML + Prometheus rule generation

    HONEST: This is a CONFIGURATION GENERATOR. Real deployment requires:
      • Alertmanager server running
      • Proper receiver configuration
      • Security audit of alert routing
    """

    # Default zone configs (PNDC Axiom 3 proportionality)
    DEFAULT_ZONE_CONFIGS = {
        SecurityZone.Z0_IMPLANT: ZoneAlertConfig(
            zone=SecurityZone.Z0_IMPLANT,
            latency_warning_s=0.5,
            latency_critical_s=2.0,
            latency_emergency_s=10.0,
            error_rate_warning=0.001,
            error_rate_critical=0.01,
            error_rate_emergency=0.05,
            duration_anomaly_warning=1.5,
            duration_anomaly_critical=3.0,
            default_severity=AlertSeverity.EMERGENCY,
            receiver="clinical-pager",
            group_wait="10s",
            repeat_interval="1h",
        ),
        SecurityZone.Z1_CLINICAL: ZoneAlertConfig(
            zone=SecurityZone.Z1_CLINICAL,
            latency_warning_s=1.0,
            latency_critical_s=5.0,
            latency_emergency_s=20.0,
            error_rate_warning=0.005,
            error_rate_critical=0.02,
            error_rate_emergency=0.10,
            duration_anomaly_warning=2.0,
            duration_anomaly_critical=4.0,
            default_severity=AlertSeverity.CRITICAL,
            receiver="clinical-team",
            group_wait="30s",
            repeat_interval="4h",
        ),
        SecurityZone.Z2_PROCESSING: ZoneAlertConfig(
            zone=SecurityZone.Z2_PROCESSING,
            latency_warning_s=2.0,
            latency_critical_s=10.0,
            latency_emergency_s=60.0,
            error_rate_warning=0.01,
            error_rate_critical=0.05,
            error_rate_emergency=0.20,
            duration_anomaly_warning=3.0,
            duration_anomaly_critical=6.0,
            default_severity=AlertSeverity.WARNING,
            receiver="platform-slack",
            group_wait="1m",
            repeat_interval="12h",
        ),
        SecurityZone.Z3_EXTERNAL: ZoneAlertConfig(
            zone=SecurityZone.Z3_EXTERNAL,
            latency_warning_s=5.0,
            latency_critical_s=30.0,
            latency_emergency_s=120.0,
            error_rate_warning=0.02,
            error_rate_critical=0.10,
            error_rate_emergency=0.30,
            duration_anomaly_warning=5.0,
            duration_anomaly_critical=10.0,
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
        self._alerts: List[TraceAlert] = []
        self._alert_timestamps: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._trace_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

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
                }

        return {"allowed": True}

    def record_alert_timestamp(self, zone: SecurityZone) -> None:
        """Record an alert emission for rate limiting."""
        self._alert_timestamps[zone.value].append(time.time())

    # ── Trace Regime Classification ──────────────────────────────────────

    def classify_trace_regime(
        self,
        zone: SecurityZone,
        module: AGIModule,
    ) -> Dict[str, Any]:
        """
        Classify trace stream regime using Hill estimator.

        From paper6 Thm 5.1.
        """
        key = f"{zone.value}:{module.value}"
        history = self._trace_history.get(key, [])

        if len(history) < 20:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_samples": len(history),
            }

        # Use duration as signal
        durations = [h.get("duration_ms", 0.0) for h in history[-100:]]
        median = np.median(durations)
        deviations = [abs(d - median) for d in durations if abs(d - median) > 0.01]

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
            description = "θ > 1: stable trace pattern."
        elif theta_hat > 0.0:
            regime = QualityRegime.REGULARLY_VARYING
            description = "θ ∈ (0,1): fluctuating trace pattern."
        else:
            regime = QualityRegime.DEGENERATE
            description = "θ → 0: polarized trace pattern."

        return {
            "regime": regime.name,
            "theta_hat": theta_hat,
            "description": description,
            "n_samples": len(durations),
        }

    # ── Trace Evaluation ─────────────────────────────────────────────────

    def evaluate_trace(
        self,
        zone: SecurityZone,
        module: AGIModule,
        trace_id: str,
        duration_ms: float,
        error: bool = False,
        baseline_duration_ms: Optional[float] = None,
    ) -> Optional[TraceAlert]:
        """
        Evaluate a trace span against zone thresholds.

        Args:
            zone: Security zone
            module: AGI module
            trace_id: Jaeger trace ID
            duration_ms: Span duration in milliseconds
            error: Whether the span had an error
            baseline_duration_ms: Optional baseline for anomaly detection

        Returns:
            TraceAlert if threshold exceeded, None otherwise
        """
        # Sanitize duration
        duration_ms = _safe_unit_scalar(
            duration_ms / 1000.0, name="duration_s", logger=logger, default=0.0
        )
        if duration_ms is None:
            duration_ms = 0.0
        duration_s = duration_ms  # Already scaled

        # Record trace history
        key = f"{zone.value}:{module.value}"
        self._trace_history[key].append({
            "trace_id": trace_id,
            "duration_ms": duration_ms * 1000,
            "error": error,
            "timestamp": time.time(),
        })
        if len(self._trace_history[key]) > 1000:
            self._trace_history[key] = self._trace_history[key][-1000:]

        # Check rate limit
        rate_check = self.check_alert_rate(zone)
        if not rate_check["allowed"]:
            return None

        cfg = self.zone_configs.get(zone)
        if cfg is None:
            return None

        # Evaluate thresholds
        severity = None
        alert_type = None
        message = None
        metric_value = None
        threshold = None

        # 1. Latency threshold
        if duration_s >= cfg.latency_emergency_s:
            severity = AlertSeverity.EMERGENCY
            alert_type = "trace_latency_emergency"
            message = (
                f"EMERGENCY: Trace latency in {zone.value}/{module.value} "
                f"exceeds {cfg.latency_emergency_s}s (actual: {duration_s:.3f}s)"
            )
            metric_value = duration_s
            threshold = cfg.latency_emergency_s
        elif duration_s >= cfg.latency_critical_s:
            severity = AlertSeverity.CRITICAL
            alert_type = "trace_latency_critical"
            message = (
                f"CRITICAL: Trace latency in {zone.value}/{module.value} "
                f"exceeds {cfg.latency_critical_s}s (actual: {duration_s:.3f}s)"
            )
            metric_value = duration_s
            threshold = cfg.latency_critical_s
        elif duration_s >= cfg.latency_warning_s:
            severity = cfg.default_severity
            alert_type = "trace_latency_warning"
            message = (
                f"Trace latency in {zone.value}/{module.value} "
                f"exceeds {cfg.latency_warning_s}s (actual: {duration_s:.3f}s)"
            )
            metric_value = duration_s
            threshold = cfg.latency_warning_s

        # 2. Error check (overrides latency if more severe)
        if error:
            error_severity = AlertSeverity.CRITICAL
            if cfg.default_severity.value > error_severity.value:
                error_severity = cfg.default_severity

            if severity is None or error_severity.value > severity.value:
                severity = error_severity
                alert_type = "trace_error"
                message = (
                    f"Trace error in {zone.value}/{module.value}. "
                    f"Trace ID: {trace_id}"
                )

        # 3. Duration anomaly (if baseline provided)
        if baseline_duration_ms and baseline_duration_ms > 0:
            anomaly_ratio = (duration_ms * 1000) / baseline_duration_ms
            if anomaly_ratio >= cfg.duration_anomaly_critical:
                if severity is None or AlertSeverity.CRITICAL.value > severity.value:
                    severity = AlertSeverity.CRITICAL
                    alert_type = "trace_duration_anomaly"
                    message = (
                        f"Trace duration anomaly in {zone.value}/{module.value}: "
                        f"{anomaly_ratio:.1f}x baseline"
                    )
                    metric_value = anomaly_ratio
                    threshold = cfg.duration_anomaly_critical
            elif anomaly_ratio >= cfg.duration_anomaly_warning:
                if severity is None:
                    severity = cfg.default_severity
                    alert_type = "trace_duration_anomaly"
                    message = (
                        f"Trace duration anomaly in {zone.value}/{module.value}: "
                        f"{anomaly_ratio:.1f}x baseline"
                    )
                    metric_value = anomaly_ratio
                    threshold = cfg.duration_anomaly_warning

        if severity is None:
            return None

        # Check regime (escalate if degenerate)
        regime_result = self.classify_trace_regime(zone, module)
        if regime_result["regime"] == QualityRegime.DEGENERATE.name:
            if severity.value < AlertSeverity.CRITICAL.value:
                severity = AlertSeverity.CRITICAL
                message += " [DEGENERATE REGIME: paper6 Thm 5.1]"

        # Create alert
        alert = TraceAlert(
            alert_id=str(uuid.uuid4())[:8],
            zone=zone,
            module=module,
            severity=severity,
            alert_type=alert_type,
            message=message,
            trace_id=trace_id,
            metric_value=metric_value,
            threshold=threshold,
            metadata={
                "regime": regime_result["regime"],
                "theta_hat": regime_result.get("theta_hat"),
            },
        )

        self._alerts.append(alert)
        self.record_alert_timestamp(zone)

        logger.info(
            f"[TRACE ALERT] {alert.alert_id} | zone={zone.value} | "
            f"module={module.value} | severity={severity.value} | "
            f"type={alert_type}"
        )

        return alert

    # ── Alert Rule Generation ────────────────────────────────────────────

    def generate_prometheus_rules(self) -> List[Dict[str, Any]]:
        """Generate Prometheus alert rules for trace-based alerting."""
        rules = []

        for zone, cfg in self.zone_configs.items():
            zone_name = zone.value

            # Latency P95 alert
            rules.append({
                "alert": f"TraceLatencyP95_{zone_name}",
                "expr": (
                    f'histogram_quantile(0.95, '
                    f'sum(rate(sesi_agi_module_latency_seconds_bucket{{zone="{zone_name}"}}[5m])) by (le, module)) '
                    f'> {cfg.latency_warning_s}'
                ),
                "for": "2m",
                "labels": {
                    "severity": cfg.default_severity.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"High trace latency P95 in {zone_name}",
                    "description": (
                        f"Trace latency P95 in {zone_name} exceeds "
                        f"{cfg.latency_warning_s}s for 2 minutes."
                    ),
                },
            })

            # Error rate alert
            rules.append({
                "alert": f"TraceErrorRate_{zone_name}",
                "expr": (
                    f'sum(rate(sesi_agi_module_calls_total{{zone="{zone_name}",status="ERROR"}}[5m])) '
                    f'/ sum(rate(sesi_agi_module_calls_total{{zone="{zone_name}"}}[5m])) '
                    f'> {cfg.error_rate_warning}'
                ),
                "for": "5m",
                "labels": {
                    "severity": cfg.default_severity.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"High trace error rate in {zone_name}",
                    "description": (
                        f"Trace error rate in {zone_name} exceeds "
                        f"{cfg.error_rate_warning*100:.1f}%."
                    ),
                },
            })

            # Degenerate regime alert
            rules.append({
                "alert": f"TraceDegenerateRegime_{zone_name}",
                "expr": f'sesi_agi_module_regime{{zone="{zone_name}"}} == 2',
                "for": "5m",
                "labels": {
                    "severity": AlertSeverity.CRITICAL.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"Trace stream in degenerate regime ({zone_name})",
                    "description": (
                        f"paper6 Thm 5.1: trace stream in {zone_name} is degenerate "
                        f"(θ → 0, ½δ₀+½δ₁). Polarized behavior detected."
                    ),
                },
            })

        return rules

    def generate_inhibition_rules(self) -> List[Dict[str, Any]]:
        """Generate inhibition rules to prevent alert storms."""
        return [
            {
                "source_matchers": ['severity="emergency"'],
                "target_matchers": ['severity=~"critical|warning|info"'],
                "equal": ["zone", "module"],
            },
            {
                "source_matchers": ['severity="critical"'],
                "target_matchers": ['severity=~"warning|info"'],
                "equal": ["zone", "module"],
            },
            {
                "source_matchers": ['alertname=~"TraceDegenerateRegime_.*"'],
                "target_matchers": ['alertname=~"TraceLatencyP95_.*"'],
                "equal": ["zone"],
            },
        ]

    def generate_alertmanager_config(self) -> Dict[str, Any]:
        """Generate complete Alertmanager configuration."""
        routes = []
        for zone, cfg in self.zone_configs.items():
            routes.append({
                "match": {"zone": zone.value},
                "receiver": cfg.receiver,
                "group_by": ["alertname", "zone", "module"],
                "group_wait": cfg.group_wait,
                "group_interval": cfg.group_interval,
                "repeat_interval": cfg.repeat_interval,
            })

        return {
            "global": {"resolve_timeout": "5m"},
            "route": {
                "receiver": "default-receiver",
                "group_by": ["alertname", "zone", "module", "severity"],
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
                        "channel": "#clinical-traces",
                    }],
                },
                {
                    "name": "platform-slack",
                    "slack_configs": [{
                        "api_url": "${SLACK_WEBHOOK}",
                        "channel": "#platform-traces",
                    }],
                },
                {
                    "name": "platform-email",
                    "email_configs": [{"to": "platform@sesi.example.com"}],
                },
            ],
            "inhibit_rules": self.generate_inhibition_rules(),
        }

    def write_alert_rules(self, output_path: str = "./prometheus/trace-alerts.yaml") -> str:
        """Write trace alert rules to file."""
        rules = self.generate_prometheus_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {"groups": [{"name": "sesi_trace_alerts", "rules": rules}]}

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"[Alertmanager] Trace alert rules written to {output_path}")
        return str(out_path)

    def write_alertmanager_config(self, output_path: str = "./alertmanager/alertmanager.yml") -> str:
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

        logger.info(f"[Alertmanager] Config written to {output_path}")
        return str(out_path)

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return [a.to_dict() for a in self._alerts[-limit:]]

    def get_status(self) -> Dict[str, Any]:
        """Get alertmanager status."""
        return {
            "zones_configured": len(self.zone_configs),
            "total_alerts": len(self._alerts),
            "alert_rules": len(self.generate_prometheus_rules()),
            "inhibition_rules": len(self.generate_inhibition_rules()),
            "delta_min": self.compute_delta_min(),
            "trace_history_size": sum(len(h) for h in self._trace_history.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [UGD1] UNIFIED GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedGrafanaDashboard:
    """
    [D] Unified Grafana dashboard across Alloy/Tempo/OpenSearch.

    Layout:
      • Row 1: System Overview (all components health)
      • Row 2: Alloy Pipeline Metrics (per-zone throughput)
      • Row 3: Tempo Trace Metrics (latency, error, regime)
      • Row 4: OpenSearch Log Metrics (search rate, indexing)
      • Row 5: Loki Lightweight Logs (per-module log volume)
      • Row 6: Cross-Component Correlation (trace_id linking)
      • Row 7: AGI Module Health (10 modules × metrics)
    """

    def __init__(
        self,
        prometheus_uid: str = "prometheus",
        tempo_uid: str = "tempo",
        loki_uid: str = "loki",
        opensearch_uid: str = "opensearch",
    ):
        self.prometheus_uid = prometheus_uid
        self.tempo_uid = tempo_uid
        self.loki_uid = loki_uid
        self.opensearch_uid = opensearch_uid

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
        """Create a Grafana panel."""
        ds_uid = datasource_uid or self.prometheus_uid
        ds_type = "tempo" if ds_uid == self.tempo_uid else \
                  "loki" if ds_uid == self.loki_uid else \
                  "opensearch" if ds_uid == self.opensearch_uid else "prometheus"

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

    def _row_panel(self, panel_id: int, title: str, y: int) -> Dict[str, Any]:
        """Create a row panel."""
        return {
            "id": panel_id,
            "type": "row",
            "title": title,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            "collapsed": False,
            "panels": [],
        }

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete unified dashboard."""
        panels = []
        pid = 1
        y = 0

        # ── Row 1: System Overview ──────────────────────────────────────
        panels.append(self._row_panel(pid, "System Overview", y)); pid += 1; y += 1

        panels.append(self._panel(
            pid, "Alloy Pipelines Active", "stat",
            [{"expr": "count(alloy_pipeline_active)", "legendFormat": "pipelines"}],
            {"h": 5, "w": 4, "x": 0, "y": y},
        )); pid += 1

        panels.append(self._panel(
            pid, "Tempo Trace Rate", "stat",
            [{"expr": "sum(rate(tempo_ingester_traces_created_total[5m]))", "legendFormat": "traces/s"}],
            {"h": 5, "w": 4, "x": 4, "y": y},
        )); pid += 1

        panels.append(self._panel(
            pid, "OpenSearch Index Rate", "stat",
            [{"expr": "sum(rate(opensearch_indexing_index_total[5m]))", "legendFormat": "docs/s"}],
            {"h": 5, "w": 4, "x": 8, "y": y},
        )); pid += 1

        panels.append(self._panel(
            pid, "Loki Log Rate", "stat",
            [{"expr": "sum(rate(loki_distributor_lines_received_total[5m]))", "legendFormat": "lines/s"}],
            {"h": 5, "w": 4, "x": 12, "y": y},
        )); pid += 1

        panels.append(self._panel(
            pid, "Active Alerts", "stat",
            [{"expr": "sum(ALERTS{alertstate='firing'}) or vector(0)", "legendFormat": "alerts"}],
            {"h": 5, "w": 4, "x": 16, "y": y},
            extra={"fieldConfig": {"defaults": {"thresholds": {"mode": "absolute", "steps": [
                {"color": "green", "value": None},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5},
            ]}}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "δ_min Threshold", "stat",
            [{"expr": "sesi_agi_delta_min", "legendFormat": "δ_min"}],
            {"h": 5, "w": 4, "x": 20, "y": y},
        )); pid += 1

        y += 6

        # ── Row 2: Alloy Pipeline Metrics ───────────────────────────────
        panels.append(self._row_panel(pid, "Grafana Alloy Pipeline Metrics", y)); pid += 1; y += 1

        panels.append(self._panel(
            pid, "Alloy Throughput by Zone", "timeseries",
            [{"expr": "sum(rate(alloy_receiver_accepted_samples_total[5m])) by (zone)",
              "legendFormat": "{{zone}}"}],
            {"h": 8, "w": 12, "x": 0, "y": y},
        )); pid += 1

        panels.append(self._panel(
            pid, "Alloy Export Rate by Destination", "timeseries",
            [{"expr": "sum(rate(alloy_exporter_sent_total[5m])) by (exporter)",
              "legendFormat": "{{exporter}}"}],
            {"h": 8, "w": 12, "x": 12, "y": y},
        )); pid += 1

        y += 9

        # ── Row 3: Tempo Trace Metrics ──────────────────────────────────
        panels.append(self._row_panel(pid, "Tempo Distributed Trace Metrics", y)); pid += 1; y += 1

        panels.append(self._panel(
            pid, "Trace Latency P95 by Module", "timeseries",
            [{
                "expr": "histogram_quantile(0.95, sum(rate(tempo_span_duration_seconds_bucket[5m])) by (le, module))",
                "legendFormat": "{{module}}",
            }],
            {"h": 8, "w": 8, "x": 0, "y": y},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "Trace Error Rate by Zone", "timeseries",
            [{
                "expr": "sum(rate(tempo_span_status_total{status='error'}[5m])) by (zone) / sum(rate(tempo_span_status_total[5m])) by (zone)",
                "legendFormat": "{{zone}}",
            }],
            {"h": 8, "w": 8, "x": 8, "y": y},
            extra={"fieldConfig": {"defaults": {"unit": "percentunit"}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "Trace Regime per Zone", "state-timeline",
            [{
                "expr": "sesi_agi_trace_regime",
                "legendFormat": "{{zone}}",
            }],
            {"h": 8, "w": 8, "x": 16, "y": y},
        )); pid += 1

        y += 9

        # ── Row 4: OpenSearch Log Metrics ───────────────────────────────
        panels.append(self._row_panel(pid, "OpenSearch Log Search Metrics", y)); pid += 1; y += 1

        panels.append(self._panel(
            pid, "OpenSearch Query Rate", "timeseries",
            [{
                "expr": "sum(rate(opensearch_search_query_total[5m])) by (index)",
                "legendFormat": "{{index}}",
            }],
            {"h": 8, "w": 12, "x": 0, "y": y},
            datasource_uid=self.opensearch_uid,
        )); pid += 1

        panels.append(self._panel(
            pid, "OpenSearch Indexing Latency P95", "timeseries",
            [{
                "expr": "histogram_quantile(0.95, rate(opensearch_indexing_latency_seconds_bucket[5m]))",
                "legendFormat": "p95",
            }],
            {"h": 8, "w": 12, "x": 12, "y": y},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        )); pid += 1

        y += 9

        # ── Row 5: Loki Lightweight Logs ────────────────────────────────
        panels.append(self._row_panel(pid, "Loki Lightweight Log Aggregation", y)); pid += 1; y += 1

        panels.append(self._panel(
            pid, "Log Volume by Module", "timeseries",
            [{
                "expr": "sum(rate({app='sesi-agi'}[5m])) by (module)",
                "legendFormat": "{{module}}",
            }],
            {"h": 8, "w": 12, "x": 0, "y": y},
            datasource_uid=self.loki_uid,
        )); pid += 1

        panels.append(self._panel(
            pid, "Log Volume by Level", "timeseries",
            [{
                "expr": "sum(rate({app='sesi-agi'}[5m])) by (level)",
                "legendFormat": "{{level}}",
            }],
            {"h": 8, "w": 12, "x": 12, "y": y},
            datasource_uid=self.loki_uid,
        )); pid += 1

        y += 9

        # ── Row 6: Cross-Component Correlation ──────────────────────────
        panels.append(self._row_panel(pid, "Cross-Component Correlation (trace_id)", y)); pid += 1; y += 1

        panels.append(self._panel(
            pid, "Trace → Logs Correlation", "logs",
            [{
                "expr": "{app='sesi-agi'} | json | trace_id!=\"\"",
                "legendFormat": "",
            }],
            {"h": 10, "w": 24, "x": 0, "y": y},
            datasource_uid=self.loki_uid,
        )); pid += 1

        y += 11

        # ── Row 7: AGI Module Health ────────────────────────────────────
        panels.append(self._row_panel(pid, "AGI Module Health (10 Modules)", y)); pid += 1; y += 1

        # Module health grid (2 rows × 5 columns)
        module_list = list(AGIModule)
        for i, module in enumerate(module_list):
            row_idx = i // 5
            col_idx = i % 5
            x = col_idx * 5
            y_pos = y + row_idx * 6

            panels.append(self._panel(
                pid, f"{module.value.upper()}", "stat",
                [{
                    "expr": f'sum(rate(sesi_agi_module_calls_total{{module="{module.value}"}}[5m]))',
                    "legendFormat": "calls/s",
                }],
                {"h": 5, "w": 5, "x": x, "y": y_pos},
            )); pid += 1

        y += 12

        # Build dashboard
        dashboard = {
            "uid": "sesi-unified-observability-v6",
            "title": "SESI AGI Unified Observability (Alloy + Tempo + OpenSearch + Loki)",
            "tags": ["sesi", "agi", "unified", "alloy", "tempo", "opensearch", "loki"],
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
                        "name": "module",
                        "type": "custom",
                        "options": [{"text": "All", "value": ""}] +
                                   [{"text": m.value, "value": m.value} for m in AGIModule],
                        "current": {"text": "All", "value": ""},
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
                        "name": "Trace Alerts",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "ALERTS{alertname=~'Trace.*'}",
                        "iconColor": "red",
                        "titleFormat": "{{alertname}}",
                    },
                    {
                        "name": "Regime Changes",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "changes(sesi_agi_trace_regime[5m]) > 0",
                        "iconColor": "yellow",
                        "titleFormat": "Regime Change",
                    },
                ],
            },
            "links": [
                {
                    "title": "Tempo UI",
                    "url": "http://tempo-query-frontend:3200/",
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
                {
                    "title": "OpenSearch Dashboards",
                    "url": "http://opensearch-dashboards:5601/",
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
                {
                    "title": "Trace by ID",
                    "url": f"http://tempo-query-frontend:3200/explore?traceId=${{trace_id}}",
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
            ],
        }

        return dashboard

    def write_dashboard(self, output_path: str = "./grafana/unified-observability-v6.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"[Grafana] Unified dashboard written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════════
# [LL1] LOKI LIGHTWEIGHT AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LokiConfig:
    """
    Configuration for Loki lightweight log aggregation.

    HONEST: All parameters are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be validated against real load.
    """
    # Retention per zone (PNDC Axiom 3 proportionality)
    retention_z0: str = "30d"   # Implant: longest
    retention_z1: str = "14d"   # Clinical
    retention_z2: str = "7d"    # Processing
    retention_z3: str = "24h"   # External: shortest

    # Ingestion limits
    max_streams: int = 10000
    max_entries_per_batch: int = 10000
    ingestion_rate_mb: int = 10

    # Storage
    storage_backend: str = "filesystem"  # "filesystem" | "s3" | "gcs"
    storage_path: str = "/var/loki"

    # Index
    index_prefix: str = "index_"
    index_period: str = "24h"


@dataclass
class LokiLogEntry:
    """A single log entry for Loki."""
    timestamp: float
    module: AGIModule
    zone: SecurityZone
    level: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON for Loki ingestion."""
        entry = {
            "timestamp": self.timestamp,
            "module": self.module.value,
            "zone": self.zone.value,
            "level": self.level,
            "message": self.message,
        }
        if self.trace_id:
            entry["trace_id"] = self.trace_id
        if self.span_id:
            entry["span_id"] = self.span_id
        if self.attributes:
            # Sanitize attributes (PNDC Axiom 3)
            sanitized = {}
            for k, v in self.attributes.items():
                if k.lower() in ("password", "secret", "token", "ssn", "credit_card"):
                    continue
                if isinstance(v, (int, float)):
                    sanitized[k] = _safe_unit_scalar(v, name=k, logger=logger, default=0.0)
                else:
                    sanitized[k] = v
            entry.update(sanitized)
        return json.dumps(entry)


class LokiLightweightAggregator:
    """
    [D] Loki lightweight log aggregation for AGI modules.

    Features:
      • Structured JSON logging for all 10 AGI modules
      • Label-based indexing (module, zone, level, trace_id)
      • LogQL query templates for AGI-specific analysis
      • Zone-based retention (PNDC Axiom 3)
      • _safe_unit_scalar sanitization on all values
      • No-Zeno rate limiting on log emission
      • K8s Deployment + Service manifests

    HONEST: Loki is LIGHTWEIGHT; it does not replace OpenSearch
    for full-text search. It is optimized for label-based filtering
    and fast log tailing.
    """

    # Standard LogQL queries
    STANDARD_QUERIES = [
        {
            "name": "agi_errors",
            "query": '{app="sesi-agi"} | json | level="ERROR" or level="CRITICAL"',
            "description": "All ERROR and CRITICAL logs from AGI modules",
        },
        {
            "name": "safety_violations",
            "query": '{app="sesi-agi"} | json | module="safety" | event_type="violation"',
            "description": "Safety violation events",
        },
        {
            "name": "module_logs",
            "query": '{app="sesi-agi"} | json | module="$module"',
            "description": "Logs for a specific AGI module (template variable)",
        },
        {
            "name": "trace_correlation",
            "query": '{app="sesi-agi"} | json | trace_id!="" | line_format "{{.trace_id}} {{.module}} {{.message}}"',
            "description": "Logs with trace_id for trace-log correlation",
        },
        {
            "name": "zone_logs",
            "query": '{app="sesi-agi", zone="$zone"}',
            "description": "Logs for a specific zone (template variable)",
        },
        {
            "name": "reasoning_latency",
            "query": '{app="sesi-agi"} | json | module="reasoning" | line_format "{{.latency_ms}}"',
            "description": "Reasoning engine latency distribution",
        },
        {
            "name": "meta_learning_updates",
            "query": '{app="sesi-agi"} | json | module="meta_learning" | event_type="update"',
            "description": "Meta-learning update events",
        },
        {
            "name": "curiosity_triggers",
            "query": '{app="sesi-agi"} | json | module="curiosity" | event_type="trigger"',
            "description": "Epistemic curiosity trigger events",
        },
    ]

    def __init__(self, cfg: Optional[LokiConfig] = None):
        self.cfg = cfg or LokiConfig()
        self.log_buffer: deque = deque(maxlen=100000)
        self._log_timestamps: deque = deque(maxlen=100000)
        self._module_log_counts: Dict[str, int] = defaultdict(int)

        # No-Zeno parameters
        self.max_logs_per_minute = 60000
        self.min_log_interval_s = 0.001

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_log_rate(self) -> Dict[str, Any]:
        """Check if log emission is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._log_timestamps = deque(
            (t for t in self._log_timestamps if t > cutoff),
            maxlen=100000,
        )

        if len(self._log_timestamps) >= self.max_logs_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
                "current": len(self._log_timestamps),
            }

        if self._log_timestamps:
            time_since_last = now - self._log_timestamps[-1]
            if time_since_last < self.min_log_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                }

        return {"allowed": True}

    # ── Log Emission ─────────────────────────────────────────────────────

    def log_module_event(
        self,
        module: AGIModule,
        zone: SecurityZone,
        level: str,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[LokiLogEntry]:
        """
        Log an AGI module event to Loki.

        Applies:
          1. No-Zeno rate limiting
          2. _safe_unit_scalar sanitization
          3. PNDC Axiom 3 data minimization
        """
        # Check rate limit
        rate_check = self.check_log_rate()
        if not rate_check["allowed"]:
            return None

        entry = LokiLogEntry(
            timestamp=time.time(),
            module=module,
            zone=zone,
            level=level,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {},
        )

        self.log_buffer.append(entry)
        self._log_timestamps.append(time.time())
        self._module_log_counts[module.value] += 1

        return entry

    # ── Module-Specific Loggers ──────────────────────────────────────────

    def log_reasoning(self, goal: str, confidence: float, latency_ms: float,
                      zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                      trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log reasoning engine event."""
        return self.log_module_event(
            module=AGIModule.REASONING,
            zone=zone,
            level="INFO",
            message=f"Reasoning completed for goal: {goal[:100]}",
            trace_id=trace_id,
            attributes={"goal": goal[:200], "confidence": confidence, "latency_ms": latency_ms},
        )

    def log_memory(self, operation: str, key: str, size: int,
                   zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                   trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log memory operation."""
        return self.log_module_event(
            module=AGIModule.MEMORY,
            zone=zone,
            level="INFO",
            message=f"Memory {operation}: {key}",
            trace_id=trace_id,
            attributes={"key": key, "size": size, "operation": operation},
        )

    def log_perception(self, modality: str, content_length: int,
                       zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                       trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log perception event."""
        return self.log_module_event(
            module=AGIModule.PERCEPTION,
            zone=zone,
            level="INFO",
            message=f"Perception processed: {modality}",
            trace_id=trace_id,
            attributes={"modality": modality, "content_length": content_length},
        )

    def log_actuation(self, action: str, status: str,
                      zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                      trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log actuation event."""
        return self.log_module_event(
            module=AGIModule.ACTUATION,
            zone=zone,
            level="INFO",
            message=f"Actuation {action}: {status}",
            trace_id=trace_id,
            attributes={"action": action, "status": status},
        )

    def log_curiosity(self, domain: str, uncertainty: float, triggered: bool,
                      zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                      trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log curiosity event."""
        event_type = "trigger" if triggered else "evaluate"
        return self.log_module_event(
            module=AGIModule.CURIOSITY,
            zone=zone,
            level="INFO",
            message=f"Curiosity {event_type} for domain: {domain}",
            trace_id=trace_id,
            attributes={"domain": domain, "uncertainty": uncertainty, "triggered": triggered, "event_type": event_type},
        )

    def log_synthesis(self, domain: str, abstraction_level: int,
                      zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                      trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log synthesis event."""
        return self.log_module_event(
            module=AGIModule.SYNTHESIS,
            zone=zone,
            level="INFO",
            message=f"Concept synthesis for domain: {domain}",
            trace_id=trace_id,
            attributes={"domain": domain, "abstraction_level": abstraction_level},
        )

    def log_reflection(self, goal: str, success: bool, error_margin: float,
                       zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                       trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log reflection event."""
        return self.log_module_event(
            module=AGIModule.REFLECTION,
            zone=zone,
            level="INFO" if success else "WARNING",
            message=f"Reflection on goal: {goal[:50]} - {'success' if success else 'failure'}",
            trace_id=trace_id,
            attributes={"goal": goal[:200], "success": success, "error_margin": error_margin},
        )

    def log_safety(self, action: str, result: str,
                   violation_type: Optional[str] = None,
                   zone: SecurityZone = SecurityZone.Z1_CLINICAL,
                   trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log safety verification event."""
        level = "WARNING" if result == "FAIL" else "INFO"
        event_type = "violation" if result == "FAIL" else "verified"
        attributes = {"action": action, "result": result, "event_type": event_type}
        if violation_type:
            attributes["violation_type"] = violation_type

        return self.log_module_event(
            module=AGIModule.SAFETY,
            zone=zone,
            level=level,
            message=f"Safety check for {action}: {result}",
            trace_id=trace_id,
            attributes=attributes,
        )

    def log_substrate(self, priority: str, allocated_units: float,
                      zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                      trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log substrate allocation event."""
        return self.log_module_event(
            module=AGIModule.SUBSTRATE,
            zone=zone,
            level="INFO",
            message=f"Substrate allocated: {allocated_units:.2f} units (priority={priority})",
            trace_id=trace_id,
            attributes={"priority": priority, "allocated_units": allocated_units},
        )

    def log_meta_learning(self, domain: str, loss_before: float, loss_after: float,
                          zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                          trace_id: Optional[str] = None) -> Optional[LokiLogEntry]:
        """Log meta-learning event."""
        return self.log_module_event(
            module=AGIModule.META_LEARNING,
            zone=zone,
            level="INFO",
            message=f"Meta-learning update for domain: {domain}",
            trace_id=trace_id,
            attributes={
                "domain": domain,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "improvement": loss_before - loss_after,
                "event_type": "update",
            },
        )

    # ── Loki Configuration Generation ────────────────────────────────────

    def generate_loki_config(self) -> Dict[str, Any]:
        """Generate complete Loki configuration."""
        storage_config = self._get_storage_config()

        return {
            "auth_enabled": False,
            "server": {
                "http_listen_port": 3100,
                "grpc_listen_port": 9095,
                "log_level": "info",
            },
            "distributor": {
                "ring": {
                    "kvstore": {"store": "inmemory"},
                },
            },
            "ingester": {
                "wal": {
                    "enabled": True,
                    "dir": f"{self.cfg.storage_path}/wal",
                },
                "lifecycler": {
                    "ring": {
                        "kvstore": {"store": "inmemory"},
                        "replication_factor": 1,
                    },
                },
                "chunk_idle_period": "30m",
                "chunk_block_size": 262144,
                "chunk_retain_period": "1m",
                "max_transfer_retries": 0,
            },
            "limits_config": {
                "enforce_metric_name": False,
                "reject_old_samples": True,
                "reject_old_samples_max_age": "168h",
                "max_entries_limit_per_query": 5000,
                "max_streams_per_user": self.cfg.max_streams,
                "ingestion_rate_mb": self.cfg.ingestion_rate_mb,
                "ingestion_burst_size_mb": self.cfg.ingestion_rate_mb * 2,
            },
            "schema_config": {
                "configs": [{
                    "from": "2020-10-24",
                    "store": "boltdb-shipper",
                    "object_store": self.cfg.storage_backend,
                    "schema": "v11",
                    "index": {
                        "prefix": self.cfg.index_prefix,
                        "period": self.cfg.index_period,
                    },
                }],
            },
            "storage_config": {
                "boltdb_shipper": {
                    "active_index_directory": f"{self.cfg.storage_path}/boltdb-shipper-active",
                    "cache_location": f"{self.cfg.storage_path}/boltdb-shipper-cache",
                    "cache_ttl": "24h",
                    "shared_store": self.cfg.storage_backend,
                },
                **storage_config,
            },
            "chunk_store_config": {
                "max_look_back_period": "0s",
            },
            "table_manager": {
                "retention_deletes_enabled": True,
                "retention_period": "720h",
            },
            "compactor": {
                "working_directory": f"{self.cfg.storage_path}/compactor",
                "shared_store": self.cfg.storage_backend,
                "compaction_interval": "10m",
            },
            "ruler": {
                "storage": {
                    "type": "local",
                    "local": {"directory": f"{self.cfg.storage_path}/rules"},
                },
                "rule_path": f"{self.cfg.storage_path}/rules-temp",
                "enable_api": True,
            },
        }

    def _get_storage_config(self) -> Dict[str, Any]:
        """Get storage backend configuration."""
        if self.cfg.storage_backend == "s3":
            return {
                "aws": {
                    "s3": "s3://sesi-loki-logs",
                    "region": "us-east-1",
                },
            }
        elif self.cfg.storage_backend == "gcs":
            return {
                "gcs": {
                    "bucket_name": "sesi-loki-logs",
                },
            }
        else:  # filesystem
            return {
                "filesystem": {
                    "directory": f"{self.cfg.storage_path}/chunks",
                },
            }

    def generate_k8s_deployment(self) -> Dict[str, Any]:
        """Generate K8s Deployment for Loki."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "loki",
                "namespace": "sesi-agi",
                "labels": {"app.kubernetes.io/name": "loki"},
            },
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app.kubernetes.io/name": "loki"}},
                "template": {
                    "metadata": {
                        "labels": {"app.kubernetes.io/name": "loki"},
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "3100",
                        },
                    },
                    "spec": {
                        "containers": [{
                            "name": "loki",
                            "image": "grafana/loki:2.9.0",
                            "args": ["-config.file=/etc/loki/loki.yaml"],
                            "ports": [
                                {"name": "http", "containerPort": 3100},
                                {"name": "grpc", "containerPort": 9095},
                            ],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "1Gi"},
                                "limits": {"cpu": "2000m", "memory": "4Gi"},
                            },
                            "volumeMounts": [
                                {"name": "config", "mountPath": "/etc/loki", "readOnly": True},
                                {"name": "data", "mountPath": "/var/loki"},
                            ],
                        }],
                        "volumes": [
                            {"name": "config", "configMap": {"name": "loki-config"}},
                            {"name": "data", "persistentVolumeClaim": {"claimName": "loki-data"}},
                        ],
                    },
                },
            },
        }

    def generate_logql_queries(self) -> List[Dict[str, Any]]:
        """Generate LogQL query templates."""
        return self.STANDARD_QUERIES.copy()

    def write_all(self, output_dir: str = "./loki") -> Dict[str, str]:
        """Write all Loki artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Main config
        config = self.generate_loki_config()
        config_path = out_path / "loki.yaml"
        if HAS_YAML:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        files["config"] = str(config_path)

        # K8s Deployment
        k8s = self.generate_k8s_deployment()
        k8s_path = out_path / "k8s-deployment.yaml"
        if HAS_YAML:
            with open(k8s_path, "w") as f:
                yaml.dump(k8s, f, default_flow_style=False)
        else:
            with open(k8s_path, "w") as f:
                json.dump(k8s, f, indent=2)
        files["k8s_deployment"] = str(k8s_path)

        # LogQL queries
        queries = self.generate_logql_queries()
        queries_path = out_path / "logql-queries.json"
        with open(queries_path, "w") as f:
            json.dump(queries, f, indent=2)
        files["logql_queries"] = str(queries_path)

        logger.info(f"[Loki] Generated {len(files)} artifacts in {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get Loki status."""
        return {
            "storage_backend": self.cfg.storage_backend,
            "buffer_size": len(self.log_buffer),
            "total_logs": sum(self._module_log_counts.values()),
            "module_log_counts": dict(self._module_log_counts),
            "delta_min": self.compute_delta_min(),
            "max_logs_per_minute": self.max_logs_per_minute,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIUnifiedObservabilityV6:
    """
    [D] Complete unified observability suite v6.

    Integrates:
      • Trace-based Alertmanager
      • Unified Grafana dashboard (Alloy/Tempo/OpenSearch/Loki)
      • Loki lightweight log aggregation
    """

    def __init__(
        self,
        zone_configs: Optional[Dict[SecurityZone, ZoneAlertConfig]] = None,
        loki_cfg: Optional[LokiConfig] = None,
    ):
        self.alertmanager = TraceBasedAlertmanager(zone_configs)
        self.dashboard = UnifiedGrafanaDashboard()
        self.loki = LokiLightweightAggregator(loki_cfg)

    def simulate_agi_cycle(self) -> Dict[str, Any]:
        """
        Simulate a single AGI autonomous learning cycle with full observability.
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        cycle_start = time.time()

        # 1. Perception
        self.loki.log_perception(
            modality="multimodal_code_and_text",
            content_length=100,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 2. Curiosity
        uncertainty = random.uniform(0.3, 0.9)
        triggered = uncertainty > 0.5
        self.loki.log_curiosity(
            domain="quantum_topology",
            uncertainty=uncertainty,
            triggered=triggered,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 3. Reasoning
        confidence = random.uniform(0.85, 0.99)
        reasoning_latency = random.uniform(10, 500)
        self.loki.log_reasoning(
            goal="Investigate quantum topology",
            confidence=confidence,
            latency_ms=reasoning_latency,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # Evaluate trace for alerting
        self.alertmanager.evaluate_trace(
            zone=SecurityZone.Z2_PROCESSING,
            module=AGIModule.REASONING,
            trace_id=trace_id,
            duration_ms=reasoning_latency,
            error=False,
            baseline_duration_ms=100.0,
        )

        # 4. Memory operations
        self.loki.log_memory(
            operation="store", key="quantum_topology", size=1,
            zone=SecurityZone.Z2_PROCESSING, trace_id=trace_id,
        )
        self.loki.log_memory(
            operation="query", key="quantum_topology", size=1,
            zone=SecurityZone.Z2_PROCESSING, trace_id=trace_id,
        )

        # 5. Synthesis
        self.loki.log_synthesis(
            domain="quantum_topology",
            abstraction_level=2,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 6. Safety check
        self.loki.log_safety(
            action="execute_experiment",
            result="PASS",
            zone=SecurityZone.Z1_CLINICAL,
            trace_id=trace_id,
        )

        # 7. Actuation
        self.loki.log_actuation(
            action="execute_experiment",
            status="COMPLETED",
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 8. Substrate allocation
        self.loki.log_substrate(
            priority="HIGH",
            allocated_units=0.5,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 9. Reflection
        error_margin = random.uniform(0.01, 0.15)
        success = error_margin < 0.1
        self.loki.log_reflection(
            goal="Investigate quantum topology",
            success=success,
            error_margin=error_margin,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 10. Meta-learning
        if not success:
            self.loki.log_meta_learning(
                domain="quantum_topology",
                loss_before=error_margin,
                loss_after=error_margin * 0.5,
                zone=SecurityZone.Z2_PROCESSING,
                trace_id=trace_id,
            )

        cycle_latency = time.time() - cycle_start

        return {
            "trace_id": trace_id,
            "cycle_latency_s": cycle_latency,
            "logs_emitted": len(self.loki.log_buffer),
            "alerts_generated": len(self.alertmanager._alerts),
        }

    def generate_all_artifacts(self, output_dir: str = "./sesi_unified_v6") -> Dict[str, Any]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Alertmanager
        results["trace_alert_rules"] = self.alertmanager.write_alert_rules(
            str(out_path / "prometheus" / "trace-alerts.yaml")
        )
        results["alertmanager_config"] = self.alertmanager.write_alertmanager_config(
            str(out_path / "alertmanager" / "alertmanager.yml")
        )

        # Grafana dashboard
        results["unified_dashboard"] = self.dashboard.write_dashboard(
            str(out_path / "grafana" / "unified-observability-v6.json")
        )

        # Loki
        results["loki"] = self.loki.write_all(str(out_path / "loki"))

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "alertmanager": self.alertmanager.get_status(),
            "loki": self.loki.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Trace Alertmanager + Unified Grafana + Loki — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIUnifiedObservabilityV6()

    # ── Test 1: Trace-Based Alertmanager ─────────────────────────────────
    print("\n  ── Trace-Based Alertmanager ──")
    am_status = suite.alertmanager.get_status()
    print(f"    Zones configured     : {am_status['zones_configured']}")
    print(f"    Alert rules          : {am_status['alert_rules']}")
    print(f"    Inhibition rules     : {am_status['inhibition_rules']}")
    print(f"    δ_min                : {am_status['delta_min']:.6f}")

    # Show zone thresholds
    print("\n    Zone Thresholds (PNDC Axiom 3 proportionality):")
    for zone, cfg in suite.alertmanager.zone_configs.items():
        print(f"      {zone.value:20s}: latency_warn={cfg.latency_warning_s}s, "
              f"err_warn={cfg.error_rate_warning*100:.1f}%, "
              f"severity={cfg.default_severity.value}")

    # ── Test 2: Unified Grafana Dashboard ────────────────────────────────
    print("\n  ── Unified Grafana Dashboard ──")
    dashboard = suite.dashboard.generate_dashboard()
    print(f"    Dashboard title      : {dashboard['title']}")
    print(f"    Total panels         : {len(dashboard['panels'])}")
    print(f"    Template variables   : {len(dashboard['templating']['list'])}")
    print(f"    Annotations          : {len(dashboard['annotations']['list'])}")
    print(f"    External links       : {len(dashboard['links'])}")

    # ── Test 3: Loki Lightweight ─────────────────────────────────────────
    print("\n  ── Loki Lightweight Aggregator ──")
    loki_status = suite.loki.get_status()
    print(f"    Storage backend      : {loki_status['storage_backend']}")
    print(f"    Max logs/min         : {loki_status['max_logs_per_minute']}")
    print(f"    δ_min                : {loki_status['delta_min']:.6f}")

    # LogQL queries
    queries = suite.loki.generate_logql_queries()
    print(f"    LogQL queries        : {len(queries)}")
    for q in queries[:3]:
        print(f"      • {q['name']}: {q['description'][:50]}...")

    # ── Test 4: Simulate AGI Cycle ───────────────────────────────────────
    print("\n  ── Simulating AGI Cycles ──")
    for i in range(10):
        cycle = suite.simulate_agi_cycle()
        if i % 5 == 0:
            print(f"    Cycle {i+1:2d}             : trace={cycle['trace_id'][:12]}... "
                  f"latency={cycle['cycle_latency_s']*1000:.1f}ms "
                  f"logs={cycle['logs_emitted']} alerts={cycle['alerts_generated']}")

    # ── Test 5: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_unified_v6")
    print(f"    Artifacts generated  : {len(artifacts)}")
    for name, path in artifacts.items():
        if isinstance(path, dict):
            print(f"      {name:25s}: {len(path)} files")
        else:
            print(f"      {name:25s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Alert rules          : {status['alertmanager']['alert_rules']}")
    print(f"    Total alerts         : {status['alertmanager']['total_alerts']}")
    print(f"    Loki total logs      : {status['loki']['total_logs']}")
    print(f"    Loki buffer size     : {status['loki']['buffer_size']}")
    print(f"    Loki module counts   : {status['loki']['module_log_counts']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Trace-based alerts are CONFIGURATION, not autonomous response")
    print("  • Unified dashboard is VISUALIZATION, not decision-making")
    print("  • Loki is LIGHTWEIGHT; it does not replace OpenSearch full-text")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all values")
    print("  • PNDC Axiom 3 data minimization in all logs/alerts")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
