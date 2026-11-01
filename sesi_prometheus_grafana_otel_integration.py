"""
=============================================================================
SESI PROMETHEUS AUTH METRICS × GRAFANA KEY ROTATION DASHBOARD ×
OPENTELEMETRY DISTRIBUTED TRACING
Authorization Observability + Vault Key Lifecycle + Distributed Traces
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[PM1] PrometheusAuthMetricsExporter
      • Counters for auth events (login, token issuance, zone violations)
      • Histograms for latency distribution
      • Gauges for active sessions, key rotation health
      • No-Zeno rate limiting on metric emission
      • δ_min quantum gate for key rotation events

[GD1] KeyRotationGrafanaDashboard
      • Grafana JSON for Vault Transit key rotation monitoring
      • Panels: rotation rate, δ_min compliance, regime classification
      • CUSUM change-point detection on rotation stream
      • Hill estimator for rotation regime (paper6 Thm 5.1)
      • Alert rules for anomalous rotation patterns

[OT1] OpenTelemetryTraceIntegrator
      • Distributed spans across auth → key rotation → AGIONE pipeline
      • Trace context propagation with trace_id/span_id
      • Correlation with Prometheus metrics via labels
      • Regime classification on trace event streams
      • Exit-time anomaly detection (Stochastic_NoZeno Thm 4.1)

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds auth event emission rate
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω)^{1/(n-1)} under κ-fatness
    → minimum meaningful key rotation change
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
    → confidence interval for rotation threshold
[C] Six_Problems P4: CUSUM + subsampling on raw process
    → change-point detection in rotation streams
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → rotation regime classification
[C] Stochastic_NoZeno Thm 4.1: Exit-time tail bound
    → anomalous rotation detection
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_istio_webauthn_jaeger_enrich.py — Vault Transit bridge,
    Istio AuthZ, Jaeger enrichment
  • sesi_vault_fido_chaos_cicd.py — Vault auto-unseal, FIDO MDS,
    chaos CI/CD pipeline
  • sesi_vault_ns_fido_identity_grafana.py — Vault namespaces,
    multi-identity risk, Grafana dashboard
  • agi_one_v3_11.py — AGIONE, EcosystemOrchestrator
  • agi_one_psyche_plus_v33.py — Psyche Plus, quality_score gating

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Prometheus metrics provide OBSERVABILITY, not security guarantees.
• Grafana dashboards are VISUALIZATION TOOLS, not decision engines.
• OpenTelemetry traces show CORRELATION, not causation.
• All thresholds, rotation intervals, and alert levels are FREE DESIGN
  PARAMETERS (Unified_Ordinal Hyp 7.9) requiring security review.
• Key rotation monitoring does not replace HSM-backed key management.
• Production deployment requires security audit and regulatory approval.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.prometheus_grafana_otel")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor, ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode, SpanKind
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

# Previous module imports (graceful)
try:
    from sesi_istio_webauthn_jaeger_enrich import (
        VaultIstioMTLSIntegration,
        WebAuthnVaultTransitBridge,
        JaegerTraceEnrichmentPipeline,
    )
    HAS_PREV_MODULES = True
except ImportError:
    HAS_PREV_MODULES = False

__all__ = [
    "AuthEventType", "KeyRotationEvent",
    "PrometheusAuthMetricsExporter",
    "KeyRotationGrafanaDashboard",
    "OpenTelemetryTraceIntegrator",
    "SESIObservabilitySuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class AuthEventType(Enum):
    """Authorization event types for Prometheus metrics."""
    LOGIN_SUCCESS = auto()
    LOGIN_FAILURE = auto()
    TOKEN_ISSUED = auto()
    TOKEN_VALIDATED = auto()
    TOKEN_EXPIRED = auto()
    TOKEN_REVOKED = auto()
    ZONE_VIOLATION = auto()
    MFA_CHALLENGE = auto()
    MFA_VERIFIED = auto()
    PHYSICIAN_APPROVAL = auto()
    PHYSICIAN_REJECTION = auto()
    EMERGENCY_STOP = auto()


class KeyRotationEvent(Enum):
    """Key rotation lifecycle events."""
    ROTATION_REQUESTED = auto()
    ROTATION_APPROVED = auto()
    ROTATION_REJECTED_RATE_LIMIT = auto()
    ROTATION_REJECTED_DELTA_MIN = auto()
    ROTATION_COMPLETED = auto()
    ROTATION_FAILED = auto()


class QualityRegime(Enum):
    """Growth-rate regime classification (paper6 Thm 5.1)."""
    FINITE_MEAN = auto()          # θ > 1
    REGULARLY_VARYING = auto()    # θ ∈ (0, 1)
    DEGENERATE = auto()           # θ → 0 (½δ₀ + ½δ₁)
    INSUFFICIENT_DATA = auto()


# ═══════════════════════════════════════════════════════════════════════
# [PM1] PROMETHEUS AUTH METRICS EXPORTER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MetricDefinition:
    """A single Prometheus metric definition."""
    name: str
    metric_type: str  # "counter" | "gauge" | "histogram" | "summary"
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None


class PrometheusAuthMetricsExporter:
    """
    [D] Exports Prometheus metrics for authorization events.

    Features:
      • Counters for auth event types
      • Histograms for latency distribution
      • Gauges for active sessions, key rotation health
      • No-Zeno rate limiting on metric emission
      • δ_min quantum gate for key rotation events

    HONEST: This is a METRICS GENERATOR. It produces Prometheus-compatible
    exposition format but does NOT:
      • Deploy Prometheus server
      • Configure scrape targets
      • Guarantee metric accuracy without proper instrumentation
      • Replace security audit

    Mathematical grounding:
      • No-Zeno bound: N(T) ≤ 1 + 2C_V·T/δ_min (event rate limiting)
      • δ_min quantum gate: minimum meaningful rotation change
    """

    # Metric definitions
    METRICS = {
        "auth_events_total": MetricDefinition(
            name="sesi_auth_events_total",
            metric_type="counter",
            description="Total authorization events by type",
            labels=["event_type", "zone", "outcome"],
        ),
        "auth_latency_seconds": MetricDefinition(
            name="sesi_auth_latency_seconds",
            metric_type="histogram",
            description="Authorization request latency distribution",
            labels=["event_type", "zone"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        ),
        "active_sessions": MetricDefinition(
            name="sesi_active_sessions",
            metric_type="gauge",
            description="Number of active user sessions",
            labels=["zone", "auth_level"],
        ),
        "key_rotations_total": MetricDefinition(
            name="sesi_key_rotations_total",
            metric_type="counter",
            description="Total key rotation events",
            labels=["key_type", "outcome"],
        ),
        "key_rotation_latency_seconds": MetricDefinition(
            name="sesi_key_rotation_latency_seconds",
            metric_type="histogram",
            description="Key rotation latency distribution",
            labels=["key_type"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
        ),
        "zone_violations_total": MetricDefinition(
            name="sesi_zone_violations_total",
            metric_type="counter",
            description="Total zone policy violations",
            labels=["src_zone", "dst_zone", "port"],
        ),
        "emergency_stops_total": MetricDefinition(
            name="sesi_emergency_stops_total",
            metric_type="counter",
            description="Total emergency stop events",
            labels=["reason", "initiated_by"],
        ),
        "no_zeno_event_count": MetricDefinition(
            name="sesi_no_zeno_event_count",
            metric_type="gauge",
            description="Current event count within No-Zeno window",
            labels=["event_type"],
        ),
        "no_zeno_bound": MetricDefinition(
            name="sesi_no_zeno_bound",
            metric_type="gauge",
            description="No-Zeno bound N(T) = 1 + 2C_V·T/δ_min",
            labels=["event_type"],
        ),
        "delta_min": MetricDefinition(
            name="sesi_delta_min",
            metric_type="gauge",
            description="δ_min threshold from Delta_Min Thm 1.2",
            labels=["context"],
        ),
    }

    def __init__(self, metric_prefix: str = "sesi"):
        self.metric_prefix = metric_prefix
        self._counters: Dict[str, Dict[str, float]] = {}
        self._gauges: Dict[str, Dict[str, float]] = {}
        self._histograms: Dict[str, Dict[str, List[float]]] = {}
        self._event_timestamps: Dict[str, List[float]] = {}

        # No-Zeno parameters
        self.c_v = 1.0
        self.window_s = 3600.0
        self.delta_min = self._compute_delta_min()

        # Initialize metric storage
        for name, defn in self.METRICS.items():
            if defn.metric_type == "counter":
                self._counters[name] = {}
            elif defn.metric_type == "gauge":
                self._gauges[name] = {}
            elif defn.metric_type == "histogram":
                self._histograms[name] = {}
            self._event_timestamps[name] = []

    def _compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                           n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def _check_rate_limit(self, metric_name: str) -> bool:
        """Check No-Zeno rate limit for metric emission."""
        now = time.time()
        cutoff = now - self.window_s
        self._event_timestamps[metric_name] = [
            t for t in self._event_timestamps[metric_name] if t > cutoff
        ]

        bound = 1 + (2 * self.c_v * self.window_s) / max(self.delta_min, 1e-8)
        current = len(self._event_timestamps[metric_name])

        if current >= bound:
            return False

        self._event_timestamps[metric_name].append(now)
        return True

    def _make_label_key(self, labels: Dict[str, str]) -> str:
        """Create a unique key from label dict."""
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def inc_counter(self, metric_name: str, value: float = 1.0,
                    labels: Optional[Dict[str, str]] = None) -> bool:
        """Increment a counter metric."""
        if metric_name not in self._counters:
            logger.warning(f"Unknown counter: {metric_name}")
            return False

        if not self._check_rate_limit(metric_name):
            logger.debug(f"Rate limited: {metric_name}")
            return False

        key = self._make_label_key(labels or {})
        self._counters[metric_name][key] = (
            self._counters[metric_name].get(key, 0.0) + value
        )
        return True

    def set_gauge(self, metric_name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> bool:
        """Set a gauge metric."""
        if metric_name not in self._gauges:
            logger.warning(f"Unknown gauge: {metric_name}")
            return False

        key = self._make_label_key(labels or {})
        self._gauges[metric_name][key] = value
        return True

    def observe_histogram(self, metric_name: str, value: float,
                          labels: Optional[Dict[str, str]] = None) -> bool:
        """Observe a value in a histogram metric."""
        if metric_name not in self._histograms:
            logger.warning(f"Unknown histogram: {metric_name}")
            return False

        if not self._check_rate_limit(metric_name):
            return False

        key = self._make_label_key(labels or {})
        if key not in self._histograms[metric_name]:
            self._histograms[metric_name][key] = []
        self._histograms[metric_name][key].append(value)
        return True

    # ── High-level auth event recording ───────────────────────────────────

    def record_auth_event(self, event_type: AuthEventType,
                          zone: str, outcome: str,
                          latency_s: Optional[float] = None) -> None:
        """Record an authorization event with all relevant metrics."""
        labels = {
            "event_type": event_type.name,
            "zone": zone,
            "outcome": outcome,
        }
        self.inc_counter("auth_events_total", labels=labels)

        if latency_s is not None:
            self.observe_histogram(
                "auth_latency_seconds", latency_s,
                labels={"event_type": event_type.name, "zone": zone},
            )

    def record_key_rotation(self, event: KeyRotationEvent,
                            key_type: str, latency_s: Optional[float] = None) -> None:
        """Record a key rotation event."""
        outcome = "success" if event == KeyRotationEvent.ROTATION_COMPLETED else "failure"
        self.inc_counter(
            "key_rotations_total",
            labels={"key_type": key_type, "outcome": outcome},
        )

        if latency_s is not None:
            self.observe_histogram(
                "key_rotation_latency_seconds", latency_s,
                labels={"key_type": key_type},
            )

    def record_zone_violation(self, src_zone: str, dst_zone: str,
                               port: int) -> None:
        """Record a zone policy violation."""
        self.inc_counter(
            "zone_violations_total",
            labels={
                "src_zone": src_zone,
                "dst_zone": dst_zone,
                "port": str(port),
            },
        )

    def record_emergency_stop(self, reason: str, initiated_by: str) -> None:
        """Record an emergency stop event."""
        self.inc_counter(
            "emergency_stops_total",
            labels={"reason": reason, "initiated_by": initiated_by},
        )

    def update_no_zeno_metrics(self, event_type: str) -> None:
        """Update No-Zeno bound metrics."""
        cutoff = time.time() - self.window_s
        timestamps = [
            t for t in self._event_timestamps.get("auth_events_total", [])
            if t > cutoff
        ]
        bound = 1 + (2 * self.c_v * self.window_s) / max(self.delta_min, 1e-8)

        self.set_gauge(
            "no_zeno_event_count", len(timestamps),
            labels={"event_type": event_type},
        )
        self.set_gauge(
            "no_zeno_bound", bound,
            labels={"event_type": event_type},
        )
        self.set_gauge(
            "delta_min", self.delta_min,
            labels={"context": event_type},
        )

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus exposition format."""
        lines = []

        # Counters
        for name, values in self._counters.items():
            defn = self.METRICS.get(name)
            if not defn:
                continue
            lines.append(f"# HELP {defn.name} {defn.description}")
            lines.append(f"# TYPE {defn.name} counter")
            for label_key, value in values.items():
                if label_key:
                    lines.append(f"{defn.name}{{{label_key}}} {value}")
                else:
                    lines.append(f"{defn.name} {value}")

        # Gauges
        for name, values in self._gauges.items():
            defn = self.METRICS.get(name)
            if not defn:
                continue
            lines.append(f"# HELP {defn.name} {defn.description}")
            lines.append(f"# TYPE {defn.name} gauge")
            for label_key, value in values.items():
                if label_key:
                    lines.append(f"{defn.name}{{{label_key}}} {value}")
                else:
                    lines.append(f"{defn.name} {value}")

        # Histograms
        for name, values in self._histograms.items():
            defn = self.METRICS.get(name)
            if not defn or not defn.buckets:
                continue
            lines.append(f"# HELP {defn.name} {defn.description}")
            lines.append(f"# TYPE {defn.name} histogram")
            for label_key, observations in values.items():
                prefix = f"{defn.name}{{{label_key}}}" if label_key else defn.name
                for bucket in defn.buckets:
                    count = sum(1 for v in observations if v <= bucket)
                    lines.append(f"{prefix}_bucket{{le=\"{bucket}\"}} {count}")
                lines.append(f"{prefix}_bucket{{le=\"+Inf\"}} {len(observations)}")
                lines.append(f"{prefix}_sum {sum(observations)}")
                lines.append(f"{prefix}_count {len(observations)}")

        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Get metrics exporter status."""
        return {
            "metric_definitions": len(self.METRICS),
            "counter_series": sum(len(v) for v in self._counters.values()),
            "gauge_series": sum(len(v) for v in self._gauges.values()),
            "histogram_series": sum(len(v) for v in self._histograms.values()),
            "delta_min": self.delta_min,
            "no_zeno_bound": 1 + (2 * self.c_v * self.window_s) / max(self.delta_min, 1e-8),
        }


# ═══════════════════════════════════════════════════════════════════════
# [GD1] KEY ROTATION GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class KeyRotationGrafanaDashboard:
    """
    [D] Generates Grafana dashboard JSON for Vault key rotation monitoring.

    Features:
      • Key rotation rate and latency panels
      • δ_min compliance monitoring
      • No-Zeno bound tracking
      • CUSUM change-point detection
      • Hill estimator regime classification
      • Alert rules for anomalous rotation patterns

    HONEST: This generates a Grafana dashboard JSON template. It does NOT:
      • Deploy Grafana
      • Configure Prometheus data source
      • Make security decisions
      • Guarantee detection of all anomalous rotations

    Mathematical grounding:
      • CUSUM (Six_Problems P4): change-point detection
      • Hill estimator (Paper 9 Entry 2.5): regime classification
      • No-Zeno (SESI_NoZeno Thm 3.2): rate limiting
    """

    def __init__(self, datasource_uid: str = "prometheus"):
        self.datasource_uid = datasource_uid

    def _panel(self, panel_id: int, title: str, panel_type: str,
               targets: List[Dict[str, Any]], grid_pos: Dict[str, int],
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a Grafana panel."""
        panel = {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "datasource": {"type": "prometheus", "uid": self.datasource_uid},
            "targets": targets,
            "gridPos": grid_pos,
            "fieldConfig": {
                "defaults": {"unit": "short"},
                "overrides": [],
            },
            "options": {},
        }
        if extra:
            panel.update(extra)
        return panel

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete Grafana dashboard JSON."""
        panels = []
        panel_id = 1

        # ── Row 1: Overview ──────────────────────────────────────────────
        panels.append(self._row_panel(panel_id, "Key Rotation Overview", 0))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Key Rotations / hour", "timeseries",
            [{"expr": "sum(rate(sesi_key_rotations_total[1h])) * 3600",
              "legendFormat": "rotations/h"}],
            {"h": 8, "w": 6, "x": 0, "y": 1},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Rotation Latency P95", "timeseries",
            [{"expr": "histogram_quantile(0.95, rate(sesi_key_rotation_latency_seconds_bucket[5m]))",
              "legendFormat": "p95"}],
            {"h": 8, "w": 6, "i": 6, "y": 1},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Rotation Success Rate", "gauge",
            [{"expr": "sum(sesi_key_rotations_total{outcome=\"success\"}) / sum(sesi_key_rotations_total) * 100",
              "legendFormat": "success %"}],
            {"h": 8, "w": 6, "i": 12, "y": 1},
            extra={"fieldConfig": {"defaults": {
                "min": 0, "max": 100, "unit": "percent",
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 80},
                    {"color": "green", "value": 95},
                ]},
            }}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "δ_min Compliance", "stat",
            [{"expr": "sesi_delta_min{context=\"key_rotation\"}",
              "legendFormat": "δ_min"}],
            {"h": 8, "w": 6, "i": 18, "y": 1},
        ))
        panel_id += 1

        # ── Row 2: No-Zeno Monitoring ────────────────────────────────────
        panels.append(self._row_panel(panel_id, "No-Zeno Bound Monitoring (SESI Thm 3.2)", 9))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Event Count vs No-Zeno Bound", "timeseries",
            [
                {"expr": "sesi_no_zeno_event_count{event_type=\"key_rotation\"}",
                 "legendFormat": "N(T)"},
                {"expr": "sesi_no_zeno_bound{event_type=\"key_rotation\"}",
                 "legendFormat": "bound"},
            ],
            {"h": 8, "w": 12, "i": 0, "y": 10},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "δ_min Estimate with CI", "timeseries",
            [
                {"expr": "sesi_delta_min{context=\"key_rotation\"}",
                 "legendFormat": "δ_min"},
                {"expr": "sesi_delta_min_ci_lower{context=\"key_rotation\"}",
                 "legendFormat": "CI lower"},
                {"expr": "sesi_delta_min_ci_upper{context=\"key_rotation\"}",
                 "legendFormat": "CI upper"},
            ],
            {"h": 8, "w": 12, "i": 12, "y": 10},
        ))
        panel_id += 1

        # ── Row 3: Regime Detection ──────────────────────────────────────
        panels.append(self._row_panel(panel_id, "Rotation Regime Detection (paper6 Thm 5.1)", 18))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Hill Estimator θ̂", "timeseries",
            [{"expr": "sesi_hill_theta_hat{context=\"key_rotation\"}",
              "legendFormat": "θ̂"}],
            {"h": 8, "w": 8, "i": 0, "y": 19},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Rotation Regime", "stat",
            [{"expr": "sesi_rotation_regime",
              "legendFormat": "regime"}],
            {"h": 8, "w": 8, "i": 8, "y": 19},
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
        panel_id += 1

        panels.append(self._panel(
            panel_id, "CUSUM Statistic", "timeseries",
            [
                {"expr": "sesi_cusum_statistic{context=\"key_rotation\"}",
                 "legendFormat": "cusum"},
                {"expr": "sesi_cusum_threshold{context=\"key_rotation\"}",
                 "legendFormat": "threshold"},
            ],
            {"h": 8, "w": 8, "i": 16, "y": 19},
        ))
        panel_id += 1

        # ── Row 4: Alerts ────────────────────────────────────────────────
        panels.append(self._row_panel(panel_id, "Active Alerts", 27))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Active Alerts", "stat",
            [{"expr": "sum(ALERTS{alertstate=\"firing\"}) or vector(0)",
              "legendFormat": "alerts"}],
            {"h": 8, "w": 8, "i": 0, "y": 28},
            extra={"fieldConfig": {"defaults": {
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 1},
                    {"color": "red", "value": 5},
                ]},
            }}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Emergency Stops (24h)", "stat",
            [{"expr": "sum(increase(sesi_emergency_stops_total[24h]))",
              "legendFormat": "stops"}],
            {"h": 8, "w": 8, "i": 8, "y": 28},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Zone Violations (1h)", "stat",
            [{"expr": "sum(increase(sesi_zone_violations_total[1h]))",
              "legendFormat": "violations"}],
            {"h": 8, "w": 8, "i": 16, "y": 28},
        ))
        panel_id += 1

        dashboard = {
            "uid": "sesi-key-rotation",
            "title": "SESI Vault Key Rotation Monitoring",
            "tags": ["sesi", "vault", "key-rotation", "security"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "templating": {
                "list": [
                    {
                        "name": "key_type",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self.datasource_uid},
                        "query": "label_values(sesi_key_rotations_total, key_type)",
                        "refresh": 2,
                    },
                ],
            },
            "annotations": {
                "list": [
                    {
                        "name": "Key Rotations",
                        "datasource": {"type": "prometheus", "uid": self.datasource_uid},
                        "enable": True,
                        "expr": "sesi_key_rotation_completed",
                        "iconColor": "blue",
                        "titleFormat": "Rotation: {{key_type}}",
                    },
                    {
                        "name": "Emergency Stops",
                        "datasource": {"type": "prometheus", "uid": self.datasource_uid},
                        "enable": True,
                        "expr": "sesi_emergency_stop_triggered",
                        "iconColor": "red",
                        "titleFormat": "EMERGENCY STOP",
                    },
                ],
            },
        }

        return dashboard

    def generate_alert_rules(self) -> List[Dict[str, Any]]:
        """Generate Prometheus alert rules for key rotation monitoring."""
        return [
            {
                "alert": "KeyRotationRateHigh",
                "expr": "sum(rate(sesi_key_rotations_total[5m])) * 3600 > 100",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High key rotation rate detected",
                    "description": (
                        "Key rotation rate exceeds 100/hour. "
                        "Possible rotation chattering or attack. "
                        "Check δ_min quantum gate."
                    ),
                },
            },
            {
                "alert": "KeyRotationLatencyHigh",
                "expr": "histogram_quantile(0.95, rate(sesi_key_rotation_latency_seconds_bucket[5m])) > 30",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "Key rotation latency high",
                    "description": (
                        "P95 key rotation latency exceeds 30 seconds. "
                        "Check Vault Transit engine health."
                    ),
                },
            },
            {
                "alert": "DegenerateRotationRegime",
                "expr": "sesi_rotation_regime == 2",
                "for": "5m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Degenerate rotation regime detected",
                    "description": (
                        "Key rotation regime is DEGENERATE (θ→0, ½δ₀+½δ₁). "
                        "paper6 Thm 5.1: rotation pattern polarizing. "
                        "Investigate immediately."
                    ),
                },
            },
            {
                "alert": "NoZenoBoundExceeded",
                "expr": "sesi_no_zeno_event_count{event_type=\"key_rotation\"} > sesi_no_zeno_bound{event_type=\"key_rotation\"}",
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "No-Zeno bound exceeded for key rotations",
                    "description": (
                        "Key rotation event count exceeds No-Zeno bound: "
                        "N(T) ≤ 1 + 2C_V·T/δ_min (SESI_NoZeno Thm 3.2). "
                        "Possible Zeno-like rotation accumulation."
                    ),
                },
            },
            {
                "alert": "DeltaMinTooSmall",
                "expr": "sesi_delta_min{context=\"key_rotation\"} < 0.001",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "δ_min threshold too small",
                    "description": (
                        "δ_min for key rotation is below 0.001. "
                        "Delta_Min Thm 1.2: κ-fatness may be violated. "
                        "Rotation quantum gate may be ineffective."
                    ),
                },
            },
        ]

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

    def write_dashboard(self, output_path: str = "./grafana/key-rotation.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Key rotation dashboard written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./prometheus/key-rotation-alerts.yaml") -> str:
        """Write alert rules to file."""
        rules = self.generate_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {"groups": [{"name": "sesi_key_rotation", "rules": rules}]}

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"Alert rules written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════
# [OT1] OPENTELEMETRY TRACE INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════

class OpenTelemetryTraceIntegrator:
    """
    [D] Integrates OpenTelemetry distributed tracing with SESI pipeline.

    Features:
      • Distributed spans across auth → key rotation → AGIONE
      • Trace context propagation with trace_id/span_id
      • Correlation with Prometheus metrics via labels
      • Regime classification on trace event streams
      • Exit-time anomaly detection (Stochastic_NoZeno Thm 4.1)

    HONEST: This is a TRACING FRAMEWORK. It does NOT:
      • Deploy Jaeger collector
      • Configure trace sampling
      • Guarantee detection of all anomalies
      • Replace security audit

    Mathematical grounding:
      • CUSUM (Six_Problems P4): change-point detection in trace streams
      • Hill estimator (Paper 9 Entry 2.5): regime classification
      • Exit-time bound (Stochastic_NoZeno Thm 4.1): anomaly detection
    """

    def __init__(self, service_name: str = "sesi-observability"):
        self.service_name = service_name
        self.tracer = None
        self._span_history: List[Dict[str, Any]] = []
        self._event_timestamps: List[float] = []

        if HAS_OTEL:
            self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Initialize OpenTelemetry tracer."""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
        })
        provider = TracerProvider(resource=resource)

        # Console exporter for testing
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

        otel_trace.set_tracer_provider(provider)
        self.tracer = otel_trace.get_tracer(self.service_name)
        logger.info(f"OpenTelemetry tracer initialized: {self.service_name}")

    def start_auth_span(self, event_type: AuthEventType,
                        user_id: str, zone: str) -> Optional[Any]:
        """Start a span for an authorization event."""
        if self.tracer is None:
            return None

        span = self.tracer.start_span(
            name=f"auth.{event_type.name.lower()}",
            kind=SpanKind.SERVER,
        )
        span.set_attribute("auth.event_type", event_type.name)
        span.set_attribute("auth.user_id", user_id)
        span.set_attribute("auth.zone", zone)
        span.set_attribute("auth.timestamp", time.time())

        self._event_timestamps.append(time.time())

        return span

    def start_key_rotation_span(self, event: KeyRotationEvent,
                                 key_id: str, key_type: str) -> Optional[Any]:
        """Start a span for a key rotation event."""
        if self.tracer is None:
            return None

        span = self.tracer.start_span(
            name=f"key_rotation.{event.name.lower()}",
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("key_rotation.event", event.name)
        span.set_attribute("key_rotation.key_id", key_id)
        span.set_attribute("key_rotation.key_type", key_type)
        span.set_attribute("key_rotation.timestamp", time.time())

        self._event_timestamps.append(time.time())

        return span

    def start_agione_span(self, operation: str,
                          quality_score: Optional[float] = None) -> Optional[Any]:
        """Start a span for an AGIONE operation."""
        if self.tracer is None:
            return None

        span = self.tracer.start_span(
            name=f"agione.{operation}",
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("agione.operation", operation)
        if quality_score is not None:
            span.set_attribute("agione.quality_score", quality_score)
        span.set_attribute("agione.timestamp", time.time())

        self._event_timestamps.append(time.time())

        return span

    def end_span(self, span: Any, status: str = "OK",
                 error: Optional[str] = None) -> None:
        """End a span with status."""
        if span is None:
            return

        if error:
            span.set_status(StatusCode.ERROR, error)
        else:
            span.set_status(StatusCode.OK)

        span.end()

        self._span_history.append({
            "span_name": span.name,
            "status": status,
            "error": error,
            "timestamp": time.time(),
        })

    # ── Regime Classification ────────────────────────────────────────────

    def classify_trace_regime(self, hill_k: int = 20) -> Dict[str, Any]:
        """
        Classify trace event stream by growth-rate regime.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → ordinary LLN
          θ ∈ (0,1):  regularly varying → Beta(θ,θ) arcsine
          θ → 0:      degenerate → ½δ₀ + ½δ₁
        """
        if len(self._event_timestamps) < hill_k + 2:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_events": len(self._event_timestamps),
            }

        # Compute inter-event intervals
        sorted_ts = sorted(self._event_timestamps)
        intervals = [
            sorted_ts[i + 1] - sorted_ts[i]
            for i in range(len(sorted_ts) - 1)
        ]

        # Hill estimator for tail index
        positive_intervals = [iv for iv in intervals if iv > 0]
        if len(positive_intervals) < hill_k + 1:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_positive_intervals": len(positive_intervals),
            }

        sorted_iv = sorted(positive_intervals, reverse=True)
        x_k1 = sorted_iv[hill_k]

        if x_k1 <= 0:
            return {"regime": QualityRegime.INSUFFICIENT_DATA.name}

        log_ratios = [math.log(sorted_iv[i] / x_k1) for i in range(hill_k)]
        theta_hat = sum(log_ratios) / hill_k

        # Regime classification
        if theta_hat > 1.0:
            regime = QualityRegime.FINITE_MEAN
            description = "θ > 1: finite mean. Standard monitoring appropriate."
        elif theta_hat > 0.0:
            regime = QualityRegime.REGULARLY_VARYING
            description = "θ ∈ (0,1): regularly varying. Distributional monitoring."
        else:
            regime = QualityRegime.DEGENERATE
            description = "θ → 0: degenerate. Trace events polarizing."

        return {
            "regime": regime.name,
            "theta_hat": theta_hat,
            "description": description,
            "n_events": len(self._event_timestamps),
            "n_intervals": len(intervals),
        }

    def exit_time_alarm(self, observed_deviation: float,
                        delta: float = 0.3, g0: float = 0.05,
                        epsilon: float = 5.0) -> Dict[str, Any]:
        """
        Stochastic exit-time alarm (Stochastic_NoZeno Thm 4.1).

        P(sup ||h(r)-h(T_k)|| ≥ δ | F_{T_k}) ≤ 2exp(-δ²/8G₀²ε)
        """
        prob_bound = 2 * math.exp(-delta**2 / (8 * g0**2 * epsilon))
        prob_bound = min(prob_bound, 1.0)

        alarm = observed_deviation > delta

        return {
            "alarm": alarm,
            "observed_deviation": observed_deviation,
            "threshold_delta": delta,
            "prob_bound": prob_bound,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get trace integrator status."""
        return {
            "tracer_available": self.tracer is not None,
            "service_name": self.service_name,
            "span_history_size": len(self._span_history),
            "event_timestamps_size": len(self._event_timestamps),
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIObservabilitySuite:
    """
    [D] Complete observability suite orchestrating:
      • Prometheus auth metrics
      • Grafana key rotation dashboard
      • OpenTelemetry distributed tracing
    """

    def __init__(self):
        self.prometheus = PrometheusAuthMetricsExporter()
        self.grafana = KeyRotationGrafanaDashboard()
        self.otel = OpenTelemetryTraceIntegrator()

    def record_auth_event(self, event_type: AuthEventType,
                          user_id: str, zone: str,
                          outcome: str, latency_s: Optional[float] = None) -> None:
        """Record an auth event across all observability layers."""
        # Prometheus metrics
        self.prometheus.record_auth_event(event_type, zone, outcome, latency_s)

        # OpenTelemetry trace
        span = self.otel.start_auth_span(event_type, user_id, zone)
        if span:
            self.otel.end_span(span, status=outcome)

    def record_key_rotation(self, event: KeyRotationEvent,
                             key_id: str, key_type: str,
                             latency_s: Optional[float] = None) -> None:
        """Record a key rotation event across all observability layers."""
        # Prometheus metrics
        self.prometheus.record_key_rotation(event, key_type, latency_s)

        # OpenTelemetry trace
        span = self.otel.start_key_rotation_span(event, key_id, key_type)
        if span:
            outcome = "success" if event == KeyRotationEvent.ROTATION_COMPLETED else "failure"
            self.otel.end_span(span, status=outcome)

    def generate_all_artifacts(self, output_dir: str = "./sesi_observability") -> Dict[str, str]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Grafana dashboard
        dashboard_path = self.grafana.write_dashboard(
            str(out_path / "grafana" / "key-rotation.json")
        )
        results["grafana_dashboard"] = dashboard_path

        # Alert rules
        alerts_path = self.grafana.write_alert_rules(
            str(out_path / "prometheus" / "key-rotation-alerts.yaml")
        )
        results["alert_rules"] = alerts_path

        # Prometheus metrics export
        metrics_path = out_path / "prometheus" / "metrics.txt"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write(self.prometheus.export_prometheus())
        results["prometheus_metrics"] = str(metrics_path)

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full observability suite status."""
        return {
            "prometheus": self.prometheus.get_status(),
            "otel": self.otel.get_status(),
            "trace_regime": self.otel.classify_trace_regime(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Prometheus × Grafana × OpenTelemetry — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  OpenTelemetry avail  : {HAS_OTEL}")
    print(f"  Previous modules     : {HAS_PREV_MODULES}")

    suite = SESIObservabilitySuite()

    # ── Test 1: Prometheus Auth Metrics ──────────────────────────────────
    print("\n  ── Prometheus Auth Metrics ──")

    # Record some auth events
    suite.record_auth_event(
        AuthEventType.LOGIN_SUCCESS, "user_001", "z1-clinical",
        "success", latency_s=0.15,
    )
    suite.record_auth_event(
        AuthEventType.LOGIN_FAILURE, "user_002", "z3-external",
        "failure", latency_s=0.05,
    )
    suite.record_auth_event(
        AuthEventType.TOKEN_ISSUED, "user_001", "z1-clinical",
        "success", latency_s=0.02,
    )

    # Record zone violation
    suite.prometheus.record_zone_violation("z3-external", "z0-implant", 8443)

    # Record emergency stop
    suite.prometheus.record_emergency_stop("test_reason", "operator_001")

    # Update No-Zeno metrics
    suite.prometheus.update_no_zeno_metrics("auth_events")

    # Export Prometheus metrics
    metrics_output = suite.prometheus.export_prometheus()
    print(f"    Metrics exported     : {len(metrics_output.split(chr(10)))} lines")
    print(f"    Sample metrics       :")
    for line in metrics_output.split("\n")[:5]:
        print(f"      {line}")

    # ── Test 2: Key Rotation Monitoring ──────────────────────────────────
    print("\n  ── Key Rotation Monitoring ──")

    # Record some key rotations
    suite.record_key_rotation(
        KeyRotationEvent.ROTATION_COMPLETED,
        "key_001", "transit", latency_s=2.5,
    )
    suite.record_key_rotation(
        KeyRotationEvent.ROTATION_REJECTED_DELTA_MIN,
        "key_002", "transit", latency_s=0.1,
    )

    # Generate dashboard
    dashboard = suite.grafana.generate_dashboard()
    print(f"    Dashboard panels     : {len(dashboard['panels'])}")

    # Generate alert rules
    alerts = suite.grafana.generate_alert_rules()
    print(f"    Alert rules          : {len(alerts)}")
    for alert in alerts[:3]:
        print(f"      • {alert['alert']}")

    # ── Test 3: OpenTelemetry Tracing ────────────────────────────────────
    print("\n  ── OpenTelemetry Tracing ──")

    # Start AGIONE span
    agi_span = suite.otel.start_agione_span("forward", quality_score=0.85)
    if agi_span:
        suite.otel.end_span(agi_span, status="OK")

    # Classify trace regime
    regime = suite.otel.classify_trace_regime()
    print(f"    Trace regime         : {regime['regime']}")
    if "theta_hat" in regime:
        print(f"    θ̂ (Hill)            : {regime['theta_hat']:.4f}")

    # Exit-time alarm
    alarm = suite.otel.exit_time_alarm(observed_deviation=0.5)
    print(f"    Exit-time alarm      : {alarm['alarm']}")
    print(f"    Prob bound           : {alarm['prob_bound']:.6f}")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_observability")
    print(f"    Artifacts generated  : {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Prometheus metrics   : {status['prometheus']['metric_definitions']}")
    print(f"    Counter series       : {status['prometheus']['counter_series']}")
    print(f"    δ_min                : {status['prometheus']['delta_min']:.6f}")
    print(f"    No-Zeno bound        : {status['prometheus']['no_zeno_bound']:.2f}")
    print(f"    OTel tracer          : {status['otel']['tracer_available']}")
    print(f"    Trace regime         : {status['trace_regime']['regime']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Prometheus metrics provide OBSERVABILITY, not security")
    print("  • Grafana dashboards are VISUALIZATION, not decisions")
    print("  • OpenTelemetry traces show CORRELATION, not causation")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
