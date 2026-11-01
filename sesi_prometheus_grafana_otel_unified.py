"""
=============================================================================
SESI UNIFIED OBSERVABILITY STACK
Prometheus Enrichment Metrics + Unified Grafana Dashboard + Production OTel
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[PM1] PrometheusEnrichmentMetrics
      • Counters for enrichment events (accepted, rate-limited, rejected)
      • Histograms for enrichment latency, quality scores, regime duration
      • Gauges for active enrichments, δ_min, No-Zeno bound utilization
      • Per-domain quality score metrics (physics, fold, mental, etc.)
      • No-Zeno rate limiting on metric emission
      • _safe_unit_scalar sanitization for all quality values

[GD1] UnifiedGrafanaDashboard
      • Single-pane glass: traces + logs + metrics + service map
      • Trace-log-metric correlation via trace_id/span_id
      • Service map zone coloring (Z0-Z3)
      • Regime classification panels (paper6 Thm 5.1)
      • CUSUM change-point detection panels
      • δ_min quantum gate monitoring
      • Template variables for zone/service/trace filtering

[OT1] ProductionOTelCollectorConfig
      • Receivers: OTLP (gRPC+HTTP), Prometheus, Jaeger
      • Processors: batch, memory_limiter, filter/drop-sensitive,
        attributes/add-zone, spanmetrics, transform/add-trace-id-to-logs
      • Exporters: Prometheus, Jaeger, Loki, Alertmanager
      • Pipelines: traces, metrics, logs (zone-aware routing)
      • PNDC Axiom 3 compliance: sensitive data filtering
      • Production HA configuration (2 replicas, TLS, health checks)

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
[C] Six_Problems P4: CUSUM + subsampling
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar() sanitization
[C] PNDC Axiom 3: Necessity and Proportionality (data minimization)
[C] Comprehensive Security §5.3: Privacy preservation
[C] Unified_Ordinal Hyp 7.9: Free design parameters

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_production_trace_service_log.py — ProductionAGIONEQualityEnricher,
    UnifiedJaegerServiceMap, RealTimeLokiAggregator
  • sesi_unified_observability_stack.py — PrometheusAlertmanagerIntegration,
    UnifiedGrafanaDashboard, OpenTelemetryCollectorConfig
  • agi_one_v3_11.py — AGIONE, EcosystemOrchestrator.quality_report()
  • agi_one_psyche_plus_v33.py — AGIOnePsychePlus, _safe_unit_scalar()

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Prometheus metrics provide OBSERVABILITY, not security guarantees.
• Grafana dashboards are VISUALIZATION TOOLS, not decision engines.
• OTel Collector config is a TEMPLATE, not a deployed system.
• All thresholds, intervals, and sampling rates are FREE DESIGN
  PARAMETERS (Unified_Ordinal Hyp 7.9) requiring security review.
• quality_score sanitization via _safe_unit_scalar — NaN/Inf → 0.5.
• Sensitive data (passwords, tokens, SSN) filtered before export.
• Production deployment requires security audit and regulatory approval.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.unified_observability_v2")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ── _safe_unit_scalar fallback if Psyche Plus not available ──────────────
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
    "SecurityZone", "QualityRegime", "MetricType",
    "PrometheusEnrichmentMetrics",
    "UnifiedGrafanaDashboard",
    "ProductionOTelCollectorConfig",
    "SESIUnifiedObservabilityStack",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class SecurityZone(Enum):
    Z0_IMPLANT = "z0-implant"
    Z1_CLINICAL = "z1-clinical"
    Z2_PROCESSING = "z2-processing"
    Z3_EXTERNAL = "z3-external"


class QualityRegime(Enum):
    FINITE_MEAN = 0
    REGULARLY_VARYING = 1
    DEGENERATE = 2
    INSUFFICIENT_DATA = 3


class MetricType(Enum):
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


# ═══════════════════════════════════════════════════════════════════════
# [PM1] PROMETHEUS ENRICHMENT METRICS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MetricDefinition:
    """A single Prometheus metric definition."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None


class PrometheusEnrichmentMetrics:
    """
    [D] Prometheus metrics for enrichment events.

    Features:
      • Counters: enrichment_accepted, enrichment_rate_limited,
        enrichment_rejected, codification_events, alert_raised
      • Histograms: enrichment_latency_seconds, quality_score_distribution,
        regime_duration_seconds
      • Gauges: active_enrichments, delta_min, no_zeno_bound,
        no_zeno_event_count, quality_score_{domain}
      • No-Zeno rate limiting on emission
      • _safe_unit_scalar sanitization for all quality values

    HONEST: This is a METRICS GENERATOR. It produces Prometheus exposition
    format but does NOT deploy Prometheus or guarantee metric accuracy.

    Mathematical grounding:
      • No-Zeno bound on emission rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for meaningful metric updates (Delta_Min Thm 1.2)
    """

    METRICS = {
        # ── Counters ──────────────────────────────────────────────────
        "enrichment_accepted_total": MetricDefinition(
            name="sesi_enrichment_accepted_total",
            metric_type=MetricType.COUNTER,
            description="Total enrichment events accepted",
            labels=["zone", "service"],
        ),
        "enrichment_rate_limited_total": MetricDefinition(
            name="sesi_enrichment_rate_limited_total",
            metric_type=MetricType.COUNTER,
            description="Total enrichment events rate-limited",
            labels=["reason"],
        ),
        "enrichment_rejected_total": MetricDefinition(
            name="sesi_enrichment_rejected_total",
            metric_type=MetricType.COUNTER,
            description="Total enrichment events rejected",
            labels=["reason", "zone"],
        ),
        "codification_events_total": MetricDefinition(
            name="sesi_codification_events_total",
            metric_type=MetricType.COUNTER,
            description="Total Psyche Plus codification events",
            labels=["outcome"],  # "success" | "refused_validity" | "refused_quality"
        ),
        "alerts_raised_total": MetricDefinition(
            name="sesi_alerts_raised_total",
            metric_type=MetricType.COUNTER,
            description="Total alerts raised",
            labels=["alert_type", "severity", "zone"],
        ),

        # ── Histograms ────────────────────────────────────────────────
        "enrichment_latency_seconds": MetricDefinition(
            name="sesi_enrichment_latency_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Enrichment latency distribution",
            labels=["zone"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        ),
        "quality_score_distribution": MetricDefinition(
            name="sesi_quality_score",
            metric_type=MetricType.HISTOGRAM,
            description="Quality score distribution (sanitized via _safe_unit_scalar)",
            labels=["domain"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ),
        "regime_duration_seconds": MetricDefinition(
            name="sesi_regime_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Duration of each quality regime",
            labels=["regime"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
        ),

        # ── Gauges ────────────────────────────────────────────────────
        "active_enrichments": MetricDefinition(
            name="sesi_active_enrichments",
            metric_type=MetricType.GAUGE,
            description="Number of currently active enrichments",
            labels=[],
        ),
        "delta_min": MetricDefinition(
            name="sesi_delta_min",
            metric_type=MetricType.GAUGE,
            description="δ_min threshold from Delta_Min Thm 1.2",
            labels=["context"],
        ),
        "no_zeno_bound": MetricDefinition(
            name="sesi_no_zeno_bound",
            metric_type=MetricType.GAUGE,
            description="No-Zeno bound N(T) = 1 + 2C_V·T/δ_min",
            labels=["context"],
        ),
        "no_zeno_event_count": MetricDefinition(
            name="sesi_no_zeno_event_count",
            metric_type=MetricType.GAUGE,
            description="Current event count within No-Zeno window",
            labels=["context"],
        ),
        "quality_score_ecosystem": MetricDefinition(
            name="sesi_quality_score_ecosystem",
            metric_type=MetricType.GAUGE,
            description="Ecosystem-wide quality score (sanitized)",
            labels=[],
        ),
        "quality_regime": MetricDefinition(
            name="sesi_quality_regime",
            metric_type=MetricType.GAUGE,
            description="Current quality regime (0=FINITE_MEAN, 1=REGULARLY_VARYING, 2=DEGENERATE, 3=INSUFFICIENT_DATA)",
            labels=["domain"],
        ),
    }

    def __init__(self, metric_prefix: str = "sesi"):
        self.metric_prefix = metric_prefix
        self._counters: Dict[str, Dict[str, float]] = {
            name: {} for name, defn in self.METRICS.items()
            if defn.metric_type == MetricType.COUNTER
        }
        self._gauges: Dict[str, Dict[str, float]] = {
            name: {} for name, defn in self.METRICS.items()
            if defn.metric_type == MetricType.GAUGE
        }
        self._histograms: Dict[str, Dict[str, List[float]]] = {
            name: {} for name, defn in self.METRICS.items()
            if defn.metric_type == MetricType.HISTOGRAM
        }
        self._emission_timestamps: Dict[str, deque] = {
            name: deque(maxlen=10000) for name in self.METRICS
        }

        # No-Zeno parameters
        self.min_emission_interval_s = 0.01
        self.max_emissions_per_minute = 10000
        self.c_v = 1.0
        self.window_s = 60.0
        self.delta_min = self._compute_delta_min()

    def _compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                           n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def _check_emission_rate(self, metric_name: str) -> bool:
        """Check No-Zeno rate limit for metric emission."""
        now = time.time()
        cutoff = now - self.window_s
        timestamps = self._emission_timestamps[metric_name]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= self.max_emissions_per_minute:
            return False

        if timestamps and (now - timestamps[-1]) < self.min_emission_interval_s:
            return False

        timestamps.append(now)
        return True

    @staticmethod
    def _make_label_key(labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    # ── Counter operations ──────────────────────────────────────────────

    def inc_counter(self, metric_name: str, value: float = 1.0,
                    labels: Optional[Dict[str, str]] = None) -> bool:
        if metric_name not in self._counters:
            logger.warning(f"Unknown counter: {metric_name}")
            return False
        if not self._check_emission_rate(metric_name):
            return False
        key = self._make_label_key(labels or {})
        self._counters[metric_name][key] = (
            self._counters[metric_name].get(key, 0.0) + value
        )
        return True

    # ── Gauge operations ────────────────────────────────────────────────

    def set_gauge(self, metric_name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> bool:
        if metric_name not in self._gauges:
            logger.warning(f"Unknown gauge: {metric_name}")
            return False
        # Sanitize quality scores via _safe_unit_scalar
        if "quality" in metric_name:
            sanitized = _safe_unit_scalar(value, name=metric_name, logger=logger)
            if sanitized is None:
                sanitized = 0.5
            value = sanitized
        key = self._make_label_key(labels or {})
        self._gauges[metric_name][key] = value
        return True

    # ── Histogram operations ────────────────────────────────────────────

    def observe_histogram(self, metric_name: str, value: float,
                          labels: Optional[Dict[str, str]] = None) -> bool:
        if metric_name not in self._histograms:
            logger.warning(f"Unknown histogram: {metric_name}")
            return False
        if not self._check_emission_rate(metric_name):
            return False
        # Sanitize quality score observations
        if "quality" in metric_name:
            sanitized = _safe_unit_scalar(value, name=metric_name, logger=logger)
            if sanitized is None:
                sanitized = 0.5
            value = sanitized
        key = self._make_label_key(labels or {})
        if key not in self._histograms[metric_name]:
            self._histograms[metric_name][key] = []
        self._histograms[metric_name][key].append(value)
        return True

    # ── High-level enrichment event recording ───────────────────────────

    def record_enrichment_event(
        self,
        status: str,  # "accepted" | "rate_limited" | "rejected"
        zone: str = "z2-processing",
        service: str = "sesi-agione",
        latency_s: Optional[float] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Record a complete enrichment event across all relevant metrics."""
        labels_base = {"zone": zone, "service": service}

        if status == "accepted":
            self.inc_counter("enrichment_accepted_total", labels=labels_base)
            if latency_s is not None:
                self.observe_histogram(
                    "enrichment_latency_seconds", latency_s,
                    labels={"zone": zone},
                )
            if quality_scores:
                for domain, score in quality_scores.items():
                    self.observe_histogram(
                        "quality_score_distribution", score,
                        labels={"domain": domain},
                    )
                ecosystem = quality_scores.get("ecosystem")
                if ecosystem is not None:
                    self.set_gauge("quality_score_ecosystem", ecosystem)
            if regime:
                regime_idx = {
                    "FINITE_MEAN": 0,
                    "REGULARLY_VARYING": 1,
                    "DEGENERATE": 2,
                    "INSUFFICIENT_DATA": 3,
                }.get(regime, 3)
                self.set_gauge("quality_regime", regime_idx,
                               labels={"domain": "ecosystem"})

        elif status == "rate_limited":
            self.inc_counter(
                "enrichment_rate_limited_total",
                labels={"reason": reason or "unknown"},
            )

        elif status == "rejected":
            self.inc_counter(
                "enrichment_rejected_total",
                labels={"reason": reason or "unknown", "zone": zone},
            )

    def record_codification_event(self, outcome: str) -> None:
        """Record a Psyche Plus codification event."""
        self.inc_counter(
            "codification_events_total",
            labels={"outcome": outcome},
        )

    def record_alert(self, alert_type: str, severity: str,
                     zone: str = "z2-processing") -> None:
        """Record an alert event."""
        self.inc_counter(
            "alerts_raised_total",
            labels={"alert_type": alert_type, "severity": severity, "zone": zone},
        )

    def update_no_zeno_metrics(self, context: str, event_count: int) -> None:
        """Update No-Zeno bound metrics."""
        bound = 1 + (2 * self.c_v * self.window_s) / max(self.delta_min, 1e-8)
        self.set_gauge("no_zeno_event_count", event_count, labels={"context": context})
        self.set_gauge("no_zeno_bound", bound, labels={"context": context})
        self.set_gauge("delta_min", self.delta_min, labels={"context": context})

    # ── Prometheus exposition format ────────────────────────────────────

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
        return {
            "metric_definitions": len(self.METRICS),
            "counter_series": sum(len(v) for v in self._counters.values()),
            "gauge_series": sum(len(v) for v in self._gauges.values()),
            "histogram_series": sum(len(v) for v in self._histograms.values()),
            "delta_min": self.delta_min,
            "no_zeno_bound": 1 + (2 * self.c_v * self.window_s) / max(self.delta_min, 1e-8),
        }


# ═══════════════════════════════════════════════════════════════════════
# [GD1] UNIFIED GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class UnifiedGrafanaDashboard:
    """
    [D] Generates unified Grafana dashboard for traces+logs+metrics+service map.

    Layout:
      Row 1: System Overview (health, alerts, regime, emergency stops)
      Row 2: Enrichment Metrics (accepted/rate-limited/rejected, latency)
      Row 3: Quality Regime Detection (Hill θ̂, CUSUM, occupation fraction)
      Row 4: Service Map (zone-aware nodeGraph)
      Row 5: Trace-Log Correlation (trace count, span duration, log volume)
      Row 6: δ_min Monitoring (quantum gate, No-Zeno bound)
      Row 7: Security Events (auth, attestation, multi-identity risk)

    HONEST: Dashboard is VISUALIZATION, not decision-making.
    All thresholds are FREE DESIGN PARAMETERS.
    """

    def __init__(self, prometheus_uid: str = "prometheus",
                 jaeger_uid: str = "jaeger", loki_uid: str = "loki"):
        self.prometheus_uid = prometheus_uid
        self.jaeger_uid = jaeger_uid
        self.loki_uid = loki_uid

    def _panel(self, panel_id: int, title: str, panel_type: str,
               targets: List[Dict[str, Any]], grid_pos: Dict[str, int],
               datasource_uid: Optional[str] = None,
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    def generate_dashboard(self) -> Dict[str, Any]:
        panels = []
        pid = 1

        # ── Row 1: System Overview ──────────────────────────────────────
        panels.append(self._row_panel(pid, "System Overview", 0)); pid += 1

        panels.append(self._panel(
            pid, "System Health", "stat",
            [{"expr": "up{job='sesi-oversight'}", "legendFormat": "health"}],
            {"h": 5, "w": 4, "x": 0, "y": 1},
            extra={"fieldConfig": {"defaults": {"mappings": [
                {"type": "value", "options": {
                    "0": {"text": "DOWN", "color": "red"},
                    "1": {"text": "UP", "color": "green"}}}]}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "Active Alerts", "stat",
            [{"expr": "sum(ALERTS{alertstate='firing'}) or vector(0)",
              "legendFormat": "alerts"}],
            {"h": 5, "w": 4, "x": 4, "y": 1},
            extra={"fieldConfig": {"defaults": {"thresholds": {"mode": "absolute",
                "steps": [{"color": "green", "value": None},
                          {"color": "yellow", "value": 1},
                          {"color": "red", "value": 5}]}}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "Quality Regime (ecosystem)", "stat",
            [{"expr": "sesi_quality_regime{domain='ecosystem'}",
              "legendFormat": "regime"}],
            {"h": 5, "w": 4, "x": 8, "y": 1},
            extra={"fieldConfig": {"defaults": {"mappings": [
                {"type": "value", "options": {
                    "0": {"text": "FINITE_MEAN", "color": "green"},
                    "1": {"text": "REGULARLY_VARYING", "color": "yellow"},
                    "2": {"text": "DEGENERATE", "color": "red"},
                    "3": {"text": "INSUFFICIENT_DATA", "color": "gray"}}}]}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "Emergency Stops (24h)", "stat",
            [{"expr": "sum(increase(sesi_emergency_stops_total[24h]))",
              "legendFormat": "stops"}],
            {"h": 5, "w": 4, "x": 12, "y": 1},
        )); pid += 1

        panels.append(self._panel(
            pid, "Enrichment Rate", "timeseries",
            [{"expr": "sum(rate(sesi_enrichment_accepted_total[5m]))",
              "legendFormat": "accepted/s"},
             {"expr": "sum(rate(sesi_enrichment_rate_limited_total[5m]))",
              "legendFormat": "rate_limited/s"},
             {"expr": "sum(rate(sesi_enrichment_rejected_total[5m]))",
              "legendFormat": "rejected/s"}],
            {"h": 5, "w": 8, "x": 16, "y": 1},
        )); pid += 1

        # ── Row 2: Enrichment Metrics ───────────────────────────────────
        panels.append(self._row_panel(pid, "Enrichment Metrics", 6)); pid += 1

        panels.append(self._panel(
            pid, "Enrichment Latency P95", "timeseries",
            [{"expr": "histogram_quantile(0.95, rate(sesi_enrichment_latency_seconds_bucket[5m]))",
              "legendFormat": "p95"}],
            {"h": 8, "w": 8, "x": 0, "y": 7},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "Quality Score Distribution", "histogram",
            [{"expr": "sum(rate(sesi_quality_score_bucket[5m])) by (le)",
              "legendFormat": "{{le}}", "format": "heatmap"}],
            {"h": 8, "w": 8, "x": 8, "y": 7},
        )); pid += 1

        panels.append(self._panel(
            pid, "Codification Events", "timeseries",
            [{"expr": "sum(rate(sesi_codification_events_total[5m])) by (outcome)",
              "legendFormat": "{{outcome}}"}],
            {"h": 8, "w": 8, "x": 16, "y": 7},
        )); pid += 1

        # ── Row 3: Quality Regime Detection ─────────────────────────────
        panels.append(self._row_panel(pid, "Quality Regime Detection (paper6 Thm 5.1)", 15)); pid += 1

        panels.append(self._panel(
            pid, "Hill Estimator θ̂ (per domain)", "timeseries",
            [{"expr": "sesi_hill_theta_hat", "legendFormat": "{{domain}}"}],
            {"h": 8, "w": 8, "x": 0, "y": 16},
        )); pid += 1

        panels.append(self._panel(
            pid, "CUSUM Statistic", "timeseries",
            [{"expr": "sesi_cusum_statistic", "legendFormat": "cusum"},
             {"expr": "sesi_cusum_threshold", "legendFormat": "threshold"}],
            {"h": 8, "w": 8, "x": 8, "y": 16},
        )); pid += 1

        panels.append(self._panel(
            pid, "Occupation Fraction (½δ₀+½δ₁)", "gauge",
            [{"expr": "sesi_occupation_fraction", "legendFormat": "occupation"}],
            {"h": 8, "w": 8, "x": 16, "y": 16},
            extra={"fieldConfig": {"defaults": {"min": 0, "max": 1,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "green", "value": 0.3},
                    {"color": "red", "value": 0.7}]}}}},
        )); pid += 1

        # ── Row 4: Service Map ──────────────────────────────────────────
        panels.append(self._row_panel(pid, "Service Map (Zone-Aware)", 24)); pid += 1

        panels.append(self._panel(
            pid, "SESI Service Map", "nodeGraph",
            [{"expr": "", "legendFormat": ""}],
            {"h": 12, "w": 24, "x": 0, "y": 25},
            datasource_uid=self.jaeger_uid,
            extra={"options": {
                "nodeGrid": {"x": 0, "y": 0, "columns": 4, "rows": 1},
                "edges": [{"field": "call_count", "color": "orange"},
                          {"field": "error_rate", "color": "red"}],
                "nodes": {"main": {"color": "zone"}, "arc": "regime"},
            }},
        )); pid += 1

        # ── Row 5: Trace-Log Correlation ────────────────────────────────
        panels.append(self._row_panel(pid, "Trace-Log-Metric Correlation", 37)); pid += 1

        panels.append(self._panel(
            pid, "Trace Count", "timeseries",
            [{"expr": "sum(rate(jaeger_traces_total[5m]))",
              "legendFormat": "traces/s"}],
            {"h": 8, "w": 8, "x": 0, "y": 38},
        )); pid += 1

        panels.append(self._panel(
            pid, "Span Duration P95", "timeseries",
            [{"expr": "histogram_quantile(0.95, sum(rate(jaeger_span_duration_seconds_bucket[5m])) by (le, operation))",
              "legendFormat": "{{operation}}"}],
            {"h": 8, "w": 8, "x": 8, "y": 38},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        )); pid += 1

        panels.append(self._panel(
            pid, "Log Volume by Severity", "timeseries",
            [{"expr": "sum(rate({service='sesi-oversight'}[5m])) by (severity)",
              "legendFormat": "{{severity}}"}],
            {"h": 8, "w": 8, "x": 16, "y": 38},
            datasource_uid=self.loki_uid,
        )); pid += 1

        # ── Row 6: δ_min Monitoring ─────────────────────────────────────
        panels.append(self._row_panel(pid, "δ_min Quantum Gate (Delta_Min Thm 1.2)", 46)); pid += 1

        panels.append(self._panel(
            pid, "δ_min Estimate with CI", "timeseries",
            [{"expr": "sesi_delta_min", "legendFormat": "δ_min"},
             {"expr": "sesi_delta_min_ci_lower", "legendFormat": "CI lower"},
             {"expr": "sesi_delta_min_ci_upper", "legendFormat": "CI upper"}],
            {"h": 8, "w": 12, "x": 0, "y": 47},
        )); pid += 1

        panels.append(self._panel(
            pid, "No-Zeno Event Count vs Bound", "timeseries",
            [{"expr": "sesi_no_zeno_event_count", "legendFormat": "N(T)"},
             {"expr": "sesi_no_zeno_bound", "legendFormat": "bound"}],
            {"h": 8, "w": 12, "x": 12, "y": 47},
        )); pid += 1

        # ── Row 7: Security Events ──────────────────────────────────────
        panels.append(self._row_panel(pid, "Security Events", 55)); pid += 1

        panels.append(self._panel(
            pid, "Auth Events", "timeseries",
            [{"expr": "sum(rate(sesi_auth_events_total[5m])) by (event_type, outcome)",
              "legendFormat": "{{event_type}}-{{outcome}}"}],
            {"h": 8, "w": 12, "x": 0, "y": 56},
        )); pid += 1

        panels.append(self._panel(
            pid, "Multi-Identity Risk (Comp. Security §5.2)", "timeseries",
            [{"expr": "sesi_multi_identity_risk_score", "legendFormat": "R_id"}],
            {"h": 8, "w": 12, "x": 12, "y": 56},
        )); pid += 1

        dashboard = {
            "uid": "sesi-unified-observability-v2",
            "title": "SESI Unified Observability: Traces + Logs + Metrics + Service Map",
            "tags": ["sesi", "unified", "observability", "traces", "logs",
                     "metrics", "service-map"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "templating": {
                "list": [
                    {"name": "zone", "type": "custom",
                     "options": [
                         {"text": "All", "value": ""},
                         {"text": "Z0 Implant", "value": "z0-implant"},
                         {"text": "Z1 Clinical", "value": "z1-clinical"},
                         {"text": "Z2 Processing", "value": "z2-processing"},
                         {"text": "Z3 External", "value": "z3-external"},
                     ], "current": {"text": "All", "value": ""}},
                    {"name": "service", "type": "query",
                     "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                     "query": "label_values(sesi_enrichment_accepted_total, service)",
                     "refresh": 2},
                    {"name": "domain", "type": "query",
                     "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                     "query": "label_values(sesi_quality_score, domain)",
                     "refresh": 2},
                    {"name": "trace_id", "type": "textbox",
                     "current": {"text": "", "value": ""},
                     "label": "Trace ID (drill-down)"},
                ],
            },
            "annotations": {
                "list": [
                    {"name": "Emergency Stops",
                     "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                     "enable": True,
                     "expr": "sesi_emergency_stop_triggered",
                     "iconColor": "red", "titleFormat": "EMERGENCY STOP"},
                    {"name": "Regime Changes",
                     "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                     "enable": True,
                     "expr": "changes(sesi_quality_regime[5m]) > 0",
                     "iconColor": "yellow", "titleFormat": "Regime Change"},
                    {"name": "Codification Events",
                     "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                     "enable": True,
                     "expr": "sesi_codification_events_total{outcome='success'}",
                     "iconColor": "blue", "titleFormat": "Axiom Codified"},
                ],
            },
            "links": [
                {"title": "Jaeger Traces",
                 "url": f"http://jaeger-query:16686/search?service=sesi-oversight&tags=%7B%22trace_id%22%3A%22${{trace_id}}%22%7D",
                 "type": "link", "icon": "external link", "targetBlank": True},
                {"title": "Loki Logs",
                 "url": f"http://loki:3100/loki/api/v1/query_range?query=%7Bservice%3D%22sesi-oversight%22%2C+trace_id%3D%22${{trace_id}}%22%7D",
                 "type": "link", "icon": "external link", "targetBlank": True},
            ],
        }

        return dashboard

    def generate_alert_rules(self) -> List[Dict[str, Any]]:
        return [
            {"alert": "DegenerateRegimeDetected",
             "expr": "sesi_quality_regime{domain='ecosystem'} == 2",
             "for": "5m", "labels": {"severity": "critical"},
             "annotations": {"summary": "Degenerate quality regime detected",
                 "description": "paper6 Thm 5.1: occupation fraction polarizing (½δ₀+½δ₁)."}},
            {"alert": "HighEnrichmentRejectionRate",
             "expr": "sum(rate(sesi_enrichment_rejected_total[5m])) / sum(rate(sesi_enrichment_accepted_total[5m])) > 0.1",
             "for": "5m", "labels": {"severity": "warning"},
             "annotations": {"summary": "High enrichment rejection rate",
                 "description": "Check δ_min quantum gate and rate limiting configuration."}},
            {"alert": "NoZenoBoundExceeded",
             "expr": "sesi_no_zeno_event_count > sesi_no_zeno_bound",
             "for": "1m", "labels": {"severity": "critical"},
             "annotations": {"summary": "No-Zeno bound exceeded",
                 "description": "SESI_NoZeno Thm 3.2: event count exceeds bound."}},
            {"alert": "DeltaMinTooSmall",
             "expr": "sesi_delta_min < 0.001",
             "for": "5m", "labels": {"severity": "warning"},
             "annotations": {"summary": "δ_min threshold too small",
                 "description": "Delta_Min Thm 1.2: κ-fatness may be violated."}},
            {"alert": "HighMultiIdentityRisk",
             "expr": "sesi_multi_identity_risk_score > 0.7",
             "for": "5m", "labels": {"severity": "warning"},
             "annotations": {"summary": "High multi-identity risk",
                 "description": "Comprehensive Security §5.2: R_id > 0.7."}},
        ]

    def _row_panel(self, panel_id: int, title: str, y: int) -> Dict[str, Any]:
        return {"id": panel_id, "type": "row", "title": title,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
                "collapsed": False, "panels": []}

    def write_dashboard(self, output_path: str = "./grafana/unified-dashboard-v2.json") -> str:
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Unified dashboard written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./prometheus/unified-alerts.yaml") -> str:
        rules = self.generate_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = {"groups": [{"name": "sesi_unified", "rules": rules}]}
        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════
# [OT1] PRODUCTION OTEL COLLECTOR CONFIG
# ═══════════════════════════════════════════════════════════════════════

class ProductionOTelCollectorConfig:
    """
    [D] Production-grade OpenTelemetry Collector configuration.

    Features:
      • Receivers: OTLP (gRPC+HTTP), Prometheus, Jaeger
      • Processors: batch, memory_limiter, filter/drop-sensitive,
        attributes/add-zone, spanmetrics, transform/add-trace-id-to-logs
      • Exporters: Prometheus, Jaeger, Loki, Alertmanager
      • Pipelines: traces, metrics, logs (zone-aware routing)
      • PNDC Axiom 3 compliance: sensitive data filtering
      • Production HA: 2 replicas, TLS, health checks

    HONEST: This is a CONFIGURATION TEMPLATE. Real deployment requires
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
                            {"job_name": "sesi-oversight",
                             "scrape_interval": "15s",
                             "static_configs": [{"targets": ["sesi-oversight:9090"]}]},
                            {"job_name": "sesi-vault",
                             "scrape_interval": "30s",
                             "static_configs": [{"targets": ["vault:8200"]}]},
                            {"job_name": "sesi-agione",
                             "scrape_interval": "15s",
                             "static_configs": [{"targets": ["sesi-agione:9091"]}]},
                        ],
                    },
                },
                "jaeger": {
                    "protocols": {
                        "grpc": {"endpoint": "0.0.0.0:14250"},
                        "thrift_http": {"endpoint": "0.0.0.0:14268"},
                    },
                },
            },
            "processors": {
                "batch": {
                    "timeout": "5s",
                    "send_batch_size": 1024,
                    "send_batch_max_size": 2048,
                },
                "memory_limiter": {
                    "check_interval": "1s",
                    "limit_mib": 1024,
                    "spike_limit_mib": 256,
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
                "attributes/add-zone": {
                    "actions": [
                        {"key": "sesi.zone", "from_attribute": "zone", "action": "upsert"},
                        {"key": "sesi.service", "from_attribute": "service.name", "action": "upsert"},
                    ],
                },
                "spanmetrics": {
                    "metrics_exporter": "prometheus",
                    "latency_histogram_buckets": [
                        "1ms", "5ms", "10ms", "50ms", "100ms",
                        "500ms", "1s", "5s", "10s",
                    ],
                    "dimensions": [
                        {"name": "service.name"},
                        {"name": "sesi.zone"},
                        {"name": "http.method"},
                        {"name": "http.status_code"},
                        {"name": "quality_regime"},
                    ],
                },
                "transform/add-trace-id-to-logs": {
                    "log_statements": [{
                        "context": "log",
                        "statements": [
                            'set(attributes["trace_id"], trace_id) where trace_id != nil',
                            'set(attributes["span_id"], span_id) where span_id != nil',
                        ],
                    }],
                },
                # Zone-aware routing processor
                "attributes/zone-routing": {
                    "actions": [
                        {"key": "sesi.zone_crossing",
                         "action": "upsert",
                         "value": "true",
                         "conditional": {
                             "key": "sesi.zone",
                             "op": "exists",
                         }},
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
                    "traces": {
                        "receivers": ["otlp", "jaeger"],
                        "processors": [
                            "memory_limiter",
                            "filter/drop-sensitive",
                            "attributes/add-zone",
                            "spanmetrics",
                            "batch",
                        ],
                        "exporters": ["jaeger", "prometheus"],
                    },
                    "metrics": {
                        "receivers": ["otlp", "prometheus"],
                        "processors": ["memory_limiter", "batch"],
                        "exporters": ["prometheus", "prometheusremotewrite"],
                    },
                    "logs": {
                        "receivers": ["otlp"],
                        "processors": [
                            "memory_limiter",
                            "filter/drop-sensitive",
                            "transform/add-trace-id-to-logs",
                            "attributes/add-zone",
                            "attributes/zone-routing",
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
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "otel-collector",
                "namespace": "sesi-observability",
                "labels": {"app.kubernetes.io/name": "otel-collector"},
            },
            "spec": {
                "replicas": 2,
                "selector": {"matchLabels": {"app.kubernetes.io/name": "otel-collector"}},
                "template": {
                    "metadata": {
                        "labels": {"app.kubernetes.io/name": "otel-collector"},
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
                                {"name": "jaeger-grpc", "containerPort": 14250},
                                {"name": "jaeger-http", "containerPort": 14268},
                                {"name": "prometheus", "containerPort": 8889},
                                {"name": "metrics", "containerPort": 8888},
                                {"name": "health", "containerPort": 13133},
                            ],
                            "env": [{"name": "GOMEMLIMIT", "value": "1024MiB"}],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "1Gi"},
                                "limits": {"cpu": "2000m", "memory": "2Gi"},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/", "port": 13133},
                                "initialDelaySeconds": 30, "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/", "port": 13133},
                                "initialDelaySeconds": 10, "periodSeconds": 5,
                            },
                            "volumeMounts": [
                                {"name": "otel-config", "mountPath": "/etc/otel", "readOnly": True},
                                {"name": "otel-certs", "mountPath": "/etc/otel/certs", "readOnly": True},
                            ],
                        }],
                        "volumes": [
                            {"name": "otel-config", "configMap": {"name": "otel-collector-config"}},
                            {"name": "otel-certs", "secret": {"secretName": "otel-collector-tls"}},
                        ],
                    },
                },
            },
        }

    def write_config(self, output_path: str = "./otel/otel-collector-config.yaml") -> str:
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

    def write_k8s_manifest(self, output_path: str = "./k8s/otel-collector-deployment.yaml") -> str:
        manifest = self.generate_k8s_deployment()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(manifest, f, indent=2)
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIUnifiedObservabilityStack:
    """
    [D] Complete unified observability stack.

    Integrates:
      • Prometheus enrichment metrics
      • Unified Grafana dashboard (traces+logs+metrics+service map)
      • Production OTel Collector configuration
    """

    def __init__(self):
        self.metrics = PrometheusEnrichmentMetrics()
        self.dashboard = UnifiedGrafanaDashboard()
        self.otel_collector = ProductionOTelCollectorConfig()

    def generate_all_artifacts(self, output_dir: str = "./sesi_unified_obs") -> Dict[str, str]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results = {}

        # Grafana dashboard
        results["grafana_dashboard"] = self.dashboard.write_dashboard(
            str(out_path / "grafana" / "unified-dashboard-v2.json"))

        # Alert rules
        results["alert_rules"] = self.dashboard.write_alert_rules(
            str(out_path / "prometheus" / "unified-alerts.yaml"))

        # OTel Collector config
        results["otel_config"] = self.otel_collector.write_config(
            str(out_path / "otel" / "otel-collector-config.yaml"))

        # K8s deployment
        results["k8s_deployment"] = self.otel_collector.write_k8s_manifest(
            str(out_path / "k8s" / "otel-collector-deployment.yaml"))

        # Prometheus metrics export
        metrics_path = out_path / "prometheus" / "metrics.txt"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            f.write(self.metrics.export_prometheus())
        results["prometheus_metrics"] = str(metrics_path)

        return results

    def status_report(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.get_status(),
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
    print(" SESI Unified Observability Stack — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available: {HAS_YAML}")

    suite = SESIUnifiedObservabilityStack()

    # ── Test 1: Prometheus Enrichment Metrics ───────────────────────────
    print("\n  ── Prometheus Enrichment Metrics ──")
    metrics = suite.metrics

    # Simulate enrichment events
    for i in range(10):
        metrics.record_enrichment_event(
            status="accepted",
            zone="z2-processing",
            service=f"sesi-service-{i % 3}",
            latency_s=0.05 + i * 0.01,
            quality_scores={
                "ecosystem": 0.75 + i * 0.01,
                "physics": 0.8,
                "fold": 0.7,
                "mental": 0.65,
            },
            regime="FINITE_MEAN" if i % 2 == 0 else "REGULARLY_VARYING",
        )

    # Rate-limited event
    metrics.record_enrichment_event(
        status="rate_limited",
        reason="no_zeno_bound",
    )

    # Rejected event
    metrics.record_enrichment_event(
        status="rejected",
        zone="z3-external",
        reason="sub_delta_min",
    )

    # Codification events
    metrics.record_codification_event("success")
    metrics.record_codification_event("refused_quality")

    # Alerts
    metrics.record_alert("low_quality", "warning", "z2-processing")
    metrics.record_alert("emergency_stop", "emergency", "z0-implant")

    # No-Zeno metrics
    metrics.update_no_zeno_metrics("enrichment", 5)

    # Export
    prom_output = metrics.export_prometheus()
    print(f"    Metrics exported    : {len(prom_output.split(chr(10)))} lines")
    print(f"    Sample metrics      :")
    for line in prom_output.split("\n")[:8]:
        print(f"      {line}")

    # Test _safe_unit_scalar sanitization
    print("\n  ── _safe_unit_scalar Sanitization ──")
    nan_result = _safe_unit_scalar(float("nan"), name="test_nan", logger=logger)
    inf_result = _safe_unit_scalar(float("inf"), name="test_inf", logger=logger)
    normal_result = _safe_unit_scalar(0.85, name="test_normal", logger=logger)
    print(f"    NaN sanitized to    : {nan_result}")
    print(f"    Inf sanitized to    : {inf_result}")
    print(f"    0.85 normalized to  : {normal_result}")

    # ── Test 2: Unified Grafana Dashboard ───────────────────────────────
    print("\n  ── Unified Grafana Dashboard ──")
    dashboard = suite.dashboard.generate_dashboard()
    print(f"    Dashboard title     : {dashboard['title']}")
    print(f"    Total panels        : {len(dashboard['panels'])}")
    print(f"    Template variables  : {len(dashboard['templating']['list'])}")
    print(f"    Annotations         : {len(dashboard['annotations']['list'])}")
    print(f"    External links      : {len(dashboard['links'])}")

    alert_rules = suite.dashboard.generate_alert_rules()
    print(f"    Alert rules         : {len(alert_rules)}")
    for rule in alert_rules[:3]:
        print(f"      • {rule['alert']} ({rule['labels']['severity']})")

    # ── Test 3: OTel Collector Config ───────────────────────────────────
    print("\n  ── Production OTel Collector Config ──")
    otel_config = suite.otel_collector.generate_config()
    print(f"    Receivers           : {list(otel_config['receivers'].keys())}")
    print(f"    Processors          : {list(otel_config['processors'].keys())}")
    print(f"    Exporters           : {list(otel_config['exporters'].keys())}")
    print(f"    Pipelines           : {list(otel_config['service']['pipelines'].keys())}")

    # Check PNDC Axiom 3 compliance
    filter_proc = otel_config["processors"]["filter/drop-sensitive"]
    print(f"    Sensitive filters   : {len(filter_proc['traces']['span'])} trace, "
          f"{len(filter_proc['logs']['log_record'])} log patterns")

    # ── Test 4: Full Artifact Generation ────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_unified_obs")
    print(f"    Artifacts generated : {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:25s}: {path}")

    # ── Full Status ─────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Metric definitions  : {status['metrics']['metric_definitions']}")
    print(f"    Counter series      : {status['metrics']['counter_series']}")
    print(f"    Gauge series        : {status['metrics']['gauge_series']}")
    print(f"    Histogram series    : {status['metrics']['histogram_series']}")
    print(f"    δ_min               : {status['metrics']['delta_min']:.6f}")
    print(f"    No-Zeno bound       : {status['metrics']['no_zeno_bound']:.2f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Prometheus metrics provide OBSERVABILITY, not security")
    print("  • Grafana dashboard is VISUALIZATION, not decision-making")
    print("  • OTel Collector config is a TEMPLATE, not a deployment")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Sensitive data filtered per PNDC Axiom 3")
    print("  • quality_score sanitized via _safe_unit_scalar")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
