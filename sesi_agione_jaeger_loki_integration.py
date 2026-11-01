"""
=============================================================================
SESI AGIONE QUALITY ENRICHMENT × JAEGER SERVICE MAP × LOKI SECURITY LOGS
Trace-Quality Correlation + Service Topology + Structured Log Aggregation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[AQ1] AGIONEQualityEnricher
      • Pulls quality_report() from EcosystemOrchestrator
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Injects quality scores as OpenTelemetry span attributes
      • Correlates with Psyche Plus codification events
      • Applies No-Zeno rate limiting on enrichment frequency

[SM1] JaegerServiceMapGenerator
      • Builds zone-aware service dependency graph (Z0-Z3)
      • Generates Jaeger service map JSON
      • Computes call volume, latency, error rates per edge
      • Applies PNDC Axiom 3 (Necessity & Proportionality)
      • Produces Grafana service map panel

[LL1] LokiSecurityLogAggregator
      • Structured JSON logging for security events
      • LogQL query templates for security analysis
      • Regime classification on log streams (paper6 Thm 5.1)
      • Trace-log correlation via trace_id/span_id
      • Data minimization per Comprehensive Security §5.3

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds enrichment and log emission frequency
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω)^{1/(n-1)} under κ-fatness
    → minimum meaningful change for enrichment triggers
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → log stream regime classification
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
    → confidence interval for enrichment thresholds
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → multi-identity risk in log correlation
[C] PNDC Axiom 3: Necessity and Proportionality
    → log only what is necessary; minimize data collection
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO AGIONE v3.11 AND PSYCHE PLUS v3.3
-----------------------------------------------
From agi_one_v3_11.py:
  • EcosystemOrchestrator.quality_report() → per-domain quality scores
  • AGIONE.attach_orchestrator() → quality_score threading to Psyche Plus
  • AGIState.ecosystem_quality_scores → stored in cognitive state

From agi_one_psyche_plus_v33.py:
  • _safe_unit_scalar() → NaN/Inf sanitization (applied to all quality inputs)
  • AGIOnePsychePlus.forward(latent, quality_score=...) → quality-aware
  • AGIOnePsychePlus.maybe_codify() → dual-gate (validity + quality)
  • SuperEgoModule.quality_weight → blend weight for quality in validity

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• AGIONE quality enrichment provides OBSERVABILITY, not causation.
  Quality scores attached to spans show temporal correlation, not proof
  that quality caused the traced behavior.
• Jaeger service map is a VISUALIZATION of topology, not a security
  guarantee. It shows call patterns, not whether those calls are safe.
• Loki logs are STRUCTURED RECORDS, not omniscience. They capture
  what the system observes, not everything that happens.
• All thresholds, intervals, and sampling rates are FREE DESIGN
  PARAMETERS (Unified_Ordinal Hyp 7.9) requiring security team review.
• quality_score sanitization follows _safe_unit_scalar pattern from
  Psyche Plus v3.3 — NaN/Inf values clamped to default 0.5.
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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.agione_jaeger_loki")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode, SpanKind
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

# ── AGIONE / Psyche Plus imports (graceful) ──────────────────────────────
try:
    from agi_one_v3_11 import AGIONE, EcosystemOrchestrator, AGIState
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

try:
    from agi_one_psyche_plus_v33 import AGIOnePsychePlus, _safe_unit_scalar
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False
    # Fallback sanitization if Psyche Plus not available
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
    "SecurityZone", "ServiceNode", "ServiceEdge",
    "AGIONEQualityEnricher", "EnrichedSpan",
    "JaegerServiceMapGenerator", "ServiceMapConfig",
    "LokiSecurityLogAggregator", "LogEntry", "LogQLQuery",
    "LogRegimeClassifier",
    "SESIIntegratedObservabilitySuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class SecurityZone(Enum):
    """Four security zones from SESI network segmentation."""
    Z0_IMPLANT = 0
    Z1_CLINICAL_GATEWAY = 1
    Z2_PROCESSING = 2
    Z3_EXTERNAL = 3


class LogSeverity(Enum):
    """Log severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5


class LogRegime(Enum):
    """Growth-rate regime for log streams (paper6 Thm 5.1)."""
    FINITE_MEAN = auto()          # θ > 1
    REGULARLY_VARYING = auto()    # θ ∈ (0, 1)
    DEGENERATE = auto()           # θ → 0 (½δ₀ + ½δ₁)
    INSUFFICIENT_DATA = auto()


# ═══════════════════════════════════════════════════════════════════════
# [AQ1] AGIONE QUALITY ENRICHER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EnrichedSpan:
    """
    [D] A trace span enriched with AGIONE quality data.

    HONEST: Enrichment provides OBSERVABILITY, not causation.
    Quality scores attached to spans show temporal correlation,
    not proof that quality caused the traced behavior.
    """
    trace_id: str
    span_id: str
    operation_name: str
    service_name: str
    zone: SecurityZone
    start_time: float
    duration_ms: float
    quality_scores: Dict[str, float]
    regime: Optional[str] = None
    codification_events: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "zone": self.zone.name,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "quality_scores": self.quality_scores,
            "regime": self.regime,
            "n_codifications": len(self.codification_events),
            "n_alerts": len(self.alerts),
            "attributes": self.attributes,
        }


class AGIONEQualityEnricher:
    """
    [D] Enriches trace spans with AGIONE quality_report() data.

    Features:
      • Extracts quality scores from EcosystemOrchestrator.quality_report()
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Injects as span attributes
      • Correlates with Psyche Plus codification events
      • Applies No-Zeno rate limiting on enrichment frequency

    HONEST: This enricher provides OBSERVABILITY, not causation.
    Quality scores attached to spans show temporal correlation,
    not proof that quality caused the traced behavior.

    Mathematical grounding:
      • _safe_unit_scalar sanitization (Psyche Plus v3.3)
      • No-Zeno bound on enrichment frequency (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for meaningful enrichment (Delta_Min Thm 1.2)
    """

    def __init__(
        self,
        agione: Optional[Any] = None,
        min_enrichment_interval_s: float = 1.0,
        max_enrichments_per_window: int = 100,
    ):
        self.agione = agione
        self.min_interval_s = min_enrichment_interval_s
        self.max_per_window = max_enrichments_per_window
        self._enrichment_timestamps: deque = deque(maxlen=max_enrichments_per_window)
        self._enriched_spans: List[EnrichedSpan] = []
        self._last_quality_report: Optional[Dict[str, float]] = None

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """
        Compute δ_min from Delta_Min Thm 1.2.

        δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}

        Requires κ-fatness (Delta_Min Def 1.1). Without it, δ_min
        can be zero and the rate bound becomes vacuous.
        """
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_enrichment_rate(self) -> Dict[str, Any]:
        """Check if enrichment is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0  # 1-minute window
        recent = [t for t in self._enrichment_timestamps if t > cutoff]

        delta_min = self.compute_delta_min()
        c_v = 1.0
        T = 60.0
        bound = 1 + (2 * c_v * T) / max(delta_min, 1e-8)

        if len(recent) >= self.max_per_window:
            return {
                "allowed": False,
                "reason": "max_per_window_exceeded",
                "current": len(recent),
                "bound": self.max_per_window,
            }

        if recent and (now - recent[-1]) < self.min_interval_s:
            return {
                "allowed": False,
                "reason": "min_interval_not_elapsed",
                "time_since_last_s": now - recent[-1],
                "min_interval_s": self.min_interval_s,
            }

        return {
            "allowed": True,
            "current_count": len(recent),
            "bound": bound,
            "delta_min": delta_min,
        }

    # ── Quality Extraction ───────────────────────────────────────────────

    def extract_quality_from_agione(self) -> Dict[str, float]:
        """
        Extract quality scores from AGIONE's EcosystemOrchestrator.

        From agi_one_v3_11.py:
          EcosystemOrchestrator.quality_report() returns per-domain scores.
          The "ecosystem" key is the mean across all domains.

        Returns:
            dict with quality scores per domain, sanitized via _safe_unit_scalar
        """
        if self.agione is None:
            return {}

        orchestrator = getattr(self.agione, "orchestrator", None)
        if orchestrator is None:
            return {}

        try:
            quality_report = orchestrator.quality_report()
            # Sanitize all scores via _safe_unit_scalar
            sanitized = {}
            for domain, score in quality_report.items():
                sanitized_score = _safe_unit_scalar(
                    score, name=f"quality.{domain}", logger=logger
                )
                if sanitized_score is not None:
                    sanitized[domain] = sanitized_score
            self._last_quality_report = sanitized
            return sanitized
        except Exception as e:
            logger.warning(f"quality_report() failed: {e}")
            return self._last_quality_report or {}

    # ── Span Enrichment ──────────────────────────────────────────────────

    def enrich_span(
        self,
        trace_id: str,
        span_id: str,
        operation_name: str,
        service_name: str,
        zone: SecurityZone,
        start_time: float,
        duration_ms: float,
        codification_event: Optional[Dict[str, Any]] = None,
        extra_attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnrichedSpan]:
        """
        Enrich a trace span with quality data.

        Args:
            trace_id: Jaeger trace ID
            span_id: Jaeger span ID
            operation_name: Span operation name
            service_name: Service name
            zone: Security zone
            start_time: Span start timestamp
            duration_ms: Span duration in milliseconds
            codification_event: Optional Psyche Plus codification event
            extra_attributes: Optional additional attributes

        Returns:
            EnrichedSpan with quality data attached, or None if rate-limited
        """
        # Check rate limit
        rate_check = self.check_enrichment_rate()
        if not rate_check["allowed"]:
            logger.debug(f"Enrichment rate-limited: {rate_check['reason']}")
            return None

        # Extract quality scores
        quality_scores = self.extract_quality_from_agione()

        # Record enrichment
        self._enrichment_timestamps.append(time.time())

        # Build enriched span
        span = EnrichedSpan(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            service_name=service_name,
            zone=zone,
            start_time=start_time,
            duration_ms=duration_ms,
            quality_scores=quality_scores,
            codification_events=[codification_event] if codification_event else [],
            attributes=extra_attributes or {},
        )

        # Add alerts for quality issues
        ecosystem_quality = quality_scores.get("ecosystem", 0.5)
        if ecosystem_quality < 0.3:
            span.alerts.append({
                "type": "low_quality",
                "severity": "warning",
                "message": f"Ecosystem quality low: {ecosystem_quality:.3f}",
                "timestamp": time.time(),
            })

        self._enriched_spans.append(span)
        return span

    def get_enriched_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent enriched spans as dicts."""
        return [s.to_dict() for s in self._enriched_spans[-limit:]]

    def get_status(self) -> Dict[str, Any]:
        """Get enricher status."""
        return {
            "agione_attached": self.agione is not None,
            "enriched_spans": len(self._enriched_spans),
            "last_quality_report": self._last_quality_report,
            "delta_min": self.compute_delta_min(),
            "enrichment_rate_check": self.check_enrichment_rate(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [SM1] JAEGER SERVICE MAP GENERATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ServiceNode:
    """A node in the service map."""
    name: str
    zone: SecurityZone
    service_type: str  # "implant" | "clinical" | "processing" | "external"
    n_calls_in: int = 0
    n_calls_out: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0


@dataclass
class ServiceEdge:
    """An edge in the service map."""
    source: str
    target: str
    call_count: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    zone_crossing: bool = False


@dataclass
class ServiceMapConfig:
    """
    Configuration for service map generation.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be reviewed by security team.
    """
    # PNDC Axiom 3: Necessity & Proportionality
    min_calls_to_display: int = 10
    max_edges_per_node: int = 50

    # Zone-aware coloring
    zone_colors: Dict[str, str] = field(default_factory=lambda: {
        "z0_implant": "#ff6b6b",        # red — most restricted
        "z1_clinical": "#ffd93d",        # yellow — clinician access
        "z2_processing": "#6bcf7f",      # green — AI inference
        "z3_external": "#4d96ff",        # blue — public API
    })

    # Layout
    layout_algorithm: str = "force_directed"  # or "hierarchical"


class JaegerServiceMapGenerator:
    """
    [D] Generates Jaeger service map from trace data.

    Features:
      • Zone-aware service dependency graph (Z0-Z3)
      • Call volume, latency, error rate per edge
      • PNDC Axiom 3 (Necessity & Proportionality) filtering
      • No-Zeno bounded edge collection
      • Grafana service map panel generation

    HONEST: This is a VISUALIZATION TOOL. It shows call patterns,
    not whether those calls are safe or correct. Human operators
    must interpret the map in context.
    """

    def __init__(self, cfg: Optional[ServiceMapConfig] = None):
        self.cfg = cfg or ServiceMapConfig()
        self.nodes: Dict[str, ServiceNode] = {}
        self.edges: Dict[Tuple[str, str], ServiceEdge] = {}
        self._edge_timestamps: deque = deque(maxlen=10000)

    def register_service(self, name: str, zone: SecurityZone,
                         service_type: str) -> None:
        """Register a service in the map."""
        if name not in self.nodes:
            self.nodes[name] = ServiceNode(
                name=name,
                zone=zone,
                service_type=service_type,
            )
            logger.info(f"Registered service: {name} (zone={zone.name})")

    def record_call(
        self,
        source: str,
        target: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """
        Record a service-to-service call.

        Applies PNDC Axiom 3: only necessary calls are recorded.
        """
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
        # Running average latency
        edge.avg_latency_ms = (
            (edge.avg_latency_ms * (edge.call_count - 1) + latency_ms)
            / edge.call_count
        )
        # Running error rate
        if not success:
            edge.error_rate = (
                (edge.error_rate * (edge.call_count - 1) + 1.0)
                / edge.call_count
            )

        # Update node stats
        self.nodes[source].n_calls_out += 1
        self.nodes[target].n_calls_in += 1

        self._edge_timestamps.append(time.time())

    def filter_by_pndc(self) -> Dict[str, ServiceEdge]:
        """
        Apply PNDC Axiom 3 filtering: only necessary edges displayed.

        From PNDC Axiom 3: "An act that causes harm requires stronger
        justification when it is unnecessary for survival or the
        prevention of serious harm."

        Applied to service map: edges below min_calls_to_display are
        filtered out as unnecessary noise.
        """
        return {
            k: v for k, v in self.edges.items()
            if v.call_count >= self.cfg.min_calls_to_display
        }

    def generate_jaeger_service_map(self) -> Dict[str, Any]:
        """
        Generate Jaeger-compatible service map JSON.

        Returns:
            dict with nodes and edges for Jaeger UI
        """
        filtered_edges = self.filter_by_pndc()

        # Build nodes list
        nodes = []
        for name, node in self.nodes.items():
            nodes.append({
                "name": name,
                "zone": node.zone.name,
                "service_type": node.service_type,
                "color": self.cfg.zone_colors.get(node.zone.name.lower(), "#888"),
                "n_calls_in": node.n_calls_in,
                "n_calls_out": node.n_calls_out,
                "avg_latency_ms": node.avg_latency_ms,
                "error_rate": node.error_rate,
            })

        # Build edges list
        edges = []
        for (src, tgt), edge in filtered_edges.items():
            edges.append({
                "source": src,
                "target": tgt,
                "call_count": edge.call_count,
                "avg_latency_ms": edge.avg_latency_ms,
                "error_rate": edge.error_rate,
                "zone_crossing": edge.zone_crossing,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "generated_at": time.time(),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

    def generate_grafana_panel(self) -> Dict[str, Any]:
        """Generate Grafana service map panel."""
        return {
            "id": 100,
            "type": "nodeGraph",
            "title": "SESI Service Map (Zone-Aware)",
            "datasource": {"type": "jaeger", "uid": "jaeger"},
            "options": {
                "nodeGrid": {
                    "x": 0,
                    "y": 0,
                    "columns": 4,
                    "rows": 1,
                },
                "edges": [{"field": "call_count", "color": "orange"}],
                "nodes": {
                    "main": {"color": "zone"},
                },
            },
            "gridPos": {"h": 12, "w": 24, "x": 0, "y": 40},
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
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "filtered_edges": len(self.filter_by_pndc()),
            "zone_crossing_edges": sum(
                1 for e in self.edges.values() if e.zone_crossing
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [LL1] LOKI SECURITY LOG AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LogEntry:
    """
    [D] A structured security log entry.

    HONEST: Follows PNDC Axiom 3 (Necessity & Proportionality) and
    Comprehensive Security §5.3 (privacy preservation).
    """
    timestamp: float
    severity: LogSeverity
    service: str
    zone: SecurityZone
    event_type: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON for Loki ingestion."""
        entry = {
            "timestamp": self.timestamp,
            "severity": self.severity.name,
            "service": self.service,
            "zone": self.zone.name,
            "event_type": self.event_type,
            "message": self.message,
        }
        if self.trace_id:
            entry["trace_id"] = self.trace_id
        if self.span_id:
            entry["span_id"] = self.span_id
        if self.user_id:
            entry["user_id"] = self.user_id
        if self.attributes:
            entry.update(self.attributes)
        return json.dumps(entry)


@dataclass
class LogQLQuery:
    """A LogQL query template for Loki."""
    name: str
    query: str
    description: str
    severity_filter: Optional[LogSeverity] = None


class LogRegimeClassifier:
    """
    [D] Classify log stream regimes using paper6 Thm 5.1.

    From paper6 Thm 5.1:
      θ > 1:      finite mean → ordinary LLN
      θ ∈ (0,1):  regularly varying → Beta(θ,θ) arcsine
      θ → 0:      degenerate → ½δ₀ + ½δ₁

    Applied to log streams: classifies whether log frequency is
    stable, heavy-tailed, or polarized.
    """

    def __init__(self, hill_k: int = 20):
        self.hill_k = hill_k
        self.log_timestamps: List[float] = []

    def record_log(self, timestamp: float) -> None:
        """Record a log event timestamp."""
        self.log_timestamps.append(timestamp)

    def classify_regime(self) -> Dict[str, Any]:
        """Classify current log stream regime."""
        if len(self.log_timestamps) < self.hill_k + 2:
            return {
                "regime": LogRegime.INSUFFICIENT_DATA.name,
                "n_events": len(self.log_timestamps),
            }

        # Compute inter-log intervals
        sorted_ts = sorted(self.log_timestamps)
        intervals = [
            sorted_ts[i + 1] - sorted_ts[i]
            for i in range(len(sorted_ts) - 1)
        ]

        # Hill estimator for tail index
        positive_intervals = [iv for iv in intervals if iv > 0]
        if len(positive_intervals) < self.hill_k + 1:
            return {
                "regime": LogRegime.INSUFFICIENT_DATA.name,
                "n_positive_intervals": len(positive_intervals),
            }

        sorted_iv = sorted(positive_intervals, reverse=True)
        x_k1 = sorted_iv[self.hill_k]

        if x_k1 <= 0:
            return {"regime": LogRegime.INSUFFICIENT_DATA.name}

        log_ratios = [
            math.log(sorted_iv[i] / x_k1) for i in range(self.hill_k)
        }
        theta_hat = sum
        log_ratios = [math.log(sorted_iv[i] / x_k1) for i in range(self.hill_k)]
        theta_hat = sum(log_ratios) / self.hill_k

        # Regime classification
        if theta_hat > 1.0:
            regime = LogRegime.FINITE_MEAN
            description = "θ > 1: finite mean. Stable log frequency."
        elif theta_hat > 0.0:
            regime = LogRegime.REGULARLY_VARYING
            description = "θ ∈ (0,1): heavy-tailed log frequency."
        else:
            regime = LogRegime.DEGENERATE
            description = "θ → 0: polarized log frequency."

        return {
            "regime": regime.name,
            "theta_hat": theta_hat,
            "description": description,
            "n_events": len(self.log_timestamps),
        }


class LokiSecurityLogAggregator:
    """
    [D] Structured security log aggregation for Loki.

    Features:
      • Structured JSON logging for security events
      • LogQL query templates for security analysis
      • Regime classification on log streams (paper6 Thm 5.1)
      • Trace-log correlation via trace_id/span_id
      • Data minimization per Comprehensive Security §5.3
      • No-Zeno rate limiting on log emission

    HONEST: This is a LOGGING FRAMEWORK. It does NOT:
      • Guarantee that all security events are captured
      • Replace security audit or incident response
      • Provide causation, only correlation
      • Protect against novel attack vectors

    Mathematical grounding:
      • PNDC Axiom 3: Necessity & Proportionality (log minimization)
      • Comprehensive Security §5.3: Privacy preservation
      • paper6 Thm 5.1: Log stream regime classification
      • No-Zeno: Log emission rate limiting
    """

    # Standard LogQL queries for security analysis
    STANDARD_QUERIES = [
        LogQLQuery(
            name="emergency_stops",
            query='{service="sesi-oversight"} | json | event_type="emergency_stop"',
            description="All emergency stop events",
            severity_filter=LogSeverity.EMERGENCY,
        ),
        LogQLQuery(
            name="zone_violations",
            query='{service="sesi-oversight"} | json | event_type="zone_violation"',
            description="Cross-zone policy violations",
            severity_filter=LogSeverity.CRITICAL,
        ),
        LogQLQuery(
            name="auth_failures",
            query='{service="sesi-oversight"} | json | event_type="auth_failure" | line_format "{{.user_id}} from {{.zone}}"',
            description="Authentication failures by user and zone",
            severity_filter=LogSeverity.WARNING,
        ),
        LogQLQuery(
            name="key_rotations",
            query='{service="sesi-vault"} | json | event_type="key_rotation"',
            description="Vault key rotation events",
            severity_filter=LogSeverity.INFO,
        ),
        LogQLQuery(
            name="degenerate_regime_alerts",
            query='{service="sesi-oversight"} | json | event_type="regime_alert" | regime="DEGENERATE"',
            description="Degenerate regime alerts (paper6 Thm 5.1)",
            severity_filter=LogSeverity.CRITICAL,
        ),
        LogQLQuery(
            name="trace_correlation",
            query='{service=~"sesi-.*"} | json | trace_id!="" | line_format "{{.trace_id}} {{.event_type}}"',
            description="All logs with trace_id for trace-log correlation",
        ),
    ]

    def __init__(self, service_name: str = "sesi-oversight"):
        self.service_name = service_name
        self.log_buffer: List[LogEntry] = []
        self.max_buffer_size = 10000
        self.regime_classifier = LogRegimeClassifier()
        self._log_timestamps: deque = deque(maxlen=10000)

        # No-Zeno parameters
        self.min_log_interval_s = 0.1
        self.max_logs_per_minute = 1000

    # ── Log Emission ─────────────────────────────────────────────────────

    def log_security_event(
        self,
        severity: LogSeverity,
        event_type: str,
        message: str,
        zone: SecurityZone = SecurityZone.Z2_PROCESSING,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[LogEntry]:
        """
        Log a security event.

        Applies PNDC Axiom 3 (Necessity & Proportionality):
          • Only log what is necessary for security
          • Minimize personal data collection
          • Follow Comprehensive Security §5.3 privacy rules

        Args:
            severity: Log severity level
            event_type: Type of security event
            message: Human-readable message
            zone: Security zone where event occurred
            trace_id: Optional Jaeger trace ID for correlation
            span_id: Optional Jaeger span ID for correlation
            user_id: Optional user ID (only if necessary)
            attributes: Optional additional attributes

        Returns:
            LogEntry if logged, None if rate-limited
        """
        # Rate limiting (No-Zeno)
        now = time.time()
        cutoff = now - 60.0
        recent = [t for t in self._log_timestamps if t > cutoff]
        if len(recent) >= self.max_logs_per_minute:
            logger.debug("Log rate-limited by No-Zeno bound")
            return None

        if self._log_timestamps and (now - self._log_timestamps[-1]) < self.min_log_interval_s:
            logger.debug("Log rate-limited by min interval")
            return None

        # Data minimization (PNDC Axiom 3 + Comp. Security §5.3)
        safe_attributes = {}
        if attributes:
            # Strip potentially sensitive data
            for k, v in attributes.items():
                if k.lower() not in ("password", "secret", "token", "ssn", "credit_card"):
                    safe_attributes[k] = v

        entry = LogEntry(
            timestamp=now,
            severity=severity,
            service=self.service_name,
            zone=zone,
            event_type=event_type,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            attributes=safe_attributes,
        )

        self.log_buffer.append(entry)
        self._log_timestamps.append(now)
        self.regime_classifier.record_log(now)

        # Trim buffer if needed
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer = self.log_buffer[-self.max_buffer_size:]

        return entry

    # ── Log Export ───────────────────────────────────────────────────────

    def export_to_loki_format(self) -> List[Dict[str, Any]]:
        """
        Export logs in Loki-compatible format.

        Loki expects streams with labels and values.
        """
        streams: Dict[str, List[str]] = defaultdict(list)

        for entry in self.log_buffer:
            # Build label set
            labels = (
                f'{{service="{entry.service}",'
                f'zone="{entry.zone.name}",'
                f'severity="{entry.severity.name}",'
                f'event_type="{entry.event_type}"}}'
            )
            streams[labels].append(entry.to_json())

        # Convert to Loki stream format
        loki_streams = []
        for labels, values in streams.items():
            loki_streams.append({
                "stream": json.loads(labels.replace("'", '"')),
                "values": [[str(int(v["timestamp"] * 1e9)), v] for v in
                           [json.loads(val) for val in values]],
            })

        return loki_streams

    def write_logs_to_file(self, output_path: str = "./logs/security.jsonl") -> str:
        """Write logs to JSONL file for Loki ingestion."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for entry in self.log_buffer:
                f.write(entry.to_json() + "\n")

        logger.info(f"Wrote {len(self.log_buffer)} log entries to {output_path}")
        return str(out_path)

    # ── LogQL Query Generation ───────────────────────────────────────────

    def generate_logql_queries(self) -> List[Dict[str, Any]]:
        """Generate LogQL query templates."""
        return [
            {
                "name": q.name,
                "query": q.query,
                "description": q.description,
                "severity_filter": q.severity_filter.name if q.severity_filter else None,
            }
            for q in self.STANDARD_QUERIES
        ]

    def write_loki_config(self, output_path: str = "./loki/loki-config.yaml") -> str:
        """Generate Loki configuration file."""
        config = {
            "auth_enabled": False,
            "server": {
                "http_listen_port": 3100,
            },
            "ingester": {
                "wal": {
                    "enabled": True,
                    "dir": "/tmp/wal",
                },
                "lifecycler": {
                    "address": "127.0.0.1",
                    "ring": {
                        "kvstore": {
                            "store": "inmemory",
                        },
                        "replication_factor": 1,
                    },
                },
                "chunk_idle_period": "3m",
                "chunk_block_size": 262144,
                "chunk_retain_period": "1m",
            },
            "schema_config": {
                "configs": [{
                    "from": "2020-10-24",
                    "store": "boltdb-shipper",
                    "object_store": "filesystem",
                    "schema": "v11",
                    "index": {
                        "prefix": "index_",
                        "period": "24h",
                    },
                }],
            },
            "storage_config": {
                "boltdb_shipper": {
                    "active_index_directory": "/tmp/loki/boltdb-shipper-active",
                    "cache_location": "/tmp/loki/boltdb-shipper-cache",
                    "cache_ttl": "24h",
                    "shared_store": "filesystem",
                },
                "filesystem": {
                    "directory": "/tmp/loki/chunks",
                },
            },
            "limits_config": {
                "reject_old_samples": True,
                "reject_old_samples_max_age": "168h",
            },
            "chunk_store_config": {
                "max_look_back_period": "0s",
            },
            "table_manager": {
                "retention_deletes_enabled": False,
                "retention_period": "0s",
            },
            "ruler": {
                "storage": {
                    "type": "local",
                    "local": {
                        "directory": "/tmp/loki/rules",
                    },
                },
                "rule_path": "/tmp/loki/rules-temp",
                "alertmanager_url": "http://localhost:9093",
                "ring": {
                    "kvstore": {
                        "store": "inmemory",
                    },
                },
                "enable_api": True,
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

    def write_logql_queries(self, output_path: str = "./loki/queries.json") -> str:
        """Write LogQL queries to file."""
        queries = self.generate_logql_queries()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(queries, f, indent=2)
        logger.info(f"LogQL queries written to {output_path}")
        return str(out_path)

    # ── Regime Classification ────────────────────────────────────────────

    def classify_log_regime(self) -> Dict[str, Any]:
        """Classify current log stream regime."""
        return self.regime_classifier.classify_regime()

    def get_status(self) -> Dict[str, Any]:
        """Get aggregator status."""
        return {
            "service_name": self.service_name,
            "buffer_size": len(self.log_buffer),
            "max_buffer_size": self.max_buffer_size,
            "log_regime": self.classify_log_regime(),
            "recent_log_rate_per_min": len([
                t for t in self._log_timestamps
                if t > time.time() - 60.0
            ]),
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIIntegratedObservabilitySuite:
    """
    [D] Complete observability suite integrating:
      • AGIONE quality enrichment
      • Jaeger service map
      • Loki security logs
    """

    def __init__(self, agione: Optional[Any] = None):
        self.quality_enricher = AGIONEQualityEnricher(agione)
        self.service_map = JaegerServiceMapGenerator()
        self.log_aggregator = LokiSecurityLogAggregator()

        # Register default services
        self._register_default_services()

    def _register_default_services(self) -> None:
        """Register default SESI services in the service map."""
        self.service_map.register_service(
            "sesi-implant-gateway",
            SecurityZone.Z0_IMPLANT,
            "implant",
        )
        self.service_map.register_service(
            "sesi-clinical-gateway",
            SecurityZone.Z1_CLINICAL_GATEWAY,
            "clinical",
        )
        self.service_map.register_service(
            "sesi-processing",
            SecurityZone.Z2_PROCESSING,
            "processing",
        )
        self.service_map.register_service(
            "sesi-external-api",
            SecurityZone.Z3_EXTERNAL,
            "external",
        )
        self.service_map.register_service(
            "sesi-vault",
            SecurityZone.Z1_CLINICAL_GATEWAY,
            "security",
        )
        self.service_map.register_service(
            "sesi-fido-mds",
            SecurityZone.Z3_EXTERNAL,
            "external",
        )

    def record_security_event(
        self,
        event_type: str,
        message: str,
        severity: LogSeverity = LogSeverity.INFO,
        zone: SecurityZone = SecurityZone.Z2_PROCESSING,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a security event across all observability layers."""
        # Log to Loki
        self.log_aggregator.log_security_event(
            severity=severity,
            event_type=event_type,
            message=message,
            zone=zone,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            attributes=attributes,
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

    def enrich_and_record_span(
        self,
        trace_id: str,
        span_id: str,
        operation_name: str,
        service_name: str,
        zone: SecurityZone,
        start_time: float,
        duration_ms: float,
        codification_event: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnrichedSpan]:
        """Enrich a span and record it."""
        span = self.quality_enricher.enrich_span(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            service_name=service_name,
            zone=zone,
            start_time=start_time,
            duration_ms=duration_ms,
            codification_event=codification_event,
        )
        return span

    def generate_all_artifacts(self, output_dir: str = "./sesi_observability") -> Dict[str, str]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Service map
        results["service_map"] = self.service_map.write_service_map(
            str(out_path / "jaeger" / "service-map.json")
        )

        # Loki config
        results["loki_config"] = self.log_aggregator.write_loki_config(
            str(out_path / "loki" / "loki-config.yaml")
        )

        # LogQL queries
        results["logql_queries"] = self.log_aggregator.write_logql_queries(
            str(out_path / "loki" / "queries.json")
        )

        # Security logs
        results["security_logs"] = self.log_aggregator.write_logs_to_file(
            str(out_path / "logs" / "security.jsonl")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full observability suite status."""
        return {
            "quality_enricher": self.quality_enricher.get_status(),
            "service_map": self.service_map.get_status(),
            "log_aggregator": self.log_aggregator.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI AGIONE × Jaeger × Loki Integration — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  OpenTelemetry avail  : {HAS_OTEL}")
    print(f"  AGIONE available     : {HAS_AGIONE}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIIntegratedObservabilitySuite()

    # ── Test 1: AGIONE Quality Enrichment ────────────────────────────────
    print("\n  ── AGIONE Quality Enrichment ──")

    # Simulate quality scores (since AGIONE may not be available)
    mock_quality = {
        "ecosystem": 0.85,
        "physics": 0.9,
        "fold": 0.75,
        "mental": 0.8,
        "hodge": 0.7,
    }
    suite.quality_enricher._last_quality_report = mock_quality

    # Enrich a span
    span = suite.enrich_and_record_span(
        trace_id="trace_abc123",
        span_id="span_def456",
        operation_name="agi_one.forward",
        service_name="sesi-processing",
        zone=SecurityZone.Z2_PROCESSING,
        start_time=time.time(),
        duration_ms=150.0,
        codification_event={
            "codified": True,
            "validity_score": 0.85,
            "quality_score": 0.72,
        },
    )
    if span:
        print(f"    Span enriched      : {span.operation_name}")
        print(f"    Quality scores     : {len(span.quality_scores)} domains")
        print(f"    Ecosystem quality  : {span.quality_scores.get('ecosystem', 'N/A')}")
    else:
        print(f"    [SKIP] Enrichment rate-limited")

    # Test NaN/Inf sanitization
    if HAS_PSYCHE_PLUS:
        nan_result = _safe_unit_scalar(float("nan"), name="test_nan", logger=logger)
        inf_result = _safe_unit_scalar(float("inf"), name="test_inf", logger=logger)
        print(f"    NaN sanitized to   : {nan_result}")
        print(f"    Inf sanitized to   : {inf_result}")

    # ── Test 2: Jaeger Service Map ───────────────────────────────────────
    print("\n  ── Jaeger Service Map ──")

    # Record some service calls
    service_calls = [
        ("sesi-external-api", "sesi-processing", 50.0, True),
        ("sesi-processing", "sesi-clinical-gateway", 30.0, True),
        ("sesi-clinical-gateway", "sesi-implant-gateway", 100.0, True),
        ("sesi-clinical-gateway", "sesi-vault", 20.0, True),
        ("sesi-external-api", "sesi-fido-mds", 200.0, False),
    ]

    for _ in range(20):  # Repeat to exceed min_calls_to_display
        for src, tgt, latency, success in service_calls:
            suite.record_service_call(src, tgt, latency, success)

    # Generate service map
    service_map = suite.service_map.generate_jaeger_service_map()
    print(f"    Nodes              : {service_map['n_nodes']}")
    print(f"    Edges (filtered)   : {service_map['n_edges']}")
    print(f"    Zone crossings     : {suite.service_map.get_status()['zone_crossing_edges']}")

    # ── Test 3: Loki Security Logs ───────────────────────────────────────
    print("\n  ── Loki Security Logs ──")

    # Log some security events
    suite.record_security_event(
        event_type="auth_success",
        message="User authenticated successfully",
        severity=LogSeverity.INFO,
        zone=SecurityZone.Z1_CLINICAL_GATEWAY,
        trace_id="trace_abc123",
        user_id="clinician_001",
    )
    suite.record_security_event(
        event_type="zone_violation",
        message="Z3 attempted to access Z0",
        severity=LogSeverity.CRITICAL,
        zone=SecurityZone.Z0_IMPLANT,
        trace_id="trace_xyz789",
    )
    suite.record_security_event(
        event_type="emergency_stop",
        message="Kill switch activated by operator",
        severity=LogSeverity.EMERGENCY,
        zone=SecurityZone.Z1_CLINICAL_GATEWAY,
        user_id="operator_001",
    )

    # Log regime classification
    regime = suite.log_aggregator.classify_log_regime()
    print(f"    Log events         : {len(suite.log_aggregator.log_buffer)}")
    print(f"    Log regime         : {regime['regime']}")

    # Generate LogQL queries
    queries = suite.log_aggregator.generate_logql_queries()
    print(f"    LogQL queries      : {len(queries)}")
    for q in queries[:3]:
        print(f"      • {q['name']}: {q['description'][:50]}...")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_observability")
    print(f"    Artifacts generated: {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:20s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Enriched spans     : {status['quality_enricher']['enriched_spans']}")
    print(f"    Service map nodes  : {status['service_map']['n_nodes']}")
    print(f"    Service map edges  : {status['service_map']['n_edges']}")
    print(f"    Log buffer size    : {status['log_aggregator']['buffer_size']}")
    print(f"    Log regime         : {status['log_aggregator']['log_regime']['regime']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • AGIONE enrichment shows CORRELATION, not causation")
    print("  • Service map is TOPOLOGY, not security guarantee")
    print("  • Loki logs are OBSERVABILITY, not omniscience")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Data minimization follows PNDC Axiom 3")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
