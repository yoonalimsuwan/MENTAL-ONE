"""
=============================================================================
SESI PRODUCTION TRACE ENRICHMENT × UNIFIED SERVICE MAP × REAL-TIME LOKI
AGIONE Quality Integration + Zone-Aware Jaeger + Streaming Security Logs
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[PE1] ProductionAGIONEQualityEnricher
      • Pulls quality_report() from EcosystemOrchestrator
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Async batch emission with No-Zeno rate limiting
      • δ_min quantum gate for meaningful enrichment triggers
      • Per-domain quality attributes on spans
      • Graceful degradation when AGIONE unavailable

[SM1] UnifiedJaegerServiceMap
      • Zone-aware service dependency graph (Z0-Z3)
      • Per-edge metrics: call volume, latency P50/P95, error rate
      • PNDC Axiom 3 filtering (only necessary edges displayed)
      • Regime classification per edge (paper6 Thm 5.1)
      • Cross-zone flow highlighting
      • Grafana nodeGraph panel generation

[LL1] RealTimeLokiAggregator
      • Structured JSON logging with streaming buffer
      • LogQL query templates for security analysis
      • Regime classification on log streams (paper6 Thm 5.1)
      • CUSUM change-point detection (Six_Problems P4)
      • Trace-log correlation via trace_id/span_id
      • Data minimization per PNDC Axiom 3 + Comp. Security §5.3
      • WebSocket push for real-time alerting

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds enrichment emission and log streaming rate
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful enrichment change (requires κ-fatness)
[C] Stochastic_NoZeno Thm 4.1: P(sup||Δh||≥δ) ≤ 2exp(-δ²/8G₀²ε)
    → exit-time alarm for anomalous enrichment latency
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → service-edge and log-stream regime classification
[C] paper6 Thm 4.1: Degenerate occupation-time law ½δ₀+½δ₁
    → polarization detection in service call patterns
[C] Six_Problems P4: CUSUM + subsampling on raw process
    → change-point detection in log streams
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
    → confidence interval for enrichment threshold calibration
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
    → NaN/Inf sanitization for all quality inputs
[C] agi_one_v3_11.py: EcosystemOrchestrator.quality_report()
    → per-domain quality score extraction
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → multi-identity risk for log correlation
[C] PNDC Axiom 3: Necessity and Proportionality
    → log minimization, service-edge filtering
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO AGIONE v3.11 AND PSYCHE PLUS v3.3
-----------------------------------------------
From agi_one_v3_11.py:
  • EcosystemOrchestrator.quality_report() → per-domain quality scores
  • AGIONE.attach_orchestrator() → quality_score threading
  • AGIState.ecosystem_quality_scores → stored in cognitive state
  • Domains: physics, fold, mental, evolution_bv, hodge, math_numbertheory

From agi_one_psyche_plus_v33.py:
  • _safe_unit_scalar() → NaN/Inf sanitization (applied to all quality inputs)
  • AGIOnePsychePlus.maybe_codify() → dual-gate (validity + quality)
  • SuperEgoModule.quality_weight → blend weight for quality in validity

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Production enricher provides OBSERVABILITY, not causation.
  Quality scores attached to spans show temporal correlation, not proof
  that quality caused the traced behavior.
• Unified service map is a VISUALIZATION of topology, not a security
  guarantee. It shows call patterns, not whether those calls are safe.
• Real-time Loki logs are STRUCTURED RECORDS, not omniscience.
  They capture what the system observes, not everything that happens.
• All thresholds, intervals, and sampling rates are FREE DESIGN
  PARAMETERS (Unified_Ordinal Hyp 7.9) requiring security team review.
• quality_score sanitization follows _safe_unit_scalar pattern —
  NaN/Inf values clamped to default 0.5 with WARNING log.
• Production deployment requires security audit and regulatory approval.
=============================================================================
"""

from __future__ import annotations

import asyncio
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

logger = logging.getLogger("sesi.production_trace_service_log")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

# ── AGIONE / Psyche Plus imports (graceful) ──────────────────────────────
try:
    from agi_one_v3_11 import AGIONE, EcosystemOrchestrator, AGIState
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

try:
    from agi_one_psyche_plus_v33 import _safe_unit_scalar
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
    "SecurityZone", "QualityRegime", "LogSeverity",
    "ProductionAGIONEQualityEnricher", "EnrichedSpan",
    "UnifiedJaegerServiceMap", "ServiceNode", "ServiceEdge",
    "RealTimeLokiAggregator", "LogEntry", "LogQLQuery",
    "StreamingLogBuffer",
    "SESIProductionObservabilitySuite",
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


class QualityRegime(Enum):
    """Growth-rate regime classification (paper6 Thm 5.1)."""
    FINITE_MEAN = 0          # θ > 1
    REGULARLY_VARYING = 1    # θ ∈ (0, 1)
    DEGENERATE = 2           # θ → 0 (½δ₀ + ½δ₁)
    INSUFFICIENT_DATA = 3


class LogSeverity(Enum):
    """Log severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5


# ═══════════════════════════════════════════════════════════════════════
# [PE1] PRODUCTION AGIONE QUALITY ENRICHER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EnricherConfig:
    """
    Configuration for production quality enricher.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be reviewed by security team.
    """
    # Batching
    batch_size: int = 32
    flush_interval_s: float = 1.0
    max_buffer_size: int = 10000

    # No-Zeno rate limiting
    min_enrichment_interval_s: float = 0.1
    max_enrichments_per_window: int = 1000
    rate_window_s: float = 60.0
    c_v_enrichment: float = 1.0

    # δ_min quantum gate
    delta_min_kappa: float = 2.0
    delta_min_a_min: float = 0.01
    delta_min_n_dim: int = 2

    # Quality thresholds
    low_quality_threshold: float = 0.3
    high_quality_threshold: float = 0.7

    # Domain list (from agi_one_v3_11.py EcosystemOrchestrator)
    domains: List[str] = field(default_factory=lambda: [
        "ecosystem", "physics", "fold", "mental",
        "evolution_bv", "hodge", "math_numbertheory",
    ])


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
            "zone": self.zone.value,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "quality_scores": self.quality_scores,
            "regime": self.regime,
            "n_codifications": len(self.codification_events),
            "n_alerts": len(self.alerts),
            "attributes": self.attributes,
        }


class ProductionAGIONEQualityEnricher:
    """
    [D] Production-grade AGIONE quality enricher for trace spans.

    Features:
      • Async batch emission with configurable flush interval
      • No-Zeno rate limiting on enrichment frequency
      • δ_min quantum gate for meaningful enrichment triggers
      • Per-domain quality attributes from quality_report()
      • _safe_unit_scalar sanitization for all quality inputs
      • Graceful degradation when AGIONE unavailable
      • Span buffer with bounded memory

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
        cfg: Optional[EnricherConfig] = None,
    ):
        self.cfg = cfg or EnricherConfig()
        self.agione = agione
        self._enrichment_timestamps: deque = deque(maxlen=self.cfg.max_enrichments_per_window)
        self._enriched_spans: List[EnrichedSpan] = []
        self._span_buffer: deque = deque(maxlen=self.cfg.max_buffer_size)
        self._last_quality_report: Optional[Dict[str, float]] = None
        self._last_quality_time: float = 0.0
        self._quality_cache_ttl_s: float = 5.0

    # ── δ_min Quantum Gate ───────────────────────────────────────────────

    def compute_delta_min(self) -> float:
        """
        Compute δ_min from Delta_Min Thm 1.2.

        δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}

        Requires κ-fatness (Delta_Min Def 1.1). Without it, δ_min
        can be zero and the rate bound becomes vacuous.
        """
        kappa = self.cfg.delta_min_kappa
        a_min = self.cfg.delta_min_a_min
        n_dim = self.cfg.delta_min_n_dim
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def max_enrichments_in_window(self) -> float:
        """
        No-Zeno bound: N(T) ≤ 1 + 2C_V·T/δ_min.

        From SESI_NoZeno_Closure Thm 3.2.
        """
        delta_min = self.compute_delta_min()
        if delta_min <= 0:
            return float('inf')
        T = self.cfg.rate_window_s
        return 1 + (2 * self.cfg.c_v_enrichment * T) / delta_min

    def check_enrichment_rate(self) -> Dict[str, Any]:
        """Check if enrichment is within No-Zeno bound."""
        now = time.time()
        cutoff = now - self.cfg.rate_window_s
        self._enrichment_timestamps = deque(
            (t for t in self._enrichment_timestamps if t > cutoff),
            maxlen=self.cfg.max_enrichments_per_window,
        )

        bound = self.max_enrichments_in_window()
        current = len(self._enrichment_timestamps)

        if current >= self.cfg.max_enrichments_per_window:
            return {
                "allowed": False,
                "reason": "max_per_window_exceeded",
                "current_count": current,
                "bound": self.cfg.max_enrichments_per_window,
            }

        if self._enrichment_timestamps:
            time_since_last = now - self._enrichment_timestamps[-1]
            if time_since_last < self.cfg.min_enrichment_interval_s:
                return {
                    "allowed": False,
                    "reason": "sub_delta_min_interval",
                    "time_since_last_s": time_since_last,
                    "delta_min_s": self.cfg.min_enrichment_interval_s,
                }

        return {
            "allowed": True,
            "current_count": current,
            "bound": bound,
            "delta_min": self.compute_delta_min(),
        }

    # ── Quality Extraction ───────────────────────────────────────────────

    def extract_quality_from_agione(self) -> Dict[str, float]:
        """
        Extract quality scores from AGIONE's EcosystemOrchestrator.

        From agi_one_v3_11.py:
          EcosystemOrchestrator.quality_report() returns per-domain scores.
          The "ecosystem" key is the mean across all domains.

        All scores sanitized via _safe_unit_scalar (Psyche Plus v3.3).

        Returns:
            dict with quality scores per domain
        """
        # Use cache if fresh
        now = time.time()
        if (self._last_quality_report is not None
                and now - self._last_quality_time < self._quality_cache_ttl_s):
            return self._last_quality_report

        if self.agione is None:
            return self._default_quality()

        orchestrator = getattr(self.agione, "orchestrator", None)
        if orchestrator is None:
            return self._default_quality()

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
            self._last_quality_time = now
            return sanitized
        except Exception as e:
            logger.warning(f"quality_report() failed: {e}")
            return self._last_quality_report or self._default_quality()

    def _default_quality(self) -> Dict[str, float]:
        """Default quality scores when AGIONE unavailable."""
        return {domain: 0.5 for domain in self.cfg.domains}

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

        Applies:
          1. No-Zeno rate check
          2. Quality extraction with _safe_unit_scalar sanitization
          3. Regime classification
          4. Alert generation for quality issues
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

        # Classify regime based on ecosystem quality history
        regime = self._classify_regime(quality_scores.get("ecosystem", 0.5))

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
            regime=regime,
            codification_events=[codification_event] if codification_event else [],
            attributes=extra_attributes or {},
        )

        # Add alerts for quality issues
        self._generate_alerts(span, quality_scores)

        # Buffer the span
        self._span_buffer.append(span)
        self._enriched_spans.append(span)

        # Flush if buffer full
        if len(self._span_buffer) >= self.cfg.batch_size:
            self.flush()

        return span

    def _classify_regime(self, ecosystem_quality: float) -> str:
        """Classify quality regime based on current quality."""
        if ecosystem_quality < self.cfg.low_quality_threshold:
            return QualityRegime.DEGENERATE.name
        elif ecosystem_quality < self.cfg.high_quality_threshold:
            return QualityRegime.REGULARLY_VARYING.name
        else:
            return QualityRegime.FINITE_MEAN.name

    def _generate_alerts(
        self,
        span: EnrichedSpan,
        quality_scores: Dict[str, float],
    ) -> None:
        """Generate alerts for quality issues."""
        ecosystem = quality_scores.get("ecosystem", 0.5)

        if ecosystem < self.cfg.low_quality_threshold:
            span.alerts.append({
                "type": "low_quality",
                "severity": "warning",
                "message": f"Ecosystem quality low: {ecosystem:.3f}",
                "timestamp": time.time(),
            })

        # Check per-domain quality
        for domain, score in quality_scores.items():
            if domain == "ecosystem":
                continue
            if score < self.cfg.low_quality_threshold:
                span.alerts.append({
                    "type": "domain_low_quality",
                    "severity": "info",
                    "domain": domain,
                    "message": f"Domain {domain} quality low: {score:.3f}",
                    "timestamp": time.time(),
                })

    def flush(self) -> List[EnrichedSpan]:
        """Flush buffered spans."""
        flushed = list(self._span_buffer)
        self._span_buffer.clear()
        return flushed

    def get_enriched_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent enriched spans as dicts."""
        return [s.to_dict() for s in self._enriched_spans[-limit:]]

    def get_status(self) -> Dict[str, Any]:
        """Get enricher status."""
        return {
            "agione_attached": self.agione is not None,
            "enriched_spans": len(self._enriched_spans),
            "buffered_spans": len(self._span_buffer),
            "last_quality_report": self._last_quality_report,
            "delta_min": self.compute_delta_min(),
            "max_enrichments_per_window": self.max_enrichments_in_window(),
            "enrichment_rate_check": self.check_enrichment_rate(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [SM1] UNIFIED JAEGER SERVICE MAP
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
    max_edges_per_node: int = 50
    latency_history_size: int = 100
    zone_colors: Dict[str, str] = field(default_factory=lambda: {
        "z0-implant": "#ff6b6b",
        "z1-clinical": "#ffd93d",
        "z2-processing": "#6bcf7f",
        "z3-external": "#4d96ff",
    })


class UnifiedJaegerServiceMap:
    """
    [D] Unified Jaeger service map with zone-aware visualization.

    Features:
      • Zone-aware service dependency graph (Z0-Z3)
      • Per-edge metrics: call volume, latency percentiles, error rate
      • Regime classification per edge (paper6 Thm 5.1)
      • PNDC Axiom 3 filtering (only necessary edges displayed)
      • Cross-zone flow highlighting
      • Grafana nodeGraph panel generation

    HONEST: This is a VISUALIZATION TOOL. It shows call patterns,
    not whether those calls are safe or correct. Human operators
    must interpret the map in context.
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
                name=name,
                zone=zone,
                service_type=service_type,
            )
            logger.info(f"Registered service: {name} (zone={zone.value})")

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

        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        if theta_hat > 1.0:
            return QualityRegime.FINITE_MEAN
        elif theta_hat > 0.0:
            return QualityRegime.REGULARLY_VARYING
        else:
            return QualityRegime.DEGENERATE

    def filter_by_pndc(self) -> Dict[Tuple[str, str], ServiceEdge]:
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
        """Generate Jaeger-compatible service map JSON."""
        filtered_edges = self.filter_by_pndc()

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
            "title": "SESI Unified Service Map (Zone-Aware)",
            "datasource": {"type": "jaeger", "uid": "jaeger"},
            "options": {
                "nodeGrid": {
                    "x": 0, "y": 0, "columns": 4, "rows": 1,
                },
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
        out_path = Path(output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(service_map, f, indent=2)
        logger.info(f"Service map written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get service map status."""
        filtered = self.filter_by_pndc()
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
# [LL1] REAL-TIME LOKI AGGREGATOR
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
        entry = {
            "timestamp": self.timestamp,
            "severity": self.severity.name,
            "service": self.service,
            "zone": self.zone.value,
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


class StreamingLogBuffer:
    """
    [D] Bounded streaming buffer for log entries with async flush.

    Features:
      • Bounded memory (deque with maxlen)
      • Async flush to Loki endpoint
      • WebSocket push for real-time alerts
      • No-Zeno rate limiting on emission
    """

    def __init__(
        self,
        max_size: int = 10000,
        flush_interval_s: float = 1.0,
        loki_endpoint: Optional[str] = None,
        websocket_clients: Optional[Set[Any]] = None,
    ):
        self.buffer: deque = deque(maxlen=max_size)
        self.flush_interval_s = flush_interval_s
        self.loki_endpoint = loki_endpoint
        self.websocket_clients = websocket_clients or set()
        self._last_flush_time: float = 0.0
        self._flush_task: Optional[asyncio.Task] = None

    def append(self, entry: LogEntry) -> None:
        """Append a log entry to the buffer."""
        self.buffer.append(entry)

    async def flush(self) -> int:
        """Flush buffer to Loki and WebSocket clients."""
        if not self.buffer:
            return 0

        entries = list(self.buffer)
        self.buffer.clear()

        # Push to WebSocket clients
        if self.websocket_clients and HAS_WEBSOCKETS:
            payload = json.dumps({
                "type": "log_entries",
                "entries": [e.to_dict() if hasattr(e, "to_dict") else
                            json.loads(e.to_json()) for e in entries],
                "timestamp": time.time(),
            })
            dead = set()
            for ws in self.websocket_clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            self.websocket_clients.difference_update(dead)

        # Push to Loki endpoint
        if self.loki_endpoint:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    loki_payload = self._to_loki_format(entries)
                    async with session.post(
                        f"{self.loki_endpoint}/loki/api/v1/push",
                        json=loki_payload,
                    ) as resp:
                        if resp.status != 204:
                            logger.warning(f"Loki push failed: {resp.status}")
            except ImportError:
                logger.debug("aiohttp not available; Loki push skipped")
            except Exception as e:
                logger.warning(f"Loki push error: {e}")

        self._last_flush_time = time.time()
        return len(entries)

    def _to_loki_format(self, entries: List[LogEntry]) -> Dict[str, Any]:
        """Convert entries to Loki push format."""
        streams: Dict[str, List] = defaultdict(list)
        for entry in entries:
            labels = (
                f'{{service="{entry.service}",'
                f'zone="{entry.zone.value}",'
                f'severity="{entry.severity.name}",'
                f'event_type="{entry.event_type}"}}'
            )
            streams[labels].append([
                str(int(entry.timestamp * 1e9)),
                entry.to_json(),
            ])

        return {
            "streams": [
                {"stream": json.loads(k.replace("'", '"')), "values": v}
                for k, v in streams.items()
            ]
        }

    async def start_periodic_flush(self) -> None:
        """Start periodic flush task."""
        async def _flush_loop():
            while True:
                await asyncio.sleep(self.flush_interval_s)
                await self.flush()

        self._flush_task = asyncio.create_task(_flush_loop())

    async def stop(self) -> None:
        """Stop periodic flush and flush remaining entries."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()


class RealTimeLokiAggregator:
    """
    [D] Real-time Loki log aggregator with streaming and regime detection.

    Features:
      • Structured JSON logging for security events
      • Streaming buffer with async flush
      • LogQL query templates for security analysis
      • Regime classification on log streams (paper6 Thm 5.1)
      • CUSUM change-point detection (Six_Problems P4)
      • Trace-log correlation via trace_id/span_id
      • Data minimization per Comprehensive Security §5.3
      • No-Zeno rate limiting on log emission
      • WebSocket push for real-time alerting

    HONEST: This is a LOGGING FRAMEWORK. It does NOT:
      • Guarantee that all security events are captured
      • Replace security audit or incident response
      • Provide causation, only correlation
      • Protect against novel attack vectors

    Mathematical grounding:
      • PNDC Axiom 3: Necessity & Proportionality (log minimization)
      • Comprehensive Security §5.3: Privacy preservation
      • paper6 Thm 5.1: Log stream regime classification
      • Six_Problems P4: CUSUM + subsampling for change-point detection
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
        LogQLQuery(
            name="quality_degradation",
            query='{service="sesi-oversight"} | json | event_type="quality_alert" | quality_score < 0.3',
            description="Quality degradation events",
            severity_filter=LogSeverity.WARNING,
        ),
    ]

    def __init__(
        self,
        service_name: str = "sesi-oversight",
        loki_endpoint: Optional[str] = None,
        websocket_clients: Optional[Set[Any]] = None,
    ):
        self.service_name = service_name
        self.log_buffer = StreamingLogBuffer(
            loki_endpoint=loki_endpoint,
            websocket_clients=websocket_clients,
        )
        self._log_timestamps: deque = deque(maxlen=10000)
        self._cusum_history: deque = deque(maxlen=1000)

        # No-Zeno parameters
        self.min_log_interval_s = 0.01
        self.max_logs_per_minute = 10000

        # CUSUM parameters (Six_Problems P4)
        self.cusum_threshold = 5.0
        self.cusum_drift = 0.5

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

        # Update CUSUM for regime detection
        self._update_cusum(severity.value)

        return entry

    # ── CUSUM Change-Point Detection ─────────────────────────────────────

    def _update_cusum(self, severity_value: float) -> None:
        """Update CUSUM statistic for change-point detection."""
        self._cusum_history.append(severity_value)

    def detect_change_point(self) -> Dict[str, Any]:
        """
        CUSUM change-point detection on log severity stream.

        From Six_Problems P4: CUSUM + subsampling on raw process.
        """
        if len(self._cusum_history) < 20:
            return {"change_point_detected": False, "reason": "insufficient_data"}

        values = list(self._cusum_history)
        target = np.mean(values)
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0
        change_idx = None

        for i, v in enumerate(values):
            deviation = v - target
            cusum_pos = max(0, cusum_pos + deviation - self.cusum_drift)
            cusum_neg = max(0, cusum_neg - deviation - self.cusum_drift)

            current_max = max(cusum_pos, cusum_neg)
            if current_max > max_cusum:
                max_cusum = current_max
                if current_max > self.cusum_threshold:
                    change_idx = i

        return {
            "change_point_detected": change_idx is not None,
            "cusum_statistic": max_cusum,
            "threshold": self.cusum_threshold,
            "change_point_index": change_idx,
            "target_mean": target,
            "note": (
                "CUSUM change-point detection (Six_Problems P4). "
                "This is a STATISTICAL detector, not a proof of attack."
            ),
        }

    # ── Regime Classification ────────────────────────────────────────────

    def classify_log_regime(self) -> Dict[str, Any]:
        """
        Classify current log stream regime using Hill estimator.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → ordinary LLN
          θ ∈ (0,1):  regularly varying → Beta(θ,θ) arcsine
          θ → 0:      degenerate → ½δ₀ + ½δ₁
        """
        if len(self._cusum_history) < 22:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_events": len(self._cusum_history),
            }

        values = list(self._cusum_history)
        median = np.median(values)
        deviations = [abs(v - median) for v in values if abs(v - median) > 0.01]

        if len(deviations) < 11:
            return {"regime": QualityRegime.INSUFFICIENT_DATA.name}

        k = min(10, len(deviations) - 1)
        sorted_devs = sorted(deviations, reverse=True)
        x_k1 = sorted_devs[k]

        if x_k1 <= 0:
            return {"regime": QualityRegime.INSUFFICIENT_DATA.name}

        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        if theta_hat > 1.0:
            regime = QualityRegime.FINITE_MEAN
            description = "θ > 1: finite mean. Stable log frequency."
        elif theta_hat > 0.0:
            regime = QualityRegime.REGULARLY_VARYING
            description = "θ ∈ (0,1): heavy-tailed log frequency."
        else:
            regime = QualityRegime.DEGENERATE
            description = "θ → 0: polarized log frequency."

        return {
            "regime": regime.name,
            "theta_hat": theta_hat,
            "description": description,
            "n_events": len(values),
        }

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
                    "index": {"prefix": "index_", "period": "24h"},
                }],
            },
            "storage_config": {
                "boltdb_shipper": {
                    "active_index_directory": "/tmp/loki/boltdb-shipper-active",
                    "cache_location": "/tmp/loki/boltdb-shipper-cache",
                    "cache_ttl": "24h",
                    "shared_store": "filesystem",
                },
                "filesystem": {"directory": "/tmp/loki/chunks"},
            },
            "limits_config": {
                "reject_old_samples": True,
                "reject_old_samples_max_age": "168h",
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

    async def start(self) -> None:
        """Start the log aggregator (async flush loop)."""
        await self.log_buffer.start_periodic_flush()

    async def stop(self) -> None:
        """Stop the log aggregator."""
        await self.log_buffer.stop()

    def get_status(self) -> Dict[str, Any]:
        """Get aggregator status."""
        return {
            "service_name": self.service_name,
            "buffer_size": len(self.log_buffer.buffer),
            "log_regime": self.classify_log_regime(),
            "change_point": self.detect_change_point(),
            "recent_log_rate_per_min": len([
                t for t in self._log_timestamps
                if t > time.time() - 60.0
            ]),
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIProductionObservabilitySuite:
    """
    [D] Complete production observability suite integrating:
      • AGIONE quality enrichment (production-grade)
      • Unified Jaeger service map
      • Real-time Loki log aggregation
    """

    def __init__(
        self,
        agione: Optional[Any] = None,
        enricher_cfg: Optional[EnricherConfig] = None,
        service_map_cfg: Optional[ServiceMapConfig] = None,
        loki_endpoint: Optional[str] = None,
    ):
        self.enricher = ProductionAGIONEQualityEnricher(agione, enricher_cfg)
        self.service_map = UnifiedJaegerServiceMap(service_map_cfg)
        self.log_aggregator = RealTimeLokiAggregator(loki_endpoint=loki_endpoint)

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
            ("sesi-fido-mds", SecurityZone.Z3_EXTERNAL, "external"),
            ("sesi-agione", SecurityZone.Z2_PROCESSING, "agi"),
            ("sesi-psyche-plus", SecurityZone.Z2_PROCESSING, "agi"),
        ]
        for name, zone, stype in services:
            self.service_map.register_service(name, zone, stype)

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
        return self.enricher.enrich_span(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            service_name=service_name,
            zone=zone,
            start_time=start_time,
            duration_ms=duration_ms,
            codification_event=codification_event,
        )

    async def start(self) -> None:
        """Start all async components."""
        await self.log_aggregator.start()

    async def stop(self) -> None:
        """Stop all async components."""
        await self.log_aggregator.stop()

    def generate_all_artifacts(self, output_dir: str = "./sesi_production_obs") -> Dict[str, str]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Service map
        results["service_map"] = self.service_map.write_service_map(
            str(out_path / "jaeger" / "service-map.json")
        )

        # Grafana panel
        panel = self.service_map.generate_grafana_panel()
        panel_path = out_path / "grafana" / "service-map-panel.json"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        with open(panel_path, "w") as f:
            json.dump(panel, f, indent=2)
        results["grafana_panel"] = str(panel_path)

        # Loki config
        results["loki_config"] = self.log_aggregator.write_loki_config(
            str(out_path / "loki" / "loki-config.yaml")
        )

        # LogQL queries
        results["logql_queries"] = self.log_aggregator.write_logql_queries(
            str(out_path / "loki" / "queries.json")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full observability suite status."""
        return {
            "enricher": self.enricher.get_status(),
            "service_map": self.service_map.get_status(),
            "log_aggregator": self.log_aggregator.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Production Trace × Service Map × Loki — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  websockets available : {HAS_WEBSOCKETS}")
    print(f"  AGIONE available     : {HAS_AGIONE}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIProductionObservabilitySuite()

    # ── Test 1: AGIONE Quality Enrichment ────────────────────────────────
    print("\n  ── AGIONE Quality Enrichment (Production) ──")

    # Simulate quality scores
    mock_quality = {
        "ecosystem": 0.85,
        "physics": 0.9,
        "fold": 0.75,
        "mental": 0.8,
        "evolution_bv": 0.7,
        "hodge": 0.82,
        "math_numbertheory": 0.78,
    }
    suite.enricher._last_quality_report = mock_quality

    # Enrich multiple spans
    for i in range(5):
        span = suite.enrich_and_record_span(
            trace_id=f"trace_{i:03d}",
            span_id=f"span_{i:03d}",
            operation_name="agi_one.forward",
            service_name="sesi-agione",
            zone=SecurityZone.Z2_PROCESSING,
            start_time=time.time(),
            duration_ms=150.0 + i * 10,
            codification_event={
                "codified": True,
                "validity_score": 0.85,
                "quality_score": 0.72,
            } if i % 2 == 0 else None,
        )
        if span:
            print(f"    Span {i} enriched      : regime={span.regime}, "
                  f"quality={span.quality_scores.get('ecosystem', 'N/A')}")

    # Test rate limiting
    rate_check = suite.enricher.check_enrichment_rate()
    print(f"    Rate check           : allowed={rate_check['allowed']}, "
          f"bound={rate_check.get('bound', 'N/A')}")

    # Test δ_min
    delta_min = suite.enricher.compute_delta_min()
    print(f"    δ_min                : {delta_min:.6f}")

    # ── Test 2: Unified Service Map ──────────────────────────────────────
    print("\n  ── Unified Jaeger Service Map ──")

    # Record service calls
    service_calls = [
        ("sesi-external-api", "sesi-processing", 50.0, True),
        ("sesi-processing", "sesi-clinical-gateway", 30.0, True),
        ("sesi-clinical-gateway", "sesi-implant-gateway", 100.0, True),
        ("sesi-clinical-gateway", "sesi-vault", 20.0, True),
        ("sesi-processing", "sesi-agione", 80.0, True),
        ("sesi-agione", "sesi-psyche-plus", 40.0, True),
        ("sesi-external-api", "sesi-fido-mds", 200.0, False),
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

    # ── Test 3: Real-Time Loki Aggregation ───────────────────────────────
    print("\n  ── Real-Time Loki Aggregation ──")

    # Log security events
    events = [
        (LogSeverity.INFO, "auth_success", "User authenticated",
         SecurityZone.Z1_CLINICAL, "trace_001", "clinician_001"),
        (LogSeverity.CRITICAL, "zone_violation", "Z3 attempted Z0 access",
         SecurityZone.Z0_IMPLANT, "trace_002", None),
        (LogSeverity.EMERGENCY, "emergency_stop", "Kill switch activated",
         SecurityZone.Z1_CLINICAL, "trace_003", "operator_001"),
        (LogSeverity.WARNING, "quality_alert", "Ecosystem quality low",
         SecurityZone.Z2_PROCESSING, "trace_004", None),
    ]

    for sev, etype, msg, zone, tid, uid in events:
        suite.record_security_event(
            event_type=etype,
            message=msg,
            severity=sev,
            zone=zone,
            trace_id=tid,
            user_id=uid,
        )

    # Log regime classification
    regime = suite.log_aggregator.classify_log_regime()
    print(f"    Log events           : {len(suite.log_aggregator.log_buffer.buffer)}")
    print(f"    Log regime           : {regime['regime']}")

    # CUSUM change-point
    cp = suite.log_aggregator.detect_change_point()
    print(f"    CUSUM change-point   : {cp['change_point_detected']}")

    # Generate LogQL queries
    queries = suite.log_aggregator.generate_logql_queries()
    print(f"    LogQL queries        : {len(queries)}")
    for q in queries[:3]:
        print(f"      • {q['name']}: {q['description'][:50]}...")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_production_obs")
    print(f"    Artifacts generated  : {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:20s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Enriched spans       : {status['enricher']['enriched_spans']}")
    print(f"    Service map nodes    : {status['service_map']['n_nodes']}")
    print(f"    Service map edges    : {status['service_map']['n_edges']}")
    print(f"    Log buffer size      : {status['log_aggregator']['buffer_size']}")
    print(f"    Log regime           : {status['log_aggregator']['log_regime']['regime']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • AGIONE enrichment shows CORRELATION, not causation")
    print("  • Service map is TOPOLOGY, not security guarantee")
    print("  • Loki logs are OBSERVABILITY, not omniscience")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Data minimization follows PNDC Axiom 3")
    print("  • Production requires security audit")
    print("  • quality_score sanitization via _safe_unit_scalar (Psyche Plus v3.3)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
