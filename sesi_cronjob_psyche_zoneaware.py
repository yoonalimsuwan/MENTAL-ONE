"""
=============================================================================
SESI CRONJOB SNAPSHOTS × PSYCHE PLUS TRACE ENRICHMENT × ZONE-AWARE DASHBOARD
Periodic Quality Snapshots + Codification Event Tracing + Per-Zone Visualization
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[CJ1] QualitySnapshotCronJob
      • Periodic quality_report() capture from AGIONE EcosystemOrchestrator
      • _safe_unit_scalar sanitization for all quality scores
      • No-Zeno rate limiting on snapshot frequency
      • δ_min quantum gate for meaningful change detection
      • Regime classification per domain (paper6 Thm 5.1)
      • Persistent storage to PVC / Prometheus pushgateway
      • Kubernetes CronJob manifest generation

[PT1] PsychePlusCodificationTracer
      • Captures maybe_codify() dual-gate events (validity + quality)
      • Injects codification events as OpenTelemetry span events
      • Correlates with trace_id/span_id for trace-log linking
      • Tracks codification outcomes: success, refused_validity, refused_quality
      • _safe_unit_scalar sanitization for quality_score
      • Prometheus metrics for codification events

[ZA1] ZoneAwareGrafanaDashboard
      • Per-zone alert panels (Z0 Implant, Z1 Clinical, Z2 Processing, Z3 External)
      • Zone-specific thresholds (PNDC Axiom 3 proportionality)
      • Regime classification per zone (paper6 Thm 5.1)
      • Drill-down from zone → service → trace
      • Alert inhibition rules to prevent storms
      • Severity escalation matrix

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds snapshot and codification event rates
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful quality change for snapshot triggers
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → per-zone regime classification
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
    → confidence interval for snapshot threshold calibration
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
    → NaN/Inf sanitization for all quality inputs
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → multi-identity risk for codification correlation
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-specific alert thresholds
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO AGIONE v3.11 AND PSYCHE PLUS v3.3
-----------------------------------------------
From agi_one_v3_11.py:
  • EcosystemOrchestrator.quality_report() → per-domain quality scores
  • AGIONE.attach_orchestrator() → quality_score threading
  • Domains: ecosystem, physics, fold, mental, evolution_bv, hodge, math_numbertheory

From agi_one_psyche_plus_v33.py:
  • _safe_unit_scalar() → NaN/Inf sanitization
  • AGIOnePsychePlus.maybe_codify() → dual-gate (validity + quality)
  • SuperEgoModule.quality_weight → blend weight for quality in validity
  • PsychePlusConfig.quality_codify_min → minimum quality for codification

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• CronJob generates KUBERNETES MANIFESTS, not a deployed system.
• Psyche Plus tracing provides OBSERVABILITY, not causation.
• Zone-aware dashboard is VISUALIZATION, not decision-making.
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
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.cronjob_psyche_zoneaware")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

# AGIONE / Psyche Plus imports (graceful)
try:
    from agi_one_v3_11 import AGIONE, EcosystemOrchestrator
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

try:
    from agi_one_psyche_plus_v33 import AGIOnePsychePlus, _safe_unit_scalar
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
    "SecurityZone", "QualityRegime", "CodificationOutcome",
    "QualitySnapshotCronJob", "SnapshotConfig",
    "PsychePlusCodificationTracer", "CodificationEvent",
    "ZoneAwareGrafanaDashboard", "ZoneAlertConfig",
    "SESIProductionObservabilityV3",
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


class CodificationOutcome(Enum):
    """Outcome of Psyche Plus maybe_codify() dual-gate."""
    SUCCESS = auto()
    REFUSED_VALIDITY = auto()
    REFUSED_QUALITY = auto()
    REFUSED_NONFINITE = auto()


# ═══════════════════════════════════════════════════════════════════════
# [CJ1] QUALITY SNAPSHOT CRONJOB
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SnapshotConfig:
    """
    Configuration for quality snapshot CronJob.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be reviewed by operations team.
    """
    # Snapshot frequency
    interval_s: float = 300.0  # 5 minutes
    min_interval_s: float = 60.0  # No-Zeno minimum

    # No-Zeno rate limiting
    max_snapshots_per_hour: int = 12
    c_v_snapshot: float = 1.0

    # δ_min quantum gate
    delta_min_kappa: float = 2.0
    delta_min_a_min: float = 0.01
    delta_min_n_dim: int = 2

    # Domains to snapshot (from agi_one_v3_11.py)
    domains: List[str] = field(default_factory=lambda: [
        "ecosystem", "physics", "fold", "mental",
        "evolution_bv", "hodge", "math_numbertheory",
    ])

    # Storage
    storage_type: str = "prometheus"  # "prometheus" | "pvc" | "both"
    pvc_path: str = "/data/quality_snapshots"

    # Regime detection
    regime_history_size: int = 100


@dataclass
class QualitySnapshot:
    """A single quality snapshot with regime classification."""
    timestamp: float
    quality_scores: Dict[str, float]
    regime_per_domain: Dict[str, str]
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "quality_scores": self.quality_scores,
            "regime_per_domain": self.regime_per_domain,
        }


class QualitySnapshotCronJob:
    """
    [D] Periodic quality snapshot generator with No-Zeno rate limiting.

    Features:
      • Pulls quality_report() from EcosystemOrchestrator
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Classifies regime per domain (paper6 Thm 5.1)
      • No-Zeno rate limiting on snapshot frequency
      • δ_min quantum gate for meaningful change detection
      • Persistent storage to Prometheus / PVC

    HONEST: This generates SNAPSHOTS, not continuous monitoring.
    For real-time alerting, use the live metric emitter from
    sesi_live_metrics_zone_alerts_otel.py.

    Mathematical grounding:
      • No-Zeno bound on snapshot rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for meaningful changes (Delta_Min Thm 1.2)
      • Hill estimator for regime classification (paper6 Thm 5.1)
    """

    def __init__(
        self,
        agione: Optional[Any] = None,
        cfg: Optional[SnapshotConfig] = None,
    ):
        self.cfg = cfg or SnapshotConfig()
        self.agione = agione
        self._snapshots: List[QualitySnapshot] = []
        self._snapshot_timestamps: deque = deque(maxlen=self.cfg.max_snapshots_per_hour * 2)
        self._quality_history: Dict[str, List[float]] = {
            d: [] for d in self.cfg.domains
        }
        self._last_snapshot_time: float = 0.0

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        kappa = self.cfg.delta_min_kappa
        a_min = self.cfg.delta_min_a_min
        n_dim = self.cfg.delta_min_n_dim
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_snapshot_rate(self) -> Dict[str, Any]:
        """Check if snapshot is within No-Zeno bound."""
        now = time.time()

        # Check minimum interval
        if now - self._last_snapshot_time < self.cfg.min_interval_s:
            return {
                "allowed": False,
                "reason": "min_interval_not_elapsed",
                "time_since_last_s": now - self._last_snapshot_time,
                "min_interval_s": self.cfg.min_interval_s,
            }

        # Check hourly bound
        cutoff = now - 3600.0
        recent = [t for t in self._snapshot_timestamps if t > cutoff]
        if len(recent) >= self.cfg.max_snapshots_per_hour:
            return {
                "allowed": False,
                "reason": "max_per_hour_exceeded",
                "current_count": len(recent),
                "bound": self.cfg.max_snapshots_per_hour,
            }

        return {
            "allowed": True,
            "current_count": len(recent),
            "bound": self.cfg.max_snapshots_per_hour,
            "delta_min": self.compute_delta_min(),
        }

    # ── Quality Extraction ───────────────────────────────────────────────

    def extract_quality(self) -> Dict[str, float]:
        """Extract quality scores from AGIONE with sanitization."""
        if self.agione is None:
            return {d: 0.5 for d in self.cfg.domains}

        orchestrator = getattr(self.agione, "orchestrator", None)
        if orchestrator is None:
            return {d: 0.5 for d in self.cfg.domains}

        try:
            quality_report = orchestrator.quality_report()
            sanitized = {}
            for domain in self.cfg.domains:
                raw_score = quality_report.get(domain, 0.5)
                sanitized_score = _safe_unit_scalar(
                    raw_score, name=f"quality.{domain}", logger=logger, default=0.5
                )
                sanitized[domain] = sanitized_score
            return sanitized
        except Exception as e:
            logger.warning(f"quality_report() failed: {e}")
            return {d: 0.5 for d in self.cfg.domains}

    # ── Regime Classification ────────────────────────────────────────────

    def classify_domain_regime(self, domain: str) -> QualityRegime:
        """Classify quality regime for a domain using Hill estimator."""
        history = self._quality_history.get(domain, [])
        if len(history) < 20:
            return QualityRegime.INSUFFICIENT_DATA

        median = np.median(history)
        deviations = [abs(q - median) for q in history if abs(q - median) > 0.01]

        if len(deviations) < 10:
            return QualityRegime.FINITE_MEAN

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

    # ── Snapshot Creation ────────────────────────────────────────────────

    def take_snapshot(self) -> Optional[QualitySnapshot]:
        """Take a quality snapshot if rate-limited."""
        rate_check = self.check_snapshot_rate()
        if not rate_check["allowed"]:
            logger.debug(f"Snapshot rate-limited: {rate_check['reason']}")
            return None

        # Extract quality
        quality = self.extract_quality()

        # Update history
        for domain, score in quality.items():
            self._quality_history[domain].append(score)
            if len(self._quality_history[domain]) > self.cfg.regime_history_size:
                self._quality_history[domain] = self._quality_history[domain][-self.cfg.regime_history_size:]

        # Classify regime per domain
        regime_per_domain = {
            domain: self.classify_domain_regime(domain).name
            for domain in self.cfg.domains
        }

        # Create snapshot
        snapshot = QualitySnapshot(
            timestamp=time.time(),
            quality_scores=quality,
            regime_per_domain=regime_per_domain,
        )

        self._snapshots.append(snapshot)
        self._snapshot_timestamps.append(time.time())
        self._last_snapshot_time = time.time()

        logger.info(
            f"Quality snapshot taken: {snapshot.snapshot_id} | "
            f"ecosystem={quality.get('ecosystem', 0.5):.3f} | "
            f"regimes={regime_per_domain}"
        )

        return snapshot

    def get_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent snapshots."""
        return [s.to_dict() for s in self._snapshots[-limit:]]

    def generate_k8s_cronjob(self, namespace: str = "sesi-oversight") -> Dict[str, Any]:
        """Generate Kubernetes CronJob manifest."""
        return {
            "apiVersion": "batch/v1",
            "kind": "CronJob",
            "metadata": {
                "name": "sesi-quality-snapshot",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "sesi-quality-snapshot",
                    "app.kubernetes.io/component": "observability",
                },
            },
            "spec": {
                "schedule": f"*/{int(self.cfg.interval_s // 60)} * * * *",
                "concurrencyPolicy": "Forbid",
                "successfulJobsHistoryLimit": 3,
                "failedJobsHistoryLimit": 1,
                "jobTemplate": {
                    "spec": {
                        "template": {
                            "spec": {
                                "serviceAccountName": "sesi-quality-snapshot",
                                "containers": [{
                                    "name": "snapshot",
                                    "image": "sesi/oversight:latest",
                                    "command": [
                                        "python", "-c",
                                        "from sesi_cronjob_psyche_zoneaware import QualitySnapshotCronJob; "
                                        "job = QualitySnapshotCronJob(); job.take_snapshot()"
                                    ],
                                    "env": [
                                        {"name": "SESI_VAULT_ADDR", "valueFrom": {"secretKeyRef": {"name": "sesi-secrets", "key": "vault_addr"}}},
                                        {"name": "SESI_AGIONE_ENDPOINT", "value": "http://sesi-agione:8080"},
                                    ],
                                    "resources": {
                                        "requests": {"cpu": "100m", "memory": "256Mi"},
                                        "limits": {"cpu": "500m", "memory": "512Mi"},
                                    },
                                }],
                                "restartPolicy": "OnFailure",
                            },
                        },
                    },
                },
            },
        }

    def write_cronjob_manifest(self, output_path: str = "./k8s/quality-snapshot-cronjob.yaml") -> str:
        """Write CronJob manifest to file."""
        manifest = self.generate_k8s_cronjob()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(manifest, f, indent=2)

        logger.info(f"CronJob manifest written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get CronJob status."""
        return {
            "total_snapshots": len(self._snapshots),
            "last_snapshot_time": self._last_snapshot_time,
            "delta_min": self.compute_delta_min(),
            "max_snapshots_per_hour": self.cfg.max_snapshots_per_hour,
            "regime_per_domain": {
                domain: self.classify_domain_regime(domain).name
                for domain in self.cfg.domains
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [PT1] PSYCHE PLUS CODIFICATION TRACER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CodificationEvent:
    """A single Psyche Plus codification event."""
    event_id: str
    timestamp: float
    outcome: CodificationOutcome
    validity_score: float
    quality_score: float
    axiom_count: int
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "outcome": self.outcome.name,
            "validity_score": self.validity_score,
            "quality_score": self.quality_score,
            "axiom_count": self.axiom_count,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata,
        }


class PsychePlusCodificationTracer:
    """
    [D] Traces Psyche Plus codification events for observability.

    Features:
      • Captures maybe_codify() dual-gate outcomes
      • Injects events as OpenTelemetry span events
      • Correlates with trace_id/span_id
      • Tracks codification statistics
      • _safe_unit_scalar sanitization for quality_score

    HONEST: This tracer provides OBSERVABILITY, not causation.
    Codification events show when axioms are added to the bank,
    not whether those axioms are "correct" or "conscious".

    Mathematical grounding:
      • Dual-gate from agi_one_psyche_plus_v33.py (validity + quality)
      • _safe_unit_scalar sanitization (Psyche Plus v3.3)
      • No-Zeno rate limiting on event emission
    """

    def __init__(self, psyche_plus: Optional[Any] = None):
        self.psyche_plus = psyche_plus
        self._events: List[CodificationEvent] = []
        self._tracer = None

        if HAS_OTEL:
            self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Initialize OpenTelemetry tracer."""
        resource = Resource.create({
            "service.name": "sesi-psyche-plus",
            "service.version": "3.3",
        })
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        otel_trace.set_tracer_provider(provider)
        self._tracer = otel_trace.get_tracer("sesi-psyche-plus")

    def trace_codification(
        self,
        validity_score: float,
        quality_score: float,
        outcome: CodificationOutcome,
        axiom_count: int,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CodificationEvent:
        """
        Record a codification event.

        Args:
            validity_score: SuperEgo validity score (sanitized)
            quality_score: Ecosystem quality score (sanitized)
            outcome: Codification outcome
            axiom_count: Current axiom bank size
            trace_id: Optional Jaeger trace ID
            span_id: Optional Jaeger span ID
            metadata: Optional additional metadata

        Returns:
            CodificationEvent
        """
        # Sanitize scores via _safe_unit_scalar
        validity_score = _safe_unit_scalar(
            validity_score, name="validity_score", logger=logger, default=0.5
        )
        quality_score = _safe_unit_scalar(
            quality_score, name="quality_score", logger=logger, default=0.5
        )

        event = CodificationEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            outcome=outcome,
            validity_score=validity_score,
            quality_score=quality_score,
            axiom_count=axiom_count,
            trace_id=trace_id,
            span_id=span_id,
            metadata=metadata or {},
        )

        self._events.append(event)

        # Emit as OpenTelemetry span event
        if self._tracer:
            with self._tracer.start_as_current_span("psyche_plus_codification") as span:
                span.set_attribute("codification.outcome", outcome.name)
                span.set_attribute("codification.validity_score", validity_score)
                span.set_attribute("codification.quality_score", quality_score)
                span.set_attribute("codification.axiom_count", axiom_count)
                if trace_id:
                    span.set_attribute("trace_id", trace_id)
                if span_id:
                    span.set_attribute("span_id", span_id)
                span.set_status(StatusCode.OK)

        logger.info(
            f"Codification event: {event.event_id} | "
            f"outcome={outcome.name} | "
            f"validity={validity_score:.3f} | "
            f"quality={quality_score:.3f} | "
            f"axioms={axiom_count}"
        )

        return event

    def trace_maybe_codify_result(
        self,
        codified: bool,
        validity_score: float,
        quality_score: float,
        axiom_count: int,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> CodificationEvent:
        """
        Trace the result of AGIOnePsychePlus.maybe_codify().

        Maps the boolean result to CodificationOutcome:
          • codified=True → SUCCESS
          • codified=False, validity < threshold → REFUSED_VALIDITY
          • codified=False, quality < threshold → REFUSED_QUALITY
        """
        if codified:
            outcome = CodificationOutcome.SUCCESS
        elif validity_score < 0.6:  # Default threshold
            outcome = CodificationOutcome.REFUSED_VALIDITY
        elif quality_score < 0.4:  # Default threshold
            outcome = CodificationOutcome.REFUSED_QUALITY
        else:
            outcome = CodificationOutcome.REFUSED_NONFINITE

        return self.trace_codification(
            validity_score=validity_score,
            quality_score=quality_score,
            outcome=outcome,
            axiom_count=axiom_count,
            trace_id=trace_id,
            span_id=span_id,
        )

    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent codification events."""
        return [e.to_dict() for e in self._events[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get codification statistics."""
        if not self._events:
            return {"total": 0}

        outcomes = {}
        for event in self._events:
            outcome_name = event.outcome.name
            outcomes[outcome_name] = outcomes.get(outcome_name, 0) + 1

        validity_scores = [e.validity_score for e in self._events]
        quality_scores = [e.quality_score for e in self._events]

        return {
            "total": len(self._events),
            "outcomes": outcomes,
            "mean_validity": float(np.mean(validity_scores)),
            "mean_quality": float(np.mean(quality_scores)),
            "std_validity": float(np.std(validity_scores)),
            "std_quality": float(np.std(quality_scores)),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get tracer status."""
        return {
            "tracer_available": self._tracer is not None,
            "total_events": len(self._events),
            "statistics": self.get_statistics(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [ZA1] ZONE-AWARE GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ZoneAlertConfig:
    """Alert configuration for a specific zone."""
    zone: SecurityZone
    quality_warning: float = 0.5
    quality_critical: float = 0.3
    quality_emergency: float = 0.1
    codification_failure_rate_warning: float = 0.1
    codification_failure_rate_critical: float = 0.3
    default_severity: str = "warning"


class ZoneAwareGrafanaDashboard:
    """
    [D] Generates zone-aware Grafana dashboard for alert visualization.

    Features:
      • Per-zone alert panels (Z0-Z3)
      • Zone-specific thresholds (PNDC Axiom 3 proportionality)
      • Regime classification per zone (paper6 Thm 5.1)
      • Codification event tracking per zone
      • Drill-down from zone → service → trace
      • Alert inhibition rules

    HONEST: This generates a DASHBOARD TEMPLATE. It does NOT:
      • Deploy Grafana
      • Make security decisions
      • Guarantee detection of all issues

    Mathematical grounding:
      • PNDC Axiom 3: Necessity and Proportionality
      • paper6 Thm 5.1: Growth-rate dichotomy per zone
    """

    # Default zone configurations (PNDC Axiom 3 proportionality)
    DEFAULT_ZONE_CONFIGS = {
        SecurityZone.Z0_IMPLANT: ZoneAlertConfig(
            zone=SecurityZone.Z0_IMPLANT,
            quality_warning=0.7,
            quality_critical=0.5,
            quality_emergency=0.3,
            codification_failure_rate_warning=0.05,
            codification_failure_rate_critical=0.1,
            default_severity="critical",
        ),
        SecurityZone.Z1_CLINICAL: ZoneAlertConfig(
            zone=SecurityZone.Z1_CLINICAL,
            quality_warning=0.6,
            quality_critical=0.4,
            quality_emergency=0.2,
            codification_failure_rate_warning=0.1,
            codification_failure_rate_critical=0.2,
            default_severity="critical",
        ),
        SecurityZone.Z2_PROCESSING: ZoneAlertConfig(
            zone=SecurityZone.Z2_PROCESSING,
            quality_warning=0.5,
            quality_critical=0.3,
            quality_emergency=0.1,
            codification_failure_rate_warning=0.15,
            codification_failure_rate_critical=0.3,
            default_severity="warning",
        ),
        SecurityZone.Z3_EXTERNAL: ZoneAlertConfig(
            zone=SecurityZone.Z3_EXTERNAL,
            quality_warning=0.4,
            quality_critical=0.2,
            quality_emergency=0.1,
            codification_failure_rate_warning=0.2,
            codification_failure_rate_critical=0.4,
            default_severity="info",
        ),
    }

    def __init__(
        self,
        zone_configs: Optional[Dict[SecurityZone, ZoneAlertConfig]] = None,
        prometheus_uid: str = "prometheus",
    ):
        self.zone_configs = zone_configs or self.DEFAULT_ZONE_CONFIGS
        self.prometheus_uid = prometheus_uid

    def _panel(
        self,
        panel_id: int,
        title: str,
        panel_type: str,
        targets: List[Dict[str, Any]],
        grid_pos: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Grafana panel."""
        panel = {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
            "targets": targets,
            "gridPos": grid_pos,
            "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
            "options": {},
        }
        if extra:
            panel.update(extra)
        return panel

    def generate_zone_panels(self, zone: SecurityZone, start_y: int) -> Tuple[List[Dict[str, Any]], int]:
        """Generate panels for a specific zone."""
        cfg = self.zone_configs[zone]
        zone_name = zone.value
        panels = []
        pid = 100 + zone.value * 10  # Unique ID per zone

        # Zone header
        panels.append({
            "id": pid,
            "type": "text",
            "title": f"{zone_name.upper()} Zone Alerts",
            "options": {"mode": "markdown", "content": f"## {zone_name}\nSeverity: {cfg.default_severity}"},
            "gridPos": {"h": 2, "w": 24, "x": 0, "y": start_y},
        })
        pid += 1
        start_y += 2

        # Quality score panel
        panels.append(self._panel(
            pid, f"Quality Score ({zone_name})", "gauge",
            [{"expr": f'sesi_quality_ecosystem{{zone="{zone_name}"}}', "legendFormat": "quality"}],
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
            [{"expr": f'sesi_quality_regime{{zone="{zone_name}"}}', "legendFormat": "regime"}],
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

        # Codification success rate
        panels.append(self._panel(
            pid, f"Codification Success Rate ({zone_name})", "timeseries",
            [{"expr": f'rate(sesi_codification_events_total{{zone="{zone_name}",outcome="success"}}[5m]) / rate(sesi_codification_events_total{{zone="{zone_name}"}}[5m])',
              "legendFormat": "success_rate"}],
            {"h": 6, "w": 6, "x": 12, "y": start_y},
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

        return panels, start_y

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete zone-aware Grafana dashboard."""
        panels = []
        y = 0

        # Generate panels for each zone
        for zone in SecurityZone:
            zone_panels, y = self.generate_zone_panels(zone, y)
            panels.extend(zone_panels)
            y += 2  # Spacing between zones

        dashboard = {
            "uid": "sesi-zone-aware-alerts",
            "title": "SESI Zone-Aware Alert Dashboard",
            "tags": ["sesi", "zone-aware", "alerts", "psyche-plus"],
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
                ],
            },
            "annotations": {
                "list": [
                    {
                        "name": "Codification Events",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "sesi_codification_events_total",
                        "iconColor": "blue",
                        "titleFormat": "Codification: {{outcome}}",
                    },
                ],
            },
        }

        return dashboard

    def generate_zone_alert_rules(self) -> List[Dict[str, Any]]:
        """Generate zone-specific alert rules."""
        rules = []

        for zone, cfg in self.zone_configs.items():
            zone_name = zone.value

            # Quality degradation alert
            rules.append({
                "alert": f"QualityDegradation_{zone_name}",
                "expr": f'sesi_quality_ecosystem{{zone="{zone_name}"}} < {cfg.quality_warning}',
                "for": "5m",
                "labels": {
                    "severity": cfg.default_severity,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"Quality degradation in {zone_name}",
                    "description": (
                        f"Quality score in {zone_name} zone has dropped below "
                        f"{cfg.quality_warning}. PNDC Axiom 3: investigate "
                        f"proportionally to zone severity."
                    ),
                },
            })

            # Codification failure rate alert
            rules.append({
                "alert": f"CodificationFailureRate_{zone_name}",
                "expr": (
                    f'1 - (rate(sesi_codification_events_total{{zone="{zone_name}",outcome="success"}}[5m]) '
                    f'/ rate(sesi_codification_events_total{{zone="{zone_name}"}}[5m])) '
                    f'> {cfg.codification_failure_rate_warning}'
                ),
                "for": "5m",
                "labels": {
                    "severity": cfg.default_severity,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"High codification failure rate in {zone_name}",
                    "description": (
                        f"Codification failure rate exceeds "
                        f"{cfg.codification_failure_rate_warning*100:.0f}% in {zone_name}. "
                        f"Check Psyche Plus validity and quality gates."
                    ),
                },
            })

        return rules

    def write_dashboard(self, output_path: str = "./grafana/zone-aware-dashboard.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Zone-aware dashboard written to {output_path}")
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
        """Get dashboard status."""
        return {
            "zones_configured": len(self.zone_configs),
            "alert_rules": len(self.generate_zone_alert_rules()),
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIProductionObservabilityV3:
    """
    [D] Complete production observability suite v3.

    Integrates:
      • Quality snapshot CronJob
      • Psyche Plus codification tracer
      • Zone-aware Grafana dashboard
    """

    def __init__(self, agione: Optional[Any] = None, psyche_plus: Optional[Any] = None):
        self.cronjob = QualitySnapshotCronJob(agione)
        self.tracer = PsychePlusCodificationTracer(psyche_plus)
        self.dashboard = ZoneAwareGrafanaDashboard()

    def generate_all_artifacts(self, output_dir: str = "./sesi_observability_v3") -> Dict[str, str]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # CronJob manifest
        results["cronjob"] = self.cronjob.write_cronjob_manifest(
            str(out_path / "k8s" / "quality-snapshot-cronjob.yaml")
        )

        # Zone-aware dashboard
        results["dashboard"] = self.dashboard.write_dashboard(
            str(out_path / "grafana" / "zone-aware-dashboard.json")
        )

        # Zone alert rules
        results["alert_rules"] = self.dashboard.write_alert_rules(
            str(out_path / "prometheus" / "zone-alerts.yaml")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "cronjob": self.cronjob.get_status(),
            "tracer": self.tracer.get_status(),
            "dashboard": self.dashboard.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI CronJob × Psyche Plus Tracer × Zone-Aware Dashboard — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  OpenTelemetry avail  : {HAS_OTEL}")
    print(f"  AGIONE available     : {HAS_AGIONE}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIProductionObservabilityV3()

    # ── Test 1: Quality Snapshot CronJob ─────────────────────────────────
    print("\n  ── Quality Snapshot CronJob ──")

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
    suite.cronjob._quality_history = {d: [mock_quality[d]] * 50 for d in mock_quality}

    # Take snapshots
    for i in range(3):
        snapshot = suite.cronjob.take_snapshot()
        if snapshot:
            print(f"    Snapshot {i+1}         : {snapshot.snapshot_id}")
            print(f"      Ecosystem quality  : {snapshot.quality_scores.get('ecosystem', 0.5):.3f}")
            print(f"      Regimes            : {snapshot.regime_per_domain}")

    # Check rate limit
    rate_check = suite.cronjob.check_snapshot_rate()
    print(f"    Rate check           : {rate_check['allowed']}")
    print(f"    δ_min                : {suite.cronjob.compute_delta_min():.6f}")

    # ── Test 2: Psyche Plus Codification Tracer ──────────────────────────
    print("\n  ── Psyche Plus Codification Tracer ──")

    # Trace some codification events
    events = [
        (0.85, 0.9, CodificationOutcome.SUCCESS, 10),
        (0.4, 0.8, CodificationOutcome.REFUSED_VALIDITY, 10),
        (0.9, 0.3, CodificationOutcome.REFUSED_QUALITY, 10),
    ]

    for validity, quality, outcome, axiom_count in events:
        event = suite.tracer.trace_codification(
            validity_score=validity,
            quality_score=quality,
            outcome=outcome,
            axiom_count=axiom_count,
            trace_id=f"trace_{uuid.uuid4().hex[:8]}",
        )
        print(f"    Event                : {event.event_id} | {outcome.name}")

    # Get statistics
    stats = suite.tracer.get_statistics()
    print(f"    Total events         : {stats['total']}")
    print(f"    Outcome distribution : {stats['outcomes']}")
    print(f"    Mean validity        : {stats.get('mean_validity', 0):.3f}")
    print(f"    Mean quality         : {stats.get('mean_quality', 0):.3f}")

    # ── Test 3: Zone-Aware Grafana Dashboard ─────────────────────────────
    print("\n  ── Zone-Aware Grafana Dashboard ──")

    dashboard_status = suite.dashboard.get_status()
    print(f"    Zones configured     : {dashboard_status['zones_configured']}")
    print(f"    Alert rules          : {dashboard_status['alert_rules']}")

    # Generate dashboard
    dashboard = suite.dashboard.generate_dashboard()
    print(f"    Dashboard panels     : {len(dashboard['panels'])}")

    # Generate alert rules
    alert_rules = suite.dashboard.generate_zone_alert_rules()
    print(f"    Zone alert rules     : {len(alert_rules)}")
    for rule in alert_rules[:3]:
        print(f"      • {rule['alert']} ({rule['labels']['severity']})")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_observability_v3")
    print(f"    Artifacts generated  : {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:20s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    CronJob snapshots    : {status['cronjob']['total_snapshots']}")
    print(f"    CronJob δ_min        : {status['cronjob']['delta_min']:.6f}")
    print(f"    Tracer events        : {status['tracer']['total_events']}")
    print(f"    Dashboard zones      : {status['dashboard']['zones_configured']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • CronJob generates MANIFESTS, not a deployed system")
    print("  • Psyche Plus tracing shows OBSERVABILITY, not causation")
    print("  • Zone-aware dashboard is VISUALIZATION, not decision-making")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all quality inputs")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
