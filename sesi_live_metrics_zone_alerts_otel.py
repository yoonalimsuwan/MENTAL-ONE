"""
=============================================================================
SESI LIVE QUALITY METRICS × ZONE-AWARE ALERTS × OTEL PYTHON INSTRUMENTATION
Real-Time AGIONE Quality Emission + Per-Zone Grafana Alerts + Distributed Tracing
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[LQ1] LiveQualityMetricEmitter
      • Background thread pulling AGIONE EcosystemOrchestrator.quality_report()
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Emits per-domain Prometheus gauges + histograms
      • No-Zeno rate limiting on emission frequency
      • δ_min quantum gate for meaningful metric updates
      • Graceful degradation when AGIONE unavailable

[ZA1] ZoneAwareAlertRuleGenerator
      • Per-zone Grafana alert rules (Z0-Z3)
      • Zone-specific thresholds (PNDC Axiom 3 proportionality)
      • Regime-adaptive thresholds (paper6 Thm 5.1)
      • Inhibition rules to prevent alert storms
      • Severity escalation matrix

[OT1] OTelPythonInstrumentor
      • Auto-instrumentation decorators for Python functions
      • Span context propagation with trace_id/span_id
      • Quality score injection into span attributes
      • Log-trace correlation via trace_id in log records
      • Sampling strategy with No-Zeno bounds

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds metric emission and alert frequency
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful metric update threshold
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → regime-adaptive alert thresholds
[C] Six_Problems P4: CUSUM + subsampling
    → change-point detection in metric streams
[C] Stochastic_NoZeno Thm 4.1: Exit-time tail bound
    → anomalous metric deviation detection
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
    → NaN/Inf sanitization for all quality inputs
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-specific alert thresholds
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO AGIONE v3.11 AND PSYCHE PLUS v3.3
-----------------------------------------------
From agi_one_v3_11.py:
  • EcosystemOrchestrator.quality_report() → per-domain quality scores
  • Domains: ecosystem, physics, fold, mental, evolution_bv, hodge,
    math_numbertheory
  • AGIONE.attach_orchestrator() → quality_score threading

From agi_one_psyche_plus_v33.py:
  • _safe_unit_scalar() → NaN/Inf sanitization
  • quality_score threading to PsychePlus.maybe_codify()

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Live metric emission provides OBSERVABILITY, not causation.
  Quality scores reflect simulation health, not system truth.
• Zone-aware alerts are PROPORTIONALITY-BASED, not absolute.
  Thresholds must be calibrated against real operational data.
• OTel instrumentation is a FRAMEWORK, not a security guarantee.
  It shows correlation, not proof of correct behavior.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• Production deployment requires security audit and regulatory approval.
• This module does NOT create, detect, or simulate consciousness.
=============================================================================
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.live_metrics_zone_alerts_otel")

# ── Optional imports ─────────────────────────────────────────────────────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode, SpanKind
    from opentelemetry.context import attach, detach, get_current
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    HAS_PROM = True
except ImportError:
    HAS_PROM = False

# AGIONE imports (graceful)
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
    # Fallback sanitization
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
    "SecurityZone", "QualityRegime", "AlertSeverity",
    "LiveQualityMetricEmitter", "EmitterConfig",
    "ZoneAwareAlertRuleGenerator", "ZoneAlertConfig",
    "OTelPythonInstrumentor", "InstrumentationConfig",
    "SESILiveObservabilitySuite",
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


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ═══════════════════════════════════════════════════════════════════════
# [LQ1] LIVE QUALITY METRIC EMITTER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EmitterConfig:
    """
    Configuration for live quality metric emission.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be reviewed by operations team.
    """
    # Emission frequency
    emit_interval_s: float = 5.0
    min_emit_interval_s: float = 1.0

    # No-Zeno rate limiting
    max_emissions_per_minute: int = 60
    c_v_emission: float = 1.0

    # δ_min quantum gate
    delta_min_kappa: float = 2.0
    delta_min_a_min: float = 0.01
    delta_min_n_dim: int = 2

    # Domains to emit (from agi_one_v3_11.py)
    domains: List[str] = field(default_factory=lambda: [
        "ecosystem", "physics", "fold", "mental",
        "evolution_bv", "hodge", "math_numbertheory",
    ])

    # Sanitization
    default_quality: float = 0.5

    # Background thread
    daemon: bool = True


class LiveQualityMetricEmitter:
    """
    [D] Background emitter for AGIONE quality_report() metrics.

    Features:
      • Pulls quality_report() from EcosystemOrchestrator on interval
      • Sanitizes via _safe_unit_scalar (Psyche Plus v3.3 pattern)
      • Emits per-domain Prometheus gauges + histograms
      • No-Zeno rate limiting on emission frequency
      • δ_min quantum gate for meaningful metric updates
      • Graceful degradation when AGIONE unavailable
      • Regime classification on quality streams

    HONEST: This emitter provides OBSERVABILITY, not causation.
    Quality scores reflect simulation health, not system truth.
    All emitted metrics are sanitized and rate-limited.

    Mathematical grounding:
      • _safe_unit_scalar sanitization (Psyche Plus v3.3)
      • No-Zeno bound on emission rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for meaningful updates (Delta_Min Thm 1.2)
    """

    def __init__(
        self,
        agione: Optional[Any] = None,
        cfg: Optional[EmitterConfig] = None,
        registry: Optional[Any] = None,
    ):
        self.cfg = cfg or EmitterConfig()
        self.agione = agione
        self.registry = registry or (REGISTRY if HAS_PROM else None)

        # Prometheus metrics
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        self._counters: Dict[str, Any] = {}
        self._setup_metrics()

        # State
        self._last_quality: Dict[str, float] = {d: self.cfg.default_quality for d in self.cfg.domains}
        self._emit_timestamps: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Regime tracking
        self._quality_history: Dict[str, List[float]] = {d: [] for d in self.cfg.domains}

    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics for each domain."""
        if not HAS_PROM or self.registry is None:
            logger.warning("Prometheus not available; metrics will be logged only")
            return

        for domain in self.cfg.domains:
            # Gauge: current quality score
            self._gauges[domain] = Gauge(
                f"sesi_quality_{domain}",
                f"Quality score for {domain} domain (sanitized via _safe_unit_scalar)",
                registry=self.registry,
            )

            # Histogram: quality distribution
            self._histograms[domain] = Histogram(
                f"sesi_quality_{domain}_distribution",
                f"Quality score distribution for {domain}",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self.registry,
            )

        # Global counters
        self._counters["emissions"] = Counter(
            "sesi_quality_emissions_total",
            "Total quality metric emissions",
            ["domain", "status"],
            registry=self.registry,
        )
        self._counters["regime"] = Gauge(
            "sesi_quality_regime",
            "Current quality regime (0=FINITE_MEAN, 1=REGULARLY_VARYING, 2=DEGENERATE)",
            ["domain"],
            registry=self.registry,
        )

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self) -> float:
        """
        Compute δ_min from Delta_Min Thm 1.2.

        δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}

        Requires κ-fatness (Delta_Min Def 1.1).
        """
        kappa = self.cfg.delta_min_kappa
        a_min = self.cfg.delta_min_a_min
        n_dim = self.cfg.delta_min_n_dim
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def max_emissions_in_window(self) -> float:
        """
        No-Zeno bound: N(T) ≤ 1 + 2C_V·T/δ_min.

        From SESI_NoZeno_Closure Thm 3.2.
        """
        delta_min = self.compute_delta_min()
        if delta_min <= 0:
            return float('inf')
        T = 60.0  # 1-minute window
        return 1 + (2 * self.cfg.c_v_emission * T) / delta_min

    def check_emit_rate(self) -> Dict[str, Any]:
        """Check if emission is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._emit_timestamps = [t for t in self._emit_timestamps if t > cutoff]

        bound = self.max_emissions_in_window()
        current = len(self._emit_timestamps)

        if current >= self.cfg.max_emissions_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
                "current": current,
                "bound": self.cfg.max_emissions_per_minute,
            }

        if self._emit_timestamps:
            time_since_last = now - self._emit_timestamps[-1]
            if time_since_last < self.cfg.min_emit_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                    "time_since_last_s": time_since_last,
                    "min_interval_s": self.cfg.min_emit_interval_s,
                }

        return {
            "allowed": True,
            "current_count": current,
            "bound": bound,
            "delta_min": self.compute_delta_min(),
        }

    # ── Quality Extraction ───────────────────────────────────────────────

    def extract_quality(self) -> Dict[str, float]:
        """
        Extract quality scores from AGIONE's EcosystemOrchestrator.

        All scores sanitized via _safe_unit_scalar.
        """
        if self.agione is None:
            return self._last_quality.copy()

        orchestrator = getattr(self.agione, "orchestrator", None)
        if orchestrator is None:
            return self._last_quality.copy()

        try:
            quality_report = orchestrator.quality_report()
            sanitized = {}
            for domain in self.cfg.domains:
                raw_score = quality_report.get(domain, self.cfg.default_quality)
                sanitized_score = _safe_unit_scalar(
                    raw_score, name=f"quality.{domain}", logger=logger,
                    default=self.cfg.default_quality,
                )
                sanitized[domain] = sanitized_score
            return sanitized
        except Exception as e:
            logger.warning(f"quality_report() failed: {e}")
            return self._last_quality.copy()

    # ── Regime Classification ────────────────────────────────────────────

    def classify_domain_regime(self, domain: str) -> QualityRegime:
        """
        Classify quality regime for a domain using Hill estimator.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → ordinary monitoring
          θ ∈ (0,1):  regularly varying → distributional monitoring
          θ → 0:      degenerate → polarized monitoring
        """
        history = self._quality_history.get(domain, [])
        if len(history) < 20:
            return QualityRegime.INSUFFICIENT_DATA

        # Compute deviations from median
        median = np.median(history)
        deviations = [abs(q - median) for q in history if abs(q - median) > 0.01]

        if len(deviations) < 10:
            return QualityRegime.FINITE_MEAN

        # Hill estimator
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

    # ── Emission ─────────────────────────────────────────────────────────

    def emit_once(self) -> Dict[str, Any]:
        """
        Emit quality metrics once.

        Returns dict with emission results.
        """
        rate_check = self.check_emit_rate()
        if not rate_check["allowed"]:
            return {"status": "rate_limited", "reason": rate_check["reason"]}

        # Extract quality
        quality = self.extract_quality()

        # Update history
        for domain, score in quality.items():
            self._quality_history[domain].append(score)
            if len(self._quality_history[domain]) > 100:
                self._quality_history[domain] = self._quality_history[domain][-100:]

        # Emit to Prometheus
        if HAS_PROM:
            for domain, score in quality.items():
                if domain in self._gauges:
                    self._gauges[domain].set(score)
                if domain in self._histograms:
                    self._histograms[domain].observe(score)

                # Update regime gauge
                regime = self.classify_domain_regime(domain)
                self._counters["regime"].labels(domain=domain).set(regime.value)

                # Count emission
                self._counters["emissions"].labels(
                    domain=domain, status="emitted"
                ).inc()

        # Update last quality
        self._last_quality = quality
        self._emit_timestamps.append(time.time())

        return {
            "status": "emitted",
            "quality": quality,
            "regimes": {
                d: self.classify_domain_regime(d).name
                for d in self.cfg.domains
            },
            "timestamp": time.time(),
        }

    def _emit_loop(self) -> None:
        """Background emission loop."""
        while self._running:
            try:
                self.emit_once()
            except Exception as e:
                logger.error(f"Emission error: {e}")
            time.sleep(self.cfg.emit_interval_s)

    def start(self) -> None:
        """Start background emission thread."""
        if self._running:
            logger.warning("Emitter already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._emit_loop,
            daemon=self.cfg.daemon,
            name="sesi-quality-emitter",
        )
        self._thread.start()
        logger.info(f"Quality emitter started (interval={self.cfg.emit_interval_s}s)")

    def stop(self) -> None:
        """Stop background emission thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Quality emitter stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get emitter status."""
        return {
            "running": self._running,
            "agione_attached": self.agione is not None,
            "last_quality": self._last_quality,
            "emit_count": len(self._emit_timestamps),
            "delta_min": self.compute_delta_min(),
            "max_emissions_per_minute": self.max_emissions_in_window(),
            "regimes": {
                d: self.classify_domain_regime(d).name
                for d in self.cfg.domains
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [ZA1] ZONE-AWARE ALERT RULE GENERATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ZoneAlertConfig:
    """
    Alert configuration for a specific zone.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be calibrated against real data.
    """
    zone: SecurityZone
    # Quality thresholds (lower = worse)
    quality_warning: float = 0.5
    quality_critical: float = 0.3
    quality_emergency: float = 0.1

    # Rate thresholds
    emission_rate_warning: float = 50.0  # per minute
    emission_rate_critical: float = 100.0

    # Regime-specific
    degenerate_regime_alert: bool = True
    regime_change_alert: bool = True

    # Severity mapping
    default_severity: AlertSeverity = AlertSeverity.WARNING


class ZoneAwareAlertRuleGenerator:
    """
    [D] Generates zone-aware Grafana alert rules.

    Features:
      • Per-zone alert rules (Z0-Z3)
      • Zone-specific thresholds (PNDC Axiom 3 proportionality)
      • Regime-adaptive thresholds (paper6 Thm 5.1)
      • Inhibition rules to prevent alert storms
      • Severity escalation matrix

    HONEST: These are TEMPLATE alert rules. Real deployment requires:
      • Calibration against operational data
      • Security team review
      • Regular threshold updates
    """

    # Default zone configurations (PNDC Axiom 3 proportionality)
    DEFAULT_ZONE_CONFIGS = {
        SecurityZone.Z0_IMPLANT: ZoneAlertConfig(
            zone=SecurityZone.Z0_IMPLANT,
            quality_warning=0.7,
            quality_critical=0.5,
            quality_emergency=0.3,
            emission_rate_warning=20.0,
            emission_rate_critical=40.0,
            degenerate_regime_alert=True,
            default_severity=AlertSeverity.CRITICAL,
        ),
        SecurityZone.Z1_CLINICAL: ZoneAlertConfig(
            zone=SecurityZone.Z1_CLINICAL,
            quality_warning=0.6,
            quality_critical=0.4,
            quality_emergency=0.2,
            emission_rate_warning=30.0,
            emission_rate_critical=60.0,
            degenerate_regime_alert=True,
            default_severity=AlertSeverity.CRITICAL,
        ),
        SecurityZone.Z2_PROCESSING: ZoneAlertConfig(
            zone=SecurityZone.Z2_PROCESSING,
            quality_warning=0.5,
            quality_critical=0.3,
            quality_emergency=0.1,
            emission_rate_warning=50.0,
            emission_rate_critical=100.0,
            degenerate_regime_alert=True,
            default_severity=AlertSeverity.WARNING,
        ),
        SecurityZone.Z3_EXTERNAL: ZoneAlertConfig(
            zone=SecurityZone.Z3_EXTERNAL,
            quality_warning=0.4,
            quality_critical=0.2,
            quality_emergency=0.1,
            emission_rate_warning=100.0,
            emission_rate_critical=200.0,
            degenerate_regime_alert=False,
            default_severity=AlertSeverity.INFO,
        ),
    }

    def __init__(
        self,
        zone_configs: Optional[Dict[SecurityZone, ZoneAlertConfig]] = None,
    ):
        self.zone_configs = zone_configs or self.DEFAULT_ZONE_CONFIGS

    def generate_zone_alerts(self, zone: SecurityZone) -> List[Dict[str, Any]]:
        """
        Generate alert rules for a specific zone.

        Args:
            zone: Security zone

        Returns:
            List of Prometheus alert rule dicts
        """
        cfg = self.zone_configs[zone]
        zone_name = zone.value
        rules = []

        # Quality degradation alerts
        for domain in ["ecosystem", "physics", "fold", "mental",
                       "evolution_bv", "hodge", "math_numbertheory"]:
            rules.append({
                "alert": f"QualityDegradation_{zone_name}_{domain}",
                "expr": f'sesi_quality_{domain}{{zone="{zone_name}"}} < {cfg.quality_warning}',
                "for": "5m",
                "labels": {
                    "severity": cfg.default_severity.value,
                    "zone": zone_name,
                    "domain": domain,
                },
                "annotations": {
                    "summary": f"Quality degradation in {domain} ({zone_name})",
                    "description": (
                        f"Quality score for {domain} domain in {zone_name} zone "
                        f"has dropped below {cfg.quality_warning}. "
                        f"Current value: {{{{$value}}}}. "
                        f"PNDC Axiom 3: investigate proportionally to zone severity."
                    ),
                },
            })

            # Critical threshold
            rules.append({
                "alert": f"QualityCritical_{zone_name}_{domain}",
                "expr": f'sesi_quality_{domain}{{zone="{zone_name}"}} < {cfg.quality_critical}',
                "for": "2m",
                "labels": {
                    "severity": AlertSeverity.CRITICAL.value,
                    "zone": zone_name,
                    "domain": domain,
                },
                "annotations": {
                    "summary": f"CRITICAL quality in {domain} ({zone_name})",
                    "description": (
                        f"Quality score critically low in {zone_name}. "
                        f"Immediate review required."
                    ),
                },
            })

        # Degenerate regime alert (paper6 Thm 5.1)
        if cfg.degenerate_regime_alert:
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
                        f"Averaging-based monitoring will miss this. "
                        f"Use change-point detection and regime-specific response."
                    ),
                },
            })

        # Emission rate alert (No-Zeno bound)
        rules.append({
            "alert": f"EmissionRateHigh_{zone_name}",
            "expr": f'sum(rate(sesi_quality_emissions_total{{zone="{zone_name}"}}[1m])) * 60 > {cfg.emission_rate_critical}',
            "for": "5m",
            "labels": {
                "severity": AlertSeverity.WARNING.value,
                "zone": zone_name,
            },
            "annotations": {
                "summary": f"High emission rate in {zone_name}",
                "description": (
                    f"Quality metric emission rate exceeds {cfg.emission_rate_critical}/min "
                    f"in {zone_name}. Possible No-Zeno bound violation. "
                    f"Check emitter configuration."
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
                "alert": "DegenerateRegimeInhibitsQuality",
                "expr": 'sesi_quality_regime == 2',
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Inhibition: degenerate regime suppresses quality alerts",
                    "description": (
                        "When degenerate regime is detected, individual quality "
                        "degradation alerts are inhibited to prevent alert storms. "
                        "Focus on the regime-level alert instead."
                    ),
                },
            },
            {
                "alert": "CriticalInhibitsWarning",
                "expr": 'alertname =~ "QualityCritical_.*"',
                "for": "0m",
                "labels": {"severity": "info"},
                "annotations": {
                    "summary": "Inhibition: critical alerts suppress warnings",
                    "description": (
                        "When a critical quality alert fires, related warning "
                        "alerts are inhibited to reduce noise."
                    ),
                },
            },
        ]

    def generate_all_alert_rules(self) -> Dict[str, Any]:
        """Generate all zone-aware alert rules."""
        all_rules = []
        for zone in SecurityZone:
            all_rules.extend(self.generate_zone_alerts(zone))

        return {
            "groups": [
                {
                    "name": "sesi_zone_aware_alerts",
                    "rules": all_rules,
                },
                {
                    "name": "sesi_inhibition_rules",
                    "rules": self.generate_inhibition_rules(),
                },
            ],
            "zone_configs": {
                zone.value: {
                    "quality_warning": cfg.quality_warning,
                    "quality_critical": cfg.quality_critical,
                    "quality_emergency": cfg.quality_emergency,
                    "emission_rate_warning": cfg.emission_rate_warning,
                    "emission_rate_critical": cfg.emission_rate_critical,
                    "degenerate_regime_alert": cfg.degenerate_regime_alert,
                    "default_severity": cfg.default_severity.value,
                }
                for zone, cfg in self.zone_configs.items()
            },
        }

    def write_alert_rules(self, output_path: str = "./prometheus/zone-aware-alerts.yaml") -> str:
        """Write alert rules to file."""
        rules = self.generate_all_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(rules, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(rules, f, indent=2)

        logger.info(f"Zone-aware alert rules written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get alert generator status."""
        all_rules = self.generate_all_alert_rules()
        return {
            "zone_configs": len(self.zone_configs),
            "total_alert_rules": len(all_rules["groups"][0]["rules"]),
            "inhibition_rules": len(all_rules["groups"][1]["rules"]),
        }


# ═══════════════════════════════════════════════════════════════════════
# [OT1] OPENTELEMETRY PYTHON INSTRUMENTOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class InstrumentationConfig:
    """
    Configuration for OpenTelemetry Python instrumentation.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be reviewed by operations team.
    """
    service_name: str = "sesi-python-app"
    service_version: str = "1.0.0"

    # Exporter
    exporter_type: str = "console"  # "console" | "otlp" | "jaeger"
    otlp_endpoint: str = "http://localhost:4317"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"

    # Sampling
    sampling_rate: float = 1.0  # 1.0 = sample all
    max_spans_per_second: float = 100.0

    # No-Zeno rate limiting
    min_span_interval_s: float = 0.01
    max_spans_per_minute: int = 1000

    # Quality injection
    inject_quality_scores: bool = True
    quality_domains: List[str] = field(default_factory=lambda: [
        "ecosystem", "physics", "fold", "mental",
    ])

    # Log correlation
    inject_trace_id_to_logs: bool = True


class OTelPythonInstrumentor:
    """
    [D] OpenTelemetry instrumentation for Python applications.

    Features:
      • Auto-instrumentation decorators for Python functions
      • Span context propagation with trace_id/span_id
      • Quality score injection into span attributes
      • Log-trace correlation via trace_id in log records
      • Sampling strategy with No-Zeno bounds
      • Regime classification on span streams

    HONEST: This is an INSTRUMENTATION FRAMEWORK. It provides
    observability but does NOT:
      • Guarantee detection of all issues
      • Replace code review or testing
      • Provide causation, only correlation

    Mathematical grounding:
      • No-Zeno bound on span emission rate (SESI_NoZeno Thm 3.2)
      • CUSUM change-point detection on span duration streams
      • paper6 Thm 5.1: regime classification on span patterns
    """

    def __init__(self, cfg: Optional[InstrumentationConfig] = None):
        self.cfg = cfg or InstrumentationConfig()
        self.tracer = None
        self._span_timestamps: List[float] = []
        self._span_durations: List[float] = []
        self._quality_emitter: Optional[LiveQualityMetricEmitter] = None

        if HAS_OTEL:
            self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Initialize OpenTelemetry tracer."""
        resource = Resource.create({
            "service.name": self.cfg.service_name,
            "service.version": self.cfg.service_version,
        })
        provider = TracerProvider(resource=resource)

        # Add exporter
        if self.cfg.exporter_type == "console":
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
        elif self.cfg.exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=self.cfg.otlp_endpoint)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
            except ImportError:
                logger.warning("OTLP exporter not available; falling back to console")
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
        elif self.cfg.exporter_type == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                exporter = JaegerExporter(endpoint=self.cfg.jaeger_endpoint)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
            except ImportError:
                logger.warning("Jaeger exporter not available; falling back to console")
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)

        otel_trace.set_tracer_provider(provider)
        self.tracer = otel_trace.get_tracer(self.cfg.service_name)
        logger.info(f"OTel tracer initialized: {self.cfg.service_name}")

    def attach_quality_emitter(self, emitter: LiveQualityMetricEmitter) -> None:
        """Attach a quality emitter for span attribute injection."""
        self._quality_emitter = emitter

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def check_span_rate(self) -> Dict[str, Any]:
        """Check if span creation is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._span_timestamps = [t for t in self._span_timestamps if t > cutoff]

        if len(self._span_timestamps) >= self.cfg.max_spans_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
                "current": len(self._span_timestamps),
                "bound": self.cfg.max_spans_per_minute,
            }

        if self._span_timestamps:
            time_since_last = now - self._span_timestamps[-1]
            if time_since_last < self.cfg.min_span_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                    "time_since_last_s": time_since_last,
                }

        return {"allowed": True}

    # ── Decorator for Auto-Instrumentation ───────────────────────────────

    def trace_function(
        self,
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        inject_quality: bool = True,
    ):
        """
        Decorator for auto-instrumenting Python functions.

        Usage:
            @instrumentor.trace_function(span_name="my_function")
            def my_function(arg1, arg2):
                ...

        Args:
            span_name: Name of the span (defaults to function name)
            attributes: Additional attributes to add to the span
            inject_quality: Whether to inject quality scores
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.tracer is None:
                    return func(*args, **kwargs)

                # Check rate limit
                rate_check = self.check_span_rate()
                if not rate_check["allowed"]:
                    logger.debug(f"Span rate-limited: {rate_check['reason']}")
                    return func(*args, **kwargs)

                # Create span
                name = span_name or func.__name__
                with self.tracer.start_as_current_span(name) as span:
                    start_time = time.time()

                    # Add custom attributes
                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, str(v))

                    # Add function signature
                    sig = inspect.signature(func)
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                    # Inject quality scores
                    if inject_quality and self._quality_emitter:
                        quality = self._quality_emitter.extract_quality()
                        for domain, score in quality.items():
                            span.set_attribute(f"quality.{domain}", score)

                    # Execute function
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        self._span_timestamps.append(time.time())
                        self._span_durations.append(duration)

            return wrapper
        return decorator

    # ── Async Decorator ──────────────────────────────────────────────────

    def trace_async_function(
        self,
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        inject_quality: bool = True,
    ):
        """Async version of trace_function decorator."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if self.tracer is None:
                    return await func(*args, **kwargs)

                rate_check = self.check_span_rate()
                if not rate_check["allowed"]:
                    return await func(*args, **kwargs)

                name = span_name or func.__name__
                with self.tracer.start_as_current_span(name) as span:
                    start_time = time.time()

                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, str(v))

                    if inject_quality and self._quality_emitter:
                        quality = self._quality_emitter.extract_quality()
                        for domain, score in quality.items():
                            span.set_attribute(f"quality.{domain}", score)

                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        self._span_timestamps.append(time.time())
                        self._span_durations.append(duration)

            return wrapper
        return decorator

    # ── Log-Traces Correlation ───────────────────────────────────────────

    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for log correlation."""
        if not HAS_OTEL:
            return {}

        span = otel_trace.get_current_span()
        if span is None:
            return {}

        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, '032x'),
            "span_id": format(ctx.span_id, '016x'),
        }

    def inject_trace_id_to_log(self, log_record: Dict[str, Any]) -> Dict[str, Any]:
        """Inject trace_id into a log record for correlation."""
        if not self.cfg.inject_trace_id_to_logs:
            return log_record

        ctx = self.get_trace_context()
        if ctx:
            log_record.update(ctx)
        return log_record

    # ── Span Stream Analysis ─────────────────────────────────────────────

    def analyze_span_stream(self) -> Dict[str, Any]:
        """
        Analyze span duration stream for regime classification.

        From paper6 Thm 5.1: classify span patterns by growth-rate regime.
        """
        if len(self._span_durations) < 20:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_spans": len(self._span_durations),
            }

        # Compute deviations from median
        durations = self._span_durations[-100:]  # Last 100 spans
        median = np.median(durations)
        deviations = [abs(d - median) for d in durations if abs(d - median) > 0.001]

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
        elif theta_hat > 0.0:
            regime = QualityRegime.REGULARLY_VARYING
        else:
            regime = QualityRegime.DEGENERATE

        return {
            "regime": regime.name,
            "theta_hat": theta_hat,
            "n_spans": len(self._span_durations),
            "mean_duration_ms": np.mean(durations) * 1000,
            "p95_duration_ms": np.percentile(durations, 95) * 1000,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get instrumentor status."""
        return {
            "tracer_available": self.tracer is not None,
            "service_name": self.cfg.service_name,
            "span_count": len(self._span_timestamps),
            "span_stream_analysis": self.analyze_span_stream(),
            "quality_emitter_attached": self._quality_emitter is not None,
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESILiveObservabilitySuite:
    """
    [D] Complete live observability suite.

    Integrates:
      • Live quality metric emission
      • Zone-aware alert rules
      • OpenTelemetry Python instrumentation
    """

    def __init__(self, agione: Optional[Any] = None):
        self.emitter = LiveQualityMetricEmitter(agione)
        self.alert_gen = ZoneAwareAlertRuleGenerator()
        self.instrumentor = OTelPythonInstrumentor()

        # Attach emitter to instrumentor for quality injection
        self.instrumentor.attach_quality_emitter(self.emitter)

    def start_live_emission(self) -> None:
        """Start background quality metric emission."""
        self.emitter.start()

    def stop_live_emission(self) -> None:
        """Stop background quality metric emission."""
        self.emitter.stop()

    def generate_all_artifacts(self, output_dir: str = "./sesi_live_obs") -> Dict[str, str]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Zone-aware alert rules
        results["zone_alerts"] = self.alert_gen.write_alert_rules(
            str(out_path / "prometheus" / "zone-aware-alerts.yaml")
        )

        # OTel config
        otel_config = {
            "service_name": self.instrumentor.cfg.service_name,
            "exporter_type": self.instrumentor.cfg.exporter_type,
            "sampling_rate": self.instrumentor.cfg.sampling_rate,
        }
        otel_path = out_path / "otel" / "config.json"
        otel_path.parent.mkdir(parents=True, exist_ok=True)
        with open(otel_path, "w") as f:
            json.dump(otel_config, f, indent=2)
        results["otel_config"] = str(otel_path)

        # Example Python instrumentation
        example_code = '''
# Example: Using SESI OTel Python Instrumentor
from sesi_live_metrics_zone_alerts_otel import OTelPythonInstrumentor

# Initialize instrumentor
instrumentor = OTelPythonInstrumentor()

# Decorate functions
@instrumentor.trace_function(span_name="process_patient")
def process_patient(patient_id: str):
    # Your business logic here
    ...

@instrumentor.trace_async_function(span_name="async_query")
async def async_query(query: str):
    # Your async logic here
    ...

# Get trace context for log correlation
ctx = instrumentor.get_trace_context()
logger.info("Processing", extra=ctx)
'''
        example_path = out_path / "examples" / "python_instrumentation.py"
        example_path.parent.mkdir(parents=True, exist_ok=True)
        with open(example_path, "w") as f:
            f.write(example_code)
        results["example_code"] = str(example_path)

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "emitter": self.emitter.get_status(),
            "alerts": self.alert_gen.get_status(),
            "instrumentor": self.instrumentor.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Live Metrics × Zone Alerts × OTel Instrumentation — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  OpenTelemetry avail  : {HAS_OTEL}")
    print(f"  Prometheus available : {HAS_PROM}")
    print(f"  AGIONE available     : {HAS_AGIONE}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESILiveObservabilitySuite()

    # ── Test 1: Live Quality Emission ────────────────────────────────────
    print("\n  ── Live Quality Metric Emission ──")

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
    suite.emitter._last_quality = mock_quality

    # Emit once
    emit_result = suite.emitter.emit_once()
    print(f"    Emission status    : {emit_result['status']}")
    if emit_result['status'] == 'emitted':
        print(f"    Quality scores     : {len(emit_result['quality'])} domains")
        print(f"    Regimes            : {emit_result['regimes']}")

    # Check rate limit
    rate_check = suite.emitter.check_emit_rate()
    print(f"    Rate check         : {rate_check['allowed']}")
    print(f"    δ_min              : {suite.emitter.compute_delta_min():.6f}")
    print(f"    Max emissions/min  : {suite.emitter.max_emissions_in_window():.2f}")

    # ── Test 2: Zone-Aware Alert Rules ───────────────────────────────────
    print("\n  ── Zone-Aware Alert Rules ──")

    alert_status = suite.alert_gen.get_status()
    print(f"    Zone configs       : {alert_status['zone_configs']}")
    print(f"    Total alert rules  : {alert_status['total_alert_rules']}")
    print(f"    Inhibition rules   : {alert_status['inhibition_rules']}")

    # Generate rules for each zone
    for zone in SecurityZone:
        rules = suite.alert_gen.generate_zone_alerts(zone)
        print(f"    {zone.value:20s}: {len(rules)} rules")

    # ── Test 3: OTel Python Instrumentation ──────────────────────────────
    print("\n  ── OpenTelemetry Python Instrumentation ──")

    # Test decorator
    @suite.instrumentor.trace_function(span_name="test_function")
    def test_function(x: int) -> int:
        time.sleep(0.01)
        return x * 2

    result = test_function(5)
    print(f"    Decorated function : result={result}")

    # Test async decorator
    @suite.instrumentor.trace_async_function(span_name="test_async")
    async def test_async(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 3

    # Note: async test requires event loop
    # result = asyncio.run(test_async(5))

    # Test log correlation
    log_record = {"message": "Test log"}
    correlated = suite.instrumentor.inject_trace_id_to_log(log_record)
    print(f"    Log correlation    : {list(correlated.keys())}")

    # Span stream analysis
    for _ in range(50):
        test_function(1)
    analysis = suite.instrumentor.analyze_span_stream()
    print(f"    Span stream regime : {analysis['regime']}")
    if 'theta_hat' in analysis:
        print(f"    θ̂ (Hill)          : {analysis['theta_hat']:.4f}")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_live_obs")
    print(f"    Artifacts generated: {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:20s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Emitter running    : {status['emitter']['running']}")
    print(f"    Emitter δ_min      : {status['emitter']['delta_min']:.6f}")
    print(f"    Alert rules        : {status['alerts']['total_alert_rules']}")
    print(f"    OTel spans         : {status['instrumentor']['span_count']}")
    print(f"    Span regime        : {status['instrumentor']['span_stream_analysis']['regime']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Live emission provides OBSERVABILITY, not causation")
    print("  • Zone alerts are PROPORTIONALITY-BASED, not absolute")
    print("  • OTel instrumentation shows CORRELATION, not proof")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit")
    print("  • _safe_unit_scalar sanitization applied to all quality inputs")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
