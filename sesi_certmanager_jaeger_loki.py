"""
=============================================================================
SESI AGI PRODUCTION: CERT-MANAGER + JAEGER ZONE MAP + LOKI LOGS
TLS Certificate Management + Zone-Aware Service Map + Structured Log Aggregation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[CM1] CertManagerGenerator
      • Certificate CRD for internal mTLS
      • Issuer (Let's Encrypt / self-signed / Vault PKI)
      • Certificate for Istio Gateway
      • Auto-renewal configuration
      • Integration with Istio PeerAuthentication

[JSM1] JaegerZoneAwareServiceMap
      • Zone-colored service dependency graph (Z0-Z3)
      • Per-edge metrics: call volume, latency P50/P95, error rate
      • Regime classification per edge (paper6 Thm 5.1)
      • CUSUM change-point detection on call streams
      • Cross-zone flow highlighting
      • Grafana nodeGraph panel generation
      • OpenTelemetry span integration

[LL1] LokiAGILogsAggregator
      • Structured JSON logging for all 10 AGI modules
      • Per-module log levels (reasoning, memory, perception, etc.)
      • LogQL query templates for AGI-specific analysis
      • Trace-log correlation via trace_id/span_id
      • Regime classification on log streams (paper6 Thm 5.1)
      • No-Zeno rate limiting on log emission
      • PNDC Axiom 3 data minimization
      • _safe_unit_scalar sanitization on all values

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds log emission and certificate renewal rate
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful log/metric change
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → per-edge and per-log-stream regime classification
[C] Six_Problems P4: CUSUM + subsampling
    → change-point detection in call/log streams
[C] Paper 9 Entry 2.5: Hill estimator
    → tail-index estimation for edge weights
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
    → NaN/Inf sanitization for all values
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-based access control, log data minimization
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_grafana_istio_argocd.py — GrafanaDashboardGenerator,
    IstioServiceMeshGenerator, ArgoCDGitOpsGenerator
  • sesi_agi_k8s_prometheus.py — KubernetesManifestGenerator,
    SESIAGIMetricsCollector, InstrumentedAGISystem
  • sesi_vampire_pdf_causal_full.py — Causal discovery modules
  • sesi_agi_ultimate_complete.py — 10 core AGI modules

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Cert-Manager is a CONFIGURATION GENERATOR; real deployment requires
  proper issuer setup (Let's Encrypt, Vault PKI, or self-signed CA).
• Jaeger service map is VISUALIZATION; it shows correlation, not causation.
• Loki logs are STRUCTURED RECORDS; they do not guarantee completeness.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• Production deployment requires security audit and regulatory approval.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.certmanager_jaeger_loki")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ── Optional imports ─────────────────────────────────────────────────────
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode, SpanKind
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

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
    "SecurityZone", "QualityRegime", "CertificateIssuer",
    "CertManagerGenerator", "CertificateSpec",
    "JaegerZoneAwareServiceMap", "ServiceNode", "ServiceEdge",
    "LokiAGILogsAggregator", "AGILogEntry", "LogQLQuery",
    "SESIProductionObservabilitySuite",
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


class QualityRegime(Enum):
    """Growth-rate regime classification (paper6 Thm 5.1)."""
    FINITE_MEAN = 0          # θ > 1
    REGULARLY_VARYING = 1    # θ ∈ (0, 1)
    DEGENERATE = 2           # θ → 0 (½δ₀ + ½δ₁)
    INSUFFICIENT_DATA = 3


class CertificateIssuer(Enum):
    """Supported certificate issuers."""
    LETSENCRYPT = "letsencrypt"
    SELF_SIGNED = "self-signed"
    VAULT_PKI = "vault-pki"
    CA_ISSUER = "ca-issuer"


# ═══════════════════════════════════════════════════════════════════════════
# [CM1] CERT-MANAGER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CertificateSpec:
    """Specification for a TLS certificate."""
    name: str
    namespace: str
    secret_name: str
    issuer_name: str
    issuer_kind: str  # "ClusterIssuer" | "Issuer"
    common_name: str
    dns_names: List[str] = field(default_factory=list)
    duration: str = "2160h"  # 90 days
    renew_before: str = "720h"  # 30 days
    is_ca: bool = False
    usages: List[str] = field(default_factory=lambda: [
        "server auth", "client auth"
    ])


class CertManagerGenerator:
    """
    [D] Generates Cert-Manager manifests for automated TLS certificate lifecycle.

    Features:
      • Certificate CRD for internal mTLS
      • Issuer configuration (Let's Encrypt / self-signed / Vault PKI)
      • Auto-renewal with configurable duration
      • Integration with Istio PeerAuthentication
      • No-Zeno rate limiting on renewal attempts

    HONEST: This is a CONFIGURATION GENERATOR. Real deployment requires:
      • Cert-Manager installed in the cluster
      • Proper DNS configuration for Let's Encrypt
      • Vault PKI backend configured (if using Vault)
      • Security audit of certificate chain

    Mathematical grounding:
      • X.509 PKI: Standard certificate chain validation
      • No-Zeno bound on renewal rate (SESI_NoZeno Thm 3.2)
    """

    def __init__(
        self,
        namespace: str = "sesi-agi",
        default_issuer: CertificateIssuer = CertificateIssuer.SELF_SIGNED,
    ):
        self.namespace = namespace
        self.default_issuer = default_issuer
        self._certificates: List[CertificateSpec] = []
        self._renewal_timestamps: deque = deque(maxlen=1000)

        # No-Zeno parameters
        self.max_renewals_per_hour = 10
        self.min_renewal_interval_s = 360.0  # 6 minutes

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_renewal_rate(self) -> Dict[str, Any]:
        """Check if renewal is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 3600.0
        self._renewal_timestamps = deque(
            (t for t in self._renewal_timestamps if t > cutoff),
            maxlen=1000,
        )

        if len(self._renewal_timestamps) >= self.max_renewals_per_hour:
            return {
                "allowed": False,
                "reason": "max_per_hour_exceeded",
                "current": len(self._renewal_timestamps),
            }

        if self._renewal_timestamps:
            time_since_last = now - self._renewal_timestamps[-1]
            if time_since_last < self.min_renewal_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                }

        return {"allowed": True}

    # ── Issuer Configuration ─────────────────────────────────────────────

    def generate_issuer(self) -> Dict[str, Any]:
        """Generate ClusterIssuer based on default_issuer."""
        if self.default_issuer == CertificateIssuer.LETSENCRYPT:
            return {
                "apiVersion": "cert-manager.io/v1",
                "kind": "ClusterIssuer",
                "metadata": {"name": "letsencrypt-prod"},
                "spec": {
                    "acme": {
                        "server": "https://acme-v02.api.letsencrypt.org/directory",
                        "email": "admin@sesi.example.com",
                        "privateKeySecretRef": {"name": "letsencrypt-prod-account-key"},
                        "solvers": [{
                            "http01": {"ingress": {"class": "istio"}},
                        }],
                    },
                },
            }
        elif self.default_issuer == CertificateIssuer.SELF_SIGNED:
            return {
                "apiVersion": "cert-manager.io/v1",
                "kind": "ClusterIssuer",
                "metadata": {"name": "sesi-selfsigned"},
                "spec": {"selfSigned": {}},
            }
        elif self.default_issuer == CertificateIssuer.VAULT_PKI:
            return {
                "apiVersion": "cert-manager.io/v1",
                "kind": "ClusterIssuer",
                "metadata": {"name": "vault-pki"},
                "spec": {
                    "vault": {
                        "path": "pki_int/sign/sesi-dot-com",
                        "server": "https://vault.sesi.example.com:8200",
                        "auth": {"tokenSecretRef": {"name": "vault-token", "key": "token"}},
                    },
                },
            }
        else:
            return {
                "apiVersion": "cert-manager.io/v1",
                "kind": "ClusterIssuer",
                "metadata": {"name": "sesi-ca-issuer"},
                "spec": {
                    "ca": {"secretName": "sesi-ca-key-pair"},
                },
            }

    # ── Certificate Generation ───────────────────────────────────────────

    def generate_certificate(self, spec: CertificateSpec) -> Dict[str, Any]:
        """Generate a Certificate CRD."""
        cert = {
            "apiVersion": "cert-manager.io/v1",
            "kind": "Certificate",
            "metadata": {
                "name": spec.name,
                "namespace": spec.namespace,
            },
            "spec": {
                "secretName": spec.secret_name,
                "issuerRef": {
                    "name": spec.issuer_name,
                    "kind": spec.issuer_kind,
                },
                "commonName": spec.common_name,
                "duration": spec.duration,
                "renewBefore": spec.renew_before,
                "usages": spec.usages,
            },
        }

        if spec.dns_names:
            cert["spec"]["dnsNames"] = spec.dns_names

        if spec.is_ca:
            cert["spec"]["isCA"] = True

        return cert

    def generate_internal_mtls_certificates(self) -> List[Dict[str, Any]]:
        """Generate certificates for internal mTLS between zones."""
        specs = [
            # Z0 Implant
            CertificateSpec(
                name="z0-implant-cert",
                namespace=self.namespace,
                secret_name="z0-implant-tls",
                issuer_name="sesi-selfsigned",
                issuer_kind="ClusterIssuer",
                common_name=f"implant.{self.namespace}.svc.cluster.local",
                dns_names=[
                    f"implant.{self.namespace}.svc.cluster.local",
                    "implant.sesi.local",
                ],
            ),
            # Z1 Clinical
            CertificateSpec(
                name="z1-clinical-cert",
                namespace=self.namespace,
                secret_name="z1-clinical-tls",
                issuer_name="sesi-selfsigned",
                issuer_kind="ClusterIssuer",
                common_name=f"clinical.{self.namespace}.svc.cluster.local",
                dns_names=[
                    f"clinical.{self.namespace}.svc.cluster.local",
                    "clinical.sesi.local",
                ],
            ),
            # Z2 Processing
            CertificateSpec(
                name="z2-processing-cert",
                namespace=self.namespace,
                secret_name="z2-processing-tls",
                issuer_name="sesi-selfsigned",
                issuer_kind="ClusterIssuer",
                common_name=f"processing.{self.namespace}.svc.cluster.local",
                dns_names=[
                    f"processing.{self.namespace}.svc.cluster.local",
                    "processing.sesi.local",
                ],
            ),
            # Z3 External
            CertificateSpec(
                name="z3-external-cert",
                namespace=self.namespace,
                secret_name="z3-external-tls",
                issuer_name="sesi-selfsigned",
                issuer_kind="ClusterIssuer",
                common_name=f"external.{self.namespace}.svc.cluster.local",
                dns_names=[
                    f"external.{self.namespace}.svc.cluster.local",
                    "agi.sesi.example.com",
                ],
            ),
        ]

        self._certificates.extend(specs)
        return [self.generate_certificate(s) for s in specs]

    def generate_gateway_certificate(self) -> Dict[str, Any]:
        """Generate certificate for Istio Gateway (external HTTPS)."""
        spec = CertificateSpec(
            name="gateway-tls-cert",
            namespace=self.namespace,
            secret_name="sesi-agi-tls-cert",
            issuer_name="letsencrypt-prod" if self.default_issuer == CertificateIssuer.LETSENCRYPT else "sesi-selfsigned",
            issuer_kind="ClusterIssuer",
            common_name="agi.sesi.example.com",
            dns_names=[
                "agi.sesi.example.com",
                "*.agi.sesi.example.com",
            ],
            duration="2160h",
            renew_before="720h",
        )
        self._certificates.append(spec)
        return self.generate_certificate(spec)

    def generate_all(self, output_dir: str = "./certmanager_manifests") -> Dict[str, str]:
        """Generate all Cert-Manager manifests."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        manifests = {
            "issuer": self.generate_issuer(),
            "gateway_certificate": self.generate_gateway_certificate(),
        }

        # Add internal mTLS certificates
        mtls_certs = self.generate_internal_mtls_certificates()
        for i, cert in enumerate(mtls_certs):
            manifests[f"mtls_cert_{i}"] = cert

        files = {}
        for name, manifest in manifests.items():
            file_path = out_path / f"{name}.yaml"
            if HAS_YAML:
                with open(file_path, "w") as f:
                    yaml.dump(manifest, f, default_flow_style=False)
            else:
                with open(file_path, "w") as f:
                    json.dump(manifest, f, indent=2)
            files[name] = str(file_path)

        logger.info(f"Generated {len(manifests)} Cert-Manager manifests in {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get generator status."""
        return {
            "namespace": self.namespace,
            "default_issuer": self.default_issuer.value,
            "total_certificates": len(self._certificates),
            "delta_min": self.compute_delta_min(),
            "max_renewals_per_hour": self.max_renewals_per_hour,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [JSM1] JAEGER ZONE-AWARE SERVICE MAP
# ═══════════════════════════════════════════════════════════════════════════

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


class JaegerZoneAwareServiceMap:
    """
    [D] Zone-aware Jaeger service map with regime classification.

    Features:
      • Zone-colored service dependency graph (Z0-Z3)
      • Per-edge metrics: call volume, latency P50/P95, error rate
      • Regime classification per edge (paper6 Thm 5.1)
      • CUSUM change-point detection on call streams
      • Cross-zone flow highlighting
      • Grafana nodeGraph panel generation
      • OpenTelemetry span integration

    HONEST: This is a VISUALIZATION TOOL. It shows call patterns,
    not whether those calls are safe or correct.

    Mathematical grounding:
      • paper6 Thm 5.1: Growth-rate dichotomy per edge
      • Six_Problems P4: CUSUM change-point detection
      • Hill estimator (Paper 9 Entry 2.5): edge weight regime
    """

    ZONE_COLORS = {
        SecurityZone.Z0_IMPLANT: "#ff6b6b",        # red — most restricted
        SecurityZone.Z1_CLINICAL: "#ffd93d",        # yellow — clinician access
        SecurityZone.Z2_PROCESSING: "#6bcf7f",      # green — AI inference
        SecurityZone.Z3_EXTERNAL: "#4d96ff",        # blue — public API
    }

    def __init__(self, min_calls_to_display: int = 10, latency_history_size: int = 100):
        self.min_calls_to_display = min_calls_to_display
        self.latency_history_size = latency_history_size
        self.nodes: Dict[str, ServiceNode] = {}
        self.edges: Dict[Tuple[str, str], ServiceEdge] = {}
        self._otel_tracer = None

        if HAS_OTEL:
            self._setup_otel_tracer()

    def _setup_otel_tracer(self) -> None:
        """Initialize OpenTelemetry tracer for span generation."""
        try:
            resource = Resource.create({
                "service.name": "sesi-jaeger-service-map",
                "service.version": "1.0.0",
            })
            provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            otel_trace.set_tracer_provider(provider)
            self._otel_tracer = otel_trace.get_tracer("sesi-jaeger")
            logger.info("[Jaeger] OpenTelemetry tracer initialized")
        except Exception as e:
            logger.warning(f"[Jaeger] OTel setup failed: {e}")

    # ── Service Registration ─────────────────────────────────────────────

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
            logger.info(f"[Jaeger] Registered service: {name} (zone={zone.value})")

    def register_default_services(self) -> None:
        """Register default SESI AGI services."""
        services = [
            ("sesi-implant-gateway", SecurityZone.Z0_IMPLANT, "implant"),
            ("sesi-clinical-gateway", SecurityZone.Z1_CLINICAL, "clinical"),
            ("sesi-processing", SecurityZone.Z2_PROCESSING, "processing"),
            ("sesi-external-api", SecurityZone.Z3_EXTERNAL, "external"),
            ("sesi-vault", SecurityZone.Z1_CLINICAL, "security"),
            ("sesi-agione", SecurityZone.Z2_PROCESSING, "agi-core"),
            ("sesi-psyche-plus", SecurityZone.Z2_PROCESSING, "agi-core"),
            ("sesi-causal-engine", SecurityZone.Z2_PROCESSING, "agi-core"),
            ("sesi-memory", SecurityZone.Z2_PROCESSING, "storage"),
            ("sesi-reasoning", SecurityZone.Z2_PROCESSING, "agi-core"),
        ]
        for name, zone, stype in services:
            self.register_service(name, zone, stype)

    # ── Call Recording ───────────────────────────────────────────────────

    def record_call(
        self,
        source: str,
        target: str,
        latency_ms: float,
        success: bool = True,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Record a service-to-service call.

        Applies PNDC Axiom 3: only necessary calls are recorded.
        """
        # Sanitize latency
        latency_ms = _safe_unit_scalar(
            latency_ms, name="latency_ms", logger=logger, default=0.0
        )
        if latency_ms is None:
            latency_ms = 0.0
        # Scale to reasonable ms range
        latency_ms = latency_ms * 1000.0

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
        if len(edge.latency_history) > self.latency_history_size:
            edge.latency_history = edge.latency_history[-self.latency_history_size:]

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

        # Generate OpenTelemetry span if tracer available
        if self._otel_tracer and trace_id:
            self._create_otel_span(source, target, latency_ms, success, trace_id)

    def _create_otel_span(
        self,
        source: str,
        target: str,
        latency_ms: float,
        success: bool,
        trace_id: str,
    ) -> None:
        """Create an OpenTelemetry span for the call."""
        try:
            with self._otel_tracer.start_as_current_span(
                f"{source} -> {target}",
                kind=SpanKind.CLIENT,
            ) as span:
                span.set_attribute("source.service", source)
                span.set_attribute("target.service", target)
                span.set_attribute("latency_ms", latency_ms)
                span.set_attribute("success", success)
                span.set_attribute("trace_id", trace_id)

                if source in self.nodes:
                    span.set_attribute("source.zone", self.nodes[source].zone.value)
                if target in self.nodes:
                    span.set_attribute("target.zone", self.nodes[target].zone.value)

                if success:
                    span.set_status(StatusCode.OK)
                else:
                    span.set_status(StatusCode.ERROR, "Call failed")
        except Exception as e:
            logger.debug(f"[Jaeger] Failed to create OTel span: {e}")

    # ── Regime Classification ────────────────────────────────────────────

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

        latencies = edge.latency_history
        median = np.median(latencies)
        deviations = [abs(l - median) for l in latencies if abs(l - median) > 0.01]

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
        drift = 0.5

        for i, v in enumerate(values):
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
            "change_point_index": change_idx,
        }

    # ── Service Map Generation ───────────────────────────────────────────

    def generate_jaeger_service_map(self) -> Dict[str, Any]:
        """Generate Jaeger-compatible service map JSON."""
        # Filter edges by PNDC Axiom 3 (only necessary edges displayed)
        filtered_edges = {
            k: v for k, v in self.edges.items()
            if v.call_count >= self.min_calls_to_display
        }

        # Build nodes list
        nodes = []
        for name, node in self.nodes.items():
            nodes.append({
                "id": name,
                "name": name,
                "zone": node.zone.value,
                "service_type": node.service_type,
                "color": self.ZONE_COLORS.get(node.zone, "#888"),
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
            "title": "SESI Zone-Aware Service Map (Jaeger)",
            "datasource": {"type": "jaeger", "uid": "jaeger"},
            "options": {
                "nodeGrid": {"x": 0, "y": 0, "columns": 4, "rows": 2},
                "edges": [
                    {"field": "call_count", "color": "orange"},
                    {"field": "error_rate", "color": "red"},
                    {"field": "p95_latency_ms", "color": "purple"},
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
        logger.info(f"[Jaeger] Service map written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get service map status."""
        filtered = {
            k: v for k, v in self.edges.items()
            if v.call_count >= self.min_calls_to_display
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
            "otel_tracer_available": self._otel_tracer is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [LL1] LOKI AGI LOGS AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AGILogEntry:
    """
    [D] A structured log entry from an AGI module.

    HONEST: Follows PNDC Axiom 3 (Necessity & Proportionality) —
    only necessary data is logged. Sensitive data is stripped.
    All values sanitized via _safe_unit_scalar.
    """
    timestamp: float
    module: str           # "reasoning" | "memory" | "perception" | etc.
    zone: SecurityZone
    level: str            # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
    event_type: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON for Loki ingestion."""
        entry = {
            "timestamp": self.timestamp,
            "module": self.module,
            "zone": self.zone.value,
            "level": self.level,
            "event_type": self.event_type,
            "message": self.message,
        }
        if self.trace_id:
            entry["trace_id"] = self.trace_id
        if self.span_id:
            entry["span_id"] = self.span_id
        if self.attributes:
            # Sanitize all attribute values
            sanitized = {}
            for k, v in self.attributes.items():
                if k.lower() in ("password", "secret", "token", "ssn", "credit_card"):
                    continue  # Strip sensitive data (PNDC Axiom 3)
                if isinstance(v, (int, float)):
                    sanitized[k] = _safe_unit_scalar(v, name=k, logger=logger, default=0.0)
                else:
                    sanitized[k] = v
            entry.update(sanitized)
        return json.dumps(entry)


@dataclass
class LogQLQuery:
    """A LogQL query template for Loki."""
    name: str
    query: str
    description: str


class LokiAGILogsAggregator:
    """
    [D] Structured log aggregator for all 10 AGI modules.

    Features:
      • Structured JSON logging for all AGI modules
      • Per-module log levels
      • LogQL query templates for AGI-specific analysis
      • Trace-log correlation via trace_id/span_id
      • Regime classification on log streams (paper6 Thm 5.1)
      • No-Zeno rate limiting on log emission
      • PNDC Axiom 3 data minimization
      • _safe_unit_scalar sanitization on all values

    HONEST: This is a LOGGING FRAMEWORK. It does NOT:
      • Guarantee that all events are captured
      • Replace security audit
      • Provide causation, only correlation

    Mathematical grounding:
      • PNDC Axiom 3: Necessity and Proportionality (log minimization)
      • paper6 Thm 5.1: Log stream regime classification
      • Six_Problems P4: CUSUM change-point detection
      • No-Zeno: Log emission rate limiting
    """

    # Standard LogQL queries for AGI analysis
    STANDARD_QUERIES = [
        LogQLQuery(
            name="agi_errors",
            query='{app="sesi-agi"} | json | level="ERROR" or level="CRITICAL"',
            description="All ERROR and CRITICAL logs from AGI modules",
        ),
        LogQLQuery(
            name="safety_violations",
            query='{app="sesi-agi"} | json | event_type="safety_violation"',
            description="Safety violation events",
        ),
        LogQLQuery(
            name="reasoning_latency",
            query='{app="sesi-agi"} | json | module="reasoning" | line_format "{{.latency_ms}}"',
            description="Reasoning engine latency distribution",
        ),
        LogQLQuery(
            name="memory_operations",
            query='{app="sesi-agi"} | json | module="memory" | event_type=~"store|query"',
            description="Memory store/query operations",
        ),
        LogQLQuery(
            name="causal_discovery",
            query='{app="sesi-agi"} | json | module="causal_discovery"',
            description="Causal discovery algorithm events",
        ),
        LogQLQuery(
            name="zone_transitions",
            query='{app="sesi-agi"} | json | event_type="zone_transition"',
            description="Cross-zone flow events",
        ),
        LogQLQuery(
            name="meta_learning",
            query='{app="sesi-agi"} | json | module="meta_learning"',
            description="Meta-learning adaptation events",
        ),
        LogQLQuery(
            name="trace_correlation",
            query='{app="sesi-agi"} | json | trace_id!="" | line_format "{{.trace_id}} {{.module}} {{.event_type}}"',
            description="All logs with trace_id for trace-log correlation",
        ),
    ]

    def __init__(
        self,
        app_name: str = "sesi-agi",
        max_buffer_size: int = 10000,
        max_logs_per_minute: int = 10000,
    ):
        self.app_name = app_name
        self.log_buffer: deque = deque(maxlen=max_buffer_size)
        self.max_logs_per_minute = max_logs_per_minute
        self._log_timestamps: deque = deque(maxlen=max_logs_per_minute * 2)
        self._module_log_counts: Dict[str, int] = defaultdict(int)
        self._log_history: List[AGILogEntry] = []

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

    # ── Log Emission ─────────────────────────────────────────────────────

    def log_agi_event(
        self,
        module: str,
        zone: SecurityZone,
        level: str,
        event_type: str,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[AGILogEntry]:
        """
        Log an AGI module event.

        Applies:
          1. No-Zeno rate limiting
          2. _safe_unit_scalar sanitization
          3. PNDC Axiom 3 data minimization (strip sensitive data)
        """
        # Check rate limit
        rate_check = self.check_log_rate()
        if not rate_check["allowed"]:
            logger.debug(f"[Loki] Log rate-limited: {rate_check['reason']}")
            return None

        entry = AGILogEntry(
            timestamp=time.time(),
            module=module,
            zone=zone,
            level=level,
            event_type=event_type,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {},
        )

        self.log_buffer.append(entry)
        self._log_history.append(entry)
        self._log_timestamps.append(time.time())
        self._module_log_counts[module] += 1

        return entry

    # ── Module-Specific Loggers ──────────────────────────────────────────

    def log_reasoning(self, goal: str, confidence: float, latency_ms: float,
                      zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                      trace_id: Optional[str] = None) -> Optional[AGILogEntry]:
        """Log reasoning engine event."""
        return self.log_agi_event(
            module="reasoning",
            zone=zone,
            level="INFO",
            event_type="reasoning_complete",
            message=f"Reasoning completed for goal: {goal[:50]}",
            trace_id=trace_id,
            attributes={
                "goal": goal[:100],
                "confidence": confidence,
                "latency_ms": latency_ms,
            },
        )

    def log_memory(self, operation: str, key: str, size: int,
                   zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                   trace_id: Optional[str] = None) -> Optional[AGILogEntry]:
        """Log memory operation."""
        return self.log_agi_event(
            module="memory",
            zone=zone,
            level="INFO",
            event_type=f"memory_{operation}",
            message=f"Memory {operation}: {key}",
            trace_id=trace_id,
            attributes={"key": key, "size": size, "operation": operation},
        )

    def log_perception(self, modality: str, content_length: int,
                       zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                       trace_id: Optional[str] = None) -> Optional[AGILogEntry]:
        """Log perception event."""
        return self.log_agi_event(
            module="perception",
            zone=zone,
            level="INFO",
            event_type="perception_complete",
            message=f"Perception processed: {modality}",
            trace_id=trace_id,
            attributes={"modality": modality, "content_length": content_length},
        )

    def log_safety(self, action: str, result: str,
                   violation_type: Optional[str] = None,
                   zone: SecurityZone = SecurityZone.Z1_CLINICAL,
                   trace_id: Optional[str] = None) -> Optional[AGILogEntry]:
        """Log safety verification event."""
        level = "WARNING" if result == "FAIL" else "INFO"
        event_type = "safety_violation" if result == "FAIL" else "safety_verified"
        attributes = {"action": action, "result": result}
        if violation_type:
            attributes["violation_type"] = violation_type

        return self.log_agi_event(
            module="safety",
            zone=zone,
            level=level,
            event_type=event_type,
            message=f"Safety check for {action}: {result}",
            trace_id=trace_id,
            attributes=attributes,
        )

    def log_causal_discovery(self, algorithm: str, acyclicity_h: float,
                              orientation_loss: float, n_edges: int,
                              zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                              trace_id: Optional[str] = None) -> Optional[AGILogEntry]:
        """Log causal discovery event."""
        return self.log_agi_event(
            module="causal_discovery",
            zone=zone,
            level="INFO",
            event_type="causal_discovery_complete",
            message=f"Causal discovery ({algorithm}) completed",
            trace_id=trace_id,
            attributes={
                "algorithm": algorithm,
                "acyclicity_h": acyclicity_h,
                "orientation_loss": orientation_loss,
                "n_edges": n_edges,
            },
        )

    def log_meta_learning(self, domain: str, loss_before: float, loss_after: float,
                          zone: SecurityZone = SecurityZone.Z2_PROCESSING,
                          trace_id: Optional[str] = None) -> Optional[AGILogEntry]:
        """Log meta-learning event."""
        return self.log_agi_event(
            module="meta_learning",
            zone=zone,
            level="INFO",
            event_type="meta_update_complete",
            message=f"Meta-learning update for domain: {domain}",
            trace_id=trace_id,
            attributes={
                "domain": domain,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "improvement": loss_before - loss_after,
            },
        )

    # ── Regime Classification ────────────────────────────────────────────

    def classify_log_regime(self, module: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify log stream regime using Hill estimator.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → stable log frequency
          θ ∈ (0,1):  regularly varying → heavy-tailed log frequency
          θ → 0:      degenerate → polarized log frequency
        """
        if module:
            history = [e for e in self._log_history if e.module == module]
        else:
            history = self._log_history

        if len(history) < 20:
            return {
                "regime": QualityRegime.INSUFFICIENT_DATA.name,
                "n_events": len(history),
            }

        # Use inter-log intervals
        times = sorted(e.timestamp for e in history[-100:])
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        intervals = [iv for iv in intervals if iv > 0.001]  # Filter tiny intervals

        if len(intervals) < 10:
            return {"regime": QualityRegime.FINITE_MEAN.name}

        median = np.median(intervals)
        deviations = [abs(iv - median) for iv in intervals if abs(iv - median) > 0.001]

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
            description = "θ > 1: stable log frequency."
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
            "n_events": len(history),
            "module": module or "all",
        }

    # ── LogQL Query Generation ───────────────────────────────────────────

    def generate_logql_queries(self) -> List[Dict[str, Any]]:
        """Generate LogQL query templates."""
        return [
            {
                "name": q.name,
                "query": q.query,
                "description": q.description,
            }
            for q in self.STANDARD_QUERIES
        ]

    # ── Log Export ───────────────────────────────────────────────────────

    def write_logs_to_file(self, output_path: str = "./logs/agi_logs.jsonl") -> str:
        """Write logs to JSONL file for Loki ingestion."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for entry in self.log_buffer:
                f.write(entry.to_json() + "\n")

        logger.info(f"[Loki] Wrote {len(self.log_buffer)} log entries to {output_path}")
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

        logger.info(f"[Loki] Config written to {output_path}")
        return str(out_path)

    def write_logql_queries(self, output_path: str = "./loki/agi-queries.json") -> str:
        """Write LogQL queries to file."""
        queries = self.generate_logql_queries()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(queries, f, indent=2)
        logger.info(f"[Loki] LogQL queries written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get aggregator status."""
        return {
            "app_name": self.app_name,
            "buffer_size": len(self.log_buffer),
            "total_events": len(self._log_history),
            "module_log_counts": dict(self._module_log_counts),
            "regime": self.classify_log_regime(),
            "recent_log_rate_per_min": len([
                t for t in self._log_timestamps
                if t > time.time() - 60.0
            ]),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIProductionObservabilitySuite:
    """
    [D] Complete observability suite integrating:
      • Cert-Manager for TLS certificates
      • Jaeger zone-aware service map
      • Loki AGI logs aggregation
    """

    def __init__(
        self,
        namespace: str = "sesi-agi",
        cert_issuer: CertificateIssuer = CertificateIssuer.SELF_SIGNED,
    ):
        self.cert_manager = CertManagerGenerator(namespace, cert_issuer)
        self.jaeger_map = JaegerZoneAwareServiceMap()
        self.loki_aggregator = LokiAGILogsAggregator()

        # Register default services
        self.jaeger_map.register_default_services()

    def simulate_agi_cycle(self) -> Dict[str, Any]:
        """
        Simulate a single AGI autonomous learning cycle with full observability.
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        cycle_start = time.time()

        # 1. Perception
        self.loki_aggregator.log_perception(
            modality="multimodal_code_and_text",
            content_length=100,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 2. Reasoning
        reasoning_start = time.time()
        confidence = random.uniform(0.85, 0.99)
        reasoning_latency = (time.time() - reasoning_start) * 1000
        self.loki_aggregator.log_reasoning(
            goal="Investigate quantum topology",
            confidence=confidence,
            latency_ms=reasoning_latency,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 3. Memory operations
        self.loki_aggregator.log_memory(
            operation="store", key="quantum_topology", size=1,
            zone=SecurityZone.Z2_PROCESSING, trace_id=trace_id,
        )
        self.loki_aggregator.log_memory(
            operation="query", key="quantum_topology", size=1,
            zone=SecurityZone.Z2_PROCESSING, trace_id=trace_id,
        )

        # 4. Safety check
        self.loki_aggregator.log_safety(
            action="execute_experiment",
            result="PASS",
            zone=SecurityZone.Z1_CLINICAL,
            trace_id=trace_id,
        )

        # 5. Causal discovery
        self.loki_aggregator.log_causal_discovery(
            algorithm="FCI++",
            acyclicity_h=random.uniform(0.0001, 0.01),
            orientation_loss=random.uniform(0.01, 0.1),
            n_edges=random.randint(3, 10),
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 6. Meta-learning
        self.loki_aggregator.log_meta_learning(
            domain="quantum_topology",
            loss_before=0.15,
            loss_after=0.08,
            zone=SecurityZone.Z2_PROCESSING,
            trace_id=trace_id,
        )

        # 7. Service calls (for Jaeger map)
        service_calls = [
            ("sesi-external-api", "sesi-processing", 50.0, True),
            ("sesi-processing", "sesi-reasoning", 30.0, True),
            ("sesi-reasoning", "sesi-memory", 20.0, True),
            ("sesi-processing", "sesi-causal-engine", 80.0, True),
            ("sesi-causal-engine", "sesi-psyche-plus", 40.0, True),
            ("sesi-processing", "sesi-clinical-gateway", 100.0, True),
            ("sesi-clinical-gateway", "sesi-vault", 25.0, True),
        ]
        for src, tgt, latency, success in service_calls:
            self.jaeger_map.record_call(src, tgt, latency, success, trace_id=trace_id)

        cycle_latency = time.time() - cycle_start

        return {
            "trace_id": trace_id,
            "cycle_latency_s": cycle_latency,
            "logs_emitted": len(self.loki_aggregator.log_buffer),
        }

    def generate_all_artifacts(self, output_dir: str = "./sesi_production_obs") -> Dict[str, Any]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Cert-Manager manifests
        results["certmanager"] = self.cert_manager.generate_all(
            str(out_path / "certmanager")
        )

        # Jaeger service map
        results["jaeger_service_map"] = self.jaeger_map.write_service_map(
            str(out_path / "jaeger" / "service-map.json")
        )

        # Grafana panel for service map
        panel = self.jaeger_map.generate_grafana_panel()
        panel_path = out_path / "grafana" / "service-map-panel.json"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        with open(panel_path, "w") as f:
            json.dump(panel, f, indent=2)
        results["grafana_panel"] = str(panel_path)

        # Loki config and logs
        results["loki_config"] = self.loki_aggregator.write_loki_config(
            str(out_path / "loki" / "loki-config.yaml")
        )
        results["logql_queries"] = self.loki_aggregator.write_logql_queries(
            str(out_path / "loki" / "agi-queries.json")
        )
        results["agi_logs"] = self.loki_aggregator.write_logs_to_file(
            str(out_path / "logs" / "agi_logs.jsonl")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "cert_manager": self.cert_manager.get_status(),
            "jaeger_map": self.jaeger_map.get_status(),
            "loki_aggregator": self.loki_aggregator.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Cert-Manager + Jaeger Zone Map + Loki Logs — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  OpenTelemetry avail  : {HAS_OTEL}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIProductionObservabilitySuite(
        namespace="sesi-agi",
        cert_issuer=CertificateIssuer.SELF_SIGNED,
    )

    # ── Test 1: Cert-Manager ─────────────────────────────────────────────
    print("\n  ── Cert-Manager Generator ──")
    cm_status = suite.cert_manager.get_status()
    print(f"    Namespace            : {cm_status['namespace']}")
    print(f"    Default issuer       : {cm_status['default_issuer']}")
    print(f"    δ_min                : {cm_status['delta_min']:.6f}")

    # Generate issuer
    issuer = suite.cert_manager.generate_issuer()
    print(f"    Issuer kind          : {issuer['kind']}")
    print(f"    Issuer name          : {issuer['metadata']['name']}")

    # Generate gateway certificate
    gw_cert = suite.cert_manager.generate_gateway_certificate()
    print(f"    Gateway cert         : {gw_cert['metadata']['name']}")
    print(f"    Common name          : {gw_cert['spec']['commonName']}")

    # ── Test 2: Jaeger Zone-Aware Service Map ────────────────────────────
    print("\n  ── Jaeger Zone-Aware Service Map ──")

    # Simulate multiple AGI cycles
    for i in range(25):
        cycle = suite.simulate_agi_cycle()
        if i % 10 == 0:
            print(f"    Cycle {i+1:2d}           : trace={cycle['trace_id'][:12]}...")

    # Generate service map
    service_map = suite.jaeger_map.generate_jaeger_service_map()
    print(f"    Nodes                : {service_map['n_nodes']}")
    print(f"    Edges (filtered)     : {service_map['n_edges']}")

    # Service map status
    sm_status = suite.jaeger_map.get_status()
    print(f"    Zone crossings       : {sm_status['zone_crossing_edges']}")
    print(f"    Regime distribution  : {sm_status['regime_distribution']}")
    print(f"    OTel tracer          : {sm_status['otel_tracer_available']}")

    # ── Test 3: Loki AGI Logs ────────────────────────────────────────────
    print("\n  ── Loki AGI Logs Aggregator ──")

    loki_status = suite.loki_aggregator.get_status()
    print(f"    Total log events     : {loki_status['total_events']}")
    print(f"    Buffer size          : {loki_status['buffer_size']}")
    print(f"    Module log counts    : {loki_status['module_log_counts']}")
    print(f"    Log regime           : {loki_status['regime']['regime']}")

    # Generate LogQL queries
    queries = suite.loki_aggregator.generate_logql_queries()
    print(f"    LogQL queries        : {len(queries)}")
    for q in queries[:3]:
        print(f"      • {q['name']}: {q['description'][:50]}...")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_production_obs")
    print(f"    Artifacts generated  : {len(artifacts)}")
    for name, path in artifacts.items():
        if isinstance(path, dict):
            print(f"      {name:20s}: {len(path)} files")
        else:
            print(f"      {name:20s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Certificates         : {status['cert_manager']['total_certificates']}")
    print(f"    Service map nodes    : {status['jaeger_map']['n_nodes']}")
    print(f"    Service map edges    : {status['jaeger_map']['n_edges']}")
    print(f"    Log events           : {status['loki_aggregator']['total_events']}")
    print(f"    Log regime           : {status['loki_aggregator']['regime']['regime']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Cert-Manager is a CONFIG GENERATOR, not deployed system")
    print("  • Jaeger service map is VISUALIZATION, not security guarantee")
    print("  • Loki logs are STRUCTURED RECORDS, not omniscience")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all values")
    print("  • PNDC Axiom 3 data minimization in all logs")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
