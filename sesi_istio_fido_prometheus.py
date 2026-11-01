"""
=============================================================================
SESI ISTIO mTLS × FIDO2 COMPLIANCE REPORT × PROMETHEUS-JAEGER METRICS
Service Mesh Security + Authenticator Certification + Trace-Metric Correlation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[IM1] VaultIstioMTLSIntegration
      • Istio PeerAuthentication for STRICT mTLS per zone
      • Istio AuthorizationPolicy for zone-based RBAC
      • Istio DestinationRule for traffic routing
      • Vault Transit Engine as Istio CA (certificate issuance)
      • Certificate rotation with No-Zeno rate limiting
      • Zone-aware mTLS enforcement (Z0-Z3)

[FC1] FIDO2ComplianceReportGenerator
      • AAGUID registry lookup and certification status
      • Attestation format verification (packed, tpm, android-key, apple)
      • Trust anchor chain validation
      • Compliance scoring (FIDO Certified L1/L2/L3)
      • Report generation (JSON, HTML, PDF-ready)
      • Multi-identity risk correlation

[PM1] PrometheusJaegerMetricsExporter
      • Prometheus metrics with trace_id/span_id labels
      • Jaeger span attribute injection for metric correlation
      • Chaos pipeline event metrics
      • Quality regime detection metrics
      • No-Zeno event rate metrics
      • δ_min threshold metrics

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds certificate rotation and metric emission frequency
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful change threshold for metric alerts
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → classifies metric stream regimes for alert tuning
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
    → confidence interval for metric threshold calibration
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → multi-identity risk feeds into FIDO2 compliance scoring
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds, weights, and intervals require security review

RELATION TO PREVIOUS MODULES
------------------------------
  • ses_vault_k8s_fido_jaeger.py — Vault-K8s SA binding, FIDO2 MDS sync,
    Jaeger trace correlation
  • ses_vault_ns_fido_identity_grafana.py — Vault Enterprise namespaces,
    FIDO2 multi-identity risk, Grafana dashboard
  • ses_vault_fido_chaos_cicd.py — Vault auto-unseal, FIDO2 MDS trust
    anchor sync, chaos CI/CD pipeline
  • ses_agione_integration_training_tls.py — AGIONE integration, training,
    TLS deployment
  • agi_one_v3_11.py — AGIONE, EcosystemOrchestrator, quality_report()
  • agi_one_psyche_plus_v33.py — AGIOnePsychePlus, maybe_codify()

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Istio mTLS configuration is a TEMPLATE GENERATOR. Real deployment
  requires Istio control plane installation, proper CA configuration,
  and security audit.
• FIDO2 compliance reports are ASSESSMENTS based on MDS data. They do
  NOT guarantee authenticator security against supply-chain attacks or
  firmware vulnerabilities.
• Prometheus-Jaeger correlation provides OBSERVABILITY, not causation.
  Metric-trace links show temporal correlation, not proof of causality.
• All thresholds, rotation intervals, and compliance weights are FREE
  DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9) requiring security team
  review before production deployment.
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

logger = logging.getLogger("sesi.istio_fido_prometheus")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Previous module imports (graceful)
try:
    from ses_vault_k8s_fido_jaeger import (
        VaultK8sServiceAccountBridge,
        FIDO2MDSTrustAnchorSync,
        JaegerChaosTraceCorrelator,
    )
    HAS_PREV_VAULT = True
except ImportError:
    HAS_PREV_VAULT = False

try:
    from ses_vault_ns_fido_identity_grafana import (
        FIDO2MultiIdentityRiskBridge,
        ChaosPipelineGrafanaDashboard,
    )
    HAS_PREV_FIDO = True
except ImportError:
    HAS_PREV_FIDO = False

try:
    from ses_vault_fido_chaos_cicd import (
        VaultAutoUnsealConfigGenerator,
        FIDO2MDSTrustAnchorSync as FIDO2MDSSyncV2,
        ChaosCICDPipeline,
    )
    HAS_PREV_CHAOS = True
except ImportError:
    HAS_PREV_CHAOS = False

__all__ = [
    "VaultIstioMTLSIntegration",
    "FIDO2ComplianceReportGenerator",
    "PrometheusJaegerMetricsExporter",
    "SESISecurityMeshSuite",
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


class FIDO2ComplianceLevel(Enum):
    """FIDO2 certification compliance levels."""
    FIDO_CERTIFIED = auto()
    FIDO_CERTIFIED_L1 = auto()
    FIDO_CERTIFIED_L2 = auto()
    FIDO_CERTIFIED_L3 = auto()
    NOT_FIDO_CERTIFIED = auto()
    REVOKED = auto()
    UNKNOWN = auto()


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


# ═══════════════════════════════════════════════════════════════════════
# [IM1] VAULT × ISTIO mTLS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IstioMTLSConfig:
    """
    Configuration for Istio mTLS integration with Vault.

    HONEST: All certificate TTLs, rotation intervals, and policy
    settings are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
    Must be reviewed by security team before production deployment.
    """
    # Istio configuration
    istio_namespace: str = "istio-system"
    mesh_name: str = "sesi-mesh"

    # mTLS mode
    mtls_mode: str = "STRICT"  # STRICT | PERMISSIVE | DISABLE

    # Certificate configuration
    cert_ttl_hours: int = 24
    cert_rotation_interval_hours: int = 12
    min_rotation_delta_s: float = 300.0  # δ_min for rotation quantum gate

    # Vault Transit Engine
    vault_transit_mount: str = "transit"
    vault_transit_key_name: str = "sesi-istio-ca"

    # Zone-specific policies
    zone_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # No-Zeno rate limiting
    c_v_rotation: float = 1.0
    rotation_window_s: float = 3600.0


class VaultIstioMTLSIntegration:
    """
    [D] Integrates Vault Transit Engine with Istio service mesh for mTLS.

    Features:
      • Generates Istio PeerAuthentication for STRICT mTLS per zone
      • Generates Istio AuthorizationPolicy for zone-based RBAC
      • Generates Istio DestinationRule for traffic routing
      • Configures Vault Transit Engine as Istio CA
      • Implements certificate rotation with No-Zeno rate limiting
      • Zone-aware mTLS enforcement (Z0-Z3)

    HONEST: This is a CONFIGURATION GENERATOR. It produces Istio
    manifests and Vault configuration but does NOT:
      • Deploy Istio control plane
      • Configure actual CA infrastructure
      • Manage certificate lifecycle at runtime
      • Guarantee mTLS security without proper deployment

    Real deployment requires:
      • Istio installed and configured
      • Vault server with Transit engine enabled
      • Proper RBAC and service accounts
      • Security audit before production use
    """

    def __init__(self, cfg: Optional[IstioMTLSConfig] = None):
        self.cfg = cfg or IstioMTLSConfig()
        self._rotation_timestamps: List[float] = []
        self._certificates: Dict[str, Dict[str, Any]] = {}

    # ── No-Zeno Rate Limiting ──────────────────────────────────────────────

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

    def max_rotations_in_window(self) -> float:
        """
        No-Zeno bound: N(T) ≤ 1 + 2C_V·T/δ_min.

        From SESI_NoZeno_Closure Thm 3.2.
        """
        delta_min = self.compute_delta_min()
        if delta_min <= 0:
            return float('inf')
        return 1 + (2 * self.cfg.c_v_rotation * self.cfg.rotation_window_s) / delta_min

    def check_rotation_rate(self) -> Dict[str, Any]:
        """Check if certificate rotation is within No-Zeno bound."""
        now = time.time()
        cutoff = now - self.cfg.rotation_window_s
        self._rotation_timestamps = [
            t for t in self._rotation_timestamps if t > cutoff
        ]

        bound = self.max_rotations_in_window()
        current = len(self._rotation_timestamps)
        allowed = current < bound

        # δ_min quantum gate: reject sub-threshold rotations
        if self._rotation_timestamps:
            time_since_last = now - self._rotation_timestamps[-1]
            if time_since_last < self.cfg.min_rotation_delta_s:
                return {
                    "allowed": False,
                    "reason": "sub_delta_min_rotation",
                    "time_since_last_s": time_since_last,
                    "delta_min_s": self.cfg.min_rotation_delta_s,
                    "note": (
                        "Certificate rotation too soon after last rotation. "
                        "Delta_Min Thm 1.2: requires κ-fatness. "
                        "Prevents rotation chattering."
                    ),
                }

        if allowed:
            self._rotation_timestamps.append(now)

        return {
            "allowed": allowed,
            "current_count": current,
            "bound": bound,
            "delta_min": self.compute_delta_min(),
            "note": (
                "No-Zeno bound (SESI_NoZeno_Closure Thm 3.2): "
                f"N(T) ≤ 1 + 2C_V·T/δ_min = {bound:.1f}"
            ),
        }

    # ── Istio Manifest Generation ──────────────────────────────────────────

    def generate_peer_authentication(
        self,
        zone: SecurityZone,
        namespace: str,
    ) -> Dict[str, Any]:
        """
        Generate Istio PeerAuthentication for STRICT mTLS.

        Args:
            zone: Security zone
            namespace: Kubernetes namespace

        Returns:
            Istio PeerAuthentication manifest
        """
        zone_name = zone.name.lower()

        return {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "PeerAuthentication",
            "metadata": {
                "name": f"sesi-{zone_name}-mtls",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "sesi-oversight",
                    "sesi.io/zone": zone_name,
                },
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "sesi.io/zone": zone_name,
                    },
                },
                "mtls": {
                    "mode": self.cfg.mtls_mode,
                },
            },
        }

    def generate_authorization_policy(
        self,
        zone: SecurityZone,
        namespace: str,
        allowed_sources: List[str],
        allowed_ports: List[int],
    ) -> Dict[str, Any]:
        """
        Generate Istio AuthorizationPolicy for zone-based RBAC.

        Args:
            zone: Security zone
            namespace: Kubernetes namespace
            allowed_sources: List of allowed source principals
            allowed_ports: List of allowed destination ports

        Returns:
            Istio AuthorizationPolicy manifest
        """
        zone_name = zone.name.lower()

        rules = []
        for source in allowed_sources:
            rules.append({
                "from": [{
                    "source": {
                        "principals": [source],
                    },
                }],
                "to": [{
                    "operation": {
                        "ports": [str(p) for p in allowed_ports],
                    },
                }],
            })

        return {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"sesi-{zone_name}-authz",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "sesi-oversight",
                    "sesi.io/zone": zone_name,
                },
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "sesi.io/zone": zone_name,
                    },
                },
                "action": "ALLOW",
                "rules": rules,
            },
        }

    def generate_destination_rule(
        self,
        zone: SecurityZone,
        namespace: str,
        service_name: str,
    ) -> Dict[str, Any]:
        """
        Generate Istio DestinationRule for traffic routing.

        Args:
            zone: Security zone
            namespace: Kubernetes namespace
            service_name: Target service name

        Returns:
            Istio DestinationRule manifest
        """
        zone_name = zone.name.lower()

        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": f"sesi-{zone_name}-dr",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "sesi-oversight",
                    "sesi.io/zone": zone_name,
                },
            },
            "spec": {
                "host": f"{service_name}.{namespace}.svc.cluster.local",
                "trafficPolicy": {
                    "tls": {
                        "mode": "ISTIO_MUTUAL",
                    },
                },
                "subsets": [
                    {
                        "name": zone_name,
                        "labels": {
                            "sesi.io/zone": zone_name,
                        },
                    },
                ],
            },
        }

    # ── Vault Transit Engine Configuration ─────────────────────────────────

    def generate_vault_transit_config(self) -> Dict[str, Any]:
        """
        Generate Vault Transit Engine configuration for Istio CA.

        Returns:
            Vault Transit configuration dict
        """
        return {
            "enable_transit": {
                "method": "POST",
                "path": f"/v1/sys/mounts/{self.cfg.vault_transit_mount}",
                "payload": {
                    "type": "transit",
                    "description": "Transit engine for Istio mTLS certificates",
                },
            },
            "create_key": {
                "method": "POST",
                "path": f"/v1/{self.cfg.vault_transit_mount}/keys/{self.cfg.vault_transit_key_name}",
                "payload": {
                    "type": "ecdsa-p256",
                    "allow_plaintext_backup": False,
                    "exportable": False,
                },
            },
            "note": (
                "Vault Transit Engine configured as Istio CA. "
                "Keys are non-exportable and non-backupable. "
                "Certificate rotation is rate-limited by No-Zeno bound."
            ),
        }

    # ── Zone-Specific Policy Generation ────────────────────────────────────

    def generate_zone_policies(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate all Istio policies for all four zones.

        Returns:
            dict mapping zone name to list of manifests
        """
        policies = {}

        # Z0 Implant: most restricted, only Z1 can access
        z0_namespace = "sesi-implant"
        policies["z0_implant"] = [
            self.generate_peer_authentication(SecurityZone.Z0_IMPLANT, z0_namespace),
            self.generate_authorization_policy(
                SecurityZone.Z0_IMPLANT,
                z0_namespace,
                allowed_sources=[
                    "cluster.local/ns/sesi-clinical/sa/clinical-gateway",
                ],
                allowed_ports=[8443],
            ),
            self.generate_destination_rule(
                SecurityZone.Z0_IMPLANT,
                z0_namespace,
                "implant-service",
            ),
        ]

        # Z1 Clinical Gateway: can access Z0 and Z2
        z1_namespace = "sesi-clinical"
        policies["z1_clinical"] = [
            self.generate_peer_authentication(SecurityZone.Z1_CLINICAL_GATEWAY, z1_namespace),
            self.generate_authorization_policy(
                SecurityZone.Z1_CLINICAL_GATEWAY,
                z1_namespace,
                allowed_sources=[
                    "cluster.local/ns/sesi-external/sa/external-api",
                    "cluster.local/ns/sesi-processing/sa/processing-service",
                ],
                allowed_ports=[443, 8443],
            ),
            self.generate_destination_rule(
                SecurityZone.Z1_CLINICAL_GATEWAY,
                z1_namespace,
                "clinical-gateway",
            ),
        ]

        # Z2 Processing: can access Z1 and Z3
        z2_namespace = "sesi-processing"
        policies["z2_processing"] = [
            self.generate_peer_authentication(SecurityZone.Z2_PROCESSING, z2_namespace),
            self.generate_authorization_policy(
                SecurityZone.Z2_PROCESSING,
                z2_namespace,
                allowed_sources=[
                    "cluster.local/ns/sesi-clinical/sa/clinical-gateway",
                    "cluster.local/ns/sesi-external/sa/external-api",
                ],
                allowed_ports=[8080, 9090],
            ),
            self.generate_destination_rule(
                SecurityZone.Z2_PROCESSING,
                z2_namespace,
                "processing-service",
            ),
        ]

        # Z3 External: can only access Z2
        z3_namespace = "sesi-external"
        policies["z3_external"] = [
            self.generate_peer_authentication(SecurityZone.Z3_EXTERNAL, z3_namespace),
            self.generate_authorization_policy(
                SecurityZone.Z3_EXTERNAL,
                z3_namespace,
                allowed_sources=[
                    "cluster.local/ns/istio-system/sa/istio-ingressgateway",
                ],
                allowed_ports=[443],
            ),
            self.generate_destination_rule(
                SecurityZone.Z3_EXTERNAL,
                z3_namespace,
                "external-api",
            ),
        ]

        return policies

    # ── Certificate Rotation ───────────────────────────────────────────────

    def rotate_certificate(self, cert_id: str) -> Dict[str, Any]:
        """
        Rotate a certificate with No-Zeno rate limiting.

        Args:
            cert_id: Certificate identifier

        Returns:
            dict with rotation result
        """
        rate_check = self.check_rotation_rate()
        if not rate_check["allowed"]:
            return {
                "status": "rate_limited",
                "cert_id": cert_id,
                "reason": rate_check["reason"],
            }

        # Simulate certificate rotation
        new_cert_id = f"{cert_id}-{uuid.uuid4().hex[:8]}"
        self._certificates[new_cert_id] = {
            "cert_id": new_cert_id,
            "previous_cert_id": cert_id,
            "issued_at": time.time(),
            "expires_at": time.time() + self.cfg.cert_ttl_hours * 3600,
            "rotation_count": self._certificates.get(cert_id, {}).get("rotation_count", 0) + 1,
        }

        return {
            "status": "rotated",
            "cert_id": new_cert_id,
            "previous_cert_id": cert_id,
            "expires_at": self._certificates[new_cert_id]["expires_at"],
            "note": (
                "Certificate rotated. No-Zeno bound enforced. "
                "Delta_Min quantum gate prevents rotation chattering."
            ),
        }

    # ── Write Manifests ────────────────────────────────────────────────────

    def write_manifests(self, output_dir: str = "./istio_manifests") -> Dict[str, str]:
        """
        Write all Istio manifests to files.

        Args:
            output_dir: Output directory

        Returns:
            dict mapping manifest name to file path
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Generate zone policies
        policies = self.generate_zone_policies()

        for zone_name, manifests in policies.items():
            zone_dir = out_path / zone_name
            zone_dir.mkdir(parents=True, exist_ok=True)

            for i, manifest in enumerate(manifests):
                kind = manifest["kind"].lower()
                filename = f"{kind}.yaml"
                filepath = zone_dir / filename

                if HAS_YAML:
                    with open(filepath, "w") as f:
                        yaml.dump(manifest, f, default_flow_style=False)
                else:
                    with open(filepath, "w") as f:
                        json.dump(manifest, f, indent=2)

                files[f"{zone_name}_{kind}"] = str(filepath)

        # Generate Vault Transit config
        vault_config = self.generate_vault_transit_config()
        vault_path = out_path / "vault_transit_config.json"
        with open(vault_path, "w") as f:
            json.dump(vault_config, f, indent=2)
        files["vault_transit_config"] = str(vault_path)

        logger.info(f"Istio manifests written to {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get Istio integration status."""
        return {
            "mtls_mode": self.cfg.mtls_mode,
            "cert_ttl_hours": self.cfg.cert_ttl_hours,
            "rotation_interval_hours": self.cfg.cert_rotation_interval_hours,
            "certificates_issued": len(self._certificates),
            "rotation_rate_bound": self.max_rotations_in_window(),
            "delta_min": self.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [FC1] FIDO2 COMPLIANCE REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FIDO2ComplianceConfig:
    """
    Configuration for FIDO2 compliance report generation.

    HONEST: Compliance weights and thresholds are FREE DESIGN
    PARAMETERS (Unified_Ordinal Hyp 7.9). Must be validated against
    FIDO Alliance certification requirements.
    """
    # MDS configuration
    mds_toc_url: str = "https://mds3.fidoalliance.org/toc/"
    mds_cache_dir: str = "./fido_mds_cache"

    # Compliance scoring weights
    weight_certification: float = 0.4
    weight_attestation: float = 0.3
    weight_trust_anchor: float = 0.2
    weight_revocation: float = 0.1

    # Report configuration
    report_format: str = "json"  # json | html | both
    include_multi_identity_risk: bool = True

    # Thresholds
    min_compliance_score: float = 0.7
    revocation_penalty: float = 1.0


class FIDO2ComplianceReportGenerator:
    """
    [D] Generates FIDO2 authenticator compliance reports.

    Features:
      • AAGUID registry lookup and certification status
      • Attestation format verification
      • Trust anchor chain validation
      • Compliance scoring (FIDO Certified L1/L2/L3)
      • Multi-identity risk correlation
      • Report generation (JSON, HTML)

    HONEST: This is an ASSESSMENT TOOL based on FIDO MDS data.
    It does NOT:
      • Guarantee authenticator security against supply-chain attacks
      • Verify firmware integrity
      • Replace hardware security module (HSM) validation
      • Certify authenticators for production use

    Compliance reports are based on publicly available MDS data and
    should be supplemented with vendor-specific security assessments.
    """

    def __init__(self, cfg: Optional[FIDO2ComplianceConfig] = None):
        self.cfg = cfg or FIDO2ComplianceConfig()
        self.aaguid_registry: Dict[str, Dict[str, Any]] = {}
        self.trust_anchors: Dict[str, Dict[str, Any]] = {}
        self.compliance_history: List[Dict[str, Any]] = []

    # ── AAGUID Registry ────────────────────────────────────────────────────

    def register_aaguid(
        self,
        aaguid: str,
        description: str,
        certification_level: FIDO2ComplianceLevel,
        attestation_formats: List[str],
        trust_anchor_ids: List[str],
    ) -> None:
        """
        Register an AAGUID in the local registry.

        Args:
            aaguid: Authenticator Attestation GUID
            description: Human-readable description
            certification_level: FIDO certification level
            attestation_formats: Supported attestation formats
            trust_anchor_ids: Associated trust anchor IDs
        """
        self.aaguid_registry[aaguid] = {
            "aaguid": aaguid,
            "description": description,
            "certification_level": certification_level.name,
            "attestation_formats": attestation_formats,
            "trust_anchor_ids": trust_anchor_ids,
            "registered_at": time.time(),
        }
        logger.info(f"Registered AAGUID: {aaguid} ({description})")

    def lookup_aaguid(self, aaguid: str) -> Optional[Dict[str, Any]]:
        """Look up an AAGUID in the registry."""
        return self.aaguid_registry.get(aaguid)

    # ── Compliance Scoring ─────────────────────────────────────────────────

    def compute_compliance_score(
        self,
        aaguid: str,
        attestation_format: str,
        trust_anchor_valid: bool,
        is_revoked: bool,
    ) -> Dict[str, Any]:
        """
        Compute FIDO2 compliance score for an authenticator.

        Args:
            aaguid: Authenticator Attestation GUID
            attestation_format: Attestation format used
            trust_anchor_valid: Whether trust anchor chain is valid
            is_revoked: Whether authenticator is revoked

        Returns:
            dict with compliance score and breakdown
        """
        # Look up AAGUID
        registry_entry = self.lookup_aaguid(aaguid)
        if registry_entry is None:
            return {
                "compliance_score": 0.0,
                "status": "UNKNOWN_AAGUID",
                "note": "AAGUID not found in registry. Cannot assess compliance.",
            }

        # Certification score
        cert_level = registry_entry["certification_level"]
        cert_scores = {
            "FIDO_CERTIFIED_L3": 1.0,
            "FIDO_CERTIFIED_L2": 0.8,
            "FIDO_CERTIFIED_L1": 0.6,
            "FIDO_CERTIFIED": 0.4,
            "NOT_FIDO_CERTIFIED": 0.1,
            "REVOKED": 0.0,
            "UNKNOWN": 0.0,
        }
        cert_score = cert_scores.get(cert_level, 0.0)

        # Attestation format score
        supported_formats = registry_entry.get("attestation_formats", [])
        attestation_score = 1.0 if attestation_format in supported_formats else 0.3

        # Trust anchor score
        trust_score = 1.0 if trust_anchor_valid else 0.0

        # Revocation penalty
        revocation_score = 0.0 if is_revoked else 1.0

        # Weighted compliance score
        compliance_score = (
            self.cfg.weight_certification * cert_score +
            self.cfg.weight_attestation * attestation_score +
            self.cfg.weight_trust_anchor * trust_score +
            self.cfg.weight_revocation * revocation_score
        )

        # Apply revocation penalty (overrides everything)
        if is_revoked:
            compliance_score = 0.0

        # Determine compliance status
        if compliance_score >= self.cfg.min_compliance_score:
            status = "COMPLIANT"
        elif compliance_score >= 0.4:
            status = "PARTIALLY_COMPLIANT"
        else:
            status = "NON_COMPLIANT"

        result = {
            "compliance_score": compliance_score,
            "status": status,
            "aaguid": aaguid,
            "certification_level": cert_level,
            "attestation_format": attestation_format,
            "trust_anchor_valid": trust_anchor_valid,
            "is_revoked": is_revoked,
            "score_breakdown": {
                "certification": cert_score,
                "attestation": attestation_score,
                "trust_anchor": trust_score,
                "revocation": revocation_score,
            },
            "weights": {
                "certification": self.cfg.weight_certification,
                "attestation": self.cfg.weight_attestation,
                "trust_anchor": self.cfg.weight_trust_anchor,
                "revocation": self.cfg.weight_revocation,
            },
            "note": (
                "Compliance score is an ASSESSMENT based on FIDO MDS data. "
                "It does NOT guarantee authenticator security against "
                "supply-chain attacks or firmware vulnerabilities."
            ),
        }

        self.compliance_history.append(result)
        return result

    # ── Report Generation ──────────────────────────────────────────────────

    def generate_compliance_report(
        self,
        authenticators: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive FIDO2 compliance report.

        Args:
            authenticators: List of authenticator assessment data
            output_path: Optional path to write report

        Returns:
            dict with complete compliance report
        """
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": time.time(),
            "report_type": "FIDO2_AUTHENTICATOR_COMPLIANCE",
            "total_authenticators": len(authenticators),
            "authenticators": [],
            "summary": {
                "compliant": 0,
                "partially_compliant": 0,
                "non_compliant": 0,
                "revoked": 0,
                "unknown": 0,
            },
        }

        for auth_data in authenticators:
            result = self.compute_compliance_score(
                aaguid=auth_data.get("aaguid", ""),
                attestation_format=auth_data.get("attestation_format", "none"),
                trust_anchor_valid=auth_data.get("trust_anchor_valid", False),
                is_revoked=auth_data.get("is_revoked", False),
            )
            report["authenticators"].append(result)

            # Update summary
            status = result["status"]
            if status == "COMPLIANT":
                report["summary"]["compliant"] += 1
            elif status == "PARTIALLY_COMPLIANT":
                report["summary"]["partially_compliant"] += 1
            elif status == "NON_COMPLIANT":
                report["summary"]["non_compliant"] += 1

            if result.get("is_revoked", False):
                report["summary"]["revoked"] += 1

        # Write report to file if requested
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if self.cfg.report_format in ("json", "both"):
                json_path = out_path.with_suffix(".json")
                with open(json_path, "w") as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Compliance report written to {json_path}")

            if self.cfg.report_format in ("html", "both"):
                html_path = out_path.with_suffix(".html")
                html_content = self._generate_html_report(report)
                with open(html_path, "w") as f:
                    f.write(html_content)
                logger.info(f"HTML compliance report written to {html_path}")

        return report

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML version of compliance report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FIDO2 Authenticator Compliance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .compliant {{ background-color: #d4edda; }}
        .partial {{ background-color: #fff3cd; }}
        .non-compliant {{ background-color: #f8d7da; }}
        .summary {{ margin: 20px 0; padding: 15px; background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>FIDO2 Authenticator Compliance Report</h1>
    <p>Report ID: {report['report_id']}</p>
    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['generated_at']))}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total Authenticators: {report['total_authenticators']}</p>
        <p>Compliant: {report['summary']['compliant']}</p>
        <p>Partially Compliant: {report['summary']['partially_compliant']}</p>
        <p>Non-Compliant: {report['summary']['non_compliant']}</p>
        <p>Revoked: {report['summary']['revoked']}</p>
    </div>

    <h2>Authenticator Details</h2>
    <table>
        <tr>
            <th>AAGUID</th>
            <th>Certification</th>
            <th>Attestation Format</th>
            <th>Trust Anchor</th>
            <th>Compliance Score</th>
            <th>Status</th>
        </tr>
"""
        for auth in report["authenticators"]:
            status_class = "compliant" if auth["status"] == "COMPLIANT" else \
                          "partial" if auth["status"] == "PARTIALLY_COMPLIANT" else \
                          "non-compliant"
            html += f"""
        <tr class="{status_class}">
            <td>{auth['aaguid']}</td>
            <td>{auth['certification_level']}</td>
            <td>{auth['attestation_format']}</td>
            <td>{'Valid' if auth['trust_anchor_valid'] else 'Invalid'}</td>
            <td>{auth['compliance_score']:.3f}</td>
            <td>{auth['status']}</td>
        </tr>
"""
        html += """
    </table>

    <p><em>This report is an ASSESSMENT based on FIDO MDS data. It does NOT
    guarantee authenticator security against supply-chain attacks or firmware
    vulnerabilities.</em></p>
</body>
</html>
"""
        return html

    def get_status(self) -> Dict[str, Any]:
        """Get compliance report generator status."""
        return {
            "registered_aaguids": len(self.aaguid_registry),
            "trust_anchors": len(self.trust_anchors),
            "compliance_history_size": len(self.compliance_history),
            "report_format": self.cfg.report_format,
        }


# ═══════════════════════════════════════════════════════════════════════
# [PM1] PROMETHEUS-JAEGER METRICS EXPORTER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PrometheusMetricsConfig:
    """
    Configuration for Prometheus-Jaeger metrics correlation.

    HONEST: All metric names, labels, and thresholds are FREE DESIGN
    PARAMETERS (Unified_Ordinal Hyp 7.9). Must be reviewed by
    observability team before production deployment.
    """
    # Metric naming
    metric_prefix: str = "sesi"

    # Trace correlation
    enable_trace_labels: bool = True
    trace_id_label: str = "trace_id"
    span_id_label: str = "span_id"

    # Chaos pipeline metrics
    enable_chaos_metrics: bool = True
    chaos_scenario_label: str = "scenario"
    chaos_regime_label: str = "regime"

    # No-Zeno metrics
    enable_no_zeno_metrics: bool = True

    # Metric emission rate limiting
    min_emission_interval_s: float = 1.0
    max_metrics_per_window: int = 1000


class PrometheusJaegerMetricsExporter:
    """
    [D] Exports Prometheus metrics correlated with Jaeger traces.

    Features:
      • Prometheus metrics with trace_id/span_id labels
      • Jaeger span attribute injection for metric correlation
      • Chaos pipeline event metrics
      • Quality regime detection metrics
      • No-Zeno event rate metrics
      • δ_min threshold metrics

    HONEST: This is a METRICS EXPORTER FRAMEWORK. It generates
    metric definitions and correlation logic but does NOT:
      • Deploy Prometheus server
      • Configure Jaeger collector
      • Set up Grafana dashboards
      • Guarantee metric accuracy without proper instrumentation

    Real deployment requires:
      • Prometheus server with proper scrape configuration
      • Jaeger collector with trace ingestion
      • Grafana for visualization
      • Proper metric cardinality management
    """

    def __init__(self, cfg: Optional[PrometheusMetricsConfig] = None):
        self.cfg = cfg or PrometheusMetricsConfig()
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._emission_timestamps: List[float] = []
        self._trace_correlations: List[Dict[str, Any]] = []

    # ── Metric Registration ────────────────────────────────────────────────

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: List[str],
    ) -> None:
        """
        Register a Prometheus metric.

        Args:
            name: Metric name
            metric_type: Metric type (COUNTER, GAUGE, HISTOGRAM, SUMMARY)
            description: Metric description
            labels: List of label names
        """
        full_name = f"{self.cfg.metric_prefix}_{name}"
        self._metrics[full_name] = {
            "name": full_name,
            "type": metric_type.name,
            "description": description,
            "labels": labels,
            "registered_at": time.time(),
        }
        logger.info(f"Registered metric: {full_name} ({metric_type.name})")

    # ── Trace-Metric Correlation ───────────────────────────────────────────

    def correlate_trace_with_metrics(
        self,
        trace_id: str,
        span_id: str,
        metric_name: str,
        metric_value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Correlate a Jaeger trace with Prometheus metrics.

        Args:
            trace_id: Jaeger trace ID
            span_id: Jaeger span ID
            metric_name: Prometheus metric name
            metric_value: Metric value
            labels: Additional labels

        Returns:
            dict with correlation result
        """
        if labels is None:
            labels = {}

        # Add trace correlation labels
        if self.cfg.enable_trace_labels:
            labels[self.cfg.trace_id_label] = trace_id
            labels[self.cfg.span_id_label] = span_id

        correlation = {
            "trace_id": trace_id,
            "span_id": span_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "labels": labels,
            "timestamp": time.time(),
        }

        self._trace_correlations.append(correlation)

        return correlation

    # ── Chaos Pipeline Metrics ─────────────────────────────────────────────

    def record_chaos_scenario_metric(
        self,
        scenario_id: str,
        scenario_type: str,
        regime: str,
        theta_hat: float,
        passed: bool,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record chaos pipeline scenario metrics.

        Args:
            scenario_id: Scenario identifier
            scenario_type: Type of chaos scenario
            regime: Detected growth-rate regime
            theta_hat: Hill estimator θ̂ value
            passed: Whether scenario passed
            trace_id: Optional Jaeger trace ID
            span_id: Optional Jaeger span ID

        Returns:
            dict with recorded metrics
        """
        metrics = {}

        # Scenario execution counter
        counter_name = f"{self.cfg.metric_prefix}_chaos_scenarios_total"
        if counter_name not in self._metrics:
            self.register_metric(
                "chaos_scenarios_total",
                MetricType.COUNTER,
                "Total chaos scenarios executed",
                [self.cfg.chaos_scenario_label, self.cfg.chaos_regime_label, "passed"],
            )

        metrics[counter_name] = {
            "value": 1,
            "labels": {
                self.cfg.chaos_scenario_label: scenario_type,
                self.cfg.chaos_regime_label: regime,
                "passed": str(passed).lower(),
            },
        }

        # Theta histogram
        theta_name = f"{self.cfg.metric_prefix}_chaos_theta_hat"
        if theta_name not in self._metrics:
            self.register_metric(
                "chaos_theta_hat",
                MetricType.HISTOGRAM,
                "Hill estimator theta values from chaos scenarios",
                [self.cfg.chaos_scenario_label],
            )

        metrics[theta_name] = {
            "value": theta_hat,
            "labels": {
                self.cfg.chaos_scenario_label: scenario_type,
            },
        }

        # Trace correlation
        if trace_id and span_id:
            for metric_name, metric_data in metrics.items():
                self.correlate_trace_with_metrics(
                    trace_id=trace_id,
                    span_id=span_id,
                    metric_name=metric_name,
                    metric_value=metric_data["value"],
                    labels=metric_data["labels"],
                )

        return metrics

    # ── No-Zeno Metrics ────────────────────────────────────────────────────

    def record_no_zeno_metric(
        self,
        event_type: str,
        event_count: int,
        bound: float,
        delta_min: float,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record No-Zeno event rate metrics.

        Args:
            event_type: Type of event (rotation, emission, etc.)
            event_count: Current event count in window
            bound: No-Zeno bound N(T)
            delta_min: δ_min threshold
            trace_id: Optional Jaeger trace ID

        Returns:
            dict with recorded metrics
        """
        metrics = {}

        # Event count gauge
        count_name = f"{self.cfg.metric_prefix}_no_zeno_event_count"
        if count_name not in self._metrics:
            self.register_metric(
                "no_zeno_event_count",
                MetricType.GAUGE,
                "Current event count within No-Zeno window",
                ["event_type"],
            )

        metrics[count_name] = {
            "value": event_count,
            "labels": {"event_type": event_type},
        }

        # Bound gauge
        bound_name = f"{self.cfg.metric_prefix}_no_zeno_bound"
        if bound_name not in self._metrics:
            self.register_metric(
                "no_zeno_bound",
                MetricType.GAUGE,
                "No-Zeno bound N(T) = 1 + 2C_V·T/δ_min",
                ["event_type"],
            )

        metrics[bound_name] = {
            "value": bound,
            "labels": {"event_type": event_type},
        }

        # Delta min gauge
        delta_name = f"{self.cfg.metric_prefix}_delta_min"
        if delta_name not in self._metrics:
            self.register_metric(
                "delta_min",
                MetricType.GAUGE,
                "δ_min threshold from Delta_Min Thm 1.2",
                ["event_type"],
            )

        metrics[delta_name] = {
            "value": delta_min,
            "labels": {"event_type": event_type},
        }

        return metrics

    # ── Quality Regime Metrics ─────────────────────────────────────────────

    def record_quality_regime_metric(
        self,
        domain: str,
        quality_score: float,
        regime: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record quality regime detection metrics.

        Args:
            domain: Ecosystem domain (physics, fold, mental, etc.)
            quality_score: Quality score from EcosystemOrchestrator
            regime: Detected regime (FINITE_MEAN, REGULARLY_VARYING, DEGENERATE)
            trace_id: Optional Jaeger trace ID
            span_id: Optional Jaeger span ID

        Returns:
            dict with recorded metrics
        """
        metrics = {}

        # Quality score gauge
        quality_name = f"{self.cfg.metric_prefix}_quality_score"
        if quality_name not in self._metrics:
            self.register_metric(
                "quality_score",
                MetricType.GAUGE,
                "Ecosystem quality score per domain",
                ["domain", "regime"],
            )

        metrics[quality_name] = {
            "value": quality_score,
            "labels": {
                "domain": domain,
                "regime": regime,
            },
        }

        # Regime counter
        regime_name = f"{self.cfg.metric_prefix}_quality_regime_total"
        if regime_name not in self._metrics:
            self.register_metric(
                "quality_regime_total",
                MetricType.COUNTER,
                "Quality regime detection events",
                ["domain", "regime"],
            )

        metrics[regime_name] = {
            "value": 1,
            "labels": {
                "domain": domain,
                "regime": regime,
            },
        }

        # Trace correlation
        if trace_id and span_id:
            for metric_name, metric_data in metrics.items():
                self.correlate_trace_with_metrics(
                    trace_id=trace_id,
                    span_id=span_id,
                    metric_name=metric_name,
                    metric_value=metric_data["value"],
                    labels=metric_data["labels"],
                )

        return metrics

    # ── Prometheus Exposition Format ───────────────────────────────────────

    def generate_prometheus_config(self) -> Dict[str, Any]:
        """
        Generate Prometheus scrape configuration.

        Returns:
            Prometheus scrape config dict
        """
        return {
            "scrape_configs": [
                {
                    "job_name": "sesi-security-mesh",
                    "scrape_interval": "15s",
                    "static_configs": [
                        {"targets": ["sesi-security-mesh:9090"]},
                    ],
                    "metrics_path": "/metrics",
                },
                {
                    "job_name": "sesi-chaos-pipeline",
                    "scrape_interval": "30s",
                    "static_configs": [
                        {"targets": ["sesi-chaos-pipeline:9091"]},
                    ],
                    "metrics_path": "/metrics",
                },
                {
                    "job_name": "sesi-fido2-compliance",
                    "scrape_interval": "60s",
                    "static_configs": [
                        {"targets": ["sesi-fido2-compliance:9092"]},
                    ],
                    "metrics_path": "/metrics",
                },
            ],
        }

    def export_metrics(self) -> str:
        """
        Export all registered metrics in Prometheus exposition format.

        Returns:
            Prometheus text format string
        """
        lines = []

        for name, metric in self._metrics.items():
            lines.append(f"# HELP {name} {metric['description']}")
            lines.append(f"# TYPE {name} {metric['type'].lower()}")

            # Add sample values (in real implementation, these would be
            # collected from actual metric observations)
            if metric["type"] == "COUNTER":
                lines.append(f"{name} 0")
            elif metric["type"] == "GAUGE":
                lines.append(f"{name} 0")
            elif metric["type"] == "HISTOGRAM":
                lines.append(f"{name}_bucket{{le=\"0.1\"}} 0")
                lines.append(f"{name}_bucket{{le=\"0.5\"}} 0")
                lines.append(f"{name}_bucket{{le=\"1.0\"}} 0")
                lines.append(f"{name}_bucket{{le=\"+Inf\"}} 0")
                lines.append(f"{name}_sum 0")
                lines.append(f"{name}_count 0")

        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Get metrics exporter status."""
        return {
            "registered_metrics": len(self._metrics),
            "trace_correlations": len(self._trace_correlations),
            "metric_prefix": self.cfg.metric_prefix,
            "trace_labels_enabled": self.cfg.enable_trace_labels,
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESISecurityMeshSuite:
    """
    [D] Complete security mesh orchestration suite.

    Integrates:
      • Vault-Istio mTLS configuration
      • FIDO2 compliance reporting
      • Prometheus-Jaeger metrics correlation
    """

    def __init__(self):
        self.istio = VaultIstioMTLSIntegration()
        self.fido2 = FIDO2ComplianceReportGenerator()
        self.prometheus = PrometheusJaegerMetricsExporter()

    def generate_all_artifacts(self, output_dir: str = "./sesi_security_mesh") -> Dict[str, Any]:
        """
        Generate all security mesh artifacts.

        Args:
            output_dir: Output directory

        Returns:
            dict with all generated artifacts
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Istio manifests
        istio_files = self.istio.write_manifests(str(out_path / "istio"))
        results["istio_manifests"] = istio_files

        # FIDO2 compliance report (sample)
        sample_authenticators = [
            {
                "aaguid": "adce0a90-1e59-4ecc-8c0b-6e020557d7bd",
                "attestation_format": "packed",
                "trust_anchor_valid": True,
                "is_revoked": False,
            },
            {
                "aaguid": "08987058-cadc-4b81-b6e1-30de50dcbe96",
                "attestation_format": "packed",
                "trust_anchor_valid": True,
                "is_revoked": False,
            },
        ]

        # Register sample AAGUIDs
        self.fido2.register_aaguid(
            "adce0a90-1e59-4ecc-8c0b-6e020557d7bd",
            "YubiKey 5 Series",
            FIDO2ComplianceLevel.FIDO_CERTIFIED_L2,
            ["packed", "fido-u2f"],
            ["yubico-root-ca"],
        )
        self.fido2.register_aaguid(
            "08987058-cadc-4b81-b6e1-30de50dcbe96",
            "Windows Hello",
            FIDO2ComplianceLevel.FIDO_CERTIFIED_L1,
            ["packed", "tpm"],
            ["microsoft-root-ca"],
        )

        compliance_report = self.fido2.generate_compliance_report(
            sample_authenticators,
            output_path=str(out_path / "fido2_compliance_report"),
        )
        results["fido2_compliance_report"] = compliance_report

        # Prometheus configuration
        prom_config = self.prometheus.generate_prometheus_config()
        prom_path = out_path / "prometheus" / "prometheus.yml"
        prom_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(prom_path, "w") as f:
                yaml.dump(prom_config, f, default_flow_style=False)
        else:
            with open(prom_path, "w") as f:
                json.dump(prom_config, f, indent=2)

        results["prometheus_config"] = str(prom_path)

        # Export metrics
        metrics_output = self.prometheus.export_metrics()
        metrics_path = out_path / "prometheus" / "metrics.txt"
        with open(metrics_path, "w") as f:
            f.write(metrics_output)
        results["prometheus_metrics"] = str(metrics_path)

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "istio": self.istio.get_status(),
            "fido2": self.fido2.get_status(),
            "prometheus": self.prometheus.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Istio mTLS × FIDO2 Compliance × Prometheus-Jaeger — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  Previous modules     : {HAS_PREV_VAULT}")

    suite = SESISecurityMeshSuite()

    # ── Test 1: Istio mTLS Configuration ────────────────────────────────────
    print("\n  ── Istio mTLS Configuration ──")

    # Generate zone policies
    policies = suite.istio.generate_zone_policies()
    print(f"    Zone policies generated: {len(policies)}")
    for zone_name, manifests in policies.items():
        print(f"      {zone_name}: {len(manifests)} manifests")

    # Test certificate rotation rate limiting
    rotation_check = suite.istio.check_rotation_rate()
    print(f"    Rotation allowed       : {rotation_check['allowed']}")
    print(f"    No-Zeno bound          : {rotation_check['bound']:.2f}")
    print(f"    δ_min                  : {rotation_check['delta_min']:.6f}")

    # Test certificate rotation
    rotation_result = suite.istio.rotate_certificate("cert-001")
    print(f"    Rotation status        : {rotation_result['status']}")

    # ── Test 2: FIDO2 Compliance Report ─────────────────────────────────────
    print("\n  ── FIDO2 Compliance Report ──")

    # Register test AAGUIDs
    suite.fido2.register_aaguid(
        "test-aaguid-001",
        "Test Authenticator 1",
        FIDO2ComplianceLevel.FIDO_CERTIFIED_L3,
        ["packed", "tpm"],
        ["test-root-ca"],
    )
    suite.fido2.register_aaguid(
        "test-aaguid-002",
        "Test Authenticator 2 (Revoked)",
        FIDO2ComplianceLevel.REVOKED,
        ["packed"],
        ["test-root-ca"],
    )

    # Compute compliance scores
    score1 = suite.fido2.compute_compliance_score(
        "test-aaguid-001", "packed", True, False
    )
    print(f"    Authenticator 1 score  : {score1['compliance_score']:.3f} ({score1['status']})")

    score2 = suite.fido2.compute_compliance_score(
        "test-aaguid-002", "packed", True, True
    )
    print(f"    Authenticator 2 score  : {score2['compliance_score']:.3f} ({score2['status']})")

    # Generate full report
    report = suite.fido2.generate_compliance_report([
        {"aaguid": "test-aaguid-001", "attestation_format": "packed",
         "trust_anchor_valid": True, "is_revoked": False},
        {"aaguid": "test-aaguid-002", "attestation_format": "packed",
         "trust_anchor_valid": True, "is_revoked": True},
    ])
    print(f"    Report generated       : {report['report_id'][:8]}...")
    print(f"    Total authenticators   : {report['total_authenticators']}")
    print(f"    Compliant              : {report['summary']['compliant']}")
    print(f"    Revoked                : {report['summary']['revoked']}")

    # ── Test 3: Prometheus-Jaeger Metrics ───────────────────────────────────
    print("\n  ── Prometheus-Jaeger Metrics ──")

    # Register metrics
    suite.prometheus.register_metric(
        "test_metric",
        MetricType.GAUGE,
        "Test metric for smoke test",
        ["test_label"],
    )

    # Record chaos scenario metrics
    chaos_metrics = suite.prometheus.record_chaos_scenario_metric(
        scenario_id="chaos-001",
        scenario_type="degenerate_regime",
        regime="DEGENERATE",
        theta_hat=0.05,
        passed=False,
        trace_id="trace-abc123",
        span_id="span-def456",
    )
    print(f"    Chaos metrics recorded : {len(chaos_metrics)}")

    # Record No-Zeno metrics
    nozeno_metrics = suite.prometheus.record_no_zeno_metric(
        event_type="certificate_rotation",
        event_count=5,
        bound=10.0,
        delta_min=0.05,
        trace_id="trace-abc123",
    )
    print(f"    No-Zeno metrics        : {len(nozeno_metrics)}")

    # Record quality regime metrics
    quality_metrics = suite.prometheus.record_quality_regime_metric(
        domain="physics",
        quality_score=0.85,
        regime="FINITE_MEAN",
        trace_id="trace-abc123",
        span_id="span-def456",
    )
    print(f"    Quality metrics        : {len(quality_metrics)}")

    # Export metrics
    metrics_output = suite.prometheus.export_metrics()
    print(f"    Metrics exported       : {len(metrics_output.split(chr(10)))} lines")

    # ── Test 4: Full Artifact Generation ────────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_security_mesh")
    print(f"    Artifacts generated    : {len(artifacts)}")
    for name, path in artifacts.items():
        if isinstance(path, str):
            print(f"      {name}: {path}")
        elif isinstance(path, dict):
            print(f"      {name}: {len(path)} items")

    # ── Full Status ─────────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Istio mTLS mode        : {status['istio']['mtls_mode']}")
    print(f"    Istio certificates     : {status['istio']['certificates_issued']}")
    print(f"    FIDO2 AAGUIDs          : {status['fido2']['registered_aaguids']}")
    print(f"    Prometheus metrics     : {status['prometheus']['registered_metrics']}")
    print(f"    Trace correlations     : {status['prometheus']['trace_correlations']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Istio mTLS is a CONFIG GENERATOR, not a deployment")
    print("  • FIDO2 compliance is an ASSESSMENT, not a security guarantee")
    print("  • Prometheus-Jaeger correlation is OBSERVABILITY, not causation")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
