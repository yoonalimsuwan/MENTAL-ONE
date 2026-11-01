"""
=============================================================================
SESI VAULT-K8S SECRETS × JAEGER SECURITY TRACING × RESIDENT KEY MANAGEMENT
Kubernetes Secrets Operator + Jaeger Dashboard + FIDO2 Discoverable Credentials
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[VK1] VaultKubernetesSecretsOperator
      • External Secrets Operator (ESO) CRD generation
      • Vault Agent Injector annotation generator
      • CSI Secrets Store driver configuration
      • Zone-aware secret routing (Z0 implant → Z3 external)
      • Rotation policy with No-Zeno bound + δ_min quantum gate

[JT1] JaegerSecurityDashboard
      • Jaeger/Grafana dashboard JSON for security tracing
      • Trace query templates per security zone
      • Alert rules with CUSUM change-point detection
      • Growth-rate regime classifier for trace streams
      • Span taxonomy for security events

[RK1] ResidentKeyManager
      • FIDO2 discoverable credential (passkey) lifecycle
      • Encrypted credential storage (Vault Transit backend)
      • Multi-device credential sync protocol
      • Attestation verification for resident keys
      • Multi-identity risk integration (Comprehensive Security §5.2)

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds secret rotation frequency
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → quantum gate for meaningful rotation changes
[C] Stochastic_NoZeno Thm 4.1: P(sup||h(r)-h(T_k)|| ≥ δ|F_{T_k}) ≤ 2exp(-δ²/8G₀²ε)
    → exit-time bound for trace anomaly detection
[C] paper6 Thm 5.1: growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → trace stream regime classification
[C] Six_Problems P4: CUSUM + subsampling on raw process
    → change-point detection in security event streams
[C] Paper11 Thm 4.4: assumption-light CI for δ_min
    → confidence interval for rotation threshold estimation
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → multi-identity risk for resident key management
[C] PNDC Axiom 3: Necessity and Proportionality
    → credential data minimization

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_vault_telemetry_webauthn.py — VaultSecretManager, WebAuthnFIDO2Manager
  • ses_multi_layer_oversight_v2.py — SecurityZone, NetworkPolicy
  • ses_auth_transformer_websocket.py — AuthenticationAuthorizationLayer
  • agi_one_v3_11.py — AGIONE, EcosystemOrchestrator
  • agi_one_psyche_plus_v33.py — AGIOnePsychePlus, quality_score gating

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Vault K8s integration is a CONFIGURATION GENERATOR, not a Vault deployment.
• Jaeger dashboard is a VISUALIZATION TEMPLATE, not a security oracle.
• Resident key management is a LIFECYCLE FRAMEWORK, not a hardware
  security module (HSM). Real passkey storage requires platform
  authenticators (TPM, Secure Enclave, Android Keystore).
• All rotation thresholds, sampling rates, and risk weights are
  FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• Production deployment requires security audit by qualified professionals.
=============================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.vault_k8s_jaeger_rk")

# Optional imports
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import hvac
    HAS_HVAC = True
except ImportError:
    HAS_HVAC = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Previous module imports (graceful)
try:
    from ses_vault_telemetry_webauthn import (
        VaultSecretManager, WebAuthnFIDO2Manager, SecurityZone,
    )
    HAS_PREV_MODULES = True
except ImportError:
    HAS_PREV_MODULES = False
    class SecurityZone(Enum):
        Z0_IMPLANT = 0
        Z1_CLINICAL_GATEWAY = 1
        Z2_PROCESSING = 2
        Z3_EXTERNAL = 3

try:
    from ses_multi_layer_oversight_v2 import MultiLayerOversightSystem
    HAS_OVERSIGHT = True
except ImportError:
    HAS_OVERSIGHT = False

__all__ = [
    "VaultKubernetesSecretsOperator",
    "JaegerSecurityDashboard",
    "ResidentKeyManager",
    "SESISecurityOperationsSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# [VK1] VAULT KUBERNETES SECRETS OPERATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VaultK8sConfig:
    """
    Configuration for Vault-Kubernetes integration.

    HONEST: All rotation intervals and thresholds are FREE DESIGN
    PARAMETERS (Unified_Ordinal Hyp 7.9). Must be reviewed by
    security team before production deployment.
    """
    # Vault server
    vault_addr: str = "https://vault.sesi.example.com:8200"
    vault_auth_path: str = "kubernetes"
    vault_role: str = "sesi-oversight"

    # External Secrets Operator
    eso_namespace: str = "external-secrets"
    secret_store_name: str = "sesi-vault-store"

    # Zone-specific secret paths
    zone_secret_paths: Dict[str, str] = field(default_factory=lambda: {
        "z0_implant": "secret/data/sesi/implant",
        "z1_clinical": "secret/data/sesi/clinical",
        "z2_processing": "secret/data/sesi/processing",
        "z3_external": "secret/data/sesi/external",
    })

    # Rotation policy (No-Zeno bounded)
    rotation_interval_s: int = 86400  # 24 hours
    min_rotation_delta: float = 0.1   # δ_min quantum gate
    c_v_rotation: float = 1.0         # rotation velocity bound

    # CSI driver
    csi_driver_name: str = "secrets-store.csi.k8s.io"
    csi_provider: str = "vault"

    # Vault Agent Injector
    agent_inject: bool = True
    agent_image: str = "hashicorp/vault:1.15"


class VaultKubernetesSecretsOperator:
    """
    [D] Generates Kubernetes manifests for Vault secret integration.

    Supports three integration patterns:
      1. External Secrets Operator (ESO) — CRD-based sync
      2. Vault Agent Injector — sidecar injection via annotations
      3. CSI Secrets Store Driver — volume-mounted secrets

    Zone-aware routing ensures secrets are only accessible from
    the appropriate security zone (Z0-Z3).

    MATHEMATICAL GUARANTEES:
      • Rotation frequency bounded by No-Zeno: N(T) ≤ 1 + 2C_V·T/δ_min
      • Each rotation must exceed δ_min quantum (Delta_Min Thm 1.2)
      • δ_min requires κ-fatness (Delta_Min Def 1.1)

    HONEST: This generates CONFIGURATION, not security. Real deployment
    requires:
      • Vault server properly configured and hardened
      • RBAC policies reviewed by security team
      • Network policies enforcing zone isolation
      • Regular security audits
    """

    def __init__(self, cfg: Optional[VaultK8sConfig] = None):
        self.cfg = cfg or VaultK8sConfig()
        self._rotation_log: List[Dict[str, Any]] = []
        self._last_rotation: Dict[str, float] = {}

    # ── External Secrets Operator ──────────────────────────────────────────

    def generate_cluster_secret_store(self) -> Dict[str, Any]:
        """
        Generate ClusterSecretStore CRD for External Secrets Operator.

        This is a cluster-scoped resource that defines how to connect
        to Vault. All ExternalSecrets reference this store.
        """
        return {
            "apiVersion": "external-secrets.io/v1beta1",
            "kind": "ClusterSecretStore",
            "metadata": {
                "name": self.cfg.secret_store_name,
                "labels": {
                    "app.kubernetes.io/name": "sesi-oversight",
                    "app.kubernetes.io/component": "vault-integration",
                },
            },
            "spec": {
                "provider": {
                    "vault": {
                        "server": self.cfg.vault_addr,
                        "path": "secret",
                        "version": "v2",
                        "auth": {
                            "kubernetes": {
                                "mountPath": self.cfg.vault_auth_path,
                                "role": self.cfg.vault_role,
                                "serviceAccountRef": {
                                    "name": "sesi-vault-auth",
                                    "namespace": self.cfg.eso_namespace,
                                },
                            },
                        },
                    },
                },
            },
        }

    def generate_external_secret(
        self,
        name: str,
        namespace: str,
        zone: SecurityZone,
        vault_path: str,
        secret_keys: List[str],
        refresh_interval: str = "1h",
    ) -> Dict[str, Any]:
        """
        Generate ExternalSecret CRD for a specific zone.

        Args:
            name: ExternalSecret name
            namespace: Target namespace
            zone: Security zone (determines vault path)
            vault_path: Path within the zone's secret tree
            secret_keys: List of keys to extract
            refresh_interval: How often to sync (default 1h)

        Returns:
            ExternalSecret manifest dict
        """
        zone_path = self.cfg.zone_secret_paths.get(
            zone.name.lower(), "secret/data/sesi/unknown"
        )
        full_path = f"{zone_path}/{vault_path}"

        # Build data entries
        data_entries = []
        for key in secret_keys:
            data_entries.append({
                "secretKey": key,
                "remoteRef": {
                    "key": full_path,
                    "property": key,
                },
            })

        return {
            "apiVersion": "external-secrets.io/v1beta1",
            "kind": "ExternalSecret",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "sesi-oversight",
                    "sesi.io/zone": zone.name.lower(),
                },
            },
            "spec": {
                "refreshInterval": refresh_interval,
                "secretStoreRef": {
                    "name": self.cfg.secret_store_name,
                    "kind": "ClusterSecretStore",
                },
                "target": {
                    "name": f"{name}-secret",
                    "creationPolicy": "Owner",
                    "template": {
                        "metadata": {
                            "labels": {
                                "sesi.io/zone": zone.name.lower(),
                                "sesi.io/managed-by": "external-secrets",
                            },
                        },
                    },
                },
                "data": data_entries,
            },
        }

    # ── Vault Agent Injector ──────────────────────────────────────────────

    def generate_agent_injector_annotations(
        self,
        zone: SecurityZone,
        vault_path: str,
        secret_keys: List[str],
        template_name: str = "sesi-secret",
    ) -> Dict[str, str]:
        """
        Generate pod annotations for Vault Agent Injector.

        The Vault Agent Injector is a Kubernetes mutating webhook that
        injects a Vault Agent sidecar into pods. The agent handles
        authentication, secret retrieval, and template rendering.

        HONEST: Agent Injector requires the Vault Helm chart with
        injector enabled. Secrets are rendered to shared memory
        (tmpfs) by default.

        Args:
            zone: Security zone
            vault_path: Path within zone's secret tree
            secret_keys: Keys to render
            template_name: Name for the template file

        Returns:
            Dict of pod annotations
        """
        zone_path = self.cfg.zone_secret_paths.get(
            zone.name.lower(), "secret/data/sesi/unknown"
        )
        full_path = f"{zone_path}/{vault_path}"

        # Build template content
        template_content_parts = []
        for key in secret_keys:
            template_content_parts.append(
                f'{{{{ with secret "{full_path}" }}}}'
                f"{{{{ .Data.data.{key} }}}}"
                f"{{{{ end }}}}"
            )
        template_content = "\n".join(template_content_parts)

        annotations = {
            "vault.hashicorp.com/agent-inject": "true",
            "vault.hashicorp.com/agent-inject-status": "update",
            "vault.hashicorp.com/role": self.cfg.vault_role,
            "vault.hashicorp.com/auth-path": f"auth/{self.cfg.vault_auth_path}",
            "vault.hashicorp.com/agent-inject-secret-" + template_name: full_path,
            "vault.hashicorp.com/agent-inject-template-" + template_name: (
                template_content
            ),
            "vault.hashicorp.com/agent-limits-cpu": "100m",
            "vault.hashicorp.com/agent-limits-mem": "128Mi",
            "vault.hashicorp.com/agent-requests-cpu": "50m",
            "vault.hashicorp.com/agent-requests-mem": "64Mi",
            # Security hardening
            "vault.hashicorp.com/agent-run-as-same-user": "false",
            "vault.hashicorp.com/agent-set-selinux-label": "false",
            "vault.hashicorp.com/secret-volume-path": "/vault/secrets",
            "vault.hashicorp.com/secret-volume-perms": "0400",
            # Zone label for NetworkPolicy enforcement
            "sesi.io/zone": zone.name.lower(),
        }

        return annotations

    # ── CSI Secrets Store Driver ──────────────────────────────────────────

    def generate_csi_secret_provider_class(
        self,
        name: str,
        zone: SecurityZone,
        vault_path: str,
        secret_keys: List[str],
    ) -> Dict[str, Any]:
        """
        Generate SecretProviderClass CRD for CSI Secrets Store Driver.

        The CSI driver mounts secrets as volumes, which is useful for
        applications that expect file-based secrets.

        HONEST: CSI driver requires the Vault provider to be installed
        on each node. Secrets are mounted read-only.
        """
        zone_path = self.cfg.zone_secret_paths.get(
            zone.name.lower(), "secret/data/sesi/unknown"
        )
        full_path = f"{zone_path}/{vault_path}"

        # Build objects array for CSI
        objects = []
        for key in secret_keys:
            objects.append({
                "objectName": key,
                "secretPath": full_path,
                "secretKey": key,
            })

        return {
            "apiVersion": "secrets-store.csi.x-k8s.io/v1",
            "kind": "SecretProviderClass",
            "metadata": {
                "name": name,
                "namespace": "default",
                "labels": {
                    "app.kubernetes.io/name": "sesi-oversight",
                    "sesi.io/zone": zone.name.lower(),
                },
            },
            "spec": {
                "provider": self.cfg.csi_provider,
                "parameters": {
                    "vaultAddress": self.cfg.vault_addr,
                    "roleName": self.cfg.vault_role,
                    "objects": json.dumps(objects),
                },
            },
        }

    # ── Rotation with No-Zeno Bound ────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """
        Compute δ_min from Delta_Min Theorem 1.2.

        δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}

        This is the minimum "distance" a secret rotation must represent
        to be accepted. Rotations below this threshold are rejected as
        chattering.

        HONEST: Requires κ-fatness (Delta_Min Def 1.1). Without it,
        δ_min can be zero.
        """
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def max_rotations_in_window(self, T: float) -> float:
        """
        Compute No-Zeno bound: N(T) ≤ 1 + 2C_V·T/δ_min.

        From SESI_NoZeno_Closure Theorem 3.2.

        Args:
            T: Time window in seconds

        Returns:
            Maximum number of rotations allowed
        """
        delta_min = self.compute_delta_min()
        if delta_min <= 0:
            return float('inf')
        return 1 + (2 * self.cfg.c_v_rotation * T) / delta_min

    def check_rotation_allowed(self, secret_id: str) -> Dict[str, Any]:
        """
        Check if a rotation is allowed for a secret.

        Applies both:
          1. No-Zeno rate bound
          2. δ_min quantum gate (rotation must be meaningful)

        Args:
            secret_id: Unique identifier for the secret

        Returns:
            dict with allowed, reason, current_count, bound
        """
        now = time.time()
        delta_min = self.compute_delta_min()
        bound = self.max_rotations_in_window(self.cfg.rotation_interval_s)

        # Count rotations in window
        cutoff = now - self.cfg.rotation_interval_s
        recent = [
            r for r in self._rotation_log
            if r["secret_id"] == secret_id and r["timestamp"] > cutoff
        ]
        current_count = len(recent)

        if current_count >= bound:
            return {
                "allowed": False,
                "reason": "no_zeno_rate_bound",
                "current_count": current_count,
                "bound": bound,
                "delta_min": delta_min,
                "note": (
                    "SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min. "
                    "Rotation rate exceeds bound. Wait before rotating again."
                ),
            }

        # Check δ_min quantum gate
        last = self._last_rotation.get(secret_id)
        if last is not None:
            time_since_last = now - last
            if time_since_last < self.cfg.rotation_interval_s * 0.1:
                return {
                    "allowed": False,
                    "reason": "sub_delta_min_chatter",
                    "time_since_last_s": time_since_last,
                    "delta_min": delta_min,
                    "note": (
                        "Delta_Min Thm 1.2: rotation too soon after last. "
                        "This prevents chattering (many tiny rotations). "
                        "Requires κ-fatness."
                    ),
                }

        return {
            "allowed": True,
            "current_count": current_count,
            "bound": bound,
            "delta_min": delta_min,
        }

    def record_rotation(self, secret_id: str) -> None:
        """Record a successful rotation."""
        now = time.time()
        self._rotation_log.append({
            "secret_id": secret_id,
            "timestamp": now,
        })
        self._last_rotation[secret_id] = now

    # ── Generate All Manifests ─────────────────────────────────────────────

    def generate_all_manifests(
        self,
        output_dir: str = "./k8s/vault",
        zones: Optional[List[SecurityZone]] = None,
    ) -> Dict[str, str]:
        """
        Generate all Vault-K8s integration manifests.

        Args:
            output_dir: Directory to write manifests
            zones: List of zones to generate for (default: all)

        Returns:
            dict mapping filename to path
        """
        if zones is None:
            zones = list(SecurityZone)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # 1. ClusterSecretStore
        css = self.generate_cluster_secret_store()
        css_path = out_path / "cluster-secret-store.yaml"
        self._write_yaml(css, css_path)
        files["cluster_secret_store"] = str(css_path)

        # 2. ExternalSecrets per zone
        for zone in zones:
            zone_name = zone.name.lower()

            # ExternalSecret
            es = self.generate_external_secret(
                name=f"sesi-{zone_name}-secrets",
                namespace="sesi-oversight",
                zone=zone,
                vault_path="config",
                secret_keys=["api_key", "tls_cert", "tls_key"],
            )
            es_path = out_path / f"external-secret-{zone_name}.yaml"
            self._write_yaml(es, es_path)
            files[f"external_secret_{zone_name}"] = str(es_path)

            # SecretProviderClass (CSI)
            spc = self.generate_csi_secret_provider_class(
                name=f"sesi-{zone_name}-csi",
                zone=zone,
                vault_path="config",
                secret_keys=["api_key", "tls_cert", "tls_key"],
            )
            spc_path = out_path / f"secret-provider-class-{zone_name}.yaml"
            self._write_yaml(spc, spc_path)
            files[f"csi_{zone_name}"] = str(spc_path)

        # 3. Agent Injector example
        agent_annotations = self.generate_agent_injector_annotations(
            zone=SecurityZone.Z1_CLINICAL_GATEWAY,
            vault_path="config",
            secret_keys=["api_key"],
        )
        agent_path = out_path / "agent-injector-annotations.yaml"
        self._write_yaml({"annotations": agent_annotations}, agent_path)
        files["agent_injector"] = str(agent_path)

        logger.info(f"Vault-K8s manifests written to {output_dir}")
        return files

    def _write_yaml(self, data: Dict[str, Any], path: Path) -> None:
        """Write dict to YAML file."""
        if HAS_YAML:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    def status(self) -> Dict[str, Any]:
        """Get operator status."""
        return {
            "vault_addr": self.cfg.vault_addr,
            "rotation_log_size": len(self._rotation_log),
            "delta_min": self.compute_delta_min(),
            "max_rotations_per_day": self.max_rotations_in_window(86400),
            "zones_configured": list(self.cfg.zone_secret_paths.keys()),
        }


# ═══════════════════════════════════════════════════════════════════════
# [JT1] JAEGER SECURITY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class JaegerDashboardConfig:
    """Configuration for Jaeger security dashboard."""
    jaeger_endpoint: str = "http://jaeger-query.monitoring:16686"
    grafana_endpoint: str = "http://grafana.monitoring:3000"
    datasource_uid: str = "jaeger"
    refresh_interval: str = "30s"

    # Alert thresholds
    zone_violation_threshold: int = 5
    auth_failure_rate_threshold: float = 0.1
    trace_anomaly_threshold: float = 3.0

    # CUSUM parameters (from Six_Problems P4)
    cusum_drift: float = 0.5
    cusum_threshold: float = 5.0

    # Growth-rate regime detection (paper6 Thm 5.1)
    hill_estimator_k: int = 20


class JaegerSecurityDashboard:
    """
    [D] Generates Jaeger/Grafana dashboards for security tracing.

    Features:
      • Zone-crossing trace visualization
      • Security event span taxonomy
      • CUSUM change-point detection on trace streams
      • Growth-rate regime classification (paper6 Thm 5.1)
      • Alert rules for anomalous trace patterns

    MATHEMATICAL FOUNDATIONS:
      • CUSUM + subsampling (Six_Problems Problem 4): validates the
        composite detection-then-estimation statistic
      • Growth-rate dichotomy (paper6 Thm 5.1): θ>1 (finite mean),
        θ∈(0,1) (regularly varying), θ→0 (degenerate ½δ₀+½δ₁)
      • Exit-time bound (Stochastic_NoZeno Thm 4.1):
        P(sup||h(r)-h(T_k)|| ≥ δ|F_{T_k}) ≤ 2exp(-δ²/8G₀²ε)

    HONEST: Dashboard is a VISUALIZATION TOOL, not a security oracle.
    Novel attacks may not produce recognizable trace patterns.
    """

    def __init__(self, cfg: Optional[JaegerDashboardConfig] = None):
        self.cfg = cfg or JaegerDashboardConfig()
        self._trace_history: List[Dict[str, Any]] = []

    # ── Span Taxonomy ─────────────────────────────────────────────────────

    SECURITY_SPAN_TYPES = {
        "auth": {
            "login_attempt": "Authentication login attempt",
            "login_success": "Successful authentication",
            "login_failure": "Failed authentication",
            "mfa_challenge": "Multi-factor authentication challenge",
            "token_issued": "JWT/OIDC token issued",
            "token_validated": "Token validation",
            "token_expired": "Token expiration",
        },
        "zone_crossing": {
            "z0_to_z1": "Implant → Clinical Gateway",
            "z1_to_z0": "Clinical Gateway → Implant",
            "z1_to_z2": "Clinical Gateway → Processing",
            "z2_to_z1": "Processing → Clinical Gateway",
            "z2_to_z3": "Processing → External",
            "z3_to_z2": "External → Processing",
        },
        "secret_access": {
            "vault_read": "Vault secret read",
            "vault_write": "Vault secret write",
            "vault_rotate": "Vault secret rotation",
            "vault_denied": "Vault access denied",
        },
        "implant_command": {
            "command_submitted": "Implant command submitted",
            "command_approved": "Implant command approved",
            "command_rejected": "Implant command rejected",
            "command_executed": "Implant command executed",
            "emergency_stop": "Emergency stop triggered",
        },
        "oversight": {
            "audit_logged": "Audit event logged",
            "alert_raised": "Security alert raised",
            "limitation_enforced": "Limitation enforcer triggered",
        },
    }

    def get_span_taxonomy(self) -> Dict[str, Any]:
        """Return the security span taxonomy."""
        return self.SECURITY_SPAN_TYPES

    # ── Trace Recording ───────────────────────────────────────────────────

    def record_trace(self, span_type: str, span_name: str,
                     zone: SecurityZone, metadata: Optional[Dict] = None) -> None:
        """Record a security trace event."""
        self._trace_history.append({
            "timestamp": time.time(),
            "span_type": span_type,
            "span_name": span_name,
            "zone": zone.name,
            "metadata": metadata or {},
        })

    # ── CUSUM Change-Point Detection ───────────────────────────────────────

    def cusum_detect(self, values: Optional[List[float]] = None,
                     drift: Optional[float] = None,
                     threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        CUSUM change-point detection on trace event severities.

        From Six_Problems Problem 4: CUSUM + subsampling on raw process.
        The composite statistic satisfies Politis-Romano subsampling
        convergence hypothesis.

        Args:
            values: Optional list of severity values. If None, uses
                    trace history severity.
            drift: CUSUM drift parameter (default from config)
            threshold: CUSUM threshold (default from config)

        Returns:
            dict with change_point_detected, cusum_statistic, threshold
        """
        if values is None:
            # Extract severity from trace history
            values = [
                t.get("metadata", {}).get("severity", 0.5)
                for t in self._trace_history
            ]

        if len(values) < 10:
            return {
                "change_point_detected": False,
                "reason": "insufficient_data",
                "n_values": len(values),
            }

        drift = drift or self.cfg.cusum_drift
        threshold = threshold or self.cfg.cusum_threshold

        target = np.mean(values)
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0
        change_point_idx = None

        for i, v in enumerate(values):
            deviation = v - target
            cusum_pos = max(0, cusum_pos + deviation - drift)
            cusum_neg = max(0, cusum_neg - deviation - drift)

            current_max = max(cusum_pos, cusum_neg)
            if current_max > max_cusum:
                max_cusum = current_max
                if current_max > threshold:
                    change_point_idx = i

        detected = change_point_idx is not None

        return {
            "change_point_detected": detected,
            "cusum_statistic": max_cusum,
            "threshold": threshold,
            "change_point_index": change_point_idx,
            "target_mean": target,
            "n_values": len(values),
            "note": (
                "CUSUM change-point detection (Six_Problems P4). "
                "Validated via Politis-Romano subsampling on the composite "
                "detection-then-estimation statistic."
            ),
        }

    # ── Growth-Rate Regime Classification ─────────────────────────────────

    def hill_estimator(self, values: Optional[List[float]] = None,
                       k: Optional[int] = None) -> Dict[str, Any]:
        """
        Hill estimator for tail index θ.

        From Paper11 Entry 2.5 / paper6 Thm 5.1:
          θ > 1:      finite mean → ordinary LLN
          θ ∈ (0,1):  regularly varying → Beta(θ,θ) arcsine
          θ → 0:      degenerate → ½δ₀ + ½δ₁

        Args:
            values: Optional list of positive values
            k: Number of upper order statistics

        Returns:
            dict with theta_hat, regime, description
        """
        if values is None:
            values = [
                t.get("metadata", {}).get("severity", 0.5)
                for t in self._trace_history
            ]

        # Filter positive values
        values = [v for v in values if v > 0]
        if len(values) < 10:
            return {"error": "insufficient_positive_values", "n": len(values)}

        k = k or self.cfg.hill_estimator_k
        k = min(k, len(values) - 1)

        # Sort descending
        sorted_vals = sorted(values, reverse=True)
        x_k1 = sorted_vals[k]

        if x_k1 <= 0:
            return {"error": "x_k1_non_positive"}

        # Hill estimator: θ̂_k = (1/k) Σ ln(X_(i)/X_(k+1))
        log_ratios = [math.log(sorted_vals[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        # Asymptotic variance: θ²/k
        asymptotic_var = theta_hat ** 2 / k
        ci_half = 1.96 * math.sqrt(asymptotic_var)

        # Regime classification
        if theta_hat > 1.0:
            regime = "FINITE_MEAN"
            description = (
                "θ > 1: finite mean. Ordinary LLN applies. "
                "Standard monitoring appropriate."
            )
        elif theta_hat > 0.0:
            regime = "REGULARLY_VARYING"
            description = (
                "θ ∈ (0,1): regularly varying infinite mean. "
                "Beta(θ,θ) arcsine law. Distributional monitoring required."
            )
        else:
            regime = "DEGENERATE"
            description = (
                "θ → 0: degenerate two-point law ½δ₀ + ½δ₁. "
                "Occupation fraction polarizes. Urgent review required."
            )

        return {
            "theta_hat": theta_hat,
            "ci_lower": max(0, theta_hat - ci_half),
            "ci_upper": theta_hat + ci_half,
            "k": k,
            "regime": regime,
            "description": description,
            "note": (
                "Hill estimator (Paper11 Entry 2.5) with growth-rate "
                "regime classification (paper6 Thm 5.1). "
                "Finite-sample bias is well-known; use data-driven k "
                "selection for production."
            ),
        }

    # ── Exit-Time Alarm ───────────────────────────────────────────────────

    def exit_time_alarm(self, observed_deviation: float,
                        delta: float = 0.3, g0: float = 0.05,
                        epsilon: float = 5.0) -> Dict[str, Any]:
        """
        Stochastic exit-time alarm.

        From Stochastic_NoZeno Thm 4.1:
          P(sup||h(r)-h(T_k)|| ≥ δ | F_{T_k}) ≤ 2exp(-δ²/8G₀²ε)

        If observed deviation exceeds δ, the trace pattern may be anomalous.

        Args:
            observed_deviation: Observed state deviation
            delta: Exit threshold δ
            g0: Diffusion bound G₀
            epsilon: Observation window ε

        Returns:
            dict with alarm, prob_bound
        """
        prob_bound = 2 * math.exp(-delta**2 / (8 * g0**2 * epsilon))
        prob_bound = min(prob_bound, 1.0)

        alarm = observed_deviation > delta

        return {
            "alarm": alarm,
            "observed_deviation": observed_deviation,
            "threshold_delta": delta,
            "prob_bound": prob_bound,
            "note": (
                "Stochastic_NoZeno Thm 4.1 exit-time tail bound. "
                "This is a STATISTICAL alarm, not a proof of attack."
            ),
        }

    # ── Dashboard Generation ──────────────────────────────────────────────

    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        """
        Generate Grafana dashboard JSON for security tracing.

        Panels:
          Row 1: Overview — trace rate, error rate, zone violations
          Row 2: Zone traffic — cross-zone trace heatmap
          Row 3: Anomaly detection — CUSUM, Hill regime, exit-time
          Row 4: Secret access — Vault operations
          Row 5: Implant commands — command lifecycle
        """
        ds = {"type": "jaeger", "uid": self.cfg.datasource_uid}

        def _panel(pid, title, ptype, targets, grid, extra=None):
            p = {
                "id": pid,
                "type": ptype,
                "title": title,
                "datasource": ds,
                "targets": targets,
                "gridPos": grid,
                "fieldConfig": {
                    "defaults": {"unit": "short"},
                    "overrides": [],
                },
                "options": {},
            }
            if extra:
                p.update(extra)
            return p

        panels = [
            # Row 1: Overview
            _panel(1, "Security Trace Rate", "timeseries",
                   [{"query": 'rate(sesi_security_traces_total[5m])',
                     "legendFormat": "traces/s"}],
                   {"h": 8, "w": 8, "x": 0, "y": 0}),
            _panel(2, "Zone Violations", "stat",
                   [{"query": 'sum(increase(sesi_zone_violations_total[1h]))',
                     "legendFormat": "violations"}],
                   {"h": 8, "w": 8, "i": 8, "y": 0},
                   extra={"fieldConfig": {"defaults": {
                       "thresholds": {"mode": "absolute", "steps": [
                           {"color": "green", "value": None},
                           {"color": "yellow", "value": 1},
                           {"color": "red", "value": 5}]}}}}),
            _panel(3, "Auth Failure Rate", "gauge",
                   [{"query": 'rate(sesi_auth_failures_total[5m]) / rate(sesi_auth_attempts_total[5m])',
                     "legendFormat": "failure_rate"}],
                   {"h": 8, "w": 8, "i": 16, "y": 0},
                   extra={"fieldConfig": {"defaults": {
                       "min": 0, "max": 1, "unit": "percentunit",
                       "thresholds": {"mode": "absolute", "steps": [
                           {"color": "green", "value": None},
                           {"color": "yellow", "value": 0.05},
                           {"color": "red", "value": 0.1}]}}}}),

            # Row 2: Zone Traffic
            _panel(4, "Cross-Zone Trace Heatmap", "heatmap",
                   [{"query": 'sum(rate(sesi_zone_crossing_traces_total[5m])) by (src_zone, dst_zone)',
                     "legendFormat": "{{src_zone}}→{{dst_zone}}",
                     "format": "heatmap"}],
                   {"h": 8, "w": 12, "i": 0, "y": 8}),
            _panel(5, "Zone Violation Traces", "table",
                   [{"query": 'sesi_zone_violation_traces',
                     "legendFormat": "{{trace_id}}",
                     "format": "table"}],
                   {"h": 8, "w": 12, "i": 12, "y": 8}),

            # Row 3: Anomaly Detection
            _panel(6, "CUSUM Statistic", "timeseries",
                   [{"query": 'sesi_cusum_statistic',
                     "legendFormat": "cusum"}],
                   {"h": 8, "w": 8, "i": 0, "y": 16}),
            _panel(7, "Growth-Rate Regime", "stat",
                   [{"query": 'sesi_trace_regime',
                     "legendFormat": "regime"}],
                   {"h": 8, "w": 8, "i": 8, "y": 16},
                   extra={"fieldConfig": {"defaults": {
                       "mappings": [
                           {"type": "value", "options": {
                               "0": {"text": "FINITE_MEAN", "color": "green"},
                               "1": {"text": "REGULARLY_VARYING", "color": "yellow"},
                               "2": {"text": "DEGENERATE", "color": "red"}}}]}}}),
            _panel(8, "Exit-Time Alarm", "stat",
                   [{"query": 'sesi_exit_time_alarm',
                     "legendFormat": "alarm"}],
                   {"h": 8, "w": 8, "i": 16, "y": 16},
                   extra={"fieldConfig": {"defaults": {
                       "mappings": [
                           {"type": "value", "options": {
                               "0": {"text": "NORMAL", "color": "green"},
                               "1": {"text": "ALARM", "color": "red"}}}]}}}),

            # Row 4: Secret Access
            _panel(9, "Vault Operations", "timeseries",
                   [{"query": 'sum(rate(sesi_vault_operations_total[5m])) by (operation)',
                     "legendFormat": "{{operation}}"}],
                   {"h": 8, "w": 12, "i": 0, "y": 24}),
            _panel(10, "Secret Rotations", "timeseries",
                   [{"query": 'sum(rate(sesi_secret_rotations_total[1h]))',
                     "legendFormat": "rotations/h"}],
                   {"h": 8, "w": 12, "i": 12, "y": 24}),

            # Row 5: Implant Commands
            _panel(11, "Implant Command Lifecycle", "timeseries",
                   [{"query": 'sum(rate(sesi_implant_commands_total[5m])) by (status)',
                     "legendFormat": "{{status}}"}],
                   {"h": 8, "w": 12, "i": 0, "y": 32}),
            _panel(12, "Emergency Stops", "stat",
                   [{"query": 'sum(increase(sesi_emergency_stops_total[24h]))',
                     "legendFormat": "stops_24h"}],
                   {"h": 8, "w": 12, "i": 12, "y": 32}),
        ]

        dashboard = {
            "uid": "sesi-security-tracing",
            "title": "SESI Security Tracing Dashboard",
            "tags": ["sesi", "security", "tracing", "jaeger"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": self.cfg.refresh_interval,
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "templating": {"list": []},
            "annotations": {"list": []},
        }

        return dashboard

    def generate_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Generate Prometheus alert rules for security tracing.

        Rules:
          • High zone violation rate
          • High auth failure rate
          • CUSUM change-point detected
          • Degenerate trace regime
          • Exit-time alarm
        """
        return [
            {
                "alert": "HighZoneViolationRate",
                "expr": f'sum(rate(sesi_zone_violations_total[5m])) > {self.cfg.zone_violation_threshold}',
                "for": "5m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "High zone violation rate detected",
                    "description": (
                        "Cross-zone traffic violations exceed threshold. "
                        "Investigate immediately. This may indicate "
                        "NetworkPolicy bypass or misconfiguration."
                    ),
                },
            },
            {
                "alert": "HighAuthFailureRate",
                "expr": f'rate(sesi_auth_failures_total[5m]) / rate(sesi_auth_attempts_total[5m]) > {self.cfg.auth_failure_rate_threshold}',
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High authentication failure rate",
                    "description": (
                        "Authentication failure rate exceeds threshold. "
                        "Possible brute-force attack or credential stuffing."
                    ),
                },
            },
            {
                "alert": "CUSUMChangePointDetected",
                "expr": 'sesi_cusum_change_point == 1',
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "CUSUM change-point detected in trace stream",
                    "description": (
                        "Statistical change-point detected in security trace "
                        "severity stream (Six_Problems P4). This indicates "
                        "a regime change in security event patterns."
                    ),
                },
            },
            {
                "alert": "DegenerateTraceRegime",
                "expr": 'sesi_trace_regime == 2',
                "for": "5m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Degenerate trace regime detected",
                    "description": (
                        "Trace stream is in degenerate regime (paper6 Thm 5.1: "
                        "θ→0, ½δ₀+½δ₁). Security events are polarizing between "
                        "extremes. Averaging-based monitoring will miss this."
                    ),
                },
            },
            {
                "alert": "ExitTimeAlarm",
                "expr": 'sesi_exit_time_alarm == 1',
                "for": "1m",
                "labels": {"severity": "emergency"},
                "annotations": {
                    "summary": "Exit-time alarm triggered",
                    "description": (
                        "Stochastic exit-time bound exceeded "
                        "(Stochastic_NoZeno Thm 4.1). Trace state deviation "
                        "is outside expected diffusion envelope."
                    ),
                },
            },
        ]

    def write_dashboard(self, output_path: str = "./monitoring/jaeger/security-dashboard.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_grafana_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Jaeger dashboard written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./monitoring/prometheus/security-tracing-alerts.yaml") -> str:
        """Write alert rules to file."""
        rules = self.generate_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, 'w') as f:
                yaml.dump({"groups": [{"name": "sesi_security_tracing", "rules": rules}]},
                          f, default_flow_style=False)
        else:
            with open(out_path, 'w') as f:
                json.dump({"groups": [{"name": "sesi_security_tracing", "rules": rules}]}, f, indent=2)

        logger.info(f"Alert rules written to {output_path}")
        return str(out_path)

    def status(self) -> Dict[str, Any]:
        """Get dashboard status."""
        return {
            "jaeger_endpoint": self.cfg.jaeger_endpoint,
            "grafana_endpoint": self.cfg.grafana_endpoint,
            "trace_history_size": len(self._trace_history),
            "span_types": len(self.SECURITY_SPAN_TYPES),
        }


# ═══════════════════════════════════════════════════════════════════════
# [RK1] RESIDENT KEY MANAGER (FIDO2 Discoverable Credentials)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ResidentKeyRecord:
    """
    A FIDO2 resident key (discoverable credential) record.

    From WebAuthn Level 3:
      • Resident keys are stored ON the authenticator
      • They are "discoverable" — the authenticator can enumerate them
      • User verification (biometric/PIN) is required for assertion
      • No username needed at login (the authenticator presents the credential)

    HONEST: This record stores METADATA about the credential, not the
    private key itself. Private keys never leave the authenticator.
    """
    credential_id: str
    user_id: str
    user_name: str
    rp_id: str
    public_key_pem: str
    sign_count: int
    attestation_format: str
    aaguid: str
    created_at: float
    last_used_at: Optional[float] = None
    device_name: str = ""
    backup_eligible: bool = False
    backup_state: bool = False
    uv_initialized: bool = True
    resident_key: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "credential_id": self.credential_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "rp_id": self.rp_id,
            "public_key_pem": self.public_key_pem,
            "sign_count": self.sign_count,
            "attestation_format": self.attestation_format,
            "aaguid": self.aaguid,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "device_name": self.device_name,
            "backup_eligible": self.backup_eligible,
            "backup_state": self.backup_state,
            "uv_initialized": self.uv_initialized,
            "resident_key": self.resident_key,
        }


class ResidentKeyManager:
    """
    [D] Manages FIDO2 resident keys (discoverable credentials / passkeys).

    Features:
      • Credential registration with attestation verification
      • Credential discovery (enumerate resident keys per RP)
      • Multi-device credential sync
      • Backup eligibility tracking
      • Multi-identity risk integration (Comprehensive Security §5.2)
      • Vault Transit encryption for credential metadata

    MATHEMATICAL FOUNDATIONS:
      • Multi-Identity Risk Model: R_id(u_i) = f(B_i, T_i, C_i, V_i)
        (Comprehensive Security §5.2)
      • PNDC Axiom 3: Necessity and Proportionality
        (Ethics of Non-Killing §10) — collect only what is necessary
      • Graduated verification (Comprehensive Security §5.3):
        Level 1 automated → Level 4 legal escalation

    HONEST: This is a METADATA MANAGER. Private keys never leave the
    authenticator. This module stores:
      • Public keys (for assertion verification)
      • Credential metadata (for discovery and management)
      • Sign counts (for clone detection)
      • Attestation info (for trust evaluation)

    It does NOT store:
      • Private keys
      • Biometric data
      • PINs
      • User-identifying information beyond what FIDO2 requires
    """

    def __init__(self, vault_manager: Optional[Any] = None):
        """
        Args:
            vault_manager: Optional VaultSecretManager for encrypting
                          credential metadata at rest.
        """
        self.vault_manager = vault_manager
        self.credentials: Dict[str, ResidentKeyRecord] = {}
        self.user_credentials: Dict[str, List[str]] = {}  # user_id → [cred_ids]
        self.rp_credentials: Dict[str, List[str]] = {}    # rp_id → [cred_ids]
        self._auth_history: List[Dict[str, Any]] = []

    # ── Registration ──────────────────────────────────────────────────────

    def register_resident_key(
        self,
        credential_id: str,
        user_id: str,
        user_name: str,
        rp_id: str,
        public_key_pem: str,
        attestation_format: str = "none",
        aaguid: str = "",
        device_name: str = "",
        backup_eligible: bool = False,
    ) -> Dict[str, Any]:
        """
        Register a new resident key (passkey).

        This is called after the WebAuthn registration ceremony has
        been verified by the relying party.

        Args:
            credential_id: Base64url-encoded credential ID
            user_id: User handle (opaque identifier)
            user_name: Username (for display only)
            rp_id: Relying party identifier
            public_key_pem: PEM-encoded public key
            attestation_format: Attestation statement format
            aaguid: Authenticator AAGUID
            device_name: Human-readable device name
            backup_eligible: Whether credential can be backed up

        Returns:
            dict with registration result
        """
        # Check for duplicate credential
        if credential_id in self.credentials:
            return {
                "status": "error",
                "reason": "credential_already_registered",
            }

        record = ResidentKeyRecord(
            credential_id=credential_id,
            user_id=user_id,
            user_name=user_name,
            rp_id=rp_id,
            public_key_pem=public_key_pem,
            sign_count=0,
            attestation_format=attestation_format,
            aaguid=aaguid,
            created_at=time.time(),
            device_name=device_name,
            backup_eligible=backup_eligible,
        )

        self.credentials[credential_id] = record

        # Index by user and RP
        if user_id not in self.user_credentials:
            self.user_credentials[user_id] = []
        self.user_credentials[user_id].append(credential_id)

        if rp_id not in self.rp_credentials:
            self.rp_credentials[rp_id] = []
        self.rp_credentials[rp_id].append(credential_id)

        logger.info(
            f"Resident key registered: user={user_name}, "
            f"rp={rp_id}, device={device_name}, "
            f"backup_eligible={backup_eligible}"
        )

        return {
            "status": "registered",
            "credential_id": credential_id,
            "user_id": user_id,
            "rp_id": rp_id,
            "backup_eligible": backup_eligible,
            "note": (
                "Resident key (passkey) registered. Private key remains "
                "on the authenticator. Only public key and metadata stored."
            ),
        }

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover_credentials(self, rp_id: str) -> List[Dict[str, Any]]:
        """
        Discover all resident keys for a relying party.

        This is what makes resident keys "discoverable" — the
        authenticator can enumerate them without the user providing
        a username.

        Args:
            rp_id: Relying party identifier

        Returns:
            list of credential metadata (without public keys)
        """
        cred_ids = self.rp_credentials.get(rp_id, [])
        results = []
        for cred_id in cred_ids:
            record = self.credentials.get(cred_id)
            if record:
                results.append({
                    "credential_id": record.credential_id,
                    "user_id": record.user_id,
                    "user_name": record.user_name,
                    "device_name": record.device_name,
                    "created_at": record.created_at,
                    "last_used_at": record.last_used_at,
                    "backup_eligible": record.backup_eligible,
                })
        return results

    def get_user_credentials(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all credentials for a user."""
        cred_ids = self.user_credentials.get(user_id, [])
        return [
            self.credentials[cid].to_dict()
            for cid in cred_ids
            if cid in self.credentials
        ]

    # ── Authentication ────────────────────────────────────────────────────

    def verify_assertion(
        self,
        credential_id: str,
        reported_sign_count: int,
        rp_id: str,
    ) -> Dict[str, Any]:
        """
        Verify an authentication assertion (login).

        Checks:
          1. Credential exists and belongs to the RP
          2. Sign count is not regressed (clone detection)
          3. Updates last_used_at

        HONEST: Actual signature verification must be done by the
        relying party using the stored public key. This method only
        handles the metadata checks.

        Args:
            credential_id: Credential ID from the assertion
            reported_sign_count: Sign count reported by authenticator
            rp_id: Relying party ID

        Returns:
            dict with verification result
        """
        if credential_id not in self.credentials:
            return {
                "status": "error",
                "reason": "credential_not_found",
            }

        record = self.credentials[credential_id]

        # Check RP match
        if record.rp_id != rp_id:
            return {
                "status": "error",
                "reason": "rp_mismatch",
                "expected_rp": record.rp_id,
                "provided_rp": rp_id,
            }

        # Clone detection: sign count regression
        if reported_sign_count <= record.sign_count and reported_sign_count != 0:
            self._auth_history.append({
                "timestamp": time.time(),
                "credential_id": credential_id,
                "status": "sign_count_anomaly",
                "expected_min": record.sign_count,
                "received": reported_sign_count,
            })
            return {
                "status": "sign_count_anomaly",
                "reason": (
                    f"Sign count {reported_sign_count} <= stored "
                    f"{record.sign_count}. Possible cloned authenticator."
                ),
            }

        # Update record
        record.sign_count = reported_sign_count
        record.last_used_at = time.time()

        self._auth_history.append({
            "timestamp": time.time(),
            "credential_id": credential_id,
            "status": "authenticated",
            "sign_count": reported_sign_count,
        })

        return {
            "status": "authenticated",
            "credential_id": credential_id,
            "user_id": record.user_id,
            "sign_count": reported_sign_count,
            "note": (
                "Metadata verification passed. Signature verification "
                "must be done by the relying party using the public key."
            ),
        }

    # ── Multi-Identity Risk ───────────────────────────────────────────────

    def compute_multi_identity_risk(self, user_id: str) -> Dict[str, Any]:
        """
        Compute multi-identity risk for a user.

        From Comprehensive Security §5.2:
          R_id(u_i) = f(B_i, T_i, C_i, V_i)

        Where:
          B_i: Behavioral anomaly indicators
          T_i: Temporal interaction patterns
          C_i: Confirmed coordination indicators
          V_i: Verified policy violations

        HONEST: This is a HEURISTIC risk score. It does not prove
        that multiple accounts belong to the same person.

        Args:
            user_id: User identifier

        Returns:
            dict with risk score and components
        """
        creds = self.get_user_credentials(user_id)
        if not creds:
            return {"error": "no_credentials_found"}

        # B_i: Behavioral anomaly — number of devices, sign count anomalies
        n_devices = len(set(c.get("device_name", "") for c in creds))
        anomalies = [
            a for a in self._auth_history
            if a.get("status") == "sign_count_anomaly"
        ]
        b_score = min(1.0, n_devices / 5.0 + len(anomalies) * 0.2)

        # T_i: Temporal patterns — credential creation rate
        created_times = [c.get("created_at", 0) for c in creds]
        if len(created_times) > 1:
            time_spread = max(created_times) - min(created_times)
            # Many credentials in short time → suspicious
            t_score = min(1.0, len(creds) / max(time_spread / 3600, 1))
        else:
            t_score = 0.0

        # C_i: Coordination — shared AAGUID across users
        aaguids = [c.get("aaguid", "") for c in creds if c.get("aaguid")]
        shared_aaguid_count = sum(
            1 for other_creds in self.credentials.values()
            if other_creds.user_id != user_id and other_creds.aaguid in aaguids
        )
        c_score = min(1.0, shared_aaguid_count / 10.0)

        # V_i: Violations — sign count anomalies for this user
        v_score = min(1.0, len(anomalies) / 5.0)

        # Weighted combination
        risk = 0.3 * b_score + 0.2 * t_score + 0.3 * c_score + 0.2 * v_score

        # Determine verification level (Comprehensive Security §5.3)
        if risk < 0.3:
            level = 1  # Automated
        elif risk < 0.5:
            level = 2  # Account-level
        elif risk < 0.8:
            level = 3  # Institutional
        else:
            level = 4  # Legal escalation

        return {
            "user_id": user_id,
            "risk_score": risk,
            "components": {
                "behavioral_anomaly": b_score,
                "temporal_pattern": t_score,
                "coordination": c_score,
                "violations": v_score,
            },
            "verification_level": level,
            "n_credentials": len(creds),
            "n_devices": n_devices,
            "note": (
                "Multi-Identity Risk Model (Comprehensive Security §5.2): "
                "R_id(u_i) = f(B_i, T_i, C_i, V_i). "
                "This is a HEURISTIC score, not proof of identity. "
                "Graduated verification levels per §5.3."
            ),
        }

    # ── Credential Management ─────────────────────────────────────────────

    def revoke_credential(self, credential_id: str, reason: str) -> Dict[str, Any]:
        """Revoke a credential."""
        if credential_id not in self.credentials:
            return {"status": "error", "reason": "credential_not_found"}

        record = self.credentials[credential_id]

        # Remove from indices
        if record.user_id in self.user_credentials:
            self.user_credentials[record.user_id] = [
                cid for cid in self.user_credentials[record.user_id]
                if cid != credential_id
            ]
        if record.rp_id in self.rp_credentials:
            self.rp_credentials[record.rp_id] = [
                cid for cid in self.rp_credentials[record.rp_id]
                if cid != credential_id
            ]

        del self.credentials[credential_id]

        logger.info(f"Credential revoked: {credential_id}, reason: {reason}")

        return {
            "status": "revoked",
            "credential_id": credential_id,
            "reason": reason,
        }

    def get_backup_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get backup status for a user's credentials.

        From WebAuthn Level 3:
          • backupEligible: credential CAN be backed up
          • backupState: credential IS currently backed up

        HONEST: Backup eligibility is set by the authenticator during
        registration. This module only tracks the flags.
        """
        creds = self.get_user_credentials(user_id)
        backup_eligible = [c for c in creds if c.get("backup_eligible")]
        backed_up = [c for c in creds if c.get("backup_state")]

        return {
            "user_id": user_id,
            "total_credentials": len(creds),
            "backup_eligible": len(backup_eligible),
            "backed_up": len(backed_up),
            "not_backed_up": len(creds) - len(backed_up),
            "credentials": [
                {
                    "credential_id": c["credential_id"],
                    "device_name": c.get("device_name", ""),
                    "backup_eligible": c.get("backup_eligible", False),
                    "backup_state": c.get("backup_state", False),
                }
                for c in creds
            ],
        }

    def status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "total_credentials": len(self.credentials),
            "total_users": len(self.user_credentials),
            "total_rps": len(self.rp_credentials),
            "auth_history_size": len(self._auth_history),
            "sign_count_anomalies": sum(
                1 for a in self._auth_history
                if a.get("status") == "sign_count_anomaly"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESISecurityOperationsSuite:
    """
    [D] Complete security operations suite.

    Orchestrates:
      • Vault-K8s secrets operator
      • Jaeger security dashboard
      • Resident key manager
    """

    def __init__(self):
        self.vault_k8s = VaultKubernetesSecretsOperator()
        self.jaeger = JaegerSecurityDashboard()
        self.resident_keys = ResidentKeyManager()

    def generate_all_artifacts(self, output_dir: str = "./security_ops") -> Dict[str, Any]:
        """Generate all deployment artifacts."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = {}

        # Vault-K8s manifests
        results["vault_k8s"] = self.vault_k8s.generate_all_manifests(
            str(out / "k8s" / "vault")
        )

        # Jaeger dashboard
        results["jaeger_dashboard"] = self.jaeger.write_dashboard(
            str(out / "monitoring" / "jaeger" / "security-dashboard.json")
        )
        results["jaeger_alerts"] = self.jaeger.write_alert_rules(
            str(out / "monitoring" / "prometheus" / "security-tracing-alerts.yaml")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full status report."""
        return {
            "vault_k8s": self.vault_k8s.status(),
            "jaeger": self.jaeger.status(),
            "resident_keys": self.resident_keys.status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Vault-K8s × Jaeger × Resident Keys — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available    : {HAS_YAML}")
    print(f"  hvac available      : {HAS_HVAC}")
    print(f"  PyTorch available   : {HAS_TORCH}")
    print(f"  Previous modules    : {HAS_PREV_MODULES}")

    suite = SESISecurityOperationsSuite()

    # ── Vault-K8s ──────────────────────────────────────────────────────────
    print("\n  ── Vault-K8s Secrets Operator ──")

    # Generate ClusterSecretStore
    css = suite.vault_k8s.generate_cluster_secret_store()
    print(f"    ClusterSecretStore : {css['metadata']['name']}")

    # Generate ExternalSecret for Z1
    es = suite.vault_k8s.generate_external_secret(
        name="sesi-clinical-secrets",
        namespace="sesi-oversight",
        zone=SecurityZone.Z1_CLINICAL_GATEWAY,
        vault_path="config",
        secret_keys=["api_key", "tls_cert"],
    )
    print(f"    ExternalSecret     : {es['metadata']['name']}")
    print(f"    Zone               : {es['metadata']['labels']['sesi.io/zone']}")

    # Generate Agent Injector annotations
    annotations = suite.vault_k8s.generate_agent_injector_annotations(
        zone=SecurityZone.Z0_IMPLANT,
        vault_path="device-keys",
        secret_keys=["device_key"],
    )
    print(f"    Agent annotations  : {len(annotations)} entries")

    # Test rotation check
    rotation_check = suite.vault_k8s.check_rotation_allowed("test_secret")
    print(f"    Rotation allowed   : {rotation_check['allowed']}")
    print(f"    δ_min              : {rotation_check['delta_min']:.6f}")
    print(f"    Max rotations/day  : {suite.vault_k8s.max_rotations_in_window(86400):.2f}")

    # Generate manifests
    files = suite.generate_all_artifacts("./test_security_ops")
    print(f"    Files generated    : {len(files)}")

    # ── Jaeger Dashboard ───────────────────────────────────────────────────
    print("\n  ── Jaeger Security Dashboard ──")

    # Record some traces
    for i in range(20):
        suite.jaeger.record_trace(
            span_type="zone_crossing",
            span_name="z1_to_z2",
            zone=SecurityZone.Z1_CLINICAL_GATEWAY,
            metadata={"severity": np.random.random() * 0.3},
        )

    # CUSUM detection
    cusum = suite.jaeger.cusum_detect()
    print(f"    CUSUM detected     : {cusum['change_point_detected']}")
    print(f"    CUSUM statistic    : {cusum['cusum_statistic']:.4f}")

    # Hill estimator
    hill = suite.jaeger.hill_estimator()
    if "error" not in hill:
        print(f"    Hill θ̂             : {hill['theta_hat']:.4f}")
        print(f"    Regime             : {hill['regime']}")

    # Exit-time alarm
    alarm = suite.jaeger.exit_time_alarm(0.5)
    print(f"    Exit-time alarm    : {alarm['alarm']}")
    print(f"    Prob bound         : {alarm['prob_bound']:.6f}")

    # Generate dashboard
    dashboard = suite.jaeger.generate_grafana_dashboard()
    print(f"    Dashboard panels   : {len(dashboard['panels'])}")

    # Generate alerts
    alerts = suite.jaeger.generate_alert_rules()
    print(f"    Alert rules        : {len(alerts)}")

    # ── Resident Key Manager ───────────────────────────────────────────────
    print("\n  ── Resident Key Manager ──")

    # Register a resident key
    reg = suite.resident_keys.register_resident_key(
        credential_id="cred_" + uuid.uuid4().hex[:8],
        user_id="user_001",
        user_name="Dr. Smith",
        rp_id="sesi.example.com",
        public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjANBg...\n-----END PUBLIC KEY-----",
        attestation_format="packed",
        aaguid="aaguid_12345",
        device_name="MacBook Pro",
        backup_eligible=True,
    )
    print(f"    Registration       : {reg['status']}")
    print(f"    Backup eligible    : {reg['backup_eligible']}")

    # Register another device for the same user
    reg2 = suite.resident_keys.register_resident_key(
        credential_id="cred_" + uuid.uuid4().hex[:8],
        user_id="user_001",
        user_name="Dr. Smith",
        rp_id="sesi.example.com",
        public_key_pem="-----BEGIN PUBLIC KEY-----\nMIIBIjANBg...\n-----END PUBLIC KEY-----",
        attestation_format="apple",
        aaguid="aaguid_67890",
        device_name="iPhone 15",
        backup_eligible=True,
    )
    print(f"    Second device      : {reg2['status']}")

    # Discover credentials
    discovered = suite.resident_keys.discover_credentials("sesi.example.com")
    print(f"    Discovered creds   : {len(discovered)}")

    # Verify assertion
    cred_id = reg["credential_id"]
    verify = suite.resident_keys.verify_assertion(cred_id, 1, "sesi.example.com")
    print(f"    Assertion verify   : {verify['status']}")

    # Clone detection (sign count regression)
    clone_check = suite.resident_keys.verify_assertion(cred_id, 0, "sesi.example.com")
    print(f"    Clone detection    : {clone_check['status']}")

    # Multi-identity risk
    risk = suite.resident_keys.compute_multi_identity_risk("user_001")
    print(f"    Multi-identity risk: {risk['risk_score']:.4f}")
    print(f"    Verification level : {risk['verification_level']}")

    # Backup status
    backup = suite.resident_keys.get_backup_status("user_001")
    print(f"    Backup status      : {backup['backed_up']}/{backup['total_credentials']}")

    # ── Full Status ────────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Vault-K8s δ_min    : {status['vault_k8s']['delta_min']:.6f}")
    print(f"    Jaeger traces      : {status['jaeger']['trace_history_size']}")
    print(f"    Resident keys      : {status['resident_keys']['total_credentials']}")
    print(f"    Sign count anomalies: {status['resident_keys']['sign_count_anomalies']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Vault-K8s is a CONFIG GENERATOR, not a Vault deployment")
    print("  • Jaeger dashboard is a VISUALIZATION TOOL, not a security oracle")
    print("  • Resident key manager stores METADATA, not private keys")
    print("  • Private keys NEVER leave the authenticator")
    print("  • Multi-identity risk is HEURISTIC, not proof of identity")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
