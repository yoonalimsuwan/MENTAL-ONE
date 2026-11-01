"""
=============================================================================
SESI VAULT ENTERPRISE NAMESPACES × FIDO2 MULTI-IDENTITY RISK ×
CHAOS PIPELINE GRAFANA DASHBOARD
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[VN1] VaultEnterpriseNamespaceManager
      • Namespace hierarchy: root → z0-implant / z1-clinical /
        z2-processing / z3-external
      • Per-namespace secrets engines (KV v2, Transit, Database, PKI)
      • Sentinel EGP/RGP policies per zone
      • Audit device configuration per namespace
      • Namespace-level access control (deny-by-default cross-zone)

[FI1] FIDO2MultiIdentityRiskBridge
      • Maps FIDO2 attestation data to Comprehensive Security §5.2:
        R_id(u_i) = f(B_i, T_i, C_i, V_i)
      • B_i: sign count anomalies, unusual usage patterns
      • T_i: registration timing, authentication frequency
      • C_i: shared AAGUIDs across accounts, shared attestation roots
      • V_i: revoked credentials, attestation failures
      • Graduated verification (4 levels) per §5.3
      • FIDO MDS trust anchor integration

[GD1] ChaosPipelineGrafanaDashboard
      • Grafana dashboard JSON for chaos pipeline monitoring
      • Panels: scenario execution, quality gate, Hill θ̂, CUSUM,
        degenerate regime, δ_min CI, No-Zeno bound, FIDO2 identity risk
      • Prometheus alert rules for regime detection
      • Template variables for filtering

MATHEMATICAL FOUNDATIONS
--------------------------
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i, T_i, C_i, V_i)
[C] Comprehensive Security §5.3: Graduated verification (4 levels)
[C] Comprehensive Security §8: R_t = Σ w_i·R_i unified risk model
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
[C] Paper 9 Entry 2.5: Hill estimator + domain-of-attraction test
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
[C] paper6 Thm 4.1: Degenerate occupation-time law ½δ₀+½δ₁
[C] Unified_Ordinal Hyp 7.9: Free design parameters
[C] Dual-Model §12: No mathematical definition demonstrates consciousness

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_vault_fido_chaos_cicd.py — Vault auto-unseal, FIDO2 server,
    chaos CI/CD pipeline
  • ses_terraform_fido_runbook.py — Terraform KMS, FIDO MDS, runbook
  • agi_one_v3_11.py — AGIONE, EcosystemOrchestrator
  • agi_one_psyche_plus_v33.py — Psyche Plus, quality_score gating

HONEST SCOPE
------------
• Vault Enterprise namespaces require Vault Enterprise license.
  Community edition does not support namespaces.
• Sentinel policies require Vault Enterprise. Community edition uses
  HCL policies only.
• FIDO2 multi-identity risk is a HEURISTIC composite score, not proof
  of identity fraud. Graduated verification protects privacy.
• Grafana dashboards are VISUALIZATION TEMPLATES. They display data
  but do not make security decisions.
• All thresholds, weights, and namespace configurations are FREE
  DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• This module does NOT create, detect, or simulate consciousness.
  The AGIONE ecosystem it protects is a computational architecture,
  not a conscious entity (Dual-Model §12).
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

logger = logging.getLogger("sesi.vault_ns_fido_grafana")

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

__all__ = [
    "SecurityZone", "VaultNamespaceConfig",
    "VaultEnterpriseNamespaceManager",
    "FIDO2MultiIdentityRiskBridge", "IdentityRiskComponents",
    "VerificationLevel",
    "ChaosPipelineGrafanaDashboard",
    "SESIVaultFIDOChaosGrafanaSuite",
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


class VerificationLevel(Enum):
    """
    [C] Graduated verification levels from Comprehensive Security §5.3.

    Level 1: Automated abuse detection (no real-world identity)
    Level 2: Account-level verification (when risk justifies)
    Level 3: Institutional verification (for restricted resources)
    Level 4: Legal escalation (credible evidence of serious misuse)
    """
    AUTOMATED = 1
    ACCOUNT = 2
    INSTITUTIONAL = 3
    LEGAL = 4


@dataclass
class IdentityRiskComponents:
    """
    [C] Components of multi-identity risk from Comprehensive Security §5.2.

    R_id(u_i) = f(B_i, T_i, C_i, V_i)

    Where:
      B_i: Behavioral anomaly indicators
      T_i: Temporal interaction patterns
      C_i: Confirmed coordination indicators
      V_i: Verified policy violations
    """
    behavioral: float = 0.0    # B_i: sign count anomalies, unusual patterns
    temporal: float = 0.0      # T_i: registration timing, auth frequency
    coordination: float = 0.0  # C_i: shared AAGUIDs, shared attestation roots
    violations: float = 0.0    # V_i: revoked credentials, attestation failures

    def weighted_risk(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted risk score.

        Default weights from Comprehensive Security §5.2 discussion:
          B: 0.3, T: 0.2, C: 0.3, V: 0.2

        HONEST: These weights are FREE DESIGN PARAMETERS
        (Unified_Ordinal Hyp 7.9). Must be validated against
        real multi-identity abuse data.
        """
        if weights is None:
            weights = {"B": 0.3, "T": 0.2, "C": 0.3, "V": 0.2}
        return (
            weights["B"] * self.behavioral +
            weights["T"] * self.temporal +
            weights["C"] * self.coordination +
            weights["V"] * self.violations
        )


# ═══════════════════════════════════════════════════════════════════════
# [VN1] VAULT ENTERPRISE NAMESPACE MANAGER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VaultNamespaceConfig:
    """
    Configuration for a single Vault Enterprise namespace.

    HONEST: Namespaces require Vault Enterprise. Community edition
    does not support namespaces. This configuration generates the
    API calls that WOULD create namespaces, but they will fail
    on Community edition.
    """
    path: str
    description: str
    zone: SecurityZone

    # Secrets engines to enable
    enable_kv_v2: bool = True
    enable_transit: bool = True
    enable_database: bool = False
    enable_pki: bool = False

    # KV v2 configuration
    kv_max_versions: int = 10
    kv_delete_version_after: str = "30d"

    # Transit configuration
    transit_key_type: str = "aes256-gcm96"
    transit_allow_export: bool = False

    # Sentinel policy enforcement
    sentinel_enforcement_level: str = "enforcing"

    # Audit
    audit_enabled: bool = True
    audit_log_path: str = ""


class VaultEnterpriseNamespaceManager:
    """
    [D] Manages Vault Enterprise namespace hierarchy for SESI zones.

    Namespace hierarchy:
      root/
      ├── z0-implant/          (most restricted)
      ├── z1-clinical/         (clinician access, MFA required)
      ├── z2-processing/       (AI inference, read-only clinical)
      └── z3-external/         (public API, deny all internal)

    Features:
      • Namespace creation with zone-specific configuration
      • Per-namespace secrets engines (KV v2, Transit, Database, PKI)
      • Sentinel EGP/RGP policy generation
      • Audit device configuration
      • Cross-zone access control (deny-by-default)

    HONEST: This is a CONFIGURATION GENERATOR. It produces the API
    calls and policy documents needed to configure Vault Enterprise.
    It does NOT:
      • Deploy or configure a Vault server
      • Manage TLS certificates
      • Handle IAM/service account authentication
      • Guarantee security of the resulting configuration

    Real deployment requires:
      • Vault Enterprise license
      • Proper auto-unseal configuration (KMS/HSM)
      • TLS certificates for all listeners
      • Security audit before production use
    """

    def __init__(self, vault_addr: str = "https://vault.sesi.example.com:8200"):
        self.vault_addr = vault_addr
        self.namespaces: Dict[str, VaultNamespaceConfig] = {}
        self._client = None

    def _get_client(self):
        """Get hvac client (lazy initialization)."""
        if self._client is None and HAS_HVAC:
            self._client = hvac.Client(url=self.vault_addr)
        return self._client

    def configure_default_namespaces(self) -> None:
        """
        Configure the default 4-zone namespace hierarchy.

        Zone configuration follows SESI network segmentation:
          Z0 (implant): Most restricted, no delete, max_versions=1
          Z1 (clinical): Clinician access, MFA required, dynamic DB creds
          Z2 (processing): AI inference, read-only clinical access
          Z3 (external): Public API, deny all internal access
        """
        # Z0: Implant — most restricted
        self.namespaces["z0-implant"] = VaultNamespaceConfig(
            path="z0-implant",
            description="Implant device secrets — most restricted zone",
            zone=SecurityZone.Z0_IMPLANT,
            enable_kv_v2=True,
            enable_transit=True,
            enable_database=False,
            enable_pki=False,
            kv_max_versions=1,          # Only latest version
            kv_delete_version_after="0s",  # Never delete (audit)
            transit_key_type="aes256-gcm96",
            transit_allow_export=False,  # Keys never leave Vault
            sentinel_enforcement_level="enforcing",
            audit_enabled=True,
            audit_log_path="/var/log/vault/z0-implant-audit.log",
        )

        # Z1: Clinical Gateway — clinician access
        self.namespaces["z1-clinical"] = VaultNamespaceConfig(
            path="z1-clinical",
            description="Clinical gateway secrets — MFA required",
            zone=SecurityZone.Z1_CLINICAL_GATEWAY,
            enable_kv_v2=True,
            enable_transit=True,
            enable_database=True,   # Dynamic DB credentials
            enable_pki=True,        # mTLS certificates
            kv_max_versions=5,
            kv_delete_version_after="90d",
            transit_key_type="aes256-gcm96",
            transit_allow_export=False,
            sentinel_enforcement_level="enforcing",
            audit_enabled=True,
            audit_log_path="/var/log/vault/z1-clinical-audit.log",
        )

        # Z2: Processing — AI inference
        self.namespaces["z2-processing"] = VaultNamespaceConfig(
            path="z2-processing",
            description="AI processing secrets — read-only clinical access",
            zone=SecurityZone.Z2_PROCESSING,
            enable_kv_v2=True,
            enable_transit=True,
            enable_database=True,
            enable_pki=False,
            kv_max_versions=10,
            kv_delete_version_after="30d",
            transit_key_type="aes256-gcm96",
            transit_allow_export=False,
            sentinel_enforcement_level="enforcing",
            audit_enabled=True,
            audit_log_path="/var/log/vault/z2-processing-audit.log",
        )

        # Z3: External — public API
        self.namespaces["z3-external"] = VaultNamespaceConfig(
            path="z3-external",
            description="External API secrets — deny all internal access",
            zone=SecurityZone.Z3_EXTERNAL,
            enable_kv_v2=True,
            enable_transit=False,  # No encryption keys in external zone
            enable_database=False,
            enable_pki=False,
            kv_max_versions=3,
            kv_delete_version_after="7d",
            sentinel_enforcement_level="enforcing",
            audit_enabled=True,
            audit_log_path="/var/log/vault/z3-external-audit.log",
        )

        logger.info(f"Configured {len(self.namespaces)} Vault namespaces")

    def generate_namespace_creation_commands(self) -> List[Dict[str, Any]]:
        """
        Generate Vault API commands to create namespaces.

        HONEST: These are API call specifications, not executed commands.
        Real execution requires authenticated Vault client with
        namespace creation privileges.
        """
        commands = []
        for name, cfg in self.namespaces.items():
            commands.append({
                "operation": "create_namespace",
                "method": "POST",
                "path": f"/v1/sys/namespaces/{cfg.path}",
                "payload": {
                    "custom_metadata": {
                        "zone": cfg.zone.name,
                        "description": cfg.description,
                        "sentinel_enforcement": cfg.sentinel_enforcement_level,
                    },
                },
                "note": (
                    f"Creates namespace '{cfg.path}' for zone {cfg.zone.name}. "
                    f"Requires Vault Enterprise and root/sudo token."
                ),
            })
        return commands

    def generate_secrets_engine_commands(self) -> List[Dict[str, Any]]:
        """Generate commands to enable secrets engines per namespace."""
        commands = []
        for name, cfg in self.namespaces.items():
            ns = cfg.path

            if cfg.enable_kv_v2:
                commands.append({
                    "operation": "enable_kv_v2",
                    "namespace": ns,
                    "method": "POST",
                    "path": f"/v1/sys/mounts/secret",
                    "payload": {
                        "type": "kv",
                        "options": {"version": "2"},
                        "config": {
                            "max_versions": cfg.kv_max_versions,
                            "delete_version_after": cfg.kv_delete_version_after,
                        },
                    },
                })

            if cfg.enable_transit:
                commands.append({
                    "operation": "enable_transit",
                    "namespace": ns,
                    "method": "POST",
                    "path": f"/v1/sys/mounts/transit",
                    "payload": {
                        "type": "transit",
                        "description": f"Transit encryption for {ns}",
                    },
                })
                # Create transit key
                commands.append({
                    "operation": "create_transit_key",
                    "namespace": ns,
                    "method": "POST",
                    "path": f"/v1/transit/keys/sesi-{ns}",
                    "payload": {
                        "type": cfg.transit_key_type,
                        "allow_export": cfg.transit_allow_export,
                    },
                })

            if cfg.enable_database:
                commands.append({
                    "operation": "enable_database",
                    "namespace": ns,
                    "method": "POST",
                    "path": f"/v1/sys/mounts/database",
                    "payload": {
                        "type": "database",
                        "description": f"Dynamic DB credentials for {ns}",
                    },
                })

            if cfg.enable_pki:
                commands.append({
                    "operation": "enable_pki",
                    "namespace": ns,
                    "method": "POST",
                    "path": f"/v1/sys/mounts/pki",
                    "payload": {
                        "type": "pki",
                        "config": {"max_lease_ttl": "87600h"},
                    },
                })

        return commands

    def generate_sentinel_policies(self) -> Dict[str, str]:
        """
        Generate Sentinel policies for each namespace.

        HONEST: Sentinel policies require Vault Enterprise. These are
        TEMPLATE policies that must be reviewed by security team.
        """
        policies = {}

        # Z0 Implant: No delete, no list from outside Z1
        policies["z0-implant-egp"] = '''
# SESI Z0 Implant — Sentinel EGP (Endpoint Governing Policy)
# HONEST: Template only. Review before production deployment.

import "strings"
import "regexp"

# Deny delete operations on implant secrets
deny_delete = rule when request.operation == "delete" {
    false
}

# Only allow access from Z1 clinical gateway namespace
allow_clinical_only = rule when request.operation in ["read", "update"] {
    strings.has_prefix(request.namespace.path, "z1-clinical")
}

# Deny all cross-zone access except from Z1
deny_cross_zone = rule when not allow_clinical_only {
    false
}

main = rule {
    deny_delete and deny_cross_zone
}
'''

        # Z1 Clinical: MFA required, no access from Z3
        policies["z1-clinical-egp"] = '''
# SESI Z1 Clinical Gateway — Sentinel EGP
# HONEST: Template only. Review before production deployment.

import "strings"

# Require MFA for all write operations
require_mfa = rule when request.operation in ["create", "update", "delete"] {
    request.auth.metadata.mfa_verified == "true"
}

# Deny access from Z3 external zone
deny_external = rule when strings.has_prefix(request.namespace.path, "z3-external") {
    false
}

main = rule {
    require_mfa and deny_external
}
'''

        # Z2 Processing: Read-only clinical access
        policies["z2-processing-egp"] = '''
# SESI Z2 Processing — Sentinel EGP
# HONEST: Template only. Review before production deployment.

import "strings"

# Allow read-only access to clinical secrets
allow_clinical_read = rule when
    strings.has_prefix(request.path, "secret/data/sesi/clinical") and
    request.operation == "read" {
    true
}

# Deny write access to clinical secrets
deny_clinical_write = rule when
    strings.has_prefix(request.path, "secret/data/sesi/clinical") and
    request.operation in ["create", "update", "delete"] {
    false
}

main = rule {
    allow_clinical_read or deny_clinical_write
}
'''

        # Z3 External: Deny all internal access
        policies["z3-external-egp"] = '''
# SESI Z3 External — Sentinel EGP
# HONEST: Template only. Review before production deployment.

import "strings"

# Deny all access to internal zone secrets
deny_internal = rule when
    strings.has_prefix(request.path, "secret/data/sesi/implant") or
    strings.has_prefix(request.path, "secret/data/sesi/clinical") or
    strings.has_prefix(request.path, "secret/data/sesi/processing") {
    false
}

main = rule {
    deny_internal
}
'''

        return policies

    def generate_audit_config(self) -> List[Dict[str, Any]]:
        """Generate audit device configuration per namespace."""
        configs = []
        for name, cfg in self.namespaces.items():
            if cfg.audit_enabled:
                configs.append({
                    "namespace": cfg.path,
                    "operation": "enable_audit_device",
                    "method": "POST",
                    "path": "/v1/sys/audit/audit-file",
                    "payload": {
                        "type": "file",
                        "options": {
                            "file_path": cfg.audit_log_path or f"/var/log/vault/{cfg.path}-audit.log",
                            "log_raw": False,  # Never log raw secrets
                        },
                    },
                    "note": (
                        "Audit logging enabled. log_raw=false ensures "
                        "secret values are never written to audit log."
                    ),
                })
        return configs

    def generate_all_manifests(self, output_dir: str = "./vault_namespaces") -> Dict[str, str]:
        """Generate all namespace configuration files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Namespace creation commands
        ns_commands = self.generate_namespace_creation_commands()
        ns_path = out_path / "namespace_creation.json"
        with open(ns_path, "w") as f:
            json.dump(ns_commands, f, indent=2)
        files["namespace_creation"] = str(ns_path)

        # Secrets engine commands
        se_commands = self.generate_secrets_engine_commands()
        se_path = out_path / "secrets_engines.json"
        with open(se_path, "w") as f:
            json.dump(se_commands, f, indent=2)
        files["secrets_engines"] = str(se_path)

        # Sentinel policies
        sentinel_policies = self.generate_sentinel_policies()
        for policy_name, policy_content in sentinel_policies.items():
            policy_path = out_path / f"{policy_name}.sentinel"
            with open(policy_path, "w") as f:
                f.write(policy_content)
            files[policy_name] = str(policy_path)

        # Audit configuration
        audit_configs = self.generate_audit_config()
        audit_path = out_path / "audit_config.json"
        with open(audit_path, "w") as f:
            json.dump(audit_configs, f, indent=2)
        files["audit_config"] = str(audit_path)

        # Namespace hierarchy diagram
        hierarchy = {
            "root": {
                "z0-implant": {
                    "zone": "Z0_IMPLANT",
                    "secrets_engines": ["kv-v2", "transit"],
                    "restrictions": ["no_delete", "max_versions=1", "no_export"],
                    "access": "Z1 only",
                },
                "z1-clinical": {
                    "zone": "Z1_CLINICAL_GATEWAY",
                    "secrets_engines": ["kv-v2", "transit", "database", "pki"],
                    "restrictions": ["mfa_required", "no_external_access"],
                    "access": "Clinicians with MFA",
                },
                "z2-processing": {
                    "zone": "Z2_PROCESSING",
                    "secrets_engines": ["kv-v2", "transit", "database"],
                    "restrictions": ["read_only_clinical"],
                    "access": "AI inference services",
                },
                "z3-external": {
                    "zone": "Z3_EXTERNAL",
                    "secrets_engines": ["kv-v2"],
                    "restrictions": ["deny_all_internal"],
                    "access": "Public API only",
                },
            },
        }
        hierarchy_path = out_path / "namespace_hierarchy.json"
        with open(hierarchy_path, "w") as f:
            json.dump(hierarchy, f, indent=2)
        files["namespace_hierarchy"] = str(hierarchy_path)

        logger.info(f"Vault namespace manifests written to {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get namespace manager status."""
        return {
            "vault_addr": self.vault_addr,
            "namespaces_configured": len(self.namespaces),
            "namespace_names": list(self.namespaces.keys()),
            "hvac_available": HAS_HVAC,
            "note": (
                "Vault Enterprise namespaces require Enterprise license. "
                "Community edition does not support namespaces."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [FI1] FIDO2 MULTI-IDENTITY RISK BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class FIDO2MultiIdentityRiskBridge:
    """
    [D] Bridges FIDO2 attestation data to the Multi-Identity Risk Model
    from Comprehensive Security paper §5.2.

    R_id(u_i) = f(B_i, T_i, C_i, V_i)

    Where:
      B_i: Behavioral anomaly indicators
        → FIDO2 sign count anomalies (clone detection)
        → Unusual authentication patterns
        → Rapid credential creation/deletion

      T_i: Temporal interaction patterns
        → Registration timing (many accounts in short window)
        → Authentication frequency anomalies
        → Time-of-day pattern deviations

      C_i: Confirmed coordination indicators
        → Shared AAGUIDs across accounts
        → Shared attestation root certificates
        → Coordinated registration patterns

      V_i: Verified policy violations
        → Revoked credentials
        → Attestation failures
        → Trust anchor mismatches

    Graduated Verification (Comprehensive Security §5.3):
      Level 1: Automated abuse detection (no real-world identity)
      Level 2: Account-level verification (when risk justifies)
      Level 3: Institutional verification (for restricted resources)
      Level 4: Legal escalation (credible evidence of serious misuse)

    HONEST: This is a HEURISTIC risk scoring system. It does NOT prove
    that multiple accounts belong to the same person. It identifies
    PATTERNS that warrant further investigation.

    Privacy Protection (§5.3):
      • No unnecessary biometric collection
      • No personal contact harvesting
      • No unrelated social-network data
      • Appeal mechanism for incorrectly flagged users
    """

    def __init__(self):
        self.account_profiles: Dict[str, Dict[str, Any]] = {}
        self.aaguid_registry: Dict[str, Set[str]] = {}  # aaguid → account_ids
        self.attestation_roots: Dict[str, Set[str]] = {}  # root_cert → account_ids
        self.risk_history: List[Dict[str, Any]] = []

    def register_account(
        self,
        account_id: str,
        aaguid: str,
        attestation_format: str,
        attestation_root: Optional[str] = None,
        registration_time: Optional[float] = None,
    ) -> None:
        """
        Register an account for multi-identity tracking.

        Args:
            account_id: Unique account identifier
            aaguid: FIDO2 Authenticator Attestation GUID
            attestation_format: Attestation format (packed, tpm, etc.)
            attestation_root: Root certificate hash (if available)
            registration_time: Registration timestamp
        """
        if registration_time is None:
            registration_time = time.time()

        self.account_profiles[account_id] = {
            "account_id": account_id,
            "aaguid": aaguid,
            "attestation_format": attestation_format,
            "attestation_root": attestation_root,
            "registration_time": registration_time,
            "sign_count": 0,
            "auth_count": 0,
            "last_auth_time": None,
            "auth_times": [],
            "risk_components": IdentityRiskComponents(),
            "verification_level": VerificationLevel.AUTOMATED,
            "revoked": False,
        }

        # Track AAGUID sharing
        if aaguid not in self.aaguid_registry:
            self.aaguid_registry[aaguid] = set()
        self.aaguid_registry[aaguid].add(account_id)

        # Track attestation root sharing
        if attestation_root:
            if attestation_root not in self.attestation_roots:
                self.attestation_roots[attestation_root] = set()
            self.attestation_roots[attestation_root].add(account_id)

    def record_authentication(
        self,
        account_id: str,
        sign_count: int,
        auth_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Record an authentication event and update risk components.

        Args:
            account_id: Account identifier
            sign_count: FIDO2 sign count from authenticator
            auth_time: Authentication timestamp

        Returns:
            dict with risk assessment and any anomalies detected
        """
        if account_id not in self.account_profiles:
            return {"error": "unknown_account", "account_id": account_id}

        profile = self.account_profiles[account_id]
        if auth_time is None:
            auth_time = time.time()

        anomalies = []

        # ── B_i: Behavioral anomaly detection ─────────────────────────────
        # Sign count regression → possible cloned authenticator
        if sign_count <= profile["sign_count"] and sign_count != 0:
            anomalies.append({
                "type": "sign_count_regression",
                "expected_min": profile["sign_count"],
                "received": sign_count,
                "severity": "HIGH",
                "note": (
                    "Sign count regression detected. Possible cloned "
                    "authenticator. From WebAuthn spec: sign count must "
                    "increase monotonically."
                ),
            })
            profile["risk_components"].behavioral = min(
                1.0, profile["risk_components"].behavioral + 0.3
            )

        # Unusual authentication frequency
        profile["auth_times"].append(auth_time)
        if len(profile["auth_times"]) > 10:
            recent_times = profile["auth_times"][-10:]
            intervals = [
                recent_times[i+1] - recent_times[i]
                for i in range(len(recent_times) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval < 60:  # Less than 1 minute between auths
                anomalies.append({
                    "type": "rapid_authentication",
                    "avg_interval_s": avg_interval,
                    "severity": "MEDIUM",
                })
                profile["risk_components"].behavioral = min(
                    1.0, profile["risk_components"].behavioral + 0.2
                )

        # Update profile
        profile["sign_count"] = sign_count
        profile["auth_count"] += 1
        profile["last_auth_time"] = auth_time

        # ── T_i: Temporal pattern analysis ────────────────────────────────
        # Registration timing: many accounts in short window
        registration_time = profile["registration_time"]
        recent_registrations = sum(
            1 for p in self.account_profiles.values()
            if abs(p["registration_time"] - registration_time) < 3600  # 1 hour
        )
        if recent_registrations > 5:
            profile["risk_components"].temporal = min(
                1.0, profile["risk_components"].temporal + 0.2
            )
            anomalies.append({
                "type": "registration_burst",
                "count_in_window": recent_registrations,
                "window_s": 3600,
                "severity": "MEDIUM",
            })

        # ── C_i: Coordination indicators ──────────────────────────────────
        # Shared AAGUID across multiple accounts
        aaguid = profile["aaguid"]
        accounts_with_same_aaguid = self.aaguid_registry.get(aaguid, set())
        if len(accounts_with_same_aaguid) > 3:
            profile["risk_components"].coordination = min(
                1.0, profile["risk_components"].coordination + 0.3
            )
            anomalies.append({
                "type": "shared_aaguid",
                "aaguid": aaguid,
                "n_accounts": len(accounts_with_same_aaguid),
                "severity": "MEDIUM",
                "note": (
                    "Multiple accounts share the same AAGUID. This may "
                    "indicate coordinated registration or shared device. "
                    "Further investigation recommended."
                ),
            })

        # Shared attestation root
        att_root = profile.get("attestation_root")
        if att_root and att_root in self.attestation_roots:
            accounts_with_same_root = self.attestation_roots[att_root]
            if len(accounts_with_same_root) > 5:
                profile["risk_components"].coordination = min(
                    1.0, profile["risk_components"].coordination + 0.2
                )

        # ── V_i: Verified policy violations ───────────────────────────────
        if profile["revoked"]:
            profile["risk_components"].violations = 1.0

        # Compute total risk
        risk_score = profile["risk_components"].weighted_risk()

        # Determine verification level (graduated, §5.3)
        if risk_score > 0.8:
            verification_level = VerificationLevel.LEGAL
        elif risk_score > 0.6:
            verification_level = VerificationLevel.INSTITUTIONAL
        elif risk_score > 0.4:
            verification_level = VerificationLevel.ACCOUNT
        else:
            verification_level = VerificationLevel.AUTOMATED

        profile["verification_level"] = verification_level

        result = {
            "account_id": account_id,
            "risk_score": risk_score,
            "risk_components": {
                "behavioral": profile["risk_components"].behavioral,
                "temporal": profile["risk_components"].temporal,
                "coordination": profile["risk_components"].coordination,
                "violations": profile["risk_components"].violations,
            },
            "verification_level": verification_level.name,
            "verification_level_value": verification_level.value,
            "anomalies": anomalies,
            "n_anomalies": len(anomalies),
            "note": (
                "Multi-identity risk assessment per Comprehensive Security "
                "§5.2. This is a HEURISTIC score, not proof of identity "
                "fraud. Graduated verification (§5.3) protects privacy."
            ),
        }

        self.risk_history.append(result)
        return result

    def revoke_credential(self, account_id: str, reason: str) -> Dict[str, Any]:
        """
        Revoke a credential and update violation risk.

        Args:
            account_id: Account to revoke
            reason: Revocation reason

        Returns:
            dict with revocation result
        """
        if account_id not in self.account_profiles:
            return {"error": "unknown_account"}

        profile = self.account_profiles[account_id]
        profile["revoked"] = True
        profile["risk_components"].violations = 1.0
        profile["verification_level"] = VerificationLevel.INSTITUTIONAL

        logger.warning(
            f"FIDO2 credential revoked: {account_id}, reason: {reason}"
        )

        return {
            "status": "revoked",
            "account_id": account_id,
            "reason": reason,
            "verification_level": profile["verification_level"].name,
        }

    def get_coordination_report(self) -> Dict[str, Any]:
        """
        Generate a coordination report across all accounts.

        Identifies:
          • AAGUIDs shared by multiple accounts
          • Attestation roots shared by multiple accounts
          • Registration time clusters

        HONEST: This report identifies PATTERNS, not proof of coordination.
        Shared AAGUIDs may indicate shared devices (legitimate) or
        coordinated abuse (suspicious). Context is required.
        """
        shared_aaguids = {
            aaguid: list(accounts)
            for aaguid, accounts in self.aaguid_registry.items()
            if len(accounts) > 1
        }

        shared_roots = {
            root: list(accounts)
            for root, accounts in self.attestation_roots.items()
            if len(accounts) > 1
        }

        # Registration time clustering (simple: 1-hour windows)
        registration_times = [
            p["registration_time"] for p in self.account_profiles.values()
        ]
        clusters = {}
        for t in registration_times:
            window = int(t // 3600)  # 1-hour windows
            if window not in clusters:
                clusters[window] = 0
            clusters[window] += 1

        burst_windows = {
            window: count
            for window, count in clusters.items()
            if count > 5
        }

        return {
            "total_accounts": len(self.account_profiles),
            "shared_aaguids": shared_aaguids,
            "n_shared_aaguids": len(shared_aaguids),
            "shared_attestation_roots": shared_roots,
            "n_shared_roots": len(shared_roots),
            "registration_burst_windows": burst_windows,
            "note": (
                "Coordination report identifies PATTERNS, not proof. "
                "Shared AAGUIDs may indicate shared devices (legitimate) "
                "or coordinated abuse (suspicious). Context required."
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "total_accounts": len(self.account_profiles),
            "total_aaguids_tracked": len(self.aaguid_registry),
            "total_attestation_roots": len(self.attestation_roots),
            "risk_history_size": len(self.risk_history),
            "revoked_accounts": sum(
                1 for p in self.account_profiles.values() if p["revoked"]
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [GD1] CHAOS PIPELINE GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class ChaosPipelineGrafanaDashboard:
    """
    [D] Generates Grafana dashboard JSON for chaos pipeline monitoring.

    Dashboard structure:
      Row 1: Pipeline Overview
        • Scenario execution status (pass/fail)
        • Quality gate results
        • Pipeline duration

      Row 2: Regime Detection
        • Hill estimator θ̂ distribution
        • CUSUM statistic over time
        • Regime classification (finite/regularly-varying/degenerate)

      Row 3: Degenerate Alerts
        • Occupation fraction polarization
        • Degenerate regime alerts
        • Emergency stop status

      Row 4: δ_min Monitoring
        • Subsampling CI for δ_min (Paper 11 Thm 4.4)
        • No-Zeno event count vs bound (SESI_NoZeno Thm 3.2)
        • Energy budget utilization (SESI_NoZeno Thm 4.2)

      Row 5: FIDO2 Identity Risk
        • R_id distribution across accounts
        • Verification level distribution
        • Coordination indicators

    HONEST: This generates a Grafana dashboard JSON template. It does
    NOT deploy Grafana, configure Prometheus, or make security decisions.
    All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
    """

    def __init__(self, datasource_uid: str = "prometheus"):
        self.datasource_uid = datasource_uid

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
        panels = [
            # ── Row 1: Pipeline Overview ────────────────────────────────────
            self._panel(
                1, "Chaos Scenarios Executed", "stat",
                [{"expr": "sesi_chaos_scenarios_total", "legendFormat": "total"}],
                {"h": 6, "w": 4, "x": 0, "y": 0},
            ),
            self._panel(
                2, "Scenario Pass Rate", "gauge",
                [{"expr": "sesi_chaos_scenarios_passed / sesi_chaos_scenarios_total * 100",
                  "legendFormat": "pass_rate"}],
                {"h": 6, "w": 4, "i": 4, "y": 0},
                extra={"fieldConfig": {"defaults": {
                    "min": 0, "max": 100, "unit": "percent",
                    "thresholds": {"mode": "absolute", "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 70},
                        {"color": "green", "value": 90}]}}}},
            ),
            self._panel(
                3, "Quality Gate Status", "stat",
                [{"expr": "sesi_quality_gate_passed", "legendFormat": "gate"}],
                {"h": 6, "w": 4, "i": 8, "y": 0},
                extra={"fieldConfig": {"defaults": {
                    "mappings": [
                        {"type": "value", "options": {
                            "0": {"text": "BLOCKED", "color": "red"},
                            "1": {"text": "PASSED", "color": "green"}}}]}}},
            ),
            self._panel(
                4, "Pipeline Duration", "stat",
                [{"expr": "sesi_chaos_pipeline_duration_seconds",
                  "legendFormat": "duration"}],
                {"h": 6, "w": 4, "i": 12, "y": 0},
                extra={"fieldConfig": {"defaults": {"unit": "s"}}},
            ),
            self._panel(
                5, "Active Alerts", "stat",
                [{"expr": "sesi_active_alerts", "legendFormat": "alerts"}],
                {"h": 6, "w": 4, "i": 16, "y": 0},
                extra={"fieldConfig": {"defaults": {
                    "thresholds": {"mode": "absolute", "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 1},
                        {"color": "red", "value": 5}]}}}},
            ),
            self._panel(
                6, "Emergency Stops (24h)", "stat",
                [{"expr": "sum(increase(sesi_emergency_stops_total[24h]))",
                  "legendFormat": "stops"}],
                {"h": 6, "w": 4, "i": 20, "y": 0},
            ),

            # ── Row 2: Regime Detection ────────────────────────────────────
            self._panel(
                7, "Hill Estimator θ̂ (paper6 Thm 5.1)", "timeseries",
                [{"expr": "sesi_hill_theta_hat", "legendFormat": "θ̂"}],
                {"h": 8, "w": 8, "i": 0, "y": 6},
            ),
            self._panel(
                8, "CUSUM Statistic (Six_Problems P4)", "timeseries",
                [{"expr": "sesi_cusum_statistic", "legendFormat": "cusum"},
                 {"expr": "sesi_cusum_threshold", "legendFormat": "threshold"}],
                {"h": 8, "w": 8, "i": 8, "y": 6},
            ),
            self._panel(
                9, "Regime Classification", "stat",
                [{"expr": "sesi_quality_regime", "legendFormat": "regime"}],
                {"h": 8, "w": 8, "i": 16, "y": 6},
                extra={"fieldConfig": {"defaults": {
                    "mappings": [
                        {"type": "value", "options": {
                            "0": {"text": "FINITE_MEAN", "color": "green"},
                            "1": {"text": "REGULARLY_VARYING", "color": "yellow"},
                            "2": {"text": "DEGENERATE", "color": "red"},
                            "3": {"text": "INSUFFICIENT_DATA", "color": "gray"}}}]}}},
            ),

            # ── Row 3: Degenerate Alerts ───────────────────────────────────
            self._panel(
                10, "Occupation Fraction (paper6 Thm 4.1)", "gauge",
                [{"expr": "sesi_occupation_fraction", "legendFormat": "occupation"}],
                {"h": 8, "w": 8, "i": 0, "y": 14},
                extra={"fieldConfig": {"defaults": {
                    "min": 0, "max": 1,
                    "thresholds": {"mode": "absolute", "steps": [
                        {"color": "red", "value": None},
                        {"color": "green", "value": 0.3},
                        {"color": "red", "value": 0.7}]}}}},
            ),
            self._panel(
                11, "Degenerate Regime Alerts", "timeseries",
                [{"expr": "sum(increase(sesi_degenerate_alerts_total[5m]))",
                  "legendFormat": "alerts/5m"}],
                {"h": 8, "w": 8, "i": 8, "y": 14},
            ),
            self._panel(
                12, "Kill Switch Status", "stat",
                [{"expr": "sesi_kill_switch_active", "legendFormat": "kill_switch"}],
                {"h": 8, "w": 8, "i": 16, "y": 14},
                extra={"fieldConfig": {"defaults": {
                    "mappings": [
                        {"type": "value", "options": {
                            "0": {"text": "INACTIVE", "color": "green"},
                            "1": {"text": "ACTIVE", "color": "red"}}}]}}},
            ),

            # ── Row 4: δ_min Monitoring ────────────────────────────────────
            self._panel(
                13, "δ_min Subsampling CI (Paper 11 Thm 4.4)", "timeseries",
                [{"expr": "sesi_delta_min_estimate", "legendFormat": "δ_min"},
                 {"expr": "sesi_delta_min_ci_lower", "legendFormat": "CI lower"},
                 {"expr": "sesi_delta_min_ci_upper", "legendFormat": "CI upper"}],
                {"h": 8, "w": 8, "i": 0, "y": 22},
            ),
            self._panel(
                14, "No-Zeno Event Count vs Bound (SESI Thm 3.2)", "timeseries",
                [{"expr": "sesi_event_count", "legendFormat": "N(T)"},
                 {"expr": "sesi_no_zeno_bound", "legendFormat": "bound"}],
                {"h": 8, "w": 8, "i": 8, "y": 22},
            ),
            self._panel(
                15, "Energy Budget Utilization (SESI Thm 4.2)", "gauge",
                [{"expr": "sesi_energy_spent / sesi_energy_budget * 100",
                  "legendFormat": "utilization"}],
                {"h": 8, "w": 8, "i": 16, "y": 22},
                extra={"fieldConfig": {"defaults": {
                    "min": 0, "max": 100, "unit": "percent",
                    "thresholds": {"mode": "absolute", "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 70},
                        {"color": "red", "value": 90}]}}}},
            ),

            # ── Row 5: FIDO2 Identity Risk ─────────────────────────────────
            self._panel(
                16, "R_id Distribution (Comp. Security §5.2)", "histogram",
                [{"expr": "sesi_identity_risk_score", "legendFormat": "R_id",
                  "format": "heatmap"}],
                {"h": 8, "w": 8, "i": 0, "y": 30},
            ),
            self._panel(
                17, "Verification Level Distribution (§5.3)", "piechart",
                [{"expr": "sesi_verification_level_count", "legendFormat": "{{level}}"}],
                {"h": 8, "w": 8, "i": 8, "y": 30},
            ),
            self._panel(
                18, "Coordination Indicators", "timeseries",
                [{"expr": "sesi_shared_aaguid_count", "legendFormat": "shared_aaguids"},
                 {"expr": "sesi_registration_burst_count", "legendFormat": "reg_bursts"}],
                {"h": 8, "w": 8, "i": 16, "y": 30},
            ),
        ]

        dashboard = {
            "uid": "sesi-chaos-pipeline",
            "title": "SESI Chaos Pipeline Monitoring",
            "tags": ["sesi", "chaos", "pipeline", "degenerate-regime"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "templating": {
                "list": [
                    {
                        "name": "scenario",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self.datasource_uid},
                        "query": "label_values(sesi_chaos_scenarios_total, scenario)",
                        "refresh": 2,
                    },
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
                        "name": "Chaos Scenarios",
                        "datasource": {"type": "prometheus", "uid": self.datasource_uid},
                        "enable": True,
                        "expr": "sesi_chaos_scenario_started",
                        "iconColor": "blue",
                        "titleFormat": "Scenario: {{scenario}}",
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
        """
        Generate Prometheus alert rules for chaos pipeline monitoring.

        HONEST: These are TEMPLATE alert rules. Thresholds are FREE
        DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9) and must be
        validated against real chaos pipeline data.
        """
        return [
            {
                "alert": "DegenerateRegimeDetected",
                "expr": "sesi_quality_regime == 2",
                "for": "5m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Degenerate quality regime detected",
                    "description": (
                        "Quality regime is DEGENERATE (θ→0, ½δ₀+½δ₁). "
                        "paper6 Thm 5.1: occupation fraction polarizing "
                        "between extremes. Averaging-based monitoring "
                        "will miss this. Investigate immediately."
                    ),
                },
            },
            {
                "alert": "CUSUMChangePoint",
                "expr": "sesi_cusum_statistic > sesi_cusum_threshold",
                "for": "1m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "CUSUM change-point detected",
                    "description": (
                        "CUSUM statistic exceeded threshold. "
                        "Six_Problems P4: change-point in quality stream."
                    ),
                },
            },
            {
                "alert": "NoZenoBoundExceeded",
                "expr": "sesi_event_count > sesi_no_zeno_bound",
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "No-Zeno bound exceeded",
                    "description": (
                        "Event count exceeds No-Zeno bound: "
                        "N(T) ≤ 1 + 2C_V·T/δ_min (SESI_NoZeno Thm 3.2). "
                        "Possible Zeno-like event accumulation."
                    ),
                },
            },
            {
                "alert": "EnergyBudgetExhausted",
                "expr": "sesi_energy_spent / sesi_energy_budget > 0.9",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "Energy budget nearly exhausted",
                    "description": (
                        "Energy budget utilization > 90%. "
                        "SESI_NoZeno Thm 4.2: N(T) ≤ (E_max+D)/ΔE_min."
                    ),
                },
            },
            {
                "alert": "HighIdentityRisk",
                "expr": "sesi_identity_risk_score > 0.7",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High multi-identity risk detected",
                    "description": (
                        "R_id > 0.7 for one or more accounts. "
                        "Comprehensive Security §5.2. "
                        "Graduated verification recommended (§5.3)."
                    ),
                },
            },
            {
                "alert": "SignCountRegression",
                "expr": "sesi_sign_count_regression_detected == 1",
                "for": "0m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "FIDO2 sign count regression detected",
                    "description": (
                        "Sign count regression indicates possible cloned "
                        "authenticator. Immediate investigation required."
                    ),
                },
            },
            {
                "alert": "QualityGateBlocked",
                "expr": "sesi_quality_gate_passed == 0",
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Quality gate blocked deployment",
                    "description": (
                        "Quality gate detected degenerate regime or "
                        "other critical issue. Deployment blocked."
                    ),
                },
            },
        ]

    def generate_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus scrape configuration for chaos pipeline."""
        return {
            "scrape_configs": [
                {
                    "job_name": "sesi-chaos-pipeline",
                    "scrape_interval": "15s",
                    "static_configs": [
                        {"targets": ["sesi-chaos-pipeline:9090"]},
                    ],
                    "metrics_path": "/metrics",
                },
                {
                    "job_name": "sesi-fido2-identity",
                    "scrape_interval": "30s",
                    "static_configs": [
                        {"targets": ["sesi-fido2-bridge:9091"]},
                    ],
                    "metrics_path": "/metrics",
                },
            ],
        }

    def write_dashboard(self, output_path: str = "./monitoring/grafana/chaos-pipeline.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Grafana dashboard written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./monitoring/prometheus/chaos-alerts.yaml") -> str:
        """Write alert rules to file."""
        rules = self.generate_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {"groups": [{"name": "sesi_chaos_pipeline", "rules": rules}]}

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"Alert rules written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIVaultFIDOChaosGrafanaSuite:
    """
    [D] Complete orchestration suite for Vault namespaces, FIDO2
    multi-identity risk, and chaos pipeline Grafana dashboard.
    """

    def __init__(self):
        self.vault_ns = VaultEnterpriseNamespaceManager()
        self.vault_ns.configure_default_namespaces()
        self.fido2_bridge = FIDO2MultiIdentityRiskBridge()
        self.grafana = ChaosPipelineGrafanaDashboard()

    def generate_all_artifacts(self, output_dir: str = "./sesi_security_v2") -> Dict[str, str]:
        """Generate all configuration artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Vault namespace manifests
        vault_files = self.vault_ns.generate_all_manifests(str(out_path / "vault"))
        results["vault_namespaces"] = vault_files

        # Grafana dashboard
        dashboard_path = self.grafana.write_dashboard(
            str(out_path / "grafana" / "chaos-pipeline.json")
        )
        results["grafana_dashboard"] = dashboard_path

        # Alert rules
        alerts_path = self.grafana.write_alert_rules(
            str(out_path / "prometheus" / "chaos-alerts.yaml")
        )
        results["alert_rules"] = alerts_path

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "vault_namespaces": self.vault_ns.get_status(),
            "fido2_bridge": self.fido2_bridge.get_status(),
            "grafana": {
                "dashboard_panels": 18,
                "alert_rules": 7,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Vault NS × FIDO2 Identity × Chaos Grafana — Smoke Test")
    print("=" * 72)
    print(f"  hvac available   : {HAS_HVAC}")
    print(f"  PyYAML available : {HAS_YAML}")

    suite = SESIVaultFIDOChaosGrafanaSuite()

    # ── Test 1: Vault Namespace Configuration ──────────────────────────────
    print("\n  ── Vault Enterprise Namespaces ──")
    ns_status = suite.vault_ns.get_status()
    print(f"    Namespaces configured: {ns_status['namespaces_configured']}")
    for name in ns_status["namespace_names"]:
        print(f"      • {name}")

    # Generate manifests
    vault_files = suite.vault_ns.generate_all_manifests("./test_vault_ns")
    print(f"    Manifests generated  : {len(vault_files)}")

    # Sentinel policies
    sentinel = suite.vault_ns.generate_sentinel_policies()
    print(f"    Sentinel policies    : {len(sentinel)}")
    for policy_name in sentinel:
        print(f"      • {policy_name}")

    # ── Test 2: FIDO2 Multi-Identity Risk ──────────────────────────────────
    print("\n  ── FIDO2 Multi-Identity Risk Bridge ──")

    # Register accounts
    suite.fido2_bridge.register_account(
        account_id="user_001",
        aaguid="adce0a90-1e59-4ecc-8c0b-6e020557d7bd",  # YubiKey 5
        attestation_format="packed",
        attestation_root="root_cert_hash_001",
    )
    suite.fido2_bridge.register_account(
        account_id="user_002",
        aaguid="adce0a90-1e59-4ecc-8c0b-6e020557d7bd",  # Same AAGUID
        attestation_format="packed",
        attestation_root="root_cert_hash_001",
    )
    suite.fido2_bridge.register_account(
        account_id="user_003",
        aaguid="08987058-cadc-4b81-b6e1-30de50dcbe96",  # Windows Hello
        attestation_format="packed",
        attestation_root="root_cert_hash_002",
    )

    # Record authentications
    auth1 = suite.fido2_bridge.record_authentication("user_001", sign_count=5)
    print(f"    user_001 auth        : risk={auth1['risk_score']:.3f}, "
          f"level={auth1['verification_level']}")

    auth2 = suite.fido2_bridge.record_authentication("user_002", sign_count=3)
    print(f"    user_002 auth        : risk={auth2['risk_score']:.3f}, "
          f"level={auth2['verification_level']}")

    # Sign count regression (clone detection)
    auth3 = suite.fido2_bridge.record_authentication("user_001", sign_count=2)
    print(f"    user_001 regression  : risk={auth3['risk_score']:.3f}, "
          f"anomalies={auth3['n_anomalies']}")
    if auth3["anomalies"]:
        print(f"      Anomaly: {auth3['anomalies'][0]['type']}")

    # Coordination report
    coord = suite.fido2_bridge.get_coordination_report()
    print(f"    Shared AAGUIDs       : {coord['n_shared_aaguids']}")
    print(f"    Total accounts       : {coord['total_accounts']}")

    # Revoke a credential
    revoke = suite.fido2_bridge.revoke_credential("user_003", "testing")
    print(f"    Revocation           : {revoke['status']}")

    # ── Test 3: Grafana Dashboard ──────────────────────────────────────────
    print("\n  ── Chaos Pipeline Grafana Dashboard ──")

    dashboard = suite.grafana.generate_dashboard()
    print(f"    Dashboard panels     : {len(dashboard['panels'])}")
    print(f"    Alert rules          : {len(suite.grafana.generate_alert_rules())}")
    print(f"    Template variables   : {len(dashboard['templating']['list'])}")

    # Write dashboard
    dash_path = suite.grafana.write_dashboard("./test_grafana/chaos-pipeline.json")
    print(f"    Dashboard written    : {dash_path}")

    # Write alert rules
    alerts_path = suite.grafana.write_alert_rules("./test_grafana/chaos-alerts.yaml")
    print(f"    Alert rules written  : {alerts_path}")

    # ── Full Status ─────────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Vault namespaces     : {status['vault_namespaces']['namespaces_configured']}")
    print(f"    FIDO2 accounts       : {status['fido2_bridge']['total_accounts']}")
    print(f"    FIDO2 AAGUIDs        : {status['fido2_bridge']['total_aaguids_tracked']}")
    print(f"    Grafana panels       : {status['grafana']['dashboard_panels']}")
    print(f"    Alert rules          : {status['grafana']['alert_rules']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Vault namespaces require Vault Enterprise license")
    print("  • Sentinel policies require Vault Enterprise")
    print("  • FIDO2 multi-identity risk is HEURISTIC, not proof")
    print("  • Graduated verification protects privacy (§5.3)")
    print("  • Grafana dashboards are VISUALIZATION, not decisions")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • This module does NOT create or detect consciousness")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
