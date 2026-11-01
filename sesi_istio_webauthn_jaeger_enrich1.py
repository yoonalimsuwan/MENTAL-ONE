"""
=============================================================================
SESI ISTIO EXTERNAL AUTHZ × WEBAUTHN VAULT TRANSIT × JAEGER AGIONE ENRICHMENT
=============================================================================

This module adds three production-oriented components:

1. IstioExternalAuthorizationPolicyGenerator
   - Generates Istio AuthorizationPolicy manifests for external services.
   - Intended to run on an egress gateway workload.
   - Supports zone-aware allow rules for FIDO MDS, Vault, cloud KMS, OAuth,
     Prometheus, Jaeger, and custom external APIs.

2. WebAuthnVaultTransitCredentialStore
   - Stores WebAuthn credential records encrypted by Vault Transit.
   - Private keys are NEVER stored. Only public key + metadata are persisted.
   - Uses KV v2 for storage and Transit for envelope encryption.
   - Provides development fallback only when Vault is unavailable.

3. JaegerAGIONEQualityEnrichmentPipeline
   - Enriches traces with AGIONE / EcosystemOrchestrator quality_report().
   - Supports OpenTelemetry if installed.
   - Falls back to in-memory span dictionaries for local testing.
   - Adds quality_score, Psyche Plus validity/axiom/codification metadata,
     No-Zeno trace sampling gate, and simple regime diagnostics.

Important honesty boundary:
- This is security / observability infrastructure, not consciousness creation.
- WebAuthn attestation verifies authenticator provenance, not moral authority.
- Jaeger trace enrichment shows correlation, not causation.
- Istio policies are templates and must be reviewed before production use.

Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
License   : MIT
Year      : 2026
=============================================================================
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("sesi.istio_webauthn_vault_jaeger")

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
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    StatusCode = None


# =============================================================================
# Shared utilities
# =============================================================================

class SecurityZone(Enum):
    Z0_IMPLANT = "z0-implant"
    Z1_CLINICAL_GATEWAY = "z1-clinical-gateway"
    Z2_PROCESSING = "z2-processing"
    Z3_EXTERNAL = "z3-external"


def safe_unit_scalar(x: Any, default: float = 0.5) -> float:
    """
    Psyche Plus v3.3 style sanitation:
    NaN / Inf / out-of-range values are clamped into [0, 1].
    """
    try:
        v = float(x)
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return max(0.0, min(1.0, v))


def write_struct(path: Path, data: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_YAML and path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def compute_delta_min(kappa: float = 2.0, a_min: float = 0.01, n_dim: int = 2) -> float:
    """
    Delta_Min style lower bound proxy:
        δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}

    This assumes the relevant fatness condition. If fatness fails, the
    geometric guarantee may not apply.
    """
    if n_dim <= 1:
        raise ValueError("n_dim must be >= 2")
    omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
    return (1.0 / kappa) * (a_min / omega) ** (1.0 / (n_dim - 1))


# =============================================================================
# 1. Istio AuthorizationPolicy for External Services
# =============================================================================

@dataclass
class ExternalServiceRule:
    name: str
    hosts: List[str]
    ports: List[int]
    allowed_source_zones: List[SecurityZone]
    allowed_service_accounts: List[str]
    source_namespaces: List[str]
    methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    egress_gateway_namespace: str = "istio-system"
    egress_gateway_labels: Dict[str, str] = field(default_factory=lambda: {"istio": "egressgateway"})
    action: str = "ALLOW"


class IstioExternalAuthorizationPolicyGenerator:
    """
    Generate Istio AuthorizationPolicy for external services via egress gateway.

    Recommended pattern:
    - ServiceEntry defines external host.
    - Gateway / VirtualService route egress traffic through egress gateway.
    - AuthorizationPolicy selects egress gateway and allows only specified
      source principals / namespaces / hosts / ports.

    Note:
    AuthorizationPolicy alone does not force traffic through egress gateway.
    You need ServiceEntry + VirtualService + Gateway routing as well.
    """

    def __init__(self, namespace: str = "sesi-oversight"):
        self.namespace = namespace
        self.rules: Dict[str, ExternalServiceRule] = {}
        self._install_defaults()

    def _install_defaults(self) -> None:
        self.add_rule(ExternalServiceRule(
            name="fido-mds",
            hosts=["mds3.fidoalliance.org"],
            ports=[443],
            allowed_source_zones=[SecurityZone.Z2_PROCESSING, SecurityZone.Z1_CLINICAL_GATEWAY],
            allowed_service_accounts=["sesi-fido-sync", "sesi-attestation-verifier"],
            source_namespaces=["sesi-processing", "sesi-clinical"],
            methods=["GET"],
        ))
        self.add_rule(ExternalServiceRule(
            name="vault",
            hosts=["vault.sesi.example.com"],
            ports=[8200],
            allowed_source_zones=[SecurityZone.Z1_CLINICAL_GATEWAY, SecurityZone.Z2_PROCESSING],
            allowed_service_accounts=["sesi-vault-client", "sesi-webauthn-server"],
            source_namespaces=["sesi-clinical", "sesi-processing"],
            methods=["GET", "POST", "PUT", "DELETE"],
        ))
        self.add_rule(ExternalServiceRule(
            name="cloud-kms",
            hosts=[
                "kms.us-east-1.amazonaws.com",
                "cloudkms.googleapis.com",
                "kms.cn-hangzhou.aliyuncs.com",
                "kms.tencentcloudapi.com",
                "kms.cn-north-4.myhuaweicloud.com",
            ],
            ports=[443],
            allowed_source_zones=[SecurityZone.Z1_CLINICAL_GATEWAY],
            allowed_service_accounts=["sesi-vault-unseal", "sesi-kms-operator"],
            source_namespaces=["sesi-clinical"],
            methods=["POST"],
        ))
        self.add_rule(ExternalServiceRule(
            name="observability",
            hosts=[
                "prometheus.monitoring.svc.cluster.local",
                "jaeger-collector.observability.svc.cluster.local",
                "grafana.monitoring.svc.cluster.local",
            ],
            ports=[9090, 14268, 16686, 3000],
            allowed_source_zones=[SecurityZone.Z2_PROCESSING],
            allowed_service_accounts=["sesi-metrics-exporter", "sesi-trace-enricher"],
            source_namespaces=["sesi-processing"],
            methods=["GET", "POST"],
        ))

    def add_rule(self, rule: ExternalServiceRule) -> None:
        self.rules[rule.name] = rule

    def source_principals(self, rule: ExternalServiceRule) -> List[str]:
        principals = []
        for ns in rule.source_namespaces:
            for sa in rule.allowed_service_accounts:
                principals.append(f"cluster.local/ns/{ns}/sa/{sa}")
        return principals

    def generate_authorization_policy(self, rule_name: str) -> Dict[str, Any]:
        rule = self.rules[rule_name]
        return {
            "apiVersion": "security.istio.io/v1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"sesi-external-{rule.name}",
                "namespace": rule.egress_gateway_namespace,
                "labels": {
                    "app.kubernetes.io/part-of": "sesi-oversight",
                    "sesi.io/policy-type": "external-service-authz",
                },
            },
            "spec": {
                "selector": {"matchLabels": rule.egress_gateway_labels},
                "action": rule.action,
                "rules": [
                    {
                        "from": [{"source": {"principals": self.source_principals(rule)}}],
                        "to": [{
                            "operation": {
                                "hosts": rule.hosts,
                                "ports": [str(p) for p in rule.ports],
                                "methods": rule.methods,
                            }
                        }],
                    }
                ],
            },
        }

    def generate_deny_implant_direct_external(self) -> Dict[str, Any]:
        """
        Deny direct external egress from implant zone.
        This policy is a safety guard; actual enforcement should be paired
        with NetworkPolicy and egress-gateway routing.
        """
        return {
            "apiVersion": "security.istio.io/v1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": "sesi-deny-implant-direct-external",
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {"matchLabels": {"sesi.io/zone": SecurityZone.Z0_IMPLANT.value}},
                "action": "DENY",
                "rules": [
                    {
                        "to": [{
                            "operation": {
                                "ports": ["443", "80", "8200"],
                            }
                        }]
                    }
                ],
            },
        }

    def write_all(self, output_dir: str = "./istio_external_authz") -> Dict[str, str]:
        out = Path(output_dir)
        files = {}
        for name in self.rules:
            p = out / f"authz-{name}.yaml"
            files[name] = write_struct(p, self.generate_authorization_policy(name))
        p = out / "deny-implant-direct-external.yaml"
        files["deny_implant_direct_external"] = write_struct(p, self.generate_deny_implant_direct_external())
        return files


# =============================================================================
# 2. WebAuthn + Vault Transit Credential Store
# =============================================================================

@dataclass
class VaultTransitConfig:
    vault_addr: str = "http://127.0.0.1:8200"
    vault_token_env: str = "VAULT_TOKEN"
    vault_namespace: Optional[str] = None
    kv_mount: str = "secret"
    transit_mount: str = "transit"
    transit_key: str = "sesi-webauthn"
    kv_prefix: str = "sesi/webauthn/credentials"
    dev_fallback: bool = True


class WebAuthnVaultTransitCredentialStore:
    """
    Store WebAuthn credential records encrypted with Vault Transit.

    IMPORTANT:
    - WebAuthn private keys never leave the authenticator.
    - This store persists public key + credential metadata only.
    - Vault Transit encrypts the JSON record before KV v2 storage.
    - Development fallback is not secure and is clearly marked.
    """

    FORBIDDEN_FIELDS = {
        "private_key", "privateKey", "secret", "pin", "password",
        "biometric", "biometric_template", "seed", "mnemonic",
    }

    def __init__(self, cfg: Optional[VaultTransitConfig] = None):
        self.cfg = cfg or VaultTransitConfig()
        self.client = None
        self.dev_store: Dict[str, Dict[str, Any]] = {}
        self.audit: List[Dict[str, Any]] = []

        if HAS_HVAC:
            token = os.getenv(self.cfg.vault_token_env)
            try:
                self.client = hvac.Client(
                    url=self.cfg.vault_addr,
                    token=token,
                    namespace=self.cfg.vault_namespace,
                )
                if token and self.client.is_authenticated():
                    logger.info("Vault Transit credential store connected")
                else:
                    self.client = None
                    logger.warning("Vault not authenticated; using dev fallback if enabled")
            except Exception as e:
                logger.warning("Vault connection failed: %s", e)
                self.client = None

    def _validate_record(self, record: Dict[str, Any]) -> None:
        keys = set(record.keys())
        bad = keys & self.FORBIDDEN_FIELDS
        if bad:
            raise ValueError(f"Forbidden secret/private fields in WebAuthn record: {bad}")
        if "credential_id" not in record:
            raise ValueError("credential_id is required")
        if "public_key_pem" not in record:
            raise ValueError("public_key_pem is required")

    def _b64e(self, data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    def _b64d(self, data: str) -> bytes:
        return base64.b64decode(data.encode("ascii"))

    def transit_encrypt(self, plaintext: str) -> str:
        if self.client is None:
            if not self.cfg.dev_fallback:
                raise RuntimeError("Vault unavailable and dev_fallback=False")
            return "DEV_PLAINTEXT_B64:" + self._b64e(plaintext.encode("utf-8"))

        result = self.client.secrets.transit.encrypt_data(
            name=self.cfg.transit_key,
            plaintext=self._b64e(plaintext.encode("utf-8")),
            mount_point=self.cfg.transit_mount,
        )
        return result["data"]["ciphertext"]

    def transit_decrypt(self, ciphertext: str) -> str:
        if ciphertext.startswith("DEV_PLAINTEXT_B64:"):
            if not self.cfg.dev_fallback:
                raise RuntimeError("Dev ciphertext not allowed")
            return self._b64d(ciphertext.split(":", 1)[1]).decode("utf-8")

        if self.client is None:
            raise RuntimeError("Vault unavailable; cannot decrypt real ciphertext")

        result = self.client.secrets.transit.decrypt_data(
            name=self.cfg.transit_key,
            ciphertext=ciphertext,
            mount_point=self.cfg.transit_mount,
        )
        return self._b64d(result["data"]["plaintext"]).decode("utf-8")

    def store_credential(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected record:
        {
          "credential_id": "...",
          "user_id": "...",
          "rp_id": "sesi.example.com",
          "public_key_pem": "...",
          "sign_count": 0,
          "aaguid": "...",
          "attestation_format": "packed",
          "backup_eligible": true,
          "backup_state": false
        }
        """
        self._validate_record(record)
        cid = record["credential_id"]
        now = time.time()
        payload = dict(record)
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        payload.setdefault("revoked", False)

        ciphertext = self.transit_encrypt(json.dumps(payload, sort_keys=True))
        cid_hash = hashlib.sha256(cid.encode()).hexdigest()

        kv_secret = {
            "credential_id_hash": cid_hash,
            "ciphertext": ciphertext,
            "updated_at": now,
            "storage": "vault-transit" if self.client else "dev-fallback",
        }

        if self.client is not None:
            self.client.secrets.kv.v2.create_or_update_secret(
                mount_point=self.cfg.kv_mount,
                path=f"{self.cfg.kv_prefix}/{cid_hash}",
                secret=kv_secret,
            )
        else:
            if not self.cfg.dev_fallback:
                raise RuntimeError("Vault unavailable and dev_fallback=False")
            self.dev_store[cid_hash] = kv_secret

        self.audit.append({
            "event": "store_credential",
            "credential_id_hash": cid_hash,
            "timestamp": now,
            "vault": self.client is not None,
        })

        return {
            "status": "stored",
            "credential_id_hash": cid_hash,
            "vault_protected": self.client is not None,
        }

    def load_credential(self, credential_id: str) -> Dict[str, Any]:
        cid_hash = hashlib.sha256(credential_id.encode()).hexdigest()

        if self.client is not None:
            result = self.client.secrets.kv.v2.read_secret_version(
                mount_point=self.cfg.kv_mount,
                path=f"{self.cfg.kv_prefix}/{cid_hash}",
            )
            secret = result["data"]["data"]
        else:
            secret = self.dev_store[cid_hash]

        plaintext = self.transit_decrypt(secret["ciphertext"])
        return json.loads(plaintext)

    def update_sign_count(self, credential_id: str, new_sign_count: int) -> Dict[str, Any]:
        record = self.load_credential(credential_id)
        old = int(record.get("sign_count", 0))

        if new_sign_count <= old and new_sign_count != 0:
            return {
                "status": "sign_count_anomaly",
                "old": old,
                "new": new_sign_count,
                "action": "reject_possible_clone",
            }

        record["sign_count"] = new_sign_count
        record["last_used_at"] = time.time()
        self.store_credential(record)

        return {
            "status": "updated",
            "old": old,
            "new": new_sign_count,
        }

    def revoke_credential(self, credential_id: str, reason: str) -> Dict[str, Any]:
        record = self.load_credential(credential_id)
        record["revoked"] = True
        record["revoked_reason"] = reason
        record["revoked_at"] = time.time()
        self.store_credential(record)
        return {"status": "revoked", "credential_id": credential_id}

    def status(self) -> Dict[str, Any]:
        return {
            "vault_connected": self.client is not None,
            "dev_fallback": self.cfg.dev_fallback and self.client is None,
            "dev_store_size": len(self.dev_store),
            "audit_events": len(self.audit),
            "transit_key": self.cfg.transit_key,
        }


# =============================================================================
# 3. Jaeger Trace Enrichment for AGIONE quality_report()
# =============================================================================

class NoZenoTraceGate:
    """
    Rate-limit trace emission using a No-Zeno style bound:
        N(T) ≤ 1 + 2 C_V T / δ_min
    """

    def __init__(self, window_s: float = 60.0, c_v: float = 1.0,
                 delta_min: Optional[float] = None):
        self.window_s = window_s
        self.c_v = c_v
        self.delta_min = delta_min if delta_min is not None else compute_delta_min()
        self.events: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        self.events = [t for t in self.events if now - t <= self.window_s]
        bound = 1 + 2 * self.c_v * self.window_s / max(self.delta_min, 1e-12)
        if len(self.events) >= bound:
            return False
        self.events.append(now)
        return True


class JaegerAGIONEQualityEnrichmentPipeline:
    """
    Enrich traces with AGIONE quality_report() and Psyche Plus metadata.

    Usage:
        pipeline = JaegerAGIONEQualityEnrichmentPipeline()
        pipeline.instrument_agione(agione, orchestrator=orchestrator)

    Or manually:
        pipeline.enrich(agi_state=state, orchestrator=orchestrator)

    If OpenTelemetry is installed, it creates real spans.
    Otherwise it stores span dictionaries in memory for testing.
    """

    def __init__(self, service_name: str = "sesi-agione-quality"):
        self.service_name = service_name
        self.gate = NoZenoTraceGate()
        self.memory_spans: List[Dict[str, Any]] = []
        self.quality_history: List[float] = []

        self.tracer = None
        if HAS_OTEL:
            self.tracer = trace.get_tracer(service_name)

    def extract_quality_report(
        self,
        agi_state: Any = None,
        orchestrator: Any = None,
        output: Any = None,
    ) -> Dict[str, float]:
        """
        Extract quality scores from:
        1. orchestrator.quality_report()
        2. agi_state.ecosystem_quality_scores
        3. output dict containing ecosystem_quality_scores
        """
        q: Dict[str, Any] = {}

        if orchestrator is not None and hasattr(orchestrator, "quality_report"):
            try:
                q = orchestrator.quality_report() or {}
            except Exception as e:
                logger.debug("quality_report unavailable: %s", e)

        if not q and agi_state is not None:
            q = getattr(agi_state, "ecosystem_quality_scores", {}) or {}

        if not q and isinstance(output, dict):
            q = output.get("ecosystem_quality_scores", {}) or {}

        return {k: safe_unit_scalar(v) for k, v in q.items()}

    def extract_state_obj(self, obj: Any) -> Any:
        """
        Try to find an AGIState-like object in common return shapes.
        """
        if obj is None:
            return None
        if hasattr(obj, "ecosystem_quality_scores") or hasattr(obj, "psyche_plus_validity"):
            return obj
        if isinstance(obj, dict):
            for key in ("state", "agi_state"):
                if key in obj:
                    return self.extract_state_obj(obj[key])
        if isinstance(obj, (tuple, list)):
            for item in obj:
                found = self.extract_state_obj(item)
                if found is not None:
                    return found
        return None

    def hill_regime(self) -> Dict[str, Any]:
        """
        Lightweight quality regime readout.
        This is diagnostic, not a proof.
        """
        vals = self.quality_history[-200:]
        if len(vals) < 20:
            return {"regime": "INSUFFICIENT_DATA", "n": len(vals)}

        import numpy as np
        med = float(np.median(vals))
        tail = sorted([abs(v - med) for v in vals if abs(v - med) > 1e-6], reverse=True)
        k = min(20, len(tail) - 1)
        if k < 5 or tail[k] <= 0:
            return {"regime": "INSUFFICIENT_DATA", "n": len(vals)}

        theta = sum(math.log(tail[i] / tail[k]) for i in range(k)) / k
        if theta > 1:
            regime = "FINITE_MEAN"
        elif theta > 0:
            regime = "REGULARLY_VARYING"
        else:
            regime = "DEGENERATE"

        return {"regime": regime, "theta_hat": theta, "n": len(vals)}

    def enrich(
        self,
        agi_state: Any = None,
        orchestrator: Any = None,
        output: Any = None,
        operation: str = "agione.forward",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create / enrich a trace span with quality and Psyche Plus metadata.
        """
        if not self.gate.allow():
            return {"status": "trace_rate_limited"}

        state = agi_state or self.extract_state_obj(output)
        qrep = self.extract_quality_report(state, orchestrator, output)
        ecosystem = safe_unit_scalar(qrep.get("ecosystem", 0.5))
        self.quality_history.append(ecosystem)

        span_data = {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "operation": operation,
            "timestamp": time.time(),
            "quality_report": qrep,
            "ecosystem_quality": ecosystem,
            "regime": self.hill_regime(),
            "extra": extra or {},
        }

        if state is not None:
            for attr in (
                "psyche_plus_validity",
                "psyche_plus_axiom_count",
                "psyche_plus_codified",
                "safety_score",
                "winner_module",
            ):
                if hasattr(state, attr):
                    span_data[attr] = getattr(state, attr)

        if self.tracer is not None:
            with self.tracer.start_as_current_span(operation) as span:
                span.set_attribute("agione.quality.ecosystem", ecosystem)
                for domain, score in qrep.items():
                    span.set_attribute(f"agione.quality.{domain}", score)
                for k, v in span_data.items():
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(f"agione.{k}", v)
                span.add_event("quality_report", qrep)
                if span_data["regime"].get("regime") == "DEGENERATE":
                    span.add_event("degenerate_quality_regime", span_data["regime"])
                    span.set_status(StatusCode.ERROR, "Degenerate quality regime")
        else:
            self.memory_spans.append(span_data)

        return span_data

    def instrument_agione(self, agione: Any, orchestrator: Any = None) -> bool:
        """
        Monkey-patch agione.forward to enrich traces after each call.
        Restorable by calling restore_forward().
        """
        if not hasattr(agione, "forward"):
            return False

        if getattr(agione, "_sesi_quality_trace_instrumented", False):
            return True

        original_forward = agione.forward
        pipeline = self

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            try:
                enrichment = pipeline.enrich(
                    orchestrator=orchestrator or getattr(agione, "orchestrator", None),
                    output=out,
                    operation="agione.forward",
                )
                if isinstance(out, dict):
                    out["_sesi_trace_enrichment"] = enrichment
            except Exception as e:
                logger.debug("trace enrichment failed: %s", e)
            return out

        agione._sesi_original_forward = original_forward
        agione.forward = wrapped_forward
        agione._sesi_quality_trace_instrumented = True
        return True

    def restore_forward(self, agione: Any) -> bool:
        if hasattr(agione, "_sesi_original_forward"):
            agione.forward = agione._sesi_original_forward
            delattr(agione, "_sesi_original_forward")
            agione._sesi_quality_trace_instrumented = False
            return True
        return False


# =============================================================================
# Integrated Suite
# =============================================================================

class SESIIstioWebAuthnVaultJaegerSuite:
    """
    One-stop suite for:
    - Istio external AuthorizationPolicy
    - WebAuthn Vault Transit credential storage
    - Jaeger AGIONE quality enrichment
    """

    def __init__(self):
        self.istio_authz = IstioExternalAuthorizationPolicyGenerator()
        self.webauthn_store = WebAuthnVaultTransitCredentialStore()
        self.trace_pipeline = JaegerAGIONEQualityEnrichmentPipeline()

    def generate_artifacts(self, output_dir: str = "./sesi_istio_webauthn_jaeger") -> Dict[str, Any]:
        out = Path(output_dir)
        files = {}
        files["istio_authz"] = self.istio_authz.write_all(str(out / "istio_authz"))
        return files

    def status(self) -> Dict[str, Any]:
        return {
            "istio_rules": list(self.istio_authz.rules.keys()),
            "webauthn_store": self.webauthn_store.status(),
            "trace_spans_in_memory": len(self.trace_pipeline.memory_spans),
            "otel_available": HAS_OTEL,
        }


# =============================================================================
# Smoke test
# =============================================================================

def smoke_test():
    logging.basicConfig(level=logging.INFO)

    print("=" * 72)
    print("SESI Istio External AuthZ × WebAuthn Vault Transit × Jaeger AGIONE")
    print("=" * 72)
    print(f"PyYAML available : {HAS_YAML}")
    print(f"hvac available   : {HAS_HVAC}")
    print(f"OTEL available   : {HAS_OTEL}")

    suite = SESIIstioWebAuthnVaultJaegerSuite()

    # 1. Istio external AuthorizationPolicy
    print("\n── Istio External AuthorizationPolicy ──")
    artifacts = suite.generate_artifacts("./test_sesi_istio_webauthn_jaeger")
    print(f"Generated rule sets: {len(artifacts['istio_authz'])}")
    for name, path in artifacts["istio_authz"].items():
        print(f"  {name}: {path}")

    # 2. WebAuthn Vault Transit store
    print("\n── WebAuthn Vault Transit Credential Store ──")
    record = {
        "credential_id": "cred-demo-001",
        "user_id": "clinician-001",
        "rp_id": "sesi.example.com",
        "public_key_pem": "-----BEGIN PUBLIC KEY-----\nDEMO\n-----END PUBLIC KEY-----",
        "sign_count": 0,
        "aaguid": "adce0a90-1e59-4ecc-8c0b-6e020557d7bd",
        "attestation_format": "packed",
        "backup_eligible": True,
        "backup_state": False,
    }
    stored = suite.webauthn_store.store_credential(record)
    print(f"Stored: {stored}")
    loaded = suite.webauthn_store.load_credential("cred-demo-001")
    print(f"Loaded credential user: {loaded['user_id']}")
    updated = suite.webauthn_store.update_sign_count("cred-demo-001", 1)
    print(f"Sign count update: {updated}")
    clone = suite.webauthn_store.update_sign_count("cred-demo-001", 0)
    print(f"Clone regression check: {clone}")

    # 3. Jaeger AGIONE quality enrichment
    print("\n── Jaeger AGIONE Quality Enrichment ──")

    class MockOrchestrator:
        def quality_report(self):
            return {
                "ecosystem": 0.82,
                "physics": 0.9,
                "fold": 0.78,
                "mental": 0.75,
            }

    class MockState:
        ecosystem_quality_scores = {
            "ecosystem": 0.82,
            "physics": 0.9,
            "fold": 0.78,
            "mental": 0.75,
        }
        psyche_plus_validity = 0.73
        psyche_plus_axiom_count = 12
        psyche_plus_codified = True
        safety_score = 0.91
        winner_module = "language"

    class MockAGIONE:
        def __init__(self):
            self.orchestrator = MockOrchestrator()

        def forward(self, x):
            return {"output": x, "state": MockState()}

    agi = MockAGIONE()
    ok = suite.trace_pipeline.instrument_agione(agi, orchestrator=agi.orchestrator)
    print(f"Instrumented AGIONE: {ok}")

    out = agi.forward("hello")
    enrichment = out["_sesi_trace_enrichment"]
    print(f"Trace operation: {enrichment['operation']}")
    print(f"Ecosystem quality: {enrichment['ecosystem_quality']}")
    print(f"Regime: {enrichment['regime']}")

    print("\n── Suite Status ──")
    print(json.dumps(suite.status(), indent=2))

    print("\nHONEST SCOPE")
    print("• Istio policies are templates, not a running mesh.")
    print("• Vault Transit stores encrypted WebAuthn metadata, not private keys.")
    print("• Jaeger enrichment shows correlation, not causation.")
    print("• This module does not create or detect consciousness.")


if __name__ == "__main__":
    smoke_test()
