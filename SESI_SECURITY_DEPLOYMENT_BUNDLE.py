"""
=============================================================================
SESI SECURITY DEPLOYMENT BUNDLE
Kubernetes NetworkPolicy + Istio mTLS + Grafana Security Dashboard + OIDC Auth
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE ADDS
---------------------
[1] Kubernetes / Istio mTLS zone deployment
    - Default-deny NetworkPolicy
    - Z0 implant zone isolation
    - Istio PeerAuthentication STRICT mTLS
    - Istio DestinationRule ISTIO_MUTUAL
    - Istio AuthorizationPolicy for service identity allow-listing

[2] Grafana security monitoring dashboard
    - OIDC auth failures
    - mTLS failures
    - Cross-zone denied flows
    - Implant approval queue
    - Emergency isolation
    - JWKS/OIDC health
    - Prometheus alert rules

[3] OAuth2/OIDC authentication layer
    - Bearer JWT validation via JWKS
    - issuer / audience validation
    - scope-based access control
    - role + attribute policy checks
    - clinician approval gate for implant commands
    - no autonomous actuation

IMPORTANT
---------
• Kubernetes NetworkPolicy does NOT enforce mTLS by itself.
  It controls L3/L4 traffic. mTLS is enforced by Istio/Linkerd or by
  the application TLS stack.

• This module does not control real implants. It provides secure
  deployment and approval scaffolding. Real device control requires
  certified hardware, vendor SDKs, clinical governance, and regulatory
  approval.

• All thresholds and policies are FREE DESIGN PARAMETERS.
  Review them before deployment.

LABELS:
  [C] cited / standard practice
  [D] derived integration scaffold
  [O] open / requires real-world validation
=============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("sesi.security_bundle")

# Optional dependencies
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from fastapi import FastAPI, Depends, HTTPException, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import jwt
    from jwt import PyJWKClient
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

try:
    from prometheus_client import Counter, Gauge, Histogram
    HAS_PROM = True
except ImportError:
    HAS_PROM = False


__all__ = [
    "SecurityBundleConfig",
    "OIDCConfig",
    "OIDCIdentity",
    "OIDCAuthenticator",
    "SecurityMetrics",
    "KubernetesMTLSManifestGenerator",
    "GrafanaSecurityDashboardGenerator",
    "SESISecurityFastAPIIntegration",
    "SESISecurityDeploymentBundle",
]


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class OIDCConfig:
    """
    OAuth2/OIDC configuration.

    Production examples:
      issuer_url:
        https://accounts.google.com
        https://login.microsoftonline.com/<tenant>/v2.0
        https://your-keycloak.example.com/realms/sesi

    HONEST:
      This validates tokens. It does not replace institutional identity
      governance, license verification, or clinician credentialing.
    """
    issuer_url: str = "https://your-idp.example.com/realms/sesi"
    audience: str = "sesi-neural-api"
    jwks_url: Optional[str] = None
    algorithms: List[str] = field(default_factory=lambda: ["RS256", "ES256"])

    # Required claims for high-risk actions
    require_mfa_claim: bool = True
    required_acr_for_implant: str = "urn:sesi:acr:aal3"

    # Claim names
    scopes_claim: str = "scope"
    roles_claim: str = "roles"
    groups_claim: str = "groups"
    acr_claim: str = "acr"
    amr_claim: str = "amr"

    # Token clock skew
    leeway_s: int = 30


@dataclass
class SecurityBundleConfig:
    """
    Deployment and security configuration.
    """
    namespace: str = "sesi-neural"
    app_name: str = "sesi-neural"

    # Zone labels
    implant_zone_label: str = "implant"
    clinical_zone_label: str = "clinical-gateway"
    processing_zone_label: str = "processing"
    external_zone_label: str = "external"

    # mTLS / service mesh
    enable_istio: bool = True
    istio_mode: str = "STRICT"
    implant_control_port: int = 8443
    telemetry_port: int = 9090
    api_port: int = 8080
    websocket_port: int = 8765

    # OIDC
    oidc: OIDCConfig = field(default_factory=OIDCConfig)

    # Safety
    require_clinician_approval: bool = True
    allow_autonomous_implant_actuation: bool = False

    # Output
    output_dir: str = "./sesi_security_bundle"


# =============================================================================
# OIDC IDENTITY / AUTH
# =============================================================================

@dataclass
class OIDCIdentity:
    """
    Authenticated identity extracted from a validated JWT.
    """
    subject: str
    issuer: str
    audience: str
    scopes: Set[str]
    roles: Set[str]
    groups: Set[str]
    acr: Optional[str]
    amr: List[str]
    raw_claims: Dict[str, Any]

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes

    def has_role(self, role: str) -> bool:
        return role in self.roles or "admin" in self.roles

    def is_mfa_verified(self) -> bool:
        return any(x.lower() in {"mfa", "otp", "hwk", "webauthn"} for x in self.amr)


class SecurityMetrics:
    """
    Prometheus metrics for security monitoring.
    """

    def __init__(self):
        if not HAS_PROM:
            self.enabled = False
            return

        self.enabled = True

        self.auth_success = Counter(
            "sesi_auth_success_total",
            "Successful authentications",
            ["issuer"],
        )
        self.auth_failure = Counter(
            "sesi_auth_failure_total",
            "Failed authentications",
            ["reason"],
        )
        self.authz_denied = Counter(
            "sesi_authz_denied_total",
            "Authorization denials",
            ["required_scope"],
        )
        self.mtls_failures = Counter(
            "sesi_mtls_handshake_failures_total",
            "mTLS handshake failures",
            ["src_zone", "dst_zone"],
        )
        self.cross_zone_denied = Counter(
            "sesi_cross_zone_denied_total",
            "Denied cross-zone flows",
            ["src_zone", "dst_zone"],
        )
        self.implant_pending = Gauge(
            "sesi_implant_commands_pending_approval",
            "Implant commands pending clinician approval",
        )
        self.implant_rejected = Counter(
            "sesi_implant_commands_rejected_total",
            "Rejected implant commands",
            ["reason"],
        )
        self.emergency_isolation = Gauge(
            "sesi_emergency_isolation_active",
            "Emergency isolation active flag",
        )
        self.jwks_health = Gauge(
            "sesi_jwks_health",
            "JWKS health: 1 healthy, 0 unhealthy",
        )
        self.request_latency = Histogram(
            "sesi_security_request_duration_seconds",
            "Security request latency",
            ["operation"],
        )


class OIDCAuthenticator:
    """
    OAuth2/OIDC Bearer JWT authenticator.

    Validates:
      • signature via JWKS
      • issuer
      • audience
      • exp / nbf / iat
      • scopes / roles
      • ACR/AMR for high-risk implant operations

    HONEST:
      Passing token validation does NOT imply clinical authorization.
      Clinical privileges must be verified separately.
    """

    def __init__(self, cfg: OIDCConfig, metrics: Optional[SecurityMetrics] = None):
        self.cfg = cfg
        self.metrics = metrics
        self.jwks_url = cfg.jwks_url or self._derive_jwks_url(cfg.issuer_url)
        self._jwks_client = None

        if HAS_JWT:
            try:
                self._jwks_client = PyJWKClient(self.jwks_url)
                if self.metrics and self.metrics.enabled:
                    self.metrics.jwks_health.set(1)
            except Exception as e:
                logger.warning(f"JWKS client init failed: {e}")
                if self.metrics and self.metrics.enabled:
                    self.metrics.jwks_health.set(0)
        else:
            logger.warning("PyJWT not installed; OIDC authenticator unavailable")

    @staticmethod
    def _derive_jwks_url(issuer_url: str) -> str:
        """
        Derive common OIDC JWKS URL.

        For production, prefer explicit discovery:
          {issuer}/.well-known/openid-configuration
        """
        issuer = issuer_url.rstrip("/")
        if "login.microsoftonline.com" in issuer:
            return issuer + "/discovery/v2.0/keys"
        if issuer.endswith("/realms/sesi"):
            return issuer + "/protocol/openid-connect/certs"
        return issuer + "/.well-known/jwks.json"

    def validate_token(self, token: str) -> OIDCIdentity:
        if not HAS_JWT or self._jwks_client is None:
            self._fail("jwt_library_unavailable")
            raise RuntimeError("PyJWT / JWKS client unavailable")

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.cfg.algorithms,
                audience=self.cfg.audience,
                issuer=self.cfg.issuer_url,
                leeway=self.cfg.leeway_s,
                options={
                    "require": ["exp", "iat", "iss", "aud", "sub"],
                },
            )
        except Exception as e:
            self._fail(type(e).__name__)
            raise

        scopes = self._parse_scopes(claims.get(self.cfg.scopes_claim, ""))
        roles = set(claims.get(self.cfg.roles_claim, []) or [])
        groups = set(claims.get(self.cfg.groups_claim, []) or [])
        acr = claims.get(self.cfg.acr_claim)
        amr = claims.get(self.cfg.amr_claim, []) or []

        ident = OIDCIdentity(
            subject=claims["sub"],
            issuer=claims["iss"],
            audience=claims["aud"] if isinstance(claims["aud"], str) else ",".join(claims["aud"]),
            scopes=scopes,
            roles=roles,
            groups=groups,
            acr=acr,
            amr=amr,
            raw_claims=claims,
        )

        if self.metrics and self.metrics.enabled:
            self.metrics.auth_success.labels(issuer=ident.issuer).inc()

        return ident

    def require_scope(self, identity: OIDCIdentity, scope: str) -> None:
        if not identity.has_scope(scope):
            if self.metrics and self.metrics.enabled:
                self.metrics.authz_denied.labels(required_scope=scope).inc()
            raise PermissionError(f"Missing required scope: {scope}")

    def require_implant_high_assurance(self, identity: OIDCIdentity) -> None:
        """
        High-risk implant operations require:
          • implant scope
          • clinician role
          • MFA
          • AAL3-like ACR
        """
        self.require_scope(identity, "implant:propose")

        if not identity.has_role("clinician"):
            self._fail("missing_clinician_role")
            raise PermissionError("Clinician role required")

        if self.cfg.require_mfa_claim and not identity.is_mfa_verified():
            self._fail("mfa_required")
            raise PermissionError("MFA required for implant operation")

        if identity.acr != self.cfg.required_acr_for_implant:
            self._fail("acr_insufficient")
            raise PermissionError("High-assurance ACR required for implant operation")

    def _parse_scopes(self, raw: Any) -> Set[str]:
        if isinstance(raw, str):
            return set(raw.split())
        if isinstance(raw, list):
            return set(raw)
        return set()

    def _fail(self, reason: str) -> None:
        if self.metrics and self.metrics.enabled:
            self.metrics.auth_failure.labels(reason=reason).inc()


# =============================================================================
# KUBERNETES / ISTIO MTLS MANIFEST GENERATOR
# =============================================================================

class KubernetesMTLSManifestGenerator:
    """
    Generate:
      • Namespace
      • Default-deny NetworkPolicy
      • Implant-zone NetworkPolicy
      • Clinical-gateway NetworkPolicy
      • Processing-zone NetworkPolicy
      • Istio PeerAuthentication STRICT
      • Istio DestinationRule ISTIO_MUTUAL
      • Istio AuthorizationPolicy for Z1 → Z0
    """

    def __init__(self, cfg: SecurityBundleConfig):
        self.cfg = cfg

    def namespace(self) -> Dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.cfg.namespace,
                "labels": {
                    "istio-injection": "enabled" if self.cfg.enable_istio else "disabled",
                    "sesi.io/security-zone-model": "enabled",
                },
            },
        }

    def default_deny(self) -> Dict[str, Any]:
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "sesi-default-deny",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
            },
        }

    def implant_policy(self) -> Dict[str, Any]:
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "sesi-z0-implant-isolation",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {"sesi.io/zone": self.cfg.implant_zone_label}
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [{
                    "from": [{
                        "podSelector": {
                            "matchLabels": {
                                "sesi.io/zone": self.cfg.clinical_zone_label
                            }
                        }
                    }],
                    "ports": [{
                        "protocol": "TCP",
                        "port": self.cfg.implant_control_port,
                    }],
                }],
                "egress": [{
                    "to": [{
                        "podSelector": {
                            "matchLabels": {
                                "sesi.io/zone": self.cfg.clinical_zone_label
                            }
                        }
                    }],
                    "ports": [{
                        "protocol": "TCP",
                        "port": self.cfg.implant_control_port,
                    }],
                }],
            },
        }

    def clinical_gateway_policy(self) -> Dict[str, Any]:
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "sesi-z1-clinical-gateway",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {"sesi.io/zone": self.cfg.clinical_zone_label}
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [{
                            "podSelector": {
                                "matchLabels": {
                                    "sesi.io/zone": self.cfg.processing_zone_label
                                }
                            }
                        }],
                        "ports": [{"protocol": "TCP", "port": self.cfg.api_port}],
                    },
                    {
                        "from": [{
                            "podSelector": {
                                "matchLabels": {
                                    "sesi.io/zone": self.cfg.implant_zone_label
                                }
                            }
                        }],
                        "ports": [{"protocol": "TCP", "port": self.cfg.implant_control_port}],
                    },
                ],
                "egress": [
                    {
                        "to": [{
                            "podSelector": {
                                "matchLabels": {
                                    "sesi.io/zone": self.cfg.implant_zone_label
                                }
                            }
                        }],
                        "ports": [{"protocol": "TCP", "port": self.cfg.implant_control_port}],
                    },
                    {
                        "to": [{
                            "podSelector": {
                                "matchLabels": {
                                    "sesi.io/zone": self.cfg.processing_zone_label
                                }
                            }
                        }],
                        "ports": [{"protocol": "TCP", "port": self.cfg.telemetry_port}],
                    },
                ],
            },
        }

    def processing_policy(self) -> Dict[str, Any]:
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "sesi-z2-processing",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {"sesi.io/zone": self.cfg.processing_zone_label}
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [{
                            "namespaceSelector": {
                                "matchLabels": {"kubernetes.io/metadata.name": "ingress-nginx"}
                            }
                        }],
                        "ports": [{"protocol": "TCP", "port": self.cfg.api_port}],
                    },
                    {
                        "from": [{
                            "podSelector": {
                                "matchLabels": {
                                    "sesi.io/zone": self.cfg.clinical_zone_label
                                }
                            }
                        }],
                        "ports": [{"protocol": "TCP", "port": self.cfg.telemetry_port}],
                    },
                ],
                "egress": [
                    {
                        "to": [{
                            "podSelector": {
                                "matchLabels": {
                                    "sesi.io/zone": self.cfg.clinical_zone_label
                                }
                            }
                        }],
                        "ports": [{"protocol": "TCP", "port": self.cfg.api_port}],
                    },
                    {
                        "to": [{"namespaceSelector": {}}],
                        "ports": [
                            {"protocol": "TCP", "port": 443},
                            {"protocol": "UDP", "port": 53},
                            {"protocol": "TCP", "port": 53},
                        ],
                    },
                ],
            },
        }

    def istio_peer_auth(self) -> Dict[str, Any]:
        return {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "PeerAuthentication",
            "metadata": {
                "name": "sesi-strict-mtls",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "mtls": {"mode": self.cfg.istio_mode}
            },
        }

    def istio_destination_rule(self) -> Dict[str, Any]:
        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": "sesi-istio-mutual",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "host": f"*.{self.cfg.namespace}.svc.cluster.local",
                "trafficPolicy": {
                    "tls": {"mode": "ISTIO_MUTUAL"}
                },
            },
        }

    def istio_authz_implant(self) -> Dict[str, Any]:
        return {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": "allow-clinical-gateway-to-implant",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "selector": {
                    "matchLabels": {"sesi.io/zone": self.cfg.implant_zone_label}
                },
                "action": "ALLOW",
                "rules": [{
                    "from": [{
                        "source": {
                            "principals": [
                                f"cluster.local/ns/{self.cfg.namespace}/sa/sesi-clinical-gateway"
                            ]
                        }
                    }],
                    "to": [{
                        "operation": {
                            "ports": [str(self.cfg.implant_control_port)]
                        }
                    }],
                }],
            },
        }

    def generate_all(self) -> Dict[str, Any]:
        docs = [
            self.namespace(),
            self.default_deny(),
            self.implant_policy(),
            self.clinical_gateway_policy(),
            self.processing_policy(),
        ]
        if self.cfg.enable_istio:
            docs.extend([
                self.istio_peer_auth(),
                self.istio_destination_rule(),
                self.istio_authz_implant(),
            ])
        return {"documents": docs}


# =============================================================================
# GRAFANA SECURITY DASHBOARD GENERATOR
# =============================================================================

class GrafanaSecurityDashboardGenerator:
    """
    Generate Grafana dashboard JSON + Prometheus alert rules.

    Metrics expected:
      app:
        sesi_auth_failure_total
        sesi_authz_denied_total
        sesi_mtls_handshake_failures_total
        sesi_cross_zone_denied_total
        sesi_implant_commands_pending_approval
        sesi_implant_commands_rejected_total
        sesi_emergency_isolation_active
        sesi_jwks_health

      Istio:
        istio_requests_total
        istio_tcp_connections_opened_total
    """

    def __init__(self, cfg: SecurityBundleConfig):
        self.cfg = cfg

    def dashboard(self) -> Dict[str, Any]:
        ds = {"type": "prometheus", "uid": "prometheus"}

        def panel(pid: int, title: str, expr: str, kind: str, x: int, y: int, w: int = 8, h: int = 6):
            return {
                "id": pid,
                "type": kind,
                "title": title,
                "datasource": ds,
                "targets": [{"expr": expr, "refId": "A"}],
                "gridPos": {"x": x, "y": y, "w": w, "h": h},
            }

        panels = [
            panel(1, "Auth Failures / min",
                  'sum(rate(sesi_auth_failure_total[5m])) * 60',
                  "timeseries", 0, 0),
            panel(2, "AuthZ Denials / min",
                  'sum(rate(sesi_authz_denied_total[5m])) * 60',
                  "timeseries", 8, 0),
            panel(3, "mTLS Failures / min",
                  'sum(rate(sesi_mtls_handshake_failures_total[5m])) * 60',
                  "timeseries", 16, 0),
            panel(4, "Cross-Zone Denied Flows",
                  'sum(rate(sesi_cross_zone_denied_total[5m])) by (src_zone,dst_zone)',
                  "timeseries", 0, 6, 12),
            panel(5, "Implant Commands Pending Approval",
                  'sesi_implant_commands_pending_approval',
                  "stat", 12, 6, 6),
            panel(6, "Emergency Isolation Active",
                  'sesi_emergency_isolation_active',
                  "stat", 18, 6, 6),
            panel(7, "JWKS Health",
                  'sesi_jwks_health',
                  "stat", 0, 12, 6),
            panel(8, "Implant Commands Rejected / min",
                  'sum(rate(sesi_implant_commands_rejected_total[5m])) by (reason) * 60',
                  "timeseries", 6, 12, 18),
            panel(9, "Istio 5xx Rate",
                  f'sum(rate(istio_requests_total{{destination_service_namespace="{self.cfg.namespace}",response_code=~"5.."}}[5m]))',
                  "timeseries", 0, 18, 12),
            panel(10, "Istio TCP Connections Opened",
                  f'sum(rate(istio_tcp_connections_opened_total{{destination_service_namespace="{self.cfg.namespace}"}}[5m]))',
                  "timeseries", 12, 18, 12),
        ]

        return {
            "uid": "sesi-security-monitoring",
            "title": "SESI Security Monitoring — mTLS / OIDC / Implant Zones",
            "tags": ["sesi", "security", "mtls", "oidc", "neural"],
            "schemaVersion": 39,
            "version": 1,
            "refresh": "10s",
            "time": {"from": "now-6h", "to": "now"},
            "panels": panels,
        }

    def prometheus_rules(self) -> Dict[str, Any]:
        return {
            "groups": [{
                "name": "sesi-security.rules",
                "rules": [
                    {
                        "alert": "SESI_AuthFailureSpike",
                        "expr": "sum(rate(sesi_auth_failure_total[5m])) > 1",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Authentication failure spike",
                            "description": "More than 1 auth failure/sec over 5 minutes.",
                        },
                    },
                    {
                        "alert": "SESI_mTLSFailureSpike",
                        "expr": "sum(rate(sesi_mtls_handshake_failures_total[5m])) > 0.2",
                        "for": "3m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "mTLS handshake failures detected",
                            "description": "Possible certificate issue or zone probing.",
                        },
                    },
                    {
                        "alert": "SESI_ImplantZoneDeniedFlow",
                        "expr": 'sum(rate(sesi_cross_zone_denied_total{dst_zone="Z0_IMPLANT"}[5m])) > 0',
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Denied flow targeting implant zone",
                            "description": "Investigate immediately. Z0 should only be reachable from Z1.",
                        },
                    },
                    {
                        "alert": "SESI_EmergencyIsolationActive",
                        "expr": "sesi_emergency_isolation_active == 1",
                        "for": "1m",
                        "labels": {"severity": "emergency"},
                        "annotations": {
                            "summary": "Emergency isolation active",
                            "description": "Implant zone is isolated. Clinical review required.",
                        },
                    },
                    {
                        "alert": "SESI_JWKSUnhealthy",
                        "expr": "sesi_jwks_health == 0",
                        "for": "2m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "OIDC JWKS unavailable",
                            "description": "Token validation may fail. Check IdP connectivity.",
                        },
                    },
                ],
            }]
        }


# =============================================================================
# FASTAPI OIDC INTEGRATION
# =============================================================================

class SESISecurityFastAPIIntegration:
    """
    Add OIDC-protected routes to an existing FastAPI app.

    Routes added:
      GET  /api/v1/auth/me
      POST /api/v1/implant/propose
      POST /api/v1/implant/approve
      POST /api/v1/security/emergency-isolate
      GET  /api/v1/security/status

    Implant propose/approve are workflow endpoints only.
    No direct actuation is performed here.
    """

    def __init__(
        self,
        app: Any,
        cfg: SecurityBundleConfig,
        metrics: Optional[SecurityMetrics] = None,
    ):
        if not HAS_FASTAPI:
            raise RuntimeError("FastAPI is not installed")

        self.app = app
        self.cfg = cfg
        self.metrics = metrics or SecurityMetrics()
        self.auth = OIDCAuthenticator(cfg.oidc, self.metrics)
        self.bearer = HTTPBearer(auto_error=True)
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.isolated = False

        self._install_routes()

    def _identity_from_request(
        self,
        credentials: Any,
    ) -> OIDCIdentity:
        try:
            return self.auth.validate_token(credentials.credentials)
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    def _install_routes(self):
        cfg = self.cfg
        metrics = self.metrics
        auth = self.auth
        bearer = self.bearer

        class ImplantProposal(BaseModel):
            device_id: str
            action: str
            parameters: Dict[str, float] = Field(default_factory=dict)
            patient_id: str
            rationale: str

        class ApprovalDecision(BaseModel):
            proposal_id: str
            approved: bool
            reason: str

        @self.app.get("/api/v1/auth/me")
        async def me(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
            ident = self._identity_from_request(credentials)
            return {
                "subject": ident.subject,
                "issuer": ident.issuer,
                "scopes": sorted(ident.scopes),
                "roles": sorted(ident.roles),
                "acr": ident.acr,
                "amr": ident.amr,
            }

        @self.app.post("/api/v1/implant/propose")
        async def propose_implant_command(
            req: ImplantProposal,
            credentials: HTTPAuthorizationCredentials = Depends(bearer),
        ):
            ident = self._identity_from_request(credentials)

            try:
                auth.require_implant_high_assurance(ident)
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e))

            if cfg.allow_autonomous_implant_actuation:
                # Should never happen in production
                raise HTTPException(
                    status_code=500,
                    detail="Configuration violates safety policy: autonomous actuation forbidden",
                )

            proposal_id = f"proposal-{int(time.time())}-{len(self.pending_approvals)}"
            proposal = {
                "proposal_id": proposal_id,
                "device_id": req.device_id,
                "action": req.action,
                "parameters": req.parameters,
                "patient_id": req.patient_id,
                "rationale": req.rationale,
                "proposed_by": ident.subject,
                "status": "PENDING_CLINICIAN_APPROVAL",
                "created_at": time.time(),
            }
            self.pending_approvals[proposal_id] = proposal

            if metrics.enabled:
                metrics.implant_pending.set(len(self.pending_approvals))

            return {
                "proposal_id": proposal_id,
                "status": "PENDING_CLINICIAN_APPROVAL",
                "note": "This endpoint does not actuate devices.",
            }

        @self.app.post("/api/v1/implant/approve")
        async def approve_implant_command(
            decision: ApprovalDecision,
            credentials: HTTPAuthorizationCredentials = Depends(bearer),
        ):
            ident = self._identity_from_request(credentials)
            try:
                auth.require_scope(ident, "implant:approve")
                if not ident.has_role("clinician"):
                    raise PermissionError("Clinician role required")
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e))

            proposal = self.pending_approvals.get(decision.proposal_id)
            if proposal is None:
                raise HTTPException(status_code=404, detail="Proposal not found")

            proposal["status"] = "APPROVED" if decision.approved else "REJECTED"
            proposal["reviewed_by"] = ident.subject
            proposal["review_reason"] = decision.reason
            proposal["reviewed_at"] = time.time()

            if metrics.enabled:
                metrics.implant_pending.set(
                    sum(1 for p in self.pending_approvals.values()
                        if p["status"] == "PENDING_CLINICIAN_APPROVAL")
                )

            return {
                "proposal_id": decision.proposal_id,
                "status": proposal["status"],
                "note": (
                    "Approval recorded. Execution requires separate certified "
                    "device gateway and clinical workflow."
                ),
            }

        @self.app.post("/api/v1/security/emergency-isolate")
        async def emergency_isolate(
            reason: str,
            credentials: HTTPAuthorizationCredentials = Depends(bearer),
        ):
            ident = self._identity_from_request(credentials)
            try:
                auth.require_scope(ident, "security:emergency_stop")
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e))

            self.isolated = True
            if metrics.enabled:
                metrics.emergency_isolation.set(1)

            return {
                "status": "ISOLATED",
                "reason": reason,
                "initiated_by": ident.subject,
                "timestamp": time.time(),
            }

        @self.app.get("/api/v1/security/status")
        async def security_status():
            return {
                "isolated": self.isolated,
                "pending_approvals": len(self.pending_approvals),
                "oidc_issuer": cfg.oidc.issuer_url,
                "mtls_enabled": cfg.enable_istio,
                "namespace": cfg.namespace,
                "note": "Security status only; no device actuation.",
            }


# =============================================================================
# ORCHESTRATOR / FILE GENERATION
# =============================================================================

class SESISecurityDeploymentBundle:
    """
    Generate deployment artifacts:
      • Kubernetes NetworkPolicy + Istio mTLS manifests
      • Grafana dashboard JSON
      • Prometheus alert rules
      • OIDC config example
    """

    def __init__(self, cfg: Optional[SecurityBundleConfig] = None):
        self.cfg = cfg or SecurityBundleConfig()
        self.out = Path(self.cfg.output_dir)
        self.k8s = KubernetesMTLSManifestGenerator(self.cfg)
        self.grafana = GrafanaSecurityDashboardGenerator(self.cfg)
        self.metrics = SecurityMetrics()

    def _write_yaml(self, path: Path, docs: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_YAML:
            path.write_text(
                "\n---\n".join(yaml.safe_dump(d, sort_keys=False) for d in docs),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(docs, indent=2), encoding="utf-8")

    def generate(self) -> Dict[str, str]:
        self.out.mkdir(parents=True, exist_ok=True)

        k8s_docs = self.k8s.generate_all()["documents"]
        k8s_path = self.out / "k8s" / "sesi-mtls-networkpolicies.yaml"
        self._write_yaml(k8s_path, k8s_docs)

        dashboard_path = self.out / "monitoring" / "grafana-security-dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(
            json.dumps(self.grafana.dashboard(), indent=2),
            encoding="utf-8",
        )

        rules_path = self.out / "monitoring" / "prometheus-security-rules.yaml"
        self._write_yaml(rules_path, [self.grafana.prometheus_rules()])

        oidc_example = {
            "issuer_url": self.cfg.oidc.issuer_url,
            "audience": self.cfg.oidc.audience,
            "jwks_url": self.cfg.oidc.jwks_url or self.cfg.oidc.issuer_url.rstrip("/") + "/.well-known/jwks.json",
            "required_scopes": {
                "read_status": ["security:read"],
                "propose_implant": ["implant:propose"],
                "approve_implant": ["implant:approve"],
                "emergency_stop": ["security:emergency_stop"]
            },
            "required_roles": {
                "implant_approve": ["clinician"],
                "remote_control": ["clinician"]
            },
            "required_acr_for_implant": self.cfg.oidc.required_acr_for_implant,
            "note": "Example only. Configure for your IdP."
        }
        oidc_path = self.out / "auth" / "oidc-config.example.json"
        oidc_path.parent.mkdir(parents=True, exist_ok=True)
        oidc_path.write_text(json.dumps(oidc_example, indent=2), encoding="utf-8")

        readme = self.out / "README.md"
        readme.write_text(
            "# SESI Security Deployment Bundle\n\n"
            "## Files\n"
            "- k8s/sesi-mtls-networkpolicies.yaml\n"
            "- monitoring/grafana-security-dashboard.json\n"
            "- monitoring/prometheus-security-rules.yaml\n"
            "- auth/oidc-config.example.json\n\n"
            "## Important\n"
            "NetworkPolicy enforces L3/L4 isolation. mTLS requires Istio/Linkerd "
            "or application-level certificate verification.\n\n"
            "Implant control is not performed by these manifests. They provide "
            "security scaffolding and approval workflow only.\n",
            encoding="utf-8",
        )

        return {
            "k8s": str(k8s_path),
            "grafana": str(dashboard_path),
            "prometheus_rules": str(rules_path),
            "oidc_example": str(oidc_path),
            "readme": str(readme),
        }

    def attach_to_fastapi(self, app: Any) -> SESISecurityFastAPIIntegration:
        """
        Attach OIDC-protected routes to an existing FastAPI app.
        """
        return SESISecurityFastAPIIntegration(app, self.cfg, self.metrics)


# =============================================================================
# SMOKE TEST
# =============================================================================

def smoke_test():
    print("=" * 72)
    print(" SESI Security Deployment Bundle — Smoke Test")
    print("=" * 72)
    print(f"  FastAPI available : {HAS_FASTAPI}")
    print(f"  PyJWT available   : {HAS_JWT}")
    print(f"  YAML available    : {HAS_YAML}")
    print(f"  Prometheus client : {HAS_PROM}")

    cfg = SecurityBundleConfig(
        namespace="sesi-neural",
        enable_istio=True,
        oidc=OIDCConfig(
            issuer_url="https://your-idp.example.com/realms/sesi",
            audience="sesi-neural-api",
        ),
    )

    bundle = SESISecurityDeploymentBundle(cfg)
    files = bundle.generate()

    print("\n  ── Generated Files ──")
    for name, path in files.items():
        print(f"    {name:18s}: {path}")

    print("\n  ── Kubernetes / Istio Policies ──")
    docs = bundle.k8s.generate_all()["documents"]
    for d in docs:
        print(f"    {d['kind']:24s} {d['metadata']['name']}")

    print("\n  ── Grafana Dashboard ──")
    dashboard = bundle.grafana.dashboard()
    print(f"    Dashboard title : {dashboard['title']}")
    print(f"    Panels          : {len(dashboard['panels'])}")

    print("\n  ── OIDC Layer ──")
    auth = OIDCAuthenticator(cfg.oidc, bundle.metrics)
    print(f"    Issuer          : {cfg.oidc.issuer_url}")
    print(f"    JWKS URL        : {auth.jwks_url}")
    print(f"    Required ACR    : {cfg.oidc.required_acr_for_implant}")

    if HAS_FASTAPI:
        print("\n  ── FastAPI Integration ──")
        app = FastAPI(title="SESI Test API")
        integration = bundle.attach_to_fastapi(app)
        routes = sorted([r.path for r in app.routes])
        print(f"    Routes        : {len(routes)}")
        for r in routes:
            print(f"      {r}")
    else:
        print("\n  [SKIP] FastAPI not installed")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • NetworkPolicy = L3/L4 isolation, not mTLS")
    print("  • mTLS requires Istio/Linkerd or app-level cert verification")
    print("  • OIDC verifies identity, not clinical privilege by itself")
    print("  • Implant endpoints are approval workflow only")
    print("  • No real device actuation is implemented")
    print("  • Regulatory approval is separate and mandatory")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
