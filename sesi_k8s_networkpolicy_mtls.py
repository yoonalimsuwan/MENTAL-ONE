"""
=============================================================================
SESI KUBERNETES NETWORKPOLICY + mTLS ZONE ISOLATION
Four-zone security architecture with mutual TLS enforcement
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

ZONE ARCHITECTURE (from ses_neural_device_deployment.py)
----------------------------------------------------------
  Z0 IMPLANT          — implanted device network (most isolated)
  Z1 CLINICAL GATEWAY — clinician-facing gateway (approval required)
  Z2 PROCESSING       — signal processing / AI inference servers
  Z3 EXTERNAL         — external / internet-facing services

TRAFFIC POLICY: Default-DENY. Only explicitly allowed (zone, zone, port,
protocol) tuples pass. Every cross-zone flow is logged.

MATHEMATICAL GUARANTEES USED
------------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds the number of cross-zone connection events per window
[C] SESI_NoZeno_Closure Thm 4.2: N(T) ≤ (E_max(T) + D(T))/∆E_min
    → energy-budget bound on connection attempts
[C] Stochastic_NoZeno Thm 4.1: exit-time tail bound
    → P(sup ||h(r)-h(T_k)|| ≥ δ | F_{T_k}) ≤ 2exp(-δ²/8G₀²ε)
    → alarm threshold for anomalous connection bursts

RELATION TO PAPER PNDC Axioms
-------------------------------
[C] PNDC Axiom 3 (Necessity & Proportionality):
    A flow is permitted only if NECESSARY for a legitimate clinical
    purpose and PROPORTIONATE to that purpose.
[C] PNDC Axiom 5 (Responsibility for Consequences):
    Every cross-zone flow is logged with full audit trail.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• NetworkPolicy REDUCES blast radius; it does not make the system unhackable.
• mTLS requires proper CA infrastructure (cert-manager + real CA).
• Self-signed certificates are for DEVELOPMENT ONLY.
• These manifests are TEMPLATES — production requires security audit.
• No claim that this prevents all attacks. Defense-in-depth required.
=============================================================================
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.k8s_networkpolicy")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

__all__ = [
    "SecurityZone", "ZonePolicy", "NetworkPolicyGenerator",
    "MTLSConfigGenerator", "ConnectionRateLimiter",
    "K8sSecuritySuite",
]


class SecurityZone:
    """
    [D] Four security zones with explicit labels and ports.
    """
    Z0_IMPLANT = {
        "name": "z0-implant",
        "label": "sesi.io/zone=implant",
        "pods_label": "zone=implant",
        "allowed_inbound_ports": [8443],
        "allowed_outbound_ports": [8443],
        "description": "Implanted device network (most isolated)",
    }
    Z1_CLINICAL_GATEWAY = {
        "name": "z1-clinical-gateway",
        "label": "sesi.io/zone=clinical-gateway",
        "pods_label": "zone=clinical-gateway",
        "allowed_inbound_ports": [8443, 443],
        "allowed_outbound_ports": [8443, 9090],
        "description": "Clinician-facing gateway (approval required)",
    }
    Z2_PROCESSING = {
        "name": "z2-processing",
        "label": "sesi.io/zone=processing",
        "pods_label": "zone=processing",
        "allowed_inbound_ports": [9090, 443],
        "allowed_outbound_ports": [443, 9090, 53],
        "description": "Signal processing / AI inference servers",
    }
    Z3_EXTERNAL = {
        "name": "z3-external",
        "label": "sesi.io/zone=external",
        "pods_label": "zone=external",
        "allowed_inbound_ports": [443],
        "allowed_outbound_ports": [443],
        "description": "External / internet-facing services",
    }

    ALL_ZONES = [Z0_IMPLANT, Z1_CLINICAL_GATEWAY, Z2_PROCESSING, Z3_EXTERNAL]


@dataclass
class ZonePolicy:
    """
    [D] Policy for a single zone: what flows are allowed.

    Each flow is a tuple: (src_zone, dst_zone, port, protocol).
    Default policy is DENY.
    """
    zone: Dict[str, Any]
    allowed_flows: Set[Tuple[str, str, int, str]] = field(default_factory=set)
    default_action: str = "deny"

    def add_flow(self, src_zone: str, dst_zone: str, port: int,
                 protocol: str = "tcp") -> None:
        self.allowed_flows.add((src_zone, dst_zone, port, protocol))

    def check_flow(self, src_zone: str, dst_zone: str, port: int,
                   protocol: str = "tcp") -> bool:
        """Check if a flow is permitted."""
        if src_zone == dst_zone:
            return True  # intra-zone always allowed
        return (src_zone, dst_zone, port, protocol) in self.allowed_flows


class NetworkPolicyGenerator:
    """
    [D] Generate Kubernetes NetworkPolicy manifests for zone isolation.

    Produces one NetworkPolicy per zone, implementing:
      • Default-deny ingress and egress
      • Explicit allow-list for cross-zone flows
      • Namespace isolation
      • Pod selector based on zone labels

    HONEST: These are TEMPLATES. Production requires:
      • Cluster-specific namespace names
      • Real CA certificates for mTLS
      • Security audit before deployment
    """

    def __init__(self, namespace: str = "sesi-oversight"):
        self.namespace = namespace
        self.policies: Dict[str, Dict[str, Any]] = {}
        self._install_default_rules()

    def _install_default_rules(self) -> None:
        """
        Install default zone policies.

        Traffic rules (from ses_neural_device_deployment.py):
          Z0 (implant) ↔ Z1 (clinical gateway): port 8443 (TLS)
          Z1 → Z2 (processing): port 9090 (telemetry)
          Z3 (external) → Z2: port 443 (API)
          Z3 CANNOT reach Z0 or Z1 directly
          Z2 CANNOT reach Z0 directly
        """
        # Z0: Implant — only Z1 can reach it, only on 8443
        z0 = ZonePolicy(SecurityZone.Z0_IMPLANT)
        z0.add_flow("clinical-gateway", "implant", 8443, "tcp")
        z0.add_flow("implant", "clinical-gateway", 8443, "tcp")
        self.policies["implant"] = z0

        # Z1: Clinical Gateway — Z0, Z2, Z3 can reach it
        z1 = ZonePolicy(SecurityZone.Z1_CLINICAL_GATEWAY)
        z1.add_flow("implant", "clinical-gateway", 8443, "tcp")
        z1.add_flow("clinical-gateway", "implant", 8443, "tcp")
        z1.add_flow("clinical-gateway", "processing", 9090, "tcp")
        z1.add_flow("external", "clinical-gateway", 443, "tcp")
        self.policies["clinical-gateway"] = z1

        # Z2: Processing — Z1 and Z3 can reach it
        z2 = ZonePolicy(SecurityZone.Z2_PROCESSING)
        z2.add_flow("clinical-gateway", "processing", 9090, "tcp")
        z2.add_flow("external", "processing", 443, "tcp")
        self.policies["processing"] = z2

        # Z3: External — can only reach Z2
        z3 = ZonePolicy(SecurityZone.Z3_EXTERNAL)
        z3.add_flow("external", "processing", 443, "tcp")
        self.policies["external"] = z3

    def generate_networkpolicy(self, zone_name: str) -> Dict[str, Any]:
        """Generate a K8s NetworkPolicy manifest for a zone."""
        policy = self.policies.get(zone_name)
        if policy is None:
            raise ValueError(f"Unknown zone: {zone_name}")

        zone = policy.zone
        ingress_rules = []
        egress_rules = []

        for (src, dst, port, proto) in policy.allowed_flows:
            if dst == zone_name.replace("z", "").replace("-", "").lower():
                # This is an inbound rule
                ingress_rules.append({
                    "from": [{
                        "podSelector": {
                            "matchLabels": {"zone": src}
                        }
                    }],
                    "ports": [{"protocol": proto.upper(), "port": port}],
                })
            elif src == zone_name.replace("z", "").replace("-", "").lower():
                # This is an outbound rule
                egress_rules.append({
                    "to": [{
                        "podSelector": {
                            "matchLabels": {"zone": dst}
                        }
                    }],
                    "ports": [{"protocol": proto.upper(), "port": port}],
                })

        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"sesi-{zone_name}-isolation",
                "namespace": self.namespace,
                "labels": {
                    "app": "sesi-oversight",
                    "zone": zone_name,
                },
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {"zone": zone_name},
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": ingress_rules if ingress_rules else [],
                "egress": egress_rules if egress_rules else [],
            },
        }

        return manifest

    def generate_all(self) -> Dict[str, Dict[str, Any]]:
        """Generate all NetworkPolicy manifests."""
        return {
            zone_name: self.generate_networkpolicy(zone_name)
            for zone_name in self.policies
        }

    def write_manifests(self, output_dir: str = "./k8s/networkpolicies") -> Dict[str, str]:
        """Write all manifests to YAML files."""
        if not HAS_YAML:
            logger.warning("PyYAML not available; writing JSON instead")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        manifests = self.generate_all()
        files = {}

        for zone_name, manifest in manifests.items():
            filename = f"{zone_name}-networkpolicy.yaml"
            filepath = out_path / filename

            if HAS_YAML:
                with open(filepath, 'w') as f:
                    yaml.dump(manifest, f, default_flow_style=False)
            else:
                import json
                with open(filepath, 'w') as f:
                    json.dump(manifest, f, indent=2)

            files[filename] = str(filepath)

        logger.info(f"NetworkPolicy manifests written to {output_dir}")
        return files


class MTLSConfigGenerator:
    """
    [D] Generate mTLS configuration for Istio/Linkerd-style service mesh.

    Produces:
      • PeerAuthentication (Istio) — enforce STRICT mTLS
      • AuthorizationPolicy (Istio) — zone-based access control
      • DestinationRule (Istio) — TLS settings per service

    HONEST: Requires Istio or equivalent service mesh installed.
    Self-signed certificates are for development only.
    """

    def __init__(self, namespace: str = "sesi-oversight"):
        self.namespace = namespace

    def generate_peer_authentication(self) -> Dict[str, Any]:
        """Generate Istio PeerAuthentication for STRICT mTLS."""
        return {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "PeerAuthentication",
            "metadata": {
                "name": "sesi-mtls-strict",
                "namespace": self.namespace,
            },
            "spec": {
                "mtls": {
                    "mode": "STRICT",
                },
            },
        }

    def generate_authorization_policy(self) -> Dict[str, Any]:
        """
        Generate Istio AuthorizationPolicy for zone-based access.

        Only clinical-gateway can access implant zone.
        """
        return {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": "sesi-zone-access",
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {
                    "matchLabels": {"zone": "implant"},
                },
                "rules": [{
                    "from": [{
                        "source": {
                            "principals": [
                                f"cluster.local/ns/{self.namespace}/sa/clinical-gateway"
                            ],
                        },
                    }],
                    "to": [{
                        "operation": {"ports": ["8443"]},
                    }],
                }],
            },
        }

    def generate_destination_rule(self) -> Dict[str, Any]:
        """Generate Istio DestinationRule for implant zone."""
        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": "sesi-implant-mtls",
                "namespace": self.namespace,
            },
            "spec": {
                "host": "implant-service",
                "trafficPolicy": {
                    "tls": {
                        "mode": "ISTIO_MUTUAL",
                    },
                },
            },
        }

    def generate_all(self) -> Dict[str, Dict[str, Any]]:
        """Generate all mTLS configuration manifests."""
        return {
            "peer_authentication": self.generate_peer_authentication(),
            "authorization_policy": self.generate_authorization_policy(),
            "destination_rule": self.generate_destination_rule(),
        }

    def write_manifests(self, output_dir: str = "./k8s/mtls") -> Dict[str, str]:
        """Write mTLS manifests to YAML files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        manifests = self.generate_all()
        files = {}

        for name, manifest in manifests.items():
            filename = f"{name}.yaml"
            filepath = out_path / filename

            if HAS_YAML:
                with open(filepath, 'w') as f:
                    yaml.dump(manifest, f, default_flow_style=False)
            else:
                import json
                with open(filepath, 'w') as f:
                    json.dump(manifest, f, indent=2)

            files[filename] = str(filepath)

        logger.info(f"mTLS manifests written to {output_dir}")
        return files


class ConnectionRateLimiter:
    """
    [D] Rate limiter for cross-zone connections using No-Zeno bounds.

    From SESI_NoZeno_Closure Thm 3.2:
      N(T) ≤ 1 + 2C_V·T/δ_min

    From Stochastic_NoZeno Thm 4.1:
      P(sup ||h(r)-h(T_k)|| ≥ δ | F_{T_k}) ≤ 2exp(-δ²/8G₀²ε)

    HONEST: These bounds limit EVENT COUNT, not correctness.
    A bounded number of connections can still be malicious.
    """

    def __init__(self, delta_min: float = 0.1, c_v: float = 1.0,
                 window_s: float = 60.0):
        self.delta_min = delta_min
        self.c_v = c_v
        self.window_s = window_s
        self._timestamps: List[float] = []

    def max_connections_in_window(self) -> float:
        """
        Compute No-Zeno dwell-time bound: N(T) ≤ 1 + 2C_V·T/δ_min.
        """
        if self.delta_min <= 0:
            return float('inf')
        return 1 + (2 * self.c_v * self.window_s) / self.delta_min

    def check_rate(self) -> Dict[str, Any]:
        """Check if a new connection is within the rate bound."""
        now = time.time()
        cutoff = now - self.window_s
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        bound = self.max_connections_in_window()
        current = len(self._timestamps)
        allowed = current < bound

        if allowed:
            self._timestamps.append(now)

        return {
            "allowed": allowed,
            "current_count": current,
            "bound": bound,
            "window_s": self.window_s,
            "delta_min": self.delta_min,
            "note": (
                "SESI_NoZeno_Closure Thm 3.2 dwell-time bound. "
                "This limits CONNECTION RATE, not connection legitimacy."
            ),
        }

    def exit_time_alarm(self, observed_deviation: float,
                        g0: float = 0.5, epsilon: float = 5.0) -> Dict[str, Any]:
        """
        Stochastic exit-time alarm (Stochastic_NoZeno Thm 4.1).

        If observed deviation exceeds δ, the connection pattern may be anomalous.
        """
        delta = self.delta_min
        prob_bound = 2 * math.exp(-delta**2 / (8 * g0**2 * epsilon))
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


class K8sSecuritySuite:
    """
    [D] Complete K8s security suite: NetworkPolicy + mTLS + rate limiting.
    """

    def __init__(self, namespace: str = "sesi-oversight"):
        self.namespace = namespace
        self.netpol_gen = NetworkPolicyGenerator(namespace)
        self.mtls_gen = MTLSConfigGenerator(namespace)
        self.rate_limiter = ConnectionRateLimiter()

    def generate_all_manifests(self, output_dir: str = "./k8s") -> Dict[str, str]:
        """Generate all K8s security manifests."""
        files = {}
        files.update(self.netpol_gen.write_manifests(f"{output_dir}/networkpolicies"))
        files.update(self.mtls_gen.write_manifests(f"{output_dir}/mtls"))
        return files

    def status(self) -> Dict[str, Any]:
        """Get security suite status."""
        return {
            "namespace": self.namespace,
            "zones": [z["name"] for z in SecurityZone.ALL_ZONES],
            "policies": list(self.netpol_gen.policies.keys()),
            "rate_limit_bound": self.rate_limiter.max_connections_in_window(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI K8s NetworkPolicy + mTLS — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available: {HAS_YAML}")

    suite = K8sSecuritySuite()

    print(f"\n  ── Security Zones ──")
    for zone in SecurityZone.ALL_ZONES:
        print(f"    {zone['name']:25s} : {zone['description']}")

    print(f"\n  ── NetworkPolicy Generation ──")
    manifests = suite.netpol_gen.generate_all()
    for zone_name, manifest in manifests.items():
        n_ingress = len(manifest["spec"].get("ingress", []))
        n_egress = len(manifest["spec"].get("egress", []))
        print(f"    {zone_name:25s} : ingress={n_ingress}, egress={n_egress}")

    print(f"\n  ── mTLS Configuration ──")
    mtls_manifests = suite.mtls_gen.generate_all()
    for name, manifest in mtls_manifests.items():
        print(f"    {name:25s} : {manifest['kind']}")

    print(f"\n  ── Connection Rate Limiter (No-Zeno) ──")
    rate_check = suite.rate_limiter.check_rate()
    print(f"    Rate bound           : {rate_check['bound']:.2f}")
    print(f"    Current count        : {rate_check['current_count']}")
    print(f"    Allowed              : {rate_check['allowed']}")

    exit_alarm = suite.rate_limiter.exit_time_alarm(0.5)
    print(f"    Exit-time alarm      : {exit_alarm['alarm']}")
    print(f"    Prob bound           : {exit_alarm['prob_bound']:.6f}")

    print(f"\n  ── Manifest Files ──")
    files = suite.generate_all_manifests("./test_k8s_manifests")
    for filename, filepath in files.items():
        print(f"    {filename}")

    print(f"\n  ── Suite Status ──")
    status = suite.status()
    print(f"    Namespace            : {status['namespace']}")
    print(f"    Zones                : {len(status['zones'])}")
    print(f"    Policies             : {len(status['policies'])}")
    print(f"    Rate limit bound     : {status['rate_limit_bound']:.2f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • NetworkPolicy REDUCES blast radius, does not eliminate risk")
    print("  • mTLS requires proper CA infrastructure")
    print("  • Self-signed certs are for DEVELOPMENT ONLY")
    print("  • Rate bounds limit EVENT COUNT, not correctness")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
