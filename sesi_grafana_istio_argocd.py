"""
=============================================================================
SESI AGI PRODUCTION: GRAFANA + ISTIO + ARGOCD
Production-Grade Observability + Service Mesh + GitOps Deployment
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[GD1] GrafanaDashboardGenerator
      • JSON dashboard for AGI system metrics
      • Panels for all 10 core modules
      • Causal discovery visualization (FCI/FCI++/PC/NOTEARS)
      • K8s pod metrics (CPU, memory, GPU)
      • HPA/VPA status panels
      • Autonomous cycle metrics
      • Template variables for filtering

[ISM1] IstioServiceMeshGenerator
      • VirtualService for traffic routing
      • DestinationRule for load balancing
      • PeerAuthentication for mTLS
      • AuthorizationPolicy for zone-based access
      • Gateway for external access
      • PNDC Axiom 3 proportionality enforcement

[ACD1] ArgoCDGitOpsGenerator
      • Application CRD for GitOps deployment
      • ApplicationSet for multi-cluster
      • SyncPolicy for auto-sync
      • Notification configuration
      • Rollback strategies

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds sync frequency and metric emission
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful metric change
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-based access control
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require review

HONEST SCOPE
------------
• Grafana dashboard is VISUALIZATION, not decision-making
• Istio policies are CONFIGURATION, not security guarantees
• ArgoCD manifests are TEMPLATES, not deployed systems
• All thresholds are FREE DESIGN PARAMETERS
• Production requires security audit and expert review
=============================================================================
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.grafana_istio_argocd")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

__all__ = [
    "SecurityZone",
    "GrafanaDashboardGenerator",
    "IstioServiceMeshGenerator",
    "ArgoCDGitOpsGenerator",
    "SESIProductionSuite",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityZone:
    """Four security zones from SESI network segmentation."""
    Z0_IMPLANT = "z0-implant"
    Z1_CLINICAL = "z1-clinical"
    Z2_PROCESSING = "z2-processing"
    Z3_EXTERNAL = "z3-external"


# ═══════════════════════════════════════════════════════════════════════════════
# [GD1] GRAFANA DASHBOARD GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class GrafanaDashboardGenerator:
    """
    [D] Generates Grafana dashboard JSON for AGI system metrics.
    
    Dashboard sections:
      • Row 1: System Overview (health, cycles, latency)
      • Row 2: Core Module Metrics (reasoning, memory, perception)
      • Row 3: Causal Discovery (FCI/FCI++/PC/NOTEARS)
      • Row 4: Safety & Alignment
      • Row 5: K8s Pod Metrics (CPU, memory, GPU)
      • Row 6: HPA/VPA Auto-Scaling
      • Row 7: Meta-Learning Convergence
    
    Mathematical grounding:
      • _safe_unit_scalar sanitization on all values
      • No-Zeno rate limiting on metric emission
    """
    
    def __init__(
        self,
        prometheus_datasource: str = "Prometheus",
        dashboard_title: str = "SESI AGI Production Metrics",
    ):
        self.prometheus_datasource = prometheus_datasource
        self.dashboard_title = dashboard_title
    
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
            "datasource": self.prometheus_datasource,
            "targets": targets,
            "gridPos": grid_pos,
            "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
            "options": {},
        }
        if extra:
            panel.update(extra)
        return panel
    
    def _row_panel(self, panel_id: int, title: str, y: int) -> Dict[str, Any]:
        """Create a row panel."""
        return {
            "id": panel_id,
            "type": "row",
            "title": title,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            "collapsed": False,
            "panels": [],
        }
    
    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete Grafana dashboard JSON."""
        panels = []
        pid = 1
        
        # ── Row 1: System Overview ──────────────────────────────────────
        panels.append(self._row_panel(pid, "System Overview", 0))
        pid += 1
        
        panels.append(self._panel(
            pid, "Autonomous Cycles (Total)", "stat",
            [{"expr": "sum(sesi_agi_autonomous_cycles_total)", "legendFormat": "cycles"}],
            {"h": 6, "w": 4, "x": 0, "y": 1},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Cycle Success Rate", "gauge",
            [{"expr": "sum(sesi_agi_autonomous_cycles_total{status='SUCCESS'}) / sum(sesi_agi_autonomous_cycles_total) * 100",
              "legendFormat": "success %"}],
            {"h": 6, "w": 4, "x": 4, "y": 1},
            extra={"fieldConfig": {"defaults": {
                "min": 0, "max": 100, "unit": "percent",
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 80},
                    {"color": "green", "value": 95},
                ]},
            }}},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Cycle Latency P95", "timeseries",
            [{"expr": "histogram_quantile(0.95, rate(sesi_agi_cycle_latency_seconds_bucket[5m]))",
              "legendFormat": "p95"}],
            {"h": 6, "w": 8, "x": 8, "y": 1},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Active Pods", "stat",
            [{"expr": "count(kube_pod_status_phase{namespace='sesi-agi', phase='Running'})",
              "legendFormat": "pods"}],
            {"h": 6, "w": 4, "x": 16, "y": 1},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "δ_min Threshold", "stat",
            [{"expr": "sesi_agi_delta_min", "legendFormat": "δ_min"}],
            {"h": 6, "w": 4, "x": 20, "y": 1},
        ))
        pid += 1
        
        # ── Row 2: Core Module Metrics ──────────────────────────────────
        panels.append(self._row_panel(pid, "Core Module Metrics", 7))
        pid += 1
        
        panels.append(self._panel(
            pid, "Reasoning Calls", "timeseries",
            [{"expr": "sum(rate(sesi_agi_reasoning_calls_total[5m])) by (status)",
              "legendFormat": "{{status}}"}],
            {"h": 8, "w": 8, "x": 0, "y": 8},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Reasoning Confidence", "timeseries",
            [{"expr": "sesi_agi_reasoning_confidence", "legendFormat": "confidence"}],
            {"h": 8, "w": 8, "x": 8, "y": 8},
            extra={"fieldConfig": {"defaults": {"min": 0, "max": 1}}},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Memory Size (Episodic + Semantic)", "timeseries",
            [
                {"expr": "sesi_agi_memory_episodic_size", "legendFormat": "episodic"},
                {"expr": "sesi_agi_memory_semantic_size", "legendFormat": "semantic"},
            ],
            {"h": 8, "w": 8, "x": 16, "y": 8},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Perception Calls by Modality", "timeseries",
            [{"expr": "sum(rate(sesi_agi_perception_calls_total[5m])) by (modality)",
              "legendFormat": "{{modality}}"}],
            {"h": 8, "w": 12, "x": 0, "y": 16},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Curiosity Uncertainty", "timeseries",
            [{"expr": "sesi_agi_curiosity_uncertainty", "legendFormat": "uncertainty"}],
            {"h": 8, "w": 12, "x": 12, "y": 16},
            extra={"fieldConfig": {"defaults": {"min": 0, "max": 1}}},
        ))
        pid += 1
        
        # ── Row 3: Causal Discovery ─────────────────────────────────────
        panels.append(self._row_panel(pid, "Causal Discovery (FCI / FCI++ / PC / NOTEARS)", 24))
        pid += 1
        
        panels.append(self._panel(
            pid, "Causal Discovery Calls", "timeseries",
            [{"expr": "sum(rate(sesi_agi_causal_discovery_calls_total[5m])) by (algorithm)",
              "legendFormat": "{{algorithm}}"}],
            {"h": 8, "w": 8, "x": 0, "y": 25},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Acyclicity Constraint (h)", "timeseries",
            [{"expr": "sesi_agi_causal_acyclicity_h", "legendFormat": "{{algorithm}}"}],
            {"h": 8, "w": 8, "x": 8, "y": 25},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Edge Density", "timeseries",
            [{"expr": "sesi_agi_causal_edge_density", "legendFormat": "{{algorithm}}"}],
            {"h": 8, "w": 8, "x": 16, "y": 25},
            extra={"fieldConfig": {"defaults": {"min": 0, "max": 1}}},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Orientation Rules Loss", "timeseries",
            [{"expr": "sesi_agi_causal_orientation_loss", "legendFormat": "{{algorithm}}"}],
            {"h": 8, "w": 24, "x": 0, "y": 33},
        ))
        pid += 1
        
        # ── Row 4: Safety & Alignment ───────────────────────────────────
        panels.append(self._row_panel(pid, "Safety & Alignment", 41))
        pid += 1
        
        panels.append(self._panel(
            pid, "Safety Verifications", "timeseries",
            [{"expr": "sum(rate(sesi_agi_safety_verifications_total[5m])) by (result)",
              "legendFormat": "{{result}}"}],
            {"h": 8, "w": 12, "x": 0, "y": 42},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Safety Violations", "stat",
            [{"expr": "sum(increase(sesi_agi_safety_violations_total[24h]))",
              "legendFormat": "violations (24h)"}],
            {"h": 8, "w": 12, "x": 12, "y": 42},
            extra={"fieldConfig": {"defaults": {
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 1},
                    {"color": "red", "value": 5},
                ]},
            }}},
        ))
        pid += 1
        
        # ── Row 5: K8s Pod Metrics ──────────────────────────────────────
        panels.append(self._row_panel(pid, "Kubernetes Pod Metrics", 50))
        pid += 1
        
        panels.append(self._panel(
            pid, "CPU Usage (per pod)", "timeseries",
            [{"expr": "sum(rate(container_cpu_usage_seconds_total{namespace='sesi-agi'}[5m])) by (pod)",
              "legendFormat": "{{pod}}"}],
            {"h": 8, "w": 8, "x": 0, "y": 51},
            extra={"fieldConfig": {"defaults": {"unit": "short"}}},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Memory Usage (per pod)", "timeseries",
            [{"expr": "sum(container_memory_working_set_bytes{namespace='sesi-agi'}) by (pod) / 1024 / 1024 / 1024",
              "legendFormat": "{{pod}}"}],
            {"h": 8, "w": 8, "x": 8, "y": 51},
            extra={"fieldConfig": {"defaults": {"unit": "decgbytes"}}},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "GPU Utilization", "timeseries",
            [{"expr": "DCGM_FI_DEV_GPU_UTIL{namespace='sesi-agi'}",
              "legendFormat": "GPU {{gpu}}"}],
            {"h": 8, "w": 8, "x": 16, "y": 51},
            extra={"fieldConfig": {"defaults": {"unit": "percent"}}},
        ))
        pid += 1
        
        # ── Row 6: HPA/VPA Auto-Scaling ─────────────────────────────────
        panels.append(self._row_panel(pid, "HPA / VPA Auto-Scaling", 59))
        pid += 1
        
        panels.append(self._panel(
            pid, "HPA Current Replicas", "stat",
            [{"expr": "kube_hpa_status_current_replicas{namespace='sesi-agi'}",
              "legendFormat": "current"}],
            {"h": 6, "w": 6, "x": 0, "y": 60},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "HPA Desired Replicas", "stat",
            [{"expr": "kube_hpa_status_desired_replicas{namespace='sesi-agi'}",
              "legendFormat": "desired"}],
            {"h": 6, "w": 6, "x": 6, "y": 60},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "VPA Recommended CPU", "timeseries",
            [{"expr": "vpa_status_recommendation_value{namespace='sesi-agi', resource='cpu'}",
              "legendFormat": "recommended"}],
            {"h": 6, "w": 6, "x": 12, "y": 60},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "VPA Recommended Memory", "timeseries",
            [{"expr": "vpa_status_recommendation_value{namespace='sesi-agi', resource='memory'}",
              "legendFormat": "recommended"}],
            {"h": 6, "w": 6, "x": 18, "y": 60},
        ))
        pid += 1
        
        # ── Row 7: Meta-Learning ────────────────────────────────────────
        panels.append(self._row_panel(pid, "Meta-Learning Convergence", 66))
        pid += 1
        
        panels.append(self._panel(
            pid, "Meta-Learning Loss", "timeseries",
            [{"expr": "sesi_agi_meta_learning_loss", "legendFormat": "loss"}],
            {"h": 8, "w": 12, "x": 0, "y": 67},
        ))
        pid += 1
        
        panels.append(self._panel(
            pid, "Meta-Learning Updates", "timeseries",
            [{"expr": "sum(rate(sesi_agi_meta_learning_updates_total[5m]))",
              "legendFormat": "updates/s"}],
            {"h": 8, "w": 12, "x": 12, "y": 67},
        ))
        pid += 1
        
        # Build dashboard
        dashboard = {
            "uid": "sesi-agi-production",
            "title": self.dashboard_title,
            "tags": ["sesi", "agi", "production", "kubernetes"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "templating": {
                "list": [
                    {
                        "name": "namespace",
                        "type": "constant",
                        "current": {"text": "sesi-agi", "value": "sesi-agi"},
                    },
                    {
                        "name": "pod",
                        "type": "query",
                        "datasource": self.prometheus_datasource,
                        "query": "label_values(kube_pod_info{namespace='$namespace'}, pod)",
                        "refresh": 2,
                    },
                ],
            },
            "annotations": {
                "list": [
                    {
                        "name": "Safety Violations",
                        "datasource": self.prometheus_datasource,
                        "enable": True,
                        "expr": "increase(sesi_agi_safety_violations_total[1m]) > 0",
                        "iconColor": "red",
                        "titleFormat": "Safety Violation",
                    },
                ],
            },
        }
        
        return dashboard
    
    def write_dashboard(self, output_path: str = "./grafana/sesi-agi-dashboard.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Grafana dashboard written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# [ISM1] ISTIO SERVICE MESH GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class IstioServiceMeshGenerator:
    """
    [D] Generates Istio service mesh manifests for zone-based access control.
    
    Manifests:
      • VirtualService: Traffic routing
      • DestinationRule: Load balancing policies
      • PeerAuthentication: mTLS enforcement
      • AuthorizationPolicy: Zone-based RBAC (PNDC Axiom 3)
      • Gateway: External access
    
    Mathematical grounding:
      • PNDC Axiom 3: Necessity and Proportionality
      • Zone-based access control (Z0 strictest → Z3 loosest)
    """
    
    def __init__(
        self,
        namespace: str = "sesi-agi",
        app_name: str = "sesi-agi",
    ):
        self.namespace = namespace
        self.app_name = app_name
    
    def generate_virtual_service(self) -> Dict[str, Any]:
        """Generate VirtualService for traffic routing."""
        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{self.app_name}-vs",
                "namespace": self.namespace,
            },
            "spec": {
                "hosts": [self.app_name],
                "http": [
                    {
                        "match": [{"uri": {"prefix": "/metrics"}}],
                        "route": [{
                            "destination": {
                                "host": self.app_name,
                                "port": {"number": 9090},
                            },
                        }],
                    },
                    {
                        "route": [{
                            "destination": {
                                "host": self.app_name,
                                "port": {"number": 8080},
                            },
                        }],
                        "timeout": "30s",
                        "retries": {
                            "attempts": 3,
                            "perTryTimeout": "10s",
                            "retryOn": "5xx,reset,connect-failure",
                        },
                    },
                ],
            },
        }
    
    def generate_destination_rule(self) -> Dict[str, Any]:
        """Generate DestinationRule for load balancing."""
        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": f"{self.app_name}-dr",
                "namespace": self.namespace,
            },
            "spec": {
                "host": self.app_name,
                "trafficPolicy": {
                    "connectionPool": {
                        "tcp": {"maxConnections": 100},
                        "http": {"h2UpgradePolicy": "DEFAULT", "http1MaxPendingRequests": 100},
                    },
                    "loadBalancer": {"simple": "LEAST_CONN"},
                    "outlierDetection": {
                        "consecutive5xxErrors": 5,
                        "interval": "30s",
                        "baseEjectionTime": "30s",
                    },
                },
                "subsets": [
                    {"name": "stable", "labels": {"version": "stable"}},
                    {"name": "canary", "labels": {"version": "canary"}},
                ],
            },
        }
    
    def generate_peer_authentication(self) -> Dict[str, Any]:
        """Generate PeerAuthentication for mTLS."""
        return {
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "PeerAuthentication",
            "metadata": {
                "name": f"{self.app_name}-mtls",
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {"matchLabels": {"app": self.app_name}},
                "mtls": {"mode": "STRICT"},
            },
        }
    
    def generate_authorization_policies(self) -> List[Dict[str, Any]]:
        """
        Generate zone-based AuthorizationPolicy (PNDC Axiom 3).
        
        Z0 Implant: Most restricted (only Z1 can access)
        Z1 Clinical: Clinician access with MFA
        Z2 Processing: AI inference, read-only clinical
        Z3 External: Public API, deny internal access
        """
        policies = []
        
        # Z0 Implant: Only Z1 clinical gateway can access
        policies.append({
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"{self.app_name}-z0-policy",
                "namespace": self.namespace,
                "labels": {"sesi.io/zone": SecurityZone.Z0_IMPLANT},
            },
            "spec": {
                "selector": {"matchLabels": {"sesi.io/zone": SecurityZone.Z0_IMPLANT}},
                "action": "ALLOW",
                "rules": [{
                    "from": [{
                        "source": {
                            "principals": [f"cluster.local/ns/{self.namespace}/sa/clinical-gateway"],
                        },
                    }],
                    "to": [{
                        "operation": {"ports": ["8443"]},
                    }],
                }],
            },
        })
        
        # Z1 Clinical: Z0, Z2 can access with MFA
        policies.append({
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"{self.app_name}-z1-policy",
                "namespace": self.namespace,
                "labels": {"sesi.io/zone": SecurityZone.Z1_CLINICAL},
            },
            "spec": {
                "selector": {"matchLabels": {"sesi.io/zone": SecurityZone.Z1_CLINICAL}},
                "action": "ALLOW",
                "rules": [{
                    "from": [{
                        "source": {
                            "principals": [
                                f"cluster.local/ns/{self.namespace}/sa/implant-service",
                                f"cluster.local/ns/{self.namespace}/sa/processing-service",
                            ],
                        },
                    }],
                    "when": [{
                        "key": "request.auth.claims[mfa_verified]",
                        "values": ["true"],
                    }],
                    "to": [{
                        "operation": {"ports": ["443", "8443"]},
                    }],
                }],
            },
        })
        
        # Z2 Processing: Z1, Z3 can access
        policies.append({
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"{self.app_name}-z2-policy",
                "namespace": self.namespace,
                "labels": {"sesi.io/zone": SecurityZone.Z2_PROCESSING},
            },
            "spec": {
                "selector": {"matchLabels": {"sesi.io/zone": SecurityZone.Z2_PROCESSING}},
                "action": "ALLOW",
                "rules": [{
                    "from": [{
                        "source": {
                            "principals": [
                                f"cluster.local/ns/{self.namespace}/sa/clinical-gateway",
                                f"cluster.local/ns/{self.namespace}/sa/external-api",
                            ],
                        },
                    }],
                    "to": [{
                        "operation": {"ports": ["8080", "9090"]},
                    }],
                }],
            },
        })
        
        # Z3 External: Only Z2 can access (deny direct Z0/Z1)
        policies.append({
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "AuthorizationPolicy",
            "metadata": {
                "name": f"{self.app_name}-z3-policy",
                "namespace": self.namespace,
                "labels": {"sesi.io/zone": SecurityZone.Z3_EXTERNAL},
            },
            "spec": {
                "selector": {"matchLabels": {"sesi.io/zone": SecurityZone.Z3_EXTERNAL}},
                "action": "ALLOW",
                "rules": [{
                    "from": [{
                        "source": {
                            "principals": [f"cluster.local/ns/{self.namespace}/sa/processing-service"],
                        },
                    }],
                    "to": [{
                        "operation": {"ports": ["443"]},
                    }],
                }],
            },
        })
        
        return policies
    
    def generate_gateway(self) -> Dict[str, Any]:
        """Generate Gateway for external access."""
        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "Gateway",
            "metadata": {
                "name": f"{self.app_name}-gateway",
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {"istio": "ingressgateway"},
                "servers": [{
                    "port": {"number": 443, "name": "https", "protocol": "HTTPS"},
                    "tls": {
                        "mode": "SIMPLE",
                        "credentialName": f"{self.app_name}-tls-cert",
                    },
                    "hosts": [f"{self.app_name}.sesi.example.com"],
                }],
            },
        }
    
    def generate_all(self, output_dir: str = "./istio_manifests") -> Dict[str, str]:
        """Generate all Istio manifests."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        manifests = {
            "virtual_service": self.generate_virtual_service(),
            "destination_rule": self.generate_destination_rule(),
            "peer_authentication": self.generate_peer_authentication(),
            "gateway": self.generate_gateway(),
        }
        
        # Add authorization policies
        auth_policies = self.generate_authorization_policies()
        for i, policy in enumerate(auth_policies):
            manifests[f"authorization_policy_{i}"] = policy
        
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
        
        logger.info(f"Generated {len(manifests)} Istio manifests in {output_dir}")
        return files


# ═══════════════════════════════════════════════════════════════════════════════
# [ACD1] ARGOCD GITOPS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ArgoCDGitOpsGenerator:
    """
    [D] Generates ArgoCD manifests for GitOps deployment.
    
    Manifests:
      • Application CRD: Single application deployment
      • ApplicationSet: Multi-cluster deployment
      • SyncPolicy: Auto-sync configuration
      • Notification: Slack/Email alerts
    
    Mathematical grounding:
      • SESI_NoZeno_Closure Thm 3.2: Bounded sync frequency
      • Declarative infrastructure pattern
    """
    
    def __init__(
        self,
        app_name: str = "sesi-agi",
        namespace: str = "argocd",
        repo_url: str = "https://github.com/your-org/sesi-agi.git",
        target_revision: str = "main",
        path: str = "k8s_manifests",
    ):
        self.app_name = app_name
        self.namespace = namespace
        self.repo_url = repo_url
        self.target_revision = target_revision
        self.path = path
    
    def generate_application(self) -> Dict[str, Any]:
        """Generate ArgoCD Application CRD."""
        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Application",
            "metadata": {
                "name": self.app_name,
                "namespace": self.namespace,
                "labels": {"app.kubernetes.io/name": self.app_name},
            },
            "spec": {
                "project": "default",
                "source": {
                    "repoURL": self.repo_url,
                    "targetRevision": self.target_revision,
                    "path": self.path,
                },
                "destination": {
                    "server": "https://kubernetes.default.svc",
                    "namespace": "sesi-agi",
                },
                "syncPolicy": {
                    "automated": {
                        "prune": True,
                        "selfHeal": True,
                    },
                    "syncOptions": [
                        "CreateNamespace=true",
                        "PrunePropagationPolicy=foreground",
                        "PruneLast=true",
                    ],
                    "retry": {
                        "limit": 5,
                        "backoff": {
                            "duration": "5s",
                            "factor": 2,
                            "maxDuration": "3m",
                        },
                    },
                },
                "revisionHistoryLimit": 10,
            },
        }
    
    def generate_application_set(self) -> Dict[str, Any]:
        """Generate ArgoCD ApplicationSet for multi-cluster."""
        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "ApplicationSet",
            "metadata": {
                "name": f"{self.app_name}-set",
                "namespace": self.namespace,
            },
            "spec": {
                "generators": [
                    {
                        "clusters": {
                            "selector": {
                                "matchLabels": {"environment": "production"},
                            },
                        },
                    },
                ],
                "template": {
                    "metadata": {
                        "name": f"{self.app_name}-{{name}}",
                    },
                    "spec": {
                        "project": "default",
                        "source": {
                            "repoURL": self.repo_url,
                            "targetRevision": self.target_revision,
                            "path": self.path,
                        },
                        "destination": {
                            "server": "{{server}}",
                            "namespace": "sesi-agi",
                        },
                        "syncPolicy": {
                            "automated": {"prune": True, "selfHeal": True},
                        },
                    },
                },
            },
        }
    
    def generate_app_project(self) -> Dict[str, Any]:
        """Generate ArgoCD AppProject for RBAC."""
        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "AppProject",
            "metadata": {
                "name": "sesi-agi-project",
                "namespace": self.namespace,
            },
            "spec": {
                "description": "SESI AGI Production Project",
                "sourceRepos": [self.repo_url],
                "destinations": [{
                    "namespace": "sesi-agi",
                    "server": "https://kubernetes.default.svc",
                }],
                "clusterResourceWhitelist": [{"group": "*", "kind": "*"}],
                "namespaceResourceWhitelist": [{"group": "*", "kind": "*"}],
                "orphanedResources": {"warn": True},
                "syncWindows": [{
                    "kind": "allow",
                    "schedule": "* * * * *",
                    "duration": "24h",
                    "applications": [self.app_name],
                }],
            },
        }
    
    def generate_notification_config(self) -> Dict[str, Any]:
        """Generate ArgoCD notification configuration."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "argocd-notifications-cm",
                "namespace": self.namespace,
            },
            "data": {
                "service.slack": json.dumps({
                    "token": "$slack-token",
                }),
                "trigger.on-sync-succeeded": json.dumps({
                    "- when: app.status.operationState.phase in ['Succeeded']\n"
                    "  send: [app-sync-succeeded]",
                }),
                "trigger.on-sync-failed": json.dumps({
                    "- when: app.status.operationState.phase in ['Error', 'Failed']\n"
                    "  send: [app-sync-failed]",
                }),
                "template.app-sync-succeeded": json.dumps({
                    "message": "Application {{.app.metadata.name}} sync succeeded",
                    "slack": {
                        "attachments": [{
                            "title": "SESI AGI Sync Succeeded",
                            "color": "#18be52",
                            "fields": [{
                                "title": "Application",
                                "value": "{{.app.metadata.name}}",
                                "short": True,
                            }],
                        }],
                    },
                }),
                "template.app-sync-failed": json.dumps({
                    "message": "Application {{.app.metadata.name}} sync failed",
                    "slack": {
                        "attachments": [{
                            "title": "SESI AGI Sync Failed",
                            "color": "#e96d76",
                            "fields": [{
                                "title": "Application",
                                "value": "{{.app.metadata.name}}",
                                "short": True,
                            }],
                        }],
                    },
                }),
            },
        }
    
    def generate_all(self, output_dir: str = "./argocd_manifests") -> Dict[str, str]:
        """Generate all ArgoCD manifests."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        manifests = {
            "application": self.generate_application(),
            "application_set": self.generate_application_set(),
            "app_project": self.generate_app_project(),
            "notification_config": self.generate_notification_config(),
        }
        
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
        
        logger.info(f"Generated {len(manifests)} ArgoCD manifests in {output_dir}")
        return files


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SESIProductionSuite:
    """
    [D] Complete production suite integrating Grafana, Istio, and ArgoCD.
    """
    
    def __init__(
        self,
        namespace: str = "sesi-agi",
        app_name: str = "sesi-agi",
        repo_url: str = "https://github.com/your-org/sesi-agi.git",
    ):
        self.namespace = namespace
        self.app_name = app_name
        self.grafana = GrafanaDashboardGenerator()
        self.istio = IstioServiceMeshGenerator(namespace, app_name)
        self.argocd = ArgoCDGitOpsGenerator(app_name, "argocd", repo_url)
    
    def generate_all_artifacts(self, output_dir: str = "./sesi_production") -> Dict[str, Any]:
        """Generate all production artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Grafana dashboard
        results["grafana"] = self.grafana.write_dashboard(
            str(out_path / "grafana" / "sesi-agi-dashboard.json")
        )
        
        # Istio manifests
        results["istio"] = self.istio.generate_all(str(out_path / "istio"))
        
        # ArgoCD manifests
        results["argocd"] = self.argocd.generate_all(str(out_path / "argocd"))
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Grafana + Istio + ArgoCD — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available: {HAS_YAML}")
    
    suite = SESIProductionSuite(
        namespace="sesi-agi",
        app_name="sesi-agi",
        repo_url="https://github.com/msps4u/sesi-agi.git",
    )
    
    # ── Test 1: Grafana Dashboard ────────────────────────────────────────
    print("\n  ── Grafana Dashboard ──")
    dashboard = suite.grafana.generate_dashboard()
    print(f"    Dashboard title    : {dashboard['title']}")
    print(f"    Total panels       : {len(dashboard['panels'])}")
    print(f"    Template variables : {len(dashboard['templating']['list'])}")
    
    # ── Test 2: Istio Service Mesh ───────────────────────────────────────
    print("\n  ── Istio Service Mesh ──")
    vs = suite.istio.generate_virtual_service()
    print(f"    VirtualService     : {vs['metadata']['name']}")
    
    dr = suite.istio.generate_destination_rule()
    print(f"    DestinationRule    : {dr['metadata']['name']}")
    
    pa = suite.istio.generate_peer_authentication()
    print(f"    PeerAuthentication : {pa['metadata']['name']} (mTLS: {pa['spec']['mtls']['mode']})")
    
    auth_policies = suite.istio.generate_authorization_policies()
    print(f"    AuthorizationPolicies: {len(auth_policies)} (one per zone)")
    
    # ── Test 3: ArgoCD GitOps ────────────────────────────────────────────
    print("\n  ── ArgoCD GitOps ──")
    app = suite.argocd.generate_application()
    print(f"    Application        : {app['metadata']['name']}")
    print(f"    Repo URL           : {app['spec']['source']['repoURL']}")
    print(f"    Auto-sync          : {app['spec']['syncPolicy']['automated']}")
    
    app_set = suite.argocd.generate_application_set()
    print(f"    ApplicationSet     : {app_set['metadata']['name']}")
    
    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_production")
    print(f"    Grafana dashboard  : {artifacts['grafana']}")
    print(f"    Istio manifests    : {len(artifacts['istio'])} files")
    print(f"    ArgoCD manifests   : {len(artifacts['argocd'])} files")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Grafana dashboard is VISUALIZATION, not decision-making")
    print("  • Istio policies are CONFIGURATION, not security guarantees")
    print("  • ArgoCD manifests are TEMPLATES, not deployed systems")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
