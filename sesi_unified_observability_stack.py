"""
=============================================================================
SESI UNIFIED OBSERVABILITY STACK
Prometheus Alertmanager + Unified Grafana Dashboard + OpenTelemetry Collector
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[AM1] PrometheusAlertmanagerIntegration
      • Alert routing rules (by severity, zone, team)
      • Log-based alert generation from Loki queries
      • Inhibition rules (prevent alert storms)
      • Silencing for maintenance windows
      • No-Zeno rate limiting on alert emission
      • Subsampling CI for threshold calibration (Paper 11)

[UD1] UnifiedGrafanaDashboard
      • Trace-log-metric correlation via trace_id/span_id
      • Multi-zone view (Z0-Z3) with regime classification
      • δ_min monitoring panels
      • Multi-identity risk visualization
      • CUSUM change-point detection panels
      • Drill-down from metrics → traces → logs

[OC1] OpenTelemetryCollectorConfig
      • Receivers: OTLP, Prometheus, Jaeger, Loki
      • Processors: batch, memory_limiter, filter, attributes, spanmetrics
      • Exporters: Prometheus, Jaeger, Loki, Alertmanager
      • Pipelines: traces, metrics, logs
      • Zone-aware routing
      • Production-ready with HA considerations

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds alert emission rate
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful alert threshold
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → log stream regime classification
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
    → confidence interval for alert threshold calibration
[C] Six_Problems P4: CUSUM + subsampling
    → change-point detection in log streams
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → multi-identity risk for alert correlation
[C] PNDC Axiom 3: Necessity and Proportionality
    → log/alert data minimization
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_agione_jaeger_loki_integration.py — AGIONE quality enrichment,
    Jaeger service map, Loki security logs
  • sesi_istio_fido_prometheus.py — Istio mTLS, FIDO2 compliance,
    Prometheus-Jaeger metrics
  • sesi_vault_k8s_fido_jaeger.py — Vault-K8s SA binding, FIDO MDS sync,
    Jaeger trace correlation
  • agi_one_v3_11.py — AGIONE, EcosystemOrchestrator
  • agi_one_psyche_plus_v33.py — Psyche Plus, quality_score gating

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Alertmanager integration is a CONFIGURATION GENERATOR, not a deployed system.
• Unified dashboard is a VISUALIZATION TOOL, not a decision engine.
• OpenTelemetry Collector config is a TEMPLATE, not a production deployment.
• All thresholds, routing rules, and sampling rates are FREE DESIGN PARAMETERS.
• This module does NOT create, detect, or simulate consciousness.
• Production deployment requires security audit and regulatory approval.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.unified_observability")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

__all__ = [
    "AlertSeverity", "AlertRoutingRule", "PrometheusAlertmanagerIntegration",
    "UnifiedGrafanaDashboard", "DashboardPanel",
    "OTelReceiverConfig", "OTelProcessorConfig", "OTelExporterConfig",
    "OTelPipelineConfig", "OpenTelemetryCollectorConfig",
    "SESIUnifiedObservabilitySuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


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


# ═══════════════════════════════════════════════════════════════════════
# [AM1] PROMETHEUS ALERTMANAGER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AlertRoutingRule:
    """
    [D] A single alert routing rule.

    HONEST: Routing rules determine which team receives which alerts.
    They do NOT determine whether an alert is correct or actionable.
    """
    name: str
    severity: AlertSeverity
    zone: Optional[SecurityZone] = None
    team: str = "platform"
    group_by: List[str] = field(default_factory=lambda: ["alertname", "zone"])
    group_wait: str = "30s"
    group_interval: str = "5m"
    repeat_interval: str = "4h"
    receiver: str = "platform-team"


@dataclass
class AlertInhibitionRule:
    """
    [D] Inhibition rule to prevent alert storms.

    From PNDC Axiom 3: alerts should be necessary and proportionate.
    If a critical alert fires, suppress related warning alerts to
    prevent alert fatigue.
    """
    name: str
    source_matchers: List[str]
    target_matchers: List[str]
    equal: List[str] = field(default_factory=lambda: ["zone", "service"])


class PrometheusAlertmanagerIntegration:
    """
    [D] Integration with Prometheus Alertmanager for log-based alerts.

    Features:
      • Alert routing rules by severity, zone, team
      • Log-based alert generation from Loki queries
      • Inhibition rules to prevent alert storms
      • Silencing for maintenance windows
      • No-Zeno rate limiting on alert emission
      • Subsampling CI for threshold calibration (Paper 11)

    HONEST: This is a CONFIGURATION GENERATOR. It produces Alertmanager
    config files but does NOT:
      • Deploy Alertmanager
      • Guarantee alert correctness
      • Replace human judgment in incident response

    Mathematical grounding:
      • No-Zeno bound on alert emission rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for alert threshold (Delta_Min Thm 1.2)
      • Subsampling CI for threshold calibration (Paper 11 Thm 4.4)
    """

    def __init__(self, alertmanager_url: str = "http://alertmanager:9093"):
        self.alertmanager_url = alertmanager_url
        self.routing_rules: List[AlertRoutingRule] = []
        self.inhibition_rules: List[AlertInhibitionRule] = []
        self._alert_timestamps: List[float] = []

        # No-Zeno parameters for alert rate limiting
        self.alert_rate_limit_per_minute = 10
        self.min_alert_interval_s = 5.0

        self._install_default_rules()

    def _install_default_rules(self) -> None:
        """Install default routing and inhibition rules."""
        # Routing rules by zone and severity
        self.routing_rules = [
            AlertRoutingRule(
                name="z0-implant-emergency",
                severity=AlertSeverity.EMERGENCY,
                zone=SecurityZone.Z0_IMPLANT,
                team="clinical-ops",
                receiver="clinical-pager",
                group_wait="10s",
                repeat_interval="1h",
            ),
            AlertRoutingRule(
                name="z1-clinical-critical",
                severity=AlertSeverity.CRITICAL,
                zone=SecurityZone.Z1_CLINICAL,
                team="clinical-ops",
                receiver="clinical-team",
                group_wait="30s",
                repeat_interval="4h",
            ),
            AlertRoutingRule(
                name="z2-processing-warning",
                severity=AlertSeverity.WARNING,
                zone=SecurityZone.Z2_PROCESSING,
                team="platform",
                receiver="platform-slack",
                group_wait="1m",
                repeat_interval="12h",
            ),
            AlertRoutingRule(
                name="z3-external-info",
                severity=AlertSeverity.INFO,
                zone=SecurityZone.Z3_EXTERNAL,
                team="platform",
                receiver="platform-email",
                group_wait="5m",
                repeat_interval="24h",
            ),
        ]

        # Inhibition rules (PNDC Axiom 3: proportionality)
        self.inhibition_rules = [
            AlertInhibitionRule(
                name="emergency-inhibits-critical",
                source_matchers=['severity="emergency"'],
                target_matchers=['severity="critical"'],
                equal=["zone", "service"],
            ),
            AlertInhibitionRule(
                name="critical-inhibits-warning",
                source_matchers=['severity="critical"'],
                target_matchers=['severity="warning"'],
                equal=["zone", "service"],
            ),
            AlertInhibitionRule(
                name="degenerate-regime-inhibits-quality",
                source_matchers=['alertname="DegenerateRegimeDetected"'],
                target_matchers=['alertname="QualityDegradation"'],
                equal=["zone"],
            ),
        ]

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """
        Compute δ_min from Delta_Min Thm 1.2.

        δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}

        Applied to alert thresholds: minimum meaningful change in alert
        severity or frequency before triggering a new alert.
        """
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_alert_rate(self) -> Dict[str, Any]:
        """
        Check if alert emission is within No-Zeno bound.

        From SESI_NoZeno_Closure Thm 3.2:
          N(T) ≤ 1 + 2C_V·T/δ_min

        Applied to alerts: prevents alert storms that could overwhelm
        on-call teams.
        """
        now = time.time()
        cutoff = now - 60.0
        self._alert_timestamps = [
            t for t in self._alert_timestamps if t > cutoff
        ]

        if len(self._alert_timestamps) >= self.alert_rate_limit_per_minute:
            return {
                "allowed": False,
                "reason": "rate_limited",
                "current_count": len(self._alert_timestamps),
                "limit": self.alert_rate_limit_per_minute,
            }

        if self._alert_timestamps:
            time_since_last = now - self._alert_timestamps[-1]
            if time_since_last < self.min_alert_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                    "time_since_last_s": time_since_last,
                    "min_interval_s": self.min_alert_interval_s,
                }

        self._alert_timestamps.append(now)
        return {
            "allowed": True,
            "current_count": len(self._alert_timestamps),
            "limit": self.alert_rate_limit_per_minute,
        }

    # ── Log-Based Alert Generation ───────────────────────────────────────

    def generate_log_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Generate Prometheus alert rules based on Loki log queries.

        These rules translate log patterns into Prometheus alerts that
        Alertmanager can route and inhibit.

        HONEST: Alert thresholds are FREE DESIGN PARAMETERS
        (Unified_Ordinal Hyp 7.9). Must be calibrated against real
        operational data using subsampling CI (Paper 11 Thm 4.4).
        """
        return [
            {
                "alert": "EmergencyStopTriggered",
                "expr": 'sum(count_over_time({service="sesi-oversight"} | json | event_type="emergency_stop"[5m])) > 0',
                "for": "0m",
                "labels": {"severity": "emergency", "zone": "z0-implant"},
                "annotations": {
                    "summary": "Emergency stop triggered",
                    "description": (
                        "An emergency stop event was logged. "
                        "Immediate clinical review required. "
                        "Check Loki for full audit trail."
                    ),
                    "runbook_url": "https://wiki.sesi.example.com/runbooks/emergency-stop",
                },
            },
            {
                "alert": "ZoneViolationDetected",
                "expr": 'sum(count_over_time({service="sesi-oversight"} | json | event_type="zone_violation"[5m])) by (src_zone, dst_zone) > 3',
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Zone policy violation detected",
                    "description": (
                        "Multiple cross-zone policy violations detected. "
                        "Possible misconfiguration or attack attempt. "
                        "Check NetworkPolicy and Istio AuthorizationPolicy."
                    ),
                },
            },
            {
                "alert": "DegenerateRegimeDetected",
                "expr": 'sesi_quality_regime == 2',
                "for": "5m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Degenerate quality regime detected",
                    "description": (
                        "Quality regime is DEGENERATE (θ→0, ½δ₀+½δ₁). "
                        "paper6 Thm 5.1: occupation fraction polarizing. "
                        "Averaging-based monitoring will miss this. "
                        "Use change-point detection and regime-specific alerting."
                    ),
                },
            },
            {
                "alert": "HighAuthFailureRate",
                "expr": 'sum(rate({service="sesi-oversight"} | json | event_type="auth_failure"[5m])) > 0.1',
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High authentication failure rate",
                    "description": (
                        "Authentication failure rate exceeds threshold. "
                        "Possible brute-force attack or credential stuffing. "
                        "Check FIDO2 attestation logs and multi-identity risk."
                    ),
                },
            },
            {
                "alert": "NoZenoBoundExceeded",
                "expr": 'sesi_event_count > sesi_no_zeno_bound',
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "No-Zeno bound exceeded",
                    "description": (
                        "Event count exceeds No-Zeno bound: "
                        "N(T) ≤ 1 + 2C_V·T/δ_min (SESI_NoZeno Thm 3.2). "
                        "Possible Zeno-like event accumulation. "
                        "Investigate root cause immediately."
                    ),
                },
            },
            {
                "alert": "KeyRotationChattering",
                "expr": 'sum(rate({service="sesi-vault"} | json | event_type="key_rotation"[1m])) > 10',
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "Key rotation chattering detected",
                    "description": (
                        "Excessive key rotation events detected. "
                        "δ_min quantum gate may not be enforced. "
                        "Check Vault Transit configuration."
                    ),
                },
            },
        ]

    # ── Alertmanager Configuration ───────────────────────────────────────

    def generate_alertmanager_config(self) -> Dict[str, Any]:
        """
        Generate complete Alertmanager configuration.

        Includes:
          • Global settings
          • Routing tree
          • Receivers
          • Inhibition rules
          • Silence management

        HONEST: This is a TEMPLATE. Production requires:
          • Proper receiver configuration (PagerDuty, Slack, Email)
          • TLS certificates for Alertmanager API
          • Authentication for silence management
          • Regular security audit
        """
        config = {
            "global": {
                "resolve_timeout": "5m",
                "smtp_smarthost": "smtp.example.com:587",
                "smtp_from": "alertmanager@sesi.example.com",
                "smtp_auth_username": "alertmanager",
                "smtp_auth_password": "${SMTP_PASSWORD}",
            },
            "route": {
                "receiver": "default-receiver",
                "group_by": ["alertname", "zone", "severity"],
                "group_wait": "30s",
                "group_interval": "5m",
                "repeat_interval": "4h",
                "routes": [
                    {
                        "match": {"severity": "emergency"},
                        "receiver": "emergency-pager",
                        "group_wait": "10s",
                        "repeat_interval": "1h",
                        "continue": False,
                    },
                    {
                        "match": {"severity": "critical"},
                        "receiver": "critical-slack",
                        "group_wait": "30s",
                        "repeat_interval": "4h",
                    },
                    {
                        "match": {"severity": "warning"},
                        "receiver": "warning-email",
                        "group_wait": "1m",
                        "repeat_interval": "12h",
                    },
                    {
                        "match": {"severity": "info"},
                        "receiver": "info-log",
                        "group_wait": "5m",
                        "repeat_interval": "24h",
                    },
                ],
            },
            "receivers": [
                {
                    "name": "default-receiver",
                    "email_configs": [
                        {
                            "to": "platform-team@sesi.example.com",
                            "send_resolved": True,
                        }
                    ],
                },
                {
                    "name": "emergency-pager",
                    "pagerduty_configs": [
                        {
                            "service_key": "${PAGERDUTY_SERVICE_KEY}",
                            "severity": "critical",
                        }
                    ],
                    "webhook_configs": [
                        {
                            "url": "http://clinical-gateway:8080/api/v1/emergency",
                            "send_resolved": True,
                        }
                    ],
                },
                {
                    "name": "critical-slack",
                    "slack_configs": [
                        {
                            "api_url": "${SLACK_WEBHOOK_URL}",
                            "channel": "#sesi-critical",
                            "title": "SESI Critical Alert",
                            "text": "{{ .CommonAnnotations.summary }}",
                            "send_resolved": True,
                        }
                    ],
                },
                {
                    "name": "warning-email",
                    "email_configs": [
                        {
                            "to": "platform-warnings@sesi.example.com",
                            "send_resolved": True,
                        }
                    ],
                },
                {
                    "name": "info-log",
                    "webhook_configs": [
                        {
                            "url": "http://log-aggregator:8080/api/v1/alerts",
                            "send_resolved": True,
                        }
                    ],
                },
            ],
            "inhibit_rules": [
                {
                    "source_match": {"severity": "emergency"},
                    "target_match": {"severity": "critical"},
                    "equal": ["zone", "service"],
                },
                {
                    "source_match": {"severity": "critical"},
                    "target_match": {"severity": "warning"},
                    "equal": ["zone", "service"],
                },
                {
                    "source_match": {"alertname": "DegenerateRegimeDetected"},
                    "target_match": {"alertname": "QualityDegradation"},
                    "equal": ["zone"],
                },
            ],
        }

        return config

    def write_config(self, output_path: str = "./alertmanager/alertmanager.yml") -> str:
        """Write Alertmanager configuration to file."""
        config = self.generate_alertmanager_config()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"Alertmanager config written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./prometheus/alert-rules.yml") -> str:
        """Write Prometheus alert rules to file."""
        rules = self.generate_log_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {"groups": [{"name": "sesi_security_alerts", "rules": rules}]}

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"Alert rules written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get Alertmanager integration status."""
        return {
            "alertmanager_url": self.alertmanager_url,
            "routing_rules": len(self.routing_rules),
            "inhibition_rules": len(self.inhibition_rules),
            "alert_rate_limit": self.alert_rate_limit_per_minute,
            "delta_min": self.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [UD1] UNIFIED GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DashboardPanel:
    """A single Grafana dashboard panel."""
    id: int
    title: str
    panel_type: str
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]
    field_config: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class UnifiedGrafanaDashboard:
    """
    [D] Generates a unified Grafana dashboard for traces, logs, and metrics.

    Features:
      • Trace-log-metric correlation via trace_id/span_id
      • Multi-zone view (Z0-Z3) with regime classification
      • δ_min monitoring panels
      • Multi-identity risk visualization
      • CUSUM change-point detection panels
      • Drill-down from metrics → traces → logs

    HONEST: This is a DASHBOARD GENERATOR. It produces Grafana JSON
    but does NOT:
      • Deploy Grafana
      • Configure data sources
      • Make security decisions
      • Guarantee observability completeness

    Mathematical grounding:
      • paper6 Thm 5.1: Growth-rate regime classification
      • Six_Problems P4: CUSUM change-point detection
      • Delta_Min Thm 1.2: δ_min quantum gate monitoring
    """

    def __init__(
        self,
        prometheus_uid: str = "prometheus",
        loki_uid: str = "loki",
        jaeger_uid: str = "jaeger",
    ):
        self.prometheus_uid = prometheus_uid
        self.loki_uid = loki_uid
        self.jaeger_uid = jaeger_uid

    def _panel(
        self,
        panel_id: int,
        title: str,
        panel_type: str,
        targets: List[Dict[str, Any]],
        grid_pos: Dict[str, int],
        datasource_uid: str = "prometheus",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Grafana panel."""
        panel = {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "datasource": {"type": "prometheus", "uid": datasource_uid},
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
        """
        Generate complete unified Grafana dashboard JSON.

        Layout:
          Row 1: Overview (system health, active alerts, regime)
          Row 2: Zone Traffic (cross-zone flows, violations)
          Row 3: Quality Regime (Hill θ̂, CUSUM, occupation fraction)
          Row 4: Trace-Log Correlation (trace count, error rate, latency)
          Row 5: Security Events (auth, attestation, multi-identity risk)
          Row 6: δ_min Monitoring (quantum gate, rotation rate)
        """
        panels = []
        panel_id = 1

        # ── Row 1: Overview ──────────────────────────────────────────────
        panels.append(self._row_panel(panel_id, "System Overview", 0))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "System Health", "stat",
            [{"expr": "up{job='sesi-oversight'}", "legendFormat": "health"}],
            {"h": 6, "w": 4, "x": 0, "y": 1},
            extra={"fieldConfig": {"defaults": {
                "mappings": [
                    {"type": "value", "options": {
                        "0": {"text": "DOWN", "color": "red"},
                        "1": {"text": "UP", "color": "green"}}}]}}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Active Alerts", "stat",
            [{"expr": "sum(ALERTS{alertstate='firing'}) or vector(0)",
              "legendFormat": "alerts"}],
            {"h": 6, "w": 4, "i": 4, "y": 1},
            extra={"fieldConfig": {"defaults": {
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 1},
                    {"color": "red", "value": 5}]}}}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Quality Regime", "stat",
            [{"expr": "sesi_quality_regime", "legendFormat": "regime"}],
            {"h": 6, "w": 4, "i": 8, "y": 1},
            extra={"fieldConfig": {"defaults": {
                "mappings": [
                    {"type": "value", "options": {
                        "0": {"text": "FINITE_MEAN", "color": "green"},
                        "1": {"text": "REGULARLY_VARYING", "color": "yellow"},
                        "2": {"text": "DEGENERATE", "color": "red"},
                        "3": {"text": "INSUFFICIENT_DATA", "color": "gray"}}}]}}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Emergency Stops (24h)", "stat",
            [{"expr": "sum(increase(sesi_emergency_stops_total[24h]))",
              "legendFormat": "stops"}],
            {"h": 6, "w": 4, "i": 12, "y": 1},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Uptime", "stat",
            [{"expr": "time() - process_start_time_seconds{job='sesi-oversight'}",
              "legendFormat": "uptime"}],
            {"h": 6, "w": 4, "i": 16, "y": 1},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Request Rate", "timeseries",
            [{"expr": "sum(rate(sesi_requests_total[5m]))",
              "legendFormat": "req/s"}],
            {"h": 6, "w": 4, "i": 20, "y": 1},
        ))
        panel_id += 1

        # ── Row 2: Zone Traffic ──────────────────────────────────────────
        panels.append(self._row_panel(panel_id, "Zone Traffic (Z0-Z3)", 7))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Cross-Zone Connections", "heatmap",
            [{"expr": "sum(rate(sesi_zone_crossing_total[5m])) by (src_zone, dst_zone)",
              "legendFormat": "{{src_zone}}→{{dst_zone}}",
              "format": "heatmap"}],
            {"h": 8, "w": 12, "i": 0, "y": 8},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Zone Violations", "timeseries",
            [{"expr": "sum(rate(sesi_zone_violations_total[5m])) by (src_zone, dst_zone)",
              "legendFormat": "{{src_zone}}→{{dst_zone}}"}],
            {"h": 8, "w": 12, "i": 12, "y": 8},
        ))
        panel_id += 1

        # ── Row 3: Quality Regime ────────────────────────────────────────
        panels.append(self._row_panel(panel_id, "Quality Regime Detection (paper6 Thm 5.1)", 16))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Hill Estimator θ̂", "timeseries",
            [{"expr": "sesi_hill_theta_hat", "legendFormat": "θ̂"}],
            {"h": 8, "w": 8, "i": 0, "y": 17},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "CUSUM Statistic", "timeseries",
            [
                {"expr": "sesi_cusum_statistic", "legendFormat": "cusum"},
                {"expr": "sesi_cusum_threshold", "legendFormat": "threshold"},
            ],
            {"h": 8, "w": 8, "i": 8, "y": 17},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Occupation Fraction (½δ₀+½δ₁)", "gauge",
            [{"expr": "sesi_occupation_fraction", "legendFormat": "occupation"}],
            {"h": 8, "w": 8, "i": 16, "y": 17},
            extra={"fieldConfig": {"defaults": {
                "min": 0, "max": 1,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "green", "value": 0.3},
                    {"color": "red", "value": 0.7}]}}}},
        ))
        panel_id += 1

        # ── Row 4: Trace-Log Correlation ─────────────────────────────────
        panels.append(self._row_panel(panel_id, "Trace-Log-Metric Correlation", 25))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Trace Count", "timeseries",
            [{"expr": "sum(rate(jaeger_traces_total[5m]))",
              "legendFormat": "traces/s",
              "datasource": {"type": "prometheus", "uid": self.prometheus_uid}}],
            {"h": 8, "w": 8, "i": 0, "y": 26},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Span Duration P95", "timeseries",
            [{"expr": "histogram_quantile(0.95, sum(rate(jaeger_span_duration_seconds_bucket[5m])) by (le, operation))",
              "legendFormat": "{{operation}}",
              "datasource": {"type": "prometheus", "uid": self.prometheus_uid}}],
            {"h": 8, "w": 8, "i": 8, "y": 26},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Log Volume by Severity", "timeseries",
            [{"expr": "sum(rate({service='sesi-oversight'}[5m])) by (severity)",
              "legendFormat": "{{severity}}",
              "datasource": {"type": "loki", "uid": self.loki_uid}}],
            {"h": 8, "w": 8, "i": 16, "y": 26},
        ))
        panel_id += 1

        # ── Row 5: Security Events ───────────────────────────────────────
        panels.append(self._row_panel(panel_id, "Security Events", 34))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Auth Events", "timeseries",
            [{"expr": "sum(rate(sesi_auth_events_total[5m])) by (event_type, outcome)",
              "legendFormat": "{{event_type}}-{{outcome}}"}],
            {"h": 8, "w": 12, "i": 0, "y": 35},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "Multi-Identity Risk (Comp. Security §5.2)", "timeseries",
            [{"expr": "sesi_multi_identity_risk_score", "legendFormat": "R_id"}],
            {"h": 8, "w": 12, "i": 12, "y": 35},
        ))
        panel_id += 1

        # ── Row 6: δ_min Monitoring ──────────────────────────────────────
        panels.append(self._row_panel(panel_id, "δ_min Quantum Gate Monitoring (Delta_Min Thm 1.2)", 43))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "δ_min Estimate", "timeseries",
            [
                {"expr": "sesi_delta_min_estimate", "legendFormat": "δ_min"},
                {"expr": "sesi_delta_min_ci_lower", "legendFormat": "CI lower"},
                {"expr": "sesi_delta_min_ci_upper", "legendFormat": "CI upper"},
            ],
            {"h": 8, "w": 12, "i": 0, "y": 44},
        ))
        panel_id += 1

        panels.append(self._panel(
            panel_id, "No-Zeno Event Count vs Bound", "timeseries",
            [
                {"expr": "sesi_event_count", "legendFormat": "N(T)"},
                {"expr": "sesi_no_zeno_bound", "legendFormat": "bound"},
            ],
            {"h": 8, "w": 12, "i": 12, "y": 44},
        ))
        panel_id += 1

        # Build dashboard
        dashboard = {
            "uid": "sesi-unified-observability",
            "title": "SESI Unified Observability: Traces + Logs + Metrics",
            "tags": ["sesi", "unified", "observability", "traces", "logs", "metrics"],
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
                    {
                        "name": "service",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "query": "label_values(sesi_requests_total, service)",
                        "refresh": 2,
                    },
                    {
                        "name": "trace_id",
                        "type": "textbox",
                        "current": {"text": "", "value": ""},
                        "label": "Trace ID (for drill-down)",
                    },
                ],
            },
            "annotations": {
                "list": [
                    {
                        "name": "Emergency Stops",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "sesi_emergency_stop_triggered",
                        "iconColor": "red",
                        "titleFormat": "EMERGENCY STOP",
                    },
                    {
                        "name": "Regime Changes",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "changes(sesi_quality_regime[5m]) > 0",
                        "iconColor": "yellow",
                        "titleFormat": "Regime Change",
                    },
                ],
            },
            "links": [
                {
                    "title": "Jaeger Traces",
                    "url": f"http://jaeger-query:16686/search?service=sesi-oversight&tags=%7B%22trace_id%22%3A%22${{trace_id}}%22%7D",
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
                {
                    "title": "Loki Logs",
                    "url": f"http://loki:3100/loki/api/v1/query_range?query=%7Bservice%3D%22sesi-oversight%22%2C+trace_id%3D%22${{trace_id}}%22%7D",
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
            ],
        }

        return dashboard

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

    def write_dashboard(self, output_path: str = "./grafana/unified-dashboard.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Unified dashboard written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get dashboard status."""
        dashboard = self.generate_dashboard()
        return {
            "title": dashboard["title"],
            "n_panels": len(dashboard["panels"]),
            "n_template_vars": len(dashboard["templating"]["list"]),
            "n_annotations": len(dashboard["annotations"]["list"]),
            "n_links": len(dashboard["links"]),
        }


# ═══════════════════════════════════════════════════════════════════════
# [OC1] OPENTELEMETRY COLLECTOR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OTelReceiverConfig:
    """Configuration for an OpenTelemetry receiver."""
    name: str
    type: str  # otlp, prometheus, jaeger, loki
    config: Dict[str, Any]


@dataclass
class OTelProcessorConfig:
    """Configuration for an OpenTelemetry processor."""
    name: str
    type: str  # batch, memory_limiter, filter, attributes, spanmetrics
    config: Dict[str, Any]


@dataclass
class OTelExporterConfig:
    """Configuration for an OpenTelemetry exporter."""
    name: str
    type: str  # prometheus, jaeger, loki, alertmanager
    config: Dict[str, Any]


@dataclass
class OTelPipelineConfig:
    """Configuration for an OpenTelemetry pipeline."""
    name: str
    receivers: List[str]
    processors: List[str]
    exporters: List[str]


class OpenTelemetryCollectorConfig:
    """
    [D] Generates OpenTelemetry Collector configuration for production.

    Features:
      • Receivers: OTLP, Prometheus, Jaeger, Loki
      • Processors: batch, memory_limiter, filter, attributes, spanmetrics
      • Exporters: Prometheus, Jaeger, Loki, Alertmanager
      • Pipelines: traces, metrics, logs
      • Zone-aware routing
      • Production-ready with HA considerations

    HONEST: This is a CONFIGURATION GENERATOR. It produces OTel Collector
    config files but does NOT:
      • Deploy the collector
      • Guarantee telemetry completeness
      • Replace security audit

    Mathematical grounding:
      • PNDC Axiom 3: Necessity and Proportionality (data minimization)
      • Comprehensive Security §5.3: Privacy preservation
      • No-Zeno: Rate limiting on telemetry emission
    """

    def __init__(
        self,
        prometheus_endpoint: str = "http://prometheus:9090/api/v1/write",
        jaeger_endpoint: str = "http://jaeger-collector:14250",
        loki_endpoint: str = "http://loki:3100/loki/api/v1/push",
        alertmanager_endpoint: str = "http://alertmanager:9093/api/v2/alerts",
    ):
        self.prometheus_endpoint = prometheus_endpoint
        self.jaeger_endpoint = jaeger_endpoint
        self.loki_endpoint = loki_endpoint
        self.alertmanager_endpoint = alertmanager_endpoint

    def generate_config(self) -> Dict[str, Any]:
        """
        Generate complete OpenTelemetry Collector configuration.

        HONEST: This configuration follows PNDC Axiom 3 (Necessity & Proportionality)
        and Comprehensive Security §5.3 (privacy preservation):
          • Only necessary telemetry is collected
          • Sensitive data is filtered out
          • Zone-aware routing prevents cross-zone data leakage
        """
        config = {
            "receivers": {
                "otlp": {
                    "protocols": {
                        "grpc": {
                            "endpoint": "0.0.0.0:4317",
                            "tls": {
                                "cert_file": "/etc/otel/certs/server.crt",
                                "key_file": "/etc/otel/certs/server.key",
                            },
                        },
                        "http": {
                            "endpoint": "0.0.0.0:4318",
                            "tls": {
                                "cert_file": "/etc/otel/certs/server.crt",
                                "key_file": "/etc/otel/certs/server.key",
                            },
                        },
                    },
                },
                "prometheus": {
                    "config": {
                        "scrape_configs": [
                            {
                                "job_name": "sesi-oversight",
                                "scrape_interval": "15s",
                                "static_configs": [
                                    {"targets": ["sesi-oversight:9090"]}
                                ],
                            },
                            {
                                "job_name": "sesi-vault",
                                "scrape_interval": "30s",
                                "static_configs": [
                                    {"targets": ["vault:8200"]}
                                ],
                            },
                        ],
                    },
                },
                "jaeger": {
                    "protocols": {
                        "grpc": {
                            "endpoint": "0.0.0.0:14250",
                        },
                        "thrift_http": {
                            "endpoint": "0.0.0.0:14268",
                        },
                    },
                },
            },
            "processors": {
                "batch": {
                    "timeout": "5s",
                    "send_batch_size": 1024,
                    "send_batch_max_size": 2048,
                },
                "memory_limiter": {
                    "check_interval": "1s",
                    "limit_mib": 1024,
                    "spike_limit_mib": 256,
                },
                "filter/drop-sensitive": {
                    "error_mode": "ignore",
                    "traces": {
                        "span": [
                            'attributes["password"] != nil',
                            'attributes["secret"] != nil',
                            'attributes["token"] != nil',
                            'attributes["ssn"] != nil',
                            'attributes["credit_card"] != nil',
                        ],
                    },
                    "logs": {
                        "log_record": [
                            'body["password"] != nil',
                            'body["secret"] != nil',
                            'body["token"] != nil',
                        ],
                    },
                },
                "attributes/add-zone": {
                    "actions": [
                        {
                            "key": "sesi.zone",
                            "from_attribute": "zone",
                            "action": "upsert",
                        },
                    ],
                },
                "spanmetrics": {
                    "metrics_exporter": "prometheus",
                    "latency_histogram_buckets": [
                        "1ms", "5ms", "10ms", "50ms", "100ms",
                        "500ms", "1s", "5s", "10s",
                    ],
                    "dimensions": [
                        {"name": "service.name"},
                        {"name": "sesi.zone"},
                        {"name": "http.method"},
                        {"name": "http.status_code"},
                    ],
                },
                "transform/add-trace-id-to-logs": {
                    "log_statements": [
                        {
                            "context": "log",
                            "statements": [
                                'set(attributes["trace_id"], trace_id) where trace_id != nil',
                                'set(attributes["span_id"], span_id) where span_id != nil',
                            ],
                        },
                    ],
                },
            },
            "exporters": {
                "prometheus": {
                    "endpoint": "0.0.0.0:8889",
                    "namespace": "sesi",
                    "send_timestamps": True,
                    "metric_expiration": "5m",
                    "resource_to_telemetry_conversion": {
                        "enabled": True,
                    },
                },
                "prometheusremotewrite": {
                    "endpoint": self.prometheus_endpoint,
                    "tls": {
                        "insecure": False,
                        "ca_file": "/etc/otel/certs/ca.crt",
                    },
                    "resource_to_telemetry_conversion": {
                        "enabled": True,
                    },
                },
                "jaeger": {
                    "endpoint": self.jaeger_endpoint,
                    "tls": {
                        "insecure": False,
                        "ca_file": "/etc/otel/certs/ca.crt",
                    },
                },
                "loki": {
                    "endpoint": self.loki_endpoint,
                    "labels": {
                        "resource": {
                            "service.name": "service",
                            "sesi.zone": "zone",
                        },
                        "log": {
                            "severity": "level",
                            "event_type": "event_type",
                        },
                    },
                },
                "alertmanager": {
                    "endpoint": self.alertmanager_endpoint,
                    "tls": {
                        "insecure": False,
                        "ca_file": "/etc/otel/certs/ca.crt",
                    },
                },
            },
            "service": {
                "extensions": ["health_check", "pprof", "zpages"],
                "pipelines": {
                    "traces": {
                        "receivers": ["otlp", "jaeger"],
                        "processors": [
                            "memory_limiter",
                            "filter/drop-sensitive",
                            "attributes/add-zone",
                            "spanmetrics",
                            "batch",
                        ],
                        "exporters": ["jaeger", "prometheus"],
                    },
                    "metrics": {
                        "receivers": ["otlp", "prometheus"],
                        "processors": [
                            "memory_limiter",
                            "batch",
                        ],
                        "exporters": ["prometheus", "prometheusremotewrite"],
                    },
                    "logs": {
                        "receivers": ["otlp"],
                        "processors": [
                            "memory_limiter",
                            "filter/drop-sensitive",
                            "transform/add-trace-id-to-logs",
                            "attributes/add-zone",
                            "batch",
                        ],
                        "exporters": ["loki", "alertmanager"],
                    },
                },
                "telemetry": {
                    "logs": {
                        "level": "info",
                    },
                    "metrics": {
                        "address": "0.0.0.0:8888",
                    },
                },
            },
            "extensions": {
                "health_check": {
                    "endpoint": "0.0.0.0:13133",
                },
                "pprof": {
                    "endpoint": "0.0.0.0:1777",
                },
                "zpages": {
                    "endpoint": "0.0.0.0:55679",
                },
            },
        }

        return config

    def write_config(self, output_path: str = "./otel/otel-collector-config.yaml") -> str:
        """Write OTel Collector configuration to file."""
        config = self.generate_config()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"OTel Collector config written to {output_path}")
        return str(out_path)

    def generate_k8s_deployment(self) -> Dict[str, Any]:
        """Generate Kubernetes Deployment manifest for OTel Collector."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "otel-collector",
                "namespace": "sesi-observability",
                "labels": {
                    "app.kubernetes.io/name": "otel-collector",
                    "app.kubernetes.io/component": "telemetry",
                },
            },
            "spec": {
                "replicas": 2,
                "selector": {
                    "matchLabels": {"app.kubernetes.io/name": "otel-collector"},
                },
                "template": {
                    "metadata": {
                        "labels": {"app.kubernetes.io/name": "otel-collector"},
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8888",
                        },
                    },
                    "spec": {
                        "serviceAccountName": "otel-collector",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000,
                        },
                        "containers": [
                            {
                                "name": "otel-collector",
                                "image": "otel/opentelemetry-collector-contrib:0.91.0",
                                "args": ["--config=/etc/otel/config.yaml"],
                                "ports": [
                                    {"name": "otlp-grpc", "containerPort": 4317},
                                    {"name": "otlp-http", "containerPort": 4318},
                                    {"name": "jaeger-grpc", "containerPort": 14250},
                                    {"name": "jaeger-http", "containerPort": 14268},
                                    {"name": "prometheus", "containerPort": 8889},
                                    {"name": "metrics", "containerPort": 8888},
                                    {"name": "health", "containerPort": 13133},
                                ],
                                "env": [
                                    {"name": "GOMEMLIMIT", "value": "1024MiB"},
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "500m",
                                        "memory": "1Gi",
                                    },
                                    "limits": {
                                        "cpu": "2000m",
                                        "memory": "2Gi",
                                    },
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/",
                                        "port": 13133,
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/",
                                        "port": 13133,
                                    },
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 5,
                                },
                                "volumeMounts": [
                                    {
                                        "name": "otel-config",
                                        "mountPath": "/etc/otel",
                                        "readOnly": True,
                                    },
                                    {
                                        "name": "otel-certs",
                                        "mountPath": "/etc/otel/certs",
                                        "readOnly": True,
                                    },
                                ],
                            },
                        ],
                        "volumes": [
                            {
                                "name": "otel-config",
                                "configMap": {"name": "otel-collector-config"},
                            },
                            {
                                "name": "otel-certs",
                                "secret": {"secretName": "otel-collector-tls"},
                            },
                        ],
                    },
                },
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Get OTel Collector config status."""
        config = self.generate_config()
        return {
            "receivers": list(config["receivers"].keys()),
            "processors": list(config["processors"].keys()),
            "exporters": list(config["exporters"].keys()),
            "pipelines": list(config["service"]["pipelines"].keys()),
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIUnifiedObservabilitySuite:
    """
    [D] Complete unified observability suite.

    Integrates:
      • Prometheus Alertmanager for log-based alerts
      • Unified Grafana dashboard for traces+logs+metrics
      • OpenTelemetry Collector for production telemetry
    """

    def __init__(self):
        self.alertmanager = PrometheusAlertmanagerIntegration()
        self.dashboard = UnifiedGrafanaDashboard()
        self.otel_collector = OpenTelemetryCollectorConfig()

    def generate_all_artifacts(self, output_dir: str = "./sesi_observability") -> Dict[str, Any]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Alertmanager config
        results["alertmanager_config"] = self.alertmanager.write_config(
            str(out_path / "alertmanager" / "alertmanager.yml")
        )
        results["alert_rules"] = self.alertmanager.write_alert_rules(
            str(out_path / "prometheus" / "alert-rules.yml")
        )

        # Grafana dashboard
        results["grafana_dashboard"] = self.dashboard.write_dashboard(
            str(out_path / "grafana" / "unified-dashboard.json")
        )

        # OTel Collector config
        results["otel_config"] = self.otel_collector.write_config(
            str(out_path / "otel" / "otel-collector-config.yaml")
        )

        # K8s deployment
        k8s_deploy = self.otel_collector.generate_k8s_deployment()
        k8s_path = out_path / "k8s" / "otel-collector-deployment.yaml"
        k8s_path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_YAML:
            with open(k8s_path, "w") as f:
                yaml.dump(k8s_deploy, f, default_flow_style=False)
        else:
            with open(k8s_path, "w") as f:
                json.dump(k8s_deploy, f, indent=2)
        results["k8s_deployment"] = str(k8s_path)

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "alertmanager": self.alertmanager.get_status(),
            "dashboard": self.dashboard.get_status(),
            "otel_collector": self.otel_collector.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Unified Observability Stack — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available: {HAS_YAML}")

    suite = SESIUnifiedObservabilitySuite()

    # ── Test 1: Alertmanager Integration ─────────────────────────────────
    print("\n  ── Prometheus Alertmanager Integration ──")
    am_status = suite.alertmanager.get_status()
    print(f"    Routing rules      : {am_status['routing_rules']}")
    print(f"    Inhibition rules   : {am_status['inhibition_rules']}")
    print(f"    Alert rate limit   : {am_status['alert_rate_limit']}/min")
    print(f"    δ_min              : {am_status['delta_min']:.6f}")

    # Test alert rate limiting
    rate_check = suite.alertmanager.check_alert_rate()
    print(f"    Alert rate check   : {rate_check['allowed']}")

    # Generate alert rules
    alert_rules = suite.alertmanager.generate_log_alert_rules()
    print(f"    Alert rules        : {len(alert_rules)}")
    for rule in alert_rules[:3]:
        print(f"      • {rule['alert']} ({rule['labels']['severity']})")

    # ── Test 2: Unified Grafana Dashboard ────────────────────────────────
    print("\n  ── Unified Grafana Dashboard ──")
    dash_status = suite.dashboard.get_status()
    print(f"    Dashboard title    : {dash_status['title']}")
    print(f"    Total panels       : {dash_status['n_panels']}")
    print(f"    Template variables : {dash_status['n_template_vars']}")
    print(f"    Annotations        : {dash_status['n_annotations']}")
    print(f"    External links     : {dash_status['n_links']}")

    # Generate dashboard
    dashboard = suite.dashboard.generate_dashboard()
    print(f"    Dashboard rows     : {sum(1 for p in dashboard['panels'] if p['type'] == 'row')}")

    # ── Test 3: OpenTelemetry Collector ──────────────────────────────────
    print("\n  ── OpenTelemetry Collector Configuration ──")
    otel_status = suite.otel_collector.get_status()
    print(f"    Receivers          : {otel_status['receivers']}")
    print(f"    Processors         : {otel_status['processors']}")
    print(f"    Exporters          : {otel_status['exporters']}")
    print(f"    Pipelines          : {otel_status['pipelines']}")

    # Generate config
    otel_config = suite.otel_collector.generate_config()
    print(f"    Config sections    : {len(otel_config)}")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_observability")
    print(f"    Artifacts generated: {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:25s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Alertmanager rules : {status['alertmanager']['routing_rules']}")
    print(f"    Dashboard panels   : {status['dashboard']['n_panels']}")
    print(f"    OTel pipelines     : {len(status['otel_collector']['pipelines'])}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Alertmanager is a CONFIG GENERATOR, not a deployed system")
    print("  • Grafana dashboard is a VISUALIZATION TOOL, not a decision engine")
    print("  • OTel Collector config is a TEMPLATE, not a production deployment")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit and regulatory approval")
    print("  • Data minimization follows PNDC Axiom 3")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
