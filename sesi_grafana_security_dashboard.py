"""
=============================================================================
SESI GRAFANA SECURITY MONITORING DASHBOARD
Real-time security metrics, alert rules, and anomaly detection
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[GD1] GrafanaSecurityDashboardGenerator
      • Complete Grafana dashboard JSON model
      • Panels: auth events, zone violations, alert rates, system health
      • Alert rules with thresholds

[GD2] SecurityMetricsExporter
      • Prometheus metrics for security events
      • Counters, gauges, histograms

[GD3] AnomalyDetectionPanel
      • Degenerate occupation-time law detection (paper6 Thm 4.1)
      • Growth-rate dichotomy classification (paper6 Thm 5.1)
      • Change-point detection for attack patterns

RELATION TO PAPER6 (CHANGE-POINT TOWERS)
------------------------------------------
From paper6 Thm 4.1:
  The occupation fraction f(n) does not converge a.s., but converges
  in distribution to the symmetric two-point law ½δ₀ + ½δ₁.

This means: security metrics should NOT be interpreted as "stable averages".
A system oscillating between "safe" and "under attack" states may show
a misleading average. The dashboard must show DISTRIBUTION, not just mean.

From paper6 Thm 5.1 (growth-rate dichotomy):
  θ > 1:      finite mean → ordinary monitoring
  θ ∈ (0,1):  regularly varying → distributional monitoring
  θ → 0:      degenerate → polarized monitoring (all-or-nothing)

The dashboard classifies the current security regime and adjusts
visualization accordingly.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Dashboard displays information; it does not make security decisions.
• Alert rules are HEURISTIC; novel attacks may evade detection.
• Degenerate regime detection is STATISTICAL, not deterministic.
• All thresholds are FREE DESIGN PARAMETERS.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.grafana_dashboard")

__all__ = [
    "SecurityMetricsExporter",
    "GrafanaSecurityDashboardGenerator",
    "AnomalyDetectionPanel",
    "SecurityDashboardSuite",
]


class SecurityMetricsExporter:
    """
    [D] Prometheus metrics exporter for security events.

    Metrics:
      • sesi_auth_events_total{result="success|failure"}
      • sesi_zone_violations_total{src_zone, dst_zone}
      • sesi_alerts_total{level}
      • sesi_emergency_stops_total
      • sesi_mtls_handshake_failures_total
      • sesi_connection_rate_current
      • sesi_nivarana_detection_total{hindrance}
      • sesi_cetasika_state{attribute, value}
    """

    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

    def inc_counter(self, name: str, labels: Optional[Dict[str, str]] = None,
                    value: float = 1.0) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def set_gauge(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge."""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def observe_histogram(self, name: str, value: float,
                          labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a histogram value."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []

        # Counters
        for key, value in self._counters.items():
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in self._gauges.items():
            lines.append(f"{key} {value}")

        # Histograms (simplified: just report count and sum)
        for key, values in self._histograms.items():
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")

        return "\n".join(lines)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": self._counters,
            "gauges": self._gauges,
            "histograms": {k: {"count": len(v), "sum": sum(v)}
                           for k, v in self._histograms.items()},
        }


class AnomalyDetectionPanel:
    """
    [D] Anomaly detection based on paper6 change-point theory.

    Detects:
      • Degenerate occupation-time law (½δ₀ + ½δ₁) — paper6 Thm 4.1
      • Growth-rate dichotomy — paper6 Thm 5.1
      • Change points in security event streams

    HONEST: This is a STATISTICAL detector, not a proof of attack.
    Novel attacks may not trigger detection.
    """

    def __init__(self, degenerate_threshold: float = 0.9):
        self.degenerate_threshold = degenerate_threshold
        self.event_history: List[float] = []

    def record_event(self, severity: float) -> None:
        """Record a security event with severity [0, 1]."""
        self.event_history.append(severity)

    def detect_degenerate_regime(self) -> Dict[str, Any]:
        """
        Detect if security events follow degenerate occupation-time law.

        From paper6 Thm 4.1:
          f(n) → ½δ₀ + ½δ₁ means the system oscillates between
          "all safe" and "all under attack" with no stable average.

        Detection: check if occupation fraction is near 0 or 1.
        """
        if len(self.event_history) < 10:
            return {"regime": "insufficient_data", "n_events": len(self.event_history)}

        # Occupation fraction: fraction of events above threshold
        threshold = 0.5
        above = sum(1 for e in self.event_history if e > threshold)
        occupation = above / len(self.event_history)

        # Check for degenerate polarization
        is_degenerate = occupation < 0.1 or occupation > 0.9

        # Check for heavy tails (regularly varying)
        cv = np.std(self.event_history) / (np.mean(self.event_history) + 1e-8)
        is_heavy_tailed = cv > 2.0

        if is_degenerate:
            regime = "DEGENERATE"
            clinical = (
                "Security state is POLARIZED — oscillating between extremes. "
                "AVERAGING-BASED monitoring will MISS this. "
                "Use change-point detection ('retime, don't average')."
            )
            urgency = "URGENT"
        elif is_heavy_tailed:
            regime = "REGULARLY_VARYING"
            clinical = (
                "Security events have heavy tails. "
                "Distributional monitoring (quantiles, not means) recommended."
            )
            urgency = "REVIEW"
        else:
            regime = "FINITE_MEAN"
            clinical = "Security events are well-behaved. Standard monitoring appropriate."
            urgency = "ROUTINE"

        return {
            "regime": regime,
            "occupation_fraction": occupation,
            "cv": cv,
            "clinical_interpretation": clinical,
            "urgency": urgency,
            "note": (
                "Based on paper6 Thm 4.1 (degenerate occupation-time law) "
                "and Thm 5.1 (growth-rate dichotomy). "
                "This is STATISTICAL classification, not attack detection."
            ),
        }

    def detect_change_point(self, window_size: int = 50) -> Dict[str, Any]:
        """
        Detect change points in security event stream.

        Uses simple CUSUM-style detection.

        HONEST: This is a SIMPLIFIED detector. Production should use
        proper CUSUM or Bayesian change-point detection.
        """
        if len(self.event_history) < window_size * 2:
            return {"change_point_detected": False, "reason": "insufficient_data"}

        recent = self.event_history[-window_size:]
        historical = self.event_history[-2*window_size:-window_size]

        recent_mean = np.mean(recent)
        historical_mean = np.mean(historical)
        recent_std = np.std(recent) + 1e-8

        # CUSUM-style statistic
        cusum = abs(recent_mean - historical_mean) / recent_std

        change_detected = cusum > 3.0  # threshold = 3 standard deviations

        return {
            "change_point_detected": change_detected,
            "cusum_statistic": cusum,
            "recent_mean": recent_mean,
            "historical_mean": historical_mean,
            "note": (
                "Simplified CUSUM change-point detection. "
                "For production, use proper CUSUM or Bayesian methods. "
                "Paper6 §3.2: change-point detection is for identifying "
                "regime changes, not for predicting attacks."
            ),
        }


class GrafanaSecurityDashboardGenerator:
    """
    [D] Generate Grafana dashboard JSON for security monitoring.

    Panels:
      • Row 1: Overview — auth events, zone violations, alerts
      • Row 2: Zone traffic — heatmap of cross-zone connections
      • Row 3: Anomaly detection — occupation-time law, change points
      • Row 4: Buddhist metrics — nivarana detection, cetasika state
      • Row 5: System health — kill switch, emergency stops
    """

    def __init__(self, dashboard_title: str = "SESI Security Monitoring"):
        self.dashboard_title = dashboard_title

    def _panel(self, panel_id: int, title: str, panel_type: str,
               targets: List[Dict[str, Any]], grid_pos: Dict[str, int],
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a Grafana panel."""
        panel = {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "datasource": {"type": "prometheus", "uid": "prometheus"},
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
            # ── Row 1: Overview ─────────────────────────────────────────────
            self._panel(1, "Authentication Events", "timeseries",
                        [{"expr": 'sum(rate(sesi_auth_events_total[5m])) by (result)',
                          "legendFormat": "{{result}}"}],
                        {"h": 8, "w": 8, "x": 0, "y": 0}),

            self._panel(2, "Zone Violations", "timeseries",
                        [{"expr": 'sum(rate(sesi_zone_violations_total[5m])) by (src_zone, dst_zone)',
                          "legendFormat": "{{src_zone}}→{{dst_zone}}"}],
                        {"h": 8, "w": 8, "i": 8, "y": 0}),

            self._panel(3, "Alert Rate", "stat",
                        [{"expr": 'sum(rate(sesi_alerts_total[5m]))',
                          "legendFormat": "alerts/s"}],
                        {"h": 8, "w": 8, "i": 16, "y": 0},
                        extra={"fieldConfig": {"defaults": {
                            "thresholds": {"mode": "absolute", "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 0.1},
                                {"color": "red", "value": 1.0}]}}}}),

            # ── Row 2: Zone Traffic ────────────────────────────────────────
            self._panel(4, "Cross-Zone Connections", "heatmap",
                        [{"expr": 'sum(rate(sesi_cross_zone_connections_total[5m])) by (src_zone, dst_zone)',
                          "legendFormat": "{{src_zone}}→{{dst_zone}}",
                          "format": "heatmap"}],
                        {"h": 8, "w": 12, "i": 0, "y": 8}),

            self._panel(5, "mTLS Handshake Failures", "timeseries",
                        [{"expr": 'sum(rate(sesi_mtls_handshake_failures_total[5m]))',
                          "legendFormat": "failures"}],
                        {"h": 8, "w": 12, "i": 12, "y": 8}),

            # ── Row 3: Anomaly Detection ───────────────────────────────────
            self._panel(6, "Security Regime (paper6)", "stat",
                        [{"expr": 'sesi_security_regime',
                          "legendFormat": "regime"}],
                        {"h": 8, "w": 8, "i": 0, "y": 16},
                        extra={"fieldConfig": {"defaults": {
                            "mappings": [
                                {"type": "value", "options": {
                                    "0": {"text": "FINITE_MEAN", "color": "green"},
                                    "1": {"text": "REGULARLY_VARYING", "color": "yellow"},
                                    "2": {"text": "DEGENERATE", "color": "red"}}}]}}}),

            self._panel(7, "Change-Point Detection", "stat",
                        [{"expr": 'sesi_change_point_detected',
                          "legendFormat": "change_point"}],
                        {"h": 8, "w": 8, "i": 8, "y": 16},
                        extra={"fieldConfig": {"defaults": {
                            "mappings": [
                                {"type": "value", "options": {
                                    "0": {"text": "NO CHANGE", "color": "green"},
                                    "1": {"text": "CHANGE DETECTED", "color": "red"}}}]}}}),

            self._panel(8, "Connection Rate (No-Zeno)", "gauge",
                        [{"expr": 'sesi_connection_rate_current',
                          "legendFormat": "connections"}],
                        {"h": 8, "w": 8, "i": 16, "y": 16},
                        extra={"fieldConfig": {"defaults": {
                            "min": 0, "max": 100,
                            "thresholds": {"mode": "absolute", "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 50},
                                {"color": "red", "value": 80}]}}}}),

            # ── Row 4: Buddhist Metrics ────────────────────────────────────
            self._panel(9, "Nivarana Detection", "timeseries",
                        [{"expr": 'sum(rate(sesi_nivarana_detection_total[5m])) by (hindrance)',
                          "legendFormat": "{{hindrance}}"}],
                        {"h": 8, "w": 12, "i": 0, "y": 24}),

            self._panel(10, "Cetasika State", "table",
                        [{"expr": 'sesi_cetasika_state',
                          "legendFormat": "{{attribute}}",
                          "format": "table"}],
                        {"h": 8, "w": 12, "i": 12, "y": 24}),

            # ── Row 5: System Health ───────────────────────────────────────
            self._panel(11, "Kill Switch Status", "stat",
                        [{"expr": 'sesi_kill_switch_active',
                          "legendFormat": "kill_switch"}],
                        {"h": 8, "w": 8, "i": 0, "y": 32},
                        extra={"fieldConfig": {"defaults": {
                            "mappings": [
                                {"type": "value", "options": {
                                    "0": {"text": "INACTIVE", "color": "green"},
                                    "1": {"text": "ACTIVE", "color": "red"}}}]}}}),

            self._panel(12, "Emergency Stops", "stat",
                        [{"expr": 'sum(increase(sesi_emergency_stops_total[24h]))',
                          "legendFormat": "stops_24h"}],
                        {"h": 8, "w": 8, "i": 8, "y": 32}),

            self._panel(13, "System Uptime", "stat",
                        [{"expr": 'time() - process_start_time_seconds{job="sesi-oversight"}',
                          "legendFormat": "uptime"}],
                        {"h": 8, "w": 8, "i": 16, "y": 32},
                        extra={"fieldConfig": {"defaults": {"unit": "s"}}}),
        ]

        dashboard = {
            "uid": "sesi-security",
            "title": self.dashboard_title,
            "tags": ["sesi", "security", "oversight"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "templating": {"list": []},
            "annotations": {"list": []},
        }

        return dashboard

    def generate_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Generate Prometheus alert rules for security events.

        Rules:
          • High auth failure rate
          • Zone violation detected
          • Kill switch activated
          • Degenerate security regime
        """
        return [
            {
                "alert": "HighAuthFailureRate",
                "expr": 'sum(rate(sesi_auth_events_total{result="failure"}[5m])) > 0.1',
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High authentication failure rate",
                    "description": "Auth failure rate is above 10% for 5 minutes",
                },
            },
            {
                "alert": "ZoneViolationDetected",
                "expr": 'sum(rate(sesi_zone_violations_total[5m])) > 0',
                "for": "1m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Zone violation detected",
                    "description": "Cross-zone traffic violation detected",
                },
            },
            {
                "alert": "KillSwitchActivated",
                "expr": 'sesi_kill_switch_active == 1',
                "for": "1m",
                "labels": {"severity": "emergency"},
                "annotations": {
                    "summary": "Kill switch activated",
                    "description": "Emergency shutdown has been triggered",
                },
            },
            {
                "alert": "DegenerateSecurityRegime",
                "expr": 'sesi_security_regime == 2',
                "for": "5m",
                "labels": {"severity": "critical"},
                "annotations": {
                    "summary": "Degenerate security regime detected",
                    "description": (
                        "Security events are oscillating between extremes "
                        "(paper6 Thm 4.1: degenerate occupation-time law). "
                        "Averaging-based monitoring will miss this. "
                        "Investigate immediately."
                    ),
                },
            },
        ]

    def write_dashboard(self, output_path: str = "./monitoring/grafana/security-dashboard.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Grafana dashboard written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./monitoring/prometheus/security-alerts.yaml") -> str:
        """Write Prometheus alert rules to file."""
        rules = self.generate_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_YAML:
            with open(out_path, 'w') as f:
                yaml.dump({"groups": [{"name": "sesi_security", "rules": rules}]},
                          f, default_flow_style=False)
        else:
            with open(out_path, 'w') as f:
                json.dump({"groups": [{"name": "sesi_security", "rules": rules}]}, f, indent=2)

        logger.info(f"Alert rules written to {output_path}")
        return str(out_path)


class SecurityDashboardSuite:
    """
    [D] Complete security dashboard suite.
    """

    def __init__(self):
        self.metrics_exporter = SecurityMetricsExporter()
        self.anomaly_detector = AnomalyDetectionPanel()
        self.dashboard_gen = GrafanaSecurityDashboardGenerator()

    def record_security_event(self, event_type: str, severity: float,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a security event."""
        self.anomaly_detector.record_event(severity)
        self.metrics_exporter.inc_counter("sesi_security_events_total",
                                          {"type": event_type})
        self.metrics_exporter.observe_histogram("sesi_security_event_severity",
                                                severity)

    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get anomaly detection report."""
        return {
            "regime": self.anomaly_detector.detect_degenerate_regime(),
            "change_point": self.anomaly_detector.detect_change_point(),
        }

    def generate_all(self, output_dir: str = "./monitoring") -> Dict[str, str]:
        """Generate all dashboard files."""
        files = {}
        files["dashboard"] = self.dashboard_gen.write_dashboard(
            f"{output_dir}/grafana/security-dashboard.json")
        files["alert_rules"] = self.dashboard_gen.write_alert_rules(
            f"{output_dir}/prometheus/security-alerts.yaml")
        return files


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Grafana Security Dashboard — Smoke Test")
    print("=" * 72)

    suite = SecurityDashboardSuite()

    # Simulate security events
    print(f"\n  ── Simulating Security Events ──")
    for i in range(20):
        severity = np.random.random()
        suite.record_security_event("auth_failure", severity)
    print(f"    Recorded 20 security events")

    # Check anomaly detection
    print(f"\n  ── Anomaly Detection (paper6) ──")
    report = suite.get_anomaly_report()
    print(f"    Security regime      : {report['regime']['regime']}")
    print(f"    Occupation fraction  : {report['regime']['occupation_fraction']:.3f}")
    print(f"    Change point         : {report['change_point']['change_point_detected']}")

    # Export metrics
    print(f"\n  ── Prometheus Metrics ──")
    metrics = suite.metrics_exporter.export_prometheus()
    print(f"    Metrics exported     : {len(metrics.split(chr(10)))} lines")

    # Generate dashboard
    print(f"\n  ── Dashboard Generation ──")
    files = suite.generate_all("./test_monitoring")
    for name, path in files.items():
        print(f"    {name:15s} : {path}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Dashboard displays information; does not make decisions")
    print("  • Alert rules are HEURISTIC; novel attacks may evade")
    print("  • Degenerate regime detection is STATISTICAL")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
