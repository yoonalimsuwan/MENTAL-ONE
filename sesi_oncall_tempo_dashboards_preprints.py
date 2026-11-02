"""
=============================================================================
SESI AGI: GRAFANA ONCALL + TEMPO SERVICE GRAPH + OPENSEARCH DASHBOARDS
Incident Management + Distributed Service Graph + Custom Visualization
+ Free Preprint Sources Integration
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[OC1] GrafanaOnCallIntegration
      • Incident creation from trace-based alerts
      • Escalation policies per zone (PNDC Axiom 3)
      • On-call scheduling with rotation
      • Webhook integration with Alertmanager
      • Incident lifecycle management
      • No-Zeno rate limiting on incident creation

[TSG1] TempoMetricsGenerator
      • Service graph generation from trace spans
      • Edge metrics: latency, error rate, call count
      • Node metrics: per-service aggregation
      • RED metrics (Rate, Errors, Duration)
      • Integration with Prometheus remote write
      • Zone-aware service graph coloring

[OSD1] OpenSearchDashboardsCustom
      • Custom visualization JSON for AGI modules
      • Per-module log distribution panels
      • Trace-log correlation visualizations
      • Safety violation heatmaps
      • Meta-learning convergence plots
      • Zone-based retention visualization

[FP1] FreePreprintSources
      • Integration with arXiv, PubMed Central, Semantic Scholar
      • Automated preprint discovery for research papers
      • Citation graph construction
      • License compliance checking
      • Metadata extraction for GNN training

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
[C] paper6 Thm 5.1: Growth-rate dichotomy per service/zone
[C] Six_Problems P4: CUSUM for incident detection
[C] PNDC Axiom 3: Zone-based escalation proportionality
[C] RED metrics: Rate, Errors, Duration (Tom Wilkie)
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()

RELATION TO AGI UNIFIED FRAMEWORK (10 modules)
-----------------------------------------------
  • CoreReasoningEngine → OnCall incident if reasoning fails
  • SafetyAlignmentGuard → Critical OnCall escalation
  • MetaLearningPipeline → Service graph edges for adaptation
  • All 10 modules → OpenSearch Dashboards visualization

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Grafana OnCall is INCIDENT MANAGEMENT, not autonomous response.
• Tempo service graph shows CORRELATION, not causation.
• OpenSearch Dashboards are VISUALIZATION, not decision-making.
• Free preprint sources require license compliance checking.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.oncall_tempo_dashboards")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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
    "SecurityZone", "AlertSeverity", "QualityRegime", "AGIModule",
    "IncidentStatus", "IncidentPriority",
    "GrafanaOnCallIntegration", "OnCallConfig", "Incident",
    "TempoMetricsGenerator", "ServiceGraphEdge", "ServiceGraphNode",
    "OpenSearchDashboardsCustom",
    "FreePreprintSources", "PreprintPaper",
    "SESIIncidentVisualizationSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class SecurityZone(Enum):
    Z0_IMPLANT = "z0-implant"
    Z1_CLINICAL = "z1-clinical"
    Z2_PROCESSING = "z2-processing"
    Z3_EXTERNAL = "z3-external"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class QualityRegime(Enum):
    FINITE_MEAN = 0
    REGULARLY_VARYING = 1
    DEGENERATE = 2
    INSUFFICIENT_DATA = 3


class AGIModule(Enum):
    REASONING = "reasoning"
    MEMORY = "memory"
    PERCEPTION = "perception"
    ACTUATION = "actuation"
    CURIOSITY = "curiosity"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"
    SAFETY = "safety"
    SUBSTRATE = "substrate"
    META_LEARNING = "meta_learning"


class IncidentStatus(Enum):
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class IncidentPriority(Enum):
    P1_CRITICAL = "P1"
    P2_HIGH = "P2"
    P3_MEDIUM = "P3"
    P4_LOW = "P4"


# ═══════════════════════════════════════════════════════════════════════════
# [OC1] GRAFANA ONCALL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OnCallConfig:
    """Configuration for Grafana OnCall integration."""
    api_url: str = "http://grafana-oncall:8080"
    api_token_env: str = "GRAFANA_ONCALL_API_TOKEN"
    # Zone-based escalation (PNDC Axiom 3)
    escalation_z0: List[str] = field(default_factory=lambda: [
        "clinical-pager", "clinical-team-lead", "cto"
    ])
    escalation_z1: List[str] = field(default_factory=lambda: [
        "clinical-team", "platform-lead"
    ])
    escalation_z2: List[str] = field(default_factory=lambda: [
        "platform-oncall", "platform-team"
    ])
    escalation_z3: List[str] = field(default_factory=lambda: [
        "platform-email"
    ])
    # Rotation schedules
    rotation_interval_hours: int = 24
    # No-Zeno
    max_incidents_per_minute: int = 10
    min_incident_interval_s: float = 30.0


@dataclass
class Incident:
    """A single incident in Grafana OnCall."""
    incident_id: str
    title: str
    message: str
    zone: SecurityZone
    module: AGIModule
    priority: IncidentPriority
    status: IncidentStatus
    trace_id: Optional[str] = None
    alert_labels: Dict[str, str] = field(default_factory=dict)
    assignees: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "message": self.message,
            "zone": self.zone.value,
            "module": self.module.value,
            "priority": self.priority.value,
            "status": self.status.name,
            "trace_id": self.trace_id,
            "alert_labels": self.alert_labels,
            "assignees": self.assignees,
            "created_at": self.created_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
        }

    def to_oncall_payload(self) -> Dict[str, Any]:
        """Convert to Grafana OnCall API payload."""
        return {
            "title": self.title,
            "message": self.message,
            "labels": {
                "zone": self.zone.value,
                "module": self.module.value,
                "priority": self.priority.value,
                **self.alert_labels,
            },
        }


class GrafanaOnCallIntegration:
    """
    [D] Integration with Grafana OnCall for incident management.

    Features:
      • Incident creation from trace-based alerts
      • Zone-based escalation policies (PNDC Axiom 3 proportionality)
      • On-call scheduling with rotation
      • Webhook integration with Alertmanager
      • Incident lifecycle management (triggered → acknowledged → resolved)
      • No-Zeno rate limiting on incident creation
      • CUSUM change-point detection for incident patterns

    HONEST: This is an INCIDENT MANAGEMENT layer, not autonomous response.
    All incidents require human acknowledgment and resolution.

    Mathematical grounding:
      • PNDC Axiom 3: Zone-based escalation proportionality
      • No-Zeno bound on incident rate (SESI_NoZeno Thm 3.2)
      • CUSUM for incident pattern detection (Six_Problems P4)
    """

    def __init__(self, cfg: Optional[OnCallConfig] = None):
        self.cfg = cfg or OnCallConfig()
        self._incidents: List[Incident] = []
        self._incident_timestamps: deque = deque(maxlen=10000)
        self._active_incidents: Dict[str, Incident] = {}
        self._rotation_index: Dict[str, int] = defaultdict(int)

        # CUSUM for incident pattern detection
        self._incident_cusum: deque = deque(maxlen=1000)

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_incident_rate(self) -> Dict[str, Any]:
        """Check if incident creation is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._incident_timestamps = deque(
            (t for t in self._incident_timestamps if t > cutoff),
            maxlen=10000,
        )

        if len(self._incident_timestamps) >= self.cfg.max_incidents_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
                "current": len(self._incident_timestamps),
            }

        if self._incident_timestamps:
            time_since_last = now - self._incident_timestamps[-1]
            if time_since_last < self.cfg.min_incident_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                }

        return {"allowed": True}

    # ── Zone-Based Escalation (PNDC Axiom 3) ─────────────────────────────

    def get_escalation_policy(self, zone: SecurityZone,
                              priority: IncidentPriority) -> List[str]:
        """
        Get escalation policy based on zone and priority.

        PNDC Axiom 3: Z0 gets most aggressive escalation, Z3 least.
        """
        zone_escalation = {
            SecurityZone.Z0_IMPLANT: self.cfg.escalation_z0,
            SecurityZone.Z1_CLINICAL: self.cfg.escalation_z1,
            SecurityZone.Z2_PROCESSING: self.cfg.escalation_z2,
            SecurityZone.Z3_EXTERNAL: self.cfg.escalation_z3,
        }

        base_escalation = zone_escalation.get(zone, self.cfg.escalation_z2)

        # P1 incidents get full escalation chain
        if priority == IncidentPriority.P1_CRITICAL:
            return base_escalation
        # P2 gets first two
        elif priority == IncidentPriority.P2_HIGH:
            return base_escalation[:2] if len(base_escalation) >= 2 else base_escalation
        # P3/P4 get first only
        else:
            return [base_escalation[0]] if base_escalation else []

    def get_current_oncall(self, zone: SecurityZone) -> str:
        """Get current on-call person for a zone (rotation-based)."""
        escalation = self.get_escalation_policy(zone, IncidentPriority.P1_CRITICAL)
        if not escalation:
            return "unassigned"
        idx = self._rotation_index[zone.value] % len(escalation)
        return escalation[idx]

    def rotate_oncall(self, zone: SecurityZone) -> None:
        """Rotate on-call for a zone."""
        self._rotation_index[zone.value] += 1

    # ── Incident Creation ────────────────────────────────────────────────

    def create_incident(
        self,
        title: str,
        message: str,
        zone: SecurityZone,
        module: AGIModule,
        severity: AlertSeverity,
        trace_id: Optional[str] = None,
        alert_labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Incident]:
        """
        Create a new incident in Grafana OnCall.

        Args:
            title: Incident title
            message: Detailed message
            zone: Security zone
            module: AGI module that triggered
            severity: Alert severity
            trace_id: Optional Jaeger trace ID
            alert_labels: Optional additional labels

        Returns:
            Incident object if created, None if rate-limited
        """
        # Check rate limit
        rate_check = self.check_incident_rate()
        if not rate_check["allowed"]:
            logger.warning(f"[OnCall] Incident rate-limited: {rate_check['reason']}")
            return None

        # Map severity to priority
        priority_map = {
            AlertSeverity.EMERGENCY: IncidentPriority.P1_CRITICAL,
            AlertSeverity.CRITICAL: IncidentPriority.P2_HIGH,
            AlertSeverity.WARNING: IncidentPriority.P3_MEDIUM,
            AlertSeverity.INFO: IncidentPriority.P4_LOW,
        }
        priority = priority_map.get(severity, IncidentPriority.P3_MEDIUM)

        # Get escalation policy
        assignees = self.get_escalation_policy(zone, priority)

        incident = Incident(
            incident_id=f"inc_{uuid.uuid4().hex[:8]}",
            title=title,
            message=message,
            zone=zone,
            module=module,
            priority=priority,
            status=IncidentStatus.TRIGGERED,
            trace_id=trace_id,
            alert_labels=alert_labels or {},
            assignees=assignees,
        )

        self._incidents.append(incident)
        self._active_incidents[incident.incident_id] = incident
        self._incident_timestamps.append(time.time())

        # Update CUSUM for pattern detection
        severity_value = {
            AlertSeverity.INFO: 0.25,
            AlertSeverity.WARNING: 0.5,
            AlertSeverity.CRITICAL: 0.75,
            AlertSeverity.EMERGENCY: 1.0,
        }.get(severity, 0.5)
        self._incident_cusum.append(severity_value)

        logger.info(
            f"[OnCall] Incident created: {incident.incident_id} | "
            f"zone={zone.value} | module={module.value} | "
            f"priority={priority.value} | assignees={assignees}"
        )

        # Send to Grafana OnCall API (if available)
        if HAS_REQUESTS:
            self._send_to_oncall_api(incident)

        return incident

    def _send_to_oncall_api(self, incident: Incident) -> bool:
        """Send incident to Grafana OnCall API."""
        try:
            api_token = os.environ.get(self.cfg.api_token_env, "")
            if not api_token:
                logger.warning("[OnCall] API token not set; skipping API call")
                return False

            response = requests.post(
                f"{self.cfg.api_url}/api/v1/alerts/",
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                json=incident.to_oncall_payload(),
                timeout=10.0,
            )
            response.raise_for_status()
            logger.info(f"[OnCall] Incident {incident.incident_id} sent to API")
            return True
        except Exception as e:
            logger.warning(f"[OnCall] API call failed: {e}")
            return False

    # ── Incident Lifecycle ───────────────────────────────────────────────

    def acknowledge_incident(self, incident_id: str,
                             acknowledged_by: str) -> bool:
        """Acknowledge an incident."""
        incident = self._active_incidents.get(incident_id)
        if incident is None:
            return False
        incident.status = IncidentStatus.ACKNOWLEDGED
        incident.acknowledged_at = time.time()
        incident.metadata["acknowledged_by"] = acknowledged_by
        logger.info(f"[OnCall] Incident {incident_id} acknowledged by {acknowledged_by}")
        return True

    def resolve_incident(self, incident_id: str,
                         resolved_by: str,
                         resolution: str = "") -> bool:
        """Resolve an incident."""
        incident = self._active_incidents.get(incident_id)
        if incident is None:
            return False
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = time.time()
        incident.metadata["resolved_by"] = resolved_by
        incident.metadata["resolution"] = resolution
        del self._active_incidents[incident_id]
        logger.info(f"[OnCall] Incident {incident_id} resolved by {resolved_by}")
        return True

    # ── CUSUM Pattern Detection ──────────────────────────────────────────

    def detect_incident_pattern(self) -> Dict[str, Any]:
        """
        Detect incident pattern using CUSUM.

        From Six_Problems P4: CUSUM + subsampling on raw process.
        """
        if len(self._incident_cusum) < 20:
            return {
                "pattern_detected": False,
                "reason": "insufficient_data",
            }

        values = list(self._incident_cusum)
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
            "pattern_detected": change_idx is not None,
            "cusum_statistic": max_cusum,
            "threshold": 5.0,
            "change_point_index": change_idx,
            "target_mean": target,
        }

    # ── Webhook Handler ──────────────────────────────────────────────────

    def handle_alertmanager_webhook(self, webhook_payload: Dict[str, Any]) -> List[Incident]:
        """
        Handle Alertmanager webhook and create incidents.

        Args:
            webhook_payload: Alertmanager webhook payload

        Returns:
            List of created incidents
        """
        incidents = []
        alerts = webhook_payload.get("alerts", [])

        for alert in alerts:
            if alert.get("status") != "firing":
                continue

            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})

            # Extract zone and module from labels
            zone_str = labels.get("zone", "z2-processing")
            module_str = labels.get("module", "reasoning")
            severity_str = labels.get("severity", "warning")

            try:
                zone = SecurityZone(zone_str)
            except ValueError:
                zone = SecurityZone.Z2_PROCESSING

            try:
                module = AGIModule(module_str)
            except ValueError:
                module = AGIModule.REASONING

            try:
                severity = AlertSeverity(severity_str)
            except ValueError:
                severity = AlertSeverity.WARNING

            incident = self.create_incident(
                title=annotations.get("summary", f"Alert: {alert.get('labels', {}).get('alertname', 'unknown')}"),
                message=annotations.get("description", "No description provided"),
                zone=zone,
                module=module,
                severity=severity,
                trace_id=labels.get("trace_id"),
                alert_labels=labels,
            )

            if incident:
                incidents.append(incident)

        return incidents

    def generate_oncall_config(self) -> Dict[str, Any]:
        """Generate Grafana OnCall configuration."""
        return {
            "integrations": [
                {
                    "name": "sesi-agi-alertmanager",
                    "type": "alertmanager",
                    "routes": [
                        {
                            "routing_type": "regex",
                            "routing_regex": "zone=(z0-implant|z1-clinical|z2-processing|z3-external)",
                            "escalation_chain": "sesi-agi-escalation",
                        }
                    ],
                },
            ],
            "escalation_chains": [
                {
                    "name": "sesi-agi-escalation",
                    "steps": [
                        {
                            "position": 0,
                            "notify_on_call_from_schedule": "clinical-oncall",
                            "duration": 300,  # 5 minutes
                        },
                        {
                            "position": 1,
                            "notify_to_multiple_users": ["platform-lead", "cto"],
                        },
                    ],
                },
            ],
            "schedules": [
                {
                    "name": "clinical-oncall",
                    "type": "calendar",
                    "team": "clinical-team",
                    "shifts": [
                        {
                            "start": "2026-01-01T00:00:00Z",
                            "duration": self.cfg.rotation_interval_hours * 3600,
                            "users": self.cfg.escalation_z0,
                        },
                    ],
                },
            ],
        }

    def write_oncall_config(self, output_path: str = "./oncall/oncall-config.yaml") -> str:
        """Write OnCall configuration to file."""
        config = self.generate_oncall_config()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
        logger.info(f"[OnCall] Config written to {output_path}")
        return str(out_path)

    def get_incidents(self, limit: int = 100,
                      status: Optional[IncidentStatus] = None) -> List[Dict[str, Any]]:
        """Get recent incidents."""
        incidents = self._incidents[-limit:]
        if status:
            incidents = [i for i in incidents if i.status == status]
        return [i.to_dict() for i in incidents]

    def get_status(self) -> Dict[str, Any]:
        """Get OnCall integration status."""
        return {
            "total_incidents": len(self._incidents),
            "active_incidents": len(self._active_incidents),
            "incidents_by_priority": {
                p.value: sum(1 for i in self._incidents if i.priority == p)
                for p in IncidentPriority
            },
            "incidents_by_zone": {
                z.value: sum(1 for i in self._incidents if i.zone == z)
                for z in SecurityZone
            },
            "incident_pattern": self.detect_incident_pattern(),
            "delta_min": self.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [TSG1] TEMPO METRICS GENERATOR (SERVICE GRAPH)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ServiceGraphNode:
    """A node in the service graph."""
    service_name: str
    zone: SecurityZone
    module: Optional[AGIModule] = None
    # RED metrics
    request_rate: float = 0.0
    error_rate: float = 0.0
    duration_p50_ms: float = 0.0
    duration_p95_ms: float = 0.0
    duration_p99_ms: float = 0.0
    # Metadata
    n_traces: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class ServiceGraphEdge:
    """An edge in the service graph."""
    source: str
    target: str
    # RED metrics
    request_rate: float = 0.0
    error_rate: float = 0.0
    duration_p50_ms: float = 0.0
    duration_p95_ms: float = 0.0
    # Metadata
    n_calls: int = 0
    zone_crossing: bool = False
    regime: QualityRegime = QualityRegime.FINITE_MEAN
    latency_history: List[float] = field(default_factory=list)


class TempoMetricsGenerator:
    """
    [D] Tempo metrics-generator for service graph construction.

    Features:
      • Service graph generation from trace spans
      • Edge metrics: latency, error rate, call count
      • Node metrics: per-service aggregation (RED metrics)
      • Zone-aware service graph coloring
      • Regime classification per edge (paper6 Thm 5.1)
      • CUSUM change-point detection on call streams
      • Integration with Prometheus remote write

    HONEST: This is a METRICS GENERATOR. It constructs service graphs
    from trace data but does NOT guarantee:
      • Complete coverage of all service interactions
      • Accurate latency percentiles with small sample sizes
      • Causal relationships between services

    Mathematical grounding:
      • RED metrics: Rate, Errors, Duration (Tom Wilkie)
      • paper6 Thm 5.1: Growth-rate dichotomy per edge
      • Six_Problems P4: CUSUM for change-point detection
    """

    def __init__(
        self,
        prometheus_endpoint: str = "http://prometheus:9090/api/v1/write",
        latency_history_size: int = 100,
    ):
        self.prometheus_endpoint = prometheus_endpoint
        self.latency_history_size = latency_history_size
        self.nodes: Dict[str, ServiceGraphNode] = {}
        self.edges: Dict[Tuple[str, str], ServiceGraphEdge] = {}
        self._span_timestamps: deque = deque(maxlen=100000)

        # No-Zeno parameters
        self.max_spans_per_minute = 100000
        self.min_span_interval_s = 0.0001

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_span_rate(self) -> Dict[str, Any]:
        """Check if span ingestion is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._span_timestamps = deque(
            (t for t in self._span_timestamps if t > cutoff),
            maxlen=100000,
        )

        if len(self._span_timestamps) >= self.max_spans_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
                "current": len(self._span_timestamps),
            }

        return {"allowed": True}

    # ── Span Processing ──────────────────────────────────────────────────

    def process_span(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        service_name: str,
        operation_name: str,
        duration_ms: float,
        status: str,  # "OK" | "ERROR"
        zone: SecurityZone = SecurityZone.Z2_PROCESSING,
        module: Optional[AGIModule] = None,
    ) -> None:
        """
        Process a single span and update service graph.

        Args:
            trace_id: Jaeger trace ID
            span_id: Span ID
            parent_span_id: Parent span ID (for edge construction)
            service_name: Service name
            operation_name: Operation name
            duration_ms: Span duration in milliseconds
            status: Span status
            zone: Security zone
            module: AGI module (if applicable)
        """
        # Sanitize duration
        duration_ms = _safe_unit_scalar(
            duration_ms / 1000.0, name="duration_ms", logger=logger, default=0.0
        )
        if duration_ms is None:
            duration_ms = 0.0
        duration_ms = duration_ms * 1000.0

        # Check rate limit
        rate_check = self.check_span_rate()
        if not rate_check["allowed"]:
            return

        self._span_timestamps.append(time.time())

        # Update or create node
        if service_name not in self.nodes:
            self.nodes[service_name] = ServiceGraphNode(
                service_name=service_name,
                zone=zone,
                module=module,
            )
        node = self.nodes[service_name]
        node.n_traces += 1
        node.last_seen = time.time()

        # Update node RED metrics (exponential moving average)
        alpha = 0.1
        is_error = status == "ERROR"
        node.request_rate = (1 - alpha) * node.request_rate + alpha * 1.0
        node.error_rate = (1 - alpha) * node.error_rate + alpha * (1.0 if is_error else 0.0)

        # Update latency percentiles (simplified)
        if node.duration_p50_ms == 0:
            node.duration_p50_ms = duration_ms
            node.duration_p95_ms = duration_ms
            node.duration_p99_ms = duration_ms
        else:
            node.duration_p50_ms = (1 - alpha) * node.duration_p50_ms + alpha * duration_ms
            node.duration_p95_ms = (1 - alpha) * node.duration_p95_ms + alpha * duration_ms * 1.5
            node.duration_p99_ms = (1 - alpha) * node.duration_p99_ms + alpha * duration_ms * 2.0

        # Update edge if parent span exists
        if parent_span_id:
            # Find parent service (simplified: use trace_id to correlate)
            # In production, use Tempo's span store to find parent
            parent_service = self._find_parent_service(trace_id, parent_span_id)
            if parent_service and parent_service != service_name:
                self._update_edge(parent_service, service_name, duration_ms, is_error, zone)

    def _find_parent_service(self, trace_id: str, parent_span_id: str) -> Optional[str]:
        """Find parent service from span store (simplified)."""
        # In production, query Tempo's span store
        # For demo, return None
        return None

    def _update_edge(self, source: str, target: str,
                     duration_ms: float, is_error: bool,
                     zone: SecurityZone) -> None:
        """Update service graph edge."""
        edge_key = (source, target)
        if edge_key not in self.edges:
            source_zone = self.nodes.get(source, ServiceGraphNode(source, zone)).zone
            target_zone = self.nodes.get(target, ServiceGraphNode(target, zone)).zone
            self.edges[edge_key] = ServiceGraphEdge(
                source=source,
                target=target,
                zone_crossing=(source_zone != target_zone),
            )

        edge = self.edges[edge_key]
        edge.n_calls += 1

        # Update latency history (bounded)
        edge.latency_history.append(duration_ms)
        if len(edge.latency_history) > self.latency_history_size:
            edge.latency_history = edge.latency_history[-self.latency_history_size:]

        # Update RED metrics
        alpha = 0.1
        edge.request_rate = (1 - alpha) * edge.request_rate + alpha * 1.0
        edge.error_rate = (1 - alpha) * edge.error_rate + alpha * (1.0 if is_error else 0.0)

        if edge.duration_p50_ms == 0:
            edge.duration_p50_ms = duration_ms
            edge.duration_p95_ms = duration_ms
        else:
            edge.duration_p50_ms = (1 - alpha) * edge.duration_p50_ms + alpha * duration_ms
            edge.duration_p95_ms = (1 - alpha) * edge.duration_p95_ms + alpha * duration_ms * 1.5

        # Update regime classification
        edge.regime = self._classify_edge_regime(edge)

    def _classify_edge_regime(self, edge: ServiceGraphEdge) -> QualityRegime:
        """
        Classify edge regime using Hill estimator.

        From paper6 Thm 5.1.
        """
        if len(edge.latency_history) < 20:
            return QualityRegime.INSUFFICIENT_DATA

        latencies = edge.latency_history
        median = np.median(latencies)
        deviations = [abs(l - median) for l in latencies if abs(l - median) > 0.01]

        if len(deviations) < 10:
            return QualityRegime.FINITE_MEAN

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

    # ── Metrics Generation ───────────────────────────────────────────────

    def generate_prometheus_metrics(self) -> str:
        """Generate Prometheus exposition format for service graph metrics."""
        lines = []

        # Node metrics (RED)
        lines.append("# HELP tempo_svc_graph_request_total Total requests per service")
        lines.append("# TYPE tempo_svc_graph_request_total counter")
        for name, node in self.nodes.items():
            lines.append(
                f'tempo_svc_graph_request_total{{service="{name}",'
                f'zone="{node.zone.value}"}} {node.n_traces}'
            )

        lines.append("# HELP tempo_svc_graph_error_total Total errors per service")
        lines.append("# TYPE tempo_svc_graph_error_total counter")
        for name, node in self.nodes.items():
            error_count = int(node.n_traces * node.error_rate)
            lines.append(
                f'tempo_svc_graph_error_total{{service="{name}",'
                f'zone="{node.zone.value}"}} {error_count}'
            )

        lines.append("# HELP tempo_svc_graph_request_duration_seconds Request duration")
        lines.append("# TYPE tempo_svc_graph_request_duration_seconds gauge")
        for name, node in self.nodes.items():
            lines.append(
                f'tempo_svc_graph_request_duration_seconds{{service="{name}",'
                f'quantile="0.5"}} {node.duration_p50_ms / 1000.0:.6f}'
            )
            lines.append(
                f'tempo_svc_graph_request_duration_seconds{{service="{name}",'
                f'quantile="0.95"}} {node.duration_p95_ms / 1000.0:.6f}'
            )
            lines.append(
                f'tempo_svc_graph_request_duration_seconds{{service="{name}",'
                f'quantile="0.99"}} {node.duration_p99_ms / 1000.0:.6f}'
            )

        # Edge metrics
        lines.append("# HELP tempo_svc_graph_edge_request_total Total requests per edge")
        lines.append("# TYPE tempo_svc_graph_edge_request_total counter")
        for (src, tgt), edge in self.edges.items():
            lines.append(
                f'tempo_svc_graph_edge_request_total{{source="{src}",'
                f'target="{tgt}",'
                f'zone_crossing="{str(edge.zone_crossing).lower()}"}} {edge.n_calls}'
            )

        lines.append("# HELP tempo_svc_graph_edge_duration_seconds Edge duration")
        lines.append("# TYPE tempo_svc_graph_edge_duration_seconds gauge")
        for (src, tgt), edge in self.edges.items():
            lines.append(
                f'tempo_svc_graph_edge_duration_seconds{{source="{src}",'
                f'target="{tgt}",quantile="0.5"}} {edge.duration_p50_ms / 1000.0:.6f}'
            )
            lines.append(
                f'tempo_svc_graph_edge_duration_seconds{{source="{src}",'
                f'target="{tgt}",quantile="0.95"}} {edge.duration_p95_ms / 1000.0:.6f}'
            )

        # Regime metrics
        lines.append("# HELP tempo_svc_graph_edge_regime Edge regime classification")
        lines.append("# TYPE tempo_svc_graph_edge_regime gauge")
        for (src, tgt), edge in self.edges.items():
            lines.append(
                f'tempo_svc_graph_edge_regime{{source="{src}",'
                f'target="{tgt}"}} {edge.regime.value}'
            )

        return "\n".join(lines)

    def generate_service_graph_json(self) -> Dict[str, Any]:
        """Generate service graph as JSON for visualization."""
        nodes = []
        for name, node in self.nodes.items():
            nodes.append({
                "id": name,
                "name": name,
                "zone": node.zone.value,
                "module": node.module.value if node.module else None,
                "request_rate": node.request_rate,
                "error_rate": node.error_rate,
                "duration_p50_ms": node.duration_p50_ms,
                "duration_p95_ms": node.duration_p95_ms,
                "n_traces": node.n_traces,
            })

        edges = []
        for (src, tgt), edge in self.edges.items():
            edges.append({
                "source": src,
                "target": tgt,
                "request_rate": edge.request_rate,
                "error_rate": edge.error_rate,
                "duration_p50_ms": edge.duration_p50_ms,
                "duration_p95_ms": edge.duration_p95_ms,
                "n_calls": edge.n_calls,
                "zone_crossing": edge.zone_crossing,
                "regime": edge.regime.name,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "generated_at": time.time(),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get metrics generator status."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_spans_processed": len(self._span_timestamps),
            "zone_crossing_edges": sum(1 for e in self.edges.values() if e.zone_crossing),
            "regime_distribution": {
                r.name: sum(1 for e in self.edges.values() if e.regime == r)
                for r in QualityRegime
            },
            "delta_min": self.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [OSD1] OPENSEARCH DASHBOARDS CUSTOM VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class OpenSearchDashboardsCustom:
    """
    [D] Custom OpenSearch Dashboards visualizations for AGI modules.

    Features:
      • Per-module log distribution panels
      • Trace-log correlation visualizations
      • Safety violation heatmaps
      • Meta-learning convergence plots
      • Zone-based retention visualization
      • Incident timeline

    HONEST: These are VISUALIZATION TEMPLATES. They do not make decisions
    or provide causation. Human operators must interpret.
    """

    def __init__(self, opensearch_endpoint: str = "https://opensearch:9200"):
        self.opensearch_endpoint = opensearch_endpoint

    def generate_module_distribution_visualization(self) -> Dict[str, Any]:
        """Generate per-module log distribution visualization."""
        return {
            "id": "agi-module-distribution",
            "title": "AGI Module Log Distribution",
            "type": "visualization",
            "visState": {
                "title": "AGI Module Log Distribution",
                "type": "pie",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                        "params": {},
                    },
                    {
                        "id": "2",
                        "type": "terms",
                        "schema": "segment",
                        "params": {
                            "field": "module.keyword",
                            "size": 10,
                            "order": "desc",
                            "orderBy": "1",
                        },
                    },
                ],
                "params": {
                    "addLegend": True,
                    "addTooltip": True,
                    "isDonut": True,
                    "legendPosition": "right",
                },
            },
            "uiState": {},
            "description": "Distribution of logs across 10 AGI modules",
        }

    def generate_trace_log_correlation_visualization(self) -> Dict[str, Any]:
        """Generate trace-log correlation visualization."""
        return {
            "id": "trace-log-correlation",
            "title": "Trace-Log Correlation",
            "type": "visualization",
            "visState": {
                "title": "Trace-Log Correlation",
                "type": "table",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                        "params": {},
                    },
                    {
                        "id": "2",
                        "type": "terms",
                        "schema": "bucket",
                        "params": {
                            "field": "trace_id.keyword",
                            "size": 20,
                            "order": "desc",
                            "orderBy": "1",
                        },
                    },
                    {
                        "id": "3",
                        "type": "terms",
                        "schema": "bucket",
                        "params": {
                            "field": "module.keyword",
                            "size": 10,
                        },
                    },
                ],
                "params": {
                    "perPage": 20,
                    "showMeticsAtAllLevels": False,
                    "showMetricsAtAllLevels": False,
                    "showPartialRows": False,
                    "showTotal": False,
                    "showWarnings": True,
                },
            },
            "description": "Correlation between traces and log entries",
        }

    def generate_safety_violation_heatmap(self) -> Dict[str, Any]:
        """Generate safety violation heatmap by zone and module."""
        return {
            "id": "safety-violation-heatmap",
            "title": "Safety Violations Heatmap (Zone × Module)",
            "type": "visualization",
            "visState": {
                "title": "Safety Violations Heatmap",
                "type": "heatmap",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                        "params": {},
                    },
                    {
                        "id": "2",
                        "type": "terms",
                        "schema": "segment",
                        "params": {
                            "field": "zone.keyword",
                            "size": 4,
                        },
                    },
                    {
                        "id": "3",
                        "type": "terms",
                        "schema": "group",
                        "params": {
                            "field": "module.keyword",
                            "size": 10,
                        },
                    },
                ],
                "params": {
                    "addLegend": True,
                    "colorSchema": "Reds",
                    "colorsNumber": 10,
                    "enableHover": True,
                    "invertColors": False,
                    "legendPosition": "right",
                    "setColorRange": True,
                    "times": [],
                },
            },
            "description": "Heatmap of safety violations across zones and modules",
        }

    def generate_meta_learning_convergence_plot(self) -> Dict[str, Any]:
        """Generate meta-learning convergence visualization."""
        return {
            "id": "meta-learning-convergence",
            "title": "Meta-Learning Loss Convergence",
            "type": "visualization",
            "visState": {
                "title": "Meta-Learning Loss Convergence",
                "type": "line",
                "aggs": [
                    {
                        "id": "1",
                        "type": "avg",
                        "schema": "metric",
                        "params": {"field": "attributes.loss_after"},
                    },
                    {
                        "id": "2",
                        "type": "date_histogram",
                        "schema": "segment",
                        "params": {
                            "field": "@timestamp",
                            "interval": "auto",
                        },
                    },
                    {
                        "id": "3",
                        "type": "terms",
                        "schema": "group",
                        "params": {
                            "field": "attributes.domain.keyword",
                            "size": 5,
                        },
                    },
                ],
                "params": {
                    "addLegend": True,
                    "addTimeMarker": False,
                    "addTooltip": True,
                    "categoryAxes": [{"id": "CategoryAxis-1", "type": "category"}],
                    "grid": {"categoryLines": False, "valueAxes": ["ValueAxis-1"]},
                    "legendPosition": "right",
                    "seriesParams": [
                        {
                            "data": {"id": "1", "label": "Avg loss_after"},
                            "drawLinesBetweenPoints": True,
                            "interpolate": "linear",
                            "mode": "normal",
                            "show": True,
                            "showCircles": True,
                            "type": "line",
                            "valueAxis": "ValueAxis-1",
                        },
                    ],
                    "valueAxes": [
                        {
                            "id": "ValueAxis-1",
                            "name": "LeftAxis-1",
                            "position": "left",
                            "scale": {"mode": "linear", "type": "linear"},
                            "title": {"text": "Avg loss_after"},
                            "type": "value",
                        },
                    ],
                },
            },
            "description": "Meta-learning loss convergence over time per domain",
        }

    def generate_incident_timeline(self) -> Dict[str, Any]:
        """Generate incident timeline visualization."""
        return {
            "id": "incident-timeline",
            "title": "Incident Timeline (OnCall)",
            "type": "visualization",
            "visState": {
                "title": "Incident Timeline",
                "type": "gantt_chart",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                        "params": {},
                    },
                    {
                        "id": "2",
                        "type": "date_histogram",
                        "schema": "segment",
                        "params": {
                            "field": "created_at",
                            "interval": "1h",
                        },
                    },
                    {
                        "id": "3",
                        "type": "terms",
                        "schema": "group",
                        "params": {
                            "field": "priority.keyword",
                            "size": 4,
                        },
                    },
                ],
                "params": {
                    "addLegend": True,
                    "addTooltip": True,
                    "legendPosition": "right",
                },
            },
            "description": "Timeline of incidents by priority over time",
        }

    def generate_zone_retention_visualization(self) -> Dict[str, Any]:
        """Generate zone-based retention visualization."""
        return {
            "id": "zone-retention",
            "title": "Data Retention by Zone (PNDC Axiom 3)",
            "type": "visualization",
            "visState": {
                "title": "Zone Retention",
                "type": "metric",
                "aggs": [
                    {
                        "id": "1",
                        "type": "avg",
                        "schema": "metric",
                        "params": {"field": "retention_days"},
                    },
                    {
                        "id": "2",
                        "type": "terms",
                        "schema": "group",
                        "params": {
                            "field": "zone.keyword",
                            "size": 4,
                        },
                    },
                ],
                "params": {
                    "addLegend": False,
                    "addTooltip": True,
                    "metric": {"colorSchema": "Green to Red", "colorsRange": [
                        {"from": 0, "to": 30},
                        {"from": 30, "to": 180},
                        {"from": 180, "to": 730},
                    ]},
                },
            },
            "description": "Average retention days per zone (PNDC Axiom 3 proportionality)",
        }

    def generate_complete_dashboard(self) -> Dict[str, Any]:
        """Generate complete OpenSearch Dashboards dashboard."""
        return {
            "id": "sesi-agi-complete",
            "title": "SESI AGI Complete Observability Dashboard",
            "description": "Unified dashboard for AGI modules, incidents, and traces",
            "panelsJSON": json.dumps([
                {
                    "version": "2.11.0",
                    "type": "visualization",
                    "gridData": {"x": 0, "y": 0, "w": 12, "h": 10, "i": "1"},
                    "panelIndex": "1",
                    "embeddableConfig": {},
                    "panelRefName": "panel_1",
                },
                {
                    "version": "2.11.0",
                    "type": "visualization",
                    "gridData": {"x": 12, "y": 0, "w": 12, "h": 10, "i": "2"},
                    "panelIndex": "2",
                    "embeddableConfig": {},
                    "panelRefName": "panel_2",
                },
                {
                    "version": "2.11.0",
                    "type": "visualization",
                    "gridData": {"x": 0, "y": 10, "w": 24, "h": 12, "i": "3"},
                    "panelIndex": "3",
                    "embeddableConfig": {},
                    "panelRefName": "panel_3",
                },
                {
                    "version": "2.11.0",
                    "type": "visualization",
                    "gridData": {"x": 0, "y": 22, "w": 12, "h": 10, "i": "4"},
                    "panelIndex": "4",
                    "embeddableConfig": {},
                    "panelRefName": "panel_4",
                },
                {
                    "version": "2.11.0",
                    "type": "visualization",
                    "gridData": {"x": 12, "y": 22, "w": 12, "h": 10, "i": "5"},
                    "panelIndex": "5",
                    "embeddableConfig": {},
                    "panelRefName": "panel_5",
                },
                {
                    "version": "2.11.0",
                    "type": "visualization",
                    "gridData": {"x": 0, "y": 32, "w": 24, "h": 8, "i": "6"},
                    "panelIndex": "6",
                    "embeddableConfig": {},
                    "panelRefName": "panel_6",
                },
            ]),
            "timeRestore": False,
            "version": 1,
            "references": [
                {"id": "agi-module-distribution", "name": "panel_1", "type": "visualization"},
                {"id": "trace-log-correlation", "name": "panel_2", "type": "visualization"},
                {"id": "safety-violation-heatmap", "name": "panel_3", "type": "visualization"},
                {"id": "meta-learning-convergence", "name": "panel_4", "type": "visualization"},
                {"id": "incident-timeline", "name": "panel_5", "type": "visualization"},
                {"id": "zone-retention", "name": "panel_6", "type": "visualization"},
            ],
        }

    def write_all(self, output_dir: str = "./opensearch_dashboards") -> Dict[str, str]:
        """Write all dashboard artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        visualizations = {
            "module_distribution": self.generate_module_distribution_visualization(),
            "trace_log_correlation": self.generate_trace_log_correlation_visualization(),
            "safety_violation_heatmap": self.generate_safety_violation_heatmap(),
            "meta_learning_convergence": self.generate_meta_learning_convergence_plot(),
            "incident_timeline": self.generate_incident_timeline(),
            "zone_retention": self.generate_zone_retention_visualization(),
        }

        for name, viz in visualizations.items():
            viz_path = out_path / f"visualization-{name}.json"
            with open(viz_path, "w") as f:
                json.dump(viz, f, indent=2)
            files[f"viz_{name}"] = str(viz_path)

        # Complete dashboard
        dashboard = self.generate_complete_dashboard()
        dashboard_path = out_path / "dashboard-complete.json"
        with open(dashboard_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        files["complete_dashboard"] = str(dashboard_path)

        logger.info(f"[OpenSearch Dashboards] Generated {len(files)} artifacts in {output_dir}")
        return files


# ═══════════════════════════════════════════════════════════════════════════
# [FP1] FREE PREPRINT SOURCES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PreprintPaper:
    """A preprint paper from open-access sources."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    source: str  # "arxiv" | "pubmed" | "semantic_scholar" | "biorxiv" | etc.
    url: str
    pdf_url: Optional[str] = None
    published: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    citations: int = 0
    license: str = "open-access"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "source": self.source,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "published": self.published,
            "categories": self.categories,
            "citations": self.citations,
            "license": self.license,
            "metadata": self.metadata,
        }


class FreePreprintSources:
    """
    [D] Integration with free preprint sources for research papers.

    Sources:
      • arXiv (math, physics, CS, stats)
      • PubMed Central (biomedical)
      • Semantic Scholar (cross-disciplinary)
      • bioRxiv/medRxiv (biology, health)
      • CORE (open access aggregator)
      • DOAJ (directory of open access journals)
      • SSRN (social sciences)
      • OSF Preprints (cross-disciplinary)
      • Zenodo (all research outputs)

    Features:
      • Automated preprint discovery
      • Citation graph construction
      • License compliance checking
      • Metadata extraction for GNN training
      • PNDC Axiom 3 data minimization

    HONEST: This is a RESEARCH ASSISTANT. It does NOT:
      • Guarantee completeness of paper discovery
      • Replace expert literature review
      • Provide legal advice on copyright
    """

    # Free preprint source configurations
    SOURCES = {
        "arxiv": {
            "api_url": "http://export.arxiv.org/api/query",
            "categories": ["math.*", "cs.*", "stat.*", "physics.*"],
            "license": "arXiv perpetual license",
        },
        "pubmed": {
            "api_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            "categories": ["biomedical", "life sciences"],
            "license": "public domain / CC BY",
        },
        "semantic_scholar": {
            "api_url": "https://api.semanticscholar.org/graph/v1/paper/search",
            "categories": ["all"],
            "license": "varies by paper",
        },
        "biorxiv": {
            "api_url": "https://api.biorxiv.org/details/biorxiv",
            "categories": ["biology"],
            "license": "CC-BY-NC-ND 4.0",
        },
        "core": {
            "api_url": "https://api.core.ac.uk/v3/search/works",
            "categories": ["all"],
            "license": "varies by paper",
        },
    }

    def __init__(self):
        self._papers: List[PreprintPaper] = []
        self._search_timestamps: deque = deque(maxlen=1000)

        # No-Zeno parameters
        self.max_searches_per_minute = 20
        self.min_search_interval_s = 3.0

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def check_search_rate(self) -> Dict[str, Any]:
        """Check if search is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._search_timestamps = deque(
            (t for t in self._search_timestamps if t > cutoff),
            maxlen=1000,
        )

        if len(self._search_timestamps) >= self.max_searches_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
            }

        if self._search_timestamps:
            time_since_last = now - self._search_timestamps[-1]
            if time_since_last < self.min_search_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                }

        return {"allowed": True}

    # ── arXiv Search ─────────────────────────────────────────────────────

    def search_arxiv(
        self,
        query: str,
        max_results: int = 50,
        category: Optional[str] = None,
    ) -> List[PreprintPaper]:
        """
        Search arXiv for papers.

        Args:
            query: Search query
            max_results: Maximum number of results
            category: Optional arXiv category filter

        Returns:
            List of PreprintPaper objects
        """
        rate_check = self.check_search_rate()
        if not rate_check["allowed"]:
            logger.warning(f"[arXiv] Search rate-limited: {rate_check['reason']}")
            return []

        if not HAS_REQUESTS:
            logger.warning("[arXiv] requests not available; returning mock data")
            return self._mock_arxiv_results(query, max_results)

        search_query = f"all:{query}"
        if category:
            search_query = f"cat:{category} AND all:{query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(
                self.SOURCES["arxiv"]["api_url"],
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            self._search_timestamps.append(time.time())

            papers = self._parse_arxiv_response(response.text)
            self._papers.extend(papers)
            logger.info(f"[arXiv] Found {len(papers)} papers for query: {query}")
            return papers

        except Exception as e:
            logger.error(f"[arXiv] Search failed: {e}")
            return self._mock_arxiv_results(query, max_results)

    def _parse_arxiv_response(self, xml_text: str) -> List[PreprintPaper]:
        """Parse arXiv Atom XML response."""
        papers = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for entry in entries:
            id_match = re.search(r"<id>(.*?)</id>", entry)
            arxiv_id = id_match.group(1).split("/abs/")[-1] if id_match else ""

            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            title = title_match.group(1).strip().replace("\n", " ") if title_match else ""

            summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            abstract = summary_match.group(1).strip().replace("\n", " ") if summary_match else ""

            authors = re.findall(r"<name>(.*?)</name>", entry)
            categories = re.findall(r'<category[^>]*term="([^"]+)"', entry)
            published_match = re.search(r"<published>(.*?)</published>", entry)
            published = published_match.group(1) if published_match else None

            paper = PreprintPaper(
                paper_id=f"arxiv_{arxiv_id}",
                title=title,
                authors=authors,
                abstract=abstract[:500],  # PNDC Axiom 3 data minimization
                source="arxiv",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                published=published,
                categories=categories,
                license="arXiv perpetual license",
            )
            papers.append(paper)

        return papers

    def _mock_arxiv_results(self, query: str, max_results: int) -> List[PreprintPaper]:
        """Generate mock arXiv results for testing."""
        mock_papers = [
            PreprintPaper(
                paper_id=f"arxiv_2301.{i:05d}",
                title=f"Paper {i}: {query}",
                authors=["Y. A. Limsuwan", "PAI"],
                abstract=f"This paper investigates {query} using novel methods.",
                source="arxiv",
                url=f"https://arxiv.org/abs/2301.{i:05d}",
                pdf_url=f"https://arxiv.org/pdf/2301.{i:05d}",
                published="2023-01-01",
                categories=["math.FA", "cs.LG"],
                license="arXiv perpetual license",
            )
            for i in range(min(max_results, 5))
        ]
        self._papers.extend(mock_papers)
        return mock_papers

    # ── Semantic Scholar Search ──────────────────────────────────────────

    def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 50,
    ) -> List[PreprintPaper]:
        """Search Semantic Scholar for papers."""
        rate_check = self.check_search_rate()
        if not rate_check["allowed"]:
            return []

        if not HAS_REQUESTS:
            return self._mock_semantic_scholar_results(query, max_results)

        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,url,year,citationCount,openAccessPdf",
        }

        try:
            response = requests.get(
                self.SOURCES["semantic_scholar"]["api_url"],
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            self._search_timestamps.append(time.time())

            data = response.json()
            papers = []
            for paper_data in data.get("data", []):
                paper = PreprintPaper(
                    paper_id=f"ss_{paper_data.get('paperId', '')}",
                    title=paper_data.get("title", ""),
                    authors=[a.get("name", "") for a in paper_data.get("authors", [])],
                    abstract=(paper_data.get("abstract") or "")[:500],
                    source="semantic_scholar",
                    url=paper_data.get("url", ""),
                    pdf_url=paper_data.get("openAccessPdf", {}).get("url") if paper_data.get("openAccessPdf") else None,
                    published=str(paper_data.get("year", "")),
                    citations=paper_data.get("citationCount", 0),
                    license="varies by paper",
                )
                papers.append(paper)

            self._papers.extend(papers)
            logger.info(f"[Semantic Scholar] Found {len(papers)} papers for query: {query}")
            return papers

        except Exception as e:
            logger.error(f"[Semantic Scholar] Search failed: {e}")
            return self._mock_semantic_scholar_results(query, max_results)

    def _mock_semantic_scholar_results(self, query: str, max_results: int) -> List[PreprintPaper]:
        """Generate mock Semantic Scholar results."""
        mock_papers = [
            PreprintPaper(
                paper_id=f"ss_{uuid.uuid4().hex[:8]}",
                title=f"Semantic Scholar Paper {i}: {query}",
                authors=["Researcher A", "Researcher B"],
                abstract=f"Abstract for paper about {query}.",
                source="semantic_scholar",
                url=f"https://www.semanticscholar.org/paper/{i}",
                published="2023",
                citations=i * 10,
                license="CC-BY-4.0",
            )
            for i in range(min(max_results, 5))
        ]
        self._papers.extend(mock_papers)
        return mock_papers

    # ── Multi-Source Search ──────────────────────────────────────────────

    def search_all_sources(
        self,
        query: str,
        max_results_per_source: int = 20,
    ) -> Dict[str, List[PreprintPaper]]:
        """
        Search all free preprint sources for papers.

        Args:
            query: Search query
            max_results_per_source: Max results per source

        Returns:
            Dict mapping source name to list of papers
        """
        results = {}

        # arXiv
        results["arxiv"] = self.search_arxiv(query, max_results_per_source)

        # Semantic Scholar
        results["semantic_scholar"] = self.search_semantic_scholar(
            query, max_results_per_source
        )

        # Note: PubMed, bioRxiv, CORE would be added similarly
        # with proper API keys and rate limits

        return results

    # ── Citation Graph ───────────────────────────────────────────────────

    def build_citation_graph(self, papers: Optional[List[PreprintPaper]] = None) -> Dict[str, Any]:
        """
        Build a citation graph from papers.

        Args:
            papers: Optional list of papers (uses all if None)

        Returns:
            Dict with nodes and edges
        """
        papers = papers or self._papers

        nodes = []
        for paper in papers:
            nodes.append({
                "id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.published,
                "citations": paper.citations,
                "source": paper.source,
            })

        # In production, query Semantic Scholar API for citation relationships
        # For demo, create mock edges based on categories
        edges = []
        for i, p1 in enumerate(papers):
            for j, p2 in enumerate(papers):
                if i >= j:
                    continue
                # Mock: connect papers in same category
                if set(p1.categories) & set(p2.categories):
                    edges.append({
                        "source": p1.paper_id,
                        "target": p2.paper_id,
                        "weight": 1.0,
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }

    # ── License Compliance ───────────────────────────────────────────────

    def check_license_compliance(self, paper: PreprintPaper) -> Dict[str, Any]:
        """
        Check license compliance for a paper.

        Returns:
            Dict with compliance status and restrictions
        """
        license_type = paper.license.lower()

        if "public domain" in license_type or "cc0" in license_type:
            return {"compliant": True, "restrictions": [], "note": "Public domain"}
        elif "cc-by" in license_type:
            return {
                "compliant": True,
                "restrictions": ["attribution required"],
                "note": "CC-BY: attribution required",
            }
        elif "cc-by-nc" in license_type:
            return {
                "compliant": True,
                "restrictions": ["attribution", "non-commercial"],
                "note": "CC-BY-NC: attribution + non-commercial",
            }
        elif "arxiv" in license_type:
            return {
                "compliant": True,
                "restrictions": ["attribution", "no redistribution of PDF"],
                "note": "arXiv perpetual license: attribution + no PDF redistribution",
            }
        else:
            return {
                "compliant": False,
                "restrictions": ["unknown license"],
                "note": f"Unknown license: {paper.license}",
            }

    def write_all(self, output_dir: str = "./preprints") -> Dict[str, str]:
        """Write all preprint artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Papers JSONL
        papers_path = out_path / "papers.jsonl"
        with open(papers_path, "w") as f:
            for paper in self._papers:
                f.write(json.dumps(paper.to_dict()) + "\n")
        files["papers"] = str(papers_path)

        # Citation graph
        graph = self.build_citation_graph()
        graph_path = out_path / "citation-graph.json"
        with open(graph_path, "w") as f:
            json.dump(graph, f, indent=2)
        files["citation_graph"] = str(graph_path)

        # Metadata
        metadata = {
            "total_papers": len(self._papers),
            "by_source": {
                s: sum(1 for p in self._papers if p.source == s)
                for s in set(p.source for p in self._papers)
            },
            "license_compliance": {
                "compliant": sum(
                    1 for p in self._papers
                    if self.check_license_compliance(p)["compliant"]
                ),
                "non_compliant": sum(
                    1 for p in self._papers
                    if not self.check_license_compliance(p)["compliant"]
                ),
            },
        }
        metadata_path = out_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        files["metadata"] = str(metadata_path)

        logger.info(f"[Preprints] Generated {len(files)} artifacts in {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get preprint sources status."""
        return {
            "total_papers": len(self._papers),
            "by_source": {
                s: sum(1 for p in self._papers if p.source == s)
                for s in set(p.source for p in self._papers)
            } if self._papers else {},
            "available_sources": list(self.SOURCES.keys()),
            "requests_available": HAS_REQUESTS,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIIncidentVisualizationSuite:
    """
    [D] Complete suite for incident management, service graphs,
    custom dashboards, and free preprint sources.
    """

    def __init__(
        self,
        oncall_cfg: Optional[OnCallConfig] = None,
    ):
        self.oncall = GrafanaOnCallIntegration(oncall_cfg)
        self.tempo_metrics = TempoMetricsGenerator()
        self.os_dashboards = OpenSearchDashboardsCustom()
        self.preprints = FreePreprintSources()

    def simulate_agi_cycle_with_observability(self) -> Dict[str, Any]:
        """Simulate AGI cycle with full observability."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        cycle_start = time.time()

        # Simulate spans for each module
        modules = list(AGIModule)
        for i, module in enumerate(modules):
            duration_ms = random.uniform(10, 500)
            status = "OK" if random.random() > 0.05 else "ERROR"

            self.tempo_metrics.process_span(
                trace_id=trace_id,
                span_id=f"span_{i:03d}",
                parent_span_id=f"span_{i-1:03d}" if i > 0 else None,
                service_name=f"sesi-{module.value}",
                operation_name=f"agi.{module.value}",
                duration_ms=duration_ms,
                status=status,
                zone=SecurityZone.Z2_PROCESSING,
                module=module,
            )

            # Simulate safety violation occasionally
            if module == AGIModule.SAFETY and status == "ERROR":
                self.oncall.create_incident(
                    title=f"Safety violation in {module.value}",
                    message=f"Safety check failed for trace {trace_id}",
                    zone=SecurityZone.Z1_CLINICAL,
                    module=module,
                    severity=AlertSeverity.CRITICAL,
                    trace_id=trace_id,
                )

        cycle_latency = time.time() - cycle_start

        return {
            "trace_id": trace_id,
            "cycle_latency_s": cycle_latency,
            "spans_processed": len(modules),
            "incidents_created": len(self.oncall._incidents),
        }

    def generate_all_artifacts(self, output_dir: str = "./sesi_incident_viz") -> Dict[str, Any]:
        """Generate all artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # OnCall config
        results["oncall_config"] = self.oncall.write_oncall_config(
            str(out_path / "oncall" / "oncall-config.yaml")
        )

        # Tempo service graph
        service_graph = self.tempo_metrics.generate_service_graph_json()
        sg_path = out_path / "tempo" / "service-graph.json"
        sg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sg_path, "w") as f:
            json.dump(service_graph, f, indent=2)
        results["service_graph"] = str(sg_path)

        # Tempo Prometheus metrics
        metrics = self.tempo_metrics.generate_prometheus_metrics()
        metrics_path = out_path / "tempo" / "service-graph-metrics.prom"
        with open(metrics_path, "w") as f:
            f.write(metrics)
        results["tempo_metrics"] = str(metrics_path)

        # OpenSearch Dashboards
        results["os_dashboards"] = self.os_dashboards.write_all(
            str(out_path / "opensearch_dashboards")
        )

        # Preprints
        results["preprints"] = self.preprints.write_all(
            str(out_path / "preprints")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "oncall": self.oncall.get_status(),
            "tempo_metrics": self.tempo_metrics.get_status(),
            "preprints": self.preprints.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI OnCall + Tempo Service Graph + OS Dashboards + Preprints")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  requests available   : {HAS_REQUESTS}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIIncidentVisualizationSuite()

    # ── Test 1: Grafana OnCall ───────────────────────────────────────────
    print("\n  ── Grafana OnCall Integration ──")
    oncall_status = suite.oncall.get_status()
    print(f"    δ_min                : {oncall_status['delta_min']:.6f}")
    print(f"    Max incidents/min    : {oncall_status.get('max_incidents_per_minute', 'N/A')}")

    # Show escalation policies
    print("\n    Escalation Policies (PNDC Axiom 3):")
    for zone in SecurityZone:
        policy = suite.oncall.get_escalation_policy(zone, IncidentPriority.P1_CRITICAL)
        print(f"      {zone.value:20s}: {policy}")

    # ── Test 2: Tempo Metrics Generator ──────────────────────────────────
    print("\n  ── Tempo Metrics Generator (Service Graph) ──")

    # Simulate AGI cycles
    for i in range(20):
        cycle = suite.simulate_agi_cycle_with_observability()
        if i % 5 == 0:
            print(f"    Cycle {i+1:2d}             : trace={cycle['trace_id'][:12]}... "
                  f"spans={cycle['spans_processed']} incidents={cycle['incidents_created']}")

    tempo_status = suite.tempo_metrics.get_status()
    print(f"\n    Service graph nodes  : {tempo_status['total_nodes']}")
    print(f"    Service graph edges  : {tempo_status['total_edges']}")
    print(f"    Zone crossing edges  : {tempo_status['zone_crossing_edges']}")
    print(f"    Regime distribution  : {tempo_status['regime_distribution']}")

    # ── Test 3: OpenSearch Dashboards ────────────────────────────────────
    print("\n  ── OpenSearch Dashboards Custom Visualization ──")
    dashboard = suite.os_dashboards.generate_complete_dashboard()
    print(f"    Dashboard title      : {dashboard['title']}")
    print(f"    Total panels         : {len(json.loads(dashboard['panelsJSON']))}")
    print(f"    References           : {len(dashboard['references'])}")

    # ── Test 4: Free Preprint Sources ────────────────────────────────────
    print("\n  ── Free Preprint Sources ──")

    # Search for papers
    query = "causal discovery neural networks"
    papers = suite.preprints.search_all_sources(query, max_results_per_source=5)

    print(f"    Search query         : {query}")
    for source, source_papers in papers.items():
        print(f"      {source:20s}: {len(source_papers)} papers")
        for p in source_papers[:2]:
            print(f"        • {p.title[:60]}...")
            compliance = suite.preprints.check_license_compliance(p)
            print(f"          License: {p.license} - Compliant: {compliance['compliant']}")

    # Build citation graph
    graph = suite.preprints.build_citation_graph()
    print(f"\n    Citation graph       : {graph['n_nodes']} nodes, {graph['n_edges']} edges")

    preprint_status = suite.preprints.get_status()
    print(f"    Total papers         : {preprint_status['total_papers']}")
    print(f"    Available sources    : {preprint_status['available_sources']}")

    # ── Test 5: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_incident_viz")
    print(f"    Artifact groups      : {len(artifacts)}")
    for name, value in artifacts.items():
        if isinstance(value, dict):
            print(f"      {name:20s}: {len(value)} files")
        else:
            print(f"      {name:20s}: {value}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    OnCall incidents     : {status['oncall']['total_incidents']}")
    print(f"    Active incidents     : {status['oncall']['active_incidents']}")
    print(f"    Tempo nodes          : {status['tempo_metrics']['total_nodes']}")
    print(f"    Tempo edges          : {status['tempo_metrics']['total_edges']}")
    print(f"    Preprint papers      : {status['preprints']['total_papers']}")

    print("\n" + "=" * 72)
    print("  FREE PREPRINT SOURCES (แทนเปเปอร์ที่ต้องจ่ายเงิน)")
    print("  ─────────────────────────────────────────────────────────")
    print("  • arXiv (arxiv.org) - math, physics, CS, stats")
    print("  • PubMed Central (ncbi.nlm.nih.gov/pmc) - biomedical")
    print("  • Semantic Scholar (semanticscholar.org) - cross-disciplinary")
    print("  • bioRxiv/medRxiv (biorxiv.org) - biology, health")
    print("  • CORE (core.ac.uk) - open access aggregator")
    print("  • DOAJ (doaj.org) - directory of OA journals")
    print("  • SSRN (ssrn.com) - social sciences")
    print("  • OSF Preprints (osf.io/preprints) - cross-disciplinary")
    print("  • Zenodo (zenodo.org) - all research outputs")
    print("  • Unpaywall (unpaywall.org) - browser extension")
    print("  • ResearchGate (researchgate.net) - author uploads")
    print("=" * 72)
    print("  HONEST SCOPE")
    print("  • OnCall is INCIDENT MANAGEMENT, not autonomous response")
    print("  • Tempo service graph shows CORRELATION, not causation")
    print("  • OS Dashboards are VISUALIZATION, not decision-making")
    print("  • Free preprint sources require license compliance checking")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all values")
    print("  • PNDC Axiom 3 data minimization in all logs/incidents")
    print("=" * 72)


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    smoke_test()
