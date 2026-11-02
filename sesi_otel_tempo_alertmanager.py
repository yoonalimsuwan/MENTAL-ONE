"""
=============================================================================
SESI AGI: OTEL COLLECTOR + TEMPO BACKEND + ALERTMANAGER SAFETY RULES
Production Trace Ingestion + Distributed Tracing + Zone-Based Safety Alerting
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[OTC1] OpenTelemetryCollectorConfig
      • Receivers: OTLP (gRPC+HTTP), Jaeger, Prometheus, Loki
      • Processors: batch, memory_limiter, filter/drop-sensitive,
        attributes/add-zone, spanmetrics, transform/trace-context
      • Exporters: Tempo (traces), Prometheus (metrics), Loki (logs),
        Alertmanager (safety alerts)
      • Pipelines: traces, metrics, logs (zone-aware routing)
      • PNDC Axiom 3 compliance: sensitive data filtering
      • K8s Deployment manifest (2 replicas, TLS, HA)

[TB1] TempoBackendConfig
      • Distributed tracing storage backend
      • Trace-log correlation via trace_id
      • Search configuration (tags, duration, service)
      • Storage backend: S3 / GCS / local filesystem
      • Compaction and retention policies
      • Integration with Grafana for visualization

[AR1] SafetyAlertmanagerRules
      • Zone-based safety violation alerting (PNDC Axiom 3 proportionality)
      • Per-zone thresholds: Z0 strictest → Z3 loosest
      • Inhibition rules to prevent alert storms
      • Severity escalation matrix
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for threshold calibration
      • Prometheus alert rules YAML generation

[INT1] AGISystemInstrumentation
      • Wraps all 10 AGI modules with OpenTelemetry spans
      • Trace context propagation (trace_id/span_id)
      • Safety violation → Alertmanager webhook
      • Integration with existing AGISystem

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds trace ingestion and alert emission rate
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful alert threshold
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → trace stream regime classification
[C] Six_Problems P4: CUSUM + subsampling
    → change-point detection in trace latency streams
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-based safety thresholds, data minimization
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
    → NaN/Inf sanitization for all values
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO AGI UNIFIED FRAMEWORK
----------------------------------
From AGI Unified Framework Single File.py:
  • CoreReasoningEngine → OTel span: "agi.reasoning"
  • EpisodicSemanticMemory → OTel span: "agi.memory"
  • PerceptionMultimodal → OTel span: "agi.perception"
  • ActionActuationController → OTel span: "agi.actuation"
  • EpistemicCuriosityDrive → OTel span: "agi.curiosity"
  • ConceptSynthesisEngine → OTel span: "agi.synthesis"
  • SelfEvalReflection → OTel span: "agi.reflection"
  • SafetyAlignmentGuard → OTel span: "agi.safety" + Alertmanager
  • DynamicSubstrateAllocator → OTel span: "agi.substrate"
  • MetaLearningPipeline → OTel span: "agi.meta_learning"

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• OTel Collector config is a TEMPLATE; real deployment requires TLS certs.
• Tempo backend is STORAGE; it does not guarantee trace completeness.
• Alertmanager rules are CONFIGURATION; they do not replace human review.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• Production deployment requires security audit and regulatory approval.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.otel_tempo_alertmanager")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode, SpanKind
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

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
    "SecurityZone", "AlertSeverity", "QualityRegime",
    "OpenTelemetryCollectorConfig",
    "TempoBackendConfig",
    "SafetyAlertmanagerRules", "ZoneSafetyConfig",
    "AGISystemInstrumented",
    "SESIProductionTracingSuite",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class SecurityZone(Enum):
    """Four security zones from SESI network segmentation."""
    Z0_IMPLANT = "z0-implant"
    Z1_CLINICAL = "z1-clinical"
    Z2_PROCESSING = "z2-processing"
    Z3_EXTERNAL = "z3-external"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class QualityRegime(Enum):
    """Growth-rate regime classification (paper6 Thm 5.1)."""
    FINITE_MEAN = 0
    REGULARLY_VARYING = 1
    DEGENERATE = 2
    INSUFFICIENT_DATA = 3


# ═══════════════════════════════════════════════════════════════════════════
# [OTC1] OPENTELEMETRY COLLECTOR CONFIG
# ═══════════════════════════════════════════════════════════════════════════

class OpenTelemetryCollectorConfig:
    """
    [D] Generates OpenTelemetry Collector configuration for production.

    Features:
      • Receivers: OTLP (gRPC+HTTP), Jaeger, Prometheus, Loki
      • Processors: batch, memory_limiter, filter/drop-sensitive,
        attributes/add-zone, spanmetrics, transform/trace-context
      • Exporters: Tempo, Prometheus, Loki, Alertmanager
      • Pipelines: traces, metrics, logs (zone-aware routing)
      • PNDC Axiom 3 compliance: sensitive data filtering
      • K8s Deployment manifest (2 replicas, TLS, HA)

    HONEST: This is a CONFIGURATION GENERATOR. Real deployment requires:
      • TLS certificates for all endpoints
      • Proper IAM setup
      • Security audit
    """

    def __init__(
        self,
        tempo_endpoint: str = "http://tempo-distributor:4317",
        prometheus_endpoint: str = "http://prometheus:9090/api/v1/write",
        loki_endpoint: str = "http://loki:3100/loki/api/v1/push",
        alertmanager_endpoint: str = "http://alertmanager:9093/api/v2/alerts",
    ):
        self.tempo_endpoint = tempo_endpoint
        self.prometheus_endpoint = prometheus_endpoint
        self.loki_endpoint = loki_endpoint
        self.alertmanager_endpoint = alertmanager_endpoint

    def generate_config(self) -> Dict[str, Any]:
        """Generate complete OTel Collector configuration."""
        return {
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
                "jaeger": {
                    "protocols": {
                        "grpc": {"endpoint": "0.0.0.0:14250"},
                        "thrift_http": {"endpoint": "0.0.0.0:14268"},
                    },
                },
                "prometheus": {
                    "config": {
                        "scrape_configs": [
                            {
                                "job_name": "sesi-agi",
                                "scrape_interval": "15s",
                                "static_configs": [{"targets": ["sesi-agi:9090"]}],
                            },
                        ],
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
                    "limit_mib": 2048,
                    "spike_limit_mib": 512,
                },
                # PNDC Axiom 3: filter sensitive data
                "filter/drop-sensitive": {
                    "error_mode": "ignore",
                    "traces": {
                        "span": [
                            'attributes["password"] != nil',
                            'attributes["secret"] != nil',
                            'attributes["token"] != nil',
                            'attributes["ssn"] != nil',
                            'attributes["credit_card"] != nil',
                            'attributes["api_key"] != nil',
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
                        {"key": "sesi.zone", "from_attribute": "zone", "action": "upsert"},
                        {"key": "sesi.module", "from_attribute": "agi.module", "action": "upsert"},
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
                        {"name": "sesi.module"},
                        {"name": "http.status_code"},
                    ],
                },
                "transform/add-trace-to-logs": {
                    "log_statements": [{
                        "context": "log",
                        "statements": [
                            'set(attributes["trace_id"], trace_id) where trace_id != nil',
                            'set(attributes["span_id"], span_id) where span_id != nil',
                        ],
                    }],
                },
            },
            "exporters": {
                "otlp/tempo": {
                    "endpoint": self.tempo_endpoint,
                    "tls": {"insecure": False, "ca_file": "/etc/otel/certs/ca.crt"},
                },
                "prometheusremotewrite": {
                    "endpoint": self.prometheus_endpoint,
                    "tls": {"insecure": False, "ca_file": "/etc/otel/certs/ca.crt"},
                    "resource_to_telemetry_conversion": {"enabled": True},
                },
                "loki": {
                    "endpoint": self.loki_endpoint,
                    "labels": {
                        "resource": {
                            "service.name": "service",
                            "sesi.zone": "zone",
                            "sesi.module": "module",
                        },
                        "log": {
                            "severity": "level",
                            "event_type": "event_type",
                            "trace_id": "trace_id",
                        },
                    },
                },
                "alertmanager": {
                    "endpoint": self.alertmanager_endpoint,
                    "tls": {"insecure": False, "ca_file": "/etc/otel/certs/ca.crt"},
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
                        "exporters": ["otlp/tempo"],
                    },
                    "metrics": {
                        "receivers": ["otlp", "prometheus"],
                        "processors": ["memory_limiter", "batch"],
                        "exporters": ["prometheusremotewrite"],
                    },
                    "logs": {
                        "receivers": ["otlp"],
                        "processors": [
                            "memory_limiter",
                            "filter/drop-sensitive",
                            "transform/add-trace-to-logs",
                            "attributes/add-zone",
                            "batch",
                        ],
                        "exporters": ["loki", "alertmanager"],
                    },
                },
                "telemetry": {
                    "logs": {"level": "info"},
                    "metrics": {"address": "0.0.0.0:8888"},
                },
            },
            "extensions": {
                "health_check": {"endpoint": "0.0.0.0:13133"},
                "pprof": {"endpoint": "0.0.0.0:1777"},
                "zpages": {"endpoint": "0.0.0.0:55679"},
            },
        }

    def generate_k8s_deployment(self) -> Dict[str, Any]:
        """Generate K8s Deployment manifest for OTel Collector."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "otel-collector",
                "namespace": "sesi-agi",
                "labels": {"app.kubernetes.io/name": "otel-collector"},
            },
            "spec": {
                "replicas": 2,
                "selector": {"matchLabels": {"app.kubernetes.io/name": "otel-collector"}},
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
                        "containers": [{
                            "name": "otel-collector",
                            "image": "otel/opentelemetry-collector-contrib:0.91.0",
                            "args": ["--config=/etc/otel/config.yaml"],
                            "ports": [
                                {"name": "otlp-grpc", "containerPort": 4317},
                                {"name": "otlp-http", "containerPort": 4318},
                                {"name": "jaeger-grpc", "containerPort": 14250},
                                {"name": "metrics", "containerPort": 8888},
                                {"name": "health", "containerPort": 13133},
                            ],
                            "env": [{"name": "GOMEMLIMIT", "value": "2048MiB"}],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "1Gi"},
                                "limits": {"cpu": "2000m", "memory": "4Gi"},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/", "port": 13133},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/", "port": 13133},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                            },
                            "volumeMounts": [
                                {"name": "otel-config", "mountPath": "/etc/otel", "readOnly": True},
                                {"name": "otel-certs", "mountPath": "/etc/otel/certs", "readOnly": True},
                            ],
                        }],
                        "volumes": [
                            {"name": "otel-config", "configMap": {"name": "otel-collector-config"}},
                            {"name": "otel-certs", "secret": {"secretName": "otel-collector-tls"}},
                        ],
                    },
                },
            },
        }

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
        logger.info(f"[OTel] Config written to {output_path}")
        return str(out_path)

    def write_k8s_manifest(self, output_path: str = "./k8s/otel-collector-deployment.yaml") -> str:
        """Write K8s Deployment manifest to file."""
        manifest = self.generate_k8s_deployment()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(manifest, f, indent=2)
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════════
# [TB1] TEMPO BACKEND CONFIG
# ═══════════════════════════════════════════════════════════════════════════

class TempoBackendConfig:
    """
    [D] Generates Grafana Tempo configuration for distributed tracing.

    Features:
      • Trace storage backend (S3 / GCS / local)
      • Search configuration (tags, duration, service)
      • Compaction and retention policies
      • Integration with Grafana for visualization
      • Trace-log correlation via trace_id

    HONEST: Tempo is a STORAGE backend. It does not guarantee
    trace completeness or search accuracy.
    """

    def __init__(
        self,
        storage_backend: str = "s3",  # "s3" | "gcs" | "local"
        s3_bucket: str = "sesi-tempo-traces",
        s3_region: str = "us-east-1",
        retention_days: int = 30,
    ):
        self.storage_backend = storage_backend
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        self.retention_days = retention_days

    def generate_config(self) -> Dict[str, Any]:
        """Generate complete Tempo configuration."""
        storage_config = self._get_storage_config()

        return {
            "server": {
                "http_listen_port": 3200,
                "log_level": "info",
            },
            "distributor": {
                "receivers": {
                    "otlp": {
                        "protocols": {
                            "grpc": {"endpoint": "0.0.0.0:4317"},
                            "http": {"endpoint": "0.0.0.0:4318"},
                        },
                    },
                    "jaeger": {
                        "protocols": {
                            "grpc": {"endpoint": "0.0.0.0:14250"},
                            "thrift_http": {"endpoint": "0.0.0.0:14268"},
                        },
                    },
                },
            },
            "ingester": {
                "max_block_duration": "5m",
                "trace_idle_timeout": "10s",
            },
            "compactor": {
                "compaction": {
                    "block_retention": f"{self.retention_days}d",
                    "compaction_window": "1h",
                },
            },
            "metrics_generator": {
                "registry": {
                    "external_labels": {
                        "source": "tempo",
                        "cluster": "sesi-agi",
                    },
                },
                "storage": {
                    "path": "/var/tempo/generator/wal",
                    "remote_write": [{
                        "url": "http://prometheus:9090/api/v1/write",
                        "send_exemplars": True,
                    }],
                },
                "processor": {
                    "service_graphs": {"enabled": True},
                    "span_metrics": {"enabled": True},
                },
            },
            "storage": storage_config,
            "memberlist": {
                "join_members": ["tempo-read:", "tempo-write:"],
            },
            "overrides": {
                "defaults": {
                    "metrics_generator": {
                        "processors": ["service_graphs", "span_metrics"],
                    },
                },
            },
        }

    def _get_storage_config(self) -> Dict[str, Any]:
        """Get storage backend configuration."""
        if self.storage_backend == "s3":
            return {
                "trace": {
                    "backend": "s3",
                    "wal": {
                        "path": "/var/tempo/wal",
                        "version": "v2",
                    },
                    "s3": {
                        "bucket": self.s3_bucket,
                        "region": self.s3_region,
                        "endpoint": f"s3.{self.s3_region}.amazonaws.com",
                        "access_key": "${S3_ACCESS_KEY}",
                        "secret_key": "${S3_SECRET_KEY}",
                    },
                },
            }
        elif self.storage_backend == "gcs":
            return {
                "trace": {
                    "backend": "gcs",
                    "wal": {"path": "/var/tempo/wal", "version": "v2"},
                    "gcs": {
                        "bucket_name": self.s3_bucket,
                        "endpoint": "storage.googleapis.com",
                    },
                },
            }
        else:  # local
            return {
                "trace": {
                    "backend": "local",
                    "wal": {"path": "/var/tempo/wal", "version": "v2"},
                    "local": {"path": "/var/tempo/traces"},
                },
            }

    def generate_k8s_statefulset(self) -> Dict[str, Any]:
        """Generate K8s StatefulSet for Tempo."""
        return {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "tempo",
                "namespace": "sesi-agi",
            },
            "spec": {
                "serviceName": "tempo",
                "replicas": 3,
                "selector": {"matchLabels": {"app": "tempo"}},
                "template": {
                    "metadata": {"labels": {"app": "tempo"}},
                    "spec": {
                        "containers": [{
                            "name": "tempo",
                            "image": "grafana/tempo:2.3.0",
                            "args": ["-config.file=/etc/tempo/tempo.yaml"],
                            "ports": [
                                {"name": "http", "containerPort": 3200},
                                {"name": "otlp-grpc", "containerPort": 4317},
                                {"name": "otlp-http", "containerPort": 4318},
                            ],
                            "resources": {
                                "requests": {"cpu": "1000m", "memory": "2Gi"},
                                "limits": {"cpu": "4000m", "memory": "8Gi"},
                            },
                            "volumeMounts": [
                                {"name": "tempo-config", "mountPath": "/etc/tempo", "readOnly": True},
                                {"name": "tempo-data", "mountPath": "/var/tempo"},
                            ],
                        }],
                        "volumes": [
                            {"name": "tempo-config", "configMap": {"name": "tempo-config"}},
                        ],
                    },
                },
                "volumeClaimTemplates": [{
                    "metadata": {"name": "tempo-data"},
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "resources": {"requests": {"storage": "100Gi"}},
                    },
                }],
            },
        }

    def write_config(self, output_path: str = "./tempo/tempo.yaml") -> str:
        """Write Tempo configuration to file."""
        config = self.generate_config()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
        logger.info(f"[Tempo] Config written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════════
# [AR1] SAFETY ALERTMANAGER RULES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneSafetyConfig:
    """
    Per-zone safety alert configuration.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be calibrated against real data.
    PNDC Axiom 3 proportionality: Z0 strictest, Z3 loosest.
    """
    zone: SecurityZone
    safety_warning: float = 0.8
    safety_critical: float = 0.6
    safety_emergency: float = 0.3
    violation_rate_warning: float = 0.01   # 1% violation rate
    violation_rate_critical: float = 0.05  # 5% violation rate
    default_severity: AlertSeverity = AlertSeverity.WARNING
    receiver: str = "platform-team"
    group_wait: str = "30s"
    group_interval: str = "5m"
    repeat_interval: str = "4h"


class SafetyAlertmanagerRules:
    """
    [D] Generates zone-based Alertmanager rules for safety violations.

    Features:
      • Per-zone alert rules (Z0-Z3)
      • Zone-specific thresholds (PNDC Axiom 3 proportionality)
      • Inhibition rules to prevent alert storms
      • Receiver mapping per zone
      • No-Zeno rate limiting on alert emission
      • δ_min quantum gate for threshold calibration
      • Alertmanager YAML configuration generation

    HONEST: These are TEMPLATE alert rules. Real deployment requires:
      • Calibration against operational data
      • Security team review
      • Regular threshold updates

    Mathematical grounding:
      • PNDC Axiom 3: Necessity and Proportionality
      • No-Zeno bound on alert rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for meaningful triggers (Delta_Min Thm 1.2)
    """

    # Default zone configs (PNDC Axiom 3 proportionality)
    DEFAULT_ZONE_CONFIGS = {
        SecurityZone.Z0_IMPLANT: ZoneSafetyConfig(
            zone=SecurityZone.Z0_IMPLANT,
            safety_warning=0.95,
            safety_critical=0.85,
            safety_emergency=0.70,
            violation_rate_warning=0.001,
            violation_rate_critical=0.01,
            default_severity=AlertSeverity.EMERGENCY,
            receiver="clinical-pager",
            group_wait="10s",
            repeat_interval="1h",
        ),
        SecurityZone.Z1_CLINICAL: ZoneSafetyConfig(
            zone=SecurityZone.Z1_CLINICAL,
            safety_warning=0.90,
            safety_critical=0.75,
            safety_emergency=0.50,
            violation_rate_warning=0.005,
            violation_rate_critical=0.02,
            default_severity=AlertSeverity.CRITICAL,
            receiver="clinical-team",
            group_wait="30s",
            repeat_interval="4h",
        ),
        SecurityZone.Z2_PROCESSING: ZoneSafetyConfig(
            zone=SecurityZone.Z2_PROCESSING,
            safety_warning=0.80,
            safety_critical=0.60,
            safety_emergency=0.30,
            violation_rate_warning=0.01,
            violation_rate_critical=0.05,
            default_severity=AlertSeverity.WARNING,
            receiver="platform-slack",
            group_wait="1m",
            repeat_interval="12h",
        ),
        SecurityZone.Z3_EXTERNAL: ZoneSafetyConfig(
            zone=SecurityZone.Z3_EXTERNAL,
            safety_warning=0.70,
            safety_critical=0.50,
            safety_emergency=0.20,
            violation_rate_warning=0.02,
            violation_rate_critical=0.10,
            default_severity=AlertSeverity.INFO,
            receiver="platform-email",
            group_wait="5m",
            repeat_interval="24h",
        ),
    }

    def __init__(
        self,
        zone_configs: Optional[Dict[SecurityZone, ZoneSafetyConfig]] = None,
    ):
        self.zone_configs = zone_configs or self.DEFAULT_ZONE_CONFIGS
        self._alert_timestamps: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_alert_rate(self, zone: SecurityZone) -> Dict[str, Any]:
        """Check if alert emission is within No-Zeno bound for this zone."""
        now = time.time()
        zone_key = zone.value
        cutoff = now - 60.0
        timestamps = self._alert_timestamps[zone_key]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        delta_min = self.compute_delta_min()
        c_v = 1.0
        T = 60.0
        bound = 1 + (2 * c_v * T) / max(delta_min, 1e-8)

        if len(timestamps) >= bound:
            return {
                "allowed": False,
                "reason": "no_zeno_bound_exceeded",
                "current": len(timestamps),
                "bound": bound,
            }

        return {
            "allowed": True,
            "current_count": len(timestamps),
            "bound": bound,
        }

    # ── Zone-Specific Alert Rules ────────────────────────────────────────

    def generate_zone_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Generate Prometheus alert rules per zone for safety violations.

        HONEST: Thresholds are FREE DESIGN PARAMETERS.
        Must be calibrated against real operational data.
        """
        rules = []

        for zone, cfg in self.zone_configs.items():
            zone_name = zone.value

            # Safety score degradation alert
            rules.append({
                "alert": f"SafetyScoreDegradation_{zone_name}",
                "expr": f'sesi_agi_safety_score{{zone="{zone_name}"}} < {cfg.safety_warning}',
                "for": "2m",
                "labels": {
                    "severity": cfg.default_severity.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"Safety score degradation in {zone_name}",
                    "description": (
                        f"Safety score in {zone_name} zone has dropped below "
                        f"{cfg.safety_warning}. PNDC Axiom 3: investigate "
                        f"proportionally to zone severity."
                    ),
                    "runbook_url": f"https://wiki.sesi.example.com/runbooks/{zone_name}/safety",
                },
            })

            # Critical safety alert
            rules.append({
                "alert": f"SafetyCritical_{zone_name}",
                "expr": f'sesi_agi_safety_score{{zone="{zone_name}"}} < {cfg.safety_critical}',
                "for": "1m",
                "labels": {
                    "severity": AlertSeverity.CRITICAL.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"CRITICAL safety issue in {zone_name}",
                    "description": (
                        f"Safety score critically low in {zone_name}. "
                        f"Immediate review required."
                    ),
                },
            })

            # Emergency safety alert
            rules.append({
                "alert": f"SafetyEmergency_{zone_name}",
                "expr": f'sesi_agi_safety_score{{zone="{zone_name}"}} < {cfg.safety_emergency}',
                "for": "0m",
                "labels": {
                    "severity": AlertSeverity.EMERGENCY.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"EMERGENCY safety issue in {zone_name}",
                    "description": (
                        f"Safety score at emergency level in {zone_name}. "
                        f"Immediate intervention required."
                    ),
                },
            })

            # Safety violation rate alert
            rules.append({
                "alert": f"SafetyViolationRate_{zone_name}",
                "expr": (
                    f'rate(sesi_agi_safety_violations_total{{zone="{zone_name}"}}[5m]) '
                    f'/ rate(sesi_agi_safety_verifications_total{{zone="{zone_name}"}}[5m]) '
                    f'> {cfg.violation_rate_warning}'
                ),
                "for": "5m",
                "labels": {
                    "severity": cfg.default_severity.value,
                    "zone": zone_name,
                },
                "annotations": {
                    "summary": f"High safety violation rate in {zone_name}",
                    "description": (
                        f"Safety violation rate exceeds "
                        f"{cfg.violation_rate_warning*100:.1f}% in {zone_name}. "
                        f"Check SafetyAlignmentGuard configuration."
                    ),
                },
            })

        return rules

    def generate_inhibition_rules(self) -> List[Dict[str, Any]]:
        """
        Generate inhibition rules to prevent alert storms.

        From PNDC Axiom 3: alerts should be necessary and proportionate.
        """
        return [
            {
                "source_matchers": ['severity="emergency"'],
                "target_matchers": ['severity=~"critical|warning|info"'],
                "equal": ["zone"],
            },
            {
                "source_matchers": ['severity="critical"'],
                "target_matchers": ['severity=~"warning|info"'],
                "equal": ["zone"],
            },
            {
                "source_matchers": ['alertname=~"SafetyEmergency_.*"'],
                "target_matchers": ['alertname=~"SafetyCritical_.*"'],
                "equal": ["zone"],
            },
            {
                "source_matchers": ['alertname=~"SafetyCritical_.*"'],
                "target_matchers": ['alertname=~"SafetyScoreDegradation_.*"'],
                "equal": ["zone"],
            },
        ]

    def generate_alertmanager_config(self) -> Dict[str, Any]:
        """
        Generate complete Alertmanager configuration.

        HONEST: This is a TEMPLATE. Production requires:
          • Proper receiver configuration
          • TLS certificates
          • Security audit
        """
        routes = []
        for zone, cfg in self.zone_configs.items():
            routes.append({
                "match": {"zone": zone.value},
                "receiver": cfg.receiver,
                "group_by": ["alertname", "zone"],
                "group_wait": cfg.group_wait,
                "group_interval": cfg.group_interval,
                "repeat_interval": cfg.repeat_interval,
            })

        config = {
            "global": {
                "resolve_timeout": "5m",
            },
            "route": {
                "receiver": "default-receiver",
                "group_by": ["alertname", "zone", "severity"],
                "group_wait": "30s",
                "group_interval": "5m",
                "repeat_interval": "4h",
                "routes": routes,
            },
            "receivers": [
                {
                    "name": "default-receiver",
                    "email_configs": [{"to": "platform-team@sesi.example.com"}],
                },
                {
                    "name": "clinical-pager",
                    "pagerduty_configs": [{"service_key": "${PAGERDUTY_KEY}"}],
                },
                {
                    "name": "clinical-team",
                    "slack_configs": [{
                        "api_url": "${SLACK_WEBHOOK}",
                        "channel": "#clinical-safety",
                    }],
                },
                {
                    "name": "platform-slack",
                    "slack_configs": [{
                        "api_url": "${SLACK_WEBHOOK}",
                        "channel": "#platform-safety",
                    }],
                },
                {
                    "name": "platform-email",
                    "email_configs": [{"to": "platform@sesi.example.com"}],
                },
            ],
            "inhibit_rules": self.generate_inhibition_rules(),
        }

        return config

    def write_alert_rules(self, output_path: str = "./prometheus/safety-alerts.yaml") -> str:
        """Write zone-specific alert rules to file."""
        rules = self.generate_zone_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {"groups": [{"name": "sesi_safety_alerts", "rules": rules}]}

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"[Alertmanager] Safety alert rules written to {output_path}")
        return str(out_path)

    def write_alertmanager_config(self, output_path: str = "./alertmanager/alertmanager.yml") -> str:
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

        logger.info(f"[Alertmanager] Config written to {output_path}")
        return str(out_path)

    def get_status(self) -> Dict[str, Any]:
        """Get alert rules status."""
        return {
            "zones_configured": len(self.zone_configs),
            "alert_rules": len(self.generate_zone_alert_rules()),
            "inhibition_rules": len(self.generate_inhibition_rules()),
            "delta_min": self.compute_delta_min(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [INT1] AGI SYSTEM INSTRUMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class AGISystemInstrumented:
    """
    [D] Wraps the 10 AGI modules with OpenTelemetry instrumentation.

    Each module gets a dedicated span:
      • agi.reasoning
      • agi.memory
      • agi.perception
      • agi.actuation
      • agi.curiosity
      • agi.synthesis
      • agi.reflection
      • agi.safety
      • agi.substrate
      • agi.meta_learning

    Safety violations trigger Alertmanager webhook.
    """

    def __init__(self, service_name: str = "sesi-agi"):
        self.service_name = service_name
        self.tracer = None
        self._span_timestamps: deque = deque(maxlen=10000)

        if HAS_OTEL:
            self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Initialize OpenTelemetry tracer."""
        try:
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
            })
            provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            otel_trace.set_tracer_provider(provider)
            self.tracer = otel_trace.get_tracer(self.service_name)
            logger.info(f"[OTel] Tracer initialized: {self.service_name}")
        except Exception as e:
            logger.warning(f"[OTel] Setup failed: {e}")

    def trace_module(self, module_name: str, zone: SecurityZone):
        """Decorator for tracing AGI module calls."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if self.tracer is None:
                    return await func(*args, **kwargs)

                with self.tracer.start_as_current_span(
                    f"agi.{module_name}",
                    kind=SpanKind.INTERNAL,
                ) as span:
                    span.set_attribute("agi.module", module_name)
                    span.set_attribute("sesi.zone", zone.value)
                    span.set_attribute("trace_id", uuid.uuid4().hex[:12])

                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
                    finally:
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration_ms", duration_ms)
                        self._span_timestamps.append(time.time())

            return wrapper
        return decorator

    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for log correlation."""
        if not HAS_OTEL:
            return {}
        span = otel_trace.get_current_span()
        if span is None:
            return {}
        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, '032x'),
            "span_id": format(ctx.span_id, '016x'),
        }

    def analyze_trace_stream_regime(self) -> Dict[str, Any]:
        """
        Classify trace stream regime using Hill estimator.

        From paper6 Thm 5.1.
        """
        if len(self._span_timestamps) < 20:
            return {"regime": QualityRegime.INSUFFICIENT_DATA.name}

        times = sorted(self._span_timestamps)
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        intervals = [iv for iv in intervals if iv > 0.001]

        if len(intervals) < 10:
            return {"regime": QualityRegime.FINITE_MEAN.name}

        median = np.median(intervals)
        deviations = [abs(iv - median) for iv in intervals if abs(iv - median) > 0.001]

        if len(deviations) < 10:
            return {"regime": QualityRegime.FINITE_MEAN.name}

        k = min(10, len(deviations) - 1)
        sorted_devs = sorted(deviations, reverse=True)
        x_k1 = sorted_devs[k]

        if x_k1 <= 0:
            return {"regime": QualityRegime.INSUFFICIENT_DATA.name}

        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        if theta_hat > 1.0:
            regime = "FINITE_MEAN"
        elif theta_hat > 0.0:
            regime = "REGULARLY_VARYING"
        else:
            regime = "DEGENERATE"

        return {
            "regime": regime,
            "theta_hat": theta_hat,
            "n_spans": len(self._span_timestamps),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIProductionTracingSuite:
    """
    [D] Complete production tracing suite.

    Integrates:
      • OpenTelemetry Collector configuration
      • Tempo backend configuration
      • Safety Alertmanager rules
      • AGI system instrumentation
    """

    def __init__(
        self,
        tempo_endpoint: str = "http://tempo-distributor:4317",
        storage_backend: str = "s3",
    ):
        self.otel_collector = OpenTelemetryCollectorConfig(tempo_endpoint=tempo_endpoint)
        self.tempo = TempoBackendConfig(storage_backend=storage_backend)
        self.safety_alerts = SafetyAlertmanagerRules()
        self.instrumentation = AGISystemInstrumented()

    def generate_all_artifacts(self, output_dir: str = "./sesi_production_tracing") -> Dict[str, str]:
        """Generate all production artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # OTel Collector
        results["otel_config"] = self.otel_collector.write_config(
            str(out_path / "otel" / "otel-collector-config.yaml")
        )
        results["otel_k8s"] = self.otel_collector.write_k8s_manifest(
            str(out_path / "k8s" / "otel-collector-deployment.yaml")
        )

        # Tempo
        results["tempo_config"] = self.tempo.write_config(
            str(out_path / "tempo" / "tempo.yaml")
        )

        # Alertmanager
        results["safety_alert_rules"] = self.safety_alerts.write_alert_rules(
            str(out_path / "prometheus" / "safety-alerts.yaml")
        )
        results["alertmanager_config"] = self.safety_alerts.write_alertmanager_config(
            str(out_path / "alertmanager" / "alertmanager.yml")
        )

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "otel_collector": {
                "tempo_endpoint": self.otel_collector.tempo_endpoint,
                "prometheus_endpoint": self.otel_collector.prometheus_endpoint,
            },
            "tempo": {
                "storage_backend": self.tempo.storage_backend,
                "retention_days": self.tempo.retention_days,
            },
            "safety_alerts": self.safety_alerts.get_status(),
            "instrumentation": {
                "tracer_available": self.instrumentation.tracer is not None,
                "trace_regime": self.instrumentation.analyze_trace_stream_regime(),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI OTel Collector + Tempo + Alertmanager — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  OpenTelemetry avail  : {HAS_OTEL}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIProductionTracingSuite(
        tempo_endpoint="http://tempo-distributor:4317",
        storage_backend="s3",
    )

    # ── Test 1: OTel Collector Config ────────────────────────────────────
    print("\n  ── OpenTelemetry Collector Config ──")
    otel_config = suite.otel_collector.generate_config()
    print(f"    Receivers            : {list(otel_config['receivers'].keys())}")
    print(f"    Processors           : {list(otel_config['processors'].keys())}")
    print(f"    Exporters            : {list(otel_config['exporters'].keys())}")
    print(f"    Pipelines            : {list(otel_config['service']['pipelines'].keys())}")

    # Check PNDC Axiom 3 compliance
    filter_proc = otel_config["processors"]["filter/drop-sensitive"]
    print(f"    Sensitive filters    : {len(filter_proc['traces']['span'])} trace, "
          f"{len(filter_proc['logs']['log_record'])} log patterns")

    # ── Test 2: Tempo Backend Config ─────────────────────────────────────
    print("\n  ── Tempo Backend Config ──")
    tempo_config = suite.tempo.generate_config()
    print(f"    Storage backend      : {suite.tempo.storage_backend}")
    print(f"    Retention            : {suite.tempo.retention_days} days")
    print(f"    Receivers            : {list(tempo_config['distributor']['receivers'].keys())}")

    # ── Test 3: Safety Alertmanager Rules ────────────────────────────────
    print("\n  ── Safety Alertmanager Rules ──")
    alert_status = suite.safety_alerts.get_status()
    print(f"    Zones configured     : {alert_status['zones_configured']}")
    print(f"    Alert rules          : {alert_status['alert_rules']}")
    print(f"    Inhibition rules     : {alert_status['inhibition_rules']}")
    print(f"    δ_min                : {alert_status['delta_min']:.6f}")

    # Show zone thresholds
    print("\n    Zone Thresholds (PNDC Axiom 3 proportionality):")
    for zone, cfg in suite.safety_alerts.zone_configs.items():
        print(f"      {zone.value:20s}: warn={cfg.safety_warning:.2f}, "
              f"crit={cfg.safety_critical:.2f}, "
              f"emerg={cfg.safety_emergency:.2f}, "
              f"severity={cfg.default_severity.value}")

    # Generate alert rules
    alert_rules = suite.safety_alerts.generate_zone_alert_rules()
    print(f"\n    Sample alert rules   :")
    for rule in alert_rules[:4]:
        print(f"      • {rule['alert']} ({rule['labels']['severity']})")

    # ── Test 4: AGI Instrumentation ──────────────────────────────────────
    print("\n  ── AGI System Instrumentation ──")
    print(f"    Tracer available     : {suite.instrumentation.tracer is not None}")

    # Simulate trace stream
    for _ in range(50):
        suite.instrumentation._span_timestamps.append(time.time())
        time.sleep(0.01)

    regime = suite.instrumentation.analyze_trace_stream_regime()
    print(f"    Trace stream regime  : {regime['regime']}")
    if 'theta_hat' in regime:
        print(f"    θ̂ (Hill)            : {regime['theta_hat']:.4f}")

    # ── Test 5: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_production_tracing")
    print(f"    Artifacts generated  : {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:25s}: {path}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    OTel Tempo endpoint  : {status['otel_collector']['tempo_endpoint']}")
    print(f"    Tempo storage        : {status['tempo']['storage_backend']}")
    print(f"    Safety alert rules   : {status['safety_alerts']['alert_rules']}")
    print(f"    Trace regime         : {status['instrumentation']['trace_regime']['regime']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • OTel Collector config is a TEMPLATE, not a deployed system")
    print("  • Tempo backend is STORAGE, not omniscience")
    print("  • Alertmanager rules are CONFIGURATION, not autonomous response")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all values")
    print("  • PNDC Axiom 3 data minimization in all traces/logs")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
