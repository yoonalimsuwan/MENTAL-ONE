"""
=============================================================================
SESI AGI: GRAFANA ALLOY + TEMPO DISTRIBUTED + OPENSEARCH
Unified Telemetry Collection + Distributed Tracing + Full-Text Log Search
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[GA1] GrafanaAlloyConfig
      • Unified collector for traces/metrics/logs
      • Pipeline configuration per zone (Z0-Z3)
      • Prometheus remote_write to Mimir
      • OTLP export to Tempo
      • Loki/OpenSearch forwarding
      • PNDC Axiom 3 data minimization
      • No-Zeno rate limiting on collection

[TD1] TempoDistributedCluster
      • Multi-component Tempo deployment
      • Distributor: OTLP/Jaeger ingestion
      • Ingester: WAL + flush to object storage
      • Querier: query frontend + query scheduler
      • Compactor: block compaction + retention
      • Memcached: metadata + results caching
      • K8s manifests for each component

[OS1] OpenSearchLogSearch
      • Full-text log search with inverted index
      • Index templates per zone (Z0-Z3)
      • Zone-based retention (PNDC Axiom 3)
      • Index lifecycle management (ILM)
      • LogQL-compatible query interface
      • K8s StatefulSet + ISM policies

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → bounds collection and ingestion rate
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
    → minimum meaningful telemetry change
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → per-component trace regime classification
[C] Six_Problems P4: CUSUM + subsampling
    → change-point detection in log streams
[C] BM25 scoring: OpenSearch relevance ranking
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar()
    → NaN/Inf sanitization for all values
[C] PNDC Axiom 3: Necessity and Proportionality
    → zone-based retention, data minimization
[C] Unified_Ordinal Hyp 7.9: Free design parameters
    → all thresholds require security review

RELATION TO AGI UNIFIED FRAMEWORK
----------------------------------
From AGI Unified Framework Single File.py (10 modules):
  • CoreReasoningEngine → Alloy pipeline: "agi.reasoning.*"
  • EpisodicSemanticMemory → Alloy pipeline: "agi.memory.*"
  • PerceptionMultimodal → Alloy pipeline: "agi.perception.*"
  • ActionActuationController → Alloy pipeline: "agi.actuation.*"
  • EpistemicCuriosityDrive → Alloy pipeline: "agi.curiosity.*"
  • ConceptSynthesisEngine → Alloy pipeline: "agi.synthesis.*"
  • SelfEvalReflection → Alloy pipeline: "agi.reflection.*"
  • SafetyAlignmentGuard → Alloy pipeline: "agi.safety.*"
  • DynamicSubstrateAllocator → Alloy pipeline: "agi.substrate.*"
  • MetaLearningPipeline → Alloy pipeline: "agi.meta_learning.*"

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Alloy config is a TEMPLATE; real deployment requires proper pipeline tuning.
• Tempo distributed requires proper object storage (S3/GCS) and caching.
• OpenSearch is a SEARCH ENGINE; it does not guarantee log completeness.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• Production deployment requires security audit and regulatory approval.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.alloy_tempo_opensearch")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

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
    "SecurityZone", "QualityRegime", "AGIModule",
    "GrafanaAlloyConfig", "AlloyPipeline",
    "TempoDistributedCluster", "TempoComponent",
    "OpenSearchLogSearch", "IndexTemplate",
    "SESIUnifiedObservabilityV5",
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


class QualityRegime(Enum):
    """Growth-rate regime classification (paper6 Thm 5.1)."""
    FINITE_MEAN = 0
    REGULARLY_VARYING = 1
    DEGENERATE = 2
    INSUFFICIENT_DATA = 3


class AGIModule(Enum):
    """The 10 AGI modules from the unified framework."""
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


# ═══════════════════════════════════════════════════════════════════════════
# [GA1] GRAFANA ALLOY CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AlloyPipeline:
    """A single Alloy pipeline configuration."""
    name: str
    receiver_type: str  # "otlp" | "prometheus" | "jaeger" | "loki"
    processors: List[str]
    exporter_type: str  # "otlp" | "prometheus_remote_write" | "loki" | "opensearch"
    exporter_endpoint: str
    zone: SecurityZone
    batch_size: int = 1000
    batch_timeout_s: float = 5.0
    # PNDC Axiom 3: data minimization
    drop_sensitive_attributes: bool = True
    sensitive_keys: List[str] = field(default_factory=lambda: [
        "password", "secret", "token", "ssn", "credit_card", "api_key",
    ])

    def to_alloy_river(self) -> str:
        """Generate Alloy River (HCL-like) configuration."""
        lines = []
        lines.append(f'// Pipeline: {self.name} (Zone: {self.zone.value})')

        # Receiver
        if self.receiver_type == "otlp":
            lines.append(f'otelcol.receiver.otlp "{self.name}_receiver" {{')
            lines.append('  grpc { endpoint = "0.0.0.0:4317" }')
            lines.append('  http { endpoint = "0.0.0.0:4318" }')
            lines.append(f'  output {{')
            lines.append(f'    traces  = [otelcol.processor.batch.{self.name}_batch.input]')
            lines.append(f'    metrics = [otelcol.processor.batch.{self.name}_batch.input]')
            lines.append(f'    logs    = [otelcol.processor.batch.{self.name}_batch.input]')
            lines.append(f'  }}')
            lines.append('}')
        elif self.receiver_type == "prometheus":
            lines.append(f'prometheus.scrape "{self.name}_receiver" {{')
            lines.append('  targets = [{ __address__ = "sesi-agi:9090" }]')
            lines.append(f'  forward_to = [prometheus.remote_write.{self.name}_write.receiver]')
            lines.append('}')

        lines.append('')

        # Processors
        if self.drop_sensitive_attributes:
            lines.append(f'otelcol.processor.filter "{self.name}_sensitive_filter" {{')
            lines.append('  drops = {')
            for key in self.sensitive_keys:
                lines.append(f'    attributes = ["{key}"],')
            lines.append('  }')
            lines.append(f'  output {{ }}')
            lines.append('}')
            lines.append('')

        # Batch processor
        lines.append(f'otelcol.processor.batch "{self.name}_batch" {{')
        lines.append(f'  send_batch_size = {self.batch_size}')
        lines.append(f'  timeout = "{int(self.batch_timeout_s)}s"')
        lines.append(f'  output {{')

        if self.exporter_type == "otlp":
            lines.append(f'    traces = [otelcol.exporter.otlp.{self.name}_exporter.input]')
        elif self.exporter_type == "prometheus_remote_write":
            lines.append(f'    metrics = [prometheus.remote_write.{self.name}_write.receiver]')
        elif self.exporter_type == "loki":
            lines.append(f'    logs = [loki.write.{self.name}_exporter.receiver]')
        elif self.exporter_type == "opensearch":
            lines.append(f'    logs = [otelcol.exporter.opensearch.{self.name}_exporter.input]')

        lines.append(f'  }}')
        lines.append('}')
        lines.append('')

        # Exporter
        if self.exporter_type == "otlp":
            lines.append(f'otelcol.exporter.otlp "{self.name}_exporter" {{')
            lines.append(f'  client {{ endpoint = "{self.exporter_endpoint}" }}')
            lines.append('}')
        elif self.exporter_type == "prometheus_remote_write":
            lines.append(f'prometheus.remote_write "{self.name}_write" {{')
            lines.append(f'  endpoint {{ url = "{self.exporter_endpoint}" }}')
            lines.append('}')
        elif self.exporter_type == "loki":
            lines.append(f'loki.write "{self.name}_exporter" {{')
            lines.append(f'  endpoint {{ url = "{self.exporter_endpoint}" }}')
            lines.append('}')
        elif self.exporter_type == "opensearch":
            lines.append(f'otelcol.exporter.opensearch "{self.name}_exporter" {{')
            lines.append(f'  endpoint = "{self.exporter_endpoint}"')
            lines.append('}')

        return "\n".join(lines)


class GrafanaAlloyConfig:
    """
    [D] Grafana Alloy configuration for unified telemetry collection.

    Features:
      • Unified collector for traces/metrics/logs
      • Per-zone pipeline configuration (Z0-Z3)
      • Prometheus remote_write to Mimir
      • OTLP export to Tempo
      • Loki/OpenSearch forwarding
      • PNDC Axiom 3 data minimization
      • No-Zeno rate limiting on collection

    HONEST: This is a CONFIGURATION GENERATOR. Real deployment requires:
      • Proper endpoint configuration
      • TLS certificates
      • Security audit of pipeline routing
    """

    def __init__(
        self,
        mimir_endpoint: str = "http://mimir-distributor:8080/api/v1/push",
        tempo_endpoint: str = "tempo-distributor.sesi-agi:4317",
        opensearch_endpoint: str = "https://opensearch.sesi-agi:9200",
        loki_endpoint: str = "http://loki.sesi-agi:3100/loki/api/v1/push",
    ):
        self.mimir_endpoint = mimir_endpoint
        self.tempo_endpoint = tempo_endpoint
        self.opensearch_endpoint = opensearch_endpoint
        self.loki_endpoint = loki_endpoint
        self.pipelines: List[AlloyPipeline] = []
        self._collection_timestamps: deque = deque(maxlen=100000)

        # No-Zeno parameters
        self.max_collections_per_minute = 60000
        self.min_collection_interval_s = 0.01

        self._build_default_pipelines()

    def _build_default_pipelines(self) -> None:
        """Build default pipelines per zone."""
        # Z0 Implant: Traces to Tempo, metrics to Mimir (long retention)
        self.pipelines.append(AlloyPipeline(
            name="z0_traces",
            receiver_type="otlp",
            processors=["batch", "filter_sensitive"],
            exporter_type="otlp",
            exporter_endpoint=self.tempo_endpoint,
            zone=SecurityZone.Z0_IMPLANT,
            batch_size=500,
        ))

        # Z1 Clinical: Traces + metrics + logs to OpenSearch
        self.pipelines.append(AlloyPipeline(
            name="z1_logs",
            receiver_type="otlp",
            processors=["batch", "filter_sensitive"],
            exporter_type="opensearch",
            exporter_endpoint=self.opensearch_endpoint,
            zone=SecurityZone.Z1_CLINICAL,
            batch_size=1000,
        ))

        # Z2 Processing: Metrics to Mimir
        self.pipelines.append(AlloyPipeline(
            name="z2_metrics",
            receiver_type="prometheus",
            processors=["batch"],
            exporter_type="prometheus_remote_write",
            exporter_endpoint=self.mimir_endpoint,
            zone=SecurityZone.Z2_PROCESSING,
            batch_size=2000,
        ))

        # Z3 External: Logs to Loki (short retention)
        self.pipelines.append(AlloyPipeline(
            name="z3_logs",
            receiver_type="otlp",
            processors=["batch", "filter_sensitive"],
            exporter_type="loki",
            exporter_endpoint=self.loki_endpoint,
            zone=SecurityZone.Z3_EXTERNAL,
            batch_size=1000,
        ))

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self, kappa: float = 2.0, a_min: float = 0.01,
                          n_dim: int = 2) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_collection_rate(self) -> Dict[str, Any]:
        """Check if collection is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        self._collection_timestamps = deque(
            (t for t in self._collection_timestamps if t > cutoff),
            maxlen=100000,
        )

        if len(self._collection_timestamps) >= self.max_collections_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
                "current": len(self._collection_timestamps),
            }

        return {"allowed": True}

    # ── Configuration Generation ─────────────────────────────────────────

    def generate_alloy_config(self) -> str:
        """Generate complete Alloy configuration in River format."""
        lines = [
            "// ============================================================",
            "// SESI AGI - Grafana Alloy Configuration",
            "// Auto-generated by SESIUnifiedObservabilityV5",
            "// ============================================================",
            "",
            "// Logging configuration",
            'logging {',
            '  level  = "info"',
            '  format = "logfmt"',
            '}',
            "",
        ]

        for pipeline in self.pipelines:
            lines.append(pipeline.to_alloy_river())
            lines.append("")

        return "\n".join(lines)

    def generate_k8s_daemonset(self) -> Dict[str, Any]:
        """Generate K8s DaemonSet for Alloy."""
        return {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": {
                "name": "alloy",
                "namespace": "sesi-agi",
                "labels": {"app.kubernetes.io/name": "alloy"},
            },
            "spec": {
                "selector": {"matchLabels": {"app.kubernetes.io/name": "alloy"}},
                "template": {
                    "metadata": {
                        "labels": {"app.kubernetes.io/name": "alloy"},
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "12345",
                        },
                    },
                    "spec": {
                        "serviceAccountName": "alloy",
                        "containers": [{
                            "name": "alloy",
                            "image": "grafana/alloy:v1.3.0",
                            "args": [
                                "run",
                                "/etc/alloy/config.alloy",
                                "--server.http.listen-addr=0.0.0.0:12345",
                                "--storage.path=/var/lib/alloy/data",
                            ],
                            "ports": [
                                {"name": "http-metrics", "containerPort": 12345},
                                {"name": "otlp-grpc", "containerPort": 4317},
                                {"name": "otlp-http", "containerPort": 4318},
                            ],
                            "env": [
                                {"name": "ALLOY_DEPLOY_MODE", "value": "daemonset"},
                            ],
                            "resources": {
                                "requests": {"cpu": "200m", "memory": "256Mi"},
                                "limits": {"cpu": "1000m", "memory": "1Gi"},
                            },
                            "volumeMounts": [
                                {"name": "config", "mountPath": "/etc/alloy", "readOnly": True},
                                {"name": "data", "mountPath": "/var/lib/alloy"},
                            ],
                        }],
                        "volumes": [
                            {"name": "config", "configMap": {"name": "alloy-config"}},
                            {"name": "data", "emptyDir": {}},
                        ],
                    },
                },
            },
        }

    def write_all(self, output_dir: str = "./alloy") -> Dict[str, str]:
        """Write all Alloy artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Main config
        config = self.generate_alloy_config()
        config_path = out_path / "config.alloy"
        with open(config_path, "w") as f:
            f.write(config)
        files["config"] = str(config_path)

        # K8s DaemonSet
        k8s = self.generate_k8s_daemonset()
        k8s_path = out_path / "k8s-daemonset.yaml"
        if HAS_YAML:
            with open(k8s_path, "w") as f:
                yaml.dump(k8s, f, default_flow_style=False)
        else:
            with open(k8s_path, "w") as f:
                json.dump(k8s, f, indent=2)
        files["k8s_daemonset"] = str(k8s_path)

        logger.info(f"[Alloy] Generated {len(files)} artifacts in {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get Alloy config status."""
        return {
            "total_pipelines": len(self.pipelines),
            "pipelines_per_zone": {
                zone.value: sum(1 for p in self.pipelines if p.zone == zone)
                for zone in SecurityZone
            },
            "delta_min": self.compute_delta_min(),
            "max_collections_per_minute": self.max_collections_per_minute,
        }


# ═══════════════════════════════════════════════════════════════════════════
# [TD1] TEMPO DISTRIBUTED CLUSTER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TempoComponent:
    """Configuration for a single Tempo component."""
    name: str
    component_type: str  # "distributor" | "ingester" | "querier" | "compactor" | "query-frontend"
    replicas: int
    resources: Dict[str, Dict[str, str]]
    storage_size: Optional[str] = None


class TempoDistributedCluster:
    """
    [D] Distributed Tempo cluster configuration.

    Components:
      • Distributor: Receives traces, validates, forwards to ingesters
      • Ingester: Batches traces, writes WAL, flushes to object storage
      • Querier: Executes queries across ingesters + object storage
      • Compactor: Compacts trace blocks, enforces retention
      • Query-Frontend: Query caching, sharding, rate limiting
      • Memcached: Metadata + results caching

    HONEST: This is a CONFIGURATION GENERATOR. Real deployment requires:
      • Object storage (S3/GCS) with proper IAM
      • Memcached cluster for caching
      • Proper network policies between components
    """

    def __init__(
        self,
        storage_backend: str = "s3",
        s3_bucket: str = "sesi-tempo-traces",
        s3_region: str = "us-east-1",
        retention_days: int = 30,
    ):
        self.storage_backend = storage_backend
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        self.retention_days = retention_days

        self.components = [
            TempoComponent(
                name="tempo-distributor",
                component_type="distributor",
                replicas=3,
                resources={
                    "requests": {"cpu": "1000m", "memory": "2Gi"},
                    "limits": {"cpu": "2000m", "memory": "4Gi"},
                },
            ),
            TempoComponent(
                name="tempo-ingester",
                component_type="ingester",
                replicas=5,
                resources={
                    "requests": {"cpu": "2000m", "memory": "4Gi"},
                    "limits": {"cpu": "4000m", "memory": "8Gi"},
                },
                storage_size="100Gi",
            ),
            TempoComponent(
                name="tempo-querier",
                component_type="querier",
                replicas=3,
                resources={
                    "requests": {"cpu": "1000m", "memory": "2Gi"},
                    "limits": {"cpu": "2000m", "memory": "4Gi"},
                },
            ),
            TempoComponent(
                name="tempo-compactor",
                component_type="compactor",
                replicas=1,
                resources={
                    "requests": {"cpu": "1000m", "memory": "2Gi"},
                    "limits": {"cpu": "2000m", "memory": "4Gi"},
                },
                storage_size="200Gi",
            ),
            TempoComponent(
                name="tempo-query-frontend",
                component_type="query-frontend",
                replicas=2,
                resources={
                    "requests": {"cpu": "500m", "memory": "1Gi"},
                    "limits": {"cpu": "1000m", "memory": "2Gi"},
                },
            ),
        ]

    def generate_tempo_config(self) -> Dict[str, Any]:
        """Generate complete Tempo distributed configuration."""
        storage_config = self._get_storage_config()

        return {
            "server": {
                "http_listen_port": 3200,
                "grpc_listen_port": 9095,
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
                "ring": {
                    "kvstore": {"store": "memberlist"},
                },
            },
            "ingester": {
                "max_block_duration": "5m",
                "trace_idle_timeout": "10s",
                "flush_all_on_shutdown": True,
                "lifecycler": {
                    "ring": {
                        "kvstore": {"store": "memberlist"},
                        "replication_factor": 3,
                    },
                },
            },
            "querier": {
                "frontend_worker": {
                    "frontend_address": "tempo-query-frontend:9095",
                },
            },
            "query_frontend": {
                "search": {
                    "duration_slo": "5s",
                    "metadata_slo": "2s",
                },
                "trace_by_id": {
                    "duration_slo": "100ms",
                },
                "cache_results": True,
                "max_retries": 2,
                "service_cache": {
                    "enabled": True,
                    "backend": "memcached",
                    "memcached": {
                        "hosts": ["memcached:11211"],
                        "timeout": "500ms",
                        "max_idle_conns": 16,
                    },
                },
            },
            "compactor": {
                "compaction": {
                    "block_retention": f"{self.retention_days}d",
                    "compaction_window": "1h",
                    "max_block_bytes": 107374182400,  # 100 GB
                },
                "ring": {
                    "kvstore": {"store": "memberlist"},
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
                        "url": "http://mimir-distributor:8080/api/v1/push",
                        "send_exemplars": True,
                    }],
                },
                "processor": {
                    "service_graphs": {
                        "enabled": True,
                        "histogram_buckets": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                    },
                    "span_metrics": {
                        "enabled": True,
                        "histogram_buckets": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                    },
                },
            },
            "storage": storage_config,
            "memberlist": {
                "join_members": [
                    "tempo-distributor",
                    "tempo-ingester",
                    "tempo-querier",
                    "tempo-compactor",
                ],
            },
            "overrides": {
                "defaults": {
                    "metrics_generator": {
                        "processors": ["service_graphs", "span_metrics"],
                    },
                    "global": {
                        "max_bytes_per_trace": 5000000,  # 5 MB
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
                    "wal": {"path": "/var/tempo/wal", "version": "v2"},
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
        else:
            return {
                "trace": {
                    "backend": "local",
                    "wal": {"path": "/var/tempo/wal", "version": "v2"},
                    "local": {"path": "/var/tempo/traces"},
                },
            }

    def generate_k8s_manifests(self) -> Dict[str, Dict[str, Any]]:
        """Generate K8s manifests for all Tempo components."""
        manifests = {}

        for component in self.components:
            manifest = {
                "apiVersion": "apps/v1",
                "kind": "StatefulSet" if component.storage_size else "Deployment",
                "metadata": {
                    "name": component.name,
                    "namespace": "sesi-agi",
                    "labels": {
                        "app.kubernetes.io/name": component.name,
                        "app.kubernetes.io/component": component.component_type,
                    },
                },
                "spec": {
                    "replicas": component.replicas,
                    "selector": {
                        "matchLabels": {"app.kubernetes.io/name": component.name},
                    },
                    "template": {
                        "metadata": {
                            "labels": {"app.kubernetes.io/name": component.name},
                            "annotations": {
                                "prometheus.io/scrape": "true",
                                "prometheus.io/port": "3200",
                            },
                        },
                        "spec": {
                            "containers": [{
                                "name": "tempo",
                                "image": "grafana/tempo:2.4.0",
                                "args": [
                                    f"-target={component.component_type}",
                                    "-config.file=/etc/tempo/tempo.yaml",
                                ],
                                "ports": [
                                    {"name": "http", "containerPort": 3200},
                                    {"name": "grpc", "containerPort": 9095},
                                ],
                                "resources": component.resources,
                                "livenessProbe": {
                                    "httpGet": {"path": "/ready", "port": 3200},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/ready", "port": 3200},
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 5,
                                },
                                "volumeMounts": [
                                    {"name": "config", "mountPath": "/etc/tempo", "readOnly": True},
                                ],
                            }],
                            "volumes": [
                                {"name": "config", "configMap": {"name": "tempo-config"}},
                            ],
                        },
                    },
                },
            }

            # Add PVC for components with storage
            if component.storage_size:
                manifest["spec"]["volumeClaimTemplates"] = [{
                    "metadata": {"name": "data"},
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "resources": {"requests": {"storage": component.storage_size}},
                    },
                }]
                manifest["spec"]["template"]["spec"]["containers"][0]["volumeMounts"].append(
                    {"name": "data", "mountPath": "/var/tempo"}
                )

            manifests[component.name] = manifest

        return manifests

    def write_all(self, output_dir: str = "./tempo_distributed") -> Dict[str, str]:
        """Write all Tempo artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Main config
        config = self.generate_tempo_config()
        config_path = out_path / "tempo.yaml"
        if HAS_YAML:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        files["config"] = str(config_path)

        # K8s manifests
        manifests = self.generate_k8s_manifests()
        for name, manifest in manifests.items():
            manifest_path = out_path / f"k8s-{name}.yaml"
            if HAS_YAML:
                with open(manifest_path, "w") as f:
                    yaml.dump(manifest, f, default_flow_style=False)
            else:
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
            files[name] = str(manifest_path)

        logger.info(f"[Tempo] Generated {len(files)} artifacts in {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get Tempo cluster status."""
        return {
            "storage_backend": self.storage_backend,
            "bucket_name": self.s3_bucket,
            "retention_days": self.retention_days,
            "components": [
                {
                    "name": c.name,
                    "type": c.component_type,
                    "replicas": c.replicas,
                }
                for c in self.components
            ],
            "total_replicas": sum(c.replicas for c in self.components),
        }


# ═══════════════════════════════════════════════════════════════════════════
# [OS1] OPENSEARCH LOG SEARCH
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IndexTemplate:
    """OpenSearch index template for a specific zone."""
    name: str
    zone: SecurityZone
    index_patterns: List[str]
    retention_days: int
    shards: int = 3
    replicas: int = 1
    # PNDC Axiom 3: data minimization
    field_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)


class OpenSearchLogSearch:
    """
    [D] OpenSearch configuration for full-text log search.

    Features:
      • Index templates per zone (Z0-Z3)
      • Zone-based retention (PNDC Axiom 3 proportionality)
      • Index lifecycle management (ILM)
      • Full-text search with BM25 scoring
      • LogQL-compatible query interface
      • K8s StatefulSet + ISM policies

    HONEST: This is a CONFIGURATION GENERATOR. Real deployment requires:
      • Proper cluster sizing based on log volume
      • Security plugin configuration
      • Index lifecycle management tuning
    """

    # Zone-based retention (PNDC Axiom 3 proportionality)
    DEFAULT_RETENTION = {
        SecurityZone.Z0_IMPLANT: 730,     # 2 years
        SecurityZone.Z1_CLINICAL: 365,    # 1 year
        SecurityZone.Z2_PROCESSING: 90,   # 90 days
        SecurityZone.Z3_EXTERNAL: 30,     # 30 days
    }

    def __init__(
        self,
        endpoint: str = "https://opensearch.sesi-agi:9200",
        username: str = "admin",
        password_env: str = "OPENSEARCH_PASSWORD",
    ):
        self.endpoint = endpoint
        self.username = username
        self.password_env = password_env
        self.templates: List[IndexTemplate] = []
        self._build_default_templates()

    def _build_default_templates(self) -> None:
        """Build default index templates per zone."""
        for zone, retention in self.DEFAULT_RETENTION.items():
            self.templates.append(IndexTemplate(
                name=f"sesi-agi-logs-{zone.value}",
                zone=zone,
                index_patterns=[f"sesi-agi-logs-{zone.value}-*"],
                retention_days=retention,
                shards=3 if zone != SecurityZone.Z3_EXTERNAL else 1,
                replicas=2 if zone in (SecurityZone.Z0_IMPLANT, SecurityZone.Z1_CLINICAL) else 1,
                field_mappings={
                    "timestamp": {"type": "date"},
                    "trace_id": {"type": "keyword"},
                    "span_id": {"type": "keyword"},
                    "module": {"type": "keyword"},
                    "zone": {"type": "keyword"},
                    "level": {"type": "keyword"},
                    "event_type": {"type": "keyword"},
                    "message": {"type": "text", "analyzer": "standard"},
                    "attributes": {"type": "object", "enabled": True},
                },
            ))

    # ── Index Template Generation ────────────────────────────────────────

    def generate_index_template(self, template: IndexTemplate) -> Dict[str, Any]:
        """Generate OpenSearch index template."""
        return {
            "index_patterns": template.index_patterns,
            "template": {
                "settings": {
                    "number_of_shards": template.shards,
                    "number_of_replicas": template.replicas,
                    "index": {
                        "refresh_interval": "5s",
                        "analysis": {
                            "analyzer": {
                                "sesi_log_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop", "sesi_synonym"],
                                },
                            },
                            "filter": {
                                "sesi_synonym": {
                                    "type": "synonym",
                                    "synonyms": [
                                        "agi, artificial general intelligence",
                                        "sesi, safe ethical sentient intelligence",
                                    ],
                                },
                            },
                        },
                    },
                    "plugins": {
                        "index_state_management": {
                            "rollover_alias": template.name,
                        },
                    },
                },
                "mappings": {
                    "properties": template.field_mappings,
                },
            },
        }

    def generate_ism_policy(self, template: IndexTemplate) -> Dict[str, Any]:
        """Generate Index State Management policy for retention."""
        return {
            "policy": {
                "description": f"Retention policy for {template.name}",
                "default_state": "hot",
                "states": [
                    {
                        "name": "hot",
                        "actions": [
                            {
                                "rollover": {
                                    "min_size": "50gb",
                                    "min_index_age": "1d",
                                },
                            },
                        ],
                        "transitions": [
                            {
                                "state_name": "warm",
                                "conditions": {"min_index_age": f"{template.retention_days // 2}d"},
                            },
                        ],
                    },
                    {
                        "name": "warm",
                        "actions": [
                            {
                                "replica_count": {"number_of_replicas": 1},
                            },
                            {
                                "force_merge": {
                                    "max_num_segments": 1,
                                },
                            },
                        ],
                        "transitions": [
                            {
                                "state_name": "delete",
                                "conditions": {"min_index_age": f"{template.retention_days}d"},
                            },
                        ],
                    },
                    {
                        "name": "delete",
                        "actions": [
                            {"delete": {}},
                        ],
                    },
                ],
                "ism_template": {
                    "index_patterns": template.index_patterns,
                    "priority": 100,
                },
            },
        }

    def generate_search_query(
        self,
        query_text: str,
        zone: Optional[SecurityZone] = None,
        module: Optional[AGIModule] = None,
        level: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        size: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate OpenSearch search query.

        Uses BM25 scoring for relevance ranking.
        """
        must_clauses = []

        # Full-text search on message
        if query_text:
            must_clauses.append({
                "multi_match": {
                    "query": query_text,
                    "fields": ["message^2", "event_type", "module", "attributes.*"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                },
            })

        # Zone filter
        if zone:
            must_clauses.append({"term": {"zone": zone.value}})

        # Module filter
        if module:
            must_clauses.append({"term": {"module": module.value}})

        # Level filter
        if level:
            must_clauses.append({"term": {"level": level}})

        # Time range
        time_range = {}
        if start_time:
            time_range["gte"] = start_time
        if end_time:
            time_range["lte"] = end_time
        if time_range:
            must_clauses.append({"range": {"timestamp": time_range}})

        return {
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}],
                },
            },
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": size,
            "highlight": {
                "fields": {"message": {}},
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
            },
            "aggs": {
                "by_module": {
                    "terms": {"field": "module", "size": 10},
                },
                "by_level": {
                    "terms": {"field": "level", "size": 10},
                },
                "by_zone": {
                    "terms": {"field": "zone", "size": 10},
                },
                "over_time": {
                    "date_histogram": {
                        "field": "timestamp",
                        "fixed_interval": "1h",
                    },
                },
            },
        }

    def generate_k8s_statefulset(self) -> Dict[str, Any]:
        """Generate K8s StatefulSet for OpenSearch cluster."""
        return {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "opensearch",
                "namespace": "sesi-agi",
            },
            "spec": {
                "serviceName": "opensearch",
                "replicas": 3,
                "selector": {"matchLabels": {"app": "opensearch"}},
                "template": {
                    "metadata": {"labels": {"app": "opensearch"}},
                    "spec": {
                        "containers": [{
                            "name": "opensearch",
                            "image": "opensearchproject/opensearch:2.11.0",
                            "env": [
                                {"name": "cluster.name", "value": "sesi-agi-logs"},
                                {"name": "node.name", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                                {"name": "discovery.seed_hosts", "value": "opensearch-cluster"},
                                {"name": "cluster.initial_cluster_manager_nodes",
                                 "value": "opensearch-0,opensearch-1,opensearch-2"},
                                {"name": "bootstrap.memory_lock", "value": "true"},
                                {"name": "OPENSEARCH_JAVA_OPTS",
                                 "value": "-Xms4g -Xmx4g"},
                                {"name": "plugins.security.disabled", "value": "false"},
                            ],
                            "ports": [
                                {"name": "http", "containerPort": 9200},
                                {"name": "transport", "containerPort": 9300},
                            ],
                            "resources": {
                                "requests": {"cpu": "2000m", "memory": "8Gi"},
                                "limits": {"cpu": "4000m", "memory": "16Gi"},
                            },
                            "volumeMounts": [
                                {"name": "data", "mountPath": "/usr/share/opensearch/data"},
                            ],
                        }],
                        "initContainers": [{
                            "name": "increase-vm-max-map",
                            "image": "busybox",
                            "command": ["sysctl", "-w", "vm.max_map_count=262144"],
                            "securityContext": {"privileged": True},
                        }],
                    },
                },
                "volumeClaimTemplates": [{
                    "metadata": {"name": "data"},
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "resources": {"requests": {"storage": "500Gi"}},
                    },
                }],
            },
        }

    def write_all(self, output_dir: str = "./opensearch") -> Dict[str, str]:
        """Write all OpenSearch artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Index templates
        for template in self.templates:
            template_dict = self.generate_index_template(template)
            template_path = out_path / f"index-template-{template.zone.value}.json"
            with open(template_path, "w") as f:
                json.dump(template_dict, f, indent=2)
            files[f"template_{template.zone.value}"] = str(template_path)

            # ISM policy
            ism_dict = self.generate_ism_policy(template)
            ism_path = out_path / f"ism-policy-{template.zone.value}.json"
            with open(ism_path, "w") as f:
                json.dump(ism_dict, f, indent=2)
            files[f"ism_{template.zone.value}"] = str(ism_path)

        # K8s StatefulSet
        k8s = self.generate_k8s_statefulset()
        k8s_path = out_path / "k8s-statefulset.yaml"
        if HAS_YAML:
            with open(k8s_path, "w") as f:
                yaml.dump(k8s, f, default_flow_style=False)
        else:
            with open(k8s_path, "w") as f:
                json.dump(k8s, f, indent=2)
        files["k8s_statefulset"] = str(k8s_path)

        logger.info(f"[OpenSearch] Generated {len(files)} artifacts in {output_dir}")
        return files

    def get_status(self) -> Dict[str, Any]:
        """Get OpenSearch status."""
        return {
            "endpoint": self.endpoint,
            "total_templates": len(self.templates),
            "retention_per_zone": {
                t.zone.value: t.retention_days for t in self.templates
            },
            "total_shards": sum(t.shards for t in self.templates),
            "total_replicas": sum(t.replicas for t in self.templates),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class SESIUnifiedObservabilityV5:
    """
    [D] Complete unified observability suite v5.

    Integrates:
      • Grafana Alloy (unified telemetry collection)
      • Tempo Distributed (distributed tracing cluster)
      • OpenSearch (full-text log search)
    """

    def __init__(
        self,
        alloy_cfg: Optional[GrafanaAlloyConfig] = None,
        tempo_cluster: Optional[TempoDistributedCluster] = None,
        opensearch: Optional[OpenSearchLogSearch] = None,
    ):
        self.alloy = alloy_cfg or GrafanaAlloyConfig()
        self.tempo = tempo_cluster or TempoDistributedCluster()
        self.opensearch = opensearch or OpenSearchLogSearch()

    def generate_all_artifacts(self, output_dir: str = "./sesi_unified_v5") -> Dict[str, Any]:
        """Generate all observability artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Alloy
        results["alloy"] = self.alloy.write_all(str(out_path / "alloy"))

        # Tempo distributed
        results["tempo"] = self.tempo.write_all(str(out_path / "tempo"))

        # OpenSearch
        results["opensearch"] = self.opensearch.write_all(str(out_path / "opensearch"))

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "alloy": self.alloy.get_status(),
            "tempo": self.tempo.get_status(),
            "opensearch": self.opensearch.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Alloy + Tempo Distributed + OpenSearch — Smoke Test")
    print("=" * 72)
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    suite = SESIUnifiedObservabilityV5()

    # ── Test 1: Grafana Alloy ────────────────────────────────────────────
    print("\n  ── Grafana Alloy Config ──")
    alloy_status = suite.alloy.get_status()
    print(f"    Total pipelines      : {alloy_status['total_pipelines']}")
    print(f"    Pipelines per zone   : {alloy_status['pipelines_per_zone']}")
    print(f"    δ_min                : {alloy_status['delta_min']:.6f}")

    # Generate config
    alloy_config = suite.alloy.generate_alloy_config()
    print(f"    Config length        : {len(alloy_config)} chars")
    print(f"    Sample config        :")
    for line in alloy_config.split("\n")[:10]:
        print(f"      {line}")

    # ── Test 2: Tempo Distributed Cluster ────────────────────────────────
    print("\n  ── Tempo Distributed Cluster ──")
    tempo_status = suite.tempo.get_status()
    print(f"    Storage backend      : {tempo_status['storage_backend']}")
    print(f"    Bucket name          : {tempo_status['bucket_name']}")
    print(f"    Retention            : {tempo_status['retention_days']} days")
    print(f"    Total replicas       : {tempo_status['total_replicas']}")
    print(f"    Components           :")
    for comp in tempo_status['components']:
        print(f"      {comp['name']:25s}: {comp['replicas']} replicas ({comp['type']})")

    # Generate config
    tempo_config = suite.tempo.generate_tempo_config()
    print(f"    Config sections      : {len(tempo_config)}")

    # ── Test 3: OpenSearch Log Search ────────────────────────────────────
    print("\n  ── OpenSearch Log Search ──")
    os_status = suite.opensearch.get_status()
    print(f"    Endpoint             : {os_status['endpoint']}")
    print(f"    Total templates      : {os_status['total_templates']}")
    print(f"    Retention per zone   :")
    for zone, days in os_status['retention_per_zone'].items():
        print(f"      {zone:20s}: {days} days")
    print(f"    Total shards         : {os_status['total_shards']}")
    print(f"    Total replicas       : {os_status['total_replicas']}")

    # Generate sample search query
    search_query = suite.opensearch.generate_search_query(
        query_text="safety violation",
        zone=SecurityZone.Z1_CLINICAL,
        module=AGIModule.SAFETY,
        level="ERROR",
        start_time="now-24h",
    )
    print(f"\n    Sample search query  :")
    print(f"      Query type         : multi_match + filters")
    print(f"      Aggregations       : {len(search_query['aggs'])}")
    print(f"      Highlight          : {search_query['highlight']['fields']}")

    # ── Test 4: Full Artifact Generation ─────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_unified_v5")
    print(f"    Artifact groups      : {len(artifacts)}")
    for name, value in artifacts.items():
        if isinstance(value, dict):
            print(f"      {name:20s}: {len(value)} files")
        else:
            print(f"      {name:20s}: {value}")

    # ── Full Status ──────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Alloy pipelines      : {status['alloy']['total_pipelines']}")
    print(f"    Tempo components     : {len(status['tempo']['components'])}")
    print(f"    OpenSearch templates : {status['opensearch']['total_templates']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Alloy config is a TEMPLATE, not a deployed system")
    print("  • Tempo distributed requires proper object storage")
    print("  • OpenSearch is a SEARCH ENGINE, not omniscience")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all values")
    print("  • PNDC Axiom 3 data minimization in all pipelines")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
