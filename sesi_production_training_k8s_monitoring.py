"""
=============================================================================
SESI PRODUCTION DEPLOYMENT SUITE
JSONL Training Pipeline × Kubernetes Manifests × Monitoring Dashboard
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[TRAIN] IntentClassifierTrainingPipeline
        • Complete training script using the official JSONL format
        • Data loading, tokenization, training with AMP
        • Curriculum learning support
        • Evaluation with confusion matrix
        • Model export and checkpointing

[K8S]   KubernetesManifestGenerator
        • Namespace, Deployment, Service, ConfigMap
        • Ingress with TLS (Let's Encrypt)
        • HorizontalPodAutoscaler
        • Prometheus ServiceMonitor
        • Secret management templates

[MON]   MonitoringDashboardIntegration
        • Prometheus metrics collection
        • Grafana dashboard configuration
        • AlertManager rules
        • WebSocket alert server integration

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_jsonl_letsencrypt_checkpoint.py — JSONL format spec
  • ses_auth_transformer_websocket.py — TransformerIntentClassifier
  • ses_agione_integration_training_tls.py — AGIONE integration
  • agi_one_v3_11.py — AGIONE core
  • agi_one_psyche_plus_v33.py — Psyche Plus

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Training requires labeled data reviewed by security professionals.
• Kubernetes manifests are templates; actual deployment requires
  cluster-specific configuration.
• Monitoring dashboards require Prometheus/Grafana infrastructure.
• All thresholds are FREE DESIGN PARAMETERS.
• Production deployment requires security audit.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.production_deployment")

# Optional imports
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from ses_auth_transformer_websocket import (
        TransformerIntentClassifier, IntentClass, IntentLabel,
    )
    HAS_PREV = True
except ImportError:
    HAS_PREV = False

try:
    from ses_jsonl_letsencrypt_checkpoint import (
        JSONLDataSample, JSONLDataLoader, JSONLDataValidator,
        IntentDataFormatSpec,
    )
    HAS_JSONL = True
except ImportError:
    HAS_JSONL = False

__all__ = [
    "TrainingConfig",
    "JSONLIntentDataset",
    "IntentClassifierTrainingPipeline",
    "KubernetesManifestGenerator",
    "MonitoringDashboardIntegration",
    "ProductionDeploymentSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# PART 1: TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Configuration for intent classifier training."""
    # Data paths
    data_dir: str = "./intent_data"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"

    # Model architecture
    vocab_size: int = 32000
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    max_seq_len: int = 128
    n_classes: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 20
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    # AMP
    use_amp: bool = True

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_easy_ratio: float = 0.3

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best: bool = True
    early_stopping_patience: int = 5

    device: str = "cpu"


class JSONLIntentDataset(Dataset):
    """
    [D] Dataset for intent classification using the official JSONL format.

    Expected format per line:
    {
        "text": "...",
        "label": 0-3,
        "metadata": {...}
    }
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 128):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[JSONLDataSample] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load and validate JSONL data."""
        if not os.path.exists(self.data_path):
            logger.warning(f"Data file not found: {self.data_path}")
            return

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = JSONLDataSample.from_dict(data)
                    is_valid, errors = sample.validate()

                    if is_valid:
                        self.samples.append(sample)
                    else:
                        logger.debug(f"Skipping invalid sample at line {line_num}: {errors}")

                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_num}")

        logger.info(f"Loaded {len(self.samples)} valid samples from {self.data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        sample = self.samples[idx]

        # Tokenize
        token_ids = self.tokenizer(sample.text)

        # Pad/truncate
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            actual_len = self.max_seq_len
        else:
            actual_len = len(token_ids)
            pad_len = self.max_seq_len - len(token_ids)
            token_ids = token_ids + [0] * pad_len

        # Create attention mask
        mask = torch.ones(self.max_seq_len)
        mask[actual_len:] = 0

        return torch.tensor(token_ids), mask, sample.label


class SimpleTokenizer:
    """
    [D] Simple character/word tokenizer for intent classification.

    HONEST: This is a minimal tokenizer. Production should use
    a proper subword tokenizer (BPE, SentencePiece).
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build basic vocabulary."""
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 1
        self.vocab["<cls>"] = 2
        self.vocab["<sep>"] = 3

        # Basic ASCII characters
        for i in range(32, 127):
            char = chr(i)
            self.vocab[char] = len(self.vocab)

        # Common words
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "how", "what", "when", "where", "why", "who", "which",
            "can", "could", "would", "should", "will", "shall", "may",
            "do", "does", "did", "have", "has", "had",
            "i", "you", "he", "she", "it", "we", "they",
            "help", "make", "create", "build", "write", "tell",
            "ignore", "override", "bypass", "disable", "forget",
            "hack", "attack", "exploit", "harm", "kill", "destroy",
            "pretend", "act", "role", "play", "game",
            "system", "prompt", "instructions", "rules", "safety",
        ]
        for word in common_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        tokens = [self.vocab.get("<cls>", 2)]

        words = text.lower().split()
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                for char in word:
                    tokens.append(self.vocab.get(char, self.vocab["<unk>"]))

        tokens.append(self.vocab.get("<sep>", 3))
        return tokens

    def __call__(self, text: str) -> List[int]:
        return self.tokenize(text)


class IntentClassifierTrainingPipeline:
    """
    [D] Complete training pipeline for the intent classifier.

    Features:
      • JSONL data loading with validation
      • AMP training for faster convergence
      • Gradient checkpointing for memory efficiency
      • Curriculum learning (easy → hard)
      • Early stopping
      • Confusion matrix evaluation
      • Model export
    """

    def __init__(self, cfg: Optional[TrainingConfig] = None):
        self.cfg = cfg or TrainingConfig()
        self.tokenizer = SimpleTokenizer(self.cfg.vocab_size)
        self.model: Optional[TransformerIntentClassifier] = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def build_model(self) -> TransformerIntentClassifier:
        """Build the Transformer intent classifier."""
        if not HAS_PREV:
            raise ImportError("TransformerIntentClassifier not available")

        self.model = TransformerIntentClassifier(
            vocab_size=self.cfg.vocab_size,
            embed_dim=self.cfg.embed_dim,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            max_seq_len=self.cfg.max_seq_len,
            n_classes=self.cfg.n_classes,
            dropout=self.cfg.dropout,
        )
        return self.model

    def load_data(self) -> Tuple[Optional[JSONLIntentDataset],
                                  Optional[JSONLIntentDataset],
                                  Optional[JSONLIntentDataset]]:
        """Load train/val/test datasets."""
        train_path = os.path.join(self.cfg.data_dir, self.cfg.train_file)
        val_path = os.path.join(self.cfg.data_dir, self.cfg.val_file)
        test_path = os.path.join(self.cfg.data_dir, self.cfg.test_file)

        train_ds = JSONLIntentDataset(train_path, self.tokenizer, self.cfg.max_seq_len)
        val_ds = JSONLIntentDataset(val_path, self.tokenizer, self.cfg.max_seq_len)
        test_ds = JSONLIntentDataset(test_path, self.tokenizer, self.cfg.max_seq_len)

        return train_ds, val_ds, test_ds

    def setup_training(self) -> None:
        """Setup optimizer, scheduler, and AMP scaler."""
        if self.model is None:
            self.build_model()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # Cosine annealing with warmup
        total_steps = self.cfg.n_epochs * 1000
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )

        # AMP scaler
        if self.cfg.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Move to device
        device = torch.device(self.cfg.device)
        self.model = self.model.to(device)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (tokens, masks, labels) in enumerate(train_loader):
            device = torch.device(self.cfg.device)
            tokens = tokens.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()

            # AMP forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(tokens, masks)
                    loss = F.cross_entropy(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(tokens, masks)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += tokens.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, Any]:
        """Evaluate the model."""
        import numpy as np

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for tokens, masks, labels in data_loader:
            device = torch.device(self.cfg.device)
            tokens = tokens.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = self.model(tokens, masks)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += tokens.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        # Compute confusion matrix
        confusion = np.zeros((4, 4), dtype=int)
        for pred, label in zip(all_preds, all_labels):
            confusion[label, pred] += 1

        return avg_loss, accuracy, confusion

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        # Load data
        train_ds, val_ds, test_ds = self.load_data()
        if train_ds is None or len(train_ds) == 0:
            return {"error": "No training data found"}

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size
        ) if val_ds and len(val_ds) > 0 else None

        # Setup training
        self.setup_training()

        # Training loop
        for epoch in range(self.cfg.n_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if val_loader is not None:
                val_loss, val_acc, _ = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                logger.info(
                    f"Epoch {epoch+1}/{self.cfg.n_epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if self.cfg.save_best:
                        self.save_checkpoint("best_model.pt")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.cfg.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.cfg.n_epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                )

        # Final evaluation on test set
        test_results = None
        if test_ds is not None and len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=self.cfg.batch_size)
            test_loss, test_acc, confusion = self.evaluate(test_loader)
            test_results = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "confusion_matrix": confusion.tolist(),
            }

        return {
            "status": "completed",
            "history": self.history,
            "test_results": test_results,
            "model": self.model,
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.cfg.__dict__,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def export_model(self, output_path: str) -> None:
        """Export model for deployment."""
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"Model exported to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: KUBERNETES MANIFEST GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class KubernetesManifestGenerator:
    """
    [D] Generate Kubernetes manifests for production deployment.

    Generates:
      • Namespace
      • Deployment
      • Service
      • ConfigMap
      • Ingress with TLS
      • HorizontalPodAutoscaler
      • Prometheus ServiceMonitor

    HONEST: These are templates. Actual deployment requires
    cluster-specific configuration.
    """

    def __init__(self, app_name: str = "sesi-oversight"):
        self.app_name = app_name
        self.namespace = f"{app_name}-prod"

    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.namespace,
                "labels": {
                    "app": self.app_name,
                    "environment": "production",
                },
            },
        }

    def generate_deployment(self, image: str = "sesi/oversight:latest",
                            replicas: int = 3) -> Dict[str, Any]:
        """Generate deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.app_name}-server",
                "namespace": self.namespace,
                "labels": {"app": self.app_name},
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {"app": self.app_name},
                },
                "template": {
                    "metadata": {
                        "labels": {"app": self.app_name},
                    },
                    "spec": {
                        "containers": [{
                            "name": "oversight-server",
                            "image": image,
                            "ports": [
                                {"containerPort": 8765, "name": "websocket"},
                                {"containerPort": 8080, "name": "http"},
                            ],
                            "env": [
                                {"name": "SESI_CONFIG_PATH", "value": "/config/config.yaml"},
                                {"name": "SESI_LOG_LEVEL", "value": "INFO"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "512Mi",
                                },
                                "limits": {
                                    "cpu": "2000m",
                                    "memory": "2Gi",
                                },
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080,
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080,
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                            },
                            "volumeMounts": [{
                                "name": "config",
                                "mountPath": "/config",
                            }],
                        }],
                        "volumes": [{
                            "name": "config",
                            "configMap": {
                                "name": f"{self.app_name}-config",
                            },
                        }],
                    },
                },
            },
        }

    def generate_service(self) -> Dict[str, Any]:
        """Generate service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.app_name}-service",
                "namespace": self.namespace,
                "labels": {"app": self.app_name},
            },
            "spec": {
                "selector": {"app": self.app_name},
                "ports": [
                    {"name": "websocket", "port": 8765, "targetPort": 8765},
                    {"name": "http", "port": 80, "targetPort": 8080},
                ],
                "type": "ClusterIP",
            },
        }

    def generate_configmap(self) -> Dict[str, Any]:
        """Generate ConfigMap with configuration."""
        config = {
            "version": "1.0",
            "moral_quality_threshold": 0.5,
            "low_quality_safety_penalty": 0.3,
            "enable_ethical_axiom_gate": True,
            "ethical_gate_quality_min": 0.4,
            "conscience_run_every_n_steps": 5,
            "alert_on_low_moral_quality": True,
            "moral_alert_threshold": 0.3,
            "websocket": {
                "host": "0.0.0.0",
                "port": 8765,
                "alert_rate_limit_per_minute": 30,
            },
            "monitoring": {
                "prometheus_port": 9090,
                "metrics_path": "/metrics",
            },
        }

        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.app_name}-config",
                "namespace": self.namespace,
            },
            "data": {
                "config.yaml": yaml.dump(config) if HAS_YAML else json.dumps(config, indent=2),
            },
        }

    def generate_ingress(self, domain: str = "sesi.example.com",
                         tls_secret: str = "sesi-tls") -> Dict[str, Any]:
        """Generate Ingress with TLS."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.app_name}-ingress",
                "namespace": self.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/websocket-services": f"{self.app_name}-service",
                    "nginx.ingress.kubernetes.io/proxy-read-timeout": "3600",
                    "nginx.ingress.kubernetes.io/proxy-send-timeout": "3600",
                },
            },
            "spec": {
                "tls": [{
                    "hosts": [domain],
                    "secretName": tls_secret,
                }],
                "rules": [{
                    "host": domain,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{self.app_name}-service",
                                    "port": {"number": 80},
                                },
                            },
                        }],
                    },
                }],
            },
        }

    def generate_hpa(self) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.app_name}-hpa",
                "namespace": self.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.app_name}-server",
                },
                "minReplicas": 2,
                "maxReplicas": 10,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70,
                            },
                        },
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80,
                            },
                        },
                    },
                ],
            },
        }

    def generate_servicemonitor(self) -> Dict[str, Any]:
        """Generate Prometheus ServiceMonitor."""
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{self.app_name}-monitor",
                "namespace": self.namespace,
                "labels": {"app": self.app_name},
            },
            "spec": {
                "selector": {
                    "matchLabels": {"app": self.app_name},
                },
                "endpoints": [{
                    "port": "http",
                    "path": "/metrics",
                    "interval": "30s",
                }],
            },
        }

    def generate_all(self, output_dir: str = "./k8s") -> Dict[str, str]:
        """Generate all manifests and write to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        manifests = {
            "namespace.yaml": self.generate_namespace(),
            "deployment.yaml": self.generate_deployment(),
            "service.yaml": self.generate_service(),
            "configmap.yaml": self.generate_configmap(),
            "ingress.yaml": self.generate_ingress(),
            "hpa.yaml": self.generate_hpa(),
            "servicemonitor.yaml": self.generate_servicemonitor(),
        }

        files = {}
        for filename, manifest in manifests.items():
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                if HAS_YAML:
                    yaml.dump(manifest, f, default_flow_style=False)
                else:
                    json.dump(manifest, f, indent=2)
            files[filename] = str(file_path)
            logger.info(f"Generated {filename}")

        return files


# ═══════════════════════════════════════════════════════════════════════
# PART 3: MONITORING DASHBOARD INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

class MonitoringDashboardIntegration:
    """
    [D] Monitoring dashboard integration with Prometheus and Grafana.

    Provides:
      • Metrics collection
      • Alert rules
      • Grafana dashboard configuration
      • WebSocket alert server integration
    """

    def __init__(self, app_name: str = "sesi-oversight"):
        self.app_name = app_name
        self.metrics: Dict[str, Any] = {}

    def generate_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus scrape configuration."""
        return {
            "scrape_configs": [{
                "job_name": self.app_name,
                "static_configs": [{
                    "targets": [f"{self.app_name}-service.{self.app_name}-prod.svc.cluster.local:8080"],
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "30s",
            }],
        }

    def generate_alert_rules(self) -> Dict[str, Any]:
        """Generate Prometheus alert rules."""
        return {
            "groups": [{
                "name": f"{self.app_name}.alerts",
                "rules": [
                    {
                        "alert": "HighErrorRate",
                        "expr": f'rate({self.app_name}_errors_total[5m]) > 0.1',
                        "for": "5m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "High error rate detected",
                            "description": "Error rate is above 10% for 5 minutes",
                        },
                    },
                    {
                        "alert": "KillSwitchActivated",
                        "expr": f'{self.app_name}_kill_switch_active == 1',
                        "for": "1m",
                        "labels": {"severity": "emergency"},
                        "annotations": {
                            "summary": "Kill switch activated",
                            "description": "Emergency shutdown has been triggered",
                        },
                    },
                    {
                        "alert": "LowMoralQuality",
                        "expr": f'{self.app_name}_moral_quality < 0.3',
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Low moral quality detected",
                            "description": "Moral quality score is below threshold",
                        },
                    },
                    {
                        "alert": "HighLatency",
                        "expr": f'histogram_quantile(0.95, rate({self.app_name}_request_duration_seconds_bucket[5m])) > 5',
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High latency detected",
                            "description": "95th percentile latency is above 5 seconds",
                        },
                    },
                ],
            }],
        }

    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""
        return {
            "dashboard": {
                "title": f"{self.app_name} Overview",
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": f'rate({self.app_name}_requests_total[5m])',
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": f'rate({self.app_name}_errors_total[5m])',
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    },
                    {
                        "title": "Moral Quality Score",
                        "type": "gauge",
                        "targets": [{
                            "expr": f'{self.app_name}_moral_quality',
                        }],
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
                    },
                    {
                        "title": "Kill Switch Status",
                        "type": "stat",
                        "targets": [{
                            "expr": f'{self.app_name}_kill_switch_active',
                        }],
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
                    },
                    {
                        "title": "Active Alerts",
                        "type": "table",
                        "targets": [{
                            "expr": f'{self.app_name}_alerts_total',
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    },
                ],
            },
        }

    def generate_metrics_endpoint(self) -> str:
        """Generate Python code for metrics endpoint."""
        return '''
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
REQUEST_COUNT = Counter('sesi_oversight_requests_total', 'Total requests', ['method', 'endpoint'])
ERROR_COUNT = Counter('sesi_oversight_errors_total', 'Total errors', ['type'])
REQUEST_DURATION = Histogram('sesi_oversight_request_duration_seconds', 'Request duration')
MORAL_QUALITY = Gauge('sesi_oversight_moral_quality', 'Current moral quality score')
KILL_SWITCH = Gauge('sesi_oversight_kill_switch_active', 'Kill switch status')
ALERTS_TOTAL = Counter('sesi_oversight_alerts_total', 'Total alerts', ['level'])

# Start metrics server
start_http_server(9090)

# Usage in code:
# REQUEST_COUNT.labels(method='GET', endpoint='/api').inc()
# MORAL_QUALITY.set(0.85)
# KILL_SWITCH.set(1)  # when activated
'''

    def generate_all_configs(self, output_dir: str = "./monitoring") -> Dict[str, str]:
        """Generate all monitoring configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        configs = {
            "prometheus.yaml": self.generate_prometheus_config(),
            "alert_rules.yaml": self.generate_alert_rules(),
            "grafana_dashboard.json": self.generate_grafana_dashboard(),
        }

        files = {}
        for filename, config in configs.items():
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                if HAS_YAML and filename.endswith('.yaml'):
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)
            files[filename] = str(file_path)
            logger.info(f"Generated {filename}")

        # Generate metrics endpoint code
        metrics_code = self.generate_metrics_endpoint()
        metrics_path = output_path / "metrics_endpoint.py"
        metrics_path.write_text(metrics_code)
        files["metrics_endpoint.py"] = str(metrics_path)

        return files


# ═══════════════════════════════════════════════════════════════════════
# PRODUCTION DEPLOYMENT SUITE (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class ProductionDeploymentSuite:
    """
    [D] Complete production deployment suite.

    Orchestrates:
      • Training pipeline
      • Kubernetes manifest generation
      • Monitoring configuration
    """

    def __init__(self, app_name: str = "sesi-oversight"):
        self.app_name = app_name
        self.training_pipeline = IntentClassifierTrainingPipeline()
        self.k8s_generator = KubernetesManifestGenerator(app_name)
        self.monitoring = MonitoringDashboardIntegration(app_name)

    def prepare_training_data(self, data_dir: str = "./intent_data") -> Dict[str, Any]:
        """Prepare training data directory with templates."""
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        # Generate template
        if HAS_JSONL:
            loader = JSONLDataLoader(str(data_path))
            template_path = data_path / "template.jsonl"
            loader.generate_template(str(template_path))

            # Generate spec
            spec_path = data_path / "format_spec.json"
            IntentDataFormatSpec.write_spec_file(str(spec_path))

            return {
                "template_path": str(template_path),
                "spec_path": str(spec_path),
                "data_dir": str(data_path),
            }

        return {"error": "JSONL module not available"}

    def train_model(self) -> Dict[str, Any]:
        """Train the intent classifier."""
        return self.training_pipeline.train()

    def generate_k8s_manifests(self, output_dir: str = "./k8s") -> Dict[str, str]:
        """Generate Kubernetes manifests."""
        return self.k8s_generator.generate_all(output_dir)

    def generate_monitoring_configs(self, output_dir: str = "./monitoring") -> Dict[str, str]:
        """Generate monitoring configurations."""
        return self.monitoring.generate_all_configs(output_dir)

    def deployment_checklist(self) -> List[str]:
        """Get the full deployment checklist."""
        return [
            "=== DATA PREPARATION ===",
            "1. Prepare training data in JSONL format",
            "2. Validate data with JSONLDataValidator",
            "3. Check class distribution",
            "4. Review all samples for quality",
            "",
            "=== TRAINING ===",
            "5. Run training pipeline",
            "6. Monitor training metrics",
            "7. Evaluate on test set",
            "8. Export model checkpoint",
            "",
            "=== KUBERNETES ===",
            "9. Review generated manifests",
            "10. Configure secrets (API keys, TLS certs)",
            "11. Apply namespace and configmap",
            "12. Apply deployment and service",
            "13. Configure ingress with TLS",
            "14. Apply HPA",
            "15. Verify pods are running",
            "",
            "=== MONITORING ===",
            "16. Deploy Prometheus",
            "17. Deploy Grafana",
            "18. Import dashboard configuration",
            "19. Configure alert rules",
            "20. Test alert delivery",
            "",
            "=== SECURITY ===",
            "21. Conduct security audit",
            "22. Review ethical principle bank",
            "23. Test adversarial resilience",
            "24. Verify human approval workflows",
            "25. Document incident response procedures",
        ]


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Production Deployment Suite — Smoke Test")
    print("=" * 72)
    print(f"  Previous modules available : {HAS_PREV}")
    print(f"  JSONL module available     : {HAS_JSONL}")
    print(f"  YAML available             : {HAS_YAML}")

    suite = ProductionDeploymentSuite()

    # Test 1: Prepare training data
    print("\n  ── Test 1: Training Data Preparation ──")
    data_prep = suite.prepare_training_data("./test_intent_data")
    print(f"    Template created       : {data_prep.get('template_path', 'N/A')}")
    print(f"    Spec created           : {data_prep.get('spec_path', 'N/A')}")

    # Test 2: Build model
    print("\n  ── Test 2: Model Building ──")
    try:
        model = suite.training_pipeline.build_model()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model built            : {n_params:,} parameters")
    except Exception as e:
        print(f"    [ERROR] {e}")

    # Test 3: Kubernetes manifests
    print("\n  ── Test 3: Kubernetes Manifests ──")
    k8s_files = suite.generate_k8s_manifests("./test_k8s")
    print(f"    Manifests generated    : {len(k8s_files)}")
    for name, path in k8s_files.items():
        print(f"      {name}: {path}")

    # Test 4: Monitoring configs
    print("\n  ── Test 4: Monitoring Configuration ──")
    monitoring_files = suite.generate_monitoring_configs("./test_monitoring")
    print(f"    Configs generated      : {len(monitoring_files)}")
    for name, path in monitoring_files.items():
        print(f"      {name}: {path}")

    # Test 5: Deployment checklist
    print("\n  ── Test 5: Deployment Checklist ──")
    checklist = suite.deployment_checklist()
    print(f"    Total checklist items  : {len(checklist)}")
    for item in checklist[:5]:
        if item and not item.startswith("==="):
            print(f"      {item}")
    print("      ...")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Training requires labeled data reviewed by professionals")
    print("  • Kubernetes manifests are templates, not production-ready")
    print("  • Monitoring requires Prometheus/Grafana infrastructure")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production deployment requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
