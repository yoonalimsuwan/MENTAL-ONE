#!/usr/bin/env python3
"""
=============================================================================
PSY WORLD MODEL + ATTENTION VISUALIZATION + BENCHMARK HARNESS
Clinical Pipeline with Ethics Guard — Data Science as Assistant
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  ETHICS FIRST — READ BEFORE ANYTHING ELSE                       ║
║                                                                     ║
║  THIS IS AN ASSISTANT TOOL, NOT A DIAGNOSTIC TOOL.                  ║
║                                                                     ║
║  • Data science assists psychiatrists — it does NOT replace them.   ║
║  • All clinical decisions must be made by qualified psychiatrists.  ║
║  • Patient data requires explicit informed consent + IRB approval.  ║
║  • NO real clinical data exists in this file — synthetic only.      ║
║  • This code runs with SYNTHETIC data for demonstration.            ║
║                                                                     ║
║  REQUIRED BEFORE USING REAL DATA:                                   ║
║    1. IRB / Ethics Committee approval                               ║
║    2. Informed consent from every patient                           ║
║    3. Treating psychiatrist approval                                ║
║    4. Data de-identification (HIPAA / GDPR / Thai PDPA)            ║
║    5. Data use agreement                                            ║
║    6. Clinical validation against gold standard                     ║
║                                                                     ║
║  THE PSYCHIATRIST IS THE FINAL DECISION MAKER. ALWAYS.             ║
╚═══════════════════════════════════════════════════════════════════════╝

WHAT THIS MODULE PROVIDES
  §1  Ethics & Consent Framework — consent tracking + ethics guard
  §2  Attention Visualization — extract + visualize transformer attention
  §3  Clinical Data Pipeline — structure for real data (synthetic fallback)
  §4  Transformer + World Model — runs with synthetic data now
  §5  Benchmark Harness — metrics + report generation
  §6  Self-Test — verifies mechanics work

WHAT THIS MODULE IS NOT
  ❌ NOT a diagnostic tool
  ❌ NOT a mind-reading system
  ❌ NOT validated on real clinical data (none exists here)
  ❌ NOT approved for clinical use
  ❌ Data science is an ASSISTANT — the psychiatrist decides

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("psy_wm_bench")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

ETHICS_BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  DATA SCIENCE IS AN ASSISTANT — THE PSYCHIATRIST DECIDES             ║
║  NO REAL CLINICAL DATA EXISTS IN THIS FILE                           ║
║  ALL OUTPUTS ARE RESEARCH HYPOTHESES, NOT DIAGNOSES                  ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def ethics_guard():
    print(ETHICS_BANNER)

# =============================================================================
# §1. ETHICS & CONSENT FRAMEWORK
# =============================================================================

class ConsentStatus(Enum):
    NOT_OBTAINED = "not_obtained"
    PENDING = "pending"
    OBTAINED = "obtained"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

@dataclass
class ConsentRecord:
    """Tracks informed consent for a single patient."""
    patient_id: str
    consent_status: ConsentStatus = ConsentStatus.NOT_OBTAINED
    consent_date: Optional[str] = None
    irb_approval: str = "NOT_OBTAINED"
    treating_psychiatrist: Optional[str] = None
    psychiatrist_approval: bool = False
    data_use_purpose: str = "research_only"
    notes: str = ""

    def is_valid(self) -> bool:
        """Check if consent is valid for data use."""
        return (self.consent_status == ConsentStatus.OBTAINED
                and self.irb_approval == "APPROVED"
                and self.psychiatrist_approval)

class EthicsGuard:
    """
    Enforces ethical requirements before any data processing.
    This is a STRUCTURAL guard — it does not replace actual IRB review.
    """
    def __init__(self):
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.blocked_operations: List[str] = []

    def register_consent(self, record: ConsentRecord) -> bool:
        self.consent_records[record.patient_id] = record
        return record.is_valid()

    def can_process(self, patient_id: str) -> Tuple[bool, str]:
        """Check if data processing is allowed for this patient."""
        if patient_id not in self.consent_records:
            return False, "NO_CONSENT_RECORD"
        record = self.consent_records[patient_id]
        if not record.is_valid():
            return False, f"INVALID_CONSENT_{record.consent_status.value}"
        return True, "APPROVED"

    def require_psychiatrist_approval(self, patient_id: str,
                                       psychiatrist_id: str) -> bool:
        """Record that treating psychiatrist has approved data use."""
        if patient_id in self.consent_records:
            self.consent_records[patient_id].treating_psychiatrist = psychiatrist_id
            self.consent_records[patient_id].psychiatrist_approval = True
            return True
        return False

    def audit_log(self) -> List[Dict]:
        """Return audit trail of all consent checks."""
        return [{"patient_id": pid, "status": rec.consent_status.value,
                 "irb": rec.irb_approval,
                 "psychiatrist_approved": rec.psychiatrist_approval}
                for pid, rec in self.consent_records.items()]

# =============================================================================
# §2. ATTENTION VISUALIZATION
# =============================================================================

class AttentionExtractor:
    """
    Extracts attention weights from transformer layers.
    This shows WHERE the model attends, not WHAT it understands.
    """
    def __init__(self):
        self.attention_weights: Dict[str, Tensor] = {}

    def hook_layer(self, name: str, layer: nn.Module):
        """Register a forward hook to capture attention weights."""
        def hook_fn(module, input, output):
            # For nn.MultiheadAttention, output is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attention_weights[name] = attn_weights.detach()
        layer.register_forward_hook(hook_fn)

    def get_attention(self, name: str) -> Optional[Tensor]:
        return self.attention_weights.get(name)

    def get_all_attention(self) -> Dict[str, Tensor]:
        return dict(self.attention_weights)

    def clear(self):
        self.attention_weights.clear()

class AttentionVisualizer:
    """
    Produces attention visualization data.
    Returns structured data, NOT images (matplotlib optional).
    """
    @staticmethod
    def attention_matrix(attn: Tensor, head: int = 0) -> np.ndarray:
        """
        Extract attention matrix for a specific head.
        attn shape: (batch, heads, seq, seq) or (batch, seq, seq)
        """
        if attn.dim() == 4:
            return attn[0, head].cpu().numpy()
        elif attn.dim() == 3:
            return attn[0].cpu().numpy()
        return attn.cpu().numpy()

    @staticmethod
    def attention_entropy(attn: Tensor) -> float:
        """Compute entropy of attention distribution (lower = more focused)."""
        attn_np = attn.cpu().numpy().flatten()
        attn_np = attn_np[attn_np > 1e-10]
        return float(-np.sum(attn_np * np.log2(attn_np)))

    @staticmethod
    def attention_focus_score(attn: Tensor) -> float:
        """Fraction of attention on top-k positions."""
        attn_flat = attn.cpu().numpy().flatten()
        k = max(1, len(attn_flat) // 10)
        top_k = np.sort(attn_flat)[-k:]
        return float(top_k.sum() / max(attn_flat.sum(), 1e-10))

    @staticmethod
    def temporal_attention_profile(attn: Tensor,
                                    feature_names: Optional[List[str]] = None
                                    ) -> Dict[str, Any]:
        """
        Analyze which time steps get most attention.
        Useful for understanding what the world model focuses on.
        """
        if attn.dim() >= 3:
            # Average over batch and heads
            attn_avg = attn.mean(dim=tuple(range(attn.dim() - 2)))
        else:
            attn_avg = attn

        attn_np = attn_avg.cpu().numpy()
        seq_len = attn_np.shape[0]

        # Attention received by each position
        received = attn_np.mean(axis=0)
        # Attention given by each position
        given = attn_np.mean(axis=1)

        profile = {
            "seq_len": seq_len,
            "attention_received": received.tolist(),
            "attention_given": given.tolist(),
            "most_attended_position": int(np.argmax(received)),
            "least_attended_position": int(np.argmin(received)),
            "entropy": float(-np.sum(received * np.log2(received + 1e-10))),
        }

        if feature_names is not None and len(feature_names) == seq_len:
            profile["feature_names"] = feature_names
            profile["top_features"] = [
                feature_names[i] for i in np.argsort(received)[-5:][::-1]
            ]

        return profile

    @staticmethod
    def saliency_map(model: nn.Module, input_tensor: Tensor,
                      target_class: Optional[int] = None) -> Tensor:
        """
        Compute gradient-based saliency map.
        Shows which input features most influence the output.
        """
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = model(input_tensor)

        if output.dim() > 1:
            if target_class is None:
                target_class = output.argmax(dim=-1).item()
            score = output[0, target_class]
        else:
            score = output.sum()

        model.zero_grad()
        score.backward()

        saliency = input_tensor.grad.abs().max(dim=-1)[0]
        return saliency.detach()

# =============================================================================
# §3. CLINICAL DATA PIPELINE
# =============================================================================

class ClinicalDataSource(Enum):
    SYNTHETIC = "synthetic"
    EEG = "eeg"
    MRI = "mri"
    CLINICAL_NOTES = "clinical_notes"
    QUESTIONNAIRE = "questionnaire"

@dataclass
class ClinicalSample:
    """A single clinical data sample."""
    sample_id: str
    patient_id: str
    timestamp: str
    data_source: ClinicalDataSource
    features: Tensor
    label: Optional[int] = None
    label_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    consent_verified: bool = False

    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "data_source": self.data_source.value,
            "features_shape": list(self.features.shape),
            "label": self.label,
            "label_name": self.label_name,
            "consent_verified": self.consent_verified,
        }

class SyntheticDataGenerator:
    """
    Generates synthetic clinical data for testing.
    This is NOT real patient data — it simulates the structure.
    """
    DISORDERS = [
        "Healthy", "MDD", "Bipolar", "Schizophrenia",
        "PTSD", "Panic", "Anxiety", "OCD",
        "Conversion", "Dissociative", "Somatic"
    ]

    def __init__(self, n_features: int = 64, seq_len: int = 20,
                 n_classes: int = 11, seed: int = 42):
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

    def generate_sample(self, label: Optional[int] = None) -> ClinicalSample:
        """Generate one synthetic sample."""
        if label is None:
            label = int(self.rng.integers(0, self.n_classes))

        # Class-specific signal pattern
        base_signal = self.rng.normal(0, 1, (self.seq_len, self.n_features))
        if label > 0:
            # Add class-specific perturbation
            perturbation = self.rng.normal(0, 0.5, (self.seq_len, self.n_features))
            base_signal += label * 0.1 * perturbation

        features = torch.tensor(base_signal, dtype=torch.float32)

        return ClinicalSample(
            sample_id=str(uuid.uuid4())[:8],
            patient_id=f"synthetic_{label}",
            timestamp=datetime.now().isoformat(),
            data_source=ClinicalDataSource.SYNTHETIC,
            features=features,
            label=label,
            label_name=self.DISORDERS[label] if label < len(self.DISORDERS) else "Unknown",
            consent_verified=True,  # Synthetic data doesn't need consent
        )

    def generate_dataset(self, n_samples: int,
                          label_distribution: Optional[Dict[int, float]] = None
                          ) -> List[ClinicalSample]:
        """Generate a dataset of synthetic samples."""
        samples = []
        for i in range(n_samples):
            if label_distribution is not None:
                label = self.rng.choice(
                    list(label_distribution.keys()),
                    p=list(label_distribution.values())
                )
            else:
                label = None
            samples.append(self.generate_sample(label))
        return samples

class ClinicalDataLoader:
    """
    Loads clinical data with consent verification.

    IMPORTANT: This is a STRUCTURE for loading real data.
    Currently only synthetic data is available.
    Real data requires:
      1. IRB approval
      2. Patient consent
      3. Psychiatrist approval
      4. Data de-identification
    """
    def __init__(self, ethics_guard: EthicsGuard):
        self.ethics = ethics_guard
        self.samples: List[ClinicalSample] = []

    def add_synthetic_data(self, n_samples: int = 100,
                            n_features: int = 64,
                            seq_len: int = 20) -> int:
        """Add synthetic data for testing."""
        gen = SyntheticDataGenerator(n_features=n_features, seq_len=seq_len)
        new_samples = gen.generate_dataset(n_samples)
        self.samples.extend(new_samples)
        return len(new_samples)

    def add_real_data(self, samples: List[ClinicalSample]) -> Tuple[int, int]:
        """
        Add real clinical data.
        Returns (n_added, n_rejected) — rejected if consent not verified.
        """
        n_added = 0
        n_rejected = 0
        for sample in samples:
            can_process, reason = self.ethics.can_process(sample.patient_id)
            if can_process:
                sample.consent_verified = True
                self.samples.append(sample)
                n_added += 1
            else:
                log.warning(f"Rejected sample {sample.sample_id}: {reason}")
                n_rejected += 1
        return n_added, n_rejected

    def get_batch(self, batch_size: int = 32,
                   require_consent: bool = True) -> Tuple[Tensor, Tensor]:
        """Get a batch of samples."""
        valid_samples = []
        for s in self.samples:
            if require_consent and not s.consent_verified:
                continue
            valid_samples.append(s)

        if len(valid_samples) == 0:
            raise ValueError("No valid samples available")

        batch = valid_samples[:batch_size]
        features = torch.stack([s.features for s in batch])
        labels = torch.tensor([s.label for s in batch], dtype=torch.long)
        return features, labels

# =============================================================================
# §4. TRANSFORMER + WORLD MODEL
# =============================================================================

class MultiHeadAttentionWithViz(nn.Module):
    """Multi-head attention that stores attention weights for visualization."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.last_attention: Optional[Tensor] = None

    def forward(self, x: Tensor,
                return_attention: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        output, attn_weights = self.attn(x, x, x, need_weights=True)
        if return_attention:
            self.last_attention = attn_weights
        return output, attn_weights if return_attention else None

class ClinicalTransformerEncoder(nn.Module):
    """Transformer encoder for clinical time series."""
    def __init__(self, n_features: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3,
                 seq_len: int = 20, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )
        self.layers = nn.ModuleList([
            MultiHeadAttentionWithViz(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers * 2)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor,
                return_attention: bool = False) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: list of (batch, heads, seq, seq) per layer
        """
        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :h.shape[1], :]

        all_attention = []
        for i, layer in enumerate(self.layers):
            # Self-attention with residual
            attn_out, attn_weights = layer(h, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                all_attention.append(attn_weights)
            h = self.norms[2*i](h + self.dropout(attn_out))

            # FFN with residual
            ffn_out = self.ffn(h)
            h = self.norms[2*i+1](h + self.dropout(ffn_out))

        return h, all_attention

class PsycheWorldModel(nn.Module):
    """
    World model for predicting psychiatric state trajectories.
    This is a RESEARCH MODEL — not a diagnostic tool.
    """
    def __init__(self, n_features: int, n_classes: int = 11,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, seq_len: int = 20,
                 n_future_steps: int = 5):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.n_future_steps = n_future_steps

        # Encoder
        self.encoder = ClinicalTransformerEncoder(
            n_features, d_model, n_heads, n_layers, seq_len
        )

        # Dynamics model (GRU for temporal evolution)
        self.dynamics = nn.GRUCell(d_model, d_model)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, n_classes),
        )

        # Uncertainty estimator
        self.uncertainty = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features),
            nn.Softplus(),
        )

    def forward(self, x: Tensor,
                return_attention: bool = False
                ) -> Dict[str, Any]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            dict with predictions, classifications, attention, uncertainty
        """
        # Encode
        encoded, attention = self.encoder(x, return_attention=return_attention)

        # Pool over sequence
        pooled = encoded.mean(dim=1)

        # Classify
        logits = self.classifier(pooled)

        # Predict future
        z = pooled
        predictions = []
        uncertainties = []
        for _ in range(self.n_future_steps):
            z = self.dynamics(z, z)
            pred = self.decoder(z)
            unc = self.uncertainty(z)
            predictions.append(pred)
            uncertainties.append(unc)

        return {
            "logits": logits,
            "predictions": torch.stack(predictions, dim=1),
            "uncertainties": torch.stack(uncertainties, dim=1),
            "attention": attention if return_attention else None,
            "pooled": pooled,
        }

# =============================================================================
# §5. BENCHMARK HARNESS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    metric: str
    value: float
    n_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

class BenchmarkHarness:
    """
    Benchmark harness for evaluating the world model.
    Compares against baselines and reports metrics.
    """
    def __init__(self, ethics_guard: Optional[EthicsGuard] = None):
        self.ethics = ethics_guard or EthicsGuard()
        self.results: List[BenchmarkResult] = []

    def compute_accuracy(self, y_true: Tensor, y_pred: Tensor) -> float:
        return float((y_pred.argmax(dim=-1) == y_true).float().mean())

    def compute_f1(self, y_true: Tensor, y_pred: Tensor,
                    n_classes: int) -> float:
        """Macro F1 score."""
        pred_class = y_pred.argmax(dim=-1)
        f1s = []
        for c in range(n_classes):
            tp = ((pred_class == c) & (y_true == c)).sum().float()
            fp = ((pred_class == c) & (y_true != c)).sum().float()
            fn = ((pred_class != c) & (y_true == c)).sum().float()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1s.append(f1)
        return float(np.mean(f1s))

    def compute_auroc(self, y_true: Tensor, y_scores: Tensor) -> float:
        """Binary AUROC (one-vs-rest for multiclass)."""
        # Simplified: use max class score
        if y_scores.dim() == 1:
            scores = y_scores
        else:
            scores = y_scores.max(dim=-1).values
        # Rank-based AUROC
        order = torch.argsort(scores, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
        pos_mask = y_true > 0
        n_pos = pos_mask.sum().float()
        n_neg = (~pos_mask).sum().float()
        if n_pos == 0 or n_neg == 0:
            return 0.5
        auc = (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)

    def run_classification_benchmark(self, model: nn.Module,
                                      dataloader: ClinicalDataLoader,
                                      n_batches: int = 10,
                                      batch_size: int = 32) -> Dict[str, Any]:
        """Run classification benchmark."""
        model.eval()
        all_true = []
        all_pred = []

        with torch.no_grad():
            for i in range(n_batches):
                try:
                    features, labels = dataloader.get_batch(batch_size)
                except ValueError:
                    break
                output = model(features)
                all_true.append(labels)
                all_pred.append(output["logits"])

        if not all_true:
            return {"status": "NO_DATA"}

        y_true = torch.cat(all_true)
        y_pred = torch.cat(all_pred)

        accuracy = self.compute_accuracy(y_true, y_pred)
        f1 = self.compute_f1(y_true, y_pred, model.n_classes)
        auroc = self.compute_auroc(y_true, y_pred)

        results = {
            "status": "OK",
            "accuracy": accuracy,
            "f1_macro": f1,
            "auroc": auroc,
            "n_samples": len(y_true),
        }

        self.results.extend([
            BenchmarkResult("classification", "accuracy", accuracy, len(y_true)),
            BenchmarkResult("classification", "f1_macro", f1, len(y_true)),
            BenchmarkResult("classification", "auroc", auroc, len(y_true)),
        ])

        return results

    def run_prediction_benchmark(self, model: nn.Module,
                                  dataloader: ClinicalDataLoader,
                                  n_batches: int = 10,
                                  batch_size: int = 32) -> Dict[str, Any]:
        """Run prediction benchmark (world model)."""
        model.eval()
        total_mse = 0.0
        total_mae = 0.0
        n_samples = 0

        with torch.no_grad():
            for i in range(n_batches):
                try:
                    features, labels = dataloader.get_batch(batch_size)
                except ValueError:
                    break
                output = model(features)
                pred = output["predictions"]
                # Use last time step as target (simplified)
                target = features[:, -1:, :]
                # Compare first prediction step
                pred_first = pred[:, 0:1, :]
                mse = F.mse_loss(pred_first, target)
                mae = F.l1_loss(pred_first, target)
                total_mse += mse.item() * len(features)
                total_mae += mae.item() * len(features)
                n_samples += len(features)

        if n_samples == 0:
            return {"status": "NO_DATA"}

        results = {
            "status": "OK",
            "mse": total_mse / n_samples,
            "mae": total_mae / n_samples,
            "n_samples": n_samples,
        }

        self.results.extend([
            BenchmarkResult("prediction", "mse", total_mse / n_samples, n_samples),
            BenchmarkResult("prediction", "mae", total_mae / n_samples, n_samples),
        ])

        return results

    def run_attention_benchmark(self, model: nn.Module,
                                 dataloader: ClinicalDataLoader,
                                 n_samples: int = 5) -> Dict[str, Any]:
        """Analyze attention patterns."""
        model.eval()
        viz = AttentionVisualizer()

        try:
            features, labels = dataloader.get_batch(n_samples)
        except ValueError:
            return {"status": "NO_DATA"}

        with torch.no_grad():
            output = model(features, return_attention=True)

        if output["attention"] is None:
            return {"status": "NO_ATTENTION"}

        # Analyze attention from last layer
        last_attn = output["attention"][-1]
        attn_matrix = viz.attention_matrix(last_attn)
        entropy = viz.attention_entropy(last_attn)
        focus = viz.attention_focus_score(last_attn)
        profile = viz.temporal_attention_profile(last_attn)

        results = {
            "status": "OK",
            "attention_entropy": entropy,
            "attention_focus_score": focus,
            "most_attended_position": profile["most_attended_position"],
            "attention_matrix_shape": attn_matrix.shape,
            "n_layers": len(output["attention"]),
        }

        self.results.extend([
            BenchmarkResult("attention", "entropy", entropy, n_samples),
            BenchmarkResult("attention", "focus_score", focus, n_samples),
        ])

        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "n_results": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "disclaimer": (
                "ALL RESULTS ARE FROM SYNTHETIC DATA. "
                "NO REAL CLINICAL DATA WAS USED. "
                "THIS IS A RESEARCH TOOL, NOT A DIAGNOSTIC TOOL. "
                "PSYCHIATRIST APPROVAL REQUIRED FOR ANY CLINICAL USE."
            ),
        }
        return report

    def save_report(self, path: str) -> None:
        """Save report to JSON file."""
        report = self.generate_report()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Benchmark report saved to {path}")

# =============================================================================
# §6. TRAINING LOOP
# =============================================================================

def train_world_model(model: PsycheWorldModel,
                       dataloader: ClinicalDataLoader,
                       n_epochs: int = 10,
                       batch_size: int = 32,
                       lr: float = 1e-3,
                       device: str = "cpu") -> Dict[str, Any]:
    """
    Train the world model on synthetic data.
    Returns training history.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        # Get all data
        try:
            features, labels = dataloader.get_batch(
                batch_size=len(dataloader.samples)
            )
        except ValueError:
            break

        # Mini-batch training
        indices = torch.randperm(len(features))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            x = features[batch_idx].to(device)
            y = labels[batch_idx].to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output["logits"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(x)
            epoch_correct += (output["logits"].argmax(-1) == y).sum().item()
            epoch_total += len(x)
            n_batches += 1

        avg_loss = epoch_loss / max(epoch_total, 1)
        accuracy = epoch_correct / max(epoch_total, 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            log.info(f"Epoch {epoch+1}/{n_epochs} | "
                     f"loss={avg_loss:.4f} | acc={accuracy:.3f}")

    return history

# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test() -> bool:
    ethics_guard()
    print("=" * 78)
    print(f"  PSY WORLD MODEL + ATTENTION VIZ + BENCHMARK v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)
    print("  ⚠️  SYNTHETIC DATA ONLY — NO REAL CLINICAL DATA")
    print("  ⚠️  DATA SCIENCE IS AN ASSISTANT — PSYCHIATRIST DECIDES")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ── T1: Ethics Guard ──────────────────────────────────────
    print("\n[T1] Ethics & Consent Framework")
    ethics = EthicsGuard()
    consent = ConsentRecord(
        patient_id="test_001",
        consent_status=ConsentStatus.OBTAINED,
        irb_approval="APPROVED",
        psychiatrist_approval=True,
    )
    ethics.register_consent(consent)
    can_proc, reason = ethics.can_process("test_001")
    rep("Valid consent approved", can_proc, reason)

    consent2 = ConsentRecord(
        patient_id="test_002",
        consent_status=ConsentStatus.NOT_OBTAINED,
    )
    ethics.register_consent(consent2)
    can_proc2, reason2 = ethics.can_process("test_002")
    rep("No consent rejected", not can_proc2, reason2)

    # ── T2: Synthetic Data Generation ─────────────────────────
    print("\n[T2] Synthetic Data Generation")
    dataloader = ClinicalDataLoader(ethics)
    n_added = dataloader.add_synthetic_data(n_samples=100)
    rep("Synthetic data generated", n_added == 100, f"n={n_added}")

    features, labels = dataloader.get_batch(32)
    rep("Batch shape correct", features.shape == (32, 20, 64),
        f"{tuple(features.shape)}")

    # ── T3: Model Construction ────────────────────────────────
    print("\n[T3] World Model Construction")
    model = PsycheWorldModel(
        n_features=64, n_classes=11,
        d_model=64, n_heads=2, n_layers=2,
        seq_len=20, n_future_steps=3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    rep("Model constructs", n_params > 0, f"{n_params:,} params")

    # ── T4: Forward Pass ──────────────────────────────────────
    print("\n[T4] Forward Pass")
    x = torch.randn(4, 20, 64)
    with torch.no_grad():
        output = model(x, return_attention=True)
    rep("Logits shape", output["logits"].shape == (4, 11),
        f"{tuple(output['logits'].shape)}")
    rep("Predictions shape", output["predictions"].shape == (4, 3, 64),
        f"{tuple(output['predictions'].shape)}")
    rep("Attention captured", output["attention"] is not None,
        f"n_layers={len(output['attention'])}")

    # ── T5: Attention Visualization ───────────────────────────
    print("\n[T5] Attention Visualization")
    viz = AttentionVisualizer()
    attn = output["attention"][-1]
    attn_matrix = viz.attention_matrix(attn)
    rep("Attention matrix shape", attn_matrix.shape == (20, 20),
        f"{attn_matrix.shape}")

    entropy = viz.attention_entropy(attn)
    rep("Attention entropy finite", math.isfinite(entropy),
        f"H={entropy:.3f}")

    focus = viz.attention_focus_score(attn)
    rep("Attention focus in [0,1]", 0 <= focus <= 1, f"focus={focus:.3f}")

    profile = viz.temporal_attention_profile(attn)
    rep("Temporal profile computed", "most_attended_position" in profile)

    # ── T6: Saliency Map ──────────────────────────────────────
    print("\n[T6] Saliency Map")
    saliency = viz.saliency_map(model, x[:1])
    rep("Saliency shape", saliency.shape == (1, 20),
        f"{tuple(saliency.shape)}")
    rep("Saliency non-negative", (saliency >= 0).all().item())

    # ── T7: Training ──────────────────────────────────────────
    print("\n[T7] Training (5 epochs)")
    history = train_world_model(
        model, dataloader, n_epochs=5, batch_size=16
    )
    rep("Training completes", len(history["loss"]) == 5)
    rep("Loss decreases", history["loss"][-1] < history["loss"][0],
        f"{history['loss'][0]:.4f} → {history['loss'][-1]:.4f}")

    # ── T8: Benchmark Harness ─────────────────────────────────
    print("\n[T8] Benchmark Harness")
    harness = BenchmarkHarness(ethics)

    cls_result = harness.run_classification_benchmark(
        model, dataloader, n_batches=5
    )
    rep("Classification benchmark runs", cls_result["status"] == "OK")
    rep("Accuracy in [0,1]", 0 <= cls_result["accuracy"] <= 1,
        f"acc={cls_result['accuracy']:.3f}")

    pred_result = harness.run_prediction_benchmark(
        model, dataloader, n_batches=5
    )
    rep("Prediction benchmark runs", pred_result["status"] == "OK")
    rep("MSE finite", math.isfinite(pred_result["mse"]))

    attn_result = harness.run_attention_benchmark(
        model, dataloader, n_samples=5
    )
    rep("Attention benchmark runs", attn_result["status"] == "OK")

    # ── T9: Report Generation ─────────────────────────────────
    print("\n[T9] Report Generation")
    report = harness.generate_report()
    rep("Report generated", report["n_results"] > 0)
    rep("Disclaimer present", "SYNTHETIC" in report["disclaimer"])

    # ── T10: Consent Verification ─────────────────────────────
    print("\n[T10] Consent Verification")
    real_sample = ClinicalSample(
        sample_id="real_001",
        patient_id="test_002",  # No consent
        timestamp=datetime.now().isoformat(),
        data_source=ClinicalDataSource.EEG,
        features=torch.randn(20, 64),
        label=0,
    )
    n_added, n_rejected = dataloader.add_real_data([real_sample])
    rep("Unconsented data rejected", n_rejected == 1 and n_added == 0,
        f"added={n_added}, rejected={n_rejected}")

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THESE TESTS PROVE
    T1   Ethics guard approves valid consent, rejects invalid
    T2   Synthetic data generation works
    T3   World model constructs correctly
    T4   Forward pass produces correct shapes
    T5   Attention visualization extracts meaningful data
    T6   Saliency maps are computable
    T7   Training converges on synthetic data
    T8   Benchmark harness computes metrics
    T9   Reports include required disclaimers
    T10  Consent verification blocks unconsented data

  WHAT THESE TESTS DO NOT PROVE
    ❌ No real clinical data was used
    ❌ No diagnostic accuracy is claimed
    ❌ No psychiatrist validation was performed
    ❌ This is NOT a diagnostic tool

  REQUIRED FOR REAL CLINICAL USE
    1. IRB / Ethics Committee approval
    2. Informed consent from every patient
    3. Treating psychiatrist approval
    4. Data de-identification
    5. Clinical validation against gold standard
    6. Regulatory approval (FDA / Thai FDA / CE)
    7. Psychiatrist oversight at every step

  THE PSYCHIATRIST IS THE FINAL DECISION MAKER. ALWAYS.
""")
    return ok

# =============================================================================
# §8. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PSY World Model + Attention Viz + Benchmark"
    )
    parser.add_argument("--test", action="store_true",
                        help="Run self-tests")
    parser.add_argument("--train", action="store_true",
                        help="Train on synthetic data")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark suite")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out", default="benchmark_report.json")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if self_test() else 1)

    ethics_guard()

    if args.train or args.benchmark:
        # Setup
        ethics = EthicsGuard()
        dataloader = ClinicalDataLoader(ethics)
        dataloader.add_synthetic_data(n_samples=args.n_samples)

        model = PsycheWorldModel(
            n_features=64, n_classes=11,
            d_model=128, n_heads=4, n_layers=3,
            seq_len=20, n_future_steps=5,
        )

        if args.train:
            print("\nTraining world model on synthetic data...")
            history = train_world_model(
                model, dataloader,
                n_epochs=args.epochs, batch_size=32
            )
            print(f"Final loss: {history['loss'][-1]:.4f}")
            print(f"Final accuracy: {history['accuracy'][-1]:.3f}")

        if args.benchmark:
            print("\nRunning benchmark suite...")
            harness = BenchmarkHarness(ethics)
            cls_result = harness.run_classification_benchmark(model, dataloader)
            pred_result = harness.run_prediction_benchmark(model, dataloader)
            attn_result = harness.run_attention_benchmark(model, dataloader)

            print(f"\nClassification: acc={cls_result['accuracy']:.3f} "
                  f"f1={cls_result['f1_macro']:.3f}")
            print(f"Prediction: mse={pred_result['mse']:.4f} "
                  f"mae={pred_result['mae']:.4f}")
            print(f"Attention: entropy={attn_result['attention_entropy']:.3f}")

            harness.save_report(args.out)
            print(f"\nReport saved to {args.out}")
        return

    parser.print_help()
    print(f"\n{ATTRIBUTION}")

if __name__ == "__main__":
    main()
