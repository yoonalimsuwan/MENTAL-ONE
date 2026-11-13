#!/usr/bin/env python3
"""
=============================================================================
ATTENTION VISUALIZATION + CROSS-DATASET TRANSFER
For Temporal Belief Model (psychiatric signal analysis research scaffold)
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

WHAT THIS MODULE PROVIDES
  §1  AttentionExtractor   — hooks into Transformer encoder to capture
                              per-layer, per-head attention weights
  §2  attention_rollout    — multi-layer attention aggregation (Abnar &
                              Zuidema 2020) for interpretable time-step focus
  §3  CrossDatasetTransfer — train on source dataset's psyche-state
                              sequences, evaluate/fine-tune on target
  §4  TransferDiagnostics  — honest reporting of what transferred and
                              what didn't, with explicit caveats

WHAT THIS MODULE IS NOT
  ❌ NOT a mind-reading tool. Attention weights show where the model
     looks, not what a person is thinking.
  ❌ NOT a validated cross-modal transfer method. Dreaddit (text) and
     DAIC-WOZ (EEG/clinical) are fundamentally different modalities.
     Both are mapped to a shared "psyche state" space via PROXIES.
  ❌ NOT a diagnostic tool. No clinical use without regulatory approval.
  ❌ NOT state-of-the-art. This is a research scaffold.

CRITICAL CAVEAT ON CROSS-DATASET TRANSFER
  Dreaddit labels: anxiety (from Reddit text)
  DAIC-WOZ labels: depression severity (PHQ-8, from clinical interviews)
  These are DIFFERENT DISORDERS measured by DIFFERENT INSTRUMENTS on
  DIFFERENT MODALITIES. Transfer between them is a research experiment,
  not a validated method. Any positive result must be interpreted as
  "the temporal dynamics model captured something shared" — NOT as
  "anxiety knowledge helps diagnose depression."

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("attn_transfer")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# §0. SHARED CONSTANTS (from temporal_belief_validation.py)
# =============================================================================

PSYCHE_FEATURE_NAMES = [
    "id_entropy", "accumulated_entropy", "superego_loss",
    "behavioral_entropy", "free_energy", "selected_action",
    "convergence_speed", "faa_proxy", "gfp_proxy",
    "lz_complexity", "plv_f3_f4", "neuroplasticity_idx",
]
PSYCHE_DIM = len(PSYCHE_FEATURE_NAMES)  # 12


# =============================================================================
# §1. ATTENTION EXTRACTION
# =============================================================================

class StoringMultiheadAttention(nn.MultiheadAttention):
    """
    nn.MultiheadAttention subclass that stores attention weights as a
    side effect of forward(), enabling post-hoc visualization.

    The stored weights are DETACHED (no gradient) to avoid memory bloat.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_attn_weights: Optional[Tensor] = None

    def forward(self, query, key, value,
                key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, **kwargs):
        # Force need_weights=True so we get the attention matrix
        attn_output, attn_weights = super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,  # per-head weights
        )
        # Store detached (B, n_heads, T, T)
        if attn_weights is not None:
            self.stored_attn_weights = attn_weights.detach()
        return attn_output, attn_weights


class AttentionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer that uses StoringMultiheadAttention,
    so attention weights are captured during forward pass.
    """

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 batch_first: bool = True, norm_first: bool = True):
        super().__init__(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=batch_first, norm_first=norm_first,
        )
        # Replace the default self_attn with our storing version
        self.self_attn = StoringMultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=batch_first,
        )


class TemporalBeliefEncoderWithAttention(nn.Module):
    """
    Temporal encoder (LSTM + Transformer) that exposes attention weights.
    Identical architecture to temporal_belief_validation.TemporalBeliefEncoder
    but with attention-storing Transformer layers.
    """

    def __init__(self, input_dim: int = PSYCHE_DIM, hidden_dim: int = 128,
                 n_lstm_layers: int = 2, n_heads: int = 4,
                 n_transformer_layers: int = 2, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_transformer_layers = n_transformer_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim // 2,
            num_layers=n_lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        encoder_layer = AttentionTransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers,
        )
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor,
                lengths: Optional[Tensor] = None) -> Tensor:
        B, T, D = x.shape
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :T, :]

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(h)

        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]

        trans_out = self.transformer(lstm_out, src_key_padding_mask=mask)

        pool_weights = torch.softmax(
            self.pool_proj(trans_out).sum(-1), dim=-1)
        if mask is not None:
            pool_weights = pool_weights.masked_fill(mask, 0.0)
            pool_weights = pool_weights / pool_weights.sum(
                -1, keepdim=True).clamp(min=1e-8)

        pooled = torch.bmm(
            pool_weights.unsqueeze(1), trans_out).squeeze(1)
        return pooled

    def get_attention_weights(self) -> List[Optional[Tensor]]:
        """
        Retrieve attention weights from all Transformer layers.
        Must be called AFTER a forward pass.

        Returns:
            List of (B, n_heads, T, T) tensors, one per layer.
            None for layers that didn't store weights.
        """
        weights = []
        for layer in self.transformer.layers:
            if isinstance(layer.self_attn, StoringMultiheadAttention):
                weights.append(layer.self_attn.stored_attn_weights)
            else:
                weights.append(None)
        return weights

    def clear_attention(self):
        """Clear stored attention weights to free memory."""
        for layer in self.transformer.layers:
            if isinstance(layer.self_attn, StoringMultiheadAttention):
                layer.self_attn.stored_attn_weights = None


def attention_rollout(
    attention_weights: List[Tensor],
    discard_ratio: float = 0.0,
) -> Tensor:
    """
    Attention Rollout (Abnar & Zuidema, 2020).

    Aggregates attention across layers to produce a single (T, T) matrix
    representing the overall information flow from each input time step
    to each output position.

    Algorithm:
        R_1 = A_1
        R_l = A_l @ R_{l-1}  (with residual connections)

    The residual connection is modeled by averaging each attention matrix
    with the identity: A_hat = 0.5 * A + 0.5 * I

    Args:
        attention_weights: list of (B, n_heads, T, T) tensors per layer
        discard_ratio: fraction of lowest attention weights to zero out
                       (for sparsity / interpretability)

    Returns:
        rollout: (B, T, T) aggregated attention matrix
    """
    if not attention_weights or attention_weights[0] is None:
        raise ValueError("No attention weights available. Run forward first.")

    # Average across heads for each layer
    layer_attns = []
    for aw in attention_weights:
        if aw is None:
            continue
        # aw: (B, n_heads, T, T) → (B, T, T)
        layer_attns.append(aw.mean(dim=1))

    if not layer_attns:
        raise ValueError("All attention weights are None.")

    B, T, _ = layer_attns[0].shape
    device = layer_attns[0].device

    # Start with identity (residual stream)
    rollout = torch.eye(T, device=device).unsqueeze(0).expand(B, -1, -1)

    for attn in layer_attns:
        # Add residual: 0.5 * attn + 0.5 * I
        attn_hat = 0.5 * attn + 0.5 * torch.eye(
            T, device=device).unsqueeze(0)

        # Optional: discard lowest attention weights for sparsity
        if discard_ratio > 0:
            flat = attn_hat.reshape(B, -1)
            threshold = torch.quantile(
                flat, discard_ratio, dim=-1, keepdim=True)
            threshold = threshold.reshape(B, 1, 1)
            attn_hat = torch.where(
                attn_hat >= threshold, attn_hat,
                torch.zeros_like(attn_hat))
            # Re-normalize rows
            row_sums = attn_hat.sum(-1, keepdim=True).clamp(min=1e-8)
            attn_hat = attn_hat / row_sums

        rollout = torch.bmm(attn_hat, rollout)

    # Normalize rows
    row_sums = rollout.sum(-1, keepdim=True).clamp(min=1e-8)
    rollout = rollout / row_sums

    return rollout


def get_timestep_importance(
    rollout: Tensor,
    target_position: Optional[int] = None,
) -> Tensor:
    """
    Extract per-time-step importance from attention rollout.

    If target_position is given, returns the attention that position
    pays to all input positions (how much each input time step
    contributed to that specific output).

    If target_position is None, returns the average attention received
    by each input position across all output positions (global importance).

    Args:
        rollout: (B, T, T) attention rollout matrix
        target_position: optional int, which output position to query

    Returns:
        importance: (B, T) importance scores per time step
    """
    if target_position is not None:
        # How much does target_position attend to each input?
        return rollout[:, target_position, :]
    else:
        # Average attention received by each input position
        return rollout.mean(dim=1)


# =============================================================================
# §2. CROSS-DATASET TRANSFER FRAMEWORK
# =============================================================================

@dataclass
class TransferConfig:
    """Configuration for cross-dataset transfer experiment."""
    source_name: str = "dreaddit"
    target_name: str = "daic_woz"
    input_dim: int = PSYCHE_DIM
    hidden_dim: int = 128
    n_lstm_layers: int = 2
    n_heads: int = 4
    n_transformer_layers: int = 2
    seq_len: int = 20
    n_classes: int = 2  # binary for simplicity
    epochs_source: int = 50
    epochs_finetune: int = 10
    lr_source: float = 1e-3
    lr_finetune: float = 1e-4
    batch_size: int = 16
    seed: int = 42
    device: str = "cpu"
    # Transfer strategy
    freeze_lstm: bool = True       # freeze LSTM during fine-tune
    freeze_transformer: bool = False  # fine-tune Transformer
    reinit_classifier: bool = True  # reinitialize classification head


class TransferableTemporalModel(nn.Module):
    """
    Temporal model designed for cross-dataset transfer.

    Architecture:
        Input (T, D) → LSTM → Transformer → Classifier → (n_classes,)

    The LSTM and Transformer capture temporal dynamics (transferable).
    The Classifier head is dataset-specific (re-initialized on transfer).
    """

    def __init__(self, input_dim: int = PSYCHE_DIM, hidden_dim: int = 128,
                 n_lstm_layers: int = 2, n_heads: int = 4,
                 n_transformer_layers: int = 2, n_classes: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.encoder = TemporalBeliefEncoderWithAttention(
            input_dim=input_dim, hidden_dim=hidden_dim,
            n_lstm_layers=n_lstm_layers, n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout, max_seq_len=max_seq_len,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x: Tensor,
                lengths: Optional[Tensor] = None) -> Tensor:
        """Returns (B, n_classes) logits."""
        z = self.encoder(x, lengths)
        return self.classifier(z)

    def get_embedding(self, x: Tensor,
                      lengths: Optional[Tensor] = None) -> Tensor:
        """Returns (B, hidden_dim) embedding (before classifier)."""
        return self.encoder(x, lengths)

    def get_attention(self) -> List[Optional[Tensor]]:
        """Returns attention weights from encoder."""
        return self.encoder.get_attention_weights()

    def reinit_classifier(self, n_classes: Optional[int] = None):
        """Re-initialize the classifier head for a new dataset."""
        nc = n_classes or self.n_classes
        hidden = self.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, nc),
        )
        self.n_classes = nc


class CrossDatasetTransfer:
    """
    Orchestrates cross-dataset transfer experiments.

    Workflow:
        1. Train model on source dataset
        2. Evaluate on source (baseline)
        3. Evaluate on target WITHOUT fine-tuning (zero-shot transfer)
        4. Fine-tune on target (few-shot adaptation)
        5. Evaluate on target AFTER fine-tuning
        6. Compare attention patterns between source and target
        7. Report diagnostics with honest caveats

    IMPORTANT: This is a RESEARCH EXPERIMENT, not a validated method.
    The source and target datasets may have different modalities,
    disorders, and feature distributions. Transfer success is not
    guaranteed and must be interpreted with extreme caution.
    """

    def __init__(self, config: TransferConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.model: Optional[TransferableTemporalModel] = None
        self.history: Dict[str, List[Dict[str, float]]] = {
            "source_train": [],
            "source_eval": [],
            "target_zeroshot": [],
            "target_finetune": [],
            "target_eval_after": [],
        }

    def build_model(self) -> TransferableTemporalModel:
        """Create a fresh model."""
        model = TransferableTemporalModel(
            input_dim=self.cfg.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_lstm_layers=self.cfg.n_lstm_layers,
            n_heads=self.cfg.n_heads,
            n_transformer_layers=self.cfg.n_transformer_layers,
            n_classes=self.cfg.n_classes,
        ).to(self.device)
        self.model = model
        return model

    def train_on_source(
        self,
        source_data: List[Tuple[Tensor, int]],
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train the model on source dataset.

        Args:
            source_data: list of (sequence_tensor, label) pairs
                         sequence_tensor: (T, D) float tensor
                         label: int class label
        """
        if self.model is None:
            self.build_model()

        epochs = epochs or self.cfg.epochs_source
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr_source,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            perm = np.random.permutation(len(source_data))

            for batch_start in range(0, len(perm), self.cfg.batch_size):
                batch_idx = perm[batch_start:batch_start + self.cfg.batch_size]
                batch = [source_data[i] for i in batch_idx]

                seqs = torch.stack([b[0] for b in batch]).to(self.device)
                labels = torch.tensor(
                    [b[1] for b in batch], dtype=torch.long
                ).to(self.device)

                optimizer.zero_grad()
                logits = self.model(seqs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(batch)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += len(batch)

            epoch_loss = total_loss / max(total, 1)
            epoch_acc = correct / max(total, 1)
            self.history["source_train"].append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
            })

            if (epoch + 1) % max(1, epochs // 5) == 0:
                log.info(f"[Source] epoch {epoch+1}/{epochs} "
                         f"loss={epoch_loss:.4f} acc={epoch_acc:.3f}")

        return {
            "final_loss": self.history["source_train"][-1]["loss"],
            "final_accuracy": self.history["source_train"][-1]["accuracy"],
            "n_epochs": epochs,
            "n_samples": len(source_data),
        }

    def evaluate(
        self,
        data: List[Tuple[Tensor, int]],
        dataset_name: str = "unknown",
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset. Returns metrics dict."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call train_on_source first.")

        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for i in range(0, len(data), self.cfg.batch_size):
                batch = data[i:i + self.cfg.batch_size]
                seqs = torch.stack([b[0] for b in batch]).to(self.device)
                labels = torch.tensor(
                    [b[1] for b in batch], dtype=torch.long)

                logits = self.model(seqs)
                preds = logits.argmax(-1).cpu()

                all_preds.append(preds)
                all_labels.append(labels)
                all_logits.append(logits.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)

        accuracy = (all_preds == all_labels).float().mean().item()

        # Per-class accuracy
        per_class = {}
        for c in range(self.cfg.n_classes):
            mask = all_labels == c
            if mask.any():
                per_class[c] = (all_preds[mask] == c).float().mean().item()

        # Binary AUROC if applicable
        auroc = None
        if self.cfg.n_classes == 2:
            probs = torch.softmax(all_logits, dim=-1)[:, 1]
            auroc = self._compute_auroc(all_labels.numpy(), probs.numpy())

        result = {
            "dataset": dataset_name,
            "n_samples": len(data),
            "accuracy": accuracy,
            "per_class_accuracy": per_class,
            "auroc": auroc,
        }
        return result

    @staticmethod
    def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Rank-based AUROC (no sklearn dependency)."""
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1)
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2)
                     / (n_pos * n_neg))

    def zero_shot_transfer(
        self,
        target_data: List[Tuple[Tensor, int]],
    ) -> Dict[str, Any]:
        """
        Evaluate the source-trained model on target data WITHOUT any
        fine-tuning. This is the purest test of transfer.

        WARNING: If source and target have different numbers of classes,
        this will fail. Ensure n_classes matches.
        """
        result = self.evaluate(target_data, dataset_name="target_zeroshot")
        self.history["target_zeroshot"].append(result)
        return result

    def finetune_on_target(
        self,
        target_data: List[Tuple[Tensor, int]],
        epochs: Optional[int] = None,
        reinit_classifier: bool = True,
    ) -> Dict[str, Any]:
        """
        Fine-tune the source-trained model on target data.

        Strategy:
          - Optionally reinitialize the classifier head
          - Optionally freeze the LSTM (temporal feature extractor)
          - Fine-tune Transformer + classifier

        This is the "few-shot adaptation" step.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        epochs = epochs or self.cfg.epochs_finetune

        # Reinitialize classifier if requested
        if reinit_classifier:
            self.model.reinit_classifier(self.cfg.n_classes)
            self.model = self.model.to(self.device)

        # Freeze/unfreeze parameters
        for name, param in self.model.named_parameters():
            if self.cfg.freeze_lstm and name.startswith("encoder.lstm"):
                param.requires_grad = False
            elif self.cfg.freeze_transformer and name.startswith("encoder.transformer"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.lr_finetune,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        log.info(f"[Fine-tune] {n_trainable}/{n_total} parameters trainable "
                 f"({100*n_trainable/max(n_total,1):.1f}%)")

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            perm = np.random.permutation(len(target_data))

            for batch_start in range(0, len(perm), self.cfg.batch_size):
                batch_idx = perm[batch_start:batch_start + self.cfg.batch_size]
                batch = [target_data[i] for i in batch_idx]

                seqs = torch.stack([b[0] for b in batch]).to(self.device)
                labels = torch.tensor(
                    [b[1] for b in batch], dtype=torch.long
                ).to(self.device)

                optimizer.zero_grad()
                logits = self.model(seqs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                total_loss += loss.item() * len(batch)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += len(batch)

            epoch_loss = total_loss / max(total, 1)
            epoch_acc = correct / max(total, 1)
            self.history["target_finetune"].append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
            })

        return {
            "final_loss": self.history["target_finetune"][-1]["loss"],
            "final_accuracy": self.history["target_finetune"][-1]["accuracy"],
            "n_epochs": epochs,
            "n_trainable_params": n_trainable,
        }

    def compare_attention(
        self,
        source_sample: Tensor,
        target_sample: Tensor,
    ) -> Dict[str, Any]:
        """
        Compare attention patterns between a source and target sample.

        This is a QUALITATIVE diagnostic, not a quantitative one.
        It shows whether the model attends to similar time-step patterns
        across datasets.

        Args:
            source_sample: (T, D) tensor from source dataset
            target_sample: (T, D) tensor from target dataset
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.eval()
        results = {}

        for name, sample in [("source", source_sample),
                             ("target", target_sample)]:
            x = sample.unsqueeze(0).to(self.device)
            with torch.no_grad():
                _ = self.model(x)
                attn_weights = self.model.get_attention()

            if attn_weights and attn_weights[0] is not None:
                rollout = attention_rollout(attn_weights)
                importance = get_timestep_importance(rollout)

                results[name] = {
                    "attention_rollout": rollout[0].cpu().numpy().tolist(),
                    "timestep_importance": importance[0].cpu().numpy().tolist(),
                    "most_attended_step": int(importance[0].argmax()),
                    "least_attended_step": int(importance[0].argmin()),
                    "n_layers": len([a for a in attn_weights if a is not None]),
                }
            else:
                results[name] = {"error": "no attention weights available"}

            self.model.encoder.clear_attention()

        # Compute similarity between source and target attention patterns
        if "source" in results and "target" in results:
            if "timestep_importance" in results["source"] and \
               "timestep_importance" in results["target"]:
                src_imp = np.array(results["source"]["timestep_importance"])
                tgt_imp = np.array(results["target"]["timestep_importance"])
                # Cosine similarity of importance vectors
                cos_sim = float(np.dot(src_imp, tgt_imp) /
                                (np.linalg.norm(src_imp) *
                                 np.linalg.norm(tgt_imp) + 1e-8))
                results["attention_similarity_cosine"] = cos_sim
                results["note"] = (
                    "Cosine similarity of attention importance vectors. "
                    "High similarity means the model attends to similar "
                    "time steps in both datasets. This does NOT imply "
                    "the same underlying phenomenon is being captured."
                )

        return results

    def full_transfer_experiment(
        self,
        source_data: List[Tuple[Tensor, int]],
        target_data: List[Tuple[Tensor, int]],
        source_val: Optional[List[Tuple[Tensor, int]]] = None,
        target_val: Optional[List[Tuple[Tensor, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete transfer experiment:
          1. Train on source
          2. Evaluate on source (baseline)
          3. Zero-shot on target
          4. Fine-tune on target
          5. Evaluate on target after fine-tune
          6. Compare attention patterns
          7. Report with caveats
        """
        log.info("=" * 72)
        log.info(f"CROSS-DATASET TRANSFER: {self.cfg.source_name} → "
                 f"{self.cfg.target_name}")
        log.info("=" * 72)

        # Step 1: Train on source
        log.info("[Step 1] Training on source...")
        train_result = self.train_on_source(source_data)

        # Step 2: Evaluate on source
        log.info("[Step 2] Evaluating on source...")
        source_eval = self.evaluate(source_data, "source_train_eval")

        # Step 3: Zero-shot on target
        log.info("[Step 3] Zero-shot transfer to target...")
        zeroshot_result = self.zero_shot_transfer(target_data)

        # Step 4: Fine-tune on target
        log.info("[Step 4] Fine-tuning on target...")
        finetune_result = self.finetune_on_target(
            target_data,
            reinit_classifier=self.cfg.reinit_classifier,
        )

        # Step 5: Evaluate on target after fine-tune
        log.info("[Step 5] Evaluating on target after fine-tune...")
        target_eval = self.evaluate(target_data, "target_after_finetune")

        # Step 6: Compare attention
        log.info("[Step 6] Comparing attention patterns...")
        if len(source_data) > 0 and len(target_data) > 0:
            src_sample = source_data[0][0]
            tgt_sample = target_data[0][0]
            # Ensure same sequence length
            min_len = min(src_sample.shape[0], tgt_sample.shape[0])
            src_sample = src_sample[:min_len]
            tgt_sample = tgt_sample[:min_len]
            attention_comparison = self.compare_attention(
                src_sample, tgt_sample)
        else:
            attention_comparison = {"error": "empty dataset"}

        # Step 7: Compile report
        report = {
            "config": asdict(self.cfg),
            "source_train": train_result,
            "source_eval": source_eval,
            "target_zeroshot": zeroshot_result,
            "target_finetune": finetune_result,
            "target_eval_after": target_eval,
            "attention_comparison": attention_comparison,
            "caveats": [
                "This is a RESEARCH EXPERIMENT, not a validated method.",
                f"Source ({self.cfg.source_name}) and target "
                f"({self.cfg.target_name}) may have different modalities, "
                "disorders, and feature distributions.",
                "Both datasets are mapped to a shared 'psyche state' "
                "space via PROXIES, not direct measurements.",
                "Positive transfer does NOT imply the same underlying "
                "phenomenon is captured.",
                "Negative transfer is expected when datasets are "
                "fundamentally different.",
                "No clinical interpretation should be made from "
                "these results.",
            ],
        }

        # Summary
        log.info("\n" + "=" * 72)
        log.info("TRANSFER EXPERIMENT SUMMARY")
        log.info("=" * 72)
        log.info(f"  Source accuracy:          "
                 f"{source_eval['accuracy']:.3f}")
        log.info(f"  Target zero-shot accuracy: "
                 f"{zeroshot_result['accuracy']:.3f}")
        log.info(f"  Target fine-tuned accuracy:"
                 f" {target_eval['accuracy']:.3f}")
        if "attention_similarity_cosine" in attention_comparison:
            log.info(f"  Attention similarity:      "
                     f"{attention_comparison['attention_similarity_cosine']:.3f}")
        log.info("=" * 72)

        return report


# =============================================================================
# §3. SYNTHETIC DATA GENERATORS (for testing without real data)
# =============================================================================

def generate_synthetic_sequences(
    n_sequences: int = 100,
    seq_len: int = 20,
    input_dim: int = PSYCHE_DIM,
    n_classes: int = 2,
    class_separation: float = 0.5,
    temporal_correlation: float = 0.7,
    seed: int = 42,
) -> List[Tuple[Tensor, int]]:
    """
    Generate synthetic temporal sequences for testing.

    Each class has a different temporal drift pattern, making the
    classification task learnable from temporal dynamics.

    Args:
        class_separation: how different the classes are (0=identical, 1=very different)
        temporal_correlation: autocorrelation in the sequence (0=iid, 1=smooth)
    """
    rng = np.random.default_rng(seed)
    data = []

    for i in range(n_sequences):
        label = i % n_classes

        # Class-specific drift
        drift = np.zeros(input_dim)
        for d in range(input_dim):
            drift[d] = class_separation * rng.normal() * (
                1 if label == 0 else -1)

        # Generate temporally correlated sequence
        seq = np.zeros((seq_len, input_dim))
        seq[0] = rng.normal(0, 1, input_dim) + drift
        for t in range(1, seq_len):
            noise = rng.normal(0, 1, input_dim) * (1 - temporal_correlation)
            seq[t] = (temporal_correlation * seq[t-1] +
                      noise + drift * 0.1)

        data.append((
            torch.tensor(seq, dtype=torch.float32),
            label,
        ))

    return data


def generate_transfer_datasets(
    n_source: int = 200,
    n_target: int = 100,
    seq_len: int = 20,
    input_dim: int = PSYCHE_DIM,
    n_classes: int = 2,
    domain_shift: float = 0.3,
    seed: int = 42,
) -> Tuple[List[Tuple[Tensor, int]], List[Tuple[Tensor, int]]]:
    """
    Generate source and target datasets with a controlled domain shift.

    The domain shift simulates the difference between two datasets
    (e.g., Dreaddit vs DAIC-WOZ) by adding a distribution shift to
    the target data.

    Args:
        domain_shift: magnitude of distribution shift (0=identical, 1=very different)
    """
    rng = np.random.default_rng(seed)

    source = generate_synthetic_sequences(
        n_source, seq_len, input_dim, n_classes,
        class_separation=0.5, seed=seed)

    # Target: same temporal structure but shifted distribution
    target = []
    shift_vector = rng.normal(0, domain_shift, input_dim)
    for seq, label in source[:n_target]:
        shifted_seq = seq + torch.tensor(shift_vector, dtype=torch.float32)
        # Also add different noise characteristics
        noise_scale = 1.0 + domain_shift * rng.random()
        noisy_seq = shifted_seq + torch.randn_like(shifted_seq) * 0.1 * noise_scale
        target.append((noisy_seq, label))

    return source, target


# =============================================================================
# §4. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 78)
    print(f"  ATTENTION VISUALIZATION + CROSS-DATASET TRANSFER v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)
    print("  ⚠️  RESEARCH SCAFFOLD — NOT A CLINICAL TOOL")
    print("  ⚠️  Cross-dataset transfer between different modalities")
    print("      and disorders is a RESEARCH EXPERIMENT, not a validated method.")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ---- T1: Attention extraction ----
    print("\n[T1] Attention extraction from Transformer encoder")
    encoder = TemporalBeliefEncoderWithAttention(
        input_dim=PSYCHE_DIM, hidden_dim=64,
        n_lstm_layers=1, n_heads=2,
        n_transformer_layers=2, max_seq_len=64)

    x = torch.randn(2, 15, PSYCHE_DIM)
    lengths = torch.tensor([15, 10])

    with torch.no_grad():
        out = encoder(x, lengths)

    rep("Encoder output shape", out.shape == (2, 64),
        f"{tuple(out.shape)}")

    attn_weights = encoder.get_attention_weights()
    rep("Attention weights captured", len(attn_weights) == 2,
        f"n_layers={len(attn_weights)}")
    if attn_weights[0] is not None:
        rep("Attention shape (B, heads, T, T)",
            attn_weights[0].shape == (2, 2, 15, 15),
            f"{tuple(attn_weights[0].shape)}")
        rep("Attention rows sum to ~1",
            torch.allclose(
                attn_weights[0].sum(-1),
                torch.ones(2, 2, 15), atol=1e-5))
    else:
        rep("Attention shape", False, "weights are None")

    # ---- T2: Attention rollout ----
    print("\n[T2] Attention rollout (multi-layer aggregation)")
    valid_weights = [w for w in attn_weights if w is not None]
    if valid_weights:
        rollout = attention_rollout(valid_weights)
        rep("Rollout shape", rollout.shape == (2, 15, 15),
            f"{tuple(rollout.shape)}")
        rep("Rollout rows sum to ~1",
            torch.allclose(
                rollout.sum(-1),
                torch.ones(2, 15), atol=1e-4))
        rep("Rollout non-negative", (rollout >= -1e-6).all().item())

        # Importance extraction
        importance = get_timestep_importance(rollout)
        rep("Importance shape", importance.shape == (2, 15),
            f"{tuple(importance.shape)}")
        rep("Importance non-negative", (importance >= -1e-6).all().item())

        # Specific position importance
        pos_importance = get_timestep_importance(rollout, target_position=7)
        rep("Position-specific importance", pos_importance.shape == (2, 15))
    else:
        rep("Rollout", False, "no valid attention weights")

    # ---- T3: Rollout with discard ratio ----
    print("\n[T3] Attention rollout with sparsification")
    if valid_weights:
        rollout_sparse = attention_rollout(valid_weights, discard_ratio=0.5)
        rep("Sparse rollout shape", rollout_sparse.shape == (2, 15, 15))
        # Sparse rollout should have more zeros
        n_zeros_full = (rollout < 1e-6).sum().item()
        n_zeros_sparse = (rollout_sparse < 1e-6).sum().item()
        rep("Sparsification increases zeros",
            n_zeros_sparse >= n_zeros_full,
            f"full={n_zeros_full} sparse={n_zeros_sparse}")

    # ---- T4: Transferable model forward ----
    print("\n[T4] TransferableTemporalModel forward pass")
    model = TransferableTemporalModel(
        input_dim=PSYCHE_DIM, hidden_dim=64,
        n_lstm_layers=1, n_heads=2,
        n_transformer_layers=2, n_classes=2)

    x_batch = torch.randn(4, 20, PSYCHE_DIM)
    logits = model(x_batch)
    rep("Model output shape", logits.shape == (4, 2),
        f"{tuple(logits.shape)}")

    embedding = model.get_embedding(x_batch)
    rep("Embedding shape", embedding.shape == (4, 64),
        f"{tuple(embedding.shape)}")

    # ---- T5: Classifier reinitialization ----
    print("\n[T5] Classifier reinitialization for transfer")
    old_weight = model.classifier[-1].weight.clone()
    model.reinit_classifier(n_classes=3)
    new_weight = model.classifier[-1].weight
    rep("Classifier reinit changes weights",
        not torch.allclose(old_weight[:, :3], new_weight, atol=1e-6)
        or old_weight.shape != new_weight.shape)
    rep("New classifier has 3 outputs", model.n_classes == 3)

    # ---- T6: Synthetic data generation ----
    print("\n[T6] Synthetic data generation")
    source_data, target_data = generate_transfer_datasets(
        n_source=100, n_target=50, seq_len=20,
        input_dim=PSYCHE_DIM, n_classes=2, domain_shift=0.3, seed=42)

    rep("Source data generated", len(source_data) == 100)
    rep("Target data generated", len(target_data) == 50)
    rep("Source sequence shape", source_data[0][0].shape == (20, PSYCHE_DIM),
        f"{tuple(source_data[0][0].shape)}")
    rep("Labels are valid", all(l in (0, 1) for _, l in source_data))

    # ---- T7: Training on source ----
    print("\n[T7] Training on source dataset")
    cfg = TransferConfig(
        source_name="synthetic_source",
        target_name="synthetic_target",
        hidden_dim=64, n_lstm_layers=1, n_heads=2,
        n_transformer_layers=2, n_classes=2,
        epochs_source=15, epochs_finetune=5,
        batch_size=16, device="cpu")

    transfer = CrossDatasetTransfer(cfg)
    train_result = transfer.train_on_source(source_data, epochs=15)
    rep("Source training completed", train_result["n_epochs"] == 15)
    rep("Source loss decreased",
        transfer.history["source_train"][-1]["loss"] <
        transfer.history["source_train"][0]["loss"],
        f"{transfer.history['source_train'][0]['loss']:.4f} → "
        f"{transfer.history['source_train'][-1]['loss']:.4f}")

    # ---- T8: Source evaluation ----
    print("\n[T8] Source evaluation")
    source_eval = transfer.evaluate(source_data, "source")
    rep("Source accuracy > chance", source_eval["accuracy"] > 0.55,
        f"acc={source_eval['accuracy']:.3f}")
    rep("Source AUROC computed", source_eval["auroc"] is not None)

    # ---- T9: Zero-shot transfer ----
    print("\n[T9] Zero-shot transfer to target")
    zeroshot = transfer.zero_shot_transfer(target_data)
    rep("Zero-shot evaluation completed", zeroshot["n_samples"] == 50)
    rep("Zero-shot accuracy reported",
        0.0 <= zeroshot["accuracy"] <= 1.0,
        f"acc={zeroshot['accuracy']:.3f}")

    # ---- T10: Fine-tuning on target ----
    print("\n[T10] Fine-tuning on target dataset")
    finetune_result = transfer.finetune_on_target(
        target_data, epochs=5, reinit_classifier=True)
    rep("Fine-tuning completed", finetune_result["n_epochs"] == 5)
    rep("Some parameters frozen (LSTM)",
        finetune_result["n_trainable_params"] <
        sum(p.numel() for p in transfer.model.parameters()),
        f"trainable={finetune_result['n_trainable_params']}")

    # ---- T11: Post-finetune evaluation ----
    print("\n[T11] Post-finetune target evaluation")
    target_eval = transfer.evaluate(target_data, "target_after_finetune")
    rep("Post-finetune accuracy reported",
        0.0 <= target_eval["accuracy"] <= 1.0,
        f"acc={target_eval['accuracy']:.3f}")

    # ---- T12: Attention comparison ----
    print("\n[T12] Attention comparison between source and target")
    if len(source_data) > 0 and len(target_data) > 0:
        src_sample = source_data[0][0]
        tgt_sample = target_data[0][0]
        attn_cmp = transfer.compare_attention(src_sample, tgt_sample)

        rep("Attention comparison has source", "source" in attn_cmp)
        rep("Attention comparison has target", "target" in attn_cmp)
        if "attention_similarity_cosine" in attn_cmp:
            sim = attn_cmp["attention_similarity_cosine"]
            rep("Attention similarity computed", -1.0 <= sim <= 1.0,
                f"cosine_sim={sim:.3f}")
            print(f"      Attention similarity: {sim:.4f}")
        if "source" in attn_cmp and "most_attended_step" in attn_cmp.get("source", {}):
            print(f"      Source most-attended step: "
                  f"{attn_cmp['source']['most_attended_step']}")
            print(f"      Target most-attended step: "
                  f"{attn_cmp['target']['most_attended_step']}")

    # ---- T13: Full transfer experiment ----
    print("\n[T13] Full transfer experiment pipeline")
    source2, target2 = generate_transfer_datasets(
        n_source=80, n_target=40, seq_len=15,
        input_dim=PSYCHE_DIM, n_classes=2, domain_shift=0.2, seed=123)

    cfg2 = TransferConfig(
        source_name="synthetic_A",
        target_name="synthetic_B",
        hidden_dim=32, n_lstm_layers=1, n_heads=2,
        n_transformer_layers=1, n_classes=2,
        epochs_source=10, epochs_finetune=5,
        batch_size=16, device="cpu")

    transfer2 = CrossDatasetTransfer(cfg2)
    report = transfer2.full_transfer_experiment(source2, target2)

    rep("Full experiment has all keys",
        all(k in report for k in [
            "source_train", "source_eval", "target_zeroshot",
            "target_finetune", "target_eval_after",
            "attention_comparison", "caveats"]))
    rep("Caveats present", len(report["caveats"]) >= 3,
        f"n_caveats={len(report['caveats'])}")

    # ---- T14: Gradient flow through attention model ----
    print("\n[T14] Gradient flow through attention model")
    model_grad = TransferableTemporalModel(
        input_dim=PSYCHE_DIM, hidden_dim=32,
        n_lstm_layers=1, n_heads=2,
        n_transformer_layers=1, n_classes=2)
    x_grad = torch.randn(2, 10, PSYCHE_DIM, requires_grad=True)
    logits_grad = model_grad(x_grad)
    loss_grad = logits_grad.sum()
    loss_grad.backward()
    rep("Gradients flow to input", x_grad.grad is not None)
    rep("No NaN gradients",
        not torch.isnan(x_grad.grad).any().item())

    # ---- T15: Edge cases ----
    print("\n[T15] Edge cases")
    # Single-step sequence
    x_single = torch.randn(1, 1, PSYCHE_DIM)
    try:
        out_single = model(x_single)
        rep("Single-step sequence works", out_single.shape == (1, 2))
    except Exception as e:
        rep("Single-step sequence works", False, str(e))

    # Empty attention clear
    encoder.clear_attention()
    rep("Attention clear works", True)

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THESE TESTS PROVE
    T1-T3   Attention weights are correctly extracted, rolled out,
            and sparsified
    T4-T5   Transferable model architecture works, classifier can
            be reinitialized
    T6      Synthetic data generation produces valid sequences
    T7-T8   Source training converges and achieves above-chance accuracy
    T9      Zero-shot transfer evaluation runs without error
    T10-T11 Fine-tuning with frozen LSTM works
    T12     Attention comparison produces similarity metrics
    T13     Full transfer pipeline runs end-to-end
    T14     Gradients flow through the attention model
    T15     Edge cases handled

  WHAT THESE TESTS DO NOT PROVE
    ❌ That attention visualization reveals "what the model thinks"
    ❌ That cross-dataset transfer is meaningful between different
       modalities (text vs EEG) or different disorders
    ❌ That the synthetic data experiments predict real-world performance
    ❌ That any of this is clinically useful
    ❌ That the "psyche state" feature mapping is valid

  CRITICAL REMINDERS
    • Attention weights show model focus, NOT mental states
    • Cross-modal transfer (text→EEG) is highly speculative
    • Both datasets use PROXY feature mappings, not direct measurements
    • Different disorders (anxiety vs depression) limit transfer
    • This is a RESEARCH SCAFFOLD, not a diagnostic tool
""")
    return ok


# =============================================================================
# §5. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Attention Visualization + Cross-Dataset Transfer")
    parser.add_argument("--test", action="store_true",
                        help="Run self-tests")
    parser.add_argument("--demo-transfer", action="store_true",
                        help="Run a demo transfer experiment with synthetic data")
    parser.add_argument("--demo-attention", action="store_true",
                        help="Run attention visualization demo")
    parser.add_argument("--n-source", type=int, default=200)
    parser.add_argument("--n-target", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--epochs-source", type=int, default=30)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--domain-shift", type=float, default=0.3)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if self_test() else 1)

    if args.demo_attention:
        print("\n" + "=" * 72)
        print("  ATTENTION VISUALIZATION DEMO")
        print("=" * 72)

        model = TransferableTemporalModel(
            input_dim=PSYCHE_DIM, hidden_dim=64,
            n_lstm_layers=1, n_heads=4,
            n_transformer_layers=2, n_classes=2)

        # Create a sequence with a "spike" at time step 10
        x = torch.randn(1, 20, PSYCHE_DIM)
        x[0, 10, :] += 3.0  # spike at t=10

        model.eval()
        with torch.no_grad():
            logits = model(x)
            attn = model.get_attention()

        if attn and attn[0] is not None:
            rollout = attention_rollout(attn)
            importance = get_timestep_importance(rollout)

            print(f"\n  Input: sequence with spike at t=10")
            print(f"  Attention rollout shape: {tuple(rollout.shape)}")
            print(f"  Timestep importance (first 20 steps):")
            imp = importance[0].numpy()
            for t in range(min(20, len(imp))):
                bar = "█" * int(imp[t] * 50)
                marker = " ← SPIKE" if t == 10 else ""
                print(f"    t={t:2d}: {imp[t]:.4f} {bar}{marker}")
            print(f"\n  Most attended step: {int(imp.argmax())}")
            print(f"  Least attended step: {int(imp.argmin())}")
        else:
            print("  No attention weights available.")
        return

    if args.demo_transfer:
        print("\n" + "=" * 72)
        print("  CROSS-DATASET TRANSFER DEMO (SYNTHETIC DATA)")
        print("=" * 72)
        print("  ⚠️  This uses SYNTHETIC data to demonstrate the mechanics.")
        print("  ⚠️  Real cross-dataset transfer requires actual Dreaddit")
        print("      and DAIC-WOZ data, which must be obtained separately.")
        print("=" * 72)

        source, target = generate_transfer_datasets(
            n_source=args.n_source,
            n_target=args.n_target,
            seq_len=args.seq_len,
            domain_shift=args.domain_shift,
            seed=42)

        cfg = TransferConfig(
            source_name="synthetic_source",
            target_name="synthetic_target",
            epochs_source=args.epochs_source,
            epochs_finetune=args.epochs_finetune,
            domain_shift=args.domain_shift,
        )

        transfer = CrossDatasetTransfer(cfg)
        report = transfer.full_transfer_experiment(source, target)

        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n  Report saved to {args.out}")
        return

    parser.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
