"""
=============================================================================
SESI AGI ADD-ON: DIFFERENTIABLE FCI × PYTORCH LIGHTNING × TENSORBOARD
Production AMP + Multi-GPU DDP + Unified AGI Adapter
=============================================================================

This file adds:
  1. Differentiable FCI Algorithm
  2. PyTorch Lightning integration
  3. TensorBoard logging

It can integrate with the uploaded AGI Unified Framework conceptually by
wrapping the original non-differentiable AGI modules as tools while keeping
the trainable cognitive/causal path fully differentiable.

Run:
  python sesi_agi_lightning_fci_tensorboard.py --devices 1
  python sesi_agi_lightning_fci_tensorboard.py --devices 2 --precision bf16-mixed

Dependencies:
  pip install torch lightning tensorboard numpy

Optional:
  pip install datasets webdataset
=============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    HAS_LIGHTNING = True
except ImportError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import TensorBoardLogger
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
        HAS_LIGHTNING = True
    except ImportError:
        HAS_LIGHTNING = False
        pl = None
        TensorBoardLogger = None
        ModelCheckpoint = None
        LearningRateMonitor = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sesi.agi.lightning_fci")


# =============================================================================
# 1. Utility
# =============================================================================

def safe_unit_scalar(x: float, default: float = 0.5) -> float:
    """Sanitize scalar into [0, 1]."""
    try:
        v = float(x)
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return max(0.0, min(1.0, v))


def hash_token(token: str, vocab_size: int) -> int:
    """Stable hashing for lightweight text embedding."""
    h = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % vocab_size


def ensure_2d_causal_data(x: torch.Tensor) -> torch.Tensor:
    """
    Convert batch data to shape (N, D).

    Accepted:
      (B, D)
      (B, T, D) -> flatten B*T
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        b, t, d = x.shape
        return x.reshape(b * t, d)
    raise ValueError(f"Expected causal data with ndim=2 or 3, got {x.shape}")


# =============================================================================
# 2. Differentiable Conditional Independence Proxy
# =============================================================================

class DifferentiableCITester(nn.Module):
    """
    Learned CI proxy.

    Input features for pair (Xi, Xj | Xcond):
      [xi, xj, cond_mean, cond_std, corr_ij]

    Output:
      dependence probability in [0, 1].
      0 ≈ independent, 1 ≈ dependent.

    Honest scope:
      This is not a formal CI test. It is a differentiable surrogate that can
      be trained from empirical correlation / residual signals.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        data: torch.Tensor,
        i: int,
        j: int,
        cond_set: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            data: (N, D)
        Returns:
            scalar dependence probability
        """
        cond_set = cond_set or []
        xi = data[:, i]
        xj = data[:, j]

        if cond_set:
            xc = data[:, cond_set]
            cond_mean = xc.mean(dim=-1)
            cond_std = xc.std(dim=-1, unbiased=False)
        else:
            cond_mean = torch.zeros_like(xi)
            cond_std = torch.zeros_like(xi)

        xi_c = xi - xi.mean()
        xj_c = xj - xj.mean()
        corr = (xi_c * xj_c).mean() / (
            xi_c.std(unbiased=False) * xj_c.std(unbiased=False) + 1e-6
        )
        corr = corr.expand_as(xi).clamp(-1.0, 1.0)

        features = torch.stack([xi, xj, cond_mean, cond_std, corr], dim=-1)
        return self.net(features).mean()


# =============================================================================
# 3. Differentiable FCI Algorithm
# =============================================================================

@dataclass
class DifferentiableFCIConfig:
    num_vars: int = 5
    alpha: float = 0.05
    max_cond_size: int = 2
    ci_samples: int = 8

    lambda_skeleton: float = 0.5
    lambda_orientation: float = 0.3
    lambda_acyclic: float = 5.0
    lambda_sparsity: float = 0.01
    lambda_ci: float = 0.5
    lambda_latent: float = 0.2

    edge_temperature: float = 8.0


class DifferentiableFCIAlgorithm(nn.Module):
    """
    Differentiable FCI-style causal discovery.

    Components:
      - Soft skeleton adjacency
      - Learned CI proxy
      - Collider orientation surrogate
      - NOTEARS acyclicity surrogate
      - Latent-confounding penalty proxy

    Honest scope:
      This is a continuous FCI-inspired relaxation. It does not replace
      discrete FCI/FCI++ PAG orientation rules for final causal claims.
    """

    def __init__(self, cfg: DifferentiableFCIConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.num_vars

        self.raw_W = nn.Parameter(torch.randn(d, d) * 0.01)
        self.ci_tester = DifferentiableCITester()

        self.register_buffer("eye", torch.eye(d))
        self._triples = self._build_triples(d)

    def _build_triples(self, d: int) -> List[Tuple[int, int, int]]:
        triples = []
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                for k in range(d):
                    if k == i or k == j:
                        continue
                    triples.append((i, j, k))
        return triples

    @property
    def W(self) -> torch.Tensor:
        """Directed weighted adjacency with zero diagonal."""
        return self.raw_W * (1.0 - self.eye)

    def edge_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.W * self.cfg.edge_temperature)

    def skeleton_prob(self) -> torch.Tensor:
        p = self.edge_prob()
        return torch.maximum(p, p.T)

    def empirical_dependence_targets(self, data: torch.Tensor) -> torch.Tensor:
        """
        Simple empirical dependence target from correlation.
        Detached target for training CI proxy.
        """
        x = data - data.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = x / std
        corr = (x.T @ x) / max(x.shape[0], 1)
        target = corr.abs().clamp(0.0, 1.0)
        return target.detach()

    def acyclicity_penalty(self) -> torch.Tensor:
        """NOTEARS h(W) = tr(exp(W ∘ W)) - d."""
        W = self.W
        d = W.shape[0]
        M = torch.linalg.matrix_exp(W * W)
        h = torch.trace(M) - d
        return h ** 2

    def collider_penalty(self) -> torch.Tensor:
        """
        Soft unshielded collider penalty.

        Encourages consistency:
          if i -> k and j -> k are strong,
          then i-j should be weak for an unshielded collider.
        """
        p = self.edge_prob()
        d = p.shape[0]
        penalty = p.new_tensor(0.0)
        count = 0

        for k in range(d):
            for i in range(d):
                if i == k:
                    continue
                for j in range(i + 1, d):
                    if j == k:
                        continue
                    collider_strength = p[i, k] * p[j, k]
                    ij_strength = torch.maximum(p[i, j], p[j, i])
                    penalty = penalty + collider_strength * ij_strength
                    count += 1

        return penalty / max(count, 1)

    def orientation_propagation_penalty(self) -> torch.Tensor:
        """
        Soft R2-R10-inspired propagation consistency.

        If i -> k and k -> j are strong, i -> j should not strongly oppose
        the induced direction. This is a surrogate, not exact PAG logic.
        """
        p = self.edge_prob()
        path = p @ p
        consistency = path * (1.0 - p)
        return consistency.mean()

    def latent_confounding_proxy(self, data: torch.Tensor) -> torch.Tensor:
        """
        Latent-confounding proxy.

        If two variables are empirically dependent but the learned skeleton
        edge is weak, a latent common cause may be plausible.

        This does NOT identify latents; it only adds a soft penalty.
        """
        target = self.empirical_dependence_targets(data)
        skeleton = self.skeleton_prob()
        residual_dependence = F.relu(target - skeleton)
        return residual_dependence.mean()

    def ci_loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        Train CI proxy and align skeleton with learned dependence.
        """
        d = data.shape[1]
        target = self.empirical_dependence_targets(data)
        skeleton = self.skeleton_prob()

        loss_ci = data.new_tensor(0.0)
        loss_skel = data.new_tensor(0.0)
        samples = min(self.cfg.ci_samples, max(1, len(self._triples)))

        for _ in range(samples):
            i, j, k = random.choice(self._triples)
            cond_set = [k] if random.random() < 0.7 else []
            pred_dep = self.ci_tester(data, i, j, cond_set)
            target_dep = target[i, j]

            loss_ci = loss_ci + F.binary_cross_entropy(
                pred_dep.clamp(1e-6, 1 - 1e-6),
                target_dep.clamp(1e-6, 1 - 1e-6),
            )

            edge_strength = skeleton[i, j]
            loss_skel = loss_skel + (edge_strength - pred_dep.detach()) ** 2

        return loss_ci / samples + loss_skel / samples

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        data = ensure_2d_causal_data(data)

        if data.shape[1] != self.cfg.num_vars:
            raise ValueError(
                f"Expected num_vars={self.cfg.num_vars}, got data.shape={data.shape}"
            )

        p = self.edge_prob()
        skeleton = self.skeleton_prob()

        fit = F.mse_loss(data @ self.W, data)
        acyclic = self.acyclicity_penalty()
        collider = self.collider_penalty()
        orientation = self.orientation_propagation_penalty()
        sparsity = self.W.abs().mean()
        ci = self.ci_loss(data)
        latent = self.latent_confounding_proxy(data)

        total = (
            fit
            + self.cfg.lambda_acyclic * acyclic
            + self.cfg.lambda_orientation * (collider + orientation)
            + self.cfg.lambda_sparsity * sparsity
            + self.cfg.lambda_ci * ci
            + self.cfg.lambda_latent * latent
        )

        return {
            "total_loss": total,
            "fit": fit,
            "acyclicity": acyclic,
            "collider": collider,
            "orientation": orientation,
            "sparsity": sparsity,
            "ci": ci,
            "latent": latent,
            "edge_prob": p,
            "skeleton": skeleton,
            "adjacency": self.W,
        }


# =============================================================================
# 4. Differentiable PC Algorithm
# =============================================================================

@dataclass
class DifferentiablePCConfig:
    num_vars: int = 5
    max_cond_size: int = 2
    samples: int = 8
    lambda_sparsity: float = 0.01
    lambda_symmetry: float = 0.1
    lambda_ci: float = 1.0


class DifferentiablePCAlgorithm(nn.Module):
    """
    Differentiable PC-style skeleton learner.

    Assumes causal sufficiency more strongly than FCI.
    """

    def __init__(self, cfg: DifferentiablePCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.num_vars
        self.skeleton_logits = nn.Parameter(torch.zeros(d, d))
        self.ci_tester = DifferentiableCITester()
        self.register_buffer("eye", torch.eye(d))

    def skeleton(self) -> torch.Tensor:
        s = torch.sigmoid(self.skeleton_logits)
        s = 0.5 * (s + s.T)
        return s * (1.0 - self.eye)

    def empirical_target(self, data: torch.Tensor) -> torch.Tensor:
        x = data - data.mean(dim=0, keepdim=True)
        x = x / x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        corr = (x.T @ x) / max(x.shape[0], 1)
        return corr.abs().clamp(0.0, 1.0).detach()

    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        data = ensure_2d_causal_data(data)
        d = data.shape[1]
        target = self.empirical_target(data)
        skeleton = self.skeleton()

        ci_loss = data.new_tensor(0.0)
        for _ in range(self.cfg.samples):
            i, j = random.sample(range(d), 2)
            others = [k for k in range(d) if k not in (i, j)]
            cond = random.sample(
                others,
                min(len(others), random.randint(0, self.cfg.max_cond_size)),
            )
            pred = self.ci_tester(data, i, j, cond)
            ci_loss = ci_loss + F.mse_loss(pred, target[i, j])

        ci_loss = ci_loss / self.cfg.samples
        sparsity = skeleton.abs().mean()
        symmetry = torch.norm(skeleton - skeleton.T, p="fro")
        fit = F.mse_loss(skeleton, target)

        total = (
            fit
            + self.cfg.lambda_ci * ci_loss
            + self.cfg.lambda_sparsity * sparsity
            + self.cfg.lambda_symmetry * symmetry
        )

        return {
            "total_loss": total,
            "skeleton": skeleton,
            "ci": ci_loss,
            "sparsity": sparsity,
            "symmetry": symmetry,
        }


# =============================================================================
# 5. Differentiable AGI Core Adapter
# =============================================================================

class DifferentiableAGIAdapter(nn.Module):
    """
    Differentiable adapter inspired by uploaded AGI Unified Framework.

    Original AGI modules are async/tool-like and non-differentiable.
    This adapter provides a trainable path:
      text -> hashed embedding -> memory/reasoning -> causal losses.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_vars: int = 5,
        vocab_size: int = 32768,
        use_pc: bool = True,
        use_fci: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_vars = num_vars
        self.vocab_size = vocab_size
        self.use_pc = use_pc
        self.use_fci = use_fci

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.memory_query = nn.Linear(embed_dim, embed_dim)
        self.memory_keys = nn.Parameter(torch.randn(4096, embed_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(4096, embed_dim) * 0.02)

        self.reasoning = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.curiosity = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.safety = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

        if use_pc:
            self.pc = DifferentiablePCAlgorithm(
                DifferentiablePCConfig(num_vars=num_vars)
            )
        else:
            self.pc = None

        if use_fci:
            self.fci = DifferentiableFCIAlgorithm(
                DifferentiableFCIConfig(num_vars=num_vars)
            )
        else:
            self.fci = None

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Lightweight differentiable text encoder.
        Token IDs are fixed hashes; embeddings are trainable.
        """
        encoded = []
        for text in texts:
            tokens = text.split() or ["<empty>"]
            ids = [hash_token(t, self.vocab_size) for t in tokens[:128]]
            token_tensor = torch.tensor(ids, dtype=torch.long, device=device)
            emb = self.token_embedding(token_tensor).mean(dim=0)
            encoded.append(emb)
        return torch.stack(encoded, dim=0)

    def memory_read(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.memory_query(x), dim=-1)
        k = F.normalize(self.memory_keys, dim=-1)
        attn = torch.softmax(q @ k.T * 10.0, dim=-1)
        return attn @ self.memory_values

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = batch["causal_data"].device
        text_emb = self.encode_text(batch["text"], device=device)

        mem = self.memory_read(text_emb)
        state = self.reasoning(torch.cat([text_emb, mem], dim=-1))

        outputs: Dict[str, torch.Tensor] = {
            "state": state,
            "curiosity": self.curiosity(state).squeeze(-1),
            "safety": self.safety(state).squeeze(-1),
        }

        causal_data = batch["causal_data"]

        if self.pc is not None:
            pc_out = self.pc(causal_data)
            outputs["pc_loss"] = pc_out["total_loss"]
            outputs["pc_skeleton"] = pc_out["skeleton"]

        if self.fci is not None:
            fci_out = self.fci(causal_data)
            outputs["fci_loss"] = fci_out["total_loss"]
            outputs["fci_acyclicity"] = fci_out["acyclicity"]
            outputs["fci_orientation"] = fci_out["orientation"]
            outputs["fci_edge_prob"] = fci_out["edge_prob"]
            outputs["fci_adjacency"] = fci_out["adjacency"]

        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = outputs["state"].pow(2).mean() * 0.01

        if "pc_loss" in outputs:
            loss = loss + 0.3 * outputs["pc_loss"]

        if "fci_loss" in outputs:
            loss = loss + outputs["fci_loss"]

        safety_loss = F.relu(0.9 - outputs["safety"]).mean()
        curiosity_reward = outputs["curiosity"].mean()

        return loss + 0.5 * safety_loss - 0.05 * curiosity_reward


# =============================================================================
# 6. HuggingFace / Mock Dataset
# =============================================================================

class MockAGIDataset(IterableDataset):
    """
    Infinite mock dataset for production smoke tests.
    Replace with HuggingFace or WebDataset in real training.
    """

    def __init__(self, batch_size: int, num_vars: int):
        self.batch_size = batch_size
        self.num_vars = num_vars

    def __iter__(self):
        while True:
            yield {
                "text": [
                    "Differentiable FCI causal graph learning",
                    "AGI unified framework with TensorBoard logging",
                    "PyTorch Lightning AMP DDP production training",
                    "Yoniso Manasikara causal analysis and proof validation",
                ][: self.batch_size],
                "causal_data": torch.randn(self.batch_size, self.num_vars),
            }


def create_dataloader(batch_size: int, num_vars: int, num_workers: int = 0):
    dataset = MockAGIDataset(batch_size=batch_size, num_vars=num_vars)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class HuggingFaceTextCausalLoader:
    """
    Optional HuggingFace streaming loader.

    This class is provided for real integration. By default, the training
    example uses MockAGIDataset to avoid requiring internet access.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        batch_size: int = 4,
        num_vars: int = 5,
        streaming: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.batch_size = batch_size
        self.num_vars = num_vars
        self.streaming = streaming
        self.rank = rank
        self.world_size = world_size
        self.dataset = None

        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
            if world_size > 1 and hasattr(ds, "shard"):
                ds = ds.shard(num_shards=world_size, index=rank)
            self.dataset = ds
        except Exception as e:
            logger.warning(f"HuggingFace load failed: {e}. Using mock data.")

    def __iter__(self):
        if self.dataset is None:
            yield from MockAGIDataset(self.batch_size, self.num_vars)
            return

        batch_texts = []
        for sample in self.dataset:
            batch_texts.append(str(sample.get(self.text_field, "")))
            if len(batch_texts) >= self.batch_size:
                yield {
                    "text": batch_texts,
                    "causal_data": torch.randn(len(batch_texts), self.num_vars),
                }
                batch_texts = []


# =============================================================================
# 7. PyTorch Lightning Module + TensorBoard Logging
# =============================================================================

if HAS_LIGHTNING:

    class SESILightningAGI(pl.LightningModule):
        """
        Lightning wrapper for Differentiable AGI Adapter.

        Logs:
          - train/loss
          - train/pc_loss
          - train/fci_loss
          - train/fci_acyclicity
          - train/fci_orientation
          - histograms of causal adjacency
        """

        def __init__(
            self,
            embed_dim: int = 256,
            num_vars: int = 5,
            lr: float = 1e-4,
            weight_decay: float = 1e-5,
            use_pc: bool = True,
            use_fci: bool = True,
            log_histograms_every_n_steps: int = 100,
        ):
            super().__init__()
            self.save_hyperparameters()

            self.model = DifferentiableAGIAdapter(
                embed_dim=embed_dim,
                num_vars=num_vars,
                use_pc=use_pc,
                use_fci=use_fci,
            )

            self.lr = lr
            self.weight_decay = weight_decay
            self.log_histograms_every_n_steps = log_histograms_every_n_steps

        def forward(self, batch):
            return self.model(batch)

        def training_step(self, batch, batch_idx):
            outputs = self(batch)
            loss = self.model.compute_loss(outputs)

            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

            if "pc_loss" in outputs:
                self.log("train/pc_loss", outputs["pc_loss"], on_step=True, on_epoch=True)

            if "fci_loss" in outputs:
                self.log("train/fci_loss", outputs["fci_loss"], on_step=True, on_epoch=True)
                self.log("train/fci_acyclicity", outputs["fci_acyclicity"], on_step=True)
                self.log("train/fci_orientation", outputs["fci_orientation"], on_step=True)

            self.log("train/safety", outputs["safety"].mean(), on_step=True)
            self.log("train/curiosity", outputs["curiosity"].mean(), on_step=True)

            if (
                self.logger is not None
                and batch_idx % self.log_histograms_every_n_steps == 0
            ):
                self._log_tensorboard_histograms(outputs)

            return loss

        def validation_step(self, batch, batch_idx):
            outputs = self(batch)
            loss = self.model.compute_loss(outputs)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.trainer.max_epochs),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        def on_train_start(self):
            if self.logger is not None:
                try:
                    self.logger.experiment.add_text(
                        "model/scope",
                        (
                            "Differentiable AGI adapter with PC/FCI continuous "
                            "surrogates. External proofs and exact PAG validation "
                            "remain separate verification steps."
                        ),
                        global_step=0,
                    )
                except Exception:
                    pass

        def _log_tensorboard_histograms(self, outputs: Dict[str, torch.Tensor]):
            try:
                exp = self.logger.experiment
                step = self.global_step

                if "fci_adjacency" in outputs:
                    exp.add_histogram(
                        "causal/fci_adjacency",
                        outputs["fci_adjacency"].detach().float().cpu(),
                        step,
                    )

                if "fci_edge_prob" in outputs:
                    exp.add_histogram(
                        "causal/fci_edge_prob",
                        outputs["fci_edge_prob"].detach().float().cpu(),
                        step,
                    )

                if "pc_skeleton" in outputs:
                    exp.add_histogram(
                        "causal/pc_skeleton",
                        outputs["pc_skeleton"].detach().float().cpu(),
                        step,
                    )
            except Exception as e:
                logger.debug(f"TensorBoard histogram logging skipped: {e}")

else:
    SESILightningAGI = None


# =============================================================================
# 8. Training Entrypoint
# =============================================================================

def build_trainer(args):
    if not HAS_LIGHTNING:
        raise RuntimeError(
            "PyTorch Lightning is not installed. Install with: pip install lightning"
        )

    logger_tb = TensorBoardLogger(
        save_dir=args.log_dir,
        name="sesi_agi_fci",
    )

    callbacks = []

    if ModelCheckpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                monitor="train/loss",
                mode="min",
                save_top_k=3,
                filename="sesi-agi-{epoch:02d}-{loss:.4f}",
            )
        )

    if LearningRateMonitor is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    strategy = "ddp" if args.devices and args.devices > 1 else "auto"

    trainer = pl.Trainer(
        accelerator="auto",
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger_tb,
        callbacks=callbacks,
        enable_checkpointing=True,
        benchmark=True,
    )

    return trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-vars", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)

    parser.add_argument("--log-dir", type=str, default="./lightning_logs")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--disable-pc", action="store_true")
    parser.add_argument("--disable-fci", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    if not HAS_LIGHTNING:
        raise RuntimeError("Please install Lightning: pip install lightning tensorboard")

    model = SESILightningAGI(
        embed_dim=args.embed_dim,
        num_vars=args.num_vars,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_pc=not args.disable_pc,
        use_fci=not args.disable_fci,
    )

    train_loader = create_dataloader(
        batch_size=args.batch_size,
        num_vars=args.num_vars,
        num_workers=args.num_workers,
    )

    trainer = build_trainer(args)

    logger.info("Starting Lightning training with TensorBoard logging")
    logger.info(
        f"devices={args.devices}, precision={args.precision}, "
        f"PC={not args.disable_pc}, FCI={not args.disable_fci}"
    )

    # Infinite IterableDataset: limit_train_batches prevents infinite epoch.
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        limit_train_batches=20,
    )


if __name__ == "__main__":
    main()
