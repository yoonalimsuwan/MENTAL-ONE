# =============================================================================
# MENTAL STRUCTURAL NEURAL OPERATOR (MSNO) - V3.0
# AI Surrogate Model Dedicated to the Enhanced MENTAL ONE Ecosystem
# =============================================================================
# Developer    : Yoon A Limsuwan
# Organization : MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026
#
# AI Development Contributors:
#   - Claude (Anthropic)  — architecture design, production hardening,
#                           loss functions, training pipeline, checkpointing
#   - GPT (OpenAI)        — algorithm consultation & code review
#   - Gemini (Google)     — supplementary research & validation strategy
#
# Description:
#   Production-grade Neural Operator that serves as the AI surrogate /
#   accelerator for the full MENTAL ONE Ecosystem (V3.0).
#
#   Operator topology (unchanged from V2):
#   1. BrainSpectralConv1D   -> EEG/MEG time-series  (mental_one,
#                               langevin_mental_bridge)
#   2. BrainGraphOperator    -> brain connectomes     (mental_one)
#   3. BrainSpatialConv3D    -> fMRI phase separation (structural_cahn_hilliard_3d)
#   4. PsycheSurrogateOp     -> Id/Ego/Superego DEQ   (psy_one_bridge_diff)
#
#   New in V3.0 (production additions):
#   - Fully implemented train_mental_surrogate() with multi-task loss
#   - validate_mental_surrogate() with per-task metrics
#   - Cosine-annealing LR scheduler + gradient clipping
#   - Mixed-precision training (torch.cuda.amp)
#   - Checkpoint save / resume with best-model tracking
#   - Structured logging (INFO / WARNING / ERROR)
#   - Graceful fallback for missing ecosystem modules
#   - MSNOTrainingConfig dataclass for clean experiment management
#   - Reproducible seeding
#   - __main__ demo with synthetic data
#
#   All topologies are modulated by the Structural Regime Field sigma(x),
#   providing O(1) inference to replace heavy BAOAB, CH-PDE, and DEQ loops.
# =============================================================================

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Optional ecosystem imports (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from one_core_mental import (
        SemanticStateContraction,
        soft_clamp,
        MENTAL_VERSION,
    )
    _HAS_ONE_CORE_MENTAL = True
except ImportError:
    _HAS_ONE_CORE_MENTAL = False
    MENTAL_VERSION = "unavailable"

    def soft_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:  # type: ignore[misc]
        c = (hi + lo) / 2.0
        s = (hi - lo) / 2.0 + 1e-8
        return c + s * torch.tanh((x - c) / s)

try:
    from mental_one import MentalONEEngine, SSCClassifier
    _HAS_MENTAL_ONE = True
except ImportError:
    _HAS_MENTAL_ONE = False

try:
    from structural_cahn_hilliard_3d import StructuralCahnHilliard3D, CahnHilliardConfig
    _HAS_CH3D = True
except ImportError:
    _HAS_CH3D = False

try:
    from psy_one_bridge_diff import PSYONEBridge, PsycheConfig
    _HAS_PSY = True
except ImportError:
    _HAS_PSY = False

try:
    from langevin_mental_bridge import LangevinMentalEvolution
    _HAS_LANGEVIN = True
except ImportError:
    _HAS_LANGEVIN = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [MSNO-V3]  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MSNO_V3")

MSNO_VERSION: str = "3.0.0"


# =============================================================================
# 0.  Configuration dataclass
# =============================================================================
@dataclass
class MSNOTrainingConfig:
    """
    Centralised experiment configuration for MSNOTrainer.

    All hyperparameters live here so a single object fully reproduces a run.
    """
    # ── model architecture ─────────────────────────────────────────────────
    eeg_channels: int    = 19
    latent_dim:   int    = 64
    modes_1d:     int    = 32
    modes_3d:     int    = 8
    action_dim:   int    = 10

    # ── training schedule ──────────────────────────────────────────────────
    epochs:       int    = 50
    batch_size:   int    = 16
    lr:           float  = 1e-4
    weight_decay: float  = 1e-5
    grad_clip:    float  = 1.0          # max-norm gradient clipping
    warmup_epochs: int   = 5            # linear warmup before cosine decay

    # ── multi-task loss weights ────────────────────────────────────────────
    lambda_eeg:   float  = 1.0
    lambda_ch3d:  float  = 1.0
    lambda_ego:   float  = 0.5

    # ── infrastructure ─────────────────────────────────────────────────────
    device:       str    = "cuda"       # "cuda" | "cpu" | "mps"
    use_amp:      bool   = True         # mixed-precision (CUDA only)
    num_workers:  int    = 4
    seed:         int    = 42
    checkpoint_dir: str  = "./msno_checkpoints"
    log_every:    int    = 10           # batches between log lines
    val_every:    int    = 5            # epochs between validation

    # ── surrogate rollout depth ────────────────────────────────────────────
    langevin_teacher_steps: int = 100   # BAOAB steps for ground-truth EEG
    ch3d_teacher_steps:     int = 100   # CH3D steps for ground-truth phase
    psyche_n_samples:       int = 32    # samples per Ego optimisation call


# =============================================================================
# 1.  Operator building blocks  (V2 → V3: no changes to forward passes)
# =============================================================================

class BrainSpectralConv1D(nn.Module):
    """
    1D Fourier Neural Operator layer modulated by the SSC stress field σ.

    Input  x : (B, C, T)  — EEG/MEG signal in channel × time format.
    Input  σ : (B, 1, 1)  — scalar structural stress, broadcast over (C, T).
    Output   : (B, C, T)  — transformed feature map.

    The σ-gate multiplies the spectral + local paths before GELU, so
    near-critical regimes (σ ≈ σ_target) are allowed full bandwidth while
    sub-critical or super-critical states are softly attenuated.
    """

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weights    = nn.Parameter(scale * torch.rand(width, width, modes, dtype=torch.cfloat))
        self.mlp        = nn.Conv1d(width, width, 1)
        self.sigma_gate = nn.Conv1d(1, width, 1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_ft   = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        m      = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.weights[:, :, :m])
        x_spec = torch.fft.irfft(out_ft, n=x.shape[-1], norm="ortho")
        x_loc  = self.mlp(x)
        s_mod  = torch.sigmoid(self.sigma_gate(sigma))
        return F.gelu(s_mod * (x_spec + x_loc))


class BrainGraphOperator(nn.Module):
    """
    Message-passing layer for brain connectome graphs.

    x          : (B, N_nodes, hidden_dim)
    edge_index : (2, E)  — COO sparse format, long tensor
    sigma      : (B, 1)  — per-graph structural stress scalar

    Aggregation is sum-based; σ scales the aggregated neighbourhood signal.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.msg_mlp  = nn.Sequential(nn.Linear(hidden_dim * 2, 128), nn.GELU(), nn.Linear(128, hidden_dim))
        self.upd_mlp  = nn.Sequential(nn.Linear(hidden_dim * 2, 128), nn.GELU(), nn.Linear(128, hidden_dim))
        self.sigma_proj = nn.Linear(1, hidden_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg_in   = torch.cat([x[:, src, :], x[:, dst, :]], dim=-1)
        messages = self.msg_mlp(msg_in)
        aggr     = torch.zeros_like(x)
        idx      = dst.unsqueeze(0).unsqueeze(-1).expand(-1, -1, messages.size(-1))
        aggr.scatter_add_(1, idx, messages)
        s_mod = torch.sigmoid(self.sigma_proj(sigma))
        return x + self.upd_mlp(torch.cat([x, s_mod * aggr], dim=-1))


class BrainSpatialConv3D(nn.Module):
    """
    3-D Fourier Neural Operator layer; surrogate for StructuralCahnHilliard3D.

    u     : (B, width, X, Y, Z)  — lifted phase-field features
    sigma : (B, 1, 1, 1, 1)      — broadcast stress scalar
    """

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale    = 1.0 / (width * width)
        shape    = (width, width, modes, modes, modes)
        self.w1  = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w2  = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w3  = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w4  = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.mlp        = nn.Conv3d(width, width, 1)
        self.sigma_gate = nn.Conv3d(1, width, 1)

    @staticmethod
    def _cmul(inp: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", inp, W)

    def forward(self, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        u_ft   = torch.fft.rfftn(u, dim=[-3, -2, -1], norm="ortho")
        out_ft = torch.zeros_like(u_ft)
        m      = self.modes
        out_ft[:, :,  :m,  :m, :m] = self._cmul(u_ft[:, :,  :m,  :m, :m], self.w1)
        out_ft[:, :, -m:,  :m, :m] = self._cmul(u_ft[:, :, -m:,  :m, :m], self.w2)
        out_ft[:, :,  :m, -m:, :m] = self._cmul(u_ft[:, :,  :m, -m:, :m], self.w3)
        out_ft[:, :, -m:, -m:, :m] = self._cmul(u_ft[:, :, -m:, -m:, :m], self.w4)
        u_spec = torch.fft.irfftn(out_ft, s=(u.shape[-3], u.shape[-2], u.shape[-1]), norm="ortho")
        u_loc  = self.mlp(u)
        s_mod  = torch.sigmoid(self.sigma_gate(sigma))
        return F.gelu(s_mod * (u_spec + u_loc))


class PsycheSurrogateOperator(nn.Module):
    """
    O(1) surrogate for the Id/Ego/Superego DEQ / Anderson-mixing loop in
    psy_one_bridge_diff.  Learns the free-energy landscape directly as a
    feed-forward mapping.

    Input  : [id_proposal ‖ superego_norm ‖ sigma]  → (B, action_dim × 2 + 1)
    Output : soft-max action distribution             → (B, action_dim)
    """

    def __init__(self, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        in_dim  = action_dim * 2 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self,
                id_proposal:   torch.Tensor,
                superego_norm: torch.Tensor,
                sigma:         torch.Tensor) -> torch.Tensor:
        inp    = torch.cat([id_proposal, superego_norm, sigma], dim=-1)
        logits = self.net(inp)
        return F.softmax(logits, dim=-1)


# =============================================================================
# 2.  MSNO Orchestrator
# =============================================================================

class MentalStructuralNeuralOperator(nn.Module):
    """
    Master surrogate / AI trainer for the MENTAL ONE ecosystem.

    Routing logic
    -------------
    predict_eeg_trajectory   → replaces 1-D BAOAB Langevin (N steps → O(1))
    predict_graph_state      → replaces connectome message passing (N iters → O(1))
    predict_spatial_phase    → replaces 4th-order CH3D PDE  (N steps → O(1))
    optimize_ego             → replaces DEQ/Anderson mixing  (N iters → O(1))

    All methods accept a ``sigma`` tensor (structural stress from the SSC
    filter) that controls the strength of spectral / local mixing, matching
    the physical intuition that near-critical dynamics need the widest
    frequency range.
    """

    def __init__(self,
                 eeg_channels: int = 19,
                 latent_dim:   int = 64,
                 modes_1d:     int = 32,
                 modes_3d:     int = 8,
                 action_dim:   int = 10) -> None:
        super().__init__()
        self.latent_dim  = latent_dim
        self.action_dim  = action_dim

        # ── 1D EEG/MEG branch ─────────────────────────────────────────────
        self.lift_1d  = nn.Conv1d(eeg_channels, latent_dim, 1)
        self.seq_op   = nn.ModuleList([BrainSpectralConv1D(latent_dim, modes_1d) for _ in range(4)])
        self.proj_1d  = nn.Conv1d(latent_dim, eeg_channels, 1)

        # ── Graph branch ──────────────────────────────────────────────────
        self.lift_graph  = nn.Linear(1, latent_dim)
        self.graph_op    = nn.ModuleList([BrainGraphOperator(latent_dim) for _ in range(3)])
        self.proj_graph  = nn.Linear(latent_dim, 1)

        # ── 3D CH phase-field branch ──────────────────────────────────────
        self.lift_3d    = nn.Conv3d(1, latent_dim, 1)
        self.spatial_op = nn.ModuleList([BrainSpatialConv3D(latent_dim, modes_3d) for _ in range(4)])
        self.proj_3d    = nn.Conv3d(latent_dim, 1, 1)

        # ── Psyche triad branch ───────────────────────────────────────────
        self.psyche_op = PsycheSurrogateOperator(action_dim, hidden_dim=128)

        logger.info(
            f"MentalStructuralNeuralOperator V{MSNO_VERSION} initialised  "
            f"| latent_dim={latent_dim}  modes_1d={modes_1d}  modes_3d={modes_3d}  "
            f"action_dim={action_dim}  "
            f"| one_core_mental={'✓' if _HAS_ONE_CORE_MENTAL else '✗ (fallback)'}"
        )

    # ------------------------------------------------------------------
    # Prediction heads
    # ------------------------------------------------------------------

    def predict_eeg_trajectory(self,
                                eeg_state: torch.Tensor,
                                sigma:     torch.Tensor) -> torch.Tensor:
        """
        O(1) surrogate for BAOAB Langevin integration.

        Args:
            eeg_state : (B, C, T)   — initial EEG/MEG state
            sigma     : (B, 1, 1)   — structural stress scalar (broadcast)
        Returns:
            (B, C, T) future state clamped to [0, 1]
        """
        x = self.lift_1d(eeg_state)
        for layer in self.seq_op:
            x = layer(x, sigma)
        return soft_clamp(self.proj_1d(x), 0.0, 1.0)

    def predict_graph_state(self,
                             node_feats:  torch.Tensor,
                             edge_index:  torch.Tensor,
                             sigma:       torch.Tensor) -> torch.Tensor:
        """
        O(1) surrogate for graph-based connectome propagation.

        Args:
            node_feats : (B, N, 1)   — per-node activity scalar
            edge_index : (2, E)      — COO edge list (long)
            sigma      : (B, 1)      — structural stress
        Returns:
            (B, N, 1) predicted node activities
        """
        x = self.lift_graph(node_feats)
        for layer in self.graph_op:
            x = layer(x, edge_index, sigma)
        return self.proj_graph(x)

    def predict_spatial_phase(self,
                               u_state: torch.Tensor,
                               sigma:   torch.Tensor) -> torch.Tensor:
        """
        O(1) surrogate for StructuralCahnHilliard3D PDE integration.

        Args:
            u_state : (B, 1, X, Y, Z)   — initial phase-field
            sigma   : (B, 1, 1, 1, 1)   — structural stress (broadcast)
        Returns:
            (B, 1, X, Y, Z) phase-field ∈ (−1, 1)
            −1 → Superego/OCD cluster  |  +1 → Id/impulsive cluster
        """
        x = self.lift_3d(u_state)
        for layer in self.spatial_op:
            x = layer(x, sigma)
        return torch.tanh(self.proj_3d(x))

    def optimize_ego(self,
                     id_prop:   torch.Tensor,
                     se_norm:   torch.Tensor,
                     sigma:     torch.Tensor) -> torch.Tensor:
        """
        O(1) surrogate for the DEQ / Anderson-mixing Ego optimisation.

        Args:
            id_prop  : (B, action_dim)  — Id-module proposals
            se_norm  : (B, action_dim)  — Superego normative policy
            sigma    : (B, 1)           — structural stress
        Returns:
            (B, action_dim) optimised action distribution
        """
        return self.psyche_op(id_prop, se_norm, sigma)

    def parameter_count(self) -> Dict[str, int]:
        counts = {
            "eeg_branch":   sum(p.numel() for p in list(self.lift_1d.parameters())
                                          + list(self.seq_op.parameters())
                                          + list(self.proj_1d.parameters())),
            "graph_branch": sum(p.numel() for p in list(self.lift_graph.parameters())
                                          + list(self.graph_op.parameters())
                                          + list(self.proj_graph.parameters())),
            "ch3d_branch":  sum(p.numel() for p in list(self.lift_3d.parameters())
                                          + list(self.spatial_op.parameters())
                                          + list(self.proj_3d.parameters())),
            "psyche_branch":sum(p.numel() for p in self.psyche_op.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


# =============================================================================
# 3.  Loss functions
# =============================================================================

class MSNOLosses:
    """
    Namespace for all surrogate loss functions.

    All losses return a scalar tensor with gradient attached.
    """

    @staticmethod
    def eeg_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Composite EEG trajectory loss.

        L_eeg = α · MSE + (1−α) · L1
        α = 0.5 balances large and small-deviation penalties equally.
        """
        mse = F.mse_loss(pred, target)
        l1  = F.l1_loss(pred, target)
        return 0.5 * mse + 0.5 * l1

    @staticmethod
    def phase_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Cahn-Hilliard phase-field loss.

        L_ch = MSE (point-wise) + λ_grad · gradient-magnitude discrepancy.
        Gradient term penalises interface diffuseness mismatch — important
        because CH interfaces are thin and MSE underweights them.
        """
        mse        = F.mse_loss(pred, target)
        # ── 3-D central-difference gradient magnitude ──────────────────────
        def _grad_mag(u: torch.Tensor) -> torch.Tensor:
            gx = u[:, :, 2:, :, :] - u[:, :, :-2, :, :]
            gy = u[:, :, :, 2:, :] - u[:, :, :, :-2, :]
            gz = u[:, :, :, :, 2:] - u[:, :, :, :, :-2]
            # trim to common shape then sum-of-squares
            s  = min(gx.shape[-3], gy.shape[-3], gz.shape[-3])
            s2 = min(gx.shape[-2], gy.shape[-2], gz.shape[-2])
            s3 = min(gx.shape[-1], gy.shape[-1], gz.shape[-1])
            return (gx[:, :, :s, :s2, :s3]**2
                    + gy[:, :, :s, :s2, :s3]**2
                    + gz[:, :, :s, :s2, :s3]**2).mean()
        grad_loss  = (_grad_mag(pred) - _grad_mag(target)).abs()
        return mse + 0.1 * grad_loss

    @staticmethod
    def ego_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        KL-divergence loss for the Ego action-distribution surrogate.

        pred   : (B, action_dim)  — MSNO soft-max output (already normalised)
        target : (B, action_dim)  — true Ego policy from PSYONEBridge
        """
        eps = 1e-8
        return F.kl_div((pred + eps).log(), target + eps, reduction="batchmean")

    @staticmethod
    def sigma_consistency_loss(sigma_pred: torch.Tensor,
                               sigma_true: torch.Tensor) -> torch.Tensor:
        """
        Optional auxiliary loss: keep MSNO's internal σ estimate consistent
        with the SSC filter output.  Weighted low (0.01) by default.
        """
        return F.mse_loss(sigma_pred, sigma_true)


# =============================================================================
# 4.  Synthetic dataset  (used for testing / pre-training when real data is
#     unavailable — replace DataLoader with OpenNeuro / TCGA loaders in prod)
# =============================================================================

class SyntheticMentalDataset(Dataset):
    """
    Synthetic benchmark dataset for smoke-testing the MSNO training pipeline.

    Each sample contains:
        eeg        : (C, T)          normalised EEG-like time-series
        eeg_target : (C, T)          simulated future state (teacher output)
        fmri_phase : (1, X, Y, Z)    random Cahn-Hilliard-like phase field
        phase_target:(1, X, Y, Z)    phase after simulated evolution
        id_prop    : (action_dim,)   uniform-random Id proposals
        se_norm    : (action_dim,)   uniform Superego normative distribution
        ego_target : (action_dim,)   normalised mixture (pseudo ground-truth)
        sigma      : ()              scalar structural stress ∈ [0.5, 2.0]
    """

    def __init__(self,
                 n_samples:   int = 512,
                 eeg_channels:int = 19,
                 n_timepoints:int = 256,
                 nx: int = 16, ny: int = 16, nz: int = 16,
                 action_dim:  int = 10,
                 seed:        int = 42) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.n  = n_samples
        self.C  = eeg_channels
        self.T  = n_timepoints
        self.nx, self.ny, self.nz = nx, ny, nz
        self.A  = action_dim

        # Pre-generate all data in RAM (small enough for smoke tests)
        def _norm_row(x: np.ndarray) -> np.ndarray:
            x = x - x.min(); d = x.max(); return x / (d + 1e-8)

        self.eeg         = rng.random((n_samples, eeg_channels, n_timepoints)).astype(np.float32)
        self.eeg_target  = np.clip(self.eeg + 0.02 * rng.standard_normal(self.eeg.shape), 0, 1).astype(np.float32)
        self.phase       = (2.0 * rng.random((n_samples, 1, nx, ny, nz)) - 1.0).astype(np.float32)
        self.phase_target= np.tanh(self.phase + 0.05 * rng.standard_normal(self.phase.shape)).astype(np.float32)
        raw_id           = rng.random((n_samples, action_dim)).astype(np.float32)
        self.id_prop     = (raw_id / (raw_id.sum(-1, keepdims=True) + 1e-8))
        raw_se           = rng.random((n_samples, action_dim)).astype(np.float32)
        self.se_norm     = (raw_se / (raw_se.sum(-1, keepdims=True) + 1e-8))
        raw_ego          = (self.id_prop + self.se_norm) / 2.0
        self.ego_target  = (raw_ego / (raw_ego.sum(-1, keepdims=True) + 1e-8))
        self.sigma       = rng.uniform(0.5, 2.0, (n_samples,)).astype(np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "eeg":          torch.from_numpy(self.eeg[idx]),
            "eeg_target":   torch.from_numpy(self.eeg_target[idx]),
            "fmri_phase":   torch.from_numpy(self.phase[idx]),
            "phase_target": torch.from_numpy(self.phase_target[idx]),
            "id_prop":      torch.from_numpy(self.id_prop[idx]),
            "se_norm":      torch.from_numpy(self.se_norm[idx]),
            "ego_target":   torch.from_numpy(self.ego_target[idx]),
            "sigma":        torch.tensor(self.sigma[idx]),
        }


# =============================================================================
# 5.  Trainer
# =============================================================================

class MSNOTrainer:
    """
    Production training loop for MentalStructuralNeuralOperator.

    Features
    --------
    * Multi-task loss (EEG + CH3D + Ego) with configurable weights
    * Linear warm-up → cosine-annealing LR schedule
    * Gradient clipping (max-norm)
    * Mixed-precision training (torch.cuda.amp)
    * Periodic validation with per-task MSE
    * Best-model checkpoint + latest checkpoint
    * Structured logging every cfg.log_every batches
    * Reproducible seeding

    Usage::

        cfg     = MSNOTrainingConfig(epochs=50, batch_size=16)
        model   = MentalStructuralNeuralOperator(...)
        trainer = MSNOTrainer(model, train_loader, val_loader, cfg)
        trainer.fit()
    """

    def __init__(self,
                 model:        MentalStructuralNeuralOperator,
                 train_loader: DataLoader,
                 val_loader:   Optional[DataLoader],
                 cfg:          MSNOTrainingConfig) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg

        # Device
        self.device = torch.device(cfg.device if torch.cuda.is_available()
                                   or cfg.device == "cpu" else "cpu")
        self.model.to(self.device)

        # Optimiser
        self.optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler: linear warm-up then cosine annealing
        total_steps   = cfg.epochs * len(train_loader)
        warmup_steps  = cfg.warmup_epochs * len(train_loader)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, _lr_lambda)

        # Mixed-precision scaler (CUDA only)
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss: float = float("inf")
        self.start_epoch:   int   = 0
        self.global_step:   int   = 0

        logger.info(
            f"MSNOTrainer ready  |  device={self.device}  amp={self.use_amp}  "
            f"epochs={cfg.epochs}  lr={cfg.lr}  grad_clip={cfg.grad_clip}"
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "latest") -> None:
        state = {
            "epoch":          epoch,
            "global_step":    self.global_step,
            "model_state":    self.model.state_dict(),
            "optimiser_state":self.optimiser.state_dict(),
            "scheduler_state":self.scheduler.state_dict(),
            "scaler_state":   self.scaler.state_dict(),
            "val_loss":       val_loss,
            "cfg":            vars(self.cfg),
            "msno_version":   MSNO_VERSION,
        }
        path = self.ckpt_dir / f"msno_{tag}.pt"
        torch.save(state, path)
        logger.info(f"  ↳ checkpoint saved → {path}  (val_loss={val_loss:.6f})")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimiser.load_state_dict(state["optimiser_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        self.start_epoch  = state["epoch"] + 1
        self.global_step  = state["global_step"]
        self.best_val_loss= state.get("val_loss", float("inf"))
        logger.info(f"Resumed from {path}  (epoch {self.start_epoch}  "
                    f"best_val={self.best_val_loss:.6f})")

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Executes one gradient update and returns a dict of scalar losses.
        """
        cfg = self.cfg

        # ── Move batch to device ──────────────────────────────────────────
        eeg        = batch["eeg"].to(self.device)          # (B, C, T)
        eeg_target = batch["eeg_target"].to(self.device)   # (B, C, T)
        phase      = batch["fmri_phase"].to(self.device)   # (B, 1, X, Y, Z)
        ph_target  = batch["phase_target"].to(self.device) # (B, 1, X, Y, Z)
        id_prop    = batch["id_prop"].to(self.device)      # (B, A)
        se_norm    = batch["se_norm"].to(self.device)      # (B, A)
        ego_tgt    = batch["ego_target"].to(self.device)   # (B, A)
        sigma_raw  = batch["sigma"].to(self.device)        # (B,)

        # Reshape sigma for each branch
        sigma_1d   = sigma_raw.view(-1, 1, 1)              # (B, 1, 1)
        sigma_3d   = sigma_raw.view(-1, 1, 1, 1, 1)        # (B, 1, 1, 1, 1)
        sigma_flat = sigma_raw.view(-1, 1)                  # (B, 1)

        self.optimiser.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            # ── Task 1: EEG trajectory ───────────────────────────────────
            eeg_pred  = self.model.predict_eeg_trajectory(eeg, sigma_1d)
            loss_eeg  = MSNOLosses.eeg_loss(eeg_pred, eeg_target)

            # ── Task 2: 3D CH phase separation ───────────────────────────
            ph_pred   = self.model.predict_spatial_phase(phase, sigma_3d)
            loss_ch3d = MSNOLosses.phase_loss(ph_pred, ph_target)

            # ── Task 3: Ego optimisation ──────────────────────────────────
            ego_pred  = self.model.optimize_ego(id_prop, se_norm, sigma_flat)
            loss_ego  = MSNOLosses.ego_loss(ego_pred, ego_tgt)

            total_loss = (cfg.lambda_eeg  * loss_eeg
                          + cfg.lambda_ch3d * loss_ch3d
                          + cfg.lambda_ego  * loss_ego)

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimiser)
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.scaler.step(self.optimiser)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1

        return {
            "total": total_loss.item(),
            "eeg":   loss_eeg.item(),
            "ch3d":  loss_ch3d.item(),
            "ego":   loss_ego.item(),
            "lr":    self.scheduler.get_last_lr()[0],
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Runs full validation set; returns mean per-task and total loss.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        cfg   = self.cfg
        sums  = {"total": 0.0, "eeg": 0.0, "ch3d": 0.0, "ego": 0.0}
        count = 0

        for batch in self.val_loader:
            eeg        = batch["eeg"].to(self.device)
            eeg_target = batch["eeg_target"].to(self.device)
            phase      = batch["fmri_phase"].to(self.device)
            ph_target  = batch["phase_target"].to(self.device)
            id_prop    = batch["id_prop"].to(self.device)
            se_norm    = batch["se_norm"].to(self.device)
            ego_tgt    = batch["ego_target"].to(self.device)
            sigma_raw  = batch["sigma"].to(self.device)

            sigma_1d   = sigma_raw.view(-1, 1, 1)
            sigma_3d   = sigma_raw.view(-1, 1, 1, 1, 1)
            sigma_flat = sigma_raw.view(-1, 1)

            with autocast(enabled=self.use_amp):
                eeg_pred  = self.model.predict_eeg_trajectory(eeg, sigma_1d)
                ph_pred   = self.model.predict_spatial_phase(phase, sigma_3d)
                ego_pred  = self.model.optimize_ego(id_prop, se_norm, sigma_flat)

                l_eeg  = MSNOLosses.eeg_loss(eeg_pred, eeg_target)
                l_ch3d = MSNOLosses.phase_loss(ph_pred, ph_target)
                l_ego  = MSNOLosses.ego_loss(ego_pred, ego_tgt)
                l_tot  = (cfg.lambda_eeg  * l_eeg
                          + cfg.lambda_ch3d * l_ch3d
                          + cfg.lambda_ego  * l_ego)

            b = eeg.size(0)
            sums["total"] += l_tot.item() * b
            sums["eeg"]   += l_eeg.item()  * b
            sums["ch3d"]  += l_ch3d.item() * b
            sums["ego"]   += l_ego.item()  * b
            count         += b

        self.model.train()
        return {k: v / max(count, 1) for k, v in sums.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """
        Full training loop with validation, checkpointing, and logging.
        """
        cfg = self.cfg
        _set_seed(cfg.seed)
        logger.info(f"{'='*64}")
        logger.info(f"  MSNO V{MSNO_VERSION}  |  Training started")
        logger.info(f"  Ecosystem: mental_one={'✓' if _HAS_MENTAL_ONE else '✗'}  "
                    f"ch3d={'✓' if _HAS_CH3D else '✗'}  "
                    f"psy={'✓' if _HAS_PSY else '✗'}  "
                    f"langevin={'✓' if _HAS_LANGEVIN else '✗'}")
        counts = self.model.parameter_count()
        logger.info(f"  Parameters: total={counts['total']:,}  "
                    f"eeg={counts['eeg_branch']:,}  "
                    f"ch3d={counts['ch3d_branch']:,}  "
                    f"psyche={counts['psyche_branch']:,}")
        logger.info(f"{'='*64}")

        self.model.train()
        t0 = time.time()

        for epoch in range(self.start_epoch, cfg.epochs):
            epoch_losses: Dict[str, List[float]] = {"total": [], "eeg": [], "ch3d": [], "ego": []}

            for step, batch in enumerate(self.train_loader):
                metrics = self._train_step(batch)
                for k in epoch_losses:
                    epoch_losses[k].append(metrics[k])

                if (step + 1) % cfg.log_every == 0:
                    avg = {k: float(np.mean(v[-cfg.log_every:])) for k, v in epoch_losses.items()}
                    elapsed = time.time() - t0
                    logger.info(
                        f"  epoch {epoch+1:3d}/{cfg.epochs}  "
                        f"step {self.global_step:6d}  "
                        f"loss={avg['total']:.5f}  "
                        f"eeg={avg['eeg']:.5f}  "
                        f"ch3d={avg['ch3d']:.5f}  "
                        f"ego={avg['ego']:.5f}  "
                        f"lr={metrics['lr']:.2e}  "
                        f"t={elapsed:.0f}s"
                    )

            # ── End-of-epoch summary ──────────────────────────────────────
            ep_avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
            logger.info(
                f"Epoch {epoch+1:3d} TRAIN  "
                f"total={ep_avg['total']:.5f}  eeg={ep_avg['eeg']:.5f}  "
                f"ch3d={ep_avg['ch3d']:.5f}  ego={ep_avg['ego']:.5f}"
            )

            # ── Validation ────────────────────────────────────────────────
            if (epoch + 1) % cfg.val_every == 0 and self.val_loader is not None:
                val_metrics = self.validate()
                logger.info(
                    f"Epoch {epoch+1:3d} VAL    "
                    f"total={val_metrics['total']:.5f}  "
                    f"eeg={val_metrics['eeg']:.5f}  "
                    f"ch3d={val_metrics['ch3d']:.5f}  "
                    f"ego={val_metrics['ego']:.5f}"
                )
                self.save_checkpoint(epoch, val_metrics["total"], tag="latest")
                if val_metrics["total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total"]
                    self.save_checkpoint(epoch, val_metrics["total"], tag="best")

        total_time = time.time() - t0
        logger.info(f"Training complete in {total_time/60:.1f} min  "
                    f"| best_val_loss={self.best_val_loss:.6f}")


# =============================================================================
# 6.  Utility: inference wrapper (for integration with MentalONEEngine)
# =============================================================================

class MSNOInference:
    """
    Thin wrapper around a trained MSNO for deployment inside the MENTAL ONE
    ecosystem.

    Example::

        inf = MSNOInference.from_checkpoint("msno_checkpoints/msno_best.pt")
        future_eeg = inf.predict_eeg(eeg_t0, sigma)
        future_u   = inf.predict_phase(u_t0, sigma)
        action     = inf.predict_ego(id_prop, se_norm, sigma)
    """

    def __init__(self,
                 model:  MentalStructuralNeuralOperator,
                 device: torch.device) -> None:
        self.model  = model.eval().to(device)
        self.device = device

    @classmethod
    def from_checkpoint(cls,
                        path: str,
                        cfg:  Optional[MSNOTrainingConfig] = None) -> "MSNOInference":
        if cfg is None:
            cfg = MSNOTrainingConfig()
        device = torch.device(cfg.device if torch.cuda.is_available()
                              or cfg.device == "cpu" else "cpu")
        model  = MentalStructuralNeuralOperator(
            eeg_channels=cfg.eeg_channels,
            latent_dim=cfg.latent_dim,
            modes_1d=cfg.modes_1d,
            modes_3d=cfg.modes_3d,
            action_dim=cfg.action_dim,
        )
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state"])
        logger.info(f"MSNOInference loaded from {path}")
        return cls(model, device)

    @torch.no_grad()
    def predict_eeg(self,
                    eeg:   torch.Tensor,
                    sigma: torch.Tensor) -> torch.Tensor:
        return self.model.predict_eeg_trajectory(
            eeg.to(self.device), sigma.to(self.device))

    @torch.no_grad()
    def predict_phase(self,
                      u:     torch.Tensor,
                      sigma: torch.Tensor) -> torch.Tensor:
        return self.model.predict_spatial_phase(
            u.to(self.device), sigma.to(self.device))

    @torch.no_grad()
    def predict_ego(self,
                    id_prop: torch.Tensor,
                    se_norm: torch.Tensor,
                    sigma:   torch.Tensor) -> torch.Tensor:
        return self.model.optimize_ego(
            id_prop.to(self.device),
            se_norm.to(self.device),
            sigma.to(self.device),
        )


# =============================================================================
# 7.  Helpers
# =============================================================================

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


def build_dataloaders(cfg: MSNOTrainingConfig,
                      val_fraction: float = 0.15
                      ) -> Tuple[DataLoader, DataLoader]:
    """
    Build synthetic train/val DataLoaders.
    Swap SyntheticMentalDataset for your real dataset here.
    """
    dataset = SyntheticMentalDataset(
        n_samples=512,
        eeg_channels=cfg.eeg_channels,
        action_dim=cfg.action_dim,
        seed=cfg.seed,
    )
    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers if cfg.device != "mps" else 0,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(cfg.device == "cuda"),
    )
    return train_loader, val_loader


# =============================================================================
# 8.  Entry point
# =============================================================================

if __name__ == "__main__":
    cfg = MSNOTrainingConfig(
        epochs        = 20,
        batch_size    = 8,
        lr            = 1e-4,
        latent_dim    = 32,       # smaller for smoke-test
        modes_1d      = 16,
        modes_3d      = 4,
        log_every     = 5,
        val_every     = 5,
        use_amp       = torch.cuda.is_available(),
        checkpoint_dir= "./msno_checkpoints",
        seed          = 42,
    )

    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model  = MentalStructuralNeuralOperator(
        eeg_channels=cfg.eeg_channels,
        latent_dim  =cfg.latent_dim,
        modes_1d    =cfg.modes_1d,
        modes_3d    =cfg.modes_3d,
        action_dim  =cfg.action_dim,
    )

    counts = model.parameter_count()
    logger.info(f"Model parameter breakdown: {counts}")

    train_loader, val_loader = build_dataloaders(cfg)

    trainer = MSNOTrainer(model, train_loader, val_loader, cfg)
    trainer.fit()

    logger.info("Smoke-test complete.  Load best model with:")
    logger.info("  inf = MSNOInference.from_checkpoint('./msno_checkpoints/msno_best.pt')")
