# =============================================================================
# PSY ONE BRIDGE
# =============================================================================
# Developer  : Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788
# GitHub     : https://github.com/yoonalimsuwan
#
# PSY ONE BRIDGE connects the Informational Mechanics of the Id, Ego, and
# Superego (information-theoretic psychoanalysis) to the MENTAL ONE engine.
#
# Theoretical Basis:
#   • "The Informational Mechanics of the Id, Ego, and Superego: Integrating
#     Psychoanalysis, Neuroscience, and Information Theory to Explain Human
#     Decision-Making" — Yoon A Limsuwan (2026)
#
# Architecture Mapping:
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  Psychoanalytic Theory        PSY ONE BRIDGE       MENTAL ONE       │
#   │  ─────────────────────        ─────────────        ──────────       │
#   │  Id (𝓘) — raw drives    ←→  IdModule            SOCController      │
#   │  Ego (𝓔) — optimizer    ←→  EgoModule           SSCClassifier      │
#   │  Superego (𝓢) — norms   ←→  SuperegoModule      DiffRGRefiner      │
#   │  Psyche Ψ(t) = ⟨𝓘,𝓔,𝓢⟩ ←→  PsycheTriad         MentalONEEngine    │
#   └─────────────────────────────────────────────────────────────────────┘
#
# Key Mathematical Constructs Implemented:
#   • Id Entropy          H(𝓘) = −∑ P(xᵢ) log₂ P(xᵢ)
#   • Superego KL Penalty L_𝓢(π) = D_KL(π ∥ π_norm)
#   • Ego Free Energy     ℱ = D_KL(Q(ϕ|μ) ∥ P(ϕ|m)) − 𝔼_Q[log P(μ|ϕ,m)]
#   • Ego Action          a* = argmin_a [ ℱ(a) + λ · L_𝓢(a) ]
#   • Psyche State        Ψ(t) = ⟨ 𝓘(t), 𝓔(t), 𝓢(t) ⟩
#   • Id Accumulation     𝓘(t) = 𝓘(0) + ∫₀ᵗ w(τ)·H(x(τ)) dτ
#
# Psychopathology Modes:
#   • MDD / Anxiety       : λ → ∞  (Superego over-regulation)
#   • Schizophrenia       : H(𝓘) → H_max, α → 0  (Id collapse)
#   • OCD                 : ℱ never converges  (infinite optimization loop)
#   • Healthy             : balanced λ, low H(𝓘), converged ℱ
#
# Dependencies (all permissive licences):
#   • PyTorch ≥ 2.0      (BSD-style)
#   • NumPy              (BSD-3-Clause)
#   • SciPy              (BSD-3-Clause)
#   • MENTAL ONE         (MIT) — mental_one.py must be importable
#
# Integration:
#   from psy_one_bridge import PSYONEBridge, PsycheTriad, PsychopathologyMode
#   bridge = PSYONEBridge(mental_one_engine)
#   result = bridge.run_psyche_cycle(eeg_state, subject_history)
#
# MIT License
# -----------
# Copyright (c) 2026 Yoon A Limsuwan / MSPS NETWORK
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# =============================================================================

from __future__ import annotations

import math
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [PSY_ONE_BRIDGE]  %(levelname)s  %(message)s",
)
logger = logging.getLogger("PSY_ONE_BRIDGE")

# ---------------------------------------------------------------------------
# Optional MENTAL ONE import
# ---------------------------------------------------------------------------
try:
    from mental_one import (
        SSCClassifier,
        SOCController,
        CSOCKernel,
        DiffRGRefiner,
        MentalHealthEvolution,
        ItoProcess,
        InterventionDesigner,
        MentalONEEngine,
        ALL_PSYCHIATRIC_DISORDERS,
        OPTIMAL_DEVICE,
    )
    HAS_MENTAL_ONE = True
    logger.info("✓ MENTAL ONE imported successfully.")
except ImportError:
    HAS_MENTAL_ONE = False
    ALL_PSYCHIATRIC_DISORDERS = [
        "MDD", "Bipolar", "Schizophrenia", "PTSD", "Panic",
        "Conversion", "Dissociative", "Somatic", "Parasomnia", "Healthy",
    ]
    OPTIMAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(
        "MENTAL ONE not found — PSY ONE BRIDGE running in standalone mode. "
        "Install mental_one.py to enable full engine integration."
    )

# =============================================================================
# 0.  Enumerations & Configuration
# =============================================================================

class PsychopathologyMode(Enum):
    """
    Preset parameter distortions that simulate clinical psychiatric profiles.

    Based on Section 5.3 of the theoretical paper:
      • HEALTHY       : balanced drives, low entropy, converged optimization
      • MDD_ANXIETY   : hyper-regularization (λ → ∞), narrow π_norm
      • SCHIZOPHRENIA : Id entropy collapse (H → H_max), Ego failure (α → 0)
      • OCD           : Ego stuck in infinite loop (ℱ never converges)
      • BIPOLAR       : oscillating λ — alternates over/under-regulation
      • PTSD          : elevated emotional salience on trauma vectors
      • CUSTOM        : user-supplied parameters
    """
    HEALTHY       = "healthy"
    MDD_ANXIETY   = "mdd_anxiety"
    SCHIZOPHRENIA = "schizophrenia"
    OCD           = "ocd"
    BIPOLAR       = "bipolar"
    PTSD          = "ptsd"
    CUSTOM        = "custom"


@dataclass
class PsycheConfig:
    """
    Full configuration for a PSY ONE BRIDGE inference cycle.

    Attributes
    ----------
    action_dim : int
        Dimensionality of the action/decision space.
    lambda_reg : float
        Superego penalty weight λ.  High → strict moral/social constraint.
    alpha_lr : float
        Ego optimization learning rate α.  High → fast adaptation.
        Neurophysiological proxy: prefrontal-subcortical white matter integrity.
    n_ego_iter : int
        Number of Ego optimization iterations per cycle.
    history_window : int
        Number of past time steps used to accumulate Id state 𝓘(t).
    emotional_salience_scale : float
        Global multiplier for emotional salience weighting w(τ).
    mode : PsychopathologyMode
        Psychopathology simulation preset.
    device : torch.device
        Compute device.
    verbose : bool
        Enable detailed logging.
    """
    action_dim             : int                   = 10
    lambda_reg             : float                 = 2.5
    alpha_lr               : float                 = 0.05
    n_ego_iter             : int                   = 50
    history_window         : int                   = 100
    emotional_salience_scale: float                = 1.0
    mode                   : PsychopathologyMode   = PsychopathologyMode.HEALTHY
    device                 : torch.device          = OPTIMAL_DEVICE
    verbose                : bool                  = False

    def apply_mode(self) -> "PsycheConfig":
        """Apply psychopathology distortions defined in the theoretical paper."""
        if self.mode == PsychopathologyMode.HEALTHY:
            pass  # default parameters represent a healthy state
        elif self.mode == PsychopathologyMode.MDD_ANXIETY:
            self.lambda_reg  = 50.0   # hyper-regularization: λ → ∞
            self.alpha_lr    = 0.005  # low adaptability
        elif self.mode == PsychopathologyMode.SCHIZOPHRENIA:
            self.lambda_reg  = 0.01   # collapsed Superego constraint
            self.alpha_lr    = 1e-5   # Ego optimization failure: α → 0
        elif self.mode == PsychopathologyMode.OCD:
            self.n_ego_iter  = 500    # compulsive repetition; ℱ never converges
            self.lambda_reg  = 15.0   # elevated Superego sensitivity
        elif self.mode == PsychopathologyMode.BIPOLAR:
            self.lambda_reg  = float(np.random.choice([0.2, 40.0]))  # oscillating
        elif self.mode == PsychopathologyMode.PTSD:
            self.emotional_salience_scale = 8.0   # hyper-salience on trauma vectors
        return self


# =============================================================================
# 1.  Id Module  —  High-Entropy Generative State Space
# =============================================================================

class IdModule(nn.Module):
    """
    Models the Id (𝓘) as a model-free generative state space.

    Theoretical role (paper §1.5 H1, §4.2):
      • Continuously accumulates historical sensory vectors x(τ) weighted by
        emotional salience w(τ).
      • Entropy H(𝓘) = −∑ P(xᵢ) log₂ P(xᵢ) measures drive unpredictability.
      • Temporal accumulation:  𝓘(t) = 𝓘(0) + ∫₀ᵗ w(τ)·H(x(τ)) dτ

    MENTAL ONE mapping:
      • SOCController.sigma()     →  structural stress ↔ Id entropy proxy
      • CSOCKernel.forward()      →  criticality weighting ↔ emotional salience
      • SOCController.temperature() →  thermal noise ↔ drive volatility
    """

    def __init__(
        self,
        action_dim    : int,
        history_window: int   = 100,
        device        : torch.device = OPTIMAL_DEVICE,
    ) -> None:
        super().__init__()
        self.action_dim     = action_dim
        self.history_window = history_window
        self.device         = device

        # Drive probability distribution over action space
        self.register_buffer(
            "drive_weights",
            torch.ones(action_dim, device=device) / action_dim,
        )
        # Circular history buffer:  (history_window, action_dim)
        self.register_buffer(
            "history_buffer",
            torch.zeros(history_window, action_dim, device=device),
        )
        self.register_buffer("_buf_ptr", torch.tensor(0, device=device))
        self.register_buffer(
            "accumulated_entropy",
            torch.tensor(0.0, device=device),
        )

        # Optional: MENTAL ONE SOC kernel for criticality-weighted salience
        if HAS_MENTAL_ONE:
            self.csoc_kernel: Optional[CSOCKernel] = CSOCKernel().to(device)
        else:
            self.csoc_kernel = None

    # ------------------------------------------------------------------
    def _criticality_weight(self, r: torch.Tensor) -> torch.Tensor:
        """
        Returns a criticality scaling factor from CSOC kernel K(r).
        Falls back to 1.0 when MENTAL ONE is unavailable.
        """
        if self.csoc_kernel is not None:
            return self.csoc_kernel(r).mean().clamp(0.1, 10.0)
        return torch.tensor(1.0, device=self.device)

    # ------------------------------------------------------------------
    def update_drive_states(
        self,
        sensory_input      : torch.Tensor,
        emotional_salience : torch.Tensor,
        salience_scale     : float = 1.0,
    ) -> None:
        """
        Update Id drive distribution from sensory input and emotional salience.

        Implements:  𝓘(t) ← 𝓘(t-1) + α · w(τ) · H(x(τ))

        Parameters
        ----------
        sensory_input      : (action_dim,) raw sensory feature vector
        emotional_salience : (action_dim,) emotional weighting per action
        salience_scale     : global scalar multiplier (elevated in PTSD mode)
        """
        sensory_input       = sensory_input.to(self.device).float()
        emotional_salience  = emotional_salience.to(self.device).float()

        if sensory_input.shape != (self.action_dim,):
            sensory_input = F.interpolate(
                sensory_input.unsqueeze(0).unsqueeze(0),
                size=self.action_dim, mode="linear", align_corners=True,
            ).squeeze()

        # Criticality-weighted salience
        r = torch.norm(sensory_input) + 1e-8
        crit_w = self._criticality_weight(r.unsqueeze(0))
        scaled_salience = emotional_salience * salience_scale * crit_w

        # Integrate into drive weights
        raw_updates = sensory_input * scaled_salience
        self.drive_weights = self.drive_weights + 0.1 * raw_updates

        # Softmax normalization → valid probability simplex
        self.drive_weights = F.softmax(self.drive_weights, dim=0)

        # Update circular history buffer
        ptr = int(self._buf_ptr.item())
        self.history_buffer[ptr] = sensory_input.detach()
        self._buf_ptr.fill_((ptr + 1) % self.history_window)

        # Accumulate temporal entropy integral
        self.accumulated_entropy = (
            self.accumulated_entropy + self.compute_entropy().detach()
        )

    # ------------------------------------------------------------------
    def compute_entropy(self) -> torch.Tensor:
        """
        Compute Shannon Entropy of current Id state.

        H(𝓘) = −∑ᵢ P(xᵢ) log₂ P(xᵢ)

        Neurophysiological proxy:
          • High H(𝓘) ↔ amygdala / striatum hyper-activation (fMRI)
          • High H(𝓘) ↔ elevated Lempel-Ziv complexity in resting EEG
        """
        p = self.drive_weights.clamp(min=1e-12)
        return -(p * torch.log2(p)).sum()

    # ------------------------------------------------------------------
    def compute_temporal_entropy(self) -> torch.Tensor:
        """
        Compute entropy over the full history buffer (temporal Id complexity).
        Captures 𝓘(t) = 𝓘(0) + ∫₀ᵗ w(τ)·H(x(τ)) dτ more directly.
        """
        buf_flat = self.history_buffer.reshape(-1)
        p = F.softmax(buf_flat, dim=0).clamp(min=1e-12)
        return -(p * torch.log2(p)).sum()

    # ------------------------------------------------------------------
    def generate_proposals(self) -> torch.Tensor:
        """Return the raw unconstrained action proposal distribution of the Id."""
        return self.drive_weights.clone()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        nn.init.constant_(self.drive_weights, 1.0 / self.action_dim)
        self.history_buffer.zero_()
        self._buf_ptr.zero_()
        self.accumulated_entropy.zero_()


# =============================================================================
# 2.  Superego Module  —  Normative Prior Constraint Matrix
# =============================================================================

class SuperegoModule(nn.Module):
    """
    Models the Superego (𝓢) as a top-down normative constraint system.

    Theoretical role (paper §1.7, §4.2):
      • Encodes societal data, moral imperatives, and long-term baselines.
      • Regulatory penalty via KL Divergence:
          L_𝓢(π) = D_KL(π(a|s) ∥ π_norm(a|s))
      • Processing error ε_𝓢 ∝ behavioral deviation ΔB  (H2)

    MENTAL ONE mapping:
      • DiffRGRefiner.forward()   →  RG smoothing ↔ Superego noise filtering
      • SSCClassifier.ref_Healthy →  healthy reference ↔ π_norm baseline
      • Frontal Alpha Asymmetry   →  high FAA ↔ elevated Superego constraint
    """

    def __init__(
        self,
        action_dim: int,
        device    : torch.device = OPTIMAL_DEVICE,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device     = device

        # π_norm: uniform baseline (updated via set_societal_baseline)
        self.register_buffer(
            "normative_policy",
            torch.ones(action_dim, device=device) / action_dim,
        )
        # Accumulated classification error  ε_𝓢
        self.register_buffer(
            "cumulative_error",
            torch.tensor(0.0, device=device),
        )

        # Optional: RG refiner for smoothing the normative distribution
        if HAS_MENTAL_ONE:
            self.rg_refiner: Optional[DiffRGRefiner] = DiffRGRefiner(
                factor=2, n_levels=1
            ).to(device)
        else:
            self.rg_refiner = None

    # ------------------------------------------------------------------
    def set_societal_baseline(
        self,
        normative_distribution: torch.Tensor,
        smooth: bool = True,
    ) -> None:
        """
        Set the societal normative policy π_norm.

        Parameters
        ----------
        normative_distribution : (action_dim,) unnormalized distribution
        smooth                 : apply RG smoothing if MENTAL ONE is available
        """
        nd = normative_distribution.to(self.device).float()
        nd = (nd / (nd.sum() + 1e-12)).clamp(min=1e-12)

        if smooth and self.rg_refiner is not None:
            nd_smooth = self.rg_refiner(nd.unsqueeze(0)).squeeze(0)
            nd = F.softmax(nd_smooth, dim=0).clamp(min=1e-12)

        self.normative_policy = nd

    # ------------------------------------------------------------------
    def evaluate_policy_divergence(self, proposed_policy: torch.Tensor) -> torch.Tensor:
        """
        Compute KL Divergence penalty between proposed and normative policies.

        L_𝓢(π) = D_KL(π ∥ π_norm) = ∑ π(a) log₂(π(a) / π_norm(a))

        Neurophysiological proxy:
          • High L_𝓢 ↔ elevated ACC activation + high Global Field Power (GFP)
          • λ parameter ↔ functional dlPFC–ACC connectivity strength
        """
        p = proposed_policy.to(self.device).float().clamp(min=1e-12)
        q = self.normative_policy.clamp(min=1e-12)
        kl = (p * torch.log2(p / q)).sum()
        return kl.clamp(min=0.0)

    # ------------------------------------------------------------------
    def register_error(self, predicted: int, actual: int) -> None:
        """
        Record a classification error to track ε_𝓢 over time.
        Per H2:  ΔB ∝ ε_𝓢
        """
        if predicted != actual:
            self.cumulative_error = self.cumulative_error + 1.0

    # ------------------------------------------------------------------
    def behavioral_entropy(self, n_decisions: int) -> torch.Tensor:
        """
        Compute behavioral deviation entropy based on accumulated error rate.
        As ε_𝓢 → ∞,  H(B) → H_max  (H2 limit).
        """
        if n_decisions == 0:
            return torch.tensor(0.0, device=self.device)
        eps = self.cumulative_error / n_decisions
        eps = eps.clamp(1e-8, 1 - 1e-8)
        h = -(eps * torch.log2(eps) + (1 - eps) * torch.log2(1 - eps))
        return h

    # ------------------------------------------------------------------
    def reset(self) -> None:
        nn.init.constant_(self.normative_policy, 1.0 / self.action_dim)
        self.cumulative_error.zero_()


# =============================================================================
# 3.  Ego Module  —  Active Inference Optimizer
# =============================================================================

class EgoModule(nn.Module):
    """
    Models the Ego (𝓔) as the central Free Energy minimization optimizer.

    Theoretical role (paper §1.5 H3, §4.2):
      • Resolves conflict between Id drives and Superego constraints.
      • Variational Free Energy:
          ℱ = D_KL(Q(ϕ|μ) ∥ P(ϕ|m)) − 𝔼_Q[log P(μ|ϕ,m)]
      • Ego action:
          a* = argmin_a [ ℱ(a) + λ · L_𝓢(a) ]

    MENTAL ONE mapping:
      • SSCClassifier.contraction_update() →  fixed-point iteration ↔ Ego loop
      • SSCClassifier.energy()             →  E(s, disorder) ↔ ℱ approximation
      • eta (step size)                    →  α learning rate ↔ neuroplasticity
    """

    def __init__(
        self,
        action_dim  : int,
        alpha_lr    : float         = 0.05,
        device      : torch.device  = OPTIMAL_DEVICE,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.alpha      = alpha_lr
        self.device     = device
        self._free_energy_history: List[float] = []

    # ------------------------------------------------------------------
    def _compute_variational_free_energy(
        self,
        q_policy  : torch.Tensor,
        p_prior   : torch.Tensor,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate Variational Free Energy.

        ℱ ≈ D_KL(Q ∥ P) − 𝔼_Q[log P(μ|ϕ,m)]

        When no external observation μ is provided, the likelihood term is
        approximated as the negative entropy of Q (maximum-entropy prior).
        """
        q = q_policy.clamp(min=1e-12)
        p = p_prior.clamp(min=1e-12)

        kl_term = (q * torch.log2(q / p)).sum()

        if observation is not None:
            obs = observation.to(self.device).float().clamp(min=1e-12)
            obs_norm = obs / obs.sum()
            log_likelihood = (q * torch.log2(obs_norm + 1e-12)).sum()
        else:
            # Approximate: 𝔼_Q[log P] ≈ H(Q) (max-entropy)
            log_likelihood = (q * torch.log2(q)).sum()  # = −H(Q)

        return kl_term - log_likelihood

    # ------------------------------------------------------------------
    def optimize_action(
        self,
        id_proposal : torch.Tensor,
        superego    : SuperegoModule,
        lambda_reg  : float,
        observation : Optional[torch.Tensor] = None,
        n_iter      : int                    = 50,
    ) -> Tuple[int, torch.Tensor, List[float]]:
        """
        Execute iterative Free Energy minimization.

        a* = argmin_a [ ℱ(a) + λ · D_KL(π ∥ π_norm) ]

        Returns
        -------
        selected_action  : int           — sampled action index
        optimized_policy : Tensor (D,)   — final probability distribution
        fe_history       : List[float]   — Free Energy per iteration
        """
        optimized_policy = id_proposal.to(self.device).float().clone()
        fe_history: List[float] = []

        for step in range(n_iter):
            op_detach = optimized_policy.detach().requires_grad_(True)

            # Free Energy
            fe = self._compute_variational_free_energy(
                op_detach, superego.normative_policy, observation
            )
            # Superego KL penalty
            superego_loss = superego.evaluate_policy_divergence(op_detach)
            total_cost = fe + lambda_reg * superego_loss

            # Analytical gradient of total cost w.r.t. log-policy
            total_cost.backward()
            with torch.no_grad():
                log_op = torch.log(optimized_policy.clamp(min=1e-12))
                log_op = log_op - self.alpha * op_detach.grad
                optimized_policy = F.softmax(log_op, dim=0)

            fe_history.append(float(total_cost.detach().item()))

        self._free_energy_history = fe_history

        selected_action = int(
            torch.multinomial(optimized_policy, num_samples=1).item()
        )
        return selected_action, optimized_policy, fe_history

    # ------------------------------------------------------------------
    def detect_ocd_loop(self, convergence_threshold: float = 0.01) -> bool:
        """
        Detect OCD-like optimization failure: ℱ never converges.

        Returns True when the last 20% of iterations show < convergence_threshold
        relative improvement — indicating the Ego is stuck in a compulsive loop.
        """
        if len(self._free_energy_history) < 10:
            return False
        tail = self._free_energy_history[int(len(self._free_energy_history) * 0.8):]
        if tail[0] == 0:
            return False
        relative_improvement = abs(tail[-1] - tail[0]) / (abs(tail[0]) + 1e-12)
        return relative_improvement < convergence_threshold

    # ------------------------------------------------------------------
    def convergence_speed(self) -> float:
        """
        Return normalized convergence speed as a proxy for α (neuroplasticity).
        Higher → faster adaptation → better prefrontal-subcortical connectivity.
        """
        if len(self._free_energy_history) < 2:
            return 0.0
        delta = abs(self._free_energy_history[0] - self._free_energy_history[-1])
        return delta / (abs(self._free_energy_history[0]) + 1e-12)


# =============================================================================
# 4.  Psyche Triad  —  Ψ(t) = ⟨ 𝓘(t), 𝓔(t), 𝓢(t) ⟩
# =============================================================================

@dataclass
class PsycheTriadState:
    """
    Full psychic state at time t.

    Attributes
    ----------
    id_entropy          : H(𝓘)                — drive unpredictability
    accumulated_entropy : ∫₀ᵗ w(τ)·H(x(τ))dτ  — temporal Id accumulation
    superego_loss       : L_𝓢                  — normative penalty
    behavioral_entropy  : H(B) ∝ ε_𝓢           — behavioral deviation
    free_energy         : ℱ (final iteration)   — Ego optimization residual
    selected_action     : a*                    — final decision output
    optimized_policy    : π*(a|s)              — final action distribution
    diagnosis           : str                   — MENTAL ONE classification
    intervention_plan   : Dict                  — treatment recommendations
    ocd_loop_detected   : bool                  — OCD marker
    convergence_speed   : float                 — neuroplasticity proxy
    fe_history          : List[float]           — Free Energy trajectory
    neurophysio_map     : Dict[str, float]      — mapped neural biomarkers
    """
    id_entropy          : float
    accumulated_entropy : float
    superego_loss       : float
    behavioral_entropy  : float
    free_energy         : float
    selected_action     : int
    optimized_policy    : np.ndarray
    diagnosis           : str                = "Unknown"
    intervention_plan   : Dict[str, Any]     = field(default_factory=dict)
    ocd_loop_detected   : bool               = False
    convergence_speed   : float              = 0.0
    fe_history          : List[float]        = field(default_factory=list)
    neurophysio_map     : Dict[str, float]   = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id_entropy"          : round(self.id_entropy, 6),
            "accumulated_entropy" : round(self.accumulated_entropy, 6),
            "superego_loss"       : round(self.superego_loss, 6),
            "behavioral_entropy"  : round(self.behavioral_entropy, 6),
            "free_energy"         : round(self.free_energy, 6),
            "selected_action"     : self.selected_action,
            "optimized_policy"    : self.optimized_policy.tolist(),
            "diagnosis"           : self.diagnosis,
            "intervention_plan"   : self.intervention_plan,
            "ocd_loop_detected"   : self.ocd_loop_detected,
            "convergence_speed"   : round(self.convergence_speed, 6),
            "fe_history"          : self.fe_history,
            "neurophysio_map"     : self.neurophysio_map,
        }


class PsycheTriad(nn.Module):
    """
    Orchestrates the full Ψ(t) = ⟨ 𝓘(t), 𝓔(t), 𝓢(t) ⟩ inference cycle.

    This is the primary interface for standalone psyche simulation, independent
    of MENTAL ONE.  Use PSYONEBridge for full engine integration.
    """

    def __init__(self, config: PsycheConfig) -> None:
        super().__init__()
        self.config = config.apply_mode()
        self.device = config.device

        self.id_module       = IdModule(config.action_dim, config.history_window, config.device)
        self.superego_module = SuperegoModule(config.action_dim, config.device)
        self.ego_module      = EgoModule(config.action_dim, config.alpha_lr, config.device)

        self._n_decisions: int = 0
        logger.info(
            f"PsycheTriad initialized  |  mode={config.mode.value}  "
            f"λ={config.lambda_reg:.2f}  α={config.alpha_lr:.4f}  "
            f"action_dim={config.action_dim}"
        )

    # ------------------------------------------------------------------
    def set_societal_baseline(self, normative_dist: torch.Tensor) -> None:
        self.superego_module.set_societal_baseline(normative_dist)

    # ------------------------------------------------------------------
    def run_inference_cycle(
        self,
        sensory_state       : torch.Tensor,
        emotional_salience  : torch.Tensor,
        observation         : Optional[torch.Tensor] = None,
        actual_action       : Optional[int]          = None,
    ) -> PsycheTriadState:
        """
        Execute one complete psychic inference cycle.

        Pipeline:
          1. Id      : update drive states → compute H(𝓘)
          2. Ego     : optimize action     → minimize ℱ + λ·L_𝓢
          3. Superego: evaluate penalty    → compute L_𝓢, H(B)
          4. Bridge  : map to neurophysio  → FAA, GFP, PLV proxies

        Parameters
        ----------
        sensory_state      : (action_dim,) current sensory feature vector
        emotional_salience : (action_dim,) emotional weighting vector
        observation        : (action_dim,) optional environmental observation μ
        actual_action      : optional true action for error tracking

        Returns
        -------
        PsycheTriadState with all computed metrics
        """
        # Step 1: Update Id
        self.id_module.update_drive_states(
            sensory_state,
            emotional_salience,
            salience_scale=self.config.emotional_salience_scale,
        )
        id_proposal         = self.id_module.generate_proposals()
        id_entropy          = float(self.id_module.compute_entropy().item())
        accumulated_entropy = float(self.id_module.accumulated_entropy.item())

        # Step 2: Ego optimization
        action, opt_policy, fe_history = self.ego_module.optimize_action(
            id_proposal  = id_proposal,
            superego     = self.superego_module,
            lambda_reg   = self.config.lambda_reg,
            observation  = observation,
            n_iter       = self.config.n_ego_iter,
        )
        free_energy      = fe_history[-1] if fe_history else 0.0
        ocd_loop         = self.ego_module.detect_ocd_loop()
        conv_speed       = self.ego_module.convergence_speed()

        # Step 3: Superego evaluation
        superego_loss    = float(
            self.superego_module.evaluate_policy_divergence(opt_policy).item()
        )
        self._n_decisions += 1
        if actual_action is not None:
            self.superego_module.register_error(action, actual_action)
        beh_entropy = float(
            self.superego_module.behavioral_entropy(self._n_decisions).item()
        )

        # Step 4: Neurophysiological proxy mapping
        neurophysio = self._compute_neurophysio_map(
            id_entropy    = id_entropy,
            superego_loss = superego_loss,
            conv_speed    = conv_speed,
            lambda_reg    = self.config.lambda_reg,
        )

        return PsycheTriadState(
            id_entropy          = id_entropy,
            accumulated_entropy = accumulated_entropy,
            superego_loss       = superego_loss,
            behavioral_entropy  = beh_entropy,
            free_energy         = free_energy,
            selected_action     = action,
            optimized_policy    = opt_policy.detach().cpu().numpy(),
            ocd_loop_detected   = ocd_loop,
            convergence_speed   = conv_speed,
            fe_history          = fe_history,
            neurophysio_map     = neurophysio,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_neurophysio_map(
        id_entropy    : float,
        superego_loss : float,
        conv_speed    : float,
        lambda_reg    : float,
    ) -> Dict[str, float]:
        """
        Map informational parameters to neurophysiological biomarker proxies.

        Based on Section 5.2 of the theoretical paper:
          • H(𝓘)  ↔  EEG Lempel-Ziv Complexity  (amygdala hyper-activation)
          • λ      ↔  dlPFC–ACC connectivity       (Frontal Alpha Asymmetry)
          • α      ↔  prefrontal white matter      (neuroplasticity index)
          • L_𝓢   ↔  Global Field Power (GFP)     (norm-violation response)
        """
        # FAA proxy: positive value → left-frontal dominance (approach motivation)
        # High λ → high FAA (Superego dominance → withdrawal / constraint)
        faa_proxy = math.tanh(lambda_reg / 10.0) * 0.5

        # GFP proxy: scales with Superego penalty
        gfp_proxy = min(superego_loss * 2.5, 10.0)

        # LZ complexity proxy: scales with Id entropy (max ≈ log2(action_dim))
        lz_complexity = id_entropy / max(math.log2(max(id_entropy, 1.0) + 1), 1.0)

        # PLV F3-F4 proxy: phase synchrony collapses in schizophrenia
        # High convergence speed → intact frontal synchrony
        plv_f3_f4 = min(conv_speed, 1.0) * 0.8

        # Neuroplasticity index: α ↔ PFC-subcortical white matter tract integrity
        neuroplasticity = min(conv_speed * 2.0, 1.0)

        return {
            "FAA_proxy"          : round(faa_proxy, 4),
            "GFP_proxy"          : round(gfp_proxy, 4),
            "LZ_complexity"      : round(lz_complexity, 4),
            "PLV_F3_F4_proxy"    : round(plv_f3_f4, 4),
            "neuroplasticity_idx": round(neuroplasticity, 4),
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.id_module.reset()
        self.superego_module.reset()
        self._n_decisions = 0
        logger.info("PsycheTriad state reset.")


# =============================================================================
# 5.  PSY ONE BRIDGE  —  Full MENTAL ONE Integration
# =============================================================================

class PSYONEBridge:
    """
    Main integration class connecting PSY ONE psyche theory to MENTAL ONE.

    When MENTAL ONE is available:
      • Uses SSCClassifier for psychiatric diagnosis (Ego-level classification)
      • Uses SOCController entropy as Id entropy ground-truth
      • Uses DiffRGRefiner to smooth the Superego normative baseline
      • Uses InterventionDesigner for treatment plan generation
      • Returns a unified PsycheTriadState enriched with clinical data

    When MENTAL ONE is unavailable:
      • Falls back to standalone PsycheTriad simulation

    Usage
    -----
    # With MENTAL ONE engine:
    engine = MentalONEEngine()
    bridge = PSYONEBridge.from_mental_one(engine, config)
    result = bridge.run_psyche_cycle(eeg_tensor, salience_vector)

    # Standalone:
    bridge = PSYONEBridge(config=PsycheConfig(action_dim=10))
    result = bridge.run_psyche_cycle(sensory_vec, salience_vec)
    """

    def __init__(
        self,
        config             : Optional[PsycheConfig] = None,
        mental_one_engine  : Optional[Any]          = None,
    ) -> None:
        self.config    = config or PsycheConfig()
        self.device    = self.config.device
        self.engine    = mental_one_engine  # MentalONEEngine instance (optional)
        self.triad     = PsycheTriad(self.config)
        self._cycle_count: int = 0

        if HAS_MENTAL_ONE and mental_one_engine is not None:
            self._setup_mental_one_bridge()
        elif HAS_MENTAL_ONE:
            logger.info(
                "MENTAL ONE available. Pass a MentalONEEngine instance via "
                "PSYONEBridge.from_mental_one() for full integration."
            )
        else:
            logger.info("PSY ONE BRIDGE running in standalone mode.")

    # ------------------------------------------------------------------
    @classmethod
    def from_mental_one(
        cls,
        engine : Any,
        config : Optional[PsycheConfig] = None,
    ) -> "PSYONEBridge":
        """
        Factory constructor when a MentalONEEngine instance is available.

        Parameters
        ----------
        engine : MentalONEEngine  (from mental_one.py)
        config : optional PsycheConfig override
        """
        cfg = config or PsycheConfig(device=OPTIMAL_DEVICE)
        bridge = cls(config=cfg, mental_one_engine=engine)
        return bridge

    # ------------------------------------------------------------------
    def _setup_mental_one_bridge(self) -> None:
        """Initialize bridge components using MENTAL ONE internals."""
        if not HAS_MENTAL_ONE:
            return
        # Replace EgoModule's alpha with SSCClassifier's eta for consistency
        if hasattr(self.engine, "classifier") and self.engine.classifier is not None:
            ssc: SSCClassifier = self.engine.classifier
            self.triad.ego_module.alpha = ssc.eta
            logger.info(f"  → Ego α synchronized to SSC η = {ssc.eta:.4f}")

        # Use SOC base temperature to calibrate Superego strictness
        if hasattr(self.engine, "soc") and self.engine.soc is not None:
            soc: SOCController = self.engine.soc
            base_lambda = soc.base_temp / 300.0 * self.config.lambda_reg
            self.config.lambda_reg = float(base_lambda)
            logger.info(f"  → λ calibrated from SOC base_temp: {base_lambda:.4f}")

        logger.info("✓ PSY ONE BRIDGE ↔ MENTAL ONE bridge established.")

    # ------------------------------------------------------------------
    def _extract_soc_id_entropy(self, eeg_state: torch.Tensor) -> float:
        """
        Extract Id entropy proxy from MENTAL ONE SOCController.

        SOC structural stress σ(x) ↔ H(𝓘) when MENTAL ONE is available.
        Falls back to PsycheTriad internal entropy otherwise.
        """
        if not HAS_MENTAL_ONE or self.engine is None:
            return float(self.triad.id_module.compute_entropy().item())

        if hasattr(self.engine, "soc") and self.engine.soc is not None:
            soc: SOCController = self.engine.soc
            sigma = soc.sigma(eeg_state.unsqueeze(0) if eeg_state.dim() == 1
                              else eeg_state)
            # Normalize σ → entropy-like value in [0, log2(action_dim)]
            max_h = math.log2(self.config.action_dim)
            h_proxy = float(torch.sigmoid(sigma).item()) * max_h
            return h_proxy

        return float(self.triad.id_module.compute_entropy().item())

    # ------------------------------------------------------------------
    def _run_mental_one_diagnosis(self, eeg_state: torch.Tensor) -> Tuple[str, Dict]:
        """
        Run MENTAL ONE classification and intervention design.

        Returns (diagnosis_str, intervention_plan_dict)
        """
        if not HAS_MENTAL_ONE or self.engine is None:
            return "Unknown (MENTAL ONE not connected)", {}

        try:
            result = self.engine.run(eeg_file=None)
            # If eeg_file is None the engine uses internal state — pass directly
        except Exception:
            pass

        # Direct SSCClassifier call
        try:
            if (hasattr(self.engine, "classifier")
                    and self.engine.classifier is not None):
                ssc: SSCClassifier = self.engine.classifier
                s = eeg_state.to(self.device).float().flatten()
                # Pad / truncate to expected size
                n_total = ssc.n_total
                if s.numel() < n_total:
                    s = F.pad(s, (0, n_total - s.numel()))
                else:
                    s = s[:n_total]
                s = s.clamp(0, 1)
                with torch.no_grad():
                    s_star   = ssc(s, n_iter=25, target="MDD", healthy="Healthy")
                    diagnosis = ssc.classify(s_star)

                # Intervention design
                plan: Dict[str, Any] = {}
                if hasattr(self.engine, "intervention"):
                    desired = getattr(ssc, "ref_Healthy", torch.zeros_like(s))
                    plan    = self.engine.intervention.design_plan(
                        diagnosis, s_star.cpu().numpy(), desired.cpu().numpy()
                    )
                return diagnosis, plan
        except Exception as e:
            logger.warning(f"MENTAL ONE classification error: {e}")

        return "Unknown", {}

    # ------------------------------------------------------------------
    def _build_sensory_from_eeg(self, eeg_state: torch.Tensor) -> torch.Tensor:
        """
        Project EEG state tensor → action_dim sensory feature vector.

        Feature extraction mirrors SSCClassifier.extract_features() but
        reduces to action_dim dimensions suitable for the Id Module.
        """
        s = eeg_state.to(self.device).float().flatten()

        # Compute frequency band powers as action-space features
        if s.numel() >= 32:
            fft   = torch.fft.rfft(s)
            freqs = torch.fft.rfftfreq(s.numel(), d=1 / 256)
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
            features = []
            for lo, hi in bands:
                mask  = (freqs >= lo) & (freqs < hi)
                power = torch.sum(torch.abs(fft[mask]) ** 2) / (mask.sum() + 1e-8)
                features.append(power)
            feat_vec = torch.stack(features)
        else:
            feat_vec = s[:min(s.numel(), 5)]

        # Pad / truncate to action_dim
        D = self.config.action_dim
        if feat_vec.numel() < D:
            feat_vec = feat_vec.repeat(math.ceil(D / feat_vec.numel()))[:D]
        else:
            feat_vec = feat_vec[:D]

        # Normalize to [0, 1]
        mn, mx = feat_vec.min(), feat_vec.max()
        feat_vec = (feat_vec - mn) / (mx - mn + 1e-12)
        return feat_vec

    # ------------------------------------------------------------------
    def run_psyche_cycle(
        self,
        eeg_or_sensory_state : torch.Tensor,
        emotional_salience   : Optional[torch.Tensor] = None,
        observation          : Optional[torch.Tensor] = None,
        actual_action        : Optional[int]          = None,
        normative_dist       : Optional[torch.Tensor] = None,
    ) -> PsycheTriadState:
        """
        Execute one complete PSY ONE ↔ MENTAL ONE psyche cycle.

        Parameters
        ----------
        eeg_or_sensory_state : EEG tensor (channels × timepoints) or sensory
                               feature vector (action_dim,)
        emotional_salience   : (action_dim,) emotional weighting; defaults to
                               uniform if None
        observation          : (action_dim,) external observation μ (optional)
        actual_action        : ground-truth action for error tracking (optional)
        normative_dist       : override societal baseline π_norm (optional)

        Returns
        -------
        PsycheTriadState with all psychic metrics and clinical data
        """
        self._cycle_count += 1
        if self.config.verbose:
            logger.info(f"Psyche cycle #{self._cycle_count} started.")

        # --- Prepare sensory input ---
        if eeg_or_sensory_state.numel() > self.config.action_dim:
            sensory_state = self._build_sensory_from_eeg(eeg_or_sensory_state)
        else:
            sensory_state = eeg_or_sensory_state.to(self.device).float()
            if sensory_state.numel() < self.config.action_dim:
                sensory_state = F.pad(
                    sensory_state,
                    (0, self.config.action_dim - sensory_state.numel()),
                )
            sensory_state = (sensory_state / (sensory_state.sum() + 1e-12)).clamp(0, 1)

        # --- Emotional salience ---
        if emotional_salience is None:
            emotional_salience = torch.ones(self.config.action_dim, device=self.device)
        emotional_salience = emotional_salience.to(self.device).float()
        if emotional_salience.numel() != self.config.action_dim:
            emotional_salience = F.interpolate(
                emotional_salience.unsqueeze(0).unsqueeze(0),
                size=self.config.action_dim, mode="linear", align_corners=True,
            ).squeeze()

        # --- Societal baseline ---
        if normative_dist is not None:
            self.triad.set_societal_baseline(normative_dist)

        # --- Id entropy from SOC (if MENTAL ONE available) ---
        soc_entropy = self._extract_soc_id_entropy(eeg_or_sensory_state)

        # --- Run PsycheTriad inference cycle ---
        state = self.triad.run_inference_cycle(
            sensory_state      = sensory_state,
            emotional_salience = emotional_salience,
            observation        = observation,
            actual_action      = actual_action,
        )

        # Override Id entropy with SOC-derived measure if available
        if HAS_MENTAL_ONE and self.engine is not None:
            state = PsycheTriadState(
                id_entropy          = soc_entropy,
                accumulated_entropy = state.accumulated_entropy,
                superego_loss       = state.superego_loss,
                behavioral_entropy  = state.behavioral_entropy,
                free_energy         = state.free_energy,
                selected_action     = state.selected_action,
                optimized_policy    = state.optimized_policy,
                ocd_loop_detected   = state.ocd_loop_detected,
                convergence_speed   = state.convergence_speed,
                fe_history          = state.fe_history,
                neurophysio_map     = state.neurophysio_map,
            )

        # --- MENTAL ONE diagnosis & intervention ---
        diagnosis, plan = self._run_mental_one_diagnosis(eeg_or_sensory_state)
        state.diagnosis         = diagnosis
        state.intervention_plan = plan

        if self.config.verbose:
            logger.info(
                f"  H(𝓘)={state.id_entropy:.4f}  "
                f"L_𝓢={state.superego_loss:.4f}  "
                f"ℱ={state.free_energy:.4f}  "
                f"a*={state.selected_action}  "
                f"Dx={state.diagnosis}"
            )

        return state

    # ------------------------------------------------------------------
    def batch_run(
        self,
        eeg_batch        : List[torch.Tensor],
        salience_batch   : Optional[List[torch.Tensor]] = None,
        reset_per_subject: bool = True,
    ) -> List[PsycheTriadState]:
        """
        Run psyche cycles over a batch of subjects.

        Parameters
        ----------
        eeg_batch         : list of EEG tensors, one per subject
        salience_batch    : optional list of salience vectors
        reset_per_subject : reset PsycheTriad state between subjects

        Returns
        -------
        List of PsycheTriadState, one per subject
        """
        results: List[PsycheTriadState] = []
        for i, eeg in enumerate(eeg_batch):
            if reset_per_subject:
                self.triad.reset()
            salience = salience_batch[i] if salience_batch else None
            state    = self.run_psyche_cycle(eeg, salience)
            results.append(state)
            if self.config.verbose:
                logger.info(f"  Subject {i+1}/{len(eeg_batch)}: {state.diagnosis}")
        return results

    # ------------------------------------------------------------------
    def generate_psychopathology_report(self, state: PsycheTriadState) -> str:
        """
        Generate a human-readable clinical psychopathology report.

        Maps PsycheTriadState metrics to clinical interpretation based on
        the theoretical thresholds defined in paper §5.3.
        """
        lines = [
            "=" * 70,
            "  PSY ONE BRIDGE  —  PSYCHOPATHOLOGY REPORT",
            "  Developer: Yoon A Limsuwan / MSPS NETWORK",
            "=" * 70,
            "",
            f"  MENTAL ONE Diagnosis   : {state.diagnosis}",
            f"  Psychopathology Mode   : {self.config.mode.value.upper()}",
            "",
            "  ── Informational Psyche Metrics ─────────────────────────",
            f"  H(𝓘)  Id Entropy        : {state.id_entropy:.4f} bits",
            f"  ∫H(𝓘) Accumulated       : {state.accumulated_entropy:.4f} bits",
            f"  L_𝓢   Superego Loss     : {state.superego_loss:.4f}",
            f"  H(B)  Behavioral Entropy: {state.behavioral_entropy:.4f} bits",
            f"  ℱ     Free Energy (final): {state.free_energy:.4f}",
            f"  a*    Selected Action   : {state.selected_action}",
            "",
            "  ── Optimization Diagnostics ─────────────────────────────",
            f"  OCD Loop Detected       : {'⚠ YES' if state.ocd_loop_detected else 'No'}",
            f"  Convergence Speed (α)   : {state.convergence_speed:.4f}",
            "",
            "  ── Neurophysiological Proxy Map (§5.2) ──────────────────",
        ]
        for k, v in state.neurophysio_map.items():
            lines.append(f"  {k:<28}: {v:.4f}")

        lines += [
            "",
            "  ── Clinical Interpretation ──────────────────────────────",
        ]

        # MDD/Anxiety: high superego loss
        if state.superego_loss > 3.0 and state.free_energy > 5.0:
            lines.append(
                "  ⚠ High L_𝓢 + high ℱ  →  Possible MDD/Anxiety profile "
                "(λ over-regulation)"
            )

        # Schizophrenia: high entropy + low convergence
        if state.id_entropy > 3.5 and state.convergence_speed < 0.05:
            lines.append(
                "  ⚠ H(𝓘) > 3.5 + low α  →  Possible Schizophrenia profile "
                "(Id entropy collapse)"
            )

        # OCD
        if state.ocd_loop_detected:
            lines.append(
                "  ⚠ OCD loop marker     →  Ego stuck in compulsive optimization "
                "(ℱ non-convergent)"
            )

        # Healthy
        if (
            state.id_entropy < 2.0
            and state.superego_loss < 1.0
            and not state.ocd_loop_detected
            and state.convergence_speed > 0.3
        ):
            lines.append("  ✓ Metrics within healthy range.")

        if state.intervention_plan:
            lines += [
                "",
                "  ── Intervention Plan (MENTAL ONE) ───────────────────────",
            ]
            for k, v in state.intervention_plan.items():
                lines.append(f"  {k}: {v}")

        lines += ["", "=" * 70]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.triad.reset()
        self._cycle_count = 0
        logger.info("PSYONEBridge state reset.")


# =============================================================================
# 6.  Longitudinal Tracker  —  Temporal Psyche Evolution
# =============================================================================

class LongitudinalPsycheTracker:
    """
    Tracks psyche state evolution across multiple time points / sessions.

    Enables:
      • Disease trajectory prediction (SOC + RG)
      • Monitoring of H(𝓘), L_𝓢, ℱ over time
      • Early detection of decompensation (entropy spikes)
      • Treatment response monitoring
    """

    def __init__(self, bridge: PSYONEBridge) -> None:
        self.bridge  = bridge
        self.history : List[PsycheTriadState] = []

    # ------------------------------------------------------------------
    def record(self, state: PsycheTriadState) -> None:
        self.history.append(state)

    # ------------------------------------------------------------------
    def run_and_record(
        self,
        eeg_state  : torch.Tensor,
        salience   : Optional[torch.Tensor] = None,
    ) -> PsycheTriadState:
        state = self.bridge.run_psyche_cycle(eeg_state, salience)
        self.record(state)
        return state

    # ------------------------------------------------------------------
    def entropy_trajectory(self) -> List[float]:
        return [s.id_entropy for s in self.history]

    def superego_trajectory(self) -> List[float]:
        return [s.superego_loss for s in self.history]

    def free_energy_trajectory(self) -> List[float]:
        return [s.free_energy for s in self.history]

    # ------------------------------------------------------------------
    def detect_decompensation(
        self,
        entropy_threshold : float = 3.5,
        window            : int   = 5,
    ) -> bool:
        """
        Return True if the last `window` cycles show H(𝓘) > threshold.
        Early decompensation marker — warrants clinical review.
        """
        if len(self.history) < window:
            return False
        tail = self.entropy_trajectory()[-window:]
        return all(h > entropy_threshold for h in tail)

    # ------------------------------------------------------------------
    def summarize(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        entropies    = self.entropy_trajectory()
        superegos    = self.superego_trajectory()
        free_energies = self.free_energy_trajectory()
        diagnoses    = [s.diagnosis for s in self.history]
        return {
            "n_cycles"            : len(self.history),
            "mean_id_entropy"     : round(float(np.mean(entropies)), 4),
            "max_id_entropy"      : round(float(np.max(entropies)), 4),
            "mean_superego_loss"  : round(float(np.mean(superegos)), 4),
            "mean_free_energy"    : round(float(np.mean(free_energies)), 4),
            "final_diagnosis"     : diagnoses[-1] if diagnoses else "Unknown",
            "decompensation_flag" : self.detect_decompensation(),
            "diagnosis_history"   : diagnoses,
        }


# =============================================================================
# 7.  Batch Benchmark Runner
# =============================================================================

class PSYONEBenchmark:
    """
    Runs structured benchmark experiments to validate the informational model.

    Implements the 3-step verification protocol from paper §5.4:
      Step 1: Empirical data gathering (simulated here with synthetic profiles)
      Step 2: Simulation parameter initialization per psychopathology mode
      Step 3: Goodness-of-fit testing (accuracy vs. expected profile)
    """

    DISORDER_PROFILES: Dict[str, Dict[str, Any]] = {
        "Healthy"       : {"mode": PsychopathologyMode.HEALTHY,       "expected_entropy": (0.5, 2.0)},
        "MDD_Anxiety"   : {"mode": PsychopathologyMode.MDD_ANXIETY,   "expected_entropy": (1.5, 3.5)},
        "Schizophrenia" : {"mode": PsychopathologyMode.SCHIZOPHRENIA, "expected_entropy": (3.0, 4.5)},
        "OCD"           : {"mode": PsychopathologyMode.OCD,           "expected_entropy": (1.5, 3.5)},
        "PTSD"          : {"mode": PsychopathologyMode.PTSD,          "expected_entropy": (2.0, 4.0)},
    }

    def __init__(self, action_dim: int = 10, n_subjects: int = 20) -> None:
        self.action_dim  = action_dim
        self.n_subjects  = n_subjects
        self.results     : Dict[str, List[PsycheTriadState]] = {}

    # ------------------------------------------------------------------
    def _generate_synthetic_eeg(
        self,
        mode  : PsychopathologyMode,
        n_ch  : int = 19,
        n_tp  : int = 256,
    ) -> torch.Tensor:
        """Generate synthetic EEG with disorder-specific spectral properties."""
        t = torch.linspace(0, 1, n_tp)
        eeg = torch.zeros(n_ch, n_tp)
        if mode == PsychopathologyMode.SCHIZOPHRENIA:
            # High-entropy: broadband noise
            eeg = torch.randn(n_ch, n_tp) * 2.0
        elif mode == PsychopathologyMode.MDD_ANXIETY:
            # Alpha suppression + beta excess
            for i in range(n_ch):
                eeg[i] = (0.3 * torch.sin(2 * math.pi * 10 * t)
                          + 1.2 * torch.sin(2 * math.pi * 20 * t)
                          + 0.2 * torch.randn(n_tp))
        elif mode == PsychopathologyMode.OCD:
            # Repetitive theta bursts
            for i in range(n_ch):
                eeg[i] = (1.5 * torch.sin(2 * math.pi * 6 * t)
                          + 0.3 * torch.randn(n_tp))
        elif mode == PsychopathologyMode.PTSD:
            # Hyper-arousal: elevated gamma
            for i in range(n_ch):
                eeg[i] = (0.5 * torch.sin(2 * math.pi * 35 * t)
                          + 0.8 * torch.randn(n_tp))
        else:
            # Healthy: dominant alpha
            for i in range(n_ch):
                eeg[i] = (1.0 * torch.sin(2 * math.pi * 10 * t)
                          + 0.3 * torch.randn(n_tp))
        return eeg.clamp(-5, 5)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        Execute full benchmark across all disorder profiles.

        Returns
        -------
        Dict with per-profile accuracy, mean metrics, and summary statistics.
        """
        logger.info("PSYONEBenchmark: starting benchmark run...")
        summary: Dict[str, Any] = {}

        for profile_name, profile in self.DISORDER_PROFILES.items():
            mode    = profile["mode"]
            lo, hi  = profile["expected_entropy"]
            config  = PsycheConfig(action_dim=self.action_dim, mode=mode)
            bridge  = PSYONEBridge(config=config)
            states  : List[PsycheTriadState] = []
            correct : int = 0

            for _ in range(self.n_subjects):
                eeg       = self._generate_synthetic_eeg(mode)
                salience  = torch.rand(self.action_dim)
                state     = bridge.run_psyche_cycle(eeg, salience)
                states.append(state)
                if lo <= state.id_entropy <= hi:
                    correct += 1
                bridge.triad.reset()

            accuracy = correct / self.n_subjects
            entropies = [s.id_entropy for s in states]
            self.results[profile_name] = states
            summary[profile_name] = {
                "accuracy"         : round(accuracy, 4),
                "mean_id_entropy"  : round(float(np.mean(entropies)), 4),
                "std_id_entropy"   : round(float(np.std(entropies)), 4),
                "expected_range"   : (lo, hi),
            }
            logger.info(
                f"  {profile_name:<16} accuracy={accuracy*100:.1f}%  "
                f"H(𝓘)={np.mean(entropies):.3f}±{np.std(entropies):.3f}"
            )

        # Overall accuracy (benchmark §5.4 target: 78–85%)
        overall_acc = float(np.mean([v["accuracy"] for v in summary.values()]))
        summary["__overall__"] = {
            "mean_accuracy" : round(overall_acc, 4),
            "n_profiles"    : len(self.DISORDER_PROFILES),
            "n_subjects"    : self.n_subjects,
            "target_range"  : "0.78–0.85",
        }
        logger.info(f"  Overall accuracy: {overall_acc*100:.1f}%")
        return summary


# =============================================================================
# 8.  Command-Line Interface
# =============================================================================

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        prog="psy_one_bridge",
        description="PSY ONE BRIDGE — Informational Psyche Engine for MENTAL ONE",
    )
    sub = parser.add_subparsers(dest="command")

    # simulate
    sim = sub.add_parser("simulate", help="Run a single psyche inference cycle")
    sim.add_argument("--mode", choices=[m.value for m in PsychopathologyMode],
                     default="healthy")
    sim.add_argument("--action_dim", type=int, default=10)
    sim.add_argument("--lambda_reg", type=float, default=2.5)
    sim.add_argument("--n_cycles", type=int, default=1)
    sim.add_argument("--verbose", action="store_true")

    # benchmark
    bench = sub.add_parser("benchmark", help="Run full benchmark suite")
    bench.add_argument("--action_dim", type=int, default=10)
    bench.add_argument("--n_subjects", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command is None:
        print("PSY ONE BRIDGE  |  Developer: Yoon A Limsuwan / MSPS NETWORK")
        print("Usage: python psy_one_bridge.py {simulate,benchmark} --help")
        return

    if args.command == "simulate":
        mode   = PsychopathologyMode(args.mode)
        config = PsycheConfig(
            action_dim  = args.action_dim,
            lambda_reg  = args.lambda_reg,
            mode        = mode,
            verbose     = args.verbose,
        )
        bridge = PSYONEBridge(config=config)
        norm   = torch.zeros(args.action_dim)
        norm[args.action_dim // 2] = 0.7
        norm   = F.softmax(norm, dim=0)
        bridge.triad.set_societal_baseline(norm)

        tracker = LongitudinalPsycheTracker(bridge)
        for cycle in range(args.n_cycles):
            eeg      = torch.randn(19, 256)
            salience = torch.rand(args.action_dim) + 0.1
            state    = tracker.run_and_record(eeg, salience)
            print(bridge.generate_psychopathology_report(state))

        if args.n_cycles > 1:
            print("\n── Longitudinal Summary ──")
            import json
            print(json.dumps(tracker.summarize(), indent=2))

    elif args.command == "benchmark":
        bench   = PSYONEBenchmark(
            action_dim = args.action_dim,
            n_subjects = args.n_subjects,
        )
        results = bench.run()
        import json
        print("\n── Benchmark Results ──")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
