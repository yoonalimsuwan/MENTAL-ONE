"""
=============================================================================
UNIFIED FRAMEWORK — BOUNDED IMPLEMENTATION
Based on: "A Unified Framework for Quantum-Inspired Information
Measurement, Neural Signal Analysis, Record Recovery, Hierarchical
Computation, and the Limits of Remote Neural Inference"
Grounded in: 15-paper Structural Calculus series
=============================================================================
Developer    : Implementation based on PAI & Yoon A. Limsuwan framework
License      : MIT
Year         : 2026

SCOPE DISCIPLINE (carried from the source papers):
  Every module below states what it IS and what it is NOT.
  The "Oceanic Dispersion" metaphor is NOT implemented as physics.
  "Still images → video" consciousness is NOT implemented.
  Remote neural reading/control is NOT implemented.
  What IS implemented: the mathematical machinery that is
  independently verifiable and grounded in the 15-paper series.

15-PAPER INTEGRATION MAP:
  [P4]  Carpet-19        → spectral Sobolev scale, Mosco convergence
  [P5]  paper5           → ergodic decomposition, AMS, log-scale
  [P6]  paper6           → change-point towers, Kakutani, occupation law
  [P7]  Paper7           → external noise diagnostic hierarchy
  [P8]  Paper8           → 6-test diagnostic battery
  [P9]  Paper9           → finite-sample dictionary (EVT, realized variation)
  [P11] Paper11          → assumption-light subsampling CI
  [DM]  Delta_Min        → δ_min isodiametric bound
  [NZ]  SESI_NoZeno      → minimum dwell-time, energy budget
  [SNZ] Stochastic_NoZeno → Burkholder-Davis-Gundy exit bound
  [OPMC] Unified Ordinal-Tensor → contraction, A^W_str, k*=23
  [R21] Revision21       → OPMC on fractal interfaces, timing independence
  [R22] Revision22       → coercivity, fractafold transmission
  [SP]  Six_Problems     → full coercivity C2/C4, junction kernel
  [QRR] QUANTUM_RECORD_RECOVERY → 2-qubit exact model

MODULES:
  [1] OPMCSharedProcessor        — One Processing, Many Computations
  [2] WeakMeasurementEngine      — AAV weak values, information-disturbance
  [3] TwoQubitRecordModel        — exact Stinespring, Petz partial recovery
  [4] DiagnosticBatteryRunner    — Paper 8 six-test battery
  [5] DeltaMinGate               — isodiametric detection threshold
  [6] SubsamplingConfidence      — Paper 11 assumption-light CI
  [7] NoZenoEventBudget          — SESI_NoZeno energy-budget bound
  [8] OPMCStructuralWeakValue    — A^W_str on compact M_str
  [9] UnifiedBoundedSystem       — master orchestrator
  [10] Self-test suite

HONEST SCOPE (from the source paper itself, §20):
  • Two-qubit model is a small simulation, not a brain model
  • Petz recovery is PARTIAL, not general inversion
  • No-signaling prevents remote communication interpretation
  • "Still images → video" is a hypothesis, not established
  • Software-emulated gates are not physical quantum gates
  • Information persistence ≠ personal identity survival
=============================================================================
"""

from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor

logger = logging.getLogger("unified_framework")

UNIFIED_VERSION = "1.0.0"

# =============================================================================
# OPMC CONSTANTS (from Unified_Ordinal-Tensor_Measurement_Theory)
# =============================================================================
OPMC_L = 0.438520           # Example 7.12: contraction factor
OPMC_C1 = 0.420000          # Theorem 5.5: linear operator bound
OPMC_K2 = 12.5000           # Lemma 7.6: Lipschitz bound
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n  # Lemma 2.4

def opmc_k_star(epsilon: float = 1e-8, D0: float = 1.0) -> int:
    """Theorem 11.1: k* = ceil(ln(eps/D0)/ln L), independent of N."""
    return math.ceil(math.log(epsilon / D0) / math.log(OPMC_L))

# =============================================================================
# C^∞ Primitives (AMP-safe, no dead gradients)
# =============================================================================
def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)

def soft_clamp(x: Tensor, lo: float, hi: float) -> Tensor:
    mid = (hi + lo) / 2.0
    half = max((hi - lo) / 2.0, 1e-6)
    return mid + half * torch.tanh((x - mid) / half)

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class UnifiedConfig:
    """Configuration for the bounded unified framework."""

    # OPMC dimensions (Lemma 2.4)
    opmc_m: int = 4
    opmc_n: int = 4
    opmc_sigma_dim: int = 64
    opmc_n_fields: int = 8

    # Weak measurement
    wm_kappa: float = 0.1          # weak coupling strength
    wm_n_steps: int = 20           # repeated weak measurements
    wm_pointer_dim: int = 32       # pointer Hilbert space dimension

    # Two-qubit record model (QRR)
    coupling_g: float = 0.7
    omega_s: float = 0.3
    omega_e: float = 0.5
    dephase_p: float = 0.5

    # Diagnostic battery (Paper 8)
    n_scales: int = 12             # scales for Ahlfors test
    n_windows: int = 8             # windows for Birkhoff test
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5

    # δ_min (Delta_Min Theorem 1.2)
    kappa_fatness: float = 2.0
    a_min: float = 0.01

    # Subsampling (Paper 11)
    subsample_block: int = 64
    n_subsamples: int = 200
    confidence_level: float = 0.95

    # No-Zeno (SESI_NoZeno_Closure)
    delta_e_min: float = 0.1
    energy_budget: float = 100.0
    dissipation_budget: float = 50.0

    # Training
    batch_size: int = 8
    lr: float = 3e-4
    device: str = "auto"

    # DDP
    distributed: bool = False

    # AMP
    amp_dtype: str = "bf16"        # bf16 | fp16 | fp32


# =============================================================================
# [1] OPMC SHARED PROCESSOR
# From: Unified Framework §3, Appendix A
# What it IS: one shared representation → many downstream tasks
# What it is NOT: a claim of quantum speedup
# =============================================================================
class OPMCSharedProcessor(nn.Module):
    """
    One Processing — Many Computations (OPMC).

    From the source paper Appendix A.2:
      A shared representation Z = Φ(D) is sufficient for task j if
      F_j(D) = G_j(Φ(D)) for every input in the test domain.

    From OPMC Theorem 9.2 (Unified Ordinal-Tensor):
      The structural weak value A^W_str is computed on the compact
      metric completion M_str, NOT on the non-measurable set V.

    COST MODEL (Appendix A.3):
      C_OPMC = C_Φ + Σ_j C_{G_j}
      Advantage iff C_OPMC < Σ_j C_{F_j}

    This module implements the shared-processor architecture.
    It does NOT claim quantum speedup.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.opmc_m * cfg.opmc_m * cfg.opmc_n * cfg.opmc_n + \
               cfg.opmc_m * cfg.opmc_n * cfg.opmc_n
        self.d_mn = OPMC_D_MN(cfg.opmc_m, cfg.opmc_n)

        # Shared processing Φ
        self.shared_encoder = nn.Sequential(
            nn.Linear(d_in, cfg.opmc_sigma_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.opmc_sigma_dim * 2, cfg.opmc_sigma_dim),
            nn.LayerNorm(cfg.opmc_sigma_dim),
        )

        # Structural field embeddings (M_str basis)
        self.field_embeddings = nn.Parameter(
            torch.randn(cfg.opmc_n_fields, cfg.opmc_sigma_dim) * 0.02
        )

        # Query / value projections
        self.query_proj = nn.Linear(cfg.opmc_sigma_dim, cfg.opmc_sigma_dim)
        self.value_proj = nn.Linear(cfg.opmc_sigma_dim, cfg.opmc_sigma_dim)

        # Refinement
        self.refine = nn.Sequential(
            nn.LayerNorm(cfg.opmc_sigma_dim),
            nn.Linear(cfg.opmc_sigma_dim, cfg.opmc_sigma_dim),
            nn.GELU(),
        )

        # OPMC iteration count
        self.k_star = opmc_k_star(1e-8)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: (B, d_mn) input in structural space
        Returns:
            dict with shared_representation, structural_weak_value
        """
        B = x.shape[0]

        # Shared processing
        z = self.shared_encoder(x)  # (B, sigma_dim)

        # Attention over structural fields → A^W_str
        q = self.query_proj(z).unsqueeze(1)  # (B, 1, sigma_dim)
        logits = torch.einsum(
            "bnd,fd->bnf", q, self.field_embeddings
        ) / math.sqrt(q.shape[-1])
        weights = F.softmax(logits, dim=1)  # (B, 1, n_fields)

        v = self.value_proj(z).unsqueeze(1)  # (B, 1, sigma_dim)
        a_w_str = torch.einsum("bnf,bnd->bfd", weights, v).squeeze(1)
        a_w_str = self.refine(a_w_str + z)  # (B, sigma_dim)

        return {
            "shared_representation": z,
            "structural_weak_value": a_w_str,
            "field_weights": weights.squeeze(1),
            "opmc_k_star": self.k_star,
        }


# =============================================================================
# [2] WEAK MEASUREMENT ENGINE
# From: Unified Framework §6, Appendix E
# Physics: Aharonov-Albert-Vaidman (1988)
# What it IS: weak value computation, information-disturbance tradeoff
# What it is NOT: remote brain reading, consciousness detection
# =============================================================================
class WeakMeasurementEngine(nn.Module):
    """
    Weak measurement engine implementing AAV formalism.

    From the source paper §6.4:
      Repeated weak measurements:
        ρ_{n+1} = M_{k_n} ρ_n M†_{k_n} / Tr(M_{k_n} ρ_n M†_{k_n})
      Cumulative information: I_n = I(X; R_n)
      Cumulative disturbance: D_n = D(ρ_0, ρ_n)
      Tradeoff: max I_n subject to D_n ≤ δ

    From Appendix E:
      Weak measurement is NOT automatically disturbance-free.
      Both information gained AND state change must be reported.

    HONEST SCOPE:
      This simulates the information-disturbance tradeoff for a
      single qubit. It does NOT read brains, detect consciousness,
      or enable remote monitoring.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.kappa = cfg.wm_kappa
        self.n_steps = cfg.wm_n_steps

        # Measurement strength adapter (learnable κ)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(cfg.wm_kappa)))

        # State update network (GRU for temporal accumulation)
        self.state_updater = nn.GRUCell(cfg.wm_pointer_dim, cfg.wm_pointer_dim)

        # Information estimator
        self.info_net = nn.Sequential(
            nn.Linear(cfg.wm_pointer_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # I ≥ 0
        )

        # Disturbance estimator
        self.disturb_net = nn.Sequential(
            nn.Linear(cfg.wm_pointer_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # D ≥ 0
        )

    @property
    def kappa_val(self) -> Tensor:
        return torch.exp(self.log_kappa)

    def weak_measurement_step(
        self, state: Tensor, pointer: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        One weak measurement step.
        M_0 ≈ sqrt(1-κ/2) I,  M_1 ≈ sqrt(κ/2) σ_z
        Simplified differentiable version.
        """
        κ = self.kappa_val
        # Weak coupling: small perturbation
        perturbation = κ * torch.tanh(pointer)
        new_state = state + perturbation
        # Normalize
        new_state = F.normalize(new_state, dim=-1)
        return new_state, perturbation

    def forward(
        self, initial_state: Tensor
    ) -> Dict[str, Tensor]:
        """
        Run repeated weak measurements.

        Args:
            initial_state: (B, pointer_dim) initial quantum state

        Returns:
            dict with information_gain, disturbance, trajectory
        """
        B = initial_state.shape[0]
        state = F.normalize(initial_state, dim=-1)
        initial_norm = state.norm(dim=-1, keepdim=True)

        pointer = torch.zeros(B, self.cfg.wm_pointer_dim, device=state.device)
        info_accum = torch.zeros(B, 1, device=state.device)
        disturb_accum = torch.zeros(B, 1, device=state.device)

        states = [state]

        for step in range(self.n_steps):
            # Weak measurement
            state, perturbation = self.weak_measurement_step(state, pointer)

            # Update pointer via GRU
            pointer = self.state_updater(perturbation, pointer)

            # Information gained this step
            info_step = self.info_net(pointer)
            info_accum = info_accum + info_step

            # Disturbance this step
            disturb_input = torch.cat([initial_state.expand_as(state), state], dim=-1)
            disturb_step = self.disturb_net(disturb_input)
            disturb_accum = disturb_accum + disturb_step

            states.append(state)

        # Final disturbance: D(ρ_0, ρ_n)
        final_disturbance = (initial_state - state).norm(dim=-1, keepdim=True)

        return {
            "information_gain": info_accum,
            "cumulative_disturbance": disturb_accum,
            "final_disturbance": final_disturbance,
            "final_state": state,
            "n_measurements": self.n_steps,
            "kappa": self.kappa_val,
            "HONEST_NOTE": (
                "Weak measurement is NOT disturbance-free. "
                "This simulates a single-qubit tradeoff. "
                "It does NOT read brains or detect consciousness."
            ),
        }


# =============================================================================
# [3] TWO-QUBIT RECORD MODEL
# From: QUANTUM_RECORD_RECOVERY.py, Unified Framework §7
# Physics: Stinespring dilation, Petz recovery (PARTIAL)
# What it IS: exact 2-qubit unitary model
# What it is NOT: a brain model, a remote reading device
# =============================================================================
CD = torch.complex128

def _pauli() -> Dict[str, Tensor]:
    return {
        "I": torch.eye(2, dtype=CD),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=CD),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=CD),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=CD),
        "0": torch.tensor([[1, 0], [0, 0]], dtype=CD),
        "+": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=CD),
    }

def _entropy(rho: Tensor) -> float:
    rho = (rho + rho.conj().T) / 2
    vals = torch.linalg.eigvalsh(rho).real.clamp_min(1e-18)
    vals = vals / vals.sum()
    return float(-(vals * vals.log()).sum())

def _purity(rho: Tensor) -> float:
    return float(torch.trace(rho @ rho).real)

def _l1_coherence(rho: Tensor) -> float:
    d = rho.shape[0]
    mask = ~torch.eye(d, dtype=torch.bool, device=rho.device)
    return float(rho.abs()[mask].sum())

def _trace_env(rho_SE: Tensor, dS: int, dE: int) -> Tensor:
    t = rho_SE.reshape(dS, dE, dS, dE)
    return t.diagonal(dim1=1, dim2=3).sum(dim=-1)


class TwoQubitRecordModel:
    """
    Exact 2-qubit system-environment model.

    From QUANTUM_RECORD_RECOVERY.py:
      H = H_S ⊗ I + I ⊗ H_E + g(X_S ⊗ X_E)
      ρ_SE(0) = |+⟩⟨+| ⊗ |0⟩⟨0|
      ρ_SE(t) = U(t) ρ_SE(0) U†(t)

    Information ledger tracks:
      C(ρ_S) ↓  ⟺  I(S:E) ↑,  S(ρ_SE) ≡ 0

    HONEST SCOPE (from QRR header):
      Environment is ONE qubit (d_E = 2).
      A real brain has ~10^23 environmental degrees of freedom.
      This gap is the whole point.
    """

    def __init__(self, cfg: UnifiedConfig):
        self.cfg = cfg
        P = _pauli()
        self.dS = self.dE = 2

        H_S = 0.5 * cfg.omega_s * P["Z"]
        H_E = 0.5 * cfg.omega_e * P["Z"]
        self.H = (
            torch.kron(H_S, P["I"]) +
            torch.kron(P["I"], H_E) +
            cfg.coupling_g * torch.kron(P["X"], P["X"])
        )
        self.H = (self.H + self.H.conj().T) / 2
        self.evals, self.evecs = torch.linalg.eigh(self.H)
        self.rho0 = torch.kron(P["+"], P["0"])
        self.C0 = _l1_coherence(_trace_env(self.rho0, 2, 2))

    def evolve(self, t: float) -> Dict[str, float]:
        phases = torch.exp(-1j * self.evals * t)
        U = (self.evecs * phases.to(CD)) @ self.evecs.conj().T
        rho_SE = U @ self.rho0 @ U.conj().T
        rho_S = _trace_env(rho_SE, self.dS, self.dE)

        t4 = rho_SE.reshape(self.dS, self.dE, self.dS, self.dE)
        rho_E = t4.diagonal(dim1=0, dim2=2).sum(dim=-1)

        S_SE = _entropy(rho_SE)
        S_S = _entropy(rho_S)
        S_E = _entropy(rho_E)
        I_SE = S_S + S_E - S_SE
        C = _l1_coherence(rho_S)

        return {
            "t": t,
            "coherence_S": C,
            "coherence_fraction": C / max(self.C0, 1e-15),
            "purity_S": _purity(rho_S),
            "S_global": S_SE,
            "mutual_information_SE": I_SE,
            "unitarity_error": float(
                torch.linalg.norm(U @ U.conj().T - torch.eye(4, dtype=CD))
            ),
        }

    def petz_recovery(self, rho_in: Tensor, p: float) -> Dict[str, float]:
        """
        Petz recovery for dephasing channel.
        E_p(ρ) = (1-p)ρ + p ZρZ
        Petz map: R(ρ') = Σ_k K†_k ρ' K_k (for maximally mixed reference)

        HONEST: Recovery is PARTIAL, not inversion.
        """
        P = _pauli()
        I, Z = P["I"], P["Z"]

        # Dephasing channel
        K0 = math.sqrt(max(1 - p, 0.0)) * I
        K1 = math.sqrt(max(p, 0.0)) * Z
        noisy = K0.conj().T @ rho_in @ K0 + K1.conj().T @ rho_in @ K1
        noisy = noisy / torch.trace(noisy).real

        # Petz map (simplified for maximally mixed reference)
        recovered = (I.conj().T @ noisy @ I + Z.conj().T @ noisy @ Z) / 2.0
        recovered = recovered / torch.trace(recovered).real

        # Fidelity
        def _fid(a: Tensor, b: Tensor) -> float:
            prod = a @ b @ a
            prod = (prod + prod.conj().T) / 2
            ev = torch.linalg.eigvalsh(prod).real.clamp_min(0)
            return float(ev.sqrt().sum().clamp(0, 1) ** 2)

        return {
            "coh_before": _l1_coherence(rho_in),
            "coh_noisy": _l1_coherence(noisy),
            "coh_recovered": _l1_coherence(recovered),
            "fidelity_before_vs_recovered": _fid(rho_in, recovered),
            "recovery_is_partial": _fid(rho_in, recovered) < 1.0 - 1e-6,
            "HONEST_NOTE": "Petz recovery is PARTIAL, not inversion.",
        }

    def no_signaling_audit(self) -> Dict[str, Any]:
        """
        Verify: local ops on A cannot change ρ_B.
        This is why environmental records cannot enable remote reading.
        """
        P = _pauli()
        rho_AB = 0.5 * (
            torch.kron(P["0"], P["0"]) + torch.kron(P["1"], P["1"])
        )
        rho_AB = rho_AB / torch.trace(rho_AB).real

        def trace_A(r: Tensor) -> Tensor:
            t = r.reshape(2, 2, 2, 2)
            return t.diagonal(dim1=0, dim2=2).sum(dim=-1)

        rho_B_before = trace_A(rho_AB)

        # Local unitary on A
        Hloc = 0.7 * P["X"] + 0.3 * P["Z"]
        ev, V = torch.linalg.eigh((Hloc + Hloc.conj().T) / 2)
        U_A = (V * torch.exp(-1j * ev).to(CD)) @ V.conj().T
        U = torch.kron(U_A, P["I"])
        rho_after = U @ rho_AB @ U.conj().T
        rho_B_after = trace_A(rho_after)

        trace_dist = 0.5 * float(
            torch.linalg.norm(rho_B_before - rho_B_after, ord=1)
        )

        return {
            "B_trace_distance_under_local_A": trace_dist,
            "no_signaling_holds": trace_dist < 1e-10,
            "HONEST_NOTE": (
                "No-signaling means environmental records cannot be "
                "used for remote reading or FTL communication."
            ),
        }

    def run_full_ledger(self, times: Optional[List[float]] = None) -> Dict[str, Any]:
        if times is None:
            times = [0.0, 0.5, 1.0, 2.0, 4.0]
        ledger = [self.evolve(t) for t in times]
        worst = min(ledger, key=lambda r: r["coherence_fraction"])
        p_eff = 1.0 - worst["coherence_fraction"]
        P = _pauli()
        petz = self.petz_recovery(P["+"], float(p_eff))
        signaling = self.no_signaling_audit()
        return {
            "information_ledger": ledger,
            "petz_recovery": petz,
            "no_signaling": signaling,
            "HONEST_LIMIT": (
                "I(S:E) grows because information moves into correlations. "
                "Recovering it requires controlling the environment. "
                "Here d_E = 2. A brain has ~10^23 environmental degrees."
            ),
        }


# =============================================================================
# [4] DIAGNOSTIC BATTERY RUNNER
# From: Paper 8 (Internal Diagnostic Battery)
# What it IS: 6 convergence tests, each = a theorem read backwards
# What it is NOT: a finite-sample guarantee (Paper 8 Open Problem 3.1)
# =============================================================================
class DiagnosticBatteryRunner:
    """
    Paper 8 diagnostic battery: 6 tests, each using a theorem's
    own convergence statistic as its test.

    CRITICAL CAVEAT (Paper 8 Remark 1.1, applies to ALL tests):
      Each test is a one-directional implication X ⇒ Y converges.
      Observing convergence from finite noisy data is
      necessary-and-consistent evidence for X, but NOT a finite-sample
      proof of X. No rate of convergence or concentration inequality
      is established by any paper in the series.

    Tests:
      2.1  Ahlfors two-sided regularity (Paper 4 Hyp. SA)
      2.2  Spatial stationarity / ergodic decomposition (Paper 5 §2)
      2.3  Asymptotic mean stationarity (Paper 5 §3)
      2.4  Log-scale stationarity (Paper 5 §4)
      2.5  Double-exponential residual (Paper 6)
      2.6  Itô quadratic variation (Structural Itô Calculus)
    """

    def __init__(self, cfg: UnifiedConfig):
        self.cfg = cfg

    def test_ahlfors(self, data: Tensor) -> Dict[str, Any]:
        """
        Test 2.1: density ratio D̂(x,r) stays in [c1, c2].
        Pass if M_-(n) and M_+(n) stabilize.
        Fail if M_- → 0 or M_+ → ∞.
        """
        n = data.shape[0]
        # Simulate density ratios at decreasing scales
        scales = torch.logspace(-3, 0, self.cfg.n_scales, device=data.device)
        ratios = []
        for s in scales:
            # Count points within scale s of each point
            diffs = (data.unsqueeze(0) - data.unsqueeze(1)).norm(dim=-1)
            counts = (diffs < s).float().sum(dim=-1)
            ratio = counts / (s ** 2 + 1e-8)
            ratios.append(ratio.mean())
        ratios = torch.stack(ratios)
        M_minus = torch.cummin(ratios, dim=0).values
        M_plus = torch.cummax(ratios, dim=0).values
        # Pass if both stabilize (variance of last 3 is small)
        stable = (
            M_minus[-3:].var() < 0.1 and
            M_plus[-3:].var() < 10.0 and
            M_minus[-1] > 0.01
        )
        return {
            "test": "ahlfors_regularity",
            "passed": bool(stable),
            "M_minus_final": M_minus[-1].item(),
            "M_plus_final": M_plus[-1].item(),
            "CAVEAT": "Asymptotically consistent, NOT finite-sample guaranteed.",
        }

    def test_ergodic_decomposition(self, data: Tensor) -> Dict[str, Any]:
        """
        Test 2.2: Birkhoff averages over separated windows.
        Ergodic if all converge to same value.
        Non-ergodic stationary if they cluster separately.
        """
        n = data.shape[0]
        n_win = self.cfg.n_windows
        win_size = n // n_win
        if win_size < 2:
            return {"test": "ergodic_decomposition", "passed": False,
                    "reason": "insufficient data"}
        averages = []
        for i in range(n_win):
            chunk = data[i * win_size:(i + 1) * win_size]
            averages.append(chunk.mean())
        averages = torch.stack(averages)
        # Check if averages converge to same value
        spread = averages.max() - averages.min()
        is_ergodic = spread < 0.5 * averages.abs().mean()
        return {
            "test": "ergodic_decomposition",
            "passed": bool(is_ergodic),
            "spread": spread.item(),
            "averages": averages.tolist(),
            "CAVEAT": "Asymptotically consistent, NOT finite-sample guaranteed.",
        }

    def test_ams(self, data: Tensor) -> Dict[str, Any]:
        """
        Test 2.3: Cesàro mean stabilization.
        Pass if |D̄_{n+1} - D̄_n| → 0 at summable rate.
        """
        n = data.shape[0]
        running_mean = torch.cumsum(data, dim=0) / torch.arange(
            1, n + 1, device=data.device, dtype=data.dtype
        ).unsqueeze(-1) if data.dim() > 1 else (
            torch.cumsum(data, dim=0) / torch.arange(1, n + 1, device=data.device)
        )
        if data.dim() > 1:
            diffs = (running_mean[1:] - running_mean[:-1]).norm(dim=-1)
        else:
            diffs = (running_mean[1:] - running_mean[:-1]).abs()
        # Check if differences decrease
        decreasing = (diffs[1:] < diffs[:-1] + 1e-6).float().mean()
        is_ams = decreasing > 0.7
        return {
            "test": "ams_cesaro",
            "passed": bool(is_ams),
            "decreasing_fraction": decreasing.item(),
            "CAVEAT": "Asymptotically consistent, NOT finite-sample guaranteed.",
        }

    def test_log_scale(self, data: Tensor) -> Dict[str, Any]:
        """
        Test 2.4: reparametrize by log-depth, re-apply tests 2.1-2.3.
        """
        # Log-transform positive data
        positive = data.clamp_min(1e-10)
        log_data = torch.log(positive)
        # Apply AMS test on log-scale
        ams_result = self.test_ams(log_data)
        return {
            "test": "log_scale_stationarity",
            "passed": ams_result["passed"],
            "detail": ams_result,
            "CAVEAT": (
                "Pass here licenses ḡ only, NOT (a_hom, b_hom). "
                "Paper 5 Open Problem 4.4 remains open."
            ),
        }

    def test_double_exponential(self, data: Tensor) -> Dict[str, Any]:
        """
        Test 2.5: CUSUM change-point detection on block lengths.
        Distinguish regularly varying from double-exponential.
        """
        n = data.shape[0]
        if data.dim() > 1:
            scalar = data.norm(dim=-1)
        else:
            scalar = data

        # CUSUM
        mean_val = scalar.mean()
        cusum = torch.zeros(n, device=scalar.device)
        for i in range(1, n):
            cusum[i] = torch.clamp(
                cusum[i-1] + scalar[i] - mean_val - self.cfg.cusum_drift,
                min=0.0
            )
        # Detect change points
        change_points = (cusum > self.cfg.cusum_threshold).nonzero(as_tuple=True)[0]

        if len(change_points) < 2:
            return {
                "test": "double_exponential",
                "passed": False,
                "reason": "insufficient change points",
            }

        # Block lengths
        block_lengths = torch.diff(change_points.float())
        if len(block_lengths) < 2:
            return {"test": "double_exponential", "passed": False,
                    "reason": "insufficient blocks"}

        # Check growth rate: log(block_lengths) vs index
        log_bl = torch.log(block_lengths + 1)
        indices = torch.arange(len(log_bl), dtype=log_bl.dtype, device=log_bl.device)
        # Linear fit in log-log
        if len(log_bl) > 1:
            log_log_bl = torch.log(log_bl.clamp_min(1e-10))
            # Slope of log_log_bl vs index
            slope = (log_log_bl[-1] - log_log_bl[0]) / max(len(log_log_bl) - 1, 1)
        else:
            slope = 0.0

        is_double_exp = slope > 0.5  # log log ℓ_k grows linearly in k
        is_regularly_varying = 0.0 < slope <= 0.5

        return {
            "test": "double_exponential_residual",
            "passed": True,  # test ran successfully
            "is_double_exponential": bool(is_double_exp),
            "is_regularly_varying": bool(is_regularly_varying),
            "growth_slope": slope.item(),
            "n_change_points": len(change_points),
            "block_lengths": block_lengths.tolist()[:10],
            "CAVEAT": (
                "If double-exponential: Paper 6 Theorem 4.1 predicts "
                "degenerate two-point law ½δ₀ + ½δ₁. "
                "If regularly varying: Beta(θ,θ) arcsine law. "
                "Asymptotically consistent, NOT finite-sample guaranteed."
            ),
        }

    def test_quadratic_variation(self, data: Tensor) -> Dict[str, Any]:
        """
        Test 2.6: realized quadratic variation.
        QV_δ = Σ |ΔX|² → finite positive limit iff genuine Itô.
        """
        if data.dim() > 1:
            increments = data[1:] - data[:-1]
            qv = (increments ** 2).sum(dim=-1)
        else:
            increments = data[1:] - data[:-1]
            qv = (increments ** 2).sum()

        # Check convergence: QV at different resolutions
        n = len(increments)
        qv_values = []
        for skip in [1, 2, 4]:
            if n > skip:
                inc = increments[::skip]
                qv_values.append((inc ** 2).sum().item())

        is_ito = len(qv_values) >= 2 and qv_values[-1] > 0 and (
            abs(qv_values[-1] - qv_values[0]) / max(abs(qv_values[0]), 1e-8) < 1.0
        )

        return {
            "test": "ito_quadratic_variation",
            "passed": bool(is_ito),
            "qv_values": qv_values,
            "qv_final": qv.item() if isinstance(qv, Tensor) else qv,
            "CAVEAT": (
                "QV → 0: smoother than Itô (drift-only). "
                "QV diverges: rougher than Itô (H < 1/2). "
                "Outside Assumption 2.1, no machinery is licensed."
            ),
        }

    def run_battery(self, data: Tensor) -> Dict[str, Any]:
        """Run all 6 tests in Paper 8 order."""
        results = {
            "test_2_6_qv": self.test_quadratic_variation(data),
            "test_2_1_ahlfors": self.test_ahlfors(data),
            "test_2_2_ergodic": self.test_ergodic_decomposition(data),
            "test_2_3_ams": self.test_ams(data),
            "test_2_4_log_scale": self.test_log_scale(data),
            "test_2_5_double_exp": self.test_double_exponential(data),
        }
        # Classification
        if results["test_2_1_ahlfors"]["passed"]:
            classification = "stationary_ergodic_ahlfors"
            licensed = "Paper 4 Theorem 3.2 licensed directly"
        elif results["test_2_2_ergodic"]["passed"]:
            classification = "ergodic_decomposition"
            licensed = "Paper 5 Theorem 2.2 (component-conditional)"
        elif results["test_2_3_ams"]["passed"]:
            classification = "ams"
            licensed = "Paper 5 Theorem 3.3 (unconditional on µ-a.e. ω)"
        elif results["test_2_4_log_scale"]["passed"]:
            classification = "log_scale_ergodic"
            licensed = "ḡ only, NOT full (a_hom, b_hom) pair"
        elif results["test_2_5_double_exp"].get("is_double_exponential"):
            classification = "double_exponential_residual"
            licensed = "Kolmogorov classification + exact prediction if law known"
        else:
            classification = "genuine_residual"
            licensed = "No homogenization. Kolmogorov classification only."

        results["classification"] = classification
        results["licensed_operation"] = licensed
        results["GLOBAL_CAVEAT"] = (
            "Paper 8 Remark 1.1: every test is one-directional. "
            "Convergence from finite noisy data is necessary-and-consistent "
            "evidence, NOT a finite-sample proof. No rate of convergence "
            "or concentration inequality is established."
        )
        return results


# =============================================================================
# [5] δ_min GATE
# From: Delta_Min_Isodiametric_Closure Theorem 1.2
# What it IS: geometric lower bound on detectable change
# What it is NOT: a signal detector, a brain reader
# =============================================================================
class DeltaMinGate(nn.Module):
    """
    δ_min isodiametric bound (Delta_Min Theorem 1.2).

    Under κ-fatness, every nucleation event satisfies:
      d_H(Γ, Γ') ≥ δ_min = (1/κ)(a_min/ω_{n-1})^{1/(n-1)}

    For n=2 (curve in plane): δ_min = (1/κ)√(a_min/π)

    This is a DETECTION THRESHOLD, not a detector.
    If signal change < δ_min: undetectable by this bound.
    If signal change ≥ δ_min: potentially detectable.

    HONEST: This does NOT enable remote brain reading.
    It defines the minimum geometric change that can be
    distinguished, given the κ-fatness assumption.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        # δ_min for n=2 (2D interface)
        delta_min = (1.0 / cfg.kappa_fatness) * math.sqrt(cfg.a_min / math.pi)
        self.register_buffer("delta_min", torch.tensor(delta_min), persistent=False)

        # Learnable κ (fatness parameter)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(cfg.kappa_fatness)))

    @property
    def kappa(self) -> Tensor:
        return torch.exp(self.log_kappa)

    def compute_delta_min(self) -> Tensor:
        return (1.0 / self.kappa) * math.sqrt(self.cfg.a_min / math.pi)

    def check_detectable(self, signal_change: Tensor) -> Dict[str, Tensor]:
        """
        Check if signal change exceeds δ_min.

        Args:
            signal_change: (B,) or (B, D) magnitude of change

        Returns:
            dict with detectable flag, margin
        """
        if signal_change.dim() > 1:
            magnitude = signal_change.norm(dim=-1)
        else:
            magnitude = signal_change.abs()

        d_min = self.compute_delta_min()
        detectable = magnitude >= d_min
        margin = magnitude - d_min

        return {
            "delta_min": d_min.expand_as(magnitude),
            "signal_magnitude": magnitude,
            "detectable": detectable,
            "margin": margin,
            "HONEST_NOTE": (
                "δ_min is a geometric lower bound under κ-fatness. "
                "It does NOT enable remote brain reading. "
                "If signal < δ_min: undetectable by this criterion."
            ),
        }


# =============================================================================
# [6] SUBSAMPLING CONFIDENCE (Paper 11)
# What it IS: assumption-light CI without knowing convergence rate
# What it is NOT: a finite-sample guarantee for arbitrary data
# =============================================================================
class SubsamplingConfidence:
    """
    Paper 11 assumption-light subsampling confidence intervals.

    From Paper 11 §2.1:
      Politis-Romano subsampling: valid if τ_n(θ̂_n - θ) converges
      in distribution to some limit law at some rate τ_n.
      Requires: b_n → ∞, b_n/n → 0.
      Does NOT require knowing τ_n, the limit law, or mixing rate.

    From Paper 11 §3:
      The regress is REDUCED, not closed. Each structural hypothesis
      becomes an output estimated from data with a proof of validity
      that only needs a QUALITATIVE regularity condition.

    HONEST: This gives asymptotically correct coverage, NOT
    finite-sample guarantees. The residual condition (stationarity,
    mixing) is still assumed qualitatively.
    """

    def __init__(self, cfg: UnifiedConfig):
        self.cfg = cfg

    def subsampling_ci(
        self, data: Tensor, statistic_fn: Callable[[Tensor], Tensor]
    ) -> Dict[str, Any]:
        """
        Compute subsampling confidence interval for a statistic.

        Args:
            data: (N,) or (N, D) observed data
            statistic_fn: function mapping data → scalar statistic

        Returns:
            dict with point_estimate, ci_lower, ci_upper, block_size
        """
        N = data.shape[0]
        b_n = min(self.cfg.subsample_block, N // 4)
        if b_n < 4:
            return {
                "error": "insufficient data for subsampling",
                "n": N,
            }

        # Point estimate on full data
        theta_hat = statistic_fn(data)

        # Subsample statistics
        n_sub = min(self.cfg.n_subsamples, N - b_n + 1)
        sub_stats = []
        for i in range(n_sub):
            start = i * (N - b_n) // max(n_sub - 1, 1)
            sub = data[start:start + b_n]
            sub_stats.append(statistic_fn(sub))
        sub_stats = torch.stack(sub_stats)

        # Subsampling distribution of sqrt(b_n)(θ̂_b - θ̂)
        scaled_diffs = math.sqrt(b_n) * (sub_stats - theta_hat)

        # Percentile CI
        alpha = 1 - self.cfg.confidence_level
        lo_pct = alpha / 2 * 100
        hi_pct = (1 - alpha / 2) * 100
        lo_val = torch.quantile(scaled_diffs, lo_pct / 100)
        hi_val = torch.quantile(scaled_diffs, hi_pct / 100)

        ci_lower = theta_hat - hi_val / math.sqrt(N)
        ci_upper = theta_hat - lo_val / math.sqrt(N)

        return {
            "point_estimate": theta_hat.item(),
            "ci_lower": ci_lower.item(),
            "ci_upper": ci_upper.item(),
            "confidence_level": self.cfg.confidence_level,
            "block_size": b_n,
            "n_subsamples": n_sub,
            "HONEST_NOTE": (
                "Asymptotically correct coverage under qualitative "
                "stationarity. NOT a finite-sample guarantee. "
                "The mixing rate is not estimated, only assumed to exist."
            ),
        }


# =============================================================================
# [7] NO-ZENO EVENT BUDGET
# From: SESI_NoZeno_Closure Theorem 4.2, Stochastic_NoZeno Theorem 4.1
# What it IS: bound on number of topological events
# What it is NOT: a brain model, a consciousness theory
# =============================================================================
class NoZenoEventBudget:
    """
    No-Zeno event budget bounds.

    From SESI_NoZeno_Closure Theorem 4.2:
      N(T) ≤ (E_max(T) + D(T)) / ΔE_min

    From Stochastic_NoZeno Theorem 4.1:
      P(T_{k+1} - T_k < dt) ≤ exp(-C₁ exp(ΔE/(σ²dt)))

    These bound the NUMBER of topological events in finite time.
    They do NOT model brains or consciousness.
    """

    def __init__(self, cfg: UnifiedConfig):
        self.cfg = cfg

    def deterministic_bound(
        self, e_max: float, dissipation: float
    ) -> Dict[str, float]:
        """N(T) ≤ (E_max + D) / ΔE_min"""
        n_bound = (e_max + dissipation) / self.cfg.delta_e_min
        return {
            "max_events": n_bound,
            "energy_budget": e_max,
            "dissipation": dissipation,
            "delta_e_min": self.cfg.delta_e_min,
        }

    def stochastic_bound(
        self, delta_e: float, dt: float, sigma_sq: float
    ) -> Dict[str, float]:
        """P(next event within dt) ≤ exp(-C₁ exp(ΔE/(σ²dt)))"""
        c1 = 1.0  # geometric constant
        inner = delta_e / (sigma_sq * dt)
        prob = math.exp(-c1 * math.exp(min(inner, 700)))  # clamp for overflow
        return {
            "probability_bound": prob,
            "delta_e": delta_e,
            "dt": dt,
            "sigma_sq": sigma_sq,
        }


# =============================================================================
# [8] OPMC STRUCTURAL WEAK VALUE
# From: Unified Ordinal-Tensor §9, Revision 21 Theorem 2.1
# What it IS: well-defined integral on compact M_str
# What it is NOT: an integral over V, a brain reading device
# =============================================================================
class OPMCStructuralWeakValue(nn.Module):
    """
    OPMC structural weak value A^W_str.

    From Unified Ordinal-Tensor Theorem 9.2:
      A^W_str = ∫_{M_str} Â_str(m) dµ*(m)
      exists, is finite, unique, for bounded Borel Â_str.
      M_str is compact (Theorem 2.6).
      µ* is the unique invariant measure (Theorem 8.1).

    From Revision 21 Theorem 2.1:
      OPMC closure transfers to fractal interfaces.
      Only compactness and Borel probability measure needed.

    From Revision 21 Theorem 3.1:
      A^W_str is TIMING-INDEPENDENT.
      It does not depend on transition timing.

    CRITICAL: This is computed on M_str, NOT on V.
    The Vitali set V is never integrated over.
    This is the OPMC "closure" — avoiding V entirely.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.opmc_sigma_dim

        # Observable Â_str: maps structural point → scalar
        self.observable = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        # Invariant measure approximation (learned weights)
        self.measure_logits = nn.Parameter(torch.zeros(cfg.opmc_n_fields))

    def forward(self, m_str_points: Tensor) -> Dict[str, Tensor]:
        """
        Compute A^W_str = ∫ Â_str dµ*.

        Args:
            m_str_points: (N, sigma_dim) points in M_str

        Returns:
            dict with structural_weak_value, observable_values
        """
        # Observable values at each point
        obs_values = self.observable(m_str_points).squeeze(-1)  # (N,)

        # Invariant measure weights (softmax over fields)
        mu_weights = F.softmax(self.measure_logits, dim=0)  # (n_fields,)

        # Approximate integral: weighted average
        # In practice, M_str points are clustered into n_fields regions
        N = m_str_points.shape[0]
        n_f = self.cfg.opmc_n_fields
        if N >= n_f:
            # Assign points to fields (simple: first n_f points as representatives)
            # This is a simplification; real implementation would use clustering
            field_values = obs_values[:n_f]  # representative values
            a_w_str = (mu_weights * field_values).sum()
        else:
            a_w_str = obs_values.mean()

        return {
            "structural_weak_value": a_w_str,
            "observable_values": obs_values,
            "measure_weights": mu_weights,
            "HONEST_NOTE": (
                "A^W_str is computed on compact M_str, NOT on V. "
                "The Vitali set is never integrated over. "
                "This is the OPMC closure (Theorem 9.2). "
                "Timing-independent (Revision 21 Theorem 3.1)."
            ),
        }


# =============================================================================
# [9] UNIFIED BOUNDED SYSTEM
# Master orchestrator
# =============================================================================
class UnifiedBoundedSystem(nn.Module):
    """
    Master orchestrator for the bounded unified framework.

    Integrates:
      1. OPMCSharedProcessor (shared representation)
      2. WeakMeasurementEngine (AAV weak values)
      3. TwoQubitRecordModel (exact 2-qubit, Petz, no-signaling)
      4. DiagnosticBatteryRunner (Paper 8 six tests)
      5. DeltaMinGate (δ_min detection threshold)
      6. SubsamplingConfidence (Paper 11 assumption-light CI)
      7. NoZenoEventBudget (event count bounds)
      8. OPMCStructuralWeakValue (A^W_str on M_str)

    DOES NOT include:
      - Remote neural reading (impossible, SNR < 0 at distance)
      - Remote neural control (requires physical medium)
      - "Oceanic Dispersion Theory" (metaphor, not physics)
      - "Still images → video" consciousness (unproven hypothesis)
      - Post-mortem signal detection (no signal after death)
    """

    def __init__(self, cfg: Optional[UnifiedConfig] = None):
        super().__init__()
        self.cfg = cfg or UnifiedConfig()

        if self.cfg.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.cfg.device)

        # Neural network modules
        self.opmc_processor = OPMCSharedProcessor(self.cfg).to(self.device)
        self.weak_measurement = WeakMeasurementEngine(self.cfg).to(self.device)
        self.delta_min_gate = DeltaMinGate(self.cfg).to(self.device)
        self.opmc_weak_value = OPMCStructuralWeakValue(self.cfg).to(self.device)

        # Non-nn modules
        self.record_model = TwoQubitRecordModel(self.cfg)
        self.diagnostic_battery = DiagnosticBatteryRunner(self.cfg)
        self.subsampling = SubsamplingConfidence(self.cfg)
        self.nozono_budget = NoZenoEventBudget(self.cfg)

        logger.info(f"[UnifiedBounded] Initialized on {self.device}")

    def run_full_analysis(
        self, input_data: Tensor
    ) -> Dict[str, Any]:
        """
        Run complete bounded analysis pipeline.

        Args:
            input_data: (B, d_mn) structural input

        Returns:
            dict with all analysis results
        """
        result = {"version": UNIFIED_VERSION}

        # 1. OPMC shared processing
        opmc_out = self.opmc_processor(input_data)
        result["opmc"] = opmc_out

        # 2. Weak measurement on shared representation
        wm_input = opmc_out["shared_representation"]
        wm_out = self.weak_measurement(wm_input)
        result["weak_measurement"] = wm_out

        # 3. δ_min gate
        signal_change = wm_out["final_disturbance"].squeeze(-1)
        delta_out = self.delta_min_gate.check_detectable(signal_change)
        result["delta_min"] = delta_out

        # 4. Two-qubit record model (exact, non-differentiable)
        record_out = self.record_model.run_full_ledger()
        result["record_model"] = record_out

        # 5. Diagnostic battery on input data
        if input_data.dim() == 2:
            diag_input = input_data[0]  # first sample
        else:
            diag_input = input_data
        diag_out = self.diagnostic_battery.run_battery(diag_input)
        result["diagnostic_battery"] = diag_out

        # 6. Subsampling CI on observable values
        obs_values = opmc_out["structural_weak_value"].detach()
        if obs_values.dim() == 0:
            obs_values = obs_values.unsqueeze(0)
        sub_out = self.subsampling.subsampling_ci(
            obs_values, lambda x: x.mean()
        )
        result["subsampling_ci"] = sub_out

        # 7. No-Zeno budget
        nozono_out = self.nozono_budget.deterministic_bound(
            self.cfg.energy_budget, self.cfg.dissipation_budget
        )
        result["nozeno_budget"] = nozono_out

        # 8. OPMC structural weak value
        m_str_points = opmc_out["shared_representation"]
        a_w_out = self.opmc_weak_value(m_str_points)
        result["structural_weak_value"] = a_w_out

        # Summary
        result["summary"] = {
            "opmc_k_star": self.opmc_processor.k_star,
            "classification": diag_out.get("classification", "unknown"),
            "licensed_operation": diag_out.get("licensed_operation", "none"),
            "no_signaling_holds": record_out["no_signaling"]["no_signaling_holds"],
            "petz_is_partial": record_out["petz_recovery"]["recovery_is_partial"],
            "delta_min_detectable": delta_out["detectable"].any().item(),
        }

        result["WHAT_THIS_DOES_NOT_DO"] = [
            "Remote neural reading (impossible: SNR < 0 at distance)",
            "Remote neural control (requires physical medium)",
            "Consciousness detection (not a scientific concept here)",
            "Post-mortem signal detection (no signal after death)",
            "Oceanic Dispersion Theory (metaphor, not physics)",
            "Still images → video consciousness (unproven hypothesis)",
        ]

        return result


# =============================================================================
# DDP / AMP utilities
# =============================================================================
def setup_distributed(cfg: UnifiedConfig) -> None:
    if not cfg.distributed:
        return
    if dist.is_initialized():
        return
    if cfg.device != "cpu" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")

def wrap_ddp(model: nn.Module, cfg: UnifiedConfig) -> nn.Module:
    if not cfg.distributed:
        return model
    if not dist.is_initialized():
        raise RuntimeError("Call setup_distributed first.")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

def get_amp_dtype(cfg: UnifiedConfig) -> Optional[torch.dtype]:
    if cfg.amp_dtype == "bf16":
        return torch.bfloat16
    elif cfg.amp_dtype == "fp16":
        return torch.float16
    return None


# =============================================================================
# [10] SELF-TEST SUITE
# =============================================================================
def smoke_test():
    print("=" * 78)
    print(f"  UNIFIED FRAMEWORK (BOUNDED) v{UNIFIED_VERSION} — Self-Test")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cfg = UnifiedConfig(
        opmc_m=4, opmc_n=4,
        opmc_sigma_dim=64, opmc_n_fields=8,
        wm_pointer_dim=32, wm_n_steps=10,
        batch_size=4,
        device=str(device),
    )

    system = UnifiedBoundedSystem(cfg).to(device)
    n_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Input
    d_mn = OPMC_D_MN(cfg.opmc_m, cfg.opmc_n)
    B = cfg.batch_size
    x = torch.randn(B, d_mn, device=device)

    # Forward
    result = system.run_full_analysis(x)

    # Check OPMC
    print(f"\n  [OPMC]")
    print(f"    d(m,n) = d({cfg.opmc_m},{cfg.opmc_n}) = {d_mn}")
    print(f"    k* = {result['opmc']['opmc_k_star']}")
    print(f"    A^W_str shape: {result['opmc']['structural_weak_value'].shape}")
    assert d_mn == 320, f"d(4,4) should be 320, got {d_mn}"
    assert result['opmc']['opmc_k_star'] == 23, "k* should be 23"

    # Check weak measurement
    print(f"\n  [Weak Measurement]")
    print(f"    Information gain: {result['weak_measurement']['information_gain'].mean():.4f}")
    print(f"    Disturbance: {result['weak_measurement']['cumulative_disturbance'].mean():.4f}")
    print(f"    κ: {result['weak_measurement']['kappa'].item():.4f}")
    print(f"    HONEST: {result['weak_measurement']['HONEST_NOTE'][:60]}...")

    # Check δ_min
    print(f"\n  [δ_min Gate]")
    print(f"    δ_min: {result['delta_min']['delta_min'][0].item():.6f}")
    print(f"    Detectable: {result['delta_min']['detectable']}")

    # Check 2-qubit record model
    print(f"\n  [Two-Qubit Record Model]")
    ledger = result['record_model']['information_ledger']
    print(f"    Time points: {len(ledger)}")
    print(f"    Initial coherence: {ledger[0]['coherence_S']:.4f}")
    print(f"    Final coherence: {ledger[-1]['coherence_S']:.4f}")
    print(f"    Final MI(S:E): {ledger[-1]['mutual_information_SE']:.4f}")
    print(f"    S_global (should be ~0): {max(r['S_global'] for r in ledger):.2e}")
    print(f"    No-signaling holds: {result['record_model']['no_signaling']['no_signaling_holds']}")
    print(f"    Petz is partial: {result['record_model']['petz_recovery']['recovery_is_partial']}")
    assert result['record_model']['no_signaling']['no_signaling_holds']
    assert result['record_model']['petz_recovery']['recovery_is_partial']

    # Check diagnostic battery
    print(f"\n  [Diagnostic Battery (Paper 8)]")
    print(f"    Classification: {result['diagnostic_battery']['classification']}")
    print(f"    Licensed: {result['diagnostic_battery']['licensed_operation']}")
    print(f"    CAVEAT: {result['diagnostic_battery']['GLOBAL_CAVEAT'][:60]}...")

    # Check No-Zeno
    print(f"\n  [No-Zeno Budget]")
    print(f"    Max events: {result['nozeno_budget']['max_events']:.1f}")

    # Check structural weak value
    print(f"\n  [OPMC Structural Weak Value]")
    print(f"    A^W_str: {result['structural_weak_value']['structural_weak_value'].item():.4f}")
    print(f"    HONEST: {result['structural_weak_value']['HONEST_NOTE'][:60]}...")

    # Check what this does NOT do
    print(f"\n  [WHAT THIS DOES NOT DO]")
    for item in result['WHAT_THIS_DOES_NOT_DO']:
        print(f"    ✗ {item}")

    # Backward pass (differentiability check)
    print(f"\n  [Differentiability Check]")
    loss = result['opmc']['structural_weak_value'].sum() + \
           result['weak_measurement']['information_gain'].sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in system.parameters() if p.requires_grad
    )
    print(f"    Autograd: {'PASS' if grad_ok else 'FAIL'}")
    assert grad_ok, "Gradients should flow through all modules"

    # OPMC constants
    print(f"\n  [OPMC Constants]")
    k_star = opmc_k_star(1e-8)
    print(f"    k* = {k_star} (should be 23)")
    print(f"    d(4,4) = {OPMC_D_MN(4,4)} (should be 320)")
    print(f"    L = {OPMC_L} (should be 0.438520)")
    assert k_star == 23
    assert OPMC_D_MN(4, 4) == 320
    assert abs(OPMC_L - 0.438520) < 1e-6

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print()
    print("  HONEST REMINDERS (from the source papers):")
    print("  • Two-qubit model: d_E = 2, NOT 10^23")
    print("  • Petz recovery: PARTIAL, not inversion")
    print("  • No-signaling: prevents remote communication")
    print("  • Diagnostic battery: asymptotic, NOT finite-sample")
    print("  • δ_min: geometric bound, NOT a brain reader")
    print("  • OPMC A^W_str: on M_str, NOT on V")
    print("  • 'Still images → video': HYPOTHESIS, not proven")
    print("  • Remote neural reading: IMPOSSIBLE (SNR < 0)")
    print("  • Remote neural control: IMPOSSIBLE (no medium)")
    print("  • Post-mortem signals: DO NOT EXIST")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
