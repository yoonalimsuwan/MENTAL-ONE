# =============================================================================
# ONE CORE MENTAL — Shared Foundation for the MENTAL ONE Ecosystem
# =============================================================================
# Developer : PAI , Yoon A Limsuwan / MSPS NETWORK
#             MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License   : MIT
# Year      : 2026
# ORCID     : 0009-0008-2374-0788
# GitHub    : yoonalimsuwan
#
# AI Co-Developers (architecture, differentiability design, integration):
#   - Claude   (Anthropic)  — CahnHilliardMentalBridge design, full
#                             differentiability audit, CSOC universality chain,
#                             cross-ecosystem integration pattern, SSC EMA
#                             boolean-buffer fix, DifferentiableSOC/RG canon
#   - GPT      (OpenAI)     — literature cross-check, numerical stability
#   - Gemini   (Google)     — operator scaffolding, structural base classes
#   - DeepSeek              — stencil verification, alternative architectures
#
# Single source of truth for components shared across:
#   mental_one.py                — psychiatric / neurological engine
#   langevin_mental_bridge.py    — BAOAB Langevin ↔ brain-state bridge
#   psy_one_bridge_diff.py       — PSY ONE fully-differentiable bridge
#   structural_langevin_mental.py — BAOAB Langevin MD integrator
#   structural_cahn_hilliard_3d.py — CH3D phase-field PDE (cross-ecosystem)
#
# This module is SEPARATE from:
#   one_core.py           — DNS/CFD continuum scale
#   one_core_fold.py      — REAL FOLD ONE protein scale
#   one_core_evolution.py — EVOLUTION ONE genomic/population scale
#
# MENTAL ONE operates at neural / psychiatric state-space scale.
#
# Shared components
# ─────────────────
#   SemanticStateContraction   — SSC EMA filter             (Paper 4)
#   CSOCBase                   — abstract CSOC base class    (Paper 4)
#   InterfaceDetectorBase      — abstract interface detector
#   StructuralItoBase          — abstract Itô correction     (Papers 2 & 3)
#   DifferentiableRG           — fully differentiable learnable RG smoother
#   DifferentiableSOC          — fully differentiable SOC dynamics (n-step)
#   CahnHilliardMentalBridge   — CH3D ↔ MENTAL ONE cross-ecosystem bridge
#   soft_clamp                 — differentiable alternative to hard .clamp()
#   get_device                 — unified hardware-backend selector
#   MENTAL_VERSION             — ecosystem-wide version string
# =============================================================================

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MENTAL_VERSION: str = "2.0.0"


# =============================================================================
# 0. Hardware-backend selector
# =============================================================================

def get_device(preferred: str = "cuda") -> torch.device:
    """
    Select the best available compute device.
    Priority: CUDA → MPS (Apple Silicon) → CPU.
    """
    p = preferred.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if p == "ascend" and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. Differentiable utility functions
# =============================================================================

def soft_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    Differentiable alternative to hard .clamp().

    Uses tanh-based smooth projection — gradient exists everywhere,
    unlike hard clamp which zeros gradient at boundaries.

        f(x) = center + scale * tanh((x - center) / scale)

    Args:
        x  : input tensor (any shape).
        lo : lower bound.
        hi : upper bound.
    """
    center = (hi + lo) / 2.0
    scale  = (hi - lo) / 2.0 + 1e-8
    return center + scale * torch.tanh((x - center) / scale)


# =============================================================================
# 2. Semantic State Contraction (SSC) — Paper 4
# =============================================================================

class SemanticStateContraction(nn.Module):
    """
    SSC EMA low-pass filter for structural stress σ  (Paper 4).

    **Canonical implementation** shared across the entire MENTAL ONE
    ecosystem.  Do NOT redefine locally in individual files.

    Fixes vs. the langevin_mental_bridge.py version
    ─────────────────────────────────────────────────
    •  Boolean ``_initialized`` buffer (not checking ``prev_sigma == 0.0``).
       The old pattern breaks when the true first stress is exactly zero.
    •  ``reset()`` clears BOTH buffer and flag — safe between independent
       patient sessions / trajectories.
    •  Buffer auto-migrates to the incoming tensor's device (CPU↔GPU safe).

    Args:
        epsilon_fp    : EMA blending factor ∈ (0, 1).
        sigma_target  : reference stress (stored for downstream use).
    """

    def __init__(
        self,
        epsilon_fp:   float = 0.0028,
        sigma_target: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 < epsilon_fp < 1.0):
            raise ValueError(f"epsilon_fp must be in (0, 1); got {epsilon_fp!r}.")
        self.eps    = epsilon_fp
        self.target = sigma_target
        self.register_buffer("prev_sigma",   torch.tensor(0.0))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset(self) -> None:
        """Reset EMA state between independent patient trajectories."""
        self.prev_sigma.zero_()
        self._initialized.fill_(False)

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_sigma : scalar stress tensor (differentiable).
        Returns:
            Filtered stress scalar.
        """
        if self.prev_sigma.device != raw_sigma.device:
            self.prev_sigma   = self.prev_sigma.to(raw_sigma.device)
            self._initialized = self._initialized.to(raw_sigma.device)

        if not self._initialized.item():
            self.prev_sigma.data = raw_sigma.detach()
            self._initialized.fill_(True)
            return raw_sigma

        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


# =============================================================================
# 3. CSOC Base — Paper 4
# =============================================================================

class CSOCBase(nn.Module, ABC):
    """
    Abstract base class for CSOC adaptive-parameter modules  (Paper 4).

    Provides the shared SSC filter, ``reset()``, and helper methods
    ``_normalised_deviation`` and ``_smooth_boost``.

    Args:
        sigma_target : reference structural stress.
        epsilon_fp   : SSC EMA blending factor.
        boost_factor : maximum parameter multiplier at high stress.
    """

    def __init__(
        self,
        sigma_target: float = 1.0,
        epsilon_fp:   float = 0.0028,
        boost_factor: float = 3.0,
    ) -> None:
        super().__init__()
        if sigma_target <= 0:
            raise ValueError(f"sigma_target must be positive; got {sigma_target!r}.")
        if boost_factor < 1.0:
            raise ValueError(f"boost_factor must be ≥ 1; got {boost_factor!r}.")
        self.sigma_target = sigma_target
        self.boost_factor = boost_factor
        self.ssc = SemanticStateContraction(epsilon_fp, sigma_target)

    def reset(self) -> None:
        """Reset SSC EMA state between independent sessions."""
        self.ssc.reset()

    def _normalised_deviation(self, sigma: torch.Tensor) -> torch.Tensor:
        """(σ − σ_target) / σ_target — scalar deviation from criticality."""
        return (sigma - self.sigma_target) / max(self.sigma_target, 1e-12)

    def _smooth_boost(self, dev: torch.Tensor) -> torch.Tensor:
        """Sigmoid boost ∈ (0, 1) for smooth parameter interpolation."""
        return torch.sigmoid(dev)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute adaptive parameters from current structural state."""


# =============================================================================
# 4. Interface Detector Base
# =============================================================================

class InterfaceDetectorBase(nn.Module, ABC):
    """
    Abstract base for differentiable interface / sharp-gradient detectors.

    Subclasses must return a tensor ∈ [0, 1], fully differentiable.
    In MENTAL ONE: detects transients, pathological spikes, phase transitions
    in EEG / brain-state vectors.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Returns mask tensor ∈ [0, 1], differentiable w.r.t. input."""


# =============================================================================
# 5. Structural Itô Base — Papers 2 & 3
# =============================================================================

class StructuralItoBase(nn.Module, ABC):
    """
    Abstract base class for Structural Itô drift-correction modules.

    Both the atomic Langevin integrator (N×3) and the brain-state integrator
    (N brain state vector) implement the same ½ G(x) ∇_x G(x) correction;
    only dimensionality and interface detector differ.

    Args:
        interface_amplification : G-field amplitude boost at interfaces.
    """

    def __init__(self, interface_amplification: float = 2.0) -> None:
        super().__init__()
        if interface_amplification < 0:
            raise ValueError(
                f"interface_amplification must be ≥ 0; got {interface_amplification!r}.")
        self.amp = interface_amplification

    def get_g_field(self, interface_mask: torch.Tensor) -> torch.Tensor:
        """G(x) = 1 + amp · mask(x).  Same formula in all domains."""
        return 1.0 + self.amp * interface_mask

    @abstractmethod
    def compute_ito_correction(
        self,
        field: torch.Tensor,
        interface_detector: InterfaceDetectorBase,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute ½ G(x) ∇_x G(x).

        Returns:
            Itô drift tensor, same shape as ``field``, **detached**.
        """


# =============================================================================
# 6. Differentiable RG smoother — learnable, shared
# =============================================================================

class DifferentiableRG(nn.Module):
    """
    Fully differentiable learnable 1-D RG smoothing kernel.

    Replaces both:
    •  ``DiffRGRefiner(factor=4, n_levels=2)`` in ``mental_one.py``
       (uses non-learnable avg_pool + interpolate)
    •  The fixed-kernel version in ``psy_one_bridge_diff.py``

    The kernel weights are ``nn.Parameter`` — end-to-end trainable.

    Args:
        kernel_size : convolution kernel length (odd recommended).
    """

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be ≥ 1; got {kernel_size!r}.")
        self.kernel_size = kernel_size
        self.weight      = nn.Parameter(torch.ones(kernel_size) / kernel_size)
        self.padding     = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : 1-D ``(T,)`` or batched ``(B, T)`` tensor.
        Returns:
            Smoothed tensor, same shape as x.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)          # (1, T)

        w   = self.weight / (self.weight.sum() + 1e-8)
        w   = w.view(1, 1, -1)          # (1, 1, K)
        out = F.conv1d(x.unsqueeze(1), w, padding=self.padding).squeeze(1)

        return out.squeeze(0) if squeeze else out


# =============================================================================
# 7. Differentiable SOC dynamics — shared, n-step, learnable
# =============================================================================

class DifferentiableSOC(nn.Module):
    """
    Fully differentiable SOC temperature modulation.

    Replaces ``SOCController.soc_evolve`` (naive random walk, breaks grad
    graph via ``.detach()`` + ``clamp``).

    ``base_temp`` and ``beta`` are ``nn.Parameter`` — the SOC temperature
    and sensitivity can be end-to-end trained.

    Args:
        base_temp : initial reference temperature.
        beta      : initial sensitivity to stress deviation.
        n_steps   : default SOC relaxation steps.
    """

    def __init__(
        self,
        base_temp: float = 300.0,
        beta:      float = 0.01,
        n_steps:   int   = 20,
    ) -> None:
        super().__init__()
        self.base_temp = nn.Parameter(torch.tensor(float(base_temp)))
        self.beta      = nn.Parameter(torch.tensor(float(beta)))
        self.n_steps   = n_steps
        # Fixed normalisation constant for `scale` in forward() — captures
        # the *initial* base_temp value but is NOT the parameter itself,
        # so it stays constant across training and doesn't cancel
        # base_temp's gradient signal out of the scale computation.
        self.register_buffer("_t_ref", torch.tensor(float(abs(base_temp)) + 1e-8))

    def forward(
        self,
        x:     torch.Tensor,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Evolve brain state / mutation load x via SOC dynamics.

        Fully differentiable — no ``.detach()`` or ``.item()`` in grad path.

        Args:
            x     : (...) float tensor (brain state mean or mu values).
            steps : override ``self.n_steps``.
        Returns:
            Evolved tensor, same shape as x.
        """
        n = steps if steps is not None else self.n_steps
        # Reference temperature is a fixed constant (the value base_temp
        # was initialised to), captured once so it does NOT track the
        # parameter — only used to non-dimensionalise T into a
        # ratio. Previously this denominator was self.base_temp.abs()
        # itself, which made T / |base_temp| algebraically cancel
        # base_temp out of `scale` entirely (scale = 1 + 0.01*beta*(sigma-1),
        # independent of base_temp's value). That left base_temp a
        # dead nn.Parameter despite the docstring's claim it's trainable
        # and meaningful. Normalising against a fixed reference instead
        # lets base_temp actually shift `scale` (and gradients w.r.t. it
        # carry real signal).
        T_ref = self._t_ref
        for _ in range(n):
            sigma = x.std() + 1e-8
            T     = self.base_temp * (1.0 + self.beta * (sigma - 1.0))
            scale = 1.0 + 0.01 * (T / T_ref - 1.0)
            x     = x * scale
            # Use soft_clamp to keep gradients at boundaries
            x     = soft_clamp(x, 0.0, 1.0)
        return x



# =============================================================================
# 8. structural_biharmonic_n — Module-level utility (shared with CH3D)
# =============================================================================

def structural_biharmonic_n(
    field: torch.Tensor,
    sigma: torch.Tensor,
    n: int,
    laplacian_fn,
) -> torch.Tensor:
    """
    Compute the n-th power of the structural Laplacian: Δ_S^n u.

    Exposed here so that ``structural_cahn_hilliard_3d.py`` and
    ``one_core.py`` can share a single canonical implementation.

    Args:
        field        : (...) input tensor.
        sigma        : (...) structural sigma-field (same shape as field).
        n            : integer order ≥ 1.
        laplacian_fn : callable(field, sigma) → tensor, same shape.
    Returns:
        Δ_S^n field, same shape as input.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    result = field
    for _ in range(n):
        result = laplacian_fn(result, sigma)
    return result


# =============================================================================
# 9. CahnHilliardMentalBridge — CH3D ↔ MENTAL ONE cross-ecosystem bridge
# =============================================================================

class CahnHilliardMentalBridge(nn.Module):
    """
    Cross-ecosystem bridge: Structural Cahn–Hilliard 3D  ↔  MENTAL ONE.

    Physical interpretation
    ───────────────────────
    The order parameter field  u(x,t) ∈ [-1, +1]³  of the CH3D solver is
    mapped to a brain-state stress signal that drives the SSC / CSOC
    pipeline inside MENTAL ONE.  The mapping is:

        σ_CH(t) = ‖∂_t u‖ₗ₂ / √(Nₓ Nᵧ N_z)          ← mean-field rate

    This scalar flows into the shared ``SemanticStateContraction`` filter,
    producing the same SSC stress σ that governs CSOC thermostat adaptation
    in ``AdvancedStructuralLangevin`` and ``SOCController``.

    Use cases
    ─────────
    1.  Psychiatric phase transitions modelled as spinodal decomposition:
        The CH order parameter encodes the spatial distribution of a
        neurochemical (e.g. dopamine gradient across cortical layers).
        Phase separation → onset of a psychiatric episode.

    2.  BV consistency: CH interface energy maps to the BV jump measure
        in ``BVConsistency`` (mental_one.py).

    3.  PFC (Phase-Field Crystal) mode: encodes neural oscillation
        pattern formation (beta/gamma synchrony → crystal lattice).

    Differentiability
    ─────────────────
    All operations are fully differentiable:
    •  ``sigma_from_ch()`` uses soft_clamp + L2-norm (autograd-safe).
    •  ``ch_to_brain_state()`` uses F.adaptive_avg_pool3d (to a small
       grid, not a single scalar — preserves coarse spatial structure)
       + linear proj.
    •  ``energy_coupling()`` returns a scalar loss for joint training.

    Args:
        state_dim          : brain-state vector length (= n_ch × n_tp).
        ssc                : shared SemanticStateContraction instance.
        coupling_strength  : weight of CH energy in joint loss.
        proj_bias          : learnable bias in the 3D→1D projection.
        pool_grid          : side length of the pooled cube grid used
                              before projection (pool_grid**3 cells).
                              Larger retains more spatial detail at the
                              cost of more projection parameters.
    """

    def __init__(
        self,
        state_dim        : int,
        ssc              : Optional["SemanticStateContraction"] = None,
        coupling_strength: float = 0.1,
        proj_bias        : bool  = True,
        pool_grid        : int   = 4,
    ) -> None:
        super().__init__()
        self.state_dim         = state_dim
        self.coupling_strength = coupling_strength
        self._pool_grid        = pool_grid

        # Shared SSC filter — reuse the one from CSOCBase if provided
        self.ssc = ssc if ssc is not None else SemanticStateContraction()

        # Learnable (pool_grid^3) → state_dim linear projection
        # (CH volume, coarse-pooled, → brain state). Previously this
        # projected from a single pooled scalar (in_features=1); see
        # ch_to_brain_state() for why that erased all spatial structure.
        self.proj = nn.Linear(pool_grid ** 3, state_dim, bias=proj_bias)

        # Learnable coupling scalar (log-parameterised for positivity)
        self.log_coupling = nn.Parameter(
            torch.tensor(float(math.log(max(coupling_strength, 1e-8))))
        )

        # Buffer: previous CH field for computing ∂_t u
        self.register_buffer("_prev_u",        torch.zeros(1))
        self.register_buffer("_u_initialized", torch.tensor(False))

    def reset(self) -> None:
        """Reset temporal state between independent simulations."""
        self._prev_u.zero_()
        self._u_initialized.fill_(False)
        self.ssc.reset()

    # ------------------------------------------------------------------
    def sigma_from_ch(self, u: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Compute SSC-filtered structural stress from CH order parameter.

        σ_raw = ‖u - u_prev‖₂ / √N   (mean temporal rate)

        Args:
            u  : (...) current CH order parameter field (any 3D shape).
            dt : time step of the CH solver (scales raw stress).
        Returns:
            σ : scalar tensor, SSC-filtered, fully differentiable.
        """
        u_flat = u.reshape(-1)

        # Migrate buffer to correct device/dtype
        if self._prev_u.device != u_flat.device or self._prev_u.dtype != u_flat.dtype:
            self._prev_u       = self._prev_u.to(u_flat.device, u_flat.dtype)
            self._u_initialized = self._u_initialized.to(u_flat.device)

        # Plain `self._prev_u = u_flat.detach().clone()` reassigns the
        # attribute and silently drops it from the registered buffers
        # (same bug already fixed in mental_one.py's SOCController.sigma()
        # — it would vanish from state_dict() and stop following later
        # .to(device) calls). Resize + copy_ in place instead.
        if not self._u_initialized.item() or self._prev_u.shape != u_flat.shape:
            if self._prev_u.shape != u_flat.shape:
                self._prev_u.resize_(u_flat.shape)
            self._prev_u.copy_(u_flat.detach())
            self._u_initialized.fill_(True)
            raw_sigma = torch.tensor(1e-6, device=u.device, dtype=u.dtype)
        else:
            diff      = u_flat - self._prev_u
            raw_sigma = soft_clamp(
                torch.sqrt((diff ** 2).mean() + 1e-12) / max(dt, 1e-8),
                0.0, 1e6,
            )
            self._prev_u.copy_(u_flat.detach())

        return self.ssc(raw_sigma)

    # ------------------------------------------------------------------
    def ch_to_brain_state(self, u: torch.Tensor) -> torch.Tensor:
        """
        Project CH 3D order parameter field → 1D brain-state vector.

        u (Nₓ, Nᵧ, N_z) → (state_dim,) via:
          1. 3D average pooling → small fixed-size pooled grid
             (POOL_GRID³ cells), preserving coarse spatial structure
          2. Learnable linear projection of the flattened grid to state_dim

        Previously step 1 pooled all the way down to a single scalar
        (adaptive_avg_pool3d(..., 1)), which meant every brain_state
        component was just an affine function of one global mean —
        no spatial information (gradients, localized phase separation,
        etc., as described above) could possibly survive. Pooling to a
        small grid instead of a point preserves coarse spatial pattern
        while still keeping the projection input size fixed regardless
        of the solver's actual (Nx, Ny, Nz) resolution.

        Fully differentiable.

        Args:
            u : (Nx, Ny, Nz) or (B, Nx, Ny, Nz) CH order parameter.
        Returns:
            brain_state : (state_dim,) or (B, state_dim).
        """
        batched = u.dim() == 4
        if not batched:
            u = u.unsqueeze(0)   # (1, Nx, Ny, Nz)

        # Pool to a small fixed grid (POOL_GRID, POOL_GRID, POOL_GRID)
        # instead of a single scalar, so coarse spatial structure
        # (e.g. a dopamine gradient across one axis) is retained.
        pooled = F.adaptive_avg_pool3d(
            u.unsqueeze(1), self._pool_grid
        )  # (B, 1, P, P, P)
        pooled_flat = pooled.flatten(1)  # (B, P*P*P)

        brain_state = self.proj(pooled_flat)  # (B, state_dim)
        # Normalise to [0, 1] range expected by MENTAL ONE
        brain_state = soft_clamp(brain_state, 0.0, 1.0)

        return brain_state if batched else brain_state.squeeze(0)

    # ------------------------------------------------------------------
    def energy_coupling(
        self,
        ch_energy: torch.Tensor,
        psy_loss : torch.Tensor,
    ) -> torch.Tensor:
        """
        Joint differentiable loss for co-training CH3D + MENTAL ONE.

        L_joint = psy_loss + exp(log_coupling) * ch_energy

        Args:
            ch_energy : scalar CH structural free energy (from ch.structural_energy).
            psy_loss  : scalar PSY ONE / MENTAL ONE loss.
        Returns:
            L_joint : scalar, backprop-ready.
        """
        w = torch.exp(self.log_coupling)
        return psy_loss + w * ch_energy

    # ------------------------------------------------------------------
    def forward(
        self,
        u       : torch.Tensor,
        ch_energy: Optional[torch.Tensor] = None,
        psy_loss : Optional[torch.Tensor] = None,
        dt      : float = 1.0,
    ) -> dict:
        """
        Full differentiable forward pass.

        Args:
            u          : CH order parameter field (Nx, Ny, Nz).
            ch_energy  : optional scalar CH free energy.
            psy_loss   : optional scalar PSY / MENTAL ONE loss.
            dt         : CH time step (for stress rate computation).
        Returns:
            dict with keys:
                'sigma'       : SSC-filtered stress scalar.
                'brain_state' : (state_dim,) projected brain-state vector.
                'joint_loss'  : optional scalar (if both ch_energy and psy_loss given).
        """
        sigma       = self.sigma_from_ch(u, dt=dt)
        brain_state = self.ch_to_brain_state(u)

        result: dict = {"sigma": sigma, "brain_state": brain_state}

        if ch_energy is not None and psy_loss is not None:
            result["joint_loss"] = self.energy_coupling(ch_energy, psy_loss)

        return result


# =============================================================================
# Module banner
# =============================================================================

logger.debug("ONE Core Mental v%s loaded.", MENTAL_VERSION)

__all__ = [
    # Version
    "MENTAL_VERSION",
    # Utilities
    "soft_clamp",
    "get_device",
    "structural_biharmonic_n",
    # Core components
    "SemanticStateContraction",
    "CSOCBase",
    "InterfaceDetectorBase",
    "StructuralItoBase",
    "DifferentiableRG",
    "DifferentiableSOC",
    # Cross-ecosystem bridge
    "CahnHilliardMentalBridge",
]
