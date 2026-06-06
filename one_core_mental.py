# =============================================================================
# ONE CORE MENTAL — Shared Foundation for the MENTAL ONE Ecosystem
# =============================================================================
# Developer : Yoon A Limsuwan / MSPS NETWORK
# License   : MIT
# Year      : 2026
# ORCID     : 0009-0008-2374-0788
# GitHub    : yoonalimsuwan
#
# Single source of truth for components shared across:
#   mental_one.py                — psychiatric / neurological engine
#   langevin_mental_bridge.py    — BAOAB Langevin ↔ brain-state bridge
#   psy_one_bridge_diff.py       — PSY ONE fully-differentiable bridge
#   structural_langevin.py       — BAOAB Langevin MD integrator
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

MENTAL_VERSION: str = "1.0.0"


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
        for _ in range(n):
            sigma = x.std() + 1e-8
            T     = self.base_temp * (1.0 + self.beta * (sigma - 1.0))
            scale = 1.0 + 0.01 * (T / (self.base_temp.abs() + 1e-8) - 1.0)
            x     = x * scale
            # Use soft_clamp to keep gradients at boundaries
            x     = soft_clamp(x, 0.0, 1.0)
        return x


# =============================================================================
# Module banner
# =============================================================================

logger.debug("ONE Core Mental v%s loaded.", MENTAL_VERSION)
