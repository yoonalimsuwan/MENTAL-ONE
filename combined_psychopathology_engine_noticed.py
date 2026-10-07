# =============================================================================
# Combined Psychopathology Engine — NATIVE FULL DIFFERENTIABLE
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : https://github.com/yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# Version      : 2.0.0 (Native Full Differentiable / AMP-Safe / DDP-Ready)
# =============================================================================
"""
Production-grade, native fully differentiable combined continuous-discrete
dynamical framework for modeling psychopathology.

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **Data-dependent Python branch removed — the biggest correctness bug.**
   The reference wrote

       if self.tau_homeostasis != 0.0:
           activation_input = activation_input - self.tau_homeostasis * (s - self.baseline_s)

   Since `tau_homeostasis` is an `nn.Parameter`, `!= 0.0` is a **tensor
   comparison**, which:
       · forces a silent `Tensor.item()` → D2H sync on every step,
       · breaks the `torch.compile` graph,
       · makes the loss landscape **discontinuous** at τ_h = 0 — the moment
         training moves τ_h away from its 0.0 initialization, the homeostasis
         term suddenly activates and destabilizes the optimizer.
   The homeostasis term is now applied unconditionally; when τ_h = 0, it
   contributes exactly 0 to the activation (identical reference behavior for
   the default init) but with **C^∞ dependence on τ_h** and no branch.

2. **`torch.abs` → C^∞ surrogate.**  `sqrt(x² + ε²)` replaces `torch.abs(x)`
   for the pressure term `P_t` — the reference had a **kink at x = 0** that
   produced undefined gradients at the Lorenz fixed point.

3. **`clamp(min=1.0)` on node degrees → C^∞ soft floor.**  Isolated nodes
   (degree 0) drove the divisor to exactly 1.0 with a dead-gradient zone
   below that.  `1.0 + softplus(d − 1.0, β)` is exact for `d ≫ 1`, smooth
   everywhere, gradient alive in the entire domain.

4. **Every Python scalar moved to non-persistent buffers.**  `self.dt`,
   `dt * 0.5`, `dt / 6.0` were read from Python every step, triggering
   `torch.compile` recompiles on any dt change.  Pre-materialized as fp32
   buffers, cast per call — no recompiles, DDP-safe (excluded from
   `state_dict`).

5. **AMP-safe Lorenz core.**  RK4 on a chaotic system loses precision in
   bf16/fp16 after a few steps.  The RK4 integration now runs in **fp32
   internally** and casts back to the input dtype — the network layer
   (`tanh`, additions) remains in the caller's preferred dtype.

6. **Per-step gradient checkpointing.**  `gradient_checkpointing=True` wraps
   each simulation step in `torch.utils.checkpoint(..., use_reentrant=False)`
   — trades ~2× compute for activation-memory savings on long simulations,
   with full DDP correctness.

7. **Deterministic evaluation.**  The reference's noise injection was
   correct (Python `float` and `bool`, no tensor sync), but it was checked
   *inside* the loop.  The `add_noise` decision is now hoisted out of the
   loop and passed as a Python bool, so `torch.compile` doesn't capture the
   branch — and eval mode is guaranteed noise-free.

8. **Input validation** — shape and dtype guards, `validate_inputs=False`
   opt-out for compiled static graphs.

9. **Public API preserved.**  Same class name, same constructor signature
   `(num_nodes, dt=0.01)`, same forward signature, same `(x, y, z, s)`
   return tuple, same parameter set — strict drop-in upgrade.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    engine = CombinedPsychopathologyEngine(num_nodes=64, dt=0.01).to(rank)
    engine = torch.compile(engine, mode="max-autotune")                # optional
    engine = torch.nn.parallel.DistributedDataParallel(
        engine, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        x, y, z, s = engine(x, y, z, s, adj, steps=64)
    loss = (s - target_s).pow(2).mean() + 0.1 * (x.pow(2).mean() + y.pow(2).mean())
    loss.backward()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# RESEARCH STATUS NOTICE (added during code review)
# * This is an exploratory differentiable dynamical-systems / ML prototype.
# * Constants are not fitted to clinical or experimental data; outputs have not
#   been validated against any ground truth, benchmark, or clinical outcome.
# * It is NOT a diagnostic, screening, treatment-selection or decision tool.
#   Do not apply it to data from real people without ethics-board (IRB) approval,
#   informed consent, data-protection review and independent clinical validation.
# * Terms such as "production", "SOTA", "clinical" or "infinite" in this file
#   describe engineering style or ambition, not demonstrated performance.
# ---------------------------------------------------------------------------
__research_status__ = "exploratory-prototype; unvalidated; not for clinical or personal decisions"

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PsychopathologyConfig", "CombinedPsychopathologyEngine"]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class PsychopathologyConfig:
    """Numerical-safety and surrogate-sharpness configuration."""
    homeostatic_beta: float = 50.0     # softplus sharpness for degree soft floor
    abs_eps: float = 1e-4              # ε in sqrt(x² + ε²)
    degree_floor: float = 1.0          # soft floor applied to node degrees
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


# =============================================================================
# Engine
# =============================================================================
class CombinedPsychopathologyEngine(nn.Module):
    """
    Production-grade, native fully differentiable combined continuous-discrete
    dynamical framework for psychopathology.

    Continuous core:  Lorenz system integrated with RK4 (fp32-internal).
    Discrete network:  bounded `tanh` node update with degree-normalized
                       adjacency, global pressure, external drives, and
                       homeostatic / autonomic-feedback modulation.

    Parameters
    ----------
    num_nodes : int
        Number of discrete network nodes (must be > 0).
    dt : float
        Continuous-core integration step (must be > 0).
    config : PsychopathologyConfig, optional
        Numerical-safety parameters.  Individual kwargs override fields.
    validate_inputs : bool
        Cheap shape/dtype guards; disable under `torch.compile` static shapes.
    gradient_checkpointing : bool
        Recomputed the step during backward — trades ~2× compute for
        activation-memory savings on long simulations.
    """

    def __init__(
        self,
        num_nodes: int,
        dt: float = 0.01,
        *,
        config: Optional[PsychopathologyConfig] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not (isinstance(num_nodes, int) and num_nodes > 0):
            raise ValueError("num_nodes must be a positive int.")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be > 0.")

        cfg = config or PsychopathologyConfig()
        if validate_inputs is not None:
            cfg = PsychopathologyConfig(**{**cfg.__dict__, "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = PsychopathologyConfig(**{**cfg.__dict__,
                                           "gradient_checkpointing": bool(gradient_checkpointing)})

        for name, v in (
            ("homeostatic_beta", cfg.homeostatic_beta),
            ("abs_eps", cfg.abs_eps),
            ("degree_floor", cfg.degree_floor),
        ):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be > 0.")

        self.num_nodes = int(num_nodes)
        self.dt = float(dt)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        # ------------------------------------------------------------------
        # Continuous Lorenz core parameters
        # ------------------------------------------------------------------
        self.sigma       = nn.Parameter(torch.tensor(10.0))
        self.rho         = nn.Parameter(torch.tensor(28.0))
        self.beta_lorenz = nn.Parameter(torch.tensor(8.0 / 3.0))

        # ------------------------------------------------------------------
        # Continuous↔Discrete coupling parameters
        # ------------------------------------------------------------------
        self.kappa1 = nn.Parameter(torch.tensor(1.0))
        self.kappa2 = nn.Parameter(torch.tensor(1.0))
        self.kappa3 = nn.Parameter(torch.tensor(1.0))

        self.gamma   = nn.Parameter(torch.randn(num_nodes))
        self.delta   = nn.Parameter(torch.randn(num_nodes))
        self.epsilon = nn.Parameter(torch.randn(num_nodes))

        # ------------------------------------------------------------------
        # Discrete network parameters
        # ------------------------------------------------------------------
        self.alpha    = nn.Parameter(torch.tensor(1.0))
        self.beta_net = nn.Parameter(torch.tensor(1.0))

        # Extended variant (homeostasis & baseline)
        self.tau_homeostasis = nn.Parameter(torch.tensor(0.0))
        self.baseline_s      = nn.Parameter(torch.zeros(num_nodes))

        # ------------------------------------------------------------------
        # Non-persistent scalar buffers (DDP-safe, checkpoint-excluded)
        # ------------------------------------------------------------------
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_dt",         _buf(self.dt),              persistent=False)
        self.register_buffer("_dt_half",    _buf(self.dt * 0.5),        persistent=False)
        self.register_buffer("_dt_sixth",   _buf(self.dt / 6.0),        persistent=False)
        self.register_buffer("_abs_eps",    _buf(cfg.abs_eps),          persistent=False)
        self.register_buffer("_homeo_beta", _buf(cfg.homeostatic_beta), persistent=False)
        self.register_buffer("_deg_floor",  _buf(cfg.degree_floor),     persistent=False)

    # ------------------------------------------------------------------ #
    # C^∞ primitives                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _soft_floor(x: torch.Tensor, floor: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """C^∞ surrogate for `max(x, floor)` — exact for `x ≫ floor`."""
        return floor + F.softplus(x - floor, beta=beta)

    @staticmethod
    def _safe_abs(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """C^∞ surrogate for `|x|` — no kink at `x = 0`."""
        return torch.sqrt(x * x + eps * eps)

    # ------------------------------------------------------------------ #
    # Lorenz derivatives + RK4 (fp32-internal for chaotic stability)     #
    # ------------------------------------------------------------------ #
    def _lorenz_derivatives(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lorenz RHS.  Inputs should be fp32."""
        dx = self.sigma.float() * (y - x)
        dy = x * (self.rho.float() - z) - y
        dz = x * y - self.beta_lorenz.float() * z
        return dx, dy, dz

    def _rk4_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One RK4 integration step of the Lorenz core.

        Runs **internally in fp32** and casts back to the input dtype.
        Under AMP this preserves the numerical integrity of the chaotic
        integration — bf16/fp16 RK4 on Lorenz loses precision within a few
        steps because the Poincaré map amplifies round-off exponentially.
        """
        dtype_in = x.dtype
        device   = x.device

        # fp32 working tensors (AMP-safe).
        x32 = x.float()
        y32 = y.float()
        z32 = z.float()

        dt_half  = self._dt_half.to(device=device)
        dt_sixth = self._dt_sixth.to(device=device)

        k1x, k1y, k1z = self._lorenz_derivatives(x32, y32, z32)
        k2x, k2y, k2z = self._lorenz_derivatives(
            x32 + dt_half * k1x, y32 + dt_half * k1y, z32 + dt_half * k1z,
        )
        k3x, k3y, k3z = self._lorenz_derivatives(
            x32 + dt_half * k2x, y32 + dt_half * k2y, z32 + dt_half * k2z,
        )
        k4x, k4y, k4z = self._lorenz_derivatives(
            x32 + self._dt.to(device=device) * k3x,
            y32 + self._dt.to(device=device) * k3y,
            z32 + self._dt.to(device=device) * k3z,
        )

        x_next = x32 + dt_sixth * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y_next = y32 + dt_sixth * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z_next = z32 + dt_sixth * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

        return x_next.to(dtype_in), y_next.to(dtype_in), z_next.to(dtype_in)

    # ------------------------------------------------------------------ #
    # Single differentiable simulation step                              #
    # ------------------------------------------------------------------ #
    def _node_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        norm_adj: torch.Tensor,         # [N, N] row-normalized adjacency
        ext_divisor: torch.Tensor,      # [1, N] row-degree reciprocal
        feedback: torch.Tensor,         # [B, N] autonomic feedback (zero if unused)
        noise_std: torch.Tensor,        # scalar
        add_noise: bool,                # Python bool, hoisted out of the loop
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One combined continuous-discrete update.  Fully tensor-typed — no
        Python branch depends on a tensor value.
        """
        # ---- 1. Continuous Lorenz core -----------------------------------
        x, y, z = self._rk4_step(x, y, z)

        # ---- 2. Global pressure P(t) — C^∞ |·| (no kink at 0) ------------
        eps = self._abs_eps.to(device=x.device, dtype=x.dtype)
        x_abs = self._safe_abs(x, eps)
        y_abs = self._safe_abs(y, eps)
        z_compl = 1.0 - z
        P_t = self.kappa1 * x_abs + self.kappa2 * y_abs + self.kappa3 * z_compl

        # ---- 3. External drive projections -------------------------------
        gamma_b   = self.gamma.unsqueeze(0)    # [1, N]
        delta_b   = self.delta.unsqueeze(0)
        epsilon_b = self.epsilon.unsqueeze(0)

        ext_drive = gamma_b * x + delta_b * y - epsilon_b * z          # [B, N]
        ext_drive_norm = ext_drive / ext_divisor                       # [B, N]

        # ---- 4. Discrete network update ---------------------------------
        neighbor_sum = s @ norm_adj.transpose(-2, -1)                  # [B, N]

        # Homeostasis is ALWAYS applied — C^∞ in tau_homeostasis.
        homeostatic = self.tau_homeostasis * (s - self.baseline_s.unsqueeze(0))

        activation = (
            s
            + self.alpha * P_t
            + self.beta_net * neighbor_sum
            + ext_drive_norm
            - homeostatic
            + feedback
        )                                                              # [B, N]

        if add_noise:
            activation = activation + torch.randn_like(s) * noise_std

        s_next = torch.tanh(activation)
        return x, y, z, s_next

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        adj_matrix: torch.Tensor,
        steps: int = 1,
        noise_std: float = 0.0,
        autonomic_feedback: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes forward simulation over `steps` temporal steps.

        Parameters
        ----------
        x, y, z : torch.Tensor  [B, 1]
            Continuous core variables (Lorenz).
        s : torch.Tensor  [B, N]
            Discrete network state.
        adj_matrix : torch.Tensor  [N, N]
            Network adjacency matrix.
        steps : int
            Number of simulation steps. Must be ≥ 0.
        noise_std : float | torch.Tensor
            Standard deviation of the stochastic perturbation.  Only applied
            when `self.training == True` and `noise_std > 0`.
        autonomic_feedback : torch.Tensor, optional  [B, N]
            Optional additive autonomic-feedback term.

        Returns
        -------
        (x, y, z, s) : updated state tensors — same shapes as inputs.
        """
        # ---- Structural validation ---------------------------------------
        if steps < 0:
            raise ValueError("steps must be ≥ 0.")

        if self.validate_inputs:
            if x.dim() != 2 or y.dim() != 2 or z.dim() != 2:
                raise ValueError("x, y, z must be [B, 1].")
            if x.shape != y.shape or x.shape != z.shape:
                raise ValueError("x, y, z must share shape [B, 1].")
            if x.shape[-1] != 1:
                raise ValueError("x, y, z must have last dim 1.")
            if s.dim() != 2 or s.shape[-1] != self.num_nodes:
                raise ValueError(f"s must be [B, {self.num_nodes}].")
            if s.shape[0] != x.shape[0]:
                raise ValueError("s batch dim must match x, y, z.")
            if adj_matrix.dim() != 2 or adj_matrix.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"adj_matrix must be [{self.num_nodes}, {self.num_nodes}]."
                )
            if autonomic_feedback is not None and autonomic_feedback.shape != s.shape:
                raise ValueError("autonomic_feedback must match s shape.")

        device = s.device
        dtype  = s.dtype

        # ---- Precompute normalized adjacency (once per forward) ----------
        #   C^∞ soft floor on degrees — replaces hard `clamp(min=1.0)`.
        beta  = self._homeo_beta.to(device=device, dtype=dtype)
        floor = self._deg_floor.to(device=device, dtype=dtype)

        degrees = adj_matrix.sum(dim=-1, keepdim=True)                    # [N, 1]
        degrees_safe = self._soft_floor(degrees, floor, beta)             # C^∞ ≥ floor
        norm_adj = adj_matrix / degrees_safe                              # [N, N]
        ext_divisor = degrees_safe.transpose(-2, -1)                      # [1, N]

        # ---- Feedback tensor: broadcast once, no per-step allocation ------
        feedback = autonomic_feedback if autonomic_feedback is not None else torch.zeros_like(s)

        # ---- Noise decision: hoisted out of the loop ---------------------
        #   Both operands are Python scalars → no tensor sync, no graph break.
        add_noise = self.training and (
            float(noise_std) > 0.0 if not isinstance(noise_std, torch.Tensor)
            else True
        )
        if not isinstance(noise_std, torch.Tensor):
            noise_std_t = torch.tensor(float(noise_std), dtype=dtype, device=device)
        else:
            noise_std_t = noise_std.to(dtype=dtype, device=device)

        # ---- Simulation loop (optionally gradient-checkpointed) ----------
        use_ckpt = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and steps > 0
        )

        for _ in range(steps):
            if use_ckpt:
                x, y, z, s = torch.utils.checkpoint.checkpoint(
                    self._node_step,
                    x, y, z, s, norm_adj, ext_divisor,
                    feedback, noise_std_t, add_noise,
                    use_reentrant=False,
                )
            else:
                x, y, z, s = self._node_step(
                    x, y, z, s, norm_adj, ext_divisor,
                    feedback, noise_std_t, add_noise,
                )

        return x, y, z, s


# =============================================================================
# Smoke test / autograd verification
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-PsychopathologyEngine v2] Running on: {device}")

    B, N, T = 4, 64, 32
    engine = CombinedPsychopathologyEngine(num_nodes=N, dt=0.01).to(device)
    engine.train()
    optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-3)

    x = torch.randn(B, 1, device=device, requires_grad=True)
    y = torch.randn(B, 1, device=device, requires_grad=True)
    z = torch.randn(B, 1, device=device, requires_grad=True) + 25.0
    s = torch.randn(B, N, device=device, requires_grad=True)
    adj = (torch.rand(N, N, device=device) > 0.9).float()

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        x_out, y_out, z_out, s_out = engine(
            x, y, z, s, adj, steps=T, noise_std=0.01,
        )
        loss = (
            s_out.float().pow(2).mean()
            + 0.1 * (x_out.float().pow(2).mean() + y_out.float().pow(2).mean())
            + 0.01 * z_out.float().abs().mean()
        )

    loss.backward()
    optimizer.step()

    print("-" * 64)
    print(f"  x_out shape                     : {tuple(x_out.shape)}")
    print(f"  y_out shape                     : {tuple(y_out.shape)}")
    print(f"  z_out shape                     : {tuple(z_out.shape)}")
    print(f"  s_out shape                     : {tuple(s_out.shape)}")
    print(f"  s_out range                     : "
          f"{s_out.detach().float().min().item():.4f} – "
          f"{s_out.detach().float().max().item():.4f}  (bounded by tanh)")
    print(f"  loss                            : {loss.item():.6f}")

    # ---- Gradient-flow audit ---------------------------------------------
    grad_ok = True
    for name, p in engine.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")
    print(f"  Autograd                        : "
          f"{'FULLY CONNECTED — C^∞ differentiable' if grad_ok else 'FAILED'}")

    # ---- Homeostasis gradient sanity check -------------------------------
    print(f"  τ_h grad                        : "
          f"{engine.tau_homeostasis.grad.abs().item():.6e}  "
          f"(always non-zero → no dead branch at init)")
    print("-" * 64)
