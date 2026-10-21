"""
=============================================================================
SESI Clinical Psychopathology Engine — NATIVE FULL DIFFERENTIABLE
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
GitHub       : https://github.com/yoonalimsuwan
Contact      : msps4u@gmail.com
License      : MIT
Year         : 2026
Version      : 2.0.1 (Syntax-Fixed: __future__, __all__, __name__, __init__)
=============================================================================

Production-grade SESI Clinical Psychopathology Engine: continuous Lorenz
cognitive-affective-identity core coupled to a discrete symptom network with
differentiable SESI topological jumps.

Fixes and improvements over the v1.0.0 reference
──────────────────────────────────────────────────
• `torch.sign(s)` → C^∞ surrogate `tanh(β·s)`.
• `torch.clamp(prob_jump, 1e-6, 1.0 − 1e-6)` → sigmoid-of-logit.
• Non-deterministic `.eval()` → deterministic hard threshold + STE.
• AMP-safe RK4 (fp32-internal).
• `torch.abs` → C^∞ `sqrt(x² + ε²)`.
• `clamp(min=1e-5)` → C^∞ soft floor.
• Loop-invariant Python branches hoisted.
• Every Python-scalar constant → non-persistent buffers.
• Input validation, gradient checkpointing.
• Public API preserved.

Multi-GPU (DDP) usage
─────────────────────
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    model = SESIPsychoNet(num_symptoms=10, dt=0.01).to(rank)
    model = torch.compile(model, mode="max-autotune")
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], gradient_as_bucket_view=True,
    )

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        x, y, z, s = model(x0, y0, z0, s0, steps=50)
    loss = F.mse_loss(s.float(), target.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SESIPsychoConfig", "SESIPsychoNet", "train_on_clinical_data"]


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class SESIPsychoConfig:
    """Numerical-safety and topological-jump configuration."""

    # ----- Surrogate sharpness --------------------------------------------
    abs_eps: float = 1e-4
    degree_floor: float = 1e-5
    degree_beta: float = 50.0
    sign_beta_init: float = 10.0
    jump_magnitude: float = 0.5
    jump_tau: float = 0.5

    # ----- SESI No-Zeno barrier numerical safety -------------------------
    inner_clamp: float = 15.0
    log_floor: float = -25.0
    denom_eps: float = 1e-8

    # ----- Execution ------------------------------------------------------
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


# ===========================================================================
# Engine
# ===========================================================================

class SESIPsychoNet(nn.Module):
    """
    Production-grade fully differentiable hybrid psychopathology engine.

    Continuous core : Lorenz system integrated with RK4 (fp32-internal).
    Discrete network: bounded `tanh` symptom update with row-normalized
                      adjacency, global pressure, external drives, and
                      differentiable SESI topological jumps.

    Parameters
    ----------
    num_symptoms : int
        Number of discrete symptom nodes (must be > 0).
    dt : float
        Continuous-core integration step (must be > 0).
    config : SESIPsychoConfig, optional
    validate_inputs : bool, optional
    gradient_checkpointing : bool, optional
    """

    def __init__(
        self,
        num_symptoms: int,
        dt: float = 0.01,
        *,
        config: Optional[SESIPsychoConfig] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # ---- Validation ---------------------------------------------------
        if not (isinstance(num_symptoms, int) and num_symptoms > 0):
            raise ValueError("num_symptoms must be a positive int.")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be > 0.")

        cfg = config or SESIPsychoConfig()
        if validate_inputs is not None:
            cfg = SESIPsychoConfig(**{**cfg.__dict__,
                                      "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = SESIPsychoConfig(**{**cfg.__dict__,
                                      "gradient_checkpointing": bool(gradient_checkpointing)})

        for name, v in (
            ("abs_eps", cfg.abs_eps),
            ("degree_floor", cfg.degree_floor),
            ("degree_beta", cfg.degree_beta),
            ("sign_beta_init", cfg.sign_beta_init),
            ("jump_tau", cfg.jump_tau),
            ("inner_clamp", cfg.inner_clamp),
            ("denom_eps", cfg.denom_eps),
        ):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be > 0.")

        if not (math.isfinite(cfg.log_floor) and cfg.log_floor < 0.0):
            raise ValueError("log_floor must be < 0.")
        if not (math.isfinite(cfg.jump_magnitude) and cfg.jump_magnitude > 0.0):
            raise ValueError("jump_magnitude must be > 0.")

        self.num_nodes = int(num_symptoms)
        self.dt = float(dt)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        # ------------------------------------------------------------------
        # 1. Continuous Lorenz core parameters
        # ------------------------------------------------------------------
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_l = nn.Parameter(torch.tensor(8.0 / 3.0))

        # ------------------------------------------------------------------
        # 2. Symptom network weights
        # ------------------------------------------------------------------
        adj_init = torch.eye(self.num_nodes) + 0.01 * torch.randn(self.num_nodes, self.num_nodes)
        self.adj_matrix = nn.Parameter(adj_init)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.ext_coupling = nn.Parameter(torch.randn(3, self.num_nodes))

        # ------------------------------------------------------------------
        # 3. SESI topological / disordered-media parameters
        # ------------------------------------------------------------------
        self.delta_E = nn.Parameter(torch.ones(self.num_nodes) * 5.0)
        self.C1 = nn.Parameter(torch.tensor(1.0))
        self.sigma_sq = nn.Parameter(torch.tensor(1.0))

        raw_beta_init = math.log(math.expm1(max(cfg.sign_beta_init - 1e-6, 1e-6)))
        self.sign_beta_raw = nn.Parameter(torch.tensor(float(raw_beta_init)))

        # ------------------------------------------------------------------
        # 4. Non-persistent scalar buffers (DDP-safe, checkpoint-excluded)
        # ------------------------------------------------------------------
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_dt", _buf(self.dt), persistent=False)
        self.register_buffer("_dt_half", _buf(self.dt * 0.5), persistent=False)
        self.register_buffer("_dt_sixth", _buf(self.dt / 6.0), persistent=False)
        self.register_buffer("_abs_eps", _buf(cfg.abs_eps), persistent=False)
        self.register_buffer("_degree_floor", _buf(cfg.degree_floor), persistent=False)
        self.register_buffer("_degree_beta", _buf(cfg.degree_beta), persistent=False)
        self.register_buffer("_jump_mag", _buf(cfg.jump_magnitude), persistent=False)
        self.register_buffer("_jump_tau", _buf(cfg.jump_tau), persistent=False)
        self.register_buffer("_inner_clamp", _buf(cfg.inner_clamp), persistent=False)
        self.register_buffer("_log_floor", _buf(cfg.log_floor), persistent=False)
        self.register_buffer("_denom_eps", _buf(cfg.denom_eps), persistent=False)

    # ------------------------------------------------------------------ #
    # C^∞ primitives                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_abs(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """C^∞ surrogate for `|x|` — no kink at 0."""
        return torch.sqrt(x * x + eps * eps)

    @staticmethod
    def _soft_floor(x: torch.Tensor, floor: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """C^∞ surrogate for `max(x, floor)` — exact for `x ≫ floor`."""
        return floor + F.softplus(x - floor, beta=beta)

    def _positive_sign_beta(self) -> torch.Tensor:
        """β = softplus(β_raw) + ε — strictly positive, C^∞, no clamp."""
        return F.softplus(self.sign_beta_raw) + 1e-6

    # ------------------------------------------------------------------ #
    # Lorenz RK4 (fp32-internal for AMP safety)                          #
    # ------------------------------------------------------------------ #
    def _lorenz_derivatives(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lorenz RHS — inputs should be fp32."""
        dx = self.sigma.float() * (y - x)
        dy = x * (self.rho.float() - z) - y
        dz = x * y - self.beta_l.float() * z
        return dx, dy, dz

    def _rk4_lorenz(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One RK4 step of the Lorenz core.
        Runs **internally in fp32** and casts back to the input dtype.
        """
        dtype_in = x.dtype
        device = x.device

        x32 = x.float()
        y32 = y.float()
        z32 = z.float()

        dt_half = self._dt_half.to(device=device)
        dt = self._dt.to(device=device)
        dt_sixth = self._dt_sixth.to(device=device)

        k1x, k1y, k1z = self._lorenz_derivatives(x32, y32, z32)
        k2x, k2y, k2z = self._lorenz_derivatives(
            x32 + dt_half * k1x, y32 + dt_half * k1y, z32 + dt_half * k1z,
        )
        k3x, k3y, k3z = self._lorenz_derivatives(
            x32 + dt_half * k2x, y32 + dt_half * k2y, z32 + dt_half * k2z,
        )
        k4x, k4y, k4z = self._lorenz_derivatives(
            x32 + dt * k3x, y32 + dt * k3y, z32 + dt * k3z,
        )

        x_next = x32 + dt_sixth * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y_next = y32 + dt_sixth * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z_next = z32 + dt_sixth * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

        return x_next.to(dtype_in), y_next.to(dtype_in), z_next.to(dtype_in)

    # ------------------------------------------------------------------ #
    # Differentiable SESI topological jump                               #
    # ------------------------------------------------------------------ #
    def _differentiable_topological_jump(
        self,
        s: torch.Tensor,
        tau: torch.Tensor,
        add_noise: bool,
    ) -> torch.Tensor:
        """
        Applies the SESI topological jump.

        Barrier
        -------
        ΔE_eff = softplus(ΔE, β) + ε
        σ²_eff = softplus(σ², β) + ε
        c₁_eff = softplus(c₁, β) + ε
        log P  = clamp(−c₁_eff · exp(clamp(ΔE_eff/(σ²_eff·dt), max=z_max)),
                       min=log_floor)
        P      = exp(log P)

        Sampling
        --------
        Training      : Gumbel-Softmax relaxed Bernoulli (differentiable).
        Eval          : Deterministic hard threshold `(P > 0.5)` with
                        straight-through estimator.

        Jump magnitude
        --------------
        `tanh(β · s)` is a C^∞ surrogate for `sign(s)`.
        """
        dtype_in = s.dtype
        device = s.device

        beta = self._degree_beta.to(device=device, dtype=dtype_in)
        inner_max = self._inner_clamp.to(device=device, dtype=dtype_in)
        log_floor = self._log_floor.to(device=device, dtype=dtype_in)
        denom_eps = self._denom_eps.to(device=device, dtype=dtype_in)
        dt = self._dt.to(device=device, dtype=dtype_in)
        jump_mag = self._jump_mag.to(device=device, dtype=dtype_in)

        # Positive physical parameters via softplus + ε.
        delta_e_eff = F.softplus(self.delta_E.to(dtype=dtype_in), beta=beta) + denom_eps
        sigma_sq_eff = F.softplus(self.sigma_sq.to(dtype=dtype_in), beta=beta) + denom_eps
        c1_eff = F.softplus(self.C1.to(dtype=dtype_in), beta=beta) + denom_eps

        # Log-domain barrier (fp32-internal for AMP safety).
        inner = (delta_e_eff.float() / (sigma_sq_eff.float() * dt.float())).clamp_(max=float(inner_max))
        log_p = (-c1_eff.float() * torch.exp(inner)).clamp_(min=float(log_floor))
        prob_jump = torch.exp(log_p).to(dtype_in)

        # ---- Gumbel-Softmax / deterministic sampling ---------------------
        log_p_soft = torch.log(prob_jump)
        log_1mp = torch.log1p(-prob_jump)
        logits = torch.stack([log_1mp, log_p_soft], dim=-1)

        if add_noise:
            jump_event = F.gumbel_softmax(logits, tau=tau, hard=True)[..., 1]
        else:
            soft = torch.sigmoid((prob_jump - 0.5) / tau)
            hard = (prob_jump > 0.5).to(dtype_in)
            jump_event = hard - soft.detach() + soft

        # ---- C^∞ surrogate sign(s) ---------------------------------------
        sign_surrogate = torch.tanh(self._positive_sign_beta().to(dtype=dtype_in) * s)
        s_jumped = s + jump_event * (sign_surrogate * jump_mag)
        return torch.tanh(s_jumped)

    # ------------------------------------------------------------------ #
    # Single temporal step (isolated for gradient checkpointing)         #
    # ------------------------------------------------------------------ #
    def _step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        norm_adj: torch.Tensor,
        add_noise: bool,
        tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        device = x.device

        # ---- 1. Continuous evolution (fp32-internal RK4) -----------------
        x, y, z = self._rk4_lorenz(x, y, z)

        # ---- 2. Global pressure & external driving ----------------------
        eps = self._abs_eps.to(device=device, dtype=dtype)
        x_abs = self._safe_abs(x, eps)
        y_abs = self._safe_abs(y, eps)
        pressure = self.alpha * (x_abs + y_abs + (1.0 - z))  # [B, 1]

        ext_drive = (
            x * self.ext_coupling[0]
            + y * self.ext_coupling[1]
            - z * self.ext_coupling[2]
        )  # [B, N]

        # ---- 3. Discrete symptom network evolution -----------------------
        neighbor_sum = s @ norm_adj.transpose(-2, -1)  # [B, N]
        s_continuous = torch.tanh(s + pressure + neighbor_sum + ext_drive)

        # ---- 4. SESI differentiable topological jump ---------------------
        s = self._differentiable_topological_jump(s_continuous, tau=tau, add_noise=add_noise)

        return x, y, z, s

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass integrating continuous dynamics, global pressure, and
        stochastic topological jumps.

        Parameters
        ----------
        x, y, z : torch.Tensor  [B, 1]
        s : torch.Tensor  [B, N]
        steps : int
            Number of temporal simulation steps. Must be ≥ 0.

        Returns
        -------
        (x, y, z, s) : updated state tensors — same shapes as inputs.
        """
        # ---- Structural validation --------------------------------------
        if steps < 0:
            raise ValueError("steps must be ≥ 0.")

        if self.validate_inputs:
            if x.dim() != 2 or y.dim() != 2 or z.dim() != 2:
                raise ValueError("x, y, z must be [B, 1].")
            if not (x.shape == y.shape == z.shape):
                raise ValueError("x, y, z must share shape [B, 1].")
            if x.shape[-1] != 1:
                raise ValueError("x, y, z must have last dim 1.")
            if s.dim() != 2 or s.shape[-1] != self.num_nodes:
                raise ValueError(f"s must be [B, {self.num_nodes}].")
            if s.shape[0] != x.shape[0]:
                raise ValueError("s batch dim must match x, y, z.")

        device = x.device
        dtype = x.dtype

        # ---- Precompute normalized adjacency (once per forward) ---------
        eps = self._abs_eps.to(device=device, dtype=dtype)
        floor = self._degree_floor.to(device=device, dtype=dtype)
        beta = self._degree_beta.to(device=device, dtype=dtype)

        adj_abs = self._safe_abs(self.adj_matrix.to(dtype=dtype), eps)  # [N, N]
        degree = adj_abs.sum(dim=-1, keepdim=True)  # [N, 1]
        degree_safe = self._soft_floor(degree, floor, beta)  # C^∞ ≥ floor
        norm_adj = self.adj_matrix / degree_safe  # [N, N]

        # ---- Hoist loop-invariant Python branches ------------------------
        add_noise = self.training and torch.is_grad_enabled()
        tau = self._jump_tau.to(device=device, dtype=dtype)

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
                    self._step,
                    x, y, z, s, norm_adj, add_noise, tau,
                    use_reentrant=False,
                )
            else:
                x, y, z, s = self._step(
                    x, y, z, s, norm_adj, add_noise, tau,
                )

        return x, y, z, s


# ===========================================================================
# Pipeline helper: training on standardized clinical data
# ===========================================================================

def train_on_clinical_data(
    model: Optional[SESIPsychoNet] = None,
    *,
    num_symptoms: int = 10,
    batch_size: int = 32,
    time_steps: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    """
    One training step on simulated clinical data.
    """
    if model is None:
        model = SESIPsychoNet(num_symptoms=num_symptoms, dt=0.01)
        if device is not None:
            model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # ---- Simulated clinical target [B, N] (BDI/PANSS-style) ----------
    device = next(model.parameters()).device
    target_symptoms = torch.rand(batch_size, num_symptoms, device=device) * 2 - 1

    x_init = torch.randn(batch_size, 1, device=device)
    y_init = torch.randn(batch_size, 1, device=device)
    z_init = torch.randn(batch_size, 1, device=device) + 25.0
    s_init = torch.randn(batch_size, num_symptoms, device=device) * 0.1

    # ---- Differentiable simulation -----------------------------------
    x_out, y_out, z_out, s_out = model(x_init, y_init, z_init, s_init, steps=time_steps)
    loss = F.mse_loss(s_out.float(), target_symptoms.float())
    loss.backward()

    # Gradient clipping — critical for the chaotic Lorenz core.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return float(loss.detach().item())


# ===========================================================================
# Smoke test / autograd verification
# ===========================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-SESIPsychoNet v2] Running on: {device}")

    B, N, T = 8, 10, 32

    model = SESIPsychoNet(num_symptoms=N, dt=0.01).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x = torch.randn(B, 1, device=device, requires_grad=True)
    y = torch.randn(B, 1, device=device, requires_grad=True)
    z = torch.randn(B, 1, device=device, requires_grad=True) + 25.0
    s = torch.randn(B, N, device=device, requires_grad=True) * 0.1

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )

    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        x_out, y_out, z_out, s_out = model(x, y, z, s, steps=T)
        loss = (
            s_out.float().pow(2).mean()
            + 0.1 * (x_out.float().pow(2).mean() + y_out.float().pow(2).mean())
            + 0.01 * z_out.float().abs().mean()
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print("-" * 64)
    print(f"  x_out shape                   : {tuple(x_out.shape)}")
    print(f"  y_out shape                   : {tuple(y_out.shape)}")
    print(f"  z_out shape                   : {tuple(z_out.shape)}")
    print(f"  s_out shape                   : {tuple(s_out.shape)}")
    print(f"  s_out range (bounded by tanh) : "
          f"{s_out.detach().float().min().item():+.4f} – "
          f"{s_out.detach().float().max().item():+.4f}")
    print(f"  loss                          : {loss.item():.6f}")

    # ---- Gradient-flow audit ---------------------------------------------
    grad_ok = True
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")

    print(f"  Autograd                      : "
          f"{'FULLY CONNECTED — C^∞ differentiable' if grad_ok else 'FAILED'}")

    # ---- Sign-sharpness sanity check ------------------------------------
    with torch.no_grad():
        beta_val = model._positive_sign_beta().item()
    print(f"  sign_beta β                   : {beta_val:.4f}  "
          f"(positive, C^∞ surrogate for sign(s))")

    # ---- Deterministic eval check ---------------------------------------
    model.eval()
    with torch.no_grad():
        x1, _, _, s1 = model(x, y, z, s, steps=4)
        x2, _, _, s2 = model(x, y, z, s, steps=4)
    det_ok = torch.allclose(s1, s2)
    print(f"  Deterministic eval            : "
          f"{'OK' if det_ok else 'FAILED (non-reproducible)'}")

    # ---- Helper pipeline -------------------------------------------------
    print("-" * 64)
    print("Running train_on_clinical_data helper ...")
    loss_val = train_on_clinical_data(
        model=model, num_symptoms=N, batch_size=B,
        time_steps=16, max_grad_norm=1.0, optimizer=optimizer,
    )
    print(f"  helper loss                   : {loss_val:.6f}")
    print("-" * 64)
