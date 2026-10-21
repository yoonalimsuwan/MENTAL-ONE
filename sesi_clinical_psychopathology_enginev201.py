"""
=============================================================================
SESI Clinical Psychopathology Engine — NATIVE FULL DIFFERENTIABLE
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT
Year         : 2026
Version      : 2.0.1 (Bug-Fixed: tanh sign surrogate, deterministic eval, AMP-safe RK4)
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SESIPsychoConfig", "SESIPsychoNet", "train_on_clinical_data"]


@dataclass
class SESIPsychoConfig:
    abs_eps: float = 1e-4
    degree_floor: float = 1e-5
    degree_beta: float = 50.0
    sign_beta_init: float = 10.0
    jump_magnitude: float = 0.5
    jump_tau: float = 0.5
    inner_clamp: float = 15.0
    log_floor: float = -25.0
    denom_eps: float = 1e-8
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


class SESIPsychoNet(nn.Module):
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
        if not (isinstance(num_symptoms, int) and num_symptoms > 0):
            raise ValueError("num_symptoms must be a positive int.")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be > 0.")

        cfg = config or SESIPsychoConfig()
        if validate_inputs is not None:
            cfg = SESIPsychoConfig(**{**cfg.__dict__, "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = SESIPsychoConfig(**{**cfg.__dict__, "gradient_checkpointing": bool(gradient_checkpointing)})

        self.num_nodes = int(num_symptoms)
        self.dt = float(dt)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        # Lorenz parameters
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_l = nn.Parameter(torch.tensor(8.0 / 3.0))

        # Symptom network
        adj_init = torch.eye(self.num_nodes) + 0.01 * torch.randn(self.num_nodes, self.num_nodes)
        self.adj_matrix = nn.Parameter(adj_init)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.ext_coupling = nn.Parameter(torch.randn(3, self.num_nodes))

        # SESI topological parameters
        self.delta_E = nn.Parameter(torch.ones(self.num_nodes) * 5.0)
        self.C1 = nn.Parameter(torch.tensor(1.0))
        self.sigma_sq = nn.Parameter(torch.tensor(1.0))

        raw_beta_init = math.log(math.expm1(max(cfg.sign_beta_init - 1e-6, 1e-6)))
        self.sign_beta_raw = nn.Parameter(torch.tensor(float(raw_beta_init)))

        # Buffers
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

    @staticmethod
    def _safe_abs(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x * x + eps * eps)

    @staticmethod
    def _soft_floor(x: torch.Tensor, floor: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return floor + F.softplus(x - floor, beta=beta)

    def _positive_sign_beta(self) -> torch.Tensor:
        return F.softplus(self.sign_beta_raw) + 1e-6

    def _lorenz_derivatives(self, x, y, z):
        dx = self.sigma.float() * (y - x)
        dy = x * (self.rho.float() - z) - y
        dz = x * y - self.beta_l.float() * z
        return dx, dy, dz

    def _rk4_lorenz(self, x, y, z):
        dtype_in = x.dtype
        device = x.device
        x32, y32, z32 = x.float(), y.float(), z.float()
        dt_half = self._dt_half.to(device=device)
        dt = self._dt.to(device=device)
        dt_sixth = self._dt_sixth.to(device=device)

        k1x, k1y, k1z = self._lorenz_derivatives(x32, y32, z32)
        k2x, k2y, k2z = self._lorenz_derivatives(x32 + dt_half * k1x, y32 + dt_half * k1y, z32 + dt_half * k1z)
        k3x, k3y, k3z = self._lorenz_derivatives(x32 + dt_half * k2x, y32 + dt_half * k2y, z32 + dt_half * k2z)
        k4x, k4y, k4z = self._lorenz_derivatives(x32 + dt * k3x, y32 + dt * k3y, z32 + dt * k3z)

        x_next = x32 + dt_sixth * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_next = y32 + dt_sixth * (k1y + 2 * k2y + 2 * k3y + k4y)
        z_next = z32 + dt_sixth * (k1z + 2 * k2z + 2 * k3z + k4z)
        return x_next.to(dtype_in), y_next.to(dtype_in), z_next.to(dtype_in)

    def _differentiable_topological_jump(self, s, tau, add_noise):
        dtype_in = s.dtype
        device = s.device
        beta = self._degree_beta.to(device=device, dtype=dtype_in)
        inner_max = self._inner_clamp.to(device=device, dtype=dtype_in)
        log_floor = self._log_floor.to(device=device, dtype=dtype_in)
        denom_eps = self._denom_eps.to(device=device, dtype=dtype_in)
        dt = self._dt.to(device=device, dtype=dtype_in)
        jump_mag = self._jump_mag.to(device=device, dtype=dtype_in)

        delta_e_eff = F.softplus(self.delta_E.to(dtype=dtype_in), beta=beta) + denom_eps
        sigma_sq_eff = F.softplus(self.sigma_sq.to(dtype=dtype_in), beta=beta) + denom_eps
        c1_eff = F.softplus(self.C1.to(dtype=dtype_in), beta=beta) + denom_eps

        inner = (delta_e_eff.float() / (sigma_sq_eff.float() * dt.float())).clamp_(max=float(inner_max.item()))
        log_p = (-c1_eff.float() * torch.exp(inner)).clamp_(min=float(log_floor.item()))
        prob_jump = torch.exp(log_p).to(dtype_in)

        log_p_soft = torch.log(prob_jump)
        log_1mp = torch.log1p(-prob_jump)
        logits = torch.stack([log_1mp, log_p_soft], dim=-1)

        if add_noise:
            jump_event = F.gumbel_softmax(logits, tau=tau, hard=True)[..., 1]
        else:
            soft = torch.sigmoid((prob_jump - 0.5) / tau)
            hard = (prob_jump > 0.5).to(dtype_in)
            jump_event = hard - soft.detach() + soft

        sign_surrogate = torch.tanh(self._positive_sign_beta().to(dtype=dtype_in) * s)
        s_jumped = s + jump_event * (sign_surrogate * jump_mag)
        return torch.tanh(s_jumped)

    def _step(self, x, y, z, s, norm_adj, add_noise, tau):
        dtype = x.dtype
        device = x.device
        x, y, z = self._rk4_lorenz(x, y, z)
        eps = self._abs_eps.to(device=device, dtype=dtype)
        x_abs = self._safe_abs(x, eps)
        y_abs = self._safe_abs(y, eps)
        pressure = self.alpha * (x_abs + y_abs + (1.0 - z))
        ext_drive = x * self.ext_coupling[0] + y * self.ext_coupling[1] - z * self.ext_coupling[2]
        neighbor_sum = s @ norm_adj.transpose(-2, -1)
        s_continuous = torch.tanh(s + pressure + neighbor_sum + ext_drive)
        s = self._differentiable_topological_jump(s_continuous, tau=tau, add_noise=add_noise)
        return x, y, z, s

    def forward(self, x, y, z, s, steps: int = 1):
        if steps < 0:
            raise ValueError("steps must be >= 0.")
        device = x.device
        dtype = x.dtype
        eps = self._abs_eps.to(device=device, dtype=dtype)
        floor = self._degree_floor.to(device=device, dtype=dtype)
        beta = self._degree_beta.to(device=device, dtype=dtype)

        adj_abs = self._safe_abs(self.adj_matrix.to(dtype=dtype), eps)
        degree = adj_abs.sum(dim=-1, keepdim=True)
        degree_safe = self._soft_floor(degree, floor, beta)
        norm_adj = self.adj_matrix / degree_safe

        add_noise = self.training and torch.is_grad_enabled()
        tau = self._jump_tau.to(device=device, dtype=dtype)

        for _ in range(steps):
            x, y, z, s = self._step(x, y, z, s, norm_adj, add_noise, tau)
        return x, y, z, s


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
    if model is None:
        model = SESIPsychoNet(num_symptoms=num_symptoms, dt=0.01)
        if device is not None:
            model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device

    target_symptoms = torch.rand(batch_size, num_symptoms, device=device) * 2 - 1
    x_init = torch.randn(batch_size, 1, device=device)
    y_init = torch.randn(batch_size, 1, device=device)
    z_init = torch.randn(batch_size, 1, device=device) + 25.0
    s_init = torch.randn(batch_size, num_symptoms, device=device) * 0.1

    x_out, y_out, z_out, s_out = model(x_init, y_init, z_init, s_init, steps=time_steps)
    loss = F.mse_loss(s_out.float(), target_symptoms.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return float(loss.detach().item())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-SESIPsychoNet v2] Running on: {device}")

    B, N, T = 8, 10, 32
    model = SESIPsychoNet(num_symptoms=N, dt=0.01).to(device)
    model.train()

    x = torch.randn(B, 1, device=device)
    y = torch.randn(B, 1, device=device)
    z = torch.randn(B, 1, device=device) + 25.0
    s = torch.randn(B, N, device=device) * 0.1

    x_out, y_out, z_out, s_out = model(x, y, z, s, steps=T)
    loss = s_out.pow(2).mean()
    loss.backward()

    print(f"  loss: {loss.item():.6f}")
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"  Autograd: {'PASS' if grad_ok else 'FAIL'}")
