"""
=============================================================================
Hybrid Dynamical Energy Network Model — NATIVE FULL DIFFERENTIABLE
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 2.0.1 (Syntax-Fixed: __future__, __all__, __name__, __init__)
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PsychoDynamoConfig", "PsychoDynamoNet"]


@dataclass
class PsychoDynamoConfig:
    degree_floor: float = 1.0
    degree_beta: float = 50.0
    abs_eps: float = 1e-4
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


class PsychoDynamoNet(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        dt: float = 0.01,
        *,
        config: Optional[PsychoDynamoConfig] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not (isinstance(num_nodes, int) and num_nodes > 0):
            raise ValueError("num_nodes must be a positive int.")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be > 0.")

        cfg = config or PsychoDynamoConfig()
        if validate_inputs is not None:
            cfg = PsychoDynamoConfig(**{**cfg.__dict__,
                                        "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = PsychoDynamoConfig(**{**cfg.__dict__,
                                        "gradient_checkpointing": bool(gradient_checkpointing)})

        for name, v in (
            ("degree_floor", cfg.degree_floor),
            ("degree_beta", cfg.degree_beta),
            ("abs_eps", cfg.abs_eps),
        ):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be > 0.")

        self.num_nodes = int(num_nodes)
        self.dt = float(dt)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        self.sigma       = nn.Parameter(torch.tensor(10.0))
        self.rho         = nn.Parameter(torch.tensor(28.0))
        self.beta_lorenz = nn.Parameter(torch.tensor(8.0 / 3.0))

        self.k1 = nn.Parameter(torch.tensor(1.0))
        self.k2 = nn.Parameter(torch.tensor(1.0))
        self.k3 = nn.Parameter(torch.tensor(1.0))

        self.gamma   = nn.Parameter(torch.randn(num_nodes))
        self.delta   = nn.Parameter(torch.randn(num_nodes))
        self.epsilon = nn.Parameter(torch.randn(num_nodes))

        self.alpha    = nn.Parameter(torch.tensor(1.0))
        self.beta_net = nn.Parameter(torch.tensor(1.0))

        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_dt",          _buf(self.dt),              persistent=False)
        self.register_buffer("_dt_half",     _buf(self.dt * 0.5),        persistent=False)
        self.register_buffer("_dt_sixth",    _buf(self.dt / 6.0),        persistent=False)
        self.register_buffer("_abs_eps",     _buf(cfg.abs_eps),          persistent=False)
        self.register_buffer("_deg_floor",   _buf(cfg.degree_floor),     persistent=False)
        self.register_buffer("_deg_beta",    _buf(cfg.degree_beta),      persistent=False)

    @staticmethod
    def _soft_floor(x: torch.Tensor, floor: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return floor + F.softplus(x - floor, beta=beta)

    @staticmethod
    def _safe_abs(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x * x + eps * eps)

    def _lorenz_derivatives(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = self.sigma.float() * (y - x)
        dy = x * (self.rho.float() - z) - y
        dz = x * y - self.beta_lorenz.float() * z
        return dx, dy, dz

    def _rk4_step(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype_in = x.dtype
        device   = x.device
        x32 = x.float()
        y32 = y.float()
        z32 = z.float()

        dt       = self._dt.to(device=device)
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
            x32 + dt * k3x, y32 + dt * k3y, z32 + dt * k3z,
        )

        x_next = x32 + dt_sixth * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y_next = y32 + dt_sixth * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z_next = z32 + dt_sixth * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

        return x_next.to(dtype_in), y_next.to(dtype_in), z_next.to(dtype_in)

    def _node_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        norm_adj: torch.Tensor,
        deg_transpose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, z = self._rk4_step(x, y, z)

        eps   = self._abs_eps.to(device=x.device, dtype=x.dtype)
        x_abs = self._safe_abs(x, eps)
        y_abs = self._safe_abs(y, eps)
        P_t = self.k1 * x_abs + self.k2 * y_abs + self.k3 * (1.0 - z)

        ext_drive = (
            self.gamma.unsqueeze(0) * x
            + self.delta.unsqueeze(0) * y
            - self.epsilon.unsqueeze(0) * z
        )
        ext_drive_norm = ext_drive / deg_transpose

        neighbor_sum = s @ norm_adj.transpose(-2, -1)

        s_next = torch.tanh(
            s
            + self.alpha * P_t
            + self.beta_net * neighbor_sum
            + ext_drive_norm
        )
        return x, y, z, s_next

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        adj_matrix: torch.Tensor,
        steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if steps < 0:
            raise ValueError("steps must be >= 0.")

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
            if adj_matrix.dim() != 2 or adj_matrix.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"adj_matrix must be [{self.num_nodes}, {self.num_nodes}]."
                )

        device = s.device
        dtype  = s.dtype

        floor = self._deg_floor.to(device=device, dtype=dtype)
        beta  = self._deg_beta.to(device=device, dtype=dtype)

        degrees = adj_matrix.sum(dim=-1, keepdim=True)
        degrees_safe = self._soft_floor(degrees, floor, beta)
        norm_adj = adj_matrix / degrees_safe
        deg_transpose = degrees_safe.transpose(-2, -1)

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
                    x, y, z, s, norm_adj, deg_transpose,
                    use_reentrant=False,
                )
            else:
                x, y, z, s = self._node_step(
                    x, y, z, s, norm_adj, deg_transpose,
                )

        return x, y, z, s


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-HybridDynamicalEnergyNetwork v2.0.1] Running on: {device}")

    B, N, T = 4, 64, 32

    engine = PsychoDynamoNet(num_nodes=N, dt=0.01).to(device)
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
        x_out, y_out, z_out, s_out = engine(x, y, z, s, adj, steps=T)
        loss = (
            s_out.float().pow(2).mean()
            + 0.1 * (x_out.float().pow(2).mean() + y_out.float().pow(2).mean())
            + 0.01 * z_out.float().abs().mean()
        )

    loss.backward()
    optimizer.step()

    print("-" * 64)
    print(f"  x_out shape              : {tuple(x_out.shape)}")
    print(f"  y_out shape              : {tuple(y_out.shape)}")
    print(f"  z_out shape              : {tuple(z_out.shape)}")
    print(f"  s_out shape              : {tuple(s_out.shape)}")
    print(f"  s_out range              : "
          f"{s_out.detach().float().min().item():.4f} – "
          f"{s_out.detach().float().max().item():.4f}  (bounded by tanh)")
    print(f"  loss                     : {loss.item():.6f}")

    grad_ok = True
    for name, p in engine.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")

    print(f"  Autograd                 : "
          f"{'FULLY CONNECTED — C^∞ differentiable' if grad_ok else 'FAILED'}")
    print("-" * 64)
