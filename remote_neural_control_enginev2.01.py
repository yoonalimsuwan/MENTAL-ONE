"""
=============================================================================
Remote Neural Control Engine — NATIVE FULL DIFFERENTIABLE
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 2.0.1 (Bug-Fixed: batch control fields, Optional[Callable])
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RemoteNeuralControlConfig",
    "RemoteNeuralControlEngine",
    "optimize_neural_control",
]


# ===========================================================================
# Reference-module shims (standalone importable)
# ===========================================================================

class PiecewiseDFTAccumulator3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class CovariantMaxwellStructuralBridge(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


@dataclass
class CahnHilliardConfig:
    dx: float = 1.0
    dt: float = 1e-3
    laplacian: str = "conv3d"
    scheme: str = "explicit"


class StructuralCahnHilliard3D(nn.Module):
    def __init__(self, cfg: CahnHilliardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        k = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        k[0, 0, 1, 1, 1] = -6.0
        for (a, b, c) in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
            k[0, 0, a, b, c] = 1.0
        self.register_buffer("_lap", k / (cfg.dx ** 2), persistent=False)
        self.register_buffer("_dt", torch.tensor(float(cfg.dt), dtype=torch.float32), persistent=False)

    def step(self, u: torch.Tensor) -> torch.Tensor:
        batched = u.dim() == 5
        u5 = u if batched else u.unsqueeze(1)
        up = F.pad(u5, (1, 1, 1, 1, 1, 1), mode="circular")
        lap = F.conv3d(up, self._lap.to(u5.dtype))
        out = u5 + self._dt.to(u5.dtype) * lap
        return out if batched else out.squeeze(1)


class AdvancedStructuralLangevin(nn.Module):
    def __init__(self, dt: float = 1e-3) -> None:
        super().__init__()
        self.register_buffer("_dt", torch.tensor(float(dt), dtype=torch.float32), persistent=False)

    def full_step(
        self, coords: torch.Tensor, velocities: torch.Tensor,
        force_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        force = force_fn(coords, velocities) if force_fn is not None else torch.zeros_like(coords)
        dt = self._dt.to(coords.dtype)
        new_vel = velocities + force * dt
        new_coords = coords + new_vel * dt
        energy = torch.zeros((), device=coords.device, dtype=coords.dtype)
        aux = torch.zeros((), device=coords.device, dtype=coords.dtype)
        return new_coords, new_vel, energy, aux


class ExactMaxwellStructuralSolver(nn.Module):
    def __init__(self, dx: float = 1.0, dt: float = 1e-3, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.register_buffer("_anchor", torch.zeros(1, dtype=torch.float32), persistent=False)
        if device is not None:
            self.to(device)

    def step(self, e_field, b_field, order_parameter):
        aux = torch.zeros((), device=e_field.device, dtype=e_field.dtype)
        return e_field, b_field, order_parameter, aux


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class RemoteNeuralControlConfig:
    num_control_channels: int = 16
    dx: float = 1.0
    dt: float = 1e-3
    power_penalty: float = 1e-4
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


# ===========================================================================
# Engine
# ===========================================================================

class RemoteNeuralControlEngine(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        num_control_channels: int = 16,
        dx: float = 1.0,
        dt: float = 1e-3,
        device: Optional[torch.device] = None,
        *,
        config: Optional[RemoteNeuralControlConfig] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()
        grid_shape = tuple(int(s) for s in grid_shape)
        if len(grid_shape) != 3 or any(s <= 0 for s in grid_shape):
            raise ValueError("grid_shape must be a positive 3-tuple.")
        if not (isinstance(num_control_channels, int) and num_control_channels > 0):
            raise ValueError("num_control_channels must be a positive int.")

        cfg = config or RemoteNeuralControlConfig()
        cfg = RemoteNeuralControlConfig(**{
            **cfg.__dict__,
            "num_control_channels": int(num_control_channels),
            "dx": float(dx),
            "dt": float(dt),
        })
        if validate_inputs is not None:
            cfg = RemoteNeuralControlConfig(**{**cfg.__dict__, "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = RemoteNeuralControlConfig(**{**cfg.__dict__, "gradient_checkpointing": bool(gradient_checkpointing)})

        self.cfg = cfg
        self.grid_shape = grid_shape
        self.num_control_channels = cfg.num_control_channels
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        ch_cfg = CahnHilliardConfig(dx=cfg.dx, dt=cfg.dt)
        self.tissue_solver = StructuralCahnHilliard3D(ch_cfg)
        self.ion_solver = AdvancedStructuralLangevin(dt=cfg.dt)
        self.maxwell_solver = ExactMaxwellStructuralSolver(dx=cfg.dx, dt=cfg.dt)

        if device is not None:
            self.tissue_solver = self.tissue_solver.to(device)
            self.ion_solver = self.ion_solver.to(device)
            self.maxwell_solver = self.maxwell_solver.to(device)

        self.em_control_weights = nn.Parameter(torch.zeros(cfg.num_control_channels, dtype=torch.float32))

        num_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.spatial_projection = nn.Linear(cfg.num_control_channels, num_voxels, bias=False)
        nn.init.xavier_normal_(self.spatial_projection.weight)

        self.register_buffer("_dt", torch.tensor(float(cfg.dt), dtype=torch.float32), persistent=False)
        self.register_buffer("_power_penalty", torch.tensor(float(cfg.power_penalty), dtype=torch.float32), persistent=False)

    def _generate_control_fields(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (isinstance(batch_size, int) and batch_size > 0):
            raise ValueError("batch_size must be a positive int.")
        base = self.spatial_projection(self.em_control_weights).view(self.grid_shape)
        base_b = base.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, *self.grid_shape)
        zeros = torch.zeros_like(base_b)
        e_field = torch.cat([base_b, zeros, zeros], dim=1)
        b_field = torch.cat([zeros, base_b, zeros], dim=1)
        return e_field, b_field

    def _coupled_step(self, e_total, b_total, tissue_phase, ion_coords, ion_vel, ion_forces_fn):
        e32, b32 = e_total.float(), b_total.float()
        tp32, ic32, iv32 = tissue_phase.float(), ion_coords.float(), ion_vel.float()
        ic32, iv32, _, _ = self.ion_solver.full_step(coords=ic32, velocities=iv32, force_fn=ion_forces_fn)
        tp32 = self.tissue_solver.step(u=tp32)
        e32, b32, tp32, _ = self.maxwell_solver.step(e_field=e32, b_field=b32, order_parameter=tp32)
        return (e32.to(e_total.dtype), b32.to(b_total.dtype), tp32.to(tissue_phase.dtype),
                ic32.to(ion_coords.dtype), iv32.to(ion_vel.dtype))

    def forward(
        self,
        ambient_e_field: torch.Tensor,
        ambient_b_field: torch.Tensor,
        tissue_phase: torch.Tensor,
        ion_coords: torch.Tensor,
        ion_vel: torch.Tensor,
        ion_forces_fn: Optional[Callable] = None,
        *,
        steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if steps < 0:
            raise ValueError("steps must be >= 0.")
        B = ambient_e_field.shape[0]
        control_e, control_b = self._generate_control_fields(batch_size=B)
        control_e = control_e.to(dtype=ambient_e_field.dtype, device=ambient_e_field.device)
        control_b = control_b.to(dtype=ambient_b_field.dtype, device=ambient_b_field.device)
        e_total = ambient_e_field + control_e
        b_total = ambient_b_field + control_b

        for _ in range(steps):
            e_total, b_total, tissue_phase, ion_coords, ion_vel = self._coupled_step(
                e_total, b_total, tissue_phase, ion_coords, ion_vel, ion_forces_fn)

        return tissue_phase, ion_coords, e_total


def optimize_neural_control(
    controller: nn.Module,
    target_tissue: torch.Tensor,
    target_ions: torch.Tensor,
    initial_state: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    ion_forces_fn: Optional[Callable] = None,
    steps: int = 1,
) -> float:
    controller.train()
    optimizer.zero_grad(set_to_none=True)
    pred_tissue, pred_ions, _ = controller(
        ambient_e_field=initial_state["ambient_e_field"],
        ambient_b_field=initial_state["ambient_b_field"],
        tissue_phase=initial_state["tissue_phase"],
        ion_coords=initial_state["ion_coords"],
        ion_vel=initial_state["ion_vel"],
        ion_forces_fn=ion_forces_fn,
        steps=steps,
    )
    loss_tissue = F.mse_loss(pred_tissue, target_tissue)
    loss_ions = F.mse_loss(pred_ions, target_ions)
    inner = controller.module if hasattr(controller, "module") else controller
    w = inner.em_control_weights
    power_penalty = (w * w).sum() * inner._power_penalty
    total_loss = loss_tissue + loss_ions + power_penalty
    total_loss.backward()
    optimizer.step()
    return float(total_loss.detach().item())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-RemoteNeuralControlEngine v2] Running on: {device}")

    D, H, W = 8, 8, 8
    controller = RemoteNeuralControlEngine(grid_shape=(D, H, W), num_control_channels=8).to(device)
    controller.train()

    B, N_IONS = 2, 16
    ambient_e = torch.randn(B, 3, D, H, W, device=device) * 0.01
    ambient_b = torch.randn(B, 3, D, H, W, device=device) * 0.01
    tissue_phase = torch.randn(B, 1, D, H, W, device=device) * 0.1
    ion_coords = torch.randn(B, N_IONS, 3, device=device) * 0.1
    ion_vel = torch.randn(B, N_IONS, 3, device=device) * 0.01

    tissue_out, ions_out, e_total = controller(
        ambient_e, ambient_b, tissue_phase, ion_coords, ion_vel, steps=4)
    loss = tissue_out.pow(2).mean() + ions_out.pow(2).mean()
    loss.backward()

    print(f"  loss: {loss.item():.6f}")
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in controller.parameters())
    print(f"  Autograd: {'PASS' if grad_ok else 'FAIL'}")
