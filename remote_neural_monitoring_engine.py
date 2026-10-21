"""
=============================================================================
Remote Neural Monitoring Engine — NATIVE FULL DIFFERENTIABLE
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 2.0.1 (Bug-Fixed: math import, DDP-safe buffers, GroupNorm)
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DifferentiableForwardPhysics",
    "EfficientInverseSolver",
    "RemoteNeuralMonitoringEndToEnd",
    "PhysicsInformedLoss",
]


# ===========================================================================
# Forward physics (analytic, differentiable)
# ===========================================================================

class DifferentiableForwardPhysics(nn.Module):
    """
    Differentiable lead field forward operator.
        sensor_signals = L @ brain_sources
    """

    def __init__(
        self,
        num_sensors: int,
        num_voxels: int,
        *,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not (isinstance(num_sensors, int) and num_sensors > 0):
            raise ValueError("num_sensors must be a positive int.")
        if not (isinstance(num_voxels, int) and num_voxels > 0):
            raise ValueError("num_voxels must be a positive int.")

        self.num_sensors = int(num_sensors)
        self.num_voxels = int(num_voxels)
        self.validate_inputs = bool(validate_inputs)

        lfm = torch.empty(num_sensors, num_voxels, dtype=torch.float32)
        nn.init.kaiming_uniform_(lfm, a=math.sqrt(5))
        self.register_buffer("lead_field", lfm, persistent=True)

    def update_lead_field(self, new_lfm: torch.Tensor) -> None:
        if not isinstance(new_lfm, torch.Tensor):
            raise ValueError("new_lfm must be a torch.Tensor.")
        if new_lfm.shape != self.lead_field.shape:
            raise ValueError(
                f"new_lfm must be [{self.num_sensors}, {self.num_voxels}]; "
                f"got {tuple(new_lfm.shape)}."
            )
        with torch.no_grad():
            self.lead_field.copy_(
                new_lfm.to(dtype=self.lead_field.dtype, device=self.lead_field.device)
            )

    def forward(self, brain_sources: torch.Tensor) -> torch.Tensor:
        if self.validate_inputs:
            if brain_sources.dim() != 3:
                raise ValueError("brain_sources must be [B, V, T].")
            if brain_sources.shape[1] != self.num_voxels:
                raise ValueError(
                    f"brain_sources must have V={self.num_voxels} voxels, "
                    f"got {brain_sources.shape[1]}."
                )
        return self.lead_field.to(brain_sources.dtype) @ brain_sources


# ===========================================================================
# Amortized inverse solver
# ===========================================================================

class EfficientInverseSolver(nn.Module):
    """
    Amortized neural inverse model: reconstructs 3D brain sources from
    external sensor signals.
    """

    def __init__(
        self,
        num_sensors: int,
        num_voxels: int,
        hidden_dim: int = 128,
        *,
        conv_groups: int = 8,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not (isinstance(num_sensors, int) and num_sensors > 0):
            raise ValueError("num_sensors must be a positive int.")
        if not (isinstance(num_voxels, int) and num_voxels > 0):
            raise ValueError("num_voxels must be a positive int.")
        if not (isinstance(hidden_dim, int) and hidden_dim > 0):
            raise ValueError("hidden_dim must be a positive int.")
        if not (isinstance(conv_groups, int) and conv_groups > 0):
            raise ValueError("conv_groups must be a positive int.")
        if num_sensors % conv_groups != 0 or hidden_dim % conv_groups != 0:
            raise ValueError("conv_groups must divide both num_sensors and hidden_dim.")

        self.num_sensors = int(num_sensors)
        self.num_voxels = int(num_voxels)
        self.hidden_dim = int(hidden_dim)
        self.validate_inputs = bool(validate_inputs)

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(num_sensors, hidden_dim, kernel_size=3, padding=1, groups=conv_groups),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
        )

        self.spatial_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, num_voxels),
        )

        self.refinement_gate = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sensor_signals: torch.Tensor) -> torch.Tensor:
        if self.validate_inputs:
            if sensor_signals.dim() != 3:
                raise ValueError("sensor_signals must be [B, S, T].")
            if sensor_signals.shape[1] != self.num_sensors:
                raise ValueError(
                    f"sensor_signals must have S={self.num_sensors}, "
                    f"got {sensor_signals.shape[1]}."
                )
        x_temp = self.temporal_encoder(sensor_signals)
        x_temp = x_temp.transpose(1, 2)
        raw_sources = self.spatial_projector(x_temp)
        raw_sources = raw_sources.transpose(1, 2)
        gate = torch.sigmoid(raw_sources * self.refinement_gate)
        return raw_sources * gate


# ===========================================================================
# End-to-end pipeline
# ===========================================================================

class RemoteNeuralMonitoringEndToEnd(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        num_voxels: int,
        hidden_dim: int = 128,
        *,
        conv_groups: int = 8,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.inverse_solver = EfficientInverseSolver(
            num_sensors, num_voxels, hidden_dim,
            conv_groups=conv_groups, validate_inputs=validate_inputs,
        )
        self.forward_physics = DifferentiableForwardPhysics(
            num_sensors, num_voxels, validate_inputs=validate_inputs,
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

    def _step(self, sensor_signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_brain_sources = self.inverse_solver(sensor_signals)
        reconstructed_signals = self.forward_physics(predicted_brain_sources)
        return predicted_brain_sources, reconstructed_signals

    def forward(self, sensor_signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.validate_inputs and sensor_signals.dim() != 3:
            raise ValueError("sensor_signals must be [B, S, T].")
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._step, sensor_signals, use_reentrant=False,
            )
        return self._step(sensor_signals)

    @torch.no_grad()
    def infer_only(self, sensor_signals: torch.Tensor) -> torch.Tensor:
        return self.inverse_solver(sensor_signals)


# ===========================================================================
# Physics-informed loss
# ===========================================================================

class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        sparsity_weight: float = 1e-3,
        tv_weight: float = 1e-4,
        *,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        if not (math.isfinite(sparsity_weight) and sparsity_weight >= 0.0):
            raise ValueError("sparsity_weight must be >= 0.")
        if not (math.isfinite(tv_weight) and tv_weight >= 0.0):
            raise ValueError("tv_weight must be >= 0.")
        if reduction not in ("none", "mean"):
            raise ValueError("reduction must be 'none' or 'mean'.")

        self.register_buffer("_lambda_sparse", torch.tensor(float(sparsity_weight)), persistent=False)
        self.register_buffer("_lambda_tv", torch.tensor(float(tv_weight)), persistent=False)
        self.reduction = reduction

    def forward(
        self,
        original_signals: torch.Tensor,
        reconstructed_signals: torch.Tensor,
        predicted_sources: torch.Tensor,
    ) -> torch.Tensor:
        dtype = predicted_sources.dtype
        device = predicted_sources.device
        lam_sparse = self._lambda_sparse.to(device=device, dtype=dtype)
        lam_tv = self._lambda_tv.to(device=device, dtype=dtype)

        diff = (reconstructed_signals - original_signals).float()
        fidelity = (diff * diff).mean(dim=(1, 2))

        sources_f = predicted_sources.float()
        sparsity = sources_f.abs().mean(dim=(1, 2))

        if sources_f.shape[-1] > 1:
            tv = (sources_f[:, :, 1:] - sources_f[:, :, :-1]).abs().mean(dim=(1, 2))
        else:
            tv = torch.zeros_like(fidelity)

        total = fidelity + lam_sparse * sparsity + lam_tv * tv
        total = total.to(dtype)

        if self.reduction == "mean":
            return total.mean()
        return total


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-RemoteNeuralMonitoring v2] Running on: {device}")

    B, S, V, T = 8, 64, 512, 128
    model = RemoteNeuralMonitoringEndToEnd(
        num_sensors=S, num_voxels=V, hidden_dim=128, conv_groups=8,
    ).to(device)
    model.train()

    criterion = PhysicsInformedLoss(sparsity_weight=1e-3, tv_weight=1e-4, reduction="none").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sensor_signals = torch.randn(B, S, T, device=device)

    optimizer.zero_grad(set_to_none=True)
    sources, recon = model(sensor_signals)
    loss_per_sample = criterion(sensor_signals, recon, sources)
    loss_scalar = loss_per_sample.float().mean()
    loss_scalar.backward()
    optimizer.step()

    print(f"  predicted_sources shape: {tuple(sources.shape)}")
    print(f"  loss (mean): {loss_scalar.item():.6f}")

    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    print(f"  Autograd: {'PASS' if grad_ok else 'FAIL'}")
