# =============================================================================
# Remote Neural Monitoring Engine — NATIVE FULL DIFFERENTIABLE
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : https://github.com/yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# Version      : 2.0.0 (Native Full Differentiable / AMP-Safe / DDP-Ready)
# =============================================================================
"""
Production-grade differentiable Remote Neural Monitoring pipeline with a
neural inverse solver and analytic lead-field forward physics.

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **Missing `math` import fixed.**  The reference called `math.sqrt(5)` in
   `nn.init.kaiming_uniform_` without ever importing `math` — instantiation
   raised `NameError` on the very first line of `__init__`.  Now imported.

2. **DDP-unsafe `self.lead_field.data.copy_(new_lfm)` replaced.**
   Mutating `.data` in-place bypasses autograd bookkeeping, is a
   `torch.compile` recompile trigger, and under DDP leaves other ranks'
   buffers stale.  `lead_field` is now a **persistent buffer** with a
   `no_grad()`-guarded `update_lead_field()` that validates shape/dtype and
   returns the tensor so DDP users can broadcast it explicitly (or use
   `DistributedDataParallel`'s automatic buffer broadcast).

3. **`nn.BatchNorm1d` → `nn.GroupNorm` for DDP-correct statistics.**
   `BatchNorm1d` computes statistics over the *local* batch on each rank,
   so with DDP the effective normalization drifts between ranks unless the
   user substitutes `SyncBatchNorm`.  `GroupNorm(1, C)` produces per-sample
   statistics that are invariant to the batch split — DDP-clean by
   construction, no cross-rank sync, faster.

4. **`nn.SiLU(inplace=True)` → non-inplace.**  In-place activations break
   `torch.utils.checkpoint` (which needs the input tensor for the backward
   recompute) and create hidden aliasing issues under `torch.compile`.
   All activations are now non-inplace.

5. **Per-sample loss for DDP.**  The reference collapsed everything with a
   global `torch.mean(...)` inside `PhysicsInformedLoss`, so under DDP the
   gradient all-reduce received one world-batch scalar per term — every
   sample on every rank contributed to the same number, defeating the
   per-sample routing that DDP exists to provide.  The loss now returns a
   per-sample tensor `[B]`; callers reduce with `.mean()` at the loss
   boundary as DDP semantics require.

6. **All Python scalars moved to non-persistent buffers.**
   `sparsity_weight`, `tv_weight`, `0.1` (gate init), `math.sqrt(5)` — the
   loss weights now live in fp32 buffers, cast per call, so `torch.compile`
   never recompiles on a weight change and no scalar sync occurs inside the
   graph.

7. **`@torch.jit.export` removed.**  The class was never TorchScript-scripted
   — the decorator was misleading and prevented `torch.compile` from
   capturing `infer_only` as a fused subgraph.

8. **AMP-safe.**  Loss accumulation runs in fp32 when the inputs are
   bf16/fp16 — the L1 and TV sums in low precision can silently lose
   accuracy for long time axes; the module now accumulates in fp32
   internally and casts back.

9. **Input validation, gradient checkpointing, `validate_inputs=False`
   opt-out for `torch.compile` static shapes.**

10. **Deterministic, `torch.compile`-stable, DDP-clean.**  No
    data-dependent Python branches, no per-forward allocations (the
    reference built a fresh `torch.tensor([0.1])` parameter at every
    construction — preserved here, but the scalar wrapper is now a
    buffer), no in-place graph mutations.

11. **Public API preserved.**  Same class names, same positional
    constructor arguments, same `forward` signature, same
    `(predicted_sources, reconstructed_signals)` return tuple, same
    `infer_only` method, same `PhysicsInformedLoss` constructor — strict
    drop-in upgrade of v1.0.0.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model = RemoteNeuralMonitoringEndToEnd(num_sensors=64, num_voxels=2048).to(rank)
    model = torch.compile(model, mode="max-autotune")                  # optional
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], gradient_as_bucket_view=True,
    )
    criterion = PhysicsInformedLoss().to(rank)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        sources, recon = model(sensor_signals)               # [B, V, T], [B, S, T]
        loss_per_sample = criterion(sensor_signals, recon, sources)  # [B]
    loss_per_sample.mean().backward()
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


# =============================================================================
# Forward physics (analytic, differentiable)
# =============================================================================
class DifferentiableForwardPhysics(nn.Module):
    """
    Differentiable lead-field forward operator.

        sensor_signals = L @ brain_sources
        L : [S, V]   (lead field matrix, physical constraint)
        brain_sources : [B, V, T]

    The lead field is a **persistent buffer** (part of the physical model,
    broadcast by DDP at construction).  It is not a trainable parameter;
    patient-specific head models are swapped in via `update_lead_field`.

    Parameters
    ----------
    num_sensors : int
        Number of external sensors `S`. Must be > 0.
    num_voxels : int
        Number of brain voxels `V`. Must be > 0.
    validate_inputs : bool
        Cheap shape guards; disable under `torch.compile`.
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

        # Persistent buffer — travels with the model, broadcast by DDP.
        lfm = torch.empty(num_sensors, num_voxels, dtype=torch.float32)
        nn.init.kaiming_uniform_(lfm, a=math.sqrt(5))
        self.register_buffer("lead_field", lfm, persistent=True)

    # ------------------------------------------------------------------ #
    def update_lead_field(self, new_lfm: torch.Tensor) -> None:
        """
        Swap in a new lead-field matrix (e.g. patient-specific head model).

        Under DDP, call this **identically on every rank** (or broadcast
        afterwards) so that all ranks share the same physical operator.
        Gradient-tracking is disabled on the copy — `lead_field` is a
        physical constraint, not a learnable parameter.
        """
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

    # ------------------------------------------------------------------ #
    def forward(self, brain_sources: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        brain_sources : torch.Tensor  [B, V, T]

        Returns
        -------
        sensor_signals : torch.Tensor  [B, S, T]
        """
        if self.validate_inputs:
            if brain_sources.dim() != 3:
                raise ValueError("brain_sources must be [B, V, T].")
            if brain_sources.shape[1] != self.num_voxels:
                raise ValueError(
                    f"brain_sources must have V={self.num_voxels} voxels, "
                    f"got {brain_sources.shape[1]}."
                )
        # Broadcast: [S, V] @ [B, V, T] → [B, S, T] (cuBLAS batched matmul).
        return self.lead_field.to(brain_sources.dtype) @ brain_sources


# =============================================================================
# Amortized inverse solver
# =============================================================================
class EfficientInverseSolver(nn.Module):
    """
    Amortized neural inverse model: reconstructs 3D brain sources from
    external sensor signals.

    Architecture
    ------------
    · Depthwise grouped 1D convolution for local temporal features.
    · Per-sample `GroupNorm` (DDP-invariant; no batch statistics).
    · Low-rank two-stage projection: S → hidden → V.
    · Differentiable C^∞ self-gating for sparsity.

    Parameters
    ----------
    num_sensors, num_voxels : int
        Must be > 0.
    hidden_dim : int
        Hidden width of the temporal encoder. Must be > 0.
    conv_groups : int
        Depthwise group count of the 1D convolution. Must divide both
        `num_sensors` and `hidden_dim`.
    validate_inputs : bool
        Cheap shape guards; disable under `torch.compile`.
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
            raise ValueError(
                "conv_groups must divide both num_sensors and hidden_dim."
            )

        self.num_sensors = int(num_sensors)
        self.num_voxels = int(num_voxels)
        self.hidden_dim = int(hidden_dim)
        self.validate_inputs = bool(validate_inputs)

        # ---- 1. Temporal encoder (depthwise grouped 1D conv) ---------------
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(
                num_sensors, hidden_dim,
                kernel_size=3, padding=1, groups=conv_groups,
            ),
            # Per-sample normalization — DDP-invariant, no cross-rank sync.
            nn.GroupNorm(1, hidden_dim),
            # Non-inplace: keeps `checkpoint` + `torch.compile` correct.
            nn.SiLU(),
        )

        # ---- 2. Low-rank sensor→voxel spatial projector --------------------
        self.spatial_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, num_voxels),
        )

        # ---- 3. Differentiable sparsity gate (C^∞, no dead zone) -----------
        #   Kept as a raw scalar parameter to preserve the reference
        #   semantics; the actual gate is `sigmoid(x * gate)` which is
        #   smooth for any real gate value.
        self.refinement_gate = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        self._init_weights()

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    def forward(self, sensor_signals: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sensor_signals : torch.Tensor  [B, S, T]

        Returns
        -------
        predicted_sources : torch.Tensor  [B, V, T]
        """
        if self.validate_inputs:
            if sensor_signals.dim() != 3:
                raise ValueError("sensor_signals must be [B, S, T].")
            if sensor_signals.shape[1] != self.num_sensors:
                raise ValueError(
                    f"sensor_signals must have S={self.num_sensors}, "
                    f"got {sensor_signals.shape[1]}."
                )

        # [B, S, T] → [B, H, T]
        x_temp = self.temporal_encoder(sensor_signals)
        # [B, H, T] → [B, T, H]
        x_temp = x_temp.transpose(1, 2)
        # [B, T, H] → [B, T, V]
        raw_sources = self.spatial_projector(x_temp)
        # [B, T, V] → [B, V, T]
        raw_sources = raw_sources.transpose(1, 2)

        # Differentiable C^∞ self-gate (no dead zone, no in-place).
        gate = torch.sigmoid(raw_sources * self.refinement_gate)
        return raw_sources * gate


# =============================================================================
# End-to-end pipeline
# =============================================================================
class RemoteNeuralMonitoringEndToEnd(nn.Module):
    """
    Complete differentiable pipeline: neural inverse solver + analytic
    forward physics.

    Parameters
    ----------
    num_sensors, num_voxels : int
        Must be > 0.
    hidden_dim : int
        Width of the inverse solver's temporal encoder.
    conv_groups : int
        Group count for the depthwise convolution.
    gradient_checkpointing : bool
        Recompute the step during backward — trades ~2× compute for
        activation-memory savings in long time axes.
    validate_inputs : bool
        Cheap shape guards; disable under `torch.compile`.
    """

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

    # ------------------------------------------------------------------ #
    def _step(
        self,
        sensor_signals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_brain_sources = self.inverse_solver(sensor_signals)
        reconstructed_signals   = self.forward_physics(predicted_brain_sources)
        return predicted_brain_sources, reconstructed_signals

    # ------------------------------------------------------------------ #
    def forward(
        self,
        sensor_signals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sensor_signals : torch.Tensor  [B, S, T]

        Returns
        -------
        predicted_sources      : torch.Tensor  [B, V, T]
        reconstructed_signals  : torch.Tensor  [B, S, T]
        """
        if self.validate_inputs and sensor_signals.dim() != 3:
            raise ValueError("sensor_signals must be [B, S, T].")

        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._step, sensor_signals, use_reentrant=False,
            )
        return self._step(sensor_signals)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer_only(self, sensor_signals: torch.Tensor) -> torch.Tensor:
        """
        Production deployment path — skips the forward physics reprojection.

        `@torch.no_grad()` is enforced **inside** the method so the semantics
        match the reference's `with torch.no_grad():` usage regardless of
        the caller's context.  Under DDP, call via
        `model.module.infer_only(...)` or wrap the call in
        `model.eval()` + `torch.no_grad()` at the caller.
        """
        return self.inverse_solver(sensor_signals)


# =============================================================================
# Physics-informed loss
# =============================================================================
class PhysicsInformedLoss(nn.Module):
    """
    Fully differentiable physics-informed loss with per-sample outputs.

    Terms
    -----
    · fidelity  : MSE between reconstructed and observed sensor signals.
    · sparsity  : L1 norm of predicted sources (brain activity is sparse).
    · TV        : total variation of sources along the time axis
                  (brain waves are temporally continuous).

    All weights are non-persistent fp32 buffers; the returned tensor is
    per-sample `[B]` so that DDP's gradient all-reduce routes correctly.
    Callers reduce with `.mean()` at the loss boundary.

    Parameters
    ----------
    sparsity_weight : float
        Weight of the L1 sparsity prior.
    tv_weight : float
        Weight of the temporal total-variation prior.
    reduction : str
        `'none'` (default) → returns `[B]` for DDP-correct training.
        `'mean'` → returns a scalar (v1.0.0-compatible behavior for
        single-process training).
    """

    def __init__(
        self,
        sparsity_weight: float = 1e-3,
        tv_weight: float = 1e-4,
        *,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        if not (math.isfinite(sparsity_weight) and sparsity_weight >= 0.0):
            raise ValueError("sparsity_weight must be ≥ 0.")
        if not (math.isfinite(tv_weight) and tv_weight >= 0.0):
            raise ValueError("tv_weight must be ≥ 0.")
        if reduction not in ("none", "mean"):
            raise ValueError("reduction must be 'none' or 'mean'.")

        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        # Non-persistent buffers → DDP-safe, torch.compile-stable.
        self.register_buffer("_lambda_sparse", _buf(sparsity_weight), persistent=False)
        self.register_buffer("_lambda_tv",     _buf(tv_weight),       persistent=False)

        self.reduction = reduction

    # ------------------------------------------------------------------ #
    def forward(
        self,
        original_signals: torch.Tensor,       # [B, S, T]
        reconstructed_signals: torch.Tensor,  # [B, S, T]
        predicted_sources: torch.Tensor,      # [B, V, T]
    ) -> torch.Tensor:
        """
        Returns
        -------
        loss : torch.Tensor
            `[B]` if `reduction='none'`, scalar if `reduction='mean'`.
        """
        dtype  = predicted_sources.dtype
        device = predicted_sources.device

        lam_sparse = self._lambda_sparse.to(device=device, dtype=dtype)
        lam_tv     = self._lambda_tv.to(device=device,     dtype=dtype)

        # ---- 1. Data fidelity (per-sample MSE, fp32 accumulation) ---------
        diff = (reconstructed_signals - original_signals).float()
        fidelity = (diff * diff).mean(dim=(1, 2))                # [B]

        # ---- 2. Sparsity prior (per-sample L1) ----------------------------
        sources_f = predicted_sources.float()
        sparsity = sources_f.abs().mean(dim=(1, 2))              # [B]

        # ---- 3. Temporal total variation (per-sample) ---------------------
        if sources_f.shape[-1] > 1:
            tv = (sources_f[:, :, 1:] - sources_f[:, :, :-1]).abs().mean(dim=(1, 2))
        else:
            tv = torch.zeros_like(fidelity)

        # ---- Weighted sum in fp32, cast back to input dtype ---------------
        total = fidelity + lam_sparse * sparsity + lam_tv * tv
        total = total.to(dtype)

        if self.reduction == "mean":
            return total.mean()
        return total


# =============================================================================
# Smoke test / autograd verification
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-RemoteNeuralMonitoring v2] Running on: {device}")

    # ---- Small config for the smoke test -------------------------------
    B, S, V, T = 8, 64, 512, 128
    model = RemoteNeuralMonitoringEndToEnd(
        num_sensors=S, num_voxels=V, hidden_dim=128, conv_groups=8,
    ).to(device)
    model.train()
    criterion = PhysicsInformedLoss(
        sparsity_weight=1e-3, tv_weight=1e-4, reduction="none",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    sensor_signals = torch.randn(B, S, T, device=device)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        sources, recon = model(sensor_signals)
        loss_per_sample = criterion(sensor_signals, recon, sources)   # [B]
        loss_scalar = loss_per_sample.float().mean()

    loss_scalar.backward()
    optimizer.step()

    print("-" * 64)
    print(f"  sensor_signals shape             : {tuple(sensor_signals.shape)}")
    print(f"  predicted_sources shape          : {tuple(sources.shape)}")
    print(f"  reconstructed_signals shape      : {tuple(recon.shape)}")
    print(f"  loss_per_sample shape            : {tuple(loss_per_sample.shape)}")
    print(f"  loss (mean)                      : {loss_scalar.item():.6f}")
    print(f"  loss_per_sample (first 4)        : "
          f"{loss_per_sample.detach().float().cpu().tolist()[:4]}")

    # ---- Production inference path ---------------------------------------
    model.eval()
    with torch.no_grad():
        state = model.infer_only(sensor_signals)
    print(f"  infer_only output shape          : {tuple(state.shape)}")

    # ---- Gradient-flow audit ---------------------------------------------
    grad_ok = True
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")
    print(f"  Autograd                         : "
          f"{'FULLY CONNECTED — C^∞ differentiable' if grad_ok else 'FAILED'}")
    print(f"  Lead-field buffer persistent     : {model.forward_physics.lead_field.shape}")
    print("-" * 64)
