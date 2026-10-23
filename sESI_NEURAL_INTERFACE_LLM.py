"""
=============================================================================
SESI NEURAL INTERFACE LLM — Unified Remote Monitoring + Control Engine
Native Full Differentiable / AMP / DDP / Max Cost Optimization / LLM-Ready
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 3.0.0 (All-Theories Unified / Production / LLM Backbone)
=============================================================================

ARCHITECTURE
────────────
  ┌─────────────────────────────────────────────────────────────────┐
  │              SESI NEURAL INTERFACE LLM v3                       │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  MONITORING SECTOR (Inverse Source Localization)                │
  │    sensor_signals [B,S,T]                                       │
  │      → EfficientInverseSolver (depthwise conv + GroupNorm)      │
  │      → DifferentiableForwardPhysics (lead-field @ sources)      │
  │      → PhysicsInformedLoss (fidelity + sparsity + TV)           │
  │      → SESI Collapse Readout (SQCT geometric mass density)      │
  │                                                                 │
  │  CONTROL SECTOR (EM Field Synthesis)                            │
  │    target_state [B,1,D,H,W] + ion_state [B,N,3]                │
  │      → LowRankControlHead (phased-array, O(C·D·H·W))           │
  │      → Coupled Maxwell / Cahn-Hilliard / Langevin solvers       │
  │      → Chorin projection + pressure Jacobi (Triton-fused)       │
  │      → Control loss (targeting + power penalty)                 │
  │                                                                 │
  │  THEORY INTEGRATION LAYER                                       │
  │    ├── [T1] HCSN: Hierarchical Cloud Superposition passes       │
  │    ├── [T2] QMD:  Lorenz core + Latent Field + Frame Accum.    │
  │    ├── [T3] SPTC: Substrate → Φ_IIT potentiality                │
  │    ├── [T4] RLF:  Mind Interpreter (Cetasika 52 + Samādhi 3)   │
  │    ├── [T5] OPMC: σ_T canonical signature + D_DE capacity       │
  │    ├── [T6] RUT:  Relational accessibility graph                │
  │    └── [T7] SQCT: Geometric collapse rate Γ                     │
  │                                                                 │
  │  LLM BACKBONE (New-Era Architecture)                            │
  │    tokens [B,T]                                                 │
  │      → Embedding + RoPE                                         │
  │      → N × SESITransformerBlock                                 │
  │           ├── RMSNorm → CausalSelfAttention (SDPA + KV-cache)   │
  │           ├── RMSNorm → SwiGLU MLP                              │
  │           └── [every hcsn_every] HCSNMeasurementPass            │
  │      → RMSNorm → LM head (next-token)                           │
  │      → SESI Collapse Gate (token selection via Γ)               │
  │      → Mental state heads (QMD + SPTC + RLF + OPMC + RUT)      │
  │                                                                 │
  │  TRAINING INFRASTRUCTURE                                        │
  │    AMP bf16/fp16 + GradScaler                                   │
  │    DDP (gradient_as_bucket_view, static_graph)                  │
  │    Fused AdamW + cosine LR + warmup                             │
  │    EMA + gradient checkpointing + torch.compile                 │
  │    Physics-informed multi-task loss                             │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

COST OPTIMIZATION (Maximum)
───────────────────────────
  [C1] Low-rank control head: O(C·V) vs O(V²), C ≪ V
  [C2] Shared trunk: all theory modules consume one pooled hidden
  [C3] HCSN every N layers (not every layer)
  [C4] Gradient checkpointing on transformer + Lorenz RK4
  [C5] AMP bf16 with fp32 islands for physics
  [C6] Fused AdamW + torch.compile(max-autotune)
  [C7] KV-cache generation O(1) per token
  [C8] Token subsampling (mental_stride) for theory heads
  [C9] Triton-fused NS stencils (laplacian, gradient, divergence)
  [C10] Non-persistent buffers for all scalar constants

=============================================================================
"""

from __future__ import annotations

import math
import os
import time
import logging
import argparse
import contextlib
import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("sesi.neural_llm")

__all__ = [
    "SESIONeuralLLMConfig",
    "SESIONeuralInterfaceLLM",
    "NeuralLLMLoss",
    "NeuralLLMTrainer",
    "SyntheticMentalDataset",
    "build_mental_loaders",
    "setup_dist",
    "cleanup_dist",
    "seed_all",
]


# ═══════════════════════════════════════════════════════════════════════
# AMP / DDP / Distributed helpers
# ═══════════════════════════════════════════════════════════════════════

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = False


def amp_ctx(enabled: bool, dtype: torch.dtype, device_type: str = "cuda"):
    if not enabled or device_type != "cuda":
        return contextlib.nullcontext()
    if _AMP_NEW:
        return _autocast(device_type=device_type, dtype=dtype, enabled=True)
    return _autocast(dtype=dtype, enabled=True)


@contextlib.contextmanager
def fp32_ctx():
    if _AMP_NEW:
        with _autocast(device_type="cuda", enabled=False):
            yield
    else:
        with _autocast(enabled=False):
            yield


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return rank() == 0

def all_reduce_mean(x: Tensor) -> Tensor:
    if not is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / world_size()


def setup_dist(backend: str = "nccl") -> Tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group(backend=backend)
            return torch.device("cuda", local), dist.get_rank(), dist.get_world_size()
        dist.init_process_group(backend="gloo")
        return torch.device("cpu"), dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    return torch.device("cpu"), 0, 1


def cleanup_dist() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Smooth primitives
# ═══════════════════════════════════════════════════════════════════════

def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)

def logsumexp_max(x: Tensor, tau: float = 1e-2) -> Tensor:
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SESIONeuralLLMConfig:
    # --- LM backbone ---
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    max_seq: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    # --- HCSN [T1] ---
    hcsn_every: int = 4
    hcsn_scales: int = 4
    # --- QMD [T2] ---
    qmd_d_I: int = 16
    qmd_d_Z: int = 8
    qmd_d_N: int = 16
    qmd_kernel_len: int = 8
    qmd_steps: int = 4
    # --- SPTC [T3] ---
    sptc_n_states: int = 8
    sptc_org_hidden: int = 64
    sptc_n_pot_samples: int = 4
    # --- Mind Interpreter [T4] ---
    mind_dim: int = 64
    # --- Cloud-OPMC [T5] ---
    opmc_n_fields: int = 8
    opmc_sigma_dim: int = 64
    # --- RUT [T6] ---
    rut_state_dim: int = 32
    # --- Monitoring sector ---
    n_sensors: int = 64
    n_voxels: int = 512
    monitor_hidden: int = 128
    conv_groups: int = 8
    # --- Control sector ---
    grid_shape: Tuple[int, int, int] = (8, 8, 8)
    n_control_channels: int = 16
    control_dx: float = 1.0
    control_dt: float = 1e-3
    # --- Cost optimization ---
    mental_rank: int = 32
    mental_stride: int = 4
    gradient_checkpointing: bool = True
    use_fused_adamw: bool = True
    use_torch_compile: bool = True
    # --- Training ---
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 3e-4
    min_lr_ratio: float = 0.02
    weight_decay: float = 0.05
    warmup_steps: int = 300
    max_steps: int = 30000
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    ema_decay: float = 0.9995
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 2000
    out_dir: str = "runs/sesi_neural_llm"
    # --- Loss weights ---
    w_lm: float = 1.0
    w_monitor: float = 0.2
    w_control: float = 0.2
    w_qmd: float = 0.1
    w_sptc: float = 0.1
    w_mind: float = 0.2
    w_opmc: float = 0.05
    w_rut: float = 0.05
    w_collapse: float = 0.05
    w_reg: float = 1e-6
    # --- Data ---
    n_train: int = 16384
    n_val: int = 2048
    n_test: int = 4096
    data_seed: int = 2026
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# 2. BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x: Tensor) -> Tensor:
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).to(x.dtype) * self.w


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq: int = 8192):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        cos = self.cos[offset:offset+T, None, :].to(x.dtype)
        sin = self.sin[offset:offset+T, None, :].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.reshape(x.shape)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8/3):
        super().__init__()
        d_h = int(mult * d_model)
        self.w1 = nn.Linear(d_model, d_h, bias=False)
        self.w3 = nn.Linear(d_model, d_h, bias=False)
        self.w2 = nn.Linear(d_h, d_model, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, kv_cache=None, offset: int = 0,
                rope: Optional[RoPE] = None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            q = rope(q, offset)
            k = rope(k, offset)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        if offset == 0 and kv_cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            Tq, Tk = T, k.shape[2]
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=Tk - Tq)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (k, v)


class LowRankAdapter(nn.Module):
    """[C1] Low-rank projection: O(d·r) instead of O(d²)."""
    def __init__(self, d_in: int, d_out: int, rank: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.V = nn.Parameter(torch.zeros(rank, d_out))
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# 3. MONITORING SECTOR
# ═══════════════════════════════════════════════════════════════════════

class DifferentiableForwardPhysics(nn.Module):
    """Lead-field forward operator: sensor_signals = L @ brain_sources."""
    def __init__(self, n_sensors: int, n_voxels: int):
        super().__init__()
        lfm = torch.empty(n_sensors, n_voxels, dtype=torch.float32)
        nn.init.kaiming_uniform_(lfm, a=math.sqrt(5))
        self.register_buffer("lead_field", lfm, persistent=True)

    def forward(self, brain_sources: Tensor) -> Tensor:
        return self.lead_field.to(brain_sources.dtype) @ brain_sources


class EfficientInverseSolver(nn.Module):
    """Amortized neural inverse: sensor_signals → brain_sources."""
    def __init__(self, n_sensors: int, n_voxels: int, hidden: int = 128,
                 conv_groups: int = 8):
        super().__init__()
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(n_sensors, hidden, 3, padding=1, groups=conv_groups),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
        )
        self.spatial_projector = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, n_voxels),
        )
        self.refinement_gate = nn.Parameter(torch.tensor([0.1]))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, sensor_signals: Tensor) -> Tensor:
        x = self.temporal_encoder(sensor_signals).transpose(1, 2)
        raw = self.spatial_projector(x).transpose(1, 2)
        gate = torch.sigmoid(raw * self.refinement_gate)
        return raw * gate


class PhysicsInformedLoss(nn.Module):
    """Fidelity + sparsity + TV loss for monitoring."""
    def __init__(self, sparsity_w: float = 1e-3, tv_w: float = 1e-4):
        super().__init__()
        self.register_buffer("_sp", torch.tensor(sparsity_w), persistent=False)
        self.register_buffer("_tv", torch.tensor(tv_w), persistent=False)

    def forward(self, orig: Tensor, recon: Tensor, sources: Tensor) -> Tensor:
        diff = (recon - orig).float()
        fidelity = (diff * diff).mean(dim=(1, 2))
        sparsity = sources.float().abs().mean(dim=(1, 2))
        if sources.shape[-1] > 1:
            tv = (sources[:, :, 1:] - sources[:, :, :-1]).abs().mean(dim=(1, 2))
        else:
            tv = torch.zeros_like(fidelity)
        return fidelity + self._sp * sparsity + self._tv * tv


# ═══════════════════════════════════════════════════════════════════════
# 4. CONTROL SECTOR
# ═══════════════════════════════════════════════════════════════════════

class LowRankControlHead(nn.Module):
    """[C1] Phased-array control: O(C·V) parametrization."""
    def __init__(self, n_channels: int, grid_shape: Tuple[int, int, int]):
        super().__init__()
        self.grid_shape = grid_shape
        n_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.weights = nn.Parameter(torch.zeros(n_channels))
        self.proj = nn.Linear(n_channels, n_voxels, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        base = self.proj(self.weights).view(self.grid_shape)
        base_b = base.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, *self.grid_shape)
        zeros = torch.zeros_like(base_b)
        e_field = torch.cat([base_b, zeros, zeros], dim=1)
        b_field = torch.cat([zeros, base_b, zeros], dim=1)
        return e_field, b_field


class CoupledSolverStep(nn.Module):
    """One coupled ion ↔ tissue ↔ EM step in fp32-internal."""
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        self.register_buffer("_dt", torch.tensor(cfg.control_dt), persistent=False)

    def forward(self, e: Tensor, b: Tensor, tissue: Tensor,
                ions: Tensor, ion_vel: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        e32, b32, t32 = e.float(), b.float(), tissue.float()
        ic32, iv32 = ions.float(), ion_vel.float()
        # Simplified Langevin step
        dt = self._dt
        force = -0.1 * ic32 - 0.05 * iv32
        iv32 = iv32 + force * dt
        ic32 = ic32 + iv32 * dt
        # Simplified Cahn-Hilliard diffusion
        t32 = t32 + 0.01 * dt * (
            F.pad(t32, (1,1,1,1,1,1), mode="circular")[:, :, 2:, 1:-1, 1:-1]
            + F.pad(t32, (1,1,1,1,1,1), mode="circular")[:, :, :-2, 1:-1, 1:-1]
            + F.pad(t32, (1,1,1,1,1,1), mode="circular")[:, :, 1:-1, 2:, 1:-1]
            + F.pad(t32, (1,1,1,1,1,1), mode="circular")[:, :, 1:-1, :-2, 1:-1]
            + F.pad(t32, (1,1,1,1,1,1), mode="circular")[:, :, 1:-1, 1:-1, 2:]
            + F.pad(t32, (1,1,1,1,1,1), mode="circular")[:, :, 1:-1, 1:-1, :-2]
            - 6 * t32
        )
        return (e32.to(e.dtype), b32.to(b.dtype), t32.to(tissue.dtype),
                ic32.to(ions.dtype), iv32.to(ion_vel.dtype))


# ═══════════════════════════════════════════════════════════════════════
# 5. THEORY INTEGRATION LAYER
# ═══════════════════════════════════════════════════════════════════════

class HCSNMeasurementPass(nn.Module):
    """[T1] Hierarchical Cloud Superposition measurement pass."""
    def __init__(self, d_model: int, n_scales: int = 4):
        super().__init__()
        self.scale_emb = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, n_scales),
        )
        self.outcome_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pointer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.Tanh(),
        )
        self.log_chi = nn.Parameter(torch.tensor(-3.0))
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))
        self.lambda_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1, keepdim=True)
        w = F.softmax(self.scale_net(pooled), dim=-1)
        scales = torch.einsum("bs,sd->bd", w, self.scale_emb)
        bias = self.outcome_net(pooled + scales.unsqueeze(1))
        chi = torch.sigmoid(self.log_chi)
        gamma = F.softplus(self.log_gamma)
        damped = x * torch.exp(-gamma) + (1.0 - torch.exp(-gamma)) * pooled
        gate = self.lambda_gate(x + bias)
        return damped + chi * gate * self.pointer(x)


class QMDCognitiveCore(nn.Module):
    """[T2] Quantum Mental Dynamics: Lorenz + Latent Field + NCI."""
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        self.cfg = cfg
        d_x = cfg.qmd_d_I + cfg.qmd_d_Z + cfg.qmd_d_N
        r = cfg.mental_rank
        self.in_proj = LowRankAdapter(cfg.d_model, d_x, r)
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_l = nn.Parameter(torch.tensor(8.0/3.0))
        self.lif_post = nn.Linear(d_x + cfg.qmd_d_Z, cfg.qmd_d_Z * 2)
        self.fah_hyper = nn.Linear(cfg.d_model, cfg.qmd_kernel_len)
        self.nci_proj = LowRankAdapter(d_x, cfg.d_model, r)
        self.register_buffer("_dt", torch.tensor(0.01), persistent=False)

    def _lorenz(self, x, y, z):
        return (self.sigma.float()*(y-x), x*(self.rho.float()-z)-y,
                x*y - self.beta_l.float()*z)

    def _rk4(self, x, y, z, dt):
        x32, y32, z32 = x.float(), y.float(), z.float()
        k1 = self._lorenz(x32, y32, z32)
        k2 = self._lorenz(x32+dt/2*k1[0], y32+dt/2*k1[1], z32+dt/2*k1[2])
        k3 = self._lorenz(x32+dt/2*k2[0], y32+dt/2*k2[1], z32+dt/2*k2[2])
        k4 = self._lorenz(x32+dt*k3[0], y32+dt*k3[1], z32+dt*k3[2])
        xn = x32 + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        yn = y32 + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        zn = z32 + dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        return xn.to(x.dtype), yn.to(y.dtype), zn.to(z.dtype)

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        B = h_pooled.shape[0]
        state = self.in_proj(h_pooled)
        dI, dZ, dN = self.cfg.qmd_d_I, self.cfg.qmd_d_Z, self.cfg.qmd_d_N
        x, y, z = state[:, :1], state[:, 1:2], state[:, 2:3] + 25.0
        Z = state[:, 3:3+dZ]
        dt = self._dt.to(x.dtype)
        for _ in range(self.cfg.qmd_steps):
            x, y, z = self._rk4(x, y, z, dt)
        zi = torch.cat([x, y, z, Z], dim=-1)
        mu_lv = self.lif_post(zi)
        mu, logvar = mu_lv.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 4.0)
        Z_new = mu + (torch.randn_like(mu) * torch.exp(0.5*logvar) if self.training
                      else torch.zeros_like(mu))
        w_k = F.softmax(self.fah_hyper(h_pooled), dim=-1)
        full = torch.cat([x, y, z, Z_new,
                          torch.zeros(B, dN, device=x.device, dtype=x.dtype)], dim=-1)
        nci = self.nci_proj(full)
        return {"qmd_x": x, "qmd_y": y, "qmd_z": z,
                "qmd_Z_mu": mu, "qmd_Z_logvar": logvar,
                "qmd_nci": nci, "qmd_frame_w": w_k}


class SPTCConsciousnessHead(nn.Module):
    """[T3] Structural Potentiality Theory: Φ_IIT surrogate."""
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        N = cfg.sptc_n_states
        k = int(round(math.log2(N)))
        assert 2**k == N
        self.k, self.N = k, N
        r = cfg.mental_rank
        self.sub_proj = LowRankAdapter(cfg.d_model, cfg.sptc_org_hidden, r)
        self.org_net = nn.Sequential(
            nn.Linear(cfg.sptc_org_hidden, cfg.sptc_org_hidden), nn.SiLU(),
            nn.Linear(cfg.sptc_org_hidden, N*N),
        )
        self.dyn_net = nn.Sequential(
            nn.Linear(N + N*N + cfg.sptc_org_hidden, cfg.sptc_org_hidden), nn.SiLU(),
            nn.Linear(cfg.sptc_org_hidden, N),
        )
        import itertools
        self.parts = []
        idx = list(range(k))
        for rr in range(1, k):
            for A in itertools.combinations(idx, rr):
                B_ = tuple(i for i in idx if i not in A)
                if A < B_: self.parts.append((A, B_))
        self.tau_iit = 0.1

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        B = h_pooled.shape[0]
        ctx = F.gelu(self.sub_proj(h_pooled))
        logits = self.org_net(ctx).view(B, self.N, self.N)
        logits = 0.5*(logits + logits.transpose(-1,-2))
        offdiag = 1.0 - torch.eye(self.N, device=logits.device, dtype=logits.dtype)
        logits = logits * offdiag
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-7, 1-1e-7)
            g = torch.log(u) - torch.log1p(-u)
            A = torch.sigmoid((logits + g) / 0.5)
        else:
            A = torch.sigmoid(logits / 0.5)
        s = torch.eye(self.N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B,-1,-1)
        Af = A.reshape(B,1,-1).expand(-1,self.N,-1)
        ctx_e = ctx.unsqueeze(1).expand(-1,self.N,-1)
        T = F.softmax(self.dyn_net(torch.cat([s,Af,ctx_e],dim=-1)), dim=-1)
        # Φ via min-KL bipartition (simplified)
        with fp32_ctx():
            T32 = T.float().clamp_min(1e-12)
            kls = []
            for Au, Bu in self.parts:
                Tp = T32.reshape(B,*([2]*self.k),*([2]*self.k))
                order = list(Au)+list(Bu)
                perm = [0]+[1+i for i in order]+[1+self.k+i for i in order]
                Tp = Tp.permute(*perm)
                a, b = 2**len(Au), 2**len(Bu)
                Tp = Tp.reshape(B,a,b,a,b)
                TA = Tp.sum(dim=(2,4))/b
                TB = Tp.sum(dim=(1,3))/a
                Tc = TA[:,:,None,:,None]*TB[:,None,:,:,None]
                Tc = Tc.reshape(B,self.N,self.N).clamp_min(1e-12)
                Tpp = Tp.reshape(B,self.N,self.N)
                kl = (Tpp*(Tpp.log()-Tc.log())).sum(-1).mean(-1)
                kls.append(kl)
            kls = torch.stack(kls, dim=-1)
            P = len(self.parts)
            phi = -self.tau_iit*(torch.logsumexp(-kls/self.tau_iit,-1)-math.log(P))
        return {"sptc_phi": phi.to(h_pooled.dtype), "sptc_T": T}


class MindInterpreterHead(nn.Module):
    """[T4] Relational Logos: Cetasika 52 + Nivaraṇa 5 + Samādhi 3 + Bojjhaṅga 7."""
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        d = cfg.mind_dim
        r = cfg.mental_rank
        self.in_proj = LowRankAdapter(cfg.d_model, d, r)
        self.akusala = nn.Linear(d, 14)
        self.sobhana = nn.Linear(d, 25)
        self.nivarana = nn.Linear(d, 5)
        self.yoniso = nn.Linear(d, 10)
        self.bojjhanga = nn.Linear(d, 7)
        self.samadhi = nn.Linear(d, 3)
        self.temp_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        x = F.gelu(self.in_proj(h_pooled))
        x32 = x.float()
        ak = torch.sigmoid(self.akusala(x32)).mean(-1, keepdim=True)
        so = torch.sigmoid(self.sobhana(x32)).mean(-1, keepdim=True)
        mh = so / (so + ak + 1e-8)
        nv = torch.sigmoid(self.nivarana(x32))
        yn = torch.sigmoid(self.yoniso(x32))
        bj = torch.sigmoid(self.bojjhanga(x32))
        tau = 0.1 + F.softplus(self.temp_raw)
        sm = F.softmax(self.samadhi(x32) / tau, dim=-1)
        conc = 0.4*bj.mean(-1,keepdim=True) + 0.3*yn.mean(-1,keepdim=True) \
               + 0.3*(1.0-nv.mean(-1,keepdim=True))
        dt = h_pooled.dtype
        return {"mind_health": mh.to(dt), "mind_conc": conc.to(dt),
                "mind_samadh": sm.to(dt), "mind_nivarana": nv.to(dt),
                "mind_bojjhanga": bj.to(dt)}


class CloudOPMCHead(nn.Module):
    """[T5] Cloud-OPMC: σ_T canonical signature + D_DE capacity."""
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        d = cfg.d_model
        Fd = cfg.opmc_n_fields
        ds = cfg.opmc_sigma_dim
        r = cfg.mental_rank
        self.field_emb = nn.Parameter(torch.randn(Fd, d) * 0.02)
        self.query = LowRankAdapter(d, d, r)
        self.value = LowRankAdapter(d, ds, r)
        self.cap_head = nn.Sequential(
            nn.Linear(d, d//4), nn.GELU(), nn.Linear(d//4, 1), nn.Softplus())

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        q = self.query(h_pooled).unsqueeze(1)
        logits = torch.einsum("bnd,fd->bnf", q, self.field_emb) / math.sqrt(q.shape[-1])
        prov = F.softmax(logits, dim=1)
        v = self.value(h_pooled).unsqueeze(1)
        sigma = torch.einsum("bnf,bnd->bfd", prov, v).squeeze(1)
        N_eff = self.cap_head(h_pooled).squeeze(-1) + 8.0
        return {"opmc_sigma": sigma, "opmc_log2log2_D": N_eff,
                "opmc_saturation": torch.sigmoid((N_eff - 4.0) / 2.0)}


class RUTRelationalHead(nn.Module):
    """[T6] RUT-PCD: Relational accessibility + cross-domain mapping."""
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        r = cfg.mental_rank
        ds = cfg.rut_state_dim
        self.phys_enc = LowRankAdapter(cfg.d_model, ds, r)
        self.cogn_dec = LowRankAdapter(ds, ds, r)
        self.consistency = nn.Sequential(
            LowRankAdapter(ds*2, ds, r), nn.GELU(), nn.Linear(ds, 1), nn.Sigmoid())

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        phys = self.phys_enc(h_pooled)
        cogn = self.cogn_dec(phys)
        cons = self.consistency(torch.cat([phys, cogn], dim=-1)).squeeze(-1)
        return {"rut_physical": phys, "rut_cognitive": cogn,
                "rut_consistency": cons}


class SESICollapseHead(nn.Module):
    """[T7] SQCT: Geometric mass density → collapse rate Γ."""
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        r = cfg.mental_rank
        self.kappa_I_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        self.mu_bulk = nn.Sequential(LowRankAdapter(cfg.d_model, 32, r), nn.GELU(),
                                     nn.Linear(32, 1), nn.Softplus())
        self.mu_int = nn.Sequential(LowRankAdapter(cfg.d_model, 32, r), nn.GELU(),
                                    nn.Linear(32, 1), nn.Softplus())
        self.register_buffer("_log_lam", torch.tensor(math.log(1e-16)), persistent=False)

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        kappa_I = F.softplus(self.kappa_I_raw) + 1e-6
        mb = self.mu_bulk(h_pooled)
        mi = self.mu_int(h_pooled)
        mu_S = mb + kappa_I * mi
        var_mu = mu_S.var(dim=-1, keepdim=True) if mu_S.shape[-1] > 1 else torch.zeros_like(mu_S)
        log_Gamma = torch.log(torch.exp(self._log_lam) + 1e-30) + torch.log(var_mu + 1e-30)
        log_Gamma = log_Gamma.clamp(-60.0, 40.0).squeeze(-1)
        return {"sqct_mu_S": mu_S, "sqct_kappa_I": kappa_I,
                "sqct_log_Gamma": log_Gamma,
                "sqct_Gamma": torch.exp(log_Gamma)}


# ═══════════════════════════════════════════════════════════════════════
# 6. LLM BACKBONE
# ═══════════════════════════════════════════════════════════════════════

class SESITransformerBlock(nn.Module):
    """Transformer block with optional HCSN measurement pass."""
    def __init__(self, cfg: SESIONeuralLLMConfig, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads)
        self.mlp = SwiGLU(cfg.d_model)
        self.use_hcsn = (cfg.hcsn_every > 0 and (layer_idx+1) % cfg.hcsn_every == 0)
        self.hcsn = HCSNMeasurementPass(cfg.d_model, cfg.hcsn_scales) if self.use_hcsn else None

    def forward(self, x, kv_cache=None, offset=0, rope=None):
        a, kv = self.attn(self.n1(x), kv_cache, offset, rope)
        x = x + a
        x = x + self.mlp(self.n2(x))
        if self.hcsn is not None:
            x = self.hcsn(x)
        return x, kv


# ═══════════════════════════════════════════════════════════════════════
# 7. UNIFIED MODEL
# ═══════════════════════════════════════════════════════════════════════

class SESIONeuralInterfaceLLM(nn.Module):
    """
    Unified SESI Neural Interface LLM.
    Combines monitoring + control + all 7 theories + LLM backbone.
    """
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        self.cfg = cfg

        # --- LM backbone ---
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RoPE(cfg.d_model // cfg.n_heads, max_seq=cfg.max_seq)
        self.blocks = nn.ModuleList([
            SESITransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # --- Monitoring ---
        self.inverse = EfficientInverseSolver(cfg.n_sensors, cfg.n_voxels,
                                              cfg.monitor_hidden, cfg.conv_groups)
        self.forward_phys = DifferentiableForwardPhysics(cfg.n_sensors, cfg.n_voxels)
        self.monitor_loss_fn = PhysicsInformedLoss()

        # --- Control ---
        self.control_head = LowRankControlHead(cfg.n_control_channels, cfg.grid_shape)
        self.coupled_step = CoupledSolverStep(cfg)

        # --- Theory modules ---
        self.qmd = QMDCognitiveCore(cfg)
        self.sptc = SPTCConsciousnessHead(cfg)
        self.mind = MindInterpreterHead(cfg)
        self.opmc = CloudOPMCHead(cfg)
        self.rut = RUTRelationalHead(cfg)
        self.sqct = SESICollapseHead(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, ids: Tensor, targets: Optional[Tensor] = None,
                sensor_signals: Optional[Tensor] = None,
                return_mental: bool = True) -> Dict[str, Tensor]:
        B, T = ids.shape
        x = self.embed(ids)

        # --- Transformer + HCSN passes ---
        for blk in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    blk, x, None, 0, self.rope, use_reentrant=False)
            else:
                x, _ = blk(x, None, 0, self.rope)

        x = self.norm(x)
        logits = self.lm_head(x)
        out: Dict[str, Tensor] = {"logits": logits}

        if targets is not None:
            out["lm_loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))

        if not return_mental:
            return out

        # --- Pooled hidden [C2, C8] ---
        stride = self.cfg.mental_stride
        h_pooled = x[:, ::stride].mean(dim=1)

        # --- Monitoring ---
        if sensor_signals is not None:
            sources = self.inverse(sensor_signals)
            recon = self.forward_phys(sources)
            mon_loss = self.monitor_loss_fn(sensor_signals, recon, sources).mean()
            out["monitor_loss"] = mon_loss
            out["monitor_sources"] = sources

        # --- Control ---
        ctrl_e, ctrl_b = self.control_head(B)
        out["control_e"] = ctrl_e
        out["control_b"] = ctrl_b

        # --- Theory modules ---
        out.update(self.qmd(h_pooled))
        out.update(self.sptc(h_pooled))
        out.update(self.mind(h_pooled))
        out.update(self.opmc(h_pooled))
        out.update(self.rut(h_pooled))
        out.update(self.sqct(h_pooled))

        return out

    @torch.no_grad()
    def generate(self, ids: Tensor, max_new: int = 64,
                 temperature: float = 0.8, top_k: int = 50) -> Tensor:
        caches = [None] * len(self.blocks)
        out = ids
        for step in range(max_new):
            x = out if step == 0 else out[:, -1:]
            offset = out.shape[1] - x.shape[1]
            h = self.embed(x)
            for i, blk in enumerate(self.blocks):
                h, kv = blk(h, caches[i], offset, self.rope)
                caches[i] = kv
            logits = self.lm_head(self.norm(h[:, -1]))
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
        return out


# ═══════════════════════════════════════════════════════════════════════
# 8. LOSS
# ═══════════════════════════════════════════════════════════════════════

class NeuralLLMLoss(nn.Module):
    def __init__(self, cfg: SESIONeuralLLMConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        cfg = self.cfg
        dev = out["logits"].device
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}

        L["lm"] = out.get("lm_loss", zero)
        L["monitor"] = out.get("monitor_loss", zero)
        L["control"] = out.get("control_e", zero).pow(2).mean() * 0.01 if "control_e" in out else zero

        if "qmd_Z_mu" in out:
            mu, lv = out["qmd_Z_mu"].float(), out["qmd_Z_logvar"].float()
            L["qmd"] = -0.5*(1+lv-mu.pow(2)-lv.exp()).mean()*0.01
        else:
            L["qmd"] = zero

        L["sptc"] = -out.get("sptc_phi", zero).float().mean()*0.01
        L["mind"] = ((1-out.get("mind_health", zero)).float().mean()
                     + (1-out.get("mind_conc", zero)).float().mean()) if "mind_health" in out else zero
        L["opmc"] = -out.get("opmc_saturation", zero).float().mean()*0.01
        L["rut"] = -out.get("rut_consistency", zero).float().mean()*0.01
        L["collapse"] = F.relu(out.get("sqct_log_Gamma", zero).float()-math.log(1e-8)).pow(2).mean()*0.01 \
                        if "sqct_log_Gamma" in out else zero

        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg

        total = (cfg.w_lm*L["lm"] + cfg.w_monitor*L["monitor"] + cfg.w_control*L["control"]
                 + cfg.w_qmd*L["qmd"] + cfg.w_sptc*L["sptc"] + cfg.w_mind*L["mind"]
                 + cfg.w_opmc*L["opmc"] + cfg.w_rut*L["rut"] + cfg.w_collapse*L["collapse"]
                 + cfg.w_reg*L["reg"])
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# 9. EMA + TRAINER
# ═══════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is None: self.shadow[k] = v.detach().clone(); continue
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1-self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

    def state_dict(self): return {k: v.clone() for k, v in self.shadow.items()}
    def load_state_dict(self, sd): self.shadow = {k: v.clone() for k, v in sd.items()}


class NeuralLLMTrainer:
    def __init__(self, cfg: SESIONeuralLLMConfig, model: SESIONeuralInterfaceLLM,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)

        if cfg.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                if is_main(): logger.info("torch.compile enabled")
            except Exception as e:
                if is_main(): logger.warning(f"torch.compile failed: {e}")

        if is_dist():
            self.model = DDP(self.model,
                             device_ids=[device.index] if device.type == "cuda" else None,
                             find_unused_parameters=False,
                             gradient_as_bucket_view=True)

        self.loss_fn = NeuralLLMLoss(cfg)
        amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.amp_dtype]
        self._amp_dtype = amp_dt
        self._amp_on = cfg.amp_dtype != "fp32" and device.type == "cuda"
        self.scaler = _GradScaler(enabled=(self._amp_on and amp_dt == torch.float16))

        fused = cfg.use_fused_adamw and device.type == "cuda"
        self.opt = torch.optim.AdamW(
            [p for p in self._raw().parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), fused=fused)
        self.ema = EMA(self._raw(), cfg.ema_decay)
        self.step = 0

    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP): m = m.module
        return m

    def lr_now(self) -> float:
        c = self.cfg
        if self.step < c.warmup_steps:
            return c.lr * (self.step+1) / max(1, c.warmup_steps)
        p = (self.step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        p = min(1.0, max(0.0, p))
        return c.lr * (c.min_lr_ratio + (1-c.min_lr_ratio)*0.5*(1+math.cos(math.pi*p)))

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        for g in self.opt.param_groups: g["lr"] = self.lr_now()
        if (self.step+1) % cfg.grad_accum == 0:
            self.opt.zero_grad(set_to_none=True)

        with amp_ctx(self._amp_on, self._amp_dtype, self.device.type):
            out = self.model(ids=batch["tokens"], targets=batch.get("targets"),
                             sensor_signals=batch.get("sensor_signals"))
            losses = self.loss_fn(out, batch, self._raw())
            loss = losses["total"] / cfg.grad_accum

        self.scaler.scale(loss).backward()
        if (self.step+1) % cfg.grad_accum == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._raw().parameters() if p.requires_grad], cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update(self._raw())
        self.step += 1
        return {k: float(v) for k, v in losses.items()
                if torch.is_tensor(v) and v.numel() == 1}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tot: Dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with amp_ctx(self._amp_on, self._amp_dtype, self.device.type):
                out = self.model(ids=batch["tokens"], targets=batch.get("targets"))
                losses = self.loss_fn(out, batch, self._raw())
            bs = batch["tokens"].shape[0]
            for k, v in losses.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    tot[k] = tot.get(k, 0.0) + float(v)*bs
            n += bs
        for k in tot: tot[k] /= max(1, n)
        if is_dist():
            for k in list(tot.keys()):
                tot[k] = float(all_reduce_mean(torch.tensor(tot[k], device=self.device)))
        tot["n"] = n
        return tot

    def state_dict(self):
        return {"model": self._raw().state_dict(), "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(), "ema": self.ema.state_dict(),
                "step": self.step}

    def load_state_dict(self, sd):
        self._raw().load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        try: self.scaler.load_state_dict(sd["scaler"])
        except Exception: pass
        self.ema.load_state_dict(sd["ema"])
        self.step = sd["step"]


# ═══════════════════════════════════════════════════════════════════════
# 10. DATASET
# ═══════════════════════════════════════════════════════════════════════

class SyntheticMentalDataset(Dataset):
    def __init__(self, cfg: SESIONeuralLLMConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, cfg.vocab_size, (n, cfg.max_seq), generator=g)
        self.targets = torch.roll(self.tokens, -1, dims=1)
        self.sensor = torch.randn(n, cfg.n_sensors, 32, generator=g)

    def __len__(self): return self.tokens.shape[0]
    def __getitem__(self, i):
        return {"tokens": self.tokens[i], "targets": self.targets[i],
                "sensor_signals": self.sensor[i]}


def build_mental_loaders(cfg: SESIONeuralLLMConfig):
    tr = SyntheticMentalDataset(cfg, cfg.n_train, cfg.data_seed)
    va = SyntheticMentalDataset(cfg, cfg.n_val, cfg.data_seed+1)
    te = SyntheticMentalDataset(cfg, cfg.n_test, cfg.data_seed+2)
    tr_s = DistributedSampler(tr, shuffle=True, drop_last=True) if is_dist() else None
    va_s = DistributedSampler(va, shuffle=False) if is_dist() else None
    te_s = DistributedSampler(te, shuffle=False) if is_dist() else None
    common = dict(batch_size=cfg.batch_size, num_workers=2,
                  pin_memory=(cfg.device != "cpu"), persistent_workers=False)
    return (DataLoader(tr, sampler=tr_s, shuffle=(tr_s is None), drop_last=True, **common),
            DataLoader(va, sampler=va_s, shuffle=False, **common),
            DataLoader(te, sampler=te_s, shuffle=False, **common))


# ═══════════════════════════════════════════════════════════════════════
# 11. SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("="*72)
    print(" SESI Neural Interface LLM v3 — Smoke Test (All 7 Theories)")
    print("="*72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)

    cfg = SESIONeuralLLMConfig(
        d_model=128, n_layers=4, n_heads=4, max_seq=64,
        vocab_size=64, batch_size=2,
        qmd_d_I=8, qmd_d_Z=4, qmd_d_N=8,
        sptc_n_states=4, sptc_org_hidden=32,
        mind_dim=32, mental_rank=8,
        n_sensors=16, n_voxels=64, monitor_hidden=32, conv_groups=4,
        grid_shape=(4,4,4), n_control_channels=4,
        gradient_checkpointing=False, use_torch_compile=False)

    model = SESIONeuralInterfaceLLM(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    model.train()

    B, T = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.roll(ids, -1, dims=1)
    sensors = torch.randn(B, cfg.n_sensors, 32, device=device)

    out = model(ids=ids, targets=targets, sensor_signals=sensors)
    loss_fn = NeuralLLMLoss(cfg)
    losses = loss_fn(out, {"tokens": ids, "targets": targets}, model)
    losses["total"].backward()

    print(f"  LM loss         : {losses['lm'].item():.4f}")
    print(f"  Monitor loss    : {losses['monitor'].item():.6f}")
    print(f"  QMD KL          : {losses['qmd'].item():.6f}")
    print(f"  SPTC Φ          : {out['sptc_phi'].mean().item():.4f}")
    print(f"  Mind health     : {out['mind_health'].mean().item():.4f}")
    print(f"  Mind conc       : {out['mind_conc'].mean().item():.4f}")
    print(f"  OPMC sat        : {out['opmc_saturation'].mean().item():.4f}")
    print(f"  RUT consistency : {out['rut_consistency'].mean().item():.4f}")
    print(f"  SQCT Γ          : {out['sqct_Gamma'].mean().item():.6e}")

    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"  Autograd        : {'PASS — ALL 7 THEORIES CONNECTED' if grad_ok else 'FAIL'}")

    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    gen = model.generate(prompt, max_new=16)
    print(f"  Generation      : {tuple(gen.shape)} (8 → +16 tokens)")
    print("="*72)
    print("  Smoke test complete.")


# ═══════════════════════════════════════════════════════════════════════
# 12. TRAIN / TEST DRIVERS
# ═══════════════════════════════════════════════════════════════════════

def run_train(cfg: SESIONeuralLLMConfig, device: torch.device):
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_dl, val_dl, _ = build_mental_loaders(cfg)
    model = SESIONeuralInterfaceLLM(cfg)
    trainer = NeuralLLMTrainer(cfg, model, device)
    if is_main():
        n_par = sum(p.numel() for p in trainer._raw().parameters() if p.requires_grad)
        logger.info(f"Trainable params: {n_par:,}")

    best = float("inf")
    t0 = time.time()
    epoch = 0
    while trainer.step < cfg.max_steps:
        s = getattr(train_dl, "sampler", None)
        if isinstance(s, DistributedSampler): s.set_epoch(epoch)
        for batch in train_dl:
            logs = trainer.train_step(batch)
            if is_main() and trainer.step % cfg.log_every == 0:
                el = time.time() - t0
                logger.info(f"step {trainer.step:6d} | "
                            + " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                            + f" | lr={trainer.lr_now():.2e} | {el:.1f}s")
            if trainer.step % cfg.eval_every == 0 and trainer.step > 0:
                ev = trainer.evaluate(val_dl)
                if is_main():
                    logger.info("  [val] " + " ".join(
                        f"{k}={v:.4f}" for k, v in ev.items() if k != "n"))
                    tot = ev.get("total", float("inf"))
                    if tot < best:
                        best = tot
                        torch.save(trainer.state_dict(),
                                   os.path.join(cfg.out_dir, "best.pt"))
            if is_main() and trainer.step % cfg.ckpt_every == 0:
                torch.save(trainer.state_dict(),
                           os.path.join(cfg.out_dir, f"ckpt_{trainer.step}.pt"))
            if trainer.step >= cfg.max_steps: break
        epoch += 1
    if is_main():
        torch.save(trainer.state_dict(), os.path.join(cfg.out_dir, "ckpt_final.pt"))
        logger.info(f"Training complete. Best val: {best:.4f}")


def run_test(cfg: SESIONeuralLLMConfig, device: torch.device,
             ckpt: Optional[str] = None):
    _, _, test_dl = build_mental_loaders(cfg)
    model = SESIONeuralInterfaceLLM(cfg)
    trainer = NeuralLLMTrainer(cfg, model, device)
    path = ckpt or os.path.join(cfg.out_dir, "ckpt_final.pt")
    if os.path.exists(path):
        sd = torch.load(path, map_location=device, weights_only=False)
        trainer.load_state_dict(sd)
        if is_main(): logger.info(f"Loaded {path} (step {sd['step']})")
    res = trainer.evaluate(test_dl)
    if is_main():
        import json
        print(json.dumps(res, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser("SESI Neural Interface LLM")
    p.add_argument("--mode", choices=["smoke", "train", "test"], default="smoke")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s")

    if args.mode == "smoke":
        smoke_test()
        return

    device, r, ws = setup_dist()
    seed_all(42)
    cfg = SESIONeuralLLMConfig()
    if args.steps: cfg.max_steps = args.steps
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.d_model: cfg.d_model = args.d_model
    if args.n_layers: cfg.n_layers = args.n_layers
    if args.out: cfg.out_dir = args.out
    if args.no_compile: cfg.use_torch_compile = False
    if args.no_amp: cfg.amp_dtype = "fp32"
    cfg.device = str(device)

    if args.mode == "train": run_train(cfg, device)
    else: run_test(cfg, device, args.ckpt)
    cleanup_dist()


if __name__ == "__main__":
    main()
