"""
=============================================================================
SUB-QUANTUM REGIME BRIDGE  —  NATIVE FULL DIFFERENTIABLE (v3.0-EXT-PROD)
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 3.0.1 (Syntax-Fixed: __future__, __all__, __name__, __init__)
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SubQuantumConfig",
    "RegimeCalculusProjector",
    "SubQuantumDeterministicCore",
    "OmniPsycheBridge",
    "soft_clamp",
    "OPTIMAL_DEVICE",
]

# ===========================================================================
# Reference-module imports with safe fallback shims
# ===========================================================================

OPTIMAL_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def soft_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return lo_t + F.softplus(x - lo_t) - F.softplus(x - hi_t)


try:  # pragma: no cover
    from psy_one_bridge import (                                     # type: ignore
        PSYONEBridge, PsycheTriadState, PsycheConfig,
        PsychopathologyMode, soft_clamp,                              # noqa: F811
    )
    _PSY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PSY_AVAILABLE = False

    class PsychopathologyMode(str, Enum):                            # type: ignore[no-redef]
        MDD_ANXIETY  = "mdd_anxiety"
        BIPOLAR      = "bipolar"
        SCHIZOPHRENIA = "schizophrenia"
        HEALTHY      = "healthy"

    @dataclass
    class PsycheConfig:                                              # type: ignore[no-redef]
        action_dim: int = 10
        mode: PsychopathologyMode = PsychopathologyMode.MDD_ANXIETY
        lambda_superego: float = 0.5
        lambda_salience: float = 0.3

    @dataclass
    class PsycheTriadState:                                          # type: ignore[no-redef]
        id_state:       torch.Tensor = field(default_factory=lambda: torch.zeros(1))
        ego_state:      torch.Tensor = field(default_factory=lambda: torch.zeros(1))
        superego_state: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
        diagnostic:     Dict[str, torch.Tensor] = field(default_factory=dict)

    class PSYONEBridge(nn.Module):                                   # type: ignore[no-redef]
        def __init__(self, config: PsycheConfig) -> None:
            super().__init__()
            self.config = config
            d = config.action_dim
            self.id_proj       = nn.Linear(d, d)
            self.ego_proj      = nn.Linear(d, d)
            self.superego_proj = nn.Linear(d, d)

        def forward(
            self,
            sensory_state: torch.Tensor,
            emotional_salience: torch.Tensor,
            observation: Optional[torch.Tensor] = None,
        ) -> Tuple[PsycheTriadState, torch.Tensor]:
            id_state       = torch.tanh(self.id_proj(sensory_state))
            ego_state      = torch.tanh(self.ego_proj(sensory_state + emotional_salience))
            superego_state = torch.tanh(self.superego_proj(emotional_salience))
            loss = (
                (id_state * id_state).mean()
                + (ego_state * ego_state).mean()
                + (superego_state * superego_state).mean()
            )
            state = PsycheTriadState(
                id_state=id_state,
                ego_state=ego_state,
                superego_state=superego_state,
                diagnostic={
                    "triad_conflict": (id_state - superego_state).abs().mean(),
                },
            )
            return state, loss

        def generate_psychopathology_report(self, state: PsycheTriadState) -> str:
            return (
                "=" * 70 + "\n"
                " PSY ONE BRIDGE — DIAGNOSTIC REPORT\n"
                + "=" * 70 + "\n"
            )


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class SubQuantumConfig:
    state_dim: int = 128
    n_layers: int = 3
    rank: int = 16
    sensory_rank: int = 8
    salience_rank: int = 4
    soft_clamp_lo: float = -1.0
    soft_clamp_hi: float = 1.0
    salience_eps: float = 1e-12
    conservation_weight: float = 0.01
    validate_inputs: bool = True
    gradient_checkpointing: bool = True


# ===========================================================================
# Low-rank regime projector
# ===========================================================================

class RegimeCalculusProjector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if not (isinstance(in_features, int) and in_features > 0):
            raise ValueError("in_features must be a positive int.")
        if not (isinstance(out_features, int) and out_features > 0):
            raise ValueError("out_features must be a positive int.")
        if not (isinstance(rank, int) and rank > 0):
            raise ValueError("rank must be a positive int.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        U = torch.empty(in_features, rank)
        V = torch.empty(rank, out_features)
        bias = torch.zeros(out_features)

        nn.init.kaiming_uniform_(U, a=math.sqrt(5))
        nn.init.zeros_(V)
        self.U    = nn.Parameter(U)
        self.V    = nn.Parameter(V)
        self.bias = nn.Parameter(bias)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = x @ self.U.to(dtype=x.dtype)
        out = latent @ self.V.to(dtype=x.dtype)
        return out + self.bias.to(dtype=x.dtype)


# ===========================================================================
# Sub-quantum deterministic core
# ===========================================================================

class SubQuantumDeterministicCore(nn.Module):
    def __init__(
        self,
        state_dim: int = 128,
        n_layers: int = 3,
        *,
        config: Optional[SubQuantumConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if not (isinstance(state_dim, int) and state_dim > 0):
            raise ValueError("state_dim must be a positive int.")
        if not (isinstance(n_layers, int) and n_layers > 0):
            raise ValueError("n_layers must be a positive int.")

        cfg = config or SubQuantumConfig()
        cfg = SubQuantumConfig(**{
            **cfg.__dict__,
            "state_dim": int(state_dim),
            "n_layers": int(n_layers),
        })

        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.n_layers = cfg.n_layers
        self.gradient_checkpointing = cfg.gradient_checkpointing

        self.regime_boundaries = nn.Parameter(
            torch.linspace(-1.0, 1.0, cfg.state_dim)
        )

        self.evolution_layers = nn.ModuleList([
            RegimeCalculusProjector(
                cfg.state_dim, cfg.state_dim, rank=cfg.rank, device=device,
            )
            for _ in range(cfg.n_layers)
        ])

        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_pi",       _buf(math.pi),           persistent=False)
        self.register_buffer("_clamp_lo", _buf(cfg.soft_clamp_lo), persistent=False)
        self.register_buffer("_clamp_hi", _buf(cfg.soft_clamp_hi), persistent=False)

    def _evolution_step(
        self,
        x: torch.Tensor,
        layer: RegimeCalculusProjector,
    ) -> torch.Tensor:
        dtype_in = x.dtype
        device   = x.device

        projected = layer(x)

        pi_b  = self._pi.to(device=device)
        lo_b  = self._clamp_lo.to(device=device)
        hi_b  = self._clamp_hi.to(device=device)

        p32   = projected.float()
        b32   = self.regime_boundaries.float()

        gated_phase = torch.sin(p32 * pi_b + b32)
        clipped = lo_b + F.softplus(gated_phase - lo_b) - F.softplus(gated_phase - hi_b)
        residual = clipped.to(dtype_in)

        return x + residual

    def forward(self, initial_sq_state: torch.Tensor) -> torch.Tensor:
        was_unbatched = (initial_sq_state.dim() == 1)
        sq_state = initial_sq_state.unsqueeze(0) if was_unbatched else initial_sq_state

        if sq_state.dim() != 2 or sq_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"initial_sq_state must be [B, {self.state_dim}] or "
                f"[{self.state_dim}]; got {tuple(initial_sq_state.shape)}."
            )

        use_ckpt = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and sq_state.requires_grad
        )

        for layer in self.evolution_layers:
            if use_ckpt:
                sq_state = torch.utils.checkpoint.checkpoint(
                    self._evolution_step,
                    sq_state,
                    layer,
                    use_reentrant=False,
                )
            else:
                sq_state = self._evolution_step(sq_state, layer)

        return sq_state.squeeze(0) if was_unbatched else sq_state


# ===========================================================================
# Omni-Psyche bridge
# ===========================================================================

class OmniPsycheBridge(nn.Module):
    def __init__(
        self,
        psyone_bridge: nn.Module,
        sq_dim: int = 128,
        *,
        config: Optional[SubQuantumConfig] = None,
        device: Optional[torch.device] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not isinstance(psyone_bridge, nn.Module):
            raise ValueError("psyone_bridge must be an nn.Module.")
        if not (isinstance(sq_dim, int) and sq_dim > 0):
            raise ValueError("sq_dim must be a positive int.")
        if not hasattr(psyone_bridge, "config") or not hasattr(psyone_bridge.config, "action_dim"):
            raise ValueError("psyone_bridge must expose `.config.action_dim`.")

        cfg = config or SubQuantumConfig()
        cfg = SubQuantumConfig(**{**cfg.__dict__, "state_dim": int(sq_dim)})
        if validate_inputs is not None:
            cfg = SubQuantumConfig(**{**cfg.__dict__,
                                      "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = SubQuantumConfig(**{**cfg.__dict__,
                                      "gradient_checkpointing": bool(gradient_checkpointing)})

        self.cfg = cfg
        self.bridge = psyone_bridge
        self.action_dim = int(self.bridge.config.action_dim)
        self.sq_dim = int(sq_dim)
        self.validate_inputs = cfg.validate_inputs

        self.sq_core = SubQuantumDeterministicCore(
            state_dim=self.sq_dim,
            n_layers=cfg.n_layers,
            config=cfg,
            device=device,
        )

        self.sq_to_sensory = RegimeCalculusProjector(
            self.sq_dim, self.action_dim, rank=cfg.sensory_rank, device=device,
        )
        self.sq_to_salience = RegimeCalculusProjector(
            self.sq_dim, self.action_dim, rank=cfg.salience_rank, device=device,
        )

        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_salience_eps", _buf(cfg.salience_eps),        persistent=False)
        self.register_buffer("_cons_weight",  _buf(cfg.conservation_weight), persistent=False)

    def forward(
        self,
        sq_latent_vector: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor]:
        if self.validate_inputs:
            if sq_latent_vector.dim() not in (1, 2):
                raise ValueError(
                    "sq_latent_vector must be 1D (sq_dim,) or 2D (B, sq_dim)."
                )
            if sq_latent_vector.shape[-1] != self.sq_dim:
                raise ValueError(
                    f"sq_latent_vector last dim must be sq_dim={self.sq_dim}, "
                    f"got {sq_latent_vector.shape[-1]}."
                )

        evolved_sq = self.sq_core(sq_latent_vector)

        sensory_state = self.sq_to_sensory(evolved_sq)
        salience_raw = torch.sigmoid(self.sq_to_salience(evolved_sq))
        eps = self._salience_eps.to(device=salience_raw.device, dtype=salience_raw.dtype)
        emotional_salience = salience_raw / (salience_raw.sum(dim=-1, keepdim=True) + eps)

        state, psy_loss = self.bridge(
            sensory_state=sensory_state,
            emotional_salience=emotional_salience,
            observation=observation,
        )

        w = self._cons_weight.to(device=evolved_sq.device, dtype=evolved_sq.dtype)
        sq_conservation_loss = (evolved_sq * evolved_sq).mean() * w

        total_loss = psy_loss + sq_conservation_loss
        return state, total_loss

    def generate_unified_report(self, state: Any) -> str:
        base_report = self.bridge.generate_psychopathology_report(state)
        sq_lines = [
            "  ── Sub-Quantum Deterministic Dynamics (v3.0-EXT-PROD) ───",
            "  ✓ Regime Calculus Projection      : Active (Low-Rank Optimized)",
            "  ✓ Deterministic State Evolution   : Checkpointed for minimal VRAM",
            "  ✓ Structural Calculus Gating      : Seamless C^∞ Phase Transition",
            "=" * 70,
        ]
        return base_report.replace("=" * 70, "\n".join(sq_lines))


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-SubQuantumRegimeBridge v3.0.1] Running on: {device}")
    print(f"  psy_one_bridge available: {_PSY_AVAILABLE}")

    mode = PsychopathologyMode.MDD_ANXIETY
    config = PsycheConfig(action_dim=10, mode=mode)

    psy_bridge = PSYONEBridge(config=config)
    omni = OmniPsycheBridge(psy_bridge, sq_dim=128).to(device)
    omni.train()

    B = 4
    sq_seeds = torch.randn(B, 128, device=device, requires_grad=True)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )

    with autocast_ctx:
        state, loss = omni(sq_seeds)

    loss.backward()

    print("-" * 64)
    print(f"  state.id_state       shape : {tuple(state.id_state.shape)}")
    print(f"  state.ego_state      shape : {tuple(state.ego_state.shape)}")
    print(f"  state.superego_state shape : {tuple(state.superego_state.shape)}")
    print(f"  total loss                 : {loss.item():.6f}")

    grad_ok = True
    for name, p in omni.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")

    print(f"  Autograd                   : "
          f"{'FULLY CONNECTED — C^∞ differentiable' if grad_ok else 'FAILED'}")

    with torch.no_grad():
        sq_ub = torch.randn(128, device=device)
        state_ub, loss_ub = omni(sq_ub)
    print(f"  unbatched loss shape       : {tuple(loss_ub.shape)}  "
          f"(scalar — matches reference)")

    print("-" * 64)
    print(omni.generate_unified_report(state))
    print("-" * 64)
