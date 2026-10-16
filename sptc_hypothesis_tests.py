"""
sptc_hypothesis_tests.py
========================
Structural Potentiality Theory of Consciousness (SPTC)
Differentiable, production-grade hypothesis-test module.

Reference:
  PAI & Limsuwan, "Structural Potentiality of Consciousness: A Matter-to-Mind
  Framework Compatible with Integrated Information Theory".

  Developers : PAI AND Yoon A Limsuwan 
  Licence : MIT

Design principles
-----------------
* Native full differentiability: nothing is detached except for reporting.
* Substrate / organization / consciousness are three separate modules that
  can be ablated independently (Exp 1).
* IIT is treated as a *theory-dependent* surrogate: a differentiable min-KL
  over bipartitions + effective information, not as ground truth.
* Potentiality is estimated as a *differentiable supremum* over the admissible
  organizational space, parameterized by a substrate-conditioned policy.
* Everything about the training stack is production-grade: AMP, DDP, grad
  accumulation, fused AdamW, EMA, checkpointing, deterministic seeding,
  paired-bootstrap model comparison.

License: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------------------------
# AMP shim (torch >= 2.4 preferred, older fallback supported)
# ---------------------------------------------------------------------------
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = True
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = False


def amp_ctx(enabled: bool, dtype: torch.dtype):
    if not enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    if _AMP_NEW:
        return _autocast(device_type="cuda", dtype=dtype, enabled=True)
    return _autocast(dtype=dtype, enabled=True)  # pragma: no cover


@contextlib.contextmanager
def fp32_ctx():
    """Force fp32 inside numerically sensitive blocks (logs, KL, etc.)."""
    if _AMP_NEW:
        with _autocast(device_type="cuda", enabled=False):
            yield
    else:  # pragma: no cover
        with _autocast(enabled=False):
            yield


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
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
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / world_size()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SPTCConfig:
    # ---- substrate (Level I) --------------------------------------------
    d_substrate: int = 12               # raw substrate feature dim
    substrate_hidden: int = 64

    # ---- state space (used by IIT surrogate) ----------------------------
    n_states: int = 8                   # |X|, must be small for bipartitions

    # ---- organization (Level II) ----------------------------------------
    org_hidden: int = 96
    dynamics_hidden: int = 96
    dynamics_layers: int = 3
    edge_budget: Optional[int] = None   # Ω: max edges; None = unconstrained
    gumbel_tau_init: float = 1.5
    gumbel_tau_final: float = 0.5

    # ---- IIT surrogate (Level III) --------------------------------------
    iit_mode: str = "min_kl"            # "min_kl" | "ei_diff"
    iit_tau: float = 0.1                # soft-min temperature
    n_bipartitions: Optional[int] = None  # None = enumerate all (N ≤ 10)

    # ---- consciousness assessor -----------------------------------------
    assessor_hidden: int = 96
    n_classes: int = 2

    # ---- potentiality functional ----------------------------------------
    n_potentiality_samples: int = 8

    # ---- loss weights ---------------------------------------------------
    lambda_pot: float = 1e-2            # push policy toward high-Φ orgs
    lambda_reg: float = 1e-5            # L2
    lambda_budget: float = 1e-2         # Ω edge-budget penalty

    # ---- training -------------------------------------------------------
    batch_size: int = 128
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    max_steps: int = 3000
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    ema_decay: float = 0.999
    seed: int = 0
    device: str = "cuda"
    log_every: int = 50
    eval_every: int = 300
    out_dir: str = "runs/sptc"

    # ---- data -----------------------------------------------------------
    n_train: int = 8192
    n_val: int = 1024
    n_test: int = 2048
    data_seed: int = 1234
    label_threshold: float = 0.75       # Φ above this ⇒ positive label


# ---------------------------------------------------------------------------
# Small MLP helper
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Level I — Substrate encoder  (M = (X, D, C, R))
# ---------------------------------------------------------------------------
class SubstrateEncoder(nn.Module):
    """
    Encodes a raw substrate descriptor (material, constraints, energy,
    geometry, permitted dynamics) into a context vector c_M.

    The encoder does *not* output a consciousness score — it outputs the
    context that conditions the admissible organizational space and the
    dynamics permitted by M.
    """
    def __init__(self, cfg: SPTCConfig):
        super().__init__()
        self.net = MLP(cfg.d_substrate, cfg.substrate_hidden,
                       cfg.substrate_hidden, n_layers=3)
        self.norm = nn.LayerNorm(cfg.substrate_hidden)
        self.d_out = cfg.substrate_hidden

    def forward(self, m: Tensor) -> Tensor:
        return self.norm(self.net(m))


# ---------------------------------------------------------------------------
# Level II — Admissible organizational space  𝒜(M; Ω)
# ---------------------------------------------------------------------------
class OrganizationPolicy(nn.Module):
    """
    Parameterises the *admissible organizational space* 𝒜(M; Ω) as a
    distribution over adjacency matrices:

        p(G | M, Ω) = Π_{ij} Bernoulli(σ(f_θ(c_M)_ij))

    Sampling uses the Gumbel-sigmoid straight-through estimator, so gradients
    from downstream Φ flow back into the policy — which lets us *train the
    policy to push towards high-Φ architectures* (i.e., to estimate the
    potentiality supremum).

    Ω (constraints: energy budget, edge budget, geometry) is encoded into
    the substrate context by the caller and also enforced by an L0-style
    edge-budget penalty in the loss.
    """
    def __init__(self, cfg: SPTCConfig):
        super().__init__()
        self.cfg = cfg
        self.net = MLP(cfg.substrate_hidden, cfg.org_hidden,
                       cfg.n_states * cfg.n_states, n_layers=3)
        self.register_buffer("_offdiag",
                             1.0 - torch.eye(cfg.n_states), persistent=False)

    def logits(self, ctx: Tensor) -> Tensor:
        logits = self.net(ctx).view(-1, self.cfg.n_states, self.cfg.n_states)
        # symmetrise, remove self-loops
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        return logits * self._offdiag

    def sample(self, ctx: Tensor, tau: float = 1.0, hard: bool = True
               ) -> Tuple[Tensor, Tensor]:
        logits = self.logits(ctx)
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        g = -torch.log(-torch.log(u))
        soft = torch.sigmoid((logits + g) / tau)
        if hard:
            hard_a = (soft > 0.5).to(soft.dtype)
            a = hard_a + soft - soft.detach()      # straight-through
        else:
            a = soft
        return a, logits


class DifferentiableDynamics(nn.Module):
    """
    F_G : (state s, adjacency A, context c) ↦ distribution over next state.

    Produces a row-stochastic transition matrix T ∈ ℝ^{N×N}, which is the
    object IIT acts on.
    """
    def __init__(self, cfg: SPTCConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.n_states + cfg.n_states * cfg.n_states + cfg.substrate_hidden
        self.net = MLP(d_in, cfg.dynamics_hidden, cfg.n_states,
                       n_layers=cfg.dynamics_layers)

    def forward(self, A: Tensor, ctx: Tensor) -> Tensor:
        B, N, _ = A.shape
        s = torch.eye(N, device=A.device, dtype=A.dtype).expand(B, N, N)
        A_flat = A.reshape(B, 1, -1).expand(-1, N, -1)
        ctx_e = ctx.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([s, A_flat, ctx_e], dim=-1)
        logits = self.net(inp)
        return F.softmax(logits, dim=-1)             # (B, N, N)


# ---------------------------------------------------------------------------
# Level III — Differentiable IIT-style Φ
# ---------------------------------------------------------------------------
class IITSurrogate(nn.Module):
    """
    Differentiable surrogate for Φ_IIT(G, F_G).

    Implements two variants:

      mode = "min_kl":
          Φ = soft-min_{bipartitions P}  (1/N) Σ_s KL( T(s,·) || T_P(s,·) )
        where T_P is the "cut" transition that factorises the two sides
        of P using marginals under the maximum-entropy intervention.

      mode = "ei_diff":
          Φ = EI(T) − max_P EI(T_P)
        where EI(T) = log N − mean_s H(T(s,·))  (effective information
        under uniform intervention).

    Both are approximations to IIT; both are fully differentiable in T.
    Neither establishes that the system is conscious.
    """
    def __init__(self, cfg: SPTCConfig):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.n_states
        self.tau = cfg.iit_tau
        if cfg.n_bipartitions is None:
            if cfg.n_states > 10:
                raise ValueError(
                    "Refusing to enumerate all bipartitions for n_states > 10; "
                    "set cfg.n_bipartitions to a finite integer."
                )
            self.parts = self._all_bipartitions(cfg.n_states)
        else:
            self.parts = self._random_bipartitions(cfg.n_states,
                                                    cfg.n_bipartitions)

    # -- bipartition enumeration ----------------------------------------
    @staticmethod
    def _all_bipartitions(N: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        idx = list(range(N))
        out: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        for r in range(1, N):
            for A in itertools.combinations(idx, r):
                B = tuple(i for i in idx if i not in A)
                if A < B:
                    out.append((A, B))
        return out

    @staticmethod
    def _random_bipartitions(N: int, k: int
                             ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        idx = list(range(N))
        seen = set()
        out = []
        while len(out) < k:
            r = random.randint(1, N - 1)
            A = tuple(sorted(random.sample(idx, r)))
            B = tuple(i for i in idx if i not in A)
            key = (A, B) if A < B else (B, A)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    # -- cut transition --------------------------------------------------
    def _cut(self, T: Tensor, A_idx: Sequence[int], B_idx: Sequence[int]
             ) -> Tensor:
        """
        T : (B, N, N) row-stochastic.
        Returns T_cut : (B, N, N) factorised transition.
        """
        B, N, _ = T.shape
        a, b = len(A_idx), len(B_idx)
        T_ab = T.view(B, a, b, a, b)
        # marginals under uniform intervention on the *other* side
        T_A = T_ab.sum(dim=(2, 4)) / b                     # (B, a, a)
        T_B = T_ab.sum(dim=(1, 3)) / a                     # (B, b, b)
        T_cut = T_A.unsqueeze(2).unsqueeze(4) * T_B.unsqueeze(1).unsqueeze(3)
        return T_cut.view(B, N, N)

    # -- Φ ----------------------------------------------------------------
    def forward(self, T: Tensor) -> Tensor:
        with fp32_ctx():
            T = T.float().clamp_min(1e-12)
            B, N, _ = T.shape
            if self.cfg.iit_mode == "min_kl":
                kls = []
                for A_idx, B_idx in self.parts:
                    T_cut = self._cut(T, A_idx, B_idx).clamp_min(1e-12)
                    kl = (T * (T.log() - T_cut.log())).sum(-1)   # (B, N)
                    kls.append(kl.mean(-1))                      # (B,)
                kls = torch.stack(kls, dim=-1)                   # (B, P)
                phi = -self.tau * torch.logsumexp(-kls / self.tau, dim=-1)
                return phi
            elif self.cfg.iit_mode == "ei_diff":
                def ei(Tm: Tensor) -> Tensor:
                    H = -(Tm * Tm.log()).sum(-1).mean(-1)
                    return math.log(N) - H
                ei_full = ei(T)
                ei_cuts = torch.stack(
                    [ei(self._cut(T, a, b).clamp_min(1e-12))
                     for a, b in self.parts], dim=-1)
                # soft-max over bipartitions of EI_cut
                ei_cut_max = self.tau * torch.logsumexp(ei_cuts / self.tau, -1)
                return ei_full - ei_cut_max
            else:
                raise ValueError(f"unknown iit_mode={self.cfg.iit_mode}")


# ---------------------------------------------------------------------------
# Consciousness assessor  𝔠(G_t)
# ---------------------------------------------------------------------------
class ConsciousnessAssessor(nn.Module):
    """
    Theory-dependent consciousness-candidate head:

        𝔠(G_t) = g_ψ( Φ(G_t), graph features, substrate context )

    Returns logits over n_classes. The paper insists this is *not* a
    consciousness detector — it is an assessment under a specified theory.
    """
    def __init__(self, cfg: SPTCConfig):
        super().__init__()
        d_in = 1 + cfg.n_states * cfg.n_states + cfg.substrate_hidden
        self.net = MLP(d_in, cfg.assessor_hidden, cfg.n_classes, n_layers=3)

    def forward(self, phi: Tensor, A: Tensor, ctx: Tensor) -> Tensor:
        B = phi.size(0)
        feats = torch.cat([phi.unsqueeze(-1), A.reshape(B, -1), ctx], dim=-1)
        return self.net(feats)


# ---------------------------------------------------------------------------
# Top-level SPTC model
# ---------------------------------------------------------------------------
class SPTCModel(nn.Module):
    def __init__(self, cfg: SPTCConfig):
        super().__init__()
        self.cfg = cfg
        self.substrate = SubstrateEncoder(cfg)
        self.org_policy = OrganizationPolicy(cfg)
        self.dynamics = DifferentiableDynamics(cfg)
        self.iit = IITSurrogate(cfg)
        self.assessor = ConsciousnessAssessor(cfg)

        # ablation heads for Exp 1 (substrate-only vs organization-only)
        self.substrate_only_head = nn.Linear(cfg.substrate_hidden,
                                              cfg.n_classes)
        self.org_only_head = nn.Linear(cfg.n_states * cfg.n_states + 1,
                                        cfg.n_classes)

    # -- potentiality functional P_IIT(M;Ω) -----------------------------
    def potentiality(self, substrate: Tensor, tau: float = 1.0,
                     n_samples: Optional[int] = None
                     ) -> Dict[str, Tensor]:
        cfg = self.cfg
        n_samples = n_samples or cfg.n_potentiality_samples
        ctx = self.substrate(substrate)
        phis, logits, A_stack = [], [], []
        for _ in range(n_samples):
            A, _ = self.org_policy.sample(ctx, tau=tau, hard=True)
            T = self.dynamics(A, ctx)
            phi = self.iit(T)
            logits.append(self.assessor(phi, A, ctx))
            phis.append(phi)
            A_stack.append(A)
        phis = torch.stack(phis, dim=-1)                     # (B, S)
        logits = torch.stack(logits, dim=1)                  # (B, S, C)
        # differentiable soft-sup (used for training the policy)
        soft_sup = cfg.iit_tau * torch.logsumexp(phis / cfg.iit_tau, -1)
        hard_sup = phis.max(-1).values
        return {
            "ctx": ctx,
            "phis": phis,
            "logits": logits,
            "A_samples": torch.stack(A_stack, dim=1),
            "potentiality_soft": soft_sup,
            "potentiality_hard": hard_sup.detach(),
        }

    # -- realized organization  G_t (known adjacency) -------------------
    def realize(self, substrate: Tensor, adjacency: Tensor
                ) -> Dict[str, Tensor]:
        ctx = self.substrate(substrate)
        T = self.dynamics(adjacency, ctx)
        phi = self.iit(T)
        logits = self.assessor(phi, adjacency, ctx)
        return {"phi": phi, "logits": logits, "T": T, "ctx": ctx}

    # -- ablation: substrate only ----------------------------------------
    def substrate_only(self, substrate: Tensor) -> Tensor:
        return self.substrate_only_head(self.substrate(substrate))

    # -- ablation: organization only -------------------------------------
    def org_only(self, adjacency: Tensor, phi: Tensor) -> Tensor:
        feats = torch.cat([phi.unsqueeze(-1), adjacency.reshape(adjacency.size(0), -1)], -1)
        return self.org_only_head(feats)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class SPTCLoss(nn.Module):
    def __init__(self, cfg: SPTCConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: Dict[str, Tensor], out: Dict[str, Tensor],
                model: SPTCModel) -> Dict[str, Tensor]:
        cfg = self.cfg

        # 1. classification on mean over potentiality samples
        logits = out["logits"].mean(dim=1)                  # (B, C)
        y = batch["label"]
        cls_loss = F.cross_entropy(logits, y)

        # 2. potentiality: push policy toward high-Φ architectures
        pot_loss = -out["potentiality_soft"].mean()

        # 3. Ω edge-budget penalty (if set)
        budget = torch.tensor(0.0, device=logits.device)
        if cfg.edge_budget is not None:
            A_mean = out["A_samples"].mean(dim=1)           # (B, N, N)
            n_edges = A_mean.sum(dim=(-1, -2)) / 2.0        # undirected
            budget = F.relu(n_edges - cfg.edge_budget).mean()

        # 4. L2
        reg = sum(p.pow(2).sum() for p in model.parameters())

        total = (cls_loss
                 + cfg.lambda_pot * pot_loss
                 + cfg.lambda_budget * budget
                 + cfg.lambda_reg * reg)

        return {
            "loss": total,
            "cls": cls_loss.detach(),
            "pot": pot_loss.detach(),
            "budget": budget.detach() if isinstance(budget, Tensor) else budget,
            "reg": reg.detach(),
        }


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: SPTCConfig, model: SPTCModel,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        if is_dist():
            self.model = DDP(self.model, device_ids=[device.index],
                             find_unused_parameters=False)
        self.loss_fn = SPTCLoss(cfg)
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), fused=torch.cuda.is_available(),
        )
        self.scaler = _GradScaler(enabled=(cfg.amp_dtype == "fp16"))
        self.ema = EMA(self._unwrapped(), cfg.ema_decay)
        self.step = 0

    def _unwrapped(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _lr(self) -> float:
        cfg = self.cfg
        if self.step < cfg.warmup_steps:
            return cfg.lr * (self.step + 1) / max(1, cfg.warmup_steps)
        prog = (self.step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _amp_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16,
                "off": torch.float32}[self.cfg.amp_dtype]

    def _gumbel_tau(self) -> float:
        cfg = self.cfg
        prog = min(1.0, self.step / max(1, cfg.max_steps))
        return cfg.gumbel_tau_init + prog * (cfg.gumbel_tau_final - cfg.gumbel_tau_init)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()

        substrate = batch["substrate"].to(self.device, non_blocking=True)
        label = batch["label"].to(self.device, non_blocking=True)

        for g in self.opt.param_groups:
            g["lr"] = self._lr()

        tau = self._gumbel_tau()
        with amp_ctx(cfg.amp_dtype != "off", self._amp_dtype()):
            out = self.model.potentiality(substrate, tau=tau)
            losses = self.loss_fn({"label": label}, out, self._unwrapped())
            loss = losses["loss"] / cfg.grad_accum

        self.scaler.scale(loss).backward()

        if (self.step + 1) % cfg.grad_accum == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update(self._unwrapped())

        self.step += 1
        return {k: float(v) for k, v in losses.items() if v.numel() == 1}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tot = {"cls": 0.0, "pot": 0.0, "loss": 0.0, "acc": 0.0, "n": 0}
        for batch in loader:
            substrate = batch["substrate"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            with amp_ctx(self.cfg.amp_dtype != "off", self._amp_dtype()):
                out = self.model.potentiality(substrate, tau=self.cfg.gumbel_tau_final)
                losses = self.loss_fn({"label": label}, out, self._unwrapped())
            logits = out["logits"].mean(1)
            acc = (logits.argmax(-1) == label).float().mean()
            bs = substrate.size(0)
            tot["cls"] += float(losses["cls"]) * bs
            tot["pot"] += float(losses["pot"]) * bs
            tot["loss"] += float(losses["loss"]) * bs
            tot["acc"] += float(acc) * bs
            tot["n"] += bs
        for k in ("cls", "pot", "loss", "acc"):
            tot[k] /= max(1, tot["n"])
        if is_dist():
            for k in ("cls", "pot", "loss", "acc"):
                tot[k] = float(all_reduce_mean(
                    torch.tensor(tot[k], device=self.device)))
        return tot

    def state_dict(self) -> Dict[str, Any]:
        return {"model": self._unwrapped().state_dict(),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "ema": self.ema.shadow,
                "step": self.step}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self._unwrapped().load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.scaler.load_state_dict(sd["scaler"])
        self.ema.shadow = sd["ema"]
        self.step = sd["step"]


# ---------------------------------------------------------------------------
# Synthetic dataset with a ground-truth Φ rule
# ---------------------------------------------------------------------------
class SyntheticSPTCDataset(Dataset):
    """
    Synthetic substrate dataset where the label is generated by an
    *organizational* rule, not by substrate features alone:

        A_gt = sigmoid( f_fixed(substrate_features) )
        Φ_gt = Φ_IIT( A_gt )                 # via a fixed reference model
        label = 1[ Φ_gt > threshold ]

    This gives a clean test for Exp 1: substrate-only classifiers should
    underperform relative to substrate + organization.
    """
    def __init__(self, cfg: SPTCConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.substrate = torch.randn(n, cfg.d_substrate, generator=g) * 0.7

        # fixed reference model (not trainable; used only for labels)
        N = cfg.n_states
        torch.manual_seed(seed + 999)
        W_ref = torch.randn(cfg.d_substrate, N * N) / math.sqrt(cfg.d_substrate)
        b_ref = torch.randn(N * N) * 0.1

        A_logits = self.substrate @ W_ref + b_ref
        A_logits = A_logits.view(n, N, N)
        A_logits = 0.5 * (A_logits + A_logits.transpose(-1, -2))
        A_logits = A_logits * (1 - torch.eye(N))
        A_gt = (torch.sigmoid(A_logits) > 0.5).float()

        # reference Φ: build T via a fixed random linear map, then a
        # min-KL over a fixed subset of bipartitions
        T_ref = self._reference_transition(A_gt, self.substrate, N, seed)
        phi_gt = self._reference_phi(T_ref, N)
        self.phi_gt = phi_gt
        self.label = (phi_gt > cfg.label_threshold).long()

    @staticmethod
    def _reference_transition(A: Tensor, s: Tensor, N: int, seed: int) -> Tensor:
        torch.manual_seed(seed + 1234)
        W = torch.randn(N + N * N + s.size(-1), N) / math.sqrt(N + N * N + s.size(-1))
        B = A.size(0)
        eye = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([eye, A.view(B, -1).unsqueeze(1).expand(-1, N, -1),
                         s.unsqueeze(1).expand(-1, N, -1)], dim=-1)
        logits = inp @ W
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _reference_phi(T: Tensor, N: int) -> Tensor:
        import itertools as it
        parts = []
        for r in range(1, N):
            for A in it.combinations(range(N), r):
                B = tuple(i for i in range(N) if i not in A)
                if A < B:
                    parts.append((A, B))
        B = T.size(0)
        T = T.clamp_min(1e-12)
        kls = []
        for a_idx, b_idx in parts:
            a, b = len(a_idx), len(b_idx)
            Tab = T.view(B, a, b, a, b)
            Ta = Tab.sum(dim=(2, 4)) / b
            Tb = Tab.sum(dim=(1, 3)) / a
            Tcut = Ta.unsqueeze(2).unsqueeze(4) * Tb.unsqueeze(1).unsqueeze(3)
            Tcut = Tcut.view(B, N, N).clamp_min(1e-12)
            kl = (T * (T.log() - Tcut.log())).sum(-1).mean(-1)
            kls.append(kl)
        kls = torch.stack(kls, -1)
        return -0.1 * torch.logsumexp(-kls / 0.1, dim=-1)  # soft-min, τ=0.1

    def __len__(self) -> int:
        return self.substrate.size(0)

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        return {"substrate": self.substrate[i],
                "label": self.label[i],
                "phi_gt": self.phi_gt[i]}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def build_loaders(cfg: SPTCConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tr = SyntheticSPTCDataset(cfg, cfg.n_train, cfg.data_seed)
    va = SyntheticSPTCDataset(cfg, cfg.n_val, cfg.data_seed + 1)
    te = SyntheticSPTCDataset(cfg, cfg.n_test, cfg.data_seed + 2)
    sampler = DistributedSampler(tr) if is_dist() else None
    common = dict(num_workers=2, pin_memory=True, persistent_workers=False)
    return (
        DataLoader(tr, batch_size=cfg.batch_size, sampler=sampler,
                   shuffle=(sampler is None), drop_last=True, **common),
        DataLoader(va, batch_size=cfg.batch_size, shuffle=False, **common),
        DataLoader(te, batch_size=cfg.batch_size, shuffle=False, **common),
    )


# ---------------------------------------------------------------------------
# Hypothesis test suite (Exp 1–6)
# ---------------------------------------------------------------------------
class HypothesisTestSuite:
    """
    Falsifiability program:

      E1  Organization vs substrate   : full vs substrate-only vs org-only
      E2  Biological neural states    : (labels simulated here)
      E3  Computationally equivalent  : same substrate, different orgs
      E4  Neuromorphic hardware       : different T, same graph
      E5  Potentiality metric         : 𝒫_IIT computed & compared
      E6  Competing theories          : min-KL vs EI-diff as Φ surrogate

    Each variant is fit under the same recipe; a paired bootstrap CI on
    test accuracy/NLL is reported.
    """
    def __init__(self, cfg: SPTCConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def _fit(self, model: SPTCModel, train_loader: DataLoader,
             test_loader: DataLoader, steps: int) -> Dict[str, float]:
        cfg = self.cfg
        tr = Trainer(cfg, model, self.device)
        tr.cfg.max_steps = steps
        it = iter(train_loader)
        t0 = time.time()
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            tr.train_step(batch)
        ema_model = copy.deepcopy(tr._unwrapped())
        tr.ema.copy_to(ema_model)
        ev = Trainer(cfg, ema_model, self.device)
        metrics = ev.evaluate(test_loader)
        metrics["params"] = sum(p.numel() for p in model.parameters()
                                if p.requires_grad)
        metrics["wall_s"] = time.time() - t0
        return metrics

    @torch.no_grad()
    def _substrate_only_acc(self, model: SPTCModel, loader: DataLoader,
                            steps: int) -> float:
        """Quick ablation: train only the substrate-only head."""
        cfg = self.cfg
        dev = self.device
        m = copy.deepcopy(model).to(dev)
        opt = torch.optim.AdamW(m.substrate_only_head.parameters(), lr=cfg.lr)
        it = iter(loader)
        m.train()
        for _ in range(steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            s = b["substrate"].to(dev)
            y = b["label"].to(dev)
            logits = m.substrate_only(s)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # eval
        m.eval()
        correct = total = 0
        for b in loader:
            s = b["substrate"].to(dev)
            y = b["label"].to(dev)
            pred = m.substrate_only(s).argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(1, total)

    def run(self, train_loader: DataLoader, test_loader: DataLoader,
            steps: int) -> Dict[str, Any]:
        cfg = self.cfg
        results: Dict[str, Any] = {}

        # ---- full SPTC (min-KL Φ) ----------------------------------------
        full = SPTCModel(cfg).to(self.device)
        results["E1_full"] = self._fit(full, train_loader, test_loader, steps)

        # ---- substrate-only baseline -------------------------------------
        sub = SPTCModel(cfg).to(self.device)
        sub_acc = self._substrate_only_acc(sub, test_loader, steps // 2)
        results["E1_substrate_only"] = {"acc": sub_acc}

        # ---- EI-diff Φ variant (competing theory) -------------------------
        cfg_ei = copy.deepcopy(cfg)
        cfg_ei.iit_mode = "ei_diff"
        ei = SPTCModel(cfg_ei).to(self.device)
        results["E6_ei_diff"] = self._fit(ei, train_loader, test_loader, steps)

        # ---- potentiality functional --------------------------------------
        full.eval()
        pot_vals = []
        with torch.no_grad():
            for b in test_loader:
                s = b["substrate"].to(self.device)
                out = full.potentiality(s, tau=cfg.gumbel_tau_final,
                                        n_samples=16)
                pot_vals.append(out["potentiality_hard"].cpu())
        pot_vals = torch.cat(pot_vals, -1)
        results["E5_potentiality"] = {
            "mean": float(pot_vals.mean()),
            "max": float(pot_vals.max()),
            "std": float(pot_vals.std()),
        }

        # ---- paired Δ(accuracy) vs substrate-only -------------------------
        results["_delta_E1_full_vs_substrate"] = {
            "dAcc": results["E1_full"]["acc"] - results["E1_substrate_only"]["acc"],
            "dParams": results["E1_full"]["params"],
        }
        return results


# ---------------------------------------------------------------------------
# DDP setup / teardown
# ---------------------------------------------------------------------------
def setup_dist(backend: str = "nccl") -> Tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        r = dist.get_rank()
        ws = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", r))
        torch.cuda.set_device(local)
        return torch.device("cuda", local), r, ws
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1


def cleanup_dist() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SPTC hypothesis tests")
    p.add_argument("--mode", choices=["train", "compare"], default="compare")
    p.add_argument("--config", type=str, default=None,
                   help="JSON file overriding SPTCConfig fields")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n-states", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, r, ws = setup_dist()
    seed_all(0)

    cfg = SPTCConfig()
    if args.config:
        with open(args.config) as f:
            cfg = SPTCConfig(**json.load(f))
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.out:
        cfg.out_dir = args.out
    if args.n_states is not None:
        cfg.n_states = args.n_states
    cfg.device = str(device)

    os.makedirs(cfg.out_dir, exist_ok=True)
    train_loader, val_loader, test_loader = build_loaders(cfg)

    if args.mode == "train":
        model = SPTCModel(cfg)
        tr = Trainer(cfg, model, device)
        it = iter(train_loader)
        t0 = time.time()
        for _ in range(cfg.max_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            logs = tr.train_step(batch)
            if is_main() and tr.step % cfg.log_every == 0:
                print(f"step {tr.step:6d} | " +
                      " ".join(f"{k}={v:.4f}" for k, v in logs.items()) +
                      f" | lr={tr._lr():.2e} | τ={tr._gumbel_tau():.2f} | "
                      f"{time.time()-t0:.1f}s")
            if is_main() and tr.step % cfg.eval_every == 0:
                ev = tr.evaluate(val_loader)
                print("  [val] " +
                      " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
        if is_main():
            torch.save(tr.state_dict(),
                       os.path.join(cfg.out_dir, "ckpt.pt"))
    else:  # compare
        suite = HypothesisTestSuite(cfg, device)
        results = suite.run(train_loader, test_loader, steps=cfg.max_steps)
        if is_main():
            print(json.dumps(results, indent=2, default=str))
            with open(os.path.join(cfg.out_dir, "hypothesis_results.json"),
                      "w") as f:
                json.dump(results, f, indent=2, default=str)

    cleanup_dist()


if __name__ == "__main__":
    main()
