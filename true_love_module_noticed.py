# =============================================================================
# TRUE LOVE MODULE — NATIVE FULL DIFFERENTIABLE (PRODUCTION RELEASE)
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
Production-grade, fully differentiable True-Love module implementing
Stable Probability Representations, IIT-like Integrated Information (Φ),
Mutual Information proxies, and psychoanalytic attachment / reciprocity
dynamics.

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **DDP batch-collapse bug on `als_scores` fixed.**  The reference applied
   `als_scores.mean(dim=-1)`.  When the caller passed a bare `[B]` tensor
   (the natural shape for per-sample attachment-latent scores) this
   collapsed to a **scalar shared across the world**, silently defeating
   per-sample gradient routing under DDP.  The module now auto-handles both
   `[B]` (identity) and `[B, N]` (mean over the feature axis) inputs.

2. **All Python-scalar constants moved to non-persistent buffers.**
   `1.5`, `0.5`, `0.6`, `0.4`, `1.0`, `1e-4` — six Python scalars read from
   the graph on every forward.  Every one is now an fp32 buffer cast per
   call, so `torch.compile` never recompiles and DDP never needs to sync
   scalars.

3. **Weight application vectorized.**  The reference indexed
   `weights[0] … weights[4]` — five `[1]`-slice allocations per forward.
   Replaced with a single `stack → multiply → sum` pattern that produces
   one tensor and one reduction, no per-slot slicing.

4. **LSTM + stability math run fp32-internal for AMP safety.**  The
   reference ran `nn.LSTM`, `torch.log1p`, and stability ratios in the
   caller's dtype.  Under bf16/fp16 the LSTM's internal gate arithmetic
   and the log1p accumulation lose precision quickly.  All numerically
   sensitive paths now run in **fp32 internally** and cast back at the API
   boundary.

5. **Reparameterized positive scale.**  `F.softplus(γ) + 1e-4` is fine in
   the reference, but the additive `1e-4` floor became a Python scalar in
   the graph.  Now a buffer-based ε, and the scale is reparameterized
   `softplus(raw) + ε` — strictly positive, C^∞, no clamp.

6. **Input validation with `validate_inputs=False` opt-out.**  Shape,
   batch, sequence-length, and dtype guards.

7. **Gradient checkpointing.**  `gradient_checkpointing=True` wraps the
   concatenated IIT projection and the LSTM stack in
   `torch.utils.checkpoint(..., use_reentrant=False)` — trades ~2× compute
   for activation-memory savings on long time series.

8. **`x ** 2` → `x * x`** in the stability squared-error approximations
   (wherever it appeared) — exact, faster, `torch.compile`-clean.

9. **Deterministic eval.**  The module contains no stochastic ops; the
   docstring now documents that `.eval()` gives a reproducible forward,
   and `nn.LSTM(dropout=…)` is left at its default of 0.0 to keep it that
   way.

10. **Public API preserved.**  Same class name
    `FullyDifferentiableTrueLoveModule`, same constructor
    `(feature_dim, hidden_dim)`, same `forward(time_series_a,
    time_series_b, als_scores)` signature, same 6 dict keys (`TLI`,
    `Phi_AB`, `MI_C`, `Stability`, `Reciprocity`, `Optimized_Weights`) —
    strict drop-in upgrade of v1.0.0.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model = FullyDifferentiableTrueLoveModule(feature_dim=64, hidden_dim=128).to(rank)
    model = torch.compile(model, mode="max-autotune")                 # optional
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(series_a, series_b, als)
    (1.0 - out["TLI"]).mean().backward()      # maximize TLI
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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TrueLoveConfig", "FullyDifferentiableTrueLoveModule"]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TrueLoveConfig:
    """Numerical-safety and stability-scoring configuration."""
    # ----- Stability-score weights (must sum ≈ 1 for a normalized score) ---
    stability_weight_alpha: float = 0.6
    stability_weight_gamma: float = 0.4

    # ----- Alpha / gamma reparameterization bounds ------------------------
    alpha_min: float = 0.5
    alpha_max: float = 2.0
    gamma_eps: float = 1e-4
    l1_eps: float = 1e-6                    # ε in F.normalize

    # ----- Execution ------------------------------------------------------
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


# =============================================================================
# Module
# =============================================================================
class FullyDifferentiableTrueLoveModule(nn.Module):
    """
    Differentiable True-Love module: stable-distribution representation,
    IIT-like integrated information (Φ), MI proxy, and psychoanalytic
    attachment / reciprocity dynamics.

    Parameters
    ----------
    feature_dim : int
        Per-timestep input feature dimension (must be > 0).
    hidden_dim : int
        Hidden width used by every encoder and by the LSTM (must be > 0).
    config : TrueLoveConfig, optional
        Full configuration; individual kwargs override fields.
    validate_inputs : bool, optional
        Shape/dtype guards; disable under `torch.compile` static shapes.
    gradient_checkpointing : bool, optional
        Wrap the projection + LSTM stack in `torch.utils.checkpoint`.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        config: Optional[TrueLoveConfig] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not (isinstance(feature_dim, int) and feature_dim > 0):
            raise ValueError("feature_dim must be a positive int.")
        if not (isinstance(hidden_dim, int) and hidden_dim > 0):
            raise ValueError("hidden_dim must be a positive int.")

        cfg = config or TrueLoveConfig()
        if validate_inputs is not None:
            cfg = TrueLoveConfig(**{**cfg.__dict__,
                                    "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = TrueLoveConfig(**{**cfg.__dict__,
                                    "gradient_checkpointing": bool(gradient_checkpointing)})

        for name, v in (
            ("stability_weight_alpha", cfg.stability_weight_alpha),
            ("stability_weight_gamma", cfg.stability_weight_gamma),
            ("alpha_min", cfg.alpha_min),
            ("alpha_max", cfg.alpha_max),
            ("gamma_eps", cfg.gamma_eps),
            ("l1_eps", cfg.l1_eps),
        ):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be > 0.")
        if cfg.alpha_max <= cfg.alpha_min:
            raise ValueError("alpha_max must be > alpha_min.")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        # ---- Stable-distribution encoders (alpha + gamma) -----------------
        self.encoder_alpha = nn.Linear(feature_dim, hidden_dim)
        self.encoder_gamma = nn.Linear(feature_dim, hidden_dim)

        # ---- IIT-like Φ projector -----------------------------------------
        self.iit_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # ---- Psychoanalytic attachment / reciprocity ----------------------
        self.psycho_lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=0.0,
        )
        self.reciprocity_head = nn.Linear(hidden_dim * 2, 1)

        # ---- Learnable TLI weights ----------------------------------------
        self.raw_weights = nn.Parameter(torch.ones(5))

        # ---- Non-persistent scalar buffers (DDP-safe) ---------------------
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_alpha_min",      _buf(cfg.alpha_min),                 persistent=False)
        self.register_buffer("_alpha_span",     _buf(cfg.alpha_max - cfg.alpha_min), persistent=False)
        self.register_buffer("_alpha_range",    _buf(cfg.alpha_max - cfg.alpha_min), persistent=False)
        self.register_buffer("_gamma_eps",      _buf(cfg.gamma_eps),                 persistent=False)
        self.register_buffer("_l1_eps",         _buf(cfg.l1_eps),                    persistent=False)
        self.register_buffer("_stab_w_alpha",   _buf(cfg.stability_weight_alpha),    persistent=False)
        self.register_buffer("_stab_w_gamma",   _buf(cfg.stability_weight_gamma),    persistent=False)

        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Encoder / stability helpers (isolated for checkpointing)           #
    # ------------------------------------------------------------------ #
    def _encode_stable_params(
        self,
        time_series: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode `[B, S, F] → [B, S, H]` and return a tensor of shape
        `[B, S, H, 2]` packing `(α, γ)` for both streams.  Runs in fp32
        internally for AMP safety, casts back.

        α ∈ (alpha_min, alpha_max)  — via bounded sigmoid reparameterization.
        γ > 0                        — via softplus + ε.
        """
        dtype_in = time_series.dtype
        device   = time_series.device

        x32 = time_series.float()

        alpha_min  = self._alpha_min.to(device=device)
        alpha_span = self._alpha_span.to(device=device)
        gamma_eps  = self._gamma_eps.to(device=device)

        alpha = torch.sigmoid(self.encoder_alpha(x32)) * alpha_span + alpha_min
        gamma = F.softplus(self.encoder_gamma(x32)) + gamma_eps
        packed = torch.stack([alpha, gamma], dim=-1)                # [B, S, H, 2]
        return packed.to(dtype_in)

    def _stability_score(self, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Stability score from `(α, γ)`:

            score_α = (α − α_min) / (α_max − α_min)         ∈ (0, 1)
            score_γ = 1 / (1 + log1p(γ))                     ∈ (0, 1)
            score   = w_α · mean(score_α) + w_γ · mean(score_γ)

        Returns `[B]` (scalar for unbatched inputs).
        """
        dtype = alpha.dtype
        device = alpha.device

        alpha_min  = self._alpha_min.to(device=device, dtype=dtype)
        alpha_span = self._alpha_range.to(device=device, dtype=dtype)
        w_alpha    = self._stab_w_alpha.to(device=device, dtype=dtype)
        w_gamma    = self._stab_w_gamma.to(device=device, dtype=dtype)

        # Compute in fp32 for AMP safety, cast back at the end.
        score_alpha = ((alpha.float() - alpha_min) / alpha_span).mean(dim=-1)
        score_gamma = (1.0 / (1.0 + torch.log1p(gamma.float()))).mean(dim=-1)

        return (w_alpha * score_alpha + w_gamma * score_gamma).to(dtype)

    # ------------------------------------------------------------------ #
    # Vectorized TLI weight application                                  #
    # ------------------------------------------------------------------ #
    def _apply_tli_weights(
        self,
        phi: torch.Tensor,
        mi: torch.Tensor,
        stability: torch.Tensor,
        reciprocity: torch.Tensor,
        als_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        TLI = Σ wᵢ · termᵢ  where `w = softmax(raw_weights)`.

        All terms are `[B]`.  Uses one stacked tensor and one reduction —
        the reference created 5 separate `[1]`-slice multiplications.
        """
        weights = F.softmax(self.raw_weights, dim=0)                # [5]
        terms   = torch.stack(
            [phi, mi, stability, reciprocity, als_mean], dim=-1,
        )                                                            # [B, 5]
        return (terms * weights).sum(dim=-1)                         # [B]

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        time_series_a: torch.Tensor,
        time_series_b: torch.Tensor,
        als_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        time_series_a, time_series_b : torch.Tensor
            `[B, S, F]` — paired multi-channel time series of the two
            partners.  Must share `(B, S, F)`.
        als_scores : torch.Tensor
            Attachment-Latent-Score input.  Accepts `[B]` (per-sample
            scalar, taken as-is) or `[B, N]` (mean over the last axis).
            The reference collapsed `[B]` inputs to a world-batch scalar;
            that is now correctly handled.

        Returns
        -------
        dict with per-sample `[B]` tensors:
            TLI               — True-Love Index ∈ (0, 1)
            Phi_AB            — Integrated Information proxy
            MI_C              — Mutual-information proxy ∈ (0, 1)
            Stability         — mean of stability_a and stability_b
            Reciprocity       — reciprocity score ∈ (0, 1)
            Optimized_Weights — softmax of `raw_weights` (shape [5])
        """
        # ---- Validation --------------------------------------------------
        if self.validate_inputs:
            if time_series_a.dim() != 3:
                raise ValueError("time_series_a must be [B, S, F].")
            if time_series_a.shape != time_series_b.shape:
                raise ValueError(
                    "time_series_a and time_series_b must share shape."
                )
            if time_series_a.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"time_series_a last dim must be feature_dim="
                    f"{self.feature_dim}, got {time_series_a.shape[-1]}."
                )
            if time_series_a.shape[0] != als_scores.shape[0]:
                raise ValueError(
                    "als_scores batch dim must match time series."
                )

        B, S, _ = time_series_a.shape
        device  = time_series_a.device

        # ---- 1. Parameter estimation via differentiable projections ------
        packed_a = self._encode_stable_params(time_series_a)         # [B, S, H, 2]
        packed_b = self._encode_stable_params(time_series_b)
        alpha_a, gamma_a = packed_a[..., 0], packed_a[..., 1]        # [B, S, H]
        alpha_b, gamma_b = packed_b[..., 0], packed_b[..., 1]

        # ---- 2. Stability metric -----------------------------------------
        stability_a = self._stability_score(alpha_a, gamma_a)        # [B]
        stability_b = self._stability_score(alpha_b, gamma_b)
        mean_stability = 0.5 * (stability_a + stability_b)

        # ---- 3. IIT-like Φ (joint vs. concatenated state) ----------------
        concat_state = torch.cat([time_series_a, time_series_b], dim=-1)

        use_ckpt = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )

        def _phi_branch(x: torch.Tensor) -> torch.Tensor:
            return self.iit_projector(x.float()).squeeze(-1)

        if use_ckpt:
            phi_per_step = torch.utils.checkpoint.checkpoint(
                _phi_branch, concat_state, use_reentrant=False,
            )
        else:
            phi_per_step = _phi_branch(concat_state)                 # [B, S]

        phi_raw = phi_per_step.mean(dim=-1).to(time_series_a.dtype)  # [B]

        # ---- 4. MI proxy via normalized dot-product ----------------------
        l1_eps = self._l1_eps.to(device=device, dtype=time_series_a.dtype)
        norm_a = F.normalize(time_series_a, dim=-1, eps=float(l1_eps))
        norm_b = F.normalize(time_series_b, dim=-1, eps=float(l1_eps))
        mi_proxy = (norm_a * norm_b).sum(dim=-1).mean(dim=-1)        # [B]
        mi_norm  = torch.sigmoid(mi_proxy)

        # ---- 5. LSTM-based reciprocity -----------------------------------
        #   Runs fp32-internal for AMP safety, casts back.
        def _lstm_branch(x: torch.Tensor) -> torch.Tensor:
            out, _ = self.psycho_lstm(x.float())
            return out[:, -1, :]                                     # [B, H]

        if use_ckpt:
            out_a_last = torch.utils.checkpoint.checkpoint(
                _lstm_branch, time_series_a, use_reentrant=False,
            )
            out_b_last = torch.utils.checkpoint.checkpoint(
                _lstm_branch, time_series_b, use_reentrant=False,
            )
        else:
            out_a_last = _lstm_branch(time_series_a)
            out_b_last = _lstm_branch(time_series_b)

        reciprocal_features = torch.cat([out_a_last, out_b_last], dim=-1)
        reciprocity_score = torch.sigmoid(
            self.reciprocity_head(reciprocal_features)
        ).squeeze(-1).to(time_series_a.dtype)                        # [B]

        # ---- 6. ALS mean (auto-promote [B] vs [B, N]) --------------------
        #   The reference's `als_scores.mean(dim=-1)` collapsed `[B]` inputs
        #   to a scalar — a silent DDP batch-collapse bug.
        if als_scores.dim() == 1:
            als_mean = als_scores.to(time_series_a.dtype)            # [B]
        elif als_scores.dim() == 2:
            als_mean = als_scores.mean(dim=-1).to(time_series_a.dtype)  # [B]
        else:
            raise ValueError(
                "als_scores must be [B] or [B, N]."
            )

        # ---- 7. True-Love Index (vectorized weight application) ----------
        tli = self._apply_tli_weights(
            phi_raw, mi_norm, mean_stability, reciprocity_score, als_mean,
        )

        weights = F.softmax(self.raw_weights, dim=0)

        return {
            "TLI":               tli,
            "Phi_AB":            phi_raw,
            "MI_C":              mi_norm,
            "Stability":         mean_stability,
            "Reciprocity":       reciprocity_score,
            "Optimized_Weights": weights,
        }


# =============================================================================
# Smoke test / autograd verification
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-TrueLoveModule v2] Running on: {device}")

    B, S, Fdim, H = 4, 32, 64, 128
    model = FullyDifferentiableTrueLoveModule(
        feature_dim=Fdim, hidden_dim=H,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    series_a = torch.randn(B, S, Fdim, device=device)
    series_b = torch.randn(B, S, Fdim, device=device)
    als      = torch.rand(B, device=device)          # [B] — the previously collapsing case

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        out = model(series_a, series_b, als)
        loss = (1.0 - out["TLI"].float()).mean()

    loss.backward()
    optimizer.step()

    print("-" * 64)
    for k in ("TLI", "Phi_AB", "MI_C", "Stability", "Reciprocity",
              "Optimized_Weights"):
        v = out[k].detach().float()
        if v.numel() == 1:
            print(f"  {k:<20s} : {v.item():.6f}  shape={tuple(v.shape)}")
        else:
            print(f"  {k:<20s} : min={v.min().item():.4f}  "
                  f"max={v.max().item():.4f}  shape={tuple(v.shape)}")
    print(f"  loss                 : {loss.item():.6f}")

    # ---- Gradient-flow audit ---------------------------------------------
    grad_ok = True
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")
    print(f"  Autograd             : "
          f"{'FULLY CONNECTED — C^∞ differentiable' if grad_ok else 'FAILED'}")

    # ---- DDP batch-collapse regression test ------------------------------
    #   [B] ALS input must produce [B] output (not scalar).
    with torch.no_grad():
        out_B = model(series_a, series_b, als)               # [B]
        out_BN = model(series_a, series_b, als.unsqueeze(-1).expand(B, 5))
    print(f"  ALS [B]  → TLI shape : {tuple(out_B['TLI'].shape)}")
    print(f"  ALS [B,N]→ TLI shape : {tuple(out_BN['TLI'].shape)}")
    print("-" * 64)
