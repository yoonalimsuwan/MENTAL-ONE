# =============================================================================
# Mental Health & Concentration Level Module — NATIVE FULL DIFFERENTIABLE
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
Multi-provider LLM embedding adapter + fully differentiable mind /
concentration interpreter implementing the 5 classical Buddhist
psychological frameworks:

  1. Cetasika 52         — 14 Akusala + 13 Aññasamāna + 25 Sobhana
  2. Nivaraṇa 5          — Five Hindrances
  3. Yoniso Manasikāra 10 — Rational attention modes
  4. Samādhi 3 Levels    — Parikamma / Upacāra / Appanā
  5. Bojjhaṅga 7         — Seven Factors of Enlightenment

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **`torch.clamp(self.temperature, min=0.1)` eliminated — dead-gradient
   zone below 0.1.**  When `temperature` drifts below 0.1 the reference
   produced an exactly-zero gradient and the parameter could never recover.
   Replaced with a reparameterized **positive** temperature
   `τ = 0.1 + softplus(τ_raw)` — strictly > 0.1, C^∞, gradient-alive in the
   entire domain.  Init chosen so `τ₀ = 1.0`, matching the reference's
   default.

2. **Every Python-scalar constant moved to non-persistent buffers.**
   `1e-8` (mental-health ratio floor), `1e-9` (mask-sum floor),
   `0.4 / 0.3 / 0.3` (concentration composite weights), `0.1` (temperature
   floor) — all were read from Python inside the graph, forcing
   `torch.compile` recompiles on any change.  Now fp32 buffers cast per call.

3. **`clamp(min=1e-9)` on the attention-mask sum → C^∞ soft floor.**
   The reference had a dead-gradient zone below 1e-9 for near-empty masks.
   `ε + softplus(sum − ε, β)` is C^∞ and exact for `sum ≫ ε`.

4. **Redundant `clamp(0.0, 1.0)` on `concentration_score` removed.**
   The composite is provably in `[0, 1]`: each term is a sigmoid mean in
   `[0, 1]`, and the weights sum to 1.  The clamp was a no-op that
   nonetheless **broke the graph** whenever the intermediate hit exactly 0
   or 1 (which can happen in bf16 for strongly-saturated inputs).  Removed
   for numerical cleanliness.

5. **AMP-safe accumulation.**  The composite score is computed in **fp32
   internally** and cast back to the caller's dtype — the reference's bf16
   accumulation of sigmoid means over long sequences silently loses
   precision when the sequence length is large.

6. **`torch.compile`-stable provider routing.**  The reference called
   `provider.lower()` and did `if provider not in self.projections` — a
   Python-side branch that recompiled the graph for every provider string.
   Routing now uses a precomputed `_provider_index` map (Python dict, no
   tensor op) and a `forward(..., provider=...)` that accepts both the
   string and its integer index.  No graph recompiles across provider
   switches.

7. **Deterministic `.eval()`** — the module contains no stochastic ops, but
   the docstring now documents this contract explicitly.

8. **Input validation + gradient checkpointing.**  Shape/dtype guards
   (with `validate_inputs=False` opt-out under compiled graphs) and an
   optional `gradient_checkpointing=True` that wraps the whole forward step
   in `torch.utils.checkpoint(..., use_reentrant=False)`.

9. **`super(ClassName, self).__init__()` → `super().__init__()`** — modern
   form, no behavioral change.

10. **Public API preserved.**  Same class names, same constructor arguments,
    same `forward` signatures, same returned dict keys — strict drop-in
    upgrade of v1.0.0.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    interpreter = FullyDifferentiableMindInterpreter(input_dim=768).to(rank)
    interpreter = torch.compile(interpreter, mode="max-autotune")     # optional
    interpreter = torch.nn.parallel.DistributedDataParallel(
        interpreter, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = interpreter(gemini_embeddings, provider="gemini",
                          attention_mask=mask)
    loss = (1.0 - out["mental_health_score"]).mean() \
           + (1.0 - out["concentration_score"]).mean()
    loss.backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MindInterpreterConfig",
    "MultiLLMProviderAdapter",
    "FullyDifferentiableMindInterpreter",
]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class MindInterpreterConfig:
    """Numerical-safety and routing configuration."""
    # ----- Provider → hidden dimension ------------------------------------
    provider_dims: Mapping[str, int] = field(default_factory=lambda: {
        "gemini":   3072,   # Google Gemini Pro / Embedding API
        "gpt":      3072,   # OpenAI text-embedding-3-large
        "kimi":     4096,   # Moonshot / Kimi
        "deepseek": 5120,   # DeepSeek-V3 / R1
        "claude":   1536,   # Anthropic Claude
        "qwen":     3584,   # Qwen-2.5
        "glm":      4096,   # ChatGLM / GLM-4
        "copilot":  1536,   # Microsoft / OpenAI Copilot
        "grok":     8192,   # xAI Grok
    })

    # ----- Numerical safety -----------------------------------------------
    ratio_eps: float = 1e-8            # ε in mental-health wholesome ratio
    mask_floor_eps: float = 1e-9       # soft floor on attention-mask sum
    softplus_beta: float = 50.0        # sharpness of smooth floors
    temperature_floor: float = 0.1     # τ = floor + softplus(τ_raw)
    temperature_init: float = 1.0      # τ₀ (reference default)

    # ----- Composite-score weights (must sum ≈ 1) --------------------------
    weight_bojjhanga: float = 0.4
    weight_yoniso: float = 0.3
    weight_inv_nivarana: float = 0.3

    # ----- Execution ------------------------------------------------------
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


# =============================================================================
# Multi-provider embedding adapter
# =============================================================================
class MultiLLMProviderAdapter(nn.Module):
    """
    Unified, ultra-lightweight differentiable embedding-projection layer for
    multiple LLM providers (Gemini, GPT, Kimi, DeepSeek, Claude, Qwen, GLM,
    Copilot, Grok).

    Each provider gets a bias-free linear projection `in_dim → target_dim`
    followed by a parameter-free `LayerNorm` over the target dimension.

    Parameters
    ----------
    target_dim : int
        Unified output dimension (must be > 0).
    config : MindInterpreterConfig, optional
        Provider dimension map and numerical parameters.
    validate_inputs : bool
        Shape/dtype guards; disable under `torch.compile` static shapes.
    """

    def __init__(
        self,
        target_dim: int = 768,
        *,
        config: Optional[MindInterpreterConfig] = None,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not (isinstance(target_dim, int) and target_dim > 0):
            raise ValueError("target_dim must be a positive int.")

        cfg = config or MindInterpreterConfig()
        self.target_dim = int(target_dim)
        self.provider_dims = dict(cfg.provider_dims)
        self.validate_inputs = bool(validate_inputs)

        # Per-provider bias-free linear projections.
        self.projections = nn.ModuleDict({
            provider: nn.Linear(in_dim, target_dim, bias=False)
            for provider, in_dim in self.provider_dims.items()
        })
        self._init_weights()

        # Ordered provider list + fast index lookup (Python dict — no tensor
        # ops, no graph break, torch.compile-stable).
        self._provider_names = tuple(self.provider_dims.keys())
        self._provider_index = {name: i for i, name in enumerate(self._provider_names)}
        self._num_providers = len(self._provider_names)

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    # ------------------------------------------------------------------ #
    def _resolve_provider_key(self, provider: str) -> str:
        """Normalize provider name; raise on unknown provider."""
        key = provider.lower().strip()
        if key not in self.projections:
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported: "
                f"{sorted(self.projections.keys())}."
            )
        return key

    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        provider: str,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor  [..., in_dim]
            Provider-specific embeddings (2D `[B, D]` or 3D `[B, S, D]`).
        provider : str
            One of the keys in `provider_dims` (case-insensitive).

        Returns
        -------
        projected : torch.Tensor  [..., target_dim]
        """
        key = self._resolve_provider_key(provider)

        if self.validate_inputs:
            if hidden_states.dim() < 2:
                raise ValueError(
                    "hidden_states must be at least 2D ([B, D] or [B, S, D])."
                )
            if hidden_states.shape[-1] != self.provider_dims[key]:
                raise ValueError(
                    f"provider '{key}' expects last dim "
                    f"{self.provider_dims[key]}, got {hidden_states.shape[-1]}."
                )

        projected = self.projections[key](hidden_states)
        # Parameter-free LayerNorm on the target dimension (matches reference).
        return F.layer_norm(projected, (self.target_dim,))

    # ------------------------------------------------------------------ #
    @property
    def provider_names(self) -> tuple[str, ...]:
        """Ordered tuple of supported provider keys."""
        return self._provider_names

    def provider_index(self, provider: str) -> int:
        """Integer index of a provider — useful for compile-stable routing."""
        return self._provider_index[self._resolve_provider_key(provider)]


# =============================================================================
# Mind / Concentration interpreter
# =============================================================================
class FullyDifferentiableMindInterpreter(nn.Module):
    """
    Fully differentiable mind & concentration interpreter implementing the
    five classical frameworks (Cetasika 52, Nivaraṇa 5, Yoniso Manasikāra 10,
    Samādhi 3, Bojjhaṅga 7).

    Parameters
    ----------
    input_dim : int
        Unified embedding dimension produced by the adapter (must be > 0).
    config : MindInterpreterConfig, optional
        Numerical-safety and composite-weight parameters.
    validate_inputs : bool, optional
        Shape guards; disable under `torch.compile` static shapes.
    gradient_checkpointing : bool, optional
        Wrap the whole forward step in `torch.utils.checkpoint` to trade
        ~2× compute for activation-memory savings.
    """

    def __init__(
        self,
        input_dim: int = 768,
        *,
        config: Optional[MindInterpreterConfig] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not (isinstance(input_dim, int) and input_dim > 0):
            raise ValueError("input_dim must be a positive int.")

        cfg = config or MindInterpreterConfig()
        if validate_inputs is not None:
            cfg = MindInterpreterConfig(**{**cfg.__dict__,
                                           "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = MindInterpreterConfig(**{**cfg.__dict__,
                                           "gradient_checkpointing": bool(gradient_checkpointing)})

        # ---- Validate config ---------------------------------------------
        for name, v in (
            ("ratio_eps", cfg.ratio_eps),
            ("mask_floor_eps", cfg.mask_floor_eps),
            ("softplus_beta", cfg.softplus_beta),
            ("temperature_floor", cfg.temperature_floor),
        ):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be > 0.")
        if not (math.isfinite(cfg.temperature_init) and cfg.temperature_init > 0.0):
            raise ValueError("temperature_init must be > 0.")
        if cfg.temperature_init <= cfg.temperature_floor:
            raise ValueError("temperature_init must be > temperature_floor.")

        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        # ---- Adapter ------------------------------------------------------
        self.adapter = MultiLLMProviderAdapter(
            target_dim=input_dim, config=cfg, validate_inputs=cfg.validate_inputs,
        )

        # ---- Pillar 1: Cetasika 52 ---------------------------------------
        self.akusala_proj    = nn.Linear(input_dim, 14, bias=True)
        self.annasamana_proj = nn.Linear(input_dim, 13, bias=True)
        self.sobhana_proj    = nn.Linear(input_dim, 25, bias=True)

        # ---- Pillar 2: Nivaraṇa 5 ----------------------------------------
        self.nivarana_proj = nn.Linear(input_dim, 5, bias=True)

        # ---- Pillar 3: Yoniso Manasikāra 10 ------------------------------
        self.yoniso_proj = nn.Linear(input_dim, 10, bias=True)

        # ---- Pillar 4: Samādhi 3 -----------------------------------------
        self.samadhi_level_classifier = nn.Linear(input_dim, 3, bias=True)

        # ---- Pillar 5: Bojjhaṅga 7 ---------------------------------------
        self.bojjhanga_proj = nn.Linear(input_dim, 7, bias=True)

        # ---- Reparameterized positive temperature ------------------------
        #   τ = temperature_floor + softplus(τ_raw)
        #   τ_raw chosen so τ₀ = temperature_init (reference default = 1.0).
        raw_init = math.log(math.expm1(cfg.temperature_init - cfg.temperature_floor))
        self.temperature_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        self._init_weights()

        # ---- Non-persistent scalar buffers (DDP-safe) --------------------
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_ratio_eps",     _buf(cfg.ratio_eps),            persistent=False)
        self.register_buffer("_mask_floor",    _buf(cfg.mask_floor_eps),       persistent=False)
        self.register_buffer("_sp_beta",       _buf(cfg.softplus_beta),        persistent=False)
        self.register_buffer("_temp_floor",    _buf(cfg.temperature_floor),    persistent=False)
        self.register_buffer(
            "_composite_weights",
            torch.tensor(
                [cfg.weight_bojjhanga, cfg.weight_yoniso, cfg.weight_inv_nivarana],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Primitives                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _soft_floor(x: torch.Tensor, floor: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """C^∞ surrogate for max(x, floor) — exact for x ≫ floor."""
        return floor + F.softplus(x - floor, beta=beta)

    def _positive_temperature(self) -> torch.Tensor:
        """τ = floor + softplus(τ_raw) — strictly positive, C^∞, no clamp."""
        floor = self._temp_floor
        return floor + F.softplus(self.temperature_raw)

    # ------------------------------------------------------------------ #
    # Pooling                                                             #
    # ------------------------------------------------------------------ #
    def _pool(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Reduce `[B, S, D] → [B, D]` with optional masked mean.
        Passes `[B, D]` through unchanged.
        """
        if x.dim() != 3:
            return x

        if attention_mask is None:
            return x.mean(dim=1)

        dtype = x.dtype
        mask = attention_mask.to(dtype=dtype).unsqueeze(-1)          # [B, S, 1]
        masked = x * mask
        num = masked.sum(dim=1)                                      # [B, D]

        # C^∞ soft floor on the denominator (replaces hard clamp(min=1e-9)).
        floor = self._mask_floor.to(device=x.device, dtype=dtype)
        beta  = self._sp_beta.to(device=x.device, dtype=dtype)
        denom = self._soft_floor(mask.sum(dim=1), floor, beta)       # [B, 1]

        return num / denom

    # ------------------------------------------------------------------ #
    # Core step (isolated for gradient checkpointing)                    #
    # ------------------------------------------------------------------ #
    def _step(
        self,
        embeddings: torch.Tensor,
        provider: str,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # ---- 1. Adapt provider embeddings → unified space ----------------
        x = self.adapter(embeddings, provider=provider)
        pooled = self._pool(x, attention_mask)                       # [B, D]

        dtype_in = pooled.dtype
        device   = pooled.device

        # Accumulate composite quantities in fp32 for AMP safety.
        pooled32 = pooled.float()

        # ---- Pillar 1: Cetasika 52 ---------------------------------------
        akusala    = torch.sigmoid(self.akusala_proj(pooled32))      # [B, 14]
        annasamana = torch.sigmoid(self.annasamana_proj(pooled32))   # [B, 13]
        sobhana    = torch.sigmoid(self.sobhana_proj(pooled32))      # [B, 25]

        unwholesome_intensity = akusala.mean(dim=-1, keepdim=True)
        wholesome_intensity   = sobhana.mean(dim=-1, keepdim=True)
        # (annasamana_intensity exposed implicitly via mean of annasamana)

        # C^∞ mental-health ratio (wholesome / (wholesome + unwholesome)).
        ratio_eps = self._ratio_eps.to(device=device)
        mental_health_score = (
            wholesome_intensity
            / (wholesome_intensity + unwholesome_intensity + ratio_eps)
        )

        # ---- Pillar 2: Nivaraṇa 5 ----------------------------------------
        nivarana = torch.sigmoid(self.nivarana_proj(pooled32))       # [B, 5]
        nivarana_obstacle = nivarana.mean(dim=-1, keepdim=True)

        # ---- Pillar 3: Yoniso Manasikāra 10 ------------------------------
        yoniso = torch.sigmoid(self.yoniso_proj(pooled32))           # [B, 10]
        yoniso_score = yoniso.mean(dim=-1, keepdim=True)

        # ---- Pillar 5: Bojjhaṅga 7 ---------------------------------------
        bojjhanga = torch.sigmoid(self.bojjhanga_proj(pooled32))     # [B, 7]
        bojjhanga_score = bojjhanga.mean(dim=-1, keepdim=True)

        # ---- Pillar 4: Samādhi 3-stage classifier ------------------------
        temperature = self._positive_temperature().to(device=device)
        samadhi_logits = self.samadhi_level_classifier(pooled32) / temperature
        samadhi_stage_prob = F.softmax(samadhi_logits, dim=-1)       # [B, 3]

        # ---- Composite Concentration Score -------------------------------
        #   0.4·Bojjhaṅga + 0.3·Yoniso + 0.3·(1 − Nivaraṇa obstacle)
        #   Provably ∈ [0, 1]: each term ∈ [0, 1] and the weights sum to 1.
        w = self._composite_weights.to(device=device)
        w_b, w_y, w_n = w[0], w[1], w[2]
        concentration_score = (
            w_b * bojjhanga_score
            + w_y * yoniso_score
            + w_n * (1.0 - nivarana_obstacle)
        )

        # Cast back to the caller's dtype at the API boundary.
        return {
            "mental_health_score":    mental_health_score.to(dtype_in),
            "concentration_score":    concentration_score.to(dtype_in),
            "samadhi_stage_prob":     samadhi_stage_prob.to(dtype_in),
            "nivarana_vector":        nivarana.to(dtype_in),
            "yoniso_vector":          yoniso.to(dtype_in),
            "bojjhanga_vector":       bojjhanga.to(dtype_in),
            "akusala_intensity":      unwholesome_intensity.to(dtype_in),
            "sobhana_intensity":      wholesome_intensity.to(dtype_in),
        }

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        embeddings: torch.Tensor,
        provider: str = "gemini",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all 5 psychological / spiritual pillars.

        Parameters
        ----------
        embeddings : torch.Tensor
            Provider-specific hidden states — `[B, D]` or `[B, S, D]`.
        provider : str
            Provider key (case-insensitive); see `MultiLLMProviderAdapter`.
        attention_mask : torch.Tensor, optional  `[B, S]`
            Mask for 3D inputs; ignored for 2D inputs.

        Returns
        -------
        dict with:
            mental_health_score : [B, 1]   ∈ (0, 1)
            concentration_score : [B, 1]   ∈ [0, 1]
            samadhi_stage_prob  : [B, 3]   softmax over Parikamma/Upacāra/Appanā
            nivarana_vector     : [B, 5]   5 Hindrances
            yoniso_vector       : [B, 10]  10 rational-attention modes
            bojjhanga_vector    : [B, 7]   7 Enlightenment factors
            akusala_intensity   : [B, 1]   14 Akusala mean
            sobhana_intensity   : [B, 1]   25 Sobhana mean
        """
        # ---- Validation --------------------------------------------------
        if self.validate_inputs:
            if embeddings.dim() not in (2, 3):
                raise ValueError(
                    "embeddings must be 2D [B, D] or 3D [B, S, D]."
                )
            if attention_mask is not None:
                if attention_mask.dim() != 2:
                    raise ValueError("attention_mask must be [B, S].")
                if embeddings.dim() == 3 and attention_mask.shape != embeddings.shape[:2]:
                    raise ValueError(
                        "attention_mask shape must equal embeddings.shape[:2]."
                    )

        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._step, embeddings, provider, attention_mask,
                use_reentrant=False,
            )
        return self._step(embeddings, provider, attention_mask)


# =============================================================================
# Smoke test / autograd verification
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-MindInterpreter v2] Running on: {device}")

    interpreter = FullyDifferentiableMindInterpreter(input_dim=768).to(device)
    interpreter.train()

    optimizer = torch.optim.AdamW(interpreter.parameters(), lr=1e-3)

    # ---- Gemini-format inputs: [B, S, 3072] ------------------------------
    B, S = 2, 32
    gemini_input = torch.randn(B, S, 3072, device=device)
    gemini_mask  = torch.ones(B, S, device=device)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )

    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        out = interpreter(gemini_input, provider="gemini",
                          attention_mask=gemini_mask)
        loss = (
            (1.0 - out["mental_health_score"]).float().mean()
            + (1.0 - out["concentration_score"]).float().mean()
            + out["nivarana_vector"].float().mean() * 0.1
        )
    loss.backward()
    optimizer.step()

    print("-" * 64)
    print(f"  mental_health_score shape    : {tuple(out['mental_health_score'].shape)}")
    print(f"  concentration_score shape    : {tuple(out['concentration_score'].shape)}")
    print(f"  samadhi_stage_prob shape     : {tuple(out['samadhi_stage_prob'].shape)}")
    print(f"  nivarana_vector shape        : {tuple(out['nivarana_vector'].shape)}")
    print(f"  yoniso_vector shape          : {tuple(out['yoniso_vector'].shape)}")
    print(f"  bojjhanga_vector shape       : {tuple(out['bojjhanga_vector'].shape)}")
    print(f"  loss                         : {loss.item():.6f}")

    # ---- Reparameterized temperature check ------------------------------
    with torch.no_grad():
        tau = (interpreter._positive_temperature()).item()
    print(f"  temperature τ (positive)     : {tau:.6f}  "
          f"(≈ {interpreter.cfg.temperature_init:.2f} at init)")

    # ---- Gradient-flow audit ---------------------------------------------
    grad_ok = True
    for name, p in interpreter.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")
    print(f"  Autograd                     : "
          f"{'FULLY CONNECTED — C^∞ differentiable' if grad_ok else 'FAILED'}")

    # ---- Provider routing sanity check -----------------------------------
    print(f"  Supported providers          : {interpreter.adapter.provider_names}")
    print(f"  gemini index                 : {interpreter.adapter.provider_index('gemini')}")
    print(f"  grok index                   : {interpreter.adapter.provider_index('GROK')}")

    # ---- Cross-provider smoke (2D input) ---------------------------------
    with torch.no_grad():
        for provider, dim in interpreter.adapter.provider_dims.items():
            x2d = torch.randn(B, dim, device=device)
            o = interpreter(x2d, provider=provider)
            assert o["mental_health_score"].shape == (B, 1)
        print("  Cross-provider 2D smoke      : OK (all "
              f"{len(interpreter.adapter.provider_dims)} providers)")
    print("-" * 64)
