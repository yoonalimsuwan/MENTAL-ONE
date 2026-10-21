"""
=============================================================================
Mental Health & Concentration Level Module — NATIVE FULL DIFFERENTIABLE
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT
Year         : 2026
Version      : 2.0.1 (Bug-Fixed: reparameterized temperature, AMP-safe)
=============================================================================
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


@dataclass
class MindInterpreterConfig:
    provider_dims: Mapping[str, int] = field(default_factory=lambda: {
        "gemini": 3072, "gpt": 3072, "kimi": 4096, "deepseek": 5120,
        "claude": 1536, "qwen": 3584, "glm": 4096, "copilot": 1536, "grok": 8192,
    })
    ratio_eps: float = 1e-8
    mask_floor_eps: float = 1e-9
    softplus_beta: float = 50.0
    temperature_floor: float = 0.1
    temperature_init: float = 1.0
    weight_bojjhanga: float = 0.4
    weight_yoniso: float = 0.3
    weight_inv_nivarana: float = 0.3
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


class MultiLLMProviderAdapter(nn.Module):
    def __init__(self, target_dim: int = 768, *, config: Optional[MindInterpreterConfig] = None,
                 validate_inputs: bool = True) -> None:
        super().__init__()
        if not (isinstance(target_dim, int) and target_dim > 0):
            raise ValueError("target_dim must be a positive int.")
        cfg = config or MindInterpreterConfig()
        self.target_dim = int(target_dim)
        self.provider_dims = dict(cfg.provider_dims)
        self.validate_inputs = bool(validate_inputs)

        self.projections = nn.ModuleDict({
            provider: nn.Linear(in_dim, target_dim, bias=False)
            for provider, in_dim in self.provider_dims.items()
        })
        self._init_weights()
        self._provider_names = tuple(self.provider_dims.keys())
        self._provider_index = {name: i for i, name in enumerate(self._provider_names)}

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def _resolve_provider_key(self, provider: str) -> str:
        key = provider.lower().strip()
        if key not in self.projections:
            raise ValueError(f"Unsupported provider '{provider}'.")
        return key

    def forward(self, hidden_states: torch.Tensor, provider: str) -> torch.Tensor:
        key = self._resolve_provider_key(provider)
        projected = self.projections[key](hidden_states)
        return F.layer_norm(projected, (self.target_dim,))

    @property
    def provider_names(self):
        return self._provider_names


class FullyDifferentiableMindInterpreter(nn.Module):
    def __init__(self, input_dim: int = 768, *, config: Optional[MindInterpreterConfig] = None,
                 validate_inputs: Optional[bool] = None, gradient_checkpointing: Optional[bool] = None) -> None:
        super().__init__()
        cfg = config or MindInterpreterConfig()
        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs

        self.adapter = MultiLLMProviderAdapter(target_dim=input_dim, config=cfg)
        self.akusala_proj = nn.Linear(input_dim, 14, bias=True)
        self.annasamana_proj = nn.Linear(input_dim, 13, bias=True)
        self.sobhana_proj = nn.Linear(input_dim, 25, bias=True)
        self.nivarana_proj = nn.Linear(input_dim, 5, bias=True)
        self.yoniso_proj = nn.Linear(input_dim, 10, bias=True)
        self.samadhi_level_classifier = nn.Linear(input_dim, 3, bias=True)
        self.bojjhanga_proj = nn.Linear(input_dim, 7, bias=True)

        raw_init = math.log(math.expm1(cfg.temperature_init - cfg.temperature_floor))
        self.temperature_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self._init_weights()

        self.register_buffer("_ratio_eps", torch.tensor(cfg.ratio_eps), persistent=False)
        self.register_buffer("_temp_floor", torch.tensor(cfg.temperature_floor), persistent=False)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _positive_temperature(self) -> torch.Tensor:
        return self._temp_floor + F.softplus(self.temperature_raw)

    def forward(self, embeddings: torch.Tensor, provider: str = "gemini",
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.adapter(embeddings, provider=provider)
        if x.dim() == 3:
            if attention_mask is not None:
                mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)
        else:
            pooled = x

        pooled32 = pooled.float()
        akusala = torch.sigmoid(self.akusala_proj(pooled32))
        sobhana = torch.sigmoid(self.sobhana_proj(pooled32))
        unwholesome_intensity = akusala.mean(dim=-1, keepdim=True)
        wholesome_intensity = sobhana.mean(dim=-1, keepdim=True)
        mental_health_score = wholesome_intensity / (wholesome_intensity + unwholesome_intensity + self._ratio_eps)

        nivarana = torch.sigmoid(self.nivarana_proj(pooled32))
        nivarana_obstacle = nivarana.mean(dim=-1, keepdim=True)
        yoniso = torch.sigmoid(self.yoniso_proj(pooled32))
        yoniso_score = yoniso.mean(dim=-1, keepdim=True)
        bojjhanga = torch.sigmoid(self.bojjhanga_proj(pooled32))
        bojjhanga_score = bojjhanga.mean(dim=-1, keepdim=True)

        temperature = self._positive_temperature()
        samadhi_logits = self.samadhi_level_classifier(pooled32) / temperature
        samadhi_stage_prob = F.softmax(samadhi_logits, dim=-1)

        concentration_score = (
            self.cfg.weight_bojjhanga * bojjhanga_score
            + self.cfg.weight_yoniso * yoniso_score
            + self.cfg.weight_inv_nivarana * (1.0 - nivarana_obstacle)
        )

        dtype_in = pooled.dtype
        return {
            "mental_health_score": mental_health_score.to(dtype_in),
            "concentration_score": concentration_score.to(dtype_in),
            "samadhi_stage_prob": samadhi_stage_prob.to(dtype_in),
            "nivarana_vector": nivarana.to(dtype_in),
            "yoniso_vector": yoniso.to(dtype_in),
            "bojjhanga_vector": bojjhanga.to(dtype_in),
            "akusala_intensity": unwholesome_intensity.to(dtype_in),
            "sobhana_intensity": wholesome_intensity.to(dtype_in),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-MindInterpreter v2] Running on: {device}")

    interpreter = FullyDifferentiableMindInterpreter(input_dim=768).to(device)
    interpreter.train()

    B, S = 2, 32
    gemini_input = torch.randn(B, S, 3072, device=device)
    out = interpreter(gemini_input, provider="gemini")

    loss = (1.0 - out["mental_health_score"]).mean() + (1.0 - out["concentration_score"]).mean()
    loss.backward()

    print(f"  mental_health_score: {out['mental_health_score'].mean().item():.4f}")
    print(f"  concentration_score: {out['concentration_score'].mean().item():.4f}")
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in interpreter.parameters())
    print(f"  Autograd: {'PASS' if grad_ok else 'FAIL'}")
