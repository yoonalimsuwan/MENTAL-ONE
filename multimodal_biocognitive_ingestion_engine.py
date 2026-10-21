"""
=============================================================================
Multimodal Bio-Cognitive Ingestion Engine — NATIVE FULL DIFFERENTIABLE
SUPER DNS ONE Cluster / ONE Ecosystem
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT
Year         : 2026
Version      : 2.0.1 (Bug-Fixed: tanh bound, positive reliability, AMP-safe)
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["IngestionConfig", "BioCognitiveIngestionEngine"]


@dataclass
class IngestionConfig:
    bio_input_dim: int = 5
    nlp_input_dim: int = 4
    dropout: float = 0.0
    reliability_eps: float = 1e-6
    reliability_init: float = 1.0
    validate_inputs: bool = True
    gradient_checkpointing: bool = False


class BioCognitiveIngestionEngine(nn.Module):
    def __init__(
        self,
        num_symptoms: int,
        latent_dim: int = 16,
        *,
        config: Optional[IngestionConfig] = None,
        dropout: Optional[float] = None,
        validate_inputs: Optional[bool] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if not (isinstance(num_symptoms, int) and num_symptoms > 0):
            raise ValueError("num_symptoms must be a positive int.")
        if not (isinstance(latent_dim, int) and latent_dim > 0):
            raise ValueError("latent_dim must be a positive int.")

        cfg = config or IngestionConfig()
        if dropout is not None:
            cfg = IngestionConfig(**{**cfg.__dict__, "dropout": float(dropout)})
        if validate_inputs is not None:
            cfg = IngestionConfig(**{**cfg.__dict__, "validate_inputs": bool(validate_inputs)})
        if gradient_checkpointing is not None:
            cfg = IngestionConfig(**{**cfg.__dict__, "gradient_checkpointing": bool(gradient_checkpointing)})

        self.num_nodes = int(num_symptoms)
        self.latent_dim = int(latent_dim)
        self.cfg = cfg
        self.validate_inputs = cfg.validate_inputs
        self.gradient_checkpointing = cfg.gradient_checkpointing

        self.bio_encoder = nn.Sequential(
            nn.Linear(cfg.bio_input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, num_symptoms),
        )
        self._init_weights(self.bio_encoder)

        self.nlp_encoder = nn.Sequential(
            nn.Linear(cfg.nlp_input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 3),
        )
        self._init_weights(self.nlp_encoder)

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        raw_init = math.log(math.expm1(max(cfg.reliability_init - cfg.reliability_eps, 1e-6)))
        self.source_reliability_raw = nn.Parameter(
            torch.full((2,), float(raw_init), dtype=torch.float32)
        )
        self.register_buffer("_rel_eps", torch.tensor(float(cfg.reliability_eps), dtype=torch.float32), persistent=False)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _reliability(self) -> torch.Tensor:
        return F.softplus(self.source_reliability_raw) + self._rel_eps

    def _bio_body(self, biometric_tensor: torch.Tensor) -> torch.Tensor:
        dtype_in = biometric_tensor.dtype
        h = self.bio_encoder(biometric_tensor.float())
        h = self.dropout(h)
        return torch.tanh(h).to(dtype_in)

    def _nlp_body(self, nlp_tensor: torch.Tensor) -> torch.Tensor:
        dtype_in = nlp_tensor.dtype
        h = self.nlp_encoder(nlp_tensor.float())
        h = self.dropout(h)
        return torch.tanh(h).to(dtype_in)

    def forward(self, biometric_tensor: torch.Tensor, nlp_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.validate_inputs:
            if biometric_tensor.dim() != 2 or nlp_tensor.dim() != 2:
                raise ValueError("Inputs must be 2D [B, D].")
            if biometric_tensor.shape[0] != nlp_tensor.shape[0]:
                raise ValueError("Batch sizes must match.")

        reliability = self._reliability()
        w_bio = reliability[0]
        w_nlp = reliability[1]

        autonomic = self._bio_body(biometric_tensor)
        cognitive = self._nlp_body(nlp_tensor)

        autonomic = autonomic * w_bio.to(dtype=autonomic.dtype, device=autonomic.device)
        cognitive = cognitive * w_nlp.to(dtype=cognitive.dtype, device=cognitive.device)
        return autonomic, cognitive


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[SESI-BioCognitiveIngestionEngine v2] Running on: {device}")

    engine = BioCognitiveIngestionEngine(num_symptoms=10, latent_dim=16).to(device)
    engine.train()

    B = 8
    bio_batch = torch.rand(B, 5, device=device) * 2.0 - 1.0
    nlp_batch = torch.rand(B, 4, device=device) * 2.0 - 1.0

    autonomic, cognitive = engine(bio_batch, nlp_batch)
    loss = autonomic.pow(2).mean() + cognitive.pow(2).mean()
    loss.backward()

    print(f"  autonomic shape: {tuple(autonomic.shape)}")
    print(f"  cognitive shape: {tuple(cognitive.shape)}")
    print(f"  loss: {loss.item():.6f}")
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in engine.parameters())
    print(f"  Autograd: {'PASS' if grad_ok else 'FAIL'}")
