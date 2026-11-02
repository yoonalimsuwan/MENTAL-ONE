"""
=============================================================================
SESI SYMBOLIC PROOF × GNN THEOREM PROVER × COUNTERFACTUAL REASONING
SymPy backend + differentiable GNN prover + SCM abduction/action/prediction
=============================================================================

Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT
Year         : 2026

Purpose
-------
Add three capabilities to the SESI / AGIONE / Yoniso-Dao science stack:

1. SymPySymbolicProver
   - Parse and verify symbolic identities.
   - Simplify / expand / factor expressions.
   - Search for numeric counterexamples.
   - Return honest proof status:
       VERIFIED_SYMBOLIC, DISPROVED_BY_COUNTEREXAMPLE, INCONCLUSIVE.

2. GNNTheoremProver
   - Graph neural network for theorem-state / proof-state scoring.
   - Predicts:
       proof_success_probability
       tactic logits
       premise usefulness scores
   - Pure PyTorch fallback; optional PyTorch Geometric can be added later.
   - AMP + DDP compatible.

3. DifferentiableCounterfactualSCM
   - Additive-noise Structural Causal Model.
   - Complete counterfactual pipeline:
       Abduction: infer exogenous noise from observed factual world.
       Action: apply do-intervention.
       Prediction: simulate counterfactual world.
   - Fully differentiable w.r.t. model parameters and continuous inputs.
   - Supports total-effect estimation.

Important
---------
This module does NOT claim:
  - machine consciousness
  - guaranteed theorem proving for arbitrary statements
  - scientific truth without human expert validation
  - medical diagnosis or treatment

All learned scores are heuristics unless backed by a formal checker.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("sesi.symbolic_gnn_counterfactual")

# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = Any

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import torch.distributed as dist
    HAS_DIST = HAS_TORCH and dist.is_available()
except ImportError:
    HAS_DIST = False


__all__ = [
    "ProofStatus",
    "ProofTier",
    "SymbolicProofResult",
    "SympySymbolicProver",
    "NeuralProofPrediction",
    "GNNTheoremProver",
    "CounterfactualResult",
    "DifferentiableCounterfactualSCM",
    "SimulationEvidence",
    "SimulationValidator",
    "ProofTrainingWrapper",
    "YonisoProofCounterfactualSuite",
]


# =============================================================================
# Enums / dataclasses
# =============================================================================

class ProofStatus(Enum):
    VERIFIED_SYMBOLIC = auto()
    DISPROVED_BY_COUNTEREXAMPLE = auto()
    INCONCLUSIVE = auto()
    PARSE_ERROR = auto()
    NOT_INSTALLED = auto()


class ProofTier(Enum):
    SPECULATIVE = 0
    SIMULATED = 1
    SYMBOLIC_CHECKED = 2
    FORMAL_PROOF = 3
    MACHINE_VERIFIED = 4


class CounterfactualStatus(Enum):
    SUCCESS = auto()
    INVALID_INTERVENTION = auto()
    MODEL_ERROR = auto()


@dataclass
class SymbolicProofResult:
    statement: str
    status: str
    tier: str
    confidence: float
    simplified_expression: Optional[str] = None
    counterexample: Optional[Dict[str, Any]] = None
    steps: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "status": self.status,
            "tier": self.tier,
            "confidence": self.confidence,
            "simplified_expression": self.simplified_expression,
            "counterexample": self.counterexample,
            "steps": self.steps,
            "assumptions": self.assumptions,
            "note": self.note,
        }


@dataclass
class NeuralProofPrediction:
    proof_probability: float
    tactic_logits: List[float]
    premise_scores: List[float]
    graph_embedding: List[float]
    note: str = "Neural proof scores are heuristic, not formal proof."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_probability": self.proof_probability,
            "tactic_logits": self.tactic_logits,
            "premise_scores": self.premise_scores,
            "graph_embedding": self.graph_embedding,
            "note": self.note,
        }


@dataclass
class CounterfactualResult:
    status: str
    factual: List[List[float]]
    counterfactual: List[List[float]]
    inferred_noise: List[List[float]]
    interventions: Dict[str, Any]
    effect: Optional[List[float]] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "factual": self.factual,
            "counterfactual": self.counterfactual,
            "inferred_noise": self.inferred_noise,
            "interventions": self.interventions,
            "effect": self.effect,
            "note": self.note,
        }


@dataclass
class SimulationEvidence:
    hypothesis: str
    n_trials: int
    success_rate: float
    mean_metric: float
    std_metric: float
    confidence: float
    passed: bool
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "n_trials": self.n_trials,
            "success_rate": self.success_rate,
            "mean_metric": self.mean_metric,
            "std_metric": self.std_metric,
            "confidence": self.confidence,
            "passed": self.passed,
            "examples": self.examples,
        }


# =============================================================================
# 1. SymPy symbolic proof backend
# =============================================================================

class SympySymbolicProver:
    """
    SymPy-backed symbolic checker.

    Can verify algebraic identities of the form:
        lhs = rhs

    It can also search for numeric counterexamples.

    Important:
    - Symbolic simplification is not equivalent to a full formal proof assistant.
    - If SymPy simplifies lhs-rhs to 0, we mark VERIFIED_SYMBOLIC.
    - If numeric counterexample is found, we mark DISPROVED_BY_COUNTEREXAMPLE.
    - Otherwise INCONCLUSIVE.
    """

    _FORBIDDEN = ["__", "import", "lambda", "exec", "eval", "open", ";", "class", "def"]

    def __init__(self, random_seed: int = 7):
        self.rng = random.Random(random_seed)
        self.symbols: Dict[str, Any] = {}

        if HAS_SYMPY:
            # Common symbols
            for name in list("abcdefghijklmnopqrstuvwxyz"):
                self.symbols[name] = sp.Symbol(name, real=True)
            for name in ["x", "y", "z", "a", "b", "c", "n", "m"]:
                self.symbols[name] = sp.Symbol(name, real=True)

    def _safe_expr(self, expr: str):
        if not HAS_SYMPY:
            raise RuntimeError("SymPy not installed")

        s = expr.strip()
        low = s.lower()
        for bad in self._FORBIDDEN:
            if bad in low:
                raise ValueError(f"Forbidden token in expression: {bad}")

        # Allow letters, digits, common math operators, parentheses, comma, dot, spaces
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s]+", s):
            raise ValueError("Expression contains unsupported characters")

        # Convert simple '=' to Eq if needed
        if "=" in s and "==" not in s:
            left, right = s.split("=", 1)
            return sp.Eq(self._safe_expr(left), self._safe_expr(right))

        local_dict = dict(self.symbols)
        local_dict.update({
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "Abs": sp.Abs,
            "pi": sp.pi,
            "E": sp.E,
            "Eq": sp.Eq,
        })

        return sp.sympify(s, locals=local_dict)

    def prove_identity(
        self,
        lhs: str,
        rhs: str,
        assumptions: Optional[List[str]] = None,
    ) -> SymbolicProofResult:
        statement = f"{lhs} = {rhs}"

        if not HAS_SYMPY:
            return SymbolicProofResult(
                statement=statement,
                status=ProofStatus.NOT_INSTALLED.name,
                tier=ProofTier.SPECULATIVE.name,
                confidence=0.0,
                note="Install SymPy with: pip install sympy",
            )

        try:
            L = self._safe_expr(lhs)
            R = self._safe_expr(rhs)
            diff = sp.simplify(L - R)

            steps = [
                f"Parse lhs: {L}",
                f"Parse rhs: {R}",
                f"Compute simplify(lhs - rhs): {diff}",
            ]

            if diff == 0:
                return SymbolicProofResult(
                    statement=statement,
                    status=ProofStatus.VERIFIED_SYMBOLIC.name,
                    tier=ProofTier.SYMBOLIC_CHECKED.name,
                    confidence=0.95,
                    simplified_expression=str(diff),
                    steps=steps,
                    assumptions=assumptions or [],
                    note="Verified by symbolic simplification. Not a full proof-assistant certificate.",
                )

            ce = self.find_counterexample(lhs, rhs)
            if ce is not None:
                return SymbolicProofResult(
                    statement=statement,
                    status=ProofStatus.DISPROVED_BY_COUNTEREXAMPLE.name,
                    tier=ProofTier.SIMULATED.name,
                    confidence=0.9,
                    simplified_expression=str(diff),
                    counterexample=ce,
                    steps=steps + ["Numeric counterexample found."],
                    assumptions=assumptions or [],
                    note="A counterexample disproves the universal identity over sampled domain.",
                )

            return SymbolicProofResult(
                statement=statement,
                status=ProofStatus.INCONCLUSIVE.name,
                tier=ProofTier.SPECULATIVE.name,
                confidence=0.4,
                simplified_expression=str(diff),
                steps=steps,
                assumptions=assumptions or [],
                note="SymPy could not verify or disprove. Human proof search required.",
            )

        except Exception as e:
            return SymbolicProofResult(
                statement=statement,
                status=ProofStatus.PARSE_ERROR.name,
                tier=ProofTier.SPECULATIVE.name,
                confidence=0.0,
                steps=[str(e)],
                note="Parse/verification error.",
            )

    def find_counterexample(
        self,
        lhs: str,
        rhs: str,
        n_trials: int = 100,
        value_range: Tuple[int, int] = (-10, 10),
    ) -> Optional[Dict[str, Any]]:
        if not HAS_SYMPY:
            return None

        L = self._safe_expr(lhs)
        R = self._safe_expr(rhs)
        expr = L - R
        free = sorted(list(expr.free_symbols), key=lambda s: s.name)

        if not free:
            val = complex(expr.evalf())
            if abs(val) > 1e-9:
                return {"values": {}, "lhs_minus_rhs": val}
            return None

        for _ in range(n_trials):
            subs = {
                s: self.rng.randint(value_range[0], value_range[1])
                for s in free
            }
            try:
                val = complex(expr.subs(subs).evalf())
                if abs(val) > 1e-8:
                    return {
                        "values": {str(k): float(v) for k, v in subs.items()},
                        "lhs_minus_rhs": val,
                    }
            except Exception:
                continue

        return None

    def derive_lemmas(self, expr: str) -> Dict[str, Any]:
        if not HAS_SYMPY:
            return {"status": "not_installed"}

        try:
            e = self._safe_expr(expr)
            return {
                "status": "ok",
                "original": str(e),
                "simplified": str(sp.simplify(e)),
                "expanded": str(sp.expand(e)),
                "factored": str(sp.factor(e)),
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


# =============================================================================
# 2. GNN neural theorem prover
# =============================================================================

class DenseGraphConv(nn.Module if HAS_TORCH else object):
    """
    Simple dense graph convolution fallback.

    nodes: (B, N, F)
    adj:   (B, N, N) weighted adjacency
    mask:  (B, N) optional node mask
    """

    def __init__(self, in_dim: int, out_dim: int):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, nodes: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            nodes = nodes * mask.unsqueeze(-1)
            adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)

        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = adj @ nodes / deg
        out = self.self_lin(nodes) + self.neigh_lin(neigh)
        out = F.gelu(self.norm(out))

        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class GNNTheoremProver(nn.Module if HAS_TORCH else object):
    """
    GNN-based neural theorem prover / tactic ranker.

    Input graph:
      nodes = theorem state, premises, definitions, lemmas
      adj   = dependency / relevance edges

    Outputs:
      proof_probability
      tactic_logits
      premise_scores
      graph_embedding

    This is a heuristic neural guide, not a formal proof checker.
    """

    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_tactics: int = 16,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        super().__init__()

        self.layers = nn.ModuleList()
        last = node_feature_dim
        for _ in range(n_layers):
            self.layers.append(DenseGraphConv(last, hidden_dim))
            last = hidden_dim

        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.proof_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.tactic_head = nn.Linear(hidden_dim, n_tactics)
        self.premise_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        nodes: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        h = nodes
        for layer in self.layers:
            h = layer(h, adj, mask)

        if mask is None:
            pooled = h.mean(dim=1)
            premise_scores = torch.sigmoid(self.premise_head(h).squeeze(-1))
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
            premise_scores = torch.sigmoid(self.premise_head(h).squeeze(-1)) * mask

        g = self.graph_proj(pooled)
        proof_prob = self.proof_head(g).squeeze(-1)
        tactic_logits = self.tactic_head(g)

        return {
            "proof_probability": proof_prob,
            "tactic_logits": tactic_logits,
            "premise_scores": premise_scores,
            "graph_embedding": g,
        }

    @torch.no_grad()
    def predict(
        self,
        nodes: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> NeuralProofPrediction:
        out = self.forward(nodes, adj, mask)
        return NeuralProofPrediction(
            proof_probability=float(out["proof_probability"][0].item()),
            tactic_logits=out["tactic_logits"][0].detach().cpu().tolist(),
            premise_scores=out["premise_scores"][0].detach().cpu().tolist(),
            graph_embedding=out["graph_embedding"][0].detach().cpu().tolist(),
        )


# =============================================================================
# 3. Differentiable counterfactual SCM
# =============================================================================

class DifferentiableCounterfactualSCM(nn.Module if HAS_TORCH else object):
    """
    Additive-noise Structural Causal Model.

    Variables are processed in topological order 0..D-1.
    Parent mask is strictly lower-triangular, ensuring acyclicity.

    Structural equation:
        x_i = f_i(parents(x)) + eps_i

    Counterfactual:
      1. Abduction:
            eps_i = observed_i - f_i(parents(observed))
      2. Action:
            set x_j := value for do(x_j=value)
      3. Prediction:
            recompute all variables under intervention using same eps

    Fully differentiable w.r.t. mechanism parameters.
    """

    def __init__(
        self,
        n_vars: int,
        hidden_dim: int = 64,
        learn_adjacency: bool = True,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        super().__init__()

        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.learn_adjacency = learn_adjacency

        mask = torch.tril(torch.ones(n_vars, n_vars), diagonal=-1)
        self.register_buffer("base_mask", mask)

        self.raw_adj = nn.Parameter(torch.randn(n_vars, n_vars) * 0.01)

        self.mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_vars, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_vars)
        ])

    def adjacency(self) -> Tensor:
        return torch.sigmoid(self.raw_adj) * self.base_mask

    def _parent_mask_i(self, i: int) -> Tensor:
        return self.adjacency()[i]

    def _predict_i(self, x: Tensor, i: int) -> Tensor:
        mask = self._parent_mask_i(i)
        return self.mechanisms[i](x * mask).squeeze(-1)

    def forward(self, noise: Tensor) -> Tensor:
        """
        Generate structural variables from exogenous noise.
        noise: (B, D)
        """
        B, D = noise.shape
        assert D == self.n_vars
        x = torch.zeros_like(noise)

        for i in range(self.n_vars):
            xi = self._predict_i(x, i) + noise[:, i]
            onehot = F.one_hot(torch.tensor(i, device=x.device), self.n_vars).float()
            x = x * (1.0 - onehot) + xi.unsqueeze(-1) * onehot

        return x

    def abduct_noise(self, observed: Tensor) -> Tensor:
        """
        Infer additive noise eps from observed factual data.
        observed: (B, D)
        """
        eps_cols = []
        for i in range(self.n_vars):
            pred_i = self._predict_i(observed, i)
            eps_cols.append(observed[:, i] - pred_i)
        return torch.stack(eps_cols, dim=-1)

    def counterfactual(
        self,
        observed: Tensor,
        interventions: Dict[int, Union[float, Tensor]],
    ) -> CounterfactualResult:
        """
        Compute counterfactual under do-interventions.

        interventions:
            {variable_index: value}
            value can be float or Tensor(B,)
        """
        try:
            eps = self.abduct_noise(observed)
            B, D = observed.shape
            x = torch.zeros_like(observed)

            for i in range(self.n_vars):
                if i in interventions:
                    val = interventions[i]
                    if isinstance(val, Tensor):
                        xi = val.to(observed.device).float()
                        if xi.ndim == 0:
                            xi = xi.expand(B)
                    else:
                        xi = torch.full((B,), float(val), device=observed.device)
                else:
                    xi = self._predict_i(x, i) + eps[:, i]

                onehot = F.one_hot(torch.tensor(i, device=x.device), self.n_vars).float()
                x = x * (1.0 - onehot) + xi.unsqueeze(-1) * onehot

            effect = (x - observed).mean(dim=0)

            return CounterfactualResult(
                status=CounterfactualStatus.SUCCESS.name,
                factual=observed.detach().cpu().tolist(),
                counterfactual=x.detach().cpu().tolist(),
                inferred_noise=eps.detach().cpu().tolist(),
                interventions={str(k): str(v) for k, v in interventions.items()},
                effect=effect.detach().cpu().tolist(),
                note="Counterfactual computed by abduction-action-prediction.",
            )

        except Exception as e:
            return CounterfactualResult(
                status=CounterfactualStatus.MODEL_ERROR.name,
                factual=observed.detach().cpu().tolist(),
                counterfactual=[],
                inferred_noise=[],
                interventions={str(k): str(v) for k, v in interventions.items()},
                note=str(e),
            )

    def total_effect(
        self,
        observed: Tensor,
        treatment_var: int,
        treatment_value: float,
        control_value: float,
        outcome_var: int,
    ) -> Dict[str, Any]:
        """
        Estimate average total effect:
            E[Y | do(T=treatment)] - E[Y | do(T=control)]
        """
        cf_treat = self.counterfactual(observed, {treatment_var: treatment_value})
        cf_ctrl = self.counterfactual(observed, {treatment_var: control_value})

        if cf_treat.status != "SUCCESS" or cf_ctrl.status != "SUCCESS":
            return {"status": "error", "treat": cf_treat.to_dict(), "control": cf_ctrl.to_dict()}

        y_t = np.array(cf_treat.counterfactual)[:, outcome_var]
        y_c = np.array(cf_ctrl.counterfactual)[:, outcome_var]
        ate = float(np.mean(y_t - y_c))

        return {
            "status": "success",
            "treatment_var": treatment_var,
            "outcome_var": outcome_var,
            "treatment_value": treatment_value,
            "control_value": control_value,
            "average_total_effect": ate,
            "note": "SCM-based counterfactual effect, not observational causality proof.",
        }


# =============================================================================
# 4. Simulation validator
# =============================================================================

class SimulationValidator:
    """
    Runs simulations as evidence, not formal proof.
    """

    def validate(
        self,
        hypothesis: str,
        simulation_fn: Callable[[], Dict[str, Any]],
        success_fn: Callable[[Dict[str, Any]], bool],
        metric_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        threshold: float = 0.8,
    ) -> SimulationEvidence:
        examples = []
        metrics = []
        successes = 0

        for _ in range(n_trials):
            try:
                out = simulation_fn()
                ok = bool(success_fn(out))
                metric = float(metric_fn(out))
                successes += int(ok)
                metrics.append(metric)
                if len(examples) < 5:
                    examples.append(out)
            except Exception as e:
                metrics.append(float("nan"))
                if len(examples) < 5:
                    examples.append({"error": str(e)})

        clean_metrics = [m for m in metrics if math.isfinite(m)]
        success_rate = successes / max(n_trials, 1)
        mean_metric = float(np.mean(clean_metrics)) if clean_metrics else 0.0
        std_metric = float(np.std(clean_metrics)) if clean_metrics else 0.0

        # Simulation confidence capped below rigorous proof.
        confidence = min(0.75, success_rate)

        return SimulationEvidence(
            hypothesis=hypothesis,
            n_trials=n_trials,
            success_rate=success_rate,
            mean_metric=mean_metric,
            std_metric=std_metric,
            confidence=confidence,
            passed=success_rate >= threshold,
            examples=examples,
        )


# =============================================================================
# 5. AMP / DDP training wrapper
# =============================================================================

class ProofTrainingWrapper:
    """
    AMP + DDP wrapper for GNN theorem prover or SCM.
    """

    def __init__(
        self,
        model: Any,
        lr: float = 1e-4,
        use_amp: bool = True,
        use_ddp: bool = True,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if use_ddp and HAS_DIST and dist.is_initialized() and self.device.type == "cuda":
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train_gnn_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        batch:
          nodes: (B,N,F)
          adj: (B,N,N)
          mask: (B,N)
          proof_label: (B,)
          tactic_label: optional (B,) long
        """
        self.model.train()
        nodes = batch["nodes"].to(self.device)
        adj = batch["adj"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        y = batch["proof_label"].to(self.device).float()

        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(nodes, adj, mask)
            prob = out["proof_probability"]
            loss = F.binary_cross_entropy(prob, y)

            if "tactic_label" in batch:
                tactic_y = batch["tactic_label"].to(self.device).long()
                loss = loss + F.cross_entropy(out["tactic_logits"], tactic_y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": float(loss.detach().cpu())}

    def train_scm_step(self, observed: Tensor) -> Dict[str, float]:
        """
        Train SCM by reconstructing observed factual data with zero-noise prior.
        """
        self.model.train()
        observed = observed.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            eps = self.model.abduct_noise(observed)
            recon = self.model(eps)
            loss = F.mse_loss(recon, observed)
            # Small noise regularization
            loss = loss + 1e-4 * eps.pow(2).mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": float(loss.detach().cpu())}


# =============================================================================
# 6. Integrated suite
# =============================================================================

class YonisoProofCounterfactualSuite:
    """
    Integration facade.

    Use this class from AGIONE / Yoniso-Dao engine:
      - symbolic_prover for exact algebraic checks
      - neural_prover for heuristic proof guidance
      - scm for counterfactual reasoning
      - simulation_validator for empirical evidence
    """

    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        n_tactics: int = 16,
        n_vars: int = 8,
        device: str = "cpu",
    ):
        self.symbolic = SympySymbolicProver()
        self.simulator = SimulationValidator()

        if HAS_TORCH:
            self.neural_prover = GNNTheoremProver(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                n_tactics=n_tactics,
            ).to(device)
            self.scm = DifferentiableCounterfactualSCM(
                n_vars=n_vars,
                hidden_dim=hidden_dim,
            ).to(device)
        else:
            self.neural_prover = None
            self.scm = None

    def full_assessment(
        self,
        lhs: str,
        rhs: str,
        graph: Optional[Dict[str, Tensor]] = None,
        observed: Optional[Tensor] = None,
        interventions: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        # 1. Symbolic proof
        result["symbolic"] = self.symbolic.prove_identity(lhs, rhs).to_dict()

        # 2. Neural proof guidance
        if graph is not None and self.neural_prover is not None:
            pred = self.neural_prover.predict(
                graph["nodes"],
                graph["adj"],
                graph.get("mask"),
            )
            result["neural"] = pred.to_dict()

        # 3. Counterfactual reasoning
        if observed is not None and interventions is not None and self.scm is not None:
            cf = self.scm.counterfactual(observed, interventions)
            result["counterfactual"] = cf.to_dict()

        result["honest_note"] = (
            "Symbolic verification is strongest when applicable. "
            "Neural proof scores are heuristic. "
            "Counterfactuals are valid only under the specified SCM."
        )
        return result


# =============================================================================
# Smoke test
# =============================================================================

def smoke_test():
    logging.basicConfig(level=logging.INFO)

    print("=" * 72)
    print(" SESI SymPy × GNN Theorem Prover × Counterfactual SCM — Smoke Test")
    print("=" * 72)
    print(f"PyTorch available : {HAS_TORCH}")
    print(f"SymPy available   : {HAS_SYMPY}")
    print(f"DDP available     : {HAS_DIST}")

    suite = YonisoProofCounterfactualSuite(
        node_feature_dim=16,
        hidden_dim=32,
        n_tactics=8,
        n_vars=5,
        device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
    )

    # ------------------------------------------------------------------
    # 1. Symbolic proof
    # ------------------------------------------------------------------
    print("\n── Symbolic proof ──")
    true_identity = suite.symbolic.prove_identity("(x-y)*(x+y)", "x**2 - y**2")
    print(true_identity.to_dict())

    false_identity = suite.symbolic.prove_identity("x+y", "x*y")
    print(false_identity.to_dict())

    print("Lemmas:", suite.symbolic.derive_lemmas("x**2 - y**2"))

    # ------------------------------------------------------------------
    # 2. GNN theorem prover
    # ------------------------------------------------------------------
    if HAS_TORCH:
        print("\n── GNN theorem prover ──")
        B, N, F_dim = 2, 6, 16
        nodes = torch.randn(B, N, F_dim)
        adj = (torch.rand(B, N, N) > 0.5).float()
        mask = torch.ones(B, N)

        pred = suite.neural_prover.predict(nodes, adj, mask)
        print("Proof probability:", pred.proof_probability)
        print("Tactic logits:", pred.tactic_logits[:3])

        trainer = ProofTrainingWrapper(
            suite.neural_prover,
            lr=1e-3,
            use_amp=True,
            use_ddp=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        batch = {
            "nodes": nodes,
            "adj": adj,
            "mask": mask,
            "proof_label": torch.tensor([1.0, 0.0]),
            "tactic_label": torch.tensor([2, 5]),
        }
        print("Train step:", trainer.train_gnn_step(batch))

    # ------------------------------------------------------------------
    # 3. Counterfactual SCM
    # ------------------------------------------------------------------
    if HAS_TORCH:
        print("\n── Counterfactual SCM ──")
        observed = torch.randn(4, 5)

        cf = suite.scm.counterfactual(
            observed,
            interventions={0: 2.5},
        )
        print("Status:", cf.status)
        print("Effect:", cf.effect)

        effect = suite.scm.total_effect(
            observed,
            treatment_var=0,
            treatment_value=2.0,
            control_value=0.0,
            outcome_var=4,
        )
        print("Total effect:", effect)

        scm_trainer = ProofTrainingWrapper(
            suite.scm,
            lr=1e-3,
            use_amp=True,
            use_ddp=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("SCM train step:", scm_trainer.train_scm_step(observed))

    # ------------------------------------------------------------------
    # 4. Simulation validation
    # ------------------------------------------------------------------
    print("\n── Simulation validation ──")

    def sim_fn():
        x = np.random.normal()
        return {"x": x, "ok": x > -1.0, "metric": abs(x)}

    evidence = suite.simulator.validate(
        hypothesis="Synthetic simulation succeeds most of the time",
        simulation_fn=sim_fn,
        success_fn=lambda out: out["ok"],
        metric_fn=lambda out: out["metric"],
        n_trials=50,
        threshold=0.7,
    )
    print(evidence.to_dict())

    print("\n" + "=" * 72)
    print("HONEST SCOPE")
    print("• SymPy can verify some symbolic identities, not arbitrary theorems.")
    print("• GNN proof scores are heuristics, not formal proofs.")
    print("• Counterfactuals are valid only under the specified SCM.")
    print("• Simulation evidence is not rigorous proof.")
    print("• This module does not create or prove consciousness.")
    print("=" * 72)


if __name__ == "__main__":
    smoke_test()
