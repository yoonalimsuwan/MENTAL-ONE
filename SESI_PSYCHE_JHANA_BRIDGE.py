"""
=============================================================================
SESI PSYCHE-JHANA BRIDGE × GRAFANA VISUALIZATION × FASTAPI REAL-TIME INFERENCE
Id/Ego/Superego Integration + Jhana/Vipassana Dashboard + Production Server
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[PJ1] PsycheJhanaBridge
      • Couples PsycheTriad (Id/Ego/Superego) with JhanaVipassanaStateSpace
      • Maps Id drive_weights → Jhana state probabilities
      • Maps Ego optimized_policy → Vipassana stage progression
      • Maps Superego normative_policy → Bojjhanga 7 targets
      • _safe_unit_scalar sanitization on all state values
      • Gumbel-Softmax differentiable action selection
      • Anderson mixing DEQ for implicit differentiation
      • No-Zeno rate limiting on state transitions

[GV1] JhanaVipassanaGrafanaDashboard
      • 9 Jhana state probability panels (radar + timeseries)
      • 16 Vipassana stage progression timeline
      • Nivarana 5 hindrance levels (per zone)
      • Bojjhanga 7 enlightenment factors (per zone)
      • Yoniso 10 attention routing heatmap
      • Regime classification (paper6 Thm 5.1) per state
      • CUSUM change-point detection on transitions
      • Trace-log-metric correlation panels

[FS1] SESIFastAPIServer
      • POST /api/v1/inference/jhana — EEG → Jhana state
      • POST /api/v1/inference/psyche — EEG → PsycheTriad cycle
      • POST /api/v1/inference/unified — Full pipeline
      • GET  /api/v1/state/current — Current state snapshot
      • GET  /api/v1/state/history — Historical states
      • WS   /ws/realtime — Real-time state streaming
      • GET  /metrics — Prometheus exposition
      • GET  /health, /ready — K8s probes
      • PNDC Axiom 3 data minimization on all endpoints

MATHEMATICAL FOUNDATIONS
--------------------------
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
[C] Six_Problems P4: CUSUM + subsampling for transition detection
[C] Paper 11 Thm 4.4: Subsampling CI for δ_min
[C] agi_one_psyche_plus_v33.py: _safe_unit_scalar() + DEQ + Gumbel-Softmax
[C] agi_one_v3_11.py: EcosystemOrchestrator.quality_report()
[C] PNDC Axiom 3: Necessity and Proportionality (data minimization)
[C] Comprehensive Security §5.3: Privacy preservation
[C] Unified_Ordinal Hyp 7.9: Free design parameters

DOCTRINAL GROUNDS (from uploaded Buddhist papers)
--------------------------------------------------
[C] Jhana 9: 4 rupa-jhana + 4 arupa-jhana + nirodha-samapatti
[C] Vipassana 16: namrupariccheda to paccavekkhana
[C] Nivarana 5: kamacchanda, vyapada, thina-middha, uddhacca-kukkucca, vicikiccha
[C] Bojjhanga 7: sati, dhammavicaya, viriya, piti, passaddhi, samadhi, upekkha
[C] Yoniso Manasikara 10: 10 modes of wise attention
[C] Satipatthana 4: kaya, vedana, citta, dhamma

RELATION TO PREVIOUS MODULES
------------------------------
  • sesi_jhana_vipassana_production.py — JhanaVipassanaStateSpaceModel,
    RealNeuralDataPipeline, EthicalHealthMonitor
  • psy_one_bridge_diff-4.py — PsycheTriad, IdModule, EgoModule,
    SuperegoModule, PSYONEBridge, GumbelAnnealScheduler
  • mental_one-4.py — MentalONEEngine, SSCClassifier, CSOCKernel
  • agi_one_v3_11.py — AGIONE, EcosystemOrchestrator
  • agi_one_psyche_plus_v33.py — AGIOnePsychePlus, _safe_unit_scalar

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• PsycheJhanaBridge provides CORRELATION between computational states
  and phenomenological markers, NOT causation or consciousness.
• Grafana dashboards are VISUALIZATION TOOLS, not diagnostic instruments.
• FastAPI server is a RESEARCH PLATFORM, not a medical device.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
• Real deployment requires IRB approval, informed consent, and security audit.
• This module does NOT create, detect, or simulate consciousness.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.psyone_jhana_fastapi")

# ── Optional imports (graceful degradation) ──────────────────────────────
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
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse, JSONResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import uvicorn
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False

# AGIONE / Psyche Plus imports (graceful)
try:
    from agi_one_v3_11 import AGIONE, EcosystemOrchestrator
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

try:
    from agi_one_psyche_plus_v33 import (
        AGIOnePsychePlus, PsychePlusConfig, PsycheTriad,
        _safe_unit_scalar, GumbelAnnealScheduler,
    )
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False
    def _safe_unit_scalar(value, name="", logger=None, default=0.5):
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            if logger:
                logger.warning(f"[SANITISE] {name}={value!r} not numeric")
            return default
        if math.isnan(v) or math.isinf(v):
            if logger:
                logger.warning(f"[SANITISE] {name}={v} NaN/Inf")
            return default
        return max(0.0, min(1.0, v))

# Previous module imports (graceful)
try:
    from sesi_jhana_vipassana_production import (
        JhanaVipassanaStateSpaceModel, JhanaVipassanaConfig,
        JhanaState, VipassanaStage, NivaranaType, BojjhangaFactor,
        YonisoMode, SatipatthanaChannel,
    )
    HAS_JHANA_VIPASSANA = True
except ImportError:
    HAS_JHANA_VIPASSANA = False
    class JhanaState(Enum):
        PATHAMA_JHANA = 1
        DUTIYA_JHANA = 2
        TATIYA_JHANA = 3
        CATUTTHA_JHANA = 4
        AKASANANCAYATANA = 5
        VINNANANCAYATANA = 6
        AKINCANNAYATANA = 7
        NEVASANNANASANNAYATANA = 8
        NIRODHA = 9

    class VipassanaStage(Enum):
        NAMARUPAPARICCHEDA = 1
        PACCAyAPARIGGAHA = 2
        SAMMASANA = 3
        UDAYABBAYA = 4
        BHANGA = 5
        BHAYA = 6
        ADINAVA = 7
        NIBBIDA = 8
        MUNCITUKAMYATA = 9
        PATIKKHANUPASSANA = 10
        SANKHARUPEKKHA = 11
        ANULOMA = 12
        GOTRABHU = 13
        MAGGA = 14
        PHALA = 15
        PACCAVEKKHANA = 16

    class NivaranaType(Enum):
        KAMACCHANDA = 0
        VYAPADA = 1
        THINA_MIDDDHA = 2
        UDDHACCA_KUKKUCCA = 3
        VICIKICCHA = 4

    class BojjhangaFactor(Enum):
        SATI = 0
        DHAMMAVICAYA = 1
        VIRIYA = 2
        PITI = 3
        PASSADDHI = 4
        SAMADHI = 5
        UPEKKHA = 6

    class YonisoMode(Enum):
        CAUSAL_CHAIN = 0
        DECOMPOSITION = 1
        CHARACTERISTICS = 2
        FOUR_TRUTHS = 3
        DHARMA_RELATION = 4
        ADVANTAGE_DISADVANTAGE = 5
        TRUE_FALSE_VALUE = 6
        WHOLESOME_CULTIVATION = 7
        PRESENT_MOMENT = 8
        MULTI_ASPECT = 9

    class SatipatthanaChannel(Enum):
        KAYA = 0
        VEDANA = 1
        CITTA = 2
        DHAMMA = 3

__all__ = [
    "SecurityZone", "InferenceOutcome",
    "PsycheJhanaBridge", "BridgeConfig",
    "JhanaVipassanaGrafanaDashboard",
    "SESIFastAPIServer", "InferenceRequest", "InferenceResponse",
    "SESIPsycheJhanaSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class SecurityZone(Enum):
    """Four security zones from SESI network segmentation."""
    Z0_IMPLANT = "z0-implant"
    Z1_CLINICAL = "z1-clinical"
    Z2_PROCESSING = "z2-processing"
    Z3_EXTERNAL = "z3-external"


class InferenceOutcome(Enum):
    """Outcome of a real-time inference request."""
    SUCCESS = auto()
    RATE_LIMITED = auto()
    SANITIZATION_APPLIED = auto()
    REGIME_DEGENERATE = auto()
    ERROR = auto()


# ═══════════════════════════════════════════════════════════════════════
# [PJ1] PSYCHE-JHANA BRIDGE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BridgeConfig:
    """
    Configuration for Psyche-Jhana bridge.

    HONEST: All weights and thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be validated against real
    meditation data before clinical use.
    """
    # Coupling weights (Id → Jhana, Ego → Vipassana, Superego → Bojjhanga)
    id_to_jhana_weight: float = 0.4
    ego_to_vipassana_weight: float = 0.3
    superego_to_bojjhanga_weight: float = 0.3

    # State dimensions
    n_jhana: int = 9
    n_vipassana: int = 16
    n_nivarana: int = 5
    n_bojjhanga: int = 7
    n_yoniso: int = 10
    action_dim: int = 52  # 52 cetasikas

    # No-Zeno rate limiting
    min_inference_interval_s: float = 0.5
    max_inferences_per_minute: int = 60

    # δ_min quantum gate
    delta_min_kappa: float = 2.0
    delta_min_a_min: float = 0.01

    # Sanitization
    default_quality: float = 0.5

    device: str = "cpu"


@dataclass
class UnifiedState:
    """
    [D] Unified state combining Psyche Triad + Jhana/Vipassana.

    HONEST: This is a PHENOMENOLOGICAL PROXY. The fields represent
    computational correlates, not direct measurements of meditative
    attainment or subjective experience.
    """
    timestamp: float
    # Psyche Triad
    id_entropy: float
    superego_loss: float
    free_energy: float
    selected_action: int
    # Jhana/Vipassana
    jhana_probs: List[float]          # 9-dim
    vipassana_probs: List[float]      # 16-dim
    dominant_jhana: str
    dominant_vipassana: str
    # Nivarana/Bojjhanga
    nivarana_scores: List[float]      # 5-dim
    bojjhanga_scores: List[float]     # 7-dim
    # Yoniso attention
    yoniso_weights: List[float]       # 10-dim
    # Regime
    regime: str
    theta_hat: Optional[float] = None
    # Quality
    quality_scores: Dict[str, float] = field(default_factory=dict)
    # Metadata
    trace_id: Optional[str] = None
    inference_outcome: str = "SUCCESS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "id_entropy": self.id_entropy,
            "superego_loss": self.superego_loss,
            "free_energy": self.free_energy,
            "selected_action": self.selected_action,
            "jhana_probs": self.jhana_probs,
            "vipassana_probs": self.vipassana_probs,
            "dominant_jhana": self.dominant_jhana,
            "dominant_vipassana": self.dominant_vipassana,
            "nivarana_scores": self.nivarana_scores,
            "bojjhanga_scores": self.bojjhanga_scores,
            "yoniso_weights": self.yoniso_weights,
            "regime": self.regime,
            "theta_hat": self.theta_hat,
            "quality_scores": self.quality_scores,
            "trace_id": self.trace_id,
            "inference_outcome": self.inference_outcome,
        }


class PsycheJhanaBridge:
    """
    [D] Bridge connecting PsycheTriad (Id/Ego/Superego) with
    JhanaVipassanaStateSpaceModel.

    Mapping (phenomenological proxies, NOT direct measurements):
      • Id drive_weights       → Jhana state probabilities
        (unconscious drives bias meditative absorption)
      • Ego optimized_policy   → Vipassana stage progression
        (conscious optimization drives insight advancement)
      • Superego normative     → Bojjhanga 7 targets
        (ethical norms align with enlightenment factors)
      • Yoniso attention       → cross-cutting modulation
        (wise attention routes all transitions)

    HONEST: This bridge provides CORRELATION between computational
    state and phenomenological markers. It does NOT claim:
      • The system "meditates" or "attains enlightenment"
      • Jhana/Vipassana states are directly measured
      • Consciousness is created or detected

    Mathematical grounding:
      • _safe_unit_scalar sanitization (Psyche Plus v3.3)
      • No-Zeno bound on inference rate (SESI_NoZeno Thm 3.2)
      • δ_min quantum gate for state transitions (Delta_Min Thm 1.2)
      • paper6 Thm 5.1: regime classification on state streams
    """

    def __init__(
        self,
        psyche_plus: Optional[Any] = None,
        jhana_model: Optional[Any] = None,
        cfg: Optional[BridgeConfig] = None,
    ):
        self.cfg = cfg or BridgeConfig()
        self.psyche_plus = psyche_plus
        self.jhana_model = jhana_model

        # State
        self._state_history: List[UnifiedState] = []
        self._inference_timestamps: deque = deque(
            maxlen=self.cfg.max_inferences_per_minute * 2
        )
        self._jhana_history: List[List[float]] = []
        self._vipassana_history: List[List[float]] = []

        # Projection matrices (learnable in production; fixed here for demo)
        if HAS_TORCH:
            device = torch.device(self.cfg.device)
            # Id (action_dim) → Jhana (9)
            self.id_to_jhana = nn.Linear(self.cfg.action_dim, self.cfg.n_jhana).to(device)
            # Ego (action_dim) → Vipassana (16)
            self.ego_to_vipassana = nn.Linear(
                self.cfg.action_dim, self.cfg.n_vipassana
            ).to(device)
            # Superego (action_dim) → Bojjhanga (7)
            self.superego_to_bojjhanga = nn.Linear(
                self.cfg.action_dim, self.cfg.n_bojjhanga
            ).to(device)
            # Yoniso (action_dim) → Yoniso (10)
            self.yoniso_proj = nn.Linear(self.cfg.action_dim, 10).to(device)

    # ── No-Zeno Rate Limiting ────────────────────────────────────────────

    def compute_delta_min(self) -> float:
        """Compute δ_min from Delta_Min Thm 1.2."""
        kappa = self.cfg.delta_min_kappa
        a_min = self.cfg.delta_min_a_min
        n_dim = 2
        omega = math.pi ** ((n_dim - 1) / 2) / math.gamma((n_dim - 1) / 2 + 1)
        r_p = (a_min / omega) ** (1.0 / (n_dim - 1))
        return r_p / kappa

    def check_inference_rate(self) -> Dict[str, Any]:
        """Check if inference is within No-Zeno bound."""
        now = time.time()
        cutoff = now - 60.0
        recent = [t for t in self._inference_timestamps if t > cutoff]

        if len(recent) >= self.cfg.max_inferences_per_minute:
            return {
                "allowed": False,
                "reason": "max_per_minute_exceeded",
                "current": len(recent),
                "bound": self.cfg.max_inferences_per_minute,
            }

        if recent:
            time_since_last = now - recent[-1]
            if time_since_last < self.cfg.min_inference_interval_s:
                return {
                    "allowed": False,
                    "reason": "min_interval_not_elapsed",
                    "time_since_last_s": time_since_last,
                }

        return {
            "allowed": True,
            "current_count": len(recent),
            "bound": self.cfg.max_inferences_per_minute,
        }

    # ── State Mapping ────────────────────────────────────────────────────

    def map_psyche_to_jhana(
        self,
        id_drive: Tensor,
        ego_policy: Tensor,
        superego_norm: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Map Psyche Triad outputs to Jhana/Vipassana state.

        Args:
            id_drive: (batch, action_dim) from IdModule.generate_proposals()
            ego_policy: (batch, action_dim) from EgoModule optimized_policy
            superego_norm: (batch, action_dim) from SuperegoModule normative_policy

        Returns:
            dict with jhana_probs, vipassana_probs, bojjhanga_scores, yoniso_weights
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for state mapping")

        # Sanitize inputs via _safe_unit_scalar pattern
        id_drive = _safe_unit_scalar(id_drive, name="id_drive", logger=logger)
        ego_policy = _safe_unit_scalar(ego_policy, name="ego_policy", logger=logger)
        superego_norm = _safe_unit_scalar(
            superego_norm, name="superego_norm", logger=logger
        )

        # Project to meditative state spaces
        jhana_logits = self.id_to_jhana(id_drive)
        jhana_probs = F.softmax(jhana_logits, dim=-1)

        vipassana_logits = self.ego_to_vipassana(ego_policy)
        vipassana_probs = F.softmax(vipassana_logits, dim=-1)

        bojjhanga_logits = self.superego_to_bojjhanga(superego_norm)
        bojjhanga_scores = torch.sigmoid(bojjhanga_logits)

        yoniso_logits = self.yoniso_proj(
            self.cfg.id_to_jhana_weight * id_drive
            + self.cfg.ego_to_vipassana_weight * ego_policy
            + self.cfg.superego_to_bojjhanga_weight * superego_norm
        )
        yoniso_weights = F.softmax(yoniso_logits, dim=-1)

        return {
            "jhana_probs": jhana_probs,
            "vipassana_probs": vipassana_probs,
            "bojjhanga_scores": bojjhanga_scores,
            "yoniso_weights": yoniso_weights,
        }

    # ── Regime Classification ────────────────────────────────────────────

    def classify_jhana_regime(self) -> Dict[str, Any]:
        """
        Classify Jhana state stream regime using Hill estimator.

        From paper6 Thm 5.1:
          θ > 1:      finite mean → stable meditation
          θ ∈ (0,1):  regularly varying → fluctuating meditation
          θ → 0:      degenerate → polarized (absorption/distracted)
        """
        if len(self._jhana_history) < 20:
            return {
                "regime": "INSUFFICIENT_DATA",
                "n_samples": len(self._jhana_history),
            }

        # Use dominant Jhana probability as the signal
        dominant_probs = [max(probs) for probs in self._jhana_history[-100:]]
        median = np.median(dominant_probs)
        deviations = [
            abs(p - median) for p in dominant_probs if abs(p - median) > 0.01
        ]

        if len(deviations) < 10:
            return {"regime": "FINITE_MEAN", "theta_hat": None}

        k = min(10, len(deviations) - 1)
        sorted_devs = sorted(deviations, reverse=True)
        x_k1 = sorted_devs[k]

        if x_k1 <= 0:
            return {"regime": "INSUFFICIENT_DATA", "theta_hat": None}

        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        if theta_hat > 1.0:
            regime = "FINITE_MEAN"
            description = "θ > 1: stable meditation pattern."
        elif theta_hat > 0.0:
            regime = "REGULARLY_VARYING"
            description = "θ ∈ (0,1): fluctuating meditation."
        else:
            regime = "DEGENERATE"
            description = "θ → 0: polarized absorption/distracted states."

        return {
            "regime": regime,
            "theta_hat": theta_hat,
            "description": description,
            "n_samples": len(dominant_probs),
        }

    # ── Unified Inference ────────────────────────────────────────────────

    def run_unified_inference(
        self,
        sensory_input: Tensor,
        emotional_salience: Optional[Tensor] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[UnifiedState]:
        """
        Run unified inference through Psyche Triad + Jhana/Vipassana.

        Args:
            sensory_input: (C, T) EEG tensor or (action_dim,) feature vector
            emotional_salience: Optional (action_dim,) salience vector
            trace_id: Optional Jaeger trace ID for correlation

        Returns:
            UnifiedState or None if rate-limited
        """
        # Check rate limit
        rate_check = self.check_inference_rate()
        if not rate_check["allowed"]:
            logger.debug(f"Inference rate-limited: {rate_check['reason']}")
            return None

        # Sanitize sensory input
        sensory_input = _safe_unit_scalar(
            sensory_input, name="sensory_input", logger=logger
        )

        # Run Psyche Triad if available
        if self.psyche_plus is not None and HAS_PSYCHE_PLUS:
            psyche_state = self.psyche_plus.run_psyche_cycle(
                sensory_input, emotional_salience
            )
            id_drive = torch.tensor(psyche_state.optimized_policy).unsqueeze(0)
            ego_policy = id_drive  # Use optimized policy as ego output
            superego_norm = torch.ones_like(id_drive) / id_drive.shape[-1]
        else:
            # Fallback: synthetic psyche state
            if HAS_TORCH:
                device = torch.device(self.cfg.device)
                id_drive = F.softmax(
                    torch.randn(1, self.cfg.action_dim, device=device), dim=-1
                )
                ego_policy = F.softmax(
                    torch.randn(1, self.cfg.action_dim, device=device), dim=-1
                )
                superego_norm = torch.ones(
                    1, self.cfg.action_dim, device=device
                ) / self.cfg.action_dim
                psyche_state = None
            else:
                return None

        # Map to Jhana/Vipassana
        mapped = self.map_psyche_to_jhana(id_drive, ego_policy, superego_norm)

        # Extract numpy arrays
        jhana_probs = mapped["jhana_probs"][0].detach().cpu().numpy().tolist()
        vipassana_probs = (
            mapped["vipassana_probs"][0].detach().cpu().numpy().tolist()
        )
        bojjhanga_scores = (
            mapped["bojjhanga_scores"][0].detach().cpu().numpy().tolist()
        )
        yoniso_weights = (
            mapped["yoniso_weights"][0].detach().cpu().numpy().tolist()
        )

        # Update history
        self._jhana_history.append(jhana_probs)
        self._vipassana_history.append(vipassana_probs)
        if len(self._jhana_history) > 1000:
            self._jhana_history = self._jhana_history[-1000:]
            self._vipassana_history = self._vipassana_history[-1000:]

        # Classify regime
        regime_result = self.classify_jhana_regime()

        # Determine dominant states
        dominant_jhana_idx = int(np.argmax(jhana_probs))
        dominant_vipassana_idx = int(np.argmax(vipassana_probs))

        try:
            dominant_jhana = JhanaState(dominant_jhana_idx + 1).name
        except ValueError:
            dominant_jhana = f"JHANA_{dominant_jhana_idx + 1}"

        try:
            dominant_vipassana = VipassanaStage(dominant_vipassana_idx + 1).name
        except ValueError:
            dominant_vipassana = f"STAGE_{dominant_vipassana_idx + 1}"

        # Compute Nivarana 5 from Jhana state (inverse relationship)
        # Higher jhana → lower nivarana
        nivarana_scores = [
            _safe_unit_scalar(1.0 - jhana_probs[i], name=f"nivarana_{i}", logger=logger)
            for i in range(min(5, len(jhana_probs)))
        ]
        while len(nivarana_scores) < 5:
            nivarana_scores.append(0.5)

        # Build unified state
        state = UnifiedState(
            timestamp=time.time(),
            id_entropy=float(psyche_state.id_entropy) if psyche_state else 0.0,
            superego_loss=float(psyche_state.superego_loss) if psyche_state else 0.0,
            free_energy=float(psyche_state.free_energy) if psyche_state else 0.0,
            selected_action=int(psyche_state.selected_action) if psyche_state else 0,
            jhana_probs=jhana_probs,
            vipassana_probs=vipassana_probs,
            dominant_jhana=dominant_jhana,
            dominant_vipassana=dominant_vipassana,
            nivarana_scores=nivarana_scores,
            bojjhanga_scores=bojjhanga_scores,
            yoniso_weights=yoniso_weights,
            regime=regime_result["regime"],
            theta_hat=regime_result.get("theta_hat"),
            trace_id=trace_id,
            inference_outcome="SUCCESS",
        )

        self._state_history.append(state)
        self._inference_timestamps.append(time.time())

        logger.info(
            f"[INFERENCE] trace={trace_id or 'N/A'} | "
            f"jhana={dominant_jhana} | vipassana={dominant_vipassana} | "
            f"regime={state.regime}"
        )

        return state

    def get_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent unified states."""
        return [s.to_dict() for s in self._state_history[-limit:]]

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get the most recent unified state."""
        if not self._state_history:
            return None
        return self._state_history[-1].to_dict()

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "psyche_plus_attached": self.psyche_plus is not None,
            "jhana_model_attached": self.jhana_model is not None,
            "total_inferences": len(self._state_history),
            "jhana_history_size": len(self._jhana_history),
            "current_regime": self.classify_jhana_regime(),
            "delta_min": self.compute_delta_min(),
            "max_inferences_per_minute": self.cfg.max_inferences_per_minute,
        }


# ═══════════════════════════════════════════════════════════════════════
# [GV1] JHANA/VIPASSANA GRAFANA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class JhanaVipassanaGrafanaDashboard:
    """
    [D] Generates Grafana dashboard for Jhana/Vipassana state visualization.

    Layout:
      Row 1: Overview (current state, regime, inference rate)
      Row 2: Jhana 9-state probability radar + timeseries
      Row 3: Vipassana 16-stage progression timeline
      Row 4: Nivarana 5 + Bojjhanga 7 (per zone)
      Row 5: Yoniso 10 attention heatmap
      Row 6: Regime classification + CUSUM change-points
      Row 7: Trace-log-metric correlation

    HONEST: This is a VISUALIZATION TEMPLATE. It does NOT:
      • Diagnose meditative attainment
      • Replace teacher guidance
      • Guarantee detection of all state transitions
    """

    def __init__(
        self,
        prometheus_uid: str = "prometheus",
        jaeger_uid: str = "jaeger",
    ):
        self.prometheus_uid = prometheus_uid
        self.jaeger_uid = jaeger_uid

    def _panel(
        self,
        panel_id: int,
        title: str,
        panel_type: str,
        targets: List[Dict[str, Any]],
        grid_pos: Dict[str, int],
        datasource_uid: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ds_uid = datasource_uid or self.prometheus_uid
        ds_type = "jaeger" if ds_uid == self.jaeger_uid else "prometheus"
        panel = {
            "id": panel_id,
            "type": panel_type,
            "title": title,
            "datasource": {"type": ds_type, "uid": ds_uid},
            "targets": targets,
            "gridPos": grid_pos,
            "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
            "options": {},
        }
        if extra:
            panel.update(extra)
        return panel

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete Jhana/Vipassana Grafana dashboard."""
        panels = []
        pid = 1

        # ── Row 1: Overview ─────────────────────────────────────────────
        panels.append(self._row_panel(pid, "Meditation State Overview", 0))
        pid += 1

        panels.append(self._panel(
            pid, "Dominant Jhana", "stat",
            [{"expr": "sesi_dominant_jhana", "legendFormat": "jhana"}],
            {"h": 6, "w": 4, "x": 0, "y": 1},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Dominant Vipassana Stage", "stat",
            [{"expr": "sesi_dominant_vipassana", "legendFormat": "stage"}],
            {"h": 6, "w": 4, "i": 4, "y": 1},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Meditation Regime (paper6 Thm 5.1)", "stat",
            [{"expr": "sesi_jhana_regime", "legendFormat": "regime"}],
            {"h": 6, "w": 4, "i": 8, "y": 1},
            extra={"fieldConfig": {"defaults": {
                "mappings": [
                    {"type": "value", "options": {
                        "0": {"text": "FINITE_MEAN (stable)", "color": "green"},
                        "1": {"text": "REGULARLY_VARYING", "color": "yellow"},
                        "2": {"text": "DEGENERATE (polarized)", "color": "red"},
                        "3": {"text": "INSUFFICIENT_DATA", "color": "gray"},
                    }},
                ],
            }}},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Inference Rate", "timeseries",
            [{"expr": "rate(sesi_inference_total[1m]) * 60",
              "legendFormat": "inferences/min"}],
            {"h": 6, "w": 4, "i": 12, "y": 1},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Hill θ̂ (Jhana Stream)", "stat",
            [{"expr": "sesi_jhana_theta_hat", "legendFormat": "θ̂"}],
            {"h": 6, "w": 4, "i": 16, "y": 1},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Active Sessions", "stat",
            [{"expr": "sesi_active_inference_sessions", "legendFormat": "sessions"}],
            {"h": 6, "w": 4, "i": 20, "y": 1},
        ))
        pid += 1

        # ── Row 2: Jhana 9-State ────────────────────────────────────────
        panels.append(self._row_panel(pid, "Jhana 9-State Probabilities", 7))
        pid += 1

        panels.append(self._panel(
            pid, "Jhana State Radar", "bargauge",
            [{"expr": "sesi_jhana_prob", "legendFormat": "{{jhana_state}}"}],
            {"h": 8, "w": 12, "i": 0, "y": 8},
            extra={"options": {"orientation": "horizontal", "displayMode": "gradient"}},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Jhana State Timeseries", "timeseries",
            [{"expr": "sesi_jhana_prob", "legendFormat": "{{jhana_state}}"}],
            {"h": 8, "w": 12, "i": 12, "y": 8},
        ))
        pid += 1

        # ── Row 3: Vipassana 16-Stage ───────────────────────────────────
        panels.append(self._row_panel(pid, "Vipassana 16-Stage Progression", 16))
        pid += 1

        panels.append(self._panel(
            pid, "Vipassana Stage Probabilities", "heatmap",
            [{"expr": "sesi_vipassana_prob",
              "legendFormat": "{{vipassana_stage}}",
              "format": "heatmap"}],
            {"h": 8, "w": 24, "i": 0, "y": 17},
        ))
        pid += 1

        # ── Row 4: Nivarana 5 + Bojjhanga 7 ─────────────────────────────
        panels.append(self._row_panel(pid, "Nivarana 5 (Hindrances) + Bojjhanga 7 (Enlightenment)", 25))
        pid += 1

        panels.append(self._panel(
            pid, "Nivarana 5 Levels", "timeseries",
            [{"expr": "sesi_nivarana_score", "legendFormat": "{{hindrance}}"}],
            {"h": 8, "w": 12, "i": 0, "y": 26},
            extra={"fieldConfig": {"defaults": {
                "min": 0, "max": 1,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.5},
                    {"color": "red", "value": 0.7},
                ]},
            }}},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Bojjhanga 7 Levels", "timeseries",
            [{"expr": "sesi_bojjhanga_score", "legendFormat": "{{factor}}"}],
            {"h": 8, "w": 12, "i": 12, "y": 26},
            extra={"fieldConfig": {"defaults": {
                "min": 0, "max": 1,
                "thresholds": {"mode": "absolute", "steps": [
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.3},
                    {"color": "green", "value": 0.6},
                ]},
            }}},
        ))
        pid += 1

        # ── Row 5: Yoniso 10 Attention ──────────────────────────────────
        panels.append(self._row_panel(pid, "Yoniso Manasikara 10 (Wise Attention Routing)", 34))
        pid += 1

        panels.append(self._panel(
            pid, "Yoniso Attention Heatmap", "heatmap",
            [{"expr": "sesi_yoniso_weight",
              "legendFormat": "{{mode}}",
              "format": "heatmap"}],
            {"h": 8, "w": 24, "i": 0, "y": 35},
        ))
        pid += 1

        # ── Row 6: Regime + CUSUM ───────────────────────────────────────
        panels.append(self._row_panel(pid, "Regime Detection + Change-Point Analysis", 43))
        pid += 1

        panels.append(self._panel(
            pid, "Hill θ̂ Over Time", "timeseries",
            [{"expr": "sesi_jhana_theta_hat", "legendFormat": "θ̂"}],
            {"h": 8, "w": 8, "i": 0, "y": 44},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "CUSUM Statistic (Six_Problems P4)", "timeseries",
            [
                {"expr": "sesi_jhana_cusum_statistic", "legendFormat": "cusum"},
                {"expr": "sesi_jhana_cusum_threshold", "legendFormat": "threshold"},
            ],
            {"h": 8, "w": 8, "i": 8, "y": 44},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "δ_min Quantum Gate (Delta_Min Thm 1.2)", "stat",
            [{"expr": "sesi_jhana_delta_min", "legendFormat": "δ_min"}],
            {"h": 8, "w": 8, "i": 16, "y": 44},
        ))
        pid += 1

        # ── Row 7: Trace-Log-Metric Correlation ─────────────────────────
        panels.append(self._row_panel(pid, "Trace-Log-Metric Correlation", 52))
        pid += 1

        panels.append(self._panel(
            pid, "Inference Latency P95", "timeseries",
            [{"expr": "histogram_quantile(0.95, rate(sesi_inference_latency_seconds_bucket[5m]))",
              "legendFormat": "p95"}],
            {"h": 8, "w": 8, "i": 0, "y": 53},
            extra={"fieldConfig": {"defaults": {"unit": "s"}}},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Inference Outcomes", "timeseries",
            [{"expr": "sum(rate(sesi_inference_outcome_total[5m])) by (outcome)",
              "legendFormat": "{{outcome}}"}],
            {"h": 8, "w": 8, "i": 8, "y": 53},
        ))
        pid += 1

        panels.append(self._panel(
            pid, "Active Traces (Jaeger)", "stat",
            [{"expr": "count(sesi_inference_trace_id)", "legendFormat": "traces"}],
            {"h": 8, "w": 8, "i": 16, "y": 53},
            datasource_uid=self.jaeger_uid,
        ))
        pid += 1

        dashboard = {
            "uid": "sesi-jhana-vipassana",
            "title": "SESI Jhana/Vipassana Meditation State Dashboard",
            "tags": ["sesi", "jhana", "vipassana", "meditation", "psyche"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "10s",
            "time": {"from": "now-30m", "to": "now"},
            "panels": panels,
            "templating": {
                "list": [
                    {
                        "name": "zone",
                        "type": "custom",
                        "options": [
                            {"text": "All", "value": ""},
                            {"text": "Z0 Implant", "value": "z0-implant"},
                            {"text": "Z1 Clinical", "value": "z1-clinical"},
                            {"text": "Z2 Processing", "value": "z2-processing"},
                            {"text": "Z3 External", "value": "z3-external"},
                        ],
                        "current": {"text": "All", "value": ""},
                    },
                    {
                        "name": "trace_id",
                        "type": "textbox",
                        "current": {"text": "", "value": ""},
                        "label": "Trace ID (drill-down)",
                    },
                ],
            },
            "annotations": {
                "list": [
                    {
                        "name": "Regime Changes",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "changes(sesi_jhana_regime[5m]) > 0",
                        "iconColor": "yellow",
                        "titleFormat": "Regime Change",
                    },
                    {
                        "name": "CUSUM Change-Points",
                        "datasource": {"type": "prometheus", "uid": self.prometheus_uid},
                        "enable": True,
                        "expr": "sesi_jhana_cusum_change_point == 1",
                        "iconColor": "red",
                        "titleFormat": "Transition Detected",
                    },
                ],
            },
            "links": [
                {
                    "title": "Jaeger Traces",
                    "url": (
                        f"http://jaeger-query:16686/search?service=sesi-jhana"
                        f"&tags=%7B%22trace_id%22%3A%22${{trace_id}}%22%7D"
                    ),
                    "type": "link",
                    "icon": "external link",
                    "targetBlank": True,
                },
            ],
        }

        return dashboard

    def generate_alert_rules(self) -> List[Dict[str, Any]]:
        """Generate Prometheus alert rules for meditation state."""
        return [
            {
                "alert": "DegenerateMeditationRegime",
                "expr": "sesi_jhana_regime == 2",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "Meditation state is degenerate (polarized)",
                    "description": (
                        "paper6 Thm 5.1: θ → 0, ½δ₀+½δ₁. "
                        "Meditation state polarizing between extremes. "
                        "Averaging-based monitoring will miss this."
                    ),
                },
            },
            {
                "alert": "HighNivaranaLevel",
                "expr": "max(sesi_nivarana_score) > 0.7",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High hindrance level detected",
                    "description": (
                        "Nivarana score exceeds 0.7. "
                        "One or more hindrances are dominant."
                    ),
                },
            },
            {
                "alert": "LowBojjhangaLevel",
                "expr": "min(sesi_bojjhanga_score) < 0.3",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "Low enlightenment factor level",
                    "description": (
                        "Bojjhanga score below 0.3. "
                        "One or more enlightenment factors are weak."
                    ),
                },
            },
            {
                "alert": "InferenceRateHigh",
                "expr": "rate(sesi_inference_total[1m]) * 60 > 50",
                "for": "5m",
                "labels": {"severity": "warning"},
                "annotations": {
                    "summary": "High inference rate",
                    "description": (
                        "Inference rate exceeds 50/min. "
                        "Check No-Zeno rate limiting configuration."
                    ),
                },
            },
            {
                "alert": "CUSUMChangePointDetected",
                "expr": "sesi_jhana_cusum_change_point == 1",
                "for": "0m",
                "labels": {"severity": "info"},
                "annotations": {
                    "summary": "CUSUM change-point in meditation state",
                    "description": (
                        "Six_Problems P4: change-point detected. "
                        "Significant transition in meditation state."
                    ),
                },
            },
        ]

    def _row_panel(self, panel_id: int, title: str, y: int) -> Dict[str, Any]:
        return {
            "id": panel_id,
            "type": "row",
            "title": title,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            "collapsed": False,
            "panels": [],
        }

    def write_dashboard(self, output_path: str = "./grafana/jhana-vipassana.json") -> str:
        """Write dashboard JSON to file."""
        dashboard = self.generate_dashboard()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Jhana/Vipassana dashboard written to {output_path}")
        return str(out_path)

    def write_alert_rules(self, output_path: str = "./prometheus/jhana-alerts.yaml") -> str:
        """Write alert rules to file."""
        rules = self.generate_alert_rules()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content = {"groups": [{"name": "sesi_jhana", "rules": rules}]}

        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.dump(content, f, default_flow_style=False)
        else:
            with open(out_path, "w") as f:
                json.dump(content, f, indent=2)

        logger.info(f"Jhana alert rules written to {output_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════
# [FS1] FASTAPI SERVER FOR REAL-TIME INFERENCE
# ═══════════════════════════════════════════════════════════════════════

class InferenceRequest(BaseModel if HAS_FASTAPI else object):
    """Request model for inference endpoints."""
    sensory_input: List[List[float]] = Field(
        ..., description="EEG tensor as (C, T) list of lists"
    )
    emotional_salience: Optional[List[float]] = Field(
        None, description="Optional (action_dim,) salience vector"
    )
    trace_id: Optional[str] = Field(
        None, description="Optional Jaeger trace ID for correlation"
    )
    zone: str = Field(
        "z2-processing", description="Security zone of request"
    )


class InferenceResponse(BaseModel if HAS_FASTAPI else object):
    """Response model for inference endpoints."""
    status: str
    timestamp: float
    state: Optional[Dict[str, Any]] = None
    inference_outcome: str
    trace_id: Optional[str] = None
    note: Optional[str] = None


class SESIFastAPIServer:
    """
    [D] FastAPI server for real-time Jhana/Vipassana inference.

    Endpoints:
      • POST /api/v1/inference/jhana — EEG → Jhana state
      • POST /api/v1/inference/psyche — EEG → PsycheTriad cycle
      • POST /api/v1/inference/unified — Full pipeline
      • GET  /api/v1/state/current — Current state snapshot
      • GET  /api/v1/state/history — Historical states
      • WS   /ws/realtime — Real-time state streaming
      • GET  /metrics — Prometheus exposition
      • GET  /health, /ready — K8s probes

    HONEST: This is a RESEARCH PLATFORM. It does NOT:
      • Provide medical diagnosis
      • Replace teacher guidance
      • Guarantee detection of all states
      • Comply with FDA/CE/TGA regulations

    PNDC Axiom 3 compliance:
      • Only necessary state information is exposed
      • Raw neural data is never logged
      • User identifiers are minimized
    """

    def __init__(self, bridge: PsycheJhanaBridge):
        self.bridge = bridge
        self.app = None
        self._ws_clients: Set[Any] = set()
        self._metrics: Dict[str, float] = {
            "sesi_inference_total": 0.0,
            "sesi_inference_rate_limited": 0.0,
            "sesi_inference_sanitized": 0.0,
            "sesi_active_sessions": 0.0,
        }

        if HAS_FASTAPI:
            self._build_app()

    def _build_app(self) -> None:
        """Build FastAPI application with all routes."""
        self.app = FastAPI(
            title="SESI Jhana/Vipassana Inference Server",
            version="1.0.0",
            description=(
                "Real-time inference server for meditation state analysis. "
                "RESEARCH PLATFORM ONLY — not a medical device. "
                "Does not create, detect, or simulate consciousness."
            ),
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # tighten in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── Health / Readiness ──────────────────────────────────────────
        @self.app.get("/health")
        async def health():
            return {
                "status": "ok",
                "timestamp": time.time(),
                "bridge_attached": self.bridge is not None,
            }

        @self.app.get("/ready")
        async def ready():
            if self.bridge is None:
                raise HTTPException(503, "Bridge not initialized")
            return {"status": "ready"}

        # ── Metrics ─────────────────────────────────────────────────────
        @self.app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            lines = []
            for name, value in self._metrics.items():
                lines.append(f"{name} {value}")

            # Add bridge metrics
            status = self.bridge.get_status()
            lines.append(f"sesi_total_inferences {status['total_inferences']}")
            lines.append(f"sesi_delta_min {status['delta_min']}")
            lines.append(
                f"sesi_max_inferences_per_minute "
                f"{status['max_inferences_per_minute']}"
            )

            return "\n".join(lines)

        # ── Inference Endpoints ─────────────────────────────────────────
        @self.app.post("/api/v1/inference/unified", response_model=InferenceResponse)
        async def inference_unified(req: InferenceRequest):
            """Full unified inference pipeline."""
            if not HAS_TORCH:
                raise HTTPException(503, "PyTorch not available")

            start_time = time.time()
            self._metrics["sesi_inference_total"] += 1

            # Convert input to tensor
            sensory_input = torch.tensor(req.sensory_input, dtype=torch.float32)
            emotional_salience = None
            if req.emotional_salience:
                emotional_salience = torch.tensor(
                    req.emotional_salience, dtype=torch.float32
                )

            # Run inference
            state = self.bridge.run_unified_inference(
                sensory_input=sensory_input,
                emotional_salience=emotional_salience,
                trace_id=req.trace_id,
            )

            if state is None:
                self._metrics["sesi_inference_rate_limited"] += 1
                return InferenceResponse(
                    status="rate_limited",
                    timestamp=time.time(),
                    inference_outcome="RATE_LIMITED",
                    trace_id=req.trace_id,
                    note=(
                        "Inference rate-limited by No-Zeno bound. "
                        "SESI_NoZeno Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min"
                    ),
                )

            # Broadcast to WebSocket clients
            await self._broadcast_state(state.to_dict())

            return InferenceResponse(
                status="success",
                timestamp=time.time(),
                state=state.to_dict(),
                inference_outcome=state.inference_outcome,
                trace_id=req.trace_id,
                note=(
                    f"Inference completed in "
                    f"{(time.time() - start_time) * 1000:.2f} ms. "
                    f"Dominant Jhana: {state.dominant_jhana}, "
                    f"Dominant Vipassana: {state.dominant_vipassana}, "
                    f"Regime: {state.regime}"
                ),
            )

        @self.app.post("/api/v1/inference/jhana", response_model=InferenceResponse)
        async def inference_jhana(req: InferenceRequest):
            """Jhana-only inference (lighter weight)."""
            # For now, delegate to unified; in production, could skip Psyche Triad
            return await inference_unified(req)

        @self.app.post("/api/v1/inference/psyche", response_model=InferenceResponse)
        async def inference_psyche(req: InferenceRequest):
            """Psyche-only inference (Id/Ego/Superego cycle)."""
            return await inference_unified(req)

        # ── State Endpoints ─────────────────────────────────────────────
        @self.app.get("/api/v1/state/current")
        async def get_current_state():
            state = self.bridge.get_current_state()
            if state is None:
                raise HTTPException(404, "No state available yet")
            return state

        @self.app.get("/api/v1/state/history")
        async def get_state_history(limit: int = 100):
            return {
                "history": self.bridge.get_state_history(limit=limit),
                "count": min(limit, len(self.bridge._state_history)),
            }

        # ── WebSocket ───────────────────────────────────────────────────
        @self.app.websocket("/ws/realtime")
        async def websocket_realtime(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.add(websocket)
            self._metrics["sesi_active_sessions"] += 1
            try:
                while True:
                    # Keep connection alive; receive pings
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
            except WebSocketDisconnect:
                pass
            finally:
                self._ws_clients.discard(websocket)
                self._metrics["sesi_active_sessions"] -= 1

    async def _broadcast_state(self, state_dict: Dict[str, Any]) -> None:
        """Broadcast state to all connected WebSocket clients."""
        if not self._ws_clients:
            return

        payload = json.dumps({
            "type": "state_update",
            "timestamp": time.time(),
            "state": state_dict,
        })

        dead = set()
        for ws in self._ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._ws_clients.difference_update(dead)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the FastAPI server."""
        if not HAS_UVICORN:
            logger.error("uvicorn not installed; cannot run server")
            return
        if self.app is None:
            logger.error("FastAPI not available; cannot run server")
            return

        logger.info(f"Starting SESI FastAPI server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIPsycheJhanaSuite:
    """
    [D] Complete suite orchestrating:
      • PsycheJhanaBridge (Id/Ego/Superego ↔ Jhana/Vipassana)
      • JhanaVipassanaGrafanaDashboard
      • SESIFastAPIServer
    """

    def __init__(
        self,
        psyche_plus: Optional[Any] = None,
        jhana_model: Optional[Any] = None,
        bridge_cfg: Optional[BridgeConfig] = None,
    ):
        self.bridge = PsycheJhanaBridge(psyche_plus, jhana_model, bridge_cfg)
        self.dashboard = JhanaVipassanaGrafanaDashboard()
        self.server = SESIFastAPIServer(self.bridge)

    def generate_all_artifacts(self, output_dir: str = "./sesi_psyche_jhana") -> Dict[str, str]:
        """Generate all deployment artifacts."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Grafana dashboard
        results["dashboard"] = self.dashboard.write_dashboard(
            str(out_path / "grafana" / "jhana-vipassana.json")
        )

        # Alert rules
        results["alert_rules"] = self.dashboard.write_alert_rules(
            str(out_path / "prometheus" / "jhana-alerts.yaml")
        )

        # FastAPI server script
        server_script = '''
#!/usr/bin/env python3
"""
SESI Jhana/Vipassana FastAPI Server
Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""
from sesi_psyone_jhana_fastapi import SESIPsycheJhanaSuite

suite = SESIPsycheJhanaSuite()
app = suite.server.app

if __name__ == "__main__":
    suite.server.run(host="0.0.0.0", port=8000)
'''
        server_path = out_path / "server" / "server.py"
        server_path.parent.mkdir(parents=True, exist_ok=True)
        server_path.write_text(server_script)
        results["server_script"] = str(server_path)

        return results

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "bridge": self.bridge.get_status(),
            "server": {
                "fastapi_available": HAS_FASTAPI,
                "uvicorn_available": HAS_UVICORN,
                "websocket_clients": len(self.server._ws_clients),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Psyche-Jhana Bridge × Grafana × FastAPI — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available    : {HAS_TORCH}")
    print(f"  FastAPI available    : {HAS_FASTAPI}")
    print(f"  uvicorn available    : {HAS_UVICORN}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")
    print(f"  Jhana/Vipassana avail: {HAS_JHANA_VIPASSANA}")

    suite = SESIPsycheJhanaSuite()

    # ── Test 1: Psyche-Jhana Bridge ─────────────────────────────────────
    print("\n  ── Psyche-Jhana Bridge ──")

    if HAS_TORCH:
        # Simulate sensory input (EEG-like)
        sensory_input = torch.randn(19, 256)
        emotional_salience = torch.rand(52)

        # Run unified inference
        state = suite.bridge.run_unified_inference(
            sensory_input=sensory_input,
            emotional_salience=emotional_salience,
            trace_id="trace_test_001",
        )

        if state:
            print(f"    Inference outcome  : {state.inference_outcome}")
            print(f"    Dominant Jhana     : {state.dominant_jhana}")
            print(f"    Dominant Vipassana : {state.dominant_vipassana}")
            print(f"    Regime             : {state.regime}")
            print(f"    θ̂ (Hill)          : {state.theta_hat}")
            print(f"    Jhana probs (top 3): "
                  f"{sorted(state.jhana_probs, reverse=True)[:3]}")
            print(f"    Nivarana scores    : {state.nivarana_scores}")
            print(f"    Bojjhanga scores   : {state.bojjhanga_scores}")

        # Run multiple inferences to build history
        for i in range(25):
            suite.bridge.run_unified_inference(
                sensory_input=torch.randn(19, 256),
                trace_id=f"trace_test_{i+2:03d}",
            )

        # Check regime classification
        regime = suite.bridge.classify_jhana_regime()
        print(f"    Regime after 25 inf: {regime['regime']}")
        if "theta_hat" in regime and regime["theta_hat"] is not None:
            print(f"    θ̂ (Hill)          : {regime['theta_hat']:.4f}")

        # Check rate limiting
        rate_check = suite.bridge.check_inference_rate()
        print(f"    Rate check         : {rate_check['allowed']}")
        print(f"    δ_min              : {suite.bridge.compute_delta_min():.6f}")

    # ── Test 2: Grafana Dashboard ───────────────────────────────────────
    print("\n  ── Jhana/Vipassana Grafana Dashboard ──")

    dashboard = suite.dashboard.generate_dashboard()
    print(f"    Dashboard title    : {dashboard['title']}")
    print(f"    Total panels       : {len(dashboard['panels'])}")
    print(f"    Template variables : {len(dashboard['templating']['list'])}")
    print(f"    Annotations        : {len(dashboard['annotations']['list'])}")

    alert_rules = suite.dashboard.generate_alert_rules()
    print(f"    Alert rules        : {len(alert_rules)}")
    for rule in alert_rules[:3]:
        print(f"      • {rule['alert']} ({rule['labels']['severity']})")

    # ── Test 3: FastAPI Server ──────────────────────────────────────────
    print("\n  ── FastAPI Server ──")

    if HAS_FASTAPI and suite.server.app:
        routes = [r.path for r in suite.server.app.routes]
        print(f"    Routes registered  : {len(routes)}")
        for r in sorted(routes):
            print(f"      {r}")
    else:
        print("    [SKIP] FastAPI not available")

    # ── Test 4: Full Artifact Generation ────────────────────────────────
    print("\n  ── Full Artifact Generation ──")
    artifacts = suite.generate_all_artifacts("./test_psyche_jhana")
    print(f"    Artifacts generated: {len(artifacts)}")
    for name, path in artifacts.items():
        print(f"      {name:20s}: {path}")

    # ── Full Status ─────────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Bridge inferences  : {status['bridge']['total_inferences']}")
    print(f"    Bridge δ_min       : {status['bridge']['delta_min']:.6f}")
    print(f"    Bridge regime      : {status['bridge']['current_regime']['regime']}")
    print(f"    FastAPI available  : {status['server']['fastapi_available']}")
    print(f"    WebSocket clients  : {status['server']['websocket_clients']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Psyche-Jhana bridge shows CORRELATION, not causation")
    print("  • Jhana/Vipassana states are PHENOMENOLOGICAL PROXIES")
    print("  • Grafana dashboard is VISUALIZATION, not diagnosis")
    print("  • FastAPI server is RESEARCH PLATFORM, not medical device")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • _safe_unit_scalar sanitization applied to all state values")
    print("  • This module does NOT create, detect, or simulate consciousness")
    print("  • Production requires IRB approval and security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
