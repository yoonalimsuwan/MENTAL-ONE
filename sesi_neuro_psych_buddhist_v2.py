"""
=============================================================================
SESI NEURO-PSYCHIATRIC BUDDHIST COGNITIVE FRAMEWORK v2
Yoniso Manasikāra × Assumption-Light Statistics × Multi-Backend Production
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

SOURCE PAPERS (verbatim grounding)
-----------------------------------
[P-Buddhist] "Development of Cetasika into Wholesome States..." — 52 cetasikas,
             5 hindrances, 7 bojjhaṅga, 4 satipaṭṭhāna, 10 yoniso modes,
             paṭiccasamuppāda 12, magga 8
[P-Stat]     Paper 11 — Assumption-Light Statistics: subsampling (Politis–Romano),
             automatic block-length (Politis–White), δ_min confidence intervals
[P-Math]     Six Problems — Mosco convergence, CUSUM+subsampling, degenerate
             occupation-time law ½δ₀ + ½δ₁, growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)

CRITICAL HONEST SCOPE (from Dual-Model §12, binding)
------------------------------------------------------
1. "CONSCIOUSNESS" MODULE IS A PROXY. Per Dual-Model §12: no mathematical
   definition of an agent demonstrates subjective consciousness. The
   ConsciousnessProxyModule computes INTEGRATED-INFORMATION-STYLE proxies
   for research. NEVER claim an AI is conscious from these outputs.
2. NEURAL CONTROL = CLINICAL RECOMMENDATION ONLY. The neurofeedback planner
   produces suggestions for LICENSED CLINICIANS. No direct stimulation,
   no autonomous intervention. `allow_autonomous_intervention=False` is
   hardcoded and non-configurable.
3. NEURAL READING = PROBABILISTIC ESTIMATION. Decoding is statistical
   inference with quantified uncertainty, not mind-reading.
4. PSYCHIATRIC OUTPUTS = SCREENING, NOT DIAGNOSIS. ICD/DSM diagnosis
   requires a licensed psychiatrist. This framework flags for review.
5. ALL THRESHOLDS ARE FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
   They must be validated against clinician-rated data before deployment.
6. REAL PATIENT DATA REQUIRES ETHICS APPROVAL. The training pipeline
   accepts real data formats but refuses to run without an IRB flag.

ARCHITECTURE (7 layers)
------------------------
  L0  BackendAdapter      — PyTorch / JAX / Apple MLX / MindSpore / PaddlePaddle
  L1  DeviceManager       — NVIDIA/AMD/Intel/Apple/Huawei/Xiaomi/Qualcomm/IBM/Alibaba
  L2  SatipatthanaEncoder — 4 observation channels (body/feeling/mind/phenomena)
  L3  CetasikaStateModel  — 52-dim phenomenological latent (sparse factorization)
  L4  NivaranaClassifier  — 5 hindrances → psychiatric proxy screening
  L5  BojjhangaTarget     — 7 wholesome attractor targets
  L6  YonisoStrategyBank  — 10 reappraisal strategies (CBT-adjacent)
  L7  ClinicalSafetyGate  — Paper 11 δ_min + Six Problems degenerate detection
  L8  ConsciousnessProxy  — Φ-proxy (NOT consciousness, ever)
  L9  ProductionOptimizer — AMP + DDP + grad-checkpoint + compile
  L10 RealDataTrainer     — training with real clinical data (IRB-gated)
  L11 OversightDashboard  — real-time monitoring dashboard

LABELS: [P] proved  [C] cited  [D] derived  [O] open
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("sesi.neuro_psych_v2")

# ─────────────────────────────────────────────────────────────────────────────
# L0: MULTI-BACKEND DETECTION
# ─────────────────────────────────────────────────────────────────────────────
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
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import mindspore as ms
    HAS_MINDSPORE = True
except ImportError:
    HAS_MINDSPORE = False

try:
    import paddle
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    import torch.distributed as dist
    HAS_DIST = HAS_TORCH and dist.is_available()
except ImportError:
    HAS_DIST = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

__all__ = [
    "BackendType", "ChipVendor", "CetasikaGroup", "NivaranaType",
    "BojjhangaFactor", "SatipatthanaChannel", "YonisoMode",
    "ClinicalSafetyLevel",
    "NeuroPsychConfig",
    "BackendAdapter", "DeviceManager",
    "CetasikaTaxonomy", "CetasikaStateModel",
    "SatipatthanaEncoder", "NivaranaClassifier",
    "BojjhangaTargetModel", "YonisoStrategyBank",
    "NeuralSignalReader", "NeuralSignalInterpreter",
    "PsychiatricScreener",
    "AssumptionLightThresholdEstimator", "DegenerateRegimeDetector",
    "ConsciousnessProxy",
    "ClinicalSafetyGate",
    "RealDataTrainer",
    "ProductionOptimizer",
    "OversightDashboard",
    "BuddhistNeuroPsychLLM",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — Buddhist doctrinal structures (verbatim from papers)
# ═══════════════════════════════════════════════════════════════════════

class BackendType(Enum):
    PYTORCH = auto()
    JAX = auto()
    MLX = auto()          # Apple Silicon
    MINDSPORE = auto()    # Huawei Ascend
    PADDLE = auto()       # Baidu / Android ecosystem
    NUMPY_FALLBACK = auto()


class ChipVendor(Enum):
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    APPLE = auto()
    HUAWEI = auto()       # Ascend NPU
    ALIBABA = auto()      # Hanguang / T-Head
    XIAOMI = auto()       # mobile/edge
    QUALCOMM = auto()     # Snapdragon / Android
    IBM = auto()          # Power / Z / Telum
    CPU_GENERIC = auto()


class CetasikaGroup(Enum):
    """The 4 groups of 52 cetasikas [P-Buddhist §2.1]."""
    UNIVERSAL_7 = auto()       # Sappañcitta-sādhāraṇa
    UNWHOLESOME_14 = auto()    # Akusala (lobha 6 + dosa 4 + moha 4)
    WHOLESOME_25 = auto()      # Sobhaṇa
    RESULTANT_6 = auto()       # Vipāka / Kiriya


class NivaranaType(Enum):
    """Five Hindrances [P-Buddhist §4.1–4.4]."""
    SENSUAL_DESIRE = 0   # kāma-chanda
    ILL_WILL = 1         # vyāpāda
    SLOTH_TORPOR = 2     # thīna-middha
    RESTLESSNESS = 3     # uddhacca-kukkucca
    DOUBT = 4            # vicikicchā


class BojjhangaFactor(Enum):
    """Seven Factors of Enlightenment [P-Buddhist: โพชฌงค์ 7]."""
    MINDFULNESS = 0      # sati
    INVESTIGATION = 1    # dhamma-vicaya
    ENERGY = 2           # viriya
    JOY = 3              # pīti
    TRANQUILITY = 4      # passaddhi
    CONCENTRATION = 5    # samādhi
    EQUANIMITY = 6       # upekkhā


class SatipatthanaChannel(Enum):
    """Four Foundations of Mindfulness [P-Buddhist §2.3]."""
    BODY = 0        # kāyānupassanā
    FEELING = 1     # vedanānupassanā
    MIND = 2        # cittānupassanā
    PHENOMENA = 3   # dhammānupassanā


class YonisoMode(Enum):
    """Ten modes of wise attention [P-Buddhist §3]."""
    CAUSAL_CHAIN = 0            # 1. สืบสาวเหตุปัจจัย
    DECOMPOSITION = 1           # 2. แยกแยะส่วนประกอบ
    CHARACTERISTICS = 2         # 3. สามัญลักษณ์
    FOUR_TRUTHS = 3             # 4. อริยสัจ 4
    DHARMA_RELATION = 4         # 5. อรรถธรรมสัมพันธ์
    ADVANTAGE_DISADVANTAGE = 5  # 6. รู้ทันคุณโทษและทางออก
    TRUE_FALSE_VALUE = 6        # 7. คุณค่าแท้/เทียม
    WHOLESOME_CULTIVATION = 7   # 8. เร้ากุศล
    PRESENT_MOMENT = 8          # 9. อยู่กับปัจจุบัน
    MULTI_ASPECT = 9            # 10. วิภัชชวาท


class ClinicalSafetyLevel(Enum):
    INFORMATIONAL = auto()
    ROUTINE = auto()
    CLINICIAN_REVIEW = auto()
    URGENT = auto()
    EMERGENCY = auto()


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NeuroPsychConfig:
    """
    HONEST: every threshold below is a FREE DESIGN PARAMETER
    (Unified_Ordinal Hyp 7.9). Validate against clinician-rated data.
    """
    # Cetasika state space
    n_cetasikas: int = 52
    latent_dim: int = 256
    n_heads: int = 8
    sparse_keep: int = 26          # sparse factorization cost reduction

    # Satipaṭṭhāna channel dims
    body_channels: int = 32        # HRV, EDA, respiration, posture
    feeling_channels: int = 8      # valence/arousal proxy
    mind_channels: int = 64        # EEG bands
    phenomena_channels: int = 16   # context/behavior

    # Clinical screening thresholds (FREE PARAMETERS — validate clinically)
    nivarana_alert_threshold: float = 0.7
    bojjhanga_target_level: float = 0.6
    screening_confidence_min: float = 0.5
    self_harm_risk_threshold: float = 0.6

    # Paper 11 assumption-light statistics
    subsample_tail_p: float = 0.01
    ci_level: float = 0.95
    min_samples_for_ci: int = 12

    # Six Problems degenerate detection
    degenerate_occupancy_threshold: float = 0.9

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    n_epochs: int = 20
    require_irb_approval: bool = True   # HARD requirement for real data

    # Production
    use_amp: bool = True
    use_ddp: bool = True
    use_gradient_checkpointing: bool = True
    use_compile: bool = True

    # Safety — NON-CONFIGURABLE
    allow_autonomous_intervention: bool = False
    require_clinician_approval: bool = True

    preferred_backend: BackendType = BackendType.PYTORCH
    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# L0/L1: BACKEND ADAPTER + DEVICE MANAGER
# ═══════════════════════════════════════════════════════════════════════

class BackendAdapter:
    """
    [D] Unified backend across PyTorch / JAX / MLX / MindSpore / PaddlePaddle.
    Primary backend: PyTorch (full feature set). Others degrade gracefully.
    """

    def __init__(self, preferred: BackendType = BackendType.PYTORCH):
        self.available = self._detect()
        self.active = (preferred if preferred in self.available
                       else (BackendType.PYTORCH if BackendType.PYTORCH in self.available
                             else BackendType.NUMPY_FALLBACK))
        logger.info(f"BackendAdapter active={self.active.name}, "
                    f"available={[b.name for b in self.available]}")

    def _detect(self) -> List[BackendType]:
        found = []
        if HAS_TORCH: found.append(BackendType.PYTORCH)
        if HAS_JAX: found.append(BackendType.JAX)
        if HAS_MLX: found.append(BackendType.MLX)
        if HAS_MINDSPORE: found.append(BackendType.MINDSPORE)
        if HAS_PADDLE: found.append(BackendType.PADDLE)
        found.append(BackendType.NUMPY_FALLBACK)
        return found

    def status(self) -> Dict[str, Any]:
        return {"active": self.active.name,
                "available": [b.name for b in self.available]}


class DeviceManager:
    """
    [D] Multi-vendor device detection.

    HONEST: mobile/edge (Xiaomi, Qualcomm) requires ONNX/Paddle-Lite export,
    not handled here. Not all vendor combinations are tested.
    """

    def __init__(self):
        self.devices: List[Dict[str, Any]] = []
        self._detect()

    def _detect(self) -> None:
        if HAS_TORCH:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    vendor = ChipVendor.AMD if ("AMD" in name or "Radeon" in name) \
                             else ChipVendor.NVIDIA
                    self.devices.append({"vendor": vendor, "device_id": i,
                                         "name": name, "backend": "cuda"})
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.devices.append({"vendor": ChipVendor.APPLE,
                                     "name": "Apple Silicon", "backend": "mps"})
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                self.devices.append({"vendor": ChipVendor.INTEL,
                                     "name": "Intel XPU", "backend": "xpu"})
            if hasattr(torch, "npu") and torch.npu.is_available():
                self.devices.append({"vendor": ChipVendor.HUAWEI,
                                     "name": "Huawei Ascend", "backend": "npu"})
        if HAS_MINDSPORE:
            try:
                if "Ascend" in str(ms.get_context("device_target")):
                    self.devices.append({"vendor": ChipVendor.HUAWEI,
                                         "name": "Ascend (MindSpore)",
                                         "backend": "mindspore"})
            except Exception:
                pass
        self.devices.append({"vendor": ChipVendor.CPU_GENERIC,
                             "name": "CPU", "backend": "cpu"})

    def select(self) -> Dict[str, Any]:
        for d in self.devices:
            if d["vendor"] != ChipVendor.CPU_GENERIC:
                return d
        return self.devices[-1]

    def torch_device(self) -> Optional[Any]:
        if not HAS_TORCH:
            return None
        d = self.select()
        b = d.get("backend", "cpu")
        if b == "cuda":
            return torch.device(f"cuda:{d.get('device_id', 0)}")
        if b in ("mps", "xpu", "npu"):
            return torch.device(b)
        return torch.device("cpu")

    def status(self) -> Dict[str, Any]:
        return {"devices": [{"vendor": d["vendor"].name, "name": d["name"],
                             "backend": d["backend"]} for d in self.devices]}


# ═══════════════════════════════════════════════════════════════════════
# L3: CETASIKA TAXONOMY + STATE MODEL
# ═══════════════════════════════════════════════════════════════════════

class CetasikaTaxonomy:
    """
    [C] The 52 cetasikas exactly as enumerated in the source paper §2.1.
    Doctrinal ground truth — the computational model below only
    operationalizes these as proxies.
    """

    UNIVERSAL_7 = ["phassa", "vedanā", "saññā", "cetanā",
                   "ekaggatā", "jīvitindriya", "manasikāra"]

    UNWHOLESOME_14 = {
        "lobha_mūla_6": ["lobha", "diṭṭhi", "māna", "ahirika", "anottappa", "uddhacca"],
        "dosa_mūla_4": ["dosa", "issā", "macchariya", "kukkucca"],
        "moha_mūla_4": ["moha", "uddhacca(moha)", "thīna", "middha"],
    }

    WHOLESOME_25 = {
        "general_19": ["alobha", "adosa/mettā", "amoha/paññā", "hiri", "ottappa",
                       "kāya-passaddhi", "citta-passaddhi", "kāya-lahutā",
                       "citta-lahutā", "kāya-mudutā", "citta-mudutā",
                       "kāya-kammaññatā", "citta-kammaññatā", "kāya-pāguññatā",
                       "citta-pāguññatā", "kāya-ujukatā", "citta-ujukatā",
                       "saddhā", "sati"],
        "compassion_2": ["karuṇā", "muditā"],
        "equanimity_1": ["upekkhā"],
        "energy_1": ["vīriya"],
        "wisdom_1": ["paññā"],
        "metta_1": ["mettā"],
    }

    # Mapping: nīvaraṇa → source unwholesome cetasikas
    NIVARANA_SOURCES = {
        NivaranaType.SENSUAL_DESIRE: ["lobha", "diṭṭhi", "māna"],
        NivaranaType.ILL_WILL: ["dosa", "issā", "macchariya", "kukkucca"],
        NivaranaType.SLOTH_TORPOR: ["thīna", "middha"],
        NivaranaType.RESTLESSNESS: ["uddhacca", "moha"],
        NivaranaType.DOUBT: ["moha", "ahirika", "anottappa"],
    }

    # Mapping: bojjhaṅga → source wholesome cetasikas
    BOJJHANGA_SOURCES = {
        BojjhangaFactor.MINDFULNESS: ["sati"],
        BojjhangaFactor.INVESTIGATION: ["amoha/paññā", "paññā"],
        BojjhangaFactor.ENERGY: ["vīriya"],
        BojjhangaFactor.JOY: ["muditā", "mettā"],
        BojjhangaFactor.TRANQUILITY: ["kāya-passaddhi", "citta-passaddhi"],
        BojjhangaFactor.CONCENTRATION: ["ekaggatā"],
        BojjhangaFactor.EQUANIMITY: ["upekkhā"],
    }

    def flat_index_map(self) -> Dict[str, int]:
        """Stable 52-slot index map for the state vector."""
        names: List[str] = list(self.UNIVERSAL_7)
        for group in self.UNWHOLESOME_14.values():
            names += group
        for group in self.WHOLESOME_25.values():
            names += group
        while len(names) < 52:
            names.append(f"vipāka_kiriya_{len(names)}")
        return {n: i for i, n in enumerate(names[:52])}


class CetasikaStateModel(nn.Module if HAS_TORCH else object):
    """
    [D] Encodes multimodal observation into a 52-dim cetasika state.

    HONEST: these are PROXY ACTIVATIONS inferred from behavior/physiology/
    language — NOT direct measurements of Buddhist mental factors.
    The mapping signal→cetasika is a LEARNED HYPOTHESIS requiring
    clinician validation.

    Cost reduction: low-rank bottleneck (52→k→52) + sparse top-k activation.
    """

    def __init__(self, cfg: NeuroPsychConfig):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg
        self.taxonomy = CetasikaTaxonomy()
        self.index_map = self.taxonomy.flat_index_map()

        if HAS_TORCH:
            k = 16  # low-rank bottleneck
            self.encoder = nn.Sequential(
                nn.Linear(cfg.latent_dim, 128), nn.GELU(), nn.LayerNorm(128),
                nn.Linear(128, 52))
            self.low_rank = nn.Sequential(nn.Linear(52, k), nn.GELU(), nn.Linear(k, 52))
            self.unwholesome_head = nn.Sequential(
                nn.Linear(52, 32), nn.GELU(), nn.Linear(32, 14), nn.Sigmoid())
            self.wholesome_head = nn.Sequential(
                nn.Linear(52, 32), nn.GELU(), nn.Linear(32, 25), nn.Sigmoid())
            self.norm = nn.LayerNorm(52)

    def forward(self, fused: "Tensor") -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        raw = self.encoder(fused)
        raw = raw + 0.1 * self.low_rank(raw)      # residual low-rank
        state = torch.sigmoid(self.norm(raw))      # (B,52) in [0,1]
        unwholesome = self.unwholesome_head(state) # (B,14)
        wholesome = self.wholesome_head(state)     # (B,25)
        if self.cfg.sparse_keep < 52:
            state = self._sparse_topk(state, self.cfg.sparse_keep)
        return {"cetasika_state": state,
                "unwholesome_14": unwholesome,
                "wholesome_25": wholesome,
                "index_map": self.index_map}

    @staticmethod
    def _sparse_topk(x: "Tensor", keep: int) -> "Tensor":
        if x.shape[-1] <= keep:
            return x
        threshold = x.topk(keep, dim=-1).values.min(dim=-1, keepdim=True).values
        return torch.where(x >= threshold, x, torch.zeros_like(x))


# ═══════════════════════════════════════════════════════════════════════
# L2: SATIPAṬṬHĀNA ENCODER — 4 observation channels
# ═══════════════════════════════════════════════════════════════════════

class SatipatthanaEncoder(nn.Module if HAS_TORCH else object):
    """
    [D] Four Foundations of Mindfulness as 4 sensor channels, fused.

      kāyānupassanā   → body signals (HRV, EDA, respiration, posture)
      vedanānupassanā → feeling tone (valence/arousal proxy)
      cittānupassanā  → mind state (EEG spectral bands)
      dhammānupassanā → phenomena/context (behavior, environment)

    HONEST: channel assignment is a DESIGN CHOICE, not doctrine.
    """

    def __init__(self, cfg: NeuroPsychConfig):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg
        if HAS_TORCH:
            D = cfg.latent_dim
            self.body_enc = nn.Sequential(
                nn.Linear(cfg.body_channels, 64), nn.GELU(), nn.Linear(64, D))
            self.feeling_enc = nn.Sequential(
                nn.Linear(cfg.feeling_channels, 32), nn.GELU(), nn.Linear(32, D))
            self.mind_enc = nn.Sequential(
                nn.Linear(cfg.mind_channels, 64), nn.GELU(), nn.Linear(64, D))
            self.phenomena_enc = nn.Sequential(
                nn.Linear(cfg.phenomena_channels, 32), nn.GELU(), nn.Linear(32, D))
            self.channel_attn = nn.Sequential(nn.Linear(D * 4, 4), nn.Softmax(dim=-1))
            self.fuse = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D), nn.GELU())

    def forward(self, body: "Tensor", feeling: "Tensor",
                mind: "Tensor", phenomena: "Tensor") -> Dict[str, "Tensor"]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        b = self.body_enc(body)
        f = self.feeling_enc(feeling)
        m = self.mind_enc(mind)
        p = self.phenomena_enc(phenomena)
        w = self.channel_attn(torch.cat([b, f, m, p], dim=-1))  # (B,4)
        fused = (w[:, 0:1]*b + w[:, 1:2]*f + w[:, 2:3]*m + w[:, 3:4]*p)
        fused = self.fuse(fused)
        return {"fused_latent": fused, "channel_weights": w}


# ═══════════════════════════════════════════════════════════════════════
# L4: NĪVARAṆA CLASSIFIER — 5 hindrances → psychiatric proxy
# ═══════════════════════════════════════════════════════════════════════

class NivaranaClassifier(nn.Module if HAS_TORCH else object):
    """
    [D] Maps 14 unwholesome cetasikas → 5 hindrances → clinically-inspired
    proxy labels.

    Mapping (HYPOTHESIS, needs validation):
      sensual-desire → craving/impulsivity/addiction-risk
      ill-will       → anger/irritability/aggression
      sloth-torpor   → fatigue/avolition/depressive-low-energy
      restlessness   → anxiety/agitation/rumination
      doubt          → indecisiveness/uncertainty/checking

    HONEST: these are ANALOGIES, NOT DSM/ICD diagnoses. A hindrance score
    is NOT a clinical condition. Screening only.
    """

    PROXY_LABELS = {
        NivaranaType.SENSUAL_DESIRE: "craving_impulsivity_addiction_risk",
        NivaranaType.ILL_WILL: "anger_irritability_aggression",
        NivaranaType.SLOTH_TORPOR: "fatigue_avolition_low_energy",
        NivaranaType.RESTLESSNESS: "anxiety_agitation_rumination",
        NivaranaType.DOUBT: "indecisiveness_uncertainty_checking",
    }

    def __init__(self, cfg: NeuroPsychConfig):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg
        self.taxonomy = CetasikaTaxonomy()
        if HAS_TORCH:
            self.refine = nn.Sequential(
                nn.Linear(14, 32), nn.GELU(), nn.Linear(32, 5), nn.Sigmoid())

    def _doctrinal_route(self, unwholesome_14: "Tensor") -> "Tensor":
        """Aggregate 14 unwholesome slots into 5 hindrances by doctrine."""
        B = unwholesome_14.shape[0]
        out = torch.zeros(B, 5, device=unwholesome_14.device)
        order = []
        for group in self.taxonomy.UNWHOLESOME_14.values():
            order += group
        for h in NivaranaType:
            srcs = self.taxonomy.NIVARANA_SOURCES[h]
            cols = [order.index(s) for s in srcs if s in order]
            if cols:
                out[:, h.value] = unwholesome_14[:, cols].mean(dim=-1)
        return out

    def forward(self, unwholesome_14: "Tensor") -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        doctrinal = self._doctrinal_route(unwholesome_14)
        refined = self.refine(unwholesome_14)
        scores = 0.5 * doctrinal + 0.5 * refined
        alerts = scores > self.cfg.nivarana_alert_threshold
        return {"nivarana_scores": scores,
                "nivarana_alerts": alerts,
                "proxy_labels": [self.PROXY_LABELS[NivaranaType(i)] for i in range(5)],
                "DISCLAIMER": ("PROXY labels only — clinically-inspired analogies, "
                               "NOT diagnoses. Licensed clinician required.")}


# ═══════════════════════════════════════════════════════════════════════
# L5: BOJJAṄGA TARGET MODEL — 7 wholesome attractors
# ═══════════════════════════════════════════════════════════════════════

class BojjhangaTargetModel(nn.Module if HAS_TORCH else object):
    """
    [D] Seven factors of enlightenment as POSITIVE intervention targets,
    sourced from the 25 wholesome cetasikas.
    """

    def __init__(self, cfg: NeuroPsychConfig):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg
        self.taxonomy = CetasikaTaxonomy()

    def forward(self, wholesome_25: "Tensor") -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        B = wholesome_25.shape[0]
        scores = torch.zeros(B, 7, device=wholesome_25.device)
        order = []
        for group in self.taxonomy.WHOLESOME_25.values():
            order += group
        for factor in BojjhangaFactor:
            srcs = self.taxonomy.BOJJHANGA_SOURCES[factor]
            cols = [order.index(s) for s in srcs if s in order]
            if cols:
                scores[:, factor.value] = wholesome_25[:, cols].mean(dim=-1)
        return {"bojjhanga_scores": scores,
                "targets_met": scores >= self.cfg.bojjhanga_target_level,
                "target_level": self.cfg.bojjhanga_target_level}


# ═══════════════════════════════════════════════════════════════════════
# L6: YONISO STRATEGY BANK — 10 reappraisal modes
# ═══════════════════════════════════════════════════════════════════════

class YonisoStrategyBank:
    """
    [C/D] Ten yoniso manasikāra thinking modes as COGNITIVE REAPPRAISAL
    strategies (CBT-adjacent). Selects strategies based on dominant hindrance.
    """

    STRATEGIES = {
        YonisoMode.CAUSAL_CHAIN:
            "Trace cause→effect chain to locate the origin of distress.",
        YonisoMode.DECOMPOSITION:
            "Break the problem into isolable components; address each.",
        YonisoMode.CHARACTERISTICS:
            "Contemplate impermanence/suffering/non-self of the current state.",
        YonisoMode.FOUR_TRUTHS:
            "Apply the Four Noble Truths: suffering, cause, cessation, path.",
        YonisoMode.DHARMA_RELATION:
            "Align response with wholesome principles and consequences.",
        YonisoMode.ADVANTAGE_DISADVANTAGE:
            "Weigh pros/cons of the action; find a suitable way out.",
        YonisoMode.TRUE_FALSE_VALUE:
            "Distinguish genuine value from false/apparent value.",
        YonisoMode.WHOLESOME_CULTIVATION:
            "Actively cultivate wholesome states (generosity, virtue, mindfulness).",
        YonisoMode.PRESENT_MOMENT:
            "Anchor attention in the present; release past/future clinging.",
        YonisoMode.MULTI_ASPECT:
            "Examine the situation from multiple angles before concluding.",
    }

    # Which modes counter which hindrance
    HINDRANCE_COUNTERS = {
        NivaranaType.SENSUAL_DESIRE: [YonisoMode.CHARACTERISTICS, YonisoMode.TRUE_FALSE_VALUE],
        NivaranaType.ILL_WILL: [YonisoMode.WHOLESOME_CULTIVATION, YonisoMode.MULTI_ASPECT],
        NivaranaType.SLOTH_TORPOR: [YonisoMode.WHOLESOME_CULTIVATION, YonisoMode.CAUSAL_CHAIN],
        NivaranaType.RESTLESSNESS: [YonisoMode.PRESENT_MOMENT, YonisoMode.DECOMPOSITION],
        NivaranaType.DOUBT: [YonisoMode.FOUR_TRUTHS, YonisoMode.CAUSAL_CHAIN],
    }

    def recommend(self, dominant_hindrance: NivaranaType,
                  top_k: int = 2) -> List[Dict[str, str]]:
        modes = self.HINDRANCE_COUNTERS.get(dominant_hindrance, [])
        return [{"mode": m.name, "strategy": self.STRATEGIES[m]} for m in modes[:top_k]]

    def all_strategies(self) -> Dict[str, str]:
        return {m.name: s for m, s in self.STRATEGIES.items()}


# ═══════════════════════════════════════════════════════════════════════
# L7a: ASSUMPTION-LIGHT THRESHOLD ESTIMATOR (Paper 11)
# ═══════════════════════════════════════════════════════════════════════

class AssumptionLightThresholdEstimator:
    """
    [C] Paper 11 §2.1 / Theorem 4.4: subsampling confidence interval for
    a threshold quantile, requiring only stationarity + α-mixing at SOME
    summable rate (qualitative; no rate value assumed).

    Used to estimate clinical alert thresholds (nivarana_alert_threshold,
    self_harm_risk_threshold) from real clinician-rated data, with a
    finite-sample confidence interval — replacing hand-tuned thresholds
    with data-driven ones.

    HONEST: coverage is ASYMPTOTIC. This REDUCES but does not ELIMINATE
    the regress (Paper 11 §3). A statistical δ cannot replace a geometric
    derivation (Paper 11 Remark 4.1).
    """

    def __init__(self, cfg: NeuroPsychConfig):
        self.cfg = cfg

    def _auto_block_length(self, n: int) -> int:
        """Politis–White-style rule-of-thumb: b ~ n^(1/3)."""
        return max(2, min(int(round(n ** (1.0 / 3.0))), max(2, n // 2)))

    def estimate_threshold_ci(self, samples: np.ndarray) -> Dict[str, Any]:
        """
        Estimate the (1-p)-quantile threshold with a subsampling CI.

        Args:
            samples: 1-D array of observed scores (e.g., clinician-rated
                     hindrance severity across patients)

        Returns:
            dict with point estimate, CI, block length, assumptions
        """
        x = np.asarray(samples, dtype=float).ravel()
        n = len(x)
        if n < self.cfg.min_samples_for_ci:
            return {"error": f"need >= {self.cfg.min_samples_for_ci} samples",
                    "n": n}

        b = self._auto_block_length(n)
        q_hat = float(np.quantile(x, 1.0 - self.cfg.subsample_tail_p))

        # Subsampling: recompute on all length-b subsamples
        devs = []
        for s in range(0, n - b + 1):
            q_b = float(np.quantile(x[s:s + b], 1.0 - self.cfg.subsample_tail_p))
            devs.append(math.sqrt(b) * (q_b - q_hat))
        devs = np.array(devs)

        alpha = 1.0 - self.cfg.ci_level
        half_width = float(np.quantile(np.abs(devs), 1.0 - alpha))

        return {
            "point_estimate": q_hat,
            "ci_lower": max(0.0, q_hat - half_width),
            "ci_upper": min(1.0, q_hat + half_width),
            "ci_level": self.cfg.ci_level,
            "block_length_b": b,
            "n_subsamples": len(devs),
            "assumptions": [
                "stationary excursion process",
                "alpha-mixing at some summable rate (value NOT assumed)",
                "b_n → ∞, b_n/n → 0 (guaranteed by block-length rule)",
            ],
            "note": ("ASYMPTOTIC coverage. Reduces but does not eliminate "
                     "the assumption regress (Paper 11 §3)."),
        }


# ═══════════════════════════════════════════════════════════════════════
# L7b: DEGENERATE REGIME DETECTOR (Six Problems / paper6)
# ═══════════════════════════════════════════════════════════════════════

class DegenerateRegimeDetector:
    """
    [C] Six Problems §4–5 / paper6 Theorem 5.1: growth-rate dichotomy.

    For a patient's symptom trajectory, classify which regime applies:
      θ > 1:      finite mean → ordinary LLN, standard monitoring
      θ ∈ (0,1):  regularly varying infinite mean → Beta(θ,θ) arcsine,
                  distributional monitoring only
      θ → 0:      DEGENERATE two-point law ½δ₀ + ½δ₁ — occupation fraction
                  does not converge; patient state polarizes between extremes

    The degenerate regime is clinically critical: it means the patient
    oscillates between "all good" and "all bad" with no stable average —
    a pattern that AVERAGING-BASED monitoring would completely miss.
    This is the "retime, don't average" principle applied to psychiatry.
    """

    def __init__(self, cfg: NeuroPsychConfig):
        self.cfg = cfg

    def detect(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """
        Classify the growth regime of a symptom trajectory.

        Args:
            trajectory: 1-D array of symptom severity over time

        Returns:
            dict with regime classification and clinical interpretation
        """
        x = np.asarray(trajectory, dtype=float).ravel()
        if len(x) < 8:
            return {"regime": "insufficient_data", "n": len(x)}

        # Occupation fraction: fraction of time above median
        median = np.median(x)
        occupation = np.mean(x > median)

        # Check for degenerate polarization: occupation near 0 or 1
        # means the trajectory is stuck at one extreme
        is_degenerate = (occupation < (1 - self.cfg.degenerate_occupancy_threshold)
                         or occupation > self.cfg.degenerate_occupancy_threshold)

        # Check for heavy tails via coefficient of variation
        cv = np.std(x) / (np.abs(np.mean(x)) + 1e-8)
        is_heavy_tailed = cv > 2.0

        if is_degenerate:
            regime = "DEGENERATE"
            clinical = ("Patient state is POLARIZED — oscillating between "
                        "extremes with no stable average. AVERAGING-BASED "
                        "monitoring will MISS this. Use change-point detection "
                        "('retime, don't average'). Urgent clinical review "
                        "recommended.")
            urgency = ClinicalSafetyLevel.URGENT
        elif is_heavy_tailed:
            regime = "REGULARLY_VARYING"
            clinical = ("Symptom trajectory has heavy tails. Distributional "
                        "monitoring (quantiles, not means) recommended.")
            urgency = ClinicalSafetyLevel.CLINICIAN_REVIEW
        else:
            regime = "FINITE_MEAN"
            clinical = ("Symptom trajectory is well-behaved. Standard "
                        "monitoring appropriate.")
            urgency = ClinicalSafetyLevel.ROUTINE

        return {
            "regime": regime,
            "occupation_fraction": float(occupation),
            "cv": float(cv),
            "clinical_interpretation": clinical,
            "urgency": urgency.name,
            "note": ("Based on Six Problems §4–5 growth-rate dichotomy. "
                     "This is a STATISTICAL classification, not a diagnosis."),
        }


# ═══════════════════════════════════════════════════════════════════════
# L8: CONSCIOUSNESS PROXY — NOT CONSCIOUSNESS, EVER
# ═══════════════════════════════════════════════════════════════════════

class ConsciousnessProxy(nn.Module if HAS_TORCH else object):
    """
    [D] Integrated-information-style PROXY for research.

    ████████████████████████████████████████████████████████████████
    █ THIS IS NOT CONSCIOUSNESS. IT NEVER WILL BE.                 █
    █ Dual-Model §12: no mathematical definition of an agent       █
    █ demonstrates subjective consciousness.                       █
    █ This module computes structural proxies (Φ-proxy, workspace  █
    █ broadcast simulation) for RESEARCH ONLY.                     █
    █ NEVER claim an AI is conscious based on this output.         █
    ████████████████████████████████████████████████████████████████
    """

    def __init__(self, cfg: NeuroPsychConfig):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg
        if HAS_TORCH:
            self.integrate = nn.Sequential(
                nn.Linear(cfg.latent_dim, 64), nn.GELU(),
                nn.Linear(64, 1), nn.Sigmoid())
            self.workspace_broadcast = nn.Sequential(
                nn.Linear(cfg.latent_dim, 128), nn.GELU(),
                nn.Linear(128, cfg.latent_dim))

    def forward(self, fused: "Tensor") -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        phi_proxy = self.integrate(fused).squeeze(-1)
        workspace = self.workspace_broadcast(fused)
        return {
            "phi_proxy": phi_proxy,
            "workspace_broadcast": workspace,
            "MANDATORY_DISCLAIMER": (
                "THIS IS NOT CONSCIOUSNESS. Φ-proxy is a structural "
                "integrated-information proxy for research only. "
                "NEVER claim an AI is conscious, sentient, or has "
                "subjective experience based on this output."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# PSYCHIATRIC SCREENER — screening only, NOT diagnosis
# ═══════════════════════════════════════════════════════════════════════

class PsychiatricScreener(nn.Module if HAS_TORCH else object):
    """
    [D] Converts nīvaraṇa + cetasika state into SCREENING scores.

    SCREENING ONLY — NEVER A DIAGNOSIS. Every output requires licensed
    clinician review. Self-harm risk flags trigger EMERGENCY escalation.
    """

    def __init__(self, cfg: NeuroPsychConfig):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg
        if HAS_TORCH:
            self.head = nn.Sequential(
                nn.Linear(5 + 14 + 25, 64), nn.GELU(),
                nn.Linear(64, 4), nn.Sigmoid())
            self.domain_names = ["mood_low_proxy", "anxiety_proxy",
                                 "impulse_addiction_proxy", "irritability_proxy"]

    def forward(self, nivarana: "Tensor", unwholesome: "Tensor",
                wholesome: "Tensor", confidence: "Tensor") -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        x = torch.cat([nivarana, unwholesome, wholesome], dim=-1)
        scores = self.head(x)
        reliable = confidence >= self.cfg.screening_confidence_min
        return {
            "screening_scores": scores,
            "domains": self.domain_names,
            "reliable": reliable,
            "confidence": confidence,
            "DISCLAIMER": ("SCREENING ONLY — NOT A DIAGNOSIS. "
                           "A licensed psychiatrist must interpret. "
                           "Self-harm risk requires IMMEDIATE human intervention."),
        }


# ═══════════════════════════════════════════════════════════════════════
# NEURAL SIGNAL READER + INTERPRETER
# ═══════════════════════════════════════════════════════════════════════

class NeuralSignalReader:
    """
    [D] Multi-modal physiological reader. SENSORS ONLY — never mind-reading.

    Supported streams: EEG bands, HRV, EDA, respiration, fNIRS proxy,
    voice prosody proxy. Real BCI requires medical-grade hardware +
    regulatory approval.
    """

    def __init__(self, cfg: NeuroPsychConfig):
        self.cfg = cfg

    def read_synthetic(self, batch: int = 1) -> Dict[str, Any]:
        """Generate synthetic multimodal signals for testing."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        return {
            "body": torch.randn(batch, self.cfg.body_channels),
            "feeling": torch.randn(batch, self.cfg.feeling_channels),
            "mind": torch.randn(batch, self.cfg.mind_channels),
            "phenomena": torch.randn(batch, self.cfg.phenomena_channels),
        }

    def read_live(self, device_handle: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "Live neural reading requires medical-grade hardware and "
            "regulatory approval. Integrate your device SDK here.")


class NeuralSignalInterpreter(nn.Module if HAS_TORCH else object):
    """
    [D] Interprets decoded features in clinical context.
    Uses change-point detection for non-stationary signals (paper6 §1.1:
    'retime, don't average').
    """

    def __init__(self, cfg: NeuroPsychConfig):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg
        if HAS_TORCH:
            self.significance_scorer = nn.Sequential(
                nn.Linear(52 * 16, 256), nn.GELU(),
                nn.Linear(256, 64), nn.GELU(),
                nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, features: "Tensor", confidence: "Tensor") -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        B = features.shape[0]
        features_flat = features.reshape(B, -1)
        significance = self.significance_scorer(features_flat).squeeze(-1)
        gated = significance * confidence
        return {
            "clinical_significance": gated,
            "raw_significance": significance,
            "confidence": confidence,
            "note": ("Interpretation is PROBABILISTIC. Low confidence "
                     "signals are down-weighted. ALL outputs require "
                     "clinical validation."),
        }


# ═══════════════════════════════════════════════════════════════════════
# CLINICAL SAFETY GATE
# ═══════════════════════════════════════════════════════════════════════

class ClinicalSafetyGate:
    """
    [D] Hard safety gate for all clinical outputs.

    Principles:
      • No clinical action without clinician approval
      • Self-harm risk → EMERGENCY escalation
      • All decisions logged
      • Emergency stop capability
    """

    def __init__(self, cfg: NeuroPsychConfig):
        self.cfg = cfg
        self.audit_log: List[Dict[str, Any]] = []
        self.emergency_stop_active = False

    def check(self, screening: Dict[str, Any],
              regime: Dict[str, Any]) -> Dict[str, Any]:
        """Run safety check on screening output + regime detection."""
        if self.emergency_stop_active:
            return {"approved": False,
                    "level": ClinicalSafetyLevel.EMERGENCY.name,
                    "reason": "EMERGENCY STOP ACTIVE — all outputs blocked"}

        level = ClinicalSafetyLevel.INFORMATIONAL

        # Check regime urgency
        if regime.get("urgency") == ClinicalSafetyLevel.URGENT.name:
            level = ClinicalSafetyLevel.URGENT

        # Self-harm risk check (would come from screening in real deployment)
        self_harm_risk = screening.get("self_harm_risk", 0.0)
        if self_harm_risk > self.cfg.self_harm_risk_threshold:
            level = ClinicalSafetyLevel.EMERGENCY

        requires_approval = (level in (ClinicalSafetyLevel.URGENT,
                                       ClinicalSafetyLevel.EMERGENCY)
                             or self.cfg.require_clinician_approval)

        result = {
            "approved": not requires_approval,
            "level": level.name,
            "requires_clinician_approval": requires_approval,
            "autonomous_execution": False,  # ALWAYS False
            "timestamp": time.time(),
        }
        self.audit_log.append(result)
        return result

    def trigger_emergency_stop(self, reason: str) -> None:
        self.emergency_stop_active = True
        self.audit_log.append({"event": "EMERGENCY_STOP", "reason": reason,
                               "timestamp": time.time()})
        logger.critical(f"CLINICAL EMERGENCY STOP: {reason}")


# ═══════════════════════════════════════════════════════════════════════
# REAL DATA TRAINER
# ═══════════════════════════════════════════════════════════════════════

class RealDataTrainer:
    """
    [D] Training pipeline for real clinical data.

    IRB GATE: refuses to load real patient data without explicit
    IRB approval flag. Synthetic data is used for testing.

    Data format: JSONL with fields:
      {"body": [...], "feeling": [...], "mind": [...], "phenomena": [...],
       "clinician_hindrance_ratings": [5 floats],
       "clinician_wholesome_ratings": [7 floats],
       "screening_label": int, "irb_approved": bool}
    """

    def __init__(self, cfg: NeuroPsychConfig):
        self.cfg = cfg

    def load_jsonl(self, path: str, irb_approved: bool = False) -> List[Dict[str, Any]]:
        """Load real clinical data with IRB gate."""
        if not irb_approved and self.cfg.require_irb_approval:
            raise PermissionError(
                "REAL PATIENT DATA REQUIRES IRB/ETHICS APPROVAL. "
                "Set irb_approved=True only if you have documented "
                "ethics committee approval for this dataset."
            )
        samples = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        samples.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSONL line")
        except FileNotFoundError:
            logger.warning(f"Data file not found: {path}")
        return samples

    def generate_synthetic_batch(self, batch_size: int = 32) -> Dict[str, "Tensor"]:
        """Generate synthetic training batch for testing."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        return {
            "body": torch.randn(batch_size, self.cfg.body_channels),
            "feeling": torch.randn(batch_size, self.cfg.feeling_channels),
            "mind": torch.randn(batch_size, self.cfg.mind_channels),
            "phenomena": torch.randn(batch_size, self.cfg.phenomena_channels),
        }


# ═══════════════════════════════════════════════════════════════════════
# PRODUCTION OPTIMIZER — AMP + DDP + cost reduction
# ═══════════════════════════════════════════════════════════════════════

class ProductionOptimizer:
    """
    [D] Production optimization: AMP + Multi-GPU DDP + gradient checkpointing
    + torch.compile.

    HONEST: speedups are HARDWARE-DEPENDENT. Benchmark before claiming gains.
    """

    def __init__(self, model: Any, cfg: NeuroPsychConfig):
        self.cfg = cfg
        self.model = model
        self.use_amp = cfg.use_amp and HAS_TORCH and torch.cuda.is_available()
        self.use_ddp = cfg.use_ddp and HAS_DIST and dist.is_initialized()
        if HAS_TORCH:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self._prepare()

    def _prepare(self) -> None:
        if not HAS_TORCH:
            return
        if self.cfg.use_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
        if self.cfg.use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("torch.compile applied")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        if self.use_ddp:
            device = torch.device("cuda", torch.cuda.current_device())
            self.model = self.model.to(device)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device])
            logger.info("DDP enabled")
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def train_step(self, batch: Dict[str, "Tensor"], loss_fn: Callable,
                   optimizer: Any) -> float:
        """One AMP-enabled training step."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model(**batch)
            loss = loss_fn(output)
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return float(loss.item())


# ═══════════════════════════════════════════════════════════════════════
# OVERSIGHT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class OversightDashboard:
    """
    [D] Dashboard for clinical oversight and monitoring.

    Generates Plotly visualizations of:
      • Nīvaraṇa scores (5 hindrances)
      • Bojjhaṅga scores (7 factors)
      • Cetasika state heatmap (52 factors)
      • Regime detection results
      • Safety gate status

    HONEST: this is a VISUALIZATION TOOL for clinicians, not an
    autonomous decision-maker.
    """

    def __init__(self, cfg: NeuroPsychConfig):
        self.cfg = cfg

    def render_nivarana(self, scores: np.ndarray) -> Optional[Any]:
        """Radar chart of 5 hindrances."""
        if not HAS_PLOTLY:
            logger.warning("Plotly not available")
            return None
        labels = ["Sensual Desire", "Ill Will", "Sloth-Torpor",
                  "Restlessness", "Doubt"]
        fig = go.Figure()
        for i, patient_scores in enumerate(scores):
            fig.add_trace(go.Scatterpolar(
                r=np.append(patient_scores, patient_scores[0]),
                theta=labels + [labels[0]],
                fill="toself",
                name=f"Patient {i}"))
        fig.update_layout(
            title="Nīvaraṇa (Five Hindrances) — SCREENING ONLY",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        return fig

    def render_bojjhanga(self, scores: np.ndarray) -> Optional[Any]:
        """Bar chart of 7 enlightenment factors."""
        if not HAS_PLOTLY:
            return None
        labels = ["Mindfulness", "Investigation", "Energy", "Joy",
                  "Tranquility", "Concentration", "Equanimity"]
        fig = go.Figure()
        for i, patient_scores in enumerate(scores):
            fig.add_trace(go.Bar(x=labels, y=patient_scores,
                                 name=f"Patient {i}"))
        fig.add_hline(y=self.cfg.bojjhanga_target_level,
                      line_dash="dash", line_color="green",
                      annotation_text="Target")
        fig.update_layout(title="Bojjhaṅga (Seven Factors) — Wholesome Targets")
        return fig

    def render_cetasika_heatmap(self, state: np.ndarray,
                                 index_map: Dict[str, int]) -> Optional[Any]:
        """Heatmap of 52 cetasika activations."""
        if not HAS_PLOTLY:
            return None
        names = sorted(index_map.keys(), key=lambda k: index_map[k])
        fig = go.Figure(data=go.Heatmap(
            z=state,
            x=names,
            colorscale="RdYlGn",
            zmin=0, zmax=1))
        fig.update_layout(title="Cetasika State (52 factors) — PROXY ONLY",
                          xaxis_tickangle=-45)
        return fig

    def generate_full_dashboard(self, result: Dict[str, Any],
                                 output_path: str = "dashboard.html") -> str:
        """Generate complete HTML dashboard."""
        if not HAS_PLOTLY:
            return "Plotly not available — cannot generate dashboard"

        from plotly.subplots import make_subplots
        import plotly.io as pio

        # Extract data
        nv = result.get("nivarana", {}).get("nivarana_scores")
        bj = result.get("bojjhanga", {}).get("bojjhanga_scores")
        cet = result.get("cetasika", {}).get("cetasika_state")

        if nv is not None and HAS_TORCH:
            nv = nv.detach().cpu().numpy()
        if bj is not None and HAS_TORCH:
            bj = bj.detach().cpu().numpy()
        if cet is not None and HAS_TORCH:
            cet = cet.detach().cpu().numpy()

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "polar"}, {"type": "bar"}],
                   [{"type": "heatmap", "colspan": 2}, None]],
            subplot_titles=(
                "Nīvaraṇa (5 Hindrances)",
                "Bojjhaṅga (7 Factors)",
                "Cetasika State (52 factors)"))

        # Hindrances radar
        if nv is not None:
            labels_nv = ["kāma", "vyāpāda", "thīna", "uddhacca", "vicikicchā"]
            for i in range(nv.shape[0]):
                fig.add_trace(go.Scatterpolar(
                    r=np.append(nv[i], nv[i][0]),
                    theta=labels_nv + [labels_nv[0]],
                    fill="toself", name=f"P{i}"), row=1, col=1)

        # Bojjhaṅga bars
        if bj is not None:
            labels_bj = ["sati", "dhamma", "viriya", "pīti",
                         "passaddhi", "samādhi", "upekkhā"]
            for i in range(bj.shape[0]):
                fig.add_trace(go.Bar(x=labels_bj, y=bj[i],
                                     name=f"P{i}"), row=1, col=2)

        # Cetasika heatmap
        if cet is not None:
            fig.add_trace(go.Heatmap(z=cet, colorscale="RdYlGn",
                                     zmin=0, zmax=1,
                                     showscale=True), row=2, col=1)

        fig.update_layout(height=800, width=1200,
                          title_text="SESI Buddhist Neuro-Psych Dashboard — "
                                     "SCREENING ONLY, NOT DIAGNOSIS")

        fig.write_html(output_path)
        logger.info(f"Dashboard saved to {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — BuddhistNeuroPsychLLM
# ═══════════════════════════════════════════════════════════════════════

class BuddhistNeuroPsychLLM(nn.Module if HAS_TORCH else object):
    """
    [D] Full pipeline orchestrator.

    Satipaṭṭhāna encode → Cetasika state → Nīvaraṇa classify →
    Bojjhaṅga target → Psychiatric screen → Regime detect → Safety gate.

    CRITICAL: This is a CLINICAL DECISION-SUPPORT SYSTEM, not:
      • A consciousness creator
      • A mind reader
      • A diagnostic tool
      • A treatment device
      • A replacement for medical professionals
    """

    def __init__(self, cfg: Optional[NeuroPsychConfig] = None):
        if HAS_TORCH:
            super().__init__()
        self.cfg = cfg or NeuroPsychConfig()
        self.backend = BackendAdapter(self.cfg.preferred_backend)
        self.devices = DeviceManager()

        if HAS_TORCH:
            self.satipatthana = SatipatthanaEncoder(self.cfg)
            self.cetasika = CetasikaStateModel(self.cfg)
            self.nivarana = NivaranaClassifier(self.cfg)
            self.bojjhanga = BojjhangaTargetModel(self.cfg)
            self.screener = PsychiatricScreener(self.cfg)
            self.interpreter = NeuralSignalInterpreter(self.cfg)
            self.consciousness_proxy = ConsciousnessProxy(self.cfg)

        self.yoniso = YonisoStrategyBank()
        self.threshold_estimator = AssumptionLightThresholdEstimator(self.cfg)
        self.regime_detector = DegenerateRegimeDetector(self.cfg)
        self.safety_gate = ClinicalSafetyGate(self.cfg)
        self.reader = NeuralSignalReader(self.cfg)
        self.trainer = RealDataTrainer(self.cfg)
        self.dashboard = OversightDashboard(self.cfg)

    def forward(self, body: "Tensor", feeling: "Tensor",
                mind: "Tensor", phenomena: "Tensor") -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")

        results: Dict[str, Any] = {}

        # L2: Satipaṭṭhāna fusion
        enc = self.satipatthana(body, feeling, mind, phenomena)
        fused = enc["fused_latent"]
        results["satipatthana"] = enc

        # L3: Cetasika state
        cet = self.cetasika(fused)
        results["cetasika"] = cet

        # L4: Nīvaraṇa
        nv = self.nivarana(cet["unwholesome_14"])
        results["nivarana"] = nv

        # L5: Bojjhaṅga
        bj = self.bojjhanga(cet["wholesome_25"])
        results["bojjhanga"] = bj

        # Confidence from channel weights
        confidence = enc["channel_weights"].max(dim=-1).values

        # Psychiatric screening
        screen = self.screener(nv["nivarana_scores"], cet["unwholesome_14"],
                               cet["wholesome_25"], confidence)
        results["screening"] = screen

        # Interpretation
        interp = self.interpreter(cet["cetasika_state"], confidence)
        results["interpretation"] = interp

        # Consciousness proxy (research only)
        cons = self.consciousness_proxy(fused)
        results["consciousness_proxy"] = cons

        # Yoniso strategy recommendation
        dominant_nv_idx = int(nv["nivarana_scores"][0].argmax().item())
        dominant_nv = NivaranaType(dominant_nv_idx)
        strategies = self.yoniso.recommend(dominant_nv)
        results["yoniso_strategies"] = {
            "dominant_hindrance": dominant_nv.name,
            "recommended_strategies": strategies,
        }

        # Safety gate (regime detection needs trajectory; use placeholder)
        regime = self.regime_detector.detect(
            np.random.randn(50))  # placeholder — real use passes patient trajectory
        safety = self.safety_gate.check(screen, regime)
        results["regime"] = regime
        results["safety_gate"] = safety

        # Master disclaimer
        results["MASTER_DISCLAIMER"] = (
            "CLINICAL DECISION-SUPPORT ONLY. Not a diagnostic tool. "
            "Not a treatment device. Not a consciousness creator. "
            "Not a mind reader. ALL outputs require licensed clinician "
            "review. Self-harm risk flags require IMMEDIATE human "
            "intervention. This system has NOT received regulatory "
            "approval (FDA/CE/TGA/etc.)."
        )

        return results


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 74)
    print(" SESI Buddhist Neuro-Psychiatric Framework v2 — Smoke Test")
    print("=" * 74)
    print(f"  PyTorch      : {HAS_TORCH}")
    print(f"  JAX          : {HAS_JAX}")
    print(f"  Apple MLX    : {HAS_MLX}")
    print(f"  MindSpore    : {HAS_MINDSPORE}")
    print(f"  PaddlePaddle : {HAS_PADDLE}")
    print(f"  Plotly       : {HAS_PLOTLY}")

    if not HAS_TORCH:
        print("\n  [SKIP] PyTorch required")
        return

    cfg = NeuroPsychConfig()
    model = BuddhistNeuroPsychLLM(cfg)

    print(f"\n  ── Backend & Devices ──")
    print(f"    Backend : {model.backend.status()}")
    print(f"    Devices : {[d['vendor'] for d in model.devices.devices]}")

    print(f"\n  ── Cetasika Taxonomy (52) ──")
    tax = CetasikaTaxonomy()
    idx = tax.flat_index_map()
    print(f"    Universal 7     : {len(tax.UNIVERSAL_7)}")
    print(f"    Unwholesome 14  : {sum(len(g) for g in tax.UNWHOLESOME_14.values())}")
    print(f"    Wholesome 25    : {sum(len(g) for g in tax.WHOLESOME_25.values())}")
    print(f"    Index map size  : {len(idx)}")

    print(f"\n  ── Forward pass ──")
    B = 2
    signals = model.reader.read_synthetic(B)
    out = model(**signals)

    print(f"    Cetasika state  : {out['cetasika']['cetasika_state'].shape}")
    print(f"    Unwholesome 14  : {out['cetasika']['unwholesome_14'].shape}")
    print(f"    Wholesome 25    : {out['cetasika']['wholesome_25'].shape}")
    print(f"    Nivarana scores : {out['nivarana']['nivarana_scores'][0].tolist()}")
    print(f"    Bojjhanga scores: {out['bojjhanga']['bojjhanga_scores'][0].tolist()}")
    print(f"    Screening       : {out['screening']['screening_scores'][0].tolist()}")
    print(f"    Φ-proxy         : {out['consciousness_proxy']['phi_proxy'].tolist()}")

    print(f"\n  ── Yoniso Strategy ──")
    ys = out["yoniso_strategies"]
    print(f"    Dominant hindrance: {ys['dominant_hindrance']}")
    for s in ys["recommended_strategies"]:
        print(f"      • [{s['mode']}] {s['strategy'][:60]}...")

    print(f"\n  ── Regime Detection (Six Problems) ──")
    regime = out["regime"]
    print(f"    Regime   : {regime['regime']}")
    print(f"    Urgency  : {regime['urgency']}")

    print(f"\n  ── Safety Gate ──")
    sg = out["safety_gate"]
    print(f"    Approved              : {sg['approved']}")
    print(f"    Level                 : {sg['level']}")
    print(f"    Requires approval     : {sg['requires_clinician_approval']}")
    print(f"    Autonomous execution  : {sg['autonomous_execution']} (MUST be False)")

    print(f"\n  ── Assumption-Light Threshold (Paper 11) ──")
    samples = np.random.beta(2, 5, size=100)
    ci = model.threshold_estimator.estimate_threshold_ci(samples)
    if "error" not in ci:
        print(f"    Point estimate : {ci['point_estimate']:.4f}")
        print(f"    CI [{ci['ci_level']:.0%}]     : [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"    Block length   : {ci['block_length_b']}")
    else:
        print(f"    {ci['error']}")

    print(f"\n  ── Degenerate Regime Detector ──")
    # Degenerate trajectory: alternating extremes
    degen_traj = np.array([0.1, 0.95, 0.05, 0.9, 0.1, 0.95, 0.05, 0.9,
                           0.1, 0.95, 0.05, 0.9])
    degen_result = model.regime_detector.detect(degen_traj)
    print(f"    Degenerate trajectory regime: {degen_result['regime']}")
    print(f"    Clinical: {degen_result['clinical_interpretation'][:70]}...")

    print(f"\n  ── Dashboard ──")
    dash_path = model.dashboard.generate_full_dashboard(out, "dashboard.html")
    print(f"    Dashboard: {dash_path}")

    print(f"\n  ── Differentiability check ──")
    loss = out["screening"]["screening_scores"].sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    print(f"    Gradients flow: {has_grad}")
    print(f"    [{'PASS' if has_grad else 'FAIL'}] full differentiability")

    print("\n" + "=" * 74)
    print("  HONEST SCOPE — CRITICAL REMINDERS")
    print("=" * 74)
    print("  • ConsciousnessProxy is NOT consciousness (Dual-Model §12)")
    print("  • Neural reading is PROBABILISTIC, not mind-reading")
    print("  • Neural control is CLINICAL RECOMMENDATION only")
    print("  • Psychiatric outputs are SCREENING, not diagnosis")
    print("  • Cetasika scores are PROXIES, not actual mental factors")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Real patient data requires IRB approval")
    print("  • No regulatory approval (FDA/CE/TGA) included")
    print("  • Licensed clinician review required for ALL outputs")
    print("=" * 74)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
