#!/usr/bin/env python3
"""
=============================================================================
BEHAVIORAL RISK ASSESSMENT — RESEARCH SCAFFOLD
Computational modeling of impulse-control dynamics using validated instruments
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  ETHICS FIRST — READ BEFORE ANYTHING ELSE                       ║
║                                                                     ║
║  THIS IS A RESEARCH SCAFFOLD, NOT A CLINICAL OR DIAGNOSTIC TOOL.   ║
║                                                                     ║
║  • The Id/Ego/Superego framework used here is a COMPUTATIONAL       ║
║    METAPHOR for impulse-control dynamics, NOT a validated clinical  ║
║    model. Modern psychology does not use Freudian constructs as     ║
║    diagnostic criteria.                                             ║
║                                                                     ║
║  • This module does NOT categorize, label, or stigmatize any        ║
║    person's sexual behavior, substance use, or aggression.          ║
║                                                                     ║
║  • All behavioral assessments use VALIDATED RESEARCH INSTRUMENTS    ║
║    as INPUT features. No raw behavioral data is collected here.     ║
║                                                                     ║
║  • REQUIRED BEFORE ANY USE:                                         ║
║    1. IRB / Ethics Committee approval                               ║
║    2. Informed consent from every participant                       ║
║    3. Clinical oversight by qualified psychiatrist                  ║
║    4. Data de-identification (HIPAA / GDPR / Thai PDPA)            ║
║    5. This module MUST NOT be used for:                             ║
║       - Employment screening                                        ║
║       - Insurance underwriting                                      ║
║       - Legal proceedings                                           ║
║       - Any form of discrimination                                  ║
║                                                                     ║
║  THE PSYCHIATRIST IS THE FINAL DECISION MAKER. ALWAYS.             ║
╚═══════════════════════════════════════════════════════════════════════╝

VALIDATED RESEARCH INSTRUMENTS REFERENCED
─────────────────────────────────────────
[V1] Buss, A.H. & Perry, M. (1992). "The Aggression Questionnaire."
     JPSP, 63(3), 452-459.  → BPAQ: 29 items, 4 subscales
     (Physical Aggression, Verbal Aggression, Anger, Hostility)

[V2] Patton, J.H., Stanford, M.S. & Barratt, E.S. (1995).
     "Factor structure of the Barratt Impulsiveness Scale."
     J Clin Psychol, 51(6), 768-774.  → BIS-11: 30 items, 3 subscales
     (Attentional, Motor, Non-planning Impulsiveness)

[V3] Saunders, J.B. et al. (1993). "Development of the AUDIT."
     Addiction, 88(6), 791-804.  → AUDIT: 10 items, alcohol use

[V4] Skinner, H.A. (1982). "The Drug Abuse Screening Test."
     Addict Behav, 7(4), 363-371.  → DAST-10: 10 items, drug use

[V5] Whiteside, S.P. & Lynam, D.R. (2001). "The Five Factor Model
     and impulsivity: using a structural model of personality to
     understand impulsivity." Pers Individ Dif, 30(4), 669-689.
     → UPPS-P: 59 items, 5 subscales
     (Negative Urgency, Premeditation, Premeditation, Sensation Seeking, Positive Urgency)

[V6] Cybersexual behavior research:
     Wéry, A. & Billieux, J. (2017). "Problematic cybersex:
     Conceptualization, assessment, and treatment."
     Addict Behav, 64, 238-246.

COMPUTATIONAL METAPHOR MAPPING
──────────────────────────────
The Id/Ego/Superego labels are used as ORGANIZATIONAL METAPHORS
for three interacting computational systems. They do NOT imply
Freudian theory is correct or clinically validated.

  IdModule      → Impulse/Drive system: reward-seeking, urgency,
                  sensation-seeking (maps to BIS-11 Motor, UPPS-P Urgency)
  EgoModule     → Executive Control system: planning, inhibition,
                  decision-making (maps to BIS-11 Attentional/Non-planning)
  SuperEgoModule→ Normative Constraint system: social rules, moral
                  reasoning, compliance (maps to BPAQ Hostility subscale
                  inverse, normative questionnaire items)

WHAT THIS MODULE PROVIDES
─────────────────────────
  §1  Validated instrument scoring (BPAQ, BIS-11, AUDIT, DAST, UPPS-P)
  §2  Id/Ego/Superego computational metaphor for impulse-control dynamics
  §3  Transformer encoder for sequential behavioral data
  §4  World model for behavioral trajectory prediction
  §5  Risk feature integration (research only)
  §6  Self-test with synthetic data

WHAT THIS MODULE IS NOT
───────────────────────
  ❌ NOT a diagnostic tool
  ❌ NOT a categorization system for sexual behavior
  ❌ NOT a violence risk assessment tool
  ❌ NOT validated against clinical outcomes
  ❌ NOT approved for any clinical, legal, or employment use
  ❌ Does NOT replace psychiatric evaluation
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("behavioral_risk")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

ETHICS_BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  RESEARCH SCAFFOLD — NOT A CLINICAL OR DIAGNOSTIC TOOL          ║
║  ⚠️  DOES NOT CATEGORIZE OR LABEL ANY PERSON'S BEHAVIOR             ║
║  ⚠️  REQUIRES IRB APPROVAL + INFORMED CONSENT + CLINICAL OVERSIGHT  ║
║  ⚠️  THE PSYCHIATRIST IS THE FINAL DECISION MAKER. ALWAYS.          ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def ethics_guard():
    print(ETHICS_BANNER)


# =============================================================================
# §1. VALIDATED RESEARCH INSTRUMENTS
# =============================================================================

class InstrumentType(Enum):
    """Validated research instruments referenced in this module."""
    BPAQ = "buss_perry_aggression"        # [V1]
    BIS11 = "barratt_impulsiveness"       # [V2]
    AUDIT = "alcohol_use_identification"  # [V3]
    DAST10 = "drug_abuse_screening"       # [V4]
    UPPS_P = "upps_p_impulsive"           # [V5]


@dataclass
class InstrumentConfig:
    """Configuration for a validated research instrument."""
    name: str
    instrument_type: InstrumentType
    n_items: int
    max_score: float
    subscales: Dict[str, List[int]]  # subscale name -> item indices
    reference: str
    clinical_threshold: Optional[float] = None
    notes: str = ""


# Validated instrument configurations
INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "BPAQ": InstrumentConfig(
        name="Buss-Perry Aggression Questionnaire",
        instrument_type=InstrumentType.BPAQ,
        n_items=29,
        max_score=145.0,  # 5-point Likert * 29 items
        subscales={
            "physical_aggression": list(range(0, 9)),      # items 1-9
            "verbal_aggression": list(range(9, 14)),       # items 10-14
            "anger": list(range(14, 21)),                  # items 15-21
            "hostility": list(range(21, 29)),              # items 22-29
        },
        reference="Buss & Perry (1992) JPSP 63(3):452-459",
        clinical_threshold=None,  # research use only, no clinical threshold
        notes="4-point subscales. Higher scores indicate higher self-reported aggression. "
              "RESEARCH USE ONLY — not a clinical diagnostic tool.",
    ),
    "BIS11": InstrumentConfig(
        name="Barratt Impulsiveness Scale-11",
        instrument_type=InstrumentType.BIS11,
        n_items=30,
        max_score=120.0,  # 4-point Likert * 30 items
        subscales={
            "attentional": list(range(0, 8)),     # items 1-8
            "motor": list(range(8, 19)),           # items 9-19
            "non_planning": list(range(19, 30)),  # items 20-30
        },
        reference="Patton, Stanford & Barratt (1995) J Clin Psychol 51(6):768-774",
        clinical_threshold=72.0,  # commonly cited research cutoff
        notes="3 subscales. Higher scores indicate higher impulsivity. "
              "Score >= 72 sometimes used in research as high impulsivity. "
              "RESEARCH USE ONLY — not a clinical diagnostic tool.",
    ),
    "AUDIT": InstrumentConfig(
        name="Alcohol Use Disorders Identification Test",
        instrument_type=InstrumentType.AUDIT,
        n_items=10,
        max_score=40.0,  # 0-4 per item * 10 items
        subscales={
            "consumption": list(range(0, 3)),       # items 1-3
            "dependence": list(range(3, 6)),        # items 4-6
            "harmful_use": list(range(6, 10)),      # items 7-10
        },
        reference="Saunders et al. (1993) Addiction 88(6):791-804",
        clinical_threshold=8.0,  # WHO cutoff for hazardous use
        notes="WHO screening tool. Score >= 8 suggests hazardous use. "
              "RESEARCH USE ONLY — clinical interpretation requires "
              "trained professional.",
    ),
    "DAST10": InstrumentConfig(
        name="Drug Abuse Screening Test-10",
        instrument_type=InstrumentType.DAST10,
        n_items=10,
        max_score=10.0,  # binary 0/1 * 10 items
        subscales={
            "total": list(range(0, 10)),
        },
        reference="Skinner (1982) Addict Behav 7(4):363-371",
        clinical_threshold=3.0,  # commonly cited research cutoff
        notes="Binary (0/1) responses. Score >= 3 suggests possible drug "
              "use problems. RESEARCH USE ONLY — clinical interpretation "
              "requires trained professional.",
    ),
    "UPPS_P": InstrumentConfig(
        name="UPPS-P Impulsive Behavior Scale",
        instrument_type=InstrumentType.UPPS_P,
        n_items=59,
        max_score=236.0,  # 4-point Likert * 59 items
        subscales={
            "negative_urgency": list(range(0, 12)),    # items 1-12
            "lack_premeditation": list(range(12, 23)), # items 13-23
            "lack_perseverance": list(range(23, 33)),  # items 24-33
            "sensation_seeking": list(range(33, 45)),  # items 34-45
            "positive_urgency": list(range(45, 59)),   # items 46-59
        },
        reference="Whiteside & Lynam (2001) Pers Individ Dif 30(4):669-689",
        clinical_threshold=None,
        notes="5 subscales measuring distinct impulsivity facets. "
              "RESEARCH USE ONLY — not a clinical diagnostic tool.",
    ),
}


class InstrumentScorer:
    """
    Scores validated research instruments.

    IMPORTANT: This scorer computes instrument scores from PROVIDED
    item responses. It does NOT collect behavioral data. All responses
    must be obtained through proper informed consent procedures.
    """

    def __init__(self):
        self.instruments = INSTRUMENTS

    def score_instrument(
        self,
        instrument_key: str,
        responses: List[float],
    ) -> Dict[str, Any]:
        """
        Score a validated instrument from item responses.

        Args:
            instrument_key: one of BPAQ, BIS11, AUDIT, DAST10, UPPS_P
            responses: list of item responses (length must match n_items)

        Returns:
            dict with total_score, subscale_scores, clinical_note
        """
        if instrument_key not in self.instruments:
            raise ValueError(f"Unknown instrument: {instrument_key}")

        cfg = self.instruments[instrument_key]
        if len(responses) != cfg.n_items:
            raise ValueError(
                f"{cfg.name} requires {cfg.n_items} responses, got {len(responses)}"
            )

        responses_t = torch.tensor(responses, dtype=torch.float32)

        # Total score
        total = float(responses_t.sum())

        # Subscale scores
        subscale_scores = {}
        for name, indices in cfg.subscales.items():
            subscale_scores[name] = float(responses_t[indices].sum())

        # Clinical note (research context only)
        clinical_note = "RESEARCH SCORE — NOT A CLINICAL DIAGNOSIS"
        if cfg.clinical_threshold is not None and total >= cfg.clinical_threshold:
            clinical_note += f" | Score ({total:.0f}) meets research threshold ({cfg.clinical_threshold:.0f})"

        return {
            "instrument": cfg.name,
            "reference": cfg.reference,
            "total_score": total,
            "max_score": cfg.max_score,
            "subscale_scores": subscale_scores,
            "clinical_note": clinical_note,
            "disclaimer": "RESEARCH USE ONLY. Clinical interpretation requires "
                          "a qualified psychiatrist or psychologist.",
        }

    def score_all(
        self,
        responses_by_instrument: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, Any]]:
        """Score multiple instruments."""
        results = {}
        for key, responses in responses_by_instrument.items():
            try:
                results[key] = self.score_instrument(key, responses)
            except ValueError as e:
                results[key] = {"error": str(e)}
        return results


# =============================================================================
# §2. IMPULSE-CONTROL COMPUTATIONAL METAPHOR (Id/Ego/Superego)
# =============================================================================
#
# CRITICAL: The Id/Ego/Superego labels below are COMPUTATIONAL METAPHORS
# for three interacting systems in an impulse-control model. They do NOT
# imply that Freudian psychoanalytic theory is correct or clinically validated.
#
# Modern psychology uses terms like:
#   - Impulse/Drive system (roughly "Id")
#   - Executive Control system (roughly "Ego")
#   - Normative Constraint system (roughly "Superego")
#
# The Freudian labels are kept for consistency with the existing PSY ONE
# ecosystem, but the underlying math is a standard state-space model.
# =============================================================================

class ImpulseDriveModule(nn.Module):
    """
    Impulse/Drive system ("Id" metaphor).
    Models reward-seeking urgency, sensation-seeking, and drive intensity.

    Input features (from validated instruments):
    - BIS-11 Motor Impulsiveness
    - UPPS-P Negative Urgency
    - UPPS-P Sensation Seeking
    - UPPS-P Positive Urgency

    Output: drive_intensity ∈ [0, 1]
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, impulse_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            impulse_features: (batch, input_dim) normalized instrument scores
        Returns:
            drive_intensity: (batch, 1) in [0, 1]
        """
        return self.net(impulse_features)


class ExecutiveControlModule(nn.Module):
    """
    Executive Control system ("Ego" metaphor).
    Models planning, inhibition, and decision-making capacity.

    Input features (from validated instruments):
    - BIS-11 Attentional Impulsiveness (inverse)
    - BIS-11 Non-planning Impulsiveness (inverse)
    - UPPS-P Lack of Premeditation (inverse)
    - UPPS-P Lack of Perseverance (inverse)

    Output: control_capacity ∈ [0, 1]
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, control_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            control_features: (batch, input_dim) normalized instrument scores
        Returns:
            control_capacity: (batch, 1) in [0, 1]
        """
        return self.net(control_features)


class NormativeConstraintModule(nn.Module):
    """
    Normative Constraint system ("Superego" metaphor).
    Models internalized social rules, moral reasoning, and compliance.

    Input features (from validated instruments):
    - BPAQ Hostility subscale (inverse)
    - Normative compliance questionnaire items (if available)
    - Social desirability scale (if available)

    Output: constraint_strength ∈ [0, 1]
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, constraint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            constraint_features: (batch, input_dim) normalized instrument scores
        Returns:
            constraint_strength: (batch, 1) in [0, 1]
        """
        return self.net(constraint_features)


class ImpulseControlModel(nn.Module):
    """
    Integrated impulse-control model combining the three systems.

    The model computes a "regulation_index" that reflects the balance
    between impulsive drives and regulatory capacity:

        regulation_index = (control_capacity + constraint_strength) / (1 + drive_intensity)

    This is a RESEARCH METRIC, not a clinical score. It has no validated
    clinical interpretation and must not be used for any clinical purpose.
    """

    def __init__(self, input_dim: int = 22, hidden_dim: int = 64):
        super().__init__()
        # Split input into three feature groups
        self.impulse_dim = 8
        self.control_dim = 8
        self.constraint_dim = 6

        self.impulse_module = ImpulseDriveModule(self.impulse_dim, hidden_dim)
        self.control_module = ExecutiveControlModule(self.control_dim, hidden_dim)
        self.constraint_module = NormativeConstraintModule(self.constraint_dim, hidden_dim)

        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, input_dim) concatenated instrument features
        Returns:
            dict with drive, control, constraint, regulation_index
        """
        impulse_feats = features[:, :self.impulse_dim]
        control_feats = features[:, self.impulse_dim:self.impulse_dim + self.control_dim]
        constraint_feats = features[:, self.impulse_dim + self.control_dim:]

        drive = self.impulse_module(impulse_feats)
        control = self.control_module(control_feats)
        constraint = self.constraint_module(constraint_feats)

        # Regulation index: (control + constraint) / (1 + drive)
        regulation_index = (control + constraint) / (1.0 + drive)

        # Integrated output
        combined = torch.cat([drive, control, constraint], dim=-1)
        integrated = self.integration(combined)

        return {
            "drive_intensity": drive,
            "control_capacity": control,
            "constraint_strength": constraint,
            "regulation_index": regulation_index,
            "integrated_output": integrated,
        }


# =============================================================================
# §3. TRANSFORMER ENCODER FOR SEQUENTIAL BEHAVIORAL DATA
# =============================================================================

class BehavioralTransformerEncoder(nn.Module):
    """
    Transformer encoder for sequential behavioral assessment data.

    Processes time-series of instrument scores to capture temporal
    patterns in behavioral regulation. This is a RESEARCH TOOL for
    understanding temporal dynamics, not for prediction of behavior.

    Input: sequence of feature vectors from validated instruments
    Output: temporal embedding + attention weights
    """

    def __init__(
        self,
        input_dim: int = 22,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, seq_len, d_model)
            attention: optional attention weights
        """
        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :h.shape[1], :]
        h = self.encoder(h)
        return self.output_proj(h), None


# =============================================================================
# §4. WORLD MODEL FOR BEHAVIORAL TRAJECTORY PREDICTION
# =============================================================================

class BehavioralWorldModel(nn.Module):
    """
    World model for predicting behavioral regulation trajectories.

    This model predicts how the regulation_index might change over time
    based on current state. It is a RESEARCH TOOL for understanding
    temporal dynamics, not for predicting individual behavior.

    CRITICAL: This model does NOT predict whether a person will engage
    in any specific behavior. It models the TRAJECTORY of instrument
    scores over time, which is a very different thing.
    """

    def __init__(
        self,
        input_dim: int = 22,
        latent_dim: int = 128,
        output_dim: int = 3,  # drive, control, constraint
        n_future_steps: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_future_steps = n_future_steps

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Dynamics (GRU)
        self.dynamics = nn.GRUCell(latent_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_dim),
        )

        # Uncertainty
        self.uncertainty = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, output_dim),
            nn.Softplus(),
        )

    def forward(
        self, x: torch.Tensor, n_future: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) current state
        Returns:
            dict with predictions and uncertainties
        """
        if n_future is None:
            n_future = self.n_future_steps

        z = self.encoder(x)
        predictions = []
        uncertainties = []

        for _ in range(n_future):
            z = self.dynamics(z, z)
            pred = self.decoder(z)
            unc = self.uncertainty(z)
            predictions.append(pred)
            uncertainties.append(unc)

        return {
            "predictions": torch.stack(predictions, dim=1),
            "uncertainties": torch.stack(uncertainties, dim=1),
        }


# =============================================================================
# §5. RISK FEATURE INTEGRATION (RESEARCH ONLY)
# =============================================================================

class BehavioralRiskFeatureExtractor:
    """
    Extracts research features from validated instrument scores.

    IMPORTANT: This module extracts RESEARCH FEATURES from instrument
    scores. It does NOT assess, categorize, or predict any specific
    behavior. All outputs are numerical features for research analysis.
    """

    def __init__(self):
        self.scorer = InstrumentScorer()

    def extract_features(
        self,
        instrument_scores: Dict[str, Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Extract normalized feature vector from instrument scores.

        Args:
            instrument_scores: dict of instrument results from InstrumentScorer

        Returns:
            (22,) feature vector
        """
        features = []

        # BPAQ features (4 subscales, normalized)
        if "BPAQ" in instrument_scores and "subscale_scores" in instrument_scores["BPAQ"]:
            bpaq = instrument_scores["BPAQ"]["subscale_scores"]
            for key in ["physical_aggression", "verbal_aggression", "anger", "hostility"]:
                features.append(bpaq.get(key, 0.0) / 45.0)  # normalize by max
        else:
            features.extend([0.0] * 4)

        # BIS-11 features (3 subscales, normalized)
        if "BIS11" in instrument_scores and "subscale_scores" in instrument_scores["BIS11"]:
            bis11 = instrument_scores["BIS11"]["subscale_scores"]
            for key in ["attentional", "motor", "non_planning"]:
                features.append(bis11.get(key, 0.0) / 40.0)
        else:
            features.extend([0.0] * 3)

        # AUDIT features (3 subscales, normalized)
        if "AUDIT" in instrument_scores and "subscale_scores" in instrument_scores["AUDIT"]:
            audit = instrument_scores["AUDIT"]["subscale_scores"]
            for key in ["consumption", "dependence", "harmful_use"]:
                features.append(audit.get(key, 0.0) / 12.0)
        else:
            features.extend([0.0] * 3)

        # DAST-10 features (1 score, normalized)
        if "DAST10" in instrument_scores:
            features.append(instrument_scores["DAST10"].get("total_score", 0.0) / 10.0)
        else:
            features.append(0.0)

        # UPPS-P features (5 subscales, normalized)
        if "UPPS_P" in instrument_scores and "subscale_scores" in instrument_scores["UPPS_P"]:
            upps = instrument_scores["UPPS_P"]["subscale_scores"]
            for key in ["negative_urgency", "lack_premeditation",
                        "lack_perseverance", "sensation_seeking", "positive_urgency"]:
                features.append(upps.get(key, 0.0) / 56.0)
        else:
            features.extend([0.0] * 5)

        # Pad to 22 if needed
        while len(features) < 22:
            features.append(0.0)

        return torch.tensor(features[:22], dtype=torch.float32)

    def compute_research_metrics(
        self,
        features: torch.Tensor,
        model: ImpulseControlModel,
    ) -> Dict[str, Any]:
        """
        Compute research metrics from features.

        Returns a dict of research metrics. These are NOT clinical scores
        and have NO validated clinical interpretation.
        """
        with torch.no_grad():
            outputs = model(features.unsqueeze(0))

        return {
            "drive_intensity": float(outputs["drive_intensity"].item()),
            "control_capacity": float(outputs["control_capacity"].item()),
            "constraint_strength": float(outputs["constraint_strength"].item()),
            "regulation_index": float(outputs["regulation_index"].item()),
            "disclaimer": "RESEARCH METRICS ONLY. NO CLINICAL INTERPRETATION. "
                          "These values do not categorize or predict behavior.",
        }


# =============================================================================
# §6. UNIFIED BEHAVIORAL RESEARCH PIPELINE
# =============================================================================

class BehavioralResearchPipeline:
    """
    Unified pipeline for behavioral research analysis.

    WORKFLOW:
    1. Score validated instruments from provided responses
    2. Extract research features
    3. Run impulse-control model
    4. (Optional) Run transformer for temporal analysis
    5. (Optional) Run world model for trajectory prediction

    ALL OUTPUTS ARE RESEARCH DATA, NOT CLINICAL ASSESSMENTS.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.scorer = InstrumentScorer()
        self.extractor = BehavioralRiskFeatureExtractor()
        self.impulse_model = ImpulseControlModel(input_dim=22).to(self.device)
        self.transformer = BehavioralTransformerEncoder(input_dim=22).to(self.device)
        self.world_model = BehavioralWorldModel(input_dim=22).to(self.device)

    def run_analysis(
        self,
        instrument_responses: Dict[str, List[float]],
        temporal_data: Optional[List[Dict[str, List[float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Run full analysis pipeline.

        Args:
            instrument_responses: dict of instrument key -> item responses
            temporal_data: optional list of repeated assessments over time

        Returns:
            dict with research results
        """
        # Step 1: Score instruments
        scores = self.scorer.score_all(instrument_responses)

        # Step 2: Extract features
        features = self.extractor.extract_features(scores)
        features = features.to(self.device)

        # Step 3: Run impulse-control model
        metrics = self.extractor.compute_research_metrics(features, self.impulse_model)

        result = {
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "RESEARCH DATA ONLY. NOT A CLINICAL ASSESSMENT. "
                          "DO NOT USE FOR DIAGNOSIS, CATEGORIZATION, OR "
                          "ANY CLINICAL/LEGAL/EMPLOYMENT PURPOSE.",
            "instrument_scores": scores,
            "research_metrics": metrics,
        }

        # Step 4: Temporal analysis (if data available)
        if temporal_data and len(temporal_data) > 1:
            temporal_features = []
            for timepoint in temporal_data:
                tp_scores = self.scorer.score_all(timepoint)
                tp_feats = self.extractor.extract_features(tp_scores)
                temporal_features.append(tp_feats)

            seq = torch.stack(temporal_features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                temporal_output, _ = self.transformer(seq)
            result["temporal_analysis"] = {
                "n_timepoints": len(temporal_data),
                "embedding_shape": list(temporal_output.shape),
                "note": "Temporal embedding for research analysis only",
            }

        # Step 5: World model prediction
        with torch.no_grad():
            wm_output = self.world_model(features.unsqueeze(0))
        result["trajectory_prediction"] = {
            "predictions": wm_output["predictions"].cpu().numpy().tolist(),
            "uncertainties": wm_output["uncertainties"].cpu().numpy().tolist(),
            "note": "RESEARCH TRAJECTORY PREDICTION. Does NOT predict behavior. "
                    "Models instrument score trajectories only.",
        }

        return result


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test() -> bool:
    ethics_guard()
    print("=" * 78)
    print(f"  BEHAVIORAL RISK RESEARCH MODULE v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)
    print("  ⚠️  RESEARCH SCAFFOLD — NOT A CLINICAL TOOL")
    print("  ⚠️  DOES NOT CATEGORIZE OR LABEL ANY PERSON'S BEHAVIOR")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ── T1: Instrument configurations ────────────────────────────
    print("\n[T1] Validated instrument configurations")
    rep("BPAQ configured", "BPAQ" in INSTRUMENTS)
    rep("BIS-11 configured", "BIS11" in INSTRUMENTS)
    rep("AUDIT configured", "AUDIT" in INSTRUMENTS)
    rep("DAST-10 configured", "DAST10" in INSTRUMENTS)
    rep("UPPS-P configured", "UPPS_P" in INSTRUMENTS)
    rep("All instruments have references",
        all(i.reference for i in INSTRUMENTS.values()))

    # ── T2: Instrument scoring ───────────────────────────────────
    print("\n[T2] Instrument scoring")
    scorer = InstrumentScorer()

    # BPAQ: 29 items, 5-point Likert (1-5)
    bpaq_responses = [3.0] * 29
    bpaq_result = scorer.score_instrument("BPAQ", bpaq_responses)
    rep("BPAQ scores correctly", bpaq_result["total_score"] == 87.0,
        f"score={bpaq_result['total_score']}")
    rep("BPAQ has subscales", "physical_aggression" in bpaq_result["subscale_scores"])
    rep("BPAQ has research disclaimer",
        "RESEARCH" in bpaq_result["clinical_note"])

    # BIS-11: 30 items, 4-point Likert (1-4)
    bis11_responses = [2.0] * 30
    bis11_result = scorer.score_instrument("BIS11", bis11_responses)
    rep("BIS-11 scores correctly", bis11_result["total_score"] == 60.0)

    # AUDIT: 10 items, 0-4 scale
    audit_responses = [2, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    audit_result = scorer.score_instrument("AUDIT", [float(x) for x in audit_responses])
    rep("AUDIT scores correctly", audit_result["total_score"] == 6.0)
    rep("AUDIT below threshold", audit_result["total_score"] < 8.0)

    # DAST-10: 10 items, binary 0/1
    dast_responses = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    dast_result = scorer.score_instrument("DAST10", [float(x) for x in dast_responses])
    rep("DAST-10 scores correctly", dast_result["total_score"] == 2.0)

    # ── T3: Feature extraction ───────────────────────────────────
    print("\n[T3] Feature extraction")
    extractor = BehavioralRiskFeatureExtractor()
    all_scores = {
        "BPAQ": bpaq_result,
        "BIS11": bis11_result,
        "AUDIT": audit_result,
        "DAST10": dast_result,
    }
    features = extractor.extract_features(all_scores)
    rep("Feature vector shape", features.shape == (22,),
        f"shape={tuple(features.shape)}")
    rep("Features normalized", float(features.max()) <= 1.0)

    # ── T4: Impulse-control model ────────────────────────────────
    print("\n[T4] Impulse-control model (computational metaphor)")
    model = ImpulseControlModel(input_dim=22)
    with torch.no_grad():
        outputs = model(features.unsqueeze(0))
    rep("Drive intensity in [0,1]",
        0 <= outputs["drive_intensity"].item() <= 1)
    rep("Control capacity in [0,1]",
        0 <= outputs["control_capacity"].item() <= 1)
    rep("Constraint strength in [0,1]",
        0 <= outputs["constraint_strength"].item() <= 1)
    rep("Regulation index positive",
        outputs["regulation_index"].item() > 0)

    # ── T5: Transformer encoder ──────────────────────────────────
    print("\n[T5] Transformer encoder for temporal data")
    transformer = BehavioralTransformerEncoder(input_dim=22)
    seq = torch.randn(2, 10, 22)  # batch=2, seq_len=10
    with torch.no_grad():
        out, _ = transformer(seq)
    rep("Transformer output shape", out.shape == (2, 10, 128),
        f"shape={tuple(out.shape)}")

    # ── T6: World model ──────────────────────────────────────────
    print("\n[T6] World model for trajectory prediction")
    world_model = BehavioralWorldModel(input_dim=22)
    state = torch.randn(1, 22)
    with torch.no_grad():
        wm_out = world_model(state)
    rep("World model predictions shape",
        wm_out["predictions"].shape == (1, 5, 3),
        f"shape={tuple(wm_out['predictions'].shape)}")
    rep("Uncertainties positive",
        bool((wm_out["uncertainties"] >= 0).all()))

    # ── T7: Full pipeline ────────────────────────────────────────
    print("\n[T7] Full research pipeline")
    pipeline = BehavioralResearchPipeline()
    responses = {
        "BPAQ": [3.0] * 29,
        "BIS11": [2.0] * 30,
        "AUDIT": [2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "DAST10": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    result = pipeline.run_analysis(responses)
    rep("Pipeline produces result", "research_metrics" in result)
    rep("Result has disclaimer", "RESEARCH" in result["disclaimer"])
    rep("Trajectory prediction included", "trajectory_prediction" in result)

    # ── T8: Ethical guards ───────────────────────────────────────
    print("\n[T8] Ethical guards")
    rep("No categorization output", "category" not in str(result.keys()).lower())
    rep("No behavior prediction claim",
        "predict behavior" not in str(result).lower())
    rep("Research disclaimer present",
        "RESEARCH" in result.get("disclaimer", ""))

    # ── T9: Gradient flow ────────────────────────────────────────
    print("\n[T9] Gradient flow")
    features_grad = features.clone().requires_grad_(True)
    out = model(features_grad.unsqueeze(0))
    loss = out["integrated_output"].sum()
    loss.backward()
    rep("Gradients flow to input", features_grad.grad is not None)
    rep("No NaN gradients", not torch.isnan(features_grad.grad).any().item())

    # ── T10: Empty/missing data handling ─────────────────────────
    print("\n[T10] Missing data handling")
    empty_features = extractor.extract_features({})
    rep("Empty scores produce zero features", float(empty_features.abs().sum()) == 0.0)

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THIS MODULE PROVIDES
    • Validated instrument scoring (BPAQ, BIS-11, AUDIT, DAST-10, UPPS-P)
    • Impulse-control computational metaphor (Id/Ego/Superego labels)
    • Transformer encoder for temporal research analysis
    • World model for trajectory prediction (instrument scores only)
    • Research feature extraction

  WHAT THIS MODULE IS NOT
    ❌ NOT a diagnostic tool
    ❌ NOT a categorization system for sexual behavior
    ❌ NOT a violence risk assessment tool
    ❌ NOT validated against clinical outcomes
    ❌ NOT approved for clinical, legal, or employment use
    ❌ Does NOT replace psychiatric evaluation

  CRITICAL ETHICAL REQUIREMENTS
    1. IRB / Ethics Committee approval REQUIRED
    2. Informed consent from every participant REQUIRED
    3. Clinical oversight by qualified psychiatrist REQUIRED
    4. Data de-identification (HIPAA / GDPR / Thai PDPA) REQUIRED
    5. MUST NOT be used for discrimination of any kind

  THE PSYCHIATRIST IS THE FINAL DECISION MAKER. ALWAYS.
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Risk Research Module (RESEARCH ONLY)")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if self_test() else 1)

    parser.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
