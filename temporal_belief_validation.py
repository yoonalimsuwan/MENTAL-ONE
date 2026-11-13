#!/usr/bin/env python3
"""
=============================================================================
TEMPORAL BELIEF MODELING + REAL-DATA VALIDATION HARNESS
LSTM/Transformer for Psyche State Trajectories · DAIC-WOZ · Dreaddit
=============================================================================
Developers : PAI and Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

WHAT "TEMPORAL BELIEF" MEANS HERE (read this first)
---------------------------------------------------
In the Free Energy Principle / active inference literature, a "belief
state" is a probability distribution over hidden states — NOT a
colloquial "belief" or "opinion." This module models how the PSY ONE
psyche state vector evolves over time:

    b(t) = [H(𝓘), L_𝓢, ℱ, policy, FAA, GFP, LZ, PLV, ...]

The LSTM/Transformer learns p(b(t+1) | b(1:t)) — the temporal dynamics
of the computational psyche state. This is a RESEARCH TOOL for
understanding state trajectories, NOT a mind-reading device.

DATASETS (what they actually contain)
--------------------------------------
DAIC-WOZ [Gratch et al., 2014]:
  • Clinical interview video/audio/text from ~189 participants
  • PHQ-8 depression scores as labels
  • NOT freely downloadable — requires USC access request + IRB
  • Does NOT contain raw EEG — derived features only
  • This adapter maps DAIC-WOZ interview features → psyche state vectors

Dreaddit [Turc et al., 2019]:
  • ~3,500 Reddit posts labeled for anxiety (0/1)
  • Text only — no physiological signals
  • More accessible than DAIC-WOZ but lower fidelity
  • This adapter maps text sentiment features → psyche state proxy

WHAT THIS MODULE PROVIDES
  §1  TemporalBeliefModel — LSTM + Transformer hybrid for state sequences
  §2  DAICWOZAdapter — feature extraction + psyche state mapping
  §3  DreadditAdapter — text → sentiment → psyche state proxy
  §4  ValidationHarness — metrics, baselines, statistical tests
  §5  Self-test suite

WHAT THIS MODULE IS NOT
  ❌ Not a diagnostic tool
  ❌ Not validated for clinical use
  ❌ Not a mind reader
  ❌ DAIC-WOZ data NOT included (must be obtained separately)
  ❌ Dreaddit data NOT included (must be downloaded separately)
  ❌ No claim of SOTA or paradigm shift

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("temporal_belief")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# §0. PSYCHE STATE VECTOR SPECIFICATION
# =============================================================================

# The PSY ONE triad produces these features per time step.
# This is the canonical feature ordering used throughout this module.
PSYCHE_FEATURE_NAMES = [
    "id_entropy",           # H(𝓘) — Id Shannon entropy (bits)
    "accumulated_entropy",  # ∫H(𝓘)dt — accumulated Id drive
    "superego_loss",        # L_𝓢 — KL(π‖π_norm)
    "behavioral_entropy",   # H(B) — behavioral deviation entropy
    "free_energy",          # ℱ — variational free energy
    "selected_action",      # a* — argmax of policy (normalized)
    "convergence_speed",    # α — Ego optimization speed
    "faa_proxy",            # frontal alpha asymmetry proxy
    "gfp_proxy",            # global field power proxy
    "lz_complexity",        # Lempel-Ziv complexity proxy
    "plv_f3_f4",            # phase-locking value proxy
    "neuroplasticity_idx",  # neuroplasticity index
]
PSYCHE_DIM = len(PSYCHE_FEATURE_NAMES)  # 12


def psyche_state_to_vector(state_dict: Dict[str, Any]) -> np.ndarray:
    """Convert a PSY ONE PsycheTriadState dict to a fixed-length vector."""
    vec = np.zeros(PSYCHE_DIM, dtype=np.float32)
    vec[0] = state_dict.get("id_entropy", 0.0)
    vec[1] = state_dict.get("accumulated_entropy", 0.0)
    vec[2] = state_dict.get("superego_loss", 0.0)
    vec[3] = state_dict.get("behavioral_entropy", 0.0)
    vec[4] = state_dict.get("free_energy", 0.0)
    vec[5] = state_dict.get("selected_action", 0) / 10.0  # normalize
    vec[6] = state_dict.get("convergence_speed", 0.0)
    neuro = state_dict.get("neurophysio_map", {})
    vec[7] = neuro.get("FAA_proxy", 0.0)
    vec[8] = neuro.get("GFP_proxy", 0.0)
    vec[9] = neuro.get("LZ_complexity", 0.0)
    vec[10] = neuro.get("PLV_F3_F4_proxy", 0.0)
    vec[11] = neuro.get("neuroplasticity_idx", 0.0)
    return vec


# =============================================================================
# §1. TEMPORAL BELIEF MODEL — LSTM + Transformer Hybrid
# =============================================================================

class TemporalBeliefEncoder(nn.Module):
    """
    Encodes a sequence of psyche state vectors into a temporal embedding.

    Architecture:
        Input (T, D) → LSTM (bidirectional) → Transformer Encoder → (D_model,)

    The LSTM captures local sequential dependencies.
    The Transformer captures long-range temporal attention patterns.
    """

    def __init__(
        self,
        input_dim: int = PSYCHE_DIM,
        hidden_dim: int = 128,
        n_lstm_layers: int = 2,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (learnable, capped at max_seq_len)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # LSTM encoder (bidirectional)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # Temporal pooling
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, T, D) batch of psyche state sequences
            lengths: (B,) optional sequence lengths for packing
        Returns:
            (B, hidden_dim) temporal embedding
        """
        B, T, D = x.shape

        # Project input
        h = self.input_proj(x)  # (B, T, hidden)

        # Add positional encoding (truncate to actual sequence length)
        h = h + self.pos_embed[:, :T, :]

        # LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, _) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, _) = self.lstm(h)

        # Transformer (over LSTM output)
        # Create attention mask for padding
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None

        trans_out = self.transformer(lstm_out, src_key_padding_mask=mask)

        # Temporal pooling: attention-weighted mean
        pool_weights = torch.softmax(
            self.pool_proj(trans_out).sum(-1), dim=-1
        )  # (B, T)
        if mask is not None:
            pool_weights = pool_weights.masked_fill(mask, 0.0)
            pool_weights = pool_weights / pool_weights.sum(-1, keepdim=True).clamp(min=1e-8)

        pooled = torch.bmm(pool_weights.unsqueeze(1), trans_out).squeeze(1)  # (B, hidden)
        return pooled


class TemporalBeliefPredictor(nn.Module):
    """
    Predicts the next psyche state from a sequence.

    p(b(t+1) | b(1:t)) — the core temporal belief model.
    """

    def __init__(
        self,
        encoder: TemporalBeliefEncoder,
        output_dim: int = PSYCHE_DIM,
        n_future_steps: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_future_steps = n_future_steps
        hidden = encoder.hidden_dim

        self.predictor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, output_dim * n_future_steps),
        )

        # Uncertainty head: predict variance of the prediction
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, output_dim * n_future_steps),
            nn.Softplus(),  # variance must be positive
        )

    def forward(
        self, x: Tensor, lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            pred_mean: (B, n_future_steps, D) predicted next states
            pred_var:  (B, n_future_steps, D) prediction variances
        """
        z = self.encoder(x, lengths)  # (B, hidden)
        mean_flat = self.predictor(z)
        var_flat = self.uncertainty_head(z) + 1e-6  # numerical stability

        D = self.encoder.input_dim
        pred_mean = mean_flat.view(-1, self.n_future_steps, D)
        pred_var = var_flat.view(-1, self.n_future_steps, D)
        return pred_mean, pred_var


class TemporalBeliefClassifier(nn.Module):
    """
    Classifies a psyche state trajectory into a diagnostic category.

    This is a RESEARCH classifier, not a diagnostic tool.
    """

    DIAGNOSTIC_LABELS = [
        "Healthy", "MDD", "Bipolar", "Schizophrenia", "PTSD",
        "Panic", "Anxiety", "OCD", "Conversion", "Dissociative",
    ]

    def __init__(
        self,
        encoder: TemporalBeliefEncoder,
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes or len(self.DIAGNOSTIC_LABELS)
        hidden = encoder.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, self.n_classes),
        )

    def forward(
        self, x: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """Returns (B, n_classes) logits."""
        z = self.encoder(x, lengths)
        return self.classifier(z)


class TemporalBeliefModel(nn.Module):
    """
    Unified temporal belief model combining prediction and classification.

    This is the main entry point for temporal psyche state modeling.
    """

    def __init__(
        self,
        input_dim: int = PSYCHE_DIM,
        hidden_dim: int = 128,
        n_lstm_layers: int = 2,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        n_future_steps: int = 1,
        n_classes: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TemporalBeliefEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_lstm_layers=n_lstm_layers,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )
        self.predictor = TemporalBeliefPredictor(
            self.encoder, input_dim, n_future_steps
        )
        self.classifier = TemporalBeliefClassifier(
            self.encoder, n_classes
        )

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
        task: str = "both",
    ) -> Dict[str, Tensor]:
        """
        Args:
            x: (B, T, D) psyche state sequences
            lengths: (B,) sequence lengths
            task: "predict", "classify", or "both"
        Returns:
            dict with "pred_mean", "pred_var", "class_logits" as applicable
        """
        out = {}
        if task in ("predict", "both"):
            pred_mean, pred_var = self.predictor(x, lengths)
            out["pred_mean"] = pred_mean
            out["pred_var"] = pred_var
        if task in ("classify", "both"):
            logits = self.classifier(x, lengths)
            out["class_logits"] = logits
        return out


# =============================================================================
# §2. DAIC-WOZ ADAPTER
# =============================================================================

class DAICWOZAdapter:
    """
    Adapter for the DAIC-WOZ dataset [Gratch et al., 2014].

    DAIC-WOZ contains:
      • Clinical interview video/audio/text
      • PHQ-8 depression scores
      • Participant metadata

    IMPORTANT: DAIC-WOZ does NOT contain raw EEG data.
    This adapter maps interview-derived features to the PSY ONE
    psyche state space. The mapping is a RESEARCH APPROXIMATION.

    Data Access:
      DAIC-WOZ requires a data use agreement from USC.
      Visit: https://dcapswoz.ict.usc.edu/
      You must have IRB approval before using this data.

    Feature mapping (interview → psyche state):
      • Speech rate, pause duration → id_entropy proxy
      • Sentiment of responses → superego_loss proxy
      • Linguistic complexity → lz_complexity proxy
      • Vocal pitch variation → faa_proxy
      • PHQ-8 score → diagnostic label
    """

    # DAIC-WOZ metadata fields
    META_FIELDS = [
        "Participant_ID", "Gender", "Age", "PHQ8_Score",
        "Depressed", "Interview_Time",
    ]

    # Feature columns from DAIC-WOZ audio/text features
    AUDIO_FEATURES = [
        "speech_rate", "pause_duration", "pitch_mean", "pitch_std",
        "energy_mean", "energy_std", "jitter", "shimmer",
    ]
    TEXT_FEATURES = [
        "word_count", "unique_words", "sentiment_pos", "sentiment_neg",
        "first_person_pronouns", "negative_emotion_words",
    ]

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.metadata: Optional[Dict] = None
        self.features: Optional[Dict] = None

    def check_data_available(self) -> bool:
        """Check if DAIC-WOZ data is available locally."""
        if self.data_dir is None:
            return False
        meta_file = self.data_dir / "metadata.csv"
        return meta_file.exists()

    def load_metadata(self, meta_path: str) -> Dict[str, Any]:
        """
        Load DAIC-WOZ metadata from CSV.

        Expected columns: Participant_ID, Gender, Age, PHQ8_Score, Depressed
        """
        import csv
        metadata = {}
        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("Participant_ID", row.get("participant_id", ""))
                metadata[pid] = {
                    "phq8_score": float(row.get("PHQ8_Score", 0)),
                    "depressed": int(row.get("Depressed", 0)),
                    "gender": row.get("Gender", "unknown"),
                    "age": float(row.get("Age", 0)),
                }
        self.metadata = metadata
        log.info(f"[DAIC-WOZ] Loaded metadata for {len(metadata)} participants")
        return metadata

    def interview_features_to_psyche(
        self,
        audio_feats: Dict[str, float],
        text_feats: Dict[str, float],
        phq8_score: float,
    ) -> np.ndarray:
        """
        Map DAIC-WOZ interview features to PSY ONE psyche state vector.

        THIS IS A RESEARCH APPROXIMATION, NOT A VALIDATED MAPPING.
        The mapping is based on face-validity heuristics, not empirical
        calibration. Do not use for clinical purposes.
        """
        vec = np.zeros(PSYCHE_DIM, dtype=np.float32)

        # id_entropy: speech rate variability + pause irregularity
        speech_rate = audio_feats.get("speech_rate", 0.0)
        pause_dur = audio_feats.get("pause_duration", 0.0)
        vec[0] = min(4.0, abs(speech_rate - 120) / 30.0 + pause_dur / 2.0)

        # accumulated_entropy: proxy from interview length
        vec[1] = vec[0] * 1.5  # rough accumulation proxy

        # superego_loss: negative sentiment + first-person pronoun overuse
        sent_neg = text_feats.get("sentiment_neg", 0.0)
        first_person = text_feats.get("first_person_pronouns", 0.0)
        vec[2] = min(3.0, sent_neg * 2.0 + first_person * 0.5)

        # behavioral_entropy: response variability
        vec[3] = min(1.0, audio_feats.get("jitter", 0.0) * 10.0)

        # free_energy: overall distress proxy
        vec[4] = phq8_score / 24.0 * 5.0  # PHQ-8 max is 24

        # selected_action: not applicable for interview data
        vec[5] = 0.0

        # convergence_speed: speech fluency proxy
        vec[6] = max(0.0, 1.0 - pause_dur / 5.0)

        # faa_proxy: pitch asymmetry (very rough proxy)
        pitch_std = audio_feats.get("pitch_std", 0.0)
        vec[7] = pitch_std / 100.0

        # gfp_proxy: energy level
        vec[8] = audio_feats.get("energy_mean", 0.0) / 100.0

        # lz_complexity: linguistic diversity
        unique_ratio = text_feats.get("unique_words", 0) / max(
            text_feats.get("word_count", 1), 1
        )
        vec[9] = unique_ratio

        # plv_f3_f4: not available from interview
        vec[10] = 0.0

        # neuroplasticity_idx: inverse of distress
        vec[11] = max(0.0, 1.0 - phq8_score / 24.0)

        return vec

    def participant_to_trajectory(
        self,
        participant_id: str,
        features_by_timepoint: List[Dict],
    ) -> np.ndarray:
        """
        Convert a participant's multi-timepoint features to a trajectory.

        Args:
            participant_id: DAIC-WOZ participant ID
            features_by_timepoint: list of dicts, each with audio/text features

        Returns:
            (T, PSYCHE_DIM) trajectory array
        """
        if self.metadata is None or participant_id not in self.metadata:
            raise ValueError(f"No metadata for participant {participant_id}")

        phq8 = self.metadata[participant_id]["phq8_score"]
        trajectory = []
        for feats in features_by_timepoint:
            audio = {k: feats.get(k, 0.0) for k in self.AUDIO_FEATURES}
            text = {k: feats.get(k, 0.0) for k in self.TEXT_FEATURES}
            vec = self.interview_features_to_psyche(audio, text, phq8)
            trajectory.append(vec)

        return np.stack(trajectory) if trajectory else np.zeros((1, PSYCHE_DIM))


# =============================================================================
# §3. DREADDIT ADAPTER
# =============================================================================

class DreadditAdapter:
    """
    Adapter for the Dreaddit dataset [Turc et al., 2019].

    Dreaddit contains:
      • ~3,500 Reddit posts labeled for anxiety (0/1)
      • Text only — no physiological signals
      • Labels: 0 = no anxiety, 1 = anxiety

    Data Access:
      Dreaddit is available from the original paper's repository.
      Check: https://github.com/aleksandrdreaddit (verify current URL)

    Feature mapping (text → psyche state proxy):
      • Sentiment polarity → superego_loss proxy
      • Lexical diversity → lz_complexity proxy
      • Anxiety word count → id_entropy proxy
      • Post length → accumulated_entropy proxy

    IMPORTANT: This mapping is a ROUGH PROXY. Reddit text is not
    a physiological signal. The psyche state vectors derived here
    are very approximate and should only be used for testing the
    temporal model's ability to learn from sequential text features.
    """

    # Anxiety-related word lists (from LIWC-style categories)
    ANXIETY_WORDS = {
        "anxious", "worried", "nervous", "scared", "afraid", "panic",
        "stress", "stressed", "overwhelmed", "fear", "dread", "tense",
        "uneasy", "apprehensive", "restless", "agitated",
    }
    DEPRESSION_WORDS = {
        "sad", "depressed", "hopeless", "worthless", "empty", "tired",
        "exhausted", "numb", "lonely", "isolated", "helpless", "gloomy",
    }
    NEGATIVE_WORDS = ANXIETY_WORDS | DEPRESSION_WORDS

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None

    def check_data_available(self) -> bool:
        if self.data_dir is None:
            return False
        return (self.data_dir / "dreaddit.csv").exists() or \
               (self.data_dir / "train.csv").exists()

    def text_to_features(self, text: str) -> Dict[str, float]:
        """Extract simple text features from a Reddit post."""
        words = text.lower().split()
        n_words = max(len(words), 1)
        unique_words = len(set(words))

        # Anxiety word density
        anxiety_count = sum(1 for w in words if w in self.ANXIETY_WORDS)
        depression_count = sum(1 for w in words if w in self.DEPRESSION_WORDS)
        negative_count = anxiety_count + depression_count

        # Simple sentiment (very rough)
        positive_words = {"good", "great", "happy", "love", "wonderful",
                          "amazing", "fantastic", "excellent", "hopeful"}
        positive_count = sum(1 for w in words if w in positive_words)

        return {
            "word_count": n_words,
            "unique_word_ratio": unique_words / n_words,
            "anxiety_density": anxiety_count / n_words,
            "depression_density": depression_count / n_words,
            "negative_density": negative_count / n_words,
            "positive_density": positive_count / n_words,
            "avg_word_length": sum(len(w) for w in words) / n_words,
            "exclamation_rate": text.count("!") / n_words,
            "question_rate": text.count("?") / n_words,
        }

    def text_to_psyche_state(self, text: str, label: int = 0) -> np.ndarray:
        """
        Map a Dreaddit text post to a PSY ONE psyche state vector.

        THIS IS A VERY ROUGH PROXY. Reddit text is not a physiological
        signal. Use only for testing the temporal model architecture.
        """
        feats = self.text_to_features(text)
        vec = np.zeros(PSYCHE_DIM, dtype=np.float32)

        # id_entropy: anxiety + negative word density
        vec[0] = min(4.0, feats["anxiety_density"] * 20.0 +
                     feats["negative_density"] * 10.0)

        # accumulated_entropy: post length proxy
        vec[1] = min(10.0, math.log(feats["word_count"] + 1) * 1.5)

        # superego_loss: negative sentiment
        vec[2] = min(3.0, feats["negative_density"] * 15.0)

        # behavioral_entropy: exclamation + question rate
        vec[3] = min(1.0, (feats["exclamation_rate"] +
                           feats["question_rate"]) * 10.0)

        # free_energy: overall distress
        vec[4] = min(8.0, feats["negative_density"] * 30.0 + label * 2.0)

        # selected_action: label proxy
        vec[5] = label / 10.0

        # convergence_speed: inverse of distress
        vec[6] = max(0.0, 1.0 - feats["negative_density"] * 10.0)

        # faa_proxy: not available
        vec[7] = 0.0

        # gfp_proxy: not available
        vec[8] = 0.0

        # lz_complexity: lexical diversity
        vec[9] = feats["unique_word_ratio"]

        # plv_f3_f4: not available
        vec[10] = 0.0

        # neuroplasticity_idx: inverse of distress
        vec[11] = max(0.0, 1.0 - feats["negative_density"] * 5.0)

        return vec

    def load_dreaddit(self, csv_path: str) -> List[Dict]:
        """
        Load Dreaddit dataset from CSV.

        Expected columns: text, label (0=no anxiety, 1=anxiety)
        """
        import csv
        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "")
                label = int(row.get("label", 0))
                if text:
                    records.append({"text": text, "label": label})
        log.info(f"[Dreaddit] Loaded {len(records)} posts")
        return records


# =============================================================================
# §4. VALIDATION HARNESS
# =============================================================================

@dataclass
class ValidationResult:
    """Result from a single validation run."""
    dataset: str
    model: str
    metric: str
    value: float
    baseline: Optional[float] = None
    p_value: Optional[float] = None
    n_samples: int = 0
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class ValidationHarness:
    """
    Validation harness for temporal belief models.

    Compares the LSTM/Transformer model against baselines:
      • Persistence baseline: predict b(t+1) = b(t)
      • Linear baseline: predict b(t+1) = W·b(t) + c
      • Mean baseline: predict b(t+1) = mean(b(1:t))

    Metrics:
      • MSE on predicted next state
      • Trajectory correlation
      • Classification accuracy (if labels available)
      • Calibration of uncertainty estimates

    STATISTICAL NOTE: With typical dataset sizes (DAIC-WOZ ~189
    participants, Dreaddit ~3500 posts), results are PRELIMINARY.
    Proper validation requires:
      • Cross-validation (k-fold, leave-one-participant-out)
      • Statistical significance testing
      • External replication on independent datasets
      • Clinical expert review of any findings
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.results: List[ValidationResult] = []

    # ---- Baselines ----
    def persistence_baseline(
        self, trajectory: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Predict b(t+1) = b(t). Returns predictions and MSE."""
        preds = trajectory[:-1]  # predict next = current
        actuals = trajectory[1:]
        mse = float(np.mean((preds - actuals) ** 2))
        return preds, mse

    def linear_baseline(
        self, trajectory: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Fit b(t+1) = W·b(t) + c via least squares."""
        X = trajectory[:-1]
        y = trajectory[1:]
        # Add bias
        X_bias = np.hstack([X, np.ones((len(X), 1))])
        try:
            W, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
            preds = X_bias @ W
            mse = float(np.mean((preds - y) ** 2))
        except np.linalg.LinAlgError:
            preds = X  # fallback to persistence
            mse = float(np.mean((preds - y) ** 2))
        return preds, mse

    def mean_baseline(self, trajectory: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict b(t+1) = mean(b(1:t))."""
        running_mean = np.cumsum(trajectory, axis=0) / np.arange(
            1, len(trajectory) + 1
        )[:, None]
        preds = running_mean[:-1]
        actuals = trajectory[1:]
        mse = float(np.mean((preds - actuals) ** 2))
        return preds, mse

    # ---- Model evaluation ----
    def evaluate_predictor(
        self,
        model: TemporalBeliefModel,
        trajectories: List[np.ndarray],
        seq_len: int = 20,
    ) -> Dict[str, float]:
        """
        Evaluate the temporal predictor on a list of trajectories.

        Uses a sliding window of size seq_len.
        """
        model.eval()
        all_preds = []
        all_actuals = []
        all_vars = []

        with torch.no_grad():
            for traj in trajectories:
                if len(traj) < seq_len + 1:
                    continue
                for start in range(0, len(traj) - seq_len, seq_len // 2):
                    window = traj[start : start + seq_len]
                    actual_next = traj[start + seq_len]

                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                    x = x.to(self.device)

                    out = model(x, task="predict")
                    pred = out["pred_mean"][0, 0].cpu().numpy()
                    var = out["pred_var"][0, 0].cpu().numpy()

                    all_preds.append(pred)
                    all_actuals.append(actual_next)
                    all_vars.append(var)

        if not all_preds:
            return {"mse": float("nan"), "n_windows": 0}

        preds = np.stack(all_preds)
        actuals = np.stack(all_actuals)
        vars_ = np.stack(all_vars)

        mse = float(np.mean((preds - actuals) ** 2))
        mae = float(np.mean(np.abs(preds - actuals)))

        # Calibration: fraction of actuals within 2σ of prediction
        within_2sigma = np.mean(
            np.abs(preds - actuals) < 2.0 * np.sqrt(vars_ + 1e-8)
        )

        # Per-feature MSE
        per_feature_mse = np.mean((preds - actuals) ** 2, axis=0)

        return {
            "mse": mse,
            "mae": mae,
            "within_2sigma": float(within_2sigma),
            "per_feature_mse": per_feature_mse.tolist(),
            "n_windows": len(all_preds),
        }

    def evaluate_classifier(
        self,
        model: TemporalBeliefModel,
        trajectories: List[np.ndarray],
        labels: List[int],
        seq_len: int = 20,
    ) -> Dict[str, float]:
        """
        Evaluate the trajectory classifier.

        Args:
            trajectories: list of (T, D) arrays
            labels: list of integer class labels
        """
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for traj, label in zip(trajectories, labels):
                if len(traj) < seq_len:
                    # Pad short trajectories
                    pad_len = seq_len - len(traj)
                    traj = np.vstack([traj, np.zeros((pad_len, traj.shape[1]))])

                window = traj[:seq_len]
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                x = x.to(self.device)

                out = model(x, task="classify")
                all_logits.append(out["class_logits"][0].cpu())
                all_labels.append(label)

        if not all_logits:
            return {"accuracy": float("nan"), "n_samples": 0}

        logits = torch.stack(all_logits)
        labels_t = torch.tensor(all_labels, dtype=torch.long)

        preds = logits.argmax(dim=-1)
        accuracy = float((preds == labels_t).float().mean())

        # Per-class accuracy
        n_classes = logits.shape[-1]
        per_class_acc = {}
        for c in range(n_classes):
            mask = labels_t == c
            if mask.any():
                per_class_acc[c] = float((preds[mask] == c).float().mean())

        return {
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc,
            "n_samples": len(all_labels),
        }

    # ---- Full validation run ----
    def run_validation(
        self,
        model: TemporalBeliefModel,
        trajectories: List[np.ndarray],
        labels: Optional[List[int]] = None,
        dataset_name: str = "unknown",
        seq_len: int = 20,
    ) -> Dict[str, Any]:
        """
        Run full validation: baselines + model + statistical comparison.
        """
        log.info(f"[Validation] Running on {dataset_name}: "
                 f"{len(trajectories)} trajectories")

        results = {}

        # Baselines
        baseline_mses = {"persistence": [], "linear": [], "mean": []}
        for traj in trajectories:
            if len(traj) < 2:
                continue
            _, mse_p = self.persistence_baseline(traj)
            _, mse_l = self.linear_baseline(traj)
            _, mse_m = self.mean_baseline(traj)
            baseline_mses["persistence"].append(mse_p)
            baseline_mses["linear"].append(mse_l)
            baseline_mses["mean"].append(mse_m)

        results["baselines"] = {
            k: float(np.mean(v)) if v else float("nan")
            for k, v in baseline_mses.items()
        }

        # Model prediction
        pred_results = self.evaluate_predictor(model, trajectories, seq_len)
        results["prediction"] = pred_results

        # Model classification (if labels available)
        if labels is not None:
            cls_results = self.evaluate_classifier(
                model, trajectories, labels, seq_len
            )
            results["classification"] = cls_results

        # Comparison
        model_mse = pred_results.get("mse", float("nan"))
        best_baseline = min(results["baselines"].values())
        results["model_vs_best_baseline"] = (
            model_mse / best_baseline if best_baseline > 0 else float("nan")
        )
        results["improvement_pct"] = (
            (1 - model_mse / best_baseline) * 100
            if best_baseline > 0 else float("nan")
        )

        self.results.append(ValidationResult(
            dataset=dataset_name,
            model="LSTM+Transformer",
            metric="next_state_MSE",
            value=model_mse,
            baseline=best_baseline,
            n_samples=len(trajectories),
            note=f"improvement={results['improvement_pct']:.1f}%",
        ))

        return results

    def print_report(self) -> None:
        """Print a summary of all validation results."""
        print("\n" + "=" * 72)
        print("  VALIDATION REPORT — Temporal Belief Model")
        print(f"  {ATTRIBUTION}")
        print("=" * 72)
        print(f"  {'Dataset':<20} {'Metric':<20} {'Model':>10} "
              f"{'Baseline':>10} {'Improve':>10}")
        print("  " + "-" * 72)
        for r in self.results:
            improve = ""
            if r.baseline and r.baseline > 0:
                pct = (1 - r.value / r.baseline) * 100
                improve = f"{pct:+.1f}%"
            print(f"  {r.dataset:<20} {r.metric:<20} "
                  f"{r.value:>10.4f} {r.baseline or 0:>10.4f} {improve:>10}")
        print("=" * 72)
        print("""
  HONEST NOTES:
    • Results are PRELIMINARY — no external replication
    • DAIC-WOZ mapping is a research approximation, not validated
    • Dreaddit text→psyche mapping is a very rough proxy
    • No clinical interpretation should be made from these results
    • Statistical significance requires larger samples and proper
      multiple-comparison correction
""")


# =============================================================================
# §5. SYNTHETIC DATA GENERATOR (for testing without real data)
# =============================================================================

def generate_synthetic_trajectories(
    n_trajectories: int = 100,
    seq_len: int = 50,
    n_classes: int = 3,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate synthetic psyche state trajectories for testing.

    Each trajectory is a random walk with class-dependent drift.
    This is NOT real data — it's for testing the model architecture.
    """
    rng = np.random.default_rng(seed)
    trajectories = []
    labels = []

    # Class-specific drift vectors
    drifts = rng.randn(n_classes, PSYCHE_DIM) * 0.1

    for i in range(n_trajectories):
        label = i % n_classes
        drift = drifts[label]

        # Random walk with drift
        state = rng.randn(PSYCHE_DIM) * 0.5
        traj = [state.copy()]
        for t in range(seq_len - 1):
            noise = rng.randn(PSYCHE_DIM) * 0.05
            state = state + drift + noise
            # Keep values in reasonable range
            state = np.clip(state, -10, 10)
            traj.append(state.copy())

        trajectories.append(np.stack(traj))
        labels.append(label)

    return trajectories, labels


# =============================================================================
# §6. TRAINING LOOP
# =============================================================================

def train_temporal_model(
    model: TemporalBeliefModel,
    trajectories: List[np.ndarray],
    labels: Optional[List[int]] = None,
    epochs: int = 50,
    batch_size: int = 16,
    seq_len: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    task: str = "predict",
) -> Dict[str, List[float]]:
    """
    Train the temporal belief model.

    Args:
        model: TemporalBeliefModel instance
        trajectories: list of (T, D) arrays
        labels: optional class labels for classification task
        task: "predict", "classify", or "both"

    Returns:
        dict with training history
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {"loss": [], "pred_loss": [], "cls_loss": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle trajectories
        indices = np.random.permutation(len(trajectories))

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            batch_trajs = [trajectories[i] for i in batch_idx]

            # Prepare batch: extract windows
            windows = []
            targets = []
            batch_labels = []

            for traj in batch_idx:
                t = trajectories[traj]
                if len(t) < seq_len + 1:
                    continue
                start = np.random.randint(0, len(t) - seq_len)
                window = t[start : start + seq_len]
                target = t[start + seq_len]
                windows.append(window)
                targets.append(target)
                if labels is not None:
                    batch_labels.append(labels[traj])

            if not windows:
                continue

            x = torch.tensor(
                np.stack(windows), dtype=torch.float32
            ).to(dev)
            y = torch.tensor(
                np.stack(targets), dtype=torch.float32
            ).to(dev)

            optimizer.zero_grad()

            if task == "predict":
                out = model(x, task="predict")
                pred_mean = out["pred_mean"][:, 0, :]  # first future step
                pred_var = out["pred_var"][:, 0, :]

                # Gaussian NLL loss
                loss = F.gaussian_nll_loss(
                    pred_mean, y, pred_var.clamp(min=1e-6)
                )

            elif task == "classify" and labels is not None:
                y_cls = torch.tensor(batch_labels, dtype=torch.long).to(dev)
                out = model(x, task="classify")
                loss = F.cross_entropy(out["class_logits"], y_cls)

            else:  # both
                out = model(x, task="both")
                pred_mean = out["pred_mean"][:, 0, :]
                pred_var = out["pred_var"][:, 0, :]
                pred_loss = F.gaussian_nll_loss(
                    pred_mean, y, pred_var.clamp(min=1e-6)
                )
                if labels is not None and batch_labels:
                    y_cls = torch.tensor(batch_labels, dtype=torch.long).to(dev)
                    cls_loss = F.cross_entropy(out["class_logits"], y_cls)
                    loss = pred_loss + 0.5 * cls_loss
                    history["cls_loss"].append(float(cls_loss))
                else:
                    loss = pred_loss
                    history["cls_loss"].append(0.0)
                history["pred_loss"].append(float(pred_loss))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            log.info(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f}")

    return history


# =============================================================================
# §7. SELF-TEST SUITE
# =============================================================================

def self_test() -> bool:
    print("=" * 78)
    print(f"  TEMPORAL BELIEF MODEL v{VERSION} — Self-Test")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)
    print("  ⚠️  RESEARCH TOOL — NOT A DIAGNOSTIC DEVICE")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: Model construction
    print("\n[T1] Model construction")
    model = TemporalBeliefModel(
        input_dim=PSYCHE_DIM,
        hidden_dim=64,
        n_lstm_layers=2,
        n_heads=4,
        n_transformer_layers=2,
        n_future_steps=1,
        n_classes=3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    rep("Model constructs", model is not None, f"{n_params:,} params")

    # T2: Forward pass shapes
    print("\n[T2] Forward pass shapes")
    B, T = 4, 20
    x = torch.randn(B, T, PSYCHE_DIM)
    lengths = torch.tensor([T, T, T // 2, T // 4])

    out_both = model(x, lengths, task="both")
    rep("Predict output shape",
        out_both["pred_mean"].shape == (B, 1, PSYCHE_DIM),
        f"{tuple(out_both['pred_mean'].shape)}")
    rep("Predict variance shape",
        out_both["pred_var"].shape == (B, 1, PSYCHE_DIM))
    rep("Classify output shape",
        out_both["class_logits"].shape == (B, 3),
        f"{tuple(out_both['class_logits'].shape)}")
    rep("Variance positive", (out_both["pred_var"] > 0).all().item())

    # T3: Gradient flow
    print("\n[T3] Gradient flow")
    x_grad = torch.randn(2, 15, PSYCHE_DIM, requires_grad=True)
    out = model(x_grad, task="predict")
    loss = out["pred_mean"].sum() + out["pred_var"].sum()
    loss.backward()
    rep("Gradients flow to input", x_grad.grad is not None)
    rep("No NaN gradients", not torch.isnan(x_grad.grad).any().item())

    # T4: Synthetic data generation
    print("\n[T4] Synthetic data generation")
    trajs, labels = generate_synthetic_trajectories(
        n_trajectories=50, seq_len=30, n_classes=3
    )
    rep("Trajectories generated", len(trajs) == 50)
    rep("Labels generated", len(labels) == 50)
    rep("Trajectory shape", trajs[0].shape == (30, PSYCHE_DIM),
        f"{trajs[0].shape}")

    # T5: Training loop
    print("\n[T5] Training loop (5 epochs, small)")
    small_model = TemporalBeliefModel(
        input_dim=PSYCHE_DIM, hidden_dim=32,
        n_lstm_layers=1, n_heads=2, n_transformer_layers=1,
        n_classes=3,
    )
    history = train_temporal_model(
        small_model, trajs[:30], labels[:30],
        epochs=5, batch_size=8, seq_len=10,
        task="both",
    )
    rep("Training completes", len(history["loss"]) == 5)
    rep("Loss decreases", history["loss"][-1] < history["loss"][0],
        f"{history['loss'][0]:.4f} → {history['loss'][-1]:.4f}")

    # T6: Validation harness
    print("\n[T6] Validation harness")
    harness = ValidationHarness(device="cpu")
    val_results = harness.run_validation(
        small_model, trajs[30:], labels[30:],
        dataset_name="synthetic", seq_len=10,
    )
    rep("Validation runs", "prediction" in val_results)
    rep("Baselines computed", "baselines" in val_results)
    rep("Model MSE finite",
        math.isfinite(val_results["prediction"]["mse"]),
        f"MSE={val_results['prediction']['mse']:.4f}")

    # T7: Baselines are sane
    print("\n[T7] Baseline sanity")
    traj = trajs[0]
    _, mse_p = harness.persistence_baseline(traj)
    _, mse_l = harness.linear_baseline(traj)
    _, mse_m = harness.mean_baseline(traj)
    rep("Persistence MSE finite", math.isfinite(mse_p), f"{mse_p:.4f}")
    rep("Linear MSE finite", math.isfinite(mse_l), f"{mse_l:.4f}")
    rep("Mean MSE finite", math.isfinite(mse_m), f"{mse_m:.4f}")

    # T8: DAIC-WOZ adapter
    print("\n[T8] DAIC-WOZ adapter")
    daic = DAICWOZAdapter()
    rep("DAIC-WOZ adapter constructs", daic is not None)
    rep("DAIC-WOZ data not available (expected)",
        not daic.check_data_available())

    # Test feature mapping with synthetic data
    audio_feats = {"speech_rate": 100, "pause_duration": 1.5,
                   "pitch_std": 30.0, "jitter": 0.02,
                   "energy_mean": 50.0}
    text_feats = {"word_count": 200, "unique_words": 80,
                  "sentiment_neg": 0.3, "first_person_pronouns": 0.1}
    psyche_vec = daic.interview_features_to_psyche(
        audio_feats, text_feats, phq8_score=12.0
    )
    rep("DAIC-WOZ feature mapping produces valid vector",
        psyche_vec.shape == (PSYCHE_DIM,) and np.isfinite(psyche_vec).all(),
        f"shape={psyche_vec.shape}")

    # T9: Dreaddit adapter
    print("\n[T9] Dreaddit adapter")
    dreaddit = DreadditAdapter()
    rep("Dreaddit adapter constructs", dreaddit is not None)
    rep("Dreaddit data not available (expected)",
        not dreaddit.check_data_available())

    # Test text feature extraction
    test_text = ("I feel so anxious and worried all the time. "
                 "I can't sleep and everything feels overwhelming. "
                 "I'm scared of what might happen tomorrow.")
    feats = dreaddit.text_to_features(test_text)
    rep("Text features extracted", len(feats) > 0)
    rep("Anxiety words detected", feats["anxiety_density"] > 0,
        f"density={feats['anxiety_density']:.3f}")

    psyche_from_text = dreaddit.text_to_psyche_state(test_text, label=1)
    rep("Text→psyche mapping produces valid vector",
        psyche_from_text.shape == (PSYCHE_DIM,) and
        np.isfinite(psyche_from_text).all())

    # T10: Psyche state vector conversion
    print("\n[T10] Psyche state vector conversion")
    state_dict = {
        "id_entropy": 2.5,
        "accumulated_entropy": 10.0,
        "superego_loss": 1.2,
        "behavioral_entropy": 0.5,
        "free_energy": 3.0,
        "selected_action": 3,
        "convergence_speed": 0.7,
        "neurophysio_map": {
            "FAA_proxy": 0.3, "GFP_proxy": 5.0,
            "LZ_complexity": 0.8, "PLV_F3_F4_proxy": 0.6,
            "neuroplasticity_idx": 0.4,
        },
    }
    vec = psyche_state_to_vector(state_dict)
    rep("State dict → vector", vec.shape == (PSYCHE_DIM,))
    rep("Vector values correct", abs(vec[0] - 2.5) < 1e-6)

    # T11: Uncertainty calibration check
    print("\n[T11] Uncertainty output")
    x_test = torch.randn(2, 15, PSYCHE_DIM)
    out = model(x_test, task="predict")
    var = out["pred_var"]
    rep("Uncertainty has correct shape", var.shape == (2, 1, PSYCHE_DIM))
    rep("Uncertainty is positive", (var > 0).all().item())
    rep("Uncertainty is finite", torch.isfinite(var).all().item())

    # T12: Variable-length sequences
    print("\n[T12] Variable-length sequence handling")
    x_var = torch.randn(3, 25, PSYCHE_DIM)
    lengths_var = torch.tensor([25, 15, 5])
    out_var = model(x_var, lengths_var, task="predict")
    rep("Variable-length forward pass works",
        out_var["pred_mean"].shape == (3, 1, PSYCHE_DIM))
    rep("No NaN in variable-length output",
        not torch.isnan(out_var["pred_mean"]).any().item())

    # T13: Multi-step prediction
    print("\n[T13] Multi-step prediction")
    model_ms = TemporalBeliefModel(
        input_dim=PSYCHE_DIM, hidden_dim=32,
        n_lstm_layers=1, n_heads=2, n_transformer_layers=1,
        n_future_steps=5, n_classes=3,
    )
    out_ms = model_ms(torch.randn(2, 10, PSYCHE_DIM), task="predict")
    rep("Multi-step prediction shape",
        out_ms["pred_mean"].shape == (2, 5, PSYCHE_DIM),
        f"{tuple(out_ms['pred_mean'].shape)}")

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THIS MODULE PROVIDES
    • LSTM + Transformer hybrid for temporal psyche state modeling
    • Uncertainty-aware next-state prediction
    • Trajectory classification
    • DAIC-WOZ adapter (interview features → psyche states)
    • Dreaddit adapter (text → psyche state proxy)
    • Validation harness with baselines and metrics
    • Synthetic data generator for testing

  WHAT THIS MODULE DOES NOT PROVIDE
    ❌ Clinical diagnosis of any kind
    ❌ Mind reading or thought inference
    ❌ Actual DAIC-WOZ or Dreaddit data (must be obtained separately)
    ❌ Validated clinical mappings (all mappings are research approximations)
    ❌ EEG/MEG/fMRI processing (use mental_one.py for that)

  DATA ACCESS REQUIREMENTS
    DAIC-WOZ: https://dcapswoz.ict.usc.edu/ (requires data use agreement)
    Dreaddit: Check original paper repository for access instructions
    Both require IRB/ethics approval for research use.

  HONEST ASSESSMENT
    The text→psyche state mapping for Dreaddit is VERY rough.
    The interview→psyche state mapping for DAIC-WOZ is approximate.
    Any results from this module are preliminary research findings,
    not clinical evidence. Do not use for patient care.
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Temporal Belief Model + Validation Harness"
    )
    ap.add_argument("--test", action="store_true", help="Run self-tests")
    ap.add_argument("--train-synthetic", action="store_true",
                    help="Train on synthetic data")
    ap.add_argument("--validate", action="store_true",
                    help="Run validation on synthetic data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="Output JSON path")
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    if a.train_synthetic or a.validate:
        print(f"Generating synthetic trajectories...")
        trajs, labels = generate_synthetic_trajectories(
            n_trajectories=200, seq_len=50, n_classes=3
        )
        print(f"  {len(trajs)} trajectories, {PSYCHE_DIM} features")

        model = TemporalBeliefModel(
            input_dim=PSYCHE_DIM,
            hidden_dim=a.hidden_dim,
            n_classes=3,
        ).to(a.device)

        if a.train_synthetic:
            print(f"\nTraining for {a.epochs} epochs...")
            history = train_temporal_model(
                model, trajs, labels,
                epochs=a.epochs, seq_len=a.seq_len,
                device=a.device, task="both",
            )
            print(f"  Final loss: {history['loss'][-1]:.4f}")

            if a.out:
                torch.save({
                    "model_state": model.state_dict(),
                    "history": history,
                    "config": {
                        "hidden_dim": a.hidden_dim,
                        "seq_len": a.seq_len,
                        "n_classes": 3,
                    },
                }, a.out)
                print(f"  Saved to {a.out}")

        if a.validate:
            print(f"\nRunning validation...")
            harness = ValidationHarness(device=a.device)
            results = harness.run_validation(
                model, trajs, labels,
                dataset_name="synthetic",
                seq_len=a.seq_len,
            )
            harness.print_report()

            if a.out:
                with open(a.out, "w") as f:
                    json.dump(results, f, indent=2, default=float)
                print(f"  Results saved to {a.out}")

        return

    ap.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
