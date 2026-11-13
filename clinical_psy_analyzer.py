#!/usr/bin/env python3
"""
=============================================================================
CLINICAL PSYCHIATRY ANALYZER — Computational Framework
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.

"The Tao is the wellspring; the Te is the path; the Si is the proof."
"Master Confucius taught that in this world, we must do everything for the sake of a safe and peaceful society."
"Jesus Christ died on the cross to teach us that when someone strikes you on the right cheek, you should turn the other cheek to them. Therefore, if we believe as He did and follow His way, we shall enter into the fullness of Christ. If we believe in this path, we shall be saved from sin with certainty."

License    : MIT | Year: 2026

WHAT THIS IS
  A computational framework for analyzing psychiatric and neurological
  signals using the mathematical tools from the Structural Calculus series.
  This is a RESEARCH SCAFFOLD, not a diagnostic tool.

WHAT THIS IS NOT — CRITICAL DISCLAIMERS
  ❌ NOT a clinical diagnostic tool. Psychiatric diagnosis requires
     trained clinicians using DSM-5/ICD-11 criteria.
  ❌ NOT mind reading. No software can determine intent, motivation,
     or inner experience from signals alone.
  ❌ NOT validated against clinical gold standards.
  ❌ NOT approved by any regulatory body (FDA, EMA, etc.)

HONEST TOOL MAPPING
  ✅ Signal classification (EEG/MEG) — uses SSC filter, CSOC stress
  ✅ Symptom trajectory analysis — uses Paper 6 change-point detection
  ✅ Multi-scale decomposition — uses carpet graph Laplacian
  ✅ Statistical symptom profiling — uses Statistics Dictionary (Paper 9)
  ✅ World model for progression — uses Langevin dynamics as analog
  ⚠️  Fractal brain morphology — carpet operators on structural data
       (exploratory, not established)
  ❌ VOF/multiphase flow — NO genuine connection to psychiatry.
       The analogy is not a method.
  ❌ Quantum chemistry — NO connection to psychiatric signal analysis.
       Molecular-level chemistry does not inform EEG classification.

ETHICAL FRAMEWORK
  • All outputs are probabilistic hypotheses, not diagnoses
  • Requires clinical validation before any clinical use
  • Must be used only by trained professionals
  • Patient data requires appropriate consent and IRB approval
  • Results must never be used to deny care or discriminate

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("clinical_psy")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# §0. CLINICAL TAXONOMY — Structured Symptom Categories
# =============================================================================

class ClinicalDomain(Enum):
    """Top-level clinical domains."""
    NEUROLOGICAL = "neurological"
    PSYCHIATRIC = "psychiatric"
    NEURODEVELOPMENTAL = "neurodevelopmental"
    SUBSTANCE_USE = "substance_use"
    PERSONALITY = "personality"
    DISSOCIATIVE = "dissociative"
    SOMATIC = "somatic"
    SLEEP = "sleep"

class SymptomCategory(Enum):
    """Specific symptom categories within domains."""
    # Psychiatric
    MOOD_DEPRESSION = "mood_depression"
    MOOD_MANIA = "mood_mania"
    ANXIETY_GENERALIZED = "anxiety_generalized"
    ANXIETY_PANIC = "anxiety_panic"
    ANXIETY_PHOBIC = "anxiety_phobic"
    PSYCHOTIC_POSITIVE = "psychotic_positive"
    PSYCHOTIC_NEGATIVE = "psychotic_negative"
    PTSD_INTRUSION = "ptsd_intrusion"
    PTSD_AVOIDANCE = "ptsd_avoidance"
    PTSD_HYPERAROUSAL = "ptsd_hyperarousal"
    OCD_OBSESSION = "ocd_obsession"
    OCD_COMPULSION = "ocd_compulsion"
    # Behavioral
    IMPULSIVITY = "impulsivity"
    AGGRESSION = "aggression"
    RISKY_BEHAVIOR = "risky_behavior"
    HYPERSEXUALITY = "hypersexuality"
    SUBSTANCE_CRAVING = "substance_craving"
    # Cognitive
    ATTENTION_DEFICIT = "attention_deficit"
    MEMORY_IMPAIRMENT = "memory_impairment"
    EXECUTIVE_DYSFUNCTION = "executive_dysfunction"
    DISSOCIATION = "dissociation"
    # Neurological
    SEIZURE_ACTIVITY = "seizure_activity"
    MOTOR_ABORMALITY = "motor_abnormality"
    SENSORY_DISTURBANCE = "sensory_disturbance"

@dataclass
class ClinicalAssessment:
    """Structured clinical assessment output."""
    domain: ClinicalDomain
    category: SymptomCategory
    confidence: float            # 0-1, hypothesis strength
    evidence_sources: List[str]  # which signals support this
    comorbidity_flags: List[SymptomCategory]
    differential_diagnosis: List[Dict[str, Any]]
    recommendation: str
    caveats: List[str]

    def to_dict(self):
        d = asdict(self)
        d["domain"] = self.domain.value
        d["category"] = self.category.value
        d["comorbidity_flags"] = [c.value for c in self.comorbidity_flags]
        return d


# =============================================================================
# §1. SIGNAL PROCESSING LAYER — SSC + CSOC
# =============================================================================

class SSCSignalFilter(nn.Module):
    """
    Semantic State Contraction applied to clinical signals.
    Uses the EMA low-pass filter from the Structural Calculus series.

    This is a genuine application: filtering noisy EEG/MEG signals
    to extract slow-varying features for classification.
    """
    def __init__(self, epsilon_fp: float = 0.0028,
                 n_channels: int = 19, n_timepoints: int = 256):
        super().__init__()
        self.eps = epsilon_fp
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.register_buffer("prev_state", torch.zeros(n_channels, n_timepoints))
        self.register_buffer("initialized", torch.tensor(False))

    def reset(self):
        self.prev_state.zero_()
        self.initialized.fill_(False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: (batch, n_channels, n_timepoints) raw EEG/MEG
        Returns:
            filtered signal with same shape
        """
        B = signal.shape[0]
        out = torch.zeros_like(signal)
        for b in range(B):
            s = signal[b]  # (n_channels, n_timepoints)
            if not self.initialized.item() or self.prev_state.shape != s.shape:
                self.prev_state = s.detach().clone()
                self.initialized.fill_(True)
                out[b] = s
            else:
                filtered = self.prev_state + self.eps * (s - self.prev_state)
                self.prev_state = filtered.detach()
                out[b] = filtered
        return out


class CSOCStressMonitor(nn.Module):
    """
    CSOC-style adaptive monitoring for clinical signals.
    Tracks signal "stress" (deviation from baseline) and triggers
    alerts when thresholds are exceeded.

    This is a genuine application: detecting acute symptom episodes
    in continuous monitoring data.
    """
    def __init__(self, sigma_target: float = 1.0,
                 epsilon_fp: float = 0.0028,
                 boost_factor: float = 3.0):
        super().__init__()
        self.sigma_target = sigma_target
        self.eps = epsilon_fp
        self.boost_factor = boost_factor
        self.register_buffer("prev_stress", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    def reset(self):
        self.prev_stress.zero_()
        self.initialized.fill_(False)

    def forward(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            signal: (batch, n_features) clinical feature vector
        Returns:
            dict with stress, adaptive_threshold, alert_level
        """
        raw_stress = signal.std(dim=-1)  # deviation across features

        if not self.initialized.item():
            self.prev_stress = raw_stress.mean().detach()
            self.initialized.fill_(True)

        smoothed = self.prev_stress + self.eps * (raw_stress.mean() - self.prev_stress)
        self.prev_stress = smoothed.detach()

        dev = (smoothed - self.sigma_target) / max(self.sigma_target, 1e-12)
        threshold = self.sigma_target * (1 + self.boost_factor * torch.sigmoid(dev))
        alert = (smoothed > threshold).float()

        return {
            "stress": smoothed,
            "adaptive_threshold": threshold,
            "alert_level": alert,
        }


# =============================================================================
# §2. CHANGE-POINT DETECTION — Paper 6 Integration
# =============================================================================

class SymptomChangePointDetector:
    """
    Detects change points in symptom trajectories using CUSUM.
    This is from Paper 6 (Change-Point Induced Stationarity).

    Genuine application: identifying when a patient's symptom
    profile shifts significantly (episode onset, treatment response,
    relapse detection).
    """
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect(self, trajectory: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            trajectory: (n_timepoints, n_features) symptom scores over time
        Returns:
            change_points, block_lengths, classification
        """
        if len(trajectory) < 10:
            return {"status": "insufficient_data"}

        # CUSUM detection
        vals = trajectory.mean(axis=1)  # aggregate feature mean
        cusum_pos = np.zeros(len(vals))
        cusum_neg = np.zeros(len(vals))
        for i in range(1, len(vals)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (vals[i] - vals.mean()) - 0.1)
            cusum_neg[i] = max(0, cusum_neg[i-1] - (vals[i] - vals.mean()) - 0.1)
        combined = np.maximum(cusum_pos, cusum_neg)
        change_points = np.where(combined > self.threshold * combined.max())[0]

        if len(change_points) < 2:
            return {"status": "no_change_points", "n_cp": 0}

        block_lengths = np.diff(change_points)

        # Classify growth rate (Paper 6, Theorem 5.1)
        log_lengths = np.log(block_lengths + 1)
        k = np.arange(len(log_lengths))
        if len(k) > 1:
            slope = np.polyfit(k, log_lengths, 1)[0]
        else:
            slope = 0.0

        if slope > 1.5:
            classification = "double_exponential"
        elif slope > 0.5:
            classification = "regularly_varying"
        else:
            classification = "finite_mean"

        return {
            "status": "ok",
            "change_points": change_points.tolist(),
            "block_lengths": block_lengths.tolist(),
            "classification": classification,
            "slope": slope,
            "n_cp": len(change_points),
        }


# =============================================================================
# §3. MULTI-SCALE SIGNAL DECOMPOSITION — Carpet Graph
# =============================================================================

class CarpetSignalDecomposer(nn.Module):
    """
    Multi-scale decomposition of clinical signals using carpet graph
    Laplacian from Carpet-19.

    Genuine application: separating fast (neural oscillation) and slow
    (mood/behavioral) components of psychiatric signals.

    LIMITATION: This is exploratory. The carpet graph is a mathematical
    structure, not a proven model of brain connectivity.
    """
    def __init__(self, n_levels: int = 3, signal_dim: int = 19):
        super().__init__()
        self.n_levels = n_levels
        self.signal_dim = signal_dim
        # Build carpet graph adjacency for signal decomposition
        self._build_carpet_graph()

    def _build_carpet_graph(self):
        """Build level-1 carpet graph adjacency (8 cells, 3x3 minus center)."""
        # 8 cells in 3x3 grid (center removed)
        cells = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
        n = len(cells)
        adj = np.zeros((n, n))
        for i, (xi, yi) in enumerate(cells):
            for j, (xj, yj) in enumerate(cells):
                if abs(xi - xj) + abs(yi - yj) == 1:
                    adj[i, j] = 1
        self.register_buffer("adjacency", torch.tensor(adj, dtype=torch.float64))
        self.register_buffer("degree", torch.tensor(adj.sum(axis=1), dtype=torch.float64))

    def laplacian(self) -> torch.Tensor:
        return torch.diag(self.degree) - self.adjacency

    def decompose(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            signal: (batch, signal_dim) clinical features
        Returns:
            dict with coarse, medium, fine components
        """
        L = self.laplacian()
        # Project signal onto carpet graph structure
        # This is a simplified projection; full carpet analysis needs
        # proper graph signal processing
        n = self.adjacency.shape[0]
        if signal.shape[-1] != n:
            # Interpolate signal to carpet size
            signal = F.interpolate(signal.unsqueeze(1), size=n, mode='linear').squeeze(1)

        # Graph Fourier transform
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Project signal onto eigenbasis
        coeffs = signal @ eigenvectors

        # Split into frequency bands
        n_low = n // 3
        n_mid = n // 3
        coarse = coeffs[:, :n_low] @ eigenvectors[:, :n_low].T
        medium = coeffs[:, n_low:n_low+n_mid] @ eigenvectors[:, n_low:n_low+n_mid].T
        fine = coeffs[:, n_low+n_mid:] @ eigenvectors[:, n_low+n_mid:].T

        return {
            "coarse": coarse,    # slow, mood-like component
            "medium": medium,    # intermediate, cognitive component
            "fine": fine,        # fast, neural oscillation component
            "eigenvalues": eigenvalues,
        }


# =============================================================================
# §4. STATISTICAL SYMPTOM PROFILER — Paper 9 Integration
# =============================================================================

class SymptomProfiler:
    """
    Statistical profiling of symptoms using methods from Paper 9
    (Statistical Dictionary for Finite-Sample Guarantees).

    Genuine application: characterizing symptom distributions,
    identifying outliers, estimating confidence intervals for
    symptom severity scores.
    """

    @staticmethod
    def profile_symptom(scores: np.ndarray,
                        method: str = "hill") -> Dict[str, Any]:
        """
        Args:
            scores: symptom severity scores over time
            method: 'hill' for tail analysis, 'moment' for distribution
        Returns:
            statistical profile
        """
        if len(scores) < 5:
            return {"status": "insufficient_data"}

        profile = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "n": len(scores),
        }

        if method == "hill" and len(scores) > 10:
            # Hill estimator for tail index (Paper 9, Entry 2.5)
            sorted_scores = np.sort(scores)[::-1]
            k = max(5, len(sorted_scores) // 4)  # upper 25%
            top_k = sorted_scores[:k]
            threshold = sorted_scores[k]
            if threshold > 0:
                log_ratios = np.log(top_k / threshold)
                hill_theta = 1.0 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else float('inf')
                profile["tail_index_hill"] = float(hill_theta)
                profile["tail_heavy"] = hill_theta < 2.0  # heavy tail if θ < 2
            else:
                profile["tail_index_hill"] = None
                profile["tail_heavy"] = None

        return profile

    @staticmethod
    def confidence_interval(scores: np.ndarray,
                            confidence: float = 0.95,
                            method: str = "bootstrap") -> Tuple[float, float]:
        """Bootstrap confidence interval for mean symptom score."""
        n_boot = 1000
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            boot_means.append(np.mean(sample))
        alpha = 1 - confidence
        lo = np.percentile(boot_means, 100 * alpha / 2)
        hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
        return float(lo), float(hi)


# =============================================================================
# §5. CLINICAL CLASSIFIER — Multi-Modal Symptom Analysis
# =============================================================================

class ClinicalSymptomClassifier(nn.Module):
    """
    Multi-modal classifier for psychiatric symptom categories.
    Uses SSC filtering, CSOC monitoring, and carpet decomposition.

    CRITICAL: This is a RESEARCH SCAFFOLD, not a diagnostic tool.
    Clinical diagnosis requires trained professionals using DSM-5/ICD-11.
    """
    def __init__(self, n_features: int = 19, n_classes: int = 10,
                 hidden_dim: int = 128):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        # Signal processing layers
        self.ssc_filter = SSCSignalFilter(n_channels=n_features)
        self.csoc_monitor = CSOCStressMonitor()
        self.carpet_decomp = CarpetSignalDecomposer(signal_dim=n_features)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # Domain classifier (neurological vs psychiatric)
        self.domain_classifier = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, len(ClinicalDomain)),
        )

    def forward(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            signal: (batch, n_features) clinical feature vector
        Returns:
            dict with class probabilities, domain, stress metrics
        """
        # Apply SSC filter for noise reduction
        filtered = self.ssc_filter(signal)

        # CSOC stress monitoring
        stress_metrics = self.csoc_monitor(signal)

        # Carpet decomposition for multi-scale features
        decomp = self.carpet_decomp(signal)

        # Combine features
        combined = torch.cat([filtered, decomp["coarse"]], dim=-1)
        # Adjust dimensions if needed
        if combined.shape[-1] != self.n_features:
            combined = F.interpolate(combined.unsqueeze(1),
                                     size=self.n_features, mode='linear').squeeze(1)

        # Classification
        class_logits = self.classifier(combined)
        domain_logits = self.domain_classifier(combined)

        return {
            "class_logits": class_logits,
            "class_probs": F.softmax(class_logits, dim=-1),
            "domain_logits": domain_logits,
            "domain_probs": F.softmax(domain_logits, dim=-1),
            "stress": stress_metrics["stress"],
            "alert": stress_metrics["alert_level"],
            "decomposition": decomp,
        }

    def assess(self, signal: torch.Tensor,
               symptom_history: Optional[np.ndarray] = None) -> List[ClinicalAssessment]:
        """
        Generate structured clinical assessments from signal.

        Args:
            signal: (batch, n_features) clinical features
            symptom_history: optional (n_timepoints, n_features) history
        Returns:
            list of ClinicalAssessment objects
        """
        outputs = self.forward(signal)
        assessments = []

        # Map class probabilities to clinical categories
        # This mapping is PLACEHOLDER — real mapping requires clinical validation
        class_to_category = {
            0: SymptomCategory.MOOD_DEPRESSION,
            1: SymptomCategory.MOOD_MANIA,
            2: SymptomCategory.ANXIETY_GENERALIZED,
            3: SymptomCategory.ANXIETY_PANIC,
            4: SymptomCategory.PSYCHOTIC_POSITIVE,
            5: SymptomCategory.PTSD_INTRUSION,
            6: SymptomCategory.OCD_OBSESSION,
            7: SymptomCategory.IMPULSIVITY,
            8: SymptomCategory.SUBSTANCE_CRAVING,
            9: SymptomCategory.SEIZURE_ACTIVITY,
        }

        for i in range(signal.shape[0]):
            probs = outputs["class_probs"][i]
            domain_probs = outputs["domain_probs"][i]

            # Find top prediction
            top_class = int(probs.argmax().item())
            top_prob = float(probs[top_class].item())

            if top_class in class_to_category:
                category = class_to_category[top_class]
            else:
                category = SymptomCategory.ANXIETY_GENERALIZED  # default

            # Determine domain
            top_domain = int(domain_probs.argmax().item())
            domain = list(ClinicalDomain)[top_domain]

            # Change-point analysis if history provided
            comorbidities = []
            if symptom_history is not None and len(symptom_history) > 10:
                cp_detector = SymptomChangePointDetector()
                cp_result = cp_detector.detect(symptom_history)
                if cp_result["status"] == "ok" and cp_result["n_cp"] > 2:
                    # Multiple change points suggest complex presentation
                    comorbidities.append(SymptomCategory.EXECUTIVE_DYSFUNCTION)

            assessment = ClinicalAssessment(
                domain=domain,
                category=category,
                confidence=top_prob,
                evidence_sources=["EEG/MEG signal analysis"],
                comorbidity_flags=comorbidities,
                differential_diagnosis=[],
                recommendation="REQUIRES CLINICAL EVALUATION - NOT A DIAGNOSIS",
                caveats=[
                    "Research scaffold only, not validated clinically",
                    "Cannot determine intent or motivation",
                    "Requires trained clinician interpretation",
                    "Must not be used for treatment decisions without validation",
                ],
            )
            assessments.append(assessment)

        return assessments


# =============================================================================
# §6. WORLD MODEL — Symptom Progression Prediction
# =============================================================================

class SymptomWorldModel(nn.Module):
    """
    Predictive model for symptom progression.
    Uses Langevin dynamics as an analog for symptom evolution.

    This is EXPLORATORY — symptom progression is complex and depends
    on many factors not captured by mathematical models alone.
    """
    def __init__(self, n_features: int = 19, hidden_dim: int = 128):
        super().__init__()
        self.n_features = n_features

        # Encoder: current state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Dynamics: latent -> next latent (learned)
        self.dynamics = nn.GRUCell(hidden_dim, hidden_dim)

        # Decoder: latent -> next state
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_features),
            nn.Softplus(),  # uncertainty must be positive
        )

    def forward(self, state: torch.Tensor,
                n_steps: int = 10) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (batch, n_features) current symptom state
            n_steps: number of future steps to predict
        Returns:
            dict with predictions and uncertainties
        """
        batch_size = state.shape[0]
        z = self.encoder(state)

        predictions = []
        uncertainties = []

        for _ in range(n_steps):
            z = self.dynamics(z, z)
            pred = self.decoder(z)
            unc = self.uncertainty(z)
            predictions.append(pred)
            uncertainties.append(unc)

        return {
            "predictions": torch.stack(predictions, dim=1),  # (batch, n_steps, n_features)
            "uncertainties": torch.stack(uncertainties, dim=1),
        }

    def predict_trajectory(self, initial_state: torch.Tensor,
                           n_steps: int = 10) -> Dict[str, np.ndarray]:
        """Predict symptom trajectory with uncertainty bounds."""
        with torch.no_grad():
            outputs = self.forward(initial_state, n_steps)

        pred_mean = outputs["predictions"].mean(dim=0).cpu().numpy()
        pred_unc = outputs["uncertainties"].mean(dim=0).cpu().numpy()

        return {
            "mean_trajectory": pred_mean,
            "uncertainty": pred_unc,
            "upper_bound": pred_mean + 2 * pred_unc,
            "lower_bound": pred_mean - 2 * pred_unc,
        }


# =============================================================================
# §7. COMORBIDITY ANALYZER
# =============================================================================

class ComorbidityAnalyzer:
    """
    Analyzes potential comorbidities from symptom profiles.
    Uses statistical co-occurrence patterns.

    LIMITATION: Co-occurrence does not imply causation.
    Clinical interpretation required.
    """

    # Common comorbidity patterns (from clinical literature)
    COMORBIDITY_MATRIX = {
        SymptomCategory.MOOD_DEPRESSION: [
            SymptomCategory.ANXIETY_GENERALIZED,
            SymptomCategory.SUBSTANCE_CRAVING,
            SymptomCategory.SLEEP,
        ],
        SymptomCategory.PSYCHOTIC_POSITIVE: [
            SymptomCategory.SUBSTANCE_CRAVING,
            SymptomCategory.MOOD_DEPRESSION,
        ],
        SymptomCategory.PTSD_INTRUSION: [
            SymptomCategory.MOOD_DEPRESSION,
            SymptomCategory.ANXIETY_GENERALIZED,
            SymptomCategory.SUBSTANCE_CRAVING,
        ],
        SymptomCategory.IMPULSIVITY: [
            SymptomCategory.SUBSTANCE_CRAVING,
            SymptomCategory.RISKY_BEHAVIOR,
        ],
        SymptomCategory.AGGRESSION: [
            SymptomCategory.SUBSTANCE_CRAVING,
            SymptomCategory.IMPULSIVITY,
        ],
    }

    @staticmethod
    def analyze(primary_category: SymptomCategory,
                symptom_scores: Dict[SymptomCategory, float]) -> List[Dict[str, Any]]:
        """
        Args:
            primary_category: the main identified category
            symptom_scores: scores for each symptom category
        Returns:
            list of potential comorbidities with confidence
        """
        potential = ComorbidityAnalyzer.COMORBIDITY_MATRIX.get(primary_category, [])
        results = []

        for category in potential:
            if category in symptom_scores:
                score = symptom_scores[category]
                if score > 0.3:  # threshold for consideration
                    results.append({
                        "category": category.value,
                        "score": score,
                        "confidence": min(score, 1.0),
                        "note": "REQUIRES CLINICAL VERIFICATION",
                    })

        return results


# =============================================================================
# §8. FULL PIPELINE
# =============================================================================

class ClinicalPsychiatryPipeline:
    """
    Complete analysis pipeline combining all components.

    WORKFLOW:
    1. Signal preprocessing (SSC filter)
    2. Multi-scale decomposition (carpet graph)
    3. Symptom classification (neural network)
    4. Change-point detection (Paper 6)
    5. Statistical profiling (Paper 9)
    6. Comorbidity analysis
    7. Progression prediction (world model)

    OUTPUT: Structured clinical assessment with confidence levels
    and explicit caveats about limitations.
    """

    def __init__(self, n_features: int = 19, device: str = "cpu"):
        self.device = torch.device(device)
        self.n_features = n_features

        # Components
        self.classifier = ClinicalSymptomClassifier(
            n_features=n_features).to(self.device)
        self.world_model = SymptomWorldModel(
            n_features=n_features).to(self.device)
        self.cp_detector = SymptomChangePointDetector()
        self.profiler = SymptomProfiler()
        self.comorbidity_analyzer = ComorbidityAnalyzer()

    def analyze(self, signal: torch.Tensor,
                symptom_history: Optional[np.ndarray] = None,
                metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Full analysis pipeline.

        Args:
            signal: (batch, n_features) clinical signal
            symptom_history: optional (n_timepoints, n_features) history
            metadata: optional patient metadata (age, sex, etc.)
        Returns:
            comprehensive analysis report
        """
        signal = signal.to(self.device)

        # Step 1: Classification
        assessments = self.classifier.assess(signal, symptom_history)

        # Step 2: Change-point detection if history available
        change_analysis = None
        if symptom_history is not None and len(symptom_history) > 10:
            change_analysis = self.cp_detector.detect(symptom_history)

        # Step 3: Statistical profiling
        profiles = {}
        if symptom_history is not None:
            for i in range(symptom_history.shape[1]):
                cat = f"feature_{i}"
                profiles[cat] = self.profiler.profile_symptom(
                    symptom_history[:, i], method="hill")

        # Step 4: Progression prediction
        progression = self.world_model.predict_trajectory(signal, n_steps=10)

        # Step 5: Comorbidity analysis
        comorbidities = []
        if assessments:
            primary = assessments[0]
            # Use symptom scores from history if available
            if symptom_history is not None:
                scores = {cat: float(symptom_history[:, i].mean())
                          for i, cat in enumerate(
                              list(SymptomCategory)[:symptom_history.shape[1]])}
                comorbidities = self.comorbidity_analyzer.analyze(
                    primary.category, scores)

        return {
            "version": VERSION,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": "RESEARCH SCAFFOLD - NOT A CLINICAL DIAGNOSTIC TOOL",
            "assessments": [a.to_dict() for a in assessments],
            "change_analysis": change_analysis,
            "statistical_profiles": profiles,
            "progression_prediction": {
                "mean_trajectory": progression["mean_trajectory"].tolist(),
                "uncertainty": progression["uncertainty"].tolist(),
            },
            "comorbidities": comorbidities,
            "metadata": metadata,
            "limitations": [
                "Not validated against clinical gold standards",
                "Cannot determine intent, motivation, or inner experience",
                "Requires trained clinician interpretation",
                "Must not be used for treatment decisions",
                "Fractal/VOF connections are analogies, not established methods",
            ],
        }


# =============================================================================
# §9. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 88)
    print(f"  CLINICAL PSYCHIATRY ANALYZER v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 88)
    print("  ⚠️  RESEARCH SCAFFOLD - NOT A CLINICAL DIAGNOSTIC TOOL")
    print("=" * 88)

    torch.manual_seed(42)
    np.random.seed(42)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: SSC filter
    print("\n[T1] SSC Signal Filter")
    ssc = SSCSignalFilter(n_channels=19, n_timepoints=256)
    signal = torch.randn(2, 19, 256)
    filtered = ssc(signal)
    rep("SSC output shape", filtered.shape == signal.shape)
    rep("SSC reduces variance", filtered.var() < signal.var())

    # T2: CSOC stress monitor
    print("\n[T2] CSOC Stress Monitor")
    csoc = CSOCStressMonitor()
    stress = csoc(torch.randn(4, 19))
    rep("CSOC returns stress", "stress" in stress)
    rep("CSOC returns alert", "alert_level" in stress)

    # T3: Change-point detection
    print("\n[T3] Change-Point Detection (Paper 6)")
    # Create synthetic trajectory with change points
    n_t = 100
    trajectory = np.zeros((n_t, 5))
    trajectory[:30, :] = np.random.randn(30, 5) * 0.1
    trajectory[30:60, :] = np.random.randn(30, 5) * 0.1 + 2.0  # shift
    trajectory[60:, :] = np.random.randn(40, 5) * 0.1 + 1.0  # partial recovery

    cp_det = SymptomChangePointDetector(threshold=0.05)
    cp_result = cp_det.detect(trajectory)
    rep("Change-points detected", cp_result["status"] == "ok",
        f"n_cp={cp_result.get('n_cp', 0)}")
    if cp_result["status"] == "ok":
        print(f"      classification: {cp_result['classification']}")

    # T4: Carpet decomposition
    print("\n[T4] Carpet Signal Decomposition")
    carpet = CarpetSignalDecomposer(signal_dim=8)
    signal_8d = torch.randn(2, 8)
    decomp = carpet.decompose(signal_8d)
    rep("Decomposition returns components", "coarse" in decomp and "fine" in decomp)
    rep("Components have correct batch dim", decomp["coarse"].shape[0] == 2)

    # T5: Statistical profiler
    print("\n[T5] Statistical Symptom Profiler")
    scores = np.random.randn(50) * 2 + 5
    profile = SymptomProfiler.profile_symptom(scores, method="hill")
    rep("Profile computed", "mean" in profile and "std" in profile)
    ci_lo, ci_hi = SymptomProfiler.confidence_interval(scores)
    rep("CI computed", ci_lo < profile["mean"] < ci_hi)

    # T6: Clinical classifier
    print("\n[T6] Clinical Symptom Classifier")
    clf = ClinicalSymptomClassifier(n_features=19, n_classes=10)
    signal_19 = torch.randn(4, 19)
    outputs = clf(signal_19)
    rep("Classifier output shape", outputs["class_probs"].shape == (4, 10))
    rep("Domain output shape", outputs["domain_probs"].shape[0] == 4)
    rep("Probabilities sum to 1",
        torch.allclose(outputs["class_probs"].sum(dim=1),
                       torch.ones(4), atol=1e-5))

    # T7: World model
    print("\n[T7] Symptom World Model")
    wm = SymptomWorldModel(n_features=19)
    state = torch.randn(2, 19)
    pred = wm(state, n_steps=5)
    rep("World model predictions shape", pred["predictions"].shape == (2, 5, 19))
    rep("Uncertainties positive", (pred["uncertainties"] >= 0).all())

    # T8: Comorbidity analyzer
    print("\n[T8] Comorbidity Analyzer")
    scores_dict = {
        SymptomCategory.ANXIETY_GENERALIZED: 0.7,
        SymptomCategory.SUBSTANCE_CRAVING: 0.4,
    }
    comorbs = ComorbidityAnalyzer.analyze(
        SymptomCategory.MOOD_DEPRESSION, scores_dict)
    rep("Comorbidities found", len(comorbs) > 0)

    # T9: Full pipeline
    print("\n[T9] Full Pipeline Integration")
    pipeline = ClinicalPsychiatryPipeline(n_features=19)
    signal_full = torch.randn(2, 19)
    history = np.random.randn(50, 19)
    report = pipeline.analyze(signal_full, history, {"patient_id": "test_001"})
    rep("Pipeline produces report", "assessments" in report)
    rep("Report has disclaimer", "NOT A CLINICAL DIAGNOSTIC" in report["disclaimer"])
    rep("Report has limitations", len(report["limitations"]) > 0)

    print("\n" + "=" * 88)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 88)
    print("""
  WHAT THIS FRAMEWORK PROVIDES
    • Signal preprocessing (SSC filter)
    • Change-point detection for symptom trajectories (Paper 6)
    • Multi-scale decomposition (carpet graph)
    • Statistical profiling (Paper 9)
    • Symptom classification (research scaffold)
    • Progression prediction (exploratory world model)
    • Comorbidity analysis (pattern-based)

  WHAT THIS FRAMEWORK DOES NOT PROVIDE
    • Clinical diagnosis (requires trained professionals)
    • Mind reading (impossible with any technology)
    • Intent/motivation detection (not inferable from signals)
    • Treatment recommendations (requires clinical judgment)

  REQUIRED BEFORE ANY CLINICAL USE
    • Validation against DSM-5/ICD-11 gold standards
    • IRB approval and informed consent
    • Regulatory approval (FDA, EMA, etc.)
    • Integration with clinical workflow
    • Training for healthcare professionals
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Clinical Psychiatry Analyzer v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--analyze", help="Path to signal file (numpy format)")
    ap.add_argument("--output", help="Output JSON path")
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    if a.analyze:
        signal = torch.tensor(np.load(a.analyze), dtype=torch.float32)
        pipeline = ClinicalPsychiatryPipeline(n_features=signal.shape[-1])
        report = pipeline.analyze(signal)
        if a.output:
            with open(a.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {a.output}")
        else:
            print(json.dumps(report, indent=2))
        return

    ap.print_help()
    print(f"\n{ATTRIBUTION}")
    print("\n⚠️  RESEARCH SCAFFOLD - NOT A CLINICAL DIAGNOSTIC TOOL")


if __name__ == "__main__":
    main()
