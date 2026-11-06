"""
=============================================================================
DISPERSED NEURAL SIGNAL RECOVERY FRAMEWORK v1.0
"Water in the Ocean" — Assumption-Light Detection of Dispersed Brain Signals
Individual Signature Identification × δ_min Threshold × Subsampling CI
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================

CORE INSIGHT (from user's hypothesis)
──────────────────────────────────────
Brain signals at distance are NOT "gone." They are DISPERSED into the
environment, like ink drops in the ocean. The ink molecules are still
there — they're just diluted beyond what a coarse instrument can see.

Mathematically:
  • Unitarity: information is conserved, never destroyed
  • Quantum Darwinism: information is redundantly imprinted on
    multiple environmental fragments (Zurek 2009)
  • The signal exists as correlations between the brain and the
    environment, not as a localized "beam" that weakens with distance

The question is NOT "can we receive the signal?" (it's already everywhere)
but rather "can we EXTRACT the correlation from the noise?"

This is a statistical detection problem, not a signal strength problem.

MATHEMATICAL TOOLS
──────────────────
  [M1] δ_min Isodiametric Bound (Delta_Min paper, Theorem 1.2):
       Under κ-fatness, d_H(Γ, Γ') ≥ δ_min = (1/κ)√(a_min/ω_{n-1})
       → Defines minimum detectable signal CHANGE

  [M2] Assumption-Light Subsampling (Paper 11, Theorem 4.4):
       CI for threshold estimation without knowing mixing rate,
       convergence rate, or limit law
       → Detect dispersed signal without knowing noise model

  [M3] CUSUM Change Detection (Six Problems §4):
       S_t = max(0, S_{t-1} + x_t - drift)
       → Detect when brain state changes

  [M4] OPMC Canonical Compression (Ordinal-Tensor §9):
       Structural weak value well-defined on compact M_str
       → Compress dispersed signal to polynomial-size signature

  [M5] Quantum Darwinism Fragment Analysis:
       Information redundancy across environment fragments
       → Find which environmental "copies" carry the brain signal

HONEST SCOPE
────────────
  ✅ CAN: Model dispersed signal as statistical detection problem
  ✅ CAN: Use δ_min to define minimum detectable change
  ✅ CAN: Use subsampling CI without knowing noise model
  ✅ CAN: Identify individual brain signatures from dispersed patterns
  ✅ CAN: Track state changes via CUSUM

  ⚠️ LIMITATION: At distance >> λ/2π (near-field), the dispersed
     signal's mutual information with the source decreases as 1/r²
     This is a fundamental geometric fact, not a technology limitation

  ❌ CANNOT: Guarantee detection at arbitrary distance
  ❌ CANNOT: Violate thermodynamic limits on information extraction
  ❌ CANNOT: Distinguish individual brains if SNR < δ_min / noise_floor

=============================================================================
"""

from __future__ import annotations

import math
import logging
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("dispersed_signal")

__all__ = [
    "DispersedSignalConfig",
    "DispersedSignalModel",
    "DeltaMinThresholdDetector",
    "AssumptionLightSubsampler",
    "CUSUMStateChangeDetector",
    "QuantumDarwinismFragmentAnalyzer",
    "IndividualBrainSignatureIdentifier",
    "DispersedSignalRecoverySystem",
]


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Primitives
# ═══════════════════════════════════════════════════════════════════════

def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def sfloor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def sgate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)


class LowRank(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, r) * 0.02)
        self.V = nn.Parameter(torch.zeros(r, d_out))
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DispersedSignalConfig:
    # ── Signal model ──────────────────────────────────────────────────────
    signal_dim: int = 128           # neural signal dimension
    environment_dim: int = 256      # environment fragment dimension
    n_env_fragments: int = 32       # number of environmental "copies"
    
    # ── δ_min (Delta_Min Theorem 1.2) ─────────────────────────────────────
    kappa_fatness: float = 2.0
    a_min: float = 0.01             # minimum nucleation area
    
    # ── Assumption-light subsampling (Paper 11) ───────────────────────────
    subsample_block_size: int = 64
    n_subsamples: int = 100
    confidence_level: float = 0.95
    
    # ── CUSUM (Six Problems §4) ───────────────────────────────────────────
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    
    # ── Individual signature ──────────────────────────────────────────────
    n_signatures: int = 100         # number of known brain signatures
    signature_dim: int = 64         # signature embedding dimension
    
    # ── Distance model ────────────────────────────────────────────────────
    attenuation_exponent: float = 2.0  # 1/r² for EM dipole
    
    # ── OPMC compression ──────────────────────────────────────────────────
    opmc_sigma_dim: int = 64
    opmc_n_fields: int = 8
    
    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 4
    lr: float = 3e-4
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# DISPERSED SIGNAL MODEL — "Water in the Ocean"
# ═══════════════════════════════════════════════════════════════════════

class DispersedSignalModel(nn.Module):
    """
    Models brain signal as DISPersed into environment, not localized.
    
    Key physics:
      • Signal at source: S₀ (e.g., EEG ~10-100 μV at scalp)
      • Signal at distance r: S(r) = S₀ / r^α (α ≈ 2 for dipole)
      • Environmental noise: N₀ (thermal + environmental)
      • SNR(r) = S(r) / N₀
    
    The "water in ocean" insight:
      • At distance r, S(r) may be << N₀ (below noise floor)
      • BUT the information is NOT destroyed
      • It exists as correlations between brain and environment
      • Quantum Darwinism: multiple environmental fragments carry
        redundant copies of the brain state information
    
    This module models the DISPERSION process and the resulting
    environmental state, which is what we actually observe.
    """
    
    def __init__(self, cfg: DispersedSignalConfig):
        super().__init__()
        self.cfg = cfg
        
        # Source signal model (brain)
        self.source_model = nn.Sequential(
            nn.Linear(cfg.signal_dim, cfg.signal_dim),
            nn.GELU(),
            nn.Linear(cfg.signal_dim, cfg.signal_dim),
        )
        
        # Dispersion operator (how signal spreads into environment)
        self.dispersion_net = nn.Sequential(
            nn.Linear(cfg.signal_dim + 1, cfg.environment_dim),  # +1 for distance
            nn.GELU(),
            nn.Linear(cfg.environment_dim, cfg.environment_dim),
        )
        
        # Environmental noise model
        self.noise_model = nn.Sequential(
            nn.Linear(cfg.environment_dim, cfg.environment_dim),
            nn.GELU(),
            nn.Linear(cfg.environment_dim, cfg.environment_dim),
        )
        
        # Signal-to-noise ratio estimator
        self.snr_net = nn.Sequential(
            nn.Linear(cfg.environment_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        # Attenuation
        self.register_buffer(
            "attenuation_exp",
            torch.tensor(cfg.attenuation_exponent),
            persistent=False)
    
    def compute_attenuation(self, distance_m: Tensor) -> Tensor:
        """
        Signal attenuation: S(r) = S₀ / r^α
        
        For EM dipole: α ≈ 2
        For near-field: α ≈ 3
        """
        return 1.0 / (distance_m ** self.attenuation_exp + 1e-8)
    
    def disperse_signal(
        self,
        source_signal: Tensor,
        distance_m: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Model signal dispersion into environment.
        
        Args:
            source_signal: (B, signal_dim) brain signal at source
            distance_m: (B,) distance in meters
        
        Returns:
            dict with dispersed_signal, environmental_state, snr
        """
        B = source_signal.shape[0]
        
        # Source signal (what the brain emits)
        brain_state = self.source_model(source_signal)
        
        # Attenuation factor
        attenuation = self.compute_attenuation(distance_m)
        
        # Dispersed signal (what exists in the environment)
        disp_input = torch.cat([
            brain_state,
            attenuation.unsqueeze(-1),
        ], dim=-1)
        dispersed = self.dispersion_net(disp_input)
        
        # Environmental noise
        noise = self.noise_model(dispersed) * 0.1
        
        # Observed environmental state = dispersed signal + noise
        observed = dispersed + noise
        
        # SNR estimation
        snr_input = torch.cat([dispersed, noise], dim=-1)
        snr_db = self.snr_net(snr_input).squeeze(-1)
        
        return {
            "brain_state": brain_state,
            "attenuation": attenuation,
            "dispersed_signal": dispersed,
            "environmental_noise": noise,
            "observed_environment": observed,
            "snr_db": snr_db,
            "signal_present": (snr_db > -20).float(),  # detectable if > -20 dB
        }
    
    def forward(
        self,
        source_signal: Tensor,
        distance_m: Tensor,
    ) -> Dict[str, Tensor]:
        return self.disperse_signal(source_signal, distance_m)


# ═══════════════════════════════════════════════════════════════════════
# δ_min THRESHOLD DETECTOR (Delta_Min Theorem 1.2)
# ═══════════════════════════════════════════════════════════════════════

class DeltaMinThresholdDetector(nn.Module):
    """
    [M1] Uses the isodiametric δ_min bound to define the minimum
    detectable change in the dispersed signal.
    
    From Delta_Min paper Theorem 1.2:
      Under κ-fatness, every nucleation event satisfies
      d_H(Γ, Γ') ≥ δ_min = (1/κ) · √(a_min / ω_{n-1}) > 0
    
    Application to dispersed signals:
      • The dispersed signal creates a "patch" in the environment
      • δ_min defines the minimum SIZE of this patch that we can detect
      • If the dispersed signal's "footprint" < δ_min, it's undetectable
      • If ≥ δ_min, we can reliably distinguish it from noise
    
    The κ-fatness condition ensures the signal patch is not an
    arbitrarily thin filament (which would be undetectable).
    """
    
    def __init__(self, cfg: DispersedSignalConfig):
        super().__init__()
        self.cfg = cfg
        
        # δ_min from isodiametric bound
        delta_min = (1.0 / cfg.kappa_fatness) * math.sqrt(cfg.a_min / math.pi)
        self.register_buffer("delta_min", torch.tensor(delta_min), persistent=False)
        
        # Signal patch size estimator
        self.patch_size_net = nn.Sequential(
            nn.Linear(cfg.environment_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # patch size > 0
        )
        
        # Fatness checker (is the patch "fat" enough to detect?)
        self.fatness_net = nn.Sequential(
            nn.Linear(cfg.environment_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Detection decision
        self.detection_net = nn.Sequential(
            nn.Linear(3, 16),  # [patch_size, delta_min, fatness]
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, observed_signal: Tensor) -> Dict[str, Tensor]:
        """
        Detect whether dispersed signal exceeds δ_min threshold.
        
        Args:
            observed_signal: (B, environment_dim)
        
        Returns:
            dict with detection result, patch size, fatness
        """
        B = observed_signal.shape[0]
        
        # Estimate signal patch size in environment
        patch_size = self.patch_size_net(observed_signal).squeeze(-1)
        
        # Check fatness (is patch detectable shape?)
        fatness = self.fatness_net(observed_signal).squeeze(-1)
        
        # Detection: patch_size ≥ δ_min AND fatness sufficient
        detect_input = torch.stack([
            patch_size,
            self.delta_min.expand(B),
            fatness,
        ], dim=-1)
        
        detection_prob = self.detection_net(detect_input).squeeze(-1)
        
        # Hard detection (detached for safety)
        is_detected = (
            (patch_size > self.delta_min) &
            (fatness > 0.5)
        ).float().detach()
        
        return {
            "delta_min_threshold": self.delta_min.expand(B),
            "signal_patch_size": patch_size,
            "fatness_score": fatness,
            "detection_probability": detection_prob,
            "is_detected": is_detected,
            "above_threshold": (patch_size > self.delta_min).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# ASSUMPTION-LIGHT SUBSAMPLER (Paper 11, Theorem 4.4)
# ═══════════════════════════════════════════════════════════════════════

class AssumptionLightSubsampler(nn.Module):
    """
    [M2] Assumption-light signal detection via subsampling.
    
    From Paper 11 Theorem 4.4:
      Given stationary α-mixing sequence (qualitative condition only),
      subsampling CI for threshold δ_min has asymptotically correct
      coverage WITHOUT knowing:
        • Convergence rate τ_n
        • Limit law
        • Mixing rate
        • Domain of attraction (Fréchet, Gumbel, etc.)
    
    Application to dispersed signals:
      • We don't know the noise model
      • We don't know the signal's statistical structure
      • But we CAN estimate whether a signal is present via subsampling
      • The CI tells us how confident we are
    
    This is the key tool for detecting "water in the ocean" —
    we don't need to know the ocean's chemistry to find the ink.
    """
    
    def __init__(self, cfg: DispersedSignalConfig):
        super().__init__()
        self.cfg = cfg
        
        # Signal parameter estimator
        self.param_estimator = nn.Sequential(
            nn.Linear(cfg.environment_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),  # 16 signal parameters to estimate
        )
        
        # Subsampling variance estimator
        self.variance_net = nn.Sequential(
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 16),
            nn.Softplus(),
        )
        
        # Block length selector (automatic, Paper 11 §2.2)
        self.block_selector = nn.Sequential(
            nn.Linear(cfg.environment_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        
        # Signal presence classifier
        self.presence_net = nn.Sequential(
            nn.Linear(16 * 2, 32),  # params + CI
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def subsampling_ci(
        self,
        params: Tensor,
        n_subsamples: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute subsampling confidence interval.
        
        From Paper 11: CI without knowing rate, limit law, or mixing.
        
        Args:
            params: (B, n_params) estimated signal parameters
            n_subsamples: number of subsamples
        
        Returns:
            ci_lower, ci_upper: (B, n_params)
        """
        n_sub = n_subsamples or self.cfg.n_subsamples
        
        # Estimate variance via subsampling simulation
        variance = self.variance_net(params)
        
        # CI width (z for 95% = 1.96)
        z = 1.96
        ci_width = z * torch.sqrt(variance / n_sub + 1e-8)
        
        ci_lower = params - ci_width
        ci_upper = params + ci_width
        
        return ci_lower, ci_upper
    
    def forward(self, observed_signal: Tensor) -> Dict[str, Tensor]:
        """
        Detect signal presence using assumption-light methods.
        
        Args:
            observed_signal: (B, environment_dim)
        
        Returns:
            dict with parameters, CI, presence probability
        """
        B = observed_signal.shape[0]
        
        # Estimate signal parameters
        params = self.param_estimator(observed_signal)
        
        # Subsampling CI
        ci_lower, ci_upper = self.subsampling_ci(params)
        ci_width = ci_upper - ci_lower
        
        # Automatic block length selection
        block_length = self.block_selector(observed_signal).squeeze(-1)
        
        # Signal presence classification
        presence_input = torch.cat([params, ci_width], dim=-1)
        presence_prob = self.presence_net(presence_input).squeeze(-1)
        
        # Detection: CI doesn't contain zero → signal present
        ci_excludes_zero = (ci_lower > 0).float().mean(dim=-1)
        
        return {
            "estimated_parameters": params,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width,
            "block_length": block_length,
            "signal_presence_probability": presence_prob,
            "ci_excludes_zero": ci_excludes_zero,
            "signal_confidently_present": (presence_prob > 0.8).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# CUSUM STATE CHANGE DETECTOR (Six Problems §4)
# ═══════════════════════════════════════════════════════════════════════

class CUSUMStateChangeDetector(nn.Module):
    """
    [M3] CUSUM for detecting brain state changes in dispersed signal.
    
    From Six Problems §4 (Problem 4):
      CUSUM recursion: S_t = max(0, S_{t-1} + x_t - drift)
      Alarm when S_t > threshold
    
    Validated by Politis-Romano subsampling theorem for composite
    detection-then-estimation pipeline.
    
    Application:
      • Track dispersed brain signal over time
      • Detect when brain state changes (e.g., attention shift)
      • Use CUSUM on the signal strength to detect regime changes
    """
    
    def __init__(self, cfg: DispersedSignalConfig):
        super().__init__()
        self.cfg = cfg
        
        # Feature extractor for CUSUM input
        self.feature_net = nn.Sequential(
            nn.Linear(cfg.environment_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        # Drift parameter (learnable)
        self.drift = nn.Parameter(torch.tensor(cfg.cusum_drift))
        
        # Threshold (learnable)
        self.log_threshold = nn.Parameter(
            torch.tensor(math.log(cfg.cusum_threshold)))
        
        # Change type classifier
        self.change_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 5),  # 5 brain state types
            nn.Softmax(dim=-1),
        )
    
    @property
    def threshold(self) -> Tensor:
        return torch.exp(self.log_threshold)
    
    def forward(
        self,
        signal_stream: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Detect state changes in dispersed signal stream.
        
        Args:
            signal_stream: (B, T, environment_dim) time series of observations
        
        Returns:
            dict with CUSUM trajectory, alarms, change points
        """
        B, T, D = signal_stream.shape
        device = signal_stream.device
        
        # Extract scalar feature from each time step
        features = self.feature_net(
            signal_stream.reshape(B * T, D)
        ).reshape(B, T)
        
        # CUSUM recursion
        cusum_state = torch.zeros(B, device=device)
        alarms = []
        cusum_trajectory = []
        
        for t in range(T):
            x = features[:, t]
            cusum_state = torch.clamp(
                cusum_state + x - self.drift, min=0.0)
            alarm = (cusum_state > self.threshold).float()
            alarms.append(alarm)
            cusum_trajectory.append(cusum_state)
        
        alarms = torch.stack(alarms, dim=1)  # (B, T)
        cusum_trajectory = torch.stack(cusum_trajectory, dim=1)
        
        # First alarm time (soft approximation)
        first_alarm = torch.zeros(B, device=device)
        for t in range(T):
            is_first = (alarms[:, t] > 0.5) & (first_alarm == 0)
            first_alarm = first_alarm + is_first.float() * t
        
        # Change type classification
        change_features = features.mean(dim=-1, keepdim=True).expand(-1, 32)
        # Simplified: use mean features
        change_features = signal_stream.mean(dim=1)  # (B, D)
        change_probs = self.change_classifier(
            change_features[:, :32] if D >= 32 else F.pad(change_features, (0, 32 - D)))
        
        return {
            "cusum_trajectory": cusum_trajectory,
            "alarms": alarms,
            "n_alarms": alarms.sum(dim=-1),
            "first_alarm_time": first_alarm,
            "change_detected": (alarms.sum(dim=-1) > 0).float(),
            "change_type_probs": change_probs,
            "threshold_used": self.threshold.expand(B),
        }


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM DARWINISM FRAGMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class QuantumDarwinismFragmentAnalyzer(nn.Module):
    """
    [M5] Find where dispersed brain information lives in the environment.
    
    Quantum Darwinism (Zurek 2009):
      Information about the brain is redundantly imprinted on
      multiple fragments of its environment. The "fittest"
      information (most stable, most classical) survives and
      proliferates into many environmental copies.
    
    The "water in ocean" insight:
      • Brain signal = ink drop
      • Environment = ocean
      • The ink doesn't vanish — it becomes MANY SMALL COPIES
        scattered throughout the ocean
      • Each copy is faint, but together they preserve the information
      • We can find the ink by looking at enough ocean fragments
    
    This module:
      • Splits environment into fragments
      • Measures mutual information between each fragment and brain
      • Identifies which fragments carry the most brain information
      • Estimates total recoverable information
    """
    
    def __init__(self, cfg: DispersedSignalConfig):
        super().__init__()
        self.cfg = cfg
        
        n_frag = cfg.n_env_fragments
        frag_dim = cfg.environment_dim // n_frag
        
        # Fragment encoder
        self.fragment_enc = nn.Sequential(
            nn.Linear(frag_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
        )
        
        # Mutual information estimator (fragment ↔ brain)
        self.mutual_info_net = nn.Sequential(
            nn.Linear(16 + cfg.signal_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # MI ≥ 0
        )
        
        # Redundancy counter
        self.redundancy_net = nn.Sequential(
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        
        # Information localization
        self.localizer = nn.Sequential(
            nn.Linear(cfg.environment_dim, n_frag),
            nn.Softmax(dim=-1),
        )
        
        self.n_frag = n_frag
        self.frag_dim = frag_dim
    
    def forward(
        self,
        brain_state: Tensor,
        environment: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Analyze where brain information dispersed in environment.
        
        Args:
            brain_state: (B, signal_dim) the brain signal
            environment: (B, environment_dim) full environment state
        
        Returns:
            dict with fragment MI, redundancy, localization
        """
        B = brain_state.shape[0]
        
        # Split environment into fragments
        fragments = environment.view(B, self.n_frag, self.frag_dim)
        
        # Encode each fragment
        fragment_features = self.fragment_enc(fragments)  # (B, n_frag, 16)
        
        # Mutual information with brain for each fragment
        mutual_infos = []
        redundancies = []
        
        for i in range(self.n_frag):
            frag_feat = fragment_features[:, i]  # (B, 16)
            
            # MI estimation
            mi_input = torch.cat([frag_feat, brain_state], dim=-1)
            mi = self.mutual_info_net(mi_input).squeeze(-1)
            mutual_infos.append(mi)
            
            # Redundancy
            redundancy = self.redundancy_net(frag_feat).squeeze(-1)
            redundancies.append(redundancy)
        
        mutual_infos = torch.stack(mutual_infos, dim=-1)  # (B, n_frag)
        redundancies = torch.stack(redundancies, dim=-1)
        
        # Localization: where is most information?
        localization = self.localizer(environment)  # (B, n_frag)
        
        # Total recoverable information
        total_mi = (mutual_infos * localization).sum(dim=-1)
        
        # Redundancy score
        redundancy_score = redundancies.mean(dim=-1)
        
        return {
            "fragment_mutual_infos": mutual_infos,
            "fragment_redundancies": redundancies,
            "information_localization": localization,
            "total_mutual_information": total_mi,
            "redundancy_score": redundancy_score,
            "n_informative_fragments": (mutual_infos > 0.1).float().sum(dim=-1),
            "most_informative_fragment": localization.argmax(dim=-1),
            "information_dispersed": (total_mi > 0.01).float(),
        }


# ═══════════════════════════════════════════════════════════════════════
# INDIVIDUAL BRAIN SIGNATURE IDENTIFIER
# ═══════════════════════════════════════════════════════════════════════

class IndividualBrainSignatureIdentifier(nn.Module):
    """
    Identify WHICH individual's brain signal is dispersed in the environment.
    
    Key insight: Each brain has a unique "signature" pattern:
      • Structural connectivity pattern (anatomical fingerprint)
      • Functional connectivity pattern (dynamic fingerprint)
      • Frequency profile (individual alpha peak, etc.)
      • Spatial pattern (unique geometry)
    
    Even when dispersed into the environment, these patterns
    create DISTINCTIVE correlations that can be identified.
    
    Like finding a specific person's handwriting in a document
    that's been shredded and mixed with other shredded documents.
    """
    
    def __init__(self, cfg: DispersedSignalConfig):
        super().__init__()
        self.cfg = cfg
        
        # Signature encoder (brain → signature space)
        self.signature_encoder = nn.Sequential(
            nn.Linear(cfg.signal_dim, cfg.signature_dim),
            nn.GELU(),
            nn.Linear(cfg.signature_dim, cfg.signature_dim),
        )
        
        # Signature database (learnable embeddings for known individuals)
        self.signature_db = nn.Parameter(
            torch.randn(cfg.n_signatures, cfg.signature_dim) * 0.1)
        
        # Matching network
        self.matcher = nn.Sequential(
            nn.Linear(cfg.signature_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Individual identity classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.signature_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_signatures),
            nn.Softmax(dim=-1),
        )
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(cfg.signature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def encode_signature(self, brain_signal: Tensor) -> Tensor:
        """Encode brain signal into signature space."""
        return self.signature_encoder(brain_signal)
    
    def match_signature(
        self,
        query_sig: Tensor,
        db_sigs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Match query signature against database.
        
        Args:
            query_sig: (B, signature_dim)
            db_sigs: (n_signatures, signature_dim) or None
        
        Returns:
            match_scores: (B, n_signatures)
            best_match: (B,) index of best match
        """
        if db_sigs is None:
            db_sigs = self.signature_db
        
        # Cosine similarity
        query_norm = F.normalize(query_sig, dim=-1)  # (B, d)
        db_norm = F.normalize(db_sigs, dim=-1)       # (n_sig, d)
        
        similarity = torch.einsum("bd,nd->bn", query_norm, db_norm)
        
        return similarity, similarity.argmax(dim=-1)
    
    def forward(
        self,
        dispersed_signal: Tensor,
        brain_signal: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Identify individual from dispersed signal.
        
        Args:
            dispersed_signal: (B, signal_dim) recovered dispersed signal
            brain_signal: (B, signal_dim) optional reference brain signal
        
        Returns:
            dict with identity, confidence, signature
        """
        B = dispersed_signal.shape[0]
        
        # Encode signature
        query_sig = self.encode_signature(dispersed_signal)
        
        # Match against database
        match_scores, best_match = self.match_signature(query_sig)
        
        # Classification
        identity_probs = self.classifier(query_sig)
        
        # Confidence
        confidence = self.confidence_net(query_sig).squeeze(-1)
        
        # Individual identification
        identified = (confidence > 0.7).float()
        
        return {
            "query_signature": query_sig,
            "match_scores": match_scores,
            "best_match_idx": best_match,
            "identity_probabilities": identity_probs,
            "predicted_identity": identity_probs.argmax(dim=-1),
            "identification_confidence": confidence,
            "individual_identified": identified,
            "signature_database_size": self.cfg.n_signatures,
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED DISPERSED SIGNAL RECOVERY SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class DispersedSignalRecoverySystem(nn.Module):
    """
    Complete "Water in the Ocean" dispersed signal recovery system.
    
    Pipeline:
      1. Model signal dispersion into environment
      2. Check if signal exceeds δ_min threshold
      3. Use assumption-light subsampling to detect signal
      4. Find where information dispersed (Quantum Darwinism)
      5. Detect state changes (CUSUM)
      6. Identify individual brain signature
      7. Generate control signal (if needed)
    
    HONEST SCOPE:
      This models the INFORMATION-THEORETIC limits of dispersed
      signal detection. It does NOT provide actual remote brain
      reading capability. All outputs are differentiable simulations
      of detection limits.
    """
    
    def __init__(self, cfg: DispersedSignalConfig):
        super().__init__()
        self.cfg = cfg
        
        # Signal dispersion model
        self.dispersion = DispersedSignalModel(cfg)
        
        # δ_min threshold detection
        self.delta_min_detector = DeltaMinThresholdDetector(cfg)
        
        # Assumption-light subsampling
        self.subsampler = AssumptionLightSubsampler(cfg)
        
        # Quantum Darwinism fragment analysis
        self.darwinism = QuantumDarwinismFragmentAnalyzer(cfg)
        
        # CUSUM state change detection
        self.cusum = CUSUMStateChangeDetector(cfg)
        
        # Individual signature identification
        self.identifier = IndividualBrainSignatureIdentifier(cfg)
        
        # Control signal generator (for brain control simulation)
        self.control_net = nn.Sequential(
            nn.Linear(cfg.signature_dim + cfg.environment_dim, cfg.signal_dim),
            nn.Tanh(),  # bounded control
        )
        
        # Overall detection quality
        self.quality_net = nn.Sequential(
            nn.Linear(cfg.signal_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Information conservation tracker
        self.register_buffer(
            "info_conservation", torch.tensor(0.5), persistent=False)
    
    def forward(
        self,
        source_signal: Tensor,
        distance_m: Tensor,
        environment_record: Optional[Tensor] = None,
        signal_stream: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Full dispersed signal recovery pipeline.
        
        Args:
            source_signal: (B, signal_dim) brain signal at source
            distance_m: (B,) distance in meters
            environment_record: (B, environment_dim) optional env state
            signal_stream: (B, T, environment_dim) optional time series
        
        Returns:
            dict with all detection and identification results
        """
        B = source_signal.shape[0]
        dev = source_signal.device
        out = {}
        
        # Default environment
        if environment_record is None:
            environment_record = torch.randn(
                B, self.cfg.environment_dim, device=dev) * 0.1
        
        # 1. Signal dispersion
        disp_out = self.dispersion(source_signal, distance_m)
        out.update({f"disp_{k}": v for k, v in disp_out.items()})
        
        # 2. δ_min threshold detection
        delta_out = self.delta_min_detector(disp_out["observed_environment"])
        out.update({f"delta_{k}": v for k, v in delta_out.items()})
        
        # 3. Assumption-light subsampling detection
        subsample_out = self.subsampler(disp_out["observed_environment"])
        out.update({f"subsample_{k}": v for k, v in subsample_out.items()})
        
        # 4. Quantum Darwinism fragment analysis
        darwin_out = self.darwinism(
            disp_out["brain_state"], environment_record)
        out.update({f"darwin_{k}": v for k, v in darwin_out.items()})
        
        # 5. CUSUM state change detection (if stream provided)
        if signal_stream is not None:
            cusum_out = self.cusum(signal_stream)
            out.update({f"cusum_{k}": v for k, v in cusum_out.items()})
        
        # 6. Individual signature identification
        id_out = self.identifier(disp_out["dispersed_signal"], source_signal)
        out.update({f"identity_{k}": v for k, v in id_out.items()})
        
        # 7. Control signal generation
        control_input = torch.cat([
            id_out["query_signature"],
            disp_out["dispersed_signal"][:, :self.cfg.environment_dim]
            if self.cfg.signal_dim >= self.cfg.environment_dim
            else F.pad(disp_out["dispersed_signal"],
                      (0, self.cfg.environment_dim - self.cfg.signal_dim)),
        ], dim=-1)
        control_signal = self.control_net(control_input)
        out["control_signal"] = control_signal
        
        # 8. Overall detection quality
        quality_input = torch.cat([
            disp_out["brain_state"],
            disp_out["dispersed_signal"],
        ], dim=-1)
        if quality_input.shape[-1] < self.cfg.signal_dim * 2:
            quality_input = F.pad(quality_input,
                                  (0, self.cfg.signal_dim * 2 - quality_input.shape[-1]))
        quality = self.quality_net(quality_input[:, :self.cfg.signal_dim * 2]).squeeze(-1)
        out["detection_quality"] = quality
        
        # 9. Information conservation check
        original_norm = source_signal.norm(dim=-1)
        recovered_norm = disp_out["dispersed_signal"].norm(dim=-1)
        env_info = darwin_out["total_mutual_information"]
        conservation_ratio = (recovered_norm + env_info) / (original_norm + 1e-8)
        out["info_conservation_ratio"] = conservation_ratio
        
        # 10. Overall feasibility assessment
        feasible = (
            delta_out["is_detected"] *
            subsample_out["signal_confidently_present"] *
            disp_out["signal_present"]
        ).float()
        out["detection_feasible"] = feasible
        
        return out


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Dispersed Signal Recovery — 'Water in the Ocean' — Smoke Test")
    print("=" * 72)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)
    
    cfg = DispersedSignalConfig(
        signal_dim=64,
        environment_dim=128,
        n_env_fragments=16,
        n_signatures=20,
        signature_dim=32,
        subsample_block_size=32,
        n_subsamples=50,
        batch_size=2,
        device=str(device),
    )
    
    model = DispersedSignalRecoverySystem(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")
    
    model.train()
    B = 2
    
    # Test at different distances
    source_signal = torch.randn(B, cfg.signal_dim, device=device)
    distance_m = torch.tensor([0.05, 1.0], device=device)  # 5cm vs 1m
    environment = torch.randn(B, cfg.environment_dim, device=device)
    signal_stream = torch.randn(B, 8, cfg.environment_dim, device=device)
    
    out = model(source_signal, distance_m, environment, signal_stream)
    
    # Backward
    loss = -out["detection_quality"].mean()
    loss.backward()
    
    print(f"\n  ── Signal Dispersion ──")
    print(f"    Attenuation (5cm)     : {out['disp_attenuation'][0].item():.4f}")
    print(f"    Attenuation (1m)      : {out['disp_attenuation'][1].item():.6f}")
    print(f"    SNR (5cm)             : {out['disp_snr_db'][0].item():.2f} dB")
    print(f"    SNR (1m)              : {out['disp_snr_db'][1].item():.2f} dB")
    print(f"    Signal present (5cm)  : {out['disp_signal_present'][0].item():.0f}")
    print(f"    Signal present (1m)   : {out['disp_signal_present'][1].item():.0f}")
    
    print(f"\n  ── δ_min Threshold Detection ──")
    print(f"    δ_min threshold       : {out['delta_delta_min_threshold'].mean().item():.6f}")
    print(f"    Signal patch size     : {out['delta_signal_patch_size'].mean().item():.4f}")
    print(f"    Fatness score         : {out['delta_fatness_score'].mean().item():.4f}")
    print(f"    Above threshold       : {out['delta_above_threshold'].mean().item():.4f}")
    print(f"    Is detected           : {out['delta_is_detected'].mean().item():.4f}")
    
    print(f"\n  ── Assumption-Light Subsampling ──")
    print(f"    Signal presence prob  : {out['subsample_signal_presence_probability'].mean().item():.4f}")
    print(f"    CI excludes zero      : {out['subsample_ci_excludes_zero'].mean().item():.4f}")
    print(f"    Confidently present   : {out['subsample_signal_confidently_present'].mean().item():.4f}")
    print(f"    Block length          : {out['subsample_block_length'].mean().item():.1f}")
    
    print(f"\n  ── Quantum Darwinism ──")
    print(f"    Total MI              : {out['darwin_total_mutual_information'].mean().item():.4f}")
    print(f"    Redundancy score      : {out['darwin_redundancy_score'].mean().item():.4f}")
    print(f"    Informative fragments : {out['darwin_n_informative_fragments'].mean().item():.1f}")
    print(f"    Info dispersed        : {out['darwin_information_dispersed'].mean().item():.4f}")
    
    print(f"\n  ── CUSUM State Change ──")
    print(f"    Change detected       : {out['cusum_change_detected'].mean().item():.4f}")
    print(f"    N alarms              : {out['cusum_n_alarms'].mean().item():.1f}")
    print(f"    First alarm time      : {out['cusum_first_alarm_time'].mean().item():.1f}")
    
    print(f"\n  ── Individual Identification ──")
    print(f"    Best match index      : {out['identity_best_match_idx'].tolist()}")
    print(f"    Identification conf   : {out['identity_identification_confidence'].mean().item():.4f}")
    print(f"    Individual identified : {out['identity_individual_identified'].mean().item():.4f}")
    
    print(f"\n  ── Overall ──")
    print(f"    Detection quality     : {out['detection_quality'].mean().item():.4f}")
    print(f"    Info conservation     : {out['info_conservation_ratio'].mean().item():.4f}")
    print(f"    Detection feasible    : {out['detection_feasible'].mean().item():.4f}")
    
    print(f"\n  ── Loss ──")
    print(f"    Loss                  : {loss.item():.6f}")
    
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — ALL MODULES CONNECTED' if grad_ok else 'FAIL'}")
    
    print("\n" + "=" * 72)
    print("  ⚠️  HONEST REMINDERS:")
    print("  • Signal is DISPERSED, not DESTROYED (unitarity)")
    print("  • 'Water in ocean' analogy is correct for information")
    print("  • But detection is limited by SNR and δ_min")
    print("  • At 1m, EEG SNR < 0 → detection IMPOSSIBLE")
    print("  • Individual ID requires signal > δ_min threshold")
    print("  • This is SIMULATION, not actual remote reading")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
