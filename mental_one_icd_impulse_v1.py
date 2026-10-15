# =============================================================================
# MENTAL ONE – Impulse Control & Addiction Neuroimaging Module  v1.0
# =============================================================================
# Developer   : Yoon A Limsuwan / MSPS NETWORK
#               MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID       : 0009-0008-2374-0788
# GitHub      : yoonalimsuwan
# License     : MIT
# Year        : 2026
#
# AI Co-Developers:
#   - Claude (Anthropic)  — module architecture, SSC extension, biomarker
#                           feature extraction, EEG/MEG/MRI preprocessing,
#                           clinical scale scoring, risk stratification engine
#   - GPT   (OpenAI)      — literature cross-check, DSM-5/ICD-11 mapping
#   - Gemini (Google)     — multi-modal fusion scaffolding
#   - DeepSeek            — alternative classifier verification
#
# PART OF: MENTAL ONE cluster (ONE Ecosystem)
#
# PURPOSE
# ───────
# Extends mental_one.py with three DSM-5/ICD-11-aligned clinical modules
# for conditions characterised by deficient inhibitory control, assessed
# via EEG, MEG, and structural/functional MRI:
#
#   1. SubstanceUseDisorderAssessor  (DSM-5 §302-305; ICD-11 §6C4x)
#      – Drug & alcohol addiction neuroimaging biomarkers
#
#   2. CompulsiveSexualBehaviourAssessor  (ICD-11: 6C72)
#      – Hypersexuality / compulsive sexual behaviour disorder
#        (includes DSM-5 proposed hypersexual disorder criteria)
#
#   3. ImpulsiveAggressionAssessor  (DSM-5 §312.34 IED; ICD-11 §6C73)
#      – Intermittent Explosive Disorder; impulse-control deficit
#        associated with harm to others
#
# NEUROIMAGING BIOMARKERS (peer-reviewed basis)
# ─────────────────────────────────────────────
#   EEG:  P300 amplitude/latency · N200 · ERN (error-related negativity)
#         · Alpha/beta/theta band power · Frontal alpha asymmetry (FAA)
#         · Theta-beta ratio (TBR) · Go/NoGo inhibitory indices
#   MEG:  Beta-band desynchronisation · Gamma oscillations in reward/limbic
#         circuits · Prefrontal inhibitory control signals
#   MRI:  ROI grey-matter volume: OFC, vmPFC, dlPFC, ACC, amygdala,
#         nucleus accumbens / ventral striatum
#   fMRI: Resting-state connectivity (amygdala–PFC, reward circuit)
#         · Cue-reactivity in striatum/ACC
#
# CLINICAL SCREENING SCALES INCLUDED
# ────────────────────────────────────
#   SUD       : AUDIT-10, DAST-10, CAGE-4
#   CSBD      : HBI-19, SAST-R (19 items), CSBD-19
#   Aggression: BPAQ-29, OAS (Overt Aggression Scale), AQ (Aggression
#               Questionnaire short form)
#
# INTEGRATION
# ───────────
# This module imports from mental_one.py. Ensure mental_one.py is on
# PYTHONPATH or in the same directory before importing this file.
#
#   from mental_one_icd_impulse_v1 import ImpulseControlRiskEngine
#   engine = ImpulseControlRiskEngine()
#   report = engine.run(eeg_file="patient.edf", mri_file="t1.nii.gz")
#
# =============================================================================

from __future__ import annotations
import math, os, json, logging, warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("mental_one.icd_impulse")
logger.setLevel(logging.INFO)

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("MNE-Python not found. EEG/MEG loading will be limited.")

try:
    import nibabel as nib
    from nilearn import masking, image, signal
    from nilearn.connectome import ConnectivityMeasure
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    warnings.warn("Nilearn/Nibabel not found. MRI/fMRI loading will be limited.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ── Try importing MENTAL ONE base ────────────────────────────────────────────
try:
    from mental_one import (
        OPTIMAL_DEVICE, ALL_PSYCHIATRIC_DISORDERS,
        MultiModalDataLoader, SSCClassifier, CSOCKernel,
        SOCController, DifferentiableRG, MentalHealthEvolution,
        InterventionDesigner, soft_clamp,
    )
    HAS_MENTAL_ONE_BASE = True
except ImportError:
    HAS_MENTAL_ONE_BASE = False
    warnings.warn(
        "mental_one.py base not found on PYTHONPATH. "
        "Standalone mode: base classes will be defined locally."
    )
    # Minimal fallback so the module is importable in isolation
    OPTIMAL_DEVICE = torch.device("cpu")
    ALL_PSYCHIATRIC_DISORDERS = [
        'MDD', 'Bipolar', 'Schizophrenia', 'PTSD', 'Panic',
        'Conversion', 'Dissociative', 'Somatic', 'Parasomnia', 'Healthy'
    ]

    def soft_clamp(x: torch.Tensor,
                   lo: float = -10.0, hi: float = 10.0) -> torch.Tensor:
        return lo + F.softplus(x - lo) - F.softplus(x - hi)

# ── Extend disorder list ──────────────────────────────────────────────────────
IMPULSE_DISORDERS = ['SUD', 'CSBD', 'IED']
EXTENDED_DISORDERS = ALL_PSYCHIATRIC_DISORDERS + IMPULSE_DISORDERS

# =============================================================================
# 1.  CONSTANTS – Frequency bands and MRI ROI definitions
# =============================================================================
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta":  (1.0,   4.0),
    "theta":  (4.0,   8.0),
    "alpha":  (8.0,  13.0),
    "beta":   (13.0, 30.0),
    "gamma":  (30.0, 45.0),
}

# MNI-space approximate ROI centre coordinates (mm) for parcellation
# Sources: AAL2 / Harvard-Oxford cortical atlas
MRI_ROI_MNI: Dict[str, Tuple[float, float, float]] = {
    "OFC":               (  0, 42, -18),   # Orbitofrontal cortex
    "vmPFC":             (  0, 50,  -8),   # Ventromedial PFC
    "dlPFC_L":           (-40, 30,  32),   # Dorsolateral PFC – left
    "dlPFC_R":           ( 40, 30,  32),   # Dorsolateral PFC – right
    "ACC":               (  0, 32,  20),   # Anterior cingulate cortex
    "Amygdala_L":        (-24, -4, -20),   # Amygdala – left
    "Amygdala_R":        ( 24, -4, -20),   # Amygdala – right
    "NAcc_L":            (-10,  8, -10),   # Nucleus accumbens – left
    "NAcc_R":            ( 10,  8, -10),   # Nucleus accumbens – right
    "Insula_L":          (-36,  8,   4),   # Insula – left
    "Insula_R":          ( 36,  8,   4),   # Insula – right
}

# ERP component time windows (ms post-stimulus) for detection
ERP_WINDOWS_MS: Dict[str, Tuple[int, int]] = {
    "P300": (280, 450),
    "N200": (180, 280),
    "ERN":  ( 0,  100),    # post-error response-locked
    "Pe":   (100, 300),    # post-error positivity
    "N2":   (180, 280),    # NoGo N2
    "P3b":  (300, 500),    # P3b (target detection)
}


# =============================================================================
# 2.  IMPULSE/ADDICTION DATA LOADER
#     Extends MultiModalDataLoader with disorder-specific preprocessing
# =============================================================================
class ImpulseAddictionDataLoader:
    """
    Multi-modal data ingestion and preprocessing for impulse-control and
    addiction neuroimaging studies.

    Extends MultiModalDataLoader from mental_one.py with:
      • EEG P300/N200/ERN/alpha-asymmetry specific epoching & baseline correction
      • MEG beta desynchronisation extraction
      • MRI ROI-based grey-matter volume extraction
      • fMRI resting-state connectivity matrices (reward circuit)
    """

    def __init__(self, device: torch.device = OPTIMAL_DEVICE,
                 sfreq: float = 256.0):
        self.device = device
        self.sfreq  = sfreq
        if HAS_MENTAL_ONE_BASE:
            self._base = MultiModalDataLoader(device)

    # ── EEG ──────────────────────────────────────────────────────────────────
    def load_eeg_raw(self, file_path: str,
                     montage: str = 'standard_1020') -> "mne.io.Raw":
        """Load raw EEG as MNE Raw object (EDF or CSV)."""
        if not HAS_MNE:
            raise ImportError("MNE-Python required for EEG loading.")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".edf":
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif ext in (".fif", ".fif.gz"):
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        elif ext == ".csv":
            import pandas as pd
            df   = pd.read_csv(file_path)
            data = df.values.T.astype(np.float64)
            ch_names = df.columns.tolist()
            info = mne.create_info(ch_names, self.sfreq, ch_types='eeg')
            raw  = mne.io.RawArray(data, info, verbose=False)
        else:
            raise ValueError(f"Unsupported EEG format: {ext}")
        raw.pick_types(eeg=True, verbose=False)
        if raw.info['sfreq'] != self.sfreq:
            raw.resample(self.sfreq, verbose=False)
        raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
        raw.set_montage(montage, on_missing='warn', verbose=False)
        return raw

    def extract_erp_features(self, raw: "mne.io.Raw",
                              events_array: Optional[np.ndarray] = None,
                              tmin: float = -0.2, tmax: float = 0.8,
                              baseline: Tuple[float, float] = (-0.2, 0.0)
                              ) -> Dict[str, torch.Tensor]:
        """
        Extract ERP component amplitudes (P300, N200, ERN, Pe, N2, P3b)
        averaged across trials.

        Parameters
        ----------
        raw          : MNE Raw object (already filtered / resampled)
        events_array : (n_events, 3) array; if None, tries mne.find_events()
        tmin / tmax  : epoch window (seconds)
        baseline     : baseline correction window (seconds)

        Returns
        -------
        Dict of component_name → mean amplitude tensor (n_channels,)
        """
        if not HAS_MNE:
            raise ImportError("MNE-Python required.")
        if events_array is None:
            try:
                events_array = mne.find_events(raw, verbose=False)
            except Exception:
                # Fall back to synthetic single-epoch if no stim channel
                n_samps = int((tmax - tmin) * self.sfreq)
                dummy_data = raw.get_data()[:, :n_samps]
                return self._erp_from_array(dummy_data, raw.info)

        epochs = mne.Epochs(raw, events_array, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, verbose=False)
        evoked = epochs.average()
        times  = evoked.times   # (n_times,)
        data   = evoked.data    # (n_channels, n_times)

        features: Dict[str, torch.Tensor] = {}
        for comp, (t0_ms, t1_ms) in ERP_WINDOWS_MS.items():
            t0 = t0_ms / 1000.0
            t1 = t1_ms / 1000.0
            mask = (times >= t0) & (times <= t1)
            if mask.sum() == 0:
                amp = np.zeros(data.shape[0])
            else:
                amp = data[:, mask].mean(axis=1)
            features[comp] = torch.tensor(amp, dtype=torch.float32,
                                          device=self.device)
        return features

    def _erp_from_array(self, data: np.ndarray,
                        info: "mne.Info") -> Dict[str, torch.Tensor]:
        """Fallback ERP extraction when no events are present."""
        n_chan, n_tp = data.shape
        sfreq = info['sfreq']
        features: Dict[str, torch.Tensor] = {}
        for comp, (t0_ms, t1_ms) in ERP_WINDOWS_MS.items():
            i0 = int(t0_ms / 1000.0 * sfreq)
            i1 = int(t1_ms / 1000.0 * sfreq)
            i0 = max(0, min(i0, n_tp - 1))
            i1 = max(i0 + 1, min(i1, n_tp))
            amp = data[:, i0:i1].mean(axis=1)
            features[comp] = torch.tensor(amp, dtype=torch.float32,
                                          device=self.device)
        return features

    def extract_band_power(self, raw: "mne.io.Raw") -> Dict[str, torch.Tensor]:
        """
        Compute average band power (delta/theta/alpha/beta/gamma) per channel.

        Returns
        -------
        Dict of band_name → power tensor (n_channels,)
        """
        if not HAS_MNE:
            raise ImportError("MNE-Python required.")
        data = raw.get_data()  # (n_channels, n_times)
        band_powers: Dict[str, torch.Tensor] = {}
        for band, (flo, fhi) in FREQ_BANDS.items():
            raw_copy = raw.copy().filter(flo, fhi, verbose=False)
            filtered = raw_copy.get_data()
            power    = (filtered ** 2).mean(axis=1)
            band_powers[band] = torch.tensor(power, dtype=torch.float32,
                                             device=self.device)
        return band_powers

    def compute_frontal_alpha_asymmetry(
            self, band_powers: Dict[str, torch.Tensor],
            raw: "mne.io.Raw") -> torch.Tensor:
        """
        Frontal Alpha Asymmetry (FAA) index: ln(F4_alpha) - ln(F3_alpha).
        Negative FAA → relative right frontal activity → approach-withdrawal
        imbalance (associated with aggression / impulsivity).

        Returns
        -------
        faa : scalar tensor
        """
        ch_names = [ch.upper() for ch in raw.ch_names]
        alpha    = band_powers.get("alpha",
                                   torch.zeros(len(ch_names), device=self.device))
        eps = 1e-10
        f3_idx = next((i for i, c in enumerate(ch_names) if c in ("F3", "EEG F3")), None)
        f4_idx = next((i for i, c in enumerate(ch_names) if c in ("F4", "EEG F4")), None)
        if f3_idx is None or f4_idx is None:
            logger.warning("F3/F4 channels not found; FAA set to 0.")
            return torch.tensor(0.0, device=self.device)
        faa = torch.log(alpha[f4_idx] + eps) - torch.log(alpha[f3_idx] + eps)
        return faa

    def compute_theta_beta_ratio(
            self, band_powers: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Theta-Beta Ratio (TBR): mean(theta) / mean(beta).
        Elevated TBR is associated with ADHD and impulse-control deficits.

        Returns
        -------
        tbr : scalar tensor
        """
        theta = band_powers.get("theta",
                                torch.zeros(1, device=self.device)).mean()
        beta  = band_powers.get("beta",
                                torch.ones(1, device=self.device)).mean()
        eps = 1e-10
        return theta / (beta + eps)

    # ── MEG ──────────────────────────────────────────────────────────────────
    def load_meg_raw(self, file_path: str) -> "mne.io.Raw":
        """Load raw MEG data (FIF format)."""
        if not HAS_MNE:
            raise ImportError("MNE-Python required for MEG loading.")
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        raw.pick_types(meg=True, eeg=False, verbose=False)
        return raw

    def extract_meg_beta_desync(self, raw: "mne.io.Raw",
                                event_onset_sample: Optional[int] = None
                                ) -> torch.Tensor:
        """
        Compute MEG beta-band event-related desynchronisation (ERD).
        Beta ERD indexes motor/cognitive preparation and is attenuated in SUD.

        Returns
        -------
        erd : tensor (n_channels,) — negative values = desynchronisation
        """
        if not HAS_MNE:
            raise ImportError("MNE-Python required.")
        raw_beta = raw.copy().filter(13.0, 30.0, verbose=False)
        data     = raw_beta.get_data()      # (n_ch, n_times)
        n_times  = data.shape[1]
        if event_onset_sample is None:
            event_onset_sample = n_times // 2
        pre_start  = max(0, event_onset_sample - int(0.5 * self.sfreq))
        pre_end    = event_onset_sample
        post_start = event_onset_sample
        post_end   = min(n_times, event_onset_sample + int(0.5 * self.sfreq))
        baseline_power = (data[:, pre_start:pre_end] ** 2).mean(axis=1) + 1e-10
        event_power    = (data[:, post_start:post_end] ** 2).mean(axis=1)
        erd = (event_power - baseline_power) / baseline_power
        return torch.tensor(erd, dtype=torch.float32, device=self.device)

    # ── MRI (structural) ─────────────────────────────────────────────────────
    def extract_mri_roi_volumes(self, t1_file: str,
                                radius_mm: float = 8.0
                                ) -> Dict[str, torch.Tensor]:
        """
        Extract mean grey-matter signal within spherical ROIs centred at
        MNI coordinates defined in MRI_ROI_MNI.

        Requires Nilearn + Nibabel.

        Parameters
        ----------
        t1_file  : path to T1-weighted NIfTI file (or GM probability map)
        radius_mm: sphere radius around each ROI centre

        Returns
        -------
        Dict of roi_name → mean signal scalar tensor
        """
        if not HAS_NILEARN:
            raise ImportError("Nilearn + Nibabel required for MRI ROI extraction.")
        from nilearn.input_data import NiftiSpheresMasker
        coords  = list(MRI_ROI_MNI.values())
        masker  = NiftiSpheresMasker(
            seeds=coords, radius=radius_mm, standardize=True,
            memory_level=0, verbose=0
        )
        signals = masker.fit_transform(t1_file)  # (1, n_rois) for a single image
        roi_signals = signals[0] if signals.ndim == 2 else signals
        result: Dict[str, torch.Tensor] = {}
        for i, roi_name in enumerate(MRI_ROI_MNI.keys()):
            result[roi_name] = torch.tensor(
                float(roi_signals[i]), dtype=torch.float32, device=self.device
            )
        return result

    def extract_fmri_reward_connectivity(
            self, fmri_file: str,
            seed_roi: str = "NAcc_L"
    ) -> Dict[str, torch.Tensor]:
        """
        Seed-based resting-state functional connectivity from the nucleus
        accumbens (or any ROI in MRI_ROI_MNI) to all other ROIs.

        Returns
        -------
        Dict of roi_name → Pearson r connectivity scalar tensor
        """
        if not HAS_NILEARN:
            raise ImportError("Nilearn + Nibabel required for fMRI analysis.")
        from nilearn.input_data import NiftiSpheresMasker
        coords   = list(MRI_ROI_MNI.values())
        roi_names = list(MRI_ROI_MNI.keys())
        masker   = NiftiSpheresMasker(
            seeds=coords, radius=8.0, standardize=True,
            memory_level=0, verbose=0
        )
        ts = masker.fit_transform(fmri_file)  # (n_vols, n_rois)
        seed_idx  = roi_names.index(seed_roi) if seed_roi in roi_names else 0
        seed_ts   = ts[:, seed_idx]
        connectivity: Dict[str, torch.Tensor] = {}
        for i, rname in enumerate(roi_names):
            if i == seed_idx:
                r_val = 1.0
            else:
                r     = np.corrcoef(seed_ts, ts[:, i])[0, 1]
                r_val = float(r)
            connectivity[rname] = torch.tensor(r_val, dtype=torch.float32,
                                               device=self.device)
        return connectivity

    # ── Unified feature vector ────────────────────────────────────────────────
    def build_feature_vector(
        self,
        erp_feats:    Optional[Dict[str, torch.Tensor]] = None,
        band_powers:  Optional[Dict[str, torch.Tensor]] = None,
        faa:          Optional[torch.Tensor] = None,
        tbr:          Optional[torch.Tensor] = None,
        meg_erd:      Optional[torch.Tensor] = None,
        mri_vols:     Optional[Dict[str, torch.Tensor]] = None,
        fmri_conn:    Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Concatenate all available features into a single flat vector.
        Missing modalities are filled with zeros of their expected size.
        """
        parts: List[torch.Tensor] = []

        # ERP (6 components × n_channels – take mean across channels → 6 scalars)
        erp_order = ["P300", "N200", "ERN", "Pe", "N2", "P3b"]
        for comp in erp_order:
            if erp_feats and comp in erp_feats:
                parts.append(erp_feats[comp].mean().unsqueeze(0))
            else:
                parts.append(torch.zeros(1, device=self.device))

        # Band powers (5 bands)
        band_order = ["delta", "theta", "alpha", "beta", "gamma"]
        for band in band_order:
            if band_powers and band in band_powers:
                parts.append(band_powers[band].mean().unsqueeze(0))
            else:
                parts.append(torch.zeros(1, device=self.device))

        # FAA scalar
        parts.append(faa if faa is not None
                     else torch.zeros(1, device=self.device))
        # TBR scalar
        parts.append(tbr if tbr is not None
                     else torch.zeros(1, device=self.device))

        # MEG ERD (mean across channels)
        if meg_erd is not None:
            parts.append(meg_erd.mean().unsqueeze(0))
        else:
            parts.append(torch.zeros(1, device=self.device))

        # MRI ROI volumes
        for rname in MRI_ROI_MNI.keys():
            if mri_vols and rname in mri_vols:
                parts.append(mri_vols[rname].unsqueeze(0))
            else:
                parts.append(torch.zeros(1, device=self.device))

        # fMRI connectivity
        for rname in MRI_ROI_MNI.keys():
            if fmri_conn and rname in fmri_conn:
                parts.append(fmri_conn[rname].unsqueeze(0))
            else:
                parts.append(torch.zeros(1, device=self.device))

        return torch.cat(parts, dim=0)  # (FEATURE_DIM,)


# Feature vector dimension (for SSC layer sizing):
# 6 ERP + 5 band + FAA + TBR + MEG_ERD + 11 ROI vols + 11 fMRI conn = 35
FEATURE_DIM = 6 + 5 + 1 + 1 + 1 + len(MRI_ROI_MNI) + len(MRI_ROI_MNI)  # = 35


# =============================================================================
# 3.  CLINICAL SCREENING QUESTIONNAIRES
# =============================================================================
@dataclass
class ClinicalScalesResult:
    """Holds raw scores and DSM-5/ICD-11 threshold flags."""
    scale_name:      str
    items_filled:    int
    total_score:     float
    subscale_scores: Dict[str, float] = field(default_factory=dict)
    threshold_met:   bool = False
    severity:        str  = "None"   # None / Mild / Moderate / Severe


class AuditScale:
    """
    AUDIT-10 — Alcohol Use Disorders Identification Test
    Saunders et al. (1993). WHO screening tool for hazardous alcohol use.
    Threshold: ≥8 (hazardous), ≥20 (dependence)
    """
    N_ITEMS = 10

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """
        Parameters
        ----------
        responses : list of 10 integers.
                    Q1-Q8 : 0,1,2,3,4
                    Q9-Q10: 0,2,4
        """
        assert len(responses) == self.N_ITEMS, "AUDIT requires 10 responses."
        total = int(sum(responses))
        if total < 8:
            sev = "None"
        elif total < 16:
            sev = "Hazardous"
        elif total < 20:
            sev = "Harmful"
        else:
            sev = "Dependence"
        return ClinicalScalesResult(
            scale_name="AUDIT-10", items_filled=self.N_ITEMS,
            total_score=total, threshold_met=(total >= 8), severity=sev
        )


class Dast10Scale:
    """
    DAST-10 — Drug Abuse Screening Test (Skinner, 1982).
    Threshold: ≥3 (moderate), ≥6 (substantial)
    """
    N_ITEMS = 10

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """responses: binary (0/1) list of 10 items."""
        assert len(responses) == self.N_ITEMS, "DAST-10 requires 10 responses."
        total = int(sum(responses))
        if total < 1:
            sev = "None"
        elif total < 3:
            sev = "Low"
        elif total < 6:
            sev = "Moderate"
        elif total < 9:
            sev = "Substantial"
        else:
            sev = "Severe"
        return ClinicalScalesResult(
            scale_name="DAST-10", items_filled=self.N_ITEMS,
            total_score=total, threshold_met=(total >= 3), severity=sev
        )


class Cage4Scale:
    """
    CAGE-4 — brief alcohol screening (Ewing, 1984).
    Threshold: ≥2 (clinically significant)
    """
    N_ITEMS = 4

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """responses: binary (0/1) list of 4 items."""
        assert len(responses) == self.N_ITEMS, "CAGE requires 4 responses."
        total = int(sum(responses))
        return ClinicalScalesResult(
            scale_name="CAGE-4", items_filled=self.N_ITEMS,
            total_score=total, threshold_met=(total >= 2),
            severity="Significant" if total >= 2 else "None"
        )


class Hbi19Scale:
    """
    HBI-19 — Hypersexual Behavior Inventory (Reid et al., 2011).
    19 items; 3 subscales: Control, Consequences, Coping.
    Total threshold: ≥53 (hypersexual disorder criteria).
    """
    N_ITEMS    = 19
    SUBSCALES  = {
        "Control":      list(range(0, 7)),     # Q1-Q7
        "Consequences": list(range(7, 13)),    # Q8-Q13
        "Coping":       list(range(13, 19)),   # Q14-Q19
    }

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """responses: list of 19 integers (1–5 Likert scale)."""
        assert len(responses) == self.N_ITEMS, "HBI-19 requires 19 responses."
        total = float(sum(responses))
        sub: Dict[str, float] = {
            name: float(sum(responses[i] for i in idxs))
            for name, idxs in self.SUBSCALES.items()
        }
        if total < 53:
            sev = "Subclinical"
        elif total < 70:
            sev = "Moderate"
        else:
            sev = "Severe"
        return ClinicalScalesResult(
            scale_name="HBI-19", items_filled=self.N_ITEMS,
            total_score=total, subscale_scores=sub,
            threshold_met=(total >= 53), severity=sev
        )


class Sastr19Scale:
    """
    SAST-R — Sexual Addiction Screening Test-Revised (Carnes et al., 2010).
    19 items (core + women's add-on). Threshold: ≥6 (clinically significant).
    """
    N_ITEMS = 19

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """responses: binary (0/1) list of 19 items."""
        assert len(responses) == self.N_ITEMS, "SAST-R requires 19 responses."
        total = int(sum(responses))
        return ClinicalScalesResult(
            scale_name="SAST-R", items_filled=self.N_ITEMS,
            total_score=total, threshold_met=(total >= 6),
            severity="Significant" if total >= 6 else "None"
        )


class Csbd19Scale:
    """
    CSBD-19 — Compulsive Sexual Behaviour Disorder (Bőthe et al., 2020).
    19 items; 4 subscales. ICD-11 6C72 screening instrument.
    Threshold: ≥30 (borderline), ≥45 (clinical)
    """
    N_ITEMS = 19
    SUBSCALES = {
        "Salience":     list(range(0, 5)),
        "GoalControl":  list(range(5, 10)),
        "Distress":     list(range(10, 14)),
        "Consequences": list(range(14, 19)),
    }

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """responses: list of 19 integers (1–5 Likert)."""
        assert len(responses) == self.N_ITEMS, "CSBD-19 requires 19 responses."
        total = float(sum(responses))
        sub   = {
            name: float(sum(responses[i] for i in idxs))
            for name, idxs in self.SUBSCALES.items()
        }
        if total < 30:
            sev = "Subclinical"
        elif total < 45:
            sev = "Borderline"
        else:
            sev = "Clinical (ICD-11 6C72)"
        return ClinicalScalesResult(
            scale_name="CSBD-19", items_filled=self.N_ITEMS,
            total_score=total, subscale_scores=sub,
            threshold_met=(total >= 45), severity=sev
        )


class Bpaq29Scale:
    """
    BPAQ-29 — Buss-Perry Aggression Questionnaire (Buss & Perry, 1992).
    29 items; 4 subscales: Physical, Verbal, Anger, Hostility.
    Total score range 29–145. Threshold >75 = elevated aggression.
    """
    N_ITEMS = 29
    SUBSCALES = {
        "Physical":  [0,4,8,12,16,20,24,28],        # 9 items (Q1,5,9,13,17,21,25,29)
        "Verbal":    [1,5,9,13,17],                   # 5 items
        "Anger":     [2,6,10,14,18,21,25],            # 7 items
        "Hostility": [3,7,11,15,19,22,23,26,27],     # 8 items
    }

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """responses: list of 29 integers (1–5 Likert)."""
        assert len(responses) == self.N_ITEMS, "BPAQ-29 requires 29 responses."
        total = float(sum(responses))
        sub: Dict[str, float] = {}
        for name, idxs in self.SUBSCALES.items():
            idxs_valid = [i for i in idxs if i < self.N_ITEMS]
            sub[name] = float(sum(responses[i] for i in idxs_valid))
        if total < 45:
            sev = "Low"
        elif total < 75:
            sev = "Average"
        elif total < 100:
            sev = "High"
        else:
            sev = "Very High"
        return ClinicalScalesResult(
            scale_name="BPAQ-29", items_filled=self.N_ITEMS,
            total_score=total, subscale_scores=sub,
            threshold_met=(total > 75), severity=sev
        )


class OasScale:
    """
    OAS — Overt Aggression Scale (Yudofsky et al., 1986).
    4 subscales: Verbal, Against objects, Against self, Against others.
    Used in inpatient and forensic settings. Each subscale scored 0–4
    (level of severity).
    """
    SUBSCALE_NAMES = ["Verbal", "AgainstObjects", "AgainstSelf", "AgainstOthers"]

    def score(self, responses: List[int]) -> ClinicalScalesResult:
        """
        responses: list of 4 integers (0–4 severity per subscale).
        Intervention Measure is optional (omitted here).
        """
        assert len(responses) == 4, "OAS requires 4 subscale scores."
        total = float(sum(responses))
        sub   = dict(zip(self.SUBSCALE_NAMES, [float(r) for r in responses]))
        if total == 0:
            sev = "None"
        elif total <= 4:
            sev = "Mild"
        elif total <= 8:
            sev = "Moderate"
        else:
            sev = "Severe"
        return ClinicalScalesResult(
            scale_name="OAS", items_filled=4,
            total_score=total, subscale_scores=sub,
            threshold_met=(total > 4), severity=sev
        )


# =============================================================================
# 4.  SUBSTANCE USE DISORDER ASSESSOR  (DSM-5 §302-305; ICD-11 §6C4x)
# =============================================================================
class SubstanceUseDisorderAssessor(nn.Module):
    """
    Neural network assessor for Substance Use Disorder (SUD) using
    EEG/MEG/MRI-derived neuroimaging features.

    Architecture
    ────────────
    Input  : feature vector from ImpulseAddictionDataLoader.build_feature_vector()
    Output : probability distribution over [No-SUD, Mild-SUD, Moderate-SUD,
             Severe-SUD] per DSM-5 §10/11 criterion count
             (2-3: mild; 4-5: moderate; ≥6: severe)

    Key biomarker contributions (from literature)
    ──────────────────────────────────────────────
    • ↓ P300 amplitude        (reward-processing blunting)
    • ↑ theta-beta ratio      (inhibitory deficit)
    • ↓ beta ERD (MEG)        (attenuated motor/cognitive prep)
    • ↓ dlPFC grey matter     (executive control impairment)
    • ↑ NAcc/striatum signal  (cue-reactivity, reward sensitisation)
    • ↓ OFC/vmPFC volume      (decision-making deficit)
    • ↓ ACC activation        (conflict monitoring deficit)
    """

    N_CLASSES = 4
    CLASS_NAMES = ["No-SUD", "Mild-SUD", "Moderate-SUD", "Severe-SUD"]

    def __init__(self, feature_dim: int = FEATURE_DIM,
                 hidden: int = 128,
                 device: torch.device = OPTIMAL_DEVICE):
        super().__init__()
        self.device = device
        self.fc1    = nn.Linear(feature_dim, hidden)
        self.ln1    = nn.LayerNorm(hidden)
        self.fc2    = nn.Linear(hidden, hidden // 2)
        self.ln2    = nn.LayerNorm(hidden // 2)
        self.fc3    = nn.Linear(hidden // 2, self.N_CLASSES)
        self.dropout = nn.Dropout(0.3)
        # Biomarker attention weights (learned)
        self.attn   = nn.Parameter(torch.ones(feature_dim))
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_dim) or (feature_dim,)

        Returns
        -------
        logits : (batch, 4) or (4,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x * torch.softmax(self.attn, dim=0)
        x = soft_clamp(F.silu(self.ln1(self.fc1(x))))
        x = self.dropout(x)
        x = soft_clamp(F.silu(self.ln2(self.fc2(x))))
        return self.fc3(x).squeeze(0)

    def predict(self, feature_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference and return structured report.

        Returns
        -------
        dict with 'class', 'probabilities', 'biomarker_attention'
        """
        self.eval()
        with torch.no_grad():
            logits = self(feature_vector)
            probs  = torch.softmax(logits, dim=-1)
            cls_idx = int(probs.argmax().item())
            attn   = torch.softmax(self.attn, dim=0).cpu().tolist()
        return {
            "disorder":     "Substance Use Disorder (DSM-5 §302-305)",
            "icd_code":     "ICD-11 6C4x",
            "class":        self.CLASS_NAMES[cls_idx],
            "probabilities": {
                name: round(float(probs[i].item()), 4)
                for i, name in enumerate(self.CLASS_NAMES)
            },
            "confidence":   round(float(probs[cls_idx].item()), 4),
            "biomarker_attention": attn,
        }

    def suggest_screening(self) -> Dict[str, str]:
        return {
            "AUDIT-10": "Primary alcohol screening",
            "DAST-10":  "Illicit drug screening",
            "CAGE-4":   "Brief alcohol dependence screen",
        }


# =============================================================================
# 5.  COMPULSIVE SEXUAL BEHAVIOUR DISORDER ASSESSOR  (ICD-11: 6C72)
# =============================================================================
class CompulsiveSexualBehaviourAssessor(nn.Module):
    """
    Neural network assessor for Compulsive Sexual Behaviour Disorder (CSBD),
    coded ICD-11 6C72, also covering the proposed DSM-5 hypersexual disorder.

    CSBD is characterised by failure to control intense, repetitive sexual
    impulses/urges, resulting in distress or functional impairment, and is
    now a recognised ICD-11 diagnosis (in Impulse Control Disorders chapter).

    Key neuroimaging biomarkers
    ───────────────────────────
    • ↓ N200/P300 in Go/NoGo tasks  (prefrontal inhibitory deficit)
    • ↑ theta-beta ratio             (attentional / impulse-control marker)
    • ↑ ventral striatum activation  (reward hypersensitivity)
    • ↑ amygdala reactivity          (heightened affective response)
    • ↓ OFC/vmPFC-amygdala connectivity (reduced top-down control)
    • FAA (frontal alpha asymmetry)  (approach motivation imbalance)
    """

    N_CLASSES    = 3
    CLASS_NAMES  = ["No-CSBD", "Borderline-CSBD", "CSBD (ICD-11 6C72)"]

    def __init__(self, feature_dim: int = FEATURE_DIM,
                 hidden: int = 128,
                 device: torch.device = OPTIMAL_DEVICE):
        super().__init__()
        self.device  = device
        self.fc1     = nn.Linear(feature_dim, hidden)
        self.ln1     = nn.LayerNorm(hidden)
        self.fc2     = nn.Linear(hidden, hidden // 2)
        self.ln2     = nn.LayerNorm(hidden // 2)
        self.fc3     = nn.Linear(hidden // 2, self.N_CLASSES)
        self.dropout = nn.Dropout(0.3)
        # Separate inhibition and reward attention heads
        self.inhibition_attn = nn.Parameter(torch.ones(feature_dim))
        self.reward_attn     = nn.Parameter(torch.ones(feature_dim))
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Blend inhibitory and reward attention
        w_inhibit = torch.softmax(self.inhibition_attn, dim=0)
        w_reward  = torch.softmax(self.reward_attn,     dim=0)
        x = x * (0.5 * w_inhibit + 0.5 * w_reward)
        x = soft_clamp(F.silu(self.ln1(self.fc1(x))))
        x = self.dropout(x)
        x = soft_clamp(F.silu(self.ln2(self.fc2(x))))
        return self.fc3(x).squeeze(0)

    def predict(self, feature_vector: torch.Tensor) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            logits  = self(feature_vector)
            probs   = torch.softmax(logits, dim=-1)
            cls_idx = int(probs.argmax().item())
        return {
            "disorder":      "Compulsive Sexual Behaviour Disorder",
            "icd_code":      "ICD-11 6C72",
            "dsm5_proposed": "Hypersexual Disorder (proposed, not adopted)",
            "class":         self.CLASS_NAMES[cls_idx],
            "probabilities": {
                name: round(float(probs[i].item()), 4)
                for i, name in enumerate(self.CLASS_NAMES)
            },
            "confidence":    round(float(probs[cls_idx].item()), 4),
            "note": (
                "CSBD requires persistent pattern of failure to control "
                "sexual impulses causing marked distress or impairment. "
                "Neuroimaging supports, but diagnosis requires clinical interview."
            ),
        }

    def suggest_screening(self) -> Dict[str, str]:
        return {
            "CSBD-19":  "Primary ICD-11 6C72 screening scale",
            "HBI-19":   "Hypersexual Behavior Inventory (3 subscales)",
            "SAST-R":   "Sexual Addiction Screening Test-Revised",
        }


# =============================================================================
# 6.  IMPULSIVE AGGRESSION ASSESSOR  (DSM-5 §312.34 IED; ICD-11: 6C73)
# =============================================================================
class ImpulsiveAggressionAssessor(nn.Module):
    """
    Neural network assessor for Intermittent Explosive Disorder (IED) and
    impulse-control deficits associated with harm to others.

    DSM-5 §312.34: recurrent behavioural outbursts representing a failure to
    control aggressive impulses, not attributable to another mental disorder.
    ICD-11 6C73: Intermittent explosive disorder.

    Key neuroimaging biomarkers
    ───────────────────────────
    • ↓ left frontal alpha (FAA)    (approach-withdrawal imbalance)
    • ↑ theta-beta ratio            (frontal inhibitory deficit)
    • ↓ N2/P3 in NoGo tasks         (impaired motor inhibition)
    • ↑ ERN amplitude               (heightened error reactivity)
    • ↑ amygdala reactivity (fMRI)  (emotional hyper-arousal)
    • ↓ OFC grey matter volume      (impulsive decision-making)
    • ↓ amygdala–vmPFC connectivity (reduced emotional regulation)
    • ↓ ACC activation              (reduced conflict monitoring)
    """

    N_CLASSES   = 4
    CLASS_NAMES = [
        "No-IED",
        "Sub-threshold IED",
        "IED (Mild-Moderate)",
        "IED (Severe)",
    ]

    def __init__(self, feature_dim: int = FEATURE_DIM,
                 hidden: int = 128,
                 device: torch.device = OPTIMAL_DEVICE):
        super().__init__()
        self.device      = device
        self.fc1         = nn.Linear(feature_dim, hidden)
        self.ln1         = nn.LayerNorm(hidden)
        self.fc2         = nn.Linear(hidden, hidden // 2)
        self.ln2         = nn.LayerNorm(hidden // 2)
        self.fc3         = nn.Linear(hidden // 2, self.N_CLASSES)
        self.dropout     = nn.Dropout(0.3)
        # Separate attention for amygdala/limbic vs frontal/inhibitory features
        self.frontal_attn = nn.Parameter(torch.ones(feature_dim))
        self.limbic_attn  = nn.Parameter(torch.ones(feature_dim))
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        w_f = torch.softmax(self.frontal_attn, dim=0)
        w_l = torch.softmax(self.limbic_attn,  dim=0)
        x = x * (0.6 * w_f + 0.4 * w_l)   # frontal control weighted more
        x = soft_clamp(F.silu(self.ln1(self.fc1(x))))
        x = self.dropout(x)
        x = soft_clamp(F.silu(self.ln2(self.fc2(x))))
        return self.fc3(x).squeeze(0)

    def predict(self, feature_vector: torch.Tensor) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            logits  = self(feature_vector)
            probs   = torch.softmax(logits, dim=-1)
            cls_idx = int(probs.argmax().item())
        return {
            "disorder":   "Intermittent Explosive Disorder",
            "dsm5_code":  "DSM-5 §312.34",
            "icd_code":   "ICD-11 6C73",
            "class":      self.CLASS_NAMES[cls_idx],
            "probabilities": {
                name: round(float(probs[i].item()), 4)
                for i, name in enumerate(self.CLASS_NAMES)
            },
            "confidence": round(float(probs[cls_idx].item()), 4),
            "note": (
                "Neuroimaging supports risk stratification. IED diagnosis "
                "requires recurrent outbursts (≥3/year verbal or ≥3 episodes "
                "of physical aggression in 12 months per DSM-5). Clinical "
                "interview and collateral history are mandatory."
            ),
        }

    def suggest_screening(self) -> Dict[str, str]:
        return {
            "BPAQ-29": "Buss-Perry Aggression Questionnaire (4 subscales)",
            "OAS":     "Overt Aggression Scale (inpatient / forensic)",
            "AQ-SF":   "Aggression Questionnaire – Short Form",
        }


# =============================================================================
# 7.  UNIFIED IMPULSE CONTROL RISK ENGINE
# =============================================================================
@dataclass
class ImpulseRiskReport:
    """Structured output from ImpulseControlRiskEngine."""
    patient_id:       str
    modalities_used:  List[str]
    feature_vector:   List[float]
    sud_result:       Optional[Dict[str, Any]]
    csbd_result:      Optional[Dict[str, Any]]
    ied_result:       Optional[Dict[str, Any]]
    scales:           Dict[str, ClinicalScalesResult] = field(default_factory=dict)
    composite_risk:   float = 0.0
    risk_level:       str   = "Unknown"
    clinical_note:    str   = ""
    treatment_plan:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        d = dataclasses.asdict(self)
        return d


class ImpulseControlRiskEngine:
    """
    Unified clinical-grade assessment engine for impulse-control disorders
    diagnosed via multi-modal neuroimaging (EEG / MEG / MRI / fMRI).

    Integrates
    ──────────
    • ImpulseAddictionDataLoader  — data ingestion & feature extraction
    • SubstanceUseDisorderAssessor   — SUD (DSM-5 / ICD-11 6C4x)
    • CompulsiveSexualBehaviourAssessor — CSBD (ICD-11 6C72)
    • ImpulsiveAggressionAssessor   — IED (DSM-5 §312.34 / ICD-11 6C73)
    • Clinical screening scales     — AUDIT, DAST, CAGE, HBI, SAST, CSBD-19,
                                      BPAQ, OAS
    • Risk stratification           — composite score + intervention mapping

    Usage
    ──────
    engine = ImpulseControlRiskEngine()

    # Full pipeline from files:
    report = engine.run(
        patient_id  = "sub-001",
        eeg_file    = "patient.edf",          # optional
        meg_file    = "patient.fif",          # optional
        mri_file    = "patient_T1.nii.gz",    # optional
        fmri_file   = "patient_rest.nii.gz",  # optional
        scale_responses = {
            "AUDIT-10": [0,1,0,2,0,1,0,0,0,0],
            "DAST-10":  [0,1,1,0,0,0,0,0,0,0],
        }
    )
    """

    def __init__(self, device: torch.device = OPTIMAL_DEVICE,
                 sfreq: float = 256.0):
        self.device  = device
        self.loader  = ImpulseAddictionDataLoader(device=device, sfreq=sfreq)
        self.sud_net  = SubstanceUseDisorderAssessor(device=device)
        self.csbd_net = CompulsiveSexualBehaviourAssessor(device=device)
        self.ied_net  = ImpulsiveAggressionAssessor(device=device)

        # Clinical scales
        self.audit  = AuditScale()
        self.dast   = Dast10Scale()
        self.cage   = Cage4Scale()
        self.hbi    = Hbi19Scale()
        self.sastr  = Sastr19Scale()
        self.csbd19 = Csbd19Scale()
        self.bpaq   = Bpaq29Scale()
        self.oas    = OasScale()

        # Intervention database (extends InterventionDesigner from base)
        self.intervention_db: Dict[str, Dict] = {
            "SUD": {
                "medication":    ["Naltrexone", "Buprenorphine", "Acamprosate",
                                  "Methadone", "Disulfiram"],
                "psychotherapy": ["Motivational Interviewing", "CBT for SUD",
                                  "Contingency Management",
                                  "12-Step Facilitation"],
                "neurotherapy":  ["Transcranial Magnetic Stimulation (rTMS/dlPFC)",
                                  "Neurofeedback (alpha/theta protocol)"],
                "lifestyle":     ["Substance-free social network",
                                  "Aerobic exercise 150 min/week",
                                  "Sleep hygiene", "Mindfulness-Based Relapse Prevention"],
            },
            "CSBD": {
                "medication":    ["Naltrexone (off-label)", "SSRI (fluoxetine)",
                                  "N-acetylcysteine (off-label)"],
                "psychotherapy": ["CBT for compulsive sexual behaviour",
                                  "Acceptance and Commitment Therapy (ACT)",
                                  "Schema Therapy", "EMDR (if trauma-related)"],
                "neurotherapy":  ["rTMS (OFC/vmPFC target)", "Neurofeedback"],
                "lifestyle":     ["Structured daily routine",
                                  "Internet/device usage limits",
                                  "Mindfulness of urges",
                                  "Physical exercise"],
            },
            "IED": {
                "medication":    ["Fluoxetine", "Divalproex sodium",
                                  "Lithium carbonate", "Carbamazepine",
                                  "Oxcarbazepine"],
                "psychotherapy": ["CBT for anger management",
                                  "DBT skills training",
                                  "Group-based anger management",
                                  "Relaxation training"],
                "neurotherapy":  ["rTMS (left DLPFC)", "EEG neurofeedback "
                                  "(FAA normalisation protocol)"],
                "lifestyle":     ["Trigger identification and avoidance",
                                  "Aerobic exercise",
                                  "Structured de-escalation plan",
                                  "Reduced alcohol/substance use"],
            },
        }

    # ── Internal feature extraction ───────────────────────────────────────────
    def _extract_features(
        self,
        eeg_file:   Optional[str] = None,
        meg_file:   Optional[str] = None,
        mri_file:   Optional[str] = None,
        fmri_file:  Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extract feature vector from available modalities."""
        modalities_used: List[str] = []
        erp_feats = band_powers = faa = tbr = meg_erd = mri_vols = fmri_conn = None

        # EEG
        if eeg_file and os.path.exists(eeg_file):
            try:
                raw         = self.loader.load_eeg_raw(eeg_file)
                erp_feats   = self.loader.extract_erp_features(raw)
                band_powers = self.loader.extract_band_power(raw)
                faa         = self.loader.compute_frontal_alpha_asymmetry(
                                  band_powers, raw)
                tbr         = self.loader.compute_theta_beta_ratio(band_powers)
                modalities_used.append("EEG")
                logger.info("EEG features extracted from %s", eeg_file)
            except Exception as e:
                logger.warning("EEG extraction failed: %s", e)

        # MEG
        if meg_file and os.path.exists(meg_file):
            try:
                raw_meg = self.loader.load_meg_raw(meg_file)
                meg_erd = self.loader.extract_meg_beta_desync(raw_meg)
                modalities_used.append("MEG")
                logger.info("MEG features extracted from %s", meg_file)
            except Exception as e:
                logger.warning("MEG extraction failed: %s", e)

        # MRI (structural)
        if mri_file and os.path.exists(mri_file):
            try:
                mri_vols = self.loader.extract_mri_roi_volumes(mri_file)
                modalities_used.append("MRI")
                logger.info("MRI ROI volumes extracted from %s", mri_file)
            except Exception as e:
                logger.warning("MRI ROI extraction failed: %s", e)

        # fMRI (functional)
        if fmri_file and os.path.exists(fmri_file):
            try:
                fmri_conn = self.loader.extract_fmri_reward_connectivity(fmri_file)
                modalities_used.append("fMRI")
                logger.info("fMRI connectivity extracted from %s", fmri_file)
            except Exception as e:
                logger.warning("fMRI connectivity failed: %s", e)

        feat_vec = self.loader.build_feature_vector(
            erp_feats=erp_feats, band_powers=band_powers,
            faa=faa, tbr=tbr, meg_erd=meg_erd,
            mri_vols=mri_vols, fmri_conn=fmri_conn,
        )
        return feat_vec, modalities_used

    # ── Composite risk score ──────────────────────────────────────────────────
    def _composite_risk(
        self,
        sud_result:  Optional[Dict],
        csbd_result: Optional[Dict],
        ied_result:  Optional[Dict],
        scale_results: Dict[str, ClinicalScalesResult],
    ) -> Tuple[float, str]:
        """
        Weighted composite risk (0–1) combining model confidences and
        questionnaire thresholds.
        """
        risk = 0.0
        n    = 0

        def _class_risk(result: Dict, n_classes: int) -> float:
            """Normalise class index to 0-1."""
            if not result:
                return 0.0
            probs = list(result.get("probabilities", {}).values())
            if not probs:
                return 0.0
            idx   = list(result["probabilities"].keys()).index(result["class"])
            return idx / max(n_classes - 1, 1)

        if sud_result:
            risk += _class_risk(sud_result, 4); n += 1
        if csbd_result:
            risk += _class_risk(csbd_result, 3); n += 1
        if ied_result:
            risk += _class_risk(ied_result, 4); n += 1

        # Add questionnaire contribution (each positive threshold adds 0.15)
        for sr in scale_results.values():
            if sr.threshold_met:
                risk += 0.15; n += 1

        composite = min(1.0, risk / max(n, 1))
        if composite < 0.25:
            level = "Low"
        elif composite < 0.50:
            level = "Moderate"
        elif composite < 0.75:
            level = "High"
        else:
            level = "Very High"
        return round(composite, 4), level

    def _build_treatment_plan(
        self,
        sud_result:  Optional[Dict],
        csbd_result: Optional[Dict],
        ied_result:  Optional[Dict],
    ) -> Dict[str, Any]:
        """Aggregate treatment recommendations for active diagnoses."""
        plan: Dict[str, Any] = {}
        if sud_result and "No-SUD" not in sud_result.get("class", "No-SUD"):
            plan["SUD"] = self.intervention_db["SUD"]
        if csbd_result and "No-CSBD" not in csbd_result.get("class", "No-CSBD"):
            plan["CSBD"] = self.intervention_db["CSBD"]
        if ied_result and "No-IED" not in ied_result.get("class", "No-IED"):
            plan["IED"] = self.intervention_db["IED"]
        if not plan:
            plan["general"] = {
                "note": "No clinically significant impulse-control disorder detected.",
                "lifestyle": ["Regular aerobic exercise",
                              "Sleep hygiene",
                              "Mindfulness practice"],
            }
        return plan

    # ── Scale scoring router ──────────────────────────────────────────────────
    def _score_scales(
        self, scale_responses: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, ClinicalScalesResult]:
        if not scale_responses:
            return {}
        results: Dict[str, ClinicalScalesResult] = {}
        scale_map = {
            "AUDIT-10": self.audit.score,
            "DAST-10":  self.dast.score,
            "CAGE-4":   self.cage.score,
            "HBI-19":   self.hbi.score,
            "SAST-R":   self.sastr.score,
            "CSBD-19":  self.csbd19.score,
            "BPAQ-29":  self.bpaq.score,
            "OAS":      self.oas.score,
        }
        for scale_name, responses in scale_responses.items():
            if scale_name in scale_map:
                try:
                    results[scale_name] = scale_map[scale_name](responses)
                    logger.info("Scale %s scored: total=%s threshold=%s",
                                scale_name,
                                results[scale_name].total_score,
                                results[scale_name].threshold_met)
                except Exception as e:
                    logger.warning("Scale %s scoring failed: %s", scale_name, e)
            else:
                logger.warning("Unknown scale: %s (skipped)", scale_name)
        return results

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(
        self,
        patient_id:      str  = "unknown",
        eeg_file:        Optional[str] = None,
        meg_file:        Optional[str] = None,
        mri_file:        Optional[str] = None,
        fmri_file:       Optional[str] = None,
        scale_responses: Optional[Dict[str, List[int]]] = None,
        run_sud:         bool = True,
        run_csbd:        bool = True,
        run_ied:         bool = True,
    ) -> ImpulseRiskReport:
        """
        Full assessment pipeline.

        Parameters
        ----------
        patient_id       : identifier string
        eeg_file         : path to EEG file (EDF / FIF / CSV)
        meg_file         : path to MEG FIF file
        mri_file         : path to T1 NIfTI file (structural MRI / GM map)
        fmri_file        : path to resting-state fMRI NIfTI file
        scale_responses  : dict of scale_name → list of item responses
                           e.g. {"AUDIT-10": [0,1,0,2,...]}
        run_sud          : whether to run the SUD assessor
        run_csbd         : whether to run the CSBD assessor
        run_ied          : whether to run the IED assessor

        Returns
        -------
        ImpulseRiskReport
        """
        logger.info("ImpulseControlRiskEngine: starting assessment for %s", patient_id)

        # 1. Extract neuroimaging features
        feat_vec, modalities_used = self._extract_features(
            eeg_file=eeg_file, meg_file=meg_file,
            mri_file=mri_file, fmri_file=fmri_file,
        )
        fv = feat_vec.to(self.device)

        # 2. Run disorder-specific assessors
        sud_result  = self.sud_net.predict(fv)  if run_sud  else None
        csbd_result = self.csbd_net.predict(fv) if run_csbd else None
        ied_result  = self.ied_net.predict(fv)  if run_ied  else None

        # 3. Score clinical scales
        scale_results = self._score_scales(scale_responses)

        # 4. Composite risk
        composite, risk_level = self._composite_risk(
            sud_result, csbd_result, ied_result, scale_results
        )

        # 5. Treatment plan
        treatment_plan = self._build_treatment_plan(
            sud_result, csbd_result, ied_result
        )

        # 6. Clinical note
        note_parts = [
            "Multi-modal neuroimaging assessment for impulse-control disorders.",
            f"Modalities used: {', '.join(modalities_used) if modalities_used else 'none (using zero-filled features)'}.",
            "Results are supportive data for clinical evaluation.",
            "DSM-5/ICD-11 diagnosis requires licensed clinician interview.",
        ]
        if not modalities_used:
            note_parts.append(
                "WARNING: No neuroimaging files provided. All model outputs are "
                "based on zero-filled feature vectors and are not clinically "
                "meaningful. Provide EEG / MEG / MRI / fMRI files for valid results."
            )

        report = ImpulseRiskReport(
            patient_id       = patient_id,
            modalities_used  = modalities_used,
            feature_vector   = feat_vec.cpu().tolist(),
            sud_result       = sud_result,
            csbd_result      = csbd_result,
            ied_result       = ied_result,
            scales           = scale_results,
            composite_risk   = composite,
            risk_level       = risk_level,
            clinical_note    = " ".join(note_parts),
            treatment_plan   = treatment_plan,
        )
        logger.info("Assessment complete: composite_risk=%.3f risk_level=%s",
                    composite, risk_level)
        return report

    def run_from_feature_vector(
        self,
        feat_vec:        torch.Tensor,
        patient_id:      str = "unknown",
        scale_responses: Optional[Dict[str, List[int]]] = None,
        run_sud:  bool = True,
        run_csbd: bool = True,
        run_ied:  bool = True,
    ) -> ImpulseRiskReport:
        """
        Run assessment directly from a pre-computed feature vector
        (e.g. already extracted by ImpulseAddictionDataLoader).
        """
        fv = feat_vec.to(self.device)
        sud_result   = self.sud_net.predict(fv)  if run_sud  else None
        csbd_result  = self.csbd_net.predict(fv) if run_csbd else None
        ied_result   = self.ied_net.predict(fv)  if run_ied  else None
        scale_results = self._score_scales(scale_responses)
        composite, risk_level = self._composite_risk(
            sud_result, csbd_result, ied_result, scale_results
        )
        treatment_plan = self._build_treatment_plan(
            sud_result, csbd_result, ied_result
        )
        return ImpulseRiskReport(
            patient_id      = patient_id,
            modalities_used = ["precomputed"],
            feature_vector  = fv.cpu().tolist(),
            sud_result      = sud_result,
            csbd_result     = csbd_result,
            ied_result      = ied_result,
            scales          = scale_results,
            composite_risk  = composite,
            risk_level      = risk_level,
            clinical_note   = "Assessment from pre-computed feature vector.",
            treatment_plan  = treatment_plan,
        )


# =============================================================================
# 8.  MENTAL ONE INTEGRATION PATCH
#     Registers IMPULSE_DISORDERS in the base ALL_PSYCHIATRIC_DISORDERS list
#     and provides a convenience factory function.
# =============================================================================
def patch_mental_one_disorders() -> None:
    """
    Call this after importing mental_one to register the impulse-control
    disorders in the shared ALL_PSYCHIATRIC_DISORDERS list.
    Example:
        import mental_one as m1
        import mental_one_icd_impulse_v1 as imp
        imp.patch_mental_one_disorders()
        # m1.ALL_PSYCHIATRIC_DISORDERS now includes SUD, CSBD, IED
    """
    if not HAS_MENTAL_ONE_BASE:
        logger.warning("mental_one base not available; patch skipped.")
        return
    import mental_one as _m1
    for d in IMPULSE_DISORDERS:
        if d not in _m1.ALL_PSYCHIATRIC_DISORDERS:
            _m1.ALL_PSYCHIATRIC_DISORDERS.append(d)
    logger.info("Patched mental_one: disorders now = %s",
                _m1.ALL_PSYCHIATRIC_DISORDERS)


def build_engine(device: Optional[torch.device] = None) -> ImpulseControlRiskEngine:
    """
    Convenience factory: create and return ImpulseControlRiskEngine.
    """
    dev = device or OPTIMAL_DEVICE
    return ImpulseControlRiskEngine(device=dev)


# =============================================================================
# 9.  CLI  (standalone usage)
# =============================================================================
def _main() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(
        description="MENTAL ONE – Impulse Control & Addiction Module v1.0"
    )
    parser.add_argument("--patient",  default="unknown",
                        help="Patient / subject ID")
    parser.add_argument("--eeg",   default=None, help="EEG file (EDF/FIF/CSV)")
    parser.add_argument("--meg",   default=None, help="MEG FIF file")
    parser.add_argument("--mri",   default=None, help="T1 NIfTI file")
    parser.add_argument("--fmri",  default=None, help="Resting-state fMRI NIfTI")
    parser.add_argument("--scales", default=None,
                        help="JSON file with scale responses: "
                             '{\"AUDIT-10\":[0,1,...]}')
    parser.add_argument("--output", default="impulse_report.json",
                        help="Output JSON report path")
    parser.add_argument("--no_sud",  action="store_true",
                        help="Skip SUD assessment")
    parser.add_argument("--no_csbd", action="store_true",
                        help="Skip CSBD assessment")
    parser.add_argument("--no_ied",  action="store_true",
                        help="Skip IED assessment")
    args = parser.parse_args()

    scale_responses = None
    if args.scales and os.path.exists(args.scales):
        with open(args.scales) as f:
            scale_responses = json.load(f)

    engine = build_engine()
    report = engine.run(
        patient_id      = args.patient,
        eeg_file        = args.eeg,
        meg_file        = args.meg,
        mri_file        = args.mri,
        fmri_file       = args.fmri,
        scale_responses = scale_responses,
        run_sud         = not args.no_sud,
        run_csbd        = not args.no_csbd,
        run_ied         = not args.no_ied,
    )

    report_dict = report.to_dict()
    # Convert ClinicalScalesResult dataclasses
    report_dict["scales"] = {
        k: {
            "scale_name":      v.scale_name,
            "total_score":     v.total_score,
            "threshold_met":   v.threshold_met,
            "severity":        v.severity,
            "subscale_scores": v.subscale_scores,
        }
        for k, v in report.scales.items()
    }

    with open(args.output, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    print(f"[MENTAL ONE] Impulse/Addiction report saved → {args.output}")
    print(f"  Patient        : {report.patient_id}")
    print(f"  Modalities     : {', '.join(report.modalities_used) or 'none'}")
    print(f"  Composite risk : {report.composite_risk}  ({report.risk_level})")
    if report.sud_result:
        print(f"  SUD            : {report.sud_result['class']}")
    if report.csbd_result:
        print(f"  CSBD           : {report.csbd_result['class']}")
    if report.ied_result:
        print(f"  IED            : {report.ied_result['class']}")


if __name__ == "__main__":
    _main()
