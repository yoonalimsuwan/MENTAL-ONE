#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_neurochem_inference.py — v1.0.0
Multimodal neurochemical inference with explicit identifiability limits.

MRS quantification (the actual measurement) · PET target registry
EEG/MEG aperiodic E/I proxy · paired-data regression with CRLB gates
Transformer / world model / diffusion as validated research surrogates.

REFUSED: neurotransmitter concentration from EEG/MEG alone;
         "detailed molecular prediction" without a measured reference.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
THE INFORMATION-THEORETIC LIMIT
=============================================================================
Let C be a neurochemical state (concentrations, receptor occupancies,
release rates) and Y a scalp EEG/MEG measurement. The forward map
    Phi : C -> Y
is many-to-one and lossy. Most chemical directions lie in ker(D Phi):
different neurochemical configurations produce IDENTICAL scalp signals.

By the data-processing inequality, for a Markov chain C -> Y -> estimate,
    I(C ; estimate) <= I(C ; Y).
No algorithm, transformer, world model, or diffusion model can create
information about C that is absent from Y.

Therefore:
  * "GABA = 1.2 mM from this EEG" is NOT computable.
  * "dopamine release rate from this MEG" is NOT computable.
  * What IS computable: a validated PROXY, or a regression trained against
    an actual chemical measurement (MRS/PET) with reported uncertainty.

=============================================================================
CAPABILITY TIERS  (enforced)
=============================================================================
scalp_eeg / meg : aperiodic exponent (E/I proxy), band power, gamma power
                  (PV-interneuron / GABA-A correlate), MMN (NMDA correlate).
                  CANNOT report a concentration.
mrs             : ACTUAL metabolite quantification. GABA (edited), Glu, Gln,
                  NAA, Cre, Cho, mI, Lac, GSH, Tau. Reports CRLB.
                  CRLB > 20% -> unreliable by convention.
pet             : ACTUAL receptor/target binding. Requires a radiotracer.
fmri            : BOLD hemodynamics. NOT a chemical measurement.
microdialysis   : ACTUAL extracellular concentration. Invasive, research.

=============================================================================
THE ONE GENUINELY VALIDATED EEG->CHEMISTRY LINK
=============================================================================
Gao, Peterson & Voytek (2017), NeuroImage 158:70-78, "Inferring synaptic
excitation/inhibition balance from field potentials":
the aperiodic PSD exponent chi in P(f) ~ 1/f^chi tracks the E/I ratio in a
computational model of AMPA (tau ~ 2 ms) and GABA-A (tau ~ 10 ms) currents.
Steeper slope -> relatively more inhibition.

This is a PROXY with a mechanistic basis, validated in some MRS studies.
It is NOT a concentration measurement and it is not specific to a molecule.

=============================================================================
ENFORCED REFUSALS
=============================================================================
[R1] Concentration of a named neurotransmitter from EEG/MEG alone -> refused
[R2] "Detailed molecular prediction" without a measured reference -> refused
[R3] MRS quantification with CRLB > threshold reported as reliable -> refused
[R4] Regression prediction without paired training + cross-validation -> Mode E
[R5] fMRI BOLD reported as a chemical concentration -> refused
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("neurochem")
VERSION = "1.0.0"; PI = math.pi
GAMMA_HZ_PER_T = 42.577e6      # 1H gyromagnetic ratio
RESEARCH_ONLY = "research_only_not_a_diagnostic"

BANNER = f"""
=============================================================================
MSPS NEUROCHEMICAL INFERENCE v{VERSION}
MRS quantification (real measurement) · PET target registry
EEG/MEG aperiodic E/I proxy · paired regression with CRLB gates
REFUSED: neurotransmitter concentration from EEG/MEG alone
=============================================================================
"""

class Refusal(RuntimeError): pass
class NotIdentifiable(Refusal): pass
class NeedsReference(Refusal): pass
class UnreliableQuantification(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")


# =============================================================================
# §1. CAPABILITY TIERS
# =============================================================================

class Modality:
    EEG = "scalp_eeg"; MEG = "meg"; MRS = "mrs"
    PET = "pet"; FMRI = "fmri"; MICRODIALYSIS = "microdialysis"

@dataclass(frozen=True)
class Capability:
    modality: str
    measures_concentration: bool
    can: Tuple[str, ...]
    cannot: Tuple[str, ...]
    invasive: bool
    requires: Tuple[str, ...]
    note: str

CAPABILITIES: Dict[str, Capability] = {
 Modality.EEG: Capability(Modality.EEG, False,
   ("aperiodic_exponent_EI_proxy","band_power","gamma_power_GABA_correlate",
    "MMN_NMDA_correlate","P50_suppression_correlate"),
   ("metabolite_concentration","receptor_occupancy",
    "dopamine_concentration","serotonin_concentration","GABA_mM"),
   False, ("electrodes","impedance_check","emg_eog_control"),
   "cm-scale, volume-conducted. Chemical state is largely in ker(D Phi)."),
 Modality.MEG: Capability(Modality.MEG, False,
   ("aperiodic_exponent_EI_proxy","source_localised_band_power",
    "gamma_power_GABA_correlate"),
   ("metabolite_concentration","receptor_occupancy","GABA_mM"),
   False, ("shielded_room","squid_or_opm","emg_eog_control"),
   "Better source localisation than EEG; still no chemical concentration."),
 Modality.MRS: Capability(Modality.MRS, True,
   ("GABA_concentration_edited","Glu_concentration","Gln","Glx",
    "NAA","Cre","Cho","mI","Lac","GSH","Tau"),
   ("receptor_occupancy","synaptic_release_rate","dopamine_D2_binding"),
   False, ("MR_scanner","localised_sequence","basis_set",
           "water_reference_or_internal_reference","CRLB_report"),
   "THE actual non-invasive chemical measurement. Edited sequences "
   "(MEGA-PRESS) needed for GABA and GSH. CRLB > 20% is unreliable."),
 Modality.PET: Capability(Modality.PET, True,
   ("D2_D3_binding_potential","SERT_binding","dopamine_synthesis_capacity",
    "SV2A_synaptic_density","TSPO_neuroinflammation"),
   ("real_time_release_dynamics","EEG_derived_concentration"),
   False, ("radiotracer","cyclotron_or_generator","kinetic_model",
           "arterial_or_reference_input","radiation_consent"),
   "Measures a specific molecular TARGET. Requires a radiotracer."),
 Modality.FMRI: Capability(Modality.FMRI, False,
   ("BOLD_hemodynamics","functional_connectivity"),
   ("metabolite_concentration","neurotransmitter_level"),
   False, ("MR_scanner","hemodynamic_model"),
   "BOLD is hemodynamic, NOT chemical."),
 Modality.MICRODIALYSIS: Capability(Modality.MICRODIALYSIS, True,
   ("extracellular_concentration_timeseries"),
   ("non_invasive_measurement"),
   True, ("neurosurgery","clinical_indication","irb","consent"),
   "Invasive research. Actual extracellular concentration."),
}

def check_target(modality: str, target: str) -> Dict[str, Any]:
    if modality not in CAPABILITIES:
        raise Refusal(f"unknown modality '{modality}'")
    c = CAPABILITIES[modality]
    if target in c.cannot:
        raise NotIdentifiable(
            f"REFUSED: '{target}' is NOT obtainable from {modality}.\n"
            f"Cannot: {list(c.cannot)}\n"
            f"Reason: {c.note}\n"
            "By the data-processing inequality no algorithm can recover "
            "information absent from the measurement. Use MRS for "
            "concentration or PET for receptor binding.")
    if target not in c.can:
        raise NotIdentifiable(f"'{target}' not a validated target for {modality}")
    return {"allowed": True, "modality": modality, "target": target,
            "measures_concentration": c.measures_concentration}


# =============================================================================
# §2. IDENTIFIABILITY — what is in the null space
# =============================================================================

def identifiability_analysis(J: "np.ndarray", n_chemical: int,
                             labels: Optional[Sequence[str]] = None
                             ) -> Dict[str, Any]:
    """
    J : Jacobian d(measurement)/d(chemical state), shape [n_meas, n_chemical].
    Directions in the null space of J are chemically INVISIBLE.
    """
    _np()
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    tol = (s[0] if len(s) else 0.0)*max(J.shape)*np.finfo(float).eps
    rank = int(np.sum(s > max(tol, 1e-12)))
    null_dim = n_chemical - rank
    V = Vt.T
    null_basis = V[:, rank:] if null_dim > 0 else np.zeros((n_chemical, 0))
    ident = np.zeros(n_chemical)
    for i in range(n_chemical):
        # fraction of coordinate i captured by the row space
        ident[i] = float(np.sum(V[i, :rank]**2))
    names = list(labels) if labels else [f"c{i}" for i in range(n_chemical)]
    return {"n_measurements": J.shape[0], "n_chemical": n_chemical,
            "rank": rank, "null_dimension": null_dim,
            "singular_values": s.tolist(),
            "condition_number": float(s[0]/s[rank-1]) if rank else float("inf"),
            "per_variable_identifiability": dict(zip(names, ident.tolist())),
            "poorly_identified": [names[i] for i in range(n_chemical)
                                  if ident[i] < 0.5],
            "null_space_basis": null_basis.tolist(),
            "VERDICT": ("chemical state is NOT identifiable from this "
                        "measurement; only the row-space projection is"
                        if null_dim > 0 else "locally identifiable"),
            "principle": "I(C; estimate) <= I(C; Y)"}


def build_eeg_chemical_jacobian(n_eeg_features: int = 12,
                                n_chemical: int = 10) -> "np.ndarray":
    """
    Illustrative Jacobian: EEG features depend strongly on a few aggregate
    directions (E/I balance, arousal) and weakly on individual molecules.
    This is a MODEL for demonstrating identifiability, not measured data.
    """
    _np()
    rng = np.random.default_rng(0)
    J = np.zeros((n_eeg_features, n_chemical))
    # aggregate E/I direction is visible
    ei = np.zeros(n_chemical); ei[0] = 1.0; ei[1] = -0.8
    J[0] = ei
    # arousal direction weakly visible
    ar = rng.normal(0, 0.15, n_chemical)
    J[1] = ar
    # remaining features: very weak, near-null sensitivity
    J[2:] = rng.normal(0, 0.01, (n_eeg_features-2, n_chemical))
    return J


# =============================================================================
# §3. MRS — THE ACTUAL MEASUREMENT
# =============================================================================

@dataclass(frozen=True)
class Metabolite:
    name: str
    shifts_ppm: Tuple[float, ...]
    amplitudes: Tuple[float, ...]
    linewidth_hz: float
    edited: bool = False
    note: str = ""

# Chemical shifts at 1H; approximate singlet/ multiplet centres.
METABOLITES: Dict[str, Metabolite] = {
 "NAA": Metabolite("NAA", (2.008, 2.51, 2.67), (1.0, 0.12, 0.10), 6.0,
                   note="neuronal integrity marker; largest singlet"),
 "Cre": Metabolite("Cre", (3.027, 3.913), (1.0, 0.5), 6.0,
                   note="creatine + phosphocreatine; common internal reference"),
 "Cho": Metabolite("Cho", (3.19, 3.50, 4.05), (1.0, 0.15, 0.10), 6.0,
                   note="membrane turnover"),
 "mI":  Metabolite("mI", (3.56, 3.64, 4.06), (1.0, 0.8, 0.4), 8.0,
                   note="glial marker"),
 "Glu": Metabolite("Glu", (2.35, 2.22, 2.13, 3.75), (1.0, 0.7, 0.6, 0.5), 9.0,
                   note="J-coupled; overlaps Gln"),
 "Gln": Metabolite("Gln", (2.45, 2.12, 3.75), (1.0, 0.6, 0.5), 9.0,
                   note="overlaps Glu; Glx = Glu + Gln"),
 "GABA": Metabolite("GABA", (1.89, 2.28, 3.01), (1.0, 1.0, 1.0), 8.0,
                    edited=True,
                    note="J-coupled; REQUIRES edited acquisition (MEGA-PRESS)"),
 "GSH": Metabolite("GSH", (2.95, 3.77), (1.0, 0.4), 8.0, edited=True,
                   note="glutathione; requires editing"),
 "Lac": Metabolite("Lac", (1.33, 4.10), (1.0, 0.15), 6.0,
                   note="doublet J=7 Hz; anaerobic metabolism"),
 "Tau": Metabolite("Tau", (3.25, 3.42), (1.0, 1.0), 8.0, note="taurine"),
}

WATER_PPM = 4.65

def ppm_to_hz(ppm: float, field_T: float) -> float:
    """Frequency offset from water, in Hz."""
    return (WATER_PPM - ppm)*GAMMA_HZ_PER_T*field_T/1e6

def lorentzian(f: "np.ndarray", f0: float, lw: float) -> "np.ndarray":
    """Absorption-mode Lorentzian, unit peak."""
    hwhm = lw/2.0
    return hwhm**2/((f-f0)**2 + hwhm**2)

def build_basis_set(field_T: float, sw_hz: float, n_pts: int,
                    names: Optional[Sequence[str]] = None
                    ) -> Tuple["np.ndarray", List[str], "np.ndarray"]:
    """Simulated basis spectra for linear-combination fitting."""
    _np()
    f = np.linspace(-sw_hz/2, sw_hz/2, n_pts)
    use = list(names) if names else list(METABOLITES)
    B = []
    for nm in use:
        m = METABOLITES[nm]
        s = np.zeros(n_pts)
        for ppm, amp in zip(m.shifts_ppm, m.amplitudes):
            s += amp*lorentzian(f, ppm_to_hz(ppm, field_T), m.linewidth_hz)
        B.append(s/max(s.max(), 1e-12))
    return np.array(B).T, use, f

def synthesize_mrs(conc: Dict[str, float], field_T: float = 3.0,
                   sw_hz: float = 2000.0, n_pts: int = 1024,
                   noise_sd: float = 0.01, baseline_terms: int = 3,
                   seed: int = 0) -> Dict[str, Any]:
    """Generate a synthetic spectrum with known ground truth (for testing)."""
    _np()
    rng = np.random.default_rng(seed)
    B, names, f = build_basis_set(field_T, sw_hz, n_pts)
    c = np.array([conc.get(nm, 0.0) for nm in names])
    y = B @ c
    # smooth macromolecule / baseline contribution
    X = np.column_stack([np.linspace(-1, 1, n_pts)**k
                         for k in range(baseline_terms)])
    bl_coef = rng.normal(0, 0.05, baseline_terms)
    y = y + X @ bl_coef + rng.normal(0, noise_sd, n_pts)
    return {"spectrum": y, "freq_hz": f, "basis": B, "names": names,
            "true_concentrations": dict(zip(names, c.tolist())),
            "baseline_design": X, "noise_sd": noise_sd,
            "field_T": field_T}

def fit_mrs(y: "np.ndarray", B: "np.ndarray", names: Sequence[str],
            baseline_design: Optional["np.ndarray"] = None,
            noise_sd: float = 0.01, lam: float = 1e-4,
            crlb_max_pct: float = 20.0,
            edited_acquired: bool = False) -> Dict[str, Any]:
    """
    Linear-combination MRS quantification with non-negative amplitudes,
    polynomial baseline nuisance regression, and Cramer-Rao lower bounds.

    CRLB > 20% is unreliable by convention (LCModel / expert consensus).
    Edited metabolites (GABA, GSH) require an edited acquisition.
    """
    _np()
    n = len(y); K = B.shape[1]
    if baseline_design is None:
        baseline_design = np.column_stack(
            [np.linspace(-1, 1, n)**k for k in range(3)])
    P = baseline_design
    # project out baseline nuisance
    Q, _ = np.linalg.qr(P)
    Bp = B - Q @ (Q.T @ B)
    yp = y - Q @ (Q.T @ y)
    # Tikhonov-regularised normal equations
    A = Bp.T @ Bp + lam*np.eye(K)
    coef = np.linalg.solve(A, Bp.T @ yp)
    coef = np.maximum(coef, 0.0)          # non-negativity
    resid = yp - Bp @ coef
    dof = max(n - K - P.shape[1], 1)
    s2 = float(resid @ resid)/dof
    # Fisher information -> CRLB
    F = (Bp.T @ Bp)/max(noise_sd**2, 1e-18)
    C = np.linalg.pinv(F)
    crlb_pct = []
    for i in range(K):
        sd = math.sqrt(max(C[i, i], 0.0))
        crlb_pct.append(100.0*sd/max(coef[i], 1e-12) if coef[i] > 1e-9
                        else float("inf"))
    out: Dict[str, Any] = {"names": list(names),
        "concentrations": dict(zip(names, coef.tolist())),
        "crlb_percent": dict(zip(names, crlb_pct)),
        "residual_sd": math.sqrt(s2), "n_points": n,
        "method": "linear combination + baseline projection + Tikhonov",
        "units": "arbitrary (ratio to internal reference unless calibrated)"}
    unreliable, edited_missing = [], []
    for nm, cr in zip(names, crlb_pct):
        if cr > crlb_max_pct: unreliable.append(nm)
        if METABOLITES[nm].edited and not edited_acquired:
            edited_missing.append(nm)
    out["unreliable_CRLB"] = unreliable
    out["crlb_threshold_pct"] = crlb_max_pct
    out["edited_metabolites_without_editing"] = edited_missing
    out["reportable"] = [nm for nm in names
                         if nm not in unreliable and nm not in edited_missing]
    if edited_missing:
        out["REFUSED"] = (f"{edited_missing} require an EDITED acquisition "
                          "(e.g. MEGA-PRESS). Unedited quantification of "
                          "these metabolites is not reportable.")
    out["NOT_A_CONCENTRATION_IN_MM"] = (
        "values are ratios/arbitrary units unless a water or internal "
        "reference with known concentration and relaxation correction is used")
    return out

def require_reportable(fit: Dict[str, Any], metabolite: str) -> Dict[str, Any]:
    if metabolite in fit.get("edited_metabolites_without_editing", []):
        raise UnreliableQuantification(
            f"{metabolite} requires an edited acquisition (MEGA-PRESS). "
            "Unedited fitting is not reportable.")
    if metabolite in fit.get("unreliable_CRLB", []):
        raise UnreliableQuantification(
            f"{metabolite} CRLB = {fit['crlb_percent'][metabolite]:.1f}% "
            f"> {fit['crlb_threshold_pct']}%. Not reportable.")
    return {"metabolite": metabolite,
            "value": fit["concentrations"][metabolite],
            "crlb_pct": fit["crlb_percent"][metabolite],
            "reportable": True}


# =============================================================================
# §4. PET TARGET REGISTRY
# =============================================================================

PET_TRACERS: Dict[str, Dict[str, Any]] = {
 "11C-raclopride": {"target":"D2/D3 receptor", "outcome":"BP_ND",
                    "half_life_min":20.4, "reference":"cerebellum",
                    "note":"displaceable by endogenous dopamine"},
 "11C-DASB":       {"target":"serotonin transporter (SERT)", "outcome":"BP_ND",
                    "half_life_min":20.4, "reference":"cerebellum"},
 "18F-FDOPA":      {"target":"dopamine synthesis capacity", "outcome":"Ki_cer",
                    "half_life_min":109.8, "reference":"cerebellum"},
 "11C-UCB-J":      {"target":"SV2A synaptic vesicle protein", "outcome":"BP_ND",
                    "half_life_min":20.4, "reference":"centrum semiovale",
                    "note":"synaptic density marker"},
 "18F-MK-6240":    {"target":"tau neurofibrillary tangles", "outcome":"SUVR",
                    "half_life_min":109.8, "reference":"cerebellar crus"},
 "11C-PBR28":      {"target":"TSPO (neuroinflammation)", "outcome":"VT",
                    "half_life_min":20.4,
                    "note":"requires TSPO genotype consideration"},
}

def pet_quantify(tracer: str, tissue_tac: "np.ndarray",
                 reference_tac: "np.ndarray", times_min: "np.ndarray",
                 t_star: float = 20.0) -> Dict[str, Any]:
    """
    Simplified reference-tissue Logan graphical analysis for BP_ND.
    BP_ND = slope - 1 for a reference-region input.
    """
    _np()
    if tracer not in PET_TRACERS:
        raise Refusal(f"unknown tracer '{tracer}'")
    dt = np.gradient(times_min)
    int_ref = np.cumsum(reference_tac*dt)
    int_tis = np.cumsum(tissue_tac*dt)
    ok = (times_min >= t_star) & (int_ref > 1e-9)
    if ok.sum() < 5:
        raise Refusal("insufficient post-equilibrium samples for Logan fit")
    x = int_ref[ok]/reference_tac[ok]
    y = int_tis[ok]/reference_tac[ok]
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    resid = y - A@coef
    r2 = 1.0 - float(resid@resid)/max(float(np.var(y)*len(y)), 1e-12)
    return {"tracer": tracer, "target": PET_TRACERS[tracer]["target"],
            "outcome": PET_TRACERS[tracer]["outcome"],
            "logan_slope": slope, "BP_ND": slope - 1.0,
            "fit_r2": r2, "t_star_min": t_star,
            "linearity_ok": r2 > 0.98,
            "requires": ["radiotracer","radiation_consent","kinetic_model"],
            "NOTE": "simplified Logan; production needs full kinetic modelling"}


# =============================================================================
# §5. EEG/MEG FEATURES AND THE APERIODIC E/I PROXY
# =============================================================================

BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),
         "beta":(13,30),"low_gamma":(30,55),"high_gamma":(65,90)}

class EEGFeatures:
    def __init__(self, fs: float):
        self.fs = fs

    def psd(self, x: "np.ndarray", nperseg: int = 512
            ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Welch PSD."""
        _np()
        n = x.shape[-1]
        seg = min(nperseg, n)
        nseg = max(n//seg, 1)
        win = np.hanning(seg)
        acc = np.zeros(seg//2+1)
        for i in range(nseg):
            s = x[..., i*seg:(i+1)*seg]
            if s.shape[-1] < seg: break
            X = np.fft.rfft(s*win, axis=-1)
            acc = acc + (np.abs(X)**2).mean(axis=tuple(range(s.ndim-1)))
        acc /= max(nseg, 1)*max((win**2).sum(), 1e-12)
        f = np.fft.rfftfreq(seg, 1.0/self.fs)
        return f, acc

    def aperiodic_fit(self, f: "np.ndarray", psd: "np.ndarray",
                      fit_range: Tuple[float, float] = (2.0, 40.0),
                      exclude_bands: Sequence[Tuple[float,float]] = ((7,14),)
                      ) -> Dict[str, Any]:
        """
        Fit log P = offset - chi * log f over a range excluding oscillatory
        peaks. chi is the aperiodic exponent used as an E/I PROXY.
        """
        _np()
        m = (f >= fit_range[0]) & (f <= fit_range[1]) & (psd > 0)
        for lo, hi in exclude_bands:
            m &= ~((f >= lo) & (f <= hi))
        if m.sum() < 6:
            raise Refusal("too few points for aperiodic fit")
        lf = np.log10(f[m]); lp = np.log10(psd[m])
        A = np.column_stack([np.ones_like(lf), -lf])
        coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
        pred = A@coef
        resid = lp - pred
        r2 = 1.0 - float(resid@resid)/max(float(np.var(lp)*len(lp)), 1e-12)
        return {"offset": float(coef[0]), "exponent_chi": float(coef[1]),
                "fit_r2": r2, "n_points": int(m.sum()),
                "fit_range_hz": list(fit_range),
                "excluded_bands": [list(b) for b in exclude_bands],
                "interpretation": ("steeper (larger chi) -> relatively more "
                                   "inhibition; PROXY not concentration"),
                "reference": "Gao, Peterson & Voytek, NeuroImage 2017",
                "fit_quality_ok": r2 > 0.90}

    def band_power(self, f: "np.ndarray", psd: "np.ndarray"
                   ) -> Dict[str, float]:
        out = {}
        for b, (lo, hi) in BANDS.items():
            if hi >= self.fs/2: continue
            m = (f >= lo) & (f < hi)
            out[b] = float(psd[m].mean()) if m.any() else 0.0
        return out

    def ei_proxy(self, x: "np.ndarray") -> Dict[str, Any]:
        f, p = self.psd(x)
        ap = self.aperiodic_fit(f, p)
        bp = self.band_power(f, p)
        return {"aperiodic": ap, "band_power": bp,
                "EI_proxy_exponent": ap["exponent_chi"],
                "STATUS": "PROXY — not a GABA or glutamate concentration",
                "mechanistic_basis": ("AMPA tau ~2 ms vs GABA-A tau ~10 ms "
                                      "shapes the PSD slope"),
                "NOT": "a molecule-specific measurement",
                "validation_required": ["paired MRS", "cross-validation",
                                        "EMG/EOG control"]}


# =============================================================================
# §6. PAIRED MULTIMODAL REGRESSION  (the only defensible prediction path)
# =============================================================================

@dataclass
class PairedDataset:
    """Each row: EEG/MEG features + an ACTUAL MRS/PET measurement."""
    X: "np.ndarray"                 # [n_subjects, n_features]
    Y: "np.ndarray"                 # [n_subjects, n_targets] measured
    target_names: List[str]
    feature_names: List[str]
    subject_ids: List[str]
    session_ids: List[str]
    reference_modality: str = Modality.MRS
    crlb_percent: Optional["np.ndarray"] = None

class PairedRegression:
    """
    Predict a MEASURED chemical quantity from EEG/MEG features.
    Requires paired data, subject-wise cross-validation, permutation test.
    Without these the output is Mode E.
    """
    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit(self, X: "np.ndarray", Y: "np.ndarray") -> Dict[str, Any]:
        _np()
        Xm = X.mean(0); Xs = X.std(0) + 1e-9
        Xz = (X - Xm)/Xs
        A = Xz.T@Xz + self.lam*np.eye(Xz.shape[1])
        W = np.linalg.solve(A, Xz.T@(Y - Y.mean(0)))
        return {"W": W, "X_mean": Xm, "X_std": Xs,
                "Y_mean": Y.mean(0), "lam": self.lam}

    def predict(self, model: Dict, X: "np.ndarray") -> "np.ndarray":
        Xz = (X - model["X_mean"])/model["X_std"]
        return Xz@model["W"] + model["Y_mean"]

    def subject_cv(self, ds: PairedDataset, n_folds: int = 5,
                   min_subjects: int = 30) -> Dict[str, Any]:
        """
        SUBJECT-WISE folds. Trial-wise splits leak subject identity and
        inflate accuracy.
        """
        _np()
        subs = sorted(set(ds.subject_ids))
        if len(subs) < min_subjects:
            return {"valid": False,
                    "reason": f"need >= {min_subjects} subjects, got {len(subs)}",
                    "status": "MODE_E"}
        rng = np.random.default_rng(0)
        perm = rng.permutation(subs)
        folds = [set(perm[i::n_folds]) for i in range(n_folds)]
        r_list, pred_all, true_all = [], np.zeros_like(ds.Y), np.zeros_like(ds.Y)
        idx_of = {s: i for i, s in enumerate(subs)}
        for k in range(n_folds):
            te = np.array([s in folds[k] for s in ds.subject_ids])
            tr = ~te
            if tr.sum() < 5 or te.sum() < 2: continue
            m = self.fit(ds.X[tr], ds.Y[tr])
            p = self.predict(m, ds.X[te])
            pred_all[te] = p; true_all[te] = ds.Y[te]
            for j in range(ds.Y.shape[1]):
                yt, yp = ds.Y[te, j], p[:, j]
                if np.var(yt) > 1e-12:
                    r_list.append((ds.target_names[j], k,
                                   float(np.corrcoef(yt, yp)[0, 1])))
        summary = {}
        for nm in ds.target_names:
            rs = [r for (t, k, r) in r_list if t == nm]
            if rs:
                summary[nm] = {"mean_r": float(np.mean(rs)),
                               "sd_r": float(np.std(rs)),
                               "min_r": float(np.min(rs)),
                               "n_folds": len(rs),
                               "predictive": float(np.mean(rs)) > 0.3
                                             and float(np.min(rs)) > 0.0}
        return {"valid": True, "n_subjects": len(subs), "n_folds": n_folds,
                "per_target": summary, "split": "SUBJECT-WISE",
                "predictions": pred_all, "truth": true_all,
                "NOTE": ("a positive r means EEG features carry information "
                         "about the MEASURED quantity in this cohort; it does "
                         "NOT license concentration claims for new individuals "
                         "without further validation")}

    def permutation_test(self, ds: PairedDataset, n_perm: int = 500,
                         seed: int = 0) -> Dict[str, Any]:
        _np()
        rng = np.random.default_rng(seed)
        m = self.fit(ds.X, ds.Y)
        p = self.predict(m, ds.X)
        obs = float(np.mean([np.corrcoef(ds.Y[:, j], p[:, j])[0, 1]
                             for j in range(ds.Y.shape[1])
                             if np.var(ds.Y[:, j]) > 1e-12]))
        null = np.empty(n_perm)
        for i in range(n_perm):
            Yp = rng.permutation(ds.Y, axis=0)
            mm = self.fit(ds.X, Yp)
            pp = self.predict(mm, ds.X)
            null[i] = np.mean([np.corrcoef(Yp[:, j], pp[:, j])[0, 1]
                               for j in range(Yp.shape[1])
                               if np.var(Yp[:, j]) > 1e-12])
        pv = (1.0 + float(np.sum(null >= obs)))/(n_perm + 1.0)
        return {"observed_mean_r": obs, "null_mean": float(null.mean()),
                "p_value": pv, "n_perm": n_perm,
                "above_chance": pv < 0.01,
                "CAVEAT": "in-sample permutation; subject CV is the real test"}

    def gate_prediction(self, ds: PairedDataset, cv: Dict[str, Any],
                        target: str) -> Dict[str, Any]:
        if not cv.get("valid"):
            raise NeedsReference(f"prediction not certified: {cv.get('reason')}")
        s = cv["per_target"].get(target)
        if s is None or not s["predictive"]:
            raise NeedsReference(
                f"'{target}' is not predicted above chance in subject-wise CV. "
                "No prediction may be reported.")
        if ds.crlb_percent is not None:
            j = ds.target_names.index(target)
            bad = float(np.mean(ds.crlb_percent[:, j] > 20.0))
            if bad > 0.3:
                raise UnreliableQuantification(
                    f"{bad*100:.0f}% of reference measurements for '{target}' "
                    "have CRLB > 20%. The ground truth itself is unreliable.")
        return {"target": target, "certified": True,
                "mean_r": s["mean_r"], "min_r": s["min_r"],
                "reference_modality": ds.reference_modality,
                "SCOPE": "cohort-level statistical association, not an "
                         "individual concentration measurement"}


# =============================================================================
# §7. RESEARCH SURROGATES
# =============================================================================

if HAS_TORCH:
    class MultimodalTransformer(nn.Module):
        """
        Fuses EEG/MEG features with MRS/PET tokens. Predicts the MEASURED
        chemical target. A nuisance head predicts artifact so confounds can
        be inspected. RESEARCH MODEL.
        """
        def __init__(self, d_eeg: int = 24, d_mrs: int = 10,
                     d: int = 128, n_heads: int = 4, n_layers: int = 3,
                     n_targets: int = 4, max_len: int = 64):
            super().__init__()
            self.eeg_proj = nn.Linear(d_eeg, d)
            self.mrs_proj = nn.Linear(d_mrs, d)
            self.type_emb = nn.Parameter(torch.randn(2, d)*0.02)
            self.pos = nn.Parameter(torch.randn(1, max_len, d)*0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, d*4, 0.1,
                                               batch_first=True,
                                               norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d, n_targets)
            self.log_sd = nn.Linear(d, n_targets)
            self.nuisance = nn.Linear(d, 1)

        def forward(self, eeg_seq, mrs_tokens=None):
            B, L, _ = eeg_seq.shape
            h = self.eeg_proj(eeg_seq) + self.type_emb[0]
            if mrs_tokens is not None:
                mt = self.mrs_proj(mrs_tokens) + self.type_emb[1]
                h = torch.cat([h, mt], dim=1)
            h = self.trunk(h + self.pos[:, :h.shape[1]])
            pooled = h.mean(1)
            return {"pred": self.head(pooled),
                    "log_sd": self.log_sd(pooled).clamp(-6, 3),
                    "nuisance": self.nuisance(pooled),
                    "STATUS": "research surrogate, not a measurement"}

    class NeurochemWorldModel(nn.Module):
        """
        Latent dynamics over neural features and a SLOW chemical variable.
        The chemical state is a model latent constrained by measured data,
        not a readout of concentration.
        """
        def __init__(self, d_obs: int = 24, d_chem: int = 4,
                     d_act: int = 4, d: int = 128):
            super().__init__()
            self.enc = nn.Linear(d_obs + d_chem + d_act, d)
            self.gru = nn.GRUCell(d, d)
            self.obs_head = nn.Linear(d, d_obs)
            self.chem_head = nn.Linear(d, d_chem)
            self.log_sd = nn.Linear(d, d_chem)
            # chemical dynamics are SLOW (metabolic timescale)
            self.chem_decay = nn.Parameter(torch.tensor(0.98))

        def forward(self, obs, chem, act):
            h = F.gelu(self.enc(torch.cat([obs, chem, act], -1)))
            h = self.gru(h, h)
            chem_next = self.chem_decay*chem + self.chem_head(h)*0.02
            return {"obs_pred": self.obs_head(h),
                    "chem_next": chem_next,
                    "log_sd": self.log_sd(h).clamp(-6, 3),
                    "STATUS": ("latent chemical variable is model-internal; "
                               "it is NOT a measured concentration")}

    class SpectrumDiffusion(nn.Module):
        """
        Generates CANDIDATE MRS spectra for hypothesis testing and for
        testing fitting robustness. A generated spectrum is never data.
        """
        def __init__(self, d: int = 256, T: int = 100, hidden: int = 256):
            super().__init__()
            self.T, self.d = T, d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b, 0))
            self.net = nn.Sequential(
                nn.Linear(d+1, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, d))

        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            pred = self.net(torch.cat([xt, t.float()[:, None]/self.T], -1))
            return F.mse_loss(pred, eps)

        @torch.no_grad()
        def sample(self, n: int, device: str = "cpu"):
            x = torch.randn(n, self.d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, tb.float()[:, None]/self.T], -1))
                a = 1-self.betas[t]; ab = self.abar[t]
                x = (x - (1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate_spectrum": x,
                    "STATUS": "HYPOTHESIS — never a measurement"}


# =============================================================================
# §8. STATISTICS DICTIONARY
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "mrs_concentration": (("MRS","CRLB","REF"), "linear combination fitting",
                            "CRLB <= 20%; edited sequence for GABA/GSH"),
      "pet_binding": (("TRACER","KIN","CONSENT"), "reference-tissue Logan",
                      "requires radiotracer and kinetic model"),
      "ei_proxy": (("APER","ART"), "Gao-Peterson-Voytek 2017",
                   "PROXY only; not molecule-specific"),
      "eeg_to_chemistry": (("PAIRED","SUBJCV","PERM","CRLB"),
                           "paired regression",
                           "cohort association, not individual concentration"),
      "concentration_from_eeg": ((), "none",
                                 "NOT IDENTIFIABLE — refused"),
      "fmri_concentration": ((), "none", "BOLD is hemodynamic — refused"),
    }
    def __init__(self): self.log: List[Dict] = []
    def certify(self, stat: str, verified: Sequence[str], value=None) -> Dict:
        e = self.ENTRIES.get(stat); v = tuple(verified)
        if e is None:
            r = {"statistic": stat, "status": MODE_E, "reason": "no entry"}
        else:
            need, thm, cav = e
            miss = tuple(p for p in need if p not in v)
            r = {"statistic": stat, "theorem": thm, "caveat": cav,
                 "required": list(need), "verified": list(v),
                 "missing": list(miss),
                 "status": MODE_E if miss else MODE_CERTIFIED,
                 "reason": f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def report(self) -> Dict[str, Any]:
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}

PACKAGES = {
 "MRS":"MRS acquisition present","CRLB":"CRLB <= 20% reported",
 "REF":"water/internal reference with relaxation correction",
 "TRACER":"radiotracer administered","KIN":"kinetic model fitted",
 "CONSENT":"radiation/invasive consent","APER":"aperiodic fit r2 > 0.9",
 "ART":"EMG/EOG artifact controlled","PAIRED":"paired EEG+MRS/PET data",
 "SUBJCV":"subject-wise cross-validation >= 30 subjects",
 "PERM":"permutation test p < 0.01",
}


# =============================================================================
# §9. PIPELINE
# =============================================================================

class NeurochemPipeline:
    def __init__(self, out_dir: str = "runs/neurochem"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()

    def refuse_concentration_from_eeg(self, request: str) -> Dict[str, Any]:
        low = request.lower()
        blocked = ["gaba concentration from eeg","dopamine level from eeg",
                   "serotonin from meg","ความเข้มข้นจาก eeg",
                   "พยากรณ์โมเลกุลอย่างละเอียด","molecular prediction from eeg",
                   "neurotransmitter concentration from meg"]
        for b in blocked:
            if b in low:
                raise NotIdentifiable(
                    f"REFUSED: '{b}'.\n"
                    "The forward map chemistry -> scalp signal is many-to-one; "
                    "most chemical directions lie in its null space. By the "
                    "data-processing inequality no model can recover them.\n"
                    "Use MRS for concentration, PET for receptor binding, or "
                    "a paired-data regression validated subject-wise.")
        return {"allowed": True}

    def analyse_eeg(self, x: "np.ndarray", fs: float) -> Dict[str, Any]:
        self.refuse_concentration_from_eeg("eeg analysis")
        ft = EEGFeatures(fs)
        ei = ft.ei_proxy(x)
        ok = ei["aperiodic"]["fit_quality_ok"]
        cert = self.sd.certify("ei_proxy", ("APER",) if ok else (),
                               {"chi": ei["EI_proxy_exponent"]})
        self.sd.certify("concentration_from_eeg", (), {"refused": True})
        return {"ei": ei, "sd": cert,
                "CONCLUSION": ("aperiodic exponent is an E/I PROXY. "
                               "No concentration is reported."),
                "actionability": RESEARCH_ONLY}

    def analyse_mrs(self, y: "np.ndarray", B: "np.ndarray",
                    names: Sequence[str], edited: bool = False,
                    noise_sd: float = 0.01) -> Dict[str, Any]:
        fit = fit_mrs(y, B, names, noise_sd=noise_sd, edited_acquired=edited)
        good = [nm for nm in fit["reportable"]]
        cert = self.sd.certify("mrs_concentration",
                               ("MRS","CRLB") if good else (),
                               {"reportable": good})
        return {"fit": fit, "sd": cert,
                "NOTE": fit["NOT_A_CONCENTRATION_IN_MM"]}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION,
               "capabilities": {k: asdict(v) for k, v in CAPABILITIES.items()},
               "metabolites": {k: asdict(v) for k, v in METABOLITES.items()},
               "pet_tracers": PET_TRACERS, "packages": PACKAGES,
               "sd_report": self.sd.report(),
               "actionability": RESEARCH_ONLY}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_NUMPY:
        print("  numpy required. FAIL"); return False
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    rng = np.random.default_rng(0)
    pipe = NeurochemPipeline()

    print("\n[T1] CAPABILITY TIERS")
    for m in (Modality.EEG, Modality.MEG):
        for tgt in ("GABA_mM","dopamine_concentration"):
            b = False
            try: check_target(m, tgt)
            except NotIdentifiable: b = True
            rep(f"{m} REFUSES {tgt}", b)
    rep("MRS permits GABA (edited)",
        check_target(Modality.MRS, "GABA_concentration_edited")["allowed"])
    rep("PET permits D2 binding",
        check_target(Modality.PET, "D2_D3_binding_potential")["allowed"])
    b = False
    try: check_target(Modality.FMRI, "metabolite_concentration")
    except NotIdentifiable: b = True
    rep("fMRI REFUSES concentration", b)
    rep("MRS measures concentration",
        CAPABILITIES[Modality.MRS].measures_concentration)
    rep("EEG does NOT measure concentration",
        not CAPABILITIES[Modality.EEG].measures_concentration)

    print("\n[T2] IDENTIFIABILITY — the null space")
    names = ["Glu","GABA","DA","5HT","ACh","NE","GSH","Gln","Tau","mI"]
    J = build_eeg_chemical_jacobian(12, 10)
    ia = identifiability_analysis(J, 10, names)
    print(f"      rank={ia['rank']}  null_dim={ia['null_dimension']}")
    print(f"      poorly identified: {ia['poorly_identified']}")
    rep("chemical state NOT identifiable from EEG-like Jacobian",
        ia["null_dimension"] > 0)
    rep("most molecules poorly identified",
        len(ia["poorly_identified"]) >= 6)
    rep("data-processing principle stated",
        ia["principle"] == "I(C; estimate) <= I(C; Y)")
    Jfull = np.eye(10)
    ia2 = identifiability_analysis(Jfull, 10, names)
    rep("identity Jacobian IS identifiable", ia2["null_dimension"] == 0)

    print("\n[T3] MRS QUANTIFICATION — the real measurement")
    truth = {"NAA":10.0,"Cre":8.0,"Cho":2.0,"mI":5.0,
             "Glu":9.0,"Gln":3.0,"GABA":1.5,"Lac":0.5}
    syn = synthesize_mrs(truth, field_T=3.0, noise_sd=0.02, seed=1)
    print(f"      Larmor at 3T: {GAMMA_HZ_PER_T*3/1e6:.1f} MHz")
    print(f"      NAA offset from water: {ppm_to_hz(2.008,3.0):.1f} Hz")
    rep("NAA at ~337 Hz from water (3T)",
        abs(ppm_to_hz(2.008, 3.0) - 337.4) < 2.0)
    fit = fit_mrs(syn["spectrum"], syn["basis"], syn["names"],
                  baseline_design=syn["baseline_design"],
                  noise_sd=0.02, edited_acquired=True)
    for nm in ("NAA","Cre","Glu"):
        t = truth[nm]; e = fit["concentrations"][nm]
        print(f"      {nm}: true={t:.2f} est={e:.2f} "
              f"CRLB={fit['crlb_percent'][nm]:.1f}%")
        rep(f"{nm} recovered within 15%", abs(e-t)/t < 0.15,
            f"err={abs(e-t)/t*100:.1f}%")
    rep("CRLB reported for every metabolite",
        len(fit["crlb_percent"]) == len(syn["names"]))
    rep("units caveat present", "NOT_A_CONCENTRATION_IN_MM" in fit)

    print("\n[T4] EDITED-ACQUISITION GATE")
    fit_un = fit_mrs(syn["spectrum"], syn["basis"], syn["names"],
                     noise_sd=0.02, edited_acquired=False)
    print(f"      edited metabolites without editing: "
          f"{fit_un['edited_metabolites_without_editing']}")
    rep("GABA flagged without editing",
        "GABA" in fit_un["edited_metabolites_without_editing"])
    b = False
    try: require_reportable(fit_un, "GABA")
    except UnreliableQuantification: b = True
    rep("GABA REFUSED without edited acquisition", b)
    rep("GABA reportable WITH editing",
        require_reportable(fit, "GABA")["reportable"])
    noisy = synthesize_mrs(truth, noise_sd=1.0, seed=2)
    fit_noisy = fit_mrs(noisy["spectrum"], noisy["basis"], noisy["names"],
                        noise_sd=1.0, edited_acquired=True)
    rep("high noise -> some metabolites unreliable",
        len(fit_noisy["unreliable_CRLB"]) > 0,
        str(fit_noisy["unreliable_CRLB"][:3]))

    print("\n[T5] PET")
    t = np.arange(0, 90, 0.5)
    ref = 20*np.exp(-t/25) + 2
    tis = 2.4*ref*(1-np.exp(-t/8))
    pet = pet_quantify("11C-raclopride", tis, ref, t, t_star=25.0)
    print(f"      Logan slope={pet['logan_slope']:.3f} "
          f"BP_ND={pet['BP_ND']:.3f} r2={pet['fit_r2']:.4f}")
    rep("PET BP_ND computed", pet["BP_ND"] > 0)
    rep("target is D2/D3", "D2" in pet["target"])
    rep("radiotracer required", "radiotracer" in pet["requires"])
    b = False
    try: pet_quantify("unknown-tracer", tis, ref, t)
    except Refusal: b = True
    rep("unknown tracer refused", b)

    print("\n[T6] EEG APERIODIC E/I PROXY")
    fs = 256.0; n = 4096
    tt = np.arange(n)/fs
    # synthesise 1/f^chi plus an alpha peak
    def make(chi, seed):
        r = np.random.default_rng(seed)
        f = np.fft.rfftfreq(n, 1/fs)
        amp = np.zeros_like(f)
        amp[1:] = f[1:]**(-chi/2)
        ph = r.uniform(0, 2*PI, len(f))
        x = np.fft.irfft(amp*np.exp(1j*ph), n=n)
        x = x/max(np.std(x), 1e-9)
        return x + 1.5*np.sin(2*PI*10*tt)
    x_inh = make(2.4, 1); x_exc = make(1.2, 2)
    ft = EEGFeatures(fs)
    ei_i = ft.ei_proxy(x_inh); ei_e = ft.ei_proxy(x_exc)
    print(f"      steep chi={ei_i['EI_proxy_exponent']:.3f}  "
          f"shallow chi={ei_e['EI_proxy_exponent']:.3f}")
    rep("steeper spectrum gives larger chi",
        ei_i["EI_proxy_exponent"] > ei_e["EI_proxy_exponent"])
    rep("labelled PROXY not concentration",
        "PROXY" in ei_i["STATUS"])
    rep("not molecule-specific", "NOT" in ei_i)
    rep("reference cited", "2017" in ei_i["aperiodic"]["reference"])
    rep("alpha excluded from aperiodic fit",
        [7,14] in ei_i["aperiodic"]["excluded_bands"])

    print("\n[T7] PAIRED REGRESSION — the only defensible prediction")
    n_sub = 60; n_feat = 12
    X = rng.normal(0, 1, (n_sub, n_feat))
    true_w = rng.normal(0, 1, (n_feat, 2))
    Y = X@true_w + rng.normal(0, 0.5, (n_sub, 2))
    ds = PairedDataset(X, Y, ["GABA_MRS","Glu_MRS"],
                       [f"f{i}" for i in range(n_feat)],
                       [f"s{i:03d}" for i in range(n_sub)],
                       ["sess1"]*n_sub, Modality.MRS,
                       crlb_percent=np.full((n_sub,2), 8.0))
    reg = PairedRegression(lam=1.0)
    cv = reg.subject_cv(ds, n_folds=5, min_subjects=30)
    for nm, s in cv["per_target"].items():
        print(f"      {nm}: mean_r={s['mean_r']:.3f} min_r={s['min_r']:.3f}")
    rep("subject-wise CV runs", cv["valid"])
    rep("split is SUBJECT-WISE", cv["split"] == "SUBJECT-WISE")
    rep("signal detected", cv["per_target"]["GABA_MRS"]["predictive"])
    small = PairedDataset(X[:10], Y[:10], ["GABA_MRS","Glu_MRS"],
                          ds.feature_names,
                          [f"s{i}" for i in range(10)], ["sess1"]*10)
    cv_small = reg.subject_cv(small, min_subjects=30)
    rep("<30 subjects -> MODE_E", cv_small["status"] == MODE_E)
    perm = reg.permutation_test(ds, n_perm=200)
    rep("permutation test significant", perm["above_chance"],
        f"p={perm['p_value']:.4f}")
    gate = reg.gate_prediction(ds, cv, "GABA_MRS")
    rep("prediction certified with paired data", gate["certified"])
    rep("scope limited to cohort association",
        "cohort-level" in gate["SCOPE"])
    Ynoise = rng.normal(0, 1, (n_sub, 2))
    ds_noise = PairedDataset(X, Ynoise, ["GABA_MRS","Glu_MRS"],
                             ds.feature_names, ds.subject_ids,
                             ds.session_ids, Modality.MRS)
    cv_n = reg.subject_cv(ds_noise)
    b = False
    try: reg.gate_prediction(ds_noise, cv_n, "GABA_MRS")
    except NeedsReference: b = True
    rep("unpredictable target REFUSED", b)
    bad_crlb = PairedDataset(X, Y, ["GABA_MRS","Glu_MRS"], ds.feature_names,
                             ds.subject_ids, ds.session_ids, Modality.MRS,
                             crlb_percent=np.full((n_sub,2), 35.0))
    b = False
    try: reg.gate_prediction(bad_crlb, cv, "GABA_MRS")
    except UnreliableQuantification: b = True
    rep("unreliable ground truth REFUSED", b)

    print("\n[T8] SD GATE + PIPELINE")
    c1 = pipe.sd.certify("concentration_from_eeg", ())
    rep("concentration from EEG -> MODE_E", c1["status"] == MODE_E)
    c2 = pipe.sd.certify("eeg_to_chemistry", ("PAIRED","SUBJCV","PERM"))
    rep("missing CRLB -> MODE_E", c2["status"] == MODE_E)
    c3 = pipe.sd.certify("eeg_to_chemistry", ("PAIRED","SUBJCV","PERM","CRLB"))
    rep("all packages -> CERTIFIED", c3["status"] == MODE_CERTIFIED)
    b = False
    try: pipe.refuse_concentration_from_eeg("GABA concentration from EEG")
    except NotIdentifiable: b = True
    rep("explicit request REFUSED", b)
    eeg_out = pipe.analyse_eeg(x_inh, fs)
    rep("EEG analysis reports no concentration",
        "No concentration" in eeg_out["CONCLUSION"])
    mrs_out = pipe.analyse_mrs(syn["spectrum"], syn["basis"], syn["names"],
                               edited=True, noise_sd=0.02)
    rep("MRS analysis completes", "fit" in mrs_out)
    rr = pipe.report()
    rep("report written", (pipe.out/"report.json").exists())
    rep("Mode E retained", rr["sd_report"]["mode_E"] > 0)

    if HAS_TORCH:
        print("\n[T9] RESEARCH MODELS")
        torch.manual_seed(0)
        m = MultimodalTransformer(24, 10, 64, 4, 2, 2, 32)
        eeg = torch.randn(4, 16, 24); mrs = torch.randn(4, 3, 10)
        o = m(eeg, mrs)
        rep("transformer forward", o["pred"].shape == (4,2))
        rep("nuisance head present", o["nuisance"].shape == (4,1))
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        tgt = torch.randn(4,2); l0=lN=None
        for ep in range(50):
            opt.zero_grad()
            L = F.mse_loss(m(eeg, mrs)["pred"], tgt)
            L.backward(); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("transformer trains", lN < l0, f"{l0:.3f}->{lN:.3f}")
        wm = NeurochemWorldModel(24, 2, 4, 64)
        wo = wm(torch.randn(4,24), torch.randn(4,2), torch.randn(4,4))
        rep("world model chemical latent is model-internal",
            "NOT a measured concentration" in wo["STATUS"])
        df = SpectrumDiffusion(64, 30, 128)
        opt2 = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.randn(8,64); d0=dN=None
        for ep in range(40):
            opt2.zero_grad(); L = df.loss(x0); L.backward(); opt2.step()
            if ep == 0: d0 = float(L)
            dN = float(L)
        rep("diffusion trains", dN < d0, f"{d0:.3f}->{dN:.3f}")
        s = df.sample(2)
        rep("generated spectrum labelled HYPOTHESIS",
            "never a measurement" in s["STATUS"])
    else:
        print("\n[T9] torch absent — research models skipped")

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Capability tiers: EEG/MEG are proxies; MRS and PET measure chemistry.
  Identifiability analysis exposing the null space of the chemistry->signal
  Jacobian, with per-variable identifiability scores.
  MRS linear-combination quantification with baseline projection,
  non-negativity, Tikhonov regularisation, and Cramer-Rao lower bounds.
  Edited-acquisition gate: GABA and GSH are refused without MEGA-PRESS.
  PET reference-tissue Logan analysis with a tracer registry.
  EEG aperiodic exponent as an E/I PROXY (Gao-Peterson-Voytek 2017),
  with oscillatory peaks excluded from the fit.
  Paired regression predicting a MEASURED chemical target from EEG/MEG
  features, gated on subject-wise CV, permutation testing, and CRLB quality
  of the reference measurement.
  Transformer with nuisance head, slow-chemistry world model, and a spectrum
  diffusion generator — all labelled research surrogates / hypotheses.

REFUSED (all fired)
  GABA / dopamine / serotonin concentration from EEG or MEG
  fMRI BOLD reported as a chemical concentration
  GABA quantification without an edited acquisition
  MRS values with CRLB > 20% reported as reliable
  regression prediction without paired data and subject-wise CV
  prediction whose ground truth is itself unreliable

THE HONEST SUMMARY
  Mathematics interprets brain signals; it does not create chemical
  information that the signals never carried. Concentration comes from
  MRS or PET. EEG/MEG contribute a validated E/I proxy and, with paired
  training, a cohort-level statistical association — never an individual
  concentration readout.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Neurochemical inference")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","capabilities","metabolites","tracers",
                 "mrs","eeg","identifiability","packages","report"])
    ap.add_argument("--field", type=float, default=3.0)
    ap.add_argument("--edited", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    pipe = NeurochemPipeline(); out: Dict[str, Any] = {}
    if a.cmd == "capabilities":
        out = {k: asdict(v) for k, v in CAPABILITIES.items()}
    elif a.cmd == "metabolites":
        out = {k: asdict(v) for k, v in METABOLITES.items()}
    elif a.cmd == "tracers": out = PET_TRACERS
    elif a.cmd == "packages": out = PACKAGES
    elif a.cmd == "identifiability":
        names = ["Glu","GABA","DA","5HT","ACh","NE","GSH","Gln","Tau","mI"]
        out = identifiability_analysis(
            build_eeg_chemical_jacobian(12,10), 10, names)
    elif a.cmd == "mrs":
        truth = {"NAA":10.0,"Cre":8.0,"Cho":2.0,"Glu":9.0,"GABA":1.5}
        s = synthesize_mrs(truth, field_T=a.field)
        log.warning("synthetic spectrum (code test only); supply real MRS data")
        out = pipe.analyse_mrs(s["spectrum"], s["basis"], s["names"],
                               edited=a.edited)
    elif a.cmd == "eeg":
        rng = np.random.default_rng(0); fs=256.0; n=4096
        f = np.fft.rfftfreq(n, 1/fs); amp=np.zeros_like(f)
        amp[1:] = f[1:]**(-1.2)
        x = np.fft.irfft(amp*np.exp(1j*rng.uniform(0,2*PI,len(f))), n=n)
        log.warning("synthetic EEG (code test only)")
        out = pipe.analyse_eeg(x, fs)
    elif a.cmd == "report": out = pipe.report()
    print(json.dumps(out, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
