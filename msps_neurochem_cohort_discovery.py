#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_neurochem_cohort_discovery.py — v1.0.0
OPEN RESEARCH module: state-associated neurochemical profiles, cohort design,
drug-target discovery ledger, and device-class verification for delivery
research. IRB + psychiatrist + medical council gated.

PRECISION CORRECTION (enforced throughout)
  It is NOT a "different set of molecules". Dopamine is C8H11NO2 in every
  human. What differs between states and cohorts is:
      concentration, receptor density/occupancy, release kinetics,
      transporter capacity, and circuit context.
  Same molecular species. Different quantitative configuration.

SECOND CORRECTION
  Group-level differences are real and measurable (MRS/PET). They are NOT
  an individual diagnostic and NOT a per-thought molecular barcode.
  For a cohort effect size d, the discriminative overlap is
      AUC = Phi(d / sqrt(2)).
  d = 0.5  ->  AUC 0.64. Substantial overlap. Individual diagnosis REFUSED.

WHAT THIS MODULE IS FOR
  * Designing an honest MRS/PET cohort study (power, effect size, overlap).
  * Reporting group differences WITH overlap, not just p-values.
  * A drug-target ledger with real approval status.
  * Device-class verification for any delivery research (no approved
    implanted delivery indication for depression exists).
  * Gated AI surrogates for hypothesis generation.

REFUSED
  [R1] "different molecule set" framing            -> corrected, not allowed
  [R2] individual diagnosis from a metabolite level -> refused (overlap)
  [R3] per-thought molecular barcode                -> refused
  [R4] implanted delivery for depression            -> no indication
  [R5] any human-subject work without IRB/psych/council -> refused
  [R6] weapons / non-therapeutic enhancement        -> refused
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
"""
from __future__ import annotations

import argparse, json, logging, math, time
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
log = logging.getLogger("neurochem_cohort")
VERSION = "1.0.0"; PI = math.pi
RESEARCH_ONLY = "open_research_not_a_diagnostic"
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

BANNER = f"""
=============================================================================
MSPS NEUROCHEMICAL COHORT & DISCOVERY v{VERSION}
SAME molecules, DIFFERENT levels — not a different molecular set
Group differences WITH overlap (AUC) · power/sample-size · target ledger
device-class gate for delivery research · IRB + psych + council mandatory
REFUSED: individual diagnosis, per-thought barcode, implanted depression drug
=============================================================================
"""

class Refusal(RuntimeError): pass
class FramingRefusal(Refusal): pass
class OverlapRefusal(Refusal): pass
class GovernanceRefusal(Refusal): pass
class DeviceClassRefusal(Refusal): pass
class ScopeRefusal(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")


# =============================================================================
# §1. MOLECULAR IDENTITY — the precision correction, enforced
# =============================================================================

MOLECULES = {
 "dopamine":   {"formula":"C8H11NO2",       "class":"monoamine",
                "receptors":["D1","D2","D3","D4","D5"],
                "varies_by":["concentration","receptor_density",
                             "release_kinetics","DAT_capacity","circuit_site"]},
 "serotonin":  {"formula":"C10H12N2O",      "class":"monoamine",
                "receptors":["5-HT1A","5-HT1B","5-HT2A","5-HT2C","5-HT3"],
                "varies_by":["concentration","receptor_density",
                             "SERT_capacity","release_kinetics","circuit_site"]},
 "GABA":       {"formula":"C4H9NO2",        "class":"amino acid",
                "receptors":["GABA-A","GABA-B"],
                "varies_by":["concentration","subunit_composition",
                             "tonic_vs_phasic","interneuron_density"]},
 "glutamate":  {"formula":"C5H9NO4",        "class":"amino acid",
                "receptors":["NMDA","AMPA","kainate","mGluR"],
                "varies_by":["concentration","subunit_composition",
                             "EAAT_capacity","synaptic_vs_extrasynaptic"]},
 "glutathione":{"formula":"C10H17N3O6S",    "class":"tripeptide",
                "varies_by":["concentration","redox_ratio_GSH_GSSG"]},
}

def correct_framing(statement: str) -> Dict[str, Any]:
    """
    Refuse the 'different set of molecules' framing and return the correct one.
    This is a chemical fact, not a policy preference.
    """
    low = statement.lower()
    bad = ["different set of molecules","different molecules in depressed",
           "ชุดโมเลกุลที่แตกต่าง","โมเลกุลคนละชุด","molecular barcode of thought",
           "barcode ของความคิด","unique molecule for each thought",
           "happy person's molecule","โมเลกุลคนมีความสุข"]
    for b in bad:
        if b in low:
            raise FramingRefusal(
                f"REFUSED framing '{b}'.\n"
                "The molecular SPECIES are identical across people. Dopamine is "
                "C8H11NO2 in a depressed patient and in a joyful one.\n"
                "What differs QUANTITATIVELY: concentration, receptor density and "
                "occupancy, release kinetics, transporter capacity, subunit "
                "composition, and circuit context.\n"
                "Correct framing: 'state-associated neurochemical PROFILE "
                "differences at the cohort level, with reported overlap'.")
    return {"allowed": True,
            "correct_framing":"same species, different quantitative configuration"}


# =============================================================================
# §2. OVERLAP AND POWER — the numbers that decide what may be claimed
# =============================================================================

def phi(x: float) -> float:
    """Standard normal CDF via erf. No scipy needed."""
    return 0.5*(1.0 + math.erf(x/math.sqrt(2.0)))

def probit(p: float, lo: float = -8.0, hi: float = 8.0) -> float:
    """Inverse normal CDF by bisection. Robust, dependency-free."""
    if not (0.0 < p < 1.0): raise Refusal("p in (0,1)")
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if phi(mid) < p: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def overlap_from_d(d: float) -> Dict[str, Any]:
    """
    For two equal-variance normal groups separated by Cohen's d:
        AUC = Phi(d / sqrt(2))
        overlap coefficient OVL = 2 Phi(-|d|/2)
    AUC 0.5 = no discrimination. AUC 1.0 = perfect separation.
    """
    d = abs(d)
    auc = phi(d/math.sqrt(2.0))
    ovl = 2.0*phi(-d/2.0)
    return {"cohens_d": d, "AUC": auc, "overlap_coefficient": ovl,
            "pct_overlap": 100*ovl,
            "interpretation": ("no discrimination" if auc < 0.55 else
                               "weak, heavy overlap" if auc < 0.70 else
                               "moderate, still overlapping" if auc < 0.85 else
                               "strong but NOT individual-diagnostic"),
            "individual_diagnosis_supported": auc >= 0.95,
            "WHY": "group means can differ while distributions overlap heavily"}

def sample_size_two_group(d: float, alpha: float = 0.05,
                          power: float = 0.80) -> Dict[str, Any]:
    """n per group = 2 (z_{1-alpha/2} + z_{power})^2 / d^2."""
    if d <= 0: raise Refusal("d must be positive")
    za = probit(1.0-alpha/2.0); zb = probit(power)
    n = 2.0*(za+zb)**2/(d*d)
    return {"cohens_d": d, "alpha": alpha, "power": power,
            "z_alpha_2": za, "z_power": zb,
            "n_per_group": math.ceil(n), "n_total": 2*math.ceil(n),
            "formula": "n = 2(z_{1-a/2}+z_{power})^2 / d^2",
            "NOTE": "increase for dropout, scanner site effects, and covariates"}

def required_d_for_auc(target_auc: float) -> float:
    """Invert AUC = Phi(d/sqrt2)."""
    if not (0.5 < target_auc < 1.0): raise Refusal("AUC in (0.5,1)")
    return math.sqrt(2.0)*probit(target_auc)


# =============================================================================
# §3. COHORT STUDY DESIGN AND REPORTING
# =============================================================================

@dataclass
class CohortArm:
    label: str
    n: int
    mean: float
    sd: float
    metabolite: str
    region: str
    modality: str = "mrs"
    crlb_pct_mean: float = 8.0
    site_ids: List[str] = field(default_factory=list)

@dataclass
class CohortStudy:
    name: str
    case: CohortArm
    control: CohortArm
    pre_registered: bool = False
    irb_id: str = ""
    primary_endpoint: str = ""
    covariates: List[str] = field(default_factory=list)
    multi_site: bool = False

class CohortAnalysis:
    @staticmethod
    def pooled_d(a: CohortArm, b: CohortArm) -> float:
        if min(a.n, b.n) < 2: raise Refusal("need n>=2 per arm")
        sp2 = ((a.n-1)*a.sd**2 + (b.n-1)*b.sd**2)/(a.n+b.n-2)
        sp = math.sqrt(max(sp2, 1e-30))
        return (a.mean-b.mean)/sp

    @staticmethod
    def welch_t(a: CohortArm, b: CohortArm) -> Dict[str, Any]:
        va, vb = a.sd**2/a.n, b.sd**2/b.n
        se = math.sqrt(max(va+vb, 1e-30))
        t = (a.mean-b.mean)/se
        df = (va+vb)**2/max(va**2/(a.n-1)+vb**2/(b.n-1), 1e-30)
        # normal approximation for large df; flagged
        p_two = 2.0*(1.0-phi(abs(t)))
        return {"t": t, "df_welch": df, "p_normal_approx": p_two,
                "NOTE": "normal approximation; use a t-distribution for df<100"}

    @staticmethod
    def analyse(st: CohortStudy) -> Dict[str, Any]:
        if st.case.metabolite != st.control.metabolite:
            raise Refusal("arms must measure the same metabolite")
        if st.case.modality != st.control.modality:
            raise Refusal("arms must use the same modality")
        d = CohortAnalysis.pooled_d(st.case, st.control)
        ov = overlap_from_d(d)
        tt = CohortAnalysis.welch_t(st.case, st.control)
        ss = sample_size_two_group(abs(d) if abs(d) > 1e-6 else 0.1)
        adequate = min(st.case.n, st.control.n) >= ss["n_per_group"]
        crlb_ok = max(st.case.crlb_pct_mean, st.control.crlb_pct_mean) <= 20.0
        return {"study": st.name, "metabolite": st.case.metabolite,
                "region": st.case.region, "modality": st.case.modality,
                "cohens_d": d, **{f"group_{k}": v for k, v in ov.items()},
                "welch": tt,
                "required_n_per_group": ss["n_per_group"],
                "adequately_powered": adequate,
                "reference_CRLB_ok": crlb_ok,
                "pre_registered": st.pre_registered,
                "multi_site": st.multi_site,
                "CLAIM_ALLOWED": ("cohort-level association with stated overlap"
                                  if (adequate and crlb_ok) else
                                  "NONE — underpowered or unreliable reference"),
                "CLAIM_REFUSED": "individual diagnosis from this metabolite",
                "status": MODE_CERTIFIED if (adequate and crlb_ok
                          and st.pre_registered) else MODE_E}

    @staticmethod
    def refuse_individual_diagnosis(d: float, value: float,
                                    threshold: float) -> Dict[str, Any]:
        ov = overlap_from_d(d)
        if not ov["individual_diagnosis_supported"]:
            raise OverlapRefusal(
                f"REFUSED individual classification at d={d:.2f} "
                f"(AUC={ov['AUC']:.3f}, overlap={ov['pct_overlap']:.0f}%).\n"
                f"A required AUC of 0.95 needs d >= "
                f"{required_d_for_auc(0.95):.2f}.\n"
                "Group differences do not license individual diagnosis. "
                "Clinical assessment remains the diagnostic pathway.")
        return {"allowed": True, "AUC": ov["AUC"]}


# =============================================================================
# §4. MEASUREMENT CAPABILITY (aligned with msps_neurochem_inference)
# =============================================================================

MEASUREMENT = {
 "mrs": {"measures":"metabolite concentration (ratio or absolute)",
         "can":["GABA_edited","Glu","Gln","NAA","Cre","Cho","mI","GSH","Lac"],
         "cannot":["receptor_occupancy","release_kinetics","per_thought_state"],
         "requires":["localised_sequence","basis_set","CRLB<=20%",
                     "water_or_internal_reference","MEGA_PRESS_for_GABA_GSH"]},
 "pet": {"measures":"receptor/transporter binding or synthesis capacity",
         "can":["D2_D3_BPND","SERT_BPND","FDOPA_Ki","SV2A_BPND","TSPO_VT"],
         "cannot":["real_time_release","per_thought_state"],
         "requires":["radiotracer","kinetic_model","radiation_consent",
                     "arterial_or_reference_input"]},
 "eeg_meg": {"measures":"field potentials; E/I PROXY only",
             "can":["aperiodic_exponent","band_power","gamma_correlate"],
             "cannot":["any_concentration","receptor_occupancy"],
             "requires":["EMG_EOG_control","shielded_room_for_MEG"]},
 "microdialysis": {"measures":"extracellular concentration timeseries",
                   "can":["absolute_concentration","dynamics"],
                   "cannot":["non_invasive_use"],
                   "requires":["neurosurgery","clinical_indication","irb"]},
}

def check_measurement(modality: str, target: str) -> Dict[str, Any]:
    if modality not in MEASUREMENT: raise Refusal(f"unknown '{modality}'")
    m = MEASUREMENT[modality]
    if target in m["cannot"]:
        raise Refusal(f"'{target}' NOT obtainable from {modality}: {m['cannot']}")
    if target not in m["can"]:
        raise Refusal(f"'{target}' not a validated target for {modality}")
    return {"allowed": True, "modality": modality, "target": target,
            "requires": m["requires"]}


# =============================================================================
# §5. DRUG TARGET LEDGER — real approval status
# =============================================================================

@dataclass(frozen=True)
class Target:
    name: str
    molecule_system: str
    mechanism: str
    exemplar: str
    depression_status: str      # approved | approved_subset | off_label | investigational
    evidence: str               # strong | moderate | limited | preliminary
    route: str
    note: str

TARGETS: Dict[str, Target] = {
 "SERT": Target("serotonin transporter","serotonin","reuptake inhibition",
    "sertraline, escitalopram","approved","strong","oral",
    "First-line; effect size modest; relapse prevention established."),
 "NET": Target("norepinephrine transporter","norepinephrine","reuptake inhibition",
    "venlafaxine, duloxetine","approved","strong","oral",
    "SNRI class; dose-dependent dual action."),
 "NMDA": Target("NMDA receptor","glutamate","antagonism / rapid plasticity",
    "esketamine (intranasal), ketamine (IV)","approved_subset","strong",
    "intranasal / IV",
    "Esketamine FDA 2019 for TRD; REMS; >=120 min monitoring. IV ketamine off-label."),
 "GABA_A_PAM": Target("GABA-A receptor","GABA","positive allosteric modulation",
    "zuranolone, brexanolone","approved_subset","strong","oral / IV",
    "Zuranolone FDA 2023 for postpartum depression; brexanolone IV for PPD. "
    "THIS IS THE PROOF THAT NEUROCHEMISTRY-INFORMED DISCOVERY WORKS."),
 "5HT2A": Target("5-HT2A receptor","serotonin","agonism / plasticity",
    "psilocybin","investigational","moderate","oral (supervised)",
    "Investigational; supervised setting required; not approved."),
 "KAPPA": Target("kappa opioid receptor","dynorphin","antagonism",
    "aticaprant","investigational","limited","oral","Investigational anhedonia target."),
 "M1_M4": Target("muscarinic M1/M4","acetylcholine","agonism",
    "xanomeline-trospium","investigational","limited","oral",
    "Approved for schizophrenia; depression programmes investigational."),
 "AMPA_POT": Target("AMPA receptor","glutamate","potentiation",
    "investigational compounds","investigational","preliminary","oral",
    "Mechanistic hypothesis downstream of NMDA antagonism."),
}

def target_ledger() -> Dict[str, Any]:
    order = {"approved":0,"approved_subset":1,"off_label":2,"investigational":3}
    rows = sorted(TARGETS.values(), key=lambda t: (order[t.depression_status], t.name))
    return {"targets":[asdict(t) for t in rows],
            "principle":"approved and established before investigational",
            "discovery_works_example":"GABA-A PAM -> zuranolone (FDA 2023, PPD)",
            "NOT_OFFERED":["molecular emotion transfer","per-thought targeting"]}


# =============================================================================
# §6. DEVICE CLASS GATE FOR DELIVERY RESEARCH
# =============================================================================

class DeviceClass:
    ELECTRICAL = "electrical_stimulator"
    DELIVERY = "drug_delivery_implant"
    EXTERNAL = "external_non_implanted"

@dataclass(frozen=True)
class DeviceSpec:
    name: str; cls: str
    reservoir: bool; pathway: bool; release: bool
    can_deliver: bool
    indications: Tuple[str, ...]
    depression_indication: str
    note: str

DEVICES: Dict[str, DeviceSpec] = {
 "DBS": DeviceSpec("DBS",DeviceClass.ELECTRICAL,False,False,False,False,
    ("Parkinson","essential_tremor","dystonia","epilepsy","OCD"),
    "investigational","Electrodes only. NO reservoir. Cannot deliver a substance."),
 "RNS": DeviceSpec("RNS",DeviceClass.ELECTRICAL,False,False,False,False,
    ("refractory_focal_epilepsy",),"none","Responsive stimulation. No drug capability."),
 "VNS": DeviceSpec("VNS",DeviceClass.ELECTRICAL,False,False,False,False,
    ("refractory_epilepsy","TRD"),"approved","Electrical. Approved for TRD in some regions."),
 "PUMP_VERIFIED": DeviceSpec("Implanted infusion pump",DeviceClass.DELIVERY,
    True,True,True,True,("spasticity_baclofen","chronic_pain_morphine"),"none",
    "Reservoir+catheter+pump verified. NO depression indication."),
 "CED_RESEARCH": DeviceSpec("Convection-enhanced delivery",DeviceClass.DELIVERY,
    True,True,True,True,("oncology_CED","GDNF_Parkinson_trial"),"none",
    "Real engineering. Not an approved depression therapy."),
 "ESKETAMINE_IN": DeviceSpec("Esketamine intranasal",DeviceClass.EXTERNAL,
    False,False,False,False,("TRD","MDD_with_suicidality"),"approved",
    "FDA 2019. REMS site. >=120 min observation. NOT an implant."),
 "ZURANOLONE_ORAL": DeviceSpec("Zuranolone oral",DeviceClass.EXTERNAL,
    False,False,False,False,("postpartum_depression",),"approved_subset",
    "FDA 2023 for PPD. Oral. Neurochemistry-informed success."),
}

def verify_device(key: str, claim: str) -> Dict[str, Any]:
    if key not in DEVICES: raise Refusal(f"unknown device '{key}'")
    d = DEVICES[key]
    wants = any(k in claim.lower() for k in
        ("deliver","release","infuse","inject","drug","molecule","substance",
         "จ่ายยา","ส่งสาร","โมเลกุล"))
    if wants and not d.can_deliver:
        raise DeviceClassRefusal(
            f"REFUSED: '{key}' is {d.cls}. reservoir={d.reservoir}, "
            f"pathway={d.pathway}, release={d.release}.\n{d.note}\n"
            "An electrical stimulator cannot deliver a substance. "
            "Verify the actual device model and configuration.")
    return {"allowed": True, "device": asdict(d)}

def authorize_delivery_research(key: str, agent: str,
                                indication: str) -> Dict[str, Any]:
    v = verify_device(key, f"deliver {agent}")
    d = DEVICES[key]
    if d.can_deliver and d.depression_indication == "none":
        raise Refusal(
            f"'{key}' has NO depression indication. Implanted delivery for "
            "depression is not approved. This may be framed only as "
            "pre-clinical / IRB-approved investigational research with a "
            "declared indication, and may not be presented as a therapy.")
    return {"allowed": True, "agent": agent, "indication": indication,
            "scope": RESEARCH_ONLY}


# =============================================================================
# §7. GOVERNANCE — IRB, psychiatrist, medical council, risk screen
# =============================================================================

@dataclass
class Governance:
    irb_id: str = ""
    psychiatrist_id: str = ""
    medical_council_ref: str = ""
    pi_id: str = ""
    protocol_version: str = ""
    pre_registration_url: str = ""
    consent_signed: bool = False
    revocable: bool = True
    withdrawn_at: Optional[str] = None
    data_sharing: str = "deidentified"     # none | deidentified | open
    risk_screen_by: str = ""
    crisis_pathway_documented: bool = False
    ack_group_not_individual: bool = False
    ack_same_molecules: bool = False
    def valid(self) -> bool:
        return bool(self.irb_id and self.psychiatrist_id
                    and self.medical_council_ref and self.pi_id
                    and self.protocol_version and self.consent_signed
                    and self.revocable and self.withdrawn_at is None
                    and self.risk_screen_by and self.crisis_pathway_documented
                    and self.ack_group_not_individual and self.ack_same_molecules)

def authorize_study(g: Governance) -> Dict[str, Any]:
    if g.withdrawn_at:
        raise GovernanceRefusal("Consent WITHDRAWN. Stop all study procedures.")
    if not g.valid():
        raise GovernanceRefusal(
            "Study BLOCKED. Required: IRB, psychiatrist, medical council "
            "reference, PI, protocol version, signed revocable consent, "
            "named risk screen, documented crisis pathway, and explicit "
            "acknowledgement that (a) findings are GROUP-level not individual "
            "and (b) the molecules are the SAME species across people.")
    return {"allowed": True, "irb": g.irb_id, "data_sharing": g.data_sharing,
            "scope": RESEARCH_ONLY,
            "NOT_AUTHORIZED": "individual diagnosis; molecular emotion transfer"}

def crisis_resources(region: str = "TH") -> Dict[str, Any]:
    base = {"TH":{"lines":["กรมสุขภาพจิต 1323 (24 ชม.)","Samaritans Thailand 02-713-6793"],
                  "emergency":"1669 / 191"},
            "US":{"lines":["988 Suicide & Crisis Lifeline"],"emergency":"911"},
            "INTL":{"lines":["findahelpline.com"],"emergency":"local number"}}
    return base.get(region, base["INTL"])


# =============================================================================
# §8. GATED AI SURROGATES
# =============================================================================

if HAS_TORCH:
    class CohortTransformer(nn.Module):
        """
        Predicts a MEASURED metabolite from multimodal features at COHORT level.
        A nuisance head predicts site/scanner so confounds are inspectable.
        Output is a research hypothesis, never an individual diagnosis.
        """
        def __init__(self, d_in=16, d_site=4, d=64, heads=4, layers=2, max_len=64):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.site = nn.Sequential(nn.Linear(d_site,d), nn.GELU(), nn.Linear(d,d))
            self.pos = nn.Parameter(torch.randn(1,max_len,d)*0.02)
            enc = nn.TransformerEncoderLayer(d,heads,d*4,0.1,
                                             batch_first=True,norm_first=True)
            self.trunk = nn.TransformerEncoder(enc,layers)
            self.head = nn.Linear(d,1)
            self.log_sd = nn.Linear(d,1)
            self.nuisance = nn.Linear(d,1)
        def forward(self, x, site):
            B,L,_ = x.shape
            h = self.trunk(self.proj(x)+self.site(site).unsqueeze(1)+self.pos[:,:L])
            p = h.mean(1)
            return {"pred": self.head(p), "log_sd": self.log_sd(p).clamp(-6,3),
                    "nuisance": self.nuisance(p),
                    "STATUS": "cohort-level research hypothesis, NOT a diagnosis"}

    class ProfileWorldModel(nn.Module):
        """Slow neurochemical latent dynamics. Model-internal, not measured."""
        def __init__(self, d_obs=16, d_chem=4, d_act=4, d=64):
            super().__init__()
            self.enc = nn.Linear(d_obs+d_chem+d_act, d)
            self.gru = nn.GRUCell(d,d)
            self.obs = nn.Linear(d,d_obs)
            self.chem = nn.Linear(d,d_chem)
            self.decay = nn.Parameter(torch.tensor(0.98))
        def forward(self, obs, chem, act):
            h = F.gelu(self.enc(torch.cat([obs,chem,act],-1)))
            h = self.gru(h,h)
            return {"obs_pred": self.obs(h),
                    "chem_next": self.decay*chem + 0.02*self.chem(h),
                    "STATUS": "latent is model-internal; NOT a concentration"}

    class CandidateMoleculeDiffusion(nn.Module):
        """
        Generates CANDIDATE descriptor vectors for target exploration.
        A generated vector is a HYPOTHESIS. It is not a molecule, not a
        synthesis route, and not a therapeutic claim.
        """
        def __init__(self, d=32, T=60, hidden=128):
            super().__init__()
            self.T=T; self.d=d
            b=torch.linspace(1e-4,0.02,T)
            self.register_buffer("betas",b)
            self.register_buffer("abar",torch.cumprod(1-b,0))
            self.net=nn.Sequential(nn.Linear(d+1,hidden),nn.GELU(),
                                   nn.Linear(hidden,hidden),nn.GELU(),
                                   nn.Linear(hidden,d))
        def loss(self,x0):
            B=x0.shape[0]
            t=torch.randint(0,self.T,(B,),device=x0.device)
            eps=torch.randn_like(x0); ab=self.abar[t][:,None]
            xt=ab.sqrt()*x0+(1-ab).sqrt()*eps
            return F.mse_loss(self.net(torch.cat([xt,t.float()[:,None]/self.T],-1)),eps)
        @torch.no_grad()
        def sample(self,n,device="cpu"):
            x=torch.randn(n,self.d,device=device)
            for t in reversed(range(self.T)):
                tb=torch.full((n,),t,device=device,dtype=torch.long)
                eps=self.net(torch.cat([x,tb.float()[:,None]/self.T],-1))
                a=1-self.betas[t]; ab=self.abar[t]
                x=(x-(1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t>0: x=x+torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate_descriptors":x,
                    "STATUS":"HYPOTHESIS — not a molecule, not a synthesis route"}

class AIGate:
    @staticmethod
    def individual_gate(auc: float) -> Dict[str, Any]:
        if auc < 0.95:
            raise OverlapRefusal(
                f"AI individual classification REFUSED at AUC={auc:.3f}. "
                "Requires >=0.95 with held-out multi-site validation.")
        return {"allowed": True, "auc": auc}
    @staticmethod
    def target_gate(target_key: str, claim: str) -> Dict[str, Any]:
        if target_key not in TARGETS: raise Refusal("unknown target")
        t = TARGETS[target_key]
        if "cure" in claim.lower() or "รักษาหายขาด" in claim:
            raise Refusal("No target supports a cure claim. Refused.")
        return {"target": asdict(t), "claim_scope": RESEARCH_ONLY,
                "approval_status": t.depression_status}


# =============================================================================
# §9. SD GATE
# =============================================================================

class SDGate:
    ENTRIES = {
      "cohort_difference": (("POWER","CRLB","PREREG"),
                            "two-group comparison with overlap",
                            "group-level only; overlap must be reported"),
      "individual_classification": (("AUC95","XSITE","HELD"),
                                    "requires AUC>=0.95",
                                    "typical MRS d~0.5 gives AUC 0.64 -> refused"),
      "drug_target_claim": (("TRIAL","APPROVAL"), "target ledger",
                            "investigational is not approved"),
      "delivery_device": (("RESERVOIR","INDICATION"), "device class",
                          "no approved implanted depression indication"),
      "ai_profile": (("GATE","XVAL"), "surrogate", "hypothesis only"),
      "different_molecule_set": ((), "none", "REFUSED — same species"),
      "per_thought_barcode": ((), "none", "REFUSED — not identifiable"),
    }
    def __init__(self): self.log: List[Dict] = []
    def certify(self, stat, verified, value=None) -> Dict:
        e = self.ENTRIES.get(stat); v = tuple(verified)
        if e is None: r = {"statistic": stat, "status": MODE_E, "reason":"no entry"}
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
    def report(self):
        n = sum(1 for c in self.log if c["status"]==MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}

PACKAGES = {
 "POWER":"adequate n per group for the observed d",
 "CRLB":"reference measurement CRLB <= 20%",
 "PREREG":"pre-registered primary endpoint",
 "AUC95":"AUC >= 0.95 for individual classification",
 "XSITE":"held-out multi-site validation","HELD":"held-out subjects",
 "TRIAL":"trial evidence","APPROVAL":"regulatory status stated",
 "RESERVOIR":"reservoir+pathway+release verified",
 "INDICATION":"declared approved indication",
 "GATE":"physics/overlap gate passed","XVAL":"cross-validated",
}


# =============================================================================
# §10. PIPELINE
# =============================================================================

class Pipeline:
    def __init__(self, out_dir: str = "runs/neurochem_cohort"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()
    def run(self, g: Governance, st: CohortStudy) -> Dict[str, Any]:
        correct_framing("state-associated neurochemical profile differences")
        auth = authorize_study(g)
        res = CohortAnalysis.analyse(st)
        pk = ("POWER","CRLB","PREREG") if res["status"]==MODE_CERTIFIED else ()
        self.sd.certify("cohort_difference", pk,
                        {"d": res["cohens_d"], "AUC": res["group_AUC"]})
        self.sd.certify("individual_classification", (), {"AUC": res["group_AUC"]})
        self.sd.certify("different_molecule_set", (), {"refused": True})
        self.sd.certify("per_thought_barcode", (), {"refused": True})
        return {"authorization": auth, "cohort": res,
                "molecules": MOLECULES, "targets": target_ledger(),
                "crisis": crisis_resources("TH"),
                "sd_report": self.sd.report(), "scope": RESEARCH_ONLY}
    def report(self):
        rep = {"version": VERSION, "molecules": MOLECULES,
               "measurement": MEASUREMENT, "targets": target_ledger(),
               "devices": {k: asdict(v) for k,v in DEVICES.items()},
               "packages": PACKAGES, "sd_report": self.sd.report(),
               "scope": RESEARCH_ONLY}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2,
                                            default=str, ensure_ascii=False))
        return rep


# =============================================================================
# §11. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n,p,e=""):
        nonlocal ok; ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    pipe = Pipeline()
    good = Governance("IRB-2026-11","psych01","MC-2026-999","pi01","v1.0",
                      "https://osf.io/example", True, True, None, "open",
                      "psych01", True, True, True)

    print("\n[T1] FRAMING CORRECTION — same molecules")
    b = 0
    for s in ("different set of molecules","ชุดโมเลกุลที่แตกต่าง",
              "molecular barcode of thought","happy person's molecule"):
        try: correct_framing(s)
        except FramingRefusal: b += 1
    rep("all wrong framings REFUSED", b == 4, f"{b}/4")
    rep("correct framing allowed",
        correct_framing("cohort profile differences")["allowed"])
    rep("dopamine formula identical",
        MOLECULES["dopamine"]["formula"] == "C8H11NO2")
    rep("varies_by lists quantities not species",
        "concentration" in MOLECULES["dopamine"]["varies_by"])

    print("\n[T2] OVERLAP — the decisive numbers")
    for d, exp in ((0.4,0.611),(0.5,0.638),(0.8,0.714)):
        ov = overlap_from_d(d)
        print(f"      d={d:.1f}  AUC={ov['AUC']:.3f}  overlap={ov['pct_overlap']:.0f}%")
        rep(f"d={d} AUC matches Phi(d/sqrt2)", abs(ov["AUC"]-exp) < 0.005)
        rep(f"d={d} individual NOT supported",
            not ov["individual_diagnosis_supported"])
    need = required_d_for_auc(0.95)
    print(f"      AUC 0.95 requires d >= {need:.2f}")
    rep("required d for AUC .95 ~ 2.33", abs(need-2.326) < 0.03)
    b = False
    try: CohortAnalysis.refuse_individual_diagnosis(0.5, 1.2, 1.0)
    except OverlapRefusal: b = True
    rep("individual diagnosis REFUSED at d=0.5", b)

    print("\n[T3] POWER / SAMPLE SIZE")
    ss = sample_size_two_group(0.5)
    print(f"      d=0.5, alpha=.05, power=.80 -> n={ss['n_per_group']}/group")
    rep("n per group ~63", ss["n_per_group"] == 63, str(ss["n_per_group"]))
    ss8 = sample_size_two_group(0.8)
    rep("larger d needs fewer subjects", ss8["n_per_group"] < ss["n_per_group"])
    rep("probit/phi consistent", abs(phi(probit(0.975))-0.975) < 1e-6)

    print("\n[T4] COHORT ANALYSIS")
    case = CohortArm("MDD", 70, 1.10, 0.30, "GABA", "mPFC", "mrs", 9.0)
    ctrl = CohortArm("HC",  70, 1.28, 0.32, "GABA", "mPFC", "mrs", 8.0)
    st = CohortStudy("GABA-mPFC in MDD", case, ctrl, True,
                     "IRB-2026-11", "GABA concentration", ["age","sex","site"], True)
    r = CohortAnalysis.analyse(st)
    print(f"      d={r['cohens_d']:.3f}  AUC={r['group_AUC']:.3f}  "
          f"n_req={r['required_n_per_group']}  powered={r['adequately_powered']}")
    rep("d negative (case lower)", r["cohens_d"] < 0)
    rep("adequately powered", r["adequately_powered"])
    rep("claim is cohort-level", "cohort-level" in r["CLAIM_ALLOWED"])
    rep("individual claim refused", "individual" in r["CLAIM_REFUSED"])
    small = CohortStudy("underpowered",
        CohortArm("MDD",12,1.10,0.30,"GABA","mPFC"),
        CohortArm("HC",12,1.28,0.32,"GABA","mPFC"), True)
    rs = CohortAnalysis.analyse(small)
    rep("underpowered -> MODE_E", rs["status"] == MODE_E)
    rep("underpowered claim NONE", rs["CLAIM_ALLOWED"].startswith("NONE"))

    print("\n[T5] MEASUREMENT CAPABILITY")
    rep("MRS permits edited GABA",
        check_measurement("mrs","GABA_edited")["allowed"])
    b = False
    try: check_measurement("mrs","receptor_occupancy")
    except Refusal: b = True
    rep("MRS REFUSES receptor occupancy", b)
    b = False
    try: check_measurement("eeg_meg","any_concentration")
    except Refusal: b = True
    rep("EEG/MEG REFUSES concentration", b)
    rep("PET permits D2 binding",
        check_measurement("pet","D2_D3_BPND")["allowed"])

    print("\n[T6] TARGET LEDGER — discovery that worked")
    led = target_ledger()
    rep("GABA-A PAM present", "GABA_A_PAM" in TARGETS)
    rep("zuranolone approved subset",
        TARGETS["GABA_A_PAM"].depression_status == "approved_subset")
    rep("discovery-works example stated",
        "zuranolone" in led["discovery_works_example"])
    rep("psilocybin investigational",
        TARGETS["5HT2A"].depression_status == "investigational")
    b = False
    try: AIGate.target_gate("NMDA","this will cure depression")
    except Refusal: b = True
    rep("cure claim REFUSED", b)
    rep("legitimate target claim allowed",
        AIGate.target_gate("NMDA","TRD trial evidence")["approval_status"]
        == "approved_subset")

    print("\n[T7] DEVICE CLASS GATE")
    b = False
    try: verify_device("DBS","deliver dopamine")
    except DeviceClassRefusal: b = True
    rep("DBS substance claim REFUSED", b)
    rep("DBS has no reservoir", not DEVICES["DBS"].reservoir)
    rep("pump CAN deliver", DEVICES["PUMP_VERIFIED"].can_deliver)
    b = False
    try: authorize_delivery_research("PUMP_VERIFIED","dopamine","depression")
    except Refusal: b = True
    rep("implanted delivery for depression REFUSED", b)
    rep("electrical DBS claim allowed",
        verify_device("DBS","electrical stimulation")["allowed"])
    rep("zuranolone oral approved subset",
        DEVICES["ZURANOLONE_ORAL"].depression_indication == "approved_subset")

    print("\n[T8] GOVERNANCE")
    b = False
    try: authorize_study(Governance())
    except GovernanceRefusal: b = True
    rep("empty governance BLOCKED", b)
    noack = Governance("IRB","p","MC","pi","v1","",True,True,None,"open",
                       "p",True,False,True)
    b = False
    try: authorize_study(noack)
    except GovernanceRefusal: b = True
    rep("no group-vs-individual ack BLOCKED", b)
    wd = Governance("IRB","p","MC","pi","v1","",True,True,
                    "2026-11-27T00:00:00Z","open","p",True,True,True)
    b = False
    try: authorize_study(wd)
    except GovernanceRefusal: b = True
    rep("withdrawn consent STOPS study", b)
    rep("complete governance authorised", authorize_study(good)["allowed"])
    rep("crisis line present",
        "1323" in json.dumps(crisis_resources("TH"), ensure_ascii=False))

    print("\n[T9] AI GATES")
    b = False
    try: AIGate.individual_gate(0.64)
    except OverlapRefusal: b = True
    rep("AI individual gate REFUSED at AUC .64", b)
    rep("AI gate passes at AUC .96", AIGate.individual_gate(0.96)["allowed"])
    if HAS_TORCH:
        torch.manual_seed(0)
        m = CohortTransformer(16,4,64,4,2,32)
        o = m(torch.randn(4,8,16), torch.randn(4,4))
        rep("cohort transformer forward", o["pred"].shape == (4,1))
        rep("nuisance/site head present", o["nuisance"].shape == (4,1))
        rep("labelled NOT a diagnosis", "NOT a diagnosis" in o["STATUS"])
        wm = ProfileWorldModel(16,4,4,64)
        wo = wm(torch.randn(4,16),torch.randn(4,4),torch.randn(4,4))
        rep("world model latent not concentration",
            "NOT a concentration" in wo["STATUS"])
        df = CandidateMoleculeDiffusion(32,30,128)
        opt = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.randn(8,32); l0=lN=None
        for ep in range(30):
            opt.zero_grad(); L=df.loss(x0); L.backward(); opt.step()
            if ep==0: l0=float(L)
            lN=float(L)
        rep("diffusion trains", lN < l0, f"{l0:.3f}->{lN:.3f}")
        s = df.sample(2)
        rep("candidate not a molecule",
            "not a molecule" in s["STATUS"])
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T10] SD GATE + PIPELINE")
    c1 = pipe.sd.certify("individual_classification",("AUC95",))
    rep("missing XSITE/HELD -> MODE_E", c1["status"] == MODE_E)
    c2 = pipe.sd.certify("different_molecule_set",())
    rep("different molecule set -> MODE_E", c2["status"] == MODE_E)
    out = pipe.run(good, st)
    rep("pipeline completes", "cohort" in out)
    rep("scope research only", out["scope"] == RESEARCH_ONLY)
    sr = out["sd_report"]
    print(f"      SD: {sr['n']} outputs, {sr['certified']} certified, {sr['mode_E']} Mode E")
    rep("Mode E retained", sr["mode_E"] > 0)
    pipe.report()
    rep("report written", (pipe.out/"report.json").exists())

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT — OPEN RESEARCH, HONESTLY FRAMED
  Molecular identity table: SAME species, different quantitative configuration.
  Overlap mathematics: AUC = Phi(d/sqrt2), overlap coefficient, and the
  d required for any individual-level claim (AUC 0.95 needs d ~ 2.33).
  Power/sample-size for a two-group MRS/PET cohort.
  Cohort analysis reporting d WITH overlap, Welch t, CRLB quality,
  pre-registration, and an explicit refusal of individual diagnosis.
  Measurement capability tiers: MRS/PET measure chemistry; EEG/MEG do not.
  Drug-target ledger with real approval status, including the GABA-A PAM
  -> zuranolone (FDA 2023) example that neurochemistry-informed discovery
  genuinely works.
  Device-class verification: electrical stimulators cannot deliver drugs;
  implanted delivery has NO approved depression indication.
  Governance: IRB + psychiatrist + medical council + risk screen + crisis
  pathway + explicit acknowledgement of the two framing corrections.
  Gated AI surrogates: cohort transformer with a site-nuisance head,
  slow-chemistry world model, candidate-descriptor diffusion.

REFUSED (all fired)
  "different set of molecules" framing
  per-thought molecular barcode
  individual diagnosis at realistic effect sizes
  concentration from EEG/MEG; receptor occupancy from MRS
  substance delivery on an electrical stimulator
  implanted delivery for depression (no indication)
  cure claims; study without governance; study after withdrawal

WHAT THIS ENABLES
  An honest open cohort study that can find real group-level neurochemical
  differences, report their overlap, and feed a legitimate target-discovery
  pipeline — the same pipeline that produced zuranolone.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Neurochemical cohort & discovery")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","molecules","overlap","power","targets","devices",
                 "measurement","crisis","packages","report"])
    ap.add_argument("--d", type=float, default=0.5)
    ap.add_argument("--auc", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power-target", type=float, default=0.80)
    ap.add_argument("--region", default="TH")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    out: Dict[str, Any] = {}
    if a.cmd == "molecules": out = MOLECULES
    elif a.cmd == "overlap":
        out = overlap_from_d(a.d)
        if a.auc: out["required_d_for_target_auc"] = required_d_for_auc(a.auc)
    elif a.cmd == "power": out = sample_size_two_group(a.d, a.alpha, a.power_target)
    elif a.cmd == "targets": out = target_ledger()
    elif a.cmd == "devices": out = {k: asdict(v) for k,v in DEVICES.items()}
    elif a.cmd == "measurement": out = MEASUREMENT
    elif a.cmd == "crisis": out = crisis_resources(a.region)
    elif a.cmd == "packages": out = PACKAGES
    elif a.cmd == "report": out = Pipeline().report()
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str,
                                           ensure_ascii=False))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
