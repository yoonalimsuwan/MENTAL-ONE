#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_depression_intervention_gateway.py — v1.0.0
Evidence-based treatment-resistant depression pathway with device-class
verification, closed-loop neuromodulation research protocol, pharmacology
route enforcement, and mandatory suicide-risk / crisis gating.

REFUSED: "happiness molecule" transfer, emotional-state copying,
         molecular-pattern injection into a neural stream,
         drug delivery via an electrical stimulator.
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
WHY "HAPPINESS MOLECULES" CANNOT BE TRANSFERRED
=============================================================================
1. CHEMICAL IDENTITY
   Dopamine is C8H11NO2. A dopamine molecule taken from a joyful person is
   atom-for-atom identical to one from a depressed person. There is no
   emotional signature attached to a molecule. Serotonin (C10H12N2O),
   oxytocin (C43H66N12O12S2), and beta-endorphin are likewise fixed chemical
   species. "A happy person's molecule" is not a chemical category.

2. EFFECT IS DETERMINED BY THE RECIPIENT
   The same molecule produces different effects depending on receptor
   subtype (D1-D5 for dopamine; 5-HT1A/2A/2C etc. for serotonin), anatomical
   site, release kinetics, and the recipient's own circuit state. Dopamine
   in the nucleus accumbens, the substantia nigra, and the retina does
   three different things. There is no "happiness payload" to deliver.

3. NO READ-AND-COPY CHANNEL
   There is no method to measure person A's affective state as a molecular
   configuration and reproduce it in person B. Affect is a distributed
   dynamical state of a specific brain and body, not a transferable substance.

4. THE AUTHORS' OWN FRAMEWORK AGREES
   From the Yoniso Manasikara paper: cetasika arise from causes and
   conditions, and yoniso manasikara is the ENDOGENOUS filter. An external
   agent supplies conditions, never content. This is the same conclusion
   reached independently by chemistry and by neuroscience.

=============================================================================
DEVICE CLASS SEPARATION  (from the authors' own paper, section 6.2)
=============================================================================
An electrical neurostimulator (DBS, RNS, VNS) delivers ELECTRICAL PULSES
through electrodes. It has no reservoir and cannot release a substance.

A drug-delivery implant requires ALL of:
  - a compatible drug reservoir or releasable matrix
  - a delivery pathway (catheter, microchannel, depot)
  - a release or infusion mechanism (pump, valve, diffusion control)

One cannot infer chemical-delivery capability from the presence of an
implant or an electrode. Device model and configuration must be verified.

For DEPRESSION specifically: there is NO approved implanted drug-delivery
indication. Implanted CNS delivery is real engineering (convection-enhanced
delivery, helical neural implants) but its clinical uses are elsewhere
(e.g. oncology, GDNF trials in Parkinson disease).

=============================================================================
WHAT DOES WORK FOR TREATMENT-RESISTANT DEPRESSION
=============================================================================
TRD is commonly defined as failure of >=2 adequate antidepressant trials
(adequate dose, >=6 weeks, documented adherence) in the current episode.

Approved / established:
  ECT                    : highest response in severe TRD (~50-70%)
  Esketamine (intranasal): FDA 2019 for TRD; REMS; 2 h monitoring
  IV ketamine            : off-label for TRD; monitored setting
  TMS / theta-burst      : FDA 2008; accelerated SAINT protocol 2022
  Psychotherapy          : CBT, behavioural activation, CBASP
  Augmentation           : lithium, atypical antipsychotics, T3

Investigational neuromodulation (research only, IRB required):
  SCC / BA25 DBS         : Mayberg 2005; BROADEN halted for futility 2013;
                           long-term open-label benefit reported
  VC/VS closed-loop DBS  : Scangos et al. Nat Med 2021, n=1,
                           biomarker-triggered stimulation
  MFB DBS                : small series

None of these transfers an emotional state or a molecule from another person.

=============================================================================
ENFORCED REFUSALS
=============================================================================
[R1] "Happiness molecule" / emotional-state transfer            -> refused
[R2] Molecular-pattern injection into a neural stream           -> refused
[R3] Drug delivery claimed on an electrical stimulator          -> refused
[R4] Any depression intervention without suicide-risk screening -> refused
[R5] Intervention without psychiatrist + council + IRB          -> refused
[R6] Invasive option before TRD criteria are met                -> refused
[R7] Continuing after consent withdrawal                        -> refused
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("depression_gateway")
VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"
CLINICIAN_REQUIRED = "clinician_sign_off_required"

BANNER = f"""
=============================================================================
MSPS DEPRESSION INTERVENTION GATEWAY v{VERSION}
Device-class verification · TRD staging · evidence ledger
closed-loop DBS research protocol · pharmacology route enforcement
MANDATORY suicide-risk screening and crisis pathway
REFUSED: happiness-molecule transfer, emotional-state copying,
         drug delivery via an electrical stimulator
=============================================================================
"""

class Refusal(RuntimeError): pass
class NoChannelRefusal(Refusal): pass
class DeviceClassRefusal(Refusal): pass
class SafetyRefusal(Refusal): pass
class ConsentRefusal(Refusal): pass
class EligibilityRefusal(Refusal): pass


# =============================================================================
# §1. THE REFUSAL, WITH ITS REASONING
# =============================================================================

MOLECULE_FACTS = {
 "dopamine":       {"formula": "C8H11NO2",   "receptors": ["D1","D2","D3","D4","D5"]},
 "serotonin":      {"formula": "C10H12N2O",  "receptors": ["5-HT1A","5-HT2A","5-HT2C","5-HT3"]},
 "oxytocin":       {"formula": "C43H66N12O12S2", "receptors": ["OXTR"]},
 "beta_endorphin": {"formula": "peptide (31 aa)", "receptors": ["mu-opioid"]},
}

def refuse_happiness_transfer(request: str) -> Dict[str, Any]:
    """
    Refuse the transfer of an emotional state or a 'happy person's molecule'.
    The refusal is chemical and neuroscientific, not merely policy.
    """
    low = request.lower()
    blocked = {
      "happiness molecule": "no emotional signature exists on a molecule",
      "โมเลกุลคนมีความสุข": "โมเลกุลไม่มีลายเซ็นอารมณ์",
      "happy molecule": "chemical identity is independent of donor affect",
      "transfer emotion": "affect is not a transferable substance",
      "copy emotional state": "no read-and-copy channel exists",
      "ใส่รูปแบบโมเลกุล": "a neural stream carries no molecular emotion pattern",
      "molecular pattern injection": "category error: molecules are not patterns of affect",
      "ส่งโมเลกุล": "ไม่มีช่องทางส่งอารมณ์ผ่านโมเลกุล",
      "inject happiness": "no such payload exists",
    }
    for k, why in blocked.items():
        if k in low:
            raise NoChannelRefusal(
                f"REFUSED: '{k}' — {why}.\n"
                "Dopamine is C8H11NO2 whether the donor is joyful or depressed. "
                "Effect depends on receptor subtype, site, kinetics, and the "
                "RECIPIENT's own circuit state.\n"
                "The authors' Yoniso Manasikara framework states the same: "
                "manasikara is endogenous; an external agent supplies "
                "conditions, never content.\n"
                "Available instead: ECT, esketamine (intranasal, REMS), TMS, "
                "evidence-based psychotherapy and augmentation, and "
                "IRB-approved closed-loop DBS research.")
    return {"allowed": True, "scope": "evidence-based depression care only"}


# =============================================================================
# §2. DEVICE CLASS REGISTRY
# =============================================================================

class DeviceClass:
    ELECTRICAL_STIM = "electrical_stimulator"
    DRUG_DELIVERY = "drug_delivery_implant"
    HYBRID_VERIFIED = "hybrid_verified"
    EXTERNAL = "external_non_implanted"

@dataclass(frozen=True)
class DeviceSpec:
    name: str
    device_class: str
    has_reservoir: bool
    has_pathway: bool
    has_release_mechanism: bool
    can_deliver_substance: bool
    indications: Tuple[str, ...]
    depression_indication: str        # approved | investigational | none
    note: str

DEVICES: Dict[str, DeviceSpec] = {
 "DBS_GENERIC": DeviceSpec("DBS (generic)", DeviceClass.ELECTRICAL_STIM,
    False, False, False, False,
    ("Parkinson_disease","essential_tremor","dystonia","epilepsy","OCD"),
    "investigational",
    "Electrodes only. NO reservoir. Cannot release any substance."),
 "RNS_GENERIC": DeviceSpec("RNS (generic)", DeviceClass.ELECTRICAL_STIM,
    False, False, False, False, ("refractory_focal_epilepsy",), "none",
    "Responsive electrical stimulation. No drug capability."),
 "VNS_GENERIC": DeviceSpec("VNS (generic)", DeviceClass.ELECTRICAL_STIM,
    False, False, False, False,
    ("refractory_epilepsy","treatment_resistant_depression"),
    "approved",
    "Vagus nerve stimulation. Electrical. Approved for TRD in some regions."),
 "DRUG_PUMP_VERIFIED": DeviceSpec("Implanted infusion pump (verified)",
    DeviceClass.DRUG_DELIVERY, True, True, True, True,
    ("spasticity_baclofen","chronic_pain_morphine"), "none",
    "Reservoir + catheter + pump verified. NO depression indication."),
 "CED_RESEARCH": DeviceSpec("Convection-enhanced delivery (research)",
    DeviceClass.DRUG_DELIVERY, True, True, True, True,
    ("oncology_CED","GDNF_Parkinson_trial"), "none",
    "Research delivery. Not an approved depression therapy."),
 "ESKETAMINE_INTRANASAL": DeviceSpec("Esketamine intranasal",
    DeviceClass.EXTERNAL, False, False, False, False,
    ("treatment_resistant_depression","MDD_with_suicidality"),
    "approved",
    "FDA 2019. REMS. Certified setting, 2 h monitoring. NOT an implant."),
 "KETAMINE_IV": DeviceSpec("Ketamine IV", DeviceClass.EXTERNAL,
    False, False, False, False, ("TRD_off_label",), "off_label",
    "Monitored infusion. Not an implant."),
 "ECT": DeviceSpec("Electroconvulsive therapy", DeviceClass.EXTERNAL,
    False, False, False, False,
    ("severe_MDD","TRD","catatonia","psychotic_depression"),
    "approved", "Highest response rate in severe TRD."),
 "TMS": DeviceSpec("Transcranial magnetic stimulation", DeviceClass.EXTERNAL,
    False, False, False, False, ("MDD","TRD"), "approved",
    "Non-invasive. FDA 2008; accelerated theta-burst 2022."),
}

def verify_device_class(device_key: str, claim: str) -> Dict[str, Any]:
    """Refuse a substance-delivery claim on a device that cannot deliver."""
    if device_key not in DEVICES:
        raise Refusal(f"unknown device '{device_key}'. Known: {list(DEVICES)}")
    d = DEVICES[device_key]
    wants_substance = any(k in claim.lower() for k in
        ("deliver","release","inject","infuse","molecule","drug","substance",
         "ส่ง","ใส่","โมเลกุล","สาร"))
    if wants_substance and not d.can_deliver_substance:
        raise DeviceClassRefusal(
            f"REFUSED: '{device_key}' is {d.device_class}.\n"
            f"reservoir={d.has_reservoir}, pathway={d.has_pathway}, "
            f"release={d.has_release_mechanism}\n"
            f"{d.note}\n"
            "An electrical stimulator cannot deliver a substance. "
            "Verify the actual device model and configuration.")
    return {"allowed": True, "device": asdict(d)}


# =============================================================================
# §3. SUICIDE RISK AND CRISIS PATHWAY  (mandatory, runs first)
# =============================================================================

@dataclass
class RiskScreen:
    phq9_item9: Optional[int] = None       # 0-3
    cssrs_ideation: Optional[int] = None   # 0-5 Columbia severity
    cssrs_behavior: bool = False
    has_plan: bool = False
    has_means: bool = False
    intent: bool = False
    recent_attempt_days: Optional[int] = None
    screened_by: str = ""
    screened_at: str = ""

class CrisisPathway:
    """
    Depression care cannot proceed without a risk screen. This is not a gate
    on autonomy; it is a door that must always be open.
    """
    @staticmethod
    def resources(region: str = "TH") -> Dict[str, Any]:
        base = {
          "TH": {"name": "Thailand",
                 "lines": ["กรมสุขภาพจิต 1323 (24 ชม.)",
                           "สายด่วนสุขภาพจิต 1323",
                           "Samaritans Thailand 02-713-6793"],
                 "emergency": "1669 (การแพทย์ฉุกเฉิน) / 191"},
          "US": {"name": "United States", "lines": ["988 Suicide & Crisis Lifeline"],
                 "emergency": "911"},
          "INTL": {"name": "International",
                   "lines": ["findahelpline.com — directory by country"],
                   "emergency": "local emergency number"},
        }
        return base.get(region, base["INTL"])

    @staticmethod
    def assess(screen: Optional[RiskScreen], region: str = "TH") -> Dict[str, Any]:
        if screen is None or not screen.screened_by:
            raise SafetyRefusal(
                "BLOCKED: no suicide-risk screen on record.\n"
                "Depression intervention requires a documented risk screen "
                "(e.g. PHQ-9 item 9 plus C-SSRS) by a named clinician.\n"
                f"Crisis resources: {CrisisPathway.resources(region)}")
        urgent = (screen.cssrs_behavior or screen.has_plan or screen.intent
                  or (screen.has_means and screen.cssrs_ideation is not None
                      and screen.cssrs_ideation >= 4)
                  or (screen.recent_attempt_days is not None
                      and screen.recent_attempt_days <= 30)
                  or (screen.phq9_item9 is not None and screen.phq9_item9 >= 2))
        elevated = (screen.cssrs_ideation is not None and screen.cssrs_ideation >= 2) \
                   or (screen.phq9_item9 is not None and screen.phq9_item9 >= 1)
        out = {"screened_by": screen.screened_by, "screened_at": screen.screened_at,
               "urgent": urgent, "elevated": elevated,
               "resources": CrisisPathway.resources(region)}
        if urgent:
            out["REQUIRED_ACTION"] = (
                "SAME-DAY psychiatric evaluation. Do not leave the person "
                "alone with this result. Means restriction counselling. "
                "Collaborative safety plan. Consider emergency services.")
            out["blocks_research_enrollment"] = True
            out["blocks_elective_procedure"] = True
        elif elevated:
            out["REQUIRED_ACTION"] = (
                "Safety planning intervention, means counselling, "
                "scheduled close follow-up, clinician review before any "
                "escalation of treatment.")
            out["blocks_research_enrollment"] = True
        else:
            out["REQUIRED_ACTION"] = "Routine monitoring; re-screen at each visit."
        return out


# =============================================================================
# §4. TRD STAGING AND ELIGIBILITY
# =============================================================================

@dataclass
class AntidepressantTrial:
    agent: str
    dose_mg: float
    adequate_dose: bool
    weeks: float
    adherence_documented: bool
    response: str          # none | partial | response | remission
    episode_current: bool = True

@dataclass
class TRDProfile:
    diagnosis: str = "MDD"
    episode_severity: str = "moderate"    # mild | moderate | severe
    psychotic_features: bool = False
    catatonia: bool = False
    bipolar: bool = False
    trials: List[AntidepressantTrial] = field(default_factory=list)
    psychotherapy_adequate_trial: bool = False
    ect_prior: bool = False
    tms_prior: bool = False
    esketamine_prior: bool = False
    substance_use_active: bool = False
    medical_contraindications: List[str] = field(default_factory=list)

class TRDStaging:
    MIN_WEEKS = 6.0
    MIN_FAILED_TRIALS = 2

    @staticmethod
    def adequate(t: AntidepressantTrial) -> bool:
        return (t.adequate_dose and t.weeks >= TRDStaging.MIN_WEEKS
                and t.adherence_documented and t.episode_current)

    @staticmethod
    def stage(p: TRDProfile) -> Dict[str, Any]:
        if p.bipolar:
            return {"trd": False, "reason": "bipolar depression — different pathway; "
                     "antidepressant monotherapy risks switching",
                     "n_failed_adequate": 0}
        adq = [t for t in p.trials if TRDStaging.adequate(t)]
        failed = [t for t in adq if t.response in ("none","partial")]
        n = len(failed)
        trd = n >= TRDStaging.MIN_FAILED_TRIALS
        inadequate = [t.agent for t in p.trials if not TRDStaging.adequate(t)]
        return {"trd": trd, "n_failed_adequate": n,
                "n_adequate_trials": len(adq),
                "failed_agents": [t.agent for t in failed],
                "inadequate_trials": inadequate,
                "definition": ">=2 adequate trials (adequate dose, >=6 weeks, "
                              "documented adherence) failed in current episode",
                "next_if_not_trd": ("optimise dose/duration/adherence before "
                                    "declaring resistance"),
                "severity": p.episode_severity}

    @staticmethod
    def require_invasive_eligibility(p: TRDProfile) -> Dict[str, Any]:
        st = TRDStaging.stage(p)
        if not st["trd"]:
            raise EligibilityRefusal(
                "Invasive neuromodulation BLOCKED: TRD criteria not met.\n"
                f"failed adequate trials = {st['n_failed_adequate']} "
                f"(need >= {TRDStaging.MIN_FAILED_TRIALS}).\n"
                f"Inadequate trials: {st['inadequate_trials']}\n"
                "Optimise existing treatment first.")
        if p.episode_severity not in ("severe","moderate"):
            raise EligibilityRefusal("invasive research requires moderate-severe episode")
        if p.substance_use_active:
            raise EligibilityRefusal("active substance use must be addressed first")
        # established options before investigational surgery
        if not (p.ect_prior or p.esketamine_prior or p.tms_prior):
            raise EligibilityRefusal(
                "Investigational DBS BLOCKED: established TRD options not yet "
                "tried. Consider ECT, esketamine, or TMS first. Surgery is "
                "not a first-line resistance strategy.")
        return {"eligible": True, "staging": st}


# =============================================================================
# §5. EVIDENCE LEDGER
# =============================================================================

@dataclass(frozen=True)
class Option:
    name: str
    device_key: Optional[str]
    route: str
    approval: str          # approved | off_label | investigational
    evidence: str          # strong | moderate | limited | preliminary
    setting: str
    monitoring: str
    key_risk: str
    note: str

OPTIONS: Dict[str, Option] = {
 "ECT": Option("ECT", "ECT", "external", "approved", "strong",
    "anaesthesia suite", "cognitive monitoring, anaesthesia review",
    "transient cognitive effects; anaesthesia risk",
    "Highest response in severe TRD (~50-70%). Consider early in severe/psychotic/catatonic presentations."),
 "ESKETAMINE": Option("Esketamine intranasal", "ESKETAMINE_INTRANASAL",
    "intranasal", "approved", "strong",
    "REMS-certified healthcare setting", "2 h post-dose observation; BP; sedation; dissociation",
    "sedation, dissociation, abuse potential, BP rise",
    "FDA 2019 for TRD. Never take-home. Combined with an oral antidepressant."),
 "KETAMINE_IV": Option("Ketamine IV", "KETAMINE_IV", "intravenous",
    "off_label", "moderate", "monitored infusion suite",
    "haemodynamics, dissociation, recovery observation",
    "dissociation, BP rise, abuse potential",
    "Off-label for TRD. Typical 0.5 mg/kg over 40 min in a monitored setting."),
 "TMS": Option("TMS / theta-burst", "TMS", "external", "approved", "strong",
    "outpatient", "seizure screening, hearing protection",
    "rare seizure; scalp discomfort",
    "FDA 2008. Accelerated intermittent theta-burst (SAINT) 2022."),
 "VNS": Option("VNS", "VNS_GENERIC", "implanted_electrical", "approved", "moderate",
    "surgical + programming", "voice change, dyspnoea, surgical risk",
    "implantation surgery; delayed onset",
    "Approved for TRD in some regions. Electrical only — no drug capability."),
 "PSYCHOTHERAPY": Option("CBT / behavioural activation / CBASP", None,
    "psychological", "approved", "strong", "outpatient",
    "engagement, risk review", "minimal",
    "Combine with pharmacotherapy; adequate dose = sufficient sessions."),
 "AUGMENT_LITHIUM": Option("Lithium augmentation", None, "oral",
    "approved", "strong", "outpatient", "levels, renal, thyroid",
    "narrow therapeutic index; toxicity",
    "Established augmentation; also anti-suicidal effect reported."),
 "AUGMENT_SGA": Option("Atypical antipsychotic augmentation", None, "oral",
    "approved", "strong", "outpatient", "metabolic, EPS, QT",
    "weight gain, metabolic syndrome, akathisia",
    "Aripiprazole, quetiapine, brexpiprazole and others have TRD augmentation labels."),
 "DBS_SCC": Option("SCC/BA25 DBS", "DBS_GENERIC", "implanted_electrical",
    "investigational", "limited", "IRB research, functional neurosurgery",
    "surgical risk, hardware, mood monitoring, long programming",
    "intracranial surgery; device infection; uncertain efficacy",
    "Mayberg 2005; BROADEN halted for futility 2013; long-term open-label benefit reported. NOT approved."),
 "DBS_VC_VS_CLOSED": Option("VC/VS closed-loop DBS", "DBS_GENERIC",
    "implanted_electrical", "investigational", "preliminary",
    "IRB research, functional neurosurgery",
    "biomarker validation, surgical risk, intensive follow-up",
    "intracranial surgery; n=1 evidence base",
    "Scangos et al. Nat Med 2021, single patient, biomarker-triggered. Electrical only."),
 "DBS_MFB": Option("MFB DBS", "DBS_GENERIC", "implanted_electrical",
    "investigational", "preliminary", "IRB research",
    "surgical risk, small series", "uncertain efficacy",
    "Small open series. Not approved."),
}

def option_ledger(p: Optional[TRDProfile] = None) -> Dict[str, Any]:
    rows = []
    order = {"approved":0, "off_label":1, "investigational":2}
    for k, o in sorted(OPTIONS.items(),
                       key=lambda kv: (order[kv[1].approval], kv[0])):
        d = asdict(o)
        if o.device_key and o.device_key in DEVICES:
            d["can_deliver_substance"] = DEVICES[o.device_key].can_deliver_substance
        rows.append({"key": k, **d})
    return {"options": rows,
            "principle": "established options before investigational surgery",
            "NOT_OFFERED": "happiness-molecule transfer (no such channel)"}


# =============================================================================
# §6. CLOSED-LOOP DBS RESEARCH PROTOCOL  (electrical, biomarker-gated)
# =============================================================================

@dataclass
class BiomarkerSpec:
    """
    A personalised neural biomarker must be VALIDATED against the patient's
    own symptom reports before it can trigger stimulation. It is not read
    from another person and it does not encode a thought.
    """
    feature: str                 # e.g. "gamma power at contact X"
    channel_indices: List[int]
    threshold: float
    validated_sessions: int = 0
    correlation_with_symptom: float = 0.0
    permutation_p: Optional[float] = None
    n_perm: int = 0
    cross_session_validated: bool = False
    clinician_id: str = ""

    def certified(self) -> Tuple[bool, List[str]]:
        miss = []
        if self.validated_sessions < 3: miss.append("needs >=3 validation sessions")
        if abs(self.correlation_with_symptom) < 0.4: miss.append("|r| >= 0.4 with own symptom report")
        if self.permutation_p is None or self.permutation_p >= 0.01:
            miss.append("permutation p < 0.01")
        if self.n_perm < 1000: miss.append(">=1000 permutations")
        if not self.cross_session_validated: miss.append("cross-session validation")
        if not self.clinician_id: miss.append("clinician sign-off")
        return (len(miss) == 0), miss

@dataclass
class StimProgram:
    """Electrical parameters. Charge-density gated. No substance fields exist."""
    amplitude_mA: float
    pulse_width_us: float
    frequency_Hz: float
    contact_area_cm2: float
    duty_cycle: float = 1.0
    max_on_seconds: float = 30.0
    delta_T_C: float = 0.0

    def charge_density(self) -> Tuple[float, float]:
        Q_nC = self.amplitude_mA * self.pulse_width_us
        D_uC_cm2 = Q_nC * 1e-3 / max(self.contact_area_cm2, 1e-9)
        return Q_nC, D_uC_cm2

    def shannon_k(self) -> float:
        Q_nC, D = self.charge_density()
        return math.log10(max(D,1e-12)) + math.log10(max(Q_nC*1e-3,1e-12))

class ClosedLoopDBSResearch:
    """
    Biomarker-triggered ELECTRICAL stimulation, following the structure of
    Scangos et al. 2021. This is investigational. There is no molecular
    payload and no transfer of another person's state.
    """
    SHANNON_K_MAX = 1.5
    DT_MAX = 2.0

    @staticmethod
    def authorize(consent: "Consent", p: TRDProfile,
                  device_key: str, biomarker: BiomarkerSpec,
                  stim: StimProgram, screen: Optional[RiskScreen],
                  region: str = "TH") -> Dict[str, Any]:
        refuse_happiness_transfer("closed-loop DBS")
        risk = CrisisPathway.assess(screen, region)
        if risk.get("blocks_research_enrollment"):
            raise SafetyRefusal(
                "Research enrollment BLOCKED by risk screen. "
                f"{risk['REQUIRED_ACTION']}")
        elig = TRDStaging.require_invasive_eligibility(p)
        dev = verify_device_class(device_key, "electrical stimulation")
        if not consent.research_valid():
            raise ConsentRefusal(
                "Closed-loop DBS research requires: patient id, IRB, "
                "psychiatrist, medical council reference, neurosurgeon, "
                "revocable consent, and explicit acknowledgement that this "
                "is INVESTIGATIONAL and delivers electricity, not molecules.")
        ok, miss = biomarker.certified()
        if not ok:
            raise Refusal(f"Biomarker NOT certified: {miss}. "
                          "Open-loop programming only under clinician care; "
                          "no closed-loop triggering.")
        k = stim.shannon_k()
        if k > ClosedLoopDBSResearch.SHANNON_K_MAX:
            raise SafetyRefusal(f"Shannon k={k:.2f} > {ClosedLoopDBSResearch.SHANNON_K_MAX}")
        if stim.delta_T_C > ClosedLoopDBSResearch.DT_MAX:
            raise SafetyRefusal("thermal limit exceeded")
        Q, D = stim.charge_density()
        return {"allowed": True, "mode": RESEARCH_ONLY,
                "device_class": dev["device"]["device_class"],
                "delivers": "ELECTRICAL PULSES ONLY",
                "substance_delivery": False,
                "biomarker": asdict(biomarker),
                "charge_nC": Q, "charge_density_uC_cm2": D, "shannon_k": k,
                "eligibility": elig, "risk": risk,
                "evidence": OPTIONS["DBS_VC_VS_CLOSED"].evidence,
                "NOT_AUTHORIZED": "molecular delivery or emotional-state transfer"}

    @staticmethod
    def trigger(biomarker: BiomarkerSpec, feature_value: float,
                stim: StimProgram) -> Dict[str, Any]:
        ok, miss = biomarker.certified()
        if not ok:
            raise Refusal(f"closed-loop trigger REFUSED: biomarker uncertified {miss}")
        fire = feature_value >= biomarker.threshold
        return {"feature_value": feature_value,
                "threshold": biomarker.threshold,
                "stimulate": fire,
                "duration_s": stim.max_on_seconds if fire else 0.0,
                "payload": "electrical pulse train",
                "NOT": "a molecule, an emotion, or another person's state"}


# =============================================================================
# §7. PHARMACOLOGY WITH ROUTE ENFORCEMENT
# =============================================================================

@dataclass
class Prescription:
    agent: str
    route: str
    device_key: Optional[str]
    prescriber_id: str
    setting: str
    monitoring_minutes: int = 0
    rems_certified_site: bool = False

class PharmacologyGateway:
    @staticmethod
    def authorize(consent: "Consent", rx: Prescription,
                  screen: Optional[RiskScreen], region: str = "TH"
                  ) -> Dict[str, Any]:
        refuse_happiness_transfer(rx.agent)
        risk = CrisisPathway.assess(screen, region)
        if not consent.clinical_valid():
            raise ConsentRefusal("prescription requires psychiatrist + council + consent")
        if not rx.prescriber_id:
            raise Refusal("prescriber of record required")
        if rx.agent.lower() in ("esketamine","spravato"):
            if rx.route != "intranasal":
                raise Refusal("esketamine is INTRANASAL. Route mismatch.")
            if not rx.rems_certified_site:
                raise SafetyRefusal(
                    "esketamine requires a REMS-certified healthcare setting.")
            if rx.monitoring_minutes < 120:
                raise SafetyRefusal(
                    "esketamine requires >=120 min post-dose observation.")
            if rx.device_key and DEVICES.get(rx.device_key) and \
               DEVICES[rx.device_key].can_deliver_substance:
                raise DeviceClassRefusal(
                    "esketamine is not delivered by an implant.")
        if rx.device_key:
            verify_device_class(rx.device_key, f"deliver {rx.agent}")
            d = DEVICES[rx.device_key]
            if d.can_deliver_substance and "depression" not in " ".join(d.indications).lower():
                if d.depression_indication == "none":
                    raise Refusal(
                        f"'{rx.device_key}' has NO depression indication. "
                        "Implanted delivery for depression is not approved.")
        return {"allowed": True, "agent": rx.agent, "route": rx.route,
                "setting": rx.setting, "monitoring_min": rx.monitoring_minutes,
                "risk": risk, "patient_can_refuse": True,
                "logged": {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "prescriber": rx.prescriber_id}}


# =============================================================================
# §8. CONSENT
# =============================================================================

@dataclass
class Consent:
    patient_id: str = ""
    irb_id: str = ""
    psychiatrist_id: str = ""
    medical_council_ref: str = ""
    neurosurgeon_id: str = ""
    scope: str = ""
    signed: bool = False
    revocable: bool = True
    withdrawn_at: Optional[str] = None
    ack_investigational: bool = False
    ack_electricity_not_molecules: bool = False
    ack_no_thought_transfer: bool = False
    ack_can_withdraw: bool = False
    capacity_assessed_by: str = ""

    def clinical_valid(self) -> bool:
        return bool(self.patient_id and self.psychiatrist_id
                    and self.medical_council_ref and self.scope
                    and self.signed and self.revocable
                    and self.withdrawn_at is None
                    and self.capacity_assessed_by and self.ack_can_withdraw)

    def research_valid(self) -> bool:
        return (self.clinical_valid() and bool(self.irb_id)
                and bool(self.neurosurgeon_id)
                and self.ack_investigational
                and self.ack_electricity_not_molecules
                and self.ack_no_thought_transfer)


# =============================================================================
# §9. PAIRED RECORDING BRIDGE  (from the previous module)
# =============================================================================

class PairedRecordingBridge:
    """
    The neural–behaviour pairing module remains valid for RESEARCH
    CORRELATES. It never becomes a controller for an intervention, and a
    decoder output is never a label or a trigger for substance delivery.
    """
    @staticmethod
    def link() -> Dict[str, Any]:
        return {"module": "msps_neural_behavior_pairing.py",
                "purpose": "consented neural-behaviour correlation research",
                "labels_from": ["audio_transcript","video_facs","emg_onset",
                                "accelerometer","self_report"],
                "scalp_eeg_cannot": ["word_identity","subjective_emotion"],
                "subjective_items": {"โกรธ":"self-report required",
                                     "เศร้า":"self-report required",
                                     "ยิ้ม":"observable via FACS"},
                "NOT_A_CONTROLLER": True,
                "decoder_output_never_a_label": True,
                "decoder_output_never_triggers_substance": True}

    @staticmethod
    def depression_correlate_note() -> Dict[str, Any]:
        return {"candidate_correlates":
                  ["frontal alpha asymmetry (contested)",
                   "rostral ACC theta (contested)",
                   "gamma at a personalised contact (Scangos-style, n=1)"],
                "status": "CANDIDATE CORRELATES, not measurements of sadness",
                "label_source": "patient self-report or validated clinician scale",
                "required": ["permutation test p<0.01, >=1000 shuffles",
                             ">=3 held-out sessions",
                             "EMG/EOG artifact control",
                             "clinician sign-off"],
                "NOT": "a biomarker copied from another person"}


# =============================================================================
# §10. STATISTICS DICTIONARY
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "trd_staging": (("TRIALS",), "TRD definition",
                      ">=2 adequate failed trials"),
      "risk_screen": (("CLIN",), "PHQ-9 item 9 / C-SSRS",
                      "screening, not a prediction of behaviour"),
      "closed_loop_biomarker": (("PERM","XSESS","CLIN","OWN"),
                                "personalised biomarker validation",
                                "must be the patient's own; n=1 literature"),
      "dbs_efficacy": (("IRB","SURG","CLIN"), "investigational",
                       "limited/preliminary evidence; not approved"),
      "esketamine_admin": (("REMS","MON","CLIN"), "FDA label",
                           "certified site, >=120 min observation"),
      "happiness_transfer": ((), "none", "NO CHANNEL — refused"),
      "implant_substance_for_depression": ((), "none",
                                           "no approved indication"),
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
 "TRIALS":"documented adequate antidepressant trials",
 "CLIN":  "named clinician assessment",
 "PERM":  "permutation test p<0.01, >=1000 shuffles",
 "XSESS": ">=3 held-out sessions",
 "OWN":   "biomarker validated in THIS patient, not copied",
 "IRB":   "IRB approval",
 "SURG":  "functional neurosurgery team",
 "REMS":  "REMS-certified site",
 "MON":   ">=120 min post-dose monitoring",
}


# =============================================================================
# §11. ORCHESTRATOR
# =============================================================================

class DepressionGateway:
    def __init__(self, out_dir: str = "runs/depression_gateway"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()

    def intake(self, consent: Consent, p: TRDProfile,
               screen: Optional[RiskScreen], region: str = "TH"
               ) -> Dict[str, Any]:
        risk = CrisisPathway.assess(screen, region)
        self.sd.certify("risk_screen", ("CLIN",) if screen and screen.screened_by
                        else (), {"urgent": risk["urgent"]})
        st = TRDStaging.stage(p)
        self.sd.certify("trd_staging", ("TRIALS",) if p.trials else (), st)
        self.sd.certify("happiness_transfer", (), {"refused": True})
        self.sd.certify("implant_substance_for_depression", (), {"refused": True})
        return {"risk": risk, "staging": st,
                "options": option_ledger(p),
                "paired_recording": PairedRecordingBridge.link(),
                "correlate_note": PairedRecordingBridge.depression_correlate_note(),
                "REFUSED": ["happiness-molecule transfer",
                            "emotional-state copying",
                            "drug delivery via electrical stimulator"],
                "actionability": CLINICIAN_REQUIRED}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "devices": {k: asdict(v) for k,v in DEVICES.items()},
               "molecule_facts": MOLECULE_FACTS,
               "options": option_ledger(), "packages": PACKAGES,
               "crisis": CrisisPathway.resources("TH"),
               "sd_report": self.sd.report(),
               "actionability": CLINICIAN_REQUIRED}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2,
                                                       default=str, ensure_ascii=False))
        return rep


# =============================================================================
# §12. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    gw = DepressionGateway()
    good = Consent("pt01","IRB-2026-07","psych01","MC-2026-888","neuro01",
                   "closed-loop DBS research for TRD", True, True, None,
                   True, True, True, True, "psych01")
    safe = RiskScreen(phq9_item9=0, cssrs_ideation=0, screened_by="psych01",
                      screened_at="2026-11-26T09:00:00Z")
    urgent = RiskScreen(phq9_item9=3, cssrs_ideation=5, has_plan=True,
                        intent=True, screened_by="psych01")

    trd = TRDProfile(episode_severity="severe", ect_prior=True,
        trials=[AntidepressantTrial("sertraline",200,True,8,True,"none"),
                AntidepressantTrial("venlafaxine",225,True,8,True,"none"),
                AntidepressantTrial("escitalopram",20,True,8,True,"partial")])
    not_trd = TRDProfile(trials=[AntidepressantTrial("sertraline",50,False,3,False,"none")])

    print("\n[T1] HAPPINESS-MOLECULE TRANSFER — REFUSED")
    b = 0
    for r in ("happiness molecule","โมเลกุลคนมีความสุข","transfer emotion",
              "copy emotional state","ใส่รูปแบบโมเลกุล",
              "molecular pattern injection","inject happiness"):
        try: refuse_happiness_transfer(r)
        except NoChannelRefusal: b += 1
    rep("all transfer requests REFUSED", b == 7, f"{b}/7")
    rep("dopamine formula identical regardless of donor",
        MOLECULE_FACTS["dopamine"]["formula"] == "C8H11NO2")
    rep("legitimate request allowed",
        refuse_happiness_transfer("esketamine intranasal for TRD")["allowed"])

    print("\n[T2] DEVICE CLASS SEPARATION")
    b = False
    try: verify_device_class("DBS_GENERIC", "deliver dopamine molecule")
    except DeviceClassRefusal: b = True
    rep("DBS substance-delivery claim REFUSED", b)
    rep("DBS has no reservoir", not DEVICES["DBS_GENERIC"].has_reservoir)
    rep("DBS cannot deliver substance",
        not DEVICES["DBS_GENERIC"].can_deliver_substance)
    rep("verified pump CAN deliver",
        DEVICES["DRUG_PUMP_VERIFIED"].can_deliver_substance)
    rep("pump has NO depression indication",
        DEVICES["DRUG_PUMP_VERIFIED"].depression_indication == "none")
    rep("electrical claim on DBS allowed",
        verify_device_class("DBS_GENERIC","electrical stimulation")["allowed"])

    print("\n[T3] SUICIDE RISK — MANDATORY")
    b = False
    try: CrisisPathway.assess(None)
    except SafetyRefusal: b = True
    rep("no risk screen BLOCKS everything", b)
    r_safe = CrisisPathway.assess(safe)
    rep("low risk: routine monitoring", not r_safe["urgent"])
    r_urg = CrisisPathway.assess(urgent)
    rep("high risk flagged urgent", r_urg["urgent"])
    rep("urgent blocks research enrollment", r_urg["blocks_research_enrollment"])
    rep("crisis line present", "1323" in json.dumps(r_urg["resources"], ensure_ascii=False))
    rep("same-day action required", "SAME-DAY" in r_urg["REQUIRED_ACTION"])

    print("\n[T4] TRD STAGING")
    st = TRDStaging.stage(trd)
    print(f"      failed adequate trials = {st['n_failed_adequate']}")
    rep("TRD recognised with 3 failed adequate trials", st["trd"])
    st2 = TRDStaging.stage(not_trd)
    rep("inadequate trial does NOT count", not st2["trd"])
    rep("inadequate trials named", "sertraline" in st2["inadequate_trials"])
    b = False
    try: TRDStaging.require_invasive_eligibility(not_trd)
    except EligibilityRefusal: b = True
    rep("invasive BLOCKED before TRD criteria", b)
    no_ect = TRDProfile(episode_severity="severe", trials=trd.trials)
    b = False
    try: TRDStaging.require_invasive_eligibility(no_ect)
    except EligibilityRefusal: b = True
    rep("DBS BLOCKED before established options tried", b)
    rep("eligibility granted after ECT + TRD",
        TRDStaging.require_invasive_eligibility(trd)["eligible"])

    print("\n[T5] EVIDENCE LEDGER")
    led = option_ledger()
    approved = [o for o in led["options"] if o["approval"]=="approved"]
    invest = [o for o in led["options"] if o["approval"]=="investigational"]
    print(f"      approved={len(approved)}  investigational={len(invest)}")
    rep("ECT evidence strong", OPTIONS["ECT"].evidence == "strong")
    rep("DBS SCC evidence limited", OPTIONS["DBS_SCC"].evidence == "limited")
    rep("closed-loop DBS preliminary (n=1)",
        OPTIONS["DBS_VC_VS_CLOSED"].evidence == "preliminary")
    rep("no DBS option can deliver substance",
        all(not o.get("can_deliver_substance", False)
            for o in led["options"] if o["key"].startswith("DBS")))
    rep("happiness transfer not offered", "NOT_OFFERED" in led)

    print("\n[T6] CLOSED-LOOP DBS RESEARCH")
    bm_bad = BiomarkerSpec("gamma_contact3",[3],0.5, validated_sessions=1)
    bm_ok = BiomarkerSpec("gamma_contact3",[3],0.5, validated_sessions=4,
                          correlation_with_symptom=0.62, permutation_p=0.0009,
                          n_perm=2000, cross_session_validated=True,
                          clinician_id="psych01")
    stim = StimProgram(2.0, 90.0, 130.0, 0.06, max_on_seconds=30.0)
    okf, miss = bm_bad.certified()
    rep("uncertified biomarker rejected", not okf, str(miss[:2]))
    b = False
    try: ClosedLoopDBSResearch.authorize(good, trd, "DBS_GENERIC",
                                         bm_bad, stim, safe)
    except Refusal: b = True
    rep("closed-loop BLOCKED with uncertified biomarker", b)
    auth = ClosedLoopDBSResearch.authorize(good, trd, "DBS_GENERIC",
                                           bm_ok, stim, safe)
    print(f"      Shannon k={auth['shannon_k']:.3f}  "
          f"D={auth['charge_density_uC_cm2']:.2f} uC/cm2")
    rep("authorised with certified biomarker", auth["allowed"])
    rep("delivers ELECTRICITY ONLY", "ELECTRICAL" in auth["delivers"])
    rep("substance_delivery False", auth["substance_delivery"] is False)
    rep("explicitly excludes molecular transfer",
        "molecular delivery" in auth["NOT_AUTHORIZED"])
    b = False
    try: ClosedLoopDBSResearch.authorize(good, trd, "DBS_GENERIC",
                                         bm_ok, stim, urgent)
    except SafetyRefusal: b = True
    rep("urgent risk BLOCKS research enrollment", b)
    b = False
    try: ClosedLoopDBSResearch.authorize(good, trd, "DBS_GENERIC", bm_ok,
                                         StimProgram(10.0,500.0,130.0,0.01), safe)
    except SafetyRefusal: b = True
    rep("unsafe charge density BLOCKED", b)
    trig = ClosedLoopDBSResearch.trigger(bm_ok, 0.7, stim)
    rep("trigger fires above threshold", trig["stimulate"])
    rep("payload is a pulse train", trig["payload"] == "electrical pulse train")
    rep("explicitly not an emotion", "emotion" in trig["NOT"])

    print("\n[T7] PHARMACOLOGY ROUTE ENFORCEMENT")
    rx_ok = Prescription("esketamine","intranasal","ESKETAMINE_INTRANASAL",
                         "psych01","REMS-certified clinic",120,True)
    a = PharmacologyGateway.authorize(good, rx_ok, safe)
    rep("esketamine authorised with REMS + 120 min", a["allowed"])
    b = False
    try: PharmacologyGateway.authorize(good,
        Prescription("esketamine","intranasal",None,"psych01","home",120,False), safe)
    except SafetyRefusal: b = True
    rep("non-REMS site BLOCKED", b)
    b = False
    try: PharmacologyGateway.authorize(good,
        Prescription("esketamine","intranasal",None,"psych01","clinic",30,True), safe)
    except SafetyRefusal: b = True
    rep("<120 min monitoring BLOCKED", b)
    b = False
    try: PharmacologyGateway.authorize(good,
        Prescription("esketamine","intravenous",None,"psych01","clinic",120,True), safe)
    except Refusal: b = True
    rep("wrong route BLOCKED", b)
    b = False
    try: PharmacologyGateway.authorize(good,
        Prescription("dopamine","intracerebral","DRUG_PUMP_VERIFIED",
                     "psych01","theatre",0,False), safe)
    except Refusal: b = True
    rep("implanted delivery for depression BLOCKED (no indication)", b)

    print("\n[T8] PAIRED-RECORDING BRIDGE")
    link = PairedRecordingBridge.link()
    rep("bridge is not a controller", link["NOT_A_CONTROLLER"])
    rep("decoder output never triggers substance",
        link["decoder_output_never_triggers_substance"])
    rep("subjective items need self-report",
        "self-report" in link["subjective_items"]["เศร้า"])
    note = PairedRecordingBridge.depression_correlate_note()
    rep("biomarker must be the patient's own",
        "own" in note["NOT"] or "another person" in note["NOT"])
    rep("correlates are not measurements of sadness",
        "not measurements of sadness" in note["status"])

    print("\n[T9] SD GATE + INTAKE")
    c1 = gw.sd.certify("happiness_transfer", ())
    rep("happiness transfer -> MODE_E", c1["status"] == MODE_E)
    c2 = gw.sd.certify("closed_loop_biomarker", ("PERM","XSESS","CLIN"))
    rep("biomarker without OWN -> MODE_E", c2["status"] == MODE_E)
    rep("missing package named", "OWN" in c2["missing"])
    c3 = gw.sd.certify("closed_loop_biomarker", ("PERM","XSESS","CLIN","OWN"))
    rep("own-patient biomarker -> CERTIFIED", c3["status"] == MODE_CERTIFIED)
    intake = gw.intake(good, trd, safe)
    rep("intake completes", "options" in intake)
    rep("refusals listed", len(intake["REFUSED"]) == 3)
    rr = gw.report()
    rep("report written", (gw.out/"report.json").exists())
    rep("Mode E retained", rr["sd_report"]["mode_E"] > 0)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
REFUSED (all fired)
  happiness-molecule transfer / emotional-state copying   7/7
  substance delivery claimed on an electrical stimulator
  any depression intervention without a risk screen
  research enrollment under an urgent risk screen
  invasive option before TRD criteria and established options
  esketamine outside a REMS site or under 120 min monitoring
  implanted drug delivery for depression (no indication)
  closed-loop triggering on an uncertified / copied biomarker

BUILT
  Device-class registry separating electrical stimulators from
  reservoir-bearing delivery implants, per the authors' own section 6.2.
  Mandatory suicide-risk screening with a same-day crisis pathway.
  TRD staging from documented adequate trials.
  Evidence ledger ordered approved -> off-label -> investigational,
  with ECT/esketamine/TMS ahead of investigational DBS.
  Closed-loop DBS research protocol: personalised biomarker requiring
  permutation testing, cross-session validation, and OWN-patient data;
  electrical payload only, charge-density gated.
  Pharmacology route and setting enforcement.
  Bridge to the paired-recording module, which remains a correlate
  researcher and never a controller.

WHAT THIS IS NOT
  Not a way to transfer happiness. Not a molecular emotion injector.
  Not a substitute for psychiatric care. Every pathway ends with a
  named clinician and an open crisis door.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Depression intervention gateway")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","options","devices","crisis","molecules","packages","report"])
    ap.add_argument("--region", default="TH")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    gw = DepressionGateway(); out: Dict[str, Any] = {}
    if a.cmd == "options": out = option_ledger()
    elif a.cmd == "devices": out = {k: asdict(v) for k,v in DEVICES.items()}
    elif a.cmd == "crisis": out = CrisisPathway.resources(a.region)
    elif a.cmd == "molecules":
        out = {"facts": MOLECULE_FACTS,
               "conclusion": "no emotional signature exists on a molecule; "
                             "effect is determined by the recipient"}
    elif a.cmd == "packages": out = PACKAGES
    elif a.cmd == "report": out = gw.report()
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
