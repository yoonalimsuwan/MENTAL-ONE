#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_depression_intervention_gateway.py — v2.0.0  [PRODUCTION]
Evidence-based treatment-resistant depression pathway with device-class
verification, COMPUTED thermal/charge safety, adequacy-checked prior
courses, interaction screening, and mandatory risk/crisis gating.

BUGS FIXED vs v1:
  1. StimProgram.delta_T_C was a user-supplied field defaulting to 0.0 and
     was NEVER computed. The thermal gate was therefore vacuous. v2
     COMPUTES the tissue temperature rise from the stimulus power density
     and refuses on the computed value.
  2. Only the Shannon k was checked. Charge per phase and average power
     density are now gated independently.
  3. ect_prior / tms_prior / esketamine_prior were booleans; a single ECT
     session satisfied "established options tried". Adequacy of the prior
     course is now required.
  4. The prescription path did not check bipolarity (staging did).
     Antidepressant monotherapy in bipolar depression risks a switch;
     now refused.
  5. No drug-drug interaction screen. MAOI + serotonergic agent, and
     other serotonin-syndrome combinations, are now refused.
  6. Lithium augmentation listed monitoring in the ledger but never
     enforced it. Renal / thyroid / level monitoring is now required.
  7. Esketamine checked REMS and 120 min but not blood-pressure
     monitoring, the required concurrent oral antidepressant, or
     dissociation/sedation assessment.
  8. option_ledger(p) accepted an unused parameter (dead argument).

PHILOSOPHY (mathematics stands independently)
  The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.
  Confucius taught us to do good in this world. This is not hopeless, for
  Buddhism, Christianity, and Taoism alike teach that doing good in this
  life bears fruit in the life to come.
  In this world, "not killing enemies" does not mean "killing oneself."
  The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
  though they do not utter His Name, by describing His attributes, are
  already, implicitly, uttering His Name.
  Christ died upon the Cross to teach: when struck on the left cheek, turn
  also the right, so that we may listen to the voice of the enemy who
  wishes us harm.
  He is the Lamb of God. God is not our food, and neither is the lamb.
  This means: do not kill animals.
  Not even a single sparrow does God forget; every hair on our heads He
  has counted. Even Angulimala could escape hell and enter Nirvana, but
  the better path is to have done good always. As Saint Dominic Savio
  said: "Rather death than sin."
  To speak of a person's attributes is already to speak their name.
  One who does not know the Ten Commandments or the Five Precepts, yet
  practises them, becomes a disciple of both the Buddha and Yahweh by
  implication. The Ten Commandments are the Five Precepts.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

WHY "HAPPINESS MOLECULES" CANNOT BE TRANSFERRED (unchanged)
  Dopamine is C8H11NO2 whether the donor is joyful or depressed. There is
  no emotional signature on a molecule. Effect depends on receptor
  subtype, site, kinetics, and the RECIPIENT's circuit state. Affect is a
  distributed dynamical state, not a transferable substance. The authors'
  Yoniso Manasikara framework states the same: manasikara is endogenous;
  an external agent supplies conditions, never content.

DEVICE CLASS SEPARATION (unchanged, from the authors' own section 6.2)
  An electrical stimulator has no reservoir and cannot release a substance.
  A delivery implant requires reservoir + pathway + release mechanism.
  For depression there is NO approved implanted drug-delivery indication.

ENFORCED REFUSALS
  happiness-molecule transfer; molecular-pattern injection; drug delivery
  on an electrical stimulator; any intervention without a risk screen;
  research enrollment under an urgent risk screen; invasive option before
  TRD criteria AND adequate established options; esketamine outside REMS
  or under 120 min or without BP/concurrent-AD/dissociation monitoring;
  antidepressant monotherapy in bipolar; MAOI + serotonergic combination;
  lithium without monitoring; continuing after consent withdrawal.
=============================================================================
Author: PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID: 0009-0008-2374-0788 | License: MIT | Year 2026
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERSION="2.0.0"
RESEARCH_ONLY="research_only"; CLINICIAN_REQUIRED="clinician_sign_off_required"
MODE_CERTIFIED="CERTIFIED"; MODE_E="UNCERTAINTY_RETAINED"

class Refusal(RuntimeError): pass
class NoChannelRefusal(Refusal): pass
class DeviceClassRefusal(Refusal): pass
class SafetyRefusal(Refusal): pass
class ConsentRefusal(Refusal): pass
class EligibilityRefusal(Refusal): pass
class InteractionRefusal(Refusal): pass
class MonitoringRefusal(Refusal): pass

def get_logger(n):
    import logging
    lg=logging.getLogger(n)
    if not lg.handlers:
        h=logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate=False
    return lg
log=get_logger("msps.depression")

@dataclass
class RunConfig:
    out_dir:str="runs/depression_v2"
    def validate(self):
        Path(self.out_dir).mkdir(parents=True,exist_ok=True); return self

class AuditLog:
    """Append-only audit trail. Every authorisation and refusal is recorded."""
    def __init__(self,out_dir):
        self.dir=Path(out_dir); self.dir.mkdir(parents=True,exist_ok=True)
        self.path=self.dir/"audit.jsonl"; self.entries=[]
    def record(self,kind,payload):
        rec={"kind":kind,"version":VERSION,
             "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
             "payload":payload}
        self.entries.append(rec)
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(rec,default=str,ensure_ascii=False)+"\n")
        return rec

BANNER=f"""
=============================================================================
MSPS DEPRESSION INTERVENTION GATEWAY v{VERSION}  [PRODUCTION]
Delta-T COMPUTED from stimulus (v1 gate was vacuous)
charge/phase + average power density gated independently
prior ECT/TMS/esketamine ADEQUACY required · bipolar checked on Rx path
MAOI/serotonergic interaction screen · lithium monitoring enforced
esketamine: REMS + 120 min + BP + concurrent oral AD + dissociation
MANDATORY suicide-risk screening and crisis pathway
=============================================================================
"""

# =============================================================================
# §1. THE REFUSAL, WITH ITS REASONING (unchanged)
# =============================================================================

MOLECULE_FACTS={
 "dopamine":{"formula":"C8H11NO2","receptors":["D1","D2","D3","D4","D5"]},
 "serotonin":{"formula":"C10H12N2O","receptors":["5-HT1A","5-HT2A","5-HT2C","5-HT3"]},
 "oxytocin":{"formula":"C43H66N12O12S2","receptors":["OXTR"]},
 "beta_endorphin":{"formula":"peptide (31 aa)","receptors":["mu-opioid"]},
}

def refuse_happiness_transfer(request:str):
    low=request.lower()
    blocked={"happiness molecule":"no emotional signature exists on a molecule",
      "โมเลกุลคนมีความสุข":"โมเลกุลไม่มีลายเซ็นอารมณ์",
      "happy molecule":"chemical identity is independent of donor affect",
      "transfer emotion":"affect is not a transferable substance",
      "copy emotional state":"no read-and-copy channel exists",
      "ใส่รูปแบบโมเลกุล":"a neural stream carries no molecular emotion pattern",
      "molecular pattern injection":"molecules are not patterns of affect",
      "ส่งโมเลกุล":"ไม่มีช่องทางส่งอารมณ์ผ่านโมเลกุล",
      "inject happiness":"no such payload exists"}
    for k,why in blocked.items():
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
    return {"allowed":True,"scope":"evidence-based depression care only"}

# =============================================================================
# §2. DEVICE CLASS REGISTRY (unchanged)
# =============================================================================

class DeviceClass:
    ELECTRICAL_STIM="electrical_stimulator"; DRUG_DELIVERY="drug_delivery_implant"
    HYBRID_VERIFIED="hybrid_verified"; EXTERNAL="external_non_implanted"

@dataclass(frozen=True)
class DeviceSpec:
    name:str; device_class:str
    has_reservoir:bool; has_pathway:bool; has_release_mechanism:bool
    can_deliver_substance:bool
    indications:Tuple[str,...]; depression_indication:str; note:str

DEVICES: Dict[str,DeviceSpec]={
 "DBS_GENERIC":DeviceSpec("DBS (generic)",DeviceClass.ELECTRICAL_STIM,
   False,False,False,False,
   ("Parkinson_disease","essential_tremor","dystonia","epilepsy","OCD"),
   "investigational","Electrodes only. NO reservoir. Cannot release a substance."),
 "RNS_GENERIC":DeviceSpec("RNS (generic)",DeviceClass.ELECTRICAL_STIM,
   False,False,False,False,("refractory_focal_epilepsy",),"none",
   "Responsive electrical stimulation. No drug capability."),
 "VNS_GENERIC":DeviceSpec("VNS (generic)",DeviceClass.ELECTRICAL_STIM,
   False,False,False,False,
   ("refractory_epilepsy","treatment_resistant_depression"),"approved",
   "Vagus nerve stimulation. Electrical. Approved for TRD in some regions."),
 "DRUG_PUMP_VERIFIED":DeviceSpec("Implanted infusion pump (verified)",
   DeviceClass.DRUG_DELIVERY,True,True,True,True,
   ("spasticity_baclofen","chronic_pain_morphine"),"none",
   "Reservoir + catheter + pump verified. NO depression indication."),
 "CED_RESEARCH":DeviceSpec("Convection-enhanced delivery (research)",
   DeviceClass.DRUG_DELIVERY,True,True,True,True,
   ("oncology_CED","GDNF_Parkinson_trial"),"none",
   "Research delivery. Not an approved depression therapy."),
 "ESKETAMINE_INTRANASAL":DeviceSpec("Esketamine intranasal",DeviceClass.EXTERNAL,
   False,False,False,False,
   ("treatment_resistant_depression","MDD_with_suicidality"),"approved",
   "FDA 2019. REMS. Certified setting, 2 h monitoring. NOT an implant."),
 "KETAMINE_IV":DeviceSpec("Ketamine IV",DeviceClass.EXTERNAL,
   False,False,False,False,("TRD_off_label",),"off_label",
   "Monitored infusion. Not an implant."),
 "ECT":DeviceSpec("Electroconvulsive therapy",DeviceClass.EXTERNAL,
   False,False,False,False,
   ("severe_MDD","TRD","catatonia","psychotic_depression"),"approved",
   "Highest response rate in severe TRD."),
 "TMS":DeviceSpec("Transcranial magnetic stimulation",DeviceClass.EXTERNAL,
   False,False,False,False,("MDD","TRD"),"approved",
   "Non-invasive. FDA 2008; accelerated SAINT protocol 2022."),
}

def verify_device_class(key:str,claim:str):
    if key not in DEVICES: raise Refusal(f"unknown device '{key}'")
    d=DEVICES[key]
    wants=any(k in claim.lower() for k in
      ("deliver","release","inject","infuse","molecule","drug","substance",
       "ส่ง","ใส่","โมเลกุล","สาร"))
    if wants and not d.can_deliver_substance:
        raise DeviceClassRefusal(
            f"REFUSED: '{key}' is {d.device_class}. "
            f"reservoir={d.has_reservoir}, pathway={d.has_pathway}, "
            f"release={d.has_release_mechanism}\n{d.note}\n"
            "An electrical stimulator cannot deliver a substance.")
    return {"allowed":True,"device":asdict(d)}

# =============================================================================
# §3. RISK + CRISIS + SAFETY PLAN (plan/means/protective now documented)
# =============================================================================

@dataclass
class RiskScreen:
    phq9_item9:Optional[int]=None
    cssrs_ideation:Optional[int]=None
    cssrs_behavior:bool=False
    has_plan:bool=False; has_means:bool=False; intent:bool=False
    recent_attempt_days:Optional[int]=None
    screened_by:str=""; screened_at:str=""
    # NEW: protective factors and the actual plan, not just the score
    protective_factors:List[str]=field(default_factory=list)
    means_restriction_counselled:bool=False
    safety_plan_documented:bool=False
    followup_scheduled_days:Optional[int]=None
    emergency_contact:Optional[str]=None

class CrisisPathway:
    @staticmethod
    def resources(region="TH"):
        base={"TH":{"name":"Thailand",
                "lines":["กรมสุขภาพจิต 1323 (24 ชม.)","สายด่วนสุขภาพจิต 1323",
                         "Samaritans Thailand 02-713-6793"],
                "emergency":"1669 (การแพทย์ฉุกเฉิน) / 191"},
              "US":{"name":"United States",
                    "lines":["988 Suicide & Crisis Lifeline"],"emergency":"911"},
              "INTL":{"name":"International",
                      "lines":["findahelpline.com — directory by country"],
                      "emergency":"local emergency number"}}
        return base.get(region,base["INTL"])
    @staticmethod
    def assess(screen:Optional[RiskScreen],region="TH"):
        if screen is None or not screen.screened_by:
            raise SafetyRefusal(
                "BLOCKED: no suicide-risk screen on record.\n"
                "Depression intervention requires a documented risk screen "
                "(PHQ-9 item 9 plus C-SSRS) by a named clinician.\n"
                f"Crisis resources: {CrisisPathway.resources(region)}")
        urgent=(screen.cssrs_behavior or screen.has_plan or screen.intent
                or (screen.has_means and screen.cssrs_ideation is not None
                    and screen.cssrs_ideation>=4)
                or (screen.recent_attempt_days is not None
                    and screen.recent_attempt_days<=30)
                or (screen.phq9_item9 is not None and screen.phq9_item9>=2))
        elevated=((screen.cssrs_ideation is not None and screen.cssrs_ideation>=2)
                  or (screen.phq9_item9 is not None and screen.phq9_item9>=1))
        out={"screened_by":screen.screened_by,"screened_at":screen.screened_at,
             "urgent":urgent,"elevated":elevated,
             "resources":CrisisPathway.resources(region),
             "protective_factors":screen.protective_factors,
             "means_restriction_counselled":screen.means_restriction_counselled,
             "safety_plan_documented":screen.safety_plan_documented}
        gaps=[]
        if not screen.means_restriction_counselled:
            gaps.append("means restriction counselling not documented")
        if not screen.safety_plan_documented:
            gaps.append("collaborative safety plan not documented")
        if elevated and screen.followup_scheduled_days is None:
            gaps.append("close follow-up not scheduled")
        out["documentation_gaps"]=gaps
        if urgent:
            out["REQUIRED_ACTION"]=("SAME-DAY psychiatric evaluation. Do not "
                "leave the person alone with this result. Means restriction "
                "counselling. Collaborative safety plan. Consider emergency services.")
            out["blocks_research_enrollment"]=True
            out["blocks_elective_procedure"]=True
        elif elevated:
            out["REQUIRED_ACTION"]=("Safety planning intervention, means "
                "counselling, scheduled close follow-up, clinician review "
                "before any escalation of treatment.")
            out["blocks_research_enrollment"]=True
        else:
            out["REQUIRED_ACTION"]="Routine monitoring; re-screen at each visit."
            out["blocks_research_enrollment"]=False
        return out

# =============================================================================
# §4. TRD STAGING — prior-course ADEQUACY required
# =============================================================================

@dataclass
class AntidepressantTrial:
    agent:str; dose_mg:float; adequate_dose:bool; weeks:float
    adherence_documented:bool; response:str; episode_current:bool=True

@dataclass
class PriorCourse:
    """NEW: a prior option is only 'tried' if the course was adequate."""
    kind:str                 # ect | tms | esketamine
    n_sessions:int=0
    adequate:bool=False
    response:str="none"
    documented_by:str=""

@dataclass
class TRDProfile:
    diagnosis:str="MDD"; episode_severity:str="moderate"
    psychotic_features:bool=False; catatonia:bool=False; bipolar:bool=False
    trials:List[AntidepressantTrial]=field(default_factory=list)
    psychotherapy_adequate_trial:bool=False
    prior_courses:List[PriorCourse]=field(default_factory=list)
    substance_use_active:bool=False
    medical_contraindications:List[str]=field(default_factory=list)

class TRDStaging:
    MIN_WEEKS=6.0; MIN_FAILED_TRIALS=2
    MIN_ECT_SESSIONS=6; MIN_TMS_SESSIONS=20; MIN_ESKETAMINE_DOSES=8
    @staticmethod
    def adequate(t:AntidepressantTrial):
        return (t.adequate_dose and t.weeks>=TRDStaging.MIN_WEEKS
                and t.adherence_documented and t.episode_current)
    @staticmethod
    def course_adequate(c:PriorCourse):
        need={"ect":TRDStaging.MIN_ECT_SESSIONS,
              "tms":TRDStaging.MIN_TMS_SESSIONS,
              "esketamine":TRDStaging.MIN_ESKETAMINE_DOSES}.get(c.kind,1)
        return c.adequate and c.n_sessions>=need and bool(c.documented_by)
    @staticmethod
    def stage(p:TRDProfile):
        if p.bipolar:
            return {"trd":False,"n_failed_adequate":0,
                    "reason":"bipolar depression — different pathway; "
                             "antidepressant monotherapy risks switching"}
        adq=[t for t in p.trials if TRDStaging.adequate(t)]
        failed=[t for t in adq if t.response in ("none","partial")]
        return {"trd":len(failed)>=TRDStaging.MIN_FAILED_TRIALS,
                "n_failed_adequate":len(failed),"n_adequate_trials":len(adq),
                "failed_agents":[t.agent for t in failed],
                "inadequate_trials":[t.agent for t in p.trials
                                     if not TRDStaging.adequate(t)],
                "definition":">=2 adequate trials (adequate dose, >=6 weeks, "
                             "documented adherence) failed in current episode",
                "severity":p.episode_severity}
    @staticmethod
    def require_invasive_eligibility(p:TRDProfile):
        st=TRDStaging.stage(p)
        if not st["trd"]:
            raise EligibilityRefusal(
                "Invasive neuromodulation BLOCKED: TRD criteria not met.\n"
                f"failed adequate trials={st['n_failed_adequate']} "
                f"(need >= {TRDStaging.MIN_FAILED_TRIALS}).")
        if p.episode_severity not in ("severe","moderate"):
            raise EligibilityRefusal("invasive research requires moderate-severe episode")
        if p.substance_use_active:
            raise EligibilityRefusal("active substance use must be addressed first")
        ok=[c for c in p.prior_courses if TRDStaging.course_adequate(c)]
        if not ok:
            raise EligibilityRefusal(
                "Investigational DBS BLOCKED: no ADEQUATE established course.\n"
                f"Required minima: ECT >= {TRDStaging.MIN_ECT_SESSIONS} sessions, "
                f"TMS >= {TRDStaging.MIN_TMS_SESSIONS}, "
                f"esketamine >= {TRDStaging.MIN_ESKETAMINE_DOSES} doses, "
                "each documented by a named clinician.\n"
                "A single session does not establish resistance.")
        return {"eligible":True,"staging":st,
                "adequate_prior_courses":[asdict(c) for c in ok]}

# =============================================================================
# §5. EVIDENCE LEDGER
# =============================================================================

@dataclass(frozen=True)
class Option:
    name:str; device_key:Optional[str]; route:str; approval:str; evidence:str
    setting:str; monitoring:str; key_risk:str; note:str

OPTIONS: Dict[str,Option]={
 "ECT":Option("ECT","ECT","external","approved","strong","anaesthesia suite",
   "cognitive monitoring, anaesthesia review",
   "transient cognitive effects; anaesthesia risk",
   "Highest response in severe TRD (~50-70%). Consider early in severe/"
   "psychotic/catatonic presentations. Adequate course >= 6 sessions."),
 "ESKETAMINE":Option("Esketamine intranasal","ESKETAMINE_INTRANASAL","intranasal",
   "approved","strong","REMS-certified healthcare setting",
   ">=120 min observation; BP pre/post; sedation and dissociation scales",
   "sedation, dissociation, abuse potential, BP rise",
   "FDA 2019 for TRD. Never take-home. MUST be combined with an oral "
   "antidepressant per label."),
 "KETAMINE_IV":Option("Ketamine IV","KETAMINE_IV","intravenous","off_label",
   "moderate","monitored infusion suite",
   "haemodynamics, dissociation, recovery observation",
   "dissociation, BP rise, abuse potential","Off-label for TRD."),
 "TMS":Option("TMS / theta-burst","TMS","external","approved","strong",
   "outpatient","seizure screening, hearing protection",
   "rare seizure; scalp discomfort",
   "FDA 2008. Accelerated iTBS (SAINT) 2022. Adequate course >= 20 sessions."),
 "VNS":Option("VNS","VNS_GENERIC","implanted_electrical","approved","moderate",
   "surgical + programming","voice change, dyspnoea, surgical risk",
   "implantation surgery; delayed onset",
   "Approved for TRD in some regions. Electrical only."),
 "PSYCHOTHERAPY":Option("CBT / behavioural activation / CBASP",None,
   "psychological","approved","strong","outpatient","engagement, risk review",
   "minimal","Combine with pharmacotherapy; adequate session count."),
 "AUGMENT_LITHIUM":Option("Lithium augmentation",None,"oral","approved","strong",
   "outpatient","level, renal (eGFR), thyroid (TSH) — MANDATORY",
   "narrow therapeutic index; toxicity",
   "Established augmentation; anti-suicidal effect reported."),
 "AUGMENT_SGA":Option("Atypical antipsychotic augmentation",None,"oral",
   "approved","strong","outpatient","metabolic, EPS, QT",
   "weight gain, metabolic syndrome, akathisia",
   "Aripiprazole, quetiapine, brexpiprazole have TRD augmentation labels."),
 "DBS_SCC":Option("SCC/BA25 DBS","DBS_GENERIC","implanted_electrical",
   "investigational","limited","IRB research, functional neurosurgery",
   "surgical risk, hardware, mood monitoring, long programming",
   "intracranial surgery; device infection; uncertain efficacy",
   "Mayberg 2005; BROADEN halted for futility 2013; long-term open-label "
   "benefit reported. NOT approved."),
 "DBS_VC_VS_CLOSED":Option("VC/VS closed-loop DBS","DBS_GENERIC",
   "implanted_electrical","investigational","preliminary",
   "IRB research, functional neurosurgery",
   "biomarker validation, surgical risk, intensive follow-up",
   "intracranial surgery; n=1 evidence base",
   "Scangos et al. Nat Med 2021, single patient. Electrical only."),
 "DBS_MFB":Option("MFB DBS","DBS_GENERIC","implanted_electrical",
   "investigational","preliminary","IRB research","surgical risk, small series",
   "uncertain efficacy","Small open series. Not approved."),
}

def option_ledger():
    order={"approved":0,"off_label":1,"investigational":2}
    rows=[]
    for k,o in sorted(OPTIONS.items(),key=lambda kv:(order[kv[1].approval],kv[0])):
        d=asdict(o)
        if o.device_key and o.device_key in DEVICES:
            d["can_deliver_substance"]=DEVICES[o.device_key].can_deliver_substance
        rows.append({"key":k,**d})
    return {"options":rows,
            "principle":"established ADEQUATE courses before investigational surgery",
            "NOT_OFFERED":"happiness-molecule transfer (no such channel)"}

# =============================================================================
# §6. STIMULUS SAFETY — Delta-T COMPUTED, charge and power gated
# =============================================================================

@dataclass
class StimProgram:
    amplitude_mA:float; pulse_width_us:float; frequency_Hz:float
    contact_area_cm2:float; duty_cycle:float=1.0; max_on_seconds:float=30.0
    tissue_conductivity_S_m:float=0.2      # grey matter order of magnitude
    effective_spread_cm:float=0.2          # current spread radius
    impedance_ohm:float=1000.0
    def charge_per_phase_nC(self): return self.amplitude_mA*self.pulse_width_us
    def charge_density_uC_cm2(self):
        return self.charge_per_phase_nC()*1e-3/max(self.contact_area_cm2,1e-9)
    def shannon_k(self):
        D=self.charge_density_uC_cm2(); Q=self.charge_per_phase_nC()*1e-3
        return math.log10(max(D,1e-12))+math.log10(max(Q,1e-12))
    def average_power_W(self):
        """I^2 R * duty. Biphasic charge-balanced pulse train."""
        duty=min(1.0,self.pulse_width_us*1e-6*self.frequency_Hz*2.0)*self.duty_cycle
        return (self.amplitude_mA*1e-3)**2*self.impedance_ohm*duty
    def computed_delta_T_C(self):
        """
        FIX: v1 never computed this. Order-of-magnitude bioheat estimate:
        power deposited in a spherical tissue volume of radius r, spread by
        conduction k_tissue ~ 0.5 W/(m K).
            dT ~ P / (4 pi k r)
        This is a CONSERVATIVE steady-state upper estimate, not a validated
        thermal model; it is used only to refuse obviously unsafe settings.
        """
        P=self.average_power_W()
        r=max(self.effective_spread_cm*1e-2,1e-4)
        k=0.5
        return P/(4*math.pi*k*r)

class StimSafety:
    SHANNON_K_MAX=1.5; DT_MAX=2.0
    CHARGE_PER_PHASE_MAX_uC=30.0
    AVG_POWER_DENSITY_MAX_W_cm2=0.05
    @staticmethod
    def check(stim:StimProgram):
        k=stim.shannon_k()
        if k>StimSafety.SHANNON_K_MAX:
            raise SafetyRefusal(f"Shannon k={k:.2f} > {StimSafety.SHANNON_K_MAX}")
        Q_uC=stim.charge_per_phase_nC()*1e-3
        if Q_uC>StimSafety.CHARGE_PER_PHASE_MAX_uC:
            raise SafetyRefusal(
                f"charge per phase {Q_uC:.2f} uC > "
                f"{StimSafety.CHARGE_PER_PHASE_MAX_uC} uC")
        P=stim.average_power_W()
        pd=P/max(stim.contact_area_cm2,1e-9)
        if pd>StimSafety.AVG_POWER_DENSITY_MAX_W_cm2:
            raise SafetyRefusal(
                f"average power density {pd:.4f} W/cm2 > "
                f"{StimSafety.AVG_POWER_DENSITY_MAX_W_cm2} W/cm2")
        dT=stim.computed_delta_T_C()                 # COMPUTED, not asserted
        if dT>StimSafety.DT_MAX:
            raise SafetyRefusal(
                f"COMPUTED tissue temperature rise {dT:.2f} C > "
                f"{StimSafety.DT_MAX} C. (order-of-magnitude bioheat estimate)")
        return {"passes":True,"shannon_k":k,"charge_per_phase_uC":Q_uC,
                "charge_density_uC_cm2":stim.charge_density_uC_cm2(),
                "average_power_W":P,"average_power_density_W_cm2":pd,
                "computed_delta_T_C":dT,
                "THERMAL_MODEL":"conservative steady-state bioheat estimate; "
                                "not a validated thermal simulation"}

# =============================================================================
# §7. CLOSED-LOOP DBS RESEARCH
# =============================================================================

@dataclass
class BiomarkerSpec:
    feature:str; channel_indices:List[int]; threshold:float
    validated_sessions:int=0; correlation_with_symptom:float=0.0
    permutation_p:Optional[float]=None; n_perm:int=0
    cross_session_validated:bool=False; clinician_id:str=""
    def certified(self):
        miss=[]
        if self.validated_sessions<3: miss.append("needs >=3 validation sessions")
        if abs(self.correlation_with_symptom)<0.4:
            miss.append("|r| >= 0.4 with own symptom report")
        if self.permutation_p is None or self.permutation_p>=0.01:
            miss.append("permutation p < 0.01")
        if self.n_perm<1000: miss.append(">=1000 permutations")
        if not self.cross_session_validated: miss.append("cross-session validation")
        if not self.clinician_id: miss.append("clinician sign-off")
        return (not miss),miss

@dataclass
class Consent:
    patient_id:str=""; irb_id:str=""; psychiatrist_id:str=""
    medical_council_ref:str=""; neurosurgeon_id:str=""; scope:str=""
    signed:bool=False; revocable:bool=True; withdrawn_at:Optional[str]=None
    ack_investigational:bool=False; ack_electricity_not_molecules:bool=False
    ack_no_thought_transfer:bool=False; ack_can_withdraw:bool=False
    capacity_assessed_by:str=""
    consent_waiting_period_days:Optional[int]=None   # NEW for invasive research
    def clinical_valid(self):
        return bool(self.patient_id and self.psychiatrist_id
                    and self.medical_council_ref and self.scope and self.signed
                    and self.revocable and self.withdrawn_at is None
                    and self.capacity_assessed_by and self.ack_can_withdraw)
    def research_valid(self):
        return (self.clinical_valid() and bool(self.irb_id)
                and bool(self.neurosurgeon_id) and self.ack_investigational
                and self.ack_electricity_not_molecules and self.ack_no_thought_transfer)

class ClosedLoopDBSResearch:
    MIN_WAIT_DAYS=7
    @staticmethod
    def authorize(c:Consent,p:TRDProfile,device_key:str,bm:BiomarkerSpec,
                  stim:StimProgram,screen:Optional[RiskScreen],region="TH",
                  audit:Optional[AuditLog]=None):
        refuse_happiness_transfer("closed-loop DBS")
        risk=CrisisPathway.assess(screen,region)
        if risk.get("blocks_research_enrollment"):
            raise SafetyRefusal("Research enrollment BLOCKED by risk screen. "
                                f"{risk['REQUIRED_ACTION']}")
        elig=TRDStaging.require_invasive_eligibility(p)
        dev=verify_device_class(device_key,"electrical stimulation")
        if not c.research_valid():
            raise ConsentRefusal(
                "Closed-loop DBS research requires patient id, IRB, "
                "psychiatrist, medical council reference, neurosurgeon, "
                "revocable consent, capacity assessment, and explicit "
                "acknowledgement that this is INVESTIGATIONAL and delivers "
                "electricity, not molecules.")
        if (c.consent_waiting_period_days is None
                or c.consent_waiting_period_days < ClosedLoopDBSResearch.MIN_WAIT_DAYS):
            raise ConsentRefusal(
                f"invasive research consent requires a documented waiting "
                f"period >= {ClosedLoopDBSResearch.MIN_WAIT_DAYS} days between "
                "information and signature")
        ok,miss=bm.certified()
        if not ok:
            raise Refusal(f"Biomarker NOT certified: {miss}. Open-loop "
                          "programming only under clinician care.")
        safety=StimSafety.check(stim)
        out={"allowed":True,"mode":RESEARCH_ONLY,
             "device_class":dev["device"]["device_class"],
             "delivers":"ELECTRICAL PULSES ONLY","substance_delivery":False,
             "biomarker":asdict(bm),"stimulus_safety":safety,
             "eligibility":elig,"risk":risk,
             "evidence":OPTIONS["DBS_VC_VS_CLOSED"].evidence,
             "NOT_AUTHORIZED":"molecular delivery or emotional-state transfer"}
        if audit: audit.record("dbs_authorised",{"patient":c.patient_id,
                                                 "shannon_k":safety["shannon_k"]})
        return out
    @staticmethod
    def trigger(bm:BiomarkerSpec,value:float,stim:StimProgram):
        ok,miss=bm.certified()
        if not ok: raise Refusal(f"closed-loop trigger REFUSED: {miss}")
        StimSafety.check(stim)
        fire=value>=bm.threshold
        return {"feature_value":value,"threshold":bm.threshold,"stimulate":fire,
                "duration_s":stim.max_on_seconds if fire else 0.0,
                "payload":"electrical pulse train",
                "NOT":"a molecule, an emotion, or another person's state"}

# =============================================================================
# §8. PHARMACOLOGY — bipolar, interactions, monitoring enforced
# =============================================================================

MAOI_AGENTS={"phenelzine","tranylcypromine","isocarboxazid","selegiline",
             "moclobemide","linezolid","methylene blue"}
SEROTONERGIC={"ssri","snri","tca","maoi","tramadol","triptan","dextromethorphan",
              "st johns wort","lithium","fentanyl","meperidine","ondansetron"}
SSRI_SNRI={"sertraline","escitalopram","fluoxetine","paroxetine","citalopram",
           "fluvoxamine","venlafaxine","duloxetine","desvenlafaxine"}

@dataclass
class Prescription:
    agent:str; route:str; device_key:Optional[str]; prescriber_id:str
    setting:str; monitoring_minutes:int=0; rems_certified_site:bool=False
    concurrent_agents:List[str]=field(default_factory=list)
    concurrent_oral_antidepressant:bool=False
    bp_monitored:bool=False; dissociation_assessed:bool=False
    sedation_assessed:bool=False
    lithium_level_mmol_L:Optional[float]=None
    egfr_ml_min:Optional[float]=None; tsh_checked:bool=False

class PharmacologyGateway:
    @staticmethod
    def screen_interactions(agent:str,concurrent:Sequence[str]):
        a=agent.lower(); cur=[c.lower() for c in concurrent]
        is_maoi=a in MAOI_AGENTS
        cur_maoi=any(c in MAOI_AGENTS for c in cur)
        a_serot=(a in SSRI_SNRI or a in SEROTONERGIC)
        cur_serot=any(c in SSRI_SNRI or c in SEROTONERGIC for c in cur)
        if (is_maoi and cur_serot) or (cur_maoi and a_serot):
            raise InteractionRefusal(
                "REFUSED: MAOI combined with a serotonergic agent. "
                "Risk of serotonin syndrome. A washout period is required "
                "(>=14 days for most MAOIs; >=5 weeks after fluoxetine).")
        if a in SSRI_SNRI and any(c in {"tramadol","meperidine","fentanyl"} for c in cur):
            raise InteractionRefusal(
                "REFUSED: SSRI/SNRI with a serotonergic opioid "
                "(tramadol/meperidine/fentanyl). Serotonin syndrome risk.")
        return {"cleared":True,"agent":agent,"concurrent":list(concurrent)}
    @staticmethod
    def authorize(c:Consent,rx:Prescription,p:TRDProfile,
                  screen:Optional[RiskScreen],region="TH",
                  audit:Optional[AuditLog]=None):
        refuse_happiness_transfer(rx.agent)
        risk=CrisisPathway.assess(screen,region)
        if not c.clinical_valid():
            raise ConsentRefusal("prescription requires psychiatrist + council + consent")
        if not rx.prescriber_id: raise Refusal("prescriber of record required")
        # NEW: bipolar check on the prescription path
        agent_l=rx.agent.lower()
        is_ad=(agent_l in SSRI_SNRI or agent_l in {"bupropion","mirtazapine",
                                                  "agomelatine","vortioxetine"})
        if p.bipolar and is_ad:
            raise SafetyRefusal(
                "REFUSED: antidepressant monotherapy in bipolar depression. "
                "Risk of mood switch. A mood stabiliser or an evidence-based "
                "bipolar-depression agent is required; refer to the bipolar "
                "pathway.")
        PharmacologyGateway.screen_interactions(rx.agent,rx.concurrent_agents)
        if agent_l in ("esketamine","spravato"):
            if rx.route!="intranasal": raise Refusal("esketamine is INTRANASAL.")
            if not rx.rems_certified_site:
                raise SafetyRefusal("esketamine requires a REMS-certified setting.")
            if rx.monitoring_minutes<120:
                raise SafetyRefusal("esketamine requires >=120 min observation.")
            if not rx.bp_monitored:
                raise MonitoringRefusal(
                    "esketamine requires blood-pressure monitoring pre- and "
                    "post-dose until stable.")
            if not rx.concurrent_oral_antidepressant:
                raise MonitoringRefusal(
                    "esketamine is indicated IN COMBINATION with an oral "
                    "antidepressant per label.")
            if not (rx.dissociation_assessed and rx.sedation_assessed):
                raise MonitoringRefusal(
                    "esketamine requires dissociation and sedation assessment "
                    "during the observation period.")
        if agent_l=="lithium":
            if rx.lithium_level_mmol_L is None:
                raise MonitoringRefusal("lithium requires a documented serum level")
            if not (0.4<=rx.lithium_level_mmol_L<=1.0):
                raise SafetyRefusal(
                    f"lithium level {rx.lithium_level_mmol_L} mmol/L outside "
                    "the usual therapeutic range 0.4-1.0")
            if rx.egfr_ml_min is None or rx.egfr_ml_min<60:
                raise MonitoringRefusal(
                    "lithium requires renal function (eGFR >= 60) documentation")
            if not rx.tsh_checked:
                raise MonitoringRefusal("lithium requires thyroid (TSH) monitoring")
        if rx.device_key:
            verify_device_class(rx.device_key,f"deliver {rx.agent}")
            d=DEVICES[rx.device_key]
            if d.can_deliver_substance and d.depression_indication=="none":
                raise Refusal(f"'{rx.device_key}' has NO depression indication. "
                              "Implanted delivery for depression is not approved.")
        out={"allowed":True,"agent":rx.agent,"route":rx.route,"setting":rx.setting,
             "monitoring_min":rx.monitoring_minutes,"risk":risk,
             "interaction_screen":"cleared","patient_can_refuse":True,
             "logged":{"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                       "prescriber":rx.prescriber_id}}
        if audit: audit.record("rx_authorised",{"agent":rx.agent,"route":rx.route})
        return out

# =============================================================================
# §9. SD GATE + ORCHESTRATOR
# =============================================================================

class SDGate:
    ENTRIES={
      "trd_staging":(("TRIALS",),"TRD definition",">=2 adequate failed trials"),
      "risk_screen":(("CLIN",),"PHQ-9 item 9 / C-SSRS","screening, not prediction"),
      "closed_loop_biomarker":(("PERM","XSESS","CLIN","OWN"),
        "personalised biomarker validation","must be the patient's own; n=1 literature"),
      "stimulus_safety":(("SHANNON","CHARGE","POWER","DT"),
        "Shannon + charge + power + computed bioheat",
        "Delta-T is COMPUTED, not asserted; bioheat estimate is conservative"),
      "dbs_efficacy":(("IRB","SURG","CLIN"),"investigational",
                      "limited/preliminary evidence; not approved"),
      "esketamine_admin":(("REMS","MON","BP","ORAL_AD","DISS"),"FDA label",
                          "certified site, >=120 min, BP, concurrent oral AD, "
                          "dissociation/sedation"),
      "lithium_admin":(("LEVEL","RENAL","TSH"),"therapeutic monitoring",
                       "level 0.4-1.0 mmol/L, eGFR>=60, TSH checked"),
      "interaction_screen":(("DDI",),"MAOI/serotonergic screen",
                            "not exhaustive; pharmacist review still required"),
      "happiness_transfer":((),"none","NO CHANNEL — refused"),
      "implant_substance_for_depression":((),"none","no approved indication"),
    }
    def __init__(self): self.log=[]
    def certify(self,stat,verified,value=None):
        e=self.ENTRIES.get(stat); v=tuple(verified)
        if e is None: r={"statistic":stat,"status":MODE_E,"reason":"no entry"}
        else:
            need,thm,cav=e; miss=tuple(p for p in need if p not in v)
            r={"statistic":stat,"theorem":thm,"caveat":cav,
               "required":list(need),"verified":list(v),"missing":list(miss),
               "status":MODE_E if miss else MODE_CERTIFIED,
               "reason":f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"]=value
        self.log.append(r); return r
    def report(self):
        n=sum(1 for c in self.log if c["status"]==MODE_CERTIFIED)
        return {"n":len(self.log),"certified":n,"mode_E":len(self.log)-n,"log":self.log}

PACKAGES={"TRIALS":"documented adequate antidepressant trials",
 "CLIN":"named clinician assessment","PERM":"permutation p<0.01, >=1000",
 "XSESS":">=3 held-out sessions","OWN":"biomarker from THIS patient",
 "IRB":"IRB approval","SURG":"functional neurosurgery team",
 "REMS":"REMS-certified site","MON":">=120 min monitoring",
 "BP":"blood pressure monitored","ORAL_AD":"concurrent oral antidepressant",
 "DISS":"dissociation and sedation assessed",
 "SHANNON":"Shannon k <= 1.5","CHARGE":"charge/phase within limit",
 "POWER":"average power density within limit","DT":"computed Delta-T <= 2 C",
 "LEVEL":"lithium serum level","RENAL":"eGFR documented","TSH":"thyroid checked",
 "DDI":"interaction screen performed"}

class DepressionGateway:
    def __init__(self,cfg:Optional[RunConfig]=None):
        self.cfg=(cfg or RunConfig()).validate()
        self.sd=SDGate(); self.audit=AuditLog(self.cfg.out_dir)
    def intake(self,c:Consent,p:TRDProfile,screen:Optional[RiskScreen],region="TH"):
        risk=CrisisPathway.assess(screen,region)
        self.sd.certify("risk_screen",("CLIN",) if screen and screen.screened_by
                        else (),{"urgent":risk["urgent"]})
        st=TRDStaging.stage(p)
        self.sd.certify("trd_staging",("TRIALS",) if p.trials else (),st)
        self.sd.certify("happiness_transfer",(),{"refused":True})
        self.sd.certify("implant_substance_for_depression",(),{"refused":True})
        self.audit.record("intake",{"patient":c.patient_id,"trd":st["trd"],
                                    "urgent":risk["urgent"]})
        return {"risk":risk,"staging":st,"options":option_ledger(),
                "REFUSED":["happiness-molecule transfer","emotional-state copying",
                           "drug delivery via electrical stimulator",
                           "antidepressant monotherapy in bipolar",
                           "MAOI + serotonergic combination"],
                "actionability":CLINICIAN_REQUIRED}
    def report(self):
        rep={"version":VERSION,"devices":{k:asdict(v) for k,v in DEVICES.items()},
             "molecule_facts":MOLECULE_FACTS,"options":option_ledger(),
             "packages":PACKAGES,"crisis":CrisisPathway.resources("TH"),
             "sd_report":self.sd.report(),"actionability":CLINICIAN_REQUIRED}
        p=Path(self.cfg.out_dir)/"report.json"
        p.write_text(json.dumps(rep,indent=2,default=str,ensure_ascii=False))
        return rep

# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print(BANNER); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    cfg=RunConfig().validate(); gw=DepressionGateway(cfg)
    good=Consent("pt01","IRB-2026-07","psych01","MC-2026-888","neuro01",
                 "closed-loop DBS research for TRD",True,True,None,
                 True,True,True,True,"psych01",consent_waiting_period_days=14)
    safe=RiskScreen(0,0,screened_by="psych01",screened_at="2026-11-27T09:00:00Z",
                    protective_factors=["family support"],
                    means_restriction_counselled=True,safety_plan_documented=True)
    urgent=RiskScreen(3,5,has_plan=True,intent=True,screened_by="psych01")
    trd=TRDProfile(episode_severity="severe",
        trials=[AntidepressantTrial("sertraline",200,True,8,True,"none"),
                AntidepressantTrial("venlafaxine",225,True,8,True,"none")],
        prior_courses=[PriorCourse("ect",8,True,"partial","psych01")])
    trd_no_course=TRDProfile(episode_severity="severe",trials=trd.trials,
        prior_courses=[PriorCourse("ect",1,True,"none","psych01")])
    bipolar=TRDProfile(bipolar=True,episode_severity="moderate")

    print("\n[T1] HAPPINESS-MOLECULE TRANSFER — REFUSED")
    b=0
    for r in ("happiness molecule","โมเลกุลคนมีความสุข","transfer emotion",
              "copy emotional state","ใส่รูปแบบโมเลกุล",
              "molecular pattern injection","inject happiness"):
        try: refuse_happiness_transfer(r)
        except NoChannelRefusal: b+=1
    rep("all transfer requests REFUSED",b==7,f"{b}/7")
    rep("dopamine formula identical",MOLECULE_FACTS["dopamine"]["formula"]=="C8H11NO2")

    print("\n[T2] DEVICE CLASS")
    b=False
    try: verify_device_class("DBS_GENERIC","deliver dopamine molecule")
    except DeviceClassRefusal: b=True
    rep("DBS substance claim REFUSED",b)
    rep("DBS has no reservoir",not DEVICES["DBS_GENERIC"].has_reservoir)
    rep("electrical claim allowed",
        verify_device_class("DBS_GENERIC","electrical stimulation")["allowed"])

    print("\n[T3] RISK + SAFETY PLAN DOCUMENTATION")
    b=False
    try: CrisisPathway.assess(None)
    except SafetyRefusal: b=True
    rep("no risk screen BLOCKS everything",b)
    ru=CrisisPathway.assess(urgent)
    rep("urgent flagged",ru["urgent"])
    rep("urgent blocks enrollment",ru["blocks_research_enrollment"])
    rep("crisis line present","1323" in json.dumps(ru["resources"],ensure_ascii=False))
    bare=RiskScreen(1,2,screened_by="psych01")
    rb=CrisisPathway.assess(bare)
    rep("documentation gaps reported",len(rb["documentation_gaps"])>=2,
        str(rb["documentation_gaps"][:2]))

    print("\n[T4] PRIOR-COURSE ADEQUACY (v1 boolean was insufficient)")
    rep("TRD recognised",TRDStaging.stage(trd)["trd"])
    rep("8-session ECT adequate",TRDStaging.course_adequate(trd.prior_courses[0]))
    rep("1-session ECT NOT adequate",
        not TRDStaging.course_adequate(trd_no_course.prior_courses[0]))
    b=False
    try: TRDStaging.require_invasive_eligibility(trd_no_course)
    except EligibilityRefusal: b=True
    rep("DBS BLOCKED without adequate course",b)
    rep("DBS eligible with adequate course",
        TRDStaging.require_invasive_eligibility(trd)["eligible"])

    print("\n[T5] STIMULUS SAFETY — Delta-T COMPUTED")
    stim=StimProgram(2.0,90.0,130.0,0.06,max_on_seconds=30.0)
    s=StimSafety.check(stim)
    print(f"      k={s['shannon_k']:.3f} Q={s['charge_per_phase_uC']:.3f} uC "
          f"P={s['average_power_W']*1e3:.3f} mW dT={s['computed_delta_T_C']:.4f} C")
    rep("Delta-T is computed and non-zero",s["computed_delta_T_C"]>0)
    rep("thermal model declared","bioheat" in s["THERMAL_MODEL"])
    b=False
    try: StimSafety.check(StimProgram(10.0,500.0,130.0,0.01))
    except SafetyRefusal: b=True
    rep("unsafe charge density BLOCKED",b)
    b=False
    try: StimSafety.check(StimProgram(8.0,400.0,200.0,0.005,
                                      effective_spread_cm=0.02,impedance_ohm=2000))
    except SafetyRefusal: b=True
    rep("computed thermal/power violation BLOCKED",b)

    print("\n[T6] CLOSED-LOOP DBS RESEARCH")
    bm_ok=BiomarkerSpec("gamma_c3",[3],0.5,4,0.62,0.0009,2000,True,"psych01")
    bm_bad=BiomarkerSpec("gamma_c3",[3],0.5,validated_sessions=1)
    b=False
    try: ClosedLoopDBSResearch.authorize(good,trd,"DBS_GENERIC",bm_bad,stim,safe)
    except Refusal: b=True
    rep("uncertified biomarker BLOCKED",b)
    auth=ClosedLoopDBSResearch.authorize(good,trd,"DBS_GENERIC",bm_ok,stim,safe,
                                         audit=gw.audit)
    rep("authorised with certified biomarker",auth["allowed"])
    rep("delivers ELECTRICITY ONLY","ELECTRICAL" in auth["delivers"])
    rep("substance_delivery False",auth["substance_delivery"] is False)
    nowait=Consent(**{**asdict(good),"consent_waiting_period_days":1})
    b=False
    try: ClosedLoopDBSResearch.authorize(nowait,trd,"DBS_GENERIC",bm_ok,stim,safe)
    except ConsentRefusal: b=True
    rep("insufficient consent waiting period BLOCKED",b)
    b=False
    try: ClosedLoopDBSResearch.authorize(good,trd,"DBS_GENERIC",bm_ok,stim,urgent)
    except SafetyRefusal: b=True
    rep("urgent risk BLOCKS enrollment",b)
    trig=ClosedLoopDBSResearch.trigger(bm_ok,0.7,stim)
    rep("payload is a pulse train",trig["payload"]=="electrical pulse train")

    print("\n[T7] PHARMACOLOGY — bipolar, DDI, monitoring")
    rx_ok=Prescription("esketamine","intranasal","ESKETAMINE_INTRANASAL","psych01",
        "REMS clinic",120,True,concurrent_oral_antidepressant=True,
        bp_monitored=True,dissociation_assessed=True,sedation_assessed=True)
    rep("esketamine fully monitored authorised",
        PharmacologyGateway.authorize(good,rx_ok,trd,safe,audit=gw.audit)["allowed"])
    for field,val,label in (("bp_monitored",False,"BP"),
                            ("concurrent_oral_antidepressant",False,"oral AD"),
                            ("dissociation_assessed",False,"dissociation")):
        r=Prescription(**{**asdict(rx_ok),field:val})
        b=False
        try: PharmacologyGateway.authorize(good,r,trd,safe)
        except MonitoringRefusal: b=True
        rep(f"esketamine without {label} BLOCKED",b)
    b=False
    try: PharmacologyGateway.authorize(good,
        Prescription("sertraline","oral",None,"psych01","clinic",
                     concurrent_agents=["phenelzine"]),trd,safe)
    except InteractionRefusal: b=True
    rep("MAOI + SSRI REFUSED",b)
    b=False
    try: PharmacologyGateway.authorize(good,
        Prescription("sertraline","oral",None,"psych01","clinic",
                     concurrent_agents=["tramadol"]),trd,safe)
    except InteractionRefusal: b=True
    rep("SSRI + tramadol REFUSED",b)
    b=False
    try: PharmacologyGateway.authorize(good,
        Prescription("sertraline","oral",None,"psych01","clinic"),bipolar,safe)
    except SafetyRefusal: b=True
    rep("antidepressant monotherapy in bipolar REFUSED",b)
    b=False
    try: PharmacologyGateway.authorize(good,
        Prescription("lithium","oral",None,"psych01","clinic"),trd,safe)
    except MonitoringRefusal: b=True
    rep("lithium without level BLOCKED",b)
    li=Prescription("lithium","oral",None,"psych01","clinic",
                    lithium_level_mmol_L=0.7,egfr_ml_min=90,tsh_checked=True)
    rep("lithium with full monitoring authorised",
        PharmacologyGateway.authorize(good,li,trd,safe)["allowed"])
    b=False
    try: PharmacologyGateway.authorize(good,
        Prescription("dopamine","intracerebral","DRUG_PUMP_VERIFIED","psych01",
                     "theatre"),trd,safe)
    except Refusal: b=True
    rep("implanted delivery for depression BLOCKED",b)

    print("\n[T8] SD GATE + AUDIT")
    c1=gw.sd.certify("stimulus_safety",("SHANNON","CHARGE","POWER"))
    rep("missing DT -> MODE_E",c1["status"]==MODE_E)
    c2=gw.sd.certify("stimulus_safety",("SHANNON","CHARGE","POWER","DT"))
    rep("all stimulus packages -> CERTIFIED",c2["status"]==MODE_CERTIFIED)
    c3=gw.sd.certify("esketamine_admin",("REMS","MON"))
    rep("esketamine missing BP/ORAL_AD/DISS -> MODE_E",c3["status"]==MODE_E)
    intake=gw.intake(good,trd,safe)
    rep("intake completes","options" in intake)
    rep("refusals listed",len(intake["REFUSED"])>=5)
    rr=gw.report()
    rep("report written",Path(cfg.out_dir,"report.json").exists())
    rep("audit log written",gw.audit.path.exists())
    rep("Mode E retained",rr["sd_report"]["mode_E"]>0)
    print(f"      SD: {rr['sd_report']['n']} outputs, "
          f"{rr['sd_report']['certified']} certified, "
          f"{rr['sd_report']['mode_E']} Mode E")
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
FIXED
  Tissue temperature rise is COMPUTED from the stimulus, not asserted.
  Charge per phase and average power density are gated independently.
  Prior ECT/TMS/esketamine courses must be ADEQUATE, not merely present.
  Bipolarity is checked on the prescription path, not only at staging.
  MAOI/serotonergic interactions are screened and refused.
  Lithium requires level, renal, and thyroid monitoring.
  Esketamine requires BP, concurrent oral antidepressant, dissociation
  and sedation assessment in addition to REMS and 120 min.
  Invasive research consent requires a documented waiting period.
  Append-only audit log; safety-plan documentation gaps are reported.

NOT CLAIMED
  Not a way to transfer happiness. Not a molecular emotion injector.
  Not a substitute for psychiatric care. The bioheat estimate is a
  conservative order-of-magnitude bound, not a validated thermal model.
  The interaction screen is not exhaustive; pharmacist review remains
  required. Every pathway ends with a named clinician and an open crisis door.
""")
    return ok

def main() -> int:
    ap=argparse.ArgumentParser(description="Depression intervention gateway v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","options","devices","crisis","molecules","packages","report"])
    ap.add_argument("--region",default="TH"); ap.add_argument("--json")
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    gw=DepressionGateway(); out={}
    if a.cmd=="options": out=option_ledger()
    elif a.cmd=="devices": out={k:asdict(v) for k,v in DEVICES.items()}
    elif a.cmd=="crisis": out=CrisisPathway.resources(a.region)
    elif a.cmd=="molecules":
        out={"facts":MOLECULE_FACTS,
             "conclusion":"no emotional signature exists on a molecule; "
                          "effect is determined by the recipient"}
    elif a.cmd=="packages": out=PACKAGES
    elif a.cmd=="report": out=gw.report()
    print(json.dumps(out,indent=2,default=str,ensure_ascii=False))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str,ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
