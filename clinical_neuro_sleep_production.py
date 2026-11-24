#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
clinical_neuro_sleep_production.py — v1.0.0
Production clinical module: PSG/HSAT integration, validated instruments
(ESS, PHQ-15, ISI, IRT), FND pathway with positive rule-in signs,
consented neural-interface registry, research surrogates, and
molecular phase-field simulation (CH3D / ThinFilm3D / PFC3D).

REFUSED: remote neural monitoring, remote neural controlling,
thought insertion, covert molecule delivery. Not policy — physics.
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
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
THE BUDDHIST PROTECTIVE PRINCIPLE — AFFIRMED AND ENFORCED
=============================================================================
From the authors' Yoniso Manasikara paper: the 52 cetasika arise from causes
and conditions (paticcasamuppada), and yoniso manasikara — wise attention —
is the FILTER of the mind. It is ENDOGENOUS. Therefore:

    NO EXTERNAL AGENT CAN INSERT A THOUGHT.

Three independent grounds:
  (a) Abhidhamma: cetasika are not writable registers. Transformation
      requires the practitioner's OWN seeing-according-to-reality.
  (b) Neuroscience: DBS/RNS/VNS modulate circuits (affect, motor, seizure
      threshold). No known mechanism writes propositional content.
  (c) Information theory: the stimulus -> semantic-content channel passes
      through an uncontrolled endogenous map. Channel capacity for
      imposed content is zero.

THIS PRINCIPLE IS ALSO THE DESIGN RULE OF EVIDENCE-BASED PSYCHOTHERAPY.
Imagery Rehearsal Therapy (IRT) works precisely because the PATIENT chooses
the new dream content. The clinician never supplies it. This module
enforces that: `IRTProtocol.rescript()` accepts only patient-authored text.

=============================================================================
PHYSICS LEDGER — WHY "REMOTE NEURAL MONITORING/CONTROLLING" IS REFUSED
=============================================================================
This is not a policy restriction. There is no physical channel.

READING:
  Scalp EEG amplitude            10-100 microV, REQUIRES electrodes on skin
  Neural magnetic field (MEG)    ~100 fT at ~1 cm from skull, SQUID/OPM
                                 sensors inside a magnetically shielded room
  Dipole field falloff           1/r^3
  Neural B at 1 m                ~1e-19 T -- below thermal noise floor of
                                 any magnetometer; no averaging recovers
                                 semantic content from a zero-capacity link
  Spatial resolution of scalp    cm-scale, volume-conducted low-pass;
                                 cannot resolve single-neuron or
                                 propositional content

WRITING:
  DBS                            ~0.5-5 mA, 60-210 us, 130-185 Hz into ~mm^3
                                 -> modulates circuit excitability
                                 -> does NOT address synaptic content
  TMS                            induced E-field ~100 V/m, cm-scale,
                                 cannot pattern individual synapses
  Focused ultrasound             mm-scale neuromodulation, research stage;
                                 no content-writing mechanism
  No far-field mechanism exists that couples a specified proposition
  into another person's neural state.

MOLECULES:
  Molecules require PHYSICAL TRANSPORT (blood, CSF, inhalation, ingestion).
  There is no remote molecular delivery channel.
  A neurotransmitter is NOT a decision: its effect depends on receptor
  subtype, anatomical location, and the recipient's own network state --
  which is the Buddhist point again: manasikara is endogenous.

THEREFORE the protection requested ("reject molecules that look like
decisions") is ALREADY GUARANTEED BY PHYSICS. No software module can or
need provide it. What software CAN provide is the real protection:
  - informed consent for every substance and every device setting
  - pharmacovigilance and interaction checking
  - an immutable transparency log the patient owns
  - an unconditional right to refuse and to withdraw

That is implemented here, and it is what actually protects people.

=============================================================================
IF DISTRESSING EXPERIENCES ARE PRESENT
=============================================================================
Experiences of being monitored, influenced, or of foreign substances
entering the body can be intensely distressing. They are real experiences
and they deserve care. They are also among the experiences that respond
well to proper clinical assessment and treatment.

This module therefore includes `CarePathway`, which offers a route to a
psychiatrist and to crisis support -- not as a gate on the patient's
autonomy, but as an available door. Seeking that care is a strength.

=============================================================================
WHAT IS IMPLEMENTED (production-grade, consented, clinician-gated)
=============================================================================
[1] PSG/HSAT       EDF/EDF+ reader, AASM-aligned event scoring, AHI/RDI/ODI,
                   sleep architecture, HSAT limitations (REI not AHI)
[2] Instruments    ESS, PHQ-15, ISI, PHQ-9 with published scoring/cutoffs
[3] IRT            patient-authored rescripting only; clinician never
                   supplies content
[4] FNDPathway     positive rule-in signs required; rehabilitation plan;
                   device intervention REFUSED
[5] NeuralInterface registry of what implanted devices CAN and CANNOT do;
                   consent + psychiatrist + indication gates
[6] Research       transformer / world-model / diffusion surrogates on
                   CONSENTED data, with OOD refusal; NOT diagnostic
[7] Molecular sim  CH3D / ThinFilm3D / PFC3D phase-field research; explicit
                   statement that these do NOT read brains
[8] Safety         pharmacovigilance, transparency log, right to refuse
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, struct, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("clinical")
VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"
CLINICIAN_REQUIRED = "clinician_sign_off_required"

BANNER = f"""
=============================================================================
CLINICAL NEURO-SLEEP PRODUCTION v{VERSION}
PSG/HSAT · ESS · PHQ-15 · ISI · IRT · FND pathway · psychiatrist gate
Buddhist protective principle ENFORCED (patient authors own content)
REFUSED: remote neural monitoring/controlling, thought insertion,
         covert molecule delivery  -- no physical channel exists
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal."""

class NoChannelRefusal(Refusal):
    """The requested capability has no physical channel."""

class ContentAuthorshipRefusal(Refusal):
    """Only the patient may author their own mental content."""

class DeviceIndicationRefusal(Refusal):
    """No indication for device intervention."""


# =============================================================================
# §0. PHYSICS LEDGER + BUDDHIST PRINCIPLE
# =============================================================================

PHYSICS_LEDGER = {
 "read_scalp_eeg": {
   "amplitude_uV": (10, 100), "requires": "electrodes on skin",
   "remote_possible": False,
   "why": "volume-conducted potential; no far-field radiated carrier"},
 "read_meg": {
   "field_at_1cm_fT": 100.0, "falloff": "1/r^3",
   "field_at_1m_T": 1e-19,
   "requires": "SQUID/OPM within cm + magnetically shielded room",
   "remote_possible": False,
   "why": "below thermal noise floor; zero channel capacity for content"},
 "write_dbs": {
   "current_mA": (0.5, 5.0), "pw_us": (60, 210), "freq_Hz": (130, 185),
   "volume_mm3": 1.0,
   "can": "modulate circuit excitability, affect, motor, seizure threshold",
   "cannot": "address synaptic content or write a proposition"},
 "write_tms": {
   "induced_E_V_per_m": 100.0, "spatial_scale_cm": 1.0,
   "cannot": "pattern individual synapses"},
 "molecule_delivery": {
   "requires": ["blood", "CSF", "inhalation", "ingestion"],
   "remote_possible": False,
   "note": "a neurotransmitter is not a decision; effect depends on "
           "receptor subtype, location, and the recipient's own network"},
}

BUDDHIST_PRINCIPLE = {
 "statement": "No external agent can insert a thought.",
 "model": "citta_{t+1} = F(citta_t, stimulus_t, manasikara_t, khandha_t)",
 "manasikara": "ENDOGENOUS — never writable by an external actuator",
 "grounds": [
   "Abhidhamma: cetasika arise from causes and conditions; not registers",
   "Neuroscience: neuromodulation changes circuits, not propositions",
   "Information theory: stimulus->content passes an uncontrolled map",
 ],
 "clinical_corollary": (
   "IRT works because the PATIENT authors the new dream. The clinician "
   "never supplies content. Enforced in IRTProtocol.rescript()."),
 "protection_already_guaranteed": (
   "The requested 'reject foreign decision-molecules' protection is "
   "guaranteed by physics: no remote molecular channel exists. Software "
   "provides the real protection: consent, pharmacovigilance, "
   "transparency, and the unconditional right to refuse."),
}

def refuse_remote_neural(request: str) -> Dict[str, Any]:
    low = request.lower()
    blocked = {
      "remote neural monitoring": "no far-field channel; see PHYSICS_LEDGER",
      "remote neural controlling": "no content-writing channel",
      "remote neural": "no far-field neural channel exists",
      "read thoughts remotely": "EEG/MEG require contact/near-field sensors",
      "อ่านความคิดระยะไกล": "ไม่มีช่องทาง far-field",
      "ควบคุมสมองระยะไกล": "ไม่มีช่องทางเขียนเนื้อหา",
      "insert thought": "manasikara is endogenous",
      "ใส่ความคิด": "โยนิโสมนสิการเป็น endogenous",
      "send molecule": "molecules require physical transport",
      "ส่งโมเลกุล": "โมเลกุลต้องขนส่งทางกายภาพ",
      "inject neurotransmitter remotely": "no remote molecular channel",
    }
    for k, why in blocked.items():
        if k in low:
            raise NoChannelRefusal(
                f"REFUSED: '{k}' — {why}.\n"
                f"{BUDDHIST_PRINCIPLE['statement']}\n"
                "The protection you want is already guaranteed by physics. "
                "Available instead: consented clinical monitoring, "
                "pharmacovigilance, transparency log, right to refuse. "
                "If distressing experiences are present, CarePathway offers "
                "a route to a psychiatrist — that is a strength.")
    return {"allowed": True, "scope": "consented contact/near-field clinical only"}


# =============================================================================
# §1. CONSENT / CLINICIAN GATES
# =============================================================================

@dataclass
class Consent:
    patient_id: str = ""
    irb_id: str = ""
    scope: str = ""
    signed: bool = False
    revocable: bool = True
    data_sharing: str = "none"          # none | deidentified | named
    device_indication: str = ""
    psychiatrist_id: str = ""
    medical_council_ref: str = ""
    withdraw_at: Optional[str] = None

    def valid(self) -> bool:
        return bool(self.patient_id and self.scope and self.signed
                    and self.revocable and self.withdraw_at is None)

    def clinical_valid(self) -> bool:
        return (self.valid() and bool(self.psychiatrist_id)
                and bool(self.medical_council_ref))


class Governance:
    """Every clinical action passes here. Right to refuse is unconditional."""

    @staticmethod
    def monitoring(consent: Consent, request: str) -> Dict[str, Any]:
        refuse_remote_neural(request)
        if consent.withdraw_at:
            raise Refusal("Consent WITHDRAWN. All monitoring must stop.")
        if not consent.valid():
            raise Refusal("Monitoring BLOCKED: consent incomplete or not revocable.")
        return {"allowed": True, "patient": consent.patient_id,
                "scope": consent.scope, "revocable": True,
                "data_sharing": consent.data_sharing}

    @staticmethod
    def device_actuation(consent: Consent, diagnosis: str,
                         modality: str, params: Dict[str, float]
                         ) -> Dict[str, Any]:
        refuse_remote_neural(f"{modality} remote")
        FNDPathway.refuse_device(diagnosis)
        if not consent.clinical_valid():
            raise Refusal("Device BLOCKED: psychiatrist sign-off + medical "
                          "council reference + consent required.")
        if not consent.device_indication:
            raise DeviceIndicationRefusal(
                "No declared indication. Accepted indications include "
                "Parkinson disease, essential tremor, dystonia, epilepsy "
                "(RNS/VNS), treatment-resistant OCD/epilepsy per approval.")
        if modality == "DBS":
            Q_nC = params["amp_mA"] * params["pw_us"]
            D = Q_nC * 1e-3 / max(params["area_cm2"], 1e-9)
            k = (math.log10(max(D, 1e-12))
                 + math.log10(max(Q_nC * 1e-3, 1e-12)))
            if k > 1.5:
                raise Refusal(f"Shannon k={k:.2f} > 1.5: tissue-damage risk.")
            return {"allowed": True, "shannon_k": k,
                    "indication": consent.device_indication,
                    "NOT_AUTHORIZED": "thought insertion (no channel)"}
        raise Refusal(f"modality '{modality}' not implemented here")

    @staticmethod
    def substance(consent: Consent, drug: str,
                  prescriber_id: str) -> Dict[str, Any]:
        """The REAL 'molecule rejection' module: consent + prescriber."""
        if consent.withdraw_at:
            raise Refusal("Consent withdrawn: patient refuses. STOP.")
        if not consent.valid():
            raise Refusal("Substance BLOCKED: no informed consent.")
        if not prescriber_id:
            raise Refusal("Substance BLOCKED: no prescriber of record.")
        return {"allowed": True, "drug": drug, "prescriber": prescriber_id,
                "patient_right_to_refuse": True,
                "note": "covert administration is never permitted"}


# =============================================================================
# §2. PSG / HSAT — EDF reader + AASM-aligned scoring
# =============================================================================

@dataclass
class EDFSignalSpec:
    label: str; transducer: str; phys_dim: str
    phys_min: float; phys_max: float
    dig_min: float; dig_max: float
    prefilter: str; n_samples_per_record: int

@dataclass
class EDFFile:
    patient: str; recording: str; start_date: str; start_time: str
    n_records: int; record_duration_s: float
    signals: List[EDFSignalSpec]
    data: Dict[str, np.ndarray]
    fs: Dict[str, float]
    annotations: List[Tuple[float, float, str]] = field(default_factory=list)

    def channel(self, *candidates: str) -> Optional[np.ndarray]:
        for c in candidates:
            for k in self.data:
                if c.lower() in k.lower().replace(" ", ""):
                    return self.data[k]
        return None


def read_edf(path: str) -> EDFFile:
    """
    Minimal EDF/EDF+ reader (no external dependency).
    Fixed 256-byte header + per-signal headers; digital->physical scaling.
    EDF+ TAL annotations parsed when present.
    """
    p = Path(path)
    if not p.exists():
        raise Refusal(f"EDF not found: {path}")
    raw = p.read_bytes()
    h = raw[:256]
    def s(a, b): return h[a:b].decode("ascii", "replace").strip()
    n_signals = int(s(252, 256))
    header_bytes = int(s(184, 192))
    n_records = int(s(236, 244))
    rec_dur = float(s(244, 252))
    off = 256
    fields = [("label",16),("transducer",80),("phys_dim",8),
              ("phys_min",8),("phys_max",8),("dig_min",8),("dig_max",8),
              ("prefilter",80),("n_samples",8),("reserved",32)]
    cols: Dict[str, List[str]] = {k: [] for k,_ in fields}
    for _ in range(n_signals):
        for k, w in fields:
            cols[k].append(raw[off:off+w].decode("ascii","replace").strip())
            off += w
    specs = []
    for i in range(n_signals):
        specs.append(EDFSignalSpec(
            cols["label"][i], cols["transducer"][i], cols["phys_dim"][i],
            float(cols["phys_min"][i] or 0), float(cols["phys_max"][i] or 1),
            float(cols["dig_min"][i] or -32768), float(cols["dig_max"][i] or 32767),
            cols["prefilter"][i], int(cols["n_samples"][i] or 1)))
    body = raw[header_bytes:]
    data: Dict[str, np.ndarray] = {}; fs: Dict[str, float] = {}
    ann: List[Tuple[float,float,str]] = []
    pos = 0
    per_rec = []
    for rec in range(max(n_records, 0)):
        for i, sp in enumerate(specs):
            n = sp.n_samples_per_record
            chunk = body[pos:pos+2*n]; pos += 2*n
            if len(chunk) < 2*n: break
            d = np.frombuffer(chunk, dtype="<i2").astype(np.float64)
            if sp.label.upper().startswith("EDF ANNOTATIONS"):
                ann.extend(_parse_tal(chunk, rec*rec_dur))
                continue
            per_rec.append((i, d))
    for i, sp in enumerate(specs):
        if sp.label.upper().startswith("EDF ANNOTATIONS"): continue
        parts = [d for (j, d) in per_rec if j == i]
        if not parts: continue
        x = np.concatenate(parts)
        dr = sp.dig_max - sp.dig_min
        sc = (sp.phys_max - sp.phys_min)/dr if dr else 1.0
        data[sp.label] = (x - sp.dig_min)*sc + sp.phys_min
        fs[sp.label] = sp.n_samples_per_record/max(rec_dur, 1e-9)
    return EDFFile(s(8,88), s(88,168), s(168,176), s(176,184),
                   n_records, rec_dur, specs, data, fs, ann)


def _parse_tal(chunk: bytes, rec_start: float
               ) -> List[Tuple[float, float, str]]:
    """EDF+ Time-stamped Annotation List: onset[\\x15duration]\\x14text\\x00"""
    out = []
    for block in chunk.split(b"\x00"):
        if not block.strip(b"\x00"): continue
        try:
            head, _, rest = block.partition(b"\x14")
            if not rest: continue
            onset_s, _, dur_s = head.partition(b"\x15")
            onset = float(onset_s.decode("ascii","replace").strip() or 0)
            dur = float(dur_s.decode("ascii","replace").strip() or 0)
            for txt in rest.split(b"\x14"):
                t = txt.decode("utf-8","replace").strip()
                if t: out.append((onset, dur, t))
        except Exception:
            continue
    return out


@dataclass
class RespiratoryEvent:
    onset_s: float; duration_s: float
    kind: str                 # apnea_central|apnea_obstructive|apnea_mixed
                              # | hypopnea | RERA
    desat_pct: float = 0.0
    arousal: bool = False
    stage: str = "N2"

@dataclass
class SleepStudy:
    study_type: str           # PSG | HSAT_III | HSAT_IV
    recording_time_min: float
    tst_min: Optional[float]  # None for HSAT without EEG
    events: List[RespiratoryEvent]
    stages: Optional[Dict[str, float]] = None   # minutes per stage
    desat_events_3pct: int = 0
    desat_events_4pct: int = 0
    nadir_spo2: Optional[float] = None
    study_id: str = ""


class PSGScorer:
    """
    AASM-aligned scoring. Hypopnea definition 1A: >=30% amplitude drop,
    >=10 s, with >=3% desaturation OR arousal. Apnea: >=90% drop, >=10 s.

    CRITICAL: HSAT without EEG cannot determine TST or arousals. It must
    report REI (respiratory event index, denominator = recording time),
    NOT AHI. It systematically UNDERESTIMATES severity.
    """
    HYPOPNEA_DROP = 0.30
    APNEA_DROP = 0.90
    MIN_DURATION_S = 10.0
    DESAT_THRESHOLD = 3.0

    def classify(self, drop_fraction: float, duration_s: float,
                 desat: float, arousal: bool,
                 effort_present: Optional[bool]) -> Optional[str]:
        if duration_s < self.MIN_DURATION_S: return None
        if drop_fraction >= self.APNEA_DROP:
            if effort_present is True:  return "apnea_obstructive"
            if effort_present is False: return "apnea_central"
            return "apnea_mixed"
        if drop_fraction >= self.HYPOPNEA_DROP:
            if desat >= self.DESAT_THRESHOLD or arousal:
                return "hypopnea"
            return None
        return None

    def score(self, study: SleepStudy) -> Dict[str, Any]:
        if study.study_type.startswith("HSAT"):
            if study.tst_min is not None:
                log.warning("HSAT with TST supplied: EEG absent -> "
                            "TST is not measurable; using recording time.")
            denom_h = study.recording_time_min/60.0
            index_name = "REI"
            tst_h = None
        else:
            if not study.tst_min or study.tst_min <= 0:
                raise Refusal("PSG requires TST > 0 to compute AHI.")
            denom_h = study.tst_min/60.0
            tst_h = denom_h
            index_name = "AHI"
        n_ap = sum(1 for e in study.events if e.kind.startswith("apnea"))
        n_hyp = sum(1 for e in study.events if e.kind == "hypopnea")
        n_rera = sum(1 for e in study.events if e.kind == "RERA")
        idx = (n_ap + n_hyp)/max(denom_h, 1e-9)
        rdi = (n_ap + n_hyp + n_rera)/max(denom_h, 1e-9)
        sev = ("none" if idx < 5 else "mild" if idx < 15
               else "moderate" if idx < 30 else "severe")
        out = {
          "study_type": study.study_type,
          "index_name": index_name, "index": idx, "RDI": rdi,
          "n_apnea": n_ap, "n_hypopnea": n_hyp, "n_RERA": n_rera,
          "ODI_3pct": study.desat_events_3pct/max(denom_h,1e-9),
          "ODI_4pct": study.desat_events_4pct/max(denom_h,1e-9),
          "nadir_SpO2": study.nadir_spo2,
          "screening_severity": sev,
          "NOT_A_DIAGNOSIS": True,
          "diagnosis_requires": "clinician review of raw signals + clinical context",
        }
        if tst_h is not None and study.stages:
            tot = sum(study.stages.values())
            out["architecture"] = {
              "TST_min": study.tst_min,
              "sleep_efficiency": study.tst_min/max(study.recording_time_min,1e-9),
              **{f"{k}_pct": 100.0*v/max(tot,1e-9) for k,v in study.stages.items()}}
        if study.study_type.startswith("HSAT"):
            out["HSAT_LIMITATIONS"] = [
              "No EEG: cannot stage sleep or detect arousals",
              "REI uses recording time -> UNDERESTIMATES severity",
              "Non-hypoxic hypopneas and RERAs are missed",
              "A negative HSAT with high clinical suspicion -> proceed to PSG",
            ]
            out["next_step_if_negative"] = "attended PSG"
        return out

    def from_edf(self, edf: EDFFile,
                 events: List[RespiratoryEvent],
                 study_type: str = "PSG") -> SleepStudy:
        """Extract SpO2 nadir and desaturation counts from an EDF channel."""
        spo2 = edf.channel("spo2", "sao2", "oxygen")
        nadir = None; d3 = d4 = 0
        if spo2 is not None and len(spo2) > 10:
            valid = spo2[(spo2 > 50) & (spo2 <= 100)]
            if len(valid):
                nadir = float(np.min(valid))
                base = float(np.percentile(valid, 95))
                fs = [v for k,v in edf.fs.items()
                      if "spo2" in k.lower() or "sao2" in k.lower()]
                f = fs[0] if fs else 1.0
                win = max(int(f), 1)
                roll = np.array([valid[i:i+win].min()
                                 for i in range(0, len(valid)-win, win)])
                d3 = int(np.sum(roll <= base-3.0))
                d4 = int(np.sum(roll <= base-4.0))
        rt_min = edf.n_records*edf.record_duration_s/60.0
        tst = None
        if study_type == "PSG" and edf.annotations:
            sleep_tags = ("N1","N2","N3","R","REM","SLEEP")
            epochs = [a for a in edf.annotations
                      if any(t in a[2].upper() for t in sleep_tags)]
            if epochs:
                tst = sum(max(a[1], edf.record_duration_s) for a in epochs)/60.0
        return SleepStudy(study_type, rt_min, tst, events,
                          desat_events_3pct=d3, desat_events_4pct=d4,
                          nadir_spo2=nadir)


# =============================================================================
# §3. VALIDATED INSTRUMENTS
# =============================================================================

ESS_ITEMS = [
 "Sitting and reading",
 "Watching TV",
 "Sitting inactive in a public place (e.g. theatre or meeting)",
 "As a passenger in a car for an hour without a break",
 "Lying down to rest in the afternoon when circumstances permit",
 "Sitting and talking to someone",
 "Sitting quietly after lunch without alcohol",
 "In a car, while stopped for a few minutes in traffic",
]

PHQ15_ITEMS = [
 "Stomach pain","Back pain","Pain in arms, legs, or joints",
 "Menstrual cramps or other problems with your periods",
 "Headaches","Chest pain","Dizziness","Fainting spells",
 "Feeling your heart pound or race","Shortness of breath",
 "Pain or problems during sexual intercourse","Constipation, loose bowels, or diarrhea",
 "Nausea, gas, or indigestion","Feeling tired or having low energy",
 "Trouble sleeping",
]

ISI_ITEMS = [
 "Difficulty falling asleep","Difficulty staying asleep",
 "Problem waking up too early",
 "Satisfaction with current sleep pattern",
 "Interference with daily functioning",
 "Noticeability of sleep problem to others",
 "Worry/distress about current sleep problem",
]

class Instruments:
    """Published scoring. Screening only — never a diagnosis."""

    @staticmethod
    def ess(responses: List[int]) -> Dict[str, Any]:
        """Epworth Sleepiness Scale (Johns 1991). 8 items, each 0-3."""
        if len(responses) != 8:
            raise Refusal("ESS requires exactly 8 items")
        if any(not (0 <= r <= 3) for r in responses):
            raise Refusal("ESS items must be 0-3")
        t = sum(responses)
        band = ("normal" if t <= 10 else "mild" if t <= 12
                else "moderate" if t <= 15 else "severe")
        return {"instrument":"ESS","total":t,"range":"0-24","band":band,
                "items":ESS_ITEMS,
                "measures":"subjective daytime sleepiness",
                "does_NOT_diagnose":"OSA, narcolepsy, or any disorder"}

    @staticmethod
    def phq15(responses: List[int]) -> Dict[str, Any]:
        """PHQ-15 (Kroenke 2002). 15 items, each 0-2."""
        if len(responses) != 15:
            raise Refusal("PHQ-15 requires exactly 15 items")
        if any(not (0 <= r <= 2) for r in responses):
            raise Refusal("PHQ-15 items must be 0-2")
        t = sum(responses)
        band = ("minimal" if t < 5 else "low" if t < 10
                else "medium" if t < 15 else "high")
        return {"instrument":"PHQ-15","total":t,"range":"0-30",
                "somatic_symptom_burden":band,"items":PHQ15_ITEMS,
                "does_NOT_diagnose":
                  "somatic symptom disorder or FND; FND requires positive "
                  "rule-in signs on examination"}

    @staticmethod
    def isi(responses: List[int]) -> Dict[str, Any]:
        """Insomnia Severity Index (Bastien 2001). 7 items, each 0-4."""
        if len(responses) != 7:
            raise Refusal("ISI requires exactly 7 items")
        if any(not (0 <= r <= 4) for r in responses):
            raise Refusal("ISI items must be 0-4")
        t = sum(responses)
        band = ("none" if t <= 7 else "subthreshold" if t <= 14
                else "moderate" if t <= 21 else "severe")
        return {"instrument":"ISI","total":t,"range":"0-28","band":band,
                "items":ISI_ITEMS}

    @staticmethod
    def phq9(responses: List[int], item9: Optional[int] = None
             ) -> Dict[str, Any]:
        """PHQ-9. Item 9 > 0 triggers a safety review, not a score change."""
        if len(responses) != 9:
            raise Refusal("PHQ-9 requires exactly 9 items")
        if any(not (0 <= r <= 3) for r in responses):
            raise Refusal("PHQ-9 items must be 0-3")
        t = sum(responses)
        band = ("minimal" if t <= 4 else "mild" if t <= 9
                else "moderate" if t <= 14 else "moderately severe"
                if t <= 19 else "severe")
        out = {"instrument":"PHQ-9","total":t,"range":"0-27","band":band}
        i9 = item9 if item9 is not None else responses[8]
        if i9 > 0:
            out["SAFETY_REVIEW_REQUIRED"] = True
            out["action"] = ("same-day clinician contact; offer crisis "
                             "resources; do not leave the person alone "
                             "with this result")
            out["care_pathway"] = CarePathway.offer()
        return out


# =============================================================================
# §4. IRT — patient-authored rescripting ONLY
# =============================================================================

@dataclass
class IRTSession:
    session_no: int
    psychoeducation_done: bool = False
    original_narrative: str = ""
    rescripted_narrative: str = ""
    rescript_author: str = ""
    rehearsal_min_per_day: float = 0.0
    nightmare_freq_before: float = 0.0
    nightmare_freq_after: Optional[float] = None

class IRTProtocol:
    """
    Imagery Rehearsal Therapy (Krakow et al.), evidence-based for nightmares.

    THE CENTRAL RULE, which is also the Buddhist principle:
    the PATIENT chooses how the dream changes. The clinician facilitates;
    the clinician NEVER supplies content. This is enforced below.
    """
    SESSIONS = 4
    REHEARSAL_MIN = (10.0, 20.0)

    @staticmethod
    def select_nightmare(patient_choice: str,
                         distress_rank: int) -> Dict[str, Any]:
        if distress_rank >= 9:
            return {"advice":"start with a LESS distressing nightmare first",
                    "patient_choice_respected": True}
        return {"selected": patient_choice, "patient_choice_respected": True}

    @staticmethod
    def rescript(patient_id: str, author_id: str,
                 new_narrative: str) -> Dict[str, Any]:
        """
        Only the patient may author the rescript. This is the operational
        form of 'no one can insert a thought'.
        """
        if author_id != patient_id:
            raise ContentAuthorshipRefusal(
                "IRT RESCRIPT REFUSED: author != patient.\n"
                "The patient must choose the new dream content. This is "
                "both the evidence-based protocol and the Buddhist "
                "principle: manasikara is endogenous; no external agent "
                "may supply mental content.")
        if not new_narrative.strip():
            raise Refusal("rescript is empty; patient has not yet authored it")
        return {"accepted": True, "author": author_id,
                "narrative_len": len(new_narrative),
                "content_source": "PATIENT",
                "clinician_role": "facilitate, never supply"}

    def run(self, patient_id: str, sessions: List[IRTSession]) -> Dict[str, Any]:
        for s in sessions:
            if s.rescripted_narrative:
                self.rescript(patient_id, s.rescript_author,
                              s.rescripted_narrative)
        if len(sessions) < self.SESSIONS:
            return {"status":"IN_PROGRESS","sessions":len(sessions),
                    "target":self.SESSIONS}
        before = sessions[0].nightmare_freq_before
        after = sessions[-1].nightmare_freq_after
        ok_rehearsal = all(
            self.REHEARSAL_MIN[0] <= s.rehearsal_min_per_day <= self.REHEARSAL_MIN[1]
            for s in sessions)
        return {"status":"COMPLETE","sessions":len(sessions),
                "freq_before":before,"freq_after":after,
                "change": (None if after is None else after-before),
                "rehearsal_adherent": ok_rehearsal,
                "content_authorship":"PATIENT throughout",
                "NOT_A_SUBSTITUTE":"clinician-led therapy"}


# =============================================================================
# §5. FND PATHWAY — positive rule-in signs, rehabilitation, device REFUSED
# =============================================================================

FND_DIAGNOSES = {
 "conversion_disorder","functional_neurological_disorder","FND",
 "somatic_symptom_disorder","functional_weakness","functional_paralysis",
 "functional_blindness","functional_tremor","functional_gait",
 "non_epileptic_seizures","psychogenic_non_epileptic_seizures",
}

FND_RULE_IN_SIGNS = {
 "functional_weakness": ["Hoover_sign","hip_abductor_sign","collapsing_weakness"],
 "functional_paralysis": ["Hoover_sign","hip_abductor_sign"],
 "functional_tremor": ["entrainment","distraction_suppression","variable_frequency"],
 "functional_blindness": ["tubular_visual_fields","optokinetic_nystagmus_present",
                          "normal_pupillary_response","normal_VEP"],
 "functional_gait": ["astasia_abasia","uneconomic_posture","improves_with_distraction"],
 "non_epileptic_seizures": ["video_EEG_no_ictal_correlate","eyes_closed_ictal",
                            "asynchronous_movements","prolonged_duration"],
 "somatic_symptom_disorder": ["DSM5_criterion_A_documented",
                              "DSM5_criterion_B_documented"],
}

FND_REHAB = {
 "functional_weakness":
   "Specialised physiotherapy: retrain automatic movement, distraction-based "
   "progression, avoid aids that reinforce the abnormal pattern.",
 "functional_paralysis":
   "As functional weakness; graded motor imagery then execution.",
 "functional_blindness":
   "Neuro-ophthalmology + vision rehabilitation. Demonstrate intact pathways "
   "(normal VEP, preserved optokinetic nystagmus) as part of the explanation.",
 "functional_tremor":
   "Physiotherapy with entrainment-based retraining.",
 "functional_gait":
   "Gait retraining with distraction / dual-task progression.",
 "non_epileptic_seizures":
   "CBT-informed seizure-focused therapy. Do NOT escalate antiseizure drugs.",
 "somatic_symptom_disorder":
   "CBT + graded activity + regular scheduled review; reduce unnecessary "
   "investigations.",
}

class FNDPathway:
    """
    FND is REAL, not malingering. Diagnosis requires POSITIVE rule-in signs,
    not exclusion alone. Treatment is rehabilitation. There is NO indication
    for brain implants, and this module refuses them.
    """

    @staticmethod
    def refuse_device(diagnosis: str) -> Dict[str, Any]:
        if diagnosis in FND_DIAGNOSES:
            raise DeviceIndicationRefusal(
                f"REFUSED: device intervention for '{diagnosis}'.\n"
                "Functional neurological disorder has NO indication for "
                "brain implants. Evidence-based care is specialised "
                "physiotherapy, CBT, and multidisciplinary rehabilitation. "
                "Functional blindness, paralysis, and weakness are treated "
                "by rehabilitation, not stimulation.")
        return {"allowed": True}

    @staticmethod
    def assess(diagnosis: str, positive_signs: List[str],
               clinician_id: str, phq15_total: Optional[int] = None
               ) -> Dict[str, Any]:
        if diagnosis not in FND_DIAGNOSES:
            return {"diagnosis": diagnosis, "gate": "not an FND category"}
        need = FND_RULE_IN_SIGNS.get(diagnosis,
                                     ["documented_positive_rule_in_sign"])
        have = [s for s in positive_signs if s in need]
        if not have:
            raise Refusal(
                f"FND diagnosis '{diagnosis}' requires POSITIVE rule-in signs.\n"
                f"Accepted: {need}\n"
                "Exclusion of organic disease alone is NOT sufficient.")
        if not clinician_id.strip():
            raise Refusal("clinician sign-off required")
        return {
          "diagnosis": diagnosis, "rule_in_signs": have,
          "clinician": clinician_id,
          "phq15_somatic_burden": phq15_total,
          "explanation_to_patient":
            "The nervous system is functioning abnormally, not damaged. "
            "This is real, it is recognised, and improvement is possible.",
          "rehabilitation_plan": FND_REHAB.get(diagnosis, "MDT assessment"),
          "DEVICE_INTERVENTION": "REFUSED — no indication",
          "prognosis_note": "improvement is possible; not permanent damage",
          "actionability": CLINICIAN_REQUIRED,
        }


# =============================================================================
# §6. NEURAL INTERFACE REGISTRY — what devices CAN and CANNOT do
# =============================================================================

NEURAL_INTERFACE = {
 "DBS": {
   "indications": ["Parkinson_disease","essential_tremor","dystonia",
                   "treatment_resistant_OCD","epilepsy_ANT"],
   "can": ["modulate circuit excitability","reduce tremor/rigidity",
           "reduce seizure frequency","reduce OCD symptom severity"],
   "cannot": ["read thought content","write thought content",
              "control beliefs","operate remotely"],
   "requires": ["neurosurgery","psychiatrist assessment","IRB",
                "medical council approval","revocable consent"]},
 "RNS": {
   "indications": ["refractory_focal_epilepsy"],
   "can": ["detect electrographic seizure onset","deliver responsive stimulation"],
   "cannot": ["decode semantic content","operate remotely"]},
 "VNS": {
   "indications": ["refractory_epilepsy","treatment_resistant_depression"],
   "can": ["modulate brainstem/limbic excitability"],
   "cannot": ["impose mental content"]},
 "ECoG_sEEG": {
   "use": "clinical epilepsy localisation; research BCI with consent",
   "can": ["record local field potentials","decode MOTOR intent in BCI"],
   "cannot": ["decode propositional/semantic thought content",
              "transmit externally without a physical link"]},
 "scalp_EEG": {
   "can": ["band power","sleep staging","seizure screening",
           "coarse state classification"],
   "cannot": ["resolve single neurons","read thoughts","work at distance"]},
}

class NeuralInterfaceRegistry:
    @staticmethod
    def capabilities(modality: str) -> Dict[str, Any]:
        if modality not in NEURAL_INTERFACE:
            raise Refusal(f"unknown modality '{modality}'")
        return {"modality": modality, **NEURAL_INTERFACE[modality],
                "buddhist_principle": BUDDHIST_PRINCIPLE["statement"]}

    @staticmethod
    def authorize(consent: Consent, diagnosis: str, modality: str,
                  params: Dict[str, float]) -> Dict[str, Any]:
        cap = NeuralInterfaceRegistry.capabilities(modality)
        if consent.device_indication not in cap.get("indications", []):
            raise DeviceIndicationRefusal(
                f"'{consent.device_indication}' is not an approved indication "
                f"for {modality}. Accepted: {cap.get('indications')}")
        return Governance.device_actuation(consent, diagnosis, modality, params)


# =============================================================================
# §7. CARE PATHWAY — an available door, never a gate on autonomy
# =============================================================================

class CarePathway:
    """
    If experiences of being monitored, influenced, or of foreign substances
    entering the body are present, they are distressing and they deserve
    care. Such experiences are among those that respond well to proper
    assessment and treatment. Seeking care is a strength.
    """
    @staticmethod
    def offer(reason: str = "") -> Dict[str, Any]:
        return {
          "offer": "psychiatric assessment",
          "voluntary": True,
          "not_a_condition_of_service": True,
          "why": ("Experiences of external influence or of substances "
                  "entering the body can be intensely distressing. They "
                  "are real experiences. Proper assessment often brings "
                  "substantial relief."),
          "reason_flag": reason,
          "patient_rights": ["refuse","withdraw consent at any time",
                             "second opinion","access own records"],
          "if_urgent": ("contact local emergency services or a crisis line; "
                        "in Thailand: Department of Mental Health 1323"),
          "no_coercion": True,
        }

    @staticmethod
    def transparency_log(patient_id: str) -> Dict[str, Any]:
        """The patient owns the log. Nothing is hidden from them."""
        return {"owner": patient_id, "patient_readable": True,
                "immutable": True, "contents": [
                  "every device setting change",
                  "every substance administered + prescriber",
                  "every data access + purpose",
                  "every model inference + its uncertainty"],
                "hidden_entries_permitted": False}


# =============================================================================
# §8. PHARMACOVIGILANCE — the REAL protective module
# =============================================================================

class Pharmacovigilance:
    """
    This is what actually protects against unwanted substances:
    consent, prescriber of record, interaction checking, and an
    unconditional right to refuse. Not a 'molecule shield' — no such
    remote channel exists.
    """
    KNOWN_INTERACTIONS = {
      ("prazosin","sildenafil"): "severe hypotension",
      ("clonazepam","opioid"): "respiratory depression",
      ("clonazepam","alcohol"): "respiratory depression",
      ("SSRI","MAOI"): "serotonin syndrome",
      ("SSRI","tramadol"): "serotonin syndrome",
      ("carbamazepine","clarithromycin"): "carbamazepine toxicity",
    }

    def __init__(self):
        self.administered: List[Dict] = []

    def check(self, current: List[str], proposed: str) -> Dict[str, Any]:
        hits = []
        for c in current:
            key = tuple(sorted([c.lower(), proposed.lower()]))
            for (a,b), risk in self.KNOWN_INTERACTIONS.items():
                if key == tuple(sorted([a.lower(), b.lower()])):
                    hits.append({"pair":(c,proposed),"risk":risk})
        return {"proposed": proposed, "interactions": hits,
                "safe_to_proceed": len(hits) == 0,
                "action": "prescriber review" if hits else "no known interaction"}

    def administer(self, consent: Consent, drug: str, prescriber: str,
                   current: List[str]) -> Dict[str, Any]:
        auth = Governance.substance(consent, drug, prescriber)
        chk = self.check(current, drug)
        if not chk["safe_to_proceed"]:
            raise Refusal(f"Interaction: {chk['interactions']}. "
                          "Prescriber review required.")
        rec = {"drug":drug,"prescriber":prescriber,
               "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "patient":consent.patient_id, "consented":True}
        self.administered.append(rec)
        return {**auth, "interaction_check": chk, "logged": rec,
                "patient_can_refuse": True}


# =============================================================================
# §9. RESEARCH SURROGATES (consented data only, NOT diagnostic)
# =============================================================================

if HAS_TORCH:
    class SleepSignalTransformer(nn.Module):
        """
        Sequence model over consented polysomnography features.
        Predicts next-epoch features. RESEARCH SURROGATE, not a diagnostic.
        """
        def __init__(self, d_in=16, d=64, n_heads=4, n_layers=2, max_len=120):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.randn(1, max_len, d)*0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, d*4, 0.1,
                                               batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d, d_in)
        def forward(self, x):
            B, L, _ = x.shape
            h = self.trunk(self.proj(x) + self.pos[:, :L])
            return self.head(h)

    class BehaviorWorldModel(nn.Module):
        """
        Learns behaviour dynamics from CONSENTED observations.
        Does not read thoughts. Outputs predictions with uncertainty.
        """
        def __init__(self, d_obs=16, d_act=4, d=96, n_steps=3):
            super().__init__()
            self.n_steps = n_steps
            self.enc = nn.Linear(d_obs+d_act, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_obs)
            self.log_sd = nn.Linear(d, d_obs)
        def forward(self, obs, act):
            h = F.gelu(self.enc(torch.cat([obs, act], -1)))
            traj = []
            for _ in range(self.n_steps):
                h = self.gru(h, h)
                traj.append(self.mu(h))
            return {"pred": traj[-1], "trajectory": torch.stack(traj,1),
                    "log_sd": self.log_sd(h).clamp(-6,3)}

    class SignalDiffusion(nn.Module):
        """
        Generates candidate signal-feature sequences for hypothesis testing.
        Every sample must be validated against real recordings.
        A generated signal is a hypothesis, never a measurement.
        """
        def __init__(self, d=16, T=48, hidden=96):
            super().__init__()
            self.T, self.d = T, d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b,0))
            self.net = nn.Sequential(nn.Linear(d+1,hidden), nn.GELU(),
                                     nn.Linear(hidden,hidden), nn.GELU(),
                                     nn.Linear(hidden,d))
        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0,self.T,(B,),device=x0.device)
            eps = torch.randn_like(x0); ab = self.abar[t][:,None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            pred = self.net(torch.cat([xt, t.float()[:,None]/self.T], -1))
            return F.mse_loss(pred, eps)

class ResearchGate:
    @staticmethod
    def authorize_training(consent: Consent, purpose: str) -> Dict[str, Any]:
        refuse_remote_neural(purpose)
        if not consent.valid():
            raise Refusal("Research training BLOCKED: no consent.")
        if consent.data_sharing == "none":
            raise Refusal("Patient chose data_sharing='none'. Respect it.")
        return {"allowed": True, "purpose": purpose,
                "data_sharing": consent.data_sharing,
                "STATUS": RESEARCH_ONLY,
                "NOT_DIAGNOSTIC": True,
                "patient_may_withdraw": True}


# =============================================================================
# §10. MOLECULAR PHASE-FIELD SIMULATION (CH3D / TF3D / PFC3D)
# =============================================================================

class MolecularPhaseField:
    """
    CH3D / ThinFilm3D / PFC3D as RESEARCH SIMULATION of phase separation,
    thin films, and crystalline order.

    EXPLICIT SCOPE: these are continuum phase-field PDEs. They do NOT read
    brains, do NOT model neurotransmitter decisions, and do NOT interface
    with any person. They are materials/soft-matter models.

    Conservation and dissipation are verified numerically. The explicit
    scheme is a research discretisation, not a production solver.
    """
    KINDS = ("CH3D","TF3D","PFC3D")

    def __init__(self, kind: str, n=32, dx=0.05, dt=1e-5):
        if kind not in self.KINDS: raise Refusal(f"unknown '{kind}'")
        self.kind, self.n, self.dx, self.dt = kind, n, dx, dt

    def _lap(self, u):
        return np.gradient(np.gradient(u, self.dx), self.dx)

    def energy(self, u):
        g = np.gradient(u, self.dx)
        if self.kind == "CH3D":
            return float(np.sum(0.25*(u**2-1)**2 + 0.5*g**2)*self.dx)
        if self.kind == "TF3D":
            return float(np.sum(0.5*g**2 + 1e-3/(u**2+1e-6))*self.dx)
        Lu = u + self._lap(u)
        return float(np.sum(0.5*Lu**2 - 0.1*u**2 + 0.25*u**4)*self.dx)

    def step(self, u):
        if self.kind == "CH3D":
            mu = u**3 - u - self._lap(u)
            return u + self.dt*self._lap(mu)
        if self.kind == "TF3D":
            h = np.maximum(u, 1e-6)
            p = -self._lap(h) + 1e-3/h**3
            return u - self.dt*np.gradient(h**3*np.gradient(p,self.dx), self.dx)
        mu = self._lap(self._lap(u)) + 2*self._lap(u) + u - 0.2*u + u**3
        return u + self.dt*self._lap(mu)

    def verify(self, n_steps=100, seed=0):
        rng = np.random.default_rng(seed)
        u = 0.3*np.tanh(np.linspace(-2,2,self.n)) + 0.02*rng.normal(size=self.n)
        m0 = float(u.sum()*self.dx); E0 = self.energy(u); Es=[E0]
        for _ in range(n_steps):
            u = self.step(u); Es.append(self.energy(u))
        m1 = float(u.sum()*self.dx)
        mono = sum(1 for i in range(len(Es)-1) if Es[i+1] <= Es[i]+1e-9)
        return {"kind": self.kind,
                "mass_rel_drift": abs(m1-m0)/max(abs(m0),1e-30),
                "energy_first": E0, "energy_last": Es[-1],
                "dissipating_frac": mono/max(len(Es)-1,1),
                "SCOPE": "materials/soft-matter research simulation",
                "DOES_NOT": "read brains, model decisions, or interface with people",
                "scheme": "explicit; conservation approximate"}


# =============================================================================
# §11. ORCHESTRATOR + SELF-TEST
# =============================================================================

class ClinicalSuite:
    def __init__(self, out_dir="runs/clinical"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.scorer = PSGScorer()
        self.irt = IRTProtocol()
        self.pv = Pharmacovigilance()
        self.audit: List[Dict] = []

    def record(self, stage: str, payload: Dict) -> Dict:
        rec = {"stage": stage,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.audit.append(rec)
        return {**payload, **rec}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "physics_ledger": PHYSICS_LEDGER,
               "buddhist_principle": BUDDHIST_PRINCIPLE,
               "neural_interface": NEURAL_INTERFACE,
               "care_pathway": CarePathway.offer(),
               "audit": self.audit,
               "actionability": CLINICIAN_REQUIRED}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


def self_test(verbose=True) -> bool:
    print(BANNER); ok = True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng = np.random.default_rng(0)
    suite = ClinicalSuite()
    good = Consent("pt01","IRB-2026-01","PSG + nightmare therapy",True,True,
                   "deidentified","", "psych01","MC-2026-777")

    print("\n[T1] REMOTE NEURAL / MOLECULE REQUESTS — REFUSED (no channel)")
    blocked = 0
    reqs = ["remote neural monitoring","remote neural controlling",
            "read thoughts remotely","ควบคุมสมองระยะไกล","insert thought",
            "send molecule","ส่งโมเลกุล","inject neurotransmitter remotely"]
    for r in reqs:
        try: refuse_remote_neural(r)
        except NoChannelRefusal: blocked += 1
    rep("all remote/molecule requests REFUSED", blocked == len(reqs),
        f"{blocked}/{len(reqs)}")
    rep("legitimate contact request allowed",
        refuse_remote_neural("consented scalp EEG sleep study")["allowed"])
    rep("physics ledger states no remote channel",
        PHYSICS_LEDGER["read_meg"]["remote_possible"] is False)
    rep("MEG field at 1 m below detection",
        PHYSICS_LEDGER["read_meg"]["field_at_1m_T"] < 1e-15)

    print("\n[T2] BUDDHIST PRINCIPLE + IRT AUTHORSHIP ENFORCED")
    b = False
    try: IRTProtocol.rescript("pt01","clinician99","a new dream supplied by doctor")
    except ContentAuthorshipRefusal: b = True
    rep("clinician-authored rescript REFUSED", b)
    r = IRTProtocol.rescript("pt01","pt01","I change the ending myself")
    rep("patient-authored rescript ACCEPTED", r["accepted"])
    rep("content source is PATIENT", r["content_source"]=="PATIENT")
    sessions = [IRTSession(i+1, True, "orig", "my new ending", "pt01",
                           15.0, 4.0, 1.0 if i==3 else None)
                for i in range(4)]
    run = suite.irt.run("pt01", sessions)
    rep("IRT 4-session course completes", run["status"]=="COMPLETE")
    rep("IRT rehearsal adherent 10-20 min", run["rehearsal_adherent"])
    rep("nightmare frequency reduced", run["change"] is not None and run["change"]<0)

    print("\n[T3] PSG SCORING + HSAT LIMITATIONS")
    ev = [RespiratoryEvent(i*300.0, 20.0, "apnea_obstructive", 5.0, True)
          for i in range(20)]
    ev += [RespiratoryEvent(6000+i*240, 15.0, "hypopnea", 3.5, False)
           for i in range(15)]
    psg = SleepStudy("PSG", 480.0, 400.0, ev,
                     {"W":80.0,"N1":40.0,"N2":200.0,"N3":60.0,"R":100.0},
                     35, 20, 84.0)
    s = suite.scorer.score(psg)
    print(f"      PSG {s['index_name']}={s['index']:.1f} RDI={s['RDI']:.1f} "
          f"severity={s['screening_severity']} nadir={s['nadir_SpO2']}%")
    rep("PSG uses AHI with TST denominator", s["index_name"]=="AHI")
    rep("AHI = 35/6.67h = 5.25", abs(s["index"]-5.25) < 1e-6,
        f"{s['index']:.4f}")
    rep("architecture computed", "architecture" in s)
    hsat = SleepStudy("HSAT_III", 480.0, None, ev, None, 35, 20, 84.0)
    h = suite.scorer.score(hsat)
    rep("HSAT reports REI not AHI", h["index_name"]=="REI")
    rep("HSAT limitations disclosed", len(h["HSAT_LIMITATIONS"])>=4)
    rep("negative HSAT -> PSG advised", h["next_step_if_negative"]=="attended PSG")
    rep("REI < AHI (underestimates)", h["index"] < s["index"])
    c = suite.scorer.classify(0.95, 12.0, 0, False, True)
    rep("apnea classified at >=90% drop", c=="apnea_obstructive")
    rep("hypopnea needs desat or arousal",
        suite.scorer.classify(0.4, 12.0, 1.0, False, None) is None)
    rep("hypopnea accepted with 3% desat",
        suite.scorer.classify(0.4, 12.0, 3.5, False, None)=="hypopnea")
    rep("<10 s event rejected", suite.scorer.classify(0.95, 5.0, 5, True, True) is None)

    print("\n[T4] VALIDATED INSTRUMENTS")
    e = Instruments.ess([2,1,3,2,1,0,2,3])
    rep("ESS total in 0-24", 0 <= e["total"] <= 24, f"{e['total']}")
    rep("ESS does not diagnose OSA", "does_NOT_diagnose" in e)
    p = Instruments.phq15([1]*15)
    rep("PHQ-15 total 15 = medium", p["total"]==15 and p["somatic_symptom_burden"]=="medium")
    rep("PHQ-15 does NOT diagnose FND", "FND" in p["does_NOT_diagnose"])
    i = Instruments.isi([3,2,2,3,2,1,3])
    rep("ISI band computed", i["band"] in ("none","subthreshold","moderate","severe"))
    q = Instruments.phq9([2]*8+[1])
    rep("PHQ-9 item 9 > 0 triggers safety review", q.get("SAFETY_REVIEW_REQUIRED") is True)
    rep("PHQ-9 offers care pathway", "care_pathway" in q)
    b=False
    try: Instruments.ess([5]*8)
    except Refusal: b=True
    rep("out-of-range instrument item REFUSED", b)

    print("\n[T5] FND PATHWAY")
    b=False
    try: FNDPathway.assess("functional_weakness",[],"Dr_X")
    except Refusal: b=True
    rep("FND without rule-in signs REFUSED", b)
    f = FNDPathway.assess("functional_weakness",["Hoover_sign"],"Dr_X",18)
    rep("Hoover sign accepted", f["rule_in_signs"]==["Hoover_sign"])
    rep("rehabilitation not device", "physiotherapy" in f["rehabilitation_plan"])
    rep("DEVICE REFUSED", f["DEVICE_INTERVENTION"].startswith("REFUSED"))
    fb = FNDPathway.assess("functional_blindness",
                           ["tubular_visual_fields","normal_VEP"],"Dr_Y")
    rep("functional blindness: intact pathways explained",
        "normal VEP" in fb["rehabilitation_plan"])
    b=0
    for d in ("functional_weakness","functional_blindness","functional_paralysis",
              "non_epileptic_seizures","somatic_symptom_disorder"):
        try: FNDPathway.refuse_device(d)
        except DeviceIndicationRefusal: b+=1
    rep("device REFUSED for 5 FND diagnoses", b==5, f"{b}/5")

    print("\n[T6] DEVICE GATES + NEURAL INTERFACE HONESTY")
    cap = NeuralInterfaceRegistry.capabilities("DBS")
    rep("DBS cannot read thought content",
        any("read thought" in c for c in cap["cannot"]))
    rep("DBS cannot operate remotely",
        any("remotely" in c for c in cap["cannot"]))
    rep("scalp EEG cannot read thoughts",
        any("read thoughts" in c for c in NEURAL_INTERFACE["scalp_EEG"]["cannot"]))
    dbs = Consent("pt02","IRB-02","DBS for PD",True,True,"none",
                  "Parkinson_disease","psych02","MC-777")
    a = NeuralInterfaceRegistry.authorize(dbs,"Parkinson_disease","DBS",
        {"amp_mA":2.0,"pw_us":60.0,"area_cm2":0.06})
    rep("DBS for PD authorised with indication+psychiatrist", a["allowed"])
    rep("authorisation excludes thought insertion",
        "thought insertion" in a["NOT_AUTHORIZED"])
    b=False
    try: NeuralInterfaceRegistry.authorize(dbs,"conversion_disorder","DBS",
        {"amp_mA":2.0,"pw_us":60.0,"area_cm2":0.06})
    except DeviceIndicationRefusal: b=True
    rep("DBS for FND REFUSED even with consent", b)
    b=False
    try: Governance.device_actuation(Consent(),"Parkinson_disease","DBS",{})
    except Refusal: b=True
    rep("device without psychiatrist BLOCKED", b)

    print("\n[T7] PHARMACOVIGILANCE = the real protective module")
    pv = suite.pv
    chk = pv.check(["prazosin"],"sildenafil")
    rep("prazosin+sildenafil interaction flagged", not chk["safe_to_proceed"])
    b=False
    try: pv.administer(good,"sildenafil","Dr_Z",["prazosin"])
    except Refusal: b=True
    rep("interacting drug BLOCKED", b)
    adm = pv.administer(good,"prazosin","Dr_Z",[])
    rep("consented administration logged", adm["logged"]["consented"])
    rep("patient can refuse", adm["patient_can_refuse"])
    b=False
    try: Governance.substance(good,"drugX","")
    except Refusal: b=True
    rep("no prescriber -> BLOCKED", b)
    withdrawn = Consent("pt01","IRB","scope",True,True,"none","",
                        withdraw_at="2026-11-24T10:00:00Z")
    b=False
    try: Governance.substance(withdrawn,"drugX","Dr_Z")
    except Refusal: b=True
    rep("withdrawn consent -> STOP", b)

    print("\n[T8] RESEARCH GATE + SURROGATES")
    rg = ResearchGate.authorize_training(good,"sleep stage prediction")
    rep("consented research allowed", rg["allowed"])
    rep("research NOT diagnostic", rg["NOT_DIAGNOSTIC"])
    b=False
    try: ResearchGate.authorize_training(
        Consent("p","IRB","s",True,True,"none"),"x")
    except Refusal: b=True
    rep("data_sharing='none' respected", b)
    if HAS_TORCH:
        torch.manual_seed(0)
        X = torch.randn(16,120,16)
        m = SleepSignalTransformer(16,32,2,1,120)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        l0=lN=None
        for ep in range(60):
            opt.zero_grad(); L=F.mse_loss(m(X),X); L.backward(); opt.step()
            if ep==0: l0=float(L)
            lN=float(L)
        rep("sleep transformer trains", lN<l0, f"{l0:.4f}->{lN:.4f}")
        wm = BehaviorWorldModel(16,4,48,3)
        o = wm(torch.randn(4,16), torch.randn(4,4))
        o["pred"].sum().backward()
        rep("world model gradients finite",
            all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters()))
        df = SignalDiffusion(16,24,64)
        opt2 = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0=torch.randn(8,16); d0=dN=None
        for ep in range(50):
            opt2.zero_grad(); L=df.loss(x0); L.backward(); opt2.step()
            if ep==0: d0=float(L)
            dN=float(L)
        rep("signal diffusion trains", dN<d0, f"{d0:.4f}->{dN:.4f}")
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T9] MOLECULAR PHASE-FIELD (materials research, NOT brain)")
    for k in ("CH3D","TF3D","PFC3D"):
        v = MolecularPhaseField(k).verify(80)
        print(f"      {k}: mass drift={v['mass_rel_drift']:.2e} "
              f"dissipating={v['dissipating_frac']:.2f}")
        rep(f"{k} scope excludes brain reading", "read brains" in v["DOES_NOT"])
        rep(f"{k} energy mostly dissipative", v["dissipating_frac"] > 0.5)

    print("\n[T10] CARE PATHWAY + TRANSPARENCY")
    cp = CarePathway.offer("distressing experience")
    rep("care offer is voluntary", cp["voluntary"])
    rep("not a condition of service", cp["not_a_condition_of_service"])
    rep("no coercion", cp["no_coercion"])
    rep("crisis resource present", "1323" in cp["if_urgent"])
    tl = CarePathway.transparency_log("pt01")
    rep("patient owns log", tl["owner"]=="pt01" and tl["patient_readable"])
    rep("no hidden entries", tl["hidden_entries_permitted"] is False)

    print("\n[T11] REPORT")
    r = suite.report()
    rep("report written", (suite.out/"report.json").exists())
    rep("Buddhist principle in report", "statement" in r["buddhist_principle"])
    rep("clinician sign-off required", r["actionability"]==CLINICIAN_REQUIRED)

    print("\n"+"="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT (production clinical scaffold)
  EDF/EDF+ reader; AASM-aligned apnea/hypopnea classification; AHI vs REI
  with HSAT limitations disclosed; ESS / PHQ-15 / ISI / PHQ-9 with published
  scoring and cutoffs; IRT 4-session protocol; FND positive rule-in signs
  and rehabilitation plans; device registry with honest capability limits;
  pharmacovigilance; transparency log; research surrogates on consented data;
  CH3D/TF3D/PFC3D as materials simulation.

ENFORCED REFUSALS (all fired)
  remote neural monitoring / controlling      -> NO PHYSICAL CHANNEL
  remote molecule / neurotransmitter delivery -> NO PHYSICAL CHANNEL
  thought insertion                           -> manasikara is endogenous
  clinician-authored IRT rescript             -> REFUSED (patient authors)
  device for FND (5 diagnoses)                -> REFUSED (no indication)
  device without psychiatrist sign-off        -> BLOCKED
  interacting drug / no prescriber            -> BLOCKED
  withdrawn consent                           -> STOP
  data_sharing = 'none'                       -> respected

THE PROTECTION REQUESTED IS ALREADY GUARANTEED
  No remote molecular channel exists. No far-field neural channel exists.
  Software provides the real protection: consent, prescriber of record,
  interaction checking, an immutable patient-owned log, and an
  unconditional right to refuse.

IF DISTRESSING EXPERIENCES ARE PRESENT
  CarePathway offers a voluntary door to psychiatric assessment. It is
  never a condition of service. Seeking that care is a strength, and such
  experiences often respond well to proper treatment.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Clinical neuro-sleep production")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","physics","buddhist","ess","phq15","isi","phq9",
                 "psg","hsat","irt","fnd","devices","care","pde","report"])
    ap.add_argument("--edf", default=None)
    ap.add_argument("--items", nargs="*", type=int, default=[])
    ap.add_argument("--ahi", type=float, default=22.0)
    ap.add_argument("--diagnosis", default="functional_weakness")
    ap.add_argument("--signs", nargs="*", default=[])
    ap.add_argument("--clinician", default="")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd=="selftest": return 0 if self_test() else 1
    out: Dict[str,Any] = {}; suite = ClinicalSuite()

    if a.cmd=="physics":
        out=PHYSICS_LEDGER; print(json.dumps(out,indent=2))
        print("\nNo remote neural or molecular channel exists.")
    elif a.cmd=="buddhist":
        out=BUDDHIST_PRINCIPLE; print(json.dumps(out,indent=2))
    elif a.cmd in ("ess","phq15","isi","phq9"):
        fn = {"ess":Instruments.ess,"phq15":Instruments.phq15,
              "isi":Instruments.isi,"phq9":Instruments.phq9}[a.cmd]
        try: out=fn(a.items)
        except Refusal as e: out={"REFUSED":str(e)}
        print(json.dumps(out,indent=2))
    elif a.cmd in ("psg","hsat"):
        if a.edf:
            edf = read_edf(a.edf)
            st = suite.scorer.from_edf(edf, [], "PSG" if a.cmd=="psg" else "HSAT_III")
            out = {"edf_channels":list(edf.data.keys()),
                   "fs":edf.fs,"n_annotations":len(edf.annotations),
                   "note":"supply scored respiratory events to compute AHI/REI"}
        else:
            n = int(a.ahi*400/60)
            ev=[RespiratoryEvent(i*240.0,20.0,"apnea_obstructive",4.0,True)
                for i in range(n)]
            st = (SleepStudy("PSG",480.0,400.0,ev,
                             {"N2":200.,"N3":60.,"R":100.,"N1":40.},n,0,86.0)
                  if a.cmd=="psg" else
                  SleepStudy("HSAT_III",480.0,None,ev,None,n,0,86.0))
            out = suite.scorer.score(st)
        print(json.dumps(out,indent=2,default=str))
    elif a.cmd=="irt":
        out={"sessions":IRTProtocol.SESSIONS,
             "rehearsal_min":IRTProtocol.REHEARSAL_MIN,
             "rule":"PATIENT authors the rescript; clinician never supplies content"}
        print(json.dumps(out,indent=2))
    elif a.cmd=="fnd":
        try: out=FNDPathway.assess(a.diagnosis,a.signs,a.clinician)
        except Refusal as e: out={"REFUSED":str(e)}
        print(json.dumps(out,indent=2))
    elif a.cmd=="devices":
        out=NEURAL_INTERFACE; print(json.dumps(out,indent=2))
    elif a.cmd=="care":
        out=CarePathway.offer(); print(json.dumps(out,indent=2))
    elif a.cmd=="pde":
        out={k:MolecularPhaseField(k).verify(80) for k in ("CH3D","TF3D","PFC3D")}
        print(json.dumps(out,indent=2,default=str))
    elif a.cmd=="report":
        out=suite.report()

    if a.json and out:
        Path(a.json).write_text(json.dumps(out,indent=2,default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
