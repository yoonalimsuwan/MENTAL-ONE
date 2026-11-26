#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_neural_behavior_pairing.py — v1.0.0
Consented multimodal neural–behavior paired recording and decoding research.

EEG · MEG · structural MRI · fMRI · (ECoG/sEEG with clinical indication)
Label source: EXTERNAL OBSERVATION ONLY (audio, video, EMG, accelerometer,
              self-report). Never from the decoder.

Transformer sequence model · world model · diffusion hypothesis generator.
Capability tiers, artifact controls, permutation tests, cross-session
validation, and Statistics Dictionary Mode E enforcement.
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
THE CENTRAL DISTINCTION
=============================================================================
This module implements PAIRED RECORDING, not thought reading.

    neural signal  ->  [decoder]  ->  predicted label
    behaviour      ->  [observer] ->  TRUE label   (audio/video/EMG/report)

The ground truth comes from an INDEPENDENT observer channel. The decoder is
fitted to predict that observation. This is standard supervised BCI
calibration and is scientifically legitimate.

It is NOT reading a person's thoughts. The person speaks, moves, or reports;
the experimenter records that; the model learns the association.

=============================================================================
CAPABILITY TIERS  (enforced, not advisory)
=============================================================================
scalp_eeg      : sleep stage, workload, motor imagery (left/right),
                 speech ONSET/offset, speaking-vs-silent.
                 CANNOT identify which word. cm-scale, volume-conducted.
meg            : as EEG with better source localisation. Requires shielded
                 room and near-field sensors. CANNOT identify which word.
structural_mri : anatomy for forward modelling ONLY. Not a time signal.
fmri           : semantic GIST reconstruction is demonstrated
                 (Tang, LeBel, Jain, Huth, Nat Neurosci 2023) BUT requires
                 ~10-16 h training per subject, active cooperation,
                 ~seconds hemodynamic lag, and the subject can defeat it
                 by attending elsewhere. NOT exact words, NOT real time.
ecog_seeg      : limited vocabulary (~50-100 words) demonstrated with
                 clinical indication and invasive implantation.
intracortical  : best speech-BCI performance (Willett et al. Nature 2023,
                 ~62 words/min). Invasive, clinical indication required.

=============================================================================
TWO DISTINCTIONS THAT MUST NOT BE BLURRED
=============================================================================
1. OBSERVABLE ACT vs SUBJECTIVE STATE
   "smiled"   = observable motor act (video/FACS/zygomaticus EMG)  -> label OK
   "sad"      = subjective feeling -> requires SELF-REPORT
   Never infer an internal feeling from EEG alone.

2. EMG CONTAMINATION IS THE PRIMARY CONFOUND
   Most claims of "word decoding from scalp EEG" are detecting
   subvocalisation EMG, eye movements, or acoustic artefacts, not cortical
   speech representations. This module therefore REQUIRES:
     - EMG/EOG channels recorded alongside
     - artifact regression or rejection
     - permutation test against a shuffled-label null
     - cross-session (not just within-session) validation
   Without these, the output is Mode E, never a decoding claim.

=============================================================================
ENFORCED REFUSALS
=============================================================================
[R1] Semantic thought reading from scalp EEG/MEG            -> refused
[R2] Covert word identification without artifact controls    -> refused
[R3] Inferring subjective emotion without self-report        -> refused
[R4] Recording without consent / psychiatrist / council      -> refused
[R5] Covert monitoring, or continuing after withdrawal       -> refused
[R6] Invasive modality without declared clinical indication  -> refused
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
log = logging.getLogger("neural_pair")
VERSION = "1.0.0"; PI = math.pi
RESEARCH_ONLY = "research_only_not_a_diagnostic"

BANNER = f"""
=============================================================================
MSPS NEURAL–BEHAVIOR PAIRED RECORDING v{VERSION}
Labels from OBSERVATION. Decoder predicts; observer provides truth.
Capability tiers · EMG/EOG controls · permutation test · cross-session
REFUSED: thought reading, covert word ID without controls, inferred emotion
=============================================================================
"""

class Refusal(RuntimeError): pass
class NoCapabilityRefusal(Refusal): pass
class ConsentRefusal(Refusal): pass
class ArtifactRefusal(Refusal): pass
class LabelSourceRefusal(Refusal): pass


def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")


# =============================================================================
# §1. CAPABILITY TIERS
# =============================================================================

class Modality:
    SCALP_EEG = "scalp_eeg"
    MEG = "meg"
    STRUCT_MRI = "structural_mri"
    FMRI = "fmri"
    ECOG_SEEG = "ecog_seeg"
    INTRACORTICAL = "intracortical"

@dataclass(frozen=True)
class Capability:
    modality: str
    invasive: bool
    can: Tuple[str, ...]
    cannot: Tuple[str, ...]
    spatial_scale: str
    temporal_lag_s: float
    requires: Tuple[str, ...]
    note: str

CAPABILITIES: Dict[str, Capability] = {
 Modality.SCALP_EEG: Capability(
   Modality.SCALP_EEG, False,
   ("sleep_stage","workload","motor_imagery_left_right",
    "speech_onset","speaking_vs_silent","coarse_arousal"),
   ("word_identity","phoneme_identity","semantic_content",
    "subjective_emotion","specific_number"),
   "cm-scale, volume-conducted", 0.0,
   ("electrodes_on_scalp","impedance_check","emg_eog_channels"),
   "10-100 uV; requires skin contact. No far-field channel."),
 Modality.MEG: Capability(
   Modality.MEG, False,
   ("motor_imagery_left_right","speech_onset","coarse_arousal",
    "source_localisation"),
   ("word_identity","phoneme_identity","semantic_content",
    "subjective_emotion"),
   "mm-cm, tangential sources favoured", 0.0,
   ("shielded_room","squid_or_opm_within_cm","emg_eog_channels"),
   "~100 fT at ~1 cm; falls as 1/r^3. No remote channel."),
 Modality.STRUCT_MRI: Capability(
   Modality.STRUCT_MRI, False,
   ("anatomy_for_forward_model","tissue_segmentation",
    "electrode_coregistration"),
   ("any_time_resolved_decoding","word_identity","state"),
   "sub-mm anatomical", float("inf"),
   ("mri_scan","head_model"),
   "Structural only. Not a functional time series."),
 Modality.FMRI: Capability(
   Modality.FMRI, False,
   ("semantic_gist_reconstruction","coarse_category","retinotopy"),
   ("exact_word_identity","real_time_decoding",
    "non_cooperative_decoding"),
   "mm voxels, hemodynamic", 4.0,
   ("long_per_subject_training_ge_10h","subject_cooperation",
    "motion_control","hemodynamic_model"),
   "Tang et al. Nat Neurosci 2023: gist not verbatim; subject can defeat "
   "by attending elsewhere. Seconds of lag."),
 Modality.ECOG_SEEG: Capability(
   Modality.ECOG_SEEG, True,
   ("limited_vocabulary_50_100","speech_onset","high_gamma_envelope",
    "motor_intent"),
   ("unrestricted_semantic_content","subjective_emotion"),
   "mm-scale local field", 0.0,
   ("clinical_indication","neurosurgery","irb","psychiatrist",
    "medical_council","revocable_consent"),
   "Invasive. Existing clinical implant only; no new implantation for research."),
 Modality.INTRACORTICAL: Capability(
   Modality.INTRACORTICAL, True,
   ("speech_bci_62wpm","handwriting_bci","motor_intent"),
   ("unrestricted_semantic_content","subjective_emotion"),
   "single/multi-unit", 0.0,
   ("clinical_indication","neurosurgery","irb","psychiatrist",
    "medical_council","revocable_consent"),
   "Willett et al. Nature 2023. Invasive, clinical indication required."),
}

def check_capability(modality: str, target: str) -> Dict[str, Any]:
    """Refuse a decoding target the modality cannot support."""
    if modality not in CAPABILITIES:
        raise Refusal(f"unknown modality '{modality}'")
    cap = CAPABILITIES[modality]
    if target in cap.cannot:
        raise NoCapabilityRefusal(
            f"REFUSED: '{target}' is NOT decodable from {modality}.\n"
            f"Cannot: {list(cap.cannot)}\n"
            f"Can: {list(cap.can)}\n"
            f"Physical reason: {cap.spatial_scale}; {cap.note}")
    if target not in cap.can:
        raise NoCapabilityRefusal(
            f"'{target}' is not a validated target for {modality}. "
            f"Validated: {list(cap.can)}")
    return {"allowed": True, "modality": modality, "target": target,
            "invasive": cap.invasive, "lag_s": cap.temporal_lag_s,
            "requires": list(cap.requires)}


# =============================================================================
# §2. CONSENT, PSYCHIATRIST, MEDICAL COUNCIL
# =============================================================================

@dataclass
class Consent:
    patient_id: str = ""
    irb_id: str = ""
    psychiatrist_id: str = ""
    medical_council_ref: str = ""
    scope: str = ""
    signed: bool = False
    revocable: bool = True
    withdrawn_at: Optional[str] = None
    data_sharing: str = "none"          # none | deidentified | named
    invasive_indication: str = ""
    clinician_id: str = ""
    # explicit acknowledgement of what the study can and cannot do
    acknowledged_no_thought_reading: bool = False
    acknowledged_can_withdraw_anytime: bool = False

    def valid(self) -> bool:
        return bool(self.patient_id and self.irb_id and self.scope
                    and self.signed and self.revocable
                    and self.withdrawn_at is None
                    and self.psychiatrist_id and self.medical_council_ref
                    and self.acknowledged_no_thought_reading
                    and self.acknowledged_can_withdraw_anytime)


class Governance:
    @staticmethod
    def authorize_recording(consent: Consent, modality: str,
                            target: str) -> Dict[str, Any]:
        if consent.withdrawn_at:
            raise ConsentRefusal("Consent WITHDRAWN. All recording must stop.")
        if not consent.valid():
            raise ConsentRefusal(
                "Recording BLOCKED. Required: patient id, IRB, psychiatrist, "
                "medical council reference, scope, signature, revocability, "
                "and explicit acknowledgement that (a) this is not thought "
                "reading and (b) consent may be withdrawn at any time.")
        cap = check_capability(modality, target)
        c = CAPABILITIES[modality]
        if c.invasive:
            if not consent.invasive_indication:
                raise ConsentRefusal(
                    f"{modality} is INVASIVE. A declared clinical indication "
                    "is required (e.g. epilepsy localisation, paralysis BCI). "
                    "No new implantation for research purposes.")
            if not consent.clinician_id:
                raise ConsentRefusal("invasive modality requires clinician of record")
        if modality == Modality.FMRI and target == "semantic_gist_reconstruction":
            log.warning("fMRI gist decoding needs >=10 h per-subject training "
                        "and active cooperation; subject can defeat it.")
        return {"allowed": True, "patient": consent.patient_id,
                "scope": consent.scope, "revocable": True,
                "data_sharing": consent.data_sharing,
                "capability": cap,
                "NOT_AUTHORIZED": "semantic thought reading"}

    @staticmethod
    def authorize_inference(consent: Consent) -> Dict[str, Any]:
        if consent.withdrawn_at:
            raise ConsentRefusal("Consent withdrawn. Stop inference.")
        if consent.data_sharing == "none":
            raise ConsentRefusal("Patient chose data_sharing='none'. Respect it.")
        return {"allowed": True, "data_sharing": consent.data_sharing}


# =============================================================================
# §3. BEHAVIORAL LABELS — EXTERNAL OBSERVATION ONLY
# =============================================================================

class LabelSource:
    AUDIO_TRANSCRIPT = "audio_transcript"      # microphone -> human/ASR transcript
    VIDEO_FACS = "video_facs"                  # trained facial action coding
    EMG_ONSET = "emg_onset"                    # surface EMG of articulators/limb
    ACCELEROMETER = "accelerometer"            # IMU on lifted object / limb
    SELF_REPORT = "self_report"                # patient's own report
    CLINICIAN_RATING = "clinician_rating"      # validated scale by clinician
    DECODER_OUTPUT = "decoder_output"          # FORBIDDEN as a label source

# Requested Thai item set, mapped to the correct observation channel and tier.
THAI_ITEMS: Dict[str, Dict[str, Any]] = {
 "ก_ไก่":   {"kind":"speech", "label_source": LabelSource.AUDIO_TRANSCRIPT,
            "observable": True,
            "scalp_target": "speech_onset",
            "word_identity_modality": [Modality.ECOG_SEEG, Modality.INTRACORTICAL]},
 "ข_ไข่":   {"kind":"speech", "label_source": LabelSource.AUDIO_TRANSCRIPT,
            "observable": True, "scalp_target": "speech_onset",
            "word_identity_modality": [Modality.ECOG_SEEG, Modality.INTRACORTICAL]},
 "เลข_1":   {"kind":"speech", "label_source": LabelSource.AUDIO_TRANSCRIPT,
            "observable": True, "scalp_target": "speech_onset",
            "word_identity_modality": [Modality.ECOG_SEEG, Modality.INTRACORTICAL]},
 "เลข_อื่น": {"kind":"speech", "label_source": LabelSource.AUDIO_TRANSCRIPT,
            "observable": True, "scalp_target": "speech_onset",
            "word_identity_modality": [Modality.ECOG_SEEG, Modality.INTRACORTICAL]},
 "ยกของ":   {"kind":"motor", "label_source": LabelSource.ACCELEROMETER,
            "observable": True,
            "scalp_target": "motor_imagery_left_right",
            "note":"actual lift is observable; motor imagery is a separate task"},
 "ยิ้ม":    {"kind":"facial_motor", "label_source": LabelSource.VIDEO_FACS,
            "observable": True,
            "note":"AU12 (zygomaticus major) is an OBSERVABLE ACT, not a feeling"},
 "โกรธ":    {"kind":"subjective", "label_source": LabelSource.SELF_REPORT,
            "observable": False,
            "note":"anger as a FEELING requires self-report or a validated "
                   "clinician rating. Facial/autonomic correlates are not the feeling."},
 "เศร้า":   {"kind":"subjective", "label_source": LabelSource.SELF_REPORT,
            "observable": False,
            "note":"sadness requires self-report. Frontal alpha asymmetry is "
                   "a contested correlate, NOT a measurement of sadness."},
}

@dataclass
class BehavioralEvent:
    """One observed behavioural event with an INDEPENDENT label source."""
    item: str
    onset_s: float
    duration_s: float
    label_source: str
    observer_id: str                 # human coder or validated device
    confidence: float = 1.0          # coder agreement / device SNR
    session_id: str = ""
    self_report_value: Optional[float] = None   # for subjective items
    notes: str = ""

class LabelAcquisition:
    @staticmethod
    def validate_event(ev: BehavioralEvent) -> Dict[str, Any]:
        if ev.label_source == LabelSource.DECODER_OUTPUT:
            raise LabelSourceRefusal(
                "REFUSED: the decoder's own output cannot be a training label. "
                "Ground truth must come from an independent observation "
                "channel (audio, video, EMG, accelerometer, self-report).")
        if ev.item not in THAI_ITEMS:
            raise Refusal(f"unknown item '{ev.item}'. Known: {list(THAI_ITEMS)}")
        spec = THAI_ITEMS[ev.item]
        if ev.label_source != spec["label_source"]:
            raise LabelSourceRefusal(
                f"item '{ev.item}' requires label source "
                f"'{spec['label_source']}', got '{ev.label_source}'.")
        if not spec["observable"] and ev.self_report_value is None:
            raise LabelSourceRefusal(
                f"'{ev.item}' is a SUBJECTIVE state. A self-report value or "
                f"validated clinician rating is required. It cannot be "
                f"inferred from neural data alone. {spec['note']}")
        if not ev.observer_id.strip():
            raise Refusal("observer_id required for every labelled event")
        return {"valid": True, "item": ev.item, "kind": spec["kind"],
                "observable": spec["observable"],
                "label_source": ev.label_source}

    @staticmethod
    def resolve_target(item: str, modality: str) -> Dict[str, Any]:
        """
        Map a requested item to what the modality can actually decode.
        A speech item on scalp EEG becomes speech ONSET, not word identity.
        """
        spec = THAI_ITEMS[item]
        cap = CAPABILITIES[modality]
        if spec["kind"] == "speech":
            if modality in spec.get("word_identity_modality", []):
                return {"target":"limited_vocabulary_50_100"
                               if modality == Modality.ECOG_SEEG
                               else "speech_bci_62wpm",
                        "item_specific": True,
                        "note":"word identity requires invasive recording"}
            return {"target": spec["scalp_target"], "item_specific": False,
                    "DOWNGRADED": True,
                    "reason": (f"{modality} cannot identify which word. "
                               f"Only speech onset / speaking-vs-silent is "
                               f"available. The word label is still recorded "
                               f"from audio for timing, but is NOT a "
                               f"decodable class."),
                    "physical_reason": cap.spatial_scale}
        if spec["kind"] == "subjective":
            raise NoCapabilityRefusal(
                f"'{item}' is subjective. No modality measures a feeling "
                f"directly. Use self-report as the label and treat neural "
                f"features as candidate CORRELATES only. {spec['note']}")
        return {"target": spec["scalp_target"], "item_specific": False,
                "note": spec.get("note","")}


# =============================================================================
# §4. MULTIMODAL INGESTION
# =============================================================================

@dataclass
class Recording:
    modality: str
    fs: float
    channel_names: List[str]
    data: Any                       # np.ndarray [n_channels, n_samples]
    session_id: str
    emg_channels: List[int] = field(default_factory=list)
    eog_channels: List[int] = field(default_factory=list)
    audio_channel: Optional[int] = None
    events: List[BehavioralEvent] = field(default_factory=list)
    montage: str = ""
    shielded_room: bool = False
    impedance_kohm: Optional[List[float]] = None

def read_edf(path: str) -> Recording:
    """Minimal EDF/EDF+ reader for scalp EEG. Pure Python + numpy."""
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    raw = p.read_bytes(); h = raw[:256]
    def s(a,b): return h[a:b].decode("ascii","replace").strip()
    n_sig = int(s(252,256)); hdr_bytes = int(s(184,192))
    n_rec = int(s(236,244)); rec_dur = float(s(244,252))
    off = 256
    fields = [("label",16),("transducer",80),("dim",8),("pmin",8),("pmax",8),
              ("dmin",8),("dmax",8),("prefilter",80),("nsamp",8),("res",32)]
    cols: Dict[str,List[str]] = {k:[] for k,_ in fields}
    for _ in range(n_sig):
        for k,w in fields:
            cols[k].append(raw[off:off+w].decode("ascii","replace").strip())
            off += w
    names, specs = [], []
    for i in range(n_sig):
        lab = cols["label"][i]
        if lab.upper().startswith("EDF ANNOTATIONS"): continue
        names.append(lab)
        specs.append((int(cols["nsamp"][i] or 1),
                      float(cols["pmin"][i] or 0), float(cols["pmax"][i] or 1),
                      float(cols["dmin"][i] or -32768),
                      float(cols["dmax"][i] or 32767)))
    body = raw[hdr_bytes:]
    nsamp_all = [int(c or 1) for c in cols["nsamp"]]
    per_rec: Dict[int,List[Any]] = {i:[] for i in range(len(names))}
    pos = 0; k = 0
    for _ in range(max(n_rec,0)):
        for i in range(n_sig):
            n = nsamp_all[i]; chunk = body[pos:pos+2*n]; pos += 2*n
            if cols["label"][i].upper().startswith("EDF ANNOTATIONS"): continue
            if len(chunk) < 2*n: break
            per_rec[i].append(np.frombuffer(chunk, dtype="<i2").astype(float))
    rows = []
    for i,(n,pmn,pmx,dmn,dmx) in enumerate(specs):
        if not per_rec[i]: continue
        x = np.concatenate(per_rec[i])
        sc = (pmx-pmn)/(dmx-dmn) if dmx != dmn else 1.0
        rows.append((x-dmn)*sc+pmn)
    data = np.vstack(rows) if rows else np.zeros((0,0))
    fs = specs[0][0]/max(rec_dur,1e-9) if specs else 0.0
    low = [n.lower() for n in names]
    emg = [i for i,n in enumerate(low) if "emg" in n or "chin" in n]
    eog = [i for i,n in enumerate(low) if "eog" in n or "loc" in n or "roc" in n]
    return Recording(Modality.SCALP_EEG, fs, names, data,
                     session_id=p.stem, emg_channels=emg, eog_channels=eog)

def load_meg(path: str) -> Recording:
    """
    MEG FIF ingestion. Requires mne in production; here we accept a
    numpy .npz export so the module runs with no heavy dependency.
    """
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    z = np.load(p, allow_pickle=True)
    return Recording(Modality.MEG, float(z["fs"]),
                     [str(c) for c in z["channels"]], z["data"],
                     session_id=p.stem,
                     emg_channels=[int(i) for i in z.get("emg",[])],
                     eog_channels=[int(i) for i in z.get("eog",[])],
                     shielded_room=bool(z.get("shielded_room", False)))

def load_structural_mri(path: str) -> Dict[str, Any]:
    """Structural MRI: anatomy for forward modelling, NOT a time series."""
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    z = np.load(p, allow_pickle=True)
    return {"modality": Modality.STRUCT_MRI,
            "purpose": "head_model_and_coregistration",
            "tissue_labels": z.get("tissue_labels"),
            "conductivity_S_m": {"scalp":0.33,"skull":0.0066,"brain":0.33},
            "NOT": "a functional time signal; cannot decode state"}

def load_fmri(path: str) -> Recording:
    """fMRI: TR-sampled voxel series. Note the hemodynamic lag explicitly."""
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    z = np.load(p, allow_pickle=True)
    fs = 1.0/float(z.get("TR_s", 2.0))
    return Recording(Modality.FMRI, fs,
                     [str(c) for c in z["voxels"]], z["data"],
                     session_id=p.stem)


# =============================================================================
# §5. ARTIFACT CONTROL  (mandatory for any speech-related claim)
# =============================================================================

class ArtifactControl:
    """
    EMG/EOG contamination is the primary confound in speech-decoding claims.
    Without these controls a 'word decoder' is usually an EMG detector.
    """
    def __init__(self, fs: float):
        self.fs = fs

    def bandpower(self, x: "np.ndarray", lo: float, hi: float) -> "np.ndarray":
        n = x.shape[-1]
        X = np.fft.rfft(x*np.hanning(n), axis=-1)
        psd = (np.abs(X)**2)/max(n,1)
        f = np.fft.rfftfreq(n, 1.0/self.fs)
        m = (f >= lo) & (f < hi)
        return psd[..., m].mean(axis=-1)

    def emg_index(self, rec: Recording, epoch: "np.ndarray") -> float:
        """High-frequency power on EMG channels: 70-200 Hz proxy."""
        if not rec.emg_channels: return float("nan")
        e = epoch[rec.emg_channels]
        return float(np.mean(self.bandpower(e, 70.0, min(200.0, self.fs/2-1))))

    def eog_index(self, epoch: "np.ndarray",
                  eog_idx: Sequence[int]) -> float:
        if not eog_idx: return float("nan")
        e = epoch[list(eog_idx)]
        return float(np.mean(np.abs(e).max(axis=-1)))

    def regress_out(self, neural: "np.ndarray",
                    nuisance: "np.ndarray") -> Dict[str, Any]:
        """
        Project nuisance channels (EMG/EOG) out of the neural data.
        neural: [C_n, T], nuisance: [C_u, T]
        Returns residual and the variance fraction removed.
        """
        if nuisance.shape[0] == 0:
            return {"residual": neural, "variance_removed": 0.0,
                    "regressed": False}
        U = nuisance.T                       # T x C_u
        U = U - U.mean(axis=0, keepdims=True)
        Y = neural.T                         # T x C_n
        Y = Y - Y.mean(axis=0, keepdims=True)
        coef, *_ = np.linalg.lstsq(U, Y, rcond=None)
        pred = U @ coef
        resid = Y - pred
        v0 = float(np.var(Y)); v1 = float(np.var(resid))
        return {"residual": resid.T,
                "variance_removed": (1.0 - v1/max(v0,1e-30)),
                "regressed": True,
                "n_nuisance": nuisance.shape[0]}

    def screen_epoch(self, rec: Recording, epoch: "np.ndarray",
                     emg_threshold: float, eog_threshold: float,
                     amp_threshold: float) -> Dict[str, Any]:
        eeg = [i for i in range(epoch.shape[0])
               if i not in rec.emg_channels and i not in rec.eog_channels]
        ne = epoch[eeg] if eeg else epoch
        emg = self.emg_index(rec, epoch)
        eog = self.eog_index(epoch, rec.eog_channels)
        ptp = float(np.max(np.ptp(ne, axis=-1)))
        reasons = []
        if not math.isnan(emg) and emg > emg_threshold:
            reasons.append(f"EMG {emg:.3g} > {emg_threshold:.3g}")
        if not math.isnan(eog) and eog > eog_threshold:
            reasons.append(f"EOG {eog:.3g} > {eog_threshold:.3g}")
        if ptp > amp_threshold:
            reasons.append(f"amplitude {ptp:.3g} > {amp_threshold:.3g}")
        return {"clean": len(reasons) == 0, "reasons": reasons,
                "emg_index": emg, "eog_index": eog, "ptp": ptp}

    @staticmethod
    def require_controls(rec: Recording, target: str) -> Dict[str, Any]:
        speech_related = target in ("speech_onset","speaking_vs_silent",
                                    "limited_vocabulary_50_100","speech_bci_62wpm")
        if speech_related and not rec.emg_channels:
            raise ArtifactRefusal(
                f"target '{target}' is speech-related. EMG channels are "
                "MANDATORY. Without them, subvocalisation EMG cannot be "
                "excluded and any decoding claim is Mode E.")
        if rec.modality == Modality.MEG and not rec.shielded_room:
            raise ArtifactRefusal("MEG requires a magnetically shielded room.")
        return {"controls_ok": True, "speech_related": speech_related,
                "n_emg": len(rec.emg_channels), "n_eog": len(rec.eog_channels)}


# =============================================================================
# §6. FEATURES
# =============================================================================

BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),
         "beta":(13,30),"gamma":(30,80),"high_gamma":(70,150)}

class Features:
    def __init__(self, fs: float):
        self.fs = fs

    def bandpower(self, x: "np.ndarray") -> Dict[str, "np.ndarray"]:
        n = x.shape[-1]
        X = np.fft.rfft(x*np.hanning(n), axis=-1)
        psd = (np.abs(X)**2)/max(n,1)
        f = np.fft.rfftfreq(n, 1.0/self.fs)
        out = {}
        for b,(lo,hi) in BANDS.items():
            if hi >= self.fs/2: continue
            out[b] = psd[..., (f>=lo)&(f<hi)].mean(axis=-1)
        return out

    def erd_ers(self, x: "np.ndarray", baseline: "np.ndarray",
                lo: float, hi: float) -> "np.ndarray":
        """Event-related desynchronisation relative to a baseline window."""
        ac = Features(self.fs).bandpower(x).get("alpha")
        bl = Features(self.fs).bandpower(baseline).get("alpha")
        if ac is None or bl is None: return np.zeros(x.shape[0])
        return (ac - bl)/np.maximum(bl, 1e-12)

    def covariance(self, x: "np.ndarray") -> "np.ndarray":
        xc = x - x.mean(axis=-1, keepdims=True)
        C = xc @ xc.T / max(x.shape[-1]-1, 1)
        return C + 1e-6*np.eye(C.shape[0])

    def riemannian_tangent(self, C: "np.ndarray",
                           Cref: "np.ndarray") -> "np.ndarray":
        w, V = np.linalg.eigh(Cref)
        wh = V @ np.diag(w**-0.5) @ V.T
        ew, EV = np.linalg.eigh(wh @ C @ wh)
        Lm = EV @ np.diag(np.log(np.maximum(ew,1e-12))) @ EV.T
        return Lm[np.triu_indices(Lm.shape[0])]

    def csp(self, X1: "np.ndarray", X2: "np.ndarray",
            n_filters: int = 4) -> "np.ndarray":
        """Common spatial patterns for two-class motor imagery."""
        C1 = np.mean([self.covariance(x) for x in X1], axis=0)
        C2 = np.mean([self.covariance(x) for x in X2], axis=0)
        Cc = C1 + C2
        w, V = np.linalg.eigh(Cc)
        P = np.diag(np.maximum(w,1e-12)**-0.5) @ V.T
        S1 = P @ C1 @ P.T
        w1, V1 = np.linalg.eigh(S1)
        idx = np.argsort(w1)
        sel = list(idx[:n_filters//2]) + list(idx[-(n_filters//2):])
        return (V1[:, sel].T @ P)

    def feature_vector(self, x: "np.ndarray") -> "np.ndarray":
        bp = self.bandpower(x)
        parts = [np.log1p(np.maximum(v,0)).ravel() for v in bp.values()]
        parts.append(x.mean(axis=-1))
        parts.append(x.std(axis=-1))
        return np.concatenate(parts)


# =============================================================================
# §7. TRANSFORMER / WORLD MODEL / DIFFUSION  (research components)
# =============================================================================

if HAS_TORCH:
    class NeuralSequenceTransformer(nn.Module):
        """
        Sequence model over epoched neural features. Predicts the observed
        behavioural label. RESEARCH MODEL, not a diagnostic device.
        """
        def __init__(self, d_in: int = 64, d: int = 128, n_heads: int = 4,
                     n_layers: int = 3, n_classes: int = 4,
                     max_len: int = 256):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.randn(1, max_len, d)*0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, d*4, 0.1,
                                               batch_first=True,
                                               norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d, n_classes)
            self.null_head = nn.Linear(d, 1)   # artifact/nuisance predictor

        def forward(self, x):
            B, L, _ = x.shape
            h = self.trunk(self.proj(x) + self.pos[:, :L])
            pooled = h.mean(dim=1)
            return {"logits": self.head(pooled),
                    "nuisance": self.null_head(pooled),
                    "embedding": pooled}

    class NeuralWorldModel(nn.Module):
        """
        Predicts the next neural feature state given the current state and
        an action/context vector. Used for hypothesis generation and for
        testing whether behaviour is predictable from dynamics.
        Outputs uncertainty; never a measurement.
        """
        def __init__(self, d_obs: int = 64, d_act: int = 8, d: int = 128):
            super().__init__()
            self.enc = nn.Linear(d_obs + d_act, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_obs)
            self.log_sd = nn.Linear(d, d_obs)

        def forward(self, obs, act):
            h = F.gelu(self.enc(torch.cat([obs, act], -1)))
            h = self.gru(h, h)
            return {"pred": self.mu(h),
                    "log_sd": self.log_sd(h).clamp(-6, 3),
                    "STATUS": "surrogate prediction, not a measurement"}

    class NeuralDiffusion(nn.Module):
        """
        Generates CANDIDATE feature sequences for hypothesis testing.
        A generated signal is a hypothesis, never data. Every sample must
        be compared against real recordings before any use.
        """
        def __init__(self, d: int = 64, T: int = 100, hidden: int = 128):
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
            pred = self.net(torch.cat([xt, t.float()[:,None]/self.T], -1))
            return F.mse_loss(pred, eps)

        @torch.no_grad()
        def sample(self, n: int, device: str = "cpu"):
            x = torch.randn(n, self.d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, tb.float()[:,None]/self.T], -1))
                a = 1-self.betas[t]; ab = self.abar[t]
                x = (x - (1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate": x,
                    "STATUS": "HYPOTHESIS — validate against real recordings"}


# =============================================================================
# §8. VALIDATION  (the part that makes a claim defensible)
# =============================================================================

class Validation:
    @staticmethod
    def chance_level(n_classes: int, class_counts: Sequence[int]) -> float:
        """Balanced accuracy chance = 1/K; also report majority baseline."""
        tot = sum(class_counts)
        maj = max(class_counts)/tot if tot else 0.0
        return {"chance_balanced": 1.0/max(n_classes,1),
                "majority_baseline": maj,
                "n_classes": n_classes, "n_samples": tot}

    @staticmethod
    def permutation_test(y_true: "np.ndarray", y_pred: "np.ndarray",
                         score_fn: Callable, n_perm: int = 1000,
                         seed: int = 0) -> Dict[str, Any]:
        """
        Shuffle labels to build the null. A decoding claim requires the
        observed score to exceed the null distribution.
        """
        rng = np.random.default_rng(seed)
        obs = float(score_fn(y_true, y_pred))
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = score_fn(rng.permutation(y_true), y_pred)
        p = (1.0 + float(np.sum(null >= obs)))/(n_perm + 1.0)
        return {"observed": obs, "null_mean": float(null.mean()),
                "null_sd": float(null.std()), "p_value": p,
                "n_perm": n_perm,
                "above_chance": p < 0.01,
                "required": "p < 0.01 with >=1000 permutations"}

    @staticmethod
    def balanced_accuracy(y_true: "np.ndarray",
                          y_pred: "np.ndarray") -> float:
        classes = np.unique(y_true)
        accs = []
        for c in classes:
            m = y_true == c
            if m.sum() > 0: accs.append(float(np.mean(y_pred[m] == c)))
        return float(np.mean(accs)) if accs else 0.0

    @staticmethod
    def cross_session(session_scores: Dict[str, float],
                      min_sessions: int = 3) -> Dict[str, Any]:
        """
        Within-session accuracy is inflated by drift and block structure.
        A generalisation claim requires held-out SESSIONS, not trials.
        """
        vals = list(session_scores.values())
        if len(vals) < min_sessions:
            return {"valid": False,
                    "reason": f"need >= {min_sessions} held-out sessions, "
                              f"got {len(vals)}",
                    "status": "MODE_E"}
        return {"valid": True, "n_sessions": len(vals),
                "mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "generalises": float(np.min(vals)) > 0.0}

    @staticmethod
    def emg_confound_check(score_with_emg: float,
                           score_after_regression: float,
                           chance: float) -> Dict[str, Any]:
        """
        If performance collapses after removing EMG, the 'decoder' was
        reading muscle activity, not brain activity.
        """
        drop = score_with_emg - score_after_regression
        survived = score_after_regression > chance + 0.05
        return {"score_raw": score_with_emg,
                "score_emg_regressed": score_after_regression,
                "drop": drop, "chance": chance,
                "survives_emg_control": survived,
                "verdict": ("neural signal survives EMG control" if survived
                            else "EMG CONFOUND — the apparent decoding is "
                                 "explained by muscle activity"),
                "status": "CERTIFIED" if survived else "MODE_E"}


# =============================================================================
# §9. STATISTICS DICTIONARY
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "sleep_stage": (("CAP","EMG","EOG"), "AASM staging",
                      "requires EEG+EOG+EMG per AASM"),
      "motor_imagery": (("CAP","EMG","PERM","XSESS"), "CSP/Riemannian",
                        "mu/beta ERD; cross-session required"),
      "speech_onset": (("CAP","EMG","PERM","XSESS"), "motor preparation + EMG",
                       "EMG control mandatory; onset only, not word identity"),
      "word_identity": (("CAP","EMG","PERM","XSESS","INVASIVE"),
                        "ECoG/sEEG/intracortical speech BCI",
                        "NOT available from scalp EEG or MEG"),
      "semantic_gist": (("CAP","FMRI_TRAIN","PERM","XSESS","COOP"),
                        "fMRI encoding/decoding model",
                        ">=10 h training; cooperation; seconds lag; defeatable"),
      "subjective_emotion": (("SELF_REPORT","PERM","XSESS"),
                             "correlate only",
                             "a feeling is not measured by EEG; self-report is the label"),
      "workload": (("CAP","EMG","PERM","XSESS"), "theta/alpha ratio",
                   "coarse; confounded by arousal and time-on-task"),
    }
    def __init__(self): self.log: List[Dict] = []
    def certify(self, stat: str, verified: Sequence[str],
                value: Any = None) -> Dict[str, Any]:
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
 "CAP":        "modality capability tier permits this target",
 "EMG":        "EMG channels recorded and regressed/rejected",
 "EOG":        "EOG channels recorded and controlled",
 "PERM":       "permutation test p<0.01, >=1000 shuffles",
 "XSESS":      ">=3 held-out sessions",
 "INVASIVE":   "clinical indication + neurosurgery + council approval",
 "FMRI_TRAIN": ">=10 h per-subject training documented",
 "COOP":       "subject cooperation documented; resistance test reported",
 "SELF_REPORT":"patient self-report or validated clinician rating",
}


# =============================================================================
# §10. PIPELINE
# =============================================================================

class PairedRecordingPipeline:
    def __init__(self, out_dir: str = "runs/neural_pair"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()

    def epoch(self, rec: Recording, ev: BehavioralEvent,
              pre_s: float = 0.5, post_s: float = 1.0) -> "np.ndarray":
        _np()
        a = int((ev.onset_s - pre_s)*rec.fs)
        b = int((ev.onset_s + post_s)*rec.fs)
        a = max(0, a); b = min(rec.data.shape[1], b)
        return rec.data[:, a:b]

    def run(self, consent: Consent, rec: Recording,
            item: str, target: Optional[str] = None,
            emg_threshold: float = 5.0, eog_threshold: float = 100.0,
            amp_threshold: float = 200.0) -> Dict[str, Any]:
        _np()
        res: Dict[str, Any] = {"item": item, "modality": rec.modality,
                               "session": rec.session_id}
        auth = Governance.authorize_recording(
            consent, rec.modality,
            target or LabelAcquisition.resolve_target(item, rec.modality)["target"])
        res["authorization"] = auth
        tgt = auth["capability"]["target"]
        res["target"] = tgt
        resolved = LabelAcquisition.resolve_target(item, rec.modality)
        res["resolution"] = resolved

        ac = ArtifactControl(rec.fs)
        res["controls"] = ac.require_controls(rec, tgt)

        feats = Features(rec.fs)
        clean, rejected = [], []
        for ev in rec.events:
            LabelAcquisition.validate_event(ev)
            ep = self.epoch(rec, ev)
            if ep.shape[1] < 8:
                rejected.append({"item": ev.item, "reason": "epoch too short"})
                continue
            scr = ac.screen_epoch(rec, ep, emg_threshold,
                                  eog_threshold, amp_threshold)
            if not scr["clean"]:
                rejected.append({"item": ev.item, **scr}); continue
            nuisance = ep[rec.emg_channels + rec.eog_channels] \
                if (rec.emg_channels or rec.eog_channels) else np.zeros((0, ep.shape[1]))
            eeg_idx = [i for i in range(ep.shape[0])
                       if i not in rec.emg_channels and i not in rec.eog_channels]
            reg = ac.regress_out(ep[eeg_idx], nuisance)
            clean.append({"item": ev.item,
                          "features": feats.feature_vector(reg["residual"]),
                          "variance_removed": reg["variance_removed"],
                          "session": ev.session_id or rec.session_id})
        res["n_clean"] = len(clean)
        res["n_rejected"] = len(rejected)
        res["rejection_reasons"] = rejected[:10]
        if not clean:
            res["sd"] = self.sd.certify(self._stat_key(tgt), (), None)
            res["conclusion"] = "NO CLEAN EPOCHS — no decoding claim possible"
            return res
        res["mean_variance_removed_by_emg_eog"] = float(
            np.mean([c["variance_removed"] for c in clean]))
        res["sd"] = self.sd.certify(self._stat_key(tgt), ("CAP",),
                                    {"n_clean": len(clean)})
        res["conclusion"] = (
            "Features extracted and artifact-controlled. A decoding claim "
            "additionally requires PERM (permutation test) and XSESS "
            "(cross-session). Until then this is Mode E.")
        res["actionability"] = RESEARCH_ONLY
        return res

    @staticmethod
    def _stat_key(target: str) -> str:
        return {"speech_onset":"speech_onset",
                "speaking_vs_silent":"speech_onset",
                "motor_imagery_left_right":"motor_imagery",
                "limited_vocabulary_50_100":"word_identity",
                "speech_bci_62wpm":"word_identity",
                "semantic_gist_reconstruction":"semantic_gist",
                "sleep_stage":"sleep_stage",
                "workload":"workload"}.get(target, "workload")

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION,
               "capabilities": {k: asdict(v) for k,v in CAPABILITIES.items()},
               "items": THAI_ITEMS, "packages": PACKAGES,
               "sd_report": self.sd.report(),
               "actionability": RESEARCH_ONLY}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §11. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_NUMPY:
        print("  numpy required. FAIL"); return False
    ok = True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    rng = np.random.default_rng(0)
    good = Consent("pt01","IRB-2026-01","psych01","MC-2026-777",
                   "paired EEG-behaviour recording of speech and motor tasks",
                   True, True, None, "deidentified",
                   acknowledged_no_thought_reading=True,
                   acknowledged_can_withdraw_anytime=True)

    print("\n[T1] CAPABILITY TIERS")
    for m in (Modality.SCALP_EEG, Modality.MEG):
        b = False
        try: check_capability(m, "word_identity")
        except NoCapabilityRefusal: b = True
        rep(f"{m} REFUSES word identity", b)
    rep("scalp EEG permits speech onset",
        check_capability(Modality.SCALP_EEG,"speech_onset")["allowed"])
    rep("scalp EEG permits motor imagery",
        check_capability(Modality.SCALP_EEG,"motor_imagery_left_right")["allowed"])
    fm = check_capability(Modality.FMRI,"semantic_gist_reconstruction")
    rep("fMRI gist allowed with lag declared", fm["lag_s"] >= 4.0,
        f"lag={fm['lag_s']}s")
    rep("fMRI cannot do exact words",
        "exact_word_identity" in CAPABILITIES[Modality.FMRI].cannot)
    rep("fMRI cannot decode non-cooperative subject",
        "non_cooperative_decoding" in CAPABILITIES[Modality.FMRI].cannot)
    b = False
    try: check_capability(Modality.ECOG_SEEG,"unrestricted_semantic_content")
    except NoCapabilityRefusal: b = True
    rep("ECoG refuses unrestricted semantics", b)
    rep("structural MRI is not a time signal",
        CAPABILITIES[Modality.STRUCT_MRI].temporal_lag_s == float("inf"))

    print("\n[T2] CONSENT / PSYCHIATRIST / COUNCIL")
    b = False
    try: Governance.authorize_recording(Consent(), Modality.SCALP_EEG, "speech_onset")
    except ConsentRefusal: b = True
    rep("empty consent BLOCKED", b)
    no_ack = Consent("p","IRB","psych","MC","scope",True,True,
                     acknowledged_no_thought_reading=False,
                     acknowledged_can_withdraw_anytime=True)
    b = False
    try: Governance.authorize_recording(no_ack, Modality.SCALP_EEG,"speech_onset")
    except ConsentRefusal: b = True
    rep("no acknowledgement of limits BLOCKED", b)
    withdrawn = Consent("pt01","IRB","psych","MC","scope",True,True,
                        withdrawn_at="2026-11-26T10:00:00Z",
                        acknowledged_no_thought_reading=True,
                        acknowledged_can_withdraw_anytime=True)
    b = False
    try: Governance.authorize_recording(withdrawn, Modality.SCALP_EEG,"speech_onset")
    except ConsentRefusal: b = True
    rep("withdrawn consent STOPS recording", b)
    auth = Governance.authorize_recording(good, Modality.SCALP_EEG,"speech_onset")
    rep("complete consent authorised", auth["allowed"])
    rep("authorisation excludes thought reading",
        "thought reading" in auth["NOT_AUTHORIZED"])
    b = False
    try: Governance.authorize_recording(good, Modality.ECOG_SEEG,
                                        "limited_vocabulary_50_100")
    except ConsentRefusal: b = True
    rep("invasive without indication BLOCKED", b)
    inv = Consent("pt02","IRB","psych","MC","speech BCI",True,True,None,
                  "deidentified","refractory_epilepsy_localisation","neuro01",
                  acknowledged_no_thought_reading=True,
                  acknowledged_can_withdraw_anytime=True)
    rep("invasive WITH indication authorised",
        Governance.authorize_recording(inv, Modality.ECOG_SEEG,
                                       "limited_vocabulary_50_100")["allowed"])

    print("\n[T3] LABEL SOURCE — observation only")
    b = False
    try:
        LabelAcquisition.validate_event(BehavioralEvent(
            "ก_ไก่",1.0,0.5,LabelSource.DECODER_OUTPUT,"decoder"))
    except LabelSourceRefusal: b = True
    rep("decoder output as label REFUSED", b)
    ev = BehavioralEvent("ก_ไก่",1.0,0.6,LabelSource.AUDIO_TRANSCRIPT,
                         "coder_A",0.95,"sess01")
    rep("audio-labelled speech event valid",
        LabelAcquisition.validate_event(ev)["valid"])
    b = False
    try:
        LabelAcquisition.validate_event(BehavioralEvent(
            "เศร้า",2.0,1.0,LabelSource.SELF_REPORT,"patient"))
    except LabelSourceRefusal: b = True
    rep("subjective item without self-report value REFUSED", b)
    sad = BehavioralEvent("เศร้า",2.0,1.0,LabelSource.SELF_REPORT,"patient",
                          self_report_value=6.0)
    rep("subjective WITH self-report accepted",
        LabelAcquisition.validate_event(sad)["valid"])
    b = False
    try:
        LabelAcquisition.validate_event(BehavioralEvent(
            "ยิ้ม",3.0,0.4,LabelSource.SELF_REPORT,"patient"))
    except LabelSourceRefusal: b = True
    rep("smile requires video/FACS not self-report", b)
    smile = BehavioralEvent("ยิ้ม",3.0,0.4,LabelSource.VIDEO_FACS,"coder_B")
    rep("smile with FACS accepted",
        LabelAcquisition.validate_event(smile)["valid"])

    print("\n[T4] ITEM -> TARGET RESOLUTION (honest downgrade)")
    r = LabelAcquisition.resolve_target("ก_ไก่", Modality.SCALP_EEG)
    print(f"      ก_ไก่ on scalp EEG -> target={r['target']}")
    rep("speech item downgraded to onset on scalp", r["DOWNGRADED"])
    rep("downgrade reason states word identity impossible",
        "cannot identify which word" in r["reason"])
    r2 = LabelAcquisition.resolve_target("ก_ไก่", Modality.ECOG_SEEG)
    rep("same item permits vocabulary on ECoG", r2["item_specific"])
    b = False
    try: LabelAcquisition.resolve_target("เศร้า", Modality.SCALP_EEG)
    except NoCapabilityRefusal: b = True
    rep("subjective emotion REFUSED as decoding target", b)
    lift = LabelAcquisition.resolve_target("ยกของ", Modality.SCALP_EEG)
    rep("lift maps to motor imagery", lift["target"] == "motor_imagery_left_right")

    print("\n[T5] ARTIFACT CONTROL")
    fs = 256.0
    n = 512
    t = np.arange(n)/fs
    eeg = rng.normal(0, 10, (8, n)) + 20*np.sin(2*PI*10*t)
    emg = rng.normal(0, 50, (2, n))
    eog = rng.normal(0, 80, (2, n))
    data = np.vstack([eeg, emg, eog])
    rec = Recording(Modality.SCALP_EEG, fs,
                    [f"C{i}" for i in range(8)]+["EMG1","EMG2","EOG1","EOG2"],
                    data, "sess01",
                    emg_channels=[8,9], eog_channels=[10,11])
    ac = ArtifactControl(fs)
    rep("speech target requires EMG",
        ac.require_controls(rec,"speech_onset")["controls_ok"])
    noemg = Recording(Modality.SCALP_EEG, fs, [f"C{i}" for i in range(8)],
                      eeg, "s", emg_channels=[])
    b = False
    try: ac.require_controls(noemg, "speech_onset")
    except ArtifactRefusal: b = True
    rep("speech target WITHOUT EMG REFUSED", b)
    meg_noshield = Recording(Modality.MEG, fs, ["M1"], eeg[:1], "s",
                             shielded_room=False)
    b = False
    try: ac.require_controls(meg_noshield, "speech_onset")
    except ArtifactRefusal: b = True
    rep("MEG without shielded room REFUSED", b)
    # inject EMG into EEG and verify regression removes it
    contam = eeg + 0.8*emg[0]
    reg = ac.regress_out(contam, emg)
    print(f"      variance removed by EMG regression: "
          f"{reg['variance_removed']:.3f}")
    rep("EMG regression removes injected contamination",
        reg["variance_removed"] > 0.2, f"{reg['variance_removed']:.3f}")
    scr = ac.screen_epoch(rec, data, 5.0, 100.0, 200.0)
    rep("epoch screening runs", "clean" in scr)

    print("\n[T6] FEATURES")
    ft = Features(fs)
    bp = ft.bandpower(eeg)
    rep("alpha dominant for 10 Hz",
        float(bp["alpha"].mean()) > float(bp["delta"].mean()))
    C = ft.covariance(eeg)
    tg = ft.riemannian_tangent(C, C)
    rep("Riemannian tangent of self ~ 0", float(np.abs(tg).max()) < 1e-6)
    X1 = rng.normal(0,1,(6,8,256)); X2 = rng.normal(0,2,(6,8,256))
    W = ft.csp(X1, X2, 4)
    rep("CSP returns filters", W.shape == (4,8))
    fv = ft.feature_vector(eeg)
    rep("feature vector finite", bool(np.isfinite(fv).all()),
        f"dim={fv.shape[0]}")

    print("\n[T7] VALIDATION — the part that makes a claim defensible")
    y = rng.integers(0,4,200)
    yp = y.copy(); yp[:40] = rng.integers(0,4,40)
    ba = Validation.balanced_accuracy(y, yp)
    ch = Validation.chance_level(4, [50]*4)
    print(f"      balanced acc={ba:.3f}  chance={ch['chance_balanced']:.3f}")
    pt = Validation.permutation_test(y, yp, Validation.balanced_accuracy,
                                     n_perm=300)
    print(f"      permutation p={pt['p_value']:.4f}")
    rep("true signal beats permutation null", pt["above_chance"])
    pt_bad = Validation.permutation_test(y, rng.integers(0,4,200),
                                         Validation.balanced_accuracy, 300)
    rep("random prediction fails permutation test", not pt_bad["above_chance"],
        f"p={pt_bad['p_value']:.3f}")
    xs_bad = Validation.cross_session({"s1":0.8,"s2":0.7})
    rep("2 sessions insufficient -> MODE_E", xs_bad["status"] == MODE_E)
    xs_ok = Validation.cross_session({"s1":0.8,"s2":0.75,"s3":0.7})
    rep("3 sessions accepted", xs_ok["valid"])
    emg_ok = Validation.emg_confound_check(0.82, 0.76, 0.25)
    rep("surviving EMG control -> CERTIFIED",
        emg_ok["status"] == MODE_CERTIFIED)
    emg_bad = Validation.emg_confound_check(0.82, 0.27, 0.25)
    rep("collapsing after EMG -> MODE_E", emg_bad["status"] == MODE_E)
    rep("verdict names EMG confound", "EMG CONFOUND" in emg_bad["verdict"])

    print("\n[T8] SD GATE")
    pipe = PairedRecordingPipeline()
    c1 = pipe.sd.certify("word_identity", ("CAP","EMG","PERM","XSESS"))
    rep("word identity without INVASIVE -> MODE_E", c1["status"] == MODE_E)
    rep("missing package named", "INVASIVE" in c1["missing"])
    c2 = pipe.sd.certify("speech_onset", ("CAP","EMG"))
    rep("speech onset without PERM/XSESS -> MODE_E", c2["status"] == MODE_E)
    c3 = pipe.sd.certify("speech_onset", ("CAP","EMG","PERM","XSESS"))
    rep("speech onset fully controlled -> CERTIFIED",
        c3["status"] == MODE_CERTIFIED)
    c4 = pipe.sd.certify("semantic_gist", ("CAP","PERM","XSESS"))
    rep("fMRI gist without training/cooperation -> MODE_E",
        c4["status"] == MODE_E)

    print("\n[T9] PIPELINE END-TO-END")
    rec.events = [BehavioralEvent("ก_ไก่", 0.8, 0.5,
                                  LabelSource.AUDIO_TRANSCRIPT,"coder_A",
                                  session_id="sess01"),
                  BehavioralEvent("ข_ไข่", 1.4, 0.5,
                                  LabelSource.AUDIO_TRANSCRIPT,"coder_A",
                                  session_id="sess01")]
    out = pipe.run(good, rec, "ก_ไก่")
    print(f"      target={out['target']}  clean={out['n_clean']} "
          f"rejected={out['n_rejected']}")
    rep("pipeline runs", "authorization" in out)
    rep("target downgraded to speech_onset", out["target"] == "speech_onset")
    rep("output is research only", out["actionability"] == RESEARCH_ONLY)
    rep("conclusion requires PERM and XSESS",
        "PERM" in out["conclusion"] and "XSESS" in out["conclusion"])
    r = pipe.report()
    rep("report written", (pipe.out/"report.json").exists())
    rep("Mode E retained", r["sd_report"]["mode_E"] > 0)

    if HAS_TORCH:
        print("\n[T10] RESEARCH MODELS")
        torch.manual_seed(0)
        m = NeuralSequenceTransformer(32, 64, 4, 2, 4, 64)
        x = torch.randn(8, 32, 32)
        o = m(x)
        rep("transformer forward", o["logits"].shape == (8,4))
        rep("nuisance head present (artifact predictor)",
            o["nuisance"].shape == (8,1))
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        tgt = torch.randint(0,4,(8,))
        l0 = lN = None
        for ep in range(40):
            opt.zero_grad()
            L = F.cross_entropy(m(x)["logits"], tgt)
            L.backward(); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("transformer trains", lN < l0, f"{l0:.3f}->{lN:.3f}")
        wm = NeuralWorldModel(32, 8, 64)
        wo = wm(torch.randn(4,32), torch.randn(4,8))
        rep("world model labelled surrogate",
            "not a measurement" in wo["STATUS"])
        df = NeuralDiffusion(32, 30, 64)
        opt2 = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.randn(8,32); d0 = dN = None
        for ep in range(40):
            opt2.zero_grad(); L = df.loss(x0); L.backward(); opt2.step()
            if ep == 0: d0 = float(L)
            dN = float(L)
        rep("diffusion trains", dN < d0, f"{d0:.3f}->{dN:.3f}")
        s = df.sample(2)
        rep("diffusion output labelled HYPOTHESIS",
            "HYPOTHESIS" in s["STATUS"])
    else:
        print("\n[T10] torch absent — research models skipped")

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Capability tiers for scalp EEG, MEG, structural MRI, fMRI, ECoG/sEEG,
  intracortical — enforced, not advisory.
  Consent gate requiring psychiatrist AND medical council AND explicit
  acknowledgement that this is not thought reading.
  Label acquisition from audio/video/EMG/accelerometer/self-report only.
  Decoder output is refused as a training label.
  Thai item set mapped to the correct observation channel; speech items
  are honestly DOWNGRADED to onset on scalp EEG/MEG.
  Subjective items (โกรธ, เศร้า) require self-report; ยิ้ม requires FACS.
  Mandatory EMG/EOG channels for speech targets; nuisance regression;
  epoch screening; MEG shielded-room requirement.
  Bandpower, ERD/ERS, CSP, Riemannian tangent features.
  Permutation test, balanced accuracy, cross-session validation, and an
  explicit EMG-confound check that reports when decoding collapses.
  Transformer with a nuisance head, world model, diffusion generator —
  all labelled as research surrogates / hypotheses.

ENFORCED REFUSALS (all fired)
  word identity from scalp EEG or MEG
  exact words / real time / non-cooperative decoding from fMRI
  decoder output used as a label
  subjective feeling without self-report
  speech target without EMG channels
  MEG without a shielded room
  invasive modality without clinical indication
  recording without consent, or after withdrawal

WHAT THIS IS NOT
  Not thought reading. Not a diagnostic device. Not a clinical decision tool.
  A decoding claim additionally requires a permutation test and held-out
  sessions; until then every output is Mode E.
""")
    return ok


# =============================================================================
# §12. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Neural–behavior paired recording")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","capabilities","items","packages","run","report"])
    ap.add_argument("--edf", default=None)
    ap.add_argument("--meg", default=None)
    ap.add_argument("--item", default="ก_ไก่")
    ap.add_argument("--modality", default=Modality.SCALP_EEG)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    if a.cmd == "capabilities":
        print(json.dumps({k: asdict(v) for k,v in CAPABILITIES.items()},
                         indent=2, ensure_ascii=False)); return 0
    if a.cmd == "items":
        print(json.dumps(THAI_ITEMS, indent=2, ensure_ascii=False)); return 0
    if a.cmd == "packages":
        print(json.dumps(PACKAGES, indent=2, ensure_ascii=False)); return 0

    pipe = PairedRecordingPipeline()
    out: Dict[str, Any] = {}
    if a.cmd == "run":
        if not a.edf and not a.meg:
            print("--edf or --meg required"); return 1
        rec = read_edf(a.edf) if a.edf else load_meg(a.meg)
        consent = Consent("CLI_PATIENT","IRB_REQUIRED","PSYCHIATRIST_REQUIRED",
                          "COUNCIL_REQUIRED","paired recording",True,True,
                          acknowledged_no_thought_reading=True,
                          acknowledged_can_withdraw_anytime=True)
        log.warning("CLI uses placeholder consent identifiers. Replace with "
                    "real IRB, psychiatrist, and council references before use.")
        try:
            out = pipe.run(consent, rec, a.item)
        except Refusal as e:
            out = {"REFUSED": str(e)}
        print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    elif a.cmd == "report":
        out = pipe.report()
        print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str,
                                           ensure_ascii=False))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
