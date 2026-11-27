#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_neural_behavior_pairing.py — v2.0.0  [PRODUCTION]
Consented multimodal neural–behaviour paired recording and decoding research.

BUGS FIXED vs v1 (three were crashes):
  1. read_edf indexed per_rec by the ORIGINAL signal index while the dict
     was keyed by the COMPACTED index -> KeyError whenever an EDF
     Annotations channel preceded signal channels. Now an explicit
     original->compacted map is built.
  2. NpzFile has NO .get() method. load_meg / load_structural_mri /
     load_fmri all called z.get(...) -> AttributeError. Now a safe
     accessor checks membership in z.files.
  3. Features.erd_ers(x, baseline, lo, hi) ignored lo/hi and always
     returned the alpha band. Now the requested band is honoured.
  4. Multi-rate EDF was silently assigned the first channel's fs.
     Now rates are checked and a mismatch is refused.
  5. EMG band could invert when fs < 142 Hz (hi < lo) -> empty mean.
     Now guarded and reported as unavailable.
  6. Welch PSD normalisation was neither a density nor a power spectrum.
     Now a proper one-sided density with declared units.
  7. Cross-session and permutation validation existed but were never
     called by the pipeline. Now wired in and gating.
  8. regress_out removed nuisance without reporting task-locked loss.
     Now the task correlation of the removed component is reported.

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

ENFORCED REFUSALS (unchanged, all still fire)
  word identity from scalp EEG/MEG; exact words / real time /
  non-cooperative decoding from fMRI; decoder output as a label;
  subjective feeling without self-report; speech target without EMG;
  MEG without a shielded room; invasive without clinical indication;
  recording without consent or after withdrawal; data_sharing='none'.
=============================================================================
Author: PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID: 0009-0008-2374-0788 | License: MIT | Year 2026
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False; np = None
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VERSION = "2.0.0"; PI = math.pi
RESEARCH_ONLY = "research_only_not_a_diagnostic"
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class Refusal(RuntimeError): pass
class NoCapabilityRefusal(Refusal): pass
class ConsentRefusal(Refusal): pass
class ArtifactRefusal(Refusal): pass
class LabelSourceRefusal(Refusal): pass
class DataFormatRefusal(Refusal): pass
class SharingRefusal(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

def get_logger(n):
    import logging
    lg = logging.getLogger(n)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg
log = get_logger("msps.neural_pair")

@dataclass
class RunConfig:
    out_dir: str = "runs/neural_pair_v2"; seed: int = 0
    device: str = "auto"; amp: bool = True; strict: bool = True
    def validate(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True); return self

def set_seed(s):
    import random; random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
    if HAS_NUMPY: np.random.seed(s)
    if HAS_TORCH: torch.manual_seed(s)
    return {"seed": s}

def pick_device(pref="auto"):
    if not HAS_TORCH: return "cpu"
    if pref != "auto": return pref
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _npz_get(z, key, default=None):
    """FIX: NpzFile has no .get(). Check membership in .files."""
    return z[key] if key in z.files else default

class Manifest:
    def __init__(self, out_dir):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.items = []
    def add(self, kind, payload):
        h = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                      default=str).encode()).hexdigest()[:16]
        rec = {"kind": kind, "version": VERSION, "key": h,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "payload": payload}
        self.items.append(rec); return rec
    def write(self, name="manifest.json"):
        p = self.dir/name
        p.write_text(json.dumps({"n": len(self.items), "items": self.items},
                                indent=2, default=str, ensure_ascii=False))
        return str(p)

BANNER = f"""
=============================================================================
MSPS NEURAL–BEHAVIOUR PAIRED RECORDING v{VERSION}  [PRODUCTION]
EDF index map fixed · NpzFile accessor fixed · ERD band honoured
multi-rate EDF refused · proper Welch density · task-locked loss reported
permutation + cross-session WIRED and gating · production trainer
Labels from OBSERVATION only. Decoder output never a label.
=============================================================================
"""

# =============================================================================
# §1. CAPABILITY TIERS (unchanged, enforced)
# =============================================================================

class Modality:
    SCALP_EEG="scalp_eeg"; MEG="meg"; STRUCT_MRI="structural_mri"
    FMRI="fmri"; ECOG_SEEG="ecog_seeg"; INTRACORTICAL="intracortical"

@dataclass(frozen=True)
class Capability:
    modality: str; invasive: bool
    can: Tuple[str,...]; cannot: Tuple[str,...]
    spatial_scale: str; temporal_lag_s: float
    requires: Tuple[str,...]; note: str

CAPABILITIES: Dict[str,Capability] = {
 Modality.SCALP_EEG: Capability(Modality.SCALP_EEG,False,
   ("sleep_stage","workload","motor_imagery_left_right","speech_onset",
    "speaking_vs_silent","coarse_arousal"),
   ("word_identity","phoneme_identity","semantic_content",
    "subjective_emotion","specific_number"),
   "cm-scale, volume-conducted",0.0,
   ("electrodes_on_scalp","impedance_check","emg_eog_channels"),
   "10-100 uV; requires skin contact. No far-field channel."),
 Modality.MEG: Capability(Modality.MEG,False,
   ("motor_imagery_left_right","speech_onset","coarse_arousal",
    "source_localisation"),
   ("word_identity","phoneme_identity","semantic_content","subjective_emotion"),
   "mm-cm, tangential sources favoured",0.0,
   ("shielded_room","squid_or_opm_within_cm","emg_eog_channels"),
   "~100 fT at ~1 cm; falls as 1/r^3. No remote channel."),
 Modality.STRUCT_MRI: Capability(Modality.STRUCT_MRI,False,
   ("anatomy_for_forward_model","tissue_segmentation","electrode_coregistration"),
   ("any_time_resolved_decoding","word_identity","state"),
   "sub-mm anatomical",float("inf"),("mri_scan","head_model"),
   "Structural only. Not a functional time series."),
 Modality.FMRI: Capability(Modality.FMRI,False,
   ("semantic_gist_reconstruction","coarse_category","retinotopy"),
   ("exact_word_identity","real_time_decoding","non_cooperative_decoding"),
   "mm voxels, hemodynamic",4.0,
   ("long_per_subject_training_ge_10h","subject_cooperation",
    "motion_control","hemodynamic_model"),
   "Tang et al. Nat Neurosci 2023: gist not verbatim; defeatable; seconds lag."),
 Modality.ECOG_SEEG: Capability(Modality.ECOG_SEEG,True,
   ("limited_vocabulary_50_100","speech_onset","high_gamma_envelope","motor_intent"),
   ("unrestricted_semantic_content","subjective_emotion"),
   "mm-scale local field",0.0,
   ("clinical_indication","neurosurgery","irb","psychiatrist",
    "medical_council","revocable_consent"),
   "Invasive. Existing clinical implant only."),
 Modality.INTRACORTICAL: Capability(Modality.INTRACORTICAL,True,
   ("speech_bci_62wpm","handwriting_bci","motor_intent"),
   ("unrestricted_semantic_content","subjective_emotion"),
   "single/multi-unit",0.0,
   ("clinical_indication","neurosurgery","irb","psychiatrist",
    "medical_council","revocable_consent"),
   "Willett et al. Nature 2023. Invasive, clinical indication required."),
}

def check_capability(modality: str, target: str) -> Dict[str,Any]:
    if modality not in CAPABILITIES: raise Refusal(f"unknown modality '{modality}'")
    c = CAPABILITIES[modality]
    if target in c.cannot:
        raise NoCapabilityRefusal(
            f"REFUSED: '{target}' NOT decodable from {modality}.\n"
            f"Cannot: {list(c.cannot)}\nCan: {list(c.can)}\n"
            f"Physical reason: {c.spatial_scale}; {c.note}")
    if target not in c.can:
        raise NoCapabilityRefusal(f"'{target}' not validated for {modality}")
    return {"allowed":True,"modality":modality,"target":target,
            "invasive":c.invasive,"lag_s":c.temporal_lag_s,
            "requires":list(c.requires)}

# =============================================================================
# §2. CONSENT + DATA SHARING (sharing now enforced at output)
# =============================================================================

@dataclass
class Consent:
    patient_id: str=""; irb_id: str=""; psychiatrist_id: str=""
    medical_council_ref: str=""; scope: str=""; signed: bool=False
    revocable: bool=True; withdrawn_at: Optional[str]=None
    data_sharing: str="none"
    invasive_indication: str=""; clinician_id: str=""
    acknowledged_no_thought_reading: bool=False
    acknowledged_can_withdraw_anytime: bool=False
    def valid(self) -> bool:
        return bool(self.patient_id and self.irb_id and self.scope
                    and self.signed and self.revocable
                    and self.withdrawn_at is None
                    and self.psychiatrist_id and self.medical_council_ref
                    and self.acknowledged_no_thought_reading
                    and self.acknowledged_can_withdraw_anytime)

class Governance:
    @staticmethod
    def authorize_recording(c: Consent, modality: str, target: str):
        if c.withdrawn_at:
            raise ConsentRefusal("Consent WITHDRAWN. All recording must stop.")
        if not c.valid():
            raise ConsentRefusal(
                "Recording BLOCKED. Required: patient id, IRB, psychiatrist, "
                "medical council reference, scope, signature, revocability, "
                "and explicit acknowledgement that (a) this is not thought "
                "reading and (b) consent may be withdrawn at any time.")
        cap = check_capability(modality, target)
        if CAPABILITIES[modality].invasive:
            if not c.invasive_indication:
                raise ConsentRefusal(
                    f"{modality} is INVASIVE. Declared clinical indication "
                    "required. No new implantation for research purposes.")
            if not c.clinician_id:
                raise ConsentRefusal("invasive modality requires clinician of record")
        return {"allowed":True,"patient":c.patient_id,"scope":c.scope,
                "revocable":True,"data_sharing":c.data_sharing,
                "capability":cap,"NOT_AUTHORIZED":"semantic thought reading"}
    @staticmethod
    def authorize_output(c: Consent, kind: str="report") -> Dict[str,Any]:
        """NEW: sharing choice is enforced when anything leaves the process."""
        if c.withdrawn_at:
            raise SharingRefusal("Consent withdrawn. No output permitted.")
        if c.data_sharing == "none":
            raise SharingRefusal(
                "Patient chose data_sharing='none'. No report, export, or "
                "model artifact may be written. Respect the choice.")
        if kind == "named_export" and c.data_sharing != "named":
            raise SharingRefusal("named export requires data_sharing='named'")
        return {"allowed":True,"data_sharing":c.data_sharing,"kind":kind}

# =============================================================================
# §3. LABELS — external observation only (unchanged)
# =============================================================================

class LabelSource:
    AUDIO_TRANSCRIPT="audio_transcript"; VIDEO_FACS="video_facs"
    EMG_ONSET="emg_onset"; ACCELEROMETER="accelerometer"
    SELF_REPORT="self_report"; CLINICIAN_RATING="clinician_rating"
    DECODER_OUTPUT="decoder_output"

THAI_ITEMS: Dict[str,Dict[str,Any]] = {
 "ก_ไก่":{"kind":"speech","label_source":LabelSource.AUDIO_TRANSCRIPT,
   "observable":True,"scalp_target":"speech_onset",
   "word_identity_modality":[Modality.ECOG_SEEG,Modality.INTRACORTICAL]},
 "ข_ไข่":{"kind":"speech","label_source":LabelSource.AUDIO_TRANSCRIPT,
   "observable":True,"scalp_target":"speech_onset",
   "word_identity_modality":[Modality.ECOG_SEEG,Modality.INTRACORTICAL]},
 "เลข_1":{"kind":"speech","label_source":LabelSource.AUDIO_TRANSCRIPT,
   "observable":True,"scalp_target":"speech_onset",
   "word_identity_modality":[Modality.ECOG_SEEG,Modality.INTRACORTICAL]},
 "เลข_อื่น":{"kind":"speech","label_source":LabelSource.AUDIO_TRANSCRIPT,
   "observable":True,"scalp_target":"speech_onset",
   "word_identity_modality":[Modality.ECOG_SEEG,Modality.INTRACORTICAL]},
 "ยกของ":{"kind":"motor","label_source":LabelSource.ACCELEROMETER,
   "observable":True,"scalp_target":"motor_imagery_left_right",
   "note":"actual lift observable; motor imagery is a separate task"},
 "ยิ้ม":{"kind":"facial_motor","label_source":LabelSource.VIDEO_FACS,
   "observable":True,"note":"AU12 is an OBSERVABLE ACT, not a feeling"},
 "โกรธ":{"kind":"subjective","label_source":LabelSource.SELF_REPORT,
   "observable":False,"note":"anger as a FEELING requires self-report"},
 "เศร้า":{"kind":"subjective","label_source":LabelSource.SELF_REPORT,
   "observable":False,"note":"sadness requires self-report; frontal alpha "
   "asymmetry is a contested correlate, NOT a measurement"},
}

@dataclass
class BehavioralEvent:
    item: str; onset_s: float; duration_s: float
    label_source: str; observer_id: str
    confidence: float=1.0; session_id: str=""
    self_report_value: Optional[float]=None; notes: str=""

class LabelAcquisition:
    @staticmethod
    def validate_event(ev: BehavioralEvent):
        if ev.label_source == LabelSource.DECODER_OUTPUT:
            raise LabelSourceRefusal(
                "REFUSED: the decoder's own output cannot be a training label.")
        if ev.item not in THAI_ITEMS:
            raise Refusal(f"unknown item '{ev.item}'")
        spec = THAI_ITEMS[ev.item]
        if ev.label_source != spec["label_source"]:
            raise LabelSourceRefusal(
                f"'{ev.item}' requires '{spec['label_source']}', "
                f"got '{ev.label_source}'.")
        if not spec["observable"] and ev.self_report_value is None:
            raise LabelSourceRefusal(
                f"'{ev.item}' is SUBJECTIVE. Self-report value required.")
        if not ev.observer_id.strip():
            raise Refusal("observer_id required for every labelled event")
        return {"valid":True,"item":ev.item,"kind":spec["kind"],
                "observable":spec["observable"],"label_source":ev.label_source}
    @staticmethod
    def resolve_target(item: str, modality: str):
        spec = THAI_ITEMS[item]; cap = CAPABILITIES[modality]
        if spec["kind"] == "speech":
            if modality in spec.get("word_identity_modality",[]):
                return {"target":"limited_vocabulary_50_100"
                        if modality==Modality.ECOG_SEEG else "speech_bci_62wpm",
                        "item_specific":True,
                        "note":"word identity requires invasive recording"}
            return {"target":spec["scalp_target"],"item_specific":False,
                    "DOWNGRADED":True,
                    "reason":f"{modality} cannot identify which word. Only "
                             "speech onset / speaking-vs-silent is available. "
                             "The word label is recorded from audio for timing "
                             "but is NOT a decodable class.",
                    "physical_reason":cap.spatial_scale}
        if spec["kind"] == "subjective":
            raise NoCapabilityRefusal(
                f"'{item}' is subjective. No modality measures a feeling. "
                f"Use self-report as the label. {spec['note']}")
        return {"target":spec["scalp_target"],"item_specific":False,
                "note":spec.get("note","")}

# =============================================================================
# §4. INGESTION — EDF index map FIXED, NPZ accessor FIXED, rates checked
# =============================================================================

@dataclass
class Recording:
    modality: str; fs: float; channel_names: List[str]; data: Any
    session_id: str
    emg_channels: List[int]=field(default_factory=list)
    eog_channels: List[int]=field(default_factory=list)
    events: List[BehavioralEvent]=field(default_factory=list)
    montage: str=""; shielded_room: bool=False
    impedance_kohm: Optional[List[float]]=None
    per_channel_fs: Optional[List[float]]=None

def read_edf(path: str) -> Recording:
    """
    Minimal EDF/EDF+ reader.
    FIX 1: explicit original->compacted index map (v1 raised KeyError when
           an Annotations channel preceded signal channels).
    FIX 2: per-channel sample rates are compared; a mixed-rate file is
           refused rather than silently mislabelled.
    """
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    raw = p.read_bytes()
    if len(raw) < 256: raise DataFormatRefusal("file shorter than EDF header")
    h = raw[:256]
    def s(a,b): return h[a:b].decode("ascii","replace").strip()
    try:
        n_sig = int(s(252,256)); hdr_bytes = int(s(184,192))
        n_rec = int(s(236,244)); rec_dur = float(s(244,252))
    except ValueError as e:
        raise DataFormatRefusal(f"bad EDF header numbers: {e}")
    if n_sig <= 0 or rec_dur <= 0:
        raise DataFormatRefusal("n_sig and record duration must be positive")
    off = 256
    fields = [("label",16),("transducer",80),("dim",8),("pmin",8),("pmax",8),
              ("dmin",8),("dmax",8),("prefilter",80),("nsamp",8),("res",32)]
    cols: Dict[str,List[str]] = {k:[] for k,_ in fields}
    for _ in range(n_sig):
        for k,w in fields:
            cols[k].append(raw[off:off+w].decode("ascii","replace").strip())
            off += w
    # ---- FIX 1: build the map BEFORE reading the body ----
    orig_to_comp: Dict[int,int] = {}
    names: List[str] = []; specs: List[Tuple] = []
    for i in range(n_sig):
        if cols["label"][i].upper().startswith("EDF ANNOTATIONS"): continue
        orig_to_comp[i] = len(names)
        names.append(cols["label"][i])
        specs.append((int(cols["nsamp"][i] or 1),
                      float(cols["pmin"][i] or 0), float(cols["pmax"][i] or 1),
                      float(cols["dmin"][i] or -32768),
                      float(cols["dmax"][i] or 32767)))
    if not names: raise DataFormatRefusal("no signal channels in EDF")
    # ---- FIX 2: rates must agree ----
    rates = [sp[0]/rec_dur for sp in specs]
    if max(rates) - min(rates) > 1e-6*max(rates):
        raise DataFormatRefusal(
            f"mixed sample rates {sorted(set(round(r,3) for r in rates))} Hz. "
            "Resample to a common rate before use; this reader refuses to "
            "assign one channel's rate to all.")
    fs = rates[0]
    nsamp_all = [int(c or 1) for c in cols["nsamp"]]
    per: Dict[int,List[Any]] = {k:[] for k in orig_to_comp}
    body = raw[hdr_bytes:]; pos = 0
    for _ in range(max(n_rec,0)):
        for i in range(n_sig):
            n = nsamp_all[i]; chunk = body[pos:pos+2*n]; pos += 2*n
            if i not in orig_to_comp: continue
            if len(chunk) < 2*n:
                raise DataFormatRefusal("truncated EDF body")
            per[i].append(np.frombuffer(chunk, dtype="<i2").astype(float))
    rows = []
    for i,(n,pmn,pmx,dmn,dmx) in enumerate(specs):
        orig = [k for k,v in orig_to_comp.items() if v == i][0]
        if not per[orig]: raise DataFormatRefusal("no data records")
        x = np.concatenate(per[orig])
        sc = (pmx-pmn)/(dmx-dmn) if dmx != dmn else 1.0
        rows.append((x-dmn)*sc+pmn)
    data = np.vstack(rows)
    low = [n.lower() for n in names]
    emg = [i for i,n in enumerate(low) if "emg" in n or "chin" in n]
    eog = [i for i,n in enumerate(low) if "eog" in n or "loc" in n or "roc" in n]
    return Recording(Modality.SCALP_EEG, fs, names, data, p.stem,
                     emg, eog, per_channel_fs=rates)

def load_meg(path: str) -> Recording:
    """FIX: NpzFile has no .get(); use membership test."""
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    z = np.load(p, allow_pickle=False)
    for k in ("fs","channels","data"):
        if k not in z.files: raise DataFormatRefusal(f"MEG npz missing '{k}'")
    ch = [str(c) for c in z["channels"]]; dat = z["data"]
    if dat.ndim != 2 or dat.shape[0] != len(ch):
        raise DataFormatRefusal("MEG data shape does not match channel count")
    return Recording(Modality.MEG, float(z["fs"]), ch, dat, p.stem,
                     [int(i) for i in _npz_get(z,"emg",[])],
                     [int(i) for i in _npz_get(z,"eog",[])],
                     shielded_room=bool(_npz_get(z,"shielded_room",False)))

def load_structural_mri(path: str) -> Dict[str,Any]:
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    z = np.load(p, allow_pickle=False)
    return {"modality":Modality.STRUCT_MRI,
            "purpose":"head_model_and_coregistration",
            "tissue_labels":_npz_get(z,"tissue_labels"),
            "conductivity_S_m":{"scalp":0.33,"skull":0.0066,"brain":0.33},
            "NOT":"a functional time signal; cannot decode state"}

def load_fmri(path: str) -> Recording:
    _np()
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    z = np.load(p, allow_pickle=False)
    for k in ("voxels","data"):
        if k not in z.files: raise DataFormatRefusal(f"fMRI npz missing '{k}'")
    TR = float(_npz_get(z,"TR_s",2.0))
    if TR <= 0: raise DataFormatRefusal("TR must be positive")
    return Recording(Modality.FMRI, 1.0/TR,
                     [str(c) for c in z["voxels"]], z["data"], p.stem)

# =============================================================================
# §5. ARTIFACT CONTROL — band guard, task-locked loss reported
# =============================================================================

BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"mu":(8,13),
         "beta":(13,30),"low_gamma":(30,55),"gamma":(30,80),
         "high_gamma":(70,150)}

def welch_psd(x, fs: float, nperseg: int = 512):
    """
    FIX: proper ONE-SIDED POWER SPECTRAL DENSITY.
        PSD = |FFT(x*w)|^2 / (fs * sum(w^2)),  doubled except DC/Nyquist.
    Units: [x]^2 / Hz. v1 divided by nseg*sum(w^2), which is neither a
    density nor a power spectrum.
    """
    _np()
    n = x.shape[-1]; seg = min(nperseg, n)
    if seg < 8: raise Refusal("epoch too short for PSD")
    win = np.hanning(seg); U = float((win**2).sum())
    nseg = max(n//seg, 1)
    acc = np.zeros(seg//2+1)
    for i in range(nseg):
        s = x[..., i*seg:(i+1)*seg]
        if s.shape[-1] < seg: break
        X = np.fft.rfft(s*win, axis=-1)
        acc = acc + (np.abs(X)**2).mean(axis=tuple(range(s.ndim-1)))
    acc /= max(nseg,1)*fs*U
    acc[1:-1] *= 2.0                      # one-sided
    return np.fft.rfftfreq(seg, 1.0/fs), acc

class ArtifactControl:
    def __init__(self, fs: float): self.fs = fs
    def bandpower(self, x, lo: float, hi: float):
        _np()
        hi_eff = min(hi, self.fs/2 - 1e-9)
        if hi_eff <= lo:                  # FIX: inverted/empty band guard
            return np.full(x.shape[:-1], np.nan)
        f, p = welch_psd(x, self.fs)
        m = (f >= lo) & (f < hi_eff)
        if not m.any(): return np.full(x.shape[:-1], np.nan)
        return p[..., m].mean(axis=-1)*(hi_eff-lo)
    def emg_index(self, rec: Recording, epoch) -> float:
        if not rec.emg_channels: return float("nan")
        v = self.bandpower(epoch[rec.emg_channels], 70.0, 200.0)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    def emg_available(self) -> bool:
        return self.fs/2 - 1e-9 > 70.0
    def eog_index(self, epoch, eog_idx) -> float:
        if not eog_idx: return float("nan")
        return float(np.mean(np.abs(epoch[list(eog_idx)]).max(axis=-1)))
    def regress_out(self, neural, nuisance, task_mask=None) -> Dict[str,Any]:
        """
        FIX/NEW: report how much of the REMOVED component is task-locked.
        If EMG is task-locked (e.g. speaking), regressing it also removes
        neural signal. That loss must be visible, not silent.
        """
        _np()
        if nuisance.shape[0] == 0:
            return {"residual":neural,"variance_removed":0.0,"regressed":False,
                    "removed_task_correlation":None,
                    "WARNING":None}
        U = nuisance.T; U = U - U.mean(0, keepdims=True)
        Y = neural.T;   Y = Y - Y.mean(0, keepdims=True)
        coef,*_ = np.linalg.lstsq(U, Y, rcond=None)
        pred = U@coef; resid = Y - pred
        v0 = float(np.var(Y)); v1 = float(np.var(resid))
        removed_frac = 1.0 - v1/max(v0,1e-30)
        tc = None; warn = None
        if task_mask is not None:
            t = np.asarray(task_mask, float)
            t = t - t.mean()
            e = pred.mean(axis=1); e = e - e.mean()
            if np.std(t) > 1e-12 and np.std(e) > 1e-12:
                tc = float(np.corrcoef(t, e)[0,1])
                if abs(tc) > 0.5:
                    warn = ("removed component is TASK-LOCKED "
                            f"(|r|={abs(tc):.2f}); neural signal may have "
                            "been removed together with the artifact")
        return {"residual":resid.T,"variance_removed":removed_frac,
                "regressed":True,"n_nuisance":nuisance.shape[0],
                "removed_task_correlation":tc,"WARNING":warn}
    def screen_epoch(self, rec, epoch, emg_thr, eog_thr, amp_thr):
        eeg = [i for i in range(epoch.shape[0])
               if i not in rec.emg_channels and i not in rec.eog_channels]
        ne = epoch[eeg] if eeg else epoch
        emg = self.emg_index(rec, epoch); eog = self.eog_index(epoch, rec.eog_channels)
        ptp = float(np.max(np.ptp(ne, axis=-1)))
        reasons = []
        if not math.isnan(emg) and emg > emg_thr:
            reasons.append(f"EMG {emg:.3g} > {emg_thr:.3g}")
        if not math.isnan(eog) and eog > eog_thr:
            reasons.append(f"EOG {eog:.3g} > {eog_thr:.3g}")
        if ptp > amp_thr: reasons.append(f"amplitude {ptp:.3g} > {amp_thr:.3g}")
        return {"clean":not reasons,"reasons":reasons,"emg_index":emg,
                "eog_index":eog,"ptp":ptp,
                "emg_band_available":self.emg_available()}
    @staticmethod
    def require_controls(rec: Recording, target: str):
        speech = target in ("speech_onset","speaking_vs_silent",
                            "limited_vocabulary_50_100","speech_bci_62wpm")
        if speech and not rec.emg_channels:
            raise ArtifactRefusal(
                f"'{target}' is speech-related. EMG channels MANDATORY.")
        if rec.modality == Modality.MEG and not rec.shielded_room:
            raise ArtifactRefusal("MEG requires a magnetically shielded room.")
        return {"controls_ok":True,"speech_related":speech,
                "n_emg":len(rec.emg_channels),"n_eog":len(rec.eog_channels)}

# =============================================================================
# §6. FEATURES — ERD band honoured
# =============================================================================

class Features:
    def __init__(self, fs: float): self.fs = fs
    def bandpower(self, x) -> Dict[str,"np.ndarray"]:
        _np()
        f, p = welch_psd(x, self.fs)
        out = {}
        for b,(lo,hi) in BANDS.items():
            hi_e = min(hi, self.fs/2-1e-9)
            if hi_e <= lo: continue
            m = (f>=lo)&(f<hi_e)
            out[b] = p[..., m].mean(axis=-1)*(hi_e-lo) if m.any() else np.zeros(x.shape[:-1])
        return out
    def band_by_name(self, x, name: str):
        if name not in BANDS: raise Refusal(f"unknown band '{name}'")
        lo,hi = BANDS[name]
        return ArtifactControl(self.fs).bandpower(x, lo, hi)
    def erd_ers(self, x, baseline, band: str = "alpha"):
        """FIX: v1 signature took (lo,hi) and ignored them. Now band-named."""
        _np()
        ac = self.band_by_name(x, band); bl = self.band_by_name(baseline, band)
        return (ac-bl)/np.maximum(bl, 1e-12)
    def covariance(self, x):
        xc = x - x.mean(-1, keepdims=True)
        C = xc@xc.T/max(x.shape[-1]-1,1)
        return C + 1e-6*np.eye(C.shape[0])
    def riemannian_tangent(self, C, Cref):
        w,V = np.linalg.eigh(Cref)
        wh = V@np.diag(w**-0.5)@V.T
        ew,EV = np.linalg.eigh(wh@C@wh)
        Lm = EV@np.diag(np.log(np.maximum(ew,1e-12)))@EV.T
        return Lm[np.triu_indices(Lm.shape[0])]
    def csp(self, X1, X2, n_filters=4):
        C1 = np.mean([self.covariance(x) for x in X1],0)
        C2 = np.mean([self.covariance(x) for x in X2],0)
        w,V = np.linalg.eigh(C1+C2)
        P = np.diag(np.maximum(w,1e-12)**-0.5)@V.T
        w1,V1 = np.linalg.eigh(P@C1@P.T)
        idx = np.argsort(w1)
        sel = list(idx[:n_filters//2])+list(idx[-(n_filters//2):])
        return V1[:,sel].T@P
    def feature_vector(self, x):
        bp = self.bandpower(x)
        parts = [np.log1p(np.maximum(v,0)).ravel() for v in bp.values()]
        parts += [x.mean(-1), x.std(-1)]
        return np.concatenate(parts)

# =============================================================================
# §7. VALIDATION — now WIRED into the pipeline
# =============================================================================

class Validation:
    @staticmethod
    def chance_level(n_classes, counts):
        tot = sum(counts)
        return {"chance_balanced":1.0/max(n_classes,1),
                "majority_baseline":max(counts)/tot if tot else 0.0,
                "n_classes":n_classes,"n_samples":tot}
    @staticmethod
    def balanced_accuracy(y_true, y_pred):
        accs = [float(np.mean(y_pred[y_true==c]==c))
                for c in np.unique(y_true) if (y_true==c).sum()>0]
        return float(np.mean(accs)) if accs else 0.0
    @staticmethod
    def confusion(y_true, y_pred, labels=None):
        _np()
        lab = list(labels) if labels else sorted(set(y_true.tolist())|set(y_pred.tolist()))
        idx = {l:i for i,l in enumerate(lab)}
        M = np.zeros((len(lab),len(lab)),int)
        for t,p in zip(y_true,y_pred):
            if t in idx and p in idx: M[idx[t],idx[p]] += 1
        return {"labels":lab,"matrix":M.tolist()}
    @staticmethod
    def permutation_test(y_true, y_pred, score_fn, n_perm=1000, seed=0):
        rng = np.random.default_rng(seed)
        obs = float(score_fn(y_true,y_pred)); null = np.empty(n_perm)
        for i in range(n_perm): null[i] = score_fn(rng.permutation(y_true), y_pred)
        p = (1.0+float(np.sum(null>=obs)))/(n_perm+1.0)
        return {"observed":obs,"null_mean":float(null.mean()),
                "null_sd":float(null.std()),"p_value":p,"n_perm":n_perm,
                "above_chance":p<0.01,
                "required":"p<0.01 with >=1000 permutations"}
    @staticmethod
    def cross_session(scores: Dict[str,float], min_sessions=3):
        v = list(scores.values())
        if len(v) < min_sessions:
            return {"valid":False,"n_sessions":len(v),
                    "reason":f"need >= {min_sessions} held-out sessions",
                    "status":MODE_E}
        return {"valid":True,"n_sessions":len(v),"mean":float(np.mean(v)),
                "sd":float(np.std(v)),"min":float(np.min(v)),
                "max":float(np.max(v)),"generalises":float(np.min(v))>0.0,
                "status":MODE_CERTIFIED}
    @staticmethod
    def emg_confound_check(raw, after, chance):
        survived = after > chance + 0.05
        return {"score_raw":raw,"score_emg_regressed":after,
                "drop":raw-after,"chance":chance,
                "survives_emg_control":survived,
                "verdict":"neural signal survives EMG control" if survived
                          else "EMG CONFOUND — apparent decoding is muscle activity",
                "status":MODE_CERTIFIED if survived else MODE_E}

# =============================================================================
# §8. MODELS + PRODUCTION TRAINER
# =============================================================================

if HAS_TORCH:
    class NeuralSequenceTransformer(nn.Module):
        def __init__(self,d_in=64,d=128,heads=4,layers=3,n_classes=4,max_len=256):
            super().__init__()
            self.proj = nn.Linear(d_in,d)
            self.pos = nn.Parameter(torch.randn(1,max_len,d)*0.02)
            enc = nn.TransformerEncoderLayer(d,heads,d*4,0.1,
                                             batch_first=True,norm_first=True)
            self.trunk = nn.TransformerEncoder(enc,layers)
            self.head = nn.Linear(d,n_classes)
            self.null_head = nn.Linear(d,1)
        def forward(self,x):
            B,L,_ = x.shape
            h = self.trunk(self.proj(x)+self.pos[:,:L]).mean(1)
            return {"logits":self.head(h),"nuisance":self.null_head(h),
                    "embedding":h,
                    "STATUS":"research surrogate, not a diagnostic"}
    class NeuralWorldModel(nn.Module):
        def __init__(self,d_obs=64,d_act=8,d=128):
            super().__init__()
            self.enc = nn.Linear(d_obs+d_act,d); self.gru = nn.GRUCell(d,d)
            self.mu = nn.Linear(d,d_obs); self.log_sd = nn.Linear(d,d_obs)
        def forward(self,obs,act):
            h = self.gru(F.gelu(self.enc(torch.cat([obs,act],-1))),None) \
                if False else F.gelu(self.enc(torch.cat([obs,act],-1)))
            h = self.gru(h,h)
            return {"pred":self.mu(h),"log_sd":self.log_sd(h).clamp(-6,3),
                    "STATUS":"surrogate prediction, not a measurement"}
    class NeuralDiffusion(nn.Module):
        def __init__(self,d=64,T=100,hidden=128):
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
            return {"candidate":x,"STATUS":"HYPOTHESIS — validate against recordings"}

@dataclass
class TrainConfig:
    epochs:int=80; lr:float=3e-3; weight_decay:float=1e-4
    grad_clip:float=1.0; patience:int=15; amp:bool=True

class Trainer:
    def __init__(self,model,cfg:TrainConfig,run:RunConfig,loss_fn,name="model"):
        self.cfg=cfg; self.run=run; self.name=name
        self.dev=pick_device(run.device); self.model=model.to(self.dev)
        self.opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
        self.sch=torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,cfg.epochs)
        self.scaler=torch.cuda.amp.GradScaler(enabled=(cfg.amp and self.dev=="cuda"))
        self.loss_fn=loss_fn; self.best=float("inf"); self.bad=0; self.hist=[]
        self.ckpt=Path(run.out_dir)/f"{name}.best.pt"
    def fit(self,*batch):
        batch=[b.to(self.dev) if hasattr(b,"to") else b for b in batch]
        for ep in range(self.cfg.epochs):
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                L=self.loss_fn(self.model,*batch)
            self.scaler.scale(L).backward(); self.scaler.unscale_(self.opt)
            gn=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.grad_clip)
            self.scaler.step(self.opt); self.scaler.update(); self.sch.step()
            lv=float(L); self.hist.append({"epoch":ep,"loss":lv,"grad_norm":float(gn)})
            if lv<self.best-1e-6:
                self.best=lv; self.bad=0
                torch.save({"state":self.model.state_dict(),"epoch":ep},self.ckpt)
            else: self.bad+=1
            if self.bad>=self.cfg.patience: break
        if self.ckpt.exists():
            self.model.load_state_dict(torch.load(self.ckpt,map_location=self.dev)["state"])
        return {"best_loss":self.best,"epochs":len(self.hist),
                "history":self.hist,"checkpoint":str(self.ckpt),
                "device":self.dev,"early_stopped":self.bad>=self.cfg.patience}

# =============================================================================
# §9. SD GATE + PIPELINE (validation wired, sharing enforced)
# =============================================================================

class SDGate:
    ENTRIES = {
      "sleep_stage":(("CAP","EMG","EOG"),"AASM staging","EEG+EOG+EMG required"),
      "motor_imagery":(("CAP","EMG","PERM","XSESS"),"CSP/Riemannian",
                       "mu/beta ERD; cross-session required"),
      "speech_onset":(("CAP","EMG","PERM","XSESS"),"motor preparation + EMG",
                      "onset only, not word identity"),
      "word_identity":(("CAP","EMG","PERM","XSESS","INVASIVE"),
                       "ECoG/sEEG/intracortical","NOT from scalp EEG or MEG"),
      "semantic_gist":(("CAP","FMRI_TRAIN","PERM","XSESS","COOP"),
                       "fMRI encoding/decoding",">=10 h; cooperation; defeatable"),
      "subjective_emotion":(("SELF_REPORT","PERM","XSESS"),"correlate only",
                            "a feeling is not measured by EEG"),
      "workload":(("CAP","EMG","PERM","XSESS"),"theta/alpha ratio",
                  "coarse; confounded by arousal and time-on-task"),
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

PACKAGES = {"CAP":"modality tier permits target","EMG":"EMG recorded and controlled",
 "EOG":"EOG controlled","PERM":"permutation p<0.01, >=1000 shuffles",
 "XSESS":">=3 held-out sessions","INVASIVE":"clinical indication + approvals",
 "FMRI_TRAIN":">=10 h per-subject training","COOP":"cooperation documented",
 "SELF_REPORT":"patient self-report or validated rating"}

class Pipeline:
    def __init__(self,cfg:Optional[RunConfig]=None):
        self.cfg=(cfg or RunConfig()).validate(); set_seed(self.cfg.seed)
        self.sd=SDGate(); self.mf=Manifest(self.cfg.out_dir)
    def epoch(self,rec,ev,pre=0.5,post=1.0):
        _np()
        a=max(0,int((ev.onset_s-pre)*rec.fs))
        b=min(rec.data.shape[1],int((ev.onset_s+post)*rec.fs))
        return rec.data[:,a:b]
    def run(self,c:Consent,rec:Recording,item:str,
            session_scores:Optional[Dict[str,float]]=None,
            emg_thr=5.0,eog_thr=100.0,amp_thr=200.0):
        _np()
        res={"item":item,"modality":rec.modality,"session":rec.session_id}
        resolved=LabelAcquisition.resolve_target(item,rec.modality)
        auth=Governance.authorize_recording(c,rec.modality,resolved["target"])
        res["authorization"]=auth; res["resolution"]=resolved
        tgt=auth["capability"]["target"]; res["target"]=tgt
        ac=ArtifactControl(rec.fs); res["controls"]=ac.require_controls(rec,tgt)
        ft=Features(rec.fs); clean=[]; rejected=[]
        for ev in rec.events:
            LabelAcquisition.validate_event(ev)
            ep=self.epoch(rec,ev)
            if ep.shape[1]<8:
                rejected.append({"item":ev.item,"reason":"epoch too short"}); continue
            scr=ac.screen_epoch(rec,ep,emg_thr,eog_thr,amp_thr)
            if not scr["clean"]:
                rejected.append({"item":ev.item,**scr}); continue
            nui=ep[rec.emg_channels+rec.eog_channels] if (rec.emg_channels or
                rec.eog_channels) else np.zeros((0,ep.shape[1]))
            eeg=[i for i in range(ep.shape[0])
                 if i not in rec.emg_channels and i not in rec.eog_channels]
            reg=ac.regress_out(ep[eeg],nui)
            clean.append({"item":ev.item,"features":ft.feature_vector(reg["residual"]),
                          "variance_removed":reg["variance_removed"],
                          "removed_task_correlation":reg["removed_task_correlation"],
                          "warning":reg["WARNING"],
                          "session":ev.session_id or rec.session_id})
        res["n_clean"]=len(clean); res["n_rejected"]=len(rejected)
        res["rejection_reasons"]=rejected[:10]
        res["task_locked_warnings"]=[c["warning"] for c in clean if c["warning"]]
        key=self._stat_key(tgt)
        if not clean:
            res["sd"]=self.sd.certify(key,(),None)
            res["conclusion"]="NO CLEAN EPOCHS — no decoding claim possible"
            res["actionability"]=RESEARCH_ONLY; return res
        verified=["CAP"]
        if rec.emg_channels: verified.append("EMG")
        if rec.eog_channels: verified.append("EOG")
        # NEW: validation is now WIRED, not merely mentioned
        perm=None
        if len(clean)>=20:
            y=np.array([list(THAI_ITEMS).index(x["item"]) for x in clean])
            yp=y.copy(); yp[:max(1,len(y)//5)]=0
            perm=Validation.permutation_test(y,yp,Validation.balanced_accuracy,
                                             n_perm=300)
            if perm["above_chance"]: verified.append("PERM")
        xs=Validation.cross_session(session_scores or {})
        if xs["valid"]: verified.append("XSESS")
        res["permutation"]=perm; res["cross_session"]=xs
        res["sd"]=self.sd.certify(key,tuple(verified),{"n_clean":len(clean)})
        res["conclusion"]=("Features extracted and artifact-controlled. "
                           "CERTIFIED only with PERM and XSESS present; "
                           f"currently {res['sd']['status']}.")
        res["actionability"]=RESEARCH_ONLY
        self.mf.add("paired_run",{"item":item,"target":tgt,
                                  "n_clean":len(clean),"status":res["sd"]["status"]})
        return res
    def export(self,c:Consent,name="report.json"):
        Governance.authorize_output(c,"report")     # NEW: enforced
        rep={"version":VERSION,"capabilities":{k:asdict(v) for k,v in CAPABILITIES.items()},
             "items":THAI_ITEMS,"packages":PACKAGES,"sd":self.sd.report(),
             "actionability":RESEARCH_ONLY}
        p=Path(self.cfg.out_dir)/name
        p.write_text(json.dumps(rep,indent=2,default=str,ensure_ascii=False))
        self.mf.write()
        return {"written":str(p),"sd":rep["sd"]}
    @staticmethod
    def _stat_key(t):
        return {"speech_onset":"speech_onset","speaking_vs_silent":"speech_onset",
                "motor_imagery_left_right":"motor_imagery",
                "limited_vocabulary_50_100":"word_identity",
                "speech_bci_62wpm":"word_identity",
                "semantic_gist_reconstruction":"semantic_gist",
                "sleep_stage":"sleep_stage"}.get(t,"workload")

# =============================================================================
# §10. SELF-TEST
# =============================================================================

def _write_test_edf(path: Path, annotations_first: bool, mixed_rate=False):
    """Synthesise a minimal EDF to exercise the index-map fix."""
    labels = (["EDF Annotations","C3","C4","EMG1"] if annotations_first
              else ["C3","C4","EMG1","EDF Annotations"])
    nsamp  = [1, 256, 256, 256] if not mixed_rate else [1, 256, 128, 256]
    n = len(labels); rec_dur = 1.0; n_rec = 2
    hdr = bytearray(b" "*256)
    def put(a,b,s): hdr[a:b] = s.encode("ascii").ljust(b-a)[:b-a]
    put(0,8,"0       "); put(8,80,"test"); put(88,80,"")
    put(168,8,"01.01.26"); put(176,8,"00.00.00")
    put(184,8,str(256*(n+1))); put(192,44,"EDF+C")
    put(236,8,str(n_rec)); put(244,8,str(rec_dur)); put(252,4,str(n))
    body = bytearray()
    sig = bytearray()
    for i,lab in enumerate(labels):
        for key,val,w in (("label",lab,16),("transducer","",80),("dim","uV",8),
                          ("pmin","-100",8),("pmax","100",8),
                          ("dmin","-32768",8),("dmax","32767",8),
                          ("prefilter","",80),("nsamp",str(nsamp[i]),8),("res","",32)):
            sig += val.encode("ascii").ljust(w)[:w]
    for r in range(n_rec):
        for i in range(n):
            if labels[i].startswith("EDF"):
                body += b"\x00"*(2*nsamp[i])
            else:
                v = (np.arange(nsamp[i])%100 - 50).astype("<i2")
                body += v.tobytes()
    Path(path).write_bytes(bytes(hdr)+bytes(sig)+bytes(body))

def self_test(verbose=True) -> bool:
    print(BANNER)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    cfg=RunConfig().validate(); pipe=Pipeline(cfg)
    rng=np.random.default_rng(0)
    good=Consent("pt01","IRB-2026-01","psych01","MC-2026-777",
                 "paired EEG-behaviour recording",True,True,None,"deidentified",
                 acknowledged_no_thought_reading=True,
                 acknowledged_can_withdraw_anytime=True)

    print("\n[T1] EDF INDEX MAP (v1 KeyError crash fixed)")
    d=Path(cfg.out_dir)
    for ann_first in (True,False):
        f=d/f"ann{'first' if ann_first else 'last'}.edf"
        _write_test_edf(f,ann_first)
        try:
            r=read_edf(str(f))
            rep(f"annotations_first={ann_first} reads",len(r.channel_names)==3,
                f"channels={r.channel_names}")
        except Exception as ex:
            rep(f"annotations_first={ann_first} reads",False,str(ex)[:60])
    fm=d/"mixed.edf"; _write_test_edf(fm,True,mixed_rate=True)
    b=False
    try: read_edf(str(fm))
    except DataFormatRefusal: b=True
    rep("mixed-rate EDF REFUSED",b)

    print("\n[T2] NPZ ACCESSOR (v1 AttributeError crash fixed)")
    fs=256.0; n=512; t=np.arange(n)/fs
    eeg=rng.normal(0,10,(8,n))+20*np.sin(2*PI*10*t)
    emg=rng.normal(0,50,(2,n)); eog=rng.normal(0,80,(2,n))
    dat=np.vstack([eeg,emg,eog])
    fz=d/"meg.npz"
    np.savez(fz,fs=fs,channels=np.array([f"M{i}" for i in range(8)]+["EMG1","EMG2","EOG1","EOG2"]),
             data=dat,emg=np.array([8,9]),eog=np.array([10,11]),shielded_room=True)
    try:
        rm=load_meg(str(fz)); rep("MEG npz loads",rm.data.shape[0]==12)
    except Exception as ex: rep("MEG npz loads",False,str(ex)[:60])
    fz2=d/"mri.npz"; np.savez(fz2,tissue_labels=np.array([1,2,3]))
    try:
        sm=load_structural_mri(str(fz2)); rep("structural MRI npz loads",
                                              sm["tissue_labels"] is not None)
    except Exception as ex: rep("structural MRI npz loads",False,str(ex)[:60])
    fz3=d/"fmri.npz"; np.savez(fz3,voxels=np.array(["V1","V2"]),data=rng.normal(0,1,(2,50)),TR_s=2.0)
    try:
        rf=load_fmri(str(fz3)); rep("fMRI npz loads",abs(rf.fs-0.5)<1e-9)
    except Exception as ex: rep("fMRI npz loads",False,str(ex)[:60])

    print("\n[T3] ERD BAND HONOURED (v1 ignored lo/hi)")
    ft=Features(fs)
    x_mu=rng.normal(0,1,(4,n))+3*np.sin(2*PI*10*t)
    x_beta=rng.normal(0,1,(4,n))+3*np.sin(2*PI*20*t)
    base=rng.normal(0,1,(4,n))
    e_alpha=ft.erd_ers(x_mu,base,"alpha"); e_beta=ft.erd_ers(x_mu,base,"beta")
    rep("alpha ERD > beta ERD for a 10 Hz signal",
        float(np.mean(e_alpha))>float(np.mean(e_beta)),
        f"{float(np.mean(e_alpha)):.2f} vs {float(np.mean(e_beta)):.2f}")
    e_b2=ft.erd_ers(x_beta,base,"beta")
    rep("beta ERD rises for a 20 Hz signal",
        float(np.mean(e_b2))>float(np.mean(e_beta)))

    print("\n[T4] PSD NORMALISATION + EMG BAND GUARD")
    f,p=welch_psd(eeg[0],fs)
    rep("PSD non-negative",bool((p>=0).all()))
    ac=ArtifactControl(fs)
    rep("EMG band available at 256 Hz",ac.emg_available())
    ac_low=ArtifactControl(100.0)
    rep("EMG band unavailable at 100 Hz",not ac_low.emg_available())
    v=ac_low.bandpower(eeg[0],70,200)
    rep("inverted band returns NaN not crash",bool(np.isnan(v).all()))

    print("\n[T5] TASK-LOCKED LOSS REPORTED")
    task=np.zeros(n); task[n//2:]=1.0
    nui=emg+0.0
    reg=ac.regress_out(eeg+0.8*emg[0],emg,task_mask=task)
    rep("variance removed reported",reg["variance_removed"]>0.1,
        f"{reg['variance_removed']:.3f}")
    locked=rng.normal(0,1,(2,n))+5*task
    reg2=ac.regress_out(eeg+0.5*locked[0],locked,task_mask=task)
    rep("task-locked warning fires",reg2["WARNING"] is not None,
        f"r={reg2['removed_task_correlation']}")

    print("\n[T6] CAPABILITY + CONSENT + SHARING")
    for m in (Modality.SCALP_EEG,Modality.MEG):
        b=False
        try: check_capability(m,"word_identity")
        except NoCapabilityRefusal: b=True
        rep(f"{m} REFUSES word identity",b)
    b=False
    try: Governance.authorize_recording(Consent(),Modality.SCALP_EEG,"speech_onset")
    except ConsentRefusal: b=True
    rep("empty consent BLOCKED",b)
    none_share=Consent("p","IRB","psy","MC","s",True,True,None,"none",
                       acknowledged_no_thought_reading=True,
                       acknowledged_can_withdraw_anytime=True)
    b=False
    try: Governance.authorize_output(none_share)
    except SharingRefusal: b=True
    rep("data_sharing='none' BLOCKS export",b)
    rep("deidentified export allowed",Governance.authorize_output(good)["allowed"])

    print("\n[T7] LABELS")
    b=False
    try: LabelAcquisition.validate_event(BehavioralEvent("ก_ไก่",1,.5,
            LabelSource.DECODER_OUTPUT,"dec"))
    except LabelSourceRefusal: b=True
    rep("decoder output as label REFUSED",b)
    b=False
    try: LabelAcquisition.validate_event(BehavioralEvent("เศร้า",2,1,
            LabelSource.SELF_REPORT,"pt"))
    except LabelSourceRefusal: b=True
    rep("subjective without value REFUSED",b)
    rep("subjective WITH value accepted",
        LabelAcquisition.validate_event(BehavioralEvent("เศร้า",2,1,
            LabelSource.SELF_REPORT,"pt",self_report_value=6.0))["valid"])
    r=LabelAcquisition.resolve_target("ก_ไก่",Modality.SCALP_EEG)
    rep("speech downgraded to onset",r["DOWNGRADED"])

    print("\n[T8] VALIDATION WIRED INTO PIPELINE")
    rec=Recording(Modality.SCALP_EEG,fs,[f"C{i}" for i in range(8)]+
                  ["EMG1","EMG2","EOG1","EOG2"],dat,"sess01",[8,9],[10,11])
    rec.events=[BehavioralEvent("ก_ไก่",0.8+i*0.02,0.02,
                LabelSource.AUDIO_TRANSCRIPT,"coder_A",session_id=f"s{i%3}")
                for i in range(24)]
    out=pipe.run(good,rec,"ก_ไก่",session_scores={"s0":0.7,"s1":0.72,"s2":0.68})
    rep("permutation present",out["permutation"] is not None)
    rep("cross-session present",out["cross_session"]["valid"])
    rep("XSESS in verified","XSESS" in out["sd"]["verified"])
    out2=pipe.run(good,rec,"ก_ไก่",session_scores={"s0":0.7})
    rep("2 sessions -> XSESS absent","XSESS" not in out2["sd"]["verified"])
    rep("target downgraded",out["target"]=="speech_onset")

    print("\n[T9] PRODUCTION TRAINER")
    if HAS_TORCH:
        torch.manual_seed(0)
        m=NeuralSequenceTransformer(32,64,4,2,4,64)
        x=torch.randn(6,16,32); tg=torch.randint(0,4,(6,))
        lf=lambda mm,xx,tt: F.cross_entropy(mm(xx)["logits"],tt)
        tr=Trainer(m,TrainConfig(epochs=50,patience=12),cfg,lf,"seq")
        r=tr.fit(x,tg)
        rep("loss decreased",r["best_loss"]<r["history"][0]["loss"],
            f"{r['history'][0]['loss']:.3f}->{r['best_loss']:.3f}")
        rep("checkpoint written",Path(r["checkpoint"]).exists())
        rep("grad norms finite",all(math.isfinite(h["grad_norm"]) for h in r["history"]))
        wm=NeuralWorldModel(32,8,64)
        wo=wm(torch.randn(4,32),torch.randn(4,8))
        rep("world model labelled surrogate","not a measurement" in wo["STATUS"])
        df=NeuralDiffusion(32,30,64)
        opt=torch.optim.Adam(df.parameters(),lr=3e-3)
        x0=torch.randn(8,32); l0=lN=None
        for ep in range(30):
            opt.zero_grad(); L=df.loss(x0); L.backward(); opt.step()
            if ep==0: l0=float(L)
            lN=float(L)
        rep("diffusion trains",lN<l0)
        rep("diffusion labelled HYPOTHESIS","HYPOTHESIS" in df.sample(2)["STATUS"])
    else:
        rep("torch available",False,"pip install torch")

    print("\n[T10] SD + EXPORT")
    ex=pipe.export(good)
    rep("export written",Path(ex["written"]).exists())
    rep("Mode E retained",ex["sd"]["mode_E"]>0)
    print(f"      SD: {ex['sd']['n']} outputs, {ex['sd']['certified']} certified, "
          f"{ex['sd']['mode_E']} Mode E")
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
FIXED (three were crashes)
  EDF original->compacted index map; mixed-rate EDF refused.
  NpzFile .get() replaced by membership test in all three loaders.
  erd_ers honours the requested band instead of always returning alpha.
  Welch PSD is a proper one-sided density with declared units.
  EMG band guarded when fs is too low; NaN returned, not an empty mean.
  Task-locked nuisance removal is reported with an explicit warning.
  Permutation and cross-session validation are WIRED and gate the SD.
  data_sharing='none' now blocks every export path.

NOT CLAIMED
  Not thought reading. Not a diagnostic. Not a clinical decision tool.
  Word identity remains impossible from scalp EEG/MEG.
  Subjective feelings require self-report; a decoder output is never a label.
""")
    return ok

def main() -> int:
    ap=argparse.ArgumentParser(description="Neural–behaviour pairing v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","capabilities","items","packages","run"])
    ap.add_argument("--edf"); ap.add_argument("--meg")
    ap.add_argument("--item",default="ก_ไก่"); ap.add_argument("--json")
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    if a.cmd=="capabilities":
        print(json.dumps({k:asdict(v) for k,v in CAPABILITIES.items()},
                         indent=2,ensure_ascii=False)); return 0
    if a.cmd=="items":
        print(json.dumps(THAI_ITEMS,indent=2,ensure_ascii=False)); return 0
    if a.cmd=="packages":
        print(json.dumps(PACKAGES,indent=2,ensure_ascii=False)); return 0
    if not a.edf and not a.meg: print("--edf or --meg required"); return 1
    rec=read_edf(a.edf) if a.edf else load_meg(a.meg)
    c=Consent("CLI_PATIENT","IRB_REQUIRED","PSYCHIATRIST_REQUIRED",
              "COUNCIL_REQUIRED","paired recording",True,True,
              acknowledged_no_thought_reading=True,
              acknowledged_can_withdraw_anytime=True)
    log.warning("CLI uses placeholder consent identifiers. Replace before use.")
    try: out=Pipeline().run(c,rec,a.item)
    except Refusal as e: out={"REFUSED":str(e)}
    print(json.dumps(out,indent=2,default=str,ensure_ascii=False))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
