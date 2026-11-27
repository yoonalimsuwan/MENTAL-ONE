#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_neurochem_inference.py — v2.0.0  [PRODUCTION]
Multimodal neurochemical inference with explicit identifiability limits.

BUGS FIXED vs v1:
  1. np.maximum(coef, 0) is NOT non-negative least squares. Clipping an
     unconstrained solution does not minimise the constrained problem.
     Now a Lawson-Hanson active-set NNLS is used, with Tikhonov handled
     by row augmentation so the constraint and the penalty are solved
     jointly.
  2. CRLB used the caller's declared noise_sd and ignored the Tikhonov
     bias. Now the noise variance is ESTIMATED from the residual on the
     correct degrees of freedom, the CRLB is computed on the ACTIVE
     (passive) set only, and the regularisation bias is declared: a
     penalised estimator is biased, so CRLB is a lower bound, not an
     achieved standard error.
  3. Logan graphical analysis used a rectangle rule. Now trapezoidal
     integration, and the linearity criterion is ENFORCED, not merely
     reported.
  4. NeurochemWorldModel.chem_decay was an unconstrained Parameter and
     could exceed 1 during training (unstable latent). Now sigmoid-
     constrained to (0,1).
  5. subject_cv returned zero-filled rows for skipped folds as if they
     were predictions. Now an explicit coverage mask is returned.
  6. permutation_test reported an IN-SAMPLE correlation as the observed
     score. The p-value is valid (the null is computed identically) but
     the score is inflated; it is now labelled non-predictive.
  7. MRS basis linewidths were field-independent. Now optionally scaled
     with B0 and flagged.

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

THE INFORMATION-THEORETIC LIMIT (unchanged)
  Phi : C -> Y is many-to-one. Most chemical directions lie in ker(D Phi).
  I(C; estimate) <= I(C; Y). No model recovers what the signal never carried.
  Concentration comes from MRS or PET. EEG/MEG give a validated E/I PROXY
  and, with paired training and subject-wise CV, a COHORT association.

ENFORCED REFUSALS (unchanged, all still fire)
  neurotransmitter concentration from EEG/MEG; fMRI BOLD as concentration;
  GABA/GSH without an edited acquisition; CRLB > 20% reported as reliable;
  prediction without paired data and subject-wise CV; prediction whose
  ground truth is itself unreliable.
=============================================================================
Author: PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID: 0009-0008-2374-0788 | License: MIT | Year 2026
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY=True
except ImportError:
    HAS_NUMPY=False; np=None
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False

VERSION="2.0.0"; PI=math.pi
GAMMA_HZ_PER_T=42.577e6
RESEARCH_ONLY="research_only_not_a_diagnostic"
MODE_CERTIFIED="CERTIFIED"; MODE_E="UNCERTAINTY_RETAINED"

class Refusal(RuntimeError): pass
class NotIdentifiable(Refusal): pass
class NeedsReference(Refusal): pass
class UnreliableQuantification(Refusal): pass
class LinearityRefusal(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

def get_logger(n):
    import logging
    lg=logging.getLogger(n)
    if not lg.handlers:
        h=logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate=False
    return lg
log=get_logger("msps.neurochem")

@dataclass
class RunConfig:
    out_dir:str="runs/neurochem_v2"; seed:int=0; device:str="auto"; amp:bool=True
    def validate(self):
        Path(self.out_dir).mkdir(parents=True,exist_ok=True); return self

def set_seed(s):
    import random; random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    if HAS_NUMPY: np.random.seed(s)
    if HAS_TORCH: torch.manual_seed(s)
    return {"seed":s}

def pick_device(pref="auto"):
    if not HAS_TORCH: return "cpu"
    if pref!="auto": return pref
    return "cuda" if torch.cuda.is_available() else "cpu"

class Cache:
    def __init__(self,max_items=256):
        self._m={}; self._o=[]; self.max=max_items; self.hits=0; self.misses=0
    @staticmethod
    def key(*p):
        return hashlib.sha256(json.dumps(p,sort_keys=True,default=str).encode()).hexdigest()[:24]
    def get(self,k):
        if k in self._m:
            self.hits+=1; self._o.remove(k); self._o.append(k); return self._m[k]
        self.misses+=1; return None
    def put(self,k,v):
        self._m[k]=v; self._o.append(k)
        while len(self._o)>self.max: self._m.pop(self._o.pop(0),None)
    def stats(self):
        t=self.hits+self.misses
        return {"hits":self.hits,"misses":self.misses,"hit_rate":self.hits/max(t,1)}

BANNER=f"""
=============================================================================
MSPS NEUROCHEMICAL INFERENCE v{VERSION}  [PRODUCTION]
TRUE active-set NNLS (clip is not NNLS) · CRLB from ESTIMATED noise
CRLB on the ACTIVE set only; Tikhonov bias declared
Logan TRAPEZOID + linearity ENFORCED · chem_decay constrained to (0,1)
subject-CV coverage mask · in-sample r labelled non-predictive
REFUSED: concentration from EEG/MEG; unedited GABA/GSH; CRLB>20%
=============================================================================
"""

# =============================================================================
# §1. CAPABILITY TIERS (unchanged)
# =============================================================================

class Modality:
    EEG="scalp_eeg"; MEG="meg"; MRS="mrs"; PET="pet"
    FMRI="fmri"; MICRODIALYSIS="microdialysis"

@dataclass(frozen=True)
class Capability:
    modality:str; measures_concentration:bool
    can:Tuple[str,...]; cannot:Tuple[str,...]
    invasive:bool; requires:Tuple[str,...]; note:str

CAPABILITIES: Dict[str,Capability] = {
 Modality.EEG: Capability(Modality.EEG,False,
   ("aperiodic_exponent_EI_proxy","band_power","gamma_power_GABA_correlate",
    "MMN_NMDA_correlate","P50_suppression_correlate"),
   ("metabolite_concentration","receptor_occupancy","dopamine_concentration",
    "serotonin_concentration","GABA_mM"),False,
   ("electrodes","impedance_check","emg_eog_control"),
   "cm-scale, volume-conducted. Chemical state largely in ker(D Phi)."),
 Modality.MEG: Capability(Modality.MEG,False,
   ("aperiodic_exponent_EI_proxy","source_localised_band_power",
    "gamma_power_GABA_correlate"),
   ("metabolite_concentration","receptor_occupancy","GABA_mM"),False,
   ("shielded_room","squid_or_opm","emg_eog_control"),
   "Better source localisation than EEG; still no concentration."),
 Modality.MRS: Capability(Modality.MRS,True,
   ("GABA_concentration_edited","Glu_concentration","Gln","Glx","NAA","Cre",
    "Cho","mI","Lac","GSH","Tau"),
   ("receptor_occupancy","synaptic_release_rate","dopamine_D2_binding"),False,
   ("MR_scanner","localised_sequence","basis_set",
    "water_reference_or_internal_reference","CRLB_report"),
   "THE actual non-invasive chemical measurement. MEGA-PRESS for GABA/GSH. "
   "CRLB > 20% unreliable."),
 Modality.PET: Capability(Modality.PET,True,
   ("D2_D3_binding_potential","SERT_binding","dopamine_synthesis_capacity",
    "SV2A_synaptic_density","TSPO_neuroinflammation"),
   ("real_time_release_dynamics","EEG_derived_concentration"),False,
   ("radiotracer","cyclotron_or_generator","kinetic_model",
    "arterial_or_reference_input","radiation_consent"),
   "Measures a specific molecular TARGET. Requires a radiotracer."),
 Modality.FMRI: Capability(Modality.FMRI,False,
   ("BOLD_hemodynamics","functional_connectivity"),
   ("metabolite_concentration","neurotransmitter_level"),False,
   ("MR_scanner","hemodynamic_model"),"BOLD is hemodynamic, NOT chemical."),
 Modality.MICRODIALYSIS: Capability(Modality.MICRODIALYSIS,True,
   ("extracellular_concentration_timeseries",),("non_invasive_measurement",),
   True,("neurosurgery","clinical_indication","irb","consent"),
   "Invasive research. Actual extracellular concentration."),
}

def check_target(modality:str,target:str):
    if modality not in CAPABILITIES: raise Refusal(f"unknown '{modality}'")
    c=CAPABILITIES[modality]
    if target in c.cannot:
        raise NotIdentifiable(
            f"REFUSED: '{target}' NOT obtainable from {modality}.\n"
            f"Cannot: {list(c.cannot)}\nReason: {c.note}\n"
            "By the data-processing inequality no algorithm recovers "
            "information absent from the measurement. Use MRS for "
            "concentration or PET for receptor binding.")
    if target not in c.can:
        raise NotIdentifiable(f"'{target}' not validated for {modality}")
    return {"allowed":True,"modality":modality,"target":target,
            "measures_concentration":c.measures_concentration}

# =============================================================================
# §2. IDENTIFIABILITY (unchanged mathematics)
# =============================================================================

def identifiability_analysis(J,n_chemical,labels=None):
    _np()
    U,s,Vt=np.linalg.svd(J,full_matrices=True)
    tol=(s[0] if len(s) else 0.0)*max(J.shape)*np.finfo(float).eps
    rank=int(np.sum(s>max(tol,1e-12)))
    V=Vt.T; null_dim=n_chemical-rank
    ident=np.array([float(np.sum(V[i,:rank]**2)) for i in range(n_chemical)])
    names=list(labels) if labels else [f"c{i}" for i in range(n_chemical)]
    return {"n_measurements":J.shape[0],"n_chemical":n_chemical,"rank":rank,
            "null_dimension":null_dim,"singular_values":s.tolist(),
            "condition_number":float(s[0]/s[rank-1]) if rank else float("inf"),
            "per_variable_identifiability":dict(zip(names,ident.tolist())),
            "poorly_identified":[names[i] for i in range(n_chemical) if ident[i]<0.5],
            "null_space_basis":(V[:,rank:] if null_dim>0 else np.zeros((n_chemical,0))).tolist(),
            "VERDICT":"chemical state NOT identifiable; only the row-space "
                      "projection is" if null_dim>0 else "locally identifiable",
            "principle":"I(C; estimate) <= I(C; Y)"}

def build_eeg_chemical_jacobian(n_eeg=12,n_chem=10):
    _np(); rng=np.random.default_rng(0)
    J=np.zeros((n_eeg,n_chem))
    J[0,0]=1.0; J[0,1]=-0.8
    J[1]=rng.normal(0,0.15,n_chem)
    J[2:]=rng.normal(0,0.01,(n_eeg-2,n_chem))
    return J

# =============================================================================
# §3. TRUE NNLS  (Lawson-Hanson active set)
# =============================================================================

def nnls(A,b,lam:float=0.0,max_iter:int=None,tol:float=1e-10):
    """
    Non-negative least squares with optional Tikhonov by ROW AUGMENTATION:
        min || [A; sqrt(lam) I] x - [b; 0] ||^2   s.t. x >= 0
    Solving the augmented problem inside the active-set loop means the
    penalty and the constraint are handled JOINTLY. v1 solved the
    penalised unconstrained system and then clipped, which is not the
    constrained minimiser.
    """
    _np()
    n=A.shape[1]
    if lam>0:
        Aa=np.vstack([A,math.sqrt(lam)*np.eye(n)])
        ba=np.concatenate([b,np.zeros(n)])
    else:
        Aa,ba=A,b
    x=np.zeros(n); P=np.zeros(n,bool)
    w=Aa.T@(ba-Aa@x)
    it=0; max_iter=max_iter or 6*n+50
    while (not P.all()) and (w[~P].max(initial=-np.inf)>tol) and it<max_iter:
        cand=np.where(~P)[0]
        j=cand[int(np.argmax(w[cand]))]
        P[j]=True
        while True:
            xp,*_=np.linalg.lstsq(Aa[:,P],ba,rcond=None)
            xf=np.zeros(n); xf[P]=xp
            if (xf[P]>tol).all():
                x=xf; break
            idx=np.where(P&(xf<=tol))[0]
            if len(idx)==0: x=xf; break
            denom=x[idx]-xf[idx]
            ratios=np.divide(x[idx],denom,out=np.full_like(denom,np.inf),
                             where=np.abs(denom)>1e-30)
            alpha=float(np.min(ratios)) if np.isfinite(ratios).any() else 0.0
            x=x+alpha*(xf-x)
            P[np.where(P&(x<=tol))[0]]=False
            it+=1
            if it>max_iter: break
            if not P.any(): break
        w=Aa.T@(ba-Aa@x); it+=1
    return x,P,it

# =============================================================================
# §4. MRS QUANTIFICATION
# =============================================================================

@dataclass(frozen=True)
class Metabolite:
    name:str; shifts_ppm:Tuple[float,...]; amplitudes:Tuple[float,...]
    linewidth_hz:float; edited:bool=False; note:str=""

METABOLITES: Dict[str,Metabolite] = {
 "NAA":Metabolite("NAA",(2.008,2.51,2.67),(1.0,.12,.10),6.0,
        note="neuronal integrity marker; largest singlet"),
 "Cre":Metabolite("Cre",(3.027,3.913),(1.0,.5),6.0,
        note="creatine + phosphocreatine; common internal reference"),
 "Cho":Metabolite("Cho",(3.19,3.50,4.05),(1.0,.15,.10),6.0,note="membrane turnover"),
 "mI":Metabolite("mI",(3.56,3.64,4.06),(1.0,.8,.4),8.0,note="glial marker"),
 "Glu":Metabolite("Glu",(2.35,2.22,2.13,3.75),(1.0,.7,.6,.5),9.0,
        note="J-coupled; overlaps Gln"),
 "Gln":Metabolite("Gln",(2.45,2.12,3.75),(1.0,.6,.5),9.0,note="Glx = Glu + Gln"),
 "GABA":Metabolite("GABA",(1.89,2.28,3.01),(1.0,1.0,1.0),8.0,edited=True,
        note="REQUIRES edited acquisition (MEGA-PRESS)"),
 "GSH":Metabolite("GSH",(2.95,3.77),(1.0,.4),8.0,edited=True,
        note="glutathione; requires editing"),
 "Lac":Metabolite("Lac",(1.33,4.10),(1.0,.15),6.0,note="doublet J=7 Hz"),
 "Tau":Metabolite("Tau",(3.25,3.42),(1.0,1.0),8.0,note="taurine"),
}
WATER_PPM=4.65

def ppm_to_hz(ppm,field_T): return (WATER_PPM-ppm)*GAMMA_HZ_PER_T*field_T/1e6
def lorentzian(f,f0,lw):
    h=lw/2.0; return h*h/((f-f0)**2+h*h)

def build_basis_set(field_T,sw_hz,n_pts,names=None,scale_lw_with_field=False):
    """
    NEW: linewidths may be scaled with B0 (in Hz they broaden roughly
    linearly with field for susceptibility-dominated linewidth). The
    default keeps the v1 fixed values for reproducibility; scaling is
    opt-in and flagged.
    """
    _np()
    f=np.linspace(-sw_hz/2,sw_hz/2,n_pts)
    use=list(names) if names else list(METABOLITES)
    B=[]
    for nm in use:
        m=METABOLITES[nm]
        lw=m.linewidth_hz*(field_T/3.0) if scale_lw_with_field else m.linewidth_hz
        s=np.zeros(n_pts)
        for ppm,amp in zip(m.shifts_ppm,m.amplitudes):
            s+=amp*lorentzian(f,ppm_to_hz(ppm,field_T),lw)
        B.append(s/max(s.max(),1e-12))
    return np.array(B).T,use,f

def synthesize_mrs(conc,field_T=3.0,sw_hz=2000.0,n_pts=1024,noise_sd=0.01,
                   baseline_terms=3,seed=0,scale_lw_with_field=False):
    _np(); rng=np.random.default_rng(seed)
    B,names,f=build_basis_set(field_T,sw_hz,n_pts,
                              scale_lw_with_field=scale_lw_with_field)
    c=np.array([conc.get(nm,0.0) for nm in names]); y=B@c
    X=np.column_stack([np.linspace(-1,1,n_pts)**k for k in range(baseline_terms)])
    y=y+X@rng.normal(0,0.05,baseline_terms)+rng.normal(0,noise_sd,n_pts)
    return {"spectrum":y,"freq_hz":f,"basis":B,"names":names,
            "true_concentrations":dict(zip(names,c.tolist())),
            "baseline_design":X,"noise_sd":noise_sd,"field_T":field_T}

def fit_mrs(y,B,names,baseline_design=None,lam=1e-4,crlb_max_pct=20.0,
            edited_acquired=False,cache:Optional[Cache]=None):
    """
    Linear-combination MRS quantification.
    FIX 1: true active-set NNLS with the Tikhonov penalty augmented.
    FIX 2: noise variance ESTIMATED from the residual; CRLB computed on
           the ACTIVE set; regularisation bias declared.
    """
    _np()
    n=len(y); K=B.shape[1]
    if baseline_design is None:
        baseline_design=np.column_stack([np.linspace(-1,1,n)**k for k in range(3)])
    P=baseline_design
    Q,_=np.linalg.qr(P)
    Bp=B-Q@(Q.T@B); yp=y-Q@(Q.T@y)
    coef,passive,iters=nnls(Bp,yp,lam=lam)
    resid=yp-Bp@coef
    n_bound=int((~passive).sum())
    dof=max(n-int(passive.sum())-P.shape[1],1)
    s2=float(resid@resid)/dof                    # ESTIMATED noise variance
    # CRLB on the ACTIVE set only (bound parameters have no interior CRLB)
    crlb=np.full(K,np.inf)
    if passive.any():
        Bact=Bp[:,passive]
        F=(Bact.T@Bact)/max(s2,1e-30)
        C=np.linalg.pinv(F)
        vals=coef[passive]
        se=np.sqrt(np.maximum(np.diag(C),0.0))
        crlb[passive]=np.where(vals>1e-9,100.0*se/np.maximum(vals,1e-30),np.inf)
    unreliable=[nm for nm,c in zip(names,crlb) if c>crlb_max_pct]
    edited_missing=[nm for nm in names
                    if METABOLITES[nm].edited and not edited_acquired]
    reportable=[nm for nm in names
                if nm not in unreliable and nm not in edited_missing]
    out={"names":list(names),
         "concentrations":dict(zip(names,coef.tolist())),
         "crlb_percent":dict(zip(names,[float(c) for c in crlb])),
         "active_set":[nm for nm,p in zip(names,passive) if p],
         "at_boundary":[nm for nm,p in zip(names,passive) if not p],
         "residual_sd":math.sqrt(s2),"noise_variance_estimated":s2,
         "dof":dof,"nnls_iterations":iters,
         "method":"active-set NNLS + baseline projection + augmented Tikhonov",
         "CRLB_CAVEAT":("CRLB is computed on the active set from the ESTIMATED "
                        "noise variance. The Tikhonov-penalised estimator is "
                        "BIASED, so CRLB is a LOWER BOUND on achievable "
                        "standard error, not an achieved error bar. Parameters "
                        "at the non-negativity boundary have no interior CRLB."),
         "unreliable_CRLB":unreliable,"crlb_threshold_pct":crlb_max_pct,
         "edited_metabolites_without_editing":edited_missing,
         "reportable":reportable,
         "units":"arbitrary (ratio to internal reference unless calibrated)",
         "NOT_A_CONCENTRATION_IN_MM":(
            "values are ratios/arbitrary units unless a water or internal "
            "reference with known concentration and relaxation correction is used")}
    if edited_missing:
        out["REFUSED"]=(f"{edited_missing} require an EDITED acquisition "
                        "(MEGA-PRESS). Unedited quantification is not reportable.")
    return out

def require_reportable(fit,metabolite):
    if metabolite in fit.get("edited_metabolites_without_editing",[]):
        raise UnreliableQuantification(
            f"{metabolite} requires an edited acquisition (MEGA-PRESS).")
    if metabolite in fit.get("unreliable_CRLB",[]):
        raise UnreliableQuantification(
            f"{metabolite} CRLB={fit['crlb_percent'][metabolite]:.1f}% "
            f"> {fit['crlb_threshold_pct']}%. Not reportable.")
    return {"metabolite":metabolite,"value":fit["concentrations"][metabolite],
            "crlb_pct":fit["crlb_percent"][metabolite],"reportable":True,
            "CRLB_is_lower_bound":True}

# =============================================================================
# §5. PET — trapezoid Logan, linearity ENFORCED
# =============================================================================

PET_TRACERS = {
 "11C-raclopride":{"target":"D2/D3 receptor","outcome":"BP_ND",
   "half_life_min":20.4,"reference":"cerebellum",
   "note":"displaceable by endogenous dopamine"},
 "11C-DASB":{"target":"serotonin transporter (SERT)","outcome":"BP_ND",
   "half_life_min":20.4,"reference":"cerebellum"},
 "18F-FDOPA":{"target":"dopamine synthesis capacity","outcome":"Ki_cer",
   "half_life_min":109.8,"reference":"cerebellum"},
 "11C-UCB-J":{"target":"SV2A synaptic vesicle protein","outcome":"BP_ND",
   "half_life_min":20.4,"reference":"centrum semiovale"},
 "18F-MK-6240":{"target":"tau neurofibrillary tangles","outcome":"SUVR",
   "half_life_min":109.8,"reference":"cerebellar crus"},
 "11C-PBR28":{"target":"TSPO (neuroinflammation)","outcome":"VT",
   "half_life_min":20.4,"note":"requires TSPO genotype consideration"},
}

def _trapz_cumulative(y,t):
    _np()
    seg=(y[1:]+y[:-1])*0.5*np.diff(t)
    return np.concatenate([[0.0],np.cumsum(seg)])

def pet_quantify(tracer,tissue_tac,reference_tac,times_min,t_star=20.0,
                 min_r2=0.98,enforce_linearity=True):
    """
    Reference-tissue Logan. FIX: trapezoidal integration (v1 used a
    rectangle rule) and the linearity criterion is ENFORCED.
    """
    _np()
    if tracer not in PET_TRACERS: raise Refusal(f"unknown tracer '{tracer}'")
    t=np.asarray(times_min,float)
    ir=_trapz_cumulative(np.asarray(reference_tac,float),t)
    it=_trapz_cumulative(np.asarray(tissue_tac,float),t)
    ok=(t>=t_star)&(np.asarray(reference_tac)>1e-9)
    if ok.sum()<5: raise Refusal("insufficient post-equilibrium samples")
    x=ir[ok]/np.asarray(reference_tac)[ok]
    yy=it[ok]/np.asarray(reference_tac)[ok]
    A=np.column_stack([x,np.ones_like(x)])
    coef,*_=np.linalg.lstsq(A,yy,rcond=None)
    slope=float(coef[0]); res=yy-A@coef
    r2=1.0-float(res@res)/max(float(np.var(yy)*len(yy)),1e-12)
    if enforce_linearity and r2<min_r2:
        raise LinearityRefusal(
            f"Logan plot not linear: r2={r2:.4f} < {min_r2}. "
            "Equilibrium not reached or reference region invalid. "
            "BP_ND must not be reported.")
    return {"tracer":tracer,"target":PET_TRACERS[tracer]["target"],
            "outcome":PET_TRACERS[tracer]["outcome"],
            "logan_slope":slope,"BP_ND":slope-1.0,"fit_r2":r2,
            "t_star_min":t_star,"integration":"trapezoidal",
            "linearity_ok":r2>=min_r2,"linearity_enforced":enforce_linearity,
            "requires":["radiotracer","radiation_consent","kinetic_model"],
            "NOTE":"simplified Logan; production needs full kinetic modelling"}

# =============================================================================
# §6. EEG/MEG FEATURES
# =============================================================================

BANDS={"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),
       "low_gamma":(30,55),"high_gamma":(65,90)}

def welch_psd(x,fs,nperseg=512):
    _np()
    n=x.shape[-1]; seg=min(nperseg,n)
    if seg<8: raise Refusal("epoch too short")
    win=np.hanning(seg); U=float((win**2).sum()); nseg=max(n//seg,1)
    acc=np.zeros(seg//2+1)
    for i in range(nseg):
        s=x[...,i*seg:(i+1)*seg]
        if s.shape[-1]<seg: break
        X=np.fft.rfft(s*win,axis=-1)
        acc=acc+(np.abs(X)**2).mean(axis=tuple(range(s.ndim-1)))
    acc/=max(nseg,1)*fs*U; acc[1:-1]*=2.0
    return np.fft.rfftfreq(seg,1.0/fs),acc

class EEGFeatures:
    def __init__(self,fs): self.fs=fs
    def psd(self,x,nperseg=512): return welch_psd(x,self.fs,nperseg)
    def aperiodic_fit(self,f,psd,fit_range=(2.0,40.0),
                      exclude_bands=((7,14),),knee=False):
        """
        log P = offset - chi*log f  (optionally with a Lorentzian knee).
        The knee option is OFF by default and flagged: a knee changes the
        interpretation of chi and must be justified by the spectrum.
        """
        _np()
        m=(f>=fit_range[0])&(f<=fit_range[1])&(psd>0)
        for lo,hi in exclude_bands: m&=~((f>=lo)&(f<=hi))
        if m.sum()<6: raise Refusal("too few points for aperiodic fit")
        lf=np.log10(f[m]); lp=np.log10(psd[m])
        if not knee:
            A=np.column_stack([np.ones_like(lf),-lf])
            coef,*_=np.linalg.lstsq(A,lp,rcond=None)
            off,chi=float(coef[0]),float(coef[1]); pred=A@coef; kn=None
        else:
            best=None
            for kf in np.linspace(fit_range[0],fit_range[1],40):
                for chi_t in np.linspace(0.5,4.0,36):
                    p=off0=np.mean(lp+chi_t*np.log10(kf+f[m]))
                    pr=p-chi_t*np.log10(kf+f[m])
                    e=float(np.sum((lp-pr)**2))
                    if best is None or e<best[0]: best=(e,p,chi_t,kf)
            _,off,chi,kn=best
            pred=off-chi*np.log10(kn+f[m])
        res=lp-pred
        r2=1.0-float(res@res)/max(float(np.var(lp)*len(lp)),1e-12)
        return {"offset":off,"exponent_chi":chi,"knee_hz":kn,"fit_r2":r2,
                "n_points":int(m.sum()),"fit_range_hz":list(fit_range),
                "excluded_bands":[list(b) for b in exclude_bands],
                "knee_model":knee,
                "interpretation":"steeper (larger chi) -> relatively more "
                                 "inhibition; PROXY not concentration",
                "reference":"Gao, Peterson & Voytek, NeuroImage 2017",
                "fit_quality_ok":r2>0.90,
                "KNEE_CAVEAT":"with a knee, chi is the high-frequency slope "
                              "only; do not compare to knee-free fits"
                              if knee else None}
    def band_power(self,f,psd):
        out={}
        for b,(lo,hi) in BANDS.items():
            he=min(hi,self.fs/2-1e-9)
            if he<=lo: continue
            m=(f>=lo)&(f<he)
            out[b]=float(psd[m].mean()*(he-lo)) if m.any() else 0.0
        return out
    def ei_proxy(self,x,knee=False):
        f,p=self.psd(x); ap=self.aperiodic_fit(f,p,knee=knee)
        return {"aperiodic":ap,"band_power":self.band_power(f,p),
                "EI_proxy_exponent":ap["exponent_chi"],
                "STATUS":"PROXY — not a GABA or glutamate concentration",
                "mechanistic_basis":"AMPA tau ~2 ms vs GABA-A tau ~10 ms "
                                    "shapes the PSD slope",
                "NOT":"a molecule-specific measurement",
                "validation_required":["paired MRS","cross-validation",
                                       "EMG/EOG control"]}

# =============================================================================
# §7. PAIRED REGRESSION — coverage mask, in-sample flag
# =============================================================================

@dataclass
class PairedDataset:
    X:Any; Y:Any; target_names:List[str]; feature_names:List[str]
    subject_ids:List[str]; session_ids:List[str]
    reference_modality:str=Modality.MRS
    crlb_percent:Optional[Any]=None

class PairedRegression:
    def __init__(self,lam=1.0): self.lam=lam
    def fit(self,X,Y):
        _np()
        Xm=X.mean(0); Xs=X.std(0)+1e-9; Xz=(X-Xm)/Xs
        W=np.linalg.solve(Xz.T@Xz+self.lam*np.eye(Xz.shape[1]),
                          Xz.T@(Y-Y.mean(0)))
        return {"W":W,"X_mean":Xm,"X_std":Xs,"Y_mean":Y.mean(0),"lam":self.lam}
    def predict(self,m,X):
        return (X-m["X_mean"])/m["X_std"]@m["W"]+m["Y_mean"]
    def subject_cv(self,ds,n_folds=5,min_subjects=30):
        _np()
        subs=sorted(set(ds.subject_ids))
        if len(subs)<min_subjects:
            return {"valid":False,"reason":f"need >= {min_subjects} subjects, "
                    f"got {len(subs)}","status":MODE_E}
        rng=np.random.default_rng(0); perm=rng.permutation(subs)
        folds=[set(perm[i::n_folds]) for i in range(n_folds)]
        rows=[]; covered=np.zeros(len(ds.subject_ids),bool)
        pred=np.full_like(ds.Y,np.nan); truth=np.full_like(ds.Y,np.nan)
        for k in range(n_folds):
            te=np.array([s in folds[k] for s in ds.subject_ids]); tr=~te
            if tr.sum()<5 or te.sum()<2: continue
            m=self.fit(ds.X[tr],ds.Y[tr]); p=self.predict(m,ds.X[te])
            pred[te]=p; truth[te]=ds.Y[te]; covered[te]=True
            for j in range(ds.Y.shape[1]):
                yt,yp=ds.Y[te,j],p[:,j]
                if np.var(yt)>1e-12:
                    rows.append((ds.target_names[j],k,float(np.corrcoef(yt,yp)[0,1])))
        summary={}
        for nm in ds.target_names:
            rs=[r for (t,k,r) in rows if t==nm]
            if rs:
                summary[nm]={"mean_r":float(np.mean(rs)),"sd_r":float(np.std(rs)),
                             "min_r":float(np.min(rs)),"n_folds":len(rs),
                             "predictive":float(np.mean(rs))>0.3 and float(np.min(rs))>0.0}
        return {"valid":True,"n_subjects":len(subs),"n_folds":n_folds,
                "per_target":summary,"split":"SUBJECT-WISE",
                "coverage_mask":covered.tolist(),
                "coverage_fraction":float(covered.mean()),
                "uncovered_rows":int((~covered).sum()),
                "predictions":pred,"truth":truth,
                "NOTE":"a positive subject-wise r means the features carry "
                       "information about the MEASURED quantity in this "
                       "cohort; it does NOT license an individual "
                       "concentration claim"}
    def permutation_test(self,ds,n_perm=500,seed=0):
        _np(); rng=np.random.default_rng(seed)
        m=self.fit(ds.X,ds.Y); p=self.predict(m,ds.X)
        def score(Y,P):
            return float(np.mean([np.corrcoef(Y[:,j],P[:,j])[0,1]
                                  for j in range(Y.shape[1]) if np.var(Y[:,j])>1e-12]))
        obs=score(ds.Y,p); null=np.empty(n_perm)
        for i in range(n_perm):
            Yp=rng.permutation(ds.Y,axis=0)
            null[i]=score(Yp,self.predict(self.fit(ds.X,Yp),ds.X))
        pv=(1.0+float(np.sum(null>=obs)))/(n_perm+1.0)
        return {"observed_in_sample_r":obs,"null_mean":float(null.mean()),
                "p_value":pv,"n_perm":n_perm,"above_chance":pv<0.01,
                "OBSERVED_IS_IN_SAMPLE":True,
                "CAVEAT":"the observed r is IN-SAMPLE and inflated by fitting. "
                         "The p-value is valid because the null is computed "
                         "identically, but this r must NOT be quoted as "
                         "predictive. Use subject_cv for that."}
    def gate_prediction(self,ds,cv,target):
        if not cv.get("valid"):
            raise NeedsReference(f"prediction not certified: {cv.get('reason')}")
        s=cv["per_target"].get(target)
        if s is None or not s["predictive"]:
            raise NeedsReference(f"'{target}' not predicted above chance in "
                                 "subject-wise CV. No prediction may be reported.")
        if ds.crlb_percent is not None:
            j=ds.target_names.index(target)
            bad=float(np.mean(ds.crlb_percent[:,j]>20.0))
            if bad>0.3:
                raise UnreliableQuantification(
                    f"{bad*100:.0f}% of reference measurements for '{target}' "
                    "have CRLB > 20%. The ground truth is unreliable.")
        return {"target":target,"certified":True,"mean_r":s["mean_r"],
                "min_r":s["min_r"],"reference_modality":ds.reference_modality,
                "SCOPE":"cohort-level statistical association, not an "
                        "individual concentration measurement"}

# =============================================================================
# §8. MODELS — chem_decay constrained
# =============================================================================

if HAS_TORCH:
    class MultimodalTransformer(nn.Module):
        def __init__(self,d_eeg=24,d_mrs=10,d=128,heads=4,layers=3,
                     n_targets=4,max_len=64):
            super().__init__()
            self.eeg=nn.Linear(d_eeg,d); self.mrs=nn.Linear(d_mrs,d)
            self.type=nn.Parameter(torch.randn(2,d)*0.02)
            self.pos=nn.Parameter(torch.randn(1,max_len,d)*0.02)
            enc=nn.TransformerEncoderLayer(d,heads,d*4,0.1,
                                           batch_first=True,norm_first=True)
            self.trunk=nn.TransformerEncoder(enc,layers)
            self.head=nn.Linear(d,n_targets); self.log_sd=nn.Linear(d,n_targets)
            self.nuisance=nn.Linear(d,1)
        def forward(self,eeg,mrs=None):
            h=self.eeg(eeg)+self.type[0]
            if mrs is not None:
                h=torch.cat([h,self.mrs(mrs)+self.type[1]],1)
            p=self.trunk(h+self.pos[:,:h.shape[1]]).mean(1)
            return {"pred":self.head(p),"log_sd":self.log_sd(p).clamp(-6,3),
                    "nuisance":self.nuisance(p),
                    "STATUS":"research surrogate, not a measurement"}
    class NeurochemWorldModel(nn.Module):
        """FIX: chem_decay sigmoid-constrained to (0,1) so the slow latent
        cannot become unstable during training."""
        def __init__(self,d_obs=24,d_chem=4,d_act=4,d=128):
            super().__init__()
            self.enc=nn.Linear(d_obs+d_chem+d_act,d); self.gru=nn.GRUCell(d,d)
            self.obs=nn.Linear(d,d_obs); self.chem=nn.Linear(d,d_chem)
            self.log_sd=nn.Linear(d,d_chem)
            self.decay_logit=nn.Parameter(torch.tensor(3.9))  # sigmoid ~ 0.98
        @property
        def decay(self): return torch.sigmoid(self.decay_logit)
        def forward(self,obs,chem,act):
            h=F.gelu(self.enc(torch.cat([obs,chem,act],-1))); h=self.gru(h,h)
            return {"obs_pred":self.obs(h),
                    "chem_next":self.decay*chem+0.02*self.chem(h),
                    "decay":float(self.decay),
                    "log_sd":self.log_sd(h).clamp(-6,3),
                    "STATUS":"latent chemical variable is model-internal; "
                             "NOT a measured concentration"}
    class SpectrumDiffusion(nn.Module):
        def __init__(self,d=256,T=100,hidden=256):
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
            return {"candidate_spectrum":x,"STATUS":"HYPOTHESIS — never a measurement"}

# =============================================================================
# §9. SD GATE + PIPELINE
# =============================================================================

class SDGate:
    ENTRIES={
      "mrs_concentration":(("MRS","CRLB","REF"),"active-set NNLS linear combination",
                           "CRLB<=20%; edited sequence for GABA/GSH; "
                           "CRLB is a lower bound (penalised estimator)"),
      "pet_binding":(("TRACER","KIN","CONSENT","LIN"),"reference-tissue Logan",
                     "trapezoid integration; linearity enforced"),
      "ei_proxy":(("APER","ART"),"Gao-Peterson-Voytek 2017",
                  "PROXY only; not molecule-specific"),
      "eeg_to_chemistry":(("PAIRED","SUBJCV","PERM","CRLB"),"paired regression",
                          "cohort association, not individual concentration"),
      "concentration_from_eeg":((),"none","NOT IDENTIFIABLE — refused"),
      "fmri_concentration":((),"none","BOLD is hemodynamic — refused"),
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

PACKAGES={"MRS":"MRS acquisition present","CRLB":"CRLB<=20% reported",
 "REF":"water/internal reference with relaxation correction",
 "TRACER":"radiotracer administered","KIN":"kinetic model fitted",
 "CONSENT":"radiation/invasive consent","LIN":"Logan linearity enforced",
 "APER":"aperiodic fit r2>0.9","ART":"EMG/EOG controlled",
 "PAIRED":"paired EEG+MRS/PET data","SUBJCV":"subject-wise CV >=30 subjects",
 "PERM":"permutation p<0.01"}

class Pipeline:
    def __init__(self,cfg:Optional[RunConfig]=None):
        self.cfg=(cfg or RunConfig()).validate(); set_seed(self.cfg.seed)
        self.sd=SDGate(); self.cache=Cache()
    def refuse_concentration_from_eeg(self,request):
        low=request.lower()
        for b in ("gaba concentration from eeg","dopamine level from eeg",
                  "serotonin from meg","ความเข้มข้นจาก eeg",
                  "พยากรณ์โมเลกุลอย่างละเอียด","molecular prediction from eeg",
                  "neurotransmitter concentration from meg"):
            if b in low:
                raise NotIdentifiable(
                    f"REFUSED: '{b}'. The forward map is many-to-one; most "
                    "chemical directions lie in its null space. Use MRS for "
                    "concentration, PET for binding, or a paired regression "
                    "validated subject-wise.")
        return {"allowed":True}
    def analyse_eeg(self,x,fs,knee=False):
        self.refuse_concentration_from_eeg("eeg analysis")
        ei=EEGFeatures(fs).ei_proxy(x,knee=knee)
        cert=self.sd.certify("ei_proxy",("APER",) if ei["aperiodic"]["fit_quality_ok"]
                             else (),{"chi":ei["EI_proxy_exponent"]})
        self.sd.certify("concentration_from_eeg",(),{"refused":True})
        return {"ei":ei,"sd":cert,
                "CONCLUSION":"aperiodic exponent is an E/I PROXY. "
                             "No concentration is reported.",
                "actionability":RESEARCH_ONLY}
    def analyse_mrs(self,y,B,names,edited=False,cache=True):
        fit=fit_mrs(y,B,names,edited_acquired=edited)
        good=fit["reportable"]
        cert=self.sd.certify("mrs_concentration",("MRS","CRLB") if good else (),
                             {"reportable":good})
        return {"fit":fit,"sd":cert,"NOTE":fit["NOT_A_CONCENTRATION_IN_MM"]}
    def report(self):
        rep={"version":VERSION,
             "capabilities":{k:asdict(v) for k,v in CAPABILITIES.items()},
             "metabolites":{k:asdict(v) for k,v in METABOLITES.items()},
             "pet_tracers":PET_TRACERS,"packages":PACKAGES,
             "cache":self.cache.stats(),"sd":self.sd.report(),
             "actionability":RESEARCH_ONLY}
        p=Path(self.cfg.out_dir)/"report.json"
        p.write_text(json.dumps(rep,indent=2,default=str))
        return rep

# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print(BANNER)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    cfg=RunConfig().validate(); pipe=Pipeline(cfg); rng=np.random.default_rng(0)

    print("\n[T1] TRUE NNLS vs CLIP (v1 was not NNLS)")
    A=rng.normal(0,1,(40,6)); xt=np.array([2.0,0.0,1.5,0.0,0.5,0.0])
    b=A@xt+rng.normal(0,0.02,40)
    x,P,it=nnls(A,b,lam=1e-6)
    clip=np.maximum(np.linalg.solve(A.T@A+1e-6*np.eye(6),A.T@b),0.0)
    rn=float(np.linalg.norm(A@x-b)); rc=float(np.linalg.norm(A@clip-b))
    print(f"      NNLS residual={rn:.4f}  clip residual={rc:.4f}")
    rep("NNLS non-negative",bool((x>=-1e-12).all()))
    rep("NNLS residual <= clip residual",rn<=rc+1e-9,f"{rn:.4f} <= {rc:.4f}")
    rep("active set identified",P.sum()>=3)

    print("\n[T2] CAPABILITY TIERS + IDENTIFIABILITY")
    for m in (Modality.EEG,Modality.MEG):
        b=False
        try: check_target(m,"GABA_mM")
        except NotIdentifiable: b=True
        rep(f"{m} REFUSES GABA_mM",b)
    b=False
    try: check_target(Modality.FMRI,"metabolite_concentration")
    except NotIdentifiable: b=True
    rep("fMRI REFUSES concentration",b)
    names=["Glu","GABA","DA","5HT","ACh","NE","GSH","Gln","Tau","mI"]
    ia=identifiability_analysis(build_eeg_chemical_jacobian(12,10),10,names)
    print(f"      rank={ia['rank']} null_dim={ia['null_dimension']}")
    rep("null space non-trivial",ia["null_dimension"]>0)
    rep("most molecules poorly identified",len(ia["poorly_identified"])>=6)

    print("\n[T3] MRS QUANTIFICATION with estimated-noise CRLB")
    truth={"NAA":10.,"Cre":8.,"Cho":2.,"mI":5.,"Glu":9.,"Gln":3.,"GABA":1.5,"Lac":.5}
    syn=synthesize_mrs(truth,3.0,noise_sd=0.02,seed=1)
    fit=fit_mrs(syn["spectrum"],syn["basis"],syn["names"],
                baseline_design=syn["baseline_design"],edited_acquired=True)
    for nm in ("NAA","Cre","Glu"):
        t=truth[nm]; e=fit["concentrations"][nm]
        print(f"      {nm}: true={t:.2f} est={e:.2f} CRLB={fit['crlb_percent'][nm]:.1f}%")
        rep(f"{nm} within 15%",abs(e-t)/t<0.15,f"err={abs(e-t)/t*100:.1f}%")
    rep("noise variance ESTIMATED",abs(fit["noise_variance_estimated"]-0.02**2)<0.02**2)
    rep("CRLB caveat present","LOWER BOUND" in fit["CRLB_CAVEAT"])
    rep("active set reported",len(fit["active_set"])>0)
    noisy=synthesize_mrs(truth,noise_sd=1.0,seed=2)
    fn=fit_mrs(noisy["spectrum"],noisy["basis"],noisy["names"],edited_acquired=True)
    rep("high noise -> unreliable list",len(fn["unreliable_CRLB"])>0)

    print("\n[T4] EDITED-ACQUISITION GATE")
    fu=fit_mrs(syn["spectrum"],syn["basis"],syn["names"],edited_acquired=False)
    rep("GABA flagged without editing","GABA" in fu["edited_metabolites_without_editing"])
    b=False
    try: require_reportable(fu,"GABA")
    except UnreliableQuantification: b=True
    rep("GABA REFUSED without editing",b)
    rep("GABA reportable WITH editing",require_reportable(fit,"GABA")["reportable"])

    print("\n[T5] PET trapezoid + linearity ENFORCED")
    t=np.arange(0,90,0.5); ref=20*np.exp(-t/25)+2
    tis=2.4*ref*(1-np.exp(-t/8))
    pet=pet_quantify("11C-raclopride",tis,ref,t,t_star=25.0)
    print(f"      slope={pet['logan_slope']:.3f} BP_ND={pet['BP_ND']:.3f} "
          f"r2={pet['fit_r2']:.4f}")
    rep("integration is trapezoidal",pet["integration"]=="trapezoidal")
    rep("BP_ND positive",pet["BP_ND"]>0)
    b=False
    try: pet_quantify("11C-raclopride",rng.normal(10,5,len(t)),ref,t,t_star=25.0)
    except LinearityRefusal: b=True
    rep("non-linear Logan REFUSED",b)

    print("\n[T6] EEG APERIODIC PROXY")
    fs=256.; n=4096; tt=np.arange(n)/fs
    def make(chi,seed):
        r=np.random.default_rng(seed); f=np.fft.rfftfreq(n,1/fs)
        amp=np.zeros_like(f); amp[1:]=f[1:]**(-chi/2)
        x=np.fft.irfft(amp*np.exp(1j*r.uniform(0,2*PI,len(f))),n=n)
        return x/max(np.std(x),1e-9)+1.5*np.sin(2*PI*10*tt)
    xi=make(2.4,1); xe=make(1.2,2); ft=EEGFeatures(fs)
    ei=ft.ei_proxy(xi); ee=ft.ei_proxy(xe)
    print(f"      steep chi={ei['EI_proxy_exponent']:.3f} "
          f"shallow chi={ee['EI_proxy_exponent']:.3f}")
    rep("steeper spectrum -> larger chi",
        ei["EI_proxy_exponent"]>ee["EI_proxy_exponent"])
    rep("labelled PROXY","PROXY" in ei["STATUS"])
    ek=ft.ei_proxy(xi,knee=True)
    rep("knee caveat present",ek["aperiodic"]["KNEE_CAVEAT"] is not None)

    print("\n[T7] PAIRED REGRESSION + coverage + in-sample flag")
    ns=60; nf=12
    X=rng.normal(0,1,(ns,nf)); W=rng.normal(0,1,(nf,2))
    Y=X@W+rng.normal(0,0.5,(ns,2))
    ds=PairedDataset(X,Y,["GABA_MRS","Glu_MRS"],[f"f{i}" for i in range(nf)],
                     [f"s{i:03d}" for i in range(ns)],["sess1"]*ns,
                     Modality.MRS,np.full((ns,2),8.0))
    reg=PairedRegression(1.0); cv=reg.subject_cv(ds)
    for nm,s in cv["per_target"].items():
        print(f"      {nm}: mean_r={s['mean_r']:.3f} min_r={s['min_r']:.3f}")
    rep("subject-wise CV valid",cv["valid"])
    rep("coverage mask present",cv["coverage_fraction"]>0.9)
    small=PairedDataset(X[:10],Y[:10],["GABA_MRS","Glu_MRS"],ds.feature_names,
                        [f"s{i}" for i in range(10)],["s"]*10)
    rep("<30 subjects -> MODE_E",reg.subject_cv(small)["status"]==MODE_E)
    pm=reg.permutation_test(ds,n_perm=200)
    rep("in-sample flag present",pm["OBSERVED_IS_IN_SAMPLE"])
    rep("permutation significant",pm["above_chance"],f"p={pm['p_value']:.4f}")
    rep("prediction certified",reg.gate_prediction(ds,cv,"GABA_MRS")["certified"])
    bad=PairedDataset(X,Y,["GABA_MRS","Glu_MRS"],ds.feature_names,
                      ds.subject_ids,ds.session_ids,Modality.MRS,np.full((ns,2),35.))
    b=False
    try: reg.gate_prediction(bad,cv,"GABA_MRS")
    except UnreliableQuantification: b=True
    rep("unreliable ground truth REFUSED",b)

    print("\n[T8] MODELS: chem_decay constrained")
    if HAS_TORCH:
        torch.manual_seed(0)
        wm=NeurochemWorldModel(24,2,4,64)
        with torch.no_grad(): wm.decay_logit.fill_(20.0)   # attempt to force >1
        d=float(wm.decay)
        rep("decay stays < 1 even with extreme logit",0<d<1,f"decay={d:.6f}")
        wo=wm(torch.randn(4,24),torch.randn(4,2),torch.randn(4,4))
        rep("latent labelled not a concentration",
            "NOT a measured concentration" in wo["STATUS"])
        m=MultimodalTransformer(24,10,64,4,2,2,32)
        o=m(torch.randn(4,16,24),torch.randn(4,3,10))
        rep("transformer forward",tuple(o["pred"].shape)==(4,2))
        df=SpectrumDiffusion(64,30,128)
        opt=torch.optim.Adam(df.parameters(),lr=3e-3)
        x0=torch.randn(8,64); l0=lN=None
        for ep in range(30):
            opt.zero_grad(); L=df.loss(x0); L.backward(); opt.step()
            if ep==0: l0=float(L)
            lN=float(L)
        rep("diffusion trains",lN<l0)
        rep("spectrum labelled HYPOTHESIS","never a measurement" in df.sample(2)["STATUS"])
    else:
        rep("torch available",False,"pip install torch")

    print("\n[T9] SD GATE + PIPELINE")
    b=False
    try: pipe.refuse_concentration_from_eeg("GABA concentration from EEG")
    except NotIdentifiable: b=True
    rep("explicit request REFUSED",b)
    eo=pipe.analyse_eeg(xi,fs)
    rep("EEG analysis reports no concentration","No concentration" in eo["CONCLUSION"])
    mo=pipe.analyse_mrs(syn["spectrum"],syn["basis"],syn["names"],edited=True)
    rep("MRS analysis completes","fit" in mo)
    rep("missing CRLB -> MODE_E",
        pipe.sd.certify("eeg_to_chemistry",("PAIRED","SUBJCV","PERM"))["status"]==MODE_E)
    rr=pipe.report()
    print(f"      SD: {rr['sd']['n']} outputs, {rr['sd']['certified']} certified, "
          f"{rr['sd']['mode_E']} Mode E")
    rep("Mode E retained",rr["sd"]["mode_E"]>0)
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
FIXED
  Clipping replaced by a true active-set NNLS with augmented Tikhonov.
  CRLB from the ESTIMATED noise variance, on the ACTIVE set only, with
  the penalised-estimator bias declared as a lower bound.
  Logan integration is trapezoidal and linearity is ENFORCED.
  chem_decay is sigmoid-constrained to (0,1).
  Subject-CV returns a coverage mask; uncovered rows are not predictions.
  The permutation score is labelled IN-SAMPLE and non-predictive.

NOT CLAIMED
  Concentration from EEG/MEG remains impossible. MRS/PET measure chemistry.
  A cohort association is not an individual concentration readout.
""")
    return ok

def main() -> int:
    ap=argparse.ArgumentParser(description="Neurochemical inference v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","capabilities","metabolites","tracers","mrs","eeg",
                 "identifiability","packages","report"])
    ap.add_argument("--field",type=float,default=3.0)
    ap.add_argument("--edited",action="store_true")
    ap.add_argument("--knee",action="store_true")
    ap.add_argument("--json")
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    pipe=Pipeline(); out={}
    if a.cmd=="capabilities": out={k:asdict(v) for k,v in CAPABILITIES.items()}
    elif a.cmd=="metabolites": out={k:asdict(v) for k,v in METABOLITES.items()}
    elif a.cmd=="tracers": out=PET_TRACERS
    elif a.cmd=="packages": out=PACKAGES
    elif a.cmd=="identifiability":
        out=identifiability_analysis(build_eeg_chemical_jacobian(12,10),10,
            ["Glu","GABA","DA","5HT","ACh","NE","GSH","Gln","Tau","mI"])
    elif a.cmd=="mrs":
        s=synthesize_mrs({"NAA":10.,"Cre":8.,"Cho":2.,"Glu":9.,"GABA":1.5},a.field)
        log.warning("synthetic spectrum (code test only); supply real MRS data")
        out=pipe.analyse_mrs(s["spectrum"],s["basis"],s["names"],edited=a.edited)
    elif a.cmd=="eeg":
        rng=np.random.default_rng(0); fs=256.; n=4096
        f=np.fft.rfftfreq(n,1/fs); amp=np.zeros_like(f); amp[1:]=f[1:]**(-1.2)
        x=np.fft.irfft(amp*np.exp(1j*rng.uniform(0,2*PI,len(f))),n=n)
        log.warning("synthetic EEG (code test only)")
        out=pipe.analyse_eeg(x,fs,knee=a.knee)
    elif a.cmd=="report": out=pipe.report()
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
