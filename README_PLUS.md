# MENTAL ONE Ecosystem — README PLUS

**Developer:** Yoon A Limsuwan  
**Organization:** MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT  
**ORCID:** 0009-0008-2374-0788  
**GitHub:** [yoonalimsuwan](https://github.com/yoonalimsuwan)  
**License:** MIT  
**Ecosystem Version:** 2.0.0 (`MENTAL_VERSION`)  
**Year:** 2026

---

> *"A fully differentiable, physics-grounded computational framework for*
> *psychiatric simulation, brain-state classification, and AI-accelerated*
> *mental health trajectory prediction."*

---

## AI Development Contributors

All seven files in this ecosystem were co-developed with the assistance of large language model AI systems. Their contributions are explicitly credited as part of this project's open and transparent development practice.

| AI System | Developer | Primary Contributions |
|---|---|---|
| **Claude** | Anthropic | Architecture design, full differentiability audits, loss functions, training pipeline, CSOC/SSC canonical design, cross-ecosystem integration patterns, production hardening |
| **GPT** | OpenAI | Literature cross-checks, numerical stability review, DSM-5 mapping verification, algorithm consultation |
| **Gemini** | Google | Operator scaffolding, structural base classes, physical mapping references, multi-modal data pipeline design |
| **DeepSeek** | DeepSeek AI | Stencil verification, alternative integrator verification, fixed-point iteration alternatives |

---

## Table of Contents

1. [Overview](#overview)
2. [Ecosystem Architecture](#ecosystem-architecture)
3. [File Reference](#file-reference)
   - [one_core_mental.py](#1-one_core_mentalpy--foundation-layer)
   - [mental_one.py](#2-mental_onepy--psychiatric-engine)
   - [structural_langevin_mental.py](#3-structural_langevin_mentalpy--baoab-langevin-integrator)
   - [langevin_mental_bridge.py](#4-langevin_mental_bridgepy--langevin--brain-state-bridge)
   - [structural_cahn_hilliard_3d.py](#5-structural_cahn_hilliard_3dpy--3d-phase-field-pde)
   - [psy_one_bridge_diff.py](#6-psy_one_bridge_diffpy--fully-differentiable-psyche-triad)
   - [mental_structural_operator_v3.py](#7-mental_structural_operator_v3py--ai-surrogate-trainer)
4. [Dependency Graph](#dependency-graph)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Mathematical Framework](#mathematical-framework)
8. [API Reference](#api-reference)
9. [Configuration & Hyperparameters](#configuration--hyperparameters)
10. [Data Formats](#data-formats)
11. [Supported Psychiatric Disorders](#supported-psychiatric-disorders)
12. [Hardware Compatibility](#hardware-compatibility)
13. [License](#license)

---

## Overview

The **MENTAL ONE Ecosystem** is a suite of seven PyTorch-based, fully differentiable scientific simulation modules for computational psychiatry and neuroscience. It implements a novel theoretical framework — **Controlled Self-Organized Criticality (CSOC)** combined with **Semantic State Contraction (SSC)** — to model mental health as a physical system operating near criticality.

The ecosystem spans four computational layers:

| Layer | Module | Physical Analogy |
|---|---|---|
| Foundation | `one_core_mental.py` | Abstract operators & base classes |
| Psychiatric engine | `mental_one.py` | Brain-state energy landscape |
| Stochastic dynamics | `structural_langevin_mental.py` | Thermal bath + BAOAB integration |
| Brain-state bridge | `langevin_mental_bridge.py` | EEG ↔ molecular dynamics |
| Phase separation | `structural_cahn_hilliard_3d.py` | fMRI spatial phase fields |
| Psyche triad | `psy_one_bridge_diff.py` | Id / Ego / Superego DEQ system |
| AI surrogate | `mental_structural_operator_v3.py` | O(1) neural operator replacement |

Every module is **end-to-end differentiable**, enabling gradient-based training from raw clinical data (EEG, fMRI, questionnaire scores) through all physical simulation layers to diagnostic outputs.

---

## Ecosystem Architecture

```
┌─────────────────────────────────────────────────────────┐
│              mental_structural_operator_v3.py           │
│    (MSNO — AI Surrogate; O(1) replacement for all PDEs) │
└──────────────────────┬──────────────────────────────────┘
                       │ trains / accelerates
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌──────────────────────┐
│ mental_one │  │ langevin_  │  │ psy_one_bridge_diff  │
│    .py     │  │ mental_    │  │  (Id/Ego/Superego     │
│            │  │ bridge.py  │  │   DEQ triad)          │
└─────┬──────┘  └─────┬──────┘  └──────────┬───────────┘
      │               │                    │
      ▼               ▼                    ▼
┌──────────────────────────────────────────────────────┐
│         structural_langevin_mental.py                │
│     (BAOAB Langevin MD + CSOCThermostat + Itô)       │
└────────────────────────┬─────────────────────────────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
┌──────────┐   ┌──────────────────────┐  ┌───────────────┐
│ one_core │   │ structural_cahn_     │  │ one_core_*    │
│ _mental  │   │ hilliard_3d.py       │  │ (cross-       │
│ .py      │   │ (CH3D phase PDE)     │  │  ecosystem)   │
│(FOUNDATION)  └──────────────────────┘  └───────────────┘
└──────────┘
```

All arrows represent **import dependencies**. `one_core_mental.py` is the single foundation with no upstream dependencies within the ecosystem.

---

## File Reference

---

### 1. `one_core_mental.py` — Foundation Layer

**Role:** The single source of truth for all shared base classes, abstract operators, and mathematical primitives used throughout the MENTAL ONE ecosystem.

**Key exports:**

| Symbol | Type | Description |
|---|---|---|
| `MENTAL_VERSION` | `str` | Ecosystem-wide version string (`"2.0.0"`) |
| `soft_clamp(x, lo, hi)` | function | Differentiable tanh-based clamp; replaces `.clamp()` throughout |
| `get_device(preferred)` | function | Unified hardware-backend selector (CUDA / MPS / CPU) |
| `SemanticStateContraction` | `nn.Module` | SSC EMA filter — maps raw brain state → contracted semantic state |
| `CSOCBase` | abstract `nn.Module` | Abstract base for all Controlled SOC controllers |
| `InterfaceDetectorBase` | abstract `nn.Module` | Abstract base for interface / boundary detectors |
| `StructuralItoBase` | abstract `nn.Module` | Abstract base for Structural Itô noise correction |
| `DifferentiableRG` | `nn.Module` | Fully differentiable learnable Renormalization Group smoother |
| `DifferentiableSOC` | `nn.Module` | Fully differentiable n-step SOC dynamics |
| `CahnHilliardMentalBridge` | `nn.Module` | Cross-ecosystem bridge: CH3D ↔ MENTAL ONE |
| `structural_biharmonic_n` | function | Recursive structural biharmonic operator Δ_S^n |

**Design notes:**
- `SemanticStateContraction` uses a boolean `_initialized` buffer (not a Python `bool`) to ensure correct `state_dict` serialization and DDP broadcasting.
- All abstract base classes use Python `ABC` + `@abstractmethod`, enforcing interface compliance at instantiation time.
- `soft_clamp` is the ecosystem-standard numerical stability primitive: `c + s·tanh((x−c)/s)`.

---

### 2. `mental_one.py` — Psychiatric Engine

**Role:** The primary multi-modal psychiatric simulation and classification engine. Ingests EEG, MEG, fMRI, clinical scores, and genetic data; produces DSM-5 / ICD-10 diagnoses, trajectory predictions, and intervention plans.

**Key classes:**

| Class | Description |
|---|---|
| `MentalONEEngine` | Master orchestrator; entry point for all classification and simulation tasks |
| `SSCClassifier` | Energy-minimisation classifier using SSC distance to disorder reference states |
| `CSOCKernel` | Learnable CSOC kernel for adaptive SOC dynamics |
| `SOCController` | Concrete `CSOCBase` subclass; controls self-organised criticality |
| `DSM5DiagnosisEngine` | Full DSM-5 / ICD-10-11 engine (PHQ-9, GAD-7, HAMD-17, PCL-5, YMRS, PANSS) |
| `MentalHealthEvolution` | RG-smoothed disease burden trajectory prediction |
| `ItoProcess` | Per-patient Itô-process trajectory simulation |
| `InterventionDesigner` | Control-theoretic intervention designer (pharmacological, psychotherapy, environment) |
| `BVConsistency` | BV field theory check for brain network topological consistency |
| `MultiModalDataLoader` | Loader for EEG (MNE), MEG, fMRI (Nilearn), clinical, genetic data |
| `ExtremeTrainer` | Distributed training harness (DDP, AMP, gradient checkpointing, early stopping) |

**CLI usage:**

```bash
# Single GPU / CPU training
python mental_one.py train \
    --dataset modma \
    --data_dir /data \
    --subject_list sub-001 sub-002 \
    --epochs 100

# Multi-GPU DDP training (torchrun)
torchrun --nproc_per_node=4 mental_one.py train \
    --dataset modma --data_dir /data \
    --subject_list sub-001 sub-002 \
    --epochs 100 --ddp

# Inference
python mental_one.py classify -i patient.edf
python mental_one.py intervene -i state.json
```

**Supported datasets:** MODMA, OpenNeuro, custom EDF/BrainVision/EEGLab formats, TCGA (via EVOLUTION ONE), PDB (via REAL FOLD ONE).

---

### 3. `structural_langevin_mental.py` — BAOAB Langevin Integrator

**Role:** Provides the stochastic molecular-dynamics backbone for brain-state evolution. Implements the BAOAB Langevin splitting scheme adapted for the Structural Calculus framework.

**Key classes:**

| Class | Description |
|---|---|
| `AdvancedStructuralLangevin` | Main BAOAB integrator with structural operators; primary simulation engine |
| `CSOCThermostat` | Concrete `CSOCBase`; adaptive temperature controller targeting σ_target criticality |
| `StructuralItoNoise` | Concrete `StructuralItoBase`; multiplicative noise correction for non-flat manifolds |
| `InterfaceDetector` | Concrete `InterfaceDetectorBase`; detects phase boundaries in brain-state space |

**BAOAB splitting:**

```
B: half-step velocity kick  (force)
A: half-step position update
O: full Ornstein-Uhlenbeck noise step
A: half-step position update
B: half-step velocity kick  (force)
```

Each step is implemented as a differentiable `nn.Module` forward pass, enabling gradient flow through the entire trajectory for training the SSC reference states.

**Physics mapping:**

| MD quantity | Mental health quantity |
|---|---|
| Particle position `x` | Brain state vector `s ∈ ℝ^N` |
| Force `F = −∇E` | SSC energy gradient |
| Temperature `T` | Neural noise / arousal level |
| Structural stress `σ` | Distance from healthy reference state |
| Langevin `dt` | Contraction step size |

---

### 4. `langevin_mental_bridge.py` — Langevin ↔ Brain-State Bridge

**Role:** Upgrades `mental_one.py`'s original `MentalHealthEvolution` and `ItoProcess` classes to fully BAOAB-based, differentiable implementations. Provides the bridge between molecular-dynamics integration and EEG/MEG signal-space representations.

**Key classes:**

| Class | Description |
|---|---|
| `LangevinMentalEvolution` | Drop-in upgrade for `MentalHealthEvolution`; full BAOAB trajectory simulation |
| `LangevinBrainIntegrator` | Core BAOAB integrator specialised for brain-state coordinates |
| `LangevinSOCEvolve` | SOC evolution step; concrete `CSOCBase` with Langevin-coupled dynamics |
| `LangevinItoStep` | Itô correction step; concrete `StructuralItoBase` for brain-state noise |
| `BrainStateInterfaceDetector` | Brain-space interface detector; identifies disorder phase boundaries |
| `LangevinCHMentalBridge` | Coupling bridge between CH3D phase-field and Langevin brain dynamics |

**Patch utility:**

```python
from langevin_mental_bridge import patch_mental_one

engine = MentalONEEngine(...)
patch_mental_one(engine, target_disorder='MDD')
# engine.evolution is now LangevinMentalEvolution
```

---

### 5. `structural_cahn_hilliard_3d.py` — 3D Phase-Field PDE

**Role:** Implements the 4th-order Cahn-Hilliard PDE on a 3D structural (regime-dependent) manifold. In the MENTAL ONE context, it models spatial phase separation of mental states across 3D fMRI voxel grids — for example, the separation of OCD/Superego-dominant regions from Id/impulsive regions.

**Key classes:**

| Class | Description |
|---|---|
| `StructuralCahnHilliard3D` | Base class; 4th-order CH PDE with structural operators |
| `ThinFilmStructuralCahnHilliard3D` | Extension with thin-film mobility `M(u) = softplus(u)³` and Hamaker wetting term |
| `PhaseFieldCrystal3D` | PFC extension; 6th-order PDE via recursive `structural_biharmonic_n` |
| `CahnHilliardConfig` | Dataclass for all CH3D configuration parameters |
| `CahnHilliardDNSBridge` | Korteweg-stress coupling bridge to SUPER DNS ONE compressible solver |

**Laplacian backends (selectable via `CahnHilliardConfig.laplacian`):**

| Backend | Class | Notes |
|---|---|---|
| `'conv3d'` | `_Conv3dLaplacian` | GPU-parallel staggered stencil via Conv3d; default |
| `'fft'` | `_FFTLaplacian` | Spectral O(N log N); exact for periodic domains |
| `'roll'` | `_RollLaplacian` | Pure roll-based; reference implementation |

**Structural operators:**

```
∇_S u   = σ(x) · ∇u            (Structural Gradient)
div_S F = div(σ(x) · F)         (Structural Divergence)
Δ_S u   = div(σ(x) · ∇u)       (Structural Laplacian)
Δ_S² u  = Δ_S(Δ_S u)           (Structural Bi-Laplacian)

μ_R     = (u³ − u) − ε² · Δ_S u    (Chemical Potential)
∂u/∂t   = Δ_S(μ_R)                  (CH Evolution)
```

---

### 6. `psy_one_bridge_diff.py` — Fully Differentiable Psyche Triad

**Role:** Implements the Id / Ego / Superego triadic model of psychic regulation as a fully differentiable PyTorch system. The Ego module solves a DEQ-style fixed-point problem (Anderson mixing) balancing drive from the Id against normative constraints from the Superego, while keeping the full computation graph intact for gradient-based training.

**Key classes:**

| Class | Description |
|---|---|
| `PSYONEBridge` | Master bridge; orchestrates the full Id→Ego→Superego triad |
| `PsycheTriad` | Core triad module; produces `PsycheTriadState` + scalar `total_loss` |
| `IdModule` | Drive / reward-seeking impulse generator; uses Gumbel-Softmax |
| `SuperegoModule` | Normative policy module; computes KL divergence in log-space |
| `EgoModule` | DEQ-style fixed-point Ego optimiser; learnable step-size parameter |
| `AndersonMixer` | Anderson acceleration for DEQ fixed-point convergence |
| `GumbelAnnealScheduler` | Temperature annealing τ: 1.0 → 0.1 over training |
| `PsycheCahnHilliardBridge` | Coupling bridge: PSY ONE ↔ CH3D spatial phase fields |
| `LongitudinalPsycheTracker` | Multi-session psychiatric trajectory tracker |
| `PSYONEBenchmark` | Benchmarking harness for psyche triad performance |
| `PsychopathologyMode` | `Enum` of supported psychiatric conditions |
| `PsycheConfig` | Dataclass for all triad hyperparameters |

**Key differentiability upgrades (V2 → current):**

| Issue | Old approach | New approach |
|---|---|---|
| History buffer | `history_buffer[ptr] = input.detach()` (broken graph) | `SoftHistoryBuffer` — EMA with full gradient flow |
| Ego optimisation | Manual gradient loop with `.detach()` | DEQ / Anderson mixing with `torch.linalg` implicit diff |
| Action selection | `multinomial` discrete sampling (∇ = 0) | Gumbel-Softmax straight-through estimator |
| KL divergence | Raw ratio (explodes near p→0) | `F.kl_div` in log-space |

---

### 7. `mental_structural_operator_v3.py` — AI Surrogate Trainer

**Role:** A production-grade **Neural Operator** (Fourier Neural Operator architecture) that learns to replace all heavy numerical integrations in the MENTAL ONE ecosystem with O(1) forward passes. Trained once offline, then deployed as a fast surrogate for real-time inference.

**Key classes:**

| Class | Description |
|---|---|
| `MentalStructuralNeuralOperator` | Master surrogate model; four operator branches |
| `BrainSpectralConv1D` | 1D FNO layer for EEG/MEG; σ-gated spectral + local mixing |
| `BrainGraphOperator` | Message-passing graph layer for brain connectomes |
| `BrainSpatialConv3D` | 3D FNO layer for fMRI phase fields; σ-gated 8-corner FFT |
| `PsycheSurrogateOperator` | Feed-forward surrogate for Id/Ego/Superego DEQ |
| `MSNOTrainer` | Full training loop with AMP, scheduler, checkpointing, logging |
| `MSNOInference` | Lightweight inference wrapper for deployment |
| `MSNOLosses` | Namespace for all surrogate loss functions |
| `MSNOTrainingConfig` | Dataclass for all training hyperparameters |
| `SyntheticMentalDataset` | Synthetic benchmark dataset for smoke-testing |

**Surrogate mapping:**

| Surrogate method | Replaces | Complexity reduction |
|---|---|---|
| `predict_eeg_trajectory(eeg, σ)` | BAOAB Langevin N-step integration | N steps → O(1) |
| `predict_graph_state(nodes, edges, σ)` | Connectome message-passing N iterations | N iters → O(1) |
| `predict_spatial_phase(u, σ)` | 4th-order CH3D PDE N steps | N steps → O(1) |
| `optimize_ego(id, se, σ)` | DEQ / Anderson mixing N iterations | N iters → O(1) |

**Loss functions:**

| Loss | Formula | Task |
|---|---|---|
| `eeg_loss` | 0.5·MSE + 0.5·L1 | EEG trajectory |
| `phase_loss` | MSE + 0.1·‖∇pred − ∇target‖ | CH3D phase field |
| `ego_loss` | KL(pred ‖ target) | Ego action distribution |

**Training:**

```python
from mental_structural_operator_v3 import (
    MSNOTrainingConfig, MentalStructuralNeuralOperator,
    MSNOTrainer, build_dataloaders
)

cfg     = MSNOTrainingConfig(epochs=50, batch_size=16, lr=1e-4)
model   = MentalStructuralNeuralOperator(
    eeg_channels=19, latent_dim=64, modes_1d=32, modes_3d=8, action_dim=10
)
train_loader, val_loader = build_dataloaders(cfg)
trainer = MSNOTrainer(model, train_loader, val_loader, cfg)
trainer.fit()
```

**Inference after training:**

```python
from mental_structural_operator_v3 import MSNOInference

inf         = MSNOInference.from_checkpoint("msno_checkpoints/msno_best.pt")
future_eeg  = inf.predict_eeg(eeg_t0, sigma)
future_u    = inf.predict_phase(u_t0, sigma)
action_dist = inf.predict_ego(id_proposals, se_norm, sigma)
```

---

## Dependency Graph

```
one_core_mental.py          ← no upstream deps; pure PyTorch
       ↑
       ├── mental_one.py
       ├── structural_langevin_mental.py
       ├── langevin_mental_bridge.py
       ├── psy_one_bridge_diff.py
       └── structural_cahn_hilliard_3d.py
                                   ↑
                           (optional cross-ecosystem:
                            one_core.py / one_core_fold.py)

mental_structural_operator_v3.py
       ↑ (optional; graceful fallback if unavailable)
       ├── one_core_mental.py
       ├── mental_one.py
       ├── structural_cahn_hilliard_3d.py
       ├── psy_one_bridge_diff.py
       └── langevin_mental_bridge.py
```

All cross-ecosystem imports are wrapped in `try/except` with graceful fallback stubs. Each module is independently runnable.

---

## Installation

**Requirements:**

```bash
pip install torch>=2.0 numpy scipy pandas matplotlib seaborn \
            mne nilearn scikit-learn networkx biopython
```

**Optional (for full features):**

```bash
pip install optuna                        # hyperparameter tuning
pip install openmm openmm-ml              # REAL FOLD ONE integration
```

**Clone and set up:**

```bash
git clone https://github.com/yoonalimsuwan/mental-one
cd mental-one
pip install -r requirements.txt
```

**Python version:** 3.9+  
**PyTorch version:** 2.0+ (required for `torch.linalg` implicit differentiation in `psy_one_bridge_diff.py`)

---

## Quick Start

### 1. Classify a patient EEG

```python
from mental_one import MentalONEEngine

engine = MentalONEEngine(device="cuda")
engine.load_checkpoint("mental_one_best.pt")

result = engine.classify("patient_001.edf")
print(result["primary_diagnosis"])     # e.g. "MDD"
print(result["confidence"])            # e.g. 0.87
print(result["intervention_plan"])
```

### 2. Run a full psychiatric trajectory simulation

```python
from mental_one import MentalONEEngine
from langevin_mental_bridge import patch_mental_one

engine = MentalONEEngine(device="cuda")
patch_mental_one(engine, target_disorder="MDD")

trajectory = engine.simulate_trajectory(
    eeg_state=initial_eeg,   # (C, T) tensor
    n_steps=500,
    dt=0.01,
)
```

### 3. Run the Psyche Triad

```python
from psy_one_bridge_diff import PSYONEBridge, PsycheConfig

cfg    = PsycheConfig(action_dim=10, n_ego_iter=50)
bridge = PSYONEBridge(cfg, device="cuda")

state, loss = bridge(sensory_input=eeg_features)
print(state.ego_policy)         # optimised Ego action distribution
loss.backward()                 # fully differentiable end-to-end
```

### 4. Train the AI surrogate (MSNO)

```bash
python mental_structural_operator_v3.py
# trains for 20 epochs on synthetic data, saves to ./msno_checkpoints/
```

### 5. Use the trained surrogate for O(1) inference

```python
from mental_structural_operator_v3 import MSNOInference
import torch

inf   = MSNOInference.from_checkpoint("msno_checkpoints/msno_best.pt")
sigma = torch.tensor([[1.2]])   # structural stress scalar

future_eeg = inf.predict_eeg(eeg_now, sigma.unsqueeze(-1))   # (B, C, T)
future_u   = inf.predict_phase(u_now, sigma.view(1,1,1,1,1)) # (B, 1, X, Y, Z)
action     = inf.predict_ego(id_prop, se_norm, sigma)         # (B, action_dim)
```

---

## Mathematical Framework

The ecosystem is grounded in **Structural Calculus** — a novel mathematical framework combining:

- **BV Theory** (Ambrosio) — brain state functions of bounded variation with jump discontinuities at disorder phase transitions
- **Stochastic Navier–Stokes** (Flandoli) — stochastic PDE framework for brain-state fluctuations
- **Structural Derivative** — `D^s u = ∇u + [u] δ_Γ` — combines smooth gradient with a measure-theoretic jump term at interfaces

**Core operators** (all defined in `one_core_mental.py` and used throughout):

```
Structural Gradient:      ∇_S u   = σ(x) · ∇u
Structural Divergence:    div_S F = div(σ(x) · F)
Structural Laplacian:     Δ_S u   = div_S(∇_S u)
Structural Bi-Laplacian:  Δ_S² u  = Δ_S(Δ_S u)
Structural Biharmonic-n:  Δ_S^n u = Δ_S(Δ_S^(n-1) u)   [recursive]
```

where `σ(x)` is the **Structural Regime Field** — a scalar field encoding the local distance from the healthy brain-state critical point.

**CSOC universality chain** (from sub-cellular to phenomenological):

```
Protein contact networks  →  EEG neural oscillations
     →  Compressible DNS turbulence  →  Psychiatric phase fields
```

All scales are linked by the same CSOC + SSC mathematical structure, implemented as the shared base classes in `one_core_mental.py`.

---

## API Reference

### `one_core_mental.py`

```python
from one_core_mental import (
    SemanticStateContraction,   # SSC EMA filter
    CSOCBase,                   # abstract CSOC base
    InterfaceDetectorBase,      # abstract interface detector
    StructuralItoBase,          # abstract Itô correction
    DifferentiableRG,           # learnable RG smoother
    DifferentiableSOC,          # differentiable SOC dynamics
    CahnHilliardMentalBridge,   # CH3D ↔ MENTAL ONE bridge
    structural_biharmonic_n,    # Δ_S^n operator
    soft_clamp,                 # differentiable clamp
    get_device,                 # hardware selector
    MENTAL_VERSION,             # version string
)
```

### `mental_one.py`

```python
from mental_one import MentalONEEngine, SSCClassifier, DSM5DiagnosisEngine
engine = MentalONEEngine(device="cuda", enable_ch3d_bridge=True)
```

### `structural_langevin_mental.py`

```python
from structural_langevin_mental import AdvancedStructuralLangevin, CSOCThermostat
integrator = AdvancedStructuralLangevin(n_dof=64, dt=0.01, sigma_target=1.0)
result     = integrator.baoab_step(pos, vel, sigma_field)
```

### `langevin_mental_bridge.py`

```python
from langevin_mental_bridge import LangevinMentalEvolution, LangevinCHMentalBridge, patch_mental_one
evo  = LangevinMentalEvolution(soc_ctrl, rg_smoother, ssc_classifier)
traj = evo(initial_state, n_steps=200)
```

### `structural_cahn_hilliard_3d.py`

```python
from structural_cahn_hilliard_3d import StructuralCahnHilliard3D, CahnHilliardConfig
cfg    = CahnHilliardConfig(nx=32, ny=32, nz=32, eps=0.05, laplacian='fft')
solver = StructuralCahnHilliard3D(cfg)
u_next = solver.step(u_now, sigma_field)
```

### `psy_one_bridge_diff.py`

```python
from psy_one_bridge_diff import PSYONEBridge, PsycheConfig, PsychopathologyMode
cfg    = PsycheConfig(action_dim=10, mode=PsychopathologyMode.MDD)
bridge = PSYONEBridge(cfg, device="cuda")
state, loss = bridge(sensory_input)
loss.backward()
```

### `mental_structural_operator_v3.py`

```python
from mental_structural_operator_v3 import (
    MentalStructuralNeuralOperator, MSNOTrainer,
    MSNOTrainingConfig, MSNOInference, build_dataloaders
)
cfg     = MSNOTrainingConfig(epochs=50, batch_size=16)
model   = MentalStructuralNeuralOperator(latent_dim=64, modes_1d=32, modes_3d=8)
trainer = MSNOTrainer(model, *build_dataloaders(cfg), cfg)
trainer.fit()
```

---

## Configuration & Hyperparameters

### `MSNOTrainingConfig` (key fields)

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 50 | Total training epochs |
| `batch_size` | 16 | Samples per batch |
| `lr` | 1e-4 | Initial learning rate (AdamW) |
| `weight_decay` | 1e-5 | L2 regularisation |
| `grad_clip` | 1.0 | Max-norm gradient clipping |
| `warmup_epochs` | 5 | Linear warmup before cosine annealing |
| `lambda_eeg` | 1.0 | EEG trajectory loss weight |
| `lambda_ch3d` | 1.0 | CH3D phase loss weight |
| `lambda_ego` | 0.5 | Ego KL loss weight |
| `use_amp` | True | Mixed-precision (CUDA only) |
| `latent_dim` | 64 | FNO channel width |
| `modes_1d` | 32 | Fourier modes for 1D spectral layer |
| `modes_3d` | 8 | Fourier modes per axis for 3D layer |

### `CahnHilliardConfig` (key fields)

| Parameter | Default | Description |
|---|---|---|
| `nx, ny, nz` | 32, 32, 32 | Grid dimensions |
| `eps` | 0.05 | Interface thickness parameter |
| `laplacian` | `'conv3d'` | Backend: `'conv3d'`, `'fft'`, `'roll'` |
| `ssc_stabilise` | False | Enable SSC-based PFC stabilisation |

### `PsycheConfig` (key fields)

| Parameter | Default | Description |
|---|---|---|
| `action_dim` | 10 | Action space dimensionality |
| `n_ego_iter` | 50 | Max Anderson mixing iterations |
| `tau_init` | 1.0 | Gumbel-Softmax initial temperature |
| `tau_min` | 0.1 | Gumbel-Softmax minimum temperature |
| `mode` | `HEALTHY` | Starting `PsychopathologyMode` |

---

## Data Formats

| Modality | Accepted formats | Primary loader |
|---|---|---|
| EEG | `.edf`, `.bdf`, BrainVision, EEGLab | MNE-Python via `MultiModalDataLoader` |
| MEG | `.fif`, CTF `.ds` | MNE-Python |
| fMRI | NIfTI (`.nii`, `.nii.gz`) | Nilearn |
| Clinical scores | JSON (`{"PHQ9": 14, "GAD7": 10, ...}`) | `DSM5DiagnosisEngine` |
| Genetic | VCF, TCGA mutation matrices | EVOLUTION ONE bridge |
| Brain state tensor | `torch.Tensor (B, C, T)` | Direct API |
| Phase field | `torch.Tensor (B, 1, X, Y, Z)` | Direct API |

---

## Supported Psychiatric Disorders

The ecosystem currently models the following conditions via `PsychopathologyMode` and `DSM5DiagnosisEngine`:

| Code | Disorder |
|---|---|
| `MDD` | Major Depressive Disorder |
| `BD` | Bipolar Disorder |
| `SZ` | Schizophrenia |
| `OCD` | Obsessive-Compulsive Disorder |
| `PTSD` | Post-Traumatic Stress Disorder |
| `ANX` | Generalised Anxiety Disorder |
| `ASD` | Autism Spectrum Disorder |
| `ADHD` | Attention Deficit Hyperactivity Disorder |
| `HEALTHY` | Healthy reference baseline |

Clinical assessment instruments supported: PHQ-9, GAD-7, HAMD-17, PCL-5, YMRS, PANSS (ICD-10 / DSM-5).

---

## Hardware Compatibility

| Backend | Status | Notes |
|---|---|---|
| NVIDIA CUDA | ✅ Full support | AMP, DDP, gradient checkpointing |
| Apple MPS | ✅ Full support | `get_device("mps")`; `num_workers=0` required |
| CPU | ✅ Full support | All features; AMP disabled automatically |
| Intel XPU | ✅ (mental_one) | Via `torch.xpu` |
| Huawei Ascend | ✅ (mental_one) | Via `torch_npu` |
| Multi-GPU DDP | ✅ (mental_one, MSNO) | `torchrun --nproc_per_node=N` |

---

## License

```
MIT License

Copyright (c) 2026 Yoon A Limsuwan / MSPS NETWORK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*README_PLUS.md — MENTAL ONE Ecosystem V2.0.0 — Yoon A Limsuwan / MSPS NETWORK — 2026*
