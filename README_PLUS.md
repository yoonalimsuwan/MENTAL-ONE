# ONE Ecosystem

**Developer:** Yoon A Limsuwan / MSPS NETWORK
**License:** MIT
**ORCID:** [0009-0008-2374-0788](https://orcid.org/0009-0008-2374-0788)
**GitHub:** [yoonalimsuwan](https://github.com/yoonalimsuwan)
**Year:** 2026

---

## Overview

The ONE Ecosystem is a suite of fully differentiable, PyTorch-native simulation frameworks spanning multiple physical scales — from fundamental particles and cosmology, through molecular folding and atomic dynamics, to neural and psychiatric state-space modelling. Every module shares a common mathematical backbone rooted in four theoretical papers on Structural Calculus, enabling seamless cross-scale integration via the CSOC universality chain.

All frameworks are end-to-end differentiable. Gradients flow through every layer — from high-energy physics observables down to brain-state trajectories — enabling gradient-based optimisation, sensitivity analysis, and joint training across physical scales.

---

## Theoretical Foundation — The 4-Paper Structural Calculus

All modules in the ONE Ecosystem are built on the following unified theoretical framework:

| Paper | Title | Role in Ecosystem |
|---|---|---|
| **Paper 1** | Regime-Dependent Analytical Framework & Structural Operators | Defines the operator algebra underlying all dynamics |
| **Paper 2** | BV Jump Measures & Self-Evolving Interfaces | Governs discontinuous transitions at physical boundaries |
| **Paper 3** | Structural Itô Calculus & Multiplicative Noise Correction | Provides exact stochastic drift correction at interfaces |
| **Paper 4** | Controlled Self-Organised Criticality (CSOC) & Semantic State Contraction (SSC) | Adaptive thermostat and EMA stress filter — shared across all scales |

### CSOC Universality Chain

A single inheritance hierarchy connects every adaptive parameter module across all physical scales:

```
CSOCBase  (one_core_*.py)
├── CSOCThermostat         → structural_langevin_*.py   (molecular / neural Langevin)
├── SOCController          → mental_one.py / real_fold_one.py  (state-space / protein)
└── [future subclasses]    → any new physical scale
```

`CSOCBase` provides: `self.ssc` (SemanticStateContraction EMA filter), `reset()`, `_normalised_deviation()`, `_smooth_boost()`.

---

## Architecture — Physical Scales

```
STANDARD ONE                    Fundamental physics, cosmology, particle colliders
        ↕  RG flow / scaling
REAL FOLD ONE                   Molecular: proteins, nucleic acids, DNA origami
        ↕  Langevin bridge (structural_langevin_fold.py)
        ↕  BAOAB integrator / Itô correction
MENTAL ONE                      Neural: EEG/MEG brain-state dynamics, psychiatric modelling
        ↕  PSY ONE BRIDGE DIFF  Differentiable cognition–behaviour interface
        ↕  Langevin bridge (langevin_mental_bridge.py)
        ↕  BAOAB integrator / Itô correction
```

Cross-scale communication is handled by the Langevin bridge modules, which map each physical domain's state vector into a shared BAOAB dynamical framework with CSOC-adaptive temperature and Structural Itô drift correction.

---

## Module Reference

### `one_core_mental.py` — MENTAL ONE Shared Foundation

Single source of truth for all MENTAL ONE components. Never redefine these locally.

| Component | Description |
|---|---|
| `SemanticStateContraction` | SSC EMA low-pass filter for structural stress σ. Boolean `_initialized` buffer; `reset()` between sessions |
| `CSOCBase` | Abstract base for all CSOC modules. Provides `ssc`, `reset()`, `_normalised_deviation()`, `_smooth_boost()` |
| `InterfaceDetectorBase` | Abstract base for differentiable interface / transient detectors. Returns mask ∈ [0, 1] |
| `StructuralItoBase` | Abstract base for Structural Itô ½ G(x) ∇G(x) drift correction |
| `DifferentiableRG` | Learnable 1-D RG smoothing kernel. `nn.Parameter` weights — end-to-end trainable |
| `DifferentiableSOC` | Fully differentiable SOC temperature modulation. `base_temp` and `beta` are `nn.Parameter` |
| `soft_clamp` | Differentiable tanh-based clamp. Gradient exists at boundaries (unlike hard `.clamp()`) |
| `get_device` | Hardware selector: CUDA → MPS → Ascend NPU → CPU |

---

### `mental_one.py` — Psychiatric / Neural Engine

End-to-end differentiable engine for EEG/MEG-based psychiatric state modelling across 10 DSM-5 disorder categories.

**Key classes:**

| Class | Description |
|---|---|
| `MultiModalDataLoader` | EEG (EDF), MEG (FIF), fMRI (NIfTI), clinical CSV ingestion |
| `MentalHealthDataset` | PyTorch Dataset with sliding-window epoch extraction |
| `DSM5DiagnosisEngine` | Rule-based DSM-5 criteria check across 10 disorder categories |
| `SSCClassifier` | Core classifier: SSC-filtered energy landscape on brain-state manifold. Differentiable `contraction_update()` via `soft_clamp` |
| `CSOCKernel` | Learnable SOC kernel K(r) = r^{−α} exp(−r/λ). `log_alpha`, `log_lambda`, `log_scale` are `nn.Parameter` |
| `SOCController` | **Inherits `CSOCBase`.** CSOC-driven SOC with learnable `DifferentiableSOC`. No lazy init; `_diff_soc` created in `__init__` |
| `DiffRGRefiner` | Alias for `DifferentiableRG` from `one_core_mental` — backward compatible |
| `MentalHealthEvolution` | SOC + RG evolutionary engine. Forward: evolve brain-state time series |
| `ItoProcess` | Euler-Maruyama Itô SDE integrator for brain-state trajectories |
| `BVConsistency` | BV Jump Measure consistency checker (Paper 2) |
| `InterventionDesigner` | Pharmacological and non-pharmacological intervention optimisation |
| `ExtremeTrainer` | Multi-GPU training with AMP, DDP, cosine annealing |
| `MentalONEEngine` | Top-level orchestration: load data → train → diagnose → intervene |

**Supported disorders:** MDD, Bipolar, Schizophrenia, PTSD, Panic, Conversion, Dissociative, Somatic, Parasomnia, Healthy

**CLI:**
```bash
python mental_one.py classify -i patient.edf --type eeg --output report.json
python mental_one.py train -i dataset/ --epochs 100
python mental_one.py intervene -i patient.edf --disorder MDD
```

---

### `psy_one_bridge_diff.py` — Fully Differentiable Cognition–Behaviour Bridge

v2.0-DIFF. Replaces discrete sampling with Gumbel-Softmax straight-through estimator throughout.

**Key upgrades vs. v1:**

- `SoftHistoryBuffer` — learnable exponential decay (replaces fixed ring buffer)
- Gumbel-Softmax straight-through for all categorical decisions — gradient flows through action selection
- Anderson mixing DEQ-style fixed-point iteration for belief state convergence
- `behavioral_entropy` uses `soft_clamp` (no zero-gradient at boundary)
- `PSYONEBridge.classify()` returns consistent tensor type

**Key classes:**

| Class | Description |
|---|---|
| `PsychopathologyMode` | Enum: Normal, Subthreshold, Acute, Crisis, Recovery |
| `PSYONEBridge` | Main bridge: connects MENTAL ONE engine to behavioural observation space |
| `PSYONEBenchmark` | Standardised benchmark suite across disorder categories |

---

### `structural_langevin_mental.py` — BAOAB Langevin for Brain States

Fully differentiable BAOAB Langevin integrator adapted for neural state-space dynamics.

**Key classes:**

| Class | Description |
|---|---|
| `InterfaceDetector` | Per-atom soft interface mask from local distance variance. Differentiable w.r.t. coordinates |
| `CSOCThermostat` | **Inherits `CSOCBase`.** Adapts temperature T and friction γ from SSC-filtered stress |
| `StructuralItoNoise` | Multiplicative noise G(x) = 1 + amp·mask(x). Computes ½ G ∇G via autograd |
| `AdvancedStructuralLangevin` | Full BAOAB integrator: B–A–O–A–B splitting with structural force, Itô correction, CSOC thermostat |

**`baoa_step()` returns:** `(x_full, v_tilde, T, sigma)` — all tensors, fully differentiable. Call `.item()` only at print/logging sites.

---

### `langevin_mental_bridge.py` — Langevin ↔ MENTAL ONE Bridge

Drop-in replacements that upgrade MENTAL ONE's naive random-walk SOC to thermodynamically consistent BAOAB dynamics.

**Physical mapping:**

| MENTAL ONE concept | Langevin equivalent |
|---|---|
| EEG brain-state vector s ∈ ℝ^N | Coordinates x ∈ ℝ^(N×1) |
| SSC energy gradient ∇E(s) | Force F = −∇E |
| Disorder distance from Healthy | Structural stress σ |
| Contraction step size | Langevin dt |

**Key classes:**

| Class | Drop-in for | Description |
|---|---|---|
| `BrainStateInterfaceDetector` | — | 1-D EEG transient detector via local gradient magnitude |
| `LangevinBrainIntegrator` | — | 1-D BAOAB integrator for brain-state vectors |
| `LangevinSOCEvolve` | `SOCController.soc_evolve()` | BAOAB Langevin SOC evolution |
| `LangevinItoStep` | `ItoProcess.step()` | Langevin-corrected Euler-Maruyama step |
| `LangevinMentalEvolution` | `MentalHealthEvolution` | Full BAOAB-based evolution module |
| `patch_mental_one(engine)` | — | Monkey-patch live `MentalONEEngine` in-place |

**Usage:**
```python
from langevin_mental_bridge import patch_mental_one
engine = MentalONEEngine()
engine.initialise_from_dataset(dataset, subjects)
patch_mental_one(engine, target_disorder='MDD')
# engine.evolution is now LangevinMentalEvolution
```

---

### `one_core_fold.py` — REAL FOLD ONE Shared Foundation

Parallel to `one_core_mental.py` but operating at molecular/residue scale (Å, kcal/mol).

| Component | Description |
|---|---|
| `SemanticStateContraction` | SSC filter for atomic displacement stress |
| `CSOCBase` | CSOC abstract base (same interface as MENTAL ONE) |
| `InterfaceDetectorBase` | Abstract base for molecular interface detectors |
| `StructuralItoBase` | Abstract base for atomic Itô drift correction |
| `LangevinBridge` | Connects `RefinementEngine` ↔ `AdvancedStructuralLangevin` |
| `get_device` | Hardware selector |

---

### `real_fold_one.py` — Universal Full-Atom Differentiable Refinement Engine

End-to-end differentiable protein, RNA, DNA, and nucleic acid refinement. Native autograd via OpenMM-ML + TorchForce.

**Native differentiability via OpenMM-ML:**
```python
# Energy evaluation runs inside PyTorch autograd graph
solute_coords.requires_grad_(True)
E = openmm_solute_energy(solute_coords, calculator)
E.backward()   # analytical dE/dcoords — no force injection
H = torch.autograd.functional.hessian(...)   # works correctly
```

**Supported ML potentials:**

| Potential | Coverage | Notes |
|---|---|---|
| `ani2x` | C, H, N, O, S, F | Fast; good for organic molecules |
| `ani1ccx` | C, H, N, O | Coupled-cluster accuracy |
| `mace-mp-0` | All elements | Foundation model; highest accuracy |
| `aimnet2` | C, H, N, O + halogens | Balanced speed/accuracy |
| `None` | Classical AMBER only | Force-injection fallback |

**Key classes:**

| Class | Description |
|---|---|
| `RamachandranSampler` | Backbone dihedral sampling from empirical Ramachandran distributions |
| `OpenMMSystemBuilder` | Builds OpenMM systems for proteins (ff14SB), DNA/RNA (OL15), ligands (GAFF2), antibodies (MM-GBSA) |
| `_MLTorchEnergyModule` | TorchScript wrapper — ML potential inside autograd graph |
| `OpenMMEnergyCalculator` | Native differentiable energy via TorchForce; graceful fallback to force injection |
| `FastNeighborList` | O(N) neighbour list: torch-cluster → scipy KD-tree → pure PyTorch fallback |
| `CSOCKernel` | Learnable SOC kernel for structural stress modulation |
| `SOCController` | **Inherits `CSOCBase`.** CSOC-driven adaptive relaxation |
| `DiffRGRefiner` | RG coarse-graining with full-atom consistency |
| `WireframeOrigami` | DNA origami: wireframe routing, staple design, all-atom PDB, oxDNA export |
| `RosettaScorer` | Rosetta-style energy scoring interface |
| `CDRLoopModeler` | Antibody CDR loop modelling |
| `MDEngine` | Long-time MD: ps to microsecond timescales |
| `RefinementEngine` | Top-level orchestration: SSC + SOC + RG + MD + validation |
| `Trainer` | Multi-GPU training with AMP, DDP |

**Supported molecule types:** Proteins, DNA, RNA, DNA/RNA hybrids, G-quadruplexes, RNA aptamers, ligands, multimers, antibodies, DNA origami nanostructures

**CLI:**
```bash
python real_fold_one.py refine   -i input.pdb -o refined.pdb --steps 200
python real_fold_one.py refine   -i input.pdb -o refined.pdb --ml-potential mace-mp-0
python real_fold_one.py origami  --shape design.json --output origami/
python real_fold_one.py md       -i input.pdb -o traj/ --steps 100000
python real_fold_one.py validate -i input.pdb --reference ref.pdb
python real_fold_one.py train    -i pdbs/*.pdb --epochs 50
```

---

### `real_fold_one_ht.py` — High-Throughput Mutation & Epistasis Scanner

Ultra-fast scanning of single and double mutations using coarse-grained SOC + residue-type energy model.

**Capabilities:**

- Full single-mutation scan: all positions × all allowed monomers
- Targeted mutation list from JSON or CSV
- Double-mutation epistasis scan: random or user-supplied pairs
- Multi-chain support (auto-detects from PDB)
- Local relaxation window for fast ΔΔG estimation
- Multi-GPU parallel evaluation via `torch.multiprocessing`
- Checkpointing and resume
- Publication-quality plots: ΔΔG distribution, mutational landscape, position tolerance profile, epistasis distribution, additivity scatter
- DNA/RNA stacking and hydrogen-bonded base-pair pseudo-energy
- Supports proteins, DNA, RNA, and multimers

**CLI:**
```bash
python real_fold_one_ht.py --pdb 1abc.pdb --scan --output ht_results/
python real_fold_one_ht.py --pdb 1abc.pdb --mutlist mutations.json
python real_fold_one_ht.py --pdb 1abc.pdb --epistasis --max_epi 2000
python real_fold_one_ht.py --pdb 1abc.pdb --single 0:5:ALA
```

---

### `structural_langevin_fold.py` — BAOAB Langevin for Molecular Dynamics

Fully differentiable BAOAB Langevin integrator for atomic coordinates (N×3). Parallel to `structural_langevin_mental.py` but operating at molecular scale.

**Key classes:**

| Class | Description |
|---|---|
| `InterfaceDetector` | Per-atom interface mask from pairwise distance variance. Differentiable w.r.t. coords (N, 3) |
| `CSOCThermostat` | **Inherits `CSOCBase`.** CSOC adaptive temperature T and friction γ from atomic displacement stress |
| `StructuralItoNoise` | Multiplicative atomic noise G(x) = 1 + amp·mask(x). ½ G ∇G via autograd |
| `AdvancedStructuralLangevin` | Full BAOAB: bulk force + BV jump measures + Itô correction + CSOC thermostat |

---

### `standard_one.py` — Unified Differentiable Framework for Fundamental Physics

Comprehensive differentiable engine for particle physics and cosmology. Covers all four fundamental forces, Standard Model particles, collider simulation, and cosmological observations.

**Key capabilities:**

| Domain | Components |
|---|---|
| **Particle physics** | Full SM particle database; QED, QCD, electroweak matrix elements; NNLO K-factors |
| **PDF evolution** | DGLAP evolution; LHAPDF grids with error sets; neural PDF surrogate |
| **Collider simulation** | Parton shower; hadronisation (Pythia8/Herwig); differentiable fast detector simulation |
| **Collider data** | CERN Open Data (ROOT/awkward); pyhf HistFactory likelihoods |
| **Cosmology** | Planck + NASA FITS/HDF5 ingestion; full CMB pipeline (CAMB, CLASS, CosmoPower, built-in neural emulator) |
| **Compact objects** | Black-hole thermodynamics; dark matter models; vacuum energy and extraction |
| **Cross-scale analysis** | Collider–cosmic cross-correlation; toy unification; running couplings; Randall–Sundrum |
| **Structural probability** | CSOC, SSC, RG flow, BV measures — connecting quantum fields to structural calculus |
| **Statistics** | Bayesian (NUTS MCMC), frequentist (CLs, profile likelihood), structural deterministic probability; AIC/BIC/Bayes factors |

**Key classes:**

| Class | Description |
|---|---|
| `ParticleDB` | Full Standard Model particle database with quantum numbers |
| `PhysicsParameters` | Learnable SM parameters as `nn.Parameter` |
| `DGLAPEvolution` | Differentiable DGLAP PDF evolution |
| `NeuralPDF` | Neural PDF surrogate network |
| `MatrixElements` | QED, QCD, EW matrix elements with loop corrections |
| `Cosmology` | Planck/NASA data ingestion and cosmological parameter fitting |
| `DifferentiableCMB` | End-to-end differentiable CMB power spectrum |
| `CSOCKernel` | SOC kernel for structural probability |
| `SemanticStateContraction` | SSC filter (standalone, for cross-ecosystem use) |
| `BVConsistency` | BV Jump Measure consistency (Paper 2) |
| `ColliderGenerator` | Structural differentiable collider event generator |
| `BlackHoleGenerator` | Hawking radiation and thermodynamic observables |
| `DarkMatterGenerator` | DM annihilation cross-section and direct detection |
| `VacuumEnergyModel` | Casimir effect and vacuum energy density |
| `UnificationModel` | Running coupling unification with RG flow |
| `StandardOneUnified` | Top-level orchestration across all domains |
| `BayesianAnalysis` | Full Bayesian inference with NUTS and posterior predictive checks |
| `FrequentistAnalysis` | CLs limits, profile likelihood, test statistics |
| `StructuralProbability` | CSOC-based structural deterministic probability |
| `CrossCorrelationAnalyzer` | Collider–cosmic data cross-correlation |

---

## Cross-Ecosystem Integration

### REAL FOLD ONE ↔ MENTAL ONE

The two ecosystems connect via the shared Structural Calculus mathematical framework. The physical bridge is the Langevin integrator — the same BAOAB algorithm and CSOC thermostat operate identically at both molecular and neural scales, differing only in the interpretation of coordinates:

| Concept | REAL FOLD ONE | MENTAL ONE |
|---|---|---|
| Coordinates | Atomic positions (N, 3) in Å | Brain-state vector (N,) ∈ [0, 1] |
| Force | −∇E from ML potential | −∇E from SSC energy landscape |
| Stress σ | Mean atomic displacement | Disorder distance from Healthy manifold |
| Temperature T | Thermodynamic temperature (K) | SOC criticality temperature |
| Interface mask | Molecular interface (high ∇-distance variance) | EEG transient / pathological spike |

This means a **nucleic-acid substrate** (simulated by REAL FOLD ONE) and a **neural-state trajectory** (evolved by MENTAL ONE) inhabit the same dynamical formalism. The CSOC universality chain ensures that adaptive parameter modulation is mathematically identical across both scales.

### Toward Orch OR and IIT — Computational Phenomenology

The ONE Ecosystem provides a principled computational substrate for exploring two leading theories of consciousness. These investigations are framed explicitly as **computational phenomenology** — not proofs of consciousness, but quantitative explorations of the physical conditions these theories require.

**Orchestrated Objective Reduction (Orch OR — Penrose & Hameroff)**

Orch OR proposes that consciousness arises from quantum state reduction in tubulin dimers within neuronal microtubules. The relevant computational pathway within the ONE Ecosystem:

- `real_fold_one.py` — simulate tubulin dimer conformation switching (GTP → GDP hydrolysis) with MACE-MP-0 or ANI-2x ML potential at full-atom resolution
- `structural_langevin_fold.py` — drive conformational dynamics via BAOAB Langevin with CSOC-adaptive temperature, Structural Itô correction at the GTP-binding interface
- `langevin_mental_bridge.py` / `mental_one.py` — map tubulin conformational state onto brain-state evolution; the disorder distance σ tracks deviation from a coherent reference configuration
- `standard_one.py` — the RG flow and scaling operators in `DGLAPEvolution` / `DiffRGRefiner` provide a formal bridge between quantum-field–scale processes and classical conformational dynamics

*Important caveat:* The Langevin dynamics here are classical stochastic, not quantum wavefunction collapse. The framework models the classical conformational substrate that Orch OR acts upon, providing a computationally falsifiable proxy that can be compared against classical alternatives.

**Integrated Information Theory (IIT — Tononi)**

IIT proposes that consciousness is identical to integrated information Φ (phi) — a measure of irreducible causal structure. The relevant pathway:

- `mental_one.py` — the `SSCClassifier` energy landscape and `MentalHealthEvolution` trajectories define a dynamical system whose causal structure can be analysed
- The CSOC dynamics in `SOCController` are formally related to critical information integration: near the SOC fixed point, information is maximally transmitted across scales
- `standard_one.py` `StructuralProbability` and `BVConsistency` provide the information-geometric and measure-theoretic tools needed to estimate Φ-proxies (geometric mean information, φ₃ approximation) from brain-state trajectories
- A dedicated Φ-approximation module can be added that operates on the `MentalONEEngine` output to compute Φ_3 or φ_geometry from the evolved state covariance structure

*Important caveat:* Exact Φ computation has exponential complexity in system size. The ONE Ecosystem supports tractable approximations (φ₃, geometric Φ) rather than exact IIT.

### STANDARD ONE as Mathematical Unifier

`standard_one.py` plays a unique role as the cross-scale mathematical unifier. The RG flow apparatus (`DiffRGRefiner`, `DGLAPEvolution`, `UnificationModel`) describes how physical parameters — coupling constants, stress tensors, information density — run with energy/length scale. This provides a formal language for connecting:

- Quantum field fluctuations (TeV scale) → molecular conformational dynamics (Å scale) → neural population dynamics (mm scale) → cognitive behaviour (system scale)

This is the same RG philosophy underlying the CSOC universality chain: different physical systems share the same fixed-point structure when viewed at the appropriate scale.

---

## Toward Carbon-Based Computational Substrates and Next-Generation AI

The ONE Ecosystem's multi-scale differentiable architecture points toward a convergence of biological and computational intelligence that goes beyond conventional silicon-based AI.

**Carbon-based substrate simulation**

REAL FOLD ONE directly simulates the molecular building blocks of biological computation:

- **Proteins** — the machinery of cellular computation; enzyme kinetics, receptor binding, signal transduction
- **Nucleic acids** — DNA (genetic encoding), RNA (messenger and catalytic), G-quadruplexes (regulatory switching), RNA aptamers (selective binding)
- **DNA origami nanostructures** — programmable nanoscale scaffolds for positioning molecular components with sub-nanometre precision
- **Nucleobase stacking and hydrogen-bond energetics** — the physical basis of molecular recognition and information storage

The MACE-MP-0 foundation model covers all elements, including phosphorus and the nucleobases (adenine, thymine, guanine, cytosine, uracil), enabling full-atom simulation of nucleic acid dynamics at ML accuracy.

**Toward Humanoid AI with Biological Substrate**

The convergence pathway suggested by the ONE Ecosystem architecture:

```
STANDARD ONE          → physical laws governing all matter and energy
        ↓
REAL FOLD ONE         → molecular design: proteins, nucleic acids, synthetic biology
        ↓
structural_langevin   → atomic dynamics: conformational changes, binding events
        ↓
MENTAL ONE            → emergent neural computation: state-space, criticality, learning
        ↓
PSY ONE BRIDGE        → behaviour, cognition, decision-making, adaptation
```

This stack does not constitute a complete AGI architecture — the gap between differentiable simulation and genuine cognition remains an open scientific and philosophical problem. However, it provides a rigorous, physically grounded computational substrate in which hypotheses about biological computation can be formulated, tested, and refined quantitatively.

---

## Installation

### Core dependencies
```bash
pip install torch torchvision numpy scipy matplotlib seaborn pandas tqdm
```

### REAL FOLD ONE — molecular simulation
```bash
conda install -c conda-forge openmm openmm-ml openmm-torch
pip install biotite rdkit torch-cluster openff-toolkit openmmforcefields networkx
```

### MENTAL ONE — neural / psychiatric
```bash
pip install mne pyedflib nibabel scikit-learn
```

### STANDARD ONE — particle physics / cosmology
```bash
pip install uproot awkward astropy pyhf pyro-ppl
# Optional: CAMB, CLASS, Pythia8, Herwig (see licence notes)
```

---

## File Structure

```
ONE Ecosystem
│
├── Shared Foundations
│   ├── one_core_mental.py          MENTAL ONE single source of truth
│   └── one_core_fold.py            REAL FOLD ONE single source of truth
│
├── MENTAL ONE Cluster
│   ├── mental_one.py               Psychiatric / neural engine
│   ├── psy_one_bridge_diff.py      Differentiable cognition–behaviour bridge (v2.0-DIFF)
│   ├── structural_langevin_mental.py   BAOAB Langevin for brain states
│   └── langevin_mental_bridge.py   Drop-in MENTAL ONE ↔ Langevin bridge
│
├── REAL FOLD ONE Cluster
│   ├── real_fold_one.py            Full-atom differentiable refinement engine
│   ├── real_fold_one_ht.py         High-throughput mutation & epistasis scanner
│   └── structural_langevin_fold.py BAOAB Langevin for molecular dynamics
│
└── STANDARD ONE
    └── standard_one.py             Particle physics, cosmology, unification
```

---

## Differentiability Guarantees

Every computational path in the ONE Ecosystem satisfies the following:

- No `.item()` calls in gradient-path methods (`soc_evolve`, `sigma`, `baoa_step`, `final_b_step`, `contraction_update`)
- No hard `torch.clamp()` in gradient-path methods — replaced universally by `soft_clamp` (tanh-based, gradient everywhere)
- No `register_buffer(..., None)` — boolean `_initialized` buffers throughout
- No lazy `hasattr` module initialisation — all `nn.Module` submodules created in `__init__`
- `baoa_step()` returns `(positions, velocities, T, sigma)` as tensors — callers invoke `.item()` only at print/logging sites
- `DifferentiableSOC.base_temp` and `.beta` are `nn.Parameter` — trainable end-to-end
- `DifferentiableRG.weight` is `nn.Parameter` — trainable end-to-end

---

## Citation

If you use the ONE Ecosystem in your research, please cite:

```
Yoon A Limsuwan / MSPS NETWORK
ONE Ecosystem: Fully Differentiable Multi-Scale Simulation Framework
ORCID: 0009-0008-2374-0788
GitHub: https://github.com/yoonalimsuwan
MIT License, 2026
```

---

## Licence

MIT License. Copyright 2026 Yoon A Limsuwan / MSPS NETWORK.

External libraries (OpenMM, AMBER force fields, LHAPDF, Pythia8, CLASS, Herwig) retain their own licences. GPL-licensed components are optional; if linked, the combined work must comply with the GPL. To maintain a pure MIT distribution, use the built-in neural PDF surrogate, neural CMB emulator, and structural collider generator in place of GPL-licensed alternatives.

This software is intended exclusively for peaceful civilian scientific applications.
