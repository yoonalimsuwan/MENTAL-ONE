# PSY ONE BRIDGE

**Informational Psyche Engine — Connecting Psychoanalytic Theory to MENTAL ONE**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MENTAL ONE](https://img.shields.io/badge/integrates-MENTAL%20ONE-8A2BE2)](https://github.com/YoonALimsuwan/MENTAL-ONE)

**Developer:** Yoon A Limsuwan / MSPS NETWORK — *MY SOUL MOVE BY POWER OF HOLY SPIRIT*

**ORCID:** [0009-0008-2374-0788](https://orcid.org/0009-0008-2374-0788)

**GitHub:** [github.com/yoonalimsuwan](https://github.com/yoonalimsuwan)

**License:** MIT

**Year:** 2026

---

## Overview

**PSY ONE BRIDGE** is a production-grade Python module that connects the information-theoretic psychoanalytic framework — the *Informational Mechanics of the Id, Ego, and Superego* — to the **MENTAL ONE** psychiatric and neurological engine.

It translates Freud's tripartite psychic model into a rigorous computational architecture: the Id becomes a Shannon entropy state space, the Superego becomes a KL-Divergence constraint matrix, and the Ego becomes a Variational Free Energy optimizer — all differentiable, all grounded in neuroscience.

PSY ONE BRIDGE runs in two modes:

- **Integrated mode** — paired with MENTAL ONE for full clinical-grade psychiatric diagnosis, trajectory prediction, and intervention design.
- **Standalone mode** — operates independently for research simulation, psychopathology modelling, and algorithm development.

---

## Theoretical Basis

PSY ONE BRIDGE implements the mathematical framework from:

> *"The Informational Mechanics of the Id, Ego, and Superego: Integrating Psychoanalysis, Neuroscience, and Information Theory to Explain Human Decision-Making"*
> — Yoon A Limsuwan (2026)

The core postulate: **the human psyche is an information processing system**. The three structural components are redefined as:

| Freudian Construct | Informational Reinterpretation | PSY ONE Implementation |
|---|---|---|
| **Id (𝓘)** | High-entropy generative drive state space | `IdModule` — Shannon entropy + temporal accumulation |
| **Ego (𝓔)** | Free Energy minimization optimizer | `EgoModule` — Active Inference loop |
| **Superego (𝓢)** | Normative prior constraint matrix | `SuperegoModule` — KL Divergence penalty |
| **Ψ(t)** | Total psychic state `⟨𝓘, 𝓔, 𝓢⟩` | `PsycheTriad` — full cycle orchestrator |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Psychoanalytic Theory        PSY ONE BRIDGE          MENTAL ONE         │
│  ─────────────────────        ─────────────           ──────────         │
│  Id  (𝓘) — raw drives   ←→   IdModule           ←→  SOCController       │
│  Ego (𝓔) — optimizer    ←→   EgoModule           ←→  SSCClassifier       │
│  Superego (𝓢) — norms   ←→   SuperegoModule      ←→  DiffRGRefiner       │
│  Psyche Ψ(t) = ⟨𝓘,𝓔,𝓢⟩  ←→   PsycheTriad         ←→  MentalONEEngine     │
└─────────────────────────────────────────────────────────────────────────┘
                                      ↕
                         PSYONEBridge (main interface)
                                      ↕
                         LongitudinalPsycheTracker
                                      ↕
                           PSYONEBenchmark (§5.4)
```

**MENTAL ONE mapping in detail:**

| PSY ONE Module | MENTAL ONE Component | Mapping Logic |
|---|---|---|
| `IdModule.compute_entropy()` | `SOCController.sigma()` | SOC structural stress σ ↔ H(𝓘) |
| `IdModule.csoc_kernel` | `CSOCKernel` | Criticality weighting ↔ emotional salience |
| `EgoModule.optimize_action()` | `SSCClassifier.contraction_update()` | Fixed-point iteration ↔ Free Energy loop |
| `SuperegoModule.set_societal_baseline()` | `DiffRGRefiner` | RG smoothing ↔ normative filtering |
| `PSYONEBridge._run_mental_one_diagnosis()` | `MentalONEEngine.run()` | Full psychiatric classification |

---

## Key Mathematical Constructs

**Id Entropy** — measures drive unpredictability:
```
H(𝓘) = −∑ᵢ P(xᵢ) log₂ P(xᵢ)
```

**Id Temporal Accumulation** — experiential history integration:
```
𝓘(t) = 𝓘(0) + ∫₀ᵗ w(τ) · H(x(τ)) dτ
```

**Superego Normative Penalty** — KL Divergence from societal baseline:
```
L_𝓢(π) = D_KL( π(a|s) ∥ π_norm(a|s) )
```

**Ego Variational Free Energy** — the quantity the Ego minimizes:
```
ℱ = D_KL( Q(ϕ|μ) ∥ P(ϕ|m) ) − 𝔼_Q[ log P(μ|ϕ,m) ]
```

**Ego Optimal Action** — balancing drives against constraints:
```
a* = argmin_a [ ℱ(a) + λ · L_𝓢(a) ]
```

**Full Psychic State:**
```
Ψ(t) = ⟨ 𝓘(t), 𝓔(t), 𝓢(t) ⟩
```

---

## Psychopathology Simulation Modes

Based on Section 5.3 of the theoretical paper. Each mode applies precise algorithmic distortions that mirror clinical profiles:

| Mode | Algorithmic Distortion | Clinical Correlate |
|---|---|---|
| `HEALTHY` | Balanced λ, low H(𝓘), converged ℱ | Normal cognition |
| `MDD_ANXIETY` | λ → 50 (hyper-regularization), α → 0.005 | Excessive self-criticism, rigid π_norm |
| `SCHIZOPHRENIA` | λ → 0.01, α → 0 (Ego failure) | H(𝓘) → H_max, PLV F3-F4 collapse |
| `OCD` | n_iter → 500, λ → 15 (non-convergent ℱ) | Compulsive optimization loop |
| `BIPOLAR` | λ oscillates between 0.2 and 40.0 | Alternating over/under-regulation |
| `PTSD` | emotional_salience_scale → 8.0 | Hyper-salience on trauma vectors |
| `CUSTOM` | User-defined parameters | Research / clinical customization |

---

## Neurophysiological Proxy Map

PSY ONE BRIDGE maps information-theoretic parameters to observable neural biomarkers (§5.2):

| Parameter | Neurophysiological Proxy | Measurement |
|---|---|---|
| H(𝓘) Id Entropy | EEG Lempel-Ziv Complexity | Resting-state EEG |
| λ Superego weight | dlPFC–ACC functional connectivity | fMRI |
| α Ego learning rate | Prefrontal white matter integrity | DTI / neuroplasticity |
| L_𝓢 Superego loss | Global Field Power (GFP) | EEG during norm-violation tasks |
| Convergence speed | PLV F3-F4 phase synchrony | EEG (frontal coherence) |

---

## Installation

```bash
# Install dependencies
pip install torch numpy scipy

# Place in your project directory
cp psy_one_bridge.py your_project/

# For full integration, MENTAL ONE must also be importable
cp mental_one.py your_project/
```

**Requirements:**

| Package | Version | License |
|---|---|---|
| Python | ≥ 3.9 | PSF |
| PyTorch | ≥ 2.0 | BSD-style |
| NumPy | ≥ 1.24 | BSD-3-Clause |
| SciPy | ≥ 1.10 | BSD-3-Clause |
| MENTAL ONE | 2026 | MIT (optional) |

---

## Quick Start

### 1. Standalone — Single Inference Cycle

```python
import torch
import torch.nn.functional as F
from psy_one_bridge import PSYONEBridge, PsycheConfig, PsychopathologyMode

config = PsycheConfig(
    action_dim = 10,
    lambda_reg = 2.5,
    mode       = PsychopathologyMode.HEALTHY,
    verbose    = True,
)
bridge = PSYONEBridge(config=config)

# Set societal baseline (e.g. focused on action index 5)
norm = torch.zeros(10)
norm[5] = 3.0
bridge.triad.set_societal_baseline(F.softmax(norm, dim=0))

# Run one psyche cycle
eeg      = torch.randn(19, 256)       # 19-channel EEG, 256 timepoints
salience = torch.ones(10)             # uniform emotional salience

result = bridge.run_psyche_cycle(eeg, salience)
print(bridge.generate_psychopathology_report(result))
```

### 2. Integrated Mode — With MENTAL ONE

```python
from mental_one import MentalONEEngine
from psy_one_bridge import PSYONEBridge, PsycheConfig

engine = MentalONEEngine()
bridge = PSYONEBridge.from_mental_one(engine, config=PsycheConfig(action_dim=10))

result = bridge.run_psyche_cycle(eeg_tensor, salience_vector)

print(f"Diagnosis      : {result.diagnosis}")
print(f"Id Entropy     : {result.id_entropy:.4f} bits")
print(f"Superego Loss  : {result.superego_loss:.4f}")
print(f"Free Energy    : {result.free_energy:.4f}")
print(f"Selected Action: {result.selected_action}")
```

### 3. Simulating Psychopathology

```python
from psy_one_bridge import PSYONEBridge, PsycheConfig, PsychopathologyMode

for mode in [
    PsychopathologyMode.MDD_ANXIETY,
    PsychopathologyMode.SCHIZOPHRENIA,
    PsychopathologyMode.OCD,
]:
    bridge = PSYONEBridge(PsycheConfig(action_dim=10, mode=mode))
    result = bridge.run_psyche_cycle(torch.randn(19, 256))
    print(f"{mode.value:<16} H(I)={result.id_entropy:.3f}  "
          f"OCD={result.ocd_loop_detected}")
```

### 4. Longitudinal Tracking

```python
from psy_one_bridge import PSYONEBridge, PsycheConfig, LongitudinalPsycheTracker

bridge  = PSYONEBridge(PsycheConfig(action_dim=10))
tracker = LongitudinalPsycheTracker(bridge)

for session in range(10):
    eeg   = torch.randn(19, 256)
    state = tracker.run_and_record(eeg)

summary = tracker.summarize()
print(f"Mean H(I)          : {summary['mean_id_entropy']:.4f}")
print(f"Decompensation flag: {summary['decompensation_flag']}")
print(f"Final diagnosis    : {summary['final_diagnosis']}")
```

### 5. Benchmark Suite

```python
from psy_one_bridge import PSYONEBenchmark

bench   = PSYONEBenchmark(action_dim=10, n_subjects=50)
results = bench.run()

# Target accuracy: 78–85% (paper §5.4)
print(f"Overall accuracy: {results['__overall__']['mean_accuracy'] * 100:.1f}%")
```

### 6. Command-Line Interface

```bash
# Simulate a healthy psyche cycle
python psy_one_bridge.py simulate --mode healthy --n_cycles 1

# Simulate MDD profile with longitudinal tracking
python psy_one_bridge.py simulate --mode mdd_anxiety --n_cycles 10 --verbose

# Run full benchmark
python psy_one_bridge.py benchmark --action_dim 10 --n_subjects 50
```

---

## Module Reference

### `PsycheConfig`
Full configuration dataclass. Call `.apply_mode()` to automatically set disorder-specific parameters.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `action_dim` | int | 10 | Decision space dimensionality |
| `lambda_reg` | float | 2.5 | Superego penalty weight λ |
| `alpha_lr` | float | 0.05 | Ego optimization learning rate α |
| `n_ego_iter` | int | 50 | Iterations per Ego optimization loop |
| `history_window` | int | 100 | Id temporal accumulation buffer size |
| `emotional_salience_scale` | float | 1.0 | Global salience multiplier |
| `mode` | PsychopathologyMode | HEALTHY | Psychopathology simulation preset |

---

### `IdModule`
Models Id (𝓘) as a high-entropy generative state space.

| Method | Description |
|---|---|
| `update_drive_states(sensory, salience)` | Integrate new experience into drive distribution |
| `compute_entropy()` | H(𝓘) — Shannon Entropy of current drives |
| `compute_temporal_entropy()` | Entropy over full history buffer |
| `generate_proposals()` | Raw unconstrained action distribution |
| `reset()` | Clear all drive state and history |

---

### `SuperegoModule`
Models Superego (𝓢) as a normative prior constraint matrix.

| Method | Description |
|---|---|
| `set_societal_baseline(dist, smooth)` | Set π_norm with optional RG smoothing |
| `evaluate_policy_divergence(policy)` | L_𝓢 = D_KL(π ∥ π_norm) |
| `register_error(predicted, actual)` | Track ε_𝓢 for behavioral deviation analysis |
| `behavioral_entropy(n_decisions)` | H(B) ∝ ε_𝓢 — behavioral disorder metric |

---

### `EgoModule`
Models Ego (𝓔) as an Active Inference optimizer.

| Method | Description |
|---|---|
| `optimize_action(id_proposal, superego, λ)` | Minimize ℱ + λ·L_𝓢 iteratively |
| `detect_ocd_loop()` | Return True if ℱ fails to converge (OCD marker) |
| `convergence_speed()` | Neuroplasticity proxy α |

---

### `PSYONEBridge`
Main integration class.

| Method | Description |
|---|---|
| `from_mental_one(engine, config)` | Factory: construct with MentalONEEngine |
| `run_psyche_cycle(eeg, salience, ...)` | Full Ψ(t) inference cycle |
| `batch_run(eeg_list, salience_list)` | Multi-subject batch processing |
| `generate_psychopathology_report(state)` | Human-readable clinical report |
| `reset()` | Reset full bridge state |

---

### `PsycheTriadState`
Output dataclass returned by every inference cycle.

| Field | Type | Description |
|---|---|---|
| `id_entropy` | float | H(𝓘) — current drive entropy |
| `accumulated_entropy` | float | ∫w(τ)H(x)dτ — temporal Id history |
| `superego_loss` | float | L_𝓢 — normative penalty |
| `behavioral_entropy` | float | H(B) — behavioral deviation |
| `free_energy` | float | ℱ — Ego residual (final iteration) |
| `selected_action` | int | a* — final decision output |
| `optimized_policy` | ndarray | π*(a\|s) — full action distribution |
| `diagnosis` | str | MENTAL ONE psychiatric classification |
| `intervention_plan` | dict | MENTAL ONE treatment recommendations |
| `ocd_loop_detected` | bool | OCD compulsive loop marker |
| `convergence_speed` | float | Neuroplasticity proxy |
| `neurophysio_map` | dict | FAA, GFP, LZ, PLV, neuroplasticity proxies |

---

## Repository Structure

```
psy-one-bridge/
├── psy_one_bridge.py          # Main engine (this module)
├── README_PSY_ONE_BRIDGE.md   # This file
├── mental_one.py              # MENTAL ONE engine (required for integration)
└── examples/
    ├── simulate_healthy.py
    ├── simulate_mdd.py
    ├── longitudinal_track.py
    └── run_benchmark.py
```

---

## Disclaimer

PSY ONE BRIDGE is a **research and computational modelling tool**. It is not a certified medical device and is not intended for autonomous clinical diagnosis. All outputs must be reviewed and validated by qualified mental health professionals. The informational model has not yet been validated in prospective clinical trials. Future validation against real EEG/fMRI datasets is required before clinical deployment.

---

## Citation

If you use PSY ONE BRIDGE in your research, please cite:

```bibtex
@software{limsuwan2026psyone,
  author    = {Yoon A Limsuwan},
  title     = {PSY ONE BRIDGE: Informational Psyche Engine for MENTAL ONE},
  year      = {2026},
  url       = {https://github.com/YoonALimsuwan/psy-one-bridge},
  note      = {MIT License}
}

@article{limsuwan2026informational,
  author    = {Yoon A Limsuwan},
  title     = {The Informational Mechanics of the Id, Ego, and Superego:
               Integrating Psychoanalysis, Neuroscience, and Information
               Theory to Explain Human Decision-Making},
  year      = {2026},
  note      = {Independent Research, MSPS NETWORK}
}
```

---

## Acknowledgements

PSY ONE BRIDGE builds on the following open-source foundations:

| Library | License |
|---|---|
| PyTorch | BSD-style |
| NumPy | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| MENTAL ONE | MIT |

Theoretical foundations draw on the work of Karl Friston (Free Energy Principle), Claude Shannon (Information Theory), Sigmund Freud (Tripartite Model), and the neuro-psychoanalytic tradition of Fonagy, Kandel, and Carhart-Harris.

---

## License

MIT License — Copyright (c) 2026 Yoon A Limsuwan / MSPS NETWORK

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

*"We will heal every mind."* 🧠💖

