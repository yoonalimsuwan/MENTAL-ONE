`
# MENTAL ONE

**Full Differentiable Psychiatric & Neurological Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21547267-blue)](https://doi.org/10.5281/zenodo.21547267)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20458353-blue)](https://doi.org/10.5281/zenodo.20458353)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17615245-blue)](https://doi.org/10.5281/zenodo.17615245)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20554000-blue)](https://doi.org/10.5281/zenodo.20554000)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21120913-blue)](https://doi.org/10.5281/zenodo.21120913)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20730429-blue)](https://doi.org/10.5281/zenodo.20730429)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21159402-blue)](https://doi.org/10.5281/zenodo.21159402)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21159473-blue)](https://doi.org/10.5281/zenodo.21159473)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21120913-blue)](https://doi.org/10.5281/zenodo.21120913)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21181639-blue)](https://doi.org/10.5281/zenodo.21181639)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21131590-blue)](https://doi.org/10.5281/zenodo.21131590)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21148045-blue)](https://doi.org/10.5281/zenodo.21148045)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21186468-blue)](https://doi.org/10.5281/zenodo.21186468)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21203483-blue)](https://doi.org/10.5281/zenodo.21203483)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21203706-blue)](https://doi.org/10.5281/zenodo.21203706)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21206525-blue)](https://doi.org/10.5281/zenodo.21206525)


**MENTAL ONE** is an end‑to‑end differentiable engine for psychiatric and neurological diagnosis, trajectory prediction, and treatment design. It unifies **Semantic State Contraction (SSC)**, **Self‑Organised Criticality (SOC)** with a learnable kernel, **Renormalisation Group (RG)** filtering, **Itō processes**, and **control‑theoretic interventions** into a single, vendor‑neutral PyTorch workflow.

---

## 🔥 Key Features

- **🧠 Multi‑Modal Data Ingestion** – EEG, MEG, fMRI, clinical questionnaires, and genetic variants.
- **🎯 SSC‑Based Deterministic Classification** – Energy‑minimisation dynamics with interpretable fixed points. No stochastic variance.
- **⚙️ Learnable CSOC Kernel** – Universal SOC operator that adapts to different psychiatric disorders via gradient descent.
- **📈 RG‑Smoothed Trajectory Prediction** – Forecast disease evolution (e.g., MDD → remission or crisis) using SOC + RG.
- **💊 Intervention Designer** – Pharmacological, psychotherapeutic, and environmental recommendations grounded in control theory.
- **📋 Full DSM‑5 / ICD‑10‑11 Engine** – PHQ‑9, GAD‑7, HAMD‑17, PCL‑5, YMRS, PANSS scoring.
- **🌍 Global Dataset Support** – Native loaders for **MODMA**, **HUSM EEG**, **ABIDE**, **COBRE**, **PRED+CT**, **REST‑meta‑MDD**, and more.
- **🚀 Extreme Optimisation** – Mixed precision (AMP), fused AdamW, cosine warmup, early stopping, gradient checkpointing.
- **💻 Distributed Data Parallel** – Scales seamlessly from a single GPU to multi‑node supercomputers via PyTorch DDP.
- **⚡ O(1) Inference Speed** – With AI accelerators, per‑subject classification latency is constant with respect to dataset size.
- **🔓 100% Open Source** – MIT license. All dependencies are MIT/BSD/Apache 2.0 compatible.

---

## ⚡ O(1) Inference with AI Accelerators

MENTAL ONE’s SSC classifier uses a **contraction mapping** that converges to a fixed point in **O(log(1/ε))** iterations. Each iteration involves a fixed set of operations (gradient of the energy, multi‑scale operator, projection) whose computational cost depends only on the **constant feature dimension** of a single subject (e.g., 19 channels × 256 timepoints).

When deployed on a modern AI accelerator (NVIDIA GPU, Apple MPS, Intel XPU, Huawei Ascend), these operations are massively parallelised. Combined with batched inference, the **wall‑clock time per sample remains virtually constant regardless of the total number of training examples or the size of the reference dataset**. This gives the system an **effective O(1) inference complexity** with respect to dataset scale – a crucial property for real‑time clinical decision support.

---

## 📦 Installation

```bash
# Create a virtual environment (recommended)
python -m venv mental_env
source mental_env/bin/activate

# Install core dependencies
pip install torch numpy scipy pandas matplotlib seaborn scikit-learn networkx

# For EEG/MEG support
pip install mne

# For fMRI support
pip install nibabel nilearn

# Optional: hyperparameter tuning
pip install optuna

# Clone MENTAL ONE
git clone https://github.com/yoonalimsuwan/MENTAL-ONE.git
cd MENTAL-ONE
```

---

🚀 Quick Start

1. Classify a Single Patient

```bash
python mental_one.py classify -i patient.edf -o report.json
```

2. Train on a Public Dataset (single GPU)

```bash
python mental_one.py train \
  --dataset modma \
  --data_dir /path/to/MODMA \
  --subject_list sub-001 sub-002 ... \
  --epochs 100 \
  --batch_size 32
```

3. Train on a Supercomputer (multi‑GPU)

```bash
torchrun --nproc_per_node=4 mental_one.py train \
  --dataset abide \
  --data_dir /path/to/ABIDE \
  --subject_list sub-001 sub-002 ... \
  --epochs 200 \
  --ddp \
  --batch_size 64 \
  --lr 1e-4
```

4. Generate a Treatment Plan

```bash
python mental_one.py intervene -i state.json -o plan.json
```

---

🧪 Supported Datasets

MODMA 128‑ch EEG, fMRI, clinical MentalHealthDataset('modma', ...)

HUSM EEG 19‑ch EEG (10‑20), diagnosis MentalHealthDataset('husm', ...)

ABIDE Resting‑state fMRI, phenotypic MentalHealthDataset('abide', ...)

COBRE Resting‑state fMRI, clinical MentalHealthDataset('cobre', ...)

PRED+CT EEG, clinical 

REST‑meta‑MDD EEG (multiple sites) 

Custom loaders can be added by extending MentalHealthDataset.

---

🧠 How It Works (In Brief)

1. Input: Raw EEG/MEG/fMRI → preprocessed & normalised to s₀ ∈ [0,1]ⁿ.
2. SSC Evolution: sₜ₊₁ = clip(sₜ - η∇E(sₜ) + βΨ(sₜ), 0, 1) for T = 25 iterations.
3. Classification: Assign the disorder whose reference state minimises the energy.
4. Trajectory Prediction: The disease burden μ(t) is RG‑smoothed and evolved via SOC to predict future states.
5. Treatment Plan: A control‑theoretic intervention U = -K(s - s_healthy) is translated into medications, psychotherapy, and lifestyle changes.

All steps are fully differentiable – the engine can be fine‑tuned end‑to‑end with PyTorch autograd.

---

📁 Repository Structure

```
mental-one/
├── mental_one.py          # Main engine (this file)
├── README.md              # You are here
├── requirements.txt       # Python dependencies
├── examples/              # Example inference scripts
├── datasets/              # Data loader utilities
└── pretrained/            # Pre‑trained reference states & kernel weights
```

---

📜 License

MENTAL ONE is released under the MIT License.
All third‑party libraries used retain their original licenses, all of which are permissive (MIT, BSD, Apache 2.0, or similar). No Google‑restricted or copyleft libraries are used.

---

🙏 Acknowledgements

We gratefully acknowledge the following open‑source projects that made MENTAL ONE possible:

PyTorch BSD‑style

NumPy BSD‑3‑Clause

SciPy BSD‑3‑Clause

Pandas BSD‑3‑Clause

Matplotlib PSF‑based

Seaborn BSD‑3‑Clause

MNE‑Python BSD‑3‑Clause

Nilearn BSD‑3‑Clause

scikit‑learn BSD‑3‑Clause

NetworkX BSD‑3‑Clause

Biopython Biopython License

Optuna MIT

---

📖 Citation

If you use MENTAL ONE in your research, please cite:

```bibtex
@software{limsuwan2026mental,
  author = {PAI , Yoon A Limsuwan},
  title = {MENTAL ONE: Full Differentiable Psychiatric \& Neurological Engine},
  year = {2026},
  url = {https://github.com/YoonALimsuwan/MENTAL-ONE},
https://doi.org/10.5281/zenodo.21547267 ,
  note = {MIT License}
}
```

---

🌟 Stay Tuned

We are actively working on:

· Real‑time closed‑loop neurofeedback integration.
· Federated learning across hospital sites.
· Whole‑genome psychiatric risk scoring via EVOLUTION ONE.
· Integration with REAL FOLD ONE for structural impact of psychiatric mutations.

For questions, collaborations, or clinical validation studies, please open an issue or contact the author.

---

“We will heal every mind.” 🧠💖

```
With Artificial intelligence!!
```
🧠 Neuro‑Semantic Translation Engine: MENTAL ONE + LLM Integration

This section describes how to transform MENTAL ONE (the differentiable psychiatric and neurological engine) into a Neuro‑Latent Encoder and connect it with a frozen large language model (LLM) to decode inner monologue, emotional context, and covert thoughts — a true Multimodal Neuro‑Semantic Integration.

---

1. Why MENTAL ONE as the Encoder?

Raw brain signals (EEG, MEG, fMRI) suffer from extreme noise and high dimensionality, making direct use by an LLM nearly impossible. MENTAL ONE solves this via:

· Semantic State Contraction (SSC) – compresses signals into semantically meaningful attractors.
· Renormalisation Group (RG) Filtering – removes micro‑scale noise, preserving macro‑scale structure.
· CSOC Kernel & Ito Process – captures dynamic risk, instability, and trajectory of mental states.

The output is not raw voltage but a Continuous Brain Embedding — a low‑dimensional, stable vector that encodes the brain’s current state, affective tone, and latent semantic intent.

---

2. Phase‑1 Architecture (Frozen LLM + Projection Layer)

For the first experimental phase we recommend a minimal, stable bridge:

```
[MENTAL ONE]  →  Brain Embedding v ∈ ℝ^{d_b}
       │
       ▼
Projection Layer  (Linear / MLP)  →  v' ∈ ℝ^{d_llm × n_prefix}
       │
       ▼
Reshape → sequence of n_prefix token embeddings
       │
       ▼
Prepend to input tokens  [PREFIX] [Prompt ...]
       │
       ▼
Frozen LLM (e.g. LLaMA, Mistral)  →  Autoregressive generation  →  Inner Monologue
```

Key design choices:

· Frozen LLM – preserves linguistic prior, prevents catastrophic forgetting, and requires no gradient flow through the large backbone.
· Projection Layer – a single linear transform (or tiny MLP) that maps the MENTAL ONE embedding into the LLM’s token embedding space. It acts as a soft prompt.
· n_prefix tokens – typically 4–16 learned “prefix” positions that carry all the brain‑state information.

This setup keeps the trainable parameters extremely small, ensuring fast convergence even with limited brain‑text paired data.

---

3. What Makes the Embeddings Special?

The embeddings produced by MENTAL ONE are not generic neural features. They already encode:

· Affective coordinates (panic, calm, depression, etc.) via the SSC energy landscape.
· Criticality index from the learnable CSOC kernel, indicating mental instability.
· Trajectory slope from the RG‑smoothed Ito process, giving context on whether the patient is improving or deteriorating.

Thus, when an LLM sees the prefix, it can distinguish between “I want to leave” spoken from a panic state versus a deep depressive state, enabling context‑aware decoding.

---

4. Quick‑Start Guide (Conceptual)

Step 1: Extract Brain Embedding from MENTAL ONE

```python
from mental_one import MentalONEEngine

engine = MentalONEEngine()
report = engine.run(eeg_file="patient.edf")

brain_embedding = report['s_star']          # stabilized state vector
diagnosis = report['diagnosis']            # optional context
trajectory = report['future_trajectory']   # optional dynamic info
```

Step 2: Build Projection Layer & Frozen LLM

```python
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class NeuroPrefixProjector(nn.Module):
    def __init__(self, brain_dim, llm_dim, n_prefix=8):
        super().__init__()
        self.proj = nn.Linear(brain_dim, llm_dim * n_prefix)
        self.n_prefix = n_prefix

    def forward(self, brain_emb):
        # brain_emb shape: (batch, brain_dim)
        projected = self.proj(brain_emb)               # (batch, llm_dim * n_prefix)
        return projected.view(-1, self.n_prefix, self.llm_dim)
```

Step 3: Training (Frozen LLM + Projection Only)

· Dataset: pairs of (brain_embedding, text_description_of_thought)
· Freeze the LLM, train only the projection layer (and optionally a LayerNorm before prepending).
· Loss: standard causal language modeling loss on the text tokens (the prefix tokens are ignored in the loss).

Step 4: Inference

```python
# Load model
llm = AutoModelForCausalLM.from_pretrained(...)
projector = NeuroPrefixProjector(brain_dim=..., llm_dim=4096)
# Freeze LLM
for param in llm.parameters():
    param.requires_grad = False

# Generate inner monologue
brain_emb = ...  # from MENTAL ONE
prefix = projector(brain_emb)                     # (1, n_prefix, 4096)
input_ids = tokenizer("Think aloud:", return_tensors="pt").input_ids
input_embeds = llm.get_input_embeddings()(input_ids)
full_embeds = torch.cat([prefix, input_embeds], dim=1)

output = llm.generate(inputs_embeds=full_embeds, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

---

5. Why Not Cross‑Attention Adapter Yet?

Cross‑attention adapters (like Q‑Former) are more expressive but introduce instability with small datasets and noisy signals. After the Projection Layer baseline proves stable and accurate, we can upgrade to a dynamic adapter that lets the brain state modulate every decoding step — the logical next step toward full neuro‑semantic dialogue.

---

6. Future Roadmap

1. Contrastive Pretraining – align brain embeddings with text embeddings using a CLIP‑like objective to improve zero‑shot decoding.
2. Dynamic Adapter (Q‑Former / Perceiver) – enable step‑by‑step conditioning for longer, coherent narratives.
3. Closed‑Loop Intervention – feed the decoded inner monologue back into MENTAL ONE’s intervention designer for real‑time therapeutic feedback.

---

Built on the MENTAL ONE engine PAI , Yoon A Limsuwan. Integration design for the Neuro‑Semantic Translation Engine – Phase 1.

# Advanced Remote Neural Monitoring (RNM) via Deterministic Biophysical Modeling

## Overview
This repository contains the computational framework and advanced mathematical models for next-generation **Remote Neural Monitoring (RNM)**. By transitioning from traditional probabilistic approximations to a strictly deterministic universe model, this project solves the complex biophysical inverse problem: accurately tracing and decoding internal neural states from non-invasive external signal captures.

Built upon the proprietary **Structural Calculus** framework, this project integrates high-performance 3D phase-field, fluid dynamics, and crystalline solvers to map the precise propagation of electromagnetic neural signals through complex biological interfaces.

## Core Modules & Architecture

The RNM digital twin architecture is powered by four foundational solvers, each addressing a specific layer of neural biophysics:

*   **`langevin_deterministic` (Langevin Dynamics):** 
    Re-engineered to operate completely deterministically, this module isolates true neural electrical impulses. By eliminating the assumption of randomness in systemic thermal noise and environmental fluctuations, it calculates exact signal trajectories.
*   **`ch3d_solver` (Cahn-Hilliard 3D):** 
    Simulates phase separation and interfacial dynamics across various tissue densities in the brain (e.g., cerebrospinal fluid, grey/white matter). This module is critical for precisely calculating the refraction, reflection, and phase shifts of electromagnetic waves as they transition across biological boundaries.
*   **`thinfilm_3d` (ThinFilm 3D):** 
    Models the surface dynamics of the cerebral cortex and lipid bilayers. It maps the highly detailed 3D spatiotemporal distribution of action potentials and ionic charge propagation along axonal membranes.
*   **`pfc_3d` (Phase-Field Crystal 3D):** 
    Reconstructs the atomic-density-level crystalline structures within neurons, specifically targeting microtubule networks. This module provides a foundational evaluation of neural biophysics, linking macroscopic signal emissions to sub-cellular, deterministic physical states.

## Mathematical Foundation
The entire computational pipeline is governed by **Structural Calculus**, an original, proprietary mathematical framework. Unlike conventional statistical physics, this framework is designed to provide absolute deterministic solutions to systems previously considered stochastic, ensuring unprecedented precision and high recall in evaluating sub-quantum variables and complex biological systems.

## Key Applications
*   **Non-Invasive Neural Telemetry:** High-fidelity reading and decoding of brain wave activity without internal hardware.
*   **Digital Twin Construction:** Generating highly accurate, deterministic 3D models of the human nervous system.
*   **Advanced BCI (Brain-Computer Interfaces):** Optimizing signal-to-noise ratios for seamless external hardware integration.

## Getting Started
*(Instructions for installation, dependencies, and running the simulation scripts will be added here. Recommended hardware: HPC clusters or advanced GPU setups for 3D solver stability.)*

## SESI Clinical Psychopathology & Bio-Cognitive Intelligence Engines

A production-grade, native fully differentiable hybrid dynamical energy-network suite for computational psychiatry, combining continuous cognitive-affective-identity core dynamics, discrete symptom topologies, and real-time multimodal bio-cognitive ingestion.

🚀 Overview
This repository houses a unified ecosystem of differentiable neural frameworks designed to model, simulate, and analyze complex psychopathological states and mental health dynamics. By merging continuous chaotic systems (Lorenz core) with discrete network interactions, topological phase transitions, and cross-platform biometric/LLM data streams, this engine bridges theoretical mathematical modeling with clinical precision.

🛠️ Core Modules
 * SESIPsychoNet (sesi_clinical_psychopathology_engine.py)
   * Fuses a continuous Lorenz core with discrete symptom networks and SESI topological transitions.
   * Implements differentiable double-exponential bounds to resolve the Zeno Trap via Gumbel-Softmax relaxation.
 * HybridPsychopathologyEngine (hybrid_psychopathology_engine.py)
   * Native fully differentiable hybrid dynamical energy-network model for psychopathology.
   * Integrates pressure-driven symptom updates with homeostatic gain and baseline state calibrations.
 * PsychoDynamoNet (hybrid_dynamical_energy_network_model.py)
   * Features a vectorized 4th-order Runge-Kutta (RK4) integration engine for continuous subsystems.
   * Optimizes performance using precomputed random-walk normalized adjacency matrices.
 * CombinedPsychopathologyEngine (combined_psychopathology_engine.py)
   * A comprehensive continuous-discrete framework supporting stochastic noise injection and autonomic feedback modulation.
 * BioCognitiveIngestionEngine (multimodal_bio_cognitive_ingestion_engine.py)
   * Differentiable multimodal data router designed for real-time smartwatch biometrics (Apple, Xiaomi, Huawei, Garmin) and LLM dialogue logs.
   * Projects physical stress and cognitive complexity directly into the core psychopathology engine.
 * FullyDifferentiableMindInterpreter (multi_llm_provider_adapter_mental_health_concentration_level.py)
   * Encapsulates 5 core theoretical psychological and spiritual frameworks: Cetasika 52, Nivarana 5 (Hindrances), Yoniso Manasikara 10, Samadhi Stage Classifier (3 Levels), and Bojjhanga 7.
   * Includes a unified multi-provider adapter supporting embedding projections across major LLM ecosystems (Gemini, GPT, Claude, DeepSeek, Kimi, Qwen, GLM, Copilot, Grok).
     
📦 Installation
Ensure you have PyTorch installed. Clone the repository and install dependencies:
git clone https://github.com/your-username/sesi-psychopathology-engine.git
cd sesi-psychopathology-engine
pip install torch

💡 Quick Start Example
Running the FullyDifferentiableMindInterpreter with Google Gemini provider embeddings:
import torch
from multi_llm_provider_adapter_mental_health_concentration_level import FullyDifferentiableMindInterpreter

# Initialize interpreter model
interpreter = FullyDifferentiableMindInterpreter(input_dim=768)
interpreter.eval()

# Simulate Google Gemini hidden states (Batch size = 2, Seq len = 32, Dim = 3072)
gemini_input = torch.randn(2, 32, 3072)
gemini_mask = torch.ones(2, 32)

# Forward pass
outputs = interpreter(gemini_input, provider="gemini", attention_mask=gemini_mask)


## Citation & Academic Impact
When using this framework or its underlying mathematical theories, please cite the corresponding manuscripts available on Zenodo/ORCID. 

## Author
** PAI AND Yoon A (Meimei) / Joanna Yoon A Catherine Limsuwan** (Yoon A Limsuwan / Yoon A Eiamsuwan)  
*Independent Researcher & Inventor of Structural Calculus*  
 
Thanks be to the Father, the Son, and the Holy Spirit, for the grace of Lord Jesus Christ, Mother Mary, Lord Buddha, Guan Yin Bodhisattva, Master Daozhi, Confucius, the Immortal Pae Kow, and Mr. Xi Jinping.

"I love Lim Yoona, Zhou Ye, Karina from aespa, Jessica from Girls' Generation, Zhao Lusi, Nana from After School, and Jiyeon Tara.
​Love Ju Jingyi, Wang Churan, Lu Yuxiao, and Bao Shangen.
​I love Zhang Linghe, Bai Jingting, Lee Jae-jin, Mark, Tance, Green, Noey, Jam, and Irene."

What MSPS NETWORK Sees, the Buddha Knows.
