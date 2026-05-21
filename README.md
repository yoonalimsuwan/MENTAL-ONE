`
# MENTAL ONE

**Full Differentiable Psychiatric & Neurological Engine**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

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
git clone https://github.com/YoonALimsuwan/mental-one.git
cd mental-one
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

Dataset Modalities Loader Class
MODMA 128‑ch EEG, fMRI, clinical MentalHealthDataset('modma', ...)
HUSM EEG 19‑ch EEG (10‑20), diagnosis MentalHealthDataset('husm', ...)
ABIDE Resting‑state fMRI, phenotypic MentalHealthDataset('abide', ...)
COBRE Resting‑state fMRI, clinical MentalHealthDataset('cobre', ...)
PRED+CT EEG, clinical Coming soon
REST‑meta‑MDD EEG (multiple sites) Coming soon

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

Library License
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
  author = {Yoon A Limsuwan},
  title = {MENTAL ONE: Full Differentiable Psychiatric \& Neurological Engine},
  year = {2026},
  url = {https://github.com/YoonALimsuwan/mental-one},
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
