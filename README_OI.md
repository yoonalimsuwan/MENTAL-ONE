# Organoid Intelligence (OI)

**A Research Framework for Biological-Digital Intelligence**

[![Status](https://img.shields.io/badge/status-research-blue)]()
[![Domain](https://img.shields.io/badge/domain-organoid%20intelligence-purple)]()
[![AI](https://img.shields.io/badge/AI-hybrid%20intelligence-green)]()

---

## Abstract

**Organoid Intelligence (OI)** is a research framework for investigating biological neural networks as computational substrates for learning, adaptation, memory, and intelligent behavior.

The project explores the integration of neural organoids, cultured neuronal networks, high-density neural interfaces, machine learning, neuromorphic computing, and conventional digital computation.

The long-term hypothesis is that sufficiently organized biological neural systems could become components of scalable **biological-digital intelligence architectures**.

OI does **not** currently claim to constitute Artificial General Intelligence (AGI). Instead, the project investigates whether biological neural computation can provide capabilities complementary to conventional artificial intelligence.

---

# 1. Research Objectives

The project has six primary objectives:

1. Develop computational models of organoid-based neural systems.
2. Establish interfaces between biological neural networks and digital computers.
3. Study learning and adaptation in biological neural systems.
4. Develop scalable distributed OI architectures.
5. Benchmark biological computation against conventional AI and neuromorphic systems.
6. Investigate whether OI can contribute to future general-intelligence architectures.

---

# 2. System Architecture

The proposed OI architecture consists of five major layers.

```text
┌───────────────────────────────────────────────┐
│                 APPLICATION LAYER             │
│      Games • Robotics • Science • Agents      │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│             INTELLIGENCE LAYER                │
│     Reasoning • Planning • Learning           │
│     Memory • Representation • Adaptation      │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│             HYBRID AI LAYER                   │
│       Digital AI ↔ Biological Neural AI       │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│              OI COMPUTATION LAYER             │
│   Organoids • Neural Cultures • OI Networks   │
└───────────────────────┬───────────────────────┘
                        │
┌───────────────────────▼───────────────────────┐
│             INTERFACE LAYER                   │
│     Neural Sensors • Stimulation • I/O        │
└───────────────────────────────────────────────┘
```

---

# 3. Distributed OI Architecture

A future large-scale OI system could consist of many interconnected biological computational units.

```text
                 DIGITAL AI
                     │
             ┌───────▼───────┐
             │ Neural Gateway │
             └───────┬───────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
 ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
 │  OI Node  │ │  OI Node  │ │  OI Node  │
 │     A     │ │     B     │ │     C     │
 └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
       │             │             │
       └─────────────┼─────────────┘
                     │
             Distributed OI
                     │
             ┌───────▼───────┐
             │ Digital Memory│
             └───────────────┘
```

The architecture is intentionally modular.

Each OI node can be treated as an adaptive biological processing unit while the digital layer provides orchestration, storage, monitoring, and high-level computation.

---

# 4. Computational Model

A simplified OI system can be represented as a dynamical neural state:

\[
x_{t+1}=F(x_t,u_t,\theta_t,\eta_t)
\]

where:

- \(x_t\) = biological neural state
- \(u_t\) = external input
- \(\theta_t\) = adaptive network parameters
- \(\eta_t\) = biological variability/noise
- \(F\) = biological state-transition operator

The output can be represented as:

\[
y_t = G(x_t)
\]

where \(G\) maps biological neural activity into an observable computational output.

## 4.1 Plasticity Model

A simplified adaptive synaptic model can be written as:

\[
\Delta w_{ij}
=
\alpha H(x_i,x_j,r_t)
\]

where:

- \(w_{ij}\) = effective connection strength
- \(\alpha\) = adaptation rate
- \(x_i,x_j\) = neural activity
- \(r_t\) = task-dependent reinforcement signal
- \(H\) = plasticity operator

This is a conceptual mathematical abstraction rather than a complete biological model.

## 4.2 Closed-Loop Learning

The OI system can operate as a closed-loop dynamical system:

```text
Environment
     │
     ▼
Sensor
     │
     ▼
OI Neural State
     │
     ▼
Output
     │
     ▼
Environment
     │
     └──────── feedback ────────►
```

The system continuously receives observations and modifies its internal neural dynamics.

---

# 5. Hybrid OI + AI Model

A central research direction is the integration of OI with digital AI.

```text
             ┌─────────────────┐
             │   Digital AI    │
             │                 │
             │ LLM / Planning  │
             │ Reasoning       │
             └────────┬────────┘
                      │
                Neural Gateway
                      │
             ┌────────▼────────┐
             │       OI        │
             │                 │
             │ Plasticity      │
             │ Adaptation      │
             │ Biological      │
             │ Representation  │
             └─────────────────┘
```

The digital system may provide:

- language processing
- symbolic reasoning
- long-term digital memory
- planning
- data management
- computational scaling

The OI system may potentially provide:

- adaptive representations
- biological plasticity
- continual learning
- nonlinear dynamics
- energy-efficient neural computation

The goal is **complementarity**, not necessarily replacement.

---

# 6. API Design

The project proposes a hardware-independent conceptual API.

## Input

```python
oi.input(
    stimulus,
    modality="visual",
    timestamp=None
)
```

## Training

```python
oi.train(
    task,
    reward=None,
    episodes=100
)
```

## State

```python
state = oi.get_state()
```

## Neural Activity

```python
activity = oi.get_activity(
    region="global"
)
```

## Output

```python
output = oi.predict(
    stimulus
)
```

## Closed-Loop Environment

```python
oi.run(
    environment,
    steps=1000
)
```

These APIs represent a software abstraction layer. They do not imply that current organoid systems already expose such standardized interfaces.

---

# 7. Experimental Framework

Experiments should be divided into progressively more complex levels.

## Level 0 — Baseline Neural Activity

Measure:

- spontaneous activity
- firing patterns
- connectivity
- temporal stability
- response variability

## Level 1 — Simple Learning

Example tasks:

- pattern classification
- temporal sequence prediction
- stimulus-response association
- simple reinforcement learning

## Level 2 — Closed-Loop Learning

The OI receives environmental feedback.

Examples:

- Pong-like environments
- navigation
- adaptive control
- robotic sensorimotor tasks

The objective is to measure whether learning persists and generalizes.

## Level 3 — Multi-Task Learning

The same OI system is exposed to multiple tasks.

Metrics include:

- transfer learning
- catastrophic forgetting
- adaptation speed
- task generalization

## Level 4 — Hybrid AI

Connect OI with a digital AI model.

```text
Input
  │
  ▼
Digital AI
  │
  ▼
OI
  │
  ▼
Digital AI
  │
  ▼
Decision
```

## Level 5 — Distributed OI

Multiple OI nodes communicate through digital and/or biological interfaces.

Research questions include:

- distributed learning
- specialization
- information sharing
- emergent network behavior
- fault tolerance

---

# 8. Benchmark Suite

OI should be evaluated against multiple baselines rather than against a single measure of "intelligence."

## 8.1 Learning Efficiency

Measure:

\[
E_L=\frac{\text{Task Performance}}{\text{Training Cost}}
\]

Possible measurements:

- samples required
- training time
- energy consumption
- number of adaptation cycles

## 8.2 Generalization

Evaluate performance on previously unseen:

- inputs
- environments
- tasks
- task combinations

## 8.3 Continual Learning

Measure whether the system can learn:

```text
Task A → Task B → Task C
```

without catastrophic loss of Task A.

## 8.4 Adaptation Speed

Measure:

\[
T_{adapt}
=
\text{time required to reach a predefined performance threshold}
\]

## 8.5 Energy Efficiency

Compare useful computational output per unit energy:

\[
\eta =
\frac{\text{Useful Computation}}
{\text{Energy}}
\]

## 8.6 Robustness

Test performance under:

- noise
- missing inputs
- network perturbations
- component failure
- environmental changes

---

# 9. Benchmark Categories

| Category | OI | Digital AI | Neuromorphic |
|---|---:|---:|---:|
| Learning efficiency | TBD | TBD | TBD |
| Continual learning | TBD | TBD | TBD |
| Adaptation | TBD | TBD | TBD |
| Generalization | TBD | TBD | TBD |
| Energy efficiency | TBD | TBD | TBD |
| Robustness | TBD | TBD | TBD |
| Multi-task learning | TBD | TBD | TBD |

**TBD values must be experimentally measured.**

No claim of superiority should be made without controlled experiments.

---

# 10. Experimental Reproducibility

Every experiment should record:

```text
Experiment ID
Biological preparation ID
Culture conditions
Neural interface configuration
Input protocol
Training protocol
Environmental conditions
Random seed
Recording duration
Raw neural data
Processed data
Model configuration
Evaluation metrics
Statistical analysis
```

Experiments should be version-controlled whenever possible.

---

# 11. Data Pipeline

```text
Biological System
       │
       ▼
Neural Recording
       │
       ▼
Signal Processing
       │
       ▼
Feature Extraction
       │
       ▼
OI State Representation
       │
       ▼
Learning Algorithm
       │
       ▼
Prediction / Control
       │
       ▼
Evaluation
```

Raw experimental data should be preserved separately from processed datasets.

---


---

# 12. Roadmap

## Phase I — Computational Modeling

- [ ] Define OI state-space models
- [ ] Implement neural dynamics
- [ ] Implement plasticity models
- [ ] Create simulation environments
- [ ] Establish baseline benchmarks

## Phase II — Neural Interface

- [ ] Define standardized input/output representation
- [ ] Develop neural recording pipeline
- [ ] Develop stimulation interface
- [ ] Establish closed-loop experiments

## Phase III — Biological Learning

- [ ] Simple classification
- [ ] Temporal learning
- [ ] Reinforcement learning
- [ ] Continual learning
- [ ] Transfer learning

## Phase IV — Hybrid Intelligence

- [ ] Digital AI ↔ OI interface
- [ ] Shared memory architecture
- [ ] Hybrid learning
- [ ] Multi-modal input
- [ ] Agent-based environments

## Phase V — Distributed OI

- [ ] Multiple OI nodes
- [ ] Distributed learning
- [ ] Fault tolerance
- [ ] Network-level memory
- [ ] Large-scale benchmarking

## Phase VI — Advanced Intelligence Research

- [ ] Generalization across domains
- [ ] Open-ended learning
- [ ] Autonomous adaptation
- [ ] Long-horizon planning
- [ ] Investigation of potential AGI-relevant capabilities

---

# 13. Ethics and Governance

OI research requires stronger ethical oversight as biological complexity increases.

The project should establish explicit safeguards concerning:

- biological welfare
- neural complexity
- possible sentience
- experimental burden
- data governance
- responsible scaling
- human oversight

The project must not assume that increasing neural complexity automatically implies consciousness.

At the same time, the possibility of morally relevant neural states should not be ignored.

A **precautionary framework** should therefore accompany technical development.

---

# 14. Scientific Limitations

Several major limitations remain unresolved.

1. It is unknown whether organoids can achieve general intelligence.
2. Biological neural networks are highly variable.
3. Current organoids do not reproduce the full organization of a human brain.
4. Scaling neural tissue does not guarantee improved cognition.
5. Current methods for measuring biological intelligence remain incomplete.
6. The relationship between neural activity and subjective experience is not established.
7. AGI cannot be inferred from benchmark performance on a small number of tasks.

Therefore, all claims should distinguish between:

**Observed experimental results**

and

**Future hypotheses.**

---

# 15. Research Philosophy

The project is based on the following principle:

> **Intelligence may be substrate-independent, but the substrate can strongly influence how intelligence emerges.**

Digital AI demonstrates that sophisticated information processing can emerge from mathematical models running on silicon.

OI asks a complementary question:

> **Can intelligence-relevant computation emerge from engineered biological neural systems?**

The ultimate objective is to understand the computational principles shared by biological and artificial intelligence.

---

# 16. Long-Term Vision

The long-term vision is a scalable **Biological-Digital Intelligence Platform** combining:

```text
Biological Neural Computation
              +
Digital Artificial Intelligence
              +
Neuromorphic Computing
              +
Large-Scale Distributed Systems
              +
Continual Learning
              ↓
     Advanced Hybrid Intelligence
```

Such a system could potentially support applications in:

- adaptive robotics
- scientific discovery
- autonomous systems
- neuroscience
- biological computing
- intelligent control
- computational biology
- future general-intelligence research

This remains a long-term research direction rather than an established technological capability.

---

# 17. Citation

If this project is used in academic research, cite the repository according to the metadata provided in `CITATION.cff`.

```bibtex
@software{organoid_intelligence,
  title   = {Organoid Intelligence: A Research Framework for Biological-Digital Intelligence},
  author  = {OI Research Project : MSPS NETWORK},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/yoonalimsuwan/mental-one}
}
```

Replace the repository URL and author metadata with the actual project information before publication.

---

# 18. Project Status

**Current Status:** Conceptual / Research Framework

**Version:** 0.1.0

**Primary Goal:** Investigate scalable biological neural computation and its integration with artificial intelligence.

**AGI Status:** Not demonstrated.

**OI Status:** Experimental research direction.

---

## Final Statement

Organoid Intelligence represents an attempt to explore a new computational paradigm in which **living neural networks become computational substrates rather than merely biological objects of study**.

The central research challenge is not simply to build a larger neural system.

It is to determine whether biological neural computation can be **scaled, controlled, measured, reproduced, and integrated with digital intelligence** while maintaining rigorous scientific and ethical standards.

The ultimate question is:

> **Can scalable biological computation become a foundation for a new generation of intelligent machines?**

This repository provides a conceptual framework for investigating that question.
