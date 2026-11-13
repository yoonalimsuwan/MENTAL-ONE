#!/usr/bin/env python3
"""
=============================================================================
NEURO INTERPRETABILITY + DOMAIN ADAPTATION + LLM ASSESSMENT SCAFFOLD
Grad-CAM · Integrated Gradients · Adversarial Domain Adaptation
Clinical Research Framework · LLM Integration · World Model
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  CRITICAL DISCLAIMER — READ BEFORE ANYTHING ELSE                 ║
║                                                                     ║
║  THIS IS A RESEARCH SCAFFOLD. IT IS NOT:                           ║
║  ❌ A diagnostic tool                                              ║
║  ❌ A clinical assessment tool                                     ║
║  ❌ A substitute for trained medical professionals                 ║
║  ❌ A tool for making treatment decisions                          ║
║                                                                     ║
║  The clinical categories referenced below (Somatic Symptoms        ║
║  Disorder, Conversion Disorder, etc.) are DSM-5 research           ║
║  categories used ONLY as structural labels for the code.           ║
║  NO output from this system constitutes a diagnosis or             ║
║  clinical finding.                                                 ║
║                                                                     ║
║  Sleepwalking risk assessment requires clinical evaluation        ║
║  by a sleep medicine specialist. This module provides             ║
║  reference information only.                                      ║
╚═══════════════════════════════════════════════════════════════════════╝

WHAT THIS MODULE PROVIDES
  §1  GradCAM + Integrated Gradients — feature-level interpretability
  §2  Adversarial Domain Adaptation — reduce Dreaddit↔DAIC-WOZ shift
  §3  Clinical Research Framework — DSM-5 categories as data structures
  §4  LLM Integration Module — text analysis scaffold
  §5  World Model — predictive dynamics scaffold

WHAT THIS MODULE IS NOT
  ❌ Not a diagnostic tool
  ❌ Not a clinical assessment tool
  ❌ Not validated against any clinical gold standard
  ❌ Not approved by any regulatory body
  ❌ LLM module requires API keys not included

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("neuro_interp")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# §0. RESEARCH-ONLY GUARD
# =============================================================================

RESEARCH_BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  RESEARCH SCAFFOLD — NOT A CLINICAL TOOL                        ║
║  All clinical categories below are RESEARCH LABELS, not diagnoses.  ║
║  No output constitutes a medical finding.                           ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def research_guard():
    print(RESEARCH_BANNER)

# =============================================================================
# §1. GRAD-CAM + INTEGRATED GRADIENTS
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017).
    Identifies which input features the model uses for a prediction.

    Reference: Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations
    from Deep Networks via Gradient-based Localization. ICCV.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[Tensor] = None
        self.activations: Optional[Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: Tensor, target_class: Optional[int] = None
                 ) -> Tuple[Tensor, Tensor]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: input data (must require grad)
            target_class: class to explain (None = predicted class)

        Returns:
            cam: (spatial_dims) heatmap
            pred_class: predicted class index
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        if output.dim() > 1:
            pred_class = int(output.argmax(dim=-1).item())
        else:
            pred_class = 0

        if target_class is None:
            target_class = pred_class

        # Backward
        self.model.zero_grad()
        if output.dim() > 1:
            target_score = output[0, target_class]
        else:
            target_score = output.sum()
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks not triggered — check target_layer")

        # Global average pooling of gradients
        if self.gradients.dim() == 4:  # (B, C, H, W)
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
        elif self.gradients.dim() == 3:  # (B, C, L)
            weights = self.gradients.mean(dim=2, keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
        elif self.gradients.dim() == 2:  # (B, C)
            cam = (self.gradients * self.activations).sum(dim=1, keepdim=True)
        else:
            cam = self.gradients.sum(dim=0, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.squeeze(), pred_class

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al. 2017).
    Axiomatic attribution: satisfies Sensitivity and Implementation Invariance.

    Reference: Sundararajan, M. et al. (2017). Axiomatic Attribution for
    Deep Networks. ICML.
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def attribute(self, input_tensor: Tensor,
                  baseline: Optional[Tensor] = None,
                  target_class: Optional[int] = None) -> Tensor:
        """
        Compute integrated gradients attribution.

        Args:
            input_tensor: input to explain
            baseline: reference input (default: zeros)
            target_class: class to explain (None = predicted)

        Returns:
            attribution: same shape as input_tensor
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        self.model.eval()

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps + 1,
                                device=input_tensor.device)
        alphas = alphas.view(-1, *([1] * (input_tensor.dim() - 1)))

        # (n_steps+1, *input_shape)
        interpolated = baseline + alphas * (input_tensor - baseline)

        # Compute gradients at each step
        interpolated = interpolated.requires_grad_(True)
        outputs = self.model(interpolated)

        if outputs.dim() > 1:
            if target_class is None:
                target_class = int(outputs[0].argmax(dim=-1).item())
            scores = outputs[:, target_class]
        else:
            scores = outputs

        grads = torch.autograd.grad(
            scores.sum(), interpolated,
            create_graph=False, retain_graph=False
        )[0]

        # Average gradients and multiply by (input - baseline)
        avg_grads = grads.mean(dim=0)
        attribution = (input_tensor - baseline) * avg_grads

        return attribution

    def attribute_batch(self, input_batch: Tensor,
                        baseline: Optional[Tensor] = None,
                        target_classes: Optional[List[int]] = None
                        ) -> List[Tensor]:
        """Batch version of attribute()."""
        results = []
        for i in range(input_batch.shape[0]):
            tc = target_classes[i] if target_classes else None
            results.append(
                self.attribute(input_batch[i:i+1], baseline, tc)
            )
        return results


class FeatureAttributionReport:
    """Structured report of feature attributions."""

    def __init__(self, method: str, attributions: Tensor,
                 feature_names: Optional[List[str]] = None):
        self.method = method
        self.attributions = attributions
        self.feature_names = feature_names

    def top_features(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k most important features."""
        flat = self.attributions.abs().flatten()
        topk = torch.topk(flat, min(k, flat.numel()))
        results = []
        for i, (val, idx) in enumerate(zip(topk.values, topk.indices)):
            idx_flat = idx.item()
            if self.feature_names and idx_flat < len(self.feature_names):
                name = self.feature_names[idx_flat]
            else:
                name = f"feature_{idx_flat}"
            results.append((name, float(val)))
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "top_features": self.top_features(20),
            "attribution_shape": list(self.attributions.shape),
            "attribution_mean": float(self.attributions.mean()),
            "attribution_std": float(self.attributions.std()),
        }


# =============================================================================
# §2. ADVERSARIAL DOMAIN ADAPTATION
# =============================================================================

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal for adversarial domain adaptation.
    Forward: identity. Backward: multiply gradient by -lambda.

    Reference: Ganin, Y. et al. (2016). Domain-Adversarial Training of
    Neural Networks. JMLR.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainDiscriminator(nn.Module):
    """
    Discriminator for domain adaptation.
    Tries to distinguish source vs target domain.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns logit (sigmoid → probability of being source domain)."""
        return self.net(x).squeeze(-1)


class DomainAdaptationTrainer:
    """
    Adversarial domain adaptation trainer.

    Trains a feature extractor to produce domain-invariant features
    by adversarially training against a domain discriminator.

    Loss = task_loss + lambda * domain_loss

    Reference: Ganin, Y. et al. (2016). Domain-Adversarial Training of
    Neural Networks. JMLR.
    """

    def __init__(self, feature_extractor: nn.Module,
                 task_head: nn.Module,
                 discriminator: DomainDiscriminator,
                 lambda_init: float = 0.1,
                 lambda_schedule: str = "constant"):
        self.feature_extractor = feature_extractor
        self.task_head = task_head
        self.discriminator = discriminator
        self.lambda_ = lambda_init
        self.lambda_schedule = lambda_schedule
        self.global_step = 0

    def update_lambda(self):
        """Update lambda according to schedule."""
        if self.lambda_schedule == "constant":
            pass
        elif self.lambda_schedule == "linear":
            self.lambda_ = min(1.0, self.global_step / 1000.0)
        elif self.lambda_schedule == "exponential":
            self.lambda_ = 2.0 / (1.0 + math.exp(-10.0 * self.global_step / 1000.0)) - 1.0
        self.global_step += 1

    def compute_loss(self, source_data: Tensor, source_labels: Tensor,
                     target_data: Tensor,
                     criterion: nn.Module = None) -> Dict[str, Tensor]:
        """
        Compute adversarial domain adaptation loss.

        Args:
            source_data: labeled source domain data
            source_labels: source domain labels
            target_data: unlabeled target domain data

        Returns:
            dict with total_loss, task_loss, domain_loss
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Extract features
        source_features = self.feature_extractor(source_data)
        target_features = self.feature_extractor(target_data)

        # Task loss (source only)
        task_output = self.task_head(source_features)
        task_loss = criterion(task_output, source_labels)

        # Domain loss (both source and target)
        # Source domain label = 1, Target domain label = 0
        source_domain = torch.ones(source_features.shape[0],
                                   device=source_features.device)
        target_domain = torch.zeros(target_features.shape[0],
                                    device=target_features.device)

        # Apply gradient reversal
        source_features_rev = GradientReversalLayer.apply(
            source_features, self.lambda_)
        target_features_rev = GradientReversalLayer.apply(
            target_features, self.lambda_)

        source_domain_pred = self.discriminator(source_features_rev)
        target_domain_pred = self.discriminator(target_features_rev)

        domain_loss = (
            F.binary_cross_entropy_with_logits(
                source_domain_pred, source_domain) +
            F.binary_cross_entropy_with_logits(
                target_domain_pred, target_domain)
        )

        total_loss = task_loss + self.lambda_ * domain_loss

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "domain_loss": domain_loss,
            "lambda": self.lambda_,
        }

    def train_step(self, source_data, source_labels, target_data,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module = None) -> Dict[str, float]:
        """Single training step."""
        self.update_lambda()
        optimizer.zero_grad()
        losses = self.compute_loss(source_data, source_labels,
                                   target_data, criterion)
        losses["total_loss"].backward()
        optimizer.step()
        return {k: float(v) if isinstance(v, Tensor) else v
                for k, v in losses.items()}


# =============================================================================
# §3. CLINICAL RESEARCH FRAMEWORK (DSM-5 categories as data structures)
# =============================================================================

class SomaticCategory(Enum):
    """
    DSM-5 Somatic Symptom and Related Disorders.
    These are RESEARCH CATEGORIES, not diagnostic outputs.
    """
    SOMATIC_SYMPTOM_DISORDER = "somatic_symptom_disorder"
    CONVERSION_DISORDER = "conversion_disorder"
    ILLNESS_ANXIETY = "illness_anxiety_disorder"
    FACTITIOUS = "factitious_disorder"
    UNSPECIFIED = "unspecified"


class NeurologicalSymptom(Enum):
    """Neurological symptoms that may be assessed."""
    PAIN = "pain"
    MUSCLE_WEAKNESS = "muscle_weakness"
    VISION_LOSS = "vision_loss"
    SLEEPWALKING = "sleepwalking"
    SEIZURE_LIKE = "seizure_like"
    MOVEMENT_ABORMALITY = "movement_abnormality"


@dataclass
class ClinicalFeatureVector:
    """
    Structured clinical feature vector for research purposes.

    IMPORTANT: This is a data structure for RESEARCH, not a diagnostic tool.
    All values must be collected and interpreted by trained clinicians.
    """
    symptom_type: NeurologicalSymptom
    severity_score: float  # 0-10 scale, clinician-assessed
    duration_days: float
    frequency_per_week: float
    functional_impact: float  # 0-10 scale
    psychological_stressors: List[str] = field(default_factory=list)
    medical_rule_outs: List[str] = field(default_factory=list)
    notes: str = ""

    def to_tensor(self) -> Tensor:
        """Convert to tensor for model input."""
        return torch.tensor([
            self.severity_score / 10.0,
            math.log1p(self.duration_days) / 10.0,
            self.frequency_per_week / 7.0,
            self.functional_impact / 10.0,
            len(self.psychological_stressors) / 10.0,
            len(self.medical_rule_outs) / 10.0,
        ], dtype=torch.float32)


@dataclass
class SleepwalkingRiskFactors:
    """
    Risk factors for dangerous sleepwalking episodes.

    REFERENCE INFORMATION ONLY — not a risk assessment tool.
    Actual risk assessment requires clinical evaluation by a
    sleep medicine specialist.

    Reference: Stallman, H.M. & Kohler, M. (2016). Prevalence of sleepwalking:
    A systematic review and meta-analysis. Sleep Medicine, 27-28, 28-32.
    """
    # Known risk factors from literature
    sleep_deprivation: bool = False
    sedative_use: bool = False
    fever: bool = False
    stress: bool = False
    family_history: bool = False
    other_sleep_disorders: bool = False
    medications: List[str] = field(default_factory=list)

    # Environmental risk factors
    unlocked_doors: bool = False
    accessible_stairs: bool = False
    sharp_objects_nearby: bool = False
    accessible_car_keys: bool = False

    def risk_factor_count(self) -> int:
        """Count number of present risk factors."""
        count = sum([
            self.sleep_deprivation,
            self.sedative_use,
            self.fever,
            self.stress,
            self.family_history,
            self.other_sleep_disorders,
            self.unlocked_doors,
            self.accessible_stairs,
            self.sharp_objects_nearby,
            self.accessible_car_keys,
        ])
        return count

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalResearchFramework:
    """
    Framework for organizing clinical research data.

    IMPORTANT: This is a data organization framework, NOT a diagnostic tool.
    All clinical interpretations must be made by trained professionals.
    """

    # DSM-5 diagnostic criteria references (for research documentation)
    DSM5_CRITERIA = {
        SomaticCategory.SOMATIC_SYMPTOM_DISORDER: {
            "code": "300.82 (F45.1)",
            "criteria": [
                "One or more somatic symptoms that are distressing or result in significant disruption of daily life",
                "Excessive thoughts, feelings, or behaviors related to the somatic symptoms",
                "State of being symptomatic is persistent (typically > 6 months)",
            ],
            "specifiers": [
                "With predominant pain",
                "Persistent",
                "Mild/Moderate/Severe",
            ],
        },
        SomaticCategory.CONVERSION_DISORDER: {
            "code": "300.11 (F44.4-F44.7)",
            "criteria": [
                "One or more symptoms of altered voluntary motor or sensory function",
                "Clinical findings provide evidence of incompatibility between the symptom and recognized neurological or medical conditions",
                "The symptom or deficit is not better explained by another medical or mental disorder",
                "The symptom or deficit causes clinically significant distress or impairment",
            ],
            "specifiers": [
                "With weakness or paralysis",
                "With abnormal movement",
                "With swallowing symptoms",
                "With speech symptom",
                "With attacks or seizures",
                "With anesthesia or sensory loss",
                "With special sensory symptom",
            ],
        },
    }

    @staticmethod
    def get_criteria(category: SomaticCategory) -> Dict[str, Any]:
        """Get DSM-5 criteria reference for a category."""
        return ClinicalResearchFramework.DSM5_CRITERIA.get(category, {})

    @staticmethod
    def differentiate_somatic_vs_conversion(
        features: ClinicalFeatureVector,
    ) -> Dict[str, Any]:
        """
        RESEARCH ONLY: Structural comparison of features against DSM-5 criteria.

        This is NOT a diagnostic tool. It provides a structured comparison
        of features against published criteria for research documentation.

        Returns a dictionary with research notes, not a diagnosis.
        """
        result = {
            "disclaimer": "RESEARCH OUTPUT — NOT A DIAGNOSIS",
            "symptom_type": features.symptom_type.value,
            "features": {
                "severity": features.severity_score,
                "duration_days": features.duration_days,
                "functional_impact": features.functional_impact,
            },
            "somatic_symptom_disorder_criteria": ClinicalResearchFramework.get_criteria(
                SomaticCategory.SOMATIC_SYMPTOM_DISORDER),
            "conversion_disorder_criteria": ClinicalResearchFramework.get_criteria(
                SomaticCategory.CONVERSION_DISORDER),
            "research_notes": [
                "Somatic Symptom Disorder: excessive thoughts/feelings/behaviors about symptoms",
                "Conversion Disorder: incompatibility between symptom and recognized medical condition",
                "Medical rule-outs MUST be completed by qualified physicians",
                "Psychological assessment MUST be completed by qualified clinicians",
                "This comparison is for research documentation only",
            ],
        }
        return result


# =============================================================================
# §4. LLM INTEGRATION MODULE
# =============================================================================

class LLMIntegrationModule:
    """
    LLM integration for clinical text analysis.

    This module provides the STRUCTURE for LLM integration.
    Actual LLM API calls require valid API keys and are not included.

    Supported backends (when API keys are available):
    - Claude (Anthropic)
    - GPT-4 (OpenAI)
    - Gemini (Google)

    All LLM outputs are RESEARCH HYPOTHESES, not clinical findings.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self._clients = {}
        self._init_clients()

    def _init_clients(self):
        """Initialize LLM clients if API keys are available."""
        # Claude
        if "anthropic" in self.api_keys:
            try:
                import anthropic
                self._clients["claude"] = anthropic.Anthropic(
                    api_key=self.api_keys["anthropic"])
                log.info("✓ Claude API client initialized")
            except ImportError:
                log.warning("anthropic package not installed")
            except Exception as e:
                log.warning(f"Claude init failed: {e}")

        # OpenAI
        if "openai" in self.api_keys:
            try:
                import openai
                self._clients["gpt"] = openai.OpenAI(
                    api_key=self.api_keys["openai"])
                log.info("✓ GPT API client initialized")
            except ImportError:
                log.warning("openai package not installed")
            except Exception as e:
                log.warning(f"GPT init failed: {e}")

        # Gemini
        if "google" in self.api_keys:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_keys["google"])
                self._clients["gemini"] = genai.GenerativeModel("gemini-1.5-pro")
                log.info("✓ Gemini API client initialized")
            except ImportError:
                log.warning("google-generativeai package not installed")
            except Exception as e:
                log.warning(f"Gemini init failed: {e}")

    def analyze_clinical_text(self, text: str,
                              analysis_type: str = "general"
                              ) -> Dict[str, Any]:
        """
        Analyze clinical text using available LLM.

        Args:
            text: clinical text to analyze
            analysis_type: type of analysis requested

        Returns:
            dict with analysis results or error message
        """
        if not self._clients:
            return {
                "status": "unavailable",
                "error": "No LLM API keys configured. "
                         "Set api_keys in constructor.",
                "note": "LLM analysis requires valid API keys",
            }

        # Use first available client
        client_name = next(iter(self._clients))
        client = self._clients[client_name]

        prompt = self._build_prompt(text, analysis_type)

        try:
            if client_name == "claude":
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.content[0].text
            elif client_name == "gpt":
                response = client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.choices[0].message.content
            elif client_name == "gemini":
                response = client.generate_content(prompt)
                result = response.text
            else:
                result = "Unknown client"

            return {
                "status": "success",
                "client": client_name,
                "analysis_type": analysis_type,
                "result": result,
                "disclaimer": "LLM output is a RESEARCH HYPOTHESIS, "
                              "not a clinical finding",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "client": client_name,
            }

    def _build_prompt(self, text: str, analysis_type: str) -> str:
        """Build analysis prompt."""
        base_prompt = f"""
You are a research assistant analyzing clinical text for a RESEARCH STUDY.
This analysis is NOT for clinical diagnosis or treatment decisions.

Text to analyze:
{text}

Analysis type: {analysis_type}

Please provide:
1. Key features identified in the text
2. Relevant clinical categories mentioned
3. Any temporal patterns or progression noted
4. Research observations (NOT diagnoses)

IMPORTANT: Your output is a RESEARCH HYPOTHESIS only.
It must be interpreted by trained clinical professionals.
Do not provide diagnoses or treatment recommendations.
"""
        return base_prompt

    def extract_symptoms_from_text(self, text: str) -> Dict[str, Any]:
        """Extract symptom mentions from clinical text."""
        return self.analyze_clinical_text(text, "symptom_extraction")

    def summarize_clinical_notes(self, notes: str) -> Dict[str, Any]:
        """Summarize clinical notes for research purposes."""
        return self.analyze_clinical_text(notes, "summarization")


# =============================================================================
# §5. WORLD MODEL FOR PREDICTIVE DYNAMICS
# =============================================================================

class NeuroWorldModel(nn.Module):
    """
    World model for predicting neural/clinical state dynamics.

    This is a RESEARCH SCAFFOLD for studying predictive dynamics.
    It is NOT a clinical prediction tool.

    Architecture:
        Encoder → Latent dynamics (GRU) → Decoder → Prediction
    """

    def __init__(self, input_dim: int, latent_dim: int = 128,
                 output_dim: int = 6, n_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Dynamics model
        self.dynamics = nn.GRUCell(latent_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

        # Uncertainty estimator
        self.uncertainty = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, output_dim),
            nn.Softplus(),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def step(self, z: Tensor) -> Tensor:
        return self.dynamics(z, z)

    def decode(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.decoder(z)
        unc = self.uncertainty(z)
        return pred, unc

    def forward(self, x_seq: Tensor, n_future: int = 1
                ) -> Dict[str, Tensor]:
        """
        Predict future states from a sequence.

        Args:
            x_seq: (batch, time, input_dim) input sequence
            n_future: number of future steps to predict

        Returns:
            dict with predictions and uncertainties
        """
        batch_size = x_seq.shape[0]
        seq_len = x_seq.shape[1]

        # Encode initial state
        z = self.encode(x_seq[:, 0])

        # Run dynamics for sequence
        for t in range(1, seq_len):
            z = self.step(z)

        # Predict future
        predictions = []
        uncertainties = []
        for _ in range(n_future):
            z = self.step(z)
            pred, unc = self.decode(z)
            predictions.append(pred)
            uncertainties.append(unc)

        return {
            "predictions": torch.stack(predictions, dim=1),
            "uncertainties": torch.stack(uncertainties, dim=1),
        }


# =============================================================================
# §6. UNIFIED NEURO-ASSESSMENT PIPELINE
# =============================================================================

class NeuroAssessmentPipeline:
    """
    Unified pipeline combining interpretability, domain adaptation,
    clinical framework, LLM, and world model.

    RESEARCH SCAFFOLD ONLY — NOT A CLINICAL TOOL.
    """

    def __init__(self, feature_dim: int = 6, latent_dim: int = 128):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Components
        self.world_model = NeuroWorldModel(feature_dim, latent_dim)
        self.clinical_framework = ClinicalResearchFramework()
        self.llm_module = LLMIntegrationModule()

        # Interpretability tools
        self.gradcam: Optional[GradCAM] = None
        self.integrated_gradients: Optional[IntegratedGradients] = None

        # Domain adaptation
        self.domain_trainer: Optional[DomainAdaptationTrainer] = None

    def setup_interpretability(self, model: nn.Module,
                               target_layer: nn.Module):
        """Setup Grad-CAM and Integrated Gradients."""
        self.gradcam = GradCAM(model, target_layer)
        self.integrated_gradients = IntegratedGradients(model)

    def setup_domain_adaptation(self, feature_extractor: nn.Module,
                                task_head: nn.Module,
                                discriminator: DomainDiscriminator):
        """Setup adversarial domain adaptation."""
        self.domain_trainer = DomainAdaptationTrainer(
            feature_extractor, task_head, discriminator)

    def analyze_features(self, input_tensor: Tensor,
                         feature_names: Optional[List[str]] = None
                         ) -> Dict[str, Any]:
        """
        Analyze input features using interpretability tools.

        Returns attribution reports from both Grad-CAM and IG.
        """
        results = {}

        if self.gradcam is not None:
            cam, pred_class = self.gradcam.generate(input_tensor)
            results["gradcam"] = {
                "heatmap_shape": list(cam.shape),
                "predicted_class": pred_class,
                "max_activation": float(cam.max()),
            }

        if self.integrated_gradients is not None:
            attr = self.integrated_gradients.attribute(input_tensor)
            report = FeatureAttributionReport(
                "integrated_gradients", attr, feature_names)
            results["integrated_gradients"] = report.to_dict()

        return results

    def assess_clinical_features(self, features: ClinicalFeatureVector
                                 ) -> Dict[str, Any]:
        """
        Process clinical features through the research framework.

        Returns structured research data, NOT a diagnosis.
        """
        result = self.clinical_framework.differentiate_somatic_vs_conversion(
            features)
        result["feature_tensor"] = features.to_tensor().tolist()
        return result

    def assess_sleepwalking_risk(self, risk_factors: SleepwalkingRiskFactors
                                 ) -> Dict[str, Any]:
        """
        Document sleepwalking risk factors.

        REFERENCE INFORMATION ONLY — NOT a risk assessment.
        Actual risk assessment requires clinical evaluation.
        """
        return {
            "disclaimer": "REFERENCE INFORMATION — NOT A RISK ASSESSMENT",
            "risk_factor_count": risk_factors.risk_factor_count(),
            "factors": risk_factors.to_dict(),
            "clinical_note": (
                "Sleepwalking risk assessment requires evaluation by a "
                "sleep medicine specialist. This information is for "
                "research documentation only."
            ),
        }

    def run_full_assessment(self, input_tensor: Tensor,
                            clinical_features: Optional[ClinicalFeatureVector] = None,
                            sleepwalking_factors: Optional[SleepwalkingRiskFactors] = None,
                            clinical_text: Optional[str] = None
                            ) -> Dict[str, Any]:
        """
        Run full assessment pipeline.

        All outputs are RESEARCH DATA, not clinical findings.
        """
        results = {
            "disclaimer": "RESEARCH SCAFFOLD — NOT A CLINICAL TOOL",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Feature analysis
        results["feature_analysis"] = self.analyze_features(input_tensor)

        # Clinical features
        if clinical_features is not None:
            results["clinical_assessment"] = self.assess_clinical_features(
                clinical_features)

        # Sleepwalking
        if sleepwalking_factors is not None:
            results["sleepwalking_assessment"] = self.assess_sleepwalking_risk(
                sleepwalking_factors)

        # LLM analysis
        if clinical_text is not None:
            results["llm_analysis"] = self.llm_module.analyze_clinical_text(
                clinical_text)

        # World model prediction
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            wm_output = self.world_model(input_tensor, n_future=3)
        results["world_model_prediction"] = {
            "predictions": wm_output["predictions"].tolist(),
            "uncertainties": wm_output["uncertainties"].tolist(),
        }

        return results


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test() -> bool:
    research_guard()
    print("=" * 78)
    print(f"  NEURO INTERPRETABILITY + DOMAIN ADAPTATION + LLM v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: GradCAM
    print("\n[T1] Grad-CAM")
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    gradcam = GradCAM(model, model[4])  # target last ReLU
    x = torch.randn(1, 10)
    cam, pred = gradcam.generate(x)
    rep("GradCAM produces heatmap", cam.numel() > 0)
    rep("GradCAM heatmap normalized", float(cam.max()) <= 1.0 + 1e-6)
    gradcam.remove_hooks()

    # T2: Integrated Gradients
    print("\n[T2] Integrated Gradients")
    ig = IntegratedGradients(model, n_steps=20)
    attr = ig.attribute(x)
    rep("IG produces attribution", attr.shape == x.shape)
    rep("IG attribution finite", bool(torch.isfinite(attr).all()))

    # T3: Feature Attribution Report
    print("\n[T3] Feature Attribution Report")
    report = FeatureAttributionReport("ig", attr, [f"f{i}" for i in range(10)])
    top = report.top_features(5)
    rep("Report produces top features", len(top) > 0)
    rep("Report serializable", isinstance(report.to_dict(), dict))

    # T4: Domain Discriminator
    print("\n[T4] Domain Discriminator")
    disc = DomainDiscriminator(input_dim=32)
    features = torch.randn(8, 32)
    domain_pred = disc(features)
    rep("Discriminator output shape", domain_pred.shape == (8,))

    # T5: Gradient Reversal
    print("\n[T5] Gradient Reversal Layer")
    x_rev = torch.randn(4, 16, requires_grad=True)
    out = GradientReversalLayer.apply(x_rev, 1.0)
    loss = out.sum()
    loss.backward()
    rep("Gradient reversal changes sign",
        bool((x_rev.grad < 0).any()))

    # T6: Domain Adaptation Trainer
    print("\n[T6] Domain Adaptation Trainer")
    feat_ext = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
    task_head = nn.Linear(32, 3)
    disc = DomainDiscriminator(32)
    trainer = DomainAdaptationTrainer(feat_ext, task_head, disc)

    source_data = torch.randn(16, 10)
    source_labels = torch.randint(0, 3, (16,))
    target_data = torch.randn(16, 10)

    losses = trainer.compute_loss(source_data, source_labels, target_data)
    rep("Domain adaptation loss computed",
        "total_loss" in losses and "domain_loss" in losses)
    rep("Losses finite",
        bool(torch.isfinite(losses["total_loss"])) and
        bool(torch.isfinite(losses["domain_loss"])))

    # T7: Clinical Feature Vector
    print("\n[T7] Clinical Feature Vector")
    features = ClinicalFeatureVector(
        symptom_type=NeurologicalSymptom.PAIN,
        severity_score=7.5,
        duration_days=180,
        frequency_per_week=5.0,
        functional_impact=6.0,
        psychological_stressors=["work_stress", "family_issues"],
        medical_rule_outs=["neurological_exam", "imaging"],
    )
    tensor = features.to_tensor()
    rep("Feature vector to tensor", tensor.shape == (6,))
    rep("Feature values normalized",
        bool((tensor >= 0).all()) and bool((tensor <= 1.5).all()))

    # T8: Clinical Research Framework
    print("\n[T8] Clinical Research Framework")
    result = ClinicalResearchFramework.differentiate_somatic_vs_conversion(features)
    rep("Framework produces result", "research_notes" in result)
    rep("Result has disclaimer", "NOT A DIAGNOSIS" in result["disclaimer"])
    rep("DSM-5 criteria included",
        "somatic_symptom_disorder_criteria" in result and
        "conversion_disorder_criteria" in result)

    # T9: Sleepwalking Risk Factors
    print("\n[T9] Sleepwalking Risk Factors")
    sw_factors = SleepwalkingRiskFactors(
        sleep_deprivation=True,
        stress=True,
        family_history=True,
        unlocked_doors=True,
    )
    sw_result = NeuroAssessmentPipeline().assess_sleepwalking_risk(sw_factors)
    rep("Sleepwalking factors counted", sw_result["risk_factor_count"] == 4)
    rep("Sleepwalking result has disclaimer",
        "REFERENCE INFORMATION" in sw_result["disclaimer"])

    # T10: World Model
    print("\n[T10] World Model")
    wm = NeuroWorldModel(input_dim=6, latent_dim=32, output_dim=6)
    x_seq = torch.randn(2, 5, 6)  # batch=2, time=5, features=6
    with torch.no_grad():
        wm_out = wm(x_seq, n_future=3)
    rep("World model predictions shape",
        wm_out["predictions"].shape == (2, 3, 6))
    rep("World model uncertainties positive",
        bool((wm_out["uncertainties"] >= 0).all()))

    # T11: LLM Module (no API keys)
    print("\n[T11] LLM Module (no API keys)")
    llm = LLMIntegrationModule()
    llm_result = llm.analyze_clinical_text("Test clinical note")
    rep("LLM module handles missing keys",
        llm_result["status"] == "unavailable")

    # T12: Full Pipeline
    print("\n[T12] Full Assessment Pipeline")
    pipeline = NeuroAssessmentPipeline(feature_dim=6, latent_dim=32)
    pipeline.setup_interpretability(model, model[4])
    pipeline.setup_domain_adaptation(feat_ext, task_head, disc)

    input_t = torch.randn(6)
    full_result = pipeline.run_full_assessment(
        input_t,
        clinical_features=features,
        sleepwalking_factors=sw_factors,
    )
    rep("Full pipeline produces result", "feature_analysis" in full_result)
    rep("Full pipeline has disclaimer",
        "RESEARCH SCAFFOLD" in full_result["disclaimer"])

    # T13: DSM-5 Criteria
    print("\n[T13] DSM-5 Criteria Reference")
    ssd_criteria = ClinicalResearchFramework.get_criteria(
        SomaticCategory.SOMATIC_SYMPTOM_DISORDER)
    cd_criteria = ClinicalResearchFramework.get_criteria(
        SomaticCategory.CONVERSION_DISORDER)
    rep("SSD criteria loaded", "criteria" in ssd_criteria)
    rep("Conversion criteria loaded", "criteria" in cd_criteria)
    rep("SSD code present", "code" in ssd_criteria)
    rep("Conversion code present", "code" in cd_criteria)

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THIS MODULE PROVIDES
    • Grad-CAM and Integrated Gradients for feature interpretability
    • Adversarial domain adaptation framework
    • Clinical research data structures (DSM-5 criteria as reference)
    • LLM integration scaffold (requires API keys)
    • World model for predictive dynamics

  WHAT THIS MODULE IS NOT
    ❌ NOT a diagnostic tool
    ❌ NOT a clinical assessment tool
    ❌ NOT validated against clinical gold standards
    ❌ NOT approved by any regulatory body
    ❌ LLM outputs are research hypotheses, not clinical findings

  CLINICAL CAVEATS
    • Somatic Symptoms vs Conversion Disorder distinction requires
      clinical evaluation by trained professionals
    • Medical rule-outs MUST be completed before any psychological
      interpretation
    • Sleepwalking risk assessment requires sleep medicine specialist
    • No output from this system should be used for clinical decisions

  REQUIRED FOR ANY CLINICAL USE
    • IRB/Ethics Committee approval
    • Informed patient consent
    • Multi-site validation against gold standard diagnoses
    • Regulatory approval (FDA/CE/Thai FDA)
    • Oversight by qualified clinicians at every step
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neuro Interpretability + Domain Adaptation + LLM")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if self_test() else 1)

    parser.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
