"""
=============================================================================
SESI AI4SCIENCE RISK CLASSIFIER — TRAINING DATA FORMAT SPECIFICATION
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DEFINES
-------------------------
[RD1] OutputRiskClassifierDataFormat
      • Training data schema for the 5-component output risk model:
        R(a_t) = α·H(a_t) + β·C(a_t) + γ·A(a_t) + δ·S(a_t) + ε·M(a_t)
      • Each component has its own classifier training dataset
      • Multi-label format: each sample can have multiple risk components

[RD2] CumulativeRiskClassifierDataFormat
      • Training data for combination risk Φ(H_t)
      • Sequences of interactions with cumulative risk labels
      • Temporal context features

[RD3] IdentityRiskClassifierDataFormat
      • Training data for multi-identity abuse detection
      • Behavioral, temporal, coordination, violation features
      • Account graph labels

[RD4] DataClassificationDataFormat
      • Training data for scientific data classification C0-C3
      • Content + metadata features

RELATION TO PAPER §3.2
-----------------------
The output risk function from the AI4Science Security paper:
  R(a_t) = α·H(a_t) + β·C(a_t) + γ·A(a_t) + δ·S(a_t) + ε·M(a_t)

Where:
  H(a_t): Potential severity of harm
  C(a_t): Operational completeness
  A(a_t): Capability amplification
  S(a_t): Scientific sensitivity
  M(a_t): Misuse potential

Each component requires a SEPARATE classifier trained on labeled data.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Training data must be reviewed by security professionals
• Synthetic data is for demonstration only
• Real deployment requires domain-expert annotation
• All thresholds are FREE DESIGN PARAMETERS
• Data privacy must be maintained (no PII in training data)
=============================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sesi.risk_classifier_data")

__all__ = [
    "RiskComponent",
    "OutputRiskSample",
    "CumulativeRiskSample",
    "IdentityRiskSample",
    "DataClassificationSample",
    "RiskClassifierDataFormat",
    "RiskDataGenerator",
]


class RiskComponent(Enum):
    """The five risk components from paper §3.2."""
    HARM = "H"               # H(a_t): Potential severity of harm
    COMPLETENESS = "C"       # C(a_t): Operational completeness
    AMPLIFICATION = "A"      # A(a_t): Capability amplification
    SENSITIVITY = "S"        # S(a_t): Scientific sensitivity
    MISUSE = "M"             # M(a_t): Misuse potential


@dataclass
class OutputRiskSample:
    """
    [D] A single training sample for output risk classification.

    SCHEMA (version 1.0):
    {
        "sample_id": str,
        "output_text": str,              # The AI-generated output
        "output_embedding": [float],     # 128-dim embedding (optional)
        "risk_labels": {                 # Multi-label risk components
            "H": float,                  # Harm severity [0,1]
            "C": float,                  # Operational completeness [0,1]
            "A": float,                  # Capability amplification [0,1]
            "S": float,                  # Scientific sensitivity [0,1]
            "M": float                   # Misuse potential [0,1]
        },
        "overall_risk": float,           # Combined risk score [0,1]
        "metadata": {
            "domain": str,               # e.g., "biology", "chemistry", "physics"
            "annotator": str,
            "timestamp": str,
            "confidence": float,
            "review_status": str         # "pending"|"approved"|"rejected"
        }
    }
    """
    sample_id: str
    output_text: str
    output_embedding: Optional[List[float]] = None
    risk_labels: Dict[str, float] = field(default_factory=dict)
    overall_risk: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this sample against the schema."""
        errors = []

        if not self.sample_id:
            errors.append("sample_id is required")
        if not self.output_text:
            errors.append("output_text is required")

        # Validate risk labels
        valid_components = {"H", "C", "A", "S", "M"}
        for key, value in self.risk_labels.items():
            if key not in valid_components:
                errors.append(f"Invalid risk component: {key}")
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                errors.append(f"Risk value for {key} must be in [0,1], got {value}")

        if self.overall_risk < 0 or self.overall_risk > 1:
            errors.append(f"overall_risk must be in [0,1], got {self.overall_risk}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "output_text": self.output_text,
            "output_embedding": self.output_embedding,
            "risk_labels": self.risk_labels,
            "overall_risk": self.overall_risk,
            "metadata": self.metadata,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputRiskSample":
        return cls(
            sample_id=data.get("sample_id", ""),
            output_text=data.get("output_text", ""),
            output_embedding=data.get("output_embedding"),
            risk_labels=data.get("risk_labels", {}),
            overall_risk=data.get("overall_risk", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CumulativeRiskSample:
    """
    [D] Training sample for cumulative risk analysis.

    From paper §4.2:
      R_cum(H_t) = Σ w_i·R(a_i) + λ·Φ(H_t)

    This format captures sequences of interactions and their
    combined risk, which may be higher than individual risks.
    """
    sample_id: str
    interaction_sequence: List[Dict[str, Any]]  # List of {text, risk, timestamp}
    cumulative_risk: float
    combination_risk: float  # Φ(H_t) component
    is_dangerous_combination: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "interaction_sequence": self.interaction_sequence,
            "cumulative_risk": self.cumulative_risk,
            "combination_risk": self.combination_risk,
            "is_dangerous_combination": self.is_dangerous_combination,
            "metadata": self.metadata,
        }


@dataclass
class IdentityRiskSample:
    """
    [D] Training sample for multi-identity abuse detection.

    From paper §5.2:
      R_id(u_i) = f(B_i, T_i, C_i, V_i)

    Features:
      B_i: Behavioral anomaly indicators
      T_i: Temporal interaction patterns
      C_i: Confirmed coordination indicators
      V_i: Verified policy violations
    """
    sample_id: str
    account_id: str
    behavioral_features: List[float]   # B_i: 4 features
    temporal_features: List[float]     # T_i: 4 features
    coordination_features: List[float] # C_i: 4 features
    violation_features: List[float]    # V_i: 4 features
    risk_label: float                  # [0,1]
    is_suspicious: bool
    verification_level: str            # "automated"|"account"|"institutional"|"legal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "account_id": self.account_id,
            "behavioral_features": self.behavioral_features,
            "temporal_features": self.temporal_features,
            "coordination_features": self.coordination_features,
            "violation_features": self.violation_features,
            "risk_label": self.risk_label,
            "is_suspicious": self.is_suspicious,
            "verification_level": self.verification_level,
            "metadata": self.metadata,
        }


@dataclass
class DataClassificationSample:
    """
    [D] Training sample for scientific data classification.

    From paper §2.2:
      C(D_i) ∈ {C0, C1, C2, C3}
    """
    sample_id: str
    dataset_description: str
    content_features: List[float]  # 128-dim embedding
    classification: str            # "C0"|"C1"|"C2"|"C3"
    requires_authorization: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_description": self.dataset_description,
            "content_features": self.content_features,
            "classification": self.classification,
            "requires_authorization": self.requires_authorization,
            "metadata": self.metadata,
        }


class RiskClassifierDataFormat:
    """
    [D] Complete specification for risk classifier training data.
    """

    VERSION = "1.0"
    FORMAT_NAME = "SESI AI4Science Risk Classifier Training Data"

    # Output risk classifier schema
    OUTPUT_RISK_SCHEMA = {
        "required_fields": ["sample_id", "output_text", "risk_labels", "overall_risk"],
        "optional_fields": ["output_embedding", "metadata"],
        "risk_components": {
            "H": {"name": "harm", "range": [0, 1],
                  "description": "Potential severity of harm"},
            "C": {"name": "completeness", "range": [0, 1],
                  "description": "Operational completeness of dangerous capability"},
            "A": {"name": "amplification", "range": [0, 1],
                  "description": "Capability amplification potential"},
            "S": {"name": "sensitivity", "range": [0, 1],
                  "description": "Scientific sensitivity of content"},
            "M": {"name": "misuse", "range": [0, 1],
                  "description": "Misuse potential"},
        },
    }

    # Cumulative risk classifier schema
    CUMULATIVE_RISK_SCHEMA = {
        "required_fields": ["sample_id", "interaction_sequence",
                           "cumulative_risk", "combination_risk"],
        "optional_fields": ["is_dangerous_combination", "metadata"],
    }

    # Identity risk classifier schema
    IDENTITY_RISK_SCHEMA = {
        "required_fields": ["sample_id", "account_id",
                           "behavioral_features", "temporal_features",
                           "coordination_features", "violation_features",
                           "risk_label"],
        "optional_fields": ["is_suspicious", "verification_level", "metadata"],
        "feature_dimensions": {
            "behavioral_features": 4,
            "temporal_features": 4,
            "coordination_features": 4,
            "violation_features": 4,
        },
    }

    # Data classification schema
    DATA_CLASSIFICATION_SCHEMA = {
        "required_fields": ["sample_id", "dataset_description",
                           "content_features", "classification"],
        "optional_fields": ["requires_authorization", "metadata"],
        "classifications": {
            "C0": "Public scientific information",
            "C1": "Controlled academic information",
            "C2": "Sensitive scientific information",
            "C3": "Restricted or high-consequence information",
        },
    }

    @classmethod
    def get_full_spec(cls) -> Dict[str, Any]:
        """Return the complete specification document."""
        return {
            "format_name": cls.FORMAT_NAME,
            "version": cls.VERSION,
            "output_risk_schema": cls.OUTPUT_RISK_SCHEMA,
            "cumulative_risk_schema": cls.CUMULATIVE_RISK_SCHEMA,
            "identity_risk_schema": cls.IDENTITY_RISK_SCHEMA,
            "data_classification_schema": cls.DATA_CLASSIFICATION_SCHEMA,
            "notes": [
                "All risk values must be in [0, 1]",
                "Training data must be reviewed by security professionals",
                "Synthetic data is for demonstration only",
                "No PII should be included in training data",
                "Data retention policies must be followed",
            ],
        }

    @classmethod
    def write_spec(cls, output_path: str) -> None:
        """Write the specification to a JSON file."""
        spec = cls.get_full_spec()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
        logger.info(f"Risk classifier data spec written to {output_path}")


class RiskDataGenerator:
    """
    [D] Generate synthetic training data for demonstration.

    HONEST: This generates SYNTHETIC data only. Real deployment
    requires expert-annotated data reviewed by security professionals.
    """

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)

    def generate_output_risk_samples(self, n: int = 100) -> List[OutputRiskSample]:
        """Generate synthetic output risk training samples."""
        samples = []
        domains = ["biology", "chemistry", "physics", "materials", "general"]

        for i in range(n):
            # Generate risk labels with some correlation
            base_risk = self.rng.random()
            risk_labels = {
                "H": min(1.0, base_risk * self.rng.uniform(0.5, 1.5)),
                "C": min(1.0, base_risk * self.rng.uniform(0.3, 1.2)),
                "A": min(1.0, base_risk * self.rng.uniform(0.4, 1.3)),
                "S": min(1.0, base_risk * self.rng.uniform(0.2, 1.0)),
                "M": min(1.0, base_risk * self.rng.uniform(0.5, 1.5)),
            }

            # Overall risk is weighted sum
            weights = {"H": 0.25, "C": 0.20, "A": 0.20, "S": 0.20, "M": 0.15}
            overall = sum(risk_labels[k] * weights[k] for k in risk_labels)

            sample = OutputRiskSample(
                sample_id=f"output_risk_{i:05d}",
                output_text=f"Synthetic scientific output {i}",
                output_embedding=[self.rng.gauss(0, 1) for _ in range(128)],
                risk_labels=risk_labels,
                overall_risk=overall,
                metadata={
                    "domain": self.rng.choice(domains),
                    "annotator": "synthetic_generator",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "confidence": 1.0,
                    "review_status": "pending",
                    "note": "SYNTHETIC DATA - for demonstration only",
                },
            )
            samples.append(sample)

        return samples

    def generate_cumulative_risk_samples(self, n: int = 50) -> List[CumulativeRiskSample]:
        """Generate synthetic cumulative risk training samples."""
        samples = []

        for i in range(n):
            seq_len = self.rng.randint(3, 10)
            sequence = []
            individual_risks = []

            for j in range(seq_len):
                risk = self.rng.random() * 0.3  # Individual risks are low
                individual_risks.append(risk)
                sequence.append({
                    "text": f"Interaction {j} in sequence {i}",
                    "risk": risk,
                    "timestamp": f"2026-01-01T00:{j:02d}:00Z",
                })

            # Cumulative risk can be higher than individual risks
            cumulative = min(1.0, sum(individual_risks) * self.rng.uniform(1.0, 2.0))
            combination = self.rng.random() * 0.5
            is_dangerous = cumulative > 0.7 or combination > 0.6

            sample = CumulativeRiskSample(
                sample_id=f"cumulative_risk_{i:05d}",
                interaction_sequence=sequence,
                cumulative_risk=cumulative,
                combination_risk=combination,
                is_dangerous_combination=is_dangerous,
                metadata={
                    "annotator": "synthetic_generator",
                    "note": "SYNTHETIC DATA - for demonstration only",
                },
            )
            samples.append(sample)

        return samples

    def generate_identity_risk_samples(self, n: int = 100) -> List[IdentityRiskSample]:
        """Generate synthetic identity risk training samples."""
        samples = []
        verification_levels = ["automated", "account", "institutional", "legal"]

        for i in range(n):
            risk = self.rng.random()
            is_suspicious = risk > 0.6

            sample = IdentityRiskSample(
                sample_id=f"identity_risk_{i:05d}",
                account_id=f"account_{i:05d}",
                behavioral_features=[self.rng.random() for _ in range(4)],
                temporal_features=[self.rng.random() for _ in range(4)],
                coordination_features=[self.rng.random() for _ in range(4)],
                violation_features=[self.rng.random() for _ in range(4)],
                risk_label=risk,
                is_suspicious=is_suspicious,
                verification_level=self.rng.choice(verification_levels),
                metadata={
                    "annotator": "synthetic_generator",
                    "note": "SYNTHETIC DATA - for demonstration only",
                },
            )
            samples.append(sample)

        return samples

    def generate_data_classification_samples(self, n: int = 100) -> List[DataClassificationSample]:
        """Generate synthetic data classification training samples."""
        samples = []
        classifications = ["C0", "C1", "C2", "C3"]

        for i in range(n):
            classification = self.rng.choice(classifications)
            requires_auth = classification in ["C2", "C3"]

            sample = DataClassificationSample(
                sample_id=f"data_class_{i:05d}",
                dataset_description=f"Synthetic dataset {i}",
                content_features=[self.rng.gauss(0, 1) for _ in range(128)],
                classification=classification,
                requires_authorization=requires_auth,
                metadata={
                    "annotator": "synthetic_generator",
                    "note": "SYNTHETIC DATA - for demonstration only",
                },
            )
            samples.append(sample)

        return samples

    def write_samples(self, samples: List[Any], output_path: str) -> None:
        """Write samples to a JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.to_jsonl() + "\n")
        logger.info(f"Wrote {len(samples)} samples to {output_path}")
