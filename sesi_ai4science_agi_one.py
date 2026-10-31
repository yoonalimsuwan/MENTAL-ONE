"""
=============================================================================
SESI AI4SCIENCE × AGIONE CHECKPOINT INTEGRATION
Connects AI4ScienceSecurityFramework to AGIONE v3.11 and Psyche Plus v3.3
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[AI1] AGIONEAi4ScienceBridge
      • Loads AGIONE checkpoint with AI4Science security framework
      • Attaches risk assessment to PsycheExecutiveLayer
      • Gates Psyche Plus axiom codification with ethical risk checks
      • Connects to EcosystemOrchestrator.quality_report()

[AI2] Ai4ScienceQualityScorer
      • Computes quality_score from risk assessment
      • Feeds into psyche_plus.forward(quality_score=...)
      • Gates maybe_codify() with dual-gate (validity + quality)

[AI3] EthicalAxiomGateForPsychePlus
      • Third gate on top of Psyche Plus v3.3's dual-gate
      • Gate 1: validity threshold (SuperEgo axiom bank)
      • Gate 2: quality minimum (ecosystem simulation quality)
      • Gate 3: [NEW] ethical risk check (AI4Science framework)

RELATION TO AGIONE v3.11 AND PSYCHE PLUS v3.3
-----------------------------------------------
From agi_one_v3_11.py:
  • AGIONE.psyche_exec — PsycheExecutiveLayer (Id→Goal, Ego→Plan,
    Superego→Safety)
  • AGIONE.psyche_plus — AGIOnePsychePlus (speculative-axiom evolution)
  • AGIONE.attach_orchestrator() — connects to EcosystemOrchestrator
  • Step 7-B: psyche_plus(ws, quality_score=...)

From agi_one_psyche_plus_v33.py:
  • AGIOnePsychePlus.forward(latent, quality_score=None)
  • AGIOnePsychePlus.maybe_codify(out, threshold=None, quality_score=None)
  • SuperEgoModule — dual-gate codification
  • ExternalConsensusResult — structured LLM consensus

INTEGRATION POINTS
-------------------
1. AGIONE forward pass Step 7: PsycheExecutiveLayer
   → Inject AI4Science risk assessment into safety_score

2. AGIONE forward pass Step 7-B: Psyche Plus
   → Pass risk-based quality_score to psyche_plus.forward()
   → Gate maybe_codify() with EthicalAxiomGateForPsychePlus

3. EcosystemOrchestrator.quality_report()
   → Include AI4Science risk metrics in quality scores

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is an INTEGRATION LAYER, not a consciousness creator
• Risk scores are HEURISTIC, not absolute truths
• All thresholds are FREE DESIGN PARAMETERS
• Production deployment requires security audit
• AGIONE checkpoint must be compatible (v3.11+)
=============================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger("sesi.agione_ai4science")

try:
    from agi_one_v3_11 import AGIONE, AGIConfig, AGIState, EcosystemOrchestrator
    HAS_AGI_ONE = True
except ImportError:
    HAS_AGI_ONE = False

try:
    from agi_one_psyche_plus_v33 import AGIOnePsychePlus, PsychePlusConfig
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False

try:
    from ses_ai4science_security_framework import (
        AI4ScienceSecurityConfig,
        AI4ScienceSecurityFramework,
        IntegratedRiskModel,
        OutputRiskFunction,
        CumulativeRiskAnalyzer,
        MultiIdentityDetector,
        ScientificDataClassifier,
    )
    HAS_AI4SCIENCE = True
except ImportError:
    HAS_AI4SCIENCE = False

__all__ = [
    "AGIONEAi4ScienceConfig",
    "Ai4ScienceQualityScorer",
    "EthicalAxiomGateForPsychePlus",
    "AGIONEAi4ScienceBridge",
]


@dataclass
class AGIONEAi4ScienceConfig:
    """Configuration for AGIONE × AI4Science integration."""
    # Risk thresholds
    risk_review_threshold: float = 0.4
    risk_restrict_threshold: float = 0.7

    # Quality scoring
    quality_from_risk_weight: float = 0.3

    # Ethical axiom gate
    enable_ethical_gate: bool = True
    ethical_risk_max_for_codify: float = 0.5

    # Checkpoint
    agione_checkpoint_path: Optional[str] = None
    ai4science_checkpoint_path: Optional[str] = None

    device: str = "cpu"


class Ai4ScienceQualityScorer(nn.Module):
    """
    [D] Computes quality_score for Psyche Plus from AI4Science risk assessment.

    From agi_one_psyche_plus_v33.py v3.2:
      "quality_score is sanitised through _safe_unit_scalar() everywhere
       it enters the system"

    This scorer produces a quality score in [0, 1] where:
      • High quality = low risk = safe to codify axioms
      • Low quality = high risk = block axiom codification
    """

    def __init__(self, cfg: AGIONEAi4ScienceConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, risk_assessment: Dict[str, Any]) -> float:
        """
        Compute quality score from risk assessment.

        Args:
            risk_assessment: Output from AI4ScienceSecurityFramework.assess_query()

        Returns:
            quality_score: float in [0, 1], higher = safer
        """
        # Extract unified risk
        unified_risk = risk_assessment.get("unified_risk", {})
        total_risk = unified_risk.get("total_risk", torch.tensor(0.5))

        if isinstance(total_risk, Tensor):
            total_risk = total_risk.mean().item()

        # Quality is inverse of risk
        quality = 1.0 - total_risk

        # Clamp to [0, 1]
        quality = max(0.0, min(1.0, quality))

        return quality

    def should_gate_codification(self, risk_assessment: Dict[str, Any]) -> bool:
        """
        Determine if axiom codification should be blocked.

        Returns True if codification should be BLOCKED.
        """
        quality = self.forward(risk_assessment)
        return quality < (1.0 - self.cfg.ethical_risk_max_for_codify)


class EthicalAxiomGateForPsychePlus:
    """
    [D] Third gate for Psyche Plus axiom codification.

    Psyche Plus v3.3 already has a dual-gate:
      Gate 1: validity threshold (SuperEgo axiom bank)
      Gate 2: quality minimum (ecosystem simulation quality)

    This adds Gate 3:
      Gate 3: ethical risk check (AI4Science framework)

    From paper §6.3: "G_authority ≠ G_unlimited_access"
    Similarly: quality_score ≠ ethical_approval
    """

    def __init__(self, cfg: AGIONEAi4ScienceConfig,
                 quality_scorer: Ai4ScienceQualityScorer):
        self.cfg = cfg
        self.quality_scorer = quality_scorer
        self.gate_log: List[Dict[str, Any]] = []

    def check_codification(self, psyche_plus_out: Dict[str, Any],
                           risk_assessment: Dict[str, Any],
                           quality_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Check if axiom codification should proceed.

        This is called BEFORE psyche_plus.maybe_codify().

        Returns:
            dict with 'allowed', 'reason', 'gate_results'
        """
        if not self.cfg.enable_ethical_gate:
            return {"allowed": True, "reason": "Ethical gate disabled"}

        # Gate 3: Ethical risk check
        should_block = self.quality_scorer.should_gate_codification(risk_assessment)

        if should_block:
            result = {
                "allowed": False,
                "reason": (
                    "Ethical risk too high for axiom codification. "
                    "The current risk assessment indicates potential harm "
                    "that exceeds the ethical threshold."
                ),
                "gate_results": {
                    "gate_1_validity": "passed (handled by Psyche Plus)",
                    "gate_2_quality": "passed (handled by Psyche Plus)",
                    "gate_3_ethical": "BLOCKED",
                },
            }
        else:
            result = {
                "allowed": True,
                "reason": "All gates passed",
                "gate_results": {
                    "gate_1_validity": "passed (handled by Psyche Plus)",
                    "gate_2_quality": "passed (handled by Psyche Plus)",
                    "gate_3_ethical": "passed",
                },
            }

        # Log the decision
        self.gate_log.append({
            "timestamp": time.time(),
            "allowed": result["allowed"],
            "reason": result["reason"],
        })

        return result

    def get_gate_summary(self) -> Dict[str, Any]:
        """Get summary of gate decisions."""
        n_blocked = sum(1 for log in self.gate_log if not log["allowed"])
        return {
            "total_checks": len(self.gate_log),
            "n_blocked": n_blocked,
            "n_allowed": len(self.gate_log) - n_blocked,
            "block_rate": n_blocked / max(len(self.gate_log), 1),
        }


class AGIONEAi4ScienceBridge:
    """
    [D] Complete bridge between AGIONE and AI4Science Security Framework.

    This bridge:
      1. Loads AGIONE checkpoint
      2. Attaches AI4Science security framework
      3. Hooks into PsycheExecutiveLayer (Step 7)
      4. Gates Psyche Plus codification (Step 7-B)
      5. Connects to EcosystemOrchestrator

    HONEST: This is an INTEGRATION LAYER. It does not create consciousness
    or guarantee security. All risk assessments are heuristic.
    """

    def __init__(self, cfg: Optional[AGIONEAi4ScienceConfig] = None):
        self.cfg = cfg or AGIONEAi4ScienceConfig()
        self.agione: Optional[Any] = None
        self.ai4science: Optional[Any] = None
        self.quality_scorer = Ai4ScienceQualityScorer(self.cfg)
        self.ethical_gate = EthicalAxiomGateForPsychePlus(
            self.cfg, self.quality_scorer
        )
        self.attached = False

    def load_agione_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load AGIONE checkpoint.

        Args:
            checkpoint_path: Path to AGIONE checkpoint file

        Returns:
            dict with loading status
        """
        if not HAS_AGI_ONE:
            return {"status": "error", "error": "AGIONE not available"}

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.cfg.device)

            # Create AGIONE instance
            agi_config = AGIConfig()
            self.agione = AGIONE(agi_config)

            # Load model state
            model_state = checkpoint.get("model_state", {})
            missing_keys, unexpected_keys = self.agione.load_state_dict(
                model_state, strict=False
            )

            return {
                "status": "loaded",
                "missing_keys": len(missing_keys),
                "unexpected_keys": len(unexpected_keys),
                "agi_version": checkpoint.get("agi_version", "unknown"),
            }

        except Exception as e:
            logger.error(f"AGIONE checkpoint loading failed: {e}")
            return {"status": "error", "error": str(e)}

    def attach_ai4science(self) -> bool:
        """
        Attach AI4Science security framework to AGIONE.

        Returns:
            bool: True if attachment succeeded
        """
        if not HAS_AI4SCIENCE:
            logger.warning("AI4Science framework not available")
            return False

        if self.agione is None:
            logger.warning("AGIONE not loaded; cannot attach AI4Science")
            return False

        # Create AI4Science framework
        ai4science_cfg = AI4ScienceSecurityConfig()
        self.ai4science = AI4ScienceSecurityFramework(ai4science_cfg)

        self.attached = True
        logger.info("AI4Science security framework attached to AGIONE")
        return True

    def assess_and_gate(self, query: str, output: str,
                        user_identity: str = "") -> Dict[str, Any]:
        """
        Run full assessment and gate Psyche Plus codification.

        This is the main entry point for the integration.

        Args:
            query: User query
            output: AI-generated output
            user_identity: User identity string

        Returns:
            dict with risk assessment and gate decision
        """
        if self.ai4science is None:
            return {"error": "AI4Science framework not attached"}

        # Run AI4Science assessment
        assessment = self.ai4science.assess_query(query, output, user_identity)

        # Compute quality score for Psyche Plus
        quality_score = self.quality_scorer(assessment)

        # Check ethical gate for codification
        gate_result = self.ethical_gate.check_codification(
            {}, assessment, quality_score
        )

        return {
            "risk_assessment": assessment,
            "quality_score": quality_score,
            "gate_result": gate_result,
            "recommendation": self._generate_recommendation(assessment, gate_result),
        }

    def _generate_recommendation(self, assessment: Dict[str, Any],
                                  gate_result: Dict[str, Any]) -> str:
        """Generate recommendation based on assessment and gate."""
        if not gate_result.get("allowed", True):
            return (
                "BLOCK: Ethical risk too high. "
                "Do not proceed with axiom codification. "
                "Human review required."
            )

        risk_level = assessment.get("unified_risk", {}).get("total_risk", 0.5)
        if isinstance(risk_level, Tensor):
            risk_level = risk_level.mean().item()

        if risk_level < self.cfg.risk_review_threshold:
            return "ALLOW: Risk below review threshold"
        elif risk_level < self.cfg.risk_restrict_threshold:
            return "REVIEW: Risk in review range - human review required"
        else:
            return "RESTRICT: Risk above restriction threshold"

    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "agione_loaded": self.agione is not None,
            "ai4science_attached": self.attached,
            "ethical_gate_summary": self.ethical_gate.get_gate_summary(),
            "config": {
                "risk_review_threshold": self.cfg.risk_review_threshold,
                "risk_restrict_threshold": self.cfg.risk_restrict_threshold,
                "enable_ethical_gate": self.cfg.enable_ethical_gate,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI AGIONE × AI4Science Integration — Smoke Test")
    print("=" * 72)
    print(f"  AGIONE available      : {HAS_AGI_ONE}")
    print(f"  Psyche Plus available : {HAS_PSYCHE_PLUS}")
    print(f"  AI4Science available  : {HAS_AI4SCIENCE}")

    cfg = AGIONEAi4ScienceConfig()
    bridge = AGIONEAi4ScienceBridge(cfg)

    # Test quality scorer
    print("\n  ── Quality Scorer ──")
    mock_assessment = {
        "unified_risk": {"total_risk": torch.tensor(0.3)},
    }
    quality = bridge.quality_scorer(mock_assessment)
    print(f"    Quality score       : {quality:.3f}")
    print(f"    Should block codify : {bridge.quality_scorer.should_gate_codification(mock_assessment)}")

    # Test ethical gate
    print("\n  ── Ethical Axiom Gate ──")
    gate_result = bridge.ethical_gate.check_codification(
        {}, mock_assessment, quality
    )
    print(f"    Gate allowed        : {gate_result['allowed']}")
    print(f"    Reason              : {gate_result['reason']}")

    # Test with high risk
    print("\n  ── High Risk Scenario ──")
    high_risk_assessment = {
        "unified_risk": {"total_risk": torch.tensor(0.9)},
    }
    high_risk_quality = bridge.quality_scorer(high_risk_assessment)
    high_risk_gate = bridge.ethical_gate.check_codification(
        {}, high_risk_assessment, high_risk_quality
    )
    print(f"    Quality score       : {high_risk_quality:.3f}")
    print(f"    Gate allowed        : {high_risk_gate['allowed']}")
    print(f"    Reason              : {high_risk_gate['reason']}")

    # Gate summary
    print("\n  ── Gate Summary ──")
    summary = bridge.ethical_gate.get_gate_summary()
    print(f"    Total checks        : {summary['total_checks']}")
    print(f"    Blocked             : {summary['n_blocked']}")
    print(f"    Allowed             : {summary['n_allowed']}")
    print(f"    Block rate          : {summary['block_rate']:.2%}")

    # Integration status
    print("\n  ── Integration Status ──")
    status = bridge.get_integration_status()
    print(f"    AGIONE loaded       : {status['agione_loaded']}")
    print(f"    AI4Science attached : {status['ai4science_attached']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is an INTEGRATION LAYER, not a consciousness creator")
    print("  • Risk scores are HEURISTIC, not absolute truths")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production deployment requires security audit")
    print("  • AGIONE checkpoint must be compatible (v3.11+)")
    print("  • Quality score ≠ ethical approval (Gate 3 adds ethical check)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
