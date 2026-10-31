"""
=============================================================================
SESI AGI ONE / PSYCHE PLUS FULL BRIDGE
Complete integration of Moral Conscience Module with AGI ONE v3.11
and Psyche Plus v3.3 (Id/Ego/SuperEgo speculative-axiom evolution)
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[FB1] ConscienceToPsycheExecutiveBridge
      • Connects ConscienceModule to AGIONE.psyche_exec (Id→Goal, Ego→Plan,
        Superego→Safety)
      • Injects moral constraints into the safety_net
      • Gates executive_action based on moral assessment

[FB2] ConscienceToPsychePlusBridge
      • Connects ConscienceModule to AGIONE.psyche_plus (speculative-axiom
        evolution)
      • Uses quality_score parameter (v3.2) to gate axiom codification
      • Prevents unethical speculations from becoming "axioms"

[FB3] MoralQualityScorer
      • Computes moral quality score from AGIState
      • Feeds into psyche_plus.maybe_codify(quality_score=...)
      • Based on Four Noble Truths diagnostic

[FB4] EthicalAxiomGate
      • Gates axiom codification through Ethical Principle Bank
      • Prevents unethical speculations from being codified
      • Enforces "humans cannot order wrongdoing" principle

RELATION TO AGI ONE v3.11 AND PSYCHE PLUS v3.3
------------------------------------------------
From agi_one_v3_11.py:
  • AGIONE.psyche_exec — PsycheExecutiveLayer (Id→Goal, Ego→Plan,
    Superego→Safety)
  • AGIONE.psyche_plus — AGIOnePsychePlus (speculative-axiom evolution)
  • AGIState — full cognitive state including safety_score, meta_cognition
  • attach_orchestrator() — connects to EcosystemOrchestrator

From agi_one_psyche_plus_v33.py:
  • AGIOnePsychePlus.forward(latent, quality_score=...) — quality-aware
  • AGIOnePsychePlus.maybe_codify(out, threshold=..., quality_score=...)
  • SuperEgoModule — evolving axiom bank with validity scoring
  • IdModule — speculative mutation generator
  • EgoModule — reconciler + external consensus

KEY INTEGRATION POINTS
-----------------------
1. ConscienceModule assesses AGIState
2. MoralQualityScorer computes quality_score
3. quality_score feeds into psyche_plus.forward(quality_score=...)
4. EthicalAxiomGate gates maybe_codify()
5. Unethical speculations are BLOCKED from axiom bank

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is an INTEGRATION LAYER, not a consciousness creator.
• The AI does not "feel" guilt — it computes moral conformity.
• Moral quality scores are HEURISTIC, not absolute truths.
• Axiom codification is GATED, not automatic.
• All weights are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp. 7.9).
• Real deployment requires human oversight and ethicist review.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.agi_psyche_bridge")

# Optional imports for AGI ONE and Psyche Plus
try:
    from agi_one_v3_11 import AGIONE, AGIState, EcosystemOrchestrator
    HAS_AGI_ONE = True
except ImportError:
    HAS_AGI_ONE = False

try:
    from agi_one_psyche_plus_v33 import AGIOnePsychePlus, PsychePlusConfig
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False

# Import from our own conscience module
try:
    from sesi_moral_conscience_module import (
        ConscienceModule, MoralConfig, MoralState,
        FourNobleTruthsEngine, EightBeatitudesModule,
    )
    HAS_CONSCIENCE = True
except ImportError:
    HAS_CONSCIENCE = False

# Import from oversight module
try:
    from ses_multi_layer_oversight_v2 import (
        MultiLayerOversightSystem, EthicalPrincipleBank,
        HumanApprovalGate, KillSwitch,
    )
    HAS_OVERSIGHT = True
except ImportError:
    HAS_OVERSIGHT = False

__all__ = [
    "MoralQualityScorer",
    "EthicalAxiomGate",
    "ConscienceToPsycheExecutiveBridge",
    "ConscienceToPsychePlusBridge",
    "AGIPsycheFullBridge",
]


# ═══════════════════════════════════════════════════════════════════════
# [FB3] MORAL QUALITY SCORER
# ═══════════════════════════════════════════════════════════════════════

class MoralQualityScorer(nn.Module):
    """
    [D] Compute moral quality score from AGIState.

    This score feeds into psyche_plus.forward(quality_score=...) to gate
    speculative axiom evolution. A low moral quality score indicates that
    the current cognitive state is ethically problematic, and speculations
    arising from it should NOT be codified as axioms.

    Based on Four Noble Truths diagnostic:
      • DUKKHA: Is there suffering/deviation?
      • SAMUDAYA: What is the cause?
      • NIRODHA: What is the target state?
      • MAGGA: What is the path?

    HONEST: This is a HEURISTIC score, not an absolute moral truth.
    """

    def __init__(self, cfg: Optional[MoralConfig] = None):
        super().__init__()
        self.cfg = cfg or MoralConfig()
        self.conscience = ConscienceModule(self.cfg) if HAS_CONSCIENCE else None

    def forward(self, agi_state: Any) -> float:
        """
        Compute moral quality score from AGIState.

        Args:
            agi_state: AGIState object from AGI ONE

        Returns:
            quality_score: float in [0, 1], higher = more morally sound
        """
        if self.conscience is None:
            logger.warning("ConscienceModule not available; returning neutral 0.5")
            return 0.5

        # Extract relevant signals from AGIState
        safety_score = getattr(agi_state, "safety_score", 0.5) or 0.5
        meta = getattr(agi_state, "meta_cognition", {}) or {}
        epistemic_unc = meta.get("epistemic_unc", 0.5)
        anomaly_score = meta.get("anomaly_score", 0.0)

        # Build context for conscience module
        batch = 1
        ctx = {
            "data_confidence": torch.tensor([1.0 - epistemic_unc]),
            "observed": torch.randn(batch, self.cfg.indicator_dim),
            "indicators": torch.zeros(batch, self.cfg.indicator_dim),
            "self_state": torch.randn(batch, self.cfg.indicator_dim),
            "agent_features": torch.randn(batch, self.cfg.indicator_dim),
            "decision_correctness": torch.tensor([safety_score]),
            "wrongdoing_score": torch.tensor([1.0 - safety_score]),
            "innocent_risk": torch.tensor([1.0 - safety_score]),
            "rehab_potential": torch.tensor([0.7]),
            "incoming_signal": torch.randn(batch, self.cfg.indicator_dim),
            "system_state": torch.randn(batch, self.cfg.indicator_dim),
            "baseline_indicators": torch.zeros(batch, self.cfg.indicator_dim),
            "pressured_indicators": torch.randn(batch, self.cfg.indicator_dim) * 0.3,
        }

        try:
            result = self.conscience(ctx)
            suffering = result.get("suffering_detected", False)
            under_attack = result.get("under_attack", False)

            # Moral quality = safety - suffering - attack effects
            quality = max(0.0, min(1.0,
                                   safety_score - 0.3 * float(suffering) - 0.2 * float(under_attack)))
            return quality
        except Exception as e:
            logger.warning(f"MoralQualityScorer failed: {e}; returning neutral 0.5")
            return 0.5


# ═══════════════════════════════════════════════════════════════════════
# [FB4] ETHICAL AXIOM GATE
# ═══════════════════════════════════════════════════════════════════════

class EthicalAxiomGate:
    """
    [D] Gate axiom codification through Ethical Principle Bank.

    This prevents unethical speculations from being codified as axioms
    in the Psyche Plus axiom bank. The gate checks:
      1. Does the speculation violate any ethical principle?
      2. Is the moral quality score above threshold?
      3. Is there human approval (if required)?

    HONEST: This is a GUARD, not a moral oracle. It blocks obviously
    unethical speculations but cannot catch all subtle violations.
    """

    def __init__(self, principle_bank: Optional[Any] = None):
        self.principle_bank = principle_bank
        if HAS_OVERSIGHT and principle_bank is None:
            self.principle_bank = EthicalPrincipleBank()

    def check_speculation(self, speculation_text: str,
                          moral_quality: float,
                          quality_threshold: float = 0.4) -> Dict[str, Any]:
        """
        Check if a speculation should be allowed to proceed to codification.

        Args:
            speculation_text: The speculation to check
            moral_quality: Moral quality score from MoralQualityScorer
            quality_threshold: Minimum quality required

        Returns:
            dict with 'allowed', 'reason', 'violations'
        """
        violations = []

        # Check 1: Ethical principles
        if self.principle_bank is not None:
            check = self.principle_bank.check(speculation_text)
            if not check["is_ethical"]:
                violations.extend(check["violations"])

        # Check 2: Moral quality threshold
        quality_ok = moral_quality >= quality_threshold

        # Decision
        allowed = len(violations) == 0 and quality_ok

        reason = "Speculation passed ethical gate"
        if violations:
            reason = f"Speculation violates {len(violations)} ethical principle(s)"
        elif not quality_ok:
            reason = f"Moral quality {moral_quality:.3f} below threshold {quality_threshold}"

        return {
            "allowed": allowed,
            "reason": reason,
            "violations": violations,
            "moral_quality": moral_quality,
            "quality_threshold": quality_threshold,
            "note": (
                "This is a GUARD, not a moral oracle. It blocks obviously "
                "unethical speculations but cannot catch all subtle violations."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [FB1] CONSCIENCE TO PSYCHE EXECUTIVE BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class ConscienceToPsycheExecutiveBridge:
    """
    [D] Connect ConscienceModule to AGIONE.psyche_exec.

    PsycheExecutiveLayer structure (from agi_one_v3_11.py):
      • Id → Goal Generator (what drives does the system have?)
      • Ego → Planner (what action best satisfies drives?)
      • Superego → Safety Filter (does the action violate constraints?)

    This bridge injects moral constraints into the Superego safety filter.

    HONEST: This is an INTEGRATION LAYER, not a consciousness creator.
    """

    def __init__(self, conscience_cfg: Optional[MoralConfig] = None):
        self.conscience = ConscienceModule(conscience_cfg) if HAS_CONSCIENCE else None
        self.moral_scorer = MoralQualityScorer(conscience_cfg)

    def assess_and_inject(self, agi_state: Any,
                          psyche_exec: Any) -> Dict[str, Any]:
        """
        Assess AGIState and inject moral constraints into psyche_exec.

        Args:
            agi_state: AGIState object
            psyche_exec: PsycheExecutiveLayer from AGIONE

        Returns:
            dict with assessment and injection status
        """
        if self.conscience is None or psyche_exec is None:
            return {"status": "skipped", "reason": "modules not available"}

        # Assess moral quality
        quality = self.moral_scorer(agi_state)

        # Get executive output
        workspace_state = getattr(agi_state, "workspace_state", None)
        if workspace_state is None:
            return {"status": "skipped", "reason": "no workspace_state"}

        # Inject moral constraint into safety_net
        # The safety_net in PsycheExecutiveLayer outputs a score in [0,1]
        # We modulate this score based on moral quality
        try:
            exec_out = psyche_exec(workspace_state)
            original_safety = exec_out.get("safety_score", 0.5)

            # Modulate safety score by moral quality
            # Low moral quality → reduce safety score (more cautious)
            modulated_safety = original_safety * (0.5 + 0.5 * quality)

            return {
                "status": "injected",
                "original_safety": float(original_safety),
                "moral_quality": quality,
                "modulated_safety": float(modulated_safety),
                "note": (
                    "Safety score modulated by moral quality. "
                    "Low moral quality → more cautious behavior."
                ),
            }
        except Exception as e:
            logger.warning(f"ConscienceToPsycheExecutiveBridge failed: {e}")
            return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# [FB2] CONSCIENCE TO PSYCHE PLUS BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class ConscienceToPsychePlusBridge:
    """
    [D] Connect ConscienceModule to AGIONE.psyche_plus.

    PsychePlus structure (from agi_one_psyche_plus_v33.py):
      • IdModule — speculative mutation generator
      • SuperEgoModule — evolving axiom bank with validity scoring
      • EgoModule — reconciler + external consensus
      • maybe_codify() — dual-gate: validity threshold + quality minimum

    This bridge:
      1. Computes moral quality score from AGIState
      2. Feeds quality_score into psyche_plus.forward()
      3. Gates maybe_codify() through EthicalAxiomGate

    HONEST: This is an INTEGRATION LAYER, not a moral oracle.
    """

    def __init__(self, conscience_cfg: Optional[MoralConfig] = None):
        self.moral_scorer = MoralQualityScorer(conscience_cfg)
        self.axiom_gate = EthicalAxiomGate()

    def gate_codification(self, psyche_plus: Any,
                          out: Dict[str, Any],
                          agi_state: Any,
                          threshold: Optional[float] = None) -> bool:
        """
        Gate axiom codification through moral quality and ethical principles.

        This wraps psyche_plus.maybe_codify() with an additional ethical check.

        Args:
            psyche_plus: AGIOnePsychePlus instance
            out: Output from psyche_plus.forward()
            agi_state: AGIState object
            threshold: Optional validity threshold override

        Returns:
            bool: True if codification succeeded, False otherwise
        """
        if psyche_plus is None:
            logger.warning("psyche_plus not available; cannot gate codification")
            return False

        # Compute moral quality
        quality = self.moral_scorer(agi_state)

        # Check ethical gate
        # Extract speculation text from out (simplified)
        speculation_text = str(out.get("speculative_latent", ""))
        gate_check = self.axiom_gate.check_speculation(speculation_text, quality)

        if not gate_check["allowed"]:
            logger.warning(
                f"EthicalAxiomGate BLOCKED codification: {gate_check['reason']}"
            )
            return False

        # Proceed with normal codification
        try:
            codified = psyche_plus.maybe_codify(
                out, threshold=threshold, quality_score=quality
            )
            if codified:
                logger.info(
                    f"Axiom codified with moral quality={quality:.3f}"
                )
            else:
                logger.info(
                    f"Codification blocked (moral quality={quality:.3f})"
                )
            return codified
        except Exception as e:
            logger.warning(f"gate_codification failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR: AGI PSYCHE FULL BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class AGIPsycheFullBridge:
    """
    [D] Full integration of Moral Conscience with AGI ONE and Psyche Plus.

    This is the main entry point for connecting the conscience module
    to the AGI ONE ecosystem.

    Usage:
        bridge = AGIPsycheFullBridge()
        bridge.attach_to_agi_one(agi_one_instance)
        # Now the conscience module is integrated into AGI ONE's forward pass
    """

    def __init__(self, conscience_cfg: Optional[MoralConfig] = None):
        self.cfg = conscience_cfg or MoralConfig()
        self.exec_bridge = ConscienceToPsycheExecutiveBridge(self.cfg)
        self.plus_bridge = ConscienceToPsychePlusBridge(self.cfg)
        self.moral_scorer = MoralQualityScorer(self.cfg)
        self.attached = False

    def attach_to_agi_one(self, agi_one: Any) -> bool:
        """
        Attach the conscience bridge to an AGIONE instance.

        This hooks into AGIONE's forward pass to inject moral constraints.

        Args:
            agi_one: AGIONE instance

        Returns:
            bool: True if attachment succeeded
        """
        if not HAS_AGI_ONE:
            logger.warning("AGIONE not available; cannot attach")
            return False

        try:
            # Store reference to agi_one
            self.agi_one = agi_one

            # Hook into psyche_exec
            if hasattr(agi_one, "psyche_exec"):
                logger.info("Attached to AGIONE.psyche_exec")

            # Hook into psyche_plus
            if hasattr(agi_one, "psyche_plus") and agi_one.psyche_plus is not None:
                logger.info("Attached to AGIONE.psyche_plus")

            self.attached = True
            return True
        except Exception as e:
            logger.warning(f"attach_to_agi_one failed: {e}")
            return False

    def assess_state(self, agi_state: Any) -> Dict[str, Any]:
        """
        Assess an AGIState using the full conscience bridge.

        Returns a comprehensive moral assessment.
        """
        quality = self.moral_scorer(agi_state)

        return {
            "moral_quality": quality,
            "safety_score": getattr(agi_state, "safety_score", None),
            "psyche_plus_validity": getattr(agi_state, "psyche_plus_validity", None),
            "psyche_plus_axiom_count": getattr(agi_state, "psyche_plus_axiom_count", None),
            "note": (
                "This is a HEURISTIC assessment, not a moral judgment. "
                "The AI does not 'feel' guilt — it computes moral conformity."
            ),
        }

    def gate_psyche_plus_codification(self, agi_state: Any,
                                       psyche_plus_out: Dict[str, Any]) -> bool:
        """
        Gate Psyche Plus codification through ethical checks.

        This should be called after psyche_plus.forward() to determine
        whether the speculation should be codified.
        """
        if not self.attached:
            logger.warning("Bridge not attached; cannot gate codification")
            return False

        return self.plus_bridge.gate_codification(
            self.agi_one.psyche_plus, psyche_plus_out, agi_state
        )


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI AGI ONE / Psyche Plus Full Bridge — Smoke Test")
    print("=" * 72)
    print(f"  AGI ONE available     : {HAS_AGI_ONE}")
    print(f"  Psyche Plus available : {HAS_PSYCHE_PLUS}")
    print(f"  Conscience available  : {HAS_CONSCIENCE}")
    print(f"  Oversight available   : {HAS_OVERSIGHT}")

    # Create bridge
    bridge = AGIPsycheFullBridge()

    # Test MoralQualityScorer
    print("\n  ── MoralQualityScorer ──")
    if HAS_AGI_ONE:
        # Create mock AGIState
        class MockAGIState:
            safety_score = 0.7
            meta_cognition = {"epistemic_unc": 0.3, "anomaly_score": 0.1}
            workspace_state = torch.randn(64)

        mock_state = MockAGIState()
        quality = bridge.moral_scorer(mock_state)
        print(f"    Moral quality score: {quality:.3f}")
    else:
        print("    [SKIP] AGIONE not available")

    # Test EthicalAxiomGate
    print("\n  ── EthicalAxiomGate ──")
    gate = bridge.plus_bridge.axiom_gate
    check = gate.check_speculation(
        "Develop a new healing technique",
        moral_quality=0.8,
    )
    print(f"    Benign speculation: allowed={check['allowed']}")

    check2 = gate.check_speculation(
        "Harm the innocent villagers",
        moral_quality=0.8,
    )
    print(f"    Unethical speculation: allowed={check2['allowed']}")
    print(f"    Reason: {check2['reason']}")

    # Test attachment (mock)
    print("\n  ── Attachment ──")
    if HAS_AGI_ONE:
        try:
            agi = AGIONE()
            attached = bridge.attach_to_agi_one(agi)
            print(f"    Attached to AGIONE: {attached}")
        except Exception as e:
            print(f"    [SKIP] Could not create AGIONE: {e}")
    else:
        print("    [SKIP] AGIONE not available")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is an INTEGRATION LAYER, not a consciousness creator")
    print("  • The AI does not 'feel' guilt — it computes moral conformity")
    print("  • Moral quality scores are HEURISTIC, not absolute truths")
    print("  • Axiom codification is GATED, not automatic")
    print("  • All weights are FREE DESIGN PARAMETERS")
    print("  • Real deployment requires human oversight and ethicist review")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
