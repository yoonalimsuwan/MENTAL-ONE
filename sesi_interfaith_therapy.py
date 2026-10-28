"""
=============================================================================
SESI INTERFAITH THERAPY MODULE
Integrating Christian, Buddhist, Taoist, and Confucian ethical practices
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
-------------------------
[I1] Four-Tradition Ethical Framework:
     • Christianity: Love, compassion, forgiveness, neighbor-love
     • Buddhism: Compassion (karuṇā), mindfulness, reduction of suffering
     • Taoism: Wu wei, humility, harmony with natural order
     • Confucianism: Ren (humaneness), li (propriety), moral cultivation

[I2] Therapeutic Integration:
     • Identifies ETHICAL CONVERGENCE across traditions
     • Does NOT claim doctrinal identity
     • Uses "ethical universality" (paper §10) as common ground

[I3] Practical Applications:
     • Compassion cultivation (Christian + Buddhist)
     • Humility and restraint (Taoist + Christian)
     • Moral cultivation and relationships (Confucian)
     • Mindfulness and suffering reduction (Buddhist)

HONEST SCOPE (per paper §8, §14, §17)
--------------------------------------
• This module DOES NOT claim the 4 traditions are identical.
• It recognizes ETHICAL CONVERGENCE without metaphysical agreement.
• Each tradition maintains its DISTINCTIVE identity.
• A Christian interpretation of Buddhist compassion is a CHRISTIAN
  theological judgment, not a claim that Buddhism is Christian.
• This is a THERAPEUTIC FRAMEWORK, not a theological synthesis.

KEY PRINCIPLE (paper §14)
--------------------------
"The goal is not to erase differences but to determine where genuine
common ground exists and where further disagreement requires careful analysis."
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.interfaith_therapy")

__all__ = [
    "TherapeuticPractice",
    "TraditionPractice",
    "InterfaithTherapyFramework",
    "CompassionCultivation",
    "HumilityPractice",
    "MoralCultivation",
    "MindfulnessPractice",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

class Tradition(Enum):
    """The 4 traditions in this therapeutic framework."""
    CHRISTIANITY = "Christianity"
    BUDDHISM = "Buddhism"
    TAOISM = "Taoism"
    CONFUCIANISM = "Confucianism"


class TherapeuticGoal(Enum):
    """Therapeutic goals addressed by the framework."""
    COMPASSION = auto()
    HUMILITY = auto()
    MORAL_CULTIVATION = auto()
    MINDFULNESS = auto()
    RELATIONSHIP_HARMONY = auto()
    SUFFERING_REDUCTION = auto()


@dataclass
class TherapeuticPractice:
    """A specific therapeutic practice from a tradition."""
    tradition: Tradition
    name: str
    description: str
    goal: TherapeuticGoal
    source_text: str
    practical_steps: List[str]


# ═══════════════════════════════════════════════════════════════════════
# [I1] TRADITION-SPECIFIC PRACTICES
# ═══════════════════════════════════════════════════════════════════════

class TraditionPractice:
    """
    Curated therapeutic practices from each tradition.

    Sources:
      • Christianity: Gospel teachings, Pauline ethics (§3-4 of paper)
      • Buddhism: Four Noble Truths, bodhisattva ideal (§6, §8)
      • Taoism: Wu wei, natural harmony (§7)
      • Confucianism: Ren, li, moral cultivation (§7-8)
    """

    @staticmethod
    def get_christian_practices() -> List[TherapeuticPractice]:
        """Christian therapeutic practices."""
        return [
            TherapeuticPractice(
                tradition=Tradition.CHRISTIANITY,
                name="Love of Neighbor",
                description="Cultivate compassion and care for others as an expression of divine love",
                goal=TherapeuticGoal.COMPASSION,
                source_text="Gospel teaching: 'Love your neighbor as yourself'",
                practical_steps=[
                    "Reflect on encounters with others today",
                    "Identify opportunities for compassionate action",
                    "Practice empathy by imagining others' experiences",
                    "Offer concrete help to those in need",
                    "Pray or meditate on universal love",
                ],
            ),
            TherapeuticPractice(
                tradition=Tradition.CHRISTIANITY,
                name="Forgiveness",
                description="Release resentment and cultivate mercy",
                goal=TherapeuticGoal.COMPASSION,
                source_text="Lord's Prayer: 'Forgive us our trespasses as we forgive others'",
                practical_steps=[
                    "Identify sources of resentment",
                    "Reflect on your own need for forgiveness",
                    "Consciously choose to release anger",
                    "Practice empathy for those who have harmed you",
                    "Seek reconciliation where possible",
                ],
            ),
            TherapeuticPractice(
                tradition=Tradition.CHRISTIANITY,
                name="Humility before God",
                description="Recognize limits and cultivate trust",
                goal=TherapeuticGoal.HUMILITY,
                source_text="Beatitudes: 'Blessed are the poor in spirit'",
                practical_steps=[
                    "Acknowledge your limitations",
                    "Practice gratitude for gifts received",
                    "Resist the urge to control every outcome",
                    "Serve others without seeking recognition",
                    "Trust in a larger purpose",
                ],
            ),
        ]

    @staticmethod
    def get_buddhist_practices() -> List[TherapeuticPractice]:
        """Buddhist therapeutic practices."""
        return [
            TherapeuticPractice(
                tradition=Tradition.BUDDHISM,
                name="Metta (Loving-Kindness)",
                description="Cultivate unconditional goodwill toward all beings",
                goal=TherapeuticGoal.COMPASSION,
                source_text="Metta Sutta: 'May all beings be happy and safe'",
                practical_steps=[
                    "Begin with self-compassion",
                    "Extend goodwill to loved ones",
                    "Extend to neutral persons",
                    "Extend to difficult persons",
                    "Extend to all beings without distinction",
                ],
            ),
            TherapeuticPractice(
                tradition=Tradition.BUDDHISM,
                name="Mindfulness Meditation",
                description="Cultivate present-moment awareness to reduce suffering",
                goal=TherapeuticGoal.MINDFULNESS,
                source_text="Four Foundations of Mindfulness (Satipatthana)",
                practical_steps=[
                    "Sit comfortably and focus on breath",
                    "Observe body sensations without judgment",
                    "Notice thoughts and emotions as they arise",
                    "Return gently to present moment",
                    "Extend awareness to daily activities",
                ],
            ),
            TherapeuticPractice(
                tradition=Tradition.BUDDHISM,
                name="Karuṇā (Compassion)",
                description="Respond to suffering with active desire to help",
                goal=TherapeuticGoal.SUFFERING_REDUCTION,
                source_text="Bodhisattva ideal: relieve suffering of all beings",
                practical_steps=[
                    "Recognize suffering in yourself and others",
                    "Allow yourself to feel empathy",
                    "Generate desire to alleviate suffering",
                    "Take concrete action to help",
                    "Practice regularly to strengthen compassion",
                ],
            ),
        ]

    @staticmethod
    def get_taoist_practices() -> List[TherapeuticPractice]:
        """Taoist therapeutic practices."""
        return [
            TherapeuticPractice(
                tradition=Tradition.TAOISM,
                name="Wu Wei (Non-Forcing)",
                description="Act in harmony with natural flow, without excessive control",
                goal=TherapeuticGoal.HUMILITY,
                source_text="Tao Te Ching: 'The Tao does nothing, yet nothing is left undone'",
                practical_steps=[
                    "Observe natural patterns in your life",
                    "Identify where you are forcing outcomes",
                    "Practice allowing things to unfold",
                    "Act when action is natural, refrain when it is not",
                    "Cultivate patience and trust in the process",
                ],
            ),
            TherapeuticPractice(
                tradition=Tradition.TAOISM,
                name="Simplicity",
                description="Reduce complexity and return to essential nature",
                goal=TherapeuticGoal.HUMILITY,
                source_text="Tao Te Ching: 'Manifest plainness, embrace simplicity'",
                practical_steps=[
                    "Identify unnecessary complications",
                    "Reduce material possessions",
                    "Simplify daily routines",
                    "Focus on what is essential",
                    "Cultivate inner stillness",
                ],
            ),
        ]

    @staticmethod
    def get_confucian_practices() -> List[TherapeuticPractice]:
        """Confucian therapeutic practices."""
        return [
            TherapeuticPractice(
                tradition=Tradition.CONFUCIANISM,
                name="Ren (Humaneness)",
                description="Cultivate humane character and care for others",
                goal=TherapeuticGoal.MORAL_CULTIVATION,
                source_text="Analects: 'Ren is to love others'",
                practical_steps=[
                    "Practice empathy in daily interactions",
                    "Consider impact of actions on others",
                    "Cultivate virtues: kindness, respect, integrity",
                    "Reflect on relationships and responsibilities",
                    "Act with benevolence in all situations",
                ],
            ),
            TherapeuticPractice(
                tradition=Tradition.CONFUCIANISM,
                name="Li (Ritual Propriety)",
                description="Cultivate appropriate conduct in relationships",
                goal=TherapeuticGoal.RELATIONSHIP_HARMONY,
                source_text="Analects: 'Without li, respectfulness becomes labor'",
                practical_steps=[
                    "Learn appropriate forms of address",
                    "Practice respect for elders and authorities",
                    "Observe social customs and rituals",
                    "Balance formality with genuine care",
                    "Cultivate harmony through proper conduct",
                ],
            ),
            TherapeuticPractice(
                tradition=Tradition.CONFUCIANISM,
                name="Self-Cultivation",
                description="Continuous refinement of moral character",
                goal=TherapeuticGoal.MORAL_CULTIVATION,
                source_text="Great Learning: 'Cultivate personal life to regulate family'",
                practical_steps=[
                    "Daily reflection on conduct",
                    "Study moral exemplars",
                    "Practice virtues in small matters",
                    "Seek feedback from trusted others",
                    "Commit to lifelong learning",
                ],
            ),
        ]


# ═══════════════════════════════════════════════════════════════════════
# [I2] INTERFAITH THERAPY FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════

class InterfaithTherapyFramework:
    """
    [D] Unified framework integrating practices from 4 traditions.

    Method:
      1. Identify therapeutic goals
      2. Map practices from each tradition to goals
      3. Find ethical convergence (paper §9.2)
      4. Create integrated therapeutic plans
      5. Respect distinctive identity of each tradition

    HONEST SCOPE:
      • Does NOT claim traditions are identical
      • Uses ETHICAL CONVERGENCE as common ground
      • Maintains distinctive identity of each tradition
      • A Christian using Buddhist metta is making a CHRISTIAN choice
    """

    def __init__(self):
        self.practices = {
            Tradition.CHRISTIANITY: TraditionPractice.get_christian_practices(),
            Tradition.BUDDHISM: TraditionPractice.get_buddhist_practices(),
            Tradition.TAOISM: TraditionPractice.get_taoist_practices(),
            Tradition.CONFUCIANISM: TraditionPractice.get_confucian_practices(),
        }

    def get_practices_for_goal(self, goal: TherapeuticGoal) -> Dict[Tradition, List[TherapeuticPractice]]:
        """Get all practices addressing a specific therapeutic goal."""
        result = {}
        for tradition, practices in self.practices.items():
            matching = [p for p in practices if p.goal == goal]
            if matching:
                result[tradition] = matching
        return result

    def find_convergence(self, goal: TherapeuticGoal) -> Dict[str, Any]:
        """
        Find ethical convergence across traditions for a goal.

        Returns:
            dict with convergence analysis
        """
        practices_by_tradition = self.get_practices_for_goal(goal)

        if len(practices_by_tradition) < 2:
            return {
                "goal": goal.name,
                "convergence": False,
                "note": "Insufficient traditions address this goal",
            }

        # Identify common themes
        common_themes = []
        all_descriptions = []
        for tradition, practices in practices_by_tradition.items():
            for p in practices:
                all_descriptions.append(f"{tradition.value}: {p.description}")

        # Simple convergence detection (in production, use NLP)
        convergence_keywords = {
            TherapeuticGoal.COMPASSION: ["compassion", "love", "care", "kindness"],
            TherapeuticGoal.HUMILITY: ["humility", "restraint", "simplicity", "trust"],
            TherapeuticGoal.MORAL_CULTIVATION: ["cultivation", "virtue", "character", "reflection"],
            TherapeuticGoal.MINDFULNESS: ["awareness", "present", "attention", "observation"],
            TherapeuticGoal.RELATIONSHIP_HARMONY: ["relationship", "harmony", "respect", "propriety"],
            TherapeuticGoal.SUFFERING_REDUCTION: ["suffering", "relief", "alleviate", "help"],
        }

        keywords = convergence_keywords.get(goal, [])
        convergence_count = sum(
            1 for desc in all_descriptions
            if any(kw in desc.lower() for kw in keywords)
        )

        return {
            "goal": goal.name,
            "convergence": convergence_count >= 2,
            "n_traditions": len(practices_by_tradition),
            "practices": {t.value: [p.name for p in ps] for t, ps in practices_by_tradition.items()},
            "common_themes": keywords,
            "note": "Convergence detected at ETHICAL level, not metaphysical",
        }

    def create_integrated_plan(self, goals: List[TherapeuticGoal],
                               user_preference: Optional[Tradition] = None) -> Dict[str, Any]:
        """
        Create an integrated therapeutic plan.

        Args:
            goals: list of therapeutic goals
            user_preference: optional tradition preference

        Returns:
            dict with integrated plan
        """
        plan = {
            "goals": [],
            "practices": [],
            "schedule": [],
            "note": (
                "This plan integrates practices from multiple traditions. "
                "Users may adopt practices from any tradition regardless of "
                "their own religious identity (per paper §14)."
            ),
        }

        for goal in goals:
            convergence = self.find_convergence(goal)
            plan["goals"].append({
                "goal": goal.name,
                "convergence": convergence["convergence"],
                "traditions": list(convergence["practices"].keys()),
            })

            # Select practices
            practices_by_tradition = self.get_practices_for_goal(goal)
            for tradition, practices in practices_by_tradition.items():
                # If user has preference, weight toward that tradition
                if user_preference and tradition == user_preference:
                    selected = practices[:2]  # take more from preferred
                else:
                    selected = practices[:1]  # take one from others

                for p in selected:
                    plan["practices"].append({
                        "tradition": tradition.value,
                        "name": p.name,
                        "description": p.description,
                        "steps": p.practical_steps,
                    })

        return plan


# ═══════════════════════════════════════════════════════════════════════
# [I3] SPECIFIC THERAPEUTIC MODULES
# ═══════════════════════════════════════════════════════════════════════

class CompassionCultivation(nn.Module):
    """
    [D] Neural module for compassion cultivation.

    Integrates Christian love-of-neighbor and Buddhist metta/karuṇā.

    HONEST: This is a COMPUTATIONAL MODEL of compassion cultivation,
    not a claim that the model itself is compassionate.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # Self-compassion
        self.self_compassion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Compassion for others
        self.other_compassion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Universal compassion (all beings)
        self.universal_compassion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute compassion scores at three levels."""
        return {
            "self": self.self_compassion(x),
            "other": self.other_compassion(x),
            "universal": self.universal_compassion(x),
        }


class HumilityPractice(nn.Module):
    """
    [D] Neural module for humility cultivation.

    Integrates Christian humility and Taoist wu wei.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # Recognition of limits
        self.limit_recognition = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Non-forcing (wu wei)
        self.non_forcing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Service without recognition
        self.service = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {
            "limit_recognition": self.limit_recognition(x),
            "non_forcing": self.non_forcing(x),
            "service": self.service(x),
        }


class MoralCultivation(nn.Module):
    """
    [D] Neural module for moral cultivation.

    Integrates Confucian ren/li and Christian virtue ethics.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # Virtue development
        self.virtue = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 5),  # 5 virtues: kindness, respect, integrity, wisdom, courage
            nn.Softmax(dim=-1),
        )

        # Relationship harmony
        self.relationship = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {
            "virtue_distribution": self.virtue(x),
            "relationship_harmony": self.relationship(x),
        }


class MindfulnessPractice(nn.Module):
    """
    [D] Neural module for mindfulness cultivation.

    Based on Buddhist satipatthana (Four Foundations of Mindfulness).
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # Present-moment awareness
        self.present_awareness = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Non-judgmental observation
        self.non_judgment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Suffering recognition
        self.suffering_recognition = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {
            "present_awareness": self.present_awareness(x),
            "non_judgment": self.non_judgment(x),
            "suffering_recognition": self.suffering_recognition(x),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Interfaith Therapy Framework — Smoke Test")
    print("=" * 72)
    torch.manual_seed(42)

    # ── [I1] Tradition practices ──────────────────────────────────────
    print("\n  ── [I1] Tradition-Specific Practices ──")
    framework = InterfaithTherapyFramework()
    for tradition in Tradition:
        practices = framework.practices[tradition]
        print(f"    {tradition.value}: {len(practices)} practices")
        for p in practices[:2]:
            print(f"      • {p.name}: {p.goal.name}")

    # ── [I2] Convergence analysis ─────────────────────────────────────
    print("\n  ── [I2] Ethical Convergence Analysis ──")
    for goal in TherapeuticGoal:
        convergence = framework.find_convergence(goal)
        status = "✅" if convergence["convergence"] else "❌"
        print(f"    {status} {goal.name}: {convergence.get('n_traditions', 0)} traditions")

    # ── [I3] Integrated plan ──────────────────────────────────────────
    print("\n  ── [I3] Integrated Therapeutic Plan ──")
    goals = [TherapeuticGoal.COMPASSION, TherapeuticGoal.HUMILITY,
             TherapeuticGoal.MINDFULNESS]
    plan = framework.create_integrated_plan(goals, user_preference=Tradition.BUDDHISM)
    print(f"    Goals: {len(plan['goals'])}")
    print(f"    Practices: {len(plan['practices'])}")
    for p in plan["practices"][:3]:
        print(f"      • [{p['tradition']}] {p['name']}")

    # ── Neural modules ────────────────────────────────────────────────
    print("\n  ── Neural Modules ──")
    x = torch.randn(2, 128)

    compassion = CompassionCultivation()
    comp_out = compassion(x)
    print(f"    Compassion: self={comp_out['self'].mean().item():.3f}, "
          f"other={comp_out['other'].mean().item():.3f}, "
          f"universal={comp_out['universal'].mean().item():.3f}")

    humility = HumilityPractice()
    hum_out = humility(x)
    print(f"    Humility: limit={hum_out['limit_recognition'].mean().item():.3f}, "
          f"wu_wei={hum_out['non_forcing'].mean().item():.3f}")

    moral = MoralCultivation()
    mor_out = moral(x)
    print(f"    Moral: top virtue={mor_out['virtue_distribution'].argmax(dim=-1).tolist()}")

    mindfulness = MindfulnessPractice()
    mind_out = mindfulness(x)
    print(f"    Mindfulness: present={mind_out['present_awareness'].mean().item():.3f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE (per paper §8, §14, §17)")
    print("  • Does NOT claim 4 traditions are identical")
    print("  • Uses ETHICAL CONVERGENCE as common ground")
    print("  • Each tradition maintains DISTINCTIVE identity")
    print("  • Users may adopt practices regardless of religious identity")
    print("  • This is a THERAPEUTIC FRAMEWORK, not theological synthesis")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
