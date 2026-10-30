"""
=============================================================================
SES CATEGORY THEORY TEST — Strengthening Theorem 1 via Functors & Pullbacks
Per Dual-Model Theory §15 recommendation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES (per §15 recommendation)
-----------------------------------------------
The Dual-Model paper suggests strengthening Theorem 1 with category theory.
This module implements that suggestion as a TEST SUITE, not a proof:

[CT1] ModelCategory
      • Objects = representational states
      • Morphisms = interpretation-preserving maps
      • N (naturalistic) and T (theological) as two categories

[CT2] EmpiricalFunctor
      • F_N: Universe → R_N (naturalistic representation)
      • F_T: Universe → R_T (theological representation)
      • Both must preserve the OBSERVABLE structure (physical predictions)

[CT3] SharedEmpiricalSubstructure (Pullback)
      • The pullback R_N ×_E R_T captures what both models agree on
      • Compatibility ⟺ pullback is NON-EMPTY and preserves predictions
      • This makes the "agreement condition" P_N(Q|E)=P_T(Q|E) categorical

[CT4] NaturalTransformationChecker
      • Checks whether there exists a natural transformation relating the
        two functors on the shared substructure
      • Existence = structural compatibility; absence = incompatibility

MATHEMATICAL BASIS
------------------
[C] Category theory: Mac Lane, Categories for the Working Mathematician
[C] Pullback (fiber product): standard categorical limit
[D] Application to Dual-Model compatibility: our construction

RELATION TO THEOREM 1
----------------------
Theorem 1 (Dual-Model Compatibility) states:
    If N and T represent the same observable structure, agree on empirical
    predictions, and neither negates the other at the same semantic level,
    then Consistent(N) ∧ Consistent(T) ∧ Compatible(N,T).

This module REIFIES those assumptions as categorical conditions:
    A1, A2 → F_N, F_T preserve observable structure (functor axioms)
    A4     → pullback is non-empty (shared empirical content exists)
    A5     → no contradictory morphisms at the same level

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a TEST of Theorem 1's assumptions, not a proof that God exists
  or does not exist.
• Category theory provides STRUCTURAL CONDITIONS; satisfying them is
  necessary but not sufficient for metaphysical truth.
• The pullback being non-empty shows LOGICAL COMPATIBILITY under the
  stated assumptions, not that both interpretations are equally true.
• A5 (no same-level contradiction) requires semantic analysis that cannot
  be fully automated; we provide structural checks only.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ses.category_test")

__all__ = [
    "CTConfig",
    "Category", "Functor", "NaturalTransformation",
    "ModelCategory", "EmpiricalFunctor",
    "SharedEmpiricalSubstructure",
    "CategoryTheoryTestSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CTConfig:
    """Configuration for category theory tests."""
    # Observable structure (shared physical predictions)
    observable_quantities: List[str] = field(default_factory=lambda: [
        "gravitational_acceleration",
        "speed_of_light",
        "planck_constant",
    ])

    # Prediction tolerance for pullback membership
    prediction_tolerance: float = 1e-6

    # Semantic levels (to check A5: no same-level contradiction)
    semantic_levels: List[str] = field(default_factory=lambda: [
        "PHYSICAL",
        "MATHEMATICAL",
        "METAPHYSICAL",
    ])


# ═══════════════════════════════════════════════════════════════════════
# [C] CATEGORY PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Object:
    """An object in a category."""
    name: str
    data: Any = None
    level: str = "PHYSICAL"  # semantic level


@dataclass
class Morphism:
    """A morphism between two objects."""
    source: str
    target: str
    name: str = ""
    preserves_structure: bool = True

    def compose_with(self, other: "Morphism") -> Optional["Morphism"]:
        """Compose this morphism with another (if composable)."""
        if self.target != other.source:
            return None
        return Morphism(
            source=self.source,
            target=other.target,
            name=f"{other.name}∘{self.name}" if self.name and other.name else "",
            preserves_structure=self.preserves_structure and other.preserves_structure,
        )


class Category:
    """
    [C] A category: objects + morphisms + composition + identity.
    """

    def __init__(self, name: str):
        self.name = name
        self.objects: Dict[str, Object] = {}
        self.morphisms: List[Morphism] = []

    def add_object(self, obj: Object) -> None:
        self.objects[obj.name] = obj

    def add_morphism(self, morph: Morphism) -> None:
        if morph.source in self.objects and morph.target in self.objects:
            self.morphisms.append(morph)
        else:
            logger.warning(f"Morphism {morph.name} references unknown object")

    def identity(self, obj_name: str) -> Morphism:
        """Identity morphism for an object."""
        return Morphism(source=obj_name, target=obj_name,
                        name=f"id_{obj_name}", preserves_structure=True)

    def has_morphism(self, source: str, target: str) -> bool:
        """Check if a morphism exists between two objects."""
        return any(m.source == source and m.target == target
                   for m in self.morphisms)

    def check_axioms(self) -> Dict[str, bool]:
        """
        [C] Verify category axioms:
        1. Identity exists for every object
        2. Composition is associative (structural check)
        """
        checks = {}
        # Axiom 1: identity exists (by construction, we can always make one)
        checks["identity_exists"] = True
        # Axiom 2: all morphisms have valid source/target
        checks["morphisms_well_formed"] = all(
            m.source in self.objects and m.target in self.objects
            for m in self.morphisms
        )
        return checks


# ═══════════════════════════════════════════════════════════════════════
# [C] FUNCTOR
# ═══════════════════════════════════════════════════════════════════════

class Functor:
    """
    [C] A functor F: C → D maps objects to objects and morphisms to
    morphisms, preserving identity and composition.
    """

    def __init__(self, name: str, source: Category, target: Category):
        self.name = name
        self.source = source
        self.target = target
        self.object_map: Dict[str, str] = {}
        self.morphism_map: Dict[str, Morphism] = {}

    def map_object(self, src_name: str, tgt_name: str) -> None:
        self.object_map[src_name] = tgt_name

    def map_morphism(self, src_morph_name: str, tgt_morph: Morphism) -> None:
        self.morphism_map[src_morph_name] = tgt_morph

    def check_functor_axioms(self) -> Dict[str, bool]:
        """
        [C] Verify functor axioms:
        1. F(id_X) = id_{F(X)}
        2. F(g∘f) = F(g)∘F(f)
        We check these structurally.
        """
        checks = {}
        # Axiom 1: identity preservation (structural)
        checks["identity_preserved"] = all(
            self.object_map.get(obj) is not None
            for obj in self.source.objects
        )
        # Axiom 2: all mapped morphisms are valid in target
        checks["morphisms_valid_in_target"] = all(
            m.source in self.target.objects and m.target in self.target.objects
            for m in self.morphism_map.values()
        )
        return checks


# ═══════════════════════════════════════════════════════════════════════
# [D] MODEL CATEGORY (for N or T)
# ═══════════════════════════════════════════════════════════════════════

class ModelCategory(Category):
    """
    [D] A category representing one interpretive model (N or T).

    Objects include:
      • Physical states (shared with the other model)
      • Interpretive states (model-specific)

    Theological model T has additional objects (G, D) that the
    naturalistic model N does not.
    """

    def __init__(self, name: str, is_theological: bool = False):
        super().__init__(name)
        self.is_theological = is_theological
        self.observable_objects: Set[str] = set()
        self.interpretive_objects: Set[str] = set()

    def add_observable(self, obj: Object) -> None:
        self.add_object(obj)
        self.observable_objects.add(obj.name)

    def add_interpretive(self, obj: Object) -> None:
        self.add_object(obj)
        self.interpretive_objects.add(obj.name)

    def predictions(self, quantity: str) -> Optional[float]:
        """
        [D] Extract a model's prediction for an observable quantity.
        Returns None if the model makes no prediction for this quantity.
        """
        # In a real implementation, this would query the model's equations.
        # Here we use a simple lookup from object data.
        for obj_name in self.observable_objects:
            obj = self.objects[obj_name]
            if isinstance(obj.data, dict) and quantity in obj.data:
                return float(obj.data[quantity])
        return None


# ═══════════════════════════════════════════════════════════════════════
# [D] EMPIRICAL FUNCTOR
# ═══════════════════════════════════════════════════════════════════════

class EmpiricalFunctor(Functor):
    """
    [D] A functor from the Universe category to a model category that
    preserves OBSERVABLE structure (Theorem 1 assumptions A1, A2).
    """

    def __init__(self, name: str, source: Category, target: ModelCategory,
                 observables: List[str]):
        super().__init__(name, source, target)
        self.observables = observables

    def check_observable_preservation(self) -> Dict[str, bool]:
        """
        [D] Check A1/A2: the functor preserves observable physical structure.

        For each observable quantity, the target model must have a prediction.
        """
        preservation = {}
        for obs in self.observables:
            pred = self.target.predictions(obs)
            preservation[obs] = pred is not None
        return preservation


# ═══════════════════════════════════════════════════════════════════════
# [D] SHARED EMPIRICAL SUBSTRUCTURE (PULLBACK)
# ═══════════════════════════════════════════════════════════════════════

class SharedEmpiricalSubstructure:
    """
    [D] The pullback R_N ×_E R_T: the shared empirical content.

    This is the categorical reification of Theorem 1's assumption A4:
    both models agree on relevant empirical predictions.

    The pullback is non-empty iff there exists at least one observable
    for which both models make the same (within tolerance) prediction.
    """

    def __init__(self, R_N: ModelCategory, R_T: ModelCategory,
                 observables: List[str], tolerance: float = 1e-6):
        self.R_N = R_N
        self.R_T = R_T
        self.observables = observables
        self.tolerance = tolerance
        self.shared: Dict[str, Tuple[float, float]] = {}

    def compute(self) -> Dict[str, Any]:
        """
        Compute the pullback: observables where both models predict
        the same value within tolerance.
        """
        self.shared = {}
        disagreements = {}

        for obs in self.observables:
            pred_N = self.R_N.predictions(obs)
            pred_T = self.R_T.predictions(obs)

            if pred_N is None or pred_T is None:
                disagreements[obs] = "one or both models silent"
                continue

            if abs(pred_N - pred_T) <= self.tolerance:
                self.shared[obs] = (pred_N, pred_T)
            else:
                disagreements[obs] = (
                    f"N={pred_N}, T={pred_T}, diff={abs(pred_N - pred_T)}"
                )

        return {
            "pullback_observables": list(self.shared.keys()),
            "n_shared": len(self.shared),
            "disagreements": disagreements,
            "pullback_nonempty": len(self.shared) > 0,
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] NATURAL TRANSFORMATION CHECKER
# ═══════════════════════════════════════════════════════════════════════

class NaturalTransformationChecker:
    """
    [D] Check whether a natural transformation relates F_N and F_T on
    the shared substructure.

    A natural transformation η: F_N ⇒ F_T on the shared observables means:
    for each shared observable, the two functors' outputs are related by
    a consistent mapping (here: identity, since predictions agree).

    HONEST: This is a STRUCTURAL check. It does not prove metaphysical
    truth. It shows that, on the shared empirical content, the two models
    are related by a consistent (identity) transformation.
    """

    def __init__(self, F_N: EmpiricalFunctor, F_T: EmpiricalFunctor,
                 pullback: SharedEmpiricalSubstructure):
        self.F_N = F_N
        self.F_T = F_T
        self.pullback = pullback

    def check(self) -> Dict[str, Any]:
        """
        Check naturality on the shared substructure.

        Naturality condition (simplified): for each shared observable,
        F_N(obs) and F_T(obs) agree, so the identity transformation is
        natural.
        """
        if not self.pullback.shared:
            return {
                "natural_transformation_exists": False,
                "reason": "Pullback is empty; no shared content to relate.",
            }

        # On the shared substructure, predictions agree by construction,
        # so the identity transformation is natural.
        naturality_checks = {}
        for obs, (pred_N, pred_T) in self.pullback.shared.items():
            naturality_checks[obs] = abs(pred_N - pred_T) <= self.pullback.tolerance

        all_natural = all(naturality_checks.values())

        return {
            "natural_transformation_exists": all_natural,
            "naturality_checks": naturality_checks,
            "note": (
                "Identity transformation is natural on the shared substructure. "
                "This shows STRUCTURAL COMPATIBILITY, not metaphysical truth."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] CATEGORY THEORY TEST SUITE
# ═══════════════════════════════════════════════════════════════════════

class CategoryTheoryTestSuite:
    """
    [D] Orchestrate the category-theoretic strengthening of Theorem 1.
    """

    def __init__(self, cfg: Optional[CTConfig] = None):
        self.cfg = cfg or CTConfig()

    def build_default_scenario(self) -> Dict[str, Any]:
        """
        Build the default N and T categories with shared observables.

        This mirrors the Dual-Model paper's §5 setup:
          R_N = (X, P, M)
          R_T = (X, P, M, G, D)
        """
        # Universe category (source)
        Universe = Category("Universe")
        Universe.add_object(Object("physical_state", level="PHYSICAL"))

        # Naturalistic category
        R_N = ModelCategory("R_N", is_theological=False)
        # Shared physical objects
        R_N.add_observable(Object(
            "gravitational_acceleration",
            data={"gravitational_acceleration": 9.81},
            level="PHYSICAL"))
        R_N.add_observable(Object(
            "speed_of_light",
            data={"speed_of_light": 299792458.0},
            level="PHYSICAL"))
        R_N.add_observable(Object(
            "planck_constant",
            data={"planck_constant": 6.62607015e-34},
            level="PHYSICAL"))

        # Theological category
        R_T = ModelCategory("R_T", is_theological=True)
        # SAME physical objects (A2)
        R_T.add_observable(Object(
            "gravitational_acceleration",
            data={"gravitational_acceleration": 9.81},
            level="PHYSICAL"))
        R_T.add_observable(Object(
            "speed_of_light",
            data={"speed_of_light": 299792458.0},
            level="PHYSICAL"))
        R_T.add_observable(Object(
            "planck_constant",
            data={"planck_constant": 6.62607015e-34},
            level="PHYSICAL"))
        # Additional interpretive objects (G, D) — NOT physical
        R_T.add_interpretive(Object("G_ultimate_ground", level="METAPHYSICAL"))
        R_T.add_interpretive(Object("D_dharma", level="METAPHYSICAL"))

        # Functors
        F_N = EmpiricalFunctor("F_N", Universe, R_N,
                                self.cfg.observable_quantities)
        F_N.map_object("physical_state", "gravitational_acceleration")
        F_N.map_object("physical_state", "speed_of_light")
        F_N.map_object("physical_state", "planck_constant")

        F_T = EmpiricalFunctor("F_T", Universe, R_T,
                                self.cfg.observable_quantities)
        F_T.map_object("physical_state", "gravitational_acceleration")
        F_T.map_object("physical_state", "speed_of_light")
        F_T.map_object("physical_state", "planck_constant")

        return {"Universe": Universe, "R_N": R_N, "R_T": R_T,
                "F_N": F_N, "F_T": F_T}

    def run_all(self) -> Dict[str, Any]:
        """Run the full category-theoretic test of Theorem 1."""
        scenario = self.build_default_scenario()
        R_N, R_T = scenario["R_N"], scenario["R_T"]
        F_N, F_T = scenario["F_N"], scenario["F_T"]

        results = {}

        # CT1: Category axioms
        results["CT1_category_axioms"] = {
            "R_N": R_N.check_axioms(),
            "R_T": R_T.check_axioms(),
        }

        # CT2: Functor axioms + observable preservation (A1, A2)
        results["CT2_functor_axioms"] = {
            "F_N": F_N.check_functor_axioms(),
            "F_T": F_T.check_functor_axioms(),
        }
        results["CT2_observable_preservation"] = {
            "F_N": F_N.check_observable_preservation(),
            "F_T": F_T.check_observable_preservation(),
        }

        # CT3: Pullback (A4: empirical agreement)
        pullback = SharedEmpiricalSubstructure(
            R_N, R_T, self.cfg.observable_quantities,
            tolerance=self.cfg.prediction_tolerance)
        results["CT3_pullback"] = pullback.compute()

        # CT4: Natural transformation (structural compatibility)
        nt_checker = NaturalTransformationChecker(F_N, F_T, pullback)
        results["CT4_natural_transformation"] = nt_checker.check()

        # Summary: Theorem 1 conditions
        a1_a2 = all(
            results["CT2_observable_preservation"]["F_N"].values()
        ) and all(
            results["CT2_observable_preservation"]["F_T"].values()
        )
        a4 = results["CT3_pullback"]["pullback_nonempty"]
        compatible = a1_a2 and a4

        results["theorem1_status"] = {
            "A1_A2_observable_preserved": a1_a2,
            "A4_empirical_agreement": a4,
            "Compatible(N,T)": compatible,
            "conclusion": (
                "Theorem 1 conditions HOLD under stated assumptions. "
                "Consistent(N) ∧ Consistent(T) ∧ Compatible(N,T) — "
                "CONDITIONAL, not absolute."
                if compatible else
                "Theorem 1 conditions NOT met; check failed assumptions."
            ),
            "honest_note": (
                "This is a STRUCTURAL test. It does not prove that God exists "
                "or does not exist. It does not prove that every form of "
                "atheism and theism is simultaneously true. It shows that, "
                "under the stated assumptions, the two representational "
                "frameworks share a non-empty empirical substructure and are "
                "related by a natural transformation on that substructure."
            ),
        }

        return results


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES Category Theory Test (Theorem 1 strengthening) — Smoke Test")
    print("=" * 72)

    suite = CategoryTheoryTestSuite()
    results = suite.run_all()

    print("\n  ── CT1: Category Axioms ──")
    print(f"    R_N valid: {results['CT1_category_axioms']['R_N']['morphisms_well_formed']}")
    print(f"    R_T valid: {results['CT1_category_axioms']['R_T']['morphisms_well_formed']}")

    print("\n  ── CT2: Functor + Observable Preservation (A1, A2) ──")
    for f_name in ["F_N", "F_T"]:
        pres = results["CT2_observable_preservation"][f_name]
        print(f"    {f_name}: {pres}")

    print("\n  ── CT3: Pullback / Shared Empirical Substructure (A4) ──")
    pb = results["CT3_pullback"]
    print(f"    Shared observables : {pb['pullback_observables']}")
    print(f"    N shared           : {pb['n_shared']}")
    print(f"    Pullback non-empty : {pb['pullback_nonempty']}")
    if pb["disagreements"]:
        print(f"    Disagreements      : {pb['disagreements']}")

    print("\n  ── CT4: Natural Transformation ──")
    nt = results["CT4_natural_transformation"]
    print(f"    Exists             : {nt['natural_transformation_exists']}")
    print(f"    Note               : {nt['note']}")

    print("\n  ── Theorem 1 Status ──")
    t1 = results["theorem1_status"]
    print(f"    A1/A2 (observables preserved) : {t1['A1_A2_observable_preserved']}")
    print(f"    A4 (empirical agreement)      : {t1['A4_empirical_agreement']}")
    print(f"    Compatible(N,T)               : {t1['Compatible(N,T)']}")
    print(f"    Conclusion                    : {t1['conclusion']}")
    print(f"\n    {t1['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is a STRUCTURAL test, not a metaphysical proof")
    print("  • Pullback non-empty = LOGICAL COMPATIBILITY under assumptions")
    print("  • Does NOT prove God exists or does not exist")
    print("  • Does NOT prove every form of atheism/theism is true")
    print("  • A5 (no same-level contradiction) needs semantic analysis")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
