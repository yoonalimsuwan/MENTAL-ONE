"""
=============================================================================
SES HIGHER-ORDER LOGIC FRAMEWORK
Second-order and higher-order logic for Dual-Model Theory formalization
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[HOL1] Type system for higher-order logic
       • Base types: State, Question, Evidence
       • Function types: A → B
       • Predicate types: A → Prop
       • Higher-order: (A → Prop) → Prop (quantification over predicates)

[HOL2] Model classes as higher-order objects
       • Mod_N = { M : M ⊨ Axioms_N } is a SET of structures
       • Requires second-order quantification
       • Satisfaction relation ⊨ is a higher-order predicate

[HOL3] Formal language and semantics
       • First-order language L (constants, functions, relations)
       • Structure M interpreting L
       • Satisfaction M ⊨ φ defined recursively
       • Reduct M|_L' for sublanguage L' ⊆ L

RELATION TO DUAL-MODEL THEORY
------------------------------
The paper's §15 recommends "formal model-theoretic paper using first-order
logic, semantic interpretation, and category theory." This module provides
the LOGICAL FOUNDATION for that formalization, specifically:

  • First-order logic alone cannot express:
    - "For all model classes..." (needs 2nd order)
    - "There exists an interpretation function..." (needs 2nd order)
    - "The set of all structures satisfying axioms..." (needs 2nd order)

  • Higher-order logic (HOL) CAN express these, at the cost of:
    - Losing completeness (Gödel: no complete proof system for full HOL)
    - Requiring more careful type discipline

This is WHY Coq/Lean are appropriate: they are based on dependent type
theory, which naturally supports higher-order reasoning.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a PYTHON MIRROR of HOL concepts, not a proof checker.
• Actual proofs are in Coq/Lean (see sesi_coq_lean_proofs.py).
• HOL is MORE EXPRESSIVE but LESS COMPLETE than FOL.
• The formalization captures LOGICAL STRUCTURE, not metaphysical truth.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.hol")

__all__ = [
    "HOLType", "HOLTerm", "HOLFormula",
    "Signature", "Structure", "Satisfaction",
    "ModelClass", "HigherOrderFramework",
]


# ═══════════════════════════════════════════════════════════════════════
# [C] TYPE SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class HOLTypeKind(Enum):
    BASE = auto()       # e.g., State, Question, Evidence
    FUNCTION = auto()   # A → B
    PREDICATE = auto()  # A → Prop
    PROPOSITION = auto()  # Prop
    SET = auto()        # Set(A) — for model classes


@dataclass(frozen=True)
class HOLType:
    """
    [C] A type in higher-order logic.

    Base types: State, Question, Evidence, Level
    Function types: A → B
    Predicate types: A → Prop
    Proposition: Prop
    Set types: Set(A) for model classes
    """
    kind: HOLTypeKind
    name: str = ""
    domain: Optional["HOLType"] = None   # for FUNCTION/PREDICATE
    codomain: Optional["HOLType"] = None # for FUNCTION/PREDICATE
    element_type: Optional["HOLType"] = None  # for SET

    @staticmethod
    def base(name: str) -> "HOLType":
        return HOLType(HOLTypeKind.BASE, name)

    @staticmethod
    def function(domain: "HOLType", codomain: "HOLType") -> "HOLType":
        return HOLType(HOLTypeKind.FUNCTION, "", domain, codomain)

    @staticmethod
    def predicate(domain: "HOLType") -> "HOLType":
        return HOLType(HOLTypeKind.PREDICATE, "", domain,
                       HOLType(HOLTypeKind.PROPOSITION))

    @staticmethod
    def proposition() -> "HOLType":
        return HOLType(HOLTypeKind.PROPOSITION)

    @staticmethod
    def set_of(element: "HOLType") -> "HOLType":
        return HOLType(HOLTypeKind.SET, "", element_type=element)

    @property
    def order(self) -> int:
        """
        [C] The order of a type.
        Order 0: base types
        Order 1: predicates over base types, sets of base types
        Order n+1: predicates/sets over order-n types
        """
        if self.kind == HOLTypeKind.BASE:
            return 0
        elif self.kind == HOLTypeKind.PROPOSITION:
            return 0
        elif self.kind == HOLTypeKind.PREDICATE:
            return 1 + self.domain.order if self.domain else 1
        elif self.kind == HOLTypeKind.SET:
            return 1 + (self.element_type.order if self.element_type else 0)
        elif self.kind == HOLTypeKind.FUNCTION:
            return max(self.domain.order if self.domain else 0,
                       self.codomain.order if self.codomain else 0)
        return 0


# Standard types
State = HOLType.base("State")
Question = HOLType.base("Question")
Evidence = HOLType.base("Evidence")
Level = HOLType.base("Level")
Prop = HOLType.proposition()

# Higher-order types
PredicateOverState = HOLType.predicate(State)       # State → Prop
ModelClass = HOLType.set_of(PredicateOverState)     # Set(State → Prop)
InterpretationFn = HOLType.function(State, Prop)    # State → Prop


# ═══════════════════════════════════════════════════════════════════════
# [C] TERMS AND FORMULAS
# ═══════════════════════════════════════════════════════════════════════

class TermKind(Enum):
    VARIABLE = auto()
    CONSTANT = auto()
    APPLICATION = auto()
    LAMBDA = auto()


@dataclass
class HOLTerm:
    """A term in higher-order logic."""
    kind: TermKind
    name: str = ""
    hol_type: Optional[HOLType] = None
    # For APPLICATION
    func: Optional["HOLTerm"] = None
    arg: Optional["HOLTerm"] = None
    # For LAMBDA
    bound_var: Optional[str] = None
    body: Optional["HOLTerm"] = None


class FormulaKind(Enum):
    ATOMIC = auto()       # P(t1, ..., tn)
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()
    FORALL = auto()       # ∀x. φ
    EXISTS = auto()       # ∃x. φ
    FORALL_PRED = auto()  # ∀P:A→Prop. φ  (second-order)
    EXISTS_PRED = auto()  # ∃P:A→Prop. φ  (second-order)
    TRUE = auto()
    FALSE = auto()


@dataclass
class HOLFormula:
    """A formula in higher-order logic."""
    kind: FormulaKind
    # For ATOMIC
    predicate: Optional[str] = None
    args: List[HOLTerm] = field(default_factory=list)
    # For connectives
    left: Optional["HOLFormula"] = None
    right: Optional["HOLFormula"] = None
    # For quantifiers
    var_name: Optional[str] = None
    var_type: Optional[HOLType] = None
    body: Optional["HOLFormula"] = None

    @staticmethod
    def atomic(pred: str, *args: HOLTerm) -> "HOLFormula":
        return HOLFormula(FormulaKind.ATOMIC, predicate=pred, args=list(args))

    @staticmethod
    def forall(var: str, vtype: HOLType, body: "HOLFormula") -> "HOLFormula":
        return HOLFormula(FormulaKind.FORALL, var_name=var,
                          var_type=vtype, body=body)

    @staticmethod
    def exists(var: str, vtype: HOLType, body: "HOLFormula") -> "HOLFormula":
        return HOLFormula(FormulaKind.EXISTS, var_name=var,
                          var_type=vtype, body=body)

    @staticmethod
    def forall_pred(var: str, vtype: HOLType, body: "HOLFormula") -> "HOLFormula":
        """Second-order universal quantification over predicates."""
        return HOLFormula(FormulaKind.FORALL_PRED, var_name=var,
                          var_type=vtype, body=body)

    @staticmethod
    def and_(left: "HOLFormula", right: "HOLFormula") -> "HOLFormula":
        return HOLFormula(FormulaKind.AND, left=left, right=right)

    @staticmethod
    def implies(left: "HOLFormula", right: "HOLFormula") -> "HOLFormula":
        return HOLFormula(FormulaKind.IMPLIES, left=left, right=right)


# ═══════════════════════════════════════════════════════════════════════
# [C] SIGNATURE AND STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Symbol:
    name: str
    arity: int
    symbol_kind: str  # "constant", "function", "relation"


class Signature:
    """
    [C] A (higher-order) signature: the vocabulary of a formal language.

    L_phys ⊆ L_N and L_phys ⊆ L_T
    """

    def __init__(self, name: str):
        self.name = name
        self.symbols: Dict[str, Symbol] = {}
        self.sorts: Set[str] = set()

    def add_constant(self, name: str, sort: str = "State") -> None:
        self.symbols[name] = Symbol(name, 0, "constant")
        self.sorts.add(sort)

    def add_function(self, name: str, arity: int) -> None:
        self.symbols[name] = Symbol(name, arity, "function")

    def add_relation(self, name: str, arity: int) -> None:
        self.symbols[name] = Symbol(name, arity, "relation")

    def is_subsignature_of(self, other: "Signature") -> bool:
        return all(name in other.symbols for name in self.symbols)

    def intersection(self, other: "Signature") -> Set[str]:
        return set(self.symbols.keys()) & set(other.symbols.keys())


class Structure:
    """
    [C] A structure interpreting a signature.

    M = (|M|, (c^M), (f^M), (R^M))
    where |M| is the domain, c^M interprets constants, etc.
    """

    def __init__(self, name: str, signature: Signature):
        self.name = name
        self.signature = signature
        self.domain: List[Any] = []
        self.interpretation: Dict[str, Any] = {}

    def set_domain(self, domain: List[Any]) -> None:
        self.domain = domain

    def interpret_constant(self, sym: str, value: Any) -> None:
        self.interpretation[sym] = value

    def interpret_relation(self, sym: str, relation: Set[Tuple]) -> None:
        self.interpretation[sym] = relation

    def reduct(self, subsig: Signature) -> "Structure":
        """
        [C] The reduct of this structure to a subsignature.

        M|_L_phys retains only the interpretations of L_phys symbols.
        """
        if not subsig.is_subsignature_of(self.signature):
            raise ValueError(f"{subsig.name} is not a subsignature of "
                             f"{self.signature.name}")
        red = Structure(f"{self.name}|_{subsig.name}", subsig)
        red.domain = self.domain.copy()
        for sym in subsig.symbols:
            if sym in self.interpretation:
                red.interpretation[sym] = self.interpretation[sym]
        return red


# ═══════════════════════════════════════════════════════════════════════
# [C] SATISFACTION RELATION (higher-order)
# ═══════════════════════════════════════════════════════════════════════

class Satisfaction:
    """
    [C] The satisfaction relation M ⊨ φ.

    For first-order formulas: standard Tarskian semantics.
    For second-order formulas: quantification over subsets/relations.

    HONEST: Full second-order satisfaction is not computable in general.
    This provides a STRUCTURAL check, not a decision procedure.
    """

    @staticmethod
    def satisfies(structure: Structure, formula: HOLFormula,
                  assignment: Optional[Dict[str, Any]] = None) -> Optional[bool]:
        """
        Check if structure satisfies formula under assignment.

        Returns None if undecidable (which is possible for HOL).
        """
        if assignment is None:
            assignment = {}

        if formula.kind == FormulaKind.TRUE:
            return True
        elif formula.kind == FormulaKind.FALSE:
            return False
        elif formula.kind == FormulaKind.ATOMIC:
            return Satisfaction._eval_atomic(structure, formula, assignment)
        elif formula.kind == FormulaKind.AND:
            left = Satisfaction.satisfies(structure, formula.left, assignment)
            right = Satisfaction.satisfies(structure, formula.right, assignment)
            if left is None or right is None:
                return None
            return left and right
        elif formula.kind == FormulaKind.NOT:
            inner = Satisfaction.satisfies(structure, formula.left, assignment)
            return None if inner is None else not inner
        elif formula.kind == FormulaKind.FORALL:
            # First-order universal: check all domain elements
            for elem in structure.domain:
                new_assign = dict(assignment)
                new_assign[formula.var_name] = elem
                result = Satisfaction.satisfies(structure, formula.body, new_assign)
                if result is False:
                    return False
                if result is None:
                    return None
            return True
        elif formula.kind == FormulaKind.FORALL_PRED:
            # Second-order: cannot enumerate all predicates
            # Return None to indicate undecidability
            logger.info("Second-order FORALL: undecidable in general")
            return None
        else:
            return None

    @staticmethod
    def _eval_atomic(structure: Structure, formula: HOLFormula,
                     assignment: Dict[str, Any]) -> Optional[bool]:
        pred = formula.predicate
        if pred not in structure.interpretation:
            return None
        rel = structure.interpretation[pred]
        # Evaluate arguments
        args = []
        for arg in formula.args:
            if arg.kind == TermKind.VARIABLE:
                args.append(assignment.get(arg.name))
            elif arg.kind == TermKind.CONSTANT:
                args.append(structure.interpretation.get(arg.name))
            else:
                return None
        return tuple(args) in rel


# ═══════════════════════════════════════════════════════════════════════
# [D] MODEL CLASS (higher-order object)
# ═══════════════════════════════════════════════════════════════════════

class ModelClass:
    """
    [D] A class of structures satisfying axioms — a SECOND-ORDER object.

    Mod_N = { M : M ⊨ Axioms_N }

    This requires higher-order logic because we quantify over structures
    (which are themselves functions/relations).
    """

    def __init__(self, name: str, signature: Signature):
        self.name = name
        self.signature = signature
        self.axioms: List[HOLFormula] = []
        self.structures: List[Structure] = []

    def add_axiom(self, axiom: HOLFormula) -> None:
        self.axioms.append(axiom)

    def add_structure(self, structure: Structure) -> None:
        """Add a structure if it satisfies all axioms."""
        for axiom in self.axioms:
            result = Satisfaction.satisfies(structure, axiom)
            if result is False:
                logger.warning(f"Structure {structure.name} fails axiom; not added")
                return
        self.structures.append(structure)

    def check_membership(self, structure: Structure) -> Optional[bool]:
        """Check if a structure belongs to this model class."""
        for axiom in self.axioms:
            result = Satisfaction.satisfies(structure, axiom)
            if result is False:
                return False
            if result is None:
                return None  # undecidable
        return True


# ═══════════════════════════════════════════════════════════════════════
# [D] HIGHER-ORDER FRAMEWORK FOR DUAL-MODEL
# ═══════════════════════════════════════════════════════════════════════

class HigherOrderFramework:
    """
    [D] The full higher-order framework for Dual-Model Theory.

    Builds:
      • L_phys, L_N, L_T as signatures
      • Mod_N, Mod_T as model classes (second-order objects)
      • Shared empirical substructure as intersection
      • Compatibility as a higher-order property
    """

    def __init__(self):
        self.L_phys = self._build_L_phys()
        self.L_N = self._build_L_N()
        self.L_T = self._build_L_T()
        self.Mod_N = ModelClass("Mod_N", self.L_N)
        self.Mod_T = ModelClass("Mod_T", self.L_T)

    def _build_L_phys(self) -> Signature:
        L = Signature("L_phys")
        L.add_constant("c", "Real")
        L.add_constant("h", "Real")
        L.add_constant("G_newton", "Real")
        L.add_relation("Gravitates", 2)
        L.add_relation("ObeysMaxwell", 2)
        return L

    def _build_L_N(self) -> Signature:
        L = Signature("L_N")
        # Include L_phys
        for sym_name, sym in self.L_phys.symbols.items():
            L.symbols[sym_name] = sym
        L.sorts.update(self.L_phys.sorts)
        # Add naturalistic predicates
        L.add_relation("IsGroundedInPhysicalLaws", 1)
        return L

    def _build_L_T(self) -> Signature:
        L = Signature("L_T")
        # Include L_phys
        for sym_name, sym in self.L_phys.symbols.items():
            L.symbols[sym_name] = sym
        L.sorts.update(self.L_phys.sorts)
        # Add theological predicates
        L.add_relation("IsGroundedInGod", 1)
        L.add_relation("IsExpressionOfDharma", 1)
        return L

    def check_compatibility(self, M_N: Structure, M_T: Structure) -> Dict[str, Any]:
        """
        [D] Check the higher-order compatibility conditions.

        C1: M_N|_L_phys = M_T|_L_phys (identical reducts)
        C2: No L_phys-sentence true in one, false in other
        C3: Differences only outside L_phys
        """
        reduct_N = M_N.reduct(self.L_phys)
        reduct_T = M_T.reduct(self.L_phys)

        # C1: Compare reducts
        c1 = (reduct_N.domain == reduct_T.domain and
              reduct_N.interpretation == reduct_T.interpretation)

        # C3: Check extension symbols
        n_only = set(self.L_N.symbols) - set(self.L_phys.symbols)
        t_only = set(self.L_T.symbols) - set(self.L_phys.symbols)

        return {
            "C1_identical_reducts": c1,
            "C3_N_only_symbols": sorted(n_only),
            "C3_T_only_symbols": sorted(t_only),
            "compatible": c1,
            "honest_note": (
                "Higher-order compatibility check. This verifies LOGICAL "
                "STRUCTURE, not metaphysical truth. C2 requires checking "
                "all L_phys-sentences, which is undecidable for full HOL."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES Higher-Order Logic Framework — Smoke Test")
    print("=" * 72)

    # Build the higher-order framework
    hof = HigherOrderFramework()

    print("\n  ── Signatures ──")
    print(f"    L_phys symbols : {sorted(hof.L_phys.symbols.keys())}")
    print(f"    L_N symbols    : {sorted(hof.L_N.symbols.keys())}")
    print(f"    L_T symbols    : {sorted(hof.L_T.symbols.keys())}")
    print(f"    L_phys ⊆ L_N   : {hof.L_phys.is_subsignature_of(hof.L_N)}")
    print(f"    L_phys ⊆ L_T   : {hof.L_phys.is_subsignature_of(hof.L_T)}")

    print("\n  ── Type orders ──")
    print(f"    State order          : {State.order}")
    print(f"    PredicateOverState   : {PredicateOverState.order}")
    print(f"    ModelClass           : {ModelClass.order}")

    # Build structures
    print("\n  ── Structures ──")
    M_N = Structure("M_N", hof.L_N)
    M_N.set_domain(["s1", "s2"])
    M_N.interpret_constant("c", 299792458.0)
    M_N.interpret_constant("h", 6.62607015e-34)
    M_N.interpret_constant("G_newton", 6.67430e-11)
    M_N.interpret_relation("Gravitates", {("s1", "s2")})
    M_N.interpret_relation("ObeysMaxwell", {("s1", "s2")})
    M_N.interpret_relation("IsGroundedInPhysicalLaws", {("s1",), ("s2",)})

    M_T = Structure("M_T", hof.L_T)
    M_T.set_domain(["s1", "s2"])
    M_T.interpret_constant("c", 299792458.0)
    M_T.interpret_constant("h", 6.62607015e-34)
    M_T.interpret_constant("G_newton", 6.67430e-11)
    M_T.interpret_relation("Gravitates", {("s1", "s2")})
    M_T.interpret_relation("ObeysMaxwell", {("s1", "s2")})
    M_T.interpret_relation("IsGroundedInGod", {("s1",), ("s2",)})
    M_T.interpret_relation("IsExpressionOfDharma", {("s1",), ("s2",)})

    # Check compatibility
    print("\n  ── Higher-order compatibility ──")
    compat = hof.check_compatibility(M_N, M_T)
    print(f"    C1 (identical reducts) : {compat['C1_identical_reducts']}")
    print(f"    C3 (N-only symbols)    : {compat['C3_N_only_symbols']}")
    print(f"    C3 (T-only symbols)    : {compat['C3_T_only_symbols']}")
    print(f"    Compatible             : {compat['compatible']}")

    # Satisfaction check
    print("\n  ── Satisfaction ──")
    grav_formula = HOLFormula.atomic("Gravitates",
                                      HOLTerm(TermKind.CONSTANT, "s1"),
                                      HOLTerm(TermKind.CONSTANT, "s2"))
    sat_result = Satisfaction.satisfies(M_N, grav_formula)
    print(f"    M_N ⊨ Gravitates(s1,s2) : {sat_result}")

    # Second-order quantification
    print("\n  ── Second-order quantification ──")
    so_formula = HOLFormula.forall_pred(
        "P", PredicateOverState,
        HOLFormula(FormulaKind.TRUE))
    so_result = Satisfaction.satisfies(M_N, so_formula)
    print(f"    ∀P:State→Prop. TRUE : {so_result} (undecidable for full HOL)")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Python mirror of HOL concepts, NOT a proof checker")
    print("  • Actual proofs in Coq/Lean (see sesi_coq_lean_proofs.py)")
    print("  • Full HOL satisfaction is UNDECIDABLE in general")
    print("  • Formalization captures LOGICAL STRUCTURE, not metaphysical truth")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
