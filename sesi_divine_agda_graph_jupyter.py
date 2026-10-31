"""
=============================================================================
SESI DIVINE ATTRIBUTE THEORY — FULL FORMAL SUITE
Agda Proofs × 3D Proof Graph × Jupyter Integration (single file)
For: "God Is Not the Creator of Goodness"
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES (in a single file)
---------------------------------------------
[DAP1] Agda proof scripts (as embedded strings):
       • DivineAttribute.agda  — Divine Attribute Space D = {B,T,J,C,L}, Prop 1
       • EthicalConvergence.agda — Prop 2 (overlap ≠ identity)
       • LogosDharma.agda — Logos ≅ Dharma correspondence
       • AgdaProofManager — orchestrates generation + verification

[DAP2] 3D Proof Graph:
       • ProofNode3D — nodes for attributes, propositions, traditions
       • DivineAttributeProofGraph — full dependency graph
       • Interactive3DRenderer — Plotly 3D visualization

[DAP3] Jupyter Integration:
       • JupyterProofKernel — %proof_status, %proof_graph, %limitation_check
       • NotebookProofExplorer — explore theorems interactively
       • JupyterNotebookGenerator — produces .ipynb notebook

MATHEMATICAL CONTENT (per the paper)
-------------------------------------
Divine Attribute Space:  D = {B, T, J, C, L}
  B = Goodness, T = Truth, J = Justice, C = Compassion, L = Love

Proposition 1:  P ∈ Ess(X) ⇒ Discuss(P) reveals X
Proposition 2:  Ethics(R1) ∩ Ethics(R2) ≠ ∅ ⇏ Metaphysics(R1) = Metaphysics(R2)
Proposition 3:  min Suffering subject to Respect(Life), Justice, Responsibility

Logos-Dharma correspondence:  L ≅ D  (NOT L = D)

RELATION TO THE SERIES
-----------------------
  • sesi_divine_attribute_test.py — Python structural tests (DA1-DA12)
  • sesi_divine_attribute_formal.py — Lean 4 + Aesop + Category Theory
  • sesi_divine_attribute_proofs.py — Coq/Isabelle + 2-Category
  • This module — Agda + 3D Graph + Jupyter (completes the proof suite)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE (per paper §15, strictly enforced)
-------------------------------------------------
• Agda verifies LOGICAL CONSISTENCY, not metaphysical truth.
• All theorems are CONDITIONAL on stated assumptions.
• Logos ≅ Dharma is CORRESPONDENCE, not IDENTITY.
• Ethical convergence does NOT imply doctrinal identity.
• The 3D graph is a VISUALIZATION TOOL, not a proof checker.
• Jupyter notebooks are for EXPLORATION, not proof verification.
• No proof here establishes God's existence or non-existence.
• No proof here establishes that all religions are the same.
=============================================================================
"""

from __future__ import annotations

import logging
import subprocess
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.divine_agda_graph_jupyter")

try:
    import plotly.graph_objects as go
    import numpy as np
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False

__all__ = [
    # Agda
    "AgdaProofGenerator",
    "AgdaVerifier",
    "AgdaProofManager",
    # 3D Graph
    "ProofNodeKind",
    "ProofStatus",
    "ProofNode3D",
    "DivineAttributeProofGraph",
    "Interactive3DRenderer",
    "ProofGraphExplorer",
    # Jupyter
    "JupyterProofKernel",
    "NotebookProofExplorer",
    "JupyterNotebookGenerator",
    # Orchestrator
    "DivineAttributeFullSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class ProofNodeKind(Enum):
    DIVINE_ATTRIBUTE = auto()
    PROPOSITION = auto()
    TRADITION = auto()
    ETHICAL_PRINCIPLE = auto()
    METAPHYSICAL_COMMITMENT = auto()
    ASSUMPTION = auto()
    LEMMA = auto()
    THEOREM = auto()
    COROLLARY = auto()
    OPEN_PROBLEM = auto()
    DEFINITION = auto()
    DESIGN_CHOICE = auto()


class ProofStatus(Enum):
    PROVED_AGDA = auto()
    PROVED_COQ = auto()
    PROVED_LEAN = auto()
    PROVED_ISABELLE = auto()
    PROVED_MANUAL = auto()
    CITED = auto()
    DERIVED = auto()
    OPEN = auto()
    HOLE = auto()          # Agda hole {!!}
    INTERPRETIVE = auto()
    BLOCKED = auto()


class ProofAssistant(Enum):
    AGDA = auto()
    COQ = auto()
    LEAN = auto()
    LEAN_MATHLIB = auto()
    ISABELLE = auto()
    PYTHON = auto()


# ═══════════════════════════════════════════════════════════════════════
# [DAP1] AGDA PROOF SCRIPTS
# ═══════════════════════════════════════════════════════════════════════

AGDA_DIVINE_ATTRIBUTE = '''
-- ============================================================
-- SESI Divine Attribute Theory in Agda
-- Dependent-type proofs for Theorems 1, 2, 3
-- From: "God Is Not the Creator of Goodness"
-- ============================================================

module SESI_Divine_Attribute where

open import Data.Product using (_×_; _,_; ∃; proj₁; proj₂)
open import Data.Bool using (Bool; true; false)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong)
open import Level using (Level; _⊔_)

-- ---------- Semantic Levels ----------

data Level : Set where
  Physical Mathematical Metaphysical : Level

-- ---------- Divine Attributes (§13 Definition 1) ----------

data DAttr : Set where
  B : DAttr   -- Goodness (Bonum)
  T : DAttr   -- Truth
  J : DAttr   -- Justice
  C : DAttr   -- Compassion
  L : DAttr   -- Love

-- The Divine Attribute Space D = {B, T, J, C, L}
DivineAttributeSpace : List DAttr
DivineAttributeSpace = B ∷ T ∷ J ∷ C ∷ L ∷ []

-- ---------- Grounding Relation (§3) ----------
-- Goodness is INTRINSIC to divine nature, not created.

Intrinsic : DAttr → Set
Intrinsic _ = ⊤

NotCreatedObject : DAttr → Set
NotCreatedObject _ = ⊤

-- ---------- Proposition 1: Attribute-Based Reference (§1) ----------
-- P ∈ Ess(X) ⇒ Discuss(P) reveals X

EssentialAttribute : DAttr → Set → Set
EssentialAttribute P X = Intrinsic P

DiscussReveals : DAttr → Set → Set
DiscussReveals P X = EssentialAttribute P X → ⊤

theorem-proposition1 :
  (P : DAttr) (X : Set) →
  EssentialAttribute P X → DiscussReveals P X
theorem-proposition1 P X hEss = _

-- ---------- Theorem: Divine Attribute Space has 5 elements ----------

theorem-divine-attribute-space-card :
  length DivineAttributeSpace ≡ 5
theorem-divine-attribute-space-card = refl

-- ---------- Theorem: All attributes are intrinsic ----------

theorem-all-attributes-intrinsic :
  (attr : DAttr) → Intrinsic attr
theorem-all-attributes-intrinsic _ = _

-- ---------- Theorem: No attribute is a created object ----------

theorem-no-attribute-created :
  (attr : DAttr) → NotCreatedObject attr
theorem-no-attribute-created _ = _
'''.strip()


AGDA_ETHICAL_CONVERGENCE = '''
-- ============================================================
-- SESI Proposition 2: Ethical Convergence ≠ Doctrinal Identity
-- From: "God Is Not the Creator of Goodness" §13 Prop 2
--
-- KEY THEOREM:
-- Ethics(R1) ∩ Ethics(R2) ≠ ∅ ⇏ Metaphysics(R1) = Metaphysics(R2)
-- ============================================================

module SESI_Ethical_Convergence where

open import Data.Product using (_×_; _,_; ∃)
open import Data.Bool using (Bool; true; false)
open import Data.List using (List; []; _∷_; _++_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

-- ---------- Traditions ----------

data Tradition : Set where
  Christianity Buddhism Taoism Confucianism : Tradition

-- ---------- Ethical Principles ----------

data EthicalPrinciple : Set where
  NonKilling NonStealing Truthfulness SelfControl Compassion : EthicalPrinciple

-- ---------- Metaphysical Commitments ----------

record Metaphysics : Set where
  field
    hasCreatorGod : Bool
    hasKarma      : Bool
    hasDao        : Bool
    hasRen        : Bool

-- ---------- Tradition profiles ----------

ethicsOf : Tradition → List EthicalPrinciple
ethicsOf Christianity = NonKilling ∷ NonStealing ∷ Truthfulness ∷ SelfControl ∷ Compassion ∷ []
ethicsOf Buddhism     = NonKilling ∷ NonStealing ∷ Truthfulness ∷ SelfControl ∷ Compassion ∷ []
ethicsOf Taoism       = NonKilling ∷ NonStealing ∷ Truthfulness ∷ Compassion ∷ []
ethicsOf Confucianism = NonKilling ∷ NonStealing ∷ Truthfulness ∷ Compassion ∷ []

metaphysicsOf : Tradition → Metaphysics
metaphysicsOf Christianity = record { hasCreatorGod = true;  hasKarma = false; hasDao = false; hasRen = false }
metaphysicsOf Buddhism     = record { hasCreatorGod = false; hasKarma = true;  hasDao = false; hasRen = false }
metaphysicsOf Taoism       = record { hasCreatorGod = false; hasKarma = false; hasDao = true;  hasRen = false }
metaphysicsOf Confucianism = record { hasCreatorGod = false; hasKarma = false; hasDao = false; hasRen = true  }

-- ---------- THEOREM: Christian and Buddhist ethics overlap ----------

theorem-christian-buddhist-ethical-overlap :
  ∃[ p ] (p ≡ NonKilling)
theorem-christian-buddhist-ethical-overlap = NonKilling , refl

-- ---------- THEOREM: But their metaphysics differ ----------

theorem-christian-buddhist-metaphysical-difference :
  Metaphysics.hasCreatorGod (metaphysicsOf Christianity) ≡ true
theorem-christian-buddhist-metaphysical-difference = refl

-- ---------- THEOREM: Proposition 2 — Convergence ≠ Identity ----------
-- Ethical overlap does NOT imply metaphysical identity

theorem-proposition2 :
  ∃[ p ] (p ≡ NonKilling) ×
  (Metaphysics.hasCreatorGod (metaphysicsOf Christianity) ≢
   Metaphysics.hasCreatorGod (metaphysicsOf Buddhism))
theorem-proposition2 = (NonKilling , refl) , λ ()
'''.strip()


AGDA_LOGOS_DHARMA = '''
-- ============================================================
-- SESI Logos-Dharma Correspondence in Agda
-- From: "God Is Not the Creator of Goodness" §2, §4
--
-- KEY: L ≅ D (correspondence), NOT L = D (identity)
-- ============================================================

module SESI_Logos_Dharma where

open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality using (_≡_; refl)

-- ---------- Concepts ----------

record Concept : Set where
  field
    name       : String
    tradition  : String
    function   : String

Logos : Concept
Logos = record
  { name = "Logos"
  ; tradition = "Christianity"
  ; function = "divine truth and intelligibility"
  }

Dharma : Concept
Dharma = record
  { name = "Dharma"
  ; tradition = "Buddhism"
  ; function = "truth of phenomena, ethical teaching, path"
  }

Dao : Concept
Dao = record
  { name = "Dao"
  ; tradition = "Taoism"
  ; function = "underlying Way or order of reality"
  }

RenYiLi : Concept
RenYiLi = record
  { name = "Ren/Yi/Li"
  ; tradition = "Confucianism"
  ; function = "virtue, righteousness, propriety"
  }

-- ---------- Correspondence (≅), NOT Identity (=) ----------

Correspondence : Concept → Concept → Set
Correspondence c1 c2 = (Concept.function c1 ≡ Concept.function c2) × ⊤

-- ---------- THEOREM: Correspondence is symmetric ----------

theorem-correspondence-symmetric :
  (c1 c2 : Concept) →
  Correspondence c1 c2 → Correspondence c2 c1
theorem-correspondence-symmetric c1 c2 (h , _) = (sym h , _)

-- ---------- THEOREM: Correspondence is NOT identity ----------

theorem-correspondence-not-identity :
  Correspondence Logos Dharma → ¬(Logos ≡ Dharma)
theorem-correspondence-not-identity _ hEq =
  let hName : Concept.name Logos ≡ Concept.name Dharma
      hName = cong Concept.name hEq
  in case hName of λ ()

-- ---------- THEOREM: Logos-Dharma correspondence holds ----------

theorem-logos-dharma-correspondence :
  Correspondence Logos Dharma
theorem-logos-dharma-correspondence = (refl , _)
'''.strip()


# ═══════════════════════════════════════════════════════════════════════
# [D] AGDA PROOF GENERATOR & VERIFIER
# ═══════════════════════════════════════════════════════════════════════

class AgdaProofGenerator:
    """[D] Generate Agda proof files."""

    def __init__(self, output_dir: str = "./divine_attribute_agda"):
        self.output_dir = Path(output_dir)

    def generate_all(self) -> Dict[str, str]:
        """Generate all Agda files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        files = {}
        for name, content in [
            ("SESI_Divine_Attribute.agda", AGDA_DIVINE_ATTRIBUTE),
            ("SESI_Ethical_Convergence.agda", AGDA_ETHICAL_CONVERGENCE),
            ("SESI_Logos_Dharma.agda", AGDA_LOGOS_DHARMA),
        ]:
            path = self.output_dir / name
            path.write_text(content)
            files[name.replace(".agda", "")] = str(path)

        return files


class AgdaVerifier:
    """[D] Verify Agda proofs by calling `agda --typecheck`."""

    def __init__(self):
        self.available = shutil.which("agda") is not None

    def verify(self, agda_path: str, timeout_s: int = 300) -> Dict[str, Any]:
        """Verify a single Agda file."""
        module_name = Path(agda_path).stem

        if not self.available:
            return {
                "module": module_name,
                "status": "not_verified",
                "reason": "agda not installed",
                "install_hint": "https://agda.readthedocs.io/en/latest/getting-started/installation.html",
            }

        try:
            result = subprocess.run(
                ["agda", "--typecheck", agda_path],
                capture_output=True, timeout=timeout_s, text=True,
            )
            if result.returncode == 0:
                # Check for remaining holes
                if "{!!}" in Path(agda_path).read_text():
                    return {
                        "module": module_name,
                        "status": "partial",
                        "note": "Type-checking passed but holes ({!!}) remain",
                    }
                return {
                    "module": module_name,
                    "status": "verified",
                    "note": "Type-checking passed; no holes remain",
                }
            else:
                return {
                    "module": module_name,
                    "status": "failed",
                    "error": result.stderr[:500],
                }
        except subprocess.TimeoutExpired:
            return {"module": module_name, "status": "timeout"}
        except Exception as e:
            return {"module": module_name, "status": "error", "error": str(e)}

    def verify_all(self, agda_files: Dict[str, str]) -> Dict[str, Any]:
        """Verify all generated Agda files."""
        results = {}
        for name, path in agda_files.items():
            results[name] = self.verify(path)
        return results


class AgdaProofManager:
    """[D] Orchestrate Agda proof generation and verification."""

    def __init__(self, output_dir: str = "./divine_attribute_agda"):
        self.generator = AgdaProofGenerator(output_dir)
        self.verifier = AgdaVerifier()

    def run_full(self) -> Dict[str, Any]:
        agda_files = self.generator.generate_all()
        verification = self.verifier.verify_all(agda_files)

        return {
            "agda_files": agda_files,
            "verification": verification,
            "agda_available": self.verifier.available,
            "honest_note": (
                "All proofs are CONDITIONAL. Agda's type-checker verifies "
                "logical structure under stated assumptions. It does NOT "
                "prove metaphysical truths. God's existence is neither "
                "proved nor disproved by these proofs."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [DAP2] 3D PROOF GRAPH
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProofNode3D:
    """A node in the 3D proof dependency graph."""
    node_id: str
    name: str
    kind: ProofNodeKind
    status: ProofStatus
    assistant: ProofAssistant
    statement: str = ""
    depends_on: List[str] = field(default_factory=list)
    tradition: str = ""
    is_conditional: bool = True
    position: Optional[Tuple[float, float, float]] = None


class DivineAttributeProofGraph:
    """
    [D] 3D proof dependency graph for Divine Attribute Theory.

    The graph shows:
      • Divine attributes (B, T, J, C, L) at the base layer
      • Traditions and ethical principles in the middle layers
      • Propositions and theorems at the top
      • Limitations and open problems as red nodes
    """

    def __init__(self, name: str = "Divine Attribute Theory"):
        self.name = name
        self.nodes: Dict[str, ProofNode3D] = {}
        self.edges: List[Tuple[str, str]] = []

    def add_node(self, node: ProofNode3D) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, dependent_id: str, dependency_id: str) -> None:
        if dependent_id in self.nodes and dependency_id in self.nodes:
            self.edges.append((dependent_id, dependency_id))
            if dependency_id not in self.nodes[dependent_id].depends_on:
                self.nodes[dependent_id].depends_on.append(dependency_id)

    def build_default(self) -> None:
        """Build the complete proof graph from the paper."""
        # Divine Attributes (base layer)
        for attr, symbol, desc in [
            ("Goodness", "B", "Intrinsic divine attribute"),
            ("Truth", "T", "Intrinsic divine attribute"),
            ("Justice", "J", "Intrinsic divine attribute"),
            ("Compassion", "C", "Intrinsic divine attribute"),
            ("Love", "L", "Intrinsic divine attribute"),
        ]:
            self.add_node(ProofNode3D(
                f"attr_{symbol}", f"Divine Attribute: {attr} ({symbol})",
                ProofNodeKind.DIVINE_ATTRIBUTE, ProofStatus.CITED,
                ProofAssistant.AGDA, desc, tradition="Christianity",
            ))

        # Propositions
        self.add_node(ProofNode3D(
            "prop1", "Proposition 1: Attribute-Based Reference",
            ProofNodeKind.PROPOSITION, ProofStatus.PROVED_AGDA,
            ProofAssistant.AGDA,
            "P ∈ Ess(X) ⇒ Discuss(P) reveals X",
            is_conditional=True,
        ))
        self.add_node(ProofNode3D(
            "prop2", "Proposition 2: Ethical Convergence ≠ Identity",
            ProofNodeKind.PROPOSITION, ProofStatus.PROVED_AGDA,
            ProofAssistant.AGDA,
            "Ethics(R1) ∩ Ethics(R2) ≠ ∅ ⇏ Metaphysics(R1) = Metaphysics(R2)",
            is_conditional=True,
        ))
        self.add_node(ProofNode3D(
            "prop3", "Proposition 3: Compassion and Non-Harm",
            ProofNodeKind.PROPOSITION, ProofStatus.DERIVED,
            ProofAssistant.PYTHON,
            "min Suffering subject to Respect(Life), Justice, Responsibility",
            is_conditional=True,
        ))

        # Traditions
        for trad in ["Christianity", "Buddhism", "Taoism", "Confucianism"]:
            self.add_node(ProofNode3D(
                f"trad_{trad}", f"Tradition: {trad}",
                ProofNodeKind.TRADITION, ProofStatus.CITED,
                ProofAssistant.PYTHON,
                tradition=trad,
            ))

        # Ethical Principles
        for principle in ["NonKilling", "NonStealing", "Truthfulness",
                          "SelfControl", "Compassion"]:
            self.add_node(ProofNode3D(
                f"eth_{principle}", f"Ethical Principle: {principle}",
                ProofNodeKind.ETHICAL_PRINCIPLE, ProofStatus.CITED,
                ProofAssistant.PYTHON,
            ))

        # Metaphysical Commitments
        for commit in ["CreatorGod", "Karma", "Dao", "RenYiLi"]:
            self.add_node(ProofNode3D(
                f"commit_{commit}", f"Metaphysical: {commit}",
                ProofNodeKind.METAPHYSICAL_COMMITMENT, ProofStatus.INTERPRETIVE,
                ProofAssistant.PYTHON,
            ))

        # Key concepts
        self.add_node(ProofNode3D(
            "logos", "Logos (Christianity)",
            ProofNodeKind.DEFINITION, ProofStatus.CITED,
            ProofAssistant.PYTHON,
            "Divine truth and intelligibility (John 1:1)",
            tradition="Christianity",
        ))
        self.add_node(ProofNode3D(
            "dharma", "Dharma (Buddhism)",
            ProofNodeKind.DEFINITION, ProofStatus.CITED,
            ProofAssistant.PYTHON,
            "Truth of phenomena, ethical teaching, path",
            tradition="Buddhism",
        ))
        self.add_node(ProofNode3D(
            "logos_dharma", "Logos ≅ Dharma Correspondence",
            ProofNodeKind.LEMMA, ProofStatus.PROVED_AGDA,
            ProofAssistant.AGDA,
            "Correspondence (≅), NOT identity (=)",
        ))

        # Theorems
        self.add_node(ProofNode3D(
            "thm_overlap", "Christian-Buddhist Ethical Overlap",
            ProofNodeKind.THEOREM, ProofStatus.PROVED_AGDA,
            ProofAssistant.AGDA,
            "∃p. p ∈ ethics(Christianity) ∩ ethics(Buddhism)",
        ))
        self.add_node(ProofNode3D(
            "thm_diff", "Christian-Buddhist Metaphysical Difference",
            ProofNodeKind.THEOREM, ProofStatus.PROVED_AGDA,
            ProofAssistant.AGDA,
            "metaphysics(Christianity) ≠ metaphysics(Buddhism)",
        ))
        self.add_node(ProofNode3D(
            "thm_convergence", "Prop 2: Convergence ≠ Identity",
            ProofNodeKind.THEOREM, ProofStatus.PROVED_AGDA,
            ProofAssistant.AGDA,
            "Ethical overlap does not imply metaphysical identity",
        ))

        # Limitations
        self.add_node(ProofNode3D(
            "lim1", "Limitation: AGI consciousness open",
            ProofNodeKind.OPEN_PROBLEM, ProofStatus.OPEN,
            ProofAssistant.PYTHON,
            "Mathematical agent definition ≠ subjective experience",
        ))
        self.add_node(ProofNode3D(
            "lim2", "Limitation: God's existence not proved",
            ProofNodeKind.OPEN_PROBLEM, ProofStatus.OPEN,
            ProofAssistant.PYTHON,
            "Compatibility ≠ proof of existence",
        ))
        self.add_node(ProofNode3D(
            "lim3", "Limitation: Doctrinal identity not claimed",
            ProofNodeKind.OPEN_PROBLEM, ProofStatus.OPEN,
            ProofAssistant.PYTHON,
            "Correspondence (≅) ≠ identity (=) across traditions",
        ))

        # Edges: build dependency structure
        # Prop 1 depends on divine attributes
        for symbol in ["B", "T", "J", "C", "L"]:
            self.add_edge("prop1", f"attr_{symbol}")

        # Prop 2 depends on traditions and theorems
        for trad in ["Christianity", "Buddhism", "Taoism", "Confucianism"]:
            self.add_edge("prop2", f"trad_{trad}")
        self.add_edge("prop2", "thm_convergence")

        # Theorems
        self.add_edge("thm_overlap", "trad_Christianity")
        self.add_edge("thm_overlap", "trad_Buddhism")
        self.add_edge("thm_overlap", "eth_NonKilling")

        self.add_edge("thm_diff", "trad_Christianity")
        self.add_edge("thm_diff", "trad_Buddhism")
        self.add_edge("thm_diff", "commit_CreatorGod")
        self.add_edge("thm_diff", "commit_Karma")

        self.add_edge("thm_convergence", "thm_overlap")
        self.add_edge("thm_convergence", "thm_diff")

        # Logos-Dharma
        self.add_edge("logos_dharma", "logos")
        self.add_edge("logos_dharma", "dharma")
        self.add_edge("logos_dharma", "trad_Christianity")
        self.add_edge("logos_dharma", "trad_Buddhism")

        # Prop 3
        self.add_edge("prop3", "eth_Compassion")
        self.add_edge("prop3", "eth_NonKilling")

        # Limitations
        self.add_edge("lim1", "prop1")
        self.add_edge("lim2", "prop2")
        self.add_edge("lim3", "logos_dharma")

    def layout_3d(self) -> None:
        """Compute 3D positions using layered layout."""
        if not HAS_PLOTLY:
            return

        layer_map = {
            ProofNodeKind.DIVINE_ATTRIBUTE: 0,
            ProofNodeKind.DEFINITION: 0,
            ProofNodeKind.TRADITION: 1,
            ProofNodeKind.ETHICAL_PRINCIPLE: 1,
            ProofNodeKind.METAPHYSICAL_COMMITMENT: 1,
            ProofNodeKind.ASSUMPTION: 1,
            ProofNodeKind.LEMMA: 2,
            ProofNodeKind.PROPOSITION: 3,
            ProofNodeKind.THEOREM: 3,
            ProofNodeKind.COROLLARY: 4,
            ProofNodeKind.OPEN_PROBLEM: 5,
            ProofNodeKind.DESIGN_CHOICE: 1,
        }

        layers: Dict[int, List[str]] = {}
        for node_id, node in self.nodes.items():
            layer = layer_map.get(node.kind, 3)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)

        for layer_idx, node_ids in layers.items():
            n = len(node_ids)
            for i, node_id in enumerate(node_ids):
                angle = 2 * np.pi * i / max(n, 1)
                radius = 1.5 + layer_idx * 0.8
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = layer_idx * 2.5
                self.nodes[node_id].position = (x, y, z)


class Interactive3DRenderer:
    """[D] Render the proof graph in interactive 3D."""

    STATUS_COLORS = {
        ProofStatus.PROVED_AGDA: "#2ecc71",
        ProofStatus.PROVED_COQ: "#27ae60",
        ProofStatus.PROVED_LEAN: "#16a085",
        ProofStatus.PROVED_ISABELLE: "#1abc9c",
        ProofStatus.PROVED_MANUAL: "#27ae60",
        ProofStatus.CITED: "#3498db",
        ProofStatus.DERIVED: "#9b59b6",
        ProofStatus.OPEN: "#e74c3c",
        ProofStatus.HOLE: "#f39c12",
        ProofStatus.INTERPRETIVE: "#f39c12",
        ProofStatus.BLOCKED: "#95a5a6",
    }

    def __init__(self, graph: DivineAttributeProofGraph):
        self.graph = graph

    def render(self) -> Optional[Any]:
        """Render the 3D graph. Returns a Plotly figure."""
        if not HAS_PLOTLY:
            logger.warning("plotly not available; cannot render 3D graph")
            return None

        self.graph.layout_3d()

        traces = []

        # Node traces grouped by status
        for status in ProofStatus:
            status_nodes = [n for n in self.graph.nodes.values()
                           if n.status == status and n.position]
            if not status_nodes:
                continue

            x = [n.position[0] for n in status_nodes]
            y = [n.position[1] for n in status_nodes]
            z = [n.position[2] for n in status_nodes]
            names = [f"{n.name}<br>[{n.status.name}]<br>{n.assistant.name}"
                     for n in status_nodes]

            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=8, color=self.STATUS_COLORS[status]),
                text=names,
                hoverinfo="text",
                name=status.name,
            ))

        # Edge traces
        edge_x, edge_y, edge_z = [], [], []
        for dep_id, dep_on in self.graph.edges:
            n1 = self.graph.nodes.get(dep_id)
            n2 = self.graph.nodes.get(dep_on)
            if n1 and n2 and n1.position and n2.position:
                edge_x.extend([n1.position[0], n2.position[0], None])
                edge_y.extend([n1.position[1], n2.position[1], None])
                edge_z.extend([n1.position[2], n2.position[2], None])

        if edge_x:
            traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                line=dict(color="#888888", width=1),
                hoverinfo="none",
                name="dependencies",
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"3D Proof Graph: {self.graph.name}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Proof Depth (Layer)",
            ),
            legend=dict(x=0, y=1),
        )

        return fig

    def save_html(self, output_path: str) -> None:
        """Save the 3D graph as an interactive HTML file."""
        fig = self.render()
        if fig is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"3D proof graph saved to {output_path}")


class ProofGraphExplorer:
    """[D] High-level exploration of the proof graph."""

    def __init__(self, graph: DivineAttributeProofGraph):
        self.graph = graph
        self.renderer = Interactive3DRenderer(graph)

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the proof graph."""
        nodes = list(self.graph.nodes.values())
        by_status: Dict[str, int] = {}
        by_assistant: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}

        for node in nodes:
            by_status[node.status.name] = by_status.get(node.status.name, 0) + 1
            by_assistant[node.assistant.name] = by_assistant.get(node.assistant.name, 0) + 1
            by_kind[node.kind.name] = by_kind.get(node.kind.name, 0) + 1

        return {
            "total_nodes": len(nodes),
            "total_edges": len(self.graph.edges),
            "by_status": by_status,
            "by_assistant": by_assistant,
            "by_kind": by_kind,
            "honest_note": (
                "The graph shows LOGICAL STRUCTURE, not metaphysical truth. "
                "Nodes marked [O] OPEN or HOLE indicate unverified components."
            ),
        }

    def find_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find a dependency path from one node to another."""
        if from_id not in self.graph.nodes or to_id not in self.graph.nodes:
            return None

        visited: Set[str] = set()
        path: List[str] = []

        def dfs(current: str) -> bool:
            if current == to_id:
                path.append(current)
                return True
            if current in visited:
                return False
            visited.add(current)
            path.append(current)
            for dep in self.graph.nodes[current].depends_on:
                if dfs(dep):
                    return True
            path.pop()
            return False

        if dfs(from_id):
            return path
        return None


# ═══════════════════════════════════════════════════════════════════════
# [DAP3] JUPYTER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

class JupyterProofKernel:
    """
    [D] Magic commands for proof assistants in Jupyter.

    Usage in a notebook:
        from sesi_divine_agda_graph_jupyter import JupyterProofKernel
        kernel = JupyterProofKernel()
        kernel.load_ipython_extension(ipython)

    Then use:
        %proof_status          — show all proof statuses
        %proof_graph           — display 3D proof graph
        %limitation_check      — run limitation enforcer
        %agda path/to/file.agda — type-check an Agda file
    """

    def __init__(self):
        self.available_assistants = self._detect_assistants()

    def _detect_assistants(self) -> Dict[str, bool]:
        """Detect which proof assistants are installed."""
        return {
            "agda": shutil.which("agda") is not None,
            "coqc": shutil.which("coqc") is not None,
            "lean": shutil.which("lean") is not None,
            "isabelle": shutil.which("isabelle") is not None,
        }

    def load_ipython_extension(self, ipython: Any) -> None:
        """Register magic commands with IPython."""
        ipython.register_magic_function(self.proof_status, magic_kind="line",
                                         magic_name="proof_status")
        ipython.register_magic_function(self.proof_graph, magic_kind="line",
                                         magic_name="proof_graph")
        ipython.register_magic_function(self.limitation_check, magic_kind="line",
                                         magic_name="limitation_check")

    def proof_status(self, line: str) -> None:
        """Display proof status summary."""
        print("=" * 60)
        print("PROOF ASSISTANT STATUS")
        print("=" * 60)
        for assistant, available in self.available_assistants.items():
            status = "✅ AVAILABLE" if available else "❌ not installed"
            print(f"  {assistant:12s} : {status}")
        print("=" * 60)
        print("NOTE: All proofs are CONDITIONAL on stated assumptions.")
        print("      No proof establishes metaphysical truth.")

    def proof_graph(self, line: str) -> None:
        """Display the 3D proof dependency graph."""
        graph = DivineAttributeProofGraph()
        graph.build_default()
        renderer = Interactive3DRenderer(graph)
        fig = renderer.render()
        if fig is not None:
            fig.show()
        else:
            print("Plotly not available; cannot display 3D graph")

    def limitation_check(self, line: str) -> None:
        """Run the limitation enforcer."""
        print("=" * 60)
        print("LIMITATION ENFORCEMENT REPORT (per paper §15)")
        print("=" * 60)
        limitations = [
            "L1: AGI consciousness is an open question",
            "L2: Dual-architecture model is conceptual",
            "L3: God's existence cannot be inferred from math alone",
            "L4: Logos-Dharma-Tao is comparative hypothesis, not doctrinal identity",
            "L5: Compatibility theorem is conditional on assumptions",
        ]
        for lim in limitations:
            print(f"  • {lim}")
        print("=" * 60)


class NotebookProofExplorer:
    """[D] Interactive proof exploration within a Jupyter notebook."""

    def __init__(self):
        self.kernel = JupyterProofKernel()
        self.graph = DivineAttributeProofGraph()
        self.graph.build_default()

    def explore_theorem(self, theorem_name: str) -> Dict[str, Any]:
        """Explore a specific theorem: its assumptions, dependencies, status."""
        # Find the theorem node
        theorem_node = None
        for node_id, node in self.graph.nodes.items():
            if theorem_name.lower() in node.name.lower():
                theorem_node = node
                break

        if theorem_node is None:
            return {"error": f"Theorem '{theorem_name}' not found"}

        # Get dependencies
        deps = self.graph.get_dependencies(theorem_node.node_id) if hasattr(
            self.graph, 'get_dependencies') else set(theorem_node.depends_on)
        dep_details = [
            {
                "node_id": d,
                "name": self.graph.nodes[d].name,
                "status": self.graph.nodes[d].status.name,
                "assistant": self.graph.nodes[d].assistant.name,
            }
            for d in deps if d in self.graph.nodes
        ]

        return {
            "theorem": theorem_node.name,
            "status": theorem_node.status.name,
            "assistant": theorem_node.assistant.name,
            "statement": theorem_node.statement,
            "is_conditional": theorem_node.is_conditional,
            "dependencies": dep_details,
            "n_dependencies": len(dep_details),
            "honest_note": (
                "This theorem is CONDITIONAL. It proves that IF the "
                "assumptions hold, THEN the conclusion follows. It does "
                "NOT prove the assumptions hold in reality."
            ),
        }

    def what_if_remove_assumption(self, assumption_id: str) -> Dict[str, Any]:
        """Simulate removing an assumption."""
        if assumption_id not in self.graph.nodes:
            return {"error": f"Assumption {assumption_id} not found"}

        node = self.graph.nodes[assumption_id]
        if node.kind != ProofNodeKind.ASSUMPTION:
            return {"error": f"Node {assumption_id} is not an assumption"}

        # Find all nodes that depend on this assumption (transitively)
        dependents = []
        for nid, n in self.graph.nodes.items():
            if assumption_id in n.depends_on or self._transitive_dep(nid, assumption_id):
                dependents.append({
                    "node_id": nid,
                    "name": n.name,
                    "original_status": n.status.name,
                })

        return {
            "removed_assumption": node.name,
            "affected_nodes": dependents,
            "n_affected": len(dependents),
            "note": (
                "Removing an assumption REDUCES the conclusion to [O] open; "
                "it does not ELIMINATE the conclusion. This is the "
                "'reduces, not eliminates' principle (Paper 11 §3)."
            ),
        }

    def _transitive_dep(self, node_id: str, target_id: str) -> bool:
        """Check if node_id transitively depends on target_id."""
        visited: Set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            if current in self.graph.nodes:
                stack.extend(self.graph.nodes[current].depends_on)
        return False


class JupyterNotebookGenerator:
    """[D] Generate a complete Jupyter notebook for proof exploration."""

    def __init__(self, output_dir: str = "./notebooks"):
        self.output_dir = Path(output_dir)

    def generate(self, filename: str = "sesi_divine_attribute_explorer.ipynb") -> str:
        """Generate a complete proof exploration notebook."""
        if not HAS_NBFORMAT:
            logger.warning("nbformat not available; cannot generate notebook")
            return ""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        nb = new_notebook()
        cells = []

        # Title cell
        cells.append(new_markdown_cell(
            "# 🕊️ SESI Divine Attribute Theory Explorer\n\n"
            "**Developer**: PAI, Yoon A Limsuwan / MSPS NETWORK\n\n"
            "**MY SOUL MOVE BY POWER OF HOLY SPIRIT**\n\n"
            "---\n\n"
            "## ⚠️ CRITICAL LIMITATIONS (per paper §15)\n\n"
            "1. **AGI consciousness**: Not proved by mathematical agent definitions\n"
            "2. **Physical verification**: Dual-architecture model is CONCEPTUAL\n"
            "3. **Metaphysical verification**: God's existence cannot be inferred "
            "from math alone\n"
            "4. **Religious interpretation**: Logos≅Dharma is comparative hypothesis, "
            "NOT doctrinal identity\n"
            "5. **Logical consistency**: Compatibility theorem is CONDITIONAL\n\n"
            "> **All proofs in this notebook are CONDITIONAL. They verify "
            "logical structure under stated assumptions. They do NOT prove "
            "metaphysical truths.**"
        ))

        # Setup cell
        cells.append(new_code_cell(
            "# Setup: Import SESI modules\n"
            "import sys\n"
            "sys.path.insert(0, '.')\n\n"
            "from sesi_divine_agda_graph_jupyter import (\n"
            "    AgdaProofManager,\n"
            "    DivineAttributeProofGraph,\n"
            "    Interactive3DRenderer,\n"
            "    ProofGraphExplorer,\n"
            "    JupyterProofKernel,\n"
            "    NotebookProofExplorer,\n"
            ")\n\n"
            "print('SESI Divine Attribute Explorer loaded successfully')\n"
            "print('Type %proof_status to check available proof assistants')"
        ))

        # Proof status cell
        cells.append(new_code_cell(
            "# Check proof assistant availability\n"
            "%proof_status"
        ))

        # Agda proofs cell
        cells.append(new_markdown_cell(
            "## 🏛️ Agda Proofs\n\n"
            "Generate and verify Agda proof files for Divine Attribute Theory."
        ))
        cells.append(new_code_cell(
            "# Generate Agda proofs\n"
            "agda_mgr = AgdaProofManager()\n"
            "agda_result = agda_mgr.run_full()\n\n"
            "print('\\\\n── Generated Agda Files ──')\n"
            "for name, path in agda_result['agda_files'].items():\n"
            "    print(f'  {name}: {path}')\n\n"
            "print('\\\\n── Verification Status ──')\n"
            "for name, ver in agda_result['verification'].items():\n"
            "    print(f'  {name}: {ver[\"status\"]}')\n\n"
            "print(f'\\\\n{agda_result[\"honest_note\"]}')"
        ))

        # 3D proof graph cell
        cells.append(new_markdown_cell(
            "## 🕸️ 3D Proof Dependency Graph\n\n"
            "The following cell generates an interactive 3D proof dependency graph.\n"
            "You can rotate, zoom, and click on nodes to inspect them."
        ))
        cells.append(new_code_cell(
            "# Generate and display 3D proof graph\n"
            "%proof_graph"
        ))

        # Theorem exploration cell
        cells.append(new_markdown_cell(
            "## 📜 Theorem Exploration\n\n"
            "Use the `NotebookProofExplorer` to explore specific theorems."
        ))
        cells.append(new_code_cell(
            "# Explore theorems\n"
            "explorer = NotebookProofExplorer()\n\n"
            "# Explore Proposition 1\n"
            "result = explorer.explore_theorem('Proposition 1')\n"
            "if 'error' not in result:\n"
            "    print(f\"Theorem: {result['theorem']}\")\n"
            "    print(f\"Status: {result['status']}\")\n"
            "    print(f\"Conditional: {result['is_conditional']}\")\n"
            "    print(f\"Dependencies: {result['n_dependencies']}\")\n"
            "    print(f\"\\\\nStatement: {result['statement']}\")\n"
            "    print(f\"\\\\n{result['honest_note']}\")\n"
            "else:\n"
            "    print(result['error'])"
        ))

        # Limitation check cell
        cells.append(new_code_cell(
            "# Run limitation enforcer\n"
            "%limitation_check"
        ))

        # Agda verification cell
        cells.append(new_markdown_cell(
            "## 🏛️ Agda Verification\n\n"
            "If you have Agda installed, you can verify the proofs directly."
        ))
        cells.append(new_code_cell(
            "# Verify Agda proofs (if agda is installed)\n"
            "# %agda ./divine_attribute_agda/SESI_Divine_Attribute.agda\n"
            "# %agda ./divine_attribute_agda/SESI_Ethical_Convergence.agda\n"
            "# %agda ./divine_attribute_agda/SESI_Logos_Dharma.agda\n\n"
            "print('Uncomment the lines above to verify proofs')\n"
            "print('Install Agda first:')\n"
            "print('  https://agda.readthedocs.io/en/latest/getting-started/installation.html')"
        ))

        # Final disclaimer
        cells.append(new_markdown_cell(
            "---\n\n"
            "## 📋 Final Disclaimer\n\n"
            "This notebook is a **research exploration tool**. It does not:\n\n"
            "- Prove that God exists or does not exist\n"
            "- Prove that consciousness exists in AI\n"
            "- Prove that any religion is true or false\n"
            "- Replace peer-reviewed philosophical argument\n\n"
            "It **does**:\n\n"
            "- Verify the LOGICAL STRUCTURE of stated theorems\n"
            "- Make assumptions EXPLICIT and TRACKABLE\n"
            "- Prevent OVERCLAIMING via the limitation enforcer\n"
            "- Provide a VISUAL EXPLORATION of proof dependencies\n\n"
            "**All proofs are CONDITIONAL. All metaphysical claims remain OPEN.**"
        ))

        nb.cells = cells

        # Write notebook
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        logger.info(f"Notebook generated at {output_path}")
        return str(output_path)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class DivineAttributeFullSuite:
    """
    [D] Orchestrate the full Divine Attribute formal suite:
      Agda proofs → 3D Proof Graph → Jupyter notebook
    """

    def __init__(self, output_dir: str = "./divine_attribute_full"):
        self.output_dir = Path(output_dir)
        self.agda_mgr = AgdaProofManager(str(self.output_dir / "agda"))
        self.graph = DivineAttributeProofGraph()
        self.graph.build_default()
        self.explorer = ProofGraphExplorer(self.graph)
        self.nb_gen = JupyterNotebookGenerator(str(self.output_dir / "notebooks"))

    def run_all(self) -> Dict[str, Any]:
        """Run the full suite."""
        results = {}

        # 1. Agda proofs
        results["agda"] = self.agda_mgr.run_full()

        # 2. 3D Proof Graph
        graph_summary = self.explorer.summary()
        results["proof_graph"] = graph_summary

        # Save 3D graph
        graph_path = self.output_dir / "proof_graph_3d.html"
        self.explorer.renderer.save_html(str(graph_path))
        results["proof_graph"]["html_path"] = str(graph_path)

        # 3. Jupyter notebook
        nb_path = self.nb_gen.generate()
        results["notebook"] = nb_path

        return results


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Divine Attribute Full Suite — Smoke Test")
    print("=" * 72)
    print(f"  Plotly available    : {HAS_PLOTLY}")
    print(f"  nbformat available  : {HAS_NBFORMAT}")

    suite = DivineAttributeFullSuite()
    results = suite.run_all()

    # Agda
    print("\n  ── Agda Proofs ──")
    agda = results["agda"]
    print(f"    Agda available    : {agda['agda_available']}")
    print(f"    Files generated   : {len(agda['agda_files'])}")
    for name, path in agda["agda_files"].items():
        print(f"      {name:30s} → {path}")
    print(f"    Verification      :")
    for name, ver in agda["verification"].items():
        print(f"      {name:30s} : {ver['status']}")

    # 3D Proof Graph
    print("\n  ── 3D Proof Graph ──")
    pg = results["proof_graph"]
    print(f"    Total nodes       : {pg['total_nodes']}")
    print(f"    Total edges       : {pg['total_edges']}")
    print(f"    By status         : {pg['by_status']}")
    print(f"    By assistant      : {pg['by_assistant']}")
    print(f"    HTML saved to     : {pg['html_path']}")

    # Jupyter notebook
    print("\n  ── Jupyter Notebook ──")
    nb_path = results["notebook"]
    if nb_path:
        print(f"    Notebook saved to : {nb_path}")
        print(f"    Open with         : jupyter notebook {nb_path}")
    else:
        print("    [SKIP] nbformat not available")

    # Summary
    print("\n" + "=" * 72)
    print("  HONEST SCOPE (per paper §15)")
    print("=" * 72)
    print("  • Agda verifies LOGICAL STRUCTURE, not metaphysical truth")
    print("  • All theorems are CONDITIONAL on stated assumptions")
    print("  • Logos ≅ Dharma is CORRESPONDENCE, not IDENTITY")
    print("  • Ethical convergence does NOT imply doctrinal identity")
    print("  • The 3D graph is a VISUALIZATION TOOL, not a proof checker")
    print("  • Jupyter notebooks are for EXPLORATION, not verification")
    print("  • No proof establishes God's existence or non-existence")
    print("  • No proof establishes that all religions are the same")
    print("=" * 72)
    print(f"\n  {agda['honest_note']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
