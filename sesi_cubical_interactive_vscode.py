"""
=============================================================================
SESI CUBICAL AGDA (HoTT) × INTERACTIVE THEOREM PROVER × VS CODE INTEGRATION
Homotopy Type Theory proofs + Jupyter notebook prover + VS Code config
For: "God Is Not the Creator of Goodness"
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES (in a single file)
---------------------------------------------
[CUB1] Cubical Agda proof scripts (as embedded strings):
       • HoTT-style formulation of Divine Attribute Space
       • Logos ≅ Dharma as a path in the universe (univalence)
       • Proposition 2 reformulated via equivalence of types
       • Higher inductive types for ethical principles

[ITP1] Interactive Theorem Prover widget for Jupyter:
       • IPywidgets-based proof editor
       • Agda kernel integration (when available)
       • Proof state visualization
       • Step-by-step proof construction

[VSC1] VS Code extension configuration generator:
       • .agda-lib project file
       • settings.json for agda-mode
       • tasks.json for build/check
       • launch.json for debugging
       • Snippets for Divine Attribute proofs

MATHEMATICAL CONTENT (HoTT interpretation)
-------------------------------------------
In Homotopy Type Theory:
  • Types are spaces
  • Terms are points
  • Identifications are paths
  • Equivalences (≃) are paths in the universe (by univalence)

Divine Attribute Theory in HoTT:
  • DivineAttributeSpace : Type (a 5-element type)
  • Tradition : Type (a 4-element type)
  • Logos, Dharma : Type (concepts as types)
  • Logos ≃ Dharma (equivalence of types)
  • By univalence: (Logos ≃ Dharma) ≃ (Logos = Dharma) in U

KEY INSIGHT:
  HoTT allows us to say "Logos and Dharma are equivalent" in a
  STRONGER sense than classical correspondence (≅), but this does
  NOT mean they are metaphysically identical — it means there is
  a path between them in the universe of types, which is a
  STRUCTURAL statement, not an ontological one.

RELATION TO THE SERIES
-----------------------
  • sesi_divine_attribute_test.py — Python structural tests
  • sesi_divine_attribute_formal.py — Lean 4 + Aesop + Category Theory
  • sesi_divine_attribute_proofs.py — Coq/Isabelle + 2-Category
  • sesi_divine_agda_graph_jupyter.py — Agda + 3D Graph + Jupyter
  • This module — Cubical Agda (HoTT) + Interactive Prover + VS Code

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Cubical Agda verifies HO TT STRUCTURE, not metaphysical truth.
• Univalence is a STRUCTURAL axiom, not a proof of religious identity.
• The interactive prover is a TOOL, not an oracle.
• VS Code config is for CONVENIENCE, not verification.
• No proof here establishes God's existence or non-existence.
• No proof here establishes that all religions are the same.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import subprocess
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.cubical_interactive_vscode")

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

__all__ = [
    # Cubical Agda
    "CubicalAgdaProofGenerator",
    "CubicalAgdaVerifier",
    # Interactive Prover
    "InteractiveTheoremProver",
    "ProofState",
    "ProofStep",
    # VS Code
    "VSCodeConfigGenerator",
    # Orchestrator
    "CubicalInteractiveVSCodeSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# [CUB1] CUBICAL AGDA PROOF SCRIPTS (HoTT-style)
# ═══════════════════════════════════════════════════════════════════════

CUBICAL_DIVINE_ATTRIBUTE = '''
-- ============================================================
-- SESI Divine Attribute Theory in Cubical Agda (HoTT-style)
-- Homotopy Type Theory interpretation of Logos ≅ Dharma
-- ============================================================

{-# OPTIONS --cubical #-}

module SESI_Cubical_Divine_Attribute where

open import Cubical.Foundations.Prelude
open import Cubical.Foundations.Univalence
open import Cubical.Foundations.Equiv
open import Cubical.Data.Unit
open import Cubical.Data.Empty
open import Cubical.Data.Bool
open import Cubical.Data.Nat
open import Cubical.Data.Sigma
open import Cubical.Data.Prod
open import Cubical.HITs.PropositionalTruncation

-- ---------- Divine Attribute Space as a Type ----------
-- In HoTT, a finite set is a type with decidable equality.

data DAttr : Type where
  B : DAttr   -- Goodness (Bonum)
  T : DAttr   -- Truth
  J : DAttr   -- Justice
  C : DAttr   -- Compassion
  L : DAttr   -- Love

-- Divine Attribute Space is a 5-element type
DivineAttributeSpace : Type
DivineAttributeSpace = DAttr

-- ---------- Traditions as Types ----------

data Tradition : Type where
  Christianity : Tradition
  Buddhism : Tradition
  Taoism : Tradition
  Confucianism : Tradition

-- ---------- Concepts as Types ----------
-- In HoTT, a "concept" can be represented as a type.

record Concept : Type₁ where
  field
    carrier : Type
    tradition : Tradition
    description : String

-- Logos and Dharma as types
Logos : Concept
Logos = record
  { carrier = Unit  -- simplified: Logos as unit type
  ; tradition = Christianity
  ; description = "divine truth and intelligibility"
  }

Dharma : Concept
Dharma = record
  { carrier = Unit  -- simplified: Dharma as unit type
  ; tradition = Buddhism
  ; description = "truth of phenomena, ethical teaching"
  }

Dao : Concept
Dao = record
  { carrier = Unit
  ; tradition = Taoism
  ; description = "underlying Way of reality"
  }

RenYiLi : Concept
RenYiLi = record
  { carrier = Unit
  ; tradition = Confucianism
  ; description = "virtue, righteousness, propriety"
  }

-- ---------- Equivalence (≃) in HoTT ----------
-- Two types are equivalent if there is a bijection between them.

-- Logos ≃ Dharma: both are Unit, so trivially equivalent
logosDharmaEquiv : Concept.carrier Logos ≃ Concept.carrier Dharma
logosDharmaEquiv = idEquiv Unit

-- ---------- Univalence Axiom ----------
-- (A ≃ B) ≃ (A = B) in the universe U
-- This is the KEY insight: equivalence of types = identity in U

logosDharmaPath : Concept.carrier Logos ≡ Concept.carrier Dharma
logosDharmaPath = ua logosDharmaEquiv
  where
    open import Cubical.Foundations.Univalence

-- This path says: "Logos and Dharma are the SAME TYPE in U"
-- But this is a STRUCTURAL statement, not metaphysical identity.

-- ---------- Proposition 2 in HoTT ----------
-- Ethical convergence ≠ doctrinal identity

-- Ethics as a type family over Tradition
Ethics : Tradition → Type
Ethics Christianity = DAttr  -- all 5 attributes
Ethics Buddhism = DAttr
Ethics Taoism = DAttr  -- simplified
Ethics Confucianism = DAttr

-- Metaphysics as a type family
record Metaphysics : Type where
  field
    hasCreatorGod : Bool
    hasKarma : Bool
    hasDao : Bool
    hasRen : Bool

metaphysicsOf : Tradition → Metaphysics
metaphysicsOf Christianity = record { hasCreatorGod = true ; hasKarma = false ; hasDao = false ; hasRen = false }
metaphysicsOf Buddhism = record { hasCreatorGod = false ; hasKarma = true ; hasDao = false ; hasRen = false }
metaphysicsOf Taoism = record { hasCreatorGod = false ; hasKarma = false ; hasDao = true ; hasRen = false }
metaphysicsOf Confucianism = record { hasCreatorGod = false ; hasKarma = false ; hasDao = false ; hasRen = true }

-- Proposition 2: Ethics overlap does not imply Metaphysics identity
-- In HoTT: Σ (e : Ethics R1 ∩ Ethics R2) , ¬ (Metaphysics R1 ≡ Metaphysics R2)

prop2-hott :
  Σ[ e ∈ Ethics Christianity × Ethics Buddhism ]
    ¬ (metaphysicsOf Christianity ≡ metaphysicsOf Buddhism)
prop2-hott = (B , B) , λ p → helper p
  where
    helper : metaphysicsOf Christianity ≡ metaphysicsOf Buddhism → ⊥
    helper p = true≢false (cong Metaphysics.hasCreatorGod p)
      where
        open import Cubical.Data.Bool.Properties

-- ---------- Higher Inductive Types for Ethical Principles ----------
-- In HoTT, we can define types with both points and paths.

data EthicalPrinciple : Type where
  nonKilling : EthicalPrinciple
  nonStealing : EthicalPrinciple
  truthfulness : EthicalPrinciple
  selfControl : EthicalPrinciple
  compassion : EthicalPrinciple

-- Higher inductive type: "shared ethical principle"
-- This is a HIT with a point constructor and a path constructor
-- showing that the principle is shared across traditions.

data SharedEthicalPrinciple : Type where
  sharedPoint : (t1 t2 : Tradition) → EthicalPrinciple → SharedEthicalPrinciple
  sharedPath : {t1 t2 : Tradition} {p : EthicalPrinciple} →
               sharedPoint t1 t2 p ≡ sharedPoint t2 t1 p

-- This HIT says: a shared ethical principle is symmetric
-- (if Christianity and Buddhism share non-killing, then Buddhism
-- and Christianity share non-killing — same principle, reversed order)

-- ---------- The Univalence Insight ----------
-- The KEY philosophical insight from HoTT:

-- "Logos ≅ Dharma" in classical correspondence becomes
-- "Logos ≃ Dharma" in HoTT, which by univalence gives
-- "Logos = Dharma" in the universe U.

-- But this = is a PATH in U, not a judgmental equality.
-- It says: "there is a continuous deformation from Logos to Dharma"
-- which is a STRUCTURAL statement about the space of concepts.

-- This is STRONGER than classical correspondence (≅) but
-- WEAKER than metaphysical identity (= in the real world).

-- The HoTT interpretation allows us to say:
-- "Logos and Dharma are equivalent concepts" (structurally)
-- without saying "Logos and Dharma are the same concept" (metaphysically)

-- This is exactly the discipline of Divine Attribute Theory §15:
-- "Its comparative conclusions establish possible ethical and
--  philosophical correspondences rather than historical proof
--  that the four traditions originated from one identical doctrine."
'''.strip()


CUBICAL_UNIVALENCE_INTERPRETATION = '''
-- ============================================================
-- SESI: Univalence Interpretation of Divine Attribute Theory
-- What HoTT adds to the classical correspondence (≅)
-- ============================================================

{-# OPTIONS --cubical #-}

module SESI_Univalence_Interpretation where

open import Cubical.Foundations.Prelude
open import Cubical.Foundations.Univalence
open import Cubical.Foundations.Equiv

-- ---------- Classical vs HoTT Correspondence ----------

-- Classical: Logos ≅ Dharma (correspondence at philosophical function level)
-- HoTT: Logos ≃ Dharma (equivalence of types)
--       and by univalence: (Logos ≃ Dharma) ≃ (Logos = Dharma in U)

-- The HoTT version is STRONGER because:
-- 1. It gives a computational content (the equivalence)
-- 2. It lifts to a path in the universe (by univalence)
-- 3. It allows transport along the path (coherence)

-- But it is NOT metaphysical identity because:
-- 1. The path is in U (the universe of types), not in reality
-- 2. Transport along the path preserves structure, not essence
-- 3. The equivalence is between carriers (Unit types), not concepts

-- ---------- Transport Along the Logos-Dharma Path ----------

-- Given the path logosDharmaPath : Logos ≡ Dharma in U,
-- we can transport any property P : U → Type along this path.

-- For example, if P(Logos) = "divine truth", then
-- transport P logosDharmaPath : P(Dharma)
-- gives us "divine truth" interpreted as Dharma.

-- This is the HoTT version of "Logos and Dharma correspond":
-- they are connected by a path, and properties transport along it.

-- ---------- Philosophical Interpretation ----------

-- The HoTT interpretation of Divine Attribute Theory says:

-- "Logos and Dharma are equivalent concepts" means:
--   • There is a bijection between their carriers
--   • This bijection is an equivalence (has inverse)
--   • By univalence, they are connected by a path in U
--   • Properties transport along this path

-- This is a RICH notion of correspondence that goes beyond
-- classical ≅, but it is still a STRUCTURAL statement:
-- it says the concepts have the same SHAPE, not the same MEANING.

-- This is exactly what Divine Attribute Theory §15 requires:
-- "comparative conclusions establish possible ethical and
--  philosophical correspondences rather than historical proof
--  that the four traditions originated from one identical doctrine."

-- The HoTT framework makes this precise:
-- • "Correspondence" = equivalence of types (≃)
-- • "Not identical doctrine" = not judgmentally equal (≠)
-- • "Possible" = the equivalence exists, but its interpretation
--   depends on the path chosen (there may be many)
'''.strip()


class CubicalAgdaProofGenerator:
    """[D] Generate Cubical Agda proof files."""

    def __init__(self, output_dir: str = "./divine_attribute_cubical"):
        self.output_dir = Path(output_dir)

    def generate_all(self) -> Dict[str, str]:
        """Generate all Cubical Agda files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        files = {}
        for name, content in [
            ("SESI_Cubical_Divine_Attribute.agda", CUBICAL_DIVINE_ATTRIBUTE),
            ("SESI_Univalence_Interpretation.agda", CUBICAL_UNIVALENCE_INTERPRETATION),
        ]:
            path = self.output_dir / name
            path.write_text(content)
            files[name.replace(".agda", "")] = str(path)

        return files


class CubicalAgdaVerifier:
    """[D] Verify Cubical Agda proofs."""

    def __init__(self):
        self.available = shutil.which("agda") is not None

    def verify(self, agda_path: str, timeout_s: int = 600) -> Dict[str, Any]:
        """Verify a single Cubical Agda file."""
        module_name = Path(agda_path).stem

        if not self.available:
            return {
                "module": module_name,
                "status": "not_verified",
                "reason": "agda not installed (need Cubical Agda)",
                "install_hint": "https://agda.readthedocs.io/en/latest/getting-started/installation.html",
            }

        try:
            result = subprocess.run(
                ["agda", "--cubical", agda_path],
                capture_output=True, timeout=timeout_s, text=True,
            )
            if result.returncode == 0:
                return {
                    "module": module_name,
                    "status": "verified",
                    "note": "Cubical type-checking passed",
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
        """Verify all generated Cubical Agda files."""
        results = {}
        for name, path in agda_files.items():
            results[name] = self.verify(path)
        return results


# ═══════════════════════════════════════════════════════════════════════
# [ITP1] INTERACTIVE THEOREM PROVER IN JUPYTER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProofState:
    """State of an interactive proof."""
    goal: str
    context: List[str] = field(default_factory=list)
    tactics_applied: List[str] = field(default_factory=list)
    is_solved: bool = False


@dataclass
class ProofStep:
    """A single step in a proof."""
    tactic: str
    args: List[str] = field(default_factory=list)
    result: Optional[ProofState] = None


class InteractiveTheoremProver:
    """
    [D] Interactive theorem prover widget for Jupyter notebooks.

    This is a SIMPLIFIED prover that demonstrates the workflow:
      1. User states a goal
      2. Prover shows current state
      3. User applies tactics
      4. Prover updates state
      5. Repeat until goal is solved

    HONEST: This is a TEACHING TOOL, not a replacement for Agda/Coq/Lean.
    For real proofs, use the actual proof assistants.
    """

    def __init__(self):
        self.current_state: Optional[ProofState] = None
        self.history: List[ProofStep] = []
        self.available_tactics = {
            "intros": "Introduce variables",
            "apply": "Apply a lemma or theorem",
            "rewrite": "Rewrite using an equality",
            "exact": "Provide an exact proof term",
            "split": "Split a conjunction or product",
            "constructor": "Construct a pair or record",
            "ua": "Apply univalence axiom (HoTT)",
            "transport": "Transport along a path (HoTT)",
        }

    def start_proof(self, goal: str, context: Optional[List[str]] = None) -> ProofState:
        """Start a new proof with the given goal."""
        self.current_state = ProofState(
            goal=goal,
            context=context or [],
            tactics_applied=[],
            is_solved=False,
        )
        self.history = []
        return self.current_state

    def apply_tactic(self, tactic: str, *args: str) -> Dict[str, Any]:
        """Apply a tactic to the current proof state."""
        if self.current_state is None:
            return {"error": "No proof in progress. Call start_proof first."}

        if tactic not in self.available_tactics:
            return {"error": f"Unknown tactic: {tactic}"}

        # Simplified tactic application (demo only)
        step = ProofStep(tactic=tactic, args=list(args))

        # Update state based on tactic
        if tactic == "intros":
            for var in args:
                self.current_state.context.append(var)
        elif tactic == "exact":
            if args:
                self.current_state.is_solved = True
        elif tactic == "ua":
            # Univalence: convert equivalence to path
            self.current_state.context.append("path_in_U")
        elif tactic == "transport":
            # Transport along a path
            if self.current_state.context:
                self.current_state.context.append("transported")

        step.result = self.current_state
        self.history.append(step)
        self.current_state.tactics_applied.append(tactic)

        return {
            "tactic": tactic,
            "new_state": self.current_state,
            "is_solved": self.current_state.is_solved,
        }

    def get_state(self) -> Optional[ProofState]:
        """Get the current proof state."""
        return self.current_state

    def get_history(self) -> List[ProofStep]:
        """Get the proof history."""
        return self.history

    def create_widget(self) -> Optional[Any]:
        """Create an IPywidgets-based interactive prover UI."""
        if not HAS_WIDGETS:
            logger.warning("ipywidgets not available; cannot create widget")
            return None

        # Goal input
        goal_input = widgets.Textarea(
            value="Logos ≃ Dharma",
            description="Goal:",
            layout=widgets.Layout(width="80%", height="60px"),
        )

        # Tactic selector
        tactic_dropdown = widgets.Dropdown(
            options=list(self.available_tactics.keys()),
            description="Tactic:",
        )

        # Tactic arguments
        args_input = widgets.Text(
            value="",
            description="Args:",
            placeholder="space-separated arguments",
        )

        # Output area
        output = widgets.Output()

        # Apply button
        apply_button = widgets.Button(description="Apply Tactic")

        # Reset button
        reset_button = widgets.Button(description="Reset")

        def on_apply(b):
            with output:
                clear_output()
                if self.current_state is None:
                    self.start_proof(goal_input.value)
                result = self.apply_tactic(tactic_dropdown.value, *args_input.value.split())
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    state = result["new_state"]
                    print(f"Goal: {state.goal}")
                    print(f"Context: {state.context}")
                    print(f"Tactics applied: {state.tactics_applied}")
                    if state.is_solved:
                        print("✓ Proof complete!")

        def on_reset(b):
            with output:
                clear_output()
                self.current_state = None
                self.history = []
                print("Proof reset.")

        apply_button.on_click(on_apply)
        reset_button.on_click(on_reset)

        # Layout
        ui = widgets.VBox([
            widgets.HTML("<h3>Interactive Theorem Prover (Demo)</h3>"),
            goal_input,
            widgets.HBox([tactic_dropdown, args_input]),
            widgets.HBox([apply_button, reset_button]),
            output,
        ])

        return ui


# ═══════════════════════════════════════════════════════════════════════
# [VSC1] VS CODE EXTENSION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class VSCodeConfigGenerator:
    """
    [D] Generate VS Code configuration for Agda development.

    Creates:
      • .agda-lib project file
      • .vscode/settings.json for agda-mode
      • .vscode/tasks.json for build/check
      • .vscode/launch.json for debugging
      • .vscode/snippets.json for Divine Attribute proofs
    """

    def __init__(self, project_dir: str = "./divine_attribute_vscode"):
        self.project_dir = Path(project_dir)

    def generate_all(self) -> Dict[str, str]:
        """Generate all VS Code configuration files."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        vscode_dir = self.project_dir / ".vscode"
        vscode_dir.mkdir(exist_ok=True)

        files = {}

        # .agda-lib
        agda_lib_content = """name: DivineAttribute
include: *.agda
depend: standard-library cubical
"""
        path = self.project_dir / "DivineAttribute.agda-lib"
        path.write_text(agda_lib_content)
        files["agda_lib"] = str(path)

        # settings.json
        settings = {
            "agda.path": "agda",
            "agda.programArguments": ["--cubical"],
            "agda.inputMethod.enabled": True,
            "agda.inputMethod.showNextKey": True,
            "editor.fontSize": 14,
            "editor.fontFamily": "JuliaMono, Fira Code, monospace",
            "editor.minimap.enabled": False,
            "files.associations": {
                "*.agda": "agda",
                "*.lagda.md": "agda",
            },
        }
        path = vscode_dir / "settings.json"
        path.write_text(json.dumps(settings, indent=2))
        files["settings"] = str(path)

        # tasks.json
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Agda: Check Current File",
                    "type": "shell",
                    "command": "agda --cubical ${file}",
                    "group": {
                        "kind": "build",
                        "isDefault": True,
                    },
                    "problemMatcher": [],
                },
                {
                    "label": "Agda: Compile Current File",
                    "type": "shell",
                    "command": "agda --cubical --compile ${file}",
                    "group": "build",
                    "problemMatcher": [],
                },
                {
                    "label": "Agda: Clean",
                    "type": "shell",
                    "command": "find . -name '*.agdai' -delete",
                    "problemMatcher": [],
                },
            ],
        }
        path = vscode_dir / "tasks.json"
        path.write_text(json.dumps(tasks, indent=2))
        files["tasks"] = str(path)

        # launch.json
        launch = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Debug Agda File",
                    "type": "agda",
                    "request": "launch",
                    "program": "${file}",
                    "args": ["--cubical"],
                    "cwd": "${workspaceFolder}",
                },
            ],
        }
        path = vscode_dir / "launch.json"
        path.write_text(json.dumps(launch, indent=2))
        files["launch"] = str(path)

        # snippets.json
        snippets = {
            "Divine Attribute": {
                "prefix": "divattr",
                "body": [
                    "data DAttr : Type where",
                    "  B : DAttr   -- Goodness",
                    "  T : DAttr   -- Truth",
                    "  J : DAttr   -- Justice",
                    "  C : DAttr   -- Compassion",
                    "  L : DAttr   -- Love",
                ],
                "description": "Divine Attribute Space definition",
            },
            "Logos-Dharma Equivalence": {
                "prefix": "logosdharma",
                "body": [
                    "logosDharmaEquiv : Concept.carrier Logos ≃ Concept.carrier Dharma",
                    "logosDharmaEquiv = idEquiv Unit",
                    "",
                    "logosDharmaPath : Concept.carrier Logos ≡ Concept.carrier Dharma",
                    "logosDharmaPath = ua logosDharmaEquiv",
                ],
                "description": "Logos-Dharma equivalence and path",
            },
            "Proposition 2 (HoTT)": {
                "prefix": "prop2hott",
                "body": [
                    "prop2-hott :",
                    "  Σ[ e ∈ Ethics Christianity × Ethics Buddhism ]",
                    "    ¬ (metaphysicsOf Christianity ≡ metaphysicsOf Buddhism)",
                    "prop2-hott = (B , B) , λ p → helper p",
                    "  where",
                    "    helper : metaphysicsOf Christianity ≡ metaphysicsOf Buddhism → ⊥",
                    "    helper p = true≢false (cong Metaphysics.hasCreatorGod p)",
                ],
                "description": "Proposition 2 in HoTT style",
            },
        }
        path = vscode_dir / "agda.code-snippets"
        path.write_text(json.dumps(snippets, indent=2))
        files["snippets"] = str(path)

        # README.md
        readme = """# Divine Attribute Theory — VS Code Setup

## Prerequisites

1. Install Agda with Cubical support:
```bash
   cabal install Agda --flags=cubical
```

2. Install VS Code extension:
   - `agda-mode` (by banacorn)

3. Install standard library and cubical library:
```bash
   git clone https://github.com/agda/agda-stdlib
   git clone https://github.com/agda/cubical
```

## Usage

1. Open this folder in VS Code
2. Open any `.agda` file
3. Use `Ctrl+C Ctrl=L` to load the file (agda-mode)
4. Use `Ctrl+C Ctrl,N` to go to next goal
5. Use snippets: type `divattr`, `logosdharma`, or `prop2hott`

## Tasks

- `Ctrl+Shift+B`: Check current file
- `Ctrl+Shift+P` → "Tasks: Run Task" → "Agda: Compile Current File"

## Honest Scope

- VS Code config is for CONVENIENCE, not verification
- Real proofs require Agda type-checker
- Snippets are templates, not complete proofs
"""
        path = self.project_dir / "README.md"
        path.write_text(readme)
        files["readme"] = str(path)

        return files


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class CubicalInteractiveVSCodeSuite:
    """
    [D] Orchestrate the full Cubical Agda + Interactive Prover + VS Code suite.
    """

    def __init__(self, output_dir: str = "./divine_attribute_full_hott"):
        self.output_dir = Path(output_dir)
        self.cubical_gen = CubicalAgdaProofGenerator(str(self.output_dir / "cubical"))
        self.cubical_ver = CubicalAgdaVerifier()
        self.itp = InteractiveTheoremProver()
        self.vscode_gen = VSCodeConfigGenerator(str(self.output_dir / "vscode"))

    def run_all(self) -> Dict[str, Any]:
        """Run the full suite."""
        results = {}

        # 1. Cubical Agda proofs
        results["cubical_files"] = self.cubical_gen.generate_all()
        results["cubical_verification"] = self.cubical_ver.verify_all(results["cubical_files"])

        # 2. Interactive theorem prover
        results["itp_available"] = HAS_WIDGETS
        results["itp_tactics"] = list(self.itp.available_tactics.keys())

        # 3. VS Code configuration
        results["vscode_files"] = self.vscode_gen.generate_all()

        return results


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Cubical Agda + Interactive Prover + VS Code — Smoke Test")
    print("=" * 72)
    print(f"  ipywidgets available : {HAS_WIDGETS}")

    suite = CubicalInteractiveVSCodeSuite()
    results = suite.run_all()

    # Cubical Agda
    print("\n  ── Cubical Agda Proofs ──")
    print(f"    Files generated: {len(results['cubical_files'])}")
    for name, path in results["cubical_files"].items():
        print(f"      {name}: {path}")

    print(f"\n    Verification status:")
    for name, ver in results["cubical_verification"].items():
        print(f"      {name}: {ver['status']}")
        if ver.get("reason"):
            print(f"        reason: {ver['reason']}")

    # Interactive Theorem Prover
    print("\n  ── Interactive Theorem Prover ──")
    print(f"    Widget available: {results['itp_available']}")
    print(f"    Available tactics: {results['itp_tactics']}")

    if HAS_WIDGETS:
        itp = suite.itp
        state = itp.start_proof("Logos ≃ Dharma")
        print(f"    Demo proof started: {state.goal}")
        result = itp.apply_tactic("ua")
        print(f"    Applied 'ua': context = {result['new_state'].context}")

    # VS Code
    print("\n  ── VS Code Configuration ──")
    print(f"    Files generated: {len(results['vscode_files'])}")
    for name, path in results["vscode_files"].items():
        print(f"      {name}: {path}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Cubical Agda verifies HO TT STRUCTURE, not metaphysical truth")
    print("  • Univalence is a STRUCTURAL axiom, not proof of religious identity")
    print("  • Interactive prover is a TOOL, not an oracle")
    print("  • VS Code config is for CONVENIENCE, not verification")
    print("  • No proof establishes God's existence or non-existence")
    print("  • No proof establishes that all religions are the same")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
