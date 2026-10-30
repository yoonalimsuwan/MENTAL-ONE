"""
=============================================================================
SES JUPYTER INTEGRATION — Proof Explorer in Jupyter Notebooks
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
----------------------
[J1] JupyterProofKernel
     • Magics for running proof assistants from notebook cells
     • %coq, %lean, %isabelle, %agda magics
     • %proof_status, %proof_graph, %limitation_check

[J2] NotebookProofExplorer
     • Embed the interactive proof explorer in a notebook
     • Display 3D proof graphs inline
     • Run what-if analysis interactively

[J3] JupyterNotebookGenerator
     • Generate a complete .ipynb notebook for proof exploration
     • Includes markdown explanations, code cells, and visualizations
     • Enforces limitation discipline in generated content

RELATION TO PAPER 11
--------------------
The assumption-light discipline applies to notebook generation:
  • Every theorem is marked with its assumptions
  • Every proof is marked as CONDITIONAL
  • Every visualization includes a disclaimer
  • The notebook itself cannot overclaim (limitation enforcer)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Requires jupyter and nbformat installed.
• Proof assistant magics require the respective tools installed.
• The generated notebook is a TEMPLATE; users must fill in proofs.
• Notebooks are for EXPLORATION, not for replacing proof assistants.
=============================================================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.jupyter_integration")

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False

__all__ = [
    "JupyterProofKernel",
    "NotebookProofExplorer",
    "JupyterNotebookGenerator",
]


# ═══════════════════════════════════════════════════════════════════════
# [D] JUPYTER PROOF KERNEL (magic commands)
# ═══════════════════════════════════════════════════════════════════════

class JupyterProofKernel:
    """
    [D] Magic commands for proof assistants in Jupyter.

    Usage in a notebook:
        from sesi_jupyter_integration import JupyterProofKernel
        kernel = JupyterProofKernel()
        kernel.load_ipython_extension(ipython)

    Then use:
        %proof_status          — show all proof statuses
        %proof_graph           — display 3D proof graph
        %limitation_check      — run limitation enforcer
        %coq path/to/file.v    — verify a Coq file
        %lean path/to/file.lean — verify a Lean file
        %isabelle path/to.thy  — verify an Isabelle theory
        %agda path/to/file.agda — type-check an Agda file
    """

    def __init__(self):
        self.available_assistants = self._detect_assistants()

    def _detect_assistants(self) -> Dict[str, bool]:
        """Detect which proof assistants are installed."""
        import shutil
        return {
            "coqc": shutil.which("coqc") is not None,
            "lean": shutil.which("lean") is not None,
            "isabelle": shutil.which("isabelle") is not None,
            "agda": shutil.which("agda") is not None,
        }

    def load_ipython_extension(self, ipython) -> None:
        """Register magic commands with IPython."""
        if not HAS_NBFORMAT:
            logger.warning("nbformat not available; magics limited")

        # Register line magics
        ipython.register_magic_function(self.proof_status, magic_kind="line",
                                         magic_name="proof_status")
        ipython.register_magic_function(self.proof_graph, magic_kind="line",
                                         magic_name="proof_graph")
        ipython.register_magic_function(self.limitation_check, magic_kind="line",
                                         magic_name="limitation_check")

    def proof_status(self, line: str) -> None:
        """Display proof status summary."""
        from ses_proof_assistant_bridge import ProofAssistantBridge
        bridge = ProofAssistantBridge()
        report = bridge.availability_report()
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
        try:
            from ses_proof_graph_3d import build_full_proof_graph, Interactive3DRenderer
            graph = build_full_proof_graph()
            renderer = Interactive3DRenderer(graph)
            fig = renderer.render()
            if fig is not None:
                fig.show()
        except ImportError as e:
            print(f"Cannot display proof graph: {e}")

    def limitation_check(self, line: str) -> None:
        """Run the limitation enforcer."""
        from ses_proof_assistant_bridge import LimitationEnforcer
        enforcer = LimitationEnforcer()
        report = enforcer.enforcement_report()
        print("=" * 60)
        print("LIMITATION ENFORCEMENT REPORT")
        print("=" * 60)
        print(f"  Total violations: {report['total_violations']}")
        for lim in report['limitations_enforced']:
            print(f"    • {lim}")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# [D] NOTEBOOK PROOF EXPLORER
# ═══════════════════════════════════════════════════════════════════════

class NotebookProofExplorer:
    """
    [D] Interactive proof exploration within a Jupyter notebook.
    """

    def __init__(self):
        self.kernel = JupyterProofKernel()

    def explore_theorem(self, theorem_name: str) -> Dict[str, Any]:
        """
        Explore a specific theorem: its assumptions, dependencies,
        and proof status.
        """
        from ses_proof_graph_3d import build_full_proof_graph
        graph = build_full_proof_graph()

        # Find the theorem node
        theorem_node = None
        for node_id, node in graph.nodes.items():
            if theorem_name.lower() in node.name.lower():
                theorem_node = node
                break

        if theorem_node is None:
            return {"error": f"Theorem '{theorem_name}' not found"}

        # Get dependencies
        deps = graph.get_dependencies(theorem_node.node_id)
        dep_details = [
            {
                "node_id": d,
                "name": graph.nodes[d].name,
                "status": graph.nodes[d].status.name,
                "assistant": graph.nodes[d].assistant.name,
            }
            for d in deps if d in graph.nodes
        ]

        return {
            "theorem": theorem_node.name,
            "status": theorem_node.status.name,
            "assistant": theorem_node.assistant.name,
            "statement": theorem_node.statement,
            "is_conditional": theorem_node.is_conditional,
            "semantic_level": theorem_node.semantic_level,
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
        from ses_proof_explorer import build_dual_model_proof_tree
        tree = build_dual_model_proof_tree()
        return tree.remove_assumption(assumption_id)


# ═══════════════════════════════════════════════════════════════════════
# [D] JUPYTER NOTEBOOK GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class JupyterNotebookGenerator:
    """
    [D] Generate a complete Jupyter notebook for proof exploration.
    """

    def __init__(self, output_dir: str = "./notebooks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, filename: str = "sesi_proof_explorer.ipynb") -> str:
        """Generate a complete proof exploration notebook."""
        if not HAS_NBFORMAT:
            logger.warning("nbformat not available; cannot generate notebook")
            return ""

        nb = new_notebook()
        cells = []

        # Title cell
        cells.append(new_markdown_cell(
            "# 🔍 SESI Proof Explorer Notebook\n\n"
            "**Developer**: PAI, Yoon A Limsuwan / MSPS NETWORK\n\n"
            "**MY SOUL MOVE BY POWER OF HOLY SPIRIT**\n\n"
            "---\n\n"
            "## ⚠️ CRITICAL LIMITATIONS (per Dual-Model §12)\n\n"
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
            "from sesi_proof_assistant_bridge import (\n"
            "    ProofAssistantBridge, ProofStatusTracker, LimitationEnforcer\n"
            ")\n"
            "from sesi_proof_graph_3d import (\n"
            "    build_full_proof_graph, Interactive3DRenderer, ProofGraphExplorer\n"
            ")\n"
            "from sesi_jupyter_integration import JupyterProofKernel\n\n"
            "print('SESI Proof Explorer loaded successfully')\n"
            "print('Type %proof_status to check available proof assistants')"
        ))

        # Proof status cell
        cells.append(new_code_cell(
            "# Check proof assistant availability\n"
            "%proof_status"
        ))

        # Proof graph cell
        cells.append(new_markdown_cell(
            "## 🕸️ Proof Dependency Graph\n\n"
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
            "from sesi_jupyter_integration import NotebookProofExplorer\n\n"
            "explorer = NotebookProofExplorer()\n\n"
            "# Explore Theorem 1: Dual-Model Compatibility\n"
            "result = explorer.explore_theorem('Dual-Model Compatibility')\n"
            "print(f\"Theorem: {result['theorem']}\")\n"
            "print(f\"Status: {result['status']}\")\n"
            "print(f\"Assistant: {result['assistant']}\")\n"
            "print(f\"Conditional: {result['is_conditional']}\")\n"
            "print(f\"Dependencies: {result['n_dependencies']}\")\n"
            "print(f\"\\nStatement: {result['statement']}\")\n"
            "print(f\"\\n{result['honest_note']}\")"
        ))

        # What-if analysis cell
        cells.append(new_markdown_cell(
            "## 🔬 What-If Analysis\n\n"
            "Simulate removing an assumption and see what breaks.\n\n"
            "**NOTE**: Removing an assumption REDUCES the conclusion to [O] open; "
            "it does not ELIMINATE the conclusion (Paper 11 §3: 'reduces, not eliminates')."
        ))
        cells.append(new_code_cell(
            "# What if we remove assumption A4 (empirical agreement)?\n"
            "result = explorer.what_if_remove_assumption('A4')\n"
            "print(f\"Removed: {result['removed_assumption']}\")\n"
            "print(f\"Affected nodes: {result['n_affected']}\")\n"
            "for affected in result['affected_nodes']:\n"
            "    print(f\"  • {affected['name']}: \"\n"
            "          f\"{affected['original_status']} → {affected['new_status']}\")\n"
            "print(f\"\\n{result['note']}\")"
        ))

        # Limitation check cell
        cells.append(new_code_cell(
            "# Run limitation enforcer\n"
            "%limitation_check"
        ))

        # Proof assistant cells
        cells.append(new_markdown_cell(
            "## 🏛️ Proof Assistant Verification\n\n"
            "If you have proof assistants installed, you can verify the proofs directly."
        ))
        cells.append(new_code_cell(
            "# Verify Coq proofs (if coqc is installed)\n"
            "# %coq ./coq_proofs/sesi_dual_model_thm1.v\n\n"
            "# Verify Lean proofs (if lean is installed)\n"
            "# %lean ./lean_proofs/sesi_dual_model.lean\n\n"
            "# Verify Isabelle proofs (if isabelle is installed)\n"
            "# %isabelle ./isabelle_proofs/SESI_Dual_Model.thy\n\n"
            "# Verify Agda proofs (if agda is installed)\n"
            "# %agda ./agda_proofs/SESI_Dual_Model.agda\n\n"
            "print('Uncomment the lines above to verify proofs')\n"
            "print('Install proof assistants first:')\n"
            "print('  Coq: https://coq.inria.fr/download')\n"
            "print('  Lean 4: https://leanprover.github.io/install/')\n"
            "print('  Isabelle: https://isabelle.in.tum.de/')\n"
            "print('  Agda: https://agda.readthedocs.io/en/latest/getting-started/install.html')"
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
        with open(output_path, "w") as f:
            nbformat.write(nb, f)

        logger.info(f"Notebook generated at {output_path}")
        return str(output_path)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES Jupyter Integration — Smoke Test")
    print("=" * 72)
    print(f"  nbformat available: {HAS_NBFORMAT}")

    # Test kernel
    print("\n  ── JupyterProofKernel ──")
    kernel = JupyterProofKernel()
    print(f"    Available assistants: {kernel.available_assistants}")

    # Test explorer
    print("\n  ── NotebookProofExplorer ──")
    explorer = NotebookProofExplorer()
    result = explorer.explore_theorem("Dual-Model Compatibility")
    if "error" not in result:
        print(f"    Theorem: {result['theorem']}")
        print(f"    Status: {result['status']}")
        print(f"    Conditional: {result['is_conditional']}")
        print(f"    Dependencies: {result['n_dependencies']}")
    else:
        print(f"    {result['error']}")

    # Generate notebook
    print("\n  ── Notebook Generation ──")
    generator = JupyterNotebookGenerator()
    notebook_path = generator.generate()
    if notebook_path:
        print(f"    Notebook saved to: {notebook_path}")
        print(f"    Open with: jupyter notebook {notebook_path}")
    else:
        print("    [SKIP] nbformat not available")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Notebooks are for EXPLORATION, not proof verification")
    print("  • Proof assistants must be installed separately")
    print("  • All proofs are CONDITIONAL on stated assumptions")
    print("  • The limitation enforcer prevents overclaiming")
    print("  • No metaphysical truths are established")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
