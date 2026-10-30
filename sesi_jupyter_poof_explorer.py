"""
=============================================================================
SESI JUPYTER PROOF EXPLORER INTEGRATION
Notebook interface for proof graphs, proof status, and assumption sensitivity
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
• Displays interactive 3D proof graphs inside Jupyter.
• Provides dropdowns for theorem/assumption exploration.
• Shows proof status across Agda / Coq / Lean / Isabelle.
• Generates notebook cells for a reproducible formal verification session.

HONEST SCOPE
------------
This is a notebook integration layer.
It does not verify proofs by itself.
It calls proof assistant bridges when available.
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("sesi.jupyter_proof")

try:
    from IPython.display import display, HTML, Markdown
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

try:
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

try:
    import nbformat as nbf
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False

try:
    from sesi_proof_graph_3d import (
        ProofGraph3D, ProofKind, build_default_dual_model_graph)
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

__all__ = [
    "JupyterProofExplorer",
    "create_verification_notebook",
]


class JupyterProofExplorer:
    """Notebook integration for proof graph exploration."""

    def __init__(self, graph: Optional[Any] = None):
        if graph is None:
            if not HAS_GRAPH:
                raise RuntimeError("sesi_proof_graph_3d not available")
            graph = build_default_dual_model_graph()
        self.graph = graph

    def show_graph(self, highlight: Optional[list[str]] = None):
        fig = self.graph.to_plotly(highlight=highlight)
        if HAS_IPYTHON:
            display(fig)
        return fig

    def show_node(self, node_id: str):
        node = self.graph.nodes[node_id]
        html = f"""
        <div style="border:1px solid #666; border-radius:8px; padding:12px;
                    background:#111; color:#eee;">
          <h3>{node.node_id}: {node.label}</h3>
          <p><b>Kind:</b> {node.kind.name}</p>
          <p><b>Status:</b> {node.status.name}</p>
          <p><b>Assistant:</b> {node.assistant or "—"}</p>
          <p><b>Depends on:</b> {", ".join(node.depends_on) or "—"}</p>
          <p>{node.statement}</p>
        </div>
        """
        if HAS_IPYTHON:
            display(HTML(html))
        return html

    def interactive(self):
        """Create ipywidgets UI if available."""
        if not (HAS_IPYTHON and HAS_WIDGETS):
            raise RuntimeError("IPython and ipywidgets required")

        node_dropdown = widgets.Dropdown(
            options=[(f"{nid}: {n.label}", nid)
                     for nid, n in self.graph.nodes.items()],
            description="Node:",
            layout=widgets.Layout(width="90%"),
        )

        out = widgets.Output()

        def update(change):
            with out:
                out.clear_output()
                nid = change["new"]
                self.show_node(nid)
                deps = self.graph.ancestors(nid)
                display(HTML(f"<b>Ancestors:</b> {deps}"))
                self.show_graph(highlight=deps + [nid])

        node_dropdown.observe(update, names="value")

        display(node_dropdown, out)
        # initial render
        update({"new": node_dropdown.value})

    def assumption_sensitivity(self, assumption_id: str):
        result = self.graph.remove_assumption(assumption_id)
        if HAS_IPYTHON:
            display(Markdown(f"### Remove assumption `{assumption_id}`"))
            display(result)
        return result

    def proof_status_dashboard(self):
        rows = []
        for nid, node in self.graph.nodes.items():
            rows.append({
                "id": nid,
                "label": node.label,
                "kind": node.kind.name,
                "status": node.status.name,
                "assistant": node.assistant or "",
            })

        if HAS_IPYTHON:
            import pandas as pd
            df = pd.DataFrame(rows)
            display(df)
            return df
        return rows


def create_verification_notebook(
    output_path: str = "SESI_Formal_Verification.ipynb"
) -> str:
    """
    Create a reproducible notebook that:
      • imports the proof modules
      • generates Agda / Coq / Lean / Isabelle scripts
      • displays the 3D graph
      • runs available proof assistants
    """
    if not HAS_NBFORMAT:
        raise RuntimeError("nbformat required: pip install nbformat")

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""
# SESI Formal Verification Notebook

This notebook explores the formal proof structure of the Dual-Model Theory.

**Important:** all proofs are conditional logical checks.
They do not prove or disprove metaphysical existence claims.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sesi_agda_proofs import AgdaProofManager
from sesi_proof_graph_3d import build_default_dual_model_graph
from sesi_jupyter_proof_explorer import JupyterProofExplorer
"""))

    cells.append(nbf.v4.new_markdown_cell("## 1. Generate Agda proof scripts"))
    cells.append(nbf.v4.new_code_cell("""
agda = AgdaProofManager()
agda_result = agda.run(verify=True)
agda_result
"""))

    cells.append(nbf.v4.new_markdown_cell("## 2. Explore proof dependency graph in 3D"))
    cells.append(nbf.v4.new_code_cell("""
graph = build_default_dual_model_graph()
explorer = JupyterProofExplorer(graph)
explorer.show_graph()
"""))

    cells.append(nbf.v4.new_markdown_cell("## 3. Interactive node explorer"))
    cells.append(nbf.v4.new_code_cell("""
explorer.interactive()
"""))

    cells.append(nbf.v4.new_markdown_cell("## 4. Assumption sensitivity"))
    cells.append(nbf.v4.new_code_cell("""
explorer.assumption_sensitivity("A4")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Honest Scope

- Proof assistants verify logical implications.
- They do not verify that the assumptions hold in reality.
- They do not prove God exists or does not exist.
- They do not prove AI consciousness.
"""))

    nb["cells"] = cells
    nbf.write(nb, output_path)
    return output_path


def smoke_test():
    print("=" * 72)
    print("SESI Jupyter Proof Explorer — Smoke Test")
    print("=" * 72)
    print(f"IPython available  : {HAS_IPYTHON}")
    print(f"ipywidgets available: {HAS_WIDGETS}")
    print(f"nbformat available : {HAS_NBFORMAT}")
    print(f"graph module       : {HAS_GRAPH}")

    if HAS_GRAPH:
        graph = build_default_dual_model_graph()
        explorer = JupyterProofExplorer(graph)
        print(f"Graph nodes: {len(graph.nodes)}")
        print(f"Graph edges: {len(graph.edges)}")
        print("Status dashboard:")
        print(explorer.proof_status_dashboard())

    if HAS_NBFORMAT:
        path = create_verification_notebook()
        print(f"Notebook created -> {path}")
    else:
        print("Install nbformat to create notebook:")
        print("  pip install nbformat")

    print("\nHONEST SCOPE:")
    print("  Notebook integration displays proof structure; it does not itself prove theorems.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
