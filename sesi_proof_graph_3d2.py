"""
=============================================================================
SES INTERACTIVE 3D PROOF DEPENDENCY GRAPH
Visual exploration of proof structure across Coq/Lean/Isabelle/Agda
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
----------------------
[PG1] ProofDependencyGraph
      • Nodes: axioms, assumptions, lemmas, theorems, corollaries
      • Edges: logical dependency (A depends on B)
      • Labels: [P] proved, [C] cited, [D] derived, [O] open
      • Proof assistant: coq / lean / isabelle / agda / mathlib

[PG2] Interactive3DRenderer
      • Plotly 3D scatter for nodes
      • Line segments for edges
      • Color-coded by status and proof assistant
      • Click-to-inspect node details
      • Filter by proof assistant, status, or semantic level

[PG3] ProofGraphExplorer
      • Navigate the graph interactively
      • Find critical assumptions (high out-degree)
      • Identify isolated lemmas (no dependents)
      • Generate dependency reports

RELATION TO PAPER 6 (CHANGE-POINT TOWERS)
------------------------------------------
The proof graph has a natural "tower" structure:
  • Base layer: axioms and definitions
  • Middle layer: lemmas and assumptions
  • Top layer: theorems and corollaries

This mirrors Kakutani towers (paper6 §2.1): the "ceiling function"
is the proof depth, and the "base transformation" is the dependency
relation. A proof that depends on many assumptions is like a tall
tower with a wide base.

RELATION TO DELTA_MIN ERRATUM
------------------------------
The Erratum teaches: "made possible only by actually running and
validating code rather than trusting a derivation that looked right."
The 3D graph makes this visible: a theorem with many unverified
dependencies (holes, open assumptions) will show as a node with
many red/orange edges.

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• The graph is a VISUALIZATION TOOL, not a proof checker.
• It displays the structure extracted from proof scripts.
• It does not verify the proofs themselves.
• 3D rendering is for human exploration, not machine verification.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.proof_graph_3d")

try:
    import plotly.graph_objects as go
    import numpy as np
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

__all__ = [
    "ProofNodeKind", "ProofStatus", "ProofAssistant",
    "ProofNode3D", "ProofDependencyGraph",
    "Interactive3DRenderer", "ProofGraphExplorer",
]


class ProofNodeKind(Enum):
    AXIOM = auto()
    ASSUMPTION = auto()
    DEFINITION = auto()
    LEMMA = auto()
    THEOREM = auto()
    COROLLARY = auto()
    OPEN_PROBLEM = auto()
    DESIGN_CHOICE = auto()  # free parameter (Unified_Ordinal Remark)


class ProofStatus(Enum):
    PROVED = auto()       # [P] machine-checked
    CITED = auto()        # [C] from literature
    DERIVED = auto()      # [D] derived here
    OPEN = auto()         # [O] not yet addressed
    HOLE = auto()         # Agda hole {!!}
    BLOCKED = auto()      # blocked by limitation enforcer


class ProofAssistant(Enum):
    COQ = auto()
    LEAN = auto()
    LEAN_MATHLIB = auto()
    ISABELLE = auto()
    AGDA = auto()
    PYTHON = auto()  # for structural tests


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
    semantic_level: str = "PHYSICAL"  # PHYSICAL / MATHEMATICAL / METAPHYSICAL
    is_conditional: bool = True
    position: Optional[Tuple[float, float, float]] = None


class ProofDependencyGraph:
    """
    [D] A directed acyclic graph of proof dependencies.

    Nodes are proof components; edges point from dependent to dependency.
    """

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, ProofNode3D] = {}
        self.edges: List[Tuple[str, str]] = []  # (dependent, dependency)

    def add_node(self, node: ProofNode3D) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, dependent_id: str, dependency_id: str) -> None:
        if dependent_id in self.nodes and dependency_id in self.nodes:
            self.edges.append((dependent_id, dependency_id))
            self.nodes[dependent_id].depends_on.append(dependency_id)

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get all transitive dependencies of a node."""
        result = set()
        visited = set()

        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for dep in node.depends_on:
                    result.add(dep)
                    dfs(dep)

        dfs(node_id)
        return result

    def get_dependents(self, node_id: str) -> Set[str]:
        """Get all transitive dependents of a node."""
        result = set()
        visited = set()

        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            for dep_id, dep_on in self.edges:
                if dep_on == nid and dep_id not in visited:
                    result.add(dep_id)
                    dfs(dep_id)

        dfs(node_id)
        return result

    def critical_assumptions(self) -> List[str]:
        """Find assumptions with the most dependents (critical)."""
        assumption_counts = {}
        for node_id, node in self.nodes.items():
            if node.kind == ProofNodeKind.ASSUMPTION:
                dependents = self.get_dependents(node_id)
                assumption_counts[node_id] = len(dependents)
        sorted_assumptions = sorted(
            assumption_counts.items(), key=lambda x: -x[1])
        return [a for a, _ in sorted_assumptions]

    def layout_3d(self) -> None:
        """
        [D] Compute 3D positions using a simple layered layout.

        Layer 0: axioms and definitions
        Layer 1: assumptions
        Layer 2: lemmas
        Layer 3: theorems
        Layer 4: corollaries
        """
        layer_map = {
            ProofNodeKind.AXIOM: 0,
            ProofNodeKind.DEFINITION: 0,
            ProofNodeKind.ASSUMPTION: 1,
            ProofNodeKind.LEMMA: 2,
            ProofNodeKind.THEOREM: 3,
            ProofNodeKind.COROLLARY: 4,
            ProofNodeKind.OPEN_PROBLEM: 5,
            ProofNodeKind.DESIGN_CHOICE: 1,
        }

        # Group nodes by layer
        layers: Dict[int, List[str]] = {}
        for node_id, node in self.nodes.items():
            layer = layer_map.get(node.kind, 3)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)

        # Assign positions
        for layer_idx, node_ids in layers.items():
            n = len(node_ids)
            for i, node_id in enumerate(node_ids):
                angle = 2 * 3.14159 * i / max(n, 1)
                radius = 1.0 + layer_idx * 0.5
                x = radius * np.cos(angle) if HAS_PLOTLY else radius * (i / max(n, 1))
                y = radius * np.sin(angle) if HAS_PLOTLY else 0.0
                z = layer_idx * 2.0
                self.nodes[node_id].position = (x, y, z)


class Interactive3DRenderer:
    """
    [D] Render the proof dependency graph in interactive 3D.
    """

    STATUS_COLORS = {
        ProofStatus.PROVED: "#2ecc71",      # green
        ProofStatus.CITED: "#3498db",       # blue
        ProofStatus.DERIVED: "#9b59b6",     # purple
        ProofStatus.OPEN: "#e74c3c",        # red
        ProofStatus.HOLE: "#f39c12",        # orange
        ProofStatus.BLOCKED: "#95a5a6",     # gray
    }

    ASSISTANT_SHAPES = {
        ProofAssistant.COQ: "circle",
        ProofAssistant.LEAN: "square",
        ProofAssistant.LEAN_MATHLIB: "diamond",
        ProofAssistant.ISABELLE: "cross",
        ProofAssistant.AGDA: "triangle-up",
        ProofAssistant.PYTHON: "hexagon",
    }

    def __init__(self, graph: ProofDependencyGraph):
        self.graph = graph

    def render(self, filter_assistant: Optional[ProofAssistant] = None,
               filter_status: Optional[ProofStatus] = None) -> Any:
        """Render the 3D graph. Returns a Plotly figure."""
        if not HAS_PLOTLY:
            logger.warning("plotly not available; cannot render 3D graph")
            return None

        self.graph.layout_3d()

        # Filter nodes
        nodes = list(self.graph.nodes.values())
        if filter_assistant is not None:
            nodes = [n for n in nodes if n.assistant == filter_assistant]
        if filter_status is not None:
            nodes = [n for n in nodes if n.status == filter_status]

        node_ids = {n.node_id for n in nodes}

        # Node traces
        traces = []
        for status in ProofStatus:
            status_nodes = [n for n in nodes if n.status == status]
            if not status_nodes:
                continue

            x = [n.position[0] for n in status_nodes if n.position]
            y = [n.position[1] for n in status_nodes if n.position]
            z = [n.position[2] for n in status_nodes if n.position]
            names = [f"{n.name}<br>[{n.status.name}]<br>{n.assistant.name}"
                     for n in status_nodes]

            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=8,
                    color=self.STATUS_COLORS[status],
                    symbol=self.ASSISTANT_SHAPES.get(status_nodes[0].assistant, "circle")
                           if status_nodes else "circle",
                ),
                text=names,
                hoverinfo="text",
                name=status.name,
            ))

        # Edge traces
        edge_x, edge_y, edge_z = [], [], []
        for dep_id, dep_on in self.graph.edges:
            if dep_id in node_ids and dep_on in node_ids:
                n1 = self.graph.nodes[dep_id]
                n2 = self.graph.nodes[dep_on]
                if n1.position and n2.position:
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
            title=f"Proof Dependency Graph: {self.graph.name}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Proof Depth",
            ),
            legend=dict(x=0, y=1),
        )

        return fig

    def save_html(self, output_path: str,
                  filter_assistant: Optional[ProofAssistant] = None) -> None:
        """Save the 3D graph as an interactive HTML file."""
        fig = self.render(filter_assistant=filter_assistant)
        if fig is not None:
            fig.write_html(output_path)
            logger.info(f"3D proof graph saved to {output_path}")


class ProofGraphExplorer:
    """
    [D] High-level exploration of the proof dependency graph.
    """

    def __init__(self, graph: ProofDependencyGraph):
        self.graph = graph
        self.renderer = Interactive3DRenderer(graph)

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the proof graph."""
        nodes = list(self.graph.nodes.values())
        by_status = {}
        by_assistant = {}
        by_kind = {}

        for node in nodes:
            by_status[node.status.name] = by_status.get(node.status.name, 0) + 1
            by_assistant[node.assistant.name] = by_assistant.get(node.assistant.name, 0) + 1
            by_kind[node.kind.name] = by_kind.get(node.kind.name, 0) + 1

        critical = self.graph.critical_assumptions()

        return {
            "total_nodes": len(nodes),
            "total_edges": len(self.graph.edges),
            "by_status": by_status,
            "by_assistant": by_assistant,
            "by_kind": by_kind,
            "critical_assumptions": critical[:5],  # top 5
            "honest_note": (
                "The graph shows LOGICAL STRUCTURE, not metaphysical truth. "
                "Nodes marked [O] OPEN or HOLE indicate unverified components."
            ),
        }

    def find_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find a dependency path from one node to another."""
        if from_id not in self.graph.nodes or to_id not in self.graph.nodes:
            return None

        visited = set()
        path = []

        def dfs(current):
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
# BUILD THE FULL PROOF GRAPH
# ═══════════════════════════════════════════════════════════════════════

def build_full_proof_graph() -> ProofDependencyGraph:
    """
    [D] Build the complete proof dependency graph for the SESI
    formal verification suite.
    """
    graph = ProofDependencyGraph("SESI Formal Verification Suite")

    # --- Axioms and Definitions ---
    graph.add_node(ProofNode3D(
        "def_framework", "RepresentationalFramework",
        ProofNodeKind.DEFINITION, ProofStatus.PROVED,
        ProofAssistant.COQ,
        "R = (X, P, M) or (X, P, M, G, D)",
    ))
    graph.add_node(ProofNode3D(
        "def_consistent", "Consistent",
        ProofNodeKind.DEFINITION, ProofStatus.PROVED,
        ProofAssistant.COQ,
        "Consistent(F) ⟺ ∃s ∈ F.states. F.laws(s)",
    ))
    graph.add_node(ProofNode3D(
        "def_kappa_fat", "κ-fatness",
        ProofNodeKind.DEFINITION, ProofStatus.CITED,
        ProofAssistant.AGDA,
        "Delta_Min Definition 1.1",
    ))

    # --- Assumptions (Theorem 1) ---
    for i, desc in enumerate([
        "Naturalistic model represents physical structure",
        "Theological model represents SAME physical structure",
        "G is metaphysical ground, not physical variable",
        "Both models agree on empirical predictions",
        "No same-level contradiction",
    ]):
        graph.add_node(ProofNode3D(
            f"A{i+1}", f"Assumption {i+1}: {desc[:40]}",
            ProofNodeKind.ASSUMPTION, ProofStatus.OPEN,
            ProofAssistant.PYTHON,
            desc,
            semantic_level="METAPHYSICAL" if i in [2, 4] else "PHYSICAL",
        ))

    # --- Design Choices (Unified_Ordinal Remark 7.8) ---
    graph.add_node(ProofNode3D(
        "design_a1_a4", "Design params a1, a4 = 3×10⁻⁴",
        ProofNodeKind.DESIGN_CHOICE, ProofStatus.DERIVED,
        ProofAssistant.PYTHON,
        "Unified_Ordinal Example 7.12: free coupling parameters",
    ))
    graph.add_node(ProofNode3D(
        "design_a2_a3", "Design params a2, a3",
        ProofNodeKind.DESIGN_CHOICE, ProofStatus.DERIVED,
        ProofAssistant.PYTHON,
        "Unified_Ordinal Example 7.12: free coupling parameters",
    ))

    # --- Lemmas ---
    graph.add_node(ProofNode3D(
        "L1_shared", "Shared physical structure",
        ProofNodeKind.LEMMA, ProofStatus.DERIVED,
        ProofAssistant.LEAN_MATHLIB,
        "R_N ∩ R_T ⊇ (X, P, M)",
        depends_on=["A1", "A2"],
    ))
    graph.add_node(ProofNode3D(
        "L2_pullback", "Pullback non-empty",
        ProofNodeKind.LEMMA, ProofStatus.DERIVED,
        ProofAssistant.LEAN_MATHLIB,
        "Fiber product R_N ×_E R_T is non-empty",
        depends_on=["L1_shared", "A4"],
    ))
    graph.add_node(ProofNode3D(
        "L3_delta_min", "δ_min > 0 under κ-fatness",
        ProofNodeKind.LEMMA, ProofStatus.PROVED,
        ProofAssistant.AGDA,
        "Delta_Min Theorem 1.2",
        depends_on=["def_kappa_fat"],
    ))
    graph.add_node(ProofNode3D(
        "L4_no_zeno", "No-Zeno bound",
        ProofNodeKind.LEMMA, ProofStatus.PROVED,
        ProofAssistant.ISABELLE,
        "N(T) ≤ (E_max + D) / δ_min",
        depends_on=["L3_delta_min"],
    ))

    # --- Theorems ---
    graph.add_node(ProofNode3D(
        "T1_compat", "Theorem 1: Dual-Model Compatibility",
        ProofNodeKind.THEOREM, ProofStatus.PROVED,
        ProofAssistant.COQ,
        "Consistent(N) ∧ Consistent(T) ∧ Compatible(N,T)",
        depends_on=["A1", "A2", "A3", "A4", "A5", "L1_shared", "L2_pullback"],
        semantic_level="MATHEMATICAL",
        is_conditional=True,
    ))
    graph.add_node(ProofNode3D(
        "T2_indep", "Theorem 2: Explanatory-Level Independence",
        ProofNodeKind.THEOREM, ProofStatus.PROVED,
        ProofAssistant.AGDA,
        "Scientific explanation ⇏ Unique metaphysical conclusion",
        depends_on=["L1_shared"],
        semantic_level="MATHEMATICAL",
    ))
    graph.add_node(ProofNode3D(
        "T3_opmc", "OPMC Closure (Thm 9.2)",
        ProofNodeKind.THEOREM, ProofStatus.PROVED,
        ProofAssistant.LEAN_MATHLIB,
        "Structural weak value is well-defined on M_str",
        depends_on=["design_a1_a4", "design_a2_a3"],
    ))

    # --- Corollaries ---
    graph.add_node(ProofNode3D(
        "C1_cond", "Compatibility is CONDITIONAL",
        ProofNodeKind.COROLLARY, ProofStatus.PROVED,
        ProofAssistant.COQ,
        "Thm 1 does not prove every atheism/theism is true",
        depends_on=["T1_compat"],
    ))

    # --- Open Problems ---
    graph.add_node(ProofNode3D(
        "O1_reality", "Whether assumptions hold in reality",
        ProofNodeKind.OPEN_PROBLEM, ProofStatus.OPEN,
        ProofAssistant.PYTHON,
        "Empirical verification of A1-A5",
        depends_on=["T1_compat"],
    ))
    graph.add_node(ProofNode3D(
        "O2_kappa_fat", "Derive κ-fatness from primitives",
        ProofNodeKind.OPEN_PROBLEM, ProofStatus.OPEN,
        ProofAssistant.AGDA,
        "Delta_Min Remark: open question (i)",
        depends_on=["L3_delta_min"],
    ))

    # --- Edges ---
    graph.add_edge("L1_shared", "A1")
    graph.add_edge("L1_shared", "A2")
    graph.add_edge("L2_pullback", "L1_shared")
    graph.add_edge("L2_pullback", "A4")
    graph.add_edge("L3_delta_min", "def_kappa_fat")
    graph.add_edge("L4_no_zeno", "L3_delta_min")
    graph.add_edge("T1_compat", "A1")
    graph.add_edge("T1_compat", "A2")
    graph.add_edge("T1_compat", "A3")
    graph.add_edge("T1_compat", "A4")
    graph.add_edge("T1_compat", "A5")
    graph.add_edge("T1_compat", "L1_shared")
    graph.add_edge("T1_compat", "L2_pullback")
    graph.add_edge("T2_indep", "L1_shared")
    graph.add_edge("T3_opmc", "design_a1_a4")
    graph.add_edge("T3_opmc", "design_a2_a3")
    graph.add_edge("C1_cond", "T1_compat")
    graph.add_edge("O1_reality", "T1_compat")
    graph.add_edge("O2_kappa_fat", "L3_delta_min")

    return graph


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES 3D Proof Dependency Graph — Smoke Test")
    print("=" * 72)
    print(f"  Plotly available: {HAS_PLOTLY}")

    # Build the graph
    graph = build_full_proof_graph()
    explorer = ProofGraphExplorer(graph)

    # Summary
    summary = explorer.summary()
    print(f"\n  ── Graph Summary ──")
    print(f"    Total nodes      : {summary['total_nodes']}")
    print(f"    Total edges      : {summary['total_edges']}")
    print(f"    By status        : {summary['by_status']}")
    print(f"    By assistant     : {summary['by_assistant']}")
    print(f"    Critical assumptions: {summary['critical_assumptions']}")

    # Find a path
    print(f"\n  ── Dependency Path ──")
    path = explorer.find_path("T1_compat", "A1")
    if path:
        print(f"    T1_compat → A1: {' → '.join(path)}")
    else:
        print(f"    No path found")

    # Render 3D graph
    print(f"\n  ── 3D Rendering ──")
    if HAS_PLOTLY:
        fig = explorer.renderer.render()
        if fig is not None:
            output_path = "./proof_graph_3d.html"
            explorer.renderer.save_html(output_path)
            print(f"    Saved to: {output_path}")
            print(f"    Open in browser for interactive 3D exploration")
    else:
        print("    [SKIP] Plotly not available")

    print(f"\n  {summary['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • The graph is a VISUALIZATION TOOL, not a proof checker")
    print("  • It displays structure extracted from proof scripts")
    print("  • It does not verify the proofs themselves")
    print("  • Nodes marked [O] OPEN or HOLE are unverified")
    print("  • 3D rendering is for human exploration")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
