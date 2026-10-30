"""
=============================================================================
SESI INTERACTIVE 3D PROOF DEPENDENCY GRAPH
Assumption → Lemma → Theorem → Limitation visualization
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
• Builds a proof dependency graph.
• Renders it in interactive 3D with Plotly.
• Lets users inspect:
    - assumptions
    - proved / cited / derived / open nodes
    - which theorem depends on which assumptions
    - what fails if an assumption is removed

HONEST SCOPE
------------
This is a visualization and dependency-analysis tool.
It does not verify proofs. Proof verification belongs to Coq / Lean /
Isabelle / Agda.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sesi.proof_graph3d")

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

__all__ = [
    "ProofStatus",
    "ProofKind",
    "ProofNode3D",
    "ProofGraph3D",
]


class ProofStatus(Enum):
    PROVED = auto()
    CITED = auto()
    DERIVED = auto()
    ASSUMED = auto()
    OPEN = auto()
    BLOCKED = auto()


class ProofKind(Enum):
    ASSUMPTION = auto()
    DEFINITION = auto()
    LEMMA = auto()
    THEOREM = auto()
    COROLLARY = auto()
    LIMITATION = auto()
    OPEN_PROBLEM = auto()


STATUS_COLOR = {
    ProofStatus.PROVED: "#2ecc71",
    ProofStatus.CITED: "#3498db",
    ProofStatus.DERIVED: "#9b59b6",
    ProofStatus.ASSUMED: "#f1c40f",
    ProofStatus.OPEN: "#95a5a6",
    ProofStatus.BLOCKED: "#e74c3c",
}


@dataclass
class ProofNode3D:
    node_id: str
    label: str
    kind: ProofKind
    status: ProofStatus
    statement: str
    assistant: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProofGraph3D:
    """Interactive 3D proof dependency graph."""

    def __init__(self, name: str = "SESI Proof Graph"):
        self.name = name
        self.nodes: Dict[str, ProofNode3D] = {}
        self.edges: List[Tuple[str, str]] = []

    def add_node(self, node: ProofNode3D) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, src: str, dst: str) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"Unknown edge endpoint: {src}->{dst}")
        self.edges.append((src, dst))

    def connect_dependencies(self) -> None:
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep in self.nodes:
                    self.edges.append((dep, node.node_id))

    def ancestors(self, node_id: str) -> List[str]:
        """All assumptions/lemmas needed by node_id."""
        seen = set()
        out = []

        def visit(nid: str):
            if nid in seen:
                return
            seen.add(nid)
            node = self.nodes.get(nid)
            if not node:
                return
            for dep in node.depends_on:
                out.append(dep)
                visit(dep)

        visit(node_id)
        return sorted(set(out))

    def descendants(self, node_id: str) -> List[str]:
        """All nodes affected by node_id."""
        children: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for src, dst in self.edges:
            children.setdefault(src, []).append(dst)

        seen = set()
        out = []

        def visit(nid: str):
            for child in children.get(nid, []):
                if child not in seen:
                    seen.add(child)
                    out.append(child)
                    visit(child)

        visit(node_id)
        return sorted(set(out))

    def remove_assumption(self, assumption_id: str) -> Dict[str, Any]:
        node = self.nodes.get(assumption_id)
        if node is None:
            return {"error": f"unknown node {assumption_id}"}
        if node.kind != ProofKind.ASSUMPTION:
            return {"error": f"{assumption_id} is not an assumption"}

        affected = self.descendants(assumption_id)
        return {
            "removed": assumption_id,
            "affected": affected,
            "affected_count": len(affected),
            "note": (
                "Removing an assumption does not prove its dependent conclusions "
                "false; it makes them unsupported/open under this proof graph."
            ),
        }

    def layout_3d(self) -> Dict[str, Tuple[float, float, float]]:
        """Compute 3D layout. Use networkx if available, else spherical fallback."""
        ids = list(self.nodes.keys())

        if HAS_NX:
            G = nx.DiGraph()
            G.add_nodes_from(ids)
            G.add_edges_from(self.edges)
            pos = nx.spring_layout(G, dim=3, seed=42, k=1.5)
            return {nid: tuple(map(float, pos[nid])) for nid in ids}

        # fallback: place nodes on a spiral sphere
        pos = {}
        n = max(len(ids), 1)
        for i, nid in enumerate(ids):
            t = 2 * math.pi * i / n
            z = 1 - 2 * (i + 0.5) / n
            r = math.sqrt(max(0, 1 - z * z))
            pos[nid] = (r * math.cos(t), r * math.sin(t), z)
        return pos

    def to_plotly(self, highlight: Optional[List[str]] = None) -> Any:
        if not HAS_PLOTLY:
            raise RuntimeError("plotly is required: pip install plotly")

        highlight = set(highlight or [])
        pos = self.layout_3d()

        edge_x, edge_y, edge_z = [], [], []
        for src, dst in self.edges:
            x0, y0, z0 = pos[src]
            x1, y1, z1 = pos[dst]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="#888", width=2),
            hoverinfo="none",
            name="dependency",
        )

        node_x, node_y, node_z = [], [], []
        colors, sizes, texts, hovers = [], [], [], []

        for nid, node in self.nodes.items():
            x, y, z = pos[nid]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            base_color = STATUS_COLOR[node.status]
            colors.append("#ffffff" if nid in highlight else base_color)
            sizes.append(11 if nid in highlight else 7)

            texts.append(f"{nid}: {node.label}")
            hovers.append(
                f"<b>{node.label}</b><br>"
                f"id: {nid}<br>"
                f"kind: {node.kind.name}<br>"
                f"status: {node.status.name}<br>"
                f"assistant: {node.assistant or '—'}<br>"
                f"depends on: {', '.join(node.depends_on) or '—'}<br><br>"
                f"{node.statement}"
            )

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers+text",
            text=texts,
            textposition="top center",
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(size=sizes, color=colors, opacity=0.9),
            name="proof nodes",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=self.name,
            template="plotly_dark",
            showlegend=False,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=750,
        )
        return fig

    def save_html(self, path: str = "proof_graph_3d.html",
                  highlight: Optional[List[str]] = None) -> str:
        fig = self.to_plotly(highlight=highlight)
        fig.write_html(path)
        return path


def build_default_dual_model_graph() -> ProofGraph3D:
    """Example graph for the Dual-Model Theory proof structure."""
    g = ProofGraph3D("Dual-Model Theory — Proof Dependency Graph")

    def add(nid, label, kind, status, statement, deps=None, assistant=None):
        g.add_node(ProofNode3D(
            node_id=nid,
            label=label,
            kind=kind,
            status=status,
            statement=statement,
            depends_on=deps or [],
            assistant=assistant,
        ))

    add("A1", "Naturalistic physical representation",
        ProofKind.ASSUMPTION, ProofStatus.ASSUMED,
        "R_N represents observable physical structure.")
    add("A2", "Theological same physical representation",
        ProofKind.ASSUMPTION, ProofStatus.ASSUMED,
        "R_T represents the same observable physical structure.")
    add("A3", "G is metaphysical, not physical variable",
        ProofKind.ASSUMPTION, ProofStatus.ASSUMED,
        "G is not inserted into physical equations.")
    add("A4", "Empirical agreement",
        ProofKind.ASSUMPTION, ProofStatus.ASSUMED,
        "P_N(Q|E) = P_T(Q|E) on shared empirical questions.")
    add("A5", "No same-level contradiction",
        ProofKind.ASSUMPTION, ProofStatus.ASSUMED,
        "No proposition and its negation are asserted at the same semantic level.")

    add("L1", "Shared empirical substructure",
        ProofKind.LEMMA, ProofStatus.DERIVED,
        "R_N and R_T share (X, P, M).",
        deps=["A1", "A2"])
    add("L2", "Pullback non-empty",
        ProofKind.LEMMA, ProofStatus.DERIVED,
        "The shared empirical fiber product is non-empty.",
        deps=["L1", "A4"], assistant="Lean/Mathlib")

    add("T1", "Dual-Model Compatibility",
        ProofKind.THEOREM, ProofStatus.PROVED,
        "Consistent(N) ∧ Consistent(T) ∧ Compatible(N,T), conditional on A1–A5.",
        deps=["A1", "A2", "A3", "A4", "A5", "L1", "L2"],
        assistant="Agda/Coq/Isabelle")
    add("T2", "Explanatory-Level Independence",
        ProofKind.THEOREM, ProofStatus.PROVED,
        "Scientific explanation does not force a unique metaphysical conclusion.",
        deps=["L1"], assistant="Agda/Coq")

    add("Lim1", "No metaphysical existence proof",
        ProofKind.LIMITATION, ProofStatus.BLOCKED,
        "The formal system does not prove God exists or does not exist.",
        deps=["T1"])
    add("Open1", "Do assumptions hold in reality?",
        ProofKind.OPEN_PROBLEM, ProofStatus.OPEN,
        "The assumptions require philosophical/scientific justification.",
        deps=["T1"])

    g.connect_dependencies()
    return g


def smoke_test():
    print("=" * 72)
    print("SESI 3D Proof Graph — Smoke Test")
    print("=" * 72)
    print(f"Plotly available   : {HAS_PLOTLY}")
    print(f"NetworkX available : {HAS_NX}")

    g = build_default_dual_model_graph()
    print(f"Nodes: {len(g.nodes)}")
    print(f"Edges: {len(g.edges)}")

    print("\nDependencies of T1:")
    print(g.ancestors("T1"))

    print("\nRemove A4:")
    print(g.remove_assumption("A4"))

    if HAS_PLOTLY:
        path = g.save_html("dual_model_proof_graph_3d.html")
        print(f"Saved interactive graph -> {path}")
    else:
        print("Install plotly to export interactive 3D graph:")
        print("  pip install plotly networkx")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
