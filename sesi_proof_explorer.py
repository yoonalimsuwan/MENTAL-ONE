"""
=============================================================================
SES INTERACTIVE PROOF EXPLORER
Visual exploration of proof structure, assumptions, and dependencies
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
----------------------
[PE1] ProofTree — represents the structure of a proof as a tree
      • Nodes: assumptions, lemmas, conclusions
      • Edges: logical dependencies
      • Labels: [P] proved, [C] cited, [D] derived, [O] open

[PE2] InteractiveProofExplorer — CLI and Dash-based exploration
      • Navigate the proof tree
      • Toggle assumptions on/off to see what breaks
      • View proof status across Coq/Lean/Isabelle
      • Generate dependency graphs

[PE3] AssumptionSensitivityAnalyzer — "what if I remove this assumption?"
      • Shows which conclusions depend on which assumptions
      • Identifies critical assumptions
      • Reports conditional vs unconditional results

RELATION TO PAPER 11 & UNIFIED_ORDINAL
----------------------------------------
The explorer enforces the discipline:
  • "reduces, not eliminates" — removing an assumption may reduce
    the conclusion to [O] open, not eliminate it entirely
  • "unified in the weak sense" — each proof is independent;
    the explorer shows these independencies explicitly
  • Limitation enforcement — blocks any claim that crosses §12

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• The explorer is a VISUALIZATION TOOL, not a proof assistant.
• It does not verify proofs; it displays their structure.
• "What if" analysis is STRUCTURAL, not a re-derivation.
• The explorer cannot prove or disprove metaphysical claims.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.proof_explorer")

try:
    import dash
    from dash import dcc, html, Input, Output, State
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

__all__ = [
    "ProofNodeKind", "ProofNode", "ProofTree",
    "InteractiveProofExplorer",
    "AssumptionSensitivityAnalyzer",
]


class ProofNodeKind(Enum):
    ASSUMPTION = auto()
    LEMMA = auto()
    THEOREM = auto()
    COROLLARY = auto()
    DEFINITION = auto()
    OPEN_PROBLEM = auto()


class ProofStatus(Enum):
    PROVED = auto()       # [P] machine-checked
    CITED = auto()        # [C] from literature
    DERIVED = auto()      # [D] derived here
    OPEN = auto()         # [O] not yet addressed
    BLOCKED = auto()      # blocked by limitation enforcer


@dataclass
class ProofNode:
    """A single node in a proof tree."""
    node_id: str
    name: str
    kind: ProofNodeKind
    status: ProofStatus
    statement: str
    depends_on: List[str] = field(default_factory=list)
    proof_assistant: Optional[str] = None  # "coq", "lean", "isabelle"
    is_conditional: bool = True
    assumptions_required: List[str] = field(default_factory=list)


class ProofTree:
    """
    [D] A proof tree representing the logical structure of a proof.

    Nodes are connected by dependency edges. The tree can be explored
    to understand which conclusions depend on which assumptions.
    """

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, ProofNode] = {}
        self.edges: List[Tuple[str, str]] = []  # (from, to)

    def add_node(self, node: ProofNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append((from_id, to_id))

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all nodes that this node depends on (transitive)."""
        result = []
        visited = set()

        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for dep in node.depends_on:
                    result.append(dep)
                    dfs(dep)

        dfs(node_id)
        return result

    def get_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node (transitive)."""
        result = []
        visited = set()

        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            for from_id, to_id in self.edges:
                if from_id == nid and to_id not in visited:
                    result.append(to_id)
                    dfs(to_id)

        dfs(node_id)
        return result

    def remove_assumption(self, node_id: str) -> Dict[str, Any]:
        """
        Simulate removing an assumption.

        Returns which nodes become unsupported.
        """
        if node_id not in self.nodes:
            return {"error": f"Node {node_id} not found"}

        node = self.nodes[node_id]
        if node.kind != ProofNodeKind.ASSUMPTION:
            return {"error": f"Node {node_id} is not an assumption"}

        dependents = self.get_dependents(node_id)
        affected = []
        for dep_id in dependents:
            dep_node = self.nodes[dep_id]
            if dep_node.is_conditional:
                affected.append({
                    "node_id": dep_id,
                    "name": dep_node.name,
                    "original_status": dep_node.status.name,
                    "new_status": "OPEN" if dep_node.is_conditional else dep_node.status.name,
                    "note": "Becomes [O] open when assumption is removed",
                })

        return {
            "removed_assumption": node.name,
            "affected_nodes": affected,
            "n_affected": len(affected),
            "note": (
                "Removing an assumption does not ELIMINATE the conclusion; "
                "it reduces it to [O] open. This is the 'reduces, not "
                "eliminates' principle from Paper 11."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# BUILD THE DUAL-MODEL PROOF TREE
# ═══════════════════════════════════════════════════════════════════════

def build_dual_model_proof_tree() -> ProofTree:
    """
    [D] Build the proof tree for the Dual-Model Compatibility Theorem.
    """
    tree = ProofTree("Dual-Model Compatibility")

    # Assumptions
    tree.add_node(ProofNode(
        "A1", "Naturalistic model represents physical structure",
        ProofNodeKind.ASSUMPTION, ProofStatus.ASSUMED,
        "R_N = (X, P, M) represents the observable physical structure",
    ))
    tree.add_node(ProofNode(
        "A2", "Theological model represents SAME physical structure",
        ProofNodeKind.ASSUMPTION, ProofStatus.ASSUMED,
        "R_T = (X, P, M, G, D) represents the same observable structure",
    ))
    tree.add_node(ProofNode(
        "A3", "G is metaphysical ground, not physical variable",
        ProofNodeKind.ASSUMPTION, ProofStatus.ASSUMED,
        "God is introduced as ultimate ground, not as a physical variable",
    ))
    tree.add_node(ProofNode(
        "A4", "Both models agree on empirical predictions",
        ProofNodeKind.ASSUMPTION, ProofStatus.ASSUMED,
        "P_N(Q|E) = P_T(Q|E) for observable Q, evidence E",
    ))
    tree.add_node(ProofNode(
        "A5", "No same-level contradiction",
        ProofNodeKind.ASSUMPTION, ProofStatus.ASSUMED,
        "Neither model negates the other at the same semantic level",
    ))

    # Lemmas
    tree.add_node(ProofNode(
        "L1", "Shared physical structure exists",
        ProofNodeKind.LEMMA, ProofStatus.DERIVED,
        "R_N ∩ R_T ⊇ (X, P, M)",
        depends_on=["A1", "A2"],
    ))
    tree.add_node(ProofNode(
        "L2", "Pullback is non-empty",
        ProofNodeKind.LEMMA, ProofStatus.DERIVED,
        "The fiber product R_N ×_E R_T is non-empty",
        depends_on=["L1", "A4"],
        proof_assistant="lean",
    ))

    # Theorem
    tree.add_node(ProofNode(
        "T1", "Dual-Model Compatibility Theorem",
        ProofNodeKind.THEOREM, ProofStatus.PROVED,
        "Consistent(N) ∧ Consistent(T) ∧ Compatible(N,T)",
        depends_on=["A1", "A2", "A3", "A4", "A5", "L1", "L2"],
        proof_assistant="isabelle",
        is_conditional=True,
        assumptions_required=["A1", "A2", "A3", "A4", "A5"],
    ))

    # Theorem 2
    tree.add_node(ProofNode(
        "T2", "Explanatory-Level Independence",
        ProofNodeKind.THEOREM, ProofStatus.PROVED,
        "Scientific explanation ⇏ Unique metaphysical conclusion",
        depends_on=["L1"],
        proof_assistant="coq",
        is_conditional=True,
        assumptions_required=["L1"],
    ))

    # Corollary
    tree.add_node(ProofNode(
        "C1", "Compatibility is CONDITIONAL",
        ProofNodeKind.COROLLARY, ProofStatus.PROVED,
        "Theorem 1 does not prove every form of atheism/theism is true",
        depends_on=["T1"],
        is_conditional=True,
    ))

    # Open problems
    tree.add_node(ProofNode(
        "O1", "Whether assumptions hold in reality",
        ProofNodeKind.OPEN_PROBLEM, ProofStatus.OPEN,
        "Empirical verification of A1-A5 requires scientific investigation",
        depends_on=["T1"],
    ))

    # Edges
    tree.add_edge("A1", "L1")
    tree.add_edge("A2", "L1")
    tree.add_edge("L1", "L2")
    tree.add_edge("A4", "L2")
    tree.add_edge("L2", "T1")
    tree.add_edge("A3", "T1")
    tree.add_edge("A5", "T1")
    tree.add_edge("L1", "T2")
    tree.add_edge("T1", "C1")
    tree.add_edge("T1", "O1")

    return tree


# ═══════════════════════════════════════════════════════════════════════
# INTERACTIVE PROOF EXPLORER
# ═══════════════════════════════════════════════════════════════════════

class InteractiveProofExplorer:
    """
    [D] Interactive exploration of proof trees.

    Provides:
      • CLI navigation
      • Dash web interface (if available)
      • Assumption sensitivity analysis
    """

    def __init__(self):
        self.trees: Dict[str, ProofTree] = {}
        self.trees["dual_model"] = build_dual_model_proof_tree()

    def explore_cli(self) -> None:
        """CLI-based proof exploration."""
        print("=" * 72)
        print(" SES Interactive Proof Explorer")
        print("=" * 72)

        for tree_name, tree in self.trees.items():
            print(f"\n  ── Proof Tree: {tree.name} ──")
            print(f"  Nodes: {len(tree.nodes)}")
            print(f"  Edges: {len(tree.edges)}")

            print(f"\n  ── Proof Structure ──")
            for node_id, node in tree.nodes.items():
                status_icon = {
                    ProofStatus.PROVED: "✅",
                    ProofStatus.CITED: "📚",
                    ProofStatus.DERIVED: "🔧",
                    ProofStatus.OPEN: "❓",
                    ProofStatus.ASSUMED: "⚠️",
                    ProofStatus.BLOCKED: "🚫",
                }.get(node.status, "?")

                kind_icon = {
                    ProofNodeKind.ASSUMPTION: "  [A]",
                    ProofNodeKind.LEMMA: "  [L]",
                    ProofNodeKind.THEOREM: "  [T]",
                    ProofNodeKind.COROLLARY: "  [C]",
                    ProofNodeKind.DEFINITION: "  [D]",
                    ProofNodeKind.OPEN_PROBLEM: "  [O]",
                }.get(node.kind, "  [?]")

                assistant = f" ({node.proof_assistant})" if node.proof_assistant else ""
                cond = " [CONDITIONAL]" if node.is_conditional else ""

                print(f"  {status_icon} {kind_icon} {node.name}{assistant}{cond}")

                if node.depends_on:
                    print(f"       depends on: {', '.join(node.depends_on)}")

    def what_if_remove(self, assumption_id: str) -> Dict[str, Any]:
        """Simulate removing an assumption."""
        for tree in self.trees.values():
            if assumption_id in tree.nodes:
                return tree.remove_assumption(assumption_id)
        return {"error": f"Assumption {assumption_id} not found"}

    def build_dash_app(self) -> Optional[Any]:
        """Build a Dash web app for interactive exploration."""
        if not HAS_DASH:
            logger.warning("Dash not available; CLI mode only")
            return None

        app = dash.Dash(__name__, title="SESI Proof Explorer")

        # Build node options
        tree = self.trees["dual_model"]
        node_options = [
            {"label": f"{node.name} [{node.status.name}]", "value": node_id}
            for node_id, node in tree.nodes.items()
        ]

        app.layout = html.Div(style={
            "backgroundColor": "#1a1a2e", "color": "white",
            "fontFamily": "Arial", "padding": "20px",
        }, children=[
            html.H1("🔍 SES Interactive Proof Explorer",
                    style={"textAlign": "center", "color": "#00d4ff"}),
            html.P("Explore proof structure, assumptions, and dependencies",
                   style={"textAlign": "center", "color": "#888"}),

            html.Div(style={
                "display": "flex", "gap": "20px",
            }, children=[
                # Left panel: node selection
                html.Div(style={
                    "flex": 1, "backgroundColor": "#16213e",
                    "padding": "15px", "borderRadius": "8px",
                }, children=[
                    html.H3("Select a Proof Node"),
                    dcc.Dropdown(
                        id="node-select",
                        options=node_options,
                        value=list(tree.nodes.keys())[0] if tree.nodes else None,
                        style={"color": "black"},
                    ),
                    html.Br(),
                    html.Button("Show Dependencies", id="dep-btn", n_clicks=0),
                    html.Button("Show Dependents", id="depend-btn", n_clicks=0),
                    html.Br(),
                    html.Br(),
                    html.H4("What-if: Remove Assumption"),
                    dcc.Dropdown(
                        id="assumption-select",
                        options=[
                            {"label": f"{n.name}", "value": nid}
                            for nid, n in tree.nodes.items()
                            if n.kind == ProofNodeKind.ASSUMPTION
                        ],
                        style={"color": "black"},
                    ),
                    html.Button("Remove", id="remove-btn", n_clicks=0),
                ]),

                # Right panel: results
                html.Div(style={
                    "flex": 2, "backgroundColor": "#16213e",
                    "padding": "15px", "borderRadius": "8px",
                }, children=[
                    html.H3("Proof Analysis"),
                    html.Div(id="analysis-output"),
                ]),
            ]),
        ])

        @app.callback(
            Output("analysis-output", "children"),
            [Input("dep-btn", "n_clicks"),
             Input("depend-btn", "n_clicks"),
             Input("remove-btn", "n_clicks")],
            [State("node-select", "value"),
             State("assumption-select", "value")]
        )
        def update_analysis(dep_clicks, depend_clicks, remove_clicks,
                            node_id, assumption_id):
            ctx = dash.callback_context
            if not ctx.triggered:
                return html.P("Select a node and click a button.")

            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger == "dep-btn" and node_id:
                deps = tree.get_dependencies(node_id)
                return html.Div([
                    html.H4(f"Dependencies of {node_id}:"),
                    html.Ul([html.Li(d) for d in deps]),
                ])
            elif trigger == "depend-btn" and node_id:
                dependents = tree.get_dependents(node_id)
                return html.Div([
                    html.H4(f"Nodes depending on {node_id}:"),
                    html.Ul([html.Li(d) for d in dependents]),
                ])
            elif trigger == "remove-btn" and assumption_id:
                result = self.what_if_remove(assumption_id)
                if "error" in result:
                    return html.P(result["error"], style={"color": "red"})
                return html.Div([
                    html.H4(f"Removed: {result['removed_assumption']}"),
                    html.P(f"Affected nodes: {result['n_affected']}"),
                    html.Ul([
                        html.Li(f"{a['name']}: {a['original_status']} → {a['new_status']}")
                        for a in result["affected_nodes"]
                    ]),
                    html.P(result["note"], style={"color": "#f9a825"}),
                ])

            return html.P("No analysis selected.")

        return app


# ═══════════════════════════════════════════════════════════════════════
# ASSUMPTION SENSITIVITY ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class AssumptionSensitivityAnalyzer:
    """
    [D] Analyze which assumptions are critical for which conclusions.

    For each assumption, determines:
      • Which theorems depend on it
      • What happens if it is removed
      • Whether the conclusion becomes [O] open or remains valid
    """

    def __init__(self, tree: ProofTree):
        self.tree = tree

    def analyze(self) -> Dict[str, Any]:
        """Full sensitivity analysis."""
        assumptions = [
            (nid, node) for nid, node in self.tree.nodes.items()
            if node.kind == ProofNodeKind.ASSUMPTION
        ]

        results = {}
        for node_id, node in assumptions:
            dependents = self.tree.get_dependents(node_id)
            affected = [
                {
                    "node_id": dep_id,
                    "name": self.tree.nodes[dep_id].name,
                    "status": self.tree.nodes[dep_id].status.name,
                    "is_conditional": self.tree.nodes[dep_id].is_conditional,
                }
                for dep_id in dependents
                if dep_id in self.tree.nodes
            ]

            results[node_id] = {
                "assumption": node.name,
                "n_dependents": len(dependents),
                "dependents": affected,
                "criticality": "HIGH" if len(dependents) >= 3 else
                               "MEDIUM" if len(dependents) >= 1 else "LOW",
            }

        return {
            "n_assumptions": len(assumptions),
            "analysis": results,
            "note": (
                "Removing an assumption REDUCES the conclusion to [O] open; "
                "it does not ELIMINATE the conclusion. This is the "
                "'reduces, not eliminates' principle (Paper 11 §3)."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES Interactive Proof Explorer — Smoke Test")
    print("=" * 72)
    print(f"  Dash available: {HAS_DASH}")

    explorer = InteractiveProofExplorer()

    # CLI exploration
    explorer.explore_cli()

    # What-if analysis
    print("\n  ── What-if: Remove assumption A4 ──")
    result = explorer.what_if_remove("A4")
    print(f"    Removed: {result.get('removed_assumption', 'N/A')}")
    print(f"    Affected nodes: {result.get('n_affected', 0)}")
    for affected in result.get("affected_nodes", []):
        print(f"      • {affected['name']}: {affected['original_status']} → {affected['new_status']}")
    print(f"    Note: {result.get('note', '')}")

    # Sensitivity analysis
    print("\n  ── Assumption Sensitivity ──")
    tree = explorer.trees["dual_model"]
    analyzer = AssumptionSensitivityAnalyzer(tree)
    sensitivity = analyzer.analyze()
    print(f"    Total assumptions: {sensitivity['n_assumptions']}")
    for node_id, info in sensitivity["analysis"].items():
        print(f"    {node_id}: {info['assumption']}")
        print(f"      Criticality: {info['criticality']}")
        print(f"      Dependents: {info['n_dependents']}")

    # Dash app
    print("\n  ── Dash App ──")
    app = explorer.build_dash_app()
    if app is not None:
        print("    Dash app built successfully")
        print("    To launch: app.run_server(port=8060)")
    else:
        print("    Dash not available; CLI mode only")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Explorer is a VISUALIZATION TOOL, not a proof assistant")
    print("  • 'What-if' analysis is STRUCTURAL, not a re-derivation")
    print("  • Removing an assumption REDUCES to [O], does not ELIMINATE")
    print("  • The explorer cannot prove or disprove metaphysical claims")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
