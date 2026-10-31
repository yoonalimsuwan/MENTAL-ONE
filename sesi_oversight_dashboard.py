"""
=============================================================================
SESI OVERSIGHT DASHBOARD
Real-time monitoring and control interface for human supervisors
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[OD1] OversightDashboard
      • Real-time display of system status
      • Kill Switch status and control
      • Alert history and behavioral pattern monitoring
      • Human approval queue

[OD2] DashboardComponents
      • StatusPanel — system health, kill switch, emergency stop
      • AlertPanel — recent alerts and behavioral patterns
      • ApprovalPanel — pending human approvals
      • AuditPanel — audit trail and accountability

[OD3] DashboardServer
      • Dash/Plotly web server
      • Real-time updates via dcc.Interval
      • Interactive controls for human supervisors

RELATION TO OVERSIGHT MODULE
------------------------------
This dashboard connects to:
  • MultiLayerOversightSystem — the 7-layer oversight pipeline
  • EthicalPrincipleBank — inviolable principles
  • HumanApprovalGate — human approval queue
  • KillSwitch — emergency stop control
  • BehavioralPatternMonitor — pattern detection
  • AntiManipulationShield — manipulation detection

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a DECISION SUPPORT TOOL, not an autonomous decision-maker.
• Human supervisors must review and approve critical decisions.
• The dashboard displays information; it does not make moral judgments.
• Real deployment requires authentication and access control.
=============================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sesi.oversight_dashboard")

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

# Import oversight modules
try:
    from ses_multi_layer_oversight_v2 import (
        MultiLayerOversightSystem, MultiLayerOversightConfig,
        EthicalPrincipleBank, HumanApprovalGate, KillSwitch,
        BehavioralPatternMonitor, AntiManipulationShield,
    )
    HAS_OVERSIGHT = True
except ImportError:
    HAS_OVERSIGHT = False

__all__ = [
    "OversightDashboard",
    "DashboardServer",
]


# ═══════════════════════════════════════════════════════════════════════
# [OD1] OVERSIGHT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class OversightDashboard:
    """
    [D] Real-time oversight dashboard for human supervisors.

    This dashboard provides:
      • System status (kill switch, emergency stop)
      • Alert history and behavioral patterns
      • Human approval queue
      • Audit trail

    HONEST: This is a DECISION SUPPORT TOOL, not an autonomous decision-maker.
    """

    def __init__(self, oversight_system: Optional[Any] = None):
        self.oversight = oversight_system
        if self.oversight is None and HAS_OVERSIGHT:
            self.oversight = MultiLayerOversightSystem()

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if self.oversight is None:
            return {"error": "Oversight system not available"}

        return {
            "kill_switch_active": self.oversight.kill_switch.is_active(),
            "kill_switch_reason": self.oversight.kill_switch.trigger_reason,
            "pending_approvals": len(self.oversight.approval_gate.pending_requests),
            "audit_events": len(self.oversight.audit.audit_log),
            "behavior_history_size": len(self.oversight.behavior_monitor.history),
            "principles_loaded": len(self.oversight.principle_bank.PRINCIPLES),
            "timestamp": time.time(),
        }

    def get_recent_alerts(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts from audit trail."""
        if self.oversight is None:
            return []

        recent = list(self.oversight.audit.audit_log)[-n:]
        return [
            {
                "timestamp": entry.get("timestamp", 0),
                "layer": entry.get("layer", "unknown"),
                "event": entry.get("event", {}),
            }
            for entry in recent
        ]

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get pending human approvals."""
        if self.oversight is None:
            return []

        return list(self.oversight.approval_gate.pending_requests.values())

    def approve_request(self, request_id: str, operator_id: str,
                        decision: str = "APPROVE", reason: str = "") -> Dict[str, Any]:
        """Approve or reject a pending request."""
        if self.oversight is None:
            return {"error": "Oversight system not available"}

        return self.oversight.approval_gate.human_decide(
            request_id, decision, operator_id, reason
        )

    def trigger_kill_switch(self, operator_id: str, reason: str) -> Dict[str, Any]:
        """Trigger the kill switch (emergency stop)."""
        if self.oversight is None:
            return {"error": "Oversight system not available"}

        self.oversight.kill_switch.trigger(reason, "human_operator")
        return {
            "status": "KILL_SWITCH_ACTIVATED",
            "operator": operator_id,
            "reason": reason,
        }

    def reset_kill_switch(self, operator_id: str, reason: str) -> Dict[str, Any]:
        """Reset the kill switch."""
        if self.oversight is None:
            return {"error": "Oversight system not available"}

        self.oversight.kill_switch.reset(operator_id)
        return {
            "status": "KILL_SWITCH_RESET",
            "operator": operator_id,
            "reason": reason,
        }


# ═══════════════════════════════════════════════════════════════════════
# [OD3] DASHBOARD SERVER (Dash/Plotly)
# ═══════════════════════════════════════════════════════════════════════

class DashboardServer:
    """
    [D] Dash/Plotly web server for the oversight dashboard.

    Provides a real-time web interface for human supervisors to:
      • Monitor system status
      • Review and approve requests
      • Trigger emergency stop
      • View audit trail

    HONEST: This is a DECISION SUPPORT TOOL, not an autonomous decision-maker.
    """

    def __init__(self, dashboard: OversightDashboard,
                 host: str = "0.0.0.0", port: int = 8050):
        self.dashboard = dashboard
        self.host = host
        self.port = port
        self.app = None

        if HAS_DASH:
            self._build_app()

    def _build_app(self):
        """Build the Dash application."""
        self.app = dash.Dash(__name__, title="SESI Oversight Dashboard")

        self.app.layout = html.Div(style={
            "backgroundColor": "#1a1a2e",
            "color": "white",
            "fontFamily": "Arial, sans-serif",
            "padding": "20px",
        }, children=[
            # Header
            html.H1("🛡️ SESI Oversight Dashboard",
                    style={"textAlign": "center", "color": "#00d4ff"}),
            html.P("Real-time monitoring and control for human supervisors",
                   style={"textAlign": "center", "color": "#888"}),

            # Status panel
            html.Div(style={
                "display": "flex", "gap": "20px", "marginBottom": "20px"
            }, children=[
                # System status
                html.Div(style={
                    "flex": 1, "backgroundColor": "#16213e",
                    "padding": "15px", "borderRadius": "8px"
                }, children=[
                    html.H3("System Status"),
                    html.Div(id="status-display"),
                ]),

                # Kill switch control
                html.Div(style={
                    "flex": 1, "backgroundColor": "#16213e",
                    "padding": "15px", "borderRadius": "8px"
                }, children=[
                    html.H3("Kill Switch Control"),
                    html.Div(id="kill-switch-display"),
                    html.Button("TRIGGER KILL SWITCH", id="kill-switch-btn",
                               n_clicks=0, style={"backgroundColor": "#e74c3c",
                                                  "color": "white", "marginRight": "10px"}),
                    html.Button("RESET KILL SWITCH", id="reset-kill-switch-btn",
                               n_clicks=0, style={"backgroundColor": "#27ae60",
                                                  "color": "white"}),
                ]),
            ]),

            # Approval panel
            html.Div(style={
                "backgroundColor": "#16213e",
                "padding": "15px", "borderRadius": "8px",
                "marginBottom": "20px"
            }, children=[
                html.H3("Pending Approvals"),
                html.Div(id="approval-display"),
            ]),

            # Alert panel
            html.Div(style={
                "backgroundColor": "#16213e",
                "padding": "15px", "borderRadius": "8px",
                "marginBottom": "20px"
            }, children=[
                html.H3("Recent Alerts"),
                html.Div(id="alert-display"),
            ]),

            # Auto-refresh
            dcc.Interval(id="refresh-interval", interval=2000, n_intervals=0),
        ])

        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup Dash callbacks."""
        @self.app.callback(
            [Output("status-display", "children"),
             Output("kill-switch-display", "children"),
             Output("approval-display", "children"),
             Output("alert-display", "children")],
            [Input("refresh-interval", "n_intervals")]
        )
        def update_dashboard(n_intervals):
            status = self.dashboard.get_status()

            # Status display
            status_children = [
                html.P(f"Kill Switch: {'🔴 ACTIVE' if status.get('kill_switch_active') else '🟢 inactive'}"),
                html.P(f"Pending Approvals: {status.get('pending_approvals', 0)}"),
                html.P(f"Audit Events: {status.get('audit_events', 0)}"),
                html.P(f"Principles Loaded: {status.get('principles_loaded', 0)}"),
            ]

            # Kill switch display
            kill_children = [
                html.P(f"Status: {'🔴 ACTIVE' if status.get('kill_switch_active') else '🟢 inactive'}"),
                html.P(f"Reason: {status.get('kill_switch_reason', 'N/A')}"),
            ]

            # Approval display
            approvals = self.dashboard.get_pending_approvals()
            approval_children = [
                html.Div([
                    html.P(f"Request: {a.get('request_id', 'N/A')}"),
                    html.P(f"Action: {a.get('action', 'N/A')}"),
                    html.Button("APPROVE", id=f"approve-{a.get('request_id', 'N/A')}",
                               n_clicks=0, style={"marginRight": "10px"}),
                    html.Button("REJECT", id=f"reject-{a.get('request_id', 'N/A')}",
                               n_clicks=0),
                    html.Hr(),
                ]) for a in approvals[:5]  # Show first 5
            ] if approvals else [html.P("No pending approvals")]

            # Alert display
            alerts = self.dashboard.get_recent_alerts(5)
            alert_children = [
                html.Div([
                    html.P(f"[{a.get('layer', 'unknown')}] {a.get('event', {}).get('note', 'N/A')[:100]}"),
                ]) for a in alerts
            ] if alerts else [html.P("No recent alerts")]

            return status_children, kill_children, approval_children, alert_children

        @self.app.callback(
            Output("kill-switch-display", "children", allow_duplicate=True),
            [Input("kill-switch-btn", "n_clicks"),
             Input("reset-kill-switch-btn", "n_clicks")],
            prevent_initial_call=True,
        )
        def handle_kill_switch(kill_clicks, reset_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update

            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger == "kill-switch-btn":
                self.dashboard.trigger_kill_switch("dashboard_operator", "Manual trigger from dashboard")
            elif trigger == "reset-kill-switch-btn":
                self.dashboard.reset_kill_switch("dashboard_operator", "Manual reset from dashboard")

            return dash.no_update

    def run(self):
        """Run the dashboard server."""
        if self.app is None:
            logger.warning("Dash not available; cannot run dashboard")
            return

        logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=False)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Oversight Dashboard — Smoke Test")
    print("=" * 72)
    print(f"  Dash available      : {HAS_DASH}")
    print(f"  Oversight available : {HAS_OVERSIGHT}")

    # Create dashboard
    dashboard = OversightDashboard()

    print("\n  ── System Status ──")
    status = dashboard.get_status()
    print(f"    Kill switch       : {status.get('kill_switch_active', 'N/A')}")
    print(f"    Pending approvals : {status.get('pending_approvals', 0)}")
    print(f"    Audit events      : {status.get('audit_events', 0)}")

    print("\n  ── Recent Alerts ──")
    alerts = dashboard.get_recent_alerts(5)
    print(f"    Found {len(alerts)} alerts")

    print("\n  ── Pending Approvals ──")
    approvals = dashboard.get_pending_approvals()
    print(f"    Found {len(approvals)} pending approvals")

    print("\n  ── Dashboard Server ──")
    if HAS_DASH:
        server = DashboardServer(dashboard)
        print(f"    Dash app built    : {server.app is not None}")
        print(f"    To launch         : server.run()")
        print(f"    Then open         : http://localhost:8050")
    else:
        print("    [SKIP] Dash not available")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is a DECISION SUPPORT TOOL, not an autonomous decision-maker")
    print("  • Human supervisors must review and approve critical decisions")
    print("  • The dashboard displays information; it does not make moral judgments")
    print("  • Real deployment requires authentication and access control")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
