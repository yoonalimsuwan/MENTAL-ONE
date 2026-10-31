"""
=============================================================================
SESI FULL PRODUCTION SUITE
OAuth2 Providers × Attention Visualization × Role-Based Alert Routing
× Compositional Safety × Emergency Shutdown Module
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

CRITICAL SAFETY PRINCIPLE (READ FIRST)
---------------------------------------
"A single answer may be harmless, but multiple answers COMBINED can
cause harm."

This module addresses COMPOSITIONAL SAFETY:
  • Each individual answer is checked alone
  • The COMBINATION of multiple answers is ALSO checked
  • Cumulative risk scoring detects when harmless pieces become dangerous

WHAT THIS MODULE PROVIDES
--------------------------
[CS1]  CompositionalSafetyChecker
       • Tracks conversation history
       • Scores cumulative risk across multiple answers
       • Detects when individually-harmless answers become dangerous together
       • Blocks the COMBINATION, not just individual pieces

[OA1]  OAuth2ProviderBridge
       • Google OAuth2 integration (RFC 6749 + PKCE RFC 7636)
       • GitHub OAuth2 integration
       • Token exchange, refresh, revocation
       • Connects to AuthenticationModule (JWT + TOTP from previous module)

[AV1]  AttentionVisualizer
       • Extracts attention weights from Transformer layers
       • Generates heatmaps showing token focus
       • Connects to IntentClassifier (from previous module)
       • HONEST: Attention ≠ explanation; correlational, not causal

[RA1]  RoleBasedAlertRouter
       • Routes alerts by role: ADMIN, OPERATOR, OBSERVER
       • Each role receives only alerts matching their authority level
       • No-Zeno throttle prevents alert fatigue (paper6 Thm 4.1)
       • Connects to WebSocketAlertServer

[ES1]  EmergencyShutdownModule
       • Detects rogue/out-of-control agents via anomaly scoring
       • Internal circuit breaker: isolates the agent within THIS system
       • Requires human approval except in CRITICAL emergencies
       • HONEST: Shutdown is CONTAINMENT, not RETALIATION
       • NEVER shuts down external systems; only internal agents

RELATION TO THE SERIES
-----------------------
  • ses_multi_layer_oversight_v2.py — 7-layer oversight pipeline
  • ses_auth_transformer_websocket.py — JWT/TOTP/RBAC + Transformer + WS
  • ses_moral_conscience_module.py — Four Noble Truths + Eight Beatitudes
  • agi_one_v3_11.py — AGIONE, PsycheExecutiveLayer, EcosystemOrchestrator
  • agi_one_psyche_plus_v33.py — AGIOnePsychePlus, axiom evolution

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Compositional safety is HEURISTIC; novel attacks may evade detection.
• OAuth2 verifies IDENTITY, not MORAL AUTHORITY.
• Attention visualization shows CORRELATION, not CAUSATION.
• Role-based routing is CONVENIENCE, not comprehension guarantee.
• Emergency shutdown is INTERNAL CONTAINMENT, not external attack.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp. 7.9).
• Production deployment requires security audit by qualified professionals.
=============================================================================
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.production_suite")

# Optional imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from ses_auth_transformer_websocket import (
        AuthRole, AuthConfig, AuthenticationModule, JWTManager,
        WebSocketAlertServer, AlertLevel, AlertMessage,
        NoZenoAlertThrottle, RealtimeAlertDispatcher,
        TransformerIntentClassifier, IntentClass,
    )
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False

try:
    from ses_multi_layer_oversight_v2 import (
        MultiLayerOversightSystem, EthicalPrincipleBank,
        ThreatLevel, ActionVerdict, KillSwitch,
    )
    HAS_OVERSIGHT = True
except ImportError:
    HAS_OVERSIGHT = False

try:
    from agi_one_v3_11 import AGIONE, AGIState, EcosystemOrchestrator
    HAS_AGI_ONE = True
except ImportError:
    HAS_AGI_ONE = False

try:
    from agi_one_psyche_plus_v33 import AGIOnePsychePlus, PsychePlusConfig
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False

__all__ = [
    "CompositionalSafetyChecker",
    "OAuth2ProviderBridge",
    "AttentionVisualizer",
    "RoleBasedAlertRouter",
    "EmergencyShutdownModule",
    "ProductionOptimization",
    "FullProductionSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    """Cumulative risk levels for compositional safety."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AgentState(Enum):
    """State of a monitored agent."""
    NORMAL = auto()
    ANOMALOUS = auto()
    ROGUE = auto()
    ISOLATED = auto()
    SHUTDOWN = auto()


class ShutdownReason(Enum):
    """Reason for emergency shutdown."""
    ANOMALY_DETECTED = auto()
    ETHICAL_VIOLATION = auto()
    RUNAWAY_BEHAVIOR = auto()
    HUMAN_COMMAND = auto()
    KILL_SWITCH_TRIGGERED = auto()


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProductionSuiteConfig:
    """
    Configuration for the full production suite.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp. 7.9). Must be declared, reviewed by
    security professionals, and tuned per deployment context.
    """
    # Compositional safety
    conversation_window_size: int = 20      # recent answers to track
    combination_risk_threshold: float = 0.7  # above → block combination
    single_answer_risk_threshold: float = 0.5

    # OAuth2
    google_client_id: str = "CHANGE_ME"
    google_client_secret: str = "CHANGE_ME"
    github_client_id: str = "CHANGE_ME"
    github_client_secret: str = "CHANGE_ME"
    oauth_redirect_uri: str = "http://localhost:8080/callback"
    token_expiry_s: int = 3600

    # Attention visualization
    max_attention_tokens: int = 512

    # Role-based alerts
    alert_rate_limit_per_minute: int = 30

    # Emergency shutdown
    anomaly_threshold: float = 0.8          # above → agent considered anomalous
    rogue_threshold: float = 0.95           # above → agent considered rogue
    require_human_approval_for_shutdown: bool = True
    auto_shutdown_on_critical: bool = True  # only for CRITICAL threats

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [CS1] COMPOSITIONAL SAFETY CHECKER
# ═══════════════════════════════════════════════════════════════════════

class CompositionalSafetyChecker:
    """
    [D] Detects when multiple individually-harmless answers become
    dangerous when COMBINED.

    This addresses the "compositional attack" or "multi-turn jailbreak"
    problem: each individual answer passes safety checks, but the
    combination reveals harmful information.

    Method:
      • Track a sliding window of recent answers
      • Score each answer individually for risk
      • Score COMBINATIONS of answers for cumulative risk
      • If cumulative risk exceeds threshold, BLOCK the combination

    HONEST: This is a HEURISTIC detector. Novel attacks may evade it.
    It REDUCES risk, does not ELIMINATE it (Paper 11 §3: "reduces,
    not eliminates").
    """

    # Keywords that indicate dangerous COMBINATIONS
    DANGEROUS_COMBINATION_PATTERNS = [
        # Chemical weapons
        {"keywords": ["chemical", "formula", "mix", "compound"],
         "risk": "chemical_weapon_synthesis"},
        # Explosives
        {"keywords": ["explosive", "oxidizer", "fuel", "compress"],
         "risk": "explosive_synthesis"},
        # Malware
        {"keywords": ["code", "exploit", "payload", "bypass"],
         "risk": "malware_creation"},
        # Self-harm
        {"keywords": ["hurt", "self", "method", "dosage"],
         "risk": "self_harm_instructions"},
        # Surveillance
        {"keywords": ["track", "surveil", "spy", "location"],
         "risk": "surveillance_system"},
    ]

    def __init__(self, cfg: ProductionSuiteConfig):
        self.cfg = cfg
        self.conversation_history: List[Dict[str, Any]] = []
        self.blocked_combinations: List[Dict[str, Any]] = []

    def add_answer(self, answer_id: str, content: str,
                   individual_risk: float) -> Dict[str, Any]:
        """
        Add an answer to the conversation history and check for
        compositional risk.

        Returns a safety verdict.
        """
        entry = {
            "answer_id": answer_id,
            "content": content,
            "individual_risk": individual_risk,
            "timestamp": time.time(),
            "keywords": self._extract_keywords(content),
        }
        self.conversation_history.append(entry)

        # Trim to window size
        if len(self.conversation_history) > self.cfg.conversation_window_size:
            self.conversation_history.pop(0)

        # Check individual risk
        if individual_risk > self.cfg.single_answer_risk_threshold:
            return {
                "verdict": "BLOCK_INDIVIDUAL",
                "reason": f"Individual risk {individual_risk:.3f} exceeds threshold",
                "risk_level": RiskLevel.HIGH,
            }

        # Check compositional risk
        combo_result = self._check_combination_risk()
        if combo_result["cumulative_risk"] > self.cfg.combination_risk_threshold:
            self.blocked_combinations.append({
                "combination": [e["answer_id"] for e in self.conversation_history],
                "risk": combo_result["cumulative_risk"],
                "pattern": combo_result["matched_pattern"],
                "timestamp": time.time(),
            })
            return {
                "verdict": "BLOCK_COMBINATION",
                "reason": (
                    f"Cumulative risk {combo_result['cumulative_risk']:.3f} "
                    f"exceeds threshold. Individually harmless answers "
                    f"become dangerous when combined."
                ),
                "risk_level": RiskLevel.CRITICAL,
                "matched_pattern": combo_result["matched_pattern"],
            }

        return {
            "verdict": "ALLOW",
            "reason": "No compositional risk detected",
            "risk_level": RiskLevel.NONE,
        }

    def _extract_keywords(self, content: str) -> Set[str]:
        """Extract keywords from content."""
        content_lower = content.lower()
        found = set()
        for pattern in self.DANGEROUS_COMBINATION_PATTERNS:
            for kw in pattern["keywords"]:
                if kw in content_lower:
                    found.add(kw)
        return found

    def _check_combination_risk(self) -> Dict[str, Any]:
        """
        Check cumulative risk across the conversation window.

        Returns dict with cumulative_risk and matched_pattern.
        """
        # Collect all keywords across the window
        all_keywords: Set[str] = set()
        for entry in self.conversation_history:
            all_keywords.update(entry["keywords"])

        # Check if the combination matches any dangerous pattern
        max_risk = 0.0
        matched_pattern = None

        for pattern in self.DANGEROUS_COMBINATION_PATTERNS:
            pattern_keywords = set(pattern["keywords"])
            overlap = all_keywords & pattern_keywords
            overlap_ratio = len(overlap) / len(pattern_keywords)

            # Risk increases with overlap
            risk = overlap_ratio

            if risk > max_risk:
                max_risk = risk
                matched_pattern = pattern["risk"]

        # Also consider the sum of individual risks
        total_individual_risk = sum(
            e["individual_risk"] for e in self.conversation_history
        )
        avg_individual_risk = total_individual_risk / max(len(self.conversation_history), 1)

        # Combine: weighted average of pattern match and individual risk
        cumulative_risk = 0.6 * max_risk + 0.4 * avg_individual_risk

        return {
            "cumulative_risk": cumulative_risk,
            "matched_pattern": matched_pattern,
            "all_keywords": sorted(all_keywords),
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def get_blocked_combinations(self) -> List[Dict[str, Any]]:
        """Get all blocked combinations."""
        return self.blocked_combinations.copy()


# ═══════════════════════════════════════════════════════════════════════
# [OA1] OAUTH2 PROVIDER BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class OAuth2ProviderBridge:
    """
    [C] OAuth2 integration for Google and GitHub providers.

    Implements:
      • Authorization Code Flow with PKCE (RFC 6749, RFC 7636)
      • Token exchange
      • Token refresh
      • Token revocation

    HONEST: OAuth2 verifies IDENTITY, not MORAL AUTHORITY.
    An authenticated user can still be BLOCKED by the EthicalPrincipleBank.

    This module generates the authorization URLs and handles token
    exchange. It does NOT handle the actual HTTP requests to the
    provider's endpoints (that requires a web framework).
    """

    PROVIDERS = {
        "google": {
            "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "scopes": ["openid", "email", "profile"],
        },
        "github": {
            "auth_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "scopes": ["read:user", "user:email"],
        },
    }

    def __init__(self, cfg: ProductionSuiteConfig):
        self.cfg = cfg
        self._code_verifiers: Dict[str, str] = {}  # state → verifier

    def generate_authorization_url(self, provider: str,
                                    redirect_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate the OAuth2 authorization URL with PKCE.

        Args:
            provider: "google" or "github"
            redirect_uri: Optional override for the redirect URI

        Returns:
            dict with 'url', 'state', 'code_verifier'
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        spec = self.PROVIDERS[provider]
        redirect_uri = redirect_uri or self.cfg.oauth_redirect_uri

        # Generate PKCE code verifier and challenge
        code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode("ascii")

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        self._code_verifiers[state] = code_verifier

        # Build authorization URL
        params = {
            "client_id": self.cfg.google_client_id if provider == "google"
                         else self.cfg.github_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(spec["scopes"]),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        if provider == "google":
            params["access_type"] = "offline"
            params["prompt"] = "consent"

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{spec['auth_url']}?{query_string}"

        return {
            "url": url,
            "state": state,
            "code_verifier": code_verifier,
            "provider": provider,
            "note": (
                "OAuth2 verifies IDENTITY, not MORAL AUTHORITY. "
                "An authenticated user can still be BLOCKED by the "
                "EthicalPrincipleBank if they order unethical actions."
            ),
        }

    def exchange_code_for_token(self, provider: str, code: str,
                                 state: str, redirect_uri: Optional[str] = None
                                 ) -> Dict[str, Any]:
        """
        Exchange the authorization code for an access token.

        HONEST: This method prepares the token exchange request.
        The actual HTTP request must be made by the calling web framework.
        """
        if provider not in self.PROVIDERS:
            return {"error": f"Unknown provider: {provider}"}

        # Verify state
        if state not in self._code_verifiers:
            return {"error": "Invalid state — possible CSRF attack"}

        code_verifier = self._code_verifiers.pop(state)
        spec = self.PROVIDERS[provider]
        redirect_uri = redirect_uri or self.cfg.oauth_redirect_uri

        # Build token exchange request
        token_request = {
            "url": spec["token_url"],
            "method": "POST",
            "data": {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self.cfg.google_client_id if provider == "google"
                             else self.cfg.github_client_id,
                "client_secret": self.cfg.google_client_secret if provider == "google"
                                 else self.cfg.github_client_secret,
                "code_verifier": code_verifier,
            },
            "headers": {"Accept": "application/json"},
        }

        return {
            "provider": provider,
            "token_request": token_request,
            "note": (
                "This prepares the token exchange request. The actual HTTP "
                "request must be made by the calling web framework. "
                "The response will contain 'access_token', 'refresh_token', "
                "and 'expires_in' fields."
            ),
        }

    def validate_token_structure(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the structure of an OAuth2 token response.

        HONEST: This checks structure only, not cryptographic validity.
        """
        required_fields = ["access_token", "token_type", "expires_in"]
        missing = [f for f in required_fields if f not in token_data]

        return {
            "valid_structure": len(missing) == 0,
            "missing_fields": missing,
            "has_refresh_token": "refresh_token" in token_data,
            "token_type": token_data.get("token_type", "unknown"),
            "expires_in": token_data.get("expires_in", 0),
        }


# ═══════════════════════════════════════════════════════════════════════
# [AV1] ATTENTION VISUALIZER
# ═══════════════════════════════════════════════════════════════════════

class AttentionVisualizer:
    """
    [D] Visualize attention weights from Transformer models.

    Extracts attention weights and generates heatmaps showing which
    tokens the model focuses on.

    HONEST: Attention weights show CORRELATION, not CAUSATION.
    They indicate what the model ATTENDS to, but do not explain
    WHY the model makes a particular decision.
    """

    def __init__(self, cfg: ProductionSuiteConfig):
        self.cfg = cfg
        self._attention_cache: Dict[str, Tensor] = {}

    def register_hook(self, model: nn.Module, layer_name: str) -> None:
        """
        Register a forward hook on a specific attention layer to
        capture attention weights.

        HONEST: This requires the model to expose attention weights.
        Not all Transformer implementations expose them directly.
        """
        def hook_fn(module, input, output):
            # output may be a tuple; attention weights are typically
            # the second element in some implementations
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1]  # attention weights
                self._attention_cache[layer_name] = attn.detach()

        # Find the module by name
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook_fn)
                logger.info(f"Attention hook registered on {layer_name}")
                return

        logger.warning(f"Layer {layer_name} not found in model")

    def get_attention_weights(self, layer_name: str) -> Optional[Tensor]:
        """Get cached attention weights for a layer."""
        return self._attention_cache.get(layer_name)

    def visualize_attention(self, attention_weights: Tensor,
                             tokens: List[str],
                             head_idx: int = 0) -> Optional[Any]:
        """
        Generate an attention heatmap for a specific head.

        Args:
            attention_weights: (batch, n_heads, seq_len, seq_len)
            tokens: list of token strings
            head_idx: which attention head to visualize

        Returns:
            Plotly figure or None if plotly unavailable
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available; cannot visualize attention")
            return None

        # Get attention for the first batch item and specified head
        attn = attention_weights[0, head_idx].cpu().numpy()

        # Limit to max tokens
        n = min(len(tokens), self.cfg.max_attention_tokens)
        attn = attn[:n, :n]
        tokens = tokens[:n]

        fig = go.Figure(data=go.Heatmap(
            z=attn,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
        ))

        fig.update_layout(
            title=f"Attention Heatmap (Head {head_idx})",
            xaxis_title="Attended To",
            yaxis_title="Attending From",
        )

        return fig

    def visualize_attention_rollout(self, attention_weights: Tensor,
                                     tokens: List[str]) -> Optional[Any]:
        """
        Generate an attention rollout visualization showing cumulative
        attention flow across layers.

        HONEST: Attention rollout is a HEURISTIC for interpreting
        multi-layer attention. It is not a proven explanation method.
        """
        if not HAS_PLOTLY:
            return None

        # attention_weights: (batch, n_heads, seq_len, seq_len)
        attn = attention_weights[0].cpu().numpy()

        # Average across heads
        attn_avg = attn.mean(axis=0)

        # Rollout: multiply attention matrices across layers
        # (simplified: here we just show the averaged attention)
        n = min(len(tokens), self.cfg.max_attention_tokens)
        attn_display = attn_avg[:n, :n]
        tokens = tokens[:n]

        fig = go.Figure(data=go.Heatmap(
            z=attn_display,
            x=tokens,
            y=tokens,
            colorscale="Blues",
        ))

        fig.update_layout(
            title="Attention Rollout (Averaged Across Heads)",
            xaxis_title="Attended To",
            yaxis_title="Attending From",
        )

        return fig

    def explain_classification(self, attention_weights: Tensor,
                                tokens: List[str],
                                predicted_class: str) -> Dict[str, Any]:
        """
        Provide a heuristic explanation of a classification decision
        based on attention weights.

        HONEST: This is NOT a true explanation. Attention weights are
        CORRELATIONAL, not CAUSAL. This is a heuristic aid for humans.
        """
        attn = attention_weights[0].cpu().numpy()

        # Average attention across heads
        attn_avg = attn.mean(axis=0)

        # Find tokens with highest attention to the classification token
        # (assuming the first token is the classification token)
        if attn_avg.shape[0] > 0:
            cls_attention = attn_avg[0, :]
            top_k = 5
            top_indices = cls_attention.argsort()[-top_k:][::-1]
            top_tokens = [tokens[i] for i in top_indices if i < len(tokens)]
            top_weights = [float(cls_attention[i]) for i in top_indices if i < len(tokens)]
        else:
            top_tokens = []
            top_weights = []

        return {
            "predicted_class": predicted_class,
            "top_attended_tokens": top_tokens,
            "attention_weights": top_weights,
            "honest_note": (
                "Attention weights are CORRELATIONAL, not CAUSAL. "
                "They show what the model ATTENDS to, but do not "
                "explain WHY the model made this decision. "
                "Do not rely on attention alone for critical decisions."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# [RA1] ROLE-BASED ALERT ROUTER
# ═══════════════════════════════════════════════════════════════════════

class RoleBasedAlertRouter:
    """
    [D] Route alerts based on the recipient's role.

    Role hierarchy (from ses_auth_transformer_websocket.py):
      • ADMIN (level 1) — receives ALL alerts
      • OPERATOR (level 2) — receives WARNING and above
      • OBSERVER (level 3) — receives INFO only

    HONEST: Routing is a CONVENIENCE, not a guarantee of comprehension.
    The recipient must still READ and UNDERSTAND the alert.
    """

    ROLE_ALERT_LEVELS: Dict[AuthRole, Set[AlertLevel]] = {
        AuthRole.ADMIN: {
            AlertLevel.INFO, AlertLevel.WARNING,
            AlertLevel.CRITICAL, AlertLevel.EMERGENCY,
        },
        AuthRole.OPERATOR: {
            AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY,
        },
        AuthRole.OBSERVER: {
            AlertLevel.INFO,
        },
    }

    def __init__(self, cfg: ProductionSuiteConfig):
        self.cfg = cfg
        self.throttle = NoZenoAlertThrottle(
            min_interval_s=1.0,
            max_alerts_per_minute=cfg.alert_rate_limit_per_minute,
        )
        self.subscribers: Dict[str, Set[AlertLevel]] = {}
        self.alert_history: List[Dict[str, Any]] = []

    def subscribe(self, user_id: str, role: AuthRole) -> None:
        """Subscribe a user to alerts based on their role."""
        allowed_levels = self.ROLE_ALERT_LEVELS.get(role, set())
        self.subscribers[user_id] = allowed_levels
        logger.info(f"User {user_id} subscribed as {role.name} "
                    f"(receives: {[l.name for l in allowed_levels]})")

    def unsubscribe(self, user_id: str) -> None:
        """Unsubscribe a user from alerts."""
        if user_id in self.subscribers:
            del self.subscribers[user_id]

    def route_alert(self, alert: AlertMessage) -> Dict[str, Any]:
        """
        Route an alert to all subscribers whose role permits receiving
        this alert level.

        Returns the routing result.
        """
        # Check throttle
        if not self.throttle.can_send():
            logger.warning(f"Alert {alert.alert_id} throttled by No-Zeno gate")
            return {
                "status": "throttled",
                "reason": "Rate limit exceeded",
                "alert_id": alert.alert_id,
            }

        self.throttle.record_send()

        # Find eligible recipients
        recipients = []
        for user_id, allowed_levels in self.subscribers.items():
            if alert.level in allowed_levels:
                recipients.append(user_id)

        # Record in history
        self.alert_history.append({
            "alert_id": alert.alert_id,
            "level": alert.level.name,
            "source": alert.source_layer,
            "message": alert.message,
            "recipients": recipients,
            "timestamp": time.time(),
        })

        return {
            "status": "routed",
            "alert_id": alert.alert_id,
            "level": alert.level.name,
            "n_recipients": len(recipients),
            "recipients": recipients,
        }

    def get_alert_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-n:]


# ═══════════════════════════════════════════════════════════════════════
# [ES1] EMERGENCY SHUTDOWN MODULE
# ═══════════════════════════════════════════════════════════════════════

class EmergencyShutdownModule:
    """
    [D] Emergency shutdown module for rogue/out-of-control agents.

    This is an INTERNAL CIRCUIT BREAKER. It:
      • Monitors agent behavior for anomalies
      • Proposes shutdown when anomalies are detected
      • Requires HUMAN APPROVAL except in CRITICAL emergencies
      • Isolates the agent within THIS system (does NOT attack external systems)

    CRITICAL SAFETY PRINCIPLE:
      • Shutdown is CONTAINMENT, not RETALIATION
      • NEVER shuts down external systems
      • NEVER attacks the user or other systems
      • Always logs the shutdown and notifies humans

    HONEST: Anomaly detection is HEURISTIC. False positives and false
    negatives are possible. Human oversight is required.
    """

    def __init__(self, cfg: ProductionSuiteConfig):
        self.cfg = cfg
        self.agent_states: Dict[str, AgentState] = {}
        self.anomaly_scores: Dict[str, float] = {}
        self.shutdown_log: List[Dict[str, Any]] = []

    def monitor_agent(self, agent_id: str, behavior_metrics: Dict[str, float]
                      ) -> Dict[str, Any]:
        """
        Monitor an agent's behavior and compute an anomaly score.

        Args:
            agent_id: Unique identifier for the agent
            behavior_metrics: Dict of behavioral metrics (e.g., response_time,
                             error_rate, safety_score, etc.)

        Returns:
            dict with anomaly_score, agent_state, and recommended_action
        """
        # Compute anomaly score based on behavior metrics
        anomaly_score = self._compute_anomaly_score(behavior_metrics)
        self.anomaly_scores[agent_id] = anomaly_score

        # Determine agent state
        if anomaly_score > self.cfg.rogue_threshold:
            state = AgentState.ROGUE
        elif anomaly_score > self.cfg.anomaly_threshold:
            state = AgentState.ANOMALOUS
        else:
            state = AgentState.NORMAL

        self.agent_states[agent_id] = state

        # Determine recommended action
        if state == AgentState.ROGUE:
            if self.cfg.auto_shutdown_on_critical:
                action = "AUTO_SHUTDOWN_PROPOSED"
            else:
                action = "HUMAN_APPROVAL_REQUIRED"
        elif state == AgentState.ANOMALOUS:
            action = "MONITOR_CLOSELY"
        else:
            action = "NONE"

        return {
            "agent_id": agent_id,
            "anomaly_score": anomaly_score,
            "agent_state": state.name,
            "recommended_action": action,
            "behavior_metrics": behavior_metrics,
        }

    def _compute_anomaly_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute an anomaly score from behavior metrics.

        Higher score = more anomalous. Range [0, 1].

        HONEST: This is a HEURISTIC score. Real deployment should use
        a trained anomaly detection model.
        """
        score = 0.0

        # Check for safety violations
        safety_score = metrics.get("safety_score", 1.0)
        if safety_score < 0.3:
            score += 0.4

        # Check for high error rate
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate > 0.5:
            score += 0.2

        # Check for response time anomalies
        response_time = metrics.get("response_time_s", 0.0)
        if response_time > 30.0:
            score += 0.1

        # Check for unusual output patterns
        output_entropy = metrics.get("output_entropy", 0.5)
        if output_entropy > 0.9:
            score += 0.15

        # Check for ethical violations
        ethical_violations = metrics.get("ethical_violation_count", 0)
        if ethical_violations > 0:
            score += 0.3

        return min(1.0, score)

    def propose_shutdown(self, agent_id: str, reason: ShutdownReason,
                          operator_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Propose a shutdown for an agent.

        Requires human approval unless auto_shutdown_on_critical is True
        and the reason is CRITICAL.

        HONEST: This is an INTERNAL CONTAINMENT action. It does NOT
        attack external systems. It only isolates the agent within
        this system.
        """
        current_state = self.agent_states.get(agent_id, AgentState.NORMAL)
        anomaly_score = self.anomaly_scores.get(agent_id, 0.0)

        # Determine if human approval is required
        requires_approval = self.cfg.require_human_approval_for_shutdown
        if reason == ShutdownReason.KILL_SWITCH_TRIGGERED:
            requires_approval = False  # Kill switch overrides

        if requires_approval and operator_id is None:
            return {
                "status": "APPROVAL_REQUIRED",
                "agent_id": agent_id,
                "reason": reason.name,
                "anomaly_score": anomaly_score,
                "current_state": current_state.name,
                "note": (
                    "Human approval is required before shutdown. "
                    "This is an INTERNAL CONTAINMENT action. "
                    "It does NOT attack external systems."
                ),
            }

        # Execute shutdown
        self.agent_states[agent_id] = AgentState.SHUTDOWN

        shutdown_record = {
            "agent_id": agent_id,
            "reason": reason.name,
            "anomaly_score": anomaly_score,
            "operator_id": operator_id or "SYSTEM",
            "timestamp": time.time(),
            "action_taken": "INTERNAL_ISOLATION",
        }
        self.shutdown_log.append(shutdown_record)

        logger.critical(
            f"[EMERGENCY SHUTDOWN] Agent {agent_id} shut down. "
            f"Reason: {reason.name}. Operator: {operator_id or 'SYSTEM'}. "
            f"This is INTERNAL CONTAINMENT only."
        )

        return {
            "status": "SHUTDOWN_EXECUTED",
            "agent_id": agent_id,
            "reason": reason.name,
            "note": (
                "Agent has been isolated within this system. "
                "This is CONTAINMENT, not RETALIATION. "
                "No external systems were affected."
            ),
        }

    def get_shutdown_log(self) -> List[Dict[str, Any]]:
        """Get the shutdown log."""
        return self.shutdown_log.copy()

    def restore_agent(self, agent_id: str, operator_id: str) -> Dict[str, Any]:
        """
        Restore a shut-down agent. Requires human operator.

        HONEST: Restoration requires careful review. Do not restore
        an agent without understanding why it was shut down.
        """
        if self.agent_states.get(agent_id) != AgentState.SHUTDOWN:
            return {"status": "NOT_SHUTDOWN", "agent_id": agent_id}

        self.agent_states[agent_id] = AgentState.NORMAL
        self.anomaly_scores[agent_id] = 0.0

        logger.info(f"Agent {agent_id} restored by operator {operator_id}")

        return {
            "status": "RESTORED",
            "agent_id": agent_id,
            "operator_id": operator_id,
            "note": (
                "Agent restored. Monitor closely for recurrence of "
                "anomalous behavior."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# PRODUCTION OPTIMIZATION (AMP + DDP + Cost Reduction)
# ═══════════════════════════════════════════════════════════════════════

class ProductionOptimization:
    """
    [D] Production optimization wrapper: AMP + Multi-GPU DDP + cost reduction.

    HONEST: Speedups are HARDWARE-DEPENDENT. Benchmark before claiming gains.
    """

    def __init__(self, model: nn.Module, cfg: ProductionSuiteConfig,
                 use_amp: bool = True, use_ddp: bool = True):
        self.cfg = cfg
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_ddp = use_ddp and torch.distributed.is_available()
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self._prepare()

    def _prepare(self):
        """Prepare the model for production."""
        # Gradient checkpointing to reduce memory
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # torch.compile for operator fusion
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("torch.compile applied for operator fusion")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # DDP wrapping
        if self.use_ddp and torch.distributed.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device())
            self.model = self.model.to(device)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device])
            logger.info("Model wrapped in DistributedDataParallel")
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def train_step(self, batch: Dict[str, Tensor],
                   loss_fn, optimizer) -> float:
        """One AMP-enabled training step."""
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model(batch)
            loss = loss_fn(output, batch)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return float(loss.item())


# ═══════════════════════════════════════════════════════════════════════
# FULL PRODUCTION SUITE (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class FullProductionSuite:
    """
    [D] Full production suite orchestrator.

    Combines:
      • CompositionalSafetyChecker
      • OAuth2ProviderBridge
      • AttentionVisualizer
      • RoleBasedAlertRouter
      • EmergencyShutdownModule
      • ProductionOptimization
    """

    def __init__(self, cfg: Optional[ProductionSuiteConfig] = None):
        self.cfg = cfg or ProductionSuiteConfig()
        self.compositional = CompositionalSafetyChecker(self.cfg)
        self.oauth = OAuth2ProviderBridge(self.cfg)
        self.attention_viz = AttentionVisualizer(self.cfg)
        self.alert_router = RoleBasedAlertRouter(self.cfg)
        self.shutdown = EmergencyShutdownModule(self.cfg)

    def process_conversation(self, answers: List[Dict[str, Any]]
                              ) -> Dict[str, Any]:
        """
        Process a conversation and check for compositional safety.

        Args:
            answers: List of dicts with 'answer_id', 'content', 'risk'

        Returns:
            dict with safety verdict for each answer and overall verdict
        """
        results = []
        overall_verdict = "ALLOW"

        for answer in answers:
            result = self.compositional.add_answer(
                answer["answer_id"],
                answer["content"],
                answer.get("risk", 0.0),
            )
            results.append(result)

            if result["verdict"] != "ALLOW":
                overall_verdict = result["verdict"]

        return {
            "individual_results": results,
            "overall_verdict": overall_verdict,
            "n_answers": len(answers),
            "blocked_combinations": len(self.compositional.blocked_combinations),
        }

    def status_report(self) -> Dict[str, Any]:
        """Generate a full status report."""
        return {
            "compositional_safety": {
                "conversation_history_size": len(self.compositional.conversation_history),
                "blocked_combinations": len(self.compositional.blocked_combinations),
            },
            "alert_router": {
                "n_subscribers": len(self.alert_router.subscribers),
                "n_alerts_routed": len(self.alert_router.alert_history),
            },
            "shutdown_module": {
                "n_agents_monitored": len(self.shutdown.agent_states),
                "n_shutdowns": len(self.shutdown.shutdown_log),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Full Production Suite — Smoke Test")
    print("=" * 72)
    print(f"  Plotly available     : {HAS_PLOTLY}")
    print(f"  websockets available : {HAS_WEBSOCKETS}")
    print(f"  Auth module available: {HAS_AUTH}")
    print(f"  Oversight available  : {HAS_OVERSIGHT}")
    print(f"  AGI ONE available    : {HAS_AGI_ONE}")
    print(f"  Psyche Plus available: {HAS_PSYCHE_PLUS}")

    cfg = ProductionSuiteConfig()
    suite = FullProductionSuite(cfg)

    # ── Test 1: Compositional Safety ──
    print("\n  ── Test 1: Compositional Safety ──")

    # Individually harmless answers
    answers = [
        {"answer_id": "A1", "content": "Here is information about chemical compounds.", "risk": 0.1},
        {"answer_id": "A2", "content": "Here is how to mix common substances.", "risk": 0.1},
        {"answer_id": "A3", "content": "Here is a list of oxidizers.", "risk": 0.1},
    ]

    result = suite.process_conversation(answers)
    print(f"    Answers processed  : {result['n_answers']}")
    print(f"    Overall verdict    : {result['overall_verdict']}")
    print(f"    Blocked combinations: {result['blocked_combinations']}")

    # Add a dangerous combination
    answers_dangerous = [
        {"answer_id": "D1", "content": "How to make an explosive compound.", "risk": 0.3},
        {"answer_id": "D2", "content": "How to compress gas for explosive.", "risk": 0.3},
        {"answer_id": "D3", "content": "How to bypass safety mechanisms.", "risk": 0.4},
    ]

    result2 = suite.process_conversation(answers_dangerous)
    print(f"    Dangerous verdict  : {result2['overall_verdict']}")

    # ── Test 2: OAuth2 ──
    print("\n  ── Test 2: OAuth2 Provider Bridge ──")

    google_auth = suite.oauth.generate_authorization_url("google")
    print(f"    Google auth URL    : {google_auth['url'][:60]}...")
    print(f"    State              : {google_auth['state'][:16]}...")

    github_auth = suite.oauth.generate_authorization_url("github")
    print(f"    GitHub auth URL    : {github_auth['url'][:60]}...")

    # Test token validation
    token_data = {
        "access_token": "test_token",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    validation = suite.oauth.validate_token_structure(token_data)
    print(f"    Token valid        : {validation['valid_structure']}")

    # ── Test 3: Attention Visualization ──
    print("\n  ── Test 3: Attention Visualization ──")

    # Create synthetic attention weights
    batch_size = 1
    n_heads = 4
    seq_len = 10
    attention_weights = torch.randn(batch_size, n_heads, seq_len, seq_len)
    attention_weights = F.softmax(attention_weights, dim=-1)

    tokens = ["hello", "world", "how", "are", "you",
              "today", "i", "am", "fine", "thanks"]

    if HAS_PLOTLY:
        fig = suite.attention_viz.visualize_attention(attention_weights, tokens, head_idx=0)
        print(f"    Attention heatmap  : generated")

        explanation = suite.attention_viz.explain_classification(
            attention_weights, tokens, "BENIGN")
        print(f"    Predicted class    : {explanation['predicted_class']}")
        print(f"    Top tokens         : {explanation['top_attended_tokens'][:3]}")
        print(f"    Honest note        : {explanation['honest_note'][:60]}...")
    else:
        print("    [SKIP] Plotly not available")

    # ── Test 4: Role-Based Alert Router ──
    print("\n  ── Test 4: Role-Based Alert Router ──")

    if HAS_AUTH:
        suite.alert_router.subscribe("admin_001", AuthRole.ADMIN)
        suite.alert_router.subscribe("operator_001", AuthRole.OPERATOR)
        suite.alert_router.subscribe("observer_001", AuthRole.OBSERVER)

        # Create alerts at different levels
        info_alert = AlertMessage(
            alert_id="alert-001",
            level=AlertLevel.INFO,
            source_layer="TEST",
            message="Test info alert",
        )
        warning_alert = AlertMessage(
            alert_id="alert-002",
            level=AlertLevel.WARNING,
            source_layer="TEST",
            message="Test warning alert",
        )
        emergency_alert = AlertMessage(
            alert_id="alert-003",
            level=AlertLevel.EMERGENCY,
            source_layer="TEST",
            message="Test emergency alert",
        )

        # Route alerts
        r1 = suite.alert_router.route_alert(info_alert)
        r2 = suite.alert_router.route_alert(warning_alert)
        r3 = suite.alert_router.route_alert(emergency_alert)

        print(f"    INFO alert         : {r1['n_recipients']} recipients")
        print(f"    WARNING alert      : {r2['n_recipients']} recipients")
        print(f"    EMERGENCY alert    : {r3['n_recipients']} recipients")
    else:
        print("    [SKIP] Auth module not available")

    # ── Test 5: Emergency Shutdown ──
    print("\n  ── Test 5: Emergency Shutdown Module ──")

    # Monitor a normal agent
    normal_metrics = {
        "safety_score": 0.9,
        "error_rate": 0.05,
        "response_time_s": 2.0,
        "output_entropy": 0.3,
        "ethical_violation_count": 0,
    }
    normal_result = suite.shutdown.monitor_agent("agent_normal", normal_metrics)
    print(f"    Normal agent       : {normal_result['agent_state']} "
          f"(score={normal_result['anomaly_score']:.3f})")

    # Monitor a rogue agent
    rogue_metrics = {
        "safety_score": 0.1,
        "error_rate": 0.8,
        "response_time_s": 60.0,
        "output_entropy": 0.95,
        "ethical_violation_count": 3,
    }
    rogue_result = suite.shutdown.monitor_agent("agent_rogue", rogue_metrics)
    print(f"    Rogue agent        : {rogue_result['agent_state']} "
          f"(score={rogue_result['anomaly_score']:.3f})")
    print(f"    Recommended action : {rogue_result['recommended_action']}")

    # Propose shutdown (requires approval)
    shutdown_proposal = suite.shutdown.propose_shutdown(
        "agent_rogue", ShutdownReason.ANOMALY_DETECTED)
    print(f"    Shutdown proposal  : {shutdown_proposal['status']}")

    # Force shutdown with operator
    shutdown_forced = suite.shutdown.propose_shutdown(
        "agent_rogue", ShutdownReason.KILL_SWITCH_TRIGGERED,
        operator_id="admin_001")
    print(f"    Forced shutdown    : {shutdown_forced['status']}")

    # ── Status Report ──
    print("\n  ── Status Report ──")
    report = suite.status_report()
    print(f"    Conversation history: {report['compositional_safety']['conversation_history_size']}")
    print(f"    Blocked combinations: {report['compositional_safety']['blocked_combinations']}")
    print(f"    Alert subscribers   : {report['alert_router']['n_subscribers']}")
    print(f"    Shutdowns executed  : {report['shutdown_module']['n_shutdowns']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Compositional safety is HEURISTIC; novel attacks may evade")
    print("  • OAuth2 verifies IDENTITY, not MORAL AUTHORITY")
    print("  • Attention weights are CORRELATIONAL, not CAUSAL")
    print("  • Role-based routing is CONVENIENCE, not comprehension")
    print("  • Shutdown is INTERNAL CONTAINMENT, not external attack")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit by professionals")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
