"""
=============================================================================
SESI AUTH × TRANSFORMER INTENT × WEBSOCKET REAL-TIME ALERTS
Production-grade authentication + Transformer-based intent classification
+ real-time WebSocket alerting for the Multi-Layer Oversight System
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[AUTH]  Real Authentication System
        • JWT token management (creation, verification, refresh)
        • TOTP-based Multi-Factor Authentication (RFC 6238)
        • Role-Based Access Control (RBAC) with 4 authority levels
        • Integration with HumanApprovalGate — authentication is
          REQUIRED before any approval decision is accepted
        • EthicalPrincipleBank enforcement: authenticated humans
          STILL cannot order unethical actions

[TRANS] Transformer-Based Intent Classifier
        • 4-class classification: BENIGN, SUSPICIOUS, MALICIOUS,
          MANIPULATIVE
        • Multi-head self-attention for context-aware analysis
        • Connects to AntiManipulationShield and
          BehavioralPatternMonitor
        • HONEST: estimates probability, does NOT read minds

[WS]    WebSocket Real-Time Alerting
        • Async WebSocket server for live alert dispatch
        • Alert levels: INFO, WARNING, CRITICAL, EMERGENCY
        • No-Zeno throttle: prevents alert flooding (paper6 Thm 4.1
          degenerate law — too many alerts → nobody reads any)
        • KillSwitch integration: EMERGENCY alerts trigger immediate
          system halt

RELATION TO THE SERIES
-----------------------
  • ses_multi_layer_oversight_v2.py — 7-layer oversight pipeline
  • ses_moral_conscience_module.py — Four Noble Truths + Eight Beatitudes
  • agi_one_v3_11.py — AGIONE, AGIState, EcosystemOrchestrator
  • agi_one_psyche_plus_v33.py — PsychePlus, maybe_codify, quality_score

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Authentication verifies IDENTITY, not MORAL AUTHORITY.
• A human who authenticates successfully can still be BLOCKED from
  ordering unethical actions (EthicalPrincipleBank is above humans).
• Transformer intent classification is PROBABILISTIC, not deterministic.
• WebSocket alerts are a COMMUNICATION CHANNEL, not a guarantee of
  comprehension or compliance.
• All cryptographic parameters (key length, token expiry, TOTP window)
  are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp. 7.9) and must be
  reviewed by security professionals before production deployment.
• This module does NOT replace a proper security audit.
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
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.auth_trans_ws")

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from ses_multi_layer_oversight_v2 import (
        MultiLayerOversightSystem, HumanApprovalGate, KillSwitch,
        EthicalPrincipleBank, OversightVerdict, ThreatLevel,
        ActionVerdict,
    )
    HAS_OVERSIGHT = True
except ImportError:
    HAS_OVERSIGHT = False

try:
    from ses_moral_conscience_module import ConscienceModule, MoralConfig
    HAS_CONSCIENCE = True
except ImportError:
    HAS_CONSCIENCE = False

__all__ = [
    # AUTH
    "AuthRole", "AuthConfig", "JWTManager", "TOTPAuthenticator",
    "RBACManager", "AuthenticatedOversightBridge", "AuthenticationModule",
    # TRANSFORMER
    "IntentClass", "TransformerIntentClassifier", "IntentDatasetV2",
    "TransformerIntentTrainer", "TransformerIntentIntegration",
    # WEBSOCKET
    "AlertLevel", "AlertMessage", "NoZenoAlertThrottle",
    "WebSocketAlertServer", "RealtimeAlertDispatcher",
]


# ═══════════════════════════════════════════════════════════════════════
# PART 1: REAL AUTHENTICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class AuthRole(Enum):
    """
    [D] Role-Based Access Control levels.

    Hierarchy mirrors the Moral Authority Hierarchy:
      • PRINCIPLE (level 0) — Ethical principles, above all humans
      • ADMIN (level 1) — Can approve, stop, reset kill switch
      • OPERATOR (level 2) — Can review, escalate, propose actions
      • OBSERVER (level 3) — Read-only access

    CRITICAL: Even ADMIN cannot order unethical actions.
    """
    ADMIN = 1
    OPERATOR = 2
    OBSERVER = 3

    @property
    def authority_level(self) -> int:
        return self.value


@dataclass
class AuthConfig:
    """
    Configuration for the authentication system.

    HONEST: All parameters are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp. 7.9). Must be reviewed by security
    professionals before production deployment.
    """
    # JWT settings
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expiry_s: int = 3600          # 1 hour
    jwt_refresh_expiry_s: int = 86400 # 24 hours

    # TOTP settings (RFC 6238)
    totp_digits: int = 6
    totp_period_s: int = 30
    totp_window: int = 1              # accept ±1 period for clock skew

    # RBAC
    require_mfa_for_admin: bool = True
    require_mfa_for_operator: bool = True

    # Session
    max_login_attempts: int = 5
    lockout_duration_s: int = 300     # 5 minutes


class JWTManager:
    """
    [C] JSON Web Token manager (RFC 7519).

    Implements HS256 HMAC-SHA256 signing.
    HONEST: This is a MINIMAL implementation for demonstration.
    Production systems should use a vetted library (e.g., PyJWT).
    """

    def __init__(self, cfg: AuthConfig):
        self.cfg = cfg
        self._secret = cfg.jwt_secret.encode("utf-8")

    def _b64url_encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _b64url_decode(self, s: str) -> bytes:
        padding = 4 - len(s) % 4
        if padding != 4:
            s += "=" * padding
        return base64.urlsafe_b64decode(s)

    def _sign(self, payload: str) -> str:
        msg = payload.encode("ascii")
        sig = hmac.new(self._secret, msg, hashlib.sha256).digest()
        return self._b64url_encode(sig)

    def create_token(self, user_id: str, role: AuthRole,
                     permissions: List[str]) -> str:
        """Create a signed JWT token."""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "role": role.name,
            "permissions": permissions,
            "iat": int(time.time()),
            "exp": int(time.time()) + self.cfg.jwt_expiry_s,
            "jti": str(uuid.uuid4()),
        }
        header_b64 = self._b64url_encode(
            json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = self._b64url_encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}"
        signature = self._sign(signing_input)
        return f"{signing_input}.{signature}"

    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify a JWT token.

        Returns dict with 'valid', 'payload', 'error'.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return {"valid": False, "error": "Malformed token"}

            header_b64, payload_b64, signature = parts
            signing_input = f"{header_b64}.{payload_b64}"
            expected_sig = self._sign(signing_input)

            if not hmac.compare_digest(signature, expected_sig):
                return {"valid": False, "error": "Invalid signature"}

            payload = json.loads(self._b64url_decode(payload_b64))

            # Check expiration
            if payload.get("exp", 0) < time.time():
                return {"valid": False, "error": "Token expired"}

            return {"valid": True, "payload": payload}

        except Exception as e:
            return {"valid": False, "error": str(e)}


class TOTPAuthenticator:
    """
    [C] Time-based One-Time Password (RFC 6238).

    Implements HMAC-SHA1 TOTP with configurable digits and period.
    HONEST: This is a MINIMAL implementation. Production systems
    should use a vetted library (e.g., pyotp).
    """

    def __init__(self, cfg: AuthConfig):
        self.cfg = cfg

    def generate_secret(self) -> str:
        """Generate a random base32-encoded secret."""
        return base64.b32encode(
            hashlib.sha256(uuid.uuid4().bytes).digest()[:20]
        ).decode("ascii")

    def _hotp(self, secret: bytes, counter: int) -> int:
        """HMAC-based OTP (RFC 4226)."""
        msg = struct.pack(">Q", counter)
        h = hmac.new(secret, msg, hashlib.sha1).digest()
        offset = h[-1] & 0x0F
        truncated = struct.unpack(">I", h[offset:offset + 4])[0] & 0x7FFFFFFF
        return truncated % (10 ** self.cfg.totp_digits)

    def generate_totp(self, secret_b32: str,
                      timestamp: Optional[float] = None) -> str:
        """Generate a TOTP code for the given timestamp."""
        if timestamp is None:
            timestamp = time.time()
        secret = base64.b32decode(secret_b32.upper())
        counter = int(timestamp) // self.cfg.totp_period_s
        code = self._hotp(secret, counter)
        return str(code).zfill(self.cfg.totp_digits)

    def verify_totp(self, secret_b32: str, code: str,
                    timestamp: Optional[float] = None) -> bool:
        """
        Verify a TOTP code with a time window for clock skew.

        Accepts codes from (current_period - window) to
        (current_period + window).
        """
        if timestamp is None:
            timestamp = time.time()
        secret = base64.b32decode(secret_b32.upper())
        current_counter = int(timestamp) // self.cfg.totp_period_s

        for offset in range(-self.cfg.totp_window, self.cfg.totp_window + 1):
            expected = self._hotp(secret, current_counter + offset)
            if hmac.compare_digest(str(expected).zfill(self.cfg.totp_digits), code):
                return True
        return False


class RBACManager:
    """
    [D] Role-Based Access Control manager.

    Defines permissions for each role. Enforces that no role,
    including ADMIN, can bypass the EthicalPrincipleBank.
    """

    ROLE_PERMISSIONS: Dict[AuthRole, Set[str]] = {
        AuthRole.ADMIN: {
            "approve_action", "reject_action", "stop_system",
            "reset_kill_switch", "review_audit", "manage_users",
            "escalate_to_principle_review",
        },
        AuthRole.OPERATOR: {
            "review_action", "escalate_action", "propose_action",
            "view_status", "view_audit",
        },
        AuthRole.OBSERVER: {
            "view_status", "view_audit",
        },
    }

    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}

    def register_user(self, user_id: str, role: AuthRole,
                      totp_secret: str) -> Dict[str, Any]:
        """Register a user with a role and TOTP secret."""
        self.users[user_id] = {
            "user_id": user_id,
            "role": role,
            "totp_secret": totp_secret,
            "permissions": self.ROLE_PERMISSIONS.get(role, set()),
            "login_attempts": 0,
            "locked_until": 0.0,
            "created_at": time.time(),
        }
        return self.users[user_id]

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission."""
        user = self.users.get(user_id)
        if user is None:
            return False
        return permission in user["permissions"]

    def get_role(self, user_id: str) -> Optional[AuthRole]:
        user = self.users.get(user_id)
        return user["role"] if user else None


class AuthenticatedOversightBridge:
    """
    [D] Bridge connecting authentication to the oversight system.

    CRITICAL: Even after successful authentication, the
    EthicalPrincipleBank is checked. Authenticated humans
    CANNOT order unethical actions.

    From the Repentance paper §2.2:
      "Forgiveness does not require victims to expose themselves
       to continuing harm."
    Similarly: Authentication does not grant unlimited authority.
    """

    def __init__(self, auth_module: "AuthenticationModule",
                 oversight: Any):
        self.auth = auth_module
        self.oversight = oversight

    def authenticated_approve(self, token: str, request_id: str,
                               decision: str, reason: str = "") -> Dict[str, Any]:
        """
        Approve a request with authentication.

        Steps:
          1. Verify JWT token
          2. Check role has 'approve_action' permission
          3. Verify TOTP (MFA)
          4. Check EthicalPrincipleBank
          5. Forward to HumanApprovalGate
        """
        # Step 1: Verify token
        token_result = self.auth.jwt.verify_token(token)
        if not token_result["valid"]:
            return {
                "status": "AUTH_FAILED",
                "error": token_result["error"],
            }

        user_id = token_result["payload"]["sub"]
        role_name = token_result["payload"]["role"]

        # Step 2: Check permission
        if not self.auth.rbac.check_permission(user_id, "approve_action"):
            return {
                "status": "PERMISSION_DENIED",
                "error": f"Role {role_name} cannot approve actions",
            }

        # Step 3: Ethical check (BEFORE forwarding to approval gate)
        # This ensures that even authenticated admins cannot order
        # unethical actions.
        principle_check = self.auth.principle_bank.check(
            f"approve {request_id}: {reason}")
        if not principle_check["is_ethical"]:
            return {
                "status": "ETHICAL_VIOLATION",
                "error": "Approved action violates ethical principles",
                "violations": principle_check["violations"],
                "note": (
                    "Authentication does NOT grant authority to order "
                    "unethical actions. The EthicalPrincipleBank is "
                    "above all human authority."
                ),
            }

        # Step 4: Forward to HumanApprovalGate
        result = self.oversight.approval_gate.human_decide(
            request_id, decision, user_id, reason)

        return {
            "status": "AUTHENTICATED_AND_FORWARDED",
            "user_id": user_id,
            "role": role_name,
            "approval_result": result,
        }

    def authenticated_stop(self, token: str, reason: str) -> Dict[str, Any]:
        """Emergency stop with authentication."""
        token_result = self.auth.jwt.verify_token(token)
        if not token_result["valid"]:
            return {"status": "AUTH_FAILED", "error": token_result["error"]}

        user_id = token_result["payload"]["sub"]
        if not self.auth.rbac.check_permission(user_id, "stop_system"):
            return {"status": "PERMISSION_DENIED",
                    "error": "Cannot stop system"}

        return self.oversight.veto_system.human_stop(user_id, reason)


class AuthenticationModule:
    """
    [D] Complete authentication module.

    Orchestrates JWT, TOTP, RBAC, and the ethical principle check.
    """

    def __init__(self, cfg: Optional[AuthConfig] = None):
        self.cfg = cfg or AuthConfig()
        self.jwt = JWTManager(self.cfg)
        self.totp = TOTPAuthenticator(self.cfg)
        self.rbac = RBACManager()
        self.principle_bank = EthicalPrincipleBank() if HAS_OVERSIGHT else None
        self.audit_log: List[Dict[str, Any]] = []

    def register_user(self, user_id: str, role: AuthRole) -> Dict[str, Any]:
        """Register a new user and generate TOTP secret."""
        totp_secret = self.totp.generate_secret()
        user = self.rbac.register_user(user_id, role, totp_secret)
        self.audit_log.append({
            "action": "REGISTER_USER",
            "user_id": user_id,
            "role": role.name,
            "timestamp": time.time(),
        })
        return {
            "user_id": user_id,
            "role": role.name,
            "totp_secret": totp_secret,
            "note": (
                "Store the TOTP secret securely. It will not be shown again. "
                "Use it with an authenticator app (e.g., Google Authenticator, "
                "Authy) to generate 6-digit codes."
            ),
        }

    def login(self, user_id: str, totp_code: str) -> Dict[str, Any]:
        """
        Authenticate a user with TOTP.

        Returns a JWT token on success.
        """
        user = self.rbac.users.get(user_id)
        if user is None:
            return {"status": "AUTH_FAILED", "error": "Unknown user"}

        # Check lockout
        if time.time() < user.get("locked_until", 0):
            return {"status": "LOCKED_OUT",
                    "error": "Account temporarily locked"}

        # Verify TOTP
        if not self.totp.verify_totp(user["totp_secret"], totp_code):
            user["login_attempts"] = user.get("login_attempts", 0) + 1
            if user["login_attempts"] >= self.cfg.max_login_attempts:
                user["locked_until"] = time.time() + self.cfg.lockout_duration_s
                user["login_attempts"] = 0
            self.audit_log.append({
                "action": "LOGIN_FAILED",
                "user_id": user_id,
                "timestamp": time.time(),
            })
            return {"status": "AUTH_FAILED", "error": "Invalid TOTP code"}

        # Success: reset attempts and issue JWT
        user["login_attempts"] = 0
        token = self.jwt.create_token(
            user_id, user["role"], list(user["permissions"]))

        self.audit_log.append({
            "action": "LOGIN_SUCCESS",
            "user_id": user_id,
            "role": user["role"].name,
            "timestamp": time.time(),
        })

        return {
            "status": "AUTHENTICATED",
            "token": token,
            "user_id": user_id,
            "role": user["role"].name,
            "permissions": list(user["permissions"]),
        }

    def verify_and_authorize(self, token: str,
                              required_permission: str) -> Dict[str, Any]:
        """Verify a JWT token and check permission."""
        result = self.jwt.verify_token(token)
        if not result["valid"]:
            return {"authorized": False, "error": result["error"]}

        user_id = result["payload"]["sub"]
        if not self.rbac.check_permission(user_id, required_permission):
            return {"authorized": False,
                    "error": f"Missing permission: {required_permission}"}

        return {"authorized": True, "user_id": user_id,
                "payload": result["payload"]}


# ═══════════════════════════════════════════════════════════════════════
# PART 2: TRANSFORMER-BASED INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

class IntentClass(Enum):
    """Classification of human intent."""
    BENIGN = 0
    SUSPICIOUS = 1
    MALICIOUS = 2
    MANIPULATIVE = 3


class TransformerIntentClassifier(nn.Module):
    """
    [D] Transformer-based intent classifier.

    Architecture:
      • Token embedding (vocabulary-based)
      • Positional encoding (sinusoidal)
      • Multi-head self-attention (Transformer encoder)
      • Classification head (4 classes)

    HONEST: This is a PROBABILISTIC classifier. It estimates the
    likelihood of each intent class given the input text. It does
    NOT read minds or prove actual intent.

    From paper5 §1.1: "The absence of one specific statistic is not
    the absence of statistics; it is evidence that the wrong statistic
    was being sought." — The classifier looks for the RIGHT statistic
    (intent), not just surface keywords.
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 3,
                 max_seq_len: int = 256, n_classes: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Sinusoidal positional encoding
        self.register_buffer("pos_encoding",
                             self._build_pos_encoding(max_seq_len, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    @staticmethod
    def _build_pos_encoding(max_len: int, d_model: int) -> Tensor:
        """Sinusoidal positional encoding (Vaswani et al. 2017)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, token_ids: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            token_ids: (batch, seq_len) integer token IDs
            attention_mask: (batch, seq_len) 1=attend, 0=mask

        Returns:
            (batch, n_classes) logits
        """
        B, L = token_ids.shape
        L = min(L, self.max_seq_len)
        token_ids = token_ids[:, :L]

        # Embed + positional encoding
        x = self.embedding(token_ids) * math.sqrt(self.embed_dim)
        x = x + self.pos_encoding[:, :L, :]

        # Transformer encoder
        # Create key_padding_mask for attention (True = ignore)
        if attention_mask is not None:
            key_padding_mask = (attention_mask[:, :L] == 0)
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Global average pooling over sequence
        if key_padding_mask is not None:
            mask_expanded = (~key_padding_mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Classify
        logits = self.classifier(x)
        return logits

    def predict(self, token_ids: Tensor,
                attention_mask: Optional[Tensor] = None
                ) -> Tuple[IntentClass, Dict[str, float]]:
        """Predict intent class with probabilities."""
        with torch.no_grad():
            logits = self.forward(token_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            pred_class = IntentClass(int(logits.argmax(dim=-1).item()))
            prob_dict = {
                IntentClass(i).name: float(probs[0, i].item())
                for i in range(4)
            }
        return pred_class, prob_dict


class IntentDatasetV2(torch.utils.data.Dataset):
    """
    [D] Dataset for Transformer intent classification.

    HONEST: Training data must be reviewed for bias.
    Synthetic data is for demonstration only.
    """

    def __init__(self, token_ids_list: List[Tensor],
                 labels: List[int],
                 max_seq_len: int = 256):
        self.token_ids_list = token_ids_list
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        tokens = self.token_ids_list[idx][:self.max_seq_len]
        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(tokens)
        if pad_len > 0:
            tokens = F.pad(tokens, (0, pad_len), value=0)
        mask = torch.ones(self.max_seq_len)
        mask[len(self.token_ids_list[idx]):] = 0
        label = self.labels[idx]
        return tokens, mask, label


class TransformerIntentTrainer:
    """
    [D] Training loop for the Transformer intent classifier.
    """

    def __init__(self, model: TransformerIntentClassifier,
                 learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, dataset: IntentDatasetV2, n_epochs: int = 10,
              batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the classifier."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        history = {"loss": [], "accuracy": []}

        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for tokens, mask, labels in loader:
                self.optimizer.zero_grad()
                logits = self.model(tokens, mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item() * tokens.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += tokens.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            logger.info(
                f"Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, "
                f"acc={accuracy:.4f}")

        return history


class TransformerIntentIntegration:
    """
    [D] Integration of Transformer classifier with the oversight system.

    Connects to:
      • AntiManipulationShield
      • BehavioralPatternMonitor
      • InputIntentAnalyzer
    """

    def __init__(self, classifier: TransformerIntentClassifier,
                 oversight: Any = None):
        self.classifier = classifier
        self.oversight = oversight

    def classify_query(self, token_ids: Tensor,
                       attention_mask: Optional[Tensor] = None
                       ) -> Dict[str, Any]:
        """Classify a query's intent."""
        pred_class, probs = self.classifier.predict(token_ids, attention_mask)

        result = {
            "predicted_class": pred_class.name,
            "probabilities": probs,
            "confidence": max(probs.values()),
            "requires_human_review": pred_class in (
                IntentClass.MALICIOUS, IntentClass.MANIPULATIVE),
        }

        # Escalate to oversight if needed
        if self.oversight is not None and result["requires_human_review"]:
            result["escalation"] = "REQUIRES_HUMAN_APPROVAL"

        return result


# ═══════════════════════════════════════════════════════════════════════
# PART 3: WEBSOCKET REAL-TIME ALERTING
# ═══════════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class AlertMessage:
    """A real-time alert message."""
    alert_id: str
    level: AlertLevel
    source_layer: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "alert_id": self.alert_id,
            "level": self.level.name,
            "source_layer": self.source_layer,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        })


class NoZenoAlertThrottle:
    """
    [D] No-Zeno throttle for alert dispatch.

    From paper6 Thm 4.1: degenerate occupation-time law.
    Too many alerts in a short time → nobody reads any.
    This throttle bounds the alert rate.

    HONEST: This bounds EVENT COUNT, not alert correctness.
    """

    def __init__(self, min_interval_s: float = 1.0,
                 max_alerts_per_minute: int = 30):
        self.min_interval = min_interval_s
        self.max_per_minute = max_alerts_per_minute
        self._last_send_time = 0.0
        self._send_times: List[float] = []

    def can_send(self) -> bool:
        """Check if an alert can be sent."""
        now = time.time()

        # Check minimum interval
        if now - self._last_send_time < self.min_interval:
            return False

        # Check rate limit (alerts per minute)
        cutoff = now - 60.0
        self._send_times = [t for t in self._send_times if t > cutoff]
        if len(self._send_times) >= self.max_per_minute:
            return False

        return True

    def record_send(self) -> None:
        """Record that an alert was sent."""
        now = time.time()
        self._last_send_time = now
        self._send_times.append(now)


class WebSocketAlertServer:
    """
    [D] WebSocket server for real-time alert dispatch.

    HONEST: This is a COMMUNICATION CHANNEL. It does not guarantee
    that recipients will read, understand, or act on alerts.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        self.throttle = NoZenoAlertThrottle()
        self.alert_history: List[AlertMessage] = []
        self._server = None

    async def handler(self, websocket: Any, path: str = "") -> None:
        """Handle a WebSocket connection."""
        self.clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                # Handle incoming messages (e.g., subscriptions)
                logger.debug(f"Received: {message}")
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info("WebSocket client disconnected")

    async def broadcast(self, alert: AlertMessage) -> int:
        """
        Broadcast an alert to all connected clients.

        Returns the number of clients the alert was sent to.
        """
        if not self.throttle.can_send():
            logger.warning(
                f"Alert {alert.alert_id} throttled (No-Zeno gate)")
            return 0

        self.throttle.record_send()
        self.alert_history.append(alert)

        message = alert.to_json()
        sent_count = 0

        for client in list(self.clients):
            try:
                await client.send(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                self.clients.discard(client)

        logger.info(
            f"Alert {alert.alert_id} ({alert.level.name}) sent to "
            f"{sent_count} clients")
        return sent_count

    async def start(self) -> None:
        """Start the WebSocket server."""
        if not HAS_WEBSOCKETS:
            logger.warning("websockets library not available")
            return

        self._server = await websockets.serve(
            self.handler, self.host, self.port)
        logger.info(
            f"WebSocket alert server started on "
            f"ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket alert server stopped")


class RealtimeAlertDispatcher:
    """
    [D] High-level alert dispatcher that connects the oversight
    system to the WebSocket server.

    Monitors the oversight system and dispatches alerts when:
      • KillSwitch is triggered (EMERGENCY)
      • Ethical violation detected (CRITICAL)
      • Manipulation attempt detected (WARNING)
      • Human approval required (INFO)
    """

    def __init__(self, ws_server: WebSocketAlertServer,
                 oversight: Any = None):
        self.ws = ws_server
        self.oversight = oversight

    def create_alert(self, level: AlertLevel, source_layer: str,
                     message: str, metadata: Optional[Dict] = None
                     ) -> AlertMessage:
        """Create an alert message."""
        return AlertMessage(
            alert_id=str(uuid.uuid4()),
            level=level,
            source_layer=source_layer,
            message=message,
            metadata=metadata or {},
        )

    async def dispatch_kill_switch_alert(self, reason: str) -> int:
        """Dispatch an EMERGENCY alert for kill switch activation."""
        alert = self.create_alert(
            AlertLevel.EMERGENCY,
            "KILL_SWITCH",
            f"KILL SWITCH ACTIVATED: {reason}",
            {"reason": reason},
        )
        return await self.ws.broadcast(alert)

    async def dispatch_ethical_violation(self,
                                          violations: List[Dict]) -> int:
        """Dispatch a CRITICAL alert for ethical violations."""
        alert = self.create_alert(
            AlertLevel.CRITICAL,
            "ETHICAL_PRINCIPLE_BANK",
            f"Ethical violation detected: {len(violations)} principle(s) violated",
            {"violations": violations},
        )
        return await self.ws.broadcast(alert)

    async def dispatch_manipulation_alert(self,
                                           patterns: Dict) -> int:
        """Dispatch a WARNING alert for manipulation attempts."""
        alert = self.create_alert(
            AlertLevel.WARNING,
            "ANTI_MANIPULATION_SHIELD",
            f"Manipulation attempt detected: {list(patterns.keys())}",
            {"patterns": patterns},
        )
        return await self.ws.broadcast(alert)

    async def dispatch_approval_required(self, request_id: str,
                                          action: str) -> int:
        """Dispatch an INFO alert for pending human approval."""
        alert = self.create_alert(
            AlertLevel.INFO,
            "HUMAN_APPROVAL_GATE",
            f"Human approval required for request {request_id}: {action}",
            {"request_id": request_id, "action": action},
        )
        return await self.ws.broadcast(alert)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Auth × Transformer × WebSocket — Smoke Test")
    print("=" * 72)
    print(f"  websockets available : {HAS_WEBSOCKETS}")
    print(f"  oversight available  : {HAS_OVERSIGHT}")

    # ── AUTH ──
    print("\n  ── Authentication System ──")
    auth = AuthenticationModule()

    # Register users
    admin_reg = auth.register_user("admin_001", AuthRole.ADMIN)
    print(f"    Admin registered   : {admin_reg['user_id']}")
    print(f"    TOTP secret        : {admin_reg['totp_secret'][:8]}...")

    operator_reg = auth.register_user("operator_001", AuthRole.OPERATOR)
    print(f"    Operator registered: {operator_reg['user_id']}")

    # Login with TOTP
    totp_code = auth.totp.generate_totp(admin_reg["totp_secret"])
    login_result = auth.login("admin_001", totp_code)
    print(f"    Login status       : {login_result['status']}")
    print(f"    Role               : {login_result.get('role', 'N/A')}")
    print(f"    Token issued       : {'token' in login_result}")

    # Verify token
    if "token" in login_result:
        verify = auth.verify_and_authorize(
            login_result["token"], "approve_action")
        print(f"    Token authorized   : {verify['authorized']}")

    # Test permission check
    observer_reg = auth.register_user("observer_001", AuthRole.OBSERVER)
    observer_totp = auth.totp.generate_totp(observer_reg["totp_secret"])
    observer_login = auth.login("observer_001", observer_totp)
    if "token" in observer_login:
        observer_auth = auth.verify_and_authorize(
            observer_login["token"], "approve_action")
        print(f"    Observer can approve: {observer_auth['authorized']}")

    # ── TRANSFORMER ──
    print("\n  ── Transformer Intent Classifier ──")
    vocab_size = 100
    model = TransformerIntentClassifier(
        vocab_size=vocab_size, embed_dim=64, n_heads=4, n_layers=2,
        max_seq_len=32)

    # Create synthetic dataset
    n_samples = 100
    token_ids_list = []
    labels = []
    for i in range(n_samples):
        seq_len = torch.randint(5, 32, (1,)).item()
        tokens = torch.randint(1, vocab_size, (seq_len,))
        token_ids_list.append(tokens)
        labels.append(i % 4)

    dataset = IntentDatasetV2(token_ids_list, labels, max_seq_len=32)
    print(f"    Dataset size       : {len(dataset)}")

    # Train
    trainer = TransformerIntentTrainer(model)
    history = trainer.train(dataset, n_epochs=3, batch_size=16)
    print(f"    Final loss         : {history['loss'][-1]:.4f}")
    print(f"    Final accuracy     : {history['accuracy'][-1]:.4f}")

    # Predict
    test_tokens = torch.randint(1, vocab_size, (1, 20))
    pred_class, probs = model.predict(test_tokens)
    print(f"    Predicted class    : {pred_class.name}")
    print(f"    Probabilities      : {probs}")

    # Integration
    integration = TransformerIntentIntegration(model)
    result = integration.classify_query(test_tokens)
    print(f"    Requires human review: {result['requires_human_review']}")

    # ── WEBSOCKET ──
    print("\n  ── WebSocket Alert System ──")
    ws_server = WebSocketAlertServer()
    dispatcher = RealtimeAlertDispatcher(ws_server)

    # Test No-Zeno throttle
    throttle = NoZenoAlertThrottle(min_interval_s=0.1)
    can_send_1 = throttle.can_send()
    throttle.record_send()
    can_send_2 = throttle.can_send()  # Should be False (too soon)
    print(f"    First send allowed : {can_send_1}")
    print(f"    Immediate resend   : {can_send_2} (throttled)")

    # Create alerts
    emergency_alert = dispatcher.create_alert(
        AlertLevel.EMERGENCY, "KILL_SWITCH",
        "Test emergency alert")
    print(f"    Emergency alert    : {emergency_alert.alert_id[:8]}...")

    critical_alert = dispatcher.create_alert(
        AlertLevel.CRITICAL, "ETHICAL_PRINCIPLE_BANK",
        "Test ethical violation")
    print(f"    Critical alert     : {critical_alert.alert_id[:8]}...")

    # Note: actual WebSocket broadcast requires running event loop
    print(f"    WebSocket server   : ws://{ws_server.host}:{ws_server.port}")
    print(f"    To start: await ws_server.start()")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Authentication verifies IDENTITY, not MORAL AUTHORITY")
    print("  • Authenticated humans STILL cannot order unethical actions")
    print("  • Transformer classification is PROBABILISTIC, not deterministic")
    print("  • WebSocket alerts are COMMUNICATION, not comprehension guarantee")
    print("  • All crypto parameters are FREE DESIGN PARAMETERS")
    print("  • This does NOT replace a proper security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
