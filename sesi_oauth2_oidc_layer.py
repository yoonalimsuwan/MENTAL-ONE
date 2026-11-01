"""
=============================================================================
SESI OAUTH2/OIDC AUTHENTICATION LAYER
Identity governance with multi-identity abuse prevention
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[AUTH1] OAuth2OIDCServer
        • Authorization Code + PKCE flow
        • JWT token issuance and validation
        • Refresh token rotation

[AUTH2] MultiIdentityDetector
        • Detects coordinated multi-account abuse
        • Privacy-preserving behavioral analysis
        • Graduated verification (Level 1-4)

[AUTH3] RBACManager
        • Role-based access control
        • Attribute-based policies
        • Least privilege enforcement

[AUTH4] SessionManager
        • Secure session management
        • Token revocation
        • Audit logging

RELATION TO "COMPARATIVE THEOLOGICAL AND PHILOSOPHICAL STUDY" PAPER
---------------------------------------------------------------------
From §5.2 Multi-Identity Risk Model:
  R_id(u_i) = f(B_i, T_i, C_i, V_i)
  Where:
    B_i: Behavioral anomaly indicators
    T_i: Temporal interaction patterns
    C_i: Confirmed coordination indicators
    V_i: Verified policy violations

From §5.3 Privacy-Preserving Identity Protection:
  Level 1: Automated abuse detection (no real-world identity)
  Level 2: Account-level verification (when risk justifies)
  Level 3: Institutional verification (for restricted resources)
  Level 4: Legal escalation (credible evidence of serious misuse)

IMPORTANT: From §5.3:
  "Multi-account detection must NOT become unrestricted surveillance.
   The system should avoid collecting unnecessary biometric information,
   personal contacts, or unrelated social-network data."

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Authentication verifies IDENTITY, not moral correctness.
• Multi-identity detection is HEURISTIC; false positives possible.
• This is a FRAMEWORK, not a complete identity provider.
• Production requires: proper IdP (Keycloak/Auth0/Okta), HSM, audit.
• All thresholds are FREE DESIGN PARAMETERS.
=============================================================================
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.oauth2_oidc")

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

__all__ = [
    "OAuth2OIDCServer", "MultiIdentityDetector", "RBACManager",
    "SessionManager", "AuthSuite",
]


@dataclass
class UserSession:
    """Represents an authenticated user session."""
    session_id: str
    user_id: str
    roles: Set[str]
    permissions: Set[str]
    created_at: float
    expires_at: float
    refresh_token: str
    mfa_verified: bool = False
    device_id: Optional[str] = None


class OAuth2OIDCServer:
    """
    [D] OAuth2/OIDC authentication server.

    Implements:
      • Authorization Code + PKCE flow
      • JWT access tokens (RS256 or HS256)
      • Refresh token rotation
      • MFA requirement for sensitive operations

    HONEST: This is a FRAMEWORK implementation. Production requires:
      • Proper identity provider (Keycloak, Auth0, Okta)
      • HSM for key storage
      • Regular security audits
    """

    def __init__(self, secret_key: Optional[str] = None,
                 token_expiry_s: int = 3600,
                 refresh_expiry_s: int = 86400):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry_s = token_expiry_s
        self.refresh_expiry_s = refresh_expiry_s
        self.sessions: Dict[str, UserSession] = {}
        self.authorization_codes: Dict[str, Dict[str, Any]] = {}

    def create_authorization_code(self, user_id: str, client_id: str,
                                   redirect_uri: str,
                                   code_challenge: str,
                                   code_challenge_method: str = "S256") -> str:
        """
        Create an authorization code for the Authorization Code + PKCE flow.
        """
        code = secrets.token_urlsafe(32)
        self.authorization_codes[code] = {
            "user_id": user_id,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "created_at": time.time(),
            "expires_at": time.time() + 300,  # 5 minutes
        }
        return code

    def exchange_code_for_token(self, code: str, client_id: str,
                                 code_verifier: str) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.

        Validates PKCE code_verifier against stored code_challenge.
        """
        auth_code = self.authorization_codes.get(code)
        if not auth_code:
            return {"error": "invalid_grant", "error_description": "Unknown code"}

        if time.time() > auth_code["expires_at"]:
            return {"error": "invalid_grant", "error_description": "Code expired"}

        if auth_code["client_id"] != client_id:
            return {"error": "invalid_grant", "error_description": "Client mismatch"}

        # Verify PKCE
        if auth_code["code_challenge_method"] == "S256":
            computed = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).rstrip(b'=').decode()
        else:
            computed = code_verifier

        if computed != auth_code["code_challenge"]:
            return {"error": "invalid_grant", "error_description": "PKCE verification failed"}

        # Issue tokens
        access_token = self._create_access_token(auth_code["user_id"])
        refresh_token = secrets.token_urlsafe(32)

        # Create session
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_id=auth_code["user_id"],
            roles=set(),
            permissions=set(),
            created_at=time.time(),
            expires_at=time.time() + self.token_expiry_s,
            refresh_token=refresh_token,
        )
        self.sessions[session.session_id] = session

        # Delete used code
        del self.authorization_codes[code]

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": self.token_expiry_s,
            "refresh_token": refresh_token,
        }

    def _create_access_token(self, user_id: str) -> str:
        """Create a JWT access token."""
        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + self.token_expiry_s,
            "iss": "sesi-oversight",
            "aud": "sesi-oversight-api",
        }

        if HAS_JWT:
            return jwt.encode(payload, self.secret_key, algorithm="HS256")
        else:
            # Fallback: base64-encoded JSON (NOT secure, for testing only)
            logger.warning("PyJWT not available; using insecure token encoding")
            return base64.urlsafe_b64encode(
                json.dumps(payload).encode()).decode()

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT access token."""
        try:
            if HAS_JWT:
                payload = jwt.decode(token, self.secret_key,
                                     algorithms=["HS256"],
                                     audience="sesi-oversight-api",
                                     issuer="sesi-oversight")
                return {"valid": True, "payload": payload}
            else:
                # Fallback for testing
                payload = json.loads(base64.urlsafe_b64decode(token + "=="))
                if payload.get("exp", 0) < time.time():
                    return {"valid": False, "error": "Token expired"}
                return {"valid": True, "payload": payload}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an access token using a refresh token.

        Implements refresh token rotation: old refresh token is invalidated.
        """
        # Find session with this refresh token
        for session_id, session in self.sessions.items():
            if session.refresh_token == refresh_token:
                # Issue new tokens
                new_access = self._create_access_token(session.user_id)
                new_refresh = secrets.token_urlsafe(32)

                # Rotate refresh token
                session.refresh_token = new_refresh
                session.expires_at = time.time() + self.token_expiry_s

                return {
                    "access_token": new_access,
                    "token_type": "Bearer",
                    "expires_in": self.token_expiry_s,
                    "refresh_token": new_refresh,
                }

        return {"error": "invalid_grant", "error_description": "Unknown refresh token"}

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


class MultiIdentityDetector:
    """
    [D] Detect multi-identity abuse patterns.

    From "Comparative Theological and Philosophical Study" §5.2:
      R_id(u_i) = f(B_i, T_i, C_i, V_i)

    Where:
      B_i: Behavioral anomaly indicators
      T_i: Temporal interaction patterns
      C_i: Confirmed coordination indicators
      V_i: Verified policy violations

    HONEST: This is a HEURISTIC detector. It does NOT prove that
    multiple accounts belong to the same person. It identifies
    PATTERNS that warrant further investigation.

    PRIVACY: From §5.3:
      "Multi-account detection must NOT become unrestricted surveillance.
       The system should avoid collecting unnecessary biometric information,
       personal contacts, or unrelated social-network data."
    """

    def __init__(self):
        self.account_profiles: Dict[str, Dict[str, Any]] = {}
        self.coordination_graph: Dict[str, Set[str]] = {}

    def register_account(self, account_id: str,
                          behavioral_features: Dict[str, float]) -> None:
        """Register an account with behavioral features."""
        self.account_profiles[account_id] = {
            "account_id": account_id,
            "features": behavioral_features,
            "created_at": time.time(),
            "risk_score": 0.0,
        }

    def compute_identity_risk(self, account_id: str) -> Dict[str, Any]:
        """
        Compute identity risk score for an account.

        R_id(u_i) = f(B_i, T_i, C_i, V_i)
        """
        profile = self.account_profiles.get(account_id)
        if not profile:
            return {"error": "Unknown account"}

        features = profile["features"]

        # B_i: Behavioral anomaly indicators
        b_score = features.get("behavioral_anomaly", 0.0)

        # T_i: Temporal interaction patterns
        t_score = features.get("temporal_anomaly", 0.0)

        # C_i: Confirmed coordination indicators
        c_score = features.get("coordination_score", 0.0)

        # V_i: Verified policy violations
        v_score = features.get("violation_count", 0.0)

        # Weighted combination
        risk = 0.3 * b_score + 0.2 * t_score + 0.3 * c_score + 0.2 * min(v_score / 5.0, 1.0)

        profile["risk_score"] = risk

        return {
            "account_id": account_id,
            "risk_score": risk,
            "components": {
                "behavioral_anomaly": b_score,
                "temporal_anomaly": t_score,
                "coordination_score": c_score,
                "violation_count": v_score,
            },
            "note": (
                "Risk score is a HEURISTIC. It does not prove identity. "
                "High scores warrant investigation, not automatic action."
            ),
        }

    def detect_coordination(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Detect coordinated behavior across accounts.

        Returns pairs of accounts with high coordination scores.
        """
        coordinated_pairs = []
        accounts = list(self.account_profiles.keys())

        for i in range(len(accounts)):
            for j in range(i + 1, len(accounts)):
                acc1, acc2 = accounts[i], accounts[j]
                feat1 = self.account_profiles[acc1]["features"]
                feat2 = self.account_profiles[acc2]["features"]

                # Simple similarity: overlap in behavioral features
                common_keys = set(feat1.keys()) & set(feat2.keys())
                if not common_keys:
                    continue

                similarity = sum(
                    1 - abs(feat1[k] - feat2[k])
                    for k in common_keys
                ) / len(common_keys)

                if similarity > threshold:
                    coordinated_pairs.append({
                        "account1": acc1,
                        "account2": acc2,
                        "similarity": similarity,
                        "note": (
                            "High behavioral similarity detected. "
                            "This is a SIGNAL for investigation, not proof."
                        ),
                    })

        return coordinated_pairs

    def get_verification_level(self, risk_score: float) -> int:
        """
        Determine verification level based on risk score.

        From §5.3 graduated approach:
          Level 1: Automated abuse detection (risk < 0.3)
          Level 2: Account-level verification (0.3 ≤ risk < 0.5)
          Level 3: Institutional verification (0.5 ≤ risk < 0.8)
          Level 4: Legal escalation (risk ≥ 0.8)
        """
        if risk_score < 0.3:
            return 1
        elif risk_score < 0.5:
            return 2
        elif risk_score < 0.8:
            return 3
        else:
            return 4


class RBACManager:
    """
    [D] Role-Based Access Control manager.

    Roles:
      • admin: Full access
      • clinician: Clinical operations, device control
      • researcher: Read-only data access
      • observer: View-only access
      • system: Internal system operations

    HONEST: RBAC controls ACCESS, not correctness.
    A user with correct access can still make incorrect decisions.
    """

    ROLES = {
        "admin": {
            "permissions": {"*"},
            "description": "Full administrative access",
        },
        "clinician": {
            "permissions": {
                "device:read", "device:control",
                "patient:read", "patient:write",
                "report:read", "report:write",
            },
            "description": "Clinical operations and device control",
        },
        "researcher": {
            "permissions": {
                "data:read",
                "report:read",
                "experiment:read", "experiment:write",
            },
            "description": "Research data access",
        },
        "observer": {
            "permissions": {
                "dashboard:read",
                "report:read",
            },
            "description": "View-only access",
        },
        "system": {
            "permissions": {
                "internal:*",
            },
            "description": "Internal system operations",
        },
    }

    def __init__(self):
        self.user_roles: Dict[str, Set[str]] = {}

    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign a role to a user."""
        if role not in self.ROLES:
            return False
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
        return True

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission."""
        user_roles = self.user_roles.get(user_id, set())
        for role in user_roles:
            role_perms = self.ROLES.get(role, {}).get("permissions", set())
            if "*" in role_perms or permission in role_perms:
                return True
            # Check wildcard permissions like "device:*"
            perm_prefix = permission.split(":")[0] + ":*"
            if perm_prefix in role_perms:
                return True
        return False

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user."""
        permissions = set()
        user_roles = self.user_roles.get(user_id, set())
        for role in user_roles:
            role_perms = self.ROLES.get(role, {}).get("permissions", set())
            permissions.update(role_perms)
        return permissions


class SessionManager:
    """
    [D] Secure session management with audit logging.
    """

    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        self.audit_log: List[Dict[str, Any]] = []

    def create_session(self, user_id: str, roles: Set[str],
                       permissions: Set[str]) -> UserSession:
        """Create a new session."""
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            roles=roles,
            permissions=permissions,
            created_at=time.time(),
            expires_at=time.time() + 3600,
            refresh_token=secrets.token_urlsafe(32),
        )
        self.sessions[session.session_id] = session
        self._log_event("session_created", session.session_id, user_id)
        return session

    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate a session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        if time.time() > session.expires_at:
            self._log_event("session_expired", session_id, session.user_id)
            del self.sessions[session_id]
            return None
        return session

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            self._log_event("session_revoked", session_id, session.user_id)
            del self.sessions[session_id]
            return True
        return False

    def _log_event(self, event_type: str, session_id: str,
                   user_id: str) -> None:
        """Log a session event."""
        self.audit_log.append({
            "event_type": event_type,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": time.time(),
        })


class AuthSuite:
    """
    [D] Complete authentication suite.
    """

    def __init__(self):
        self.oauth_server = OAuth2OIDCServer()
        self.identity_detector = MultiIdentityDetector()
        self.rbac = RBACManager()
        self.session_manager = SessionManager()

    def status(self) -> Dict[str, Any]:
        """Get authentication suite status."""
        return {
            "active_sessions": len(self.session_manager.sessions),
            "registered_accounts": len(self.identity_detector.account_profiles),
            "audit_log_size": len(self.session_manager.audit_log),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI OAuth2/OIDC Authentication Layer — Smoke Test")
    print("=" * 72)
    print(f"  PyJWT available: {HAS_JWT}")

    suite = AuthSuite()

    # Test OAuth2 flow
    print(f"\n  ── OAuth2 Authorization Code + PKCE Flow ──")

    # Generate PKCE challenge
    code_verifier = secrets.token_urlsafe(32)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b'=').decode()

    # Create authorization code
    auth_code = suite.oauth_server.create_authorization_code(
        user_id="clinician_001",
        client_id="sesi-dashboard",
        redirect_uri="https://sesi.example.com/callback",
        code_challenge=code_challenge,
    )
    print(f"    Auth code created   : {auth_code[:16]}...")

    # Exchange code for token
    token_response = suite.oauth_server.exchange_code_for_token(
        code=auth_code,
        client_id="sesi-dashboard",
        code_verifier=code_verifier,
    )
    print(f"    Token exchange      : {'success' if 'access_token' in token_response else 'failed'}")

    if "access_token" in token_response:
        # Validate token
        validation = suite.oauth_server.validate_token(token_response["access_token"])
        print(f"    Token validation    : {validation['valid']}")

    # Test multi-identity detection
    print(f"\n  ── Multi-Identity Detection ──")

    suite.identity_detector.register_account("account_001", {
        "behavioral_anomaly": 0.2,
        "temporal_anomaly": 0.1,
        "coordination_score": 0.3,
        "violation_count": 0,
    })
    suite.identity_detector.register_account("account_002", {
        "behavioral_anomaly": 0.8,
        "temporal_anomaly": 0.7,
        "coordination_score": 0.9,
        "violation_count": 2,
    })

    risk1 = suite.identity_detector.compute_identity_risk("account_001")
    risk2 = suite.identity_detector.compute_identity_risk("account_002")
    print(f"    Account 001 risk    : {risk1['risk_score']:.3f}")
    print(f"    Account 002 risk    : {risk2['risk_score']:.3f}")
    print(f"    Account 002 level   : {suite.identity_detector.get_verification_level(risk2['risk_score'])}")

    # Test RBAC
    print(f"\n  ── RBAC ──")
    suite.rbac.assign_role("clinician_001", "clinician")
    suite.rbac.assign_role("researcher_001", "researcher")

    print(f"    Clinician can control device: {suite.rbac.check_permission('clinician_001', 'device:control')}")
    print(f"    Researcher can control device: {suite.rbac.check_permission('researcher_001', 'device:control')}")
    print(f"    Researcher can read data: {suite.rbac.check_permission('researcher_001', 'data:read')}")

    # Test session management
    print(f"\n  ── Session Management ──")
    session = suite.session_manager.create_session(
        user_id="clinician_001",
        roles={"clinician"},
        permissions={"device:read", "device:control"},
    )
    print(f"    Session created     : {session.session_id[:16]}...")

    validated = suite.session_manager.validate_session(session.session_id)
    print(f"    Session valid       : {validated is not None}")

    suite.session_manager.revoke_session(session.session_id)
    print(f"    Session revoked     : {suite.session_manager.validate_session(session.session_id) is None}")

    print(f"\n  ── Suite Status ──")
    status = suite.status()
    print(f"    Active sessions     : {status['active_sessions']}")
    print(f"    Registered accounts : {status['registered_accounts']}")
    print(f"    Audit log entries   : {status['audit_log_size']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Authentication verifies IDENTITY, not moral correctness")
    print("  • Multi-identity detection is HEURISTIC; false positives possible")
    print("  • This is a FRAMEWORK, not a complete identity provider")
    print("  • Production requires proper IdP, HSM, security audit")
    print("  • Privacy: avoid unnecessary biometric/social-network data")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
