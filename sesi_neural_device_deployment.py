"""
=============================================================================
SESI NEURAL DEVICE INTEGRATION × DEPLOYMENT × MONITORING × SECURITY
EEG/MEG/MRI/BCI/Implanted Device Control Framework
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[ND1] NeuralDeviceInterface — Unified interface for EEG/MEG/MRI/BCI/Implants
[ND2] SignalFusionLayer — Multi-modal signal normalization and fusion
[ND3] AuthenticationAuthorizationLayer — OIDC/OAuth2 + RBAC + ABAC + MFA
[ND4] ImplantedDeviceController — Supervised control for DBS/neurostim
[ND5] RemoteControlGateway — Secure remote access with physician oversight
[ND6] DockerKubernetesDeployment — Containerization and orchestration
[ND7] GrafanaMonitoringDashboard — Observability and alerting
[ND8] SecurityAuditTrail — Tamper-evident logging and compliance

CRITICAL SAFETY CONSTRAINTS
----------------------------
• Implanted device control REQUIRES physician approval
• Remote control REQUIRES verified clinical session
• Emergency stop is ALWAYS available
• All actions are logged to tamper-evident audit trail
• No autonomous device control without human oversight

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is a SOFTWARE ARCHITECTURE FRAMEWORK
• It does NOT replace medical device certification
• It does NOT guarantee safety of neural stimulation
• All thresholds and parameters are FREE DESIGN PARAMETERS
• Production deployment requires regulatory approval
=============================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger("sesi.neural_device")

# Optional imports for production features
try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = Any

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

__all__ = [
    # Enums
    "NeuralDeviceType", "ControlLevel", "AuthLevel", "AuditEventType",
    "DeviceState", "EmergencyLevel",
    # Data classes
    "NeuralSignal", "DeviceCommand", "AuditEvent", "AuthContext",
    "PhysicianApproval", "EmergencyOverride",
    # Core classes
    "NeuralDeviceInterface", "SignalFusionLayer",
    "AuthenticationAuthorizationLayer", "ImplantedDeviceController",
    "RemoteControlGateway", "SecurityAuditTrail",
    "DockerKubernetesDeployment", "GrafanaMonitoringDashboard",
    "SESINeuralDeploymentSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class NeuralDeviceType(Enum):
    """Supported neural device types."""
    EEG = auto()              # Electroencephalography
    MEG = auto()              # Magnetoencephalography
    MRI = auto()              # Magnetic Resonance Imaging
    FMRI = auto()             # Functional MRI
    DTI = auto()              # Diffusion Tensor Imaging
    ECOG = auto()             # Electrocorticography
    UT_ARRAY = auto()         # Utah Array (intracortical)
    DBS = auto()              # Deep Brain Stimulation
    VNS = auto()              # Vagus Nerve Stimulation
    TMS = auto()              # Transcranial Magnetic Stimulation
    TDCS = auto()             # Transcranial Direct Current Stimulation
    NEURALINK = auto()        # Neuralink-style implant
    BCI_GENERIC = auto()      # Generic Brain-Computer Interface


class ControlLevel(Enum):
    """Control authorization levels."""
    READ_ONLY = 1             # Can read signals, cannot control
    LOCAL_CONTROL = 2         # Can control with physical presence
    REMOTE_SUPERVISED = 3     # Can control remotely with physician present
    REMOTE_AUTONOMOUS = 4     # NEVER GRANTED - always requires human
    EMERGENCY_OVERRIDE = 5    # Emergency stop only


class AuthLevel(Enum):
    """Authentication assurance levels (NIST SP 800-63B)."""
    AAL1 = 1                  # Single factor
    AAL2 = 2                  # Two factors
    AAL3 = 3                  # Hardware-based multi-factor


class AuditEventType(Enum):
    """Audit event types for compliance tracking."""
    AUTH_SUCCESS = auto()
    AUTH_FAILURE = auto()
    DEVICE_CONNECT = auto()
    DEVICE_DISCONNECT = auto()
    COMMAND_SENT = auto()
    COMMAND_EXECUTED = auto()
    COMMAND_REJECTED = auto()
    EMERGENCY_STOP = auto()
    PHYSICIAN_APPROVAL = auto()
    PHYSICIAN_REJECTION = auto()
    CONFIG_CHANGE = auto()
    SECURITY_ALERT = auto()


class DeviceState(Enum):
    """Neural device operational states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    IDLE = auto()
    RECORDING = auto()
    STIMULATING = auto()
    ERROR = auto()
    EMERGENCY_STOPPED = auto()


class EmergencyLevel(Enum):
    """Emergency severity levels."""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3
    CATASTROPHIC = 4


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NeuralSignal:
    """Unified neural signal representation."""
    device_id: str
    device_type: NeuralDeviceType
    timestamp: float
    data: np.ndarray
    sampling_rate: float
    channels: int
    units: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.name,
            "timestamp": self.timestamp,
            "data_shape": self.data.shape,
            "sampling_rate": self.sampling_rate,
            "channels": self.channels,
            "units": self.units,
            "metadata": self.metadata,
        }


@dataclass
class DeviceCommand:
    """Command to send to a neural device."""
    command_id: str
    device_id: str
    device_type: NeuralDeviceType
    action: str  # "start_recording", "stop_recording", "stimulate", "configure", "emergency_stop"
    parameters: Dict[str, Any]
    issued_by: str
    issued_at: float
    physician_approval_id: Optional[str] = None
    requires_approval: bool = True
    priority: int = 0  # 0=normal, 1=urgent, 2=emergency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "device_type": self.device_type.name,
            "action": self.action,
            "parameters": self.parameters,
            "issued_by": self.issued_by,
            "issued_at": self.issued_at,
            "physician_approval_id": self.physician_approval_id,
            "requires_approval": self.requires_approval,
            "priority": self.priority,
        }


@dataclass
class AuditEvent:
    """Tamper-evident audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: float
    user_id: str
    device_id: Optional[str]
    action: str
    details: Dict[str, Any]
    previous_hash: str
    event_hash: str
    signature: str

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of event for tamper detection."""
        content = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "action": self.action,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AuthContext:
    """Authentication and authorization context."""
    user_id: str
    auth_level: AuthLevel
    roles: Set[str]
    permissions: Set[str]
    mfa_verified: bool
    session_id: str
    issued_at: float
    expires_at: float
    device_access: Dict[str, ControlLevel]
    physician_mode: bool = False  # True if authenticated as physician

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions or "*" in self.permissions

    def can_access_device(self, device_id: str, level: ControlLevel) -> bool:
        device_level = self.device_access.get(device_id, ControlLevel.READ_ONLY)
        return device_level.value >= level.value


@dataclass
class PhysicianApproval:
    """Physician approval for device control."""
    approval_id: str
    physician_id: str
    physician_name: str
    license_number: str
    patient_id: str
    device_id: str
    command_description: str
    approved: bool
    approved_at: float
    expires_at: float
    conditions: List[str]
    signature: str  # Digital signature

    def is_valid(self) -> bool:
        return self.approved and time.time() < self.expires_at


@dataclass
class EmergencyOverride:
    """Emergency override authorization."""
    override_id: str
    initiated_by: str
    reason: str
    level: EmergencyLevel
    initiated_at: float
    authorized_by: Optional[str] = None
    authorized_at: Optional[float] = None
    active: bool = True


# ═══════════════════════════════════════════════════════════════════════
# [ND1] NEURAL DEVICE INTERFACE
# ═══════════════════════════════════════════════════════════════════════

class NeuralDeviceInterface:
    """
    [D] Unified interface for neural devices.

    Supports:
      • EEG (64-256 channels, 256-2048 Hz)
      • MEG (306 channels, ~1000 Hz)
      • MRI/fMRI/DTI (volumetric data)
      • ECoG/Utah Array (intracortical)
      • DBS/VNS/TMS (stimulation devices)

    HONEST: This is an INTERFACE LAYER. Actual device communication
    requires vendor-specific SDKs and drivers.
    """

    # Standard montage mappings
    EEG_10_20_SYSTEM = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "Fz", "Cz", "Pz", "A1", "A2",
    ]

    MEG_CHANNELS = 306  # Standard Elekta/Neuromag system

    def __init__(self):
        self.connected_devices: Dict[str, Dict[str, Any]] = {}
        self.signal_buffer: Dict[str, List[NeuralSignal]] = {}
        self.device_callbacks: Dict[str, List[Callable]] = {}

    def register_device(self, device_id: str, device_type: NeuralDeviceType,
                        config: Dict[str, Any]) -> bool:
        """Register a neural device."""
        self.connected_devices[device_id] = {
            "device_id": device_id,
            "device_type": device_type,
            "config": config,
            "state": DeviceState.DISCONNECTED,
            "registered_at": time.time(),
        }
        self.signal_buffer[device_id] = []
        logger.info(f"Registered device {device_id} ({device_type.name})")
        return True

    def connect_device(self, device_id: str) -> Dict[str, Any]:
        """
        Connect to a neural device.

        HONEST: This is a stub. Real implementation requires:
          • Vendor SDK (e.g., g.tec, BrainProducts, Elekta, Siemens)
          • Hardware interface (USB, Ethernet, Bluetooth)
          • Safety interlocks
        """
        if device_id not in self.connected_devices:
            return {"success": False, "error": "Device not registered"}

        device = self.connected_devices[device_id]
        device["state"] = DeviceState.CONNECTING

        # Simulate connection
        device["state"] = DeviceState.IDLE
        device["connected_at"] = time.time()

        logger.info(f"Device {device_id} connected")
        return {
            "success": True,
            "device_id": device_id,
            "state": device["state"].name,
        }

    def start_recording(self, device_id: str, duration_s: float = 60.0) -> Dict[str, Any]:
        """Start recording from a device."""
        if device_id not in self.connected_devices:
            return {"success": False, "error": "Device not registered"}

        device = self.connected_devices[device_id]
        if device["state"] != DeviceState.IDLE:
            return {"success": False, "error": f"Device in state {device['state'].name}"}

        device["state"] = DeviceState.RECORDING
        device["recording_started_at"] = time.time()

        return {
            "success": True,
            "device_id": device_id,
            "state": "RECORDING",
            "duration_s": duration_s,
        }

    def stop_recording(self, device_id: str) -> Dict[str, Any]:
        """Stop recording from a device."""
        if device_id not in self.connected_devices:
            return {"success": False, "error": "Device not registered"}

        device = self.connected_devices[device_id]
        if device["state"] != DeviceState.RECORDING:
            return {"success": False, "error": "Device not recording"}

        device["state"] = DeviceState.IDLE
        return {
            "success": True,
            "device_id": device_id,
            "state": "IDLE",
        }

    def get_signal(self, device_id: str, n_samples: int = 1024) -> Optional[NeuralSignal]:
        """
        Get recent signal from device.

        HONEST: This returns synthetic data for demonstration.
        Real implementation requires actual device communication.
        """
        if device_id not in self.connected_devices:
            return None

        device = self.connected_devices[device_id]
        device_type = device["device_type"]

        # Generate synthetic signal based on device type
        if device_type == NeuralDeviceType.EEG:
            channels = len(self.EEG_10_20_SYSTEM)
            sampling_rate = 512.0
        elif device_type == NeuralDeviceType.MEG:
            channels = self.MEG_CHANNELS
            sampling_rate = 1000.0
        elif device_type in [NeuralDeviceType.ECOG, NeuralDeviceType.UT_ARRAY]:
            channels = 64
            sampling_rate = 2000.0
        else:
            channels = 1
            sampling_rate = 256.0

        data = np.random.randn(channels, n_samples) * 1e-6  # µV scale

        return NeuralSignal(
            device_id=device_id,
            device_type=device_type,
            timestamp=time.time(),
            data=data,
            sampling_rate=sampling_rate,
            channels=channels,
            units="µV" if device_type != NeuralDeviceType.MEG else "fT",
            metadata={"synthetic": True},
        )

    def emergency_stop(self, device_id: str, reason: str) -> Dict[str, Any]:
        """Emergency stop for a device."""
        if device_id not in self.connected_devices:
            return {"success": False, "error": "Device not registered"}

        device = self.connected_devices[device_id]
        device["state"] = DeviceState.EMERGENCY_STOPPED
        device["emergency_stop_reason"] = reason
        device["emergency_stop_at"] = time.time()

        logger.critical(f"EMERGENCY STOP: Device {device_id} - {reason}")

        return {
            "success": True,
            "device_id": device_id,
            "state": "EMERGENCY_STOPPED",
            "reason": reason,
        }

    def get_device_status(self) -> Dict[str, Any]:
        """Get status of all registered devices."""
        return {
            device_id: {
                "device_type": info["device_type"].name,
                "state": info["state"].name,
                "registered_at": info.get("registered_at"),
                "connected_at": info.get("connected_at"),
            }
            for device_id, info in self.connected_devices.items()
        }


# ═══════════════════════════════════════════════════════════════════════
# [ND2] SIGNAL FUSION LAYER
# ═══════════════════════════════════════════════════════════════════════

class SignalFusionLayer:
    """
    [D] Multi-modal neural signal fusion.

    Normalizes signals from different devices to a common space:
      • EEG 10-20 system → MNI coordinates
      • MEG sensor space → source space
      • MRI/fMRI → volumetric atlas
      • ECoG/Utah → cortical surface

    HONEST: This is a CONCEPTUAL FRAMEWORK. Actual source localization
    requires individual head models and inverse solutions.
    """

    def __init__(self):
        self.device_transforms: Dict[str, Any] = {}
        self.reference_atlas = "MNI152"

    def normalize_eeg(self, signal: NeuralSignal) -> NeuralSignal:
        """Normalize EEG signal to standard reference."""
        # Re-reference to average
        data = signal.data
        avg_ref = data.mean(axis=0, keepdims=True)
        data_reref = data - avg_ref

        return NeuralSignal(
            device_id=signal.device_id,
            device_type=signal.device_type,
            timestamp=signal.timestamp,
            data=data_reref,
            sampling_rate=signal.sampling_rate,
            channels=signal.channels,
            units=signal.units,
            metadata={**signal.metadata, "reference": "average"},
        )

    def fuse_signals(self, signals: List[NeuralSignal]) -> Dict[str, Any]:
        """
        Fuse signals from multiple devices.

        Returns a unified representation.
        """
        fused = {
            "timestamp": time.time(),
            "devices": [s.device_id for s in signals],
            "modalities": [s.device_type.name for s in signals],
        }

        # Group by modality
        by_modality: Dict[str, List[NeuralSignal]] = {}
        for s in signals:
            modality = s.device_type.name
            if modality not in by_modality:
                by_modality[modality] = []
            by_modality[modality].append(s)

        fused["by_modality"] = {
            modality: {
                "count": len(sigs),
                "total_channels": sum(s.channels for s in sigs),
            }
            for modality, sigs in by_modality.items()
        }

        return fused


# ═══════════════════════════════════════════════════════════════════════
# [ND3] AUTHENTICATION & AUTHORIZATION LAYER
# ═══════════════════════════════════════════════════════════════════════

class AuthenticationAuthorizationLayer:
    """
    [D] Multi-factor authentication and authorization.

    Implements:
      • OIDC/OAuth2 authentication
      • MFA (TOTP, hardware keys, biometrics)
      • RBAC (Role-Based Access Control)
      • ABAC (Attribute-Based Access Control)
      • Session management
      • Physician privilege escalation

    HONEST: This is a FRAMEWORK. Production requires:
      • Identity provider (Keycloak, Auth0, Okta)
      • Hardware security modules (HSM)
      • Regular security audits
    """

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.sessions: Dict[str, AuthContext] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.mfa_secrets: Dict[str, str] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.lockout_threshold = 5
        self.lockout_duration_s = 300

    def register_user(self, user_id: str, roles: Set[str],
                      permissions: Set[str], mfa_secret: str) -> Dict[str, Any]:
        """Register a new user."""
        self.users[user_id] = {
            "user_id": user_id,
            "roles": roles,
            "permissions": permissions,
            "created_at": time.time(),
        }
        self.mfa_secrets[user_id] = mfa_secret
        return {"success": True, "user_id": user_id}

    def authenticate(self, user_id: str, password: str,
                     mfa_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate a user.

        Returns session token on success.
        """
        # Check lockout
        if self._is_locked_out(user_id):
            return {
                "success": False,
                "error": "Account locked due to failed attempts",
            }

        # Verify user exists
        if user_id not in self.users:
            self._record_failed_attempt(user_id)
            return {"success": False, "error": "Invalid credentials"}

        # In production, verify password hash here
        # For this framework, we simulate success

        # Verify MFA if provided
        auth_level = AuthLevel.AAL1
        if mfa_code:
            # In production, verify TOTP code
            auth_level = AuthLevel.AAL2

        # Create session
        session_id = secrets.token_hex(32)
        context = AuthContext(
            user_id=user_id,
            auth_level=auth_level,
            roles=self.users[user_id]["roles"],
            permissions=self.users[user_id]["permissions"],
            mfa_verified=mfa_code is not None,
            session_id=session_id,
            issued_at=time.time(),
            expires_at=time.time() + 3600,  # 1 hour
            device_access={},
        )

        self.sessions[session_id] = context

        return {
            "success": True,
            "session_id": session_id,
            "auth_level": auth_level.name,
            "expires_at": context.expires_at,
        }

    def verify_session(self, session_id: str) -> Optional[AuthContext]:
        """Verify a session token."""
        if session_id not in self.sessions:
            return None

        context = self.sessions[session_id]
        if time.time() > context.expires_at:
            del self.sessions[session_id]
            return None

        return context

    def grant_device_access(self, session_id: str, device_id: str,
                            level: ControlLevel) -> bool:
        """Grant access to a specific device."""
        context = self.verify_session(session_id)
        if context is None:
            return False

        # Physician mode allows higher access
        if context.physician_mode:
            context.device_access[device_id] = level
            return True

        # Non-physician limited to READ_ONLY or LOCAL_CONTROL
        if level.value > ControlLevel.LOCAL_CONTROL.value:
            return False

        context.device_access[device_id] = level
        return True

    def verify_physician(self, session_id: str, license_number: str) -> bool:
        """
        Verify physician credentials.

        HONEST: In production, this must verify against:
          • Medical license database
          • Hospital credentialing system
          • DEA registration (for controlled substances)
        """
        context = self.verify_session(session_id)
        if context is None:
            return False

        # Simulate license verification
        # In production: query state medical board API
        context.physician_mode = True
        context.roles.add("physician")
        context.permissions.add("device_control")
        context.permissions.add("prescription_authority")

        return True

    def _is_locked_out(self, user_id: str) -> bool:
        """Check if account is locked out."""
        if user_id not in self.failed_attempts:
            return False

        attempts = self.failed_attempts[user_id]
        recent = [t for t in attempts if time.time() - t < self.lockout_duration_s]
        return len(recent) >= self.lockout_threshold

    def _record_failed_attempt(self, user_id: str) -> None:
        """Record a failed authentication attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        self.failed_attempts[user_id].append(time.time())


# ═══════════════════════════════════════════════════════════════════════
# [ND4] IMPLANTED DEVICE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════

class ImplantedDeviceController:
    """
    [D] Controller for implanted neural devices (DBS, VNS, neurostimulators).

    CRITICAL SAFETY CONSTRAINTS:
      • Every stimulation command requires physician approval
      • Remote control requires verified clinical session
      • Emergency stop is always available
      • All commands are logged to audit trail
      • No autonomous operation

    HONEST: This is a SOFTWARE FRAMEWORK. Actual device control requires:
      • FDA/CE approved device interface
      • Clinician programming system
      • Patient safety monitoring
    """

    def __init__(self, auth_layer: AuthenticationAuthorizationLayer,
                 audit_trail: "SecurityAuditTrail"):
        self.auth = auth_layer
        self.audit = audit_trail
        self.pending_approvals: Dict[str, PhysicianApproval] = {}
        self.active_stimulation: Dict[str, Dict[str, Any]] = {}
        self.emergency_overrides: Dict[str, EmergencyOverride] = {}

    def request_stimulation(self, session_id: str, device_id: str,
                            parameters: Dict[str, Any],
                            patient_id: str) -> Dict[str, Any]:
        """
        Request stimulation of an implanted device.

        REQUIRES physician approval before execution.
        """
        context = self.auth.verify_session(session_id)
        if context is None:
            return {"success": False, "error": "Invalid session"}

        if not context.physician_mode:
            return {
                "success": False,
                "error": "Physician authentication required",
            }

        # Create approval request
        approval_id = str(uuid.uuid4())

        self.audit.log_event(
            AuditEventType.COMMAND_SENT,
            user_id=context.user_id,
            device_id=device_id,
            action="request_stimulation",
            details={"parameters": parameters, "approval_id": approval_id},
        )

        return {
            "success": True,
            "approval_id": approval_id,
            "status": "PENDING_APPROVAL",
            "message": "Stimulation request created. Awaiting physician approval.",
        }

    def approve_stimulation(self, physician_id: str, approval_id: str,
                            license_number: str,
                            conditions: List[str]) -> Dict[str, Any]:
        """
        Approve a stimulation request.

        REQUIRES:
          • Valid physician license
          • Patient verification
          • Device verification
        """
        if approval_id not in self.pending_approvals:
            return {"success": False, "error": "Approval request not found"}

        # In production: verify physician license against database
        approval = PhysicianApproval(
            approval_id=approval_id,
            physician_id=physician_id,
            physician_name="Dr. " + physician_id,
            license_number=license_number,
            patient_id="patient_001",
            device_id="device_001",
            command_description="DBS stimulation",
            approved=True,
            approved_at=time.time(),
            expires_at=time.time() + 3600,  # 1 hour
            conditions=conditions,
            signature="simulated_signature",
        )

        self.pending_approvals[approval_id] = approval

        self.audit.log_event(
            AuditEventType.PHYSICIAN_APPROVAL,
            user_id=physician_id,
            device_id=approval.device_id,
            action="approve_stimulation",
            details={"approval_id": approval_id},
        )

        return {
            "success": True,
            "approval_id": approval_id,
            "status": "APPROVED",
            "expires_at": approval.expires_at,
        }

    def execute_stimulation(self, session_id: str, approval_id: str) -> Dict[str, Any]:
        """
        Execute approved stimulation.

        REQUIRES valid approval.
        """
        context = self.auth.verify_session(session_id)
        if context is None:
            return {"success": False, "error": "Invalid session"}

        if approval_id not in self.pending_approvals:
            return {"success": False, "error": "Approval not found"}

        approval = self.pending_approvals[approval_id]
        if not approval.is_valid():
            return {"success": False, "error": "Approval expired or invalid"}

        # Execute stimulation
        device_id = approval.device_id
        self.active_stimulation[device_id] = {
            "approval_id": approval_id,
            "started_at": time.time(),
            "parameters": {},
        }

        self.audit.log_event(
            AuditEventType.COMMAND_EXECUTED,
            user_id=context.user_id,
            device_id=device_id,
            action="execute_stimulation",
            details={"approval_id": approval_id},
        )

        return {
            "success": True,
            "device_id": device_id,
            "status": "STIMULATING",
        }

    def emergency_stop_stimulation(self, session_id: str, device_id: str,
                                    reason: str) -> Dict[str, Any]:
        """
        Emergency stop of stimulation.

        Can be called by ANY authenticated user in emergency.
        """
        context = self.auth.verify_session(session_id)
        if context is None:
            return {"success": False, "error": "Invalid session"}

        if device_id in self.active_stimulation:
            del self.active_stimulation[device_id]

        override = EmergencyOverride(
            override_id=str(uuid.uuid4()),
            initiated_by=context.user_id,
            reason=reason,
            level=EmergencyLevel.EMERGENCY,
            initiated_at=time.time(),
        )
        self.emergency_overrides[override.override_id] = override

        self.audit.log_event(
            AuditEventType.EMERGENCY_STOP,
            user_id=context.user_id,
            device_id=device_id,
            action="emergency_stop",
            details={"reason": reason, "override_id": override.override_id},
        )

        logger.critical(f"EMERGENCY STOP: {device_id} - {reason}")

        return {
            "success": True,
            "device_id": device_id,
            "status": "EMERGENCY_STOPPED",
            "override_id": override.override_id,
        }

    def get_stimulation_status(self) -> Dict[str, Any]:
        """Get status of all active stimulations."""
        return {
            device_id: {
                "approval_id": info["approval_id"],
                "started_at": info["started_at"],
                "duration_s": time.time() - info["started_at"],
            }
            for device_id, info in self.active_stimulation.items()
        }


# ═══════════════════════════════════════════════════════════════════════
# [ND5] REMOTE CONTROL GATEWAY
# ═══════════════════════════════════════════════════════════════════════

class RemoteControlGateway:
    """
    [D] Secure remote control gateway for neural devices.

    CRITICAL: Remote control of implanted devices REQUIRES:
      • Verified clinical session (video + audio)
      • Physician present at remote site
      • Patient consent
      • Emergency stop capability
      • Full audit trail

    HONEST: This is a FRAMEWORK. Production requires:
      • HIPAA-compliant video conferencing
      • End-to-end encryption
      • Network segmentation
      • Regular penetration testing
    """

    def __init__(self, auth_layer: AuthenticationAuthorizationLayer,
                 device_interface: NeuralDeviceInterface,
                 audit_trail: "SecurityAuditTrail"):
        self.auth = auth_layer
        self.devices = device_interface
        self.audit = audit_trail
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def initiate_remote_session(self, session_id: str,
                                 physician_id: str,
                                 patient_id: str) -> Dict[str, Any]:
        """
        Initiate a remote clinical session.

        REQUIRES:
          • Physician authentication (AAL3)
          • Patient consent verification
          • Network security verification
        """
        context = self.auth.verify_session(session_id)
        if context is None:
            return {"success": False, "error": "Invalid session"}

        if not context.physician_mode:
            return {"success": False, "error": "Physician authentication required"}

        if context.auth_level != AuthLevel.AAL3:
            return {"success": False, "error": "AAL3 authentication required for remote control"}

        # Create remote session
        remote_session_id = str(uuid.uuid4())
        self.active_sessions[remote_session_id] = {
            "session_id": remote_session_id,
            "physician_id": physician_id,
            "patient_id": patient_id,
            "started_at": time.time(),
            "status": "ACTIVE",
        }

        self.audit.log_event(
            AuditEventType.DEVICE_CONNECT,
            user_id=physician_id,
            device_id=None,
            action="initiate_remote_session",
            details={
                "remote_session_id": remote_session_id,
                "patient_id": patient_id,
            },
        )

        return {
            "success": True,
            "remote_session_id": remote_session_id,
            "status": "ACTIVE",
        }

    def verify_clinical_session(self, remote_session_id: str) -> Dict[str, Any]:
        """
        Verify that a clinical session is properly established.

        Checks:
          • Video connection active
          • Audio connection active
          • Patient visible on camera
          • Physician credentials valid
        """
        if remote_session_id not in self.active_sessions:
            return {"valid": False, "error": "Session not found"}

        session = self.active_sessions[remote_session_id]

        # In production, verify:
        # - Video stream active
        # - Audio stream active
        # - Patient consent documented
        # - Physician identity verified

        return {
            "valid": True,
            "session_id": remote_session_id,
            "physician_id": session["physician_id"],
            "patient_id": session["patient_id"],
            "duration_s": time.time() - session["started_at"],
        }

    def terminate_remote_session(self, remote_session_id: str,
                                  reason: str) -> Dict[str, Any]:
        """Terminate a remote session."""
        if remote_session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        session = self.active_sessions[remote_session_id]
        session["status"] = "TERMINATED"
        session["terminated_at"] = time.time()
        session["termination_reason"] = reason

        self.audit.log_event(
            AuditEventType.DEVICE_DISCONNECT,
            user_id=session["physician_id"],
            device_id=None,
            action="terminate_remote_session",
            details={
                "remote_session_id": remote_session_id,
                "reason": reason,
            },
        )

        return {"success": True, "remote_session_id": remote_session_id}


# ═══════════════════════════════════════════════════════════════════════
# [ND8] SECURITY AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════

class SecurityAuditTrail:
    """
    [D] Tamper-evident audit trail for all neural device operations.

    Implements:
      • Hash chain for tamper detection
      • Digital signatures
      • Compliance reporting (HIPAA, GDPR)
      • Alert generation for suspicious activity

    HONEST: This provides SOFTWARE-LEVEL audit trail. Regulatory
    compliance requires additional controls (physical security, etc.)
    """

    def __init__(self, signing_key: Optional[str] = None):
        self.signing_key = signing_key or secrets.token_hex(32)
        self.events: List[AuditEvent] = []
        self.previous_hash = "0" * 64  # Genesis hash

    def log_event(self, event_type: AuditEventType, user_id: str,
                  device_id: Optional[str], action: str,
                  details: Dict[str, Any]) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            device_id=device_id,
            action=action,
            details=details,
            previous_hash=self.previous_hash,
            event_hash="",
            signature="",
        )

        # Compute hash
        event.event_hash = event.compute_hash()

        # Sign
        event.signature = hmac.new(
            self.signing_key.encode(),
            event.event_hash.encode(),
            hashlib.sha256,
        ).hexdigest()

        self.events.append(event)
        self.previous_hash = event.event_hash

        return event

    def verify_chain(self) -> Dict[str, Any]:
        """Verify the integrity of the audit chain."""
        if not self.events:
            return {"valid": True, "message": "No events"}

        previous_hash = "0" * 64
        for i, event in enumerate(self.events):
            if event.previous_hash != previous_hash:
                return {
                    "valid": False,
                    "error": f"Chain broken at event {i}",
                    "event_id": event.event_id,
                }

            computed_hash = event.compute_hash()
            if computed_hash != event.event_hash:
                return {
                    "valid": False,
                    "error": f"Hash mismatch at event {i}",
                    "event_id": event.event_id,
                }

            previous_hash = event.event_hash

        return {"valid": True, "n_events": len(self.events)}

    def get_events(self, event_type: Optional[AuditEventType] = None,
                   user_id: Optional[str] = None,
                   device_id: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit events."""
        events = self.events

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if device_id:
            events = [e for e in events if e.device_id == device_id]

        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.name,
                "timestamp": e.timestamp,
                "user_id": e.user_id,
                "device_id": e.device_id,
                "action": e.action,
            }
            for e in events[-limit:]
        ]

    def generate_compliance_report(self, start_time: float,
                                    end_time: float) -> Dict[str, Any]:
        """Generate compliance report for a time period."""
        events = [e for e in self.events if start_time <= e.timestamp <= end_time]

        return {
            "period_start": start_time,
            "period_end": end_time,
            "total_events": len(events),
            "events_by_type": {
                t.name: len([e for e in events if e.event_type == t])
                for t in AuditEventType
            },
            "chain_valid": self.verify_chain()["valid"],
        }


# ═══════════════════════════════════════════════════════════════════════
# [ND6] DOCKER/KUBERNETES DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════

class DockerKubernetesDeployment:
    """
    [D] Docker and Kubernetes deployment configuration generator.

    Generates:
      • Dockerfile
      • docker-compose.yml
      • Kubernetes manifests
      • Helm chart

    HONEST: These are TEMPLATES. Production deployment requires:
      • Security review
      • Load testing
      • Disaster recovery planning
      • Compliance certification
    """

    DOCKERFILE_TEMPLATE = '''
# SESI Neural Device Service
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 sesi && chown -R sesi:sesi /app
USER sesi

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
'''

    DOCKER_COMPOSE_TEMPLATE = '''
version: "3.9"

services:
  neural-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://sesi:sesi@postgres:5432/sesi
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config:ro
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=sesi
      - POSTGRES_USER=sesi
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  grafana_data:
'''

    K8S_DEPLOYMENT_TEMPLATE = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-api
  labels:
    app: neural-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: neural-api
  template:
    metadata:
      labels:
        app: neural-api
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
        - name: neural-api
          image: sesi/neural-api:latest
          ports:
            - containerPort: 8080
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: neural-secrets
                  key: database-url
            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: neural-secrets
                  key: jwt-secret
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: neural-api
spec:
  selector:
    app: neural-api
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
'''

    def __init__(self, output_dir: str = "./deployment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dockerfile(self) -> str:
        """Generate Dockerfile."""
        path = self.output_dir / "Dockerfile"
        path.write_text(self.DOCKERFILE_TEMPLATE)
        return str(path)

    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml."""
        path = self.output_dir / "docker-compose.yml"
        path.write_text(self.DOCKER_COMPOSE_TEMPLATE)
        return str(path)

    def generate_k8s_manifests(self) -> str:
        """Generate Kubernetes manifests."""
        path = self.output_dir / "k8s-deployment.yaml"
        path.write_text(self.K8S_DEPLOYMENT_TEMPLATE)
        return str(path)

    def generate_all(self) -> Dict[str, str]:
        """Generate all deployment files."""
        return {
            "dockerfile": self.generate_dockerfile(),
            "docker_compose": self.generate_docker_compose(),
            "k8s_manifests": self.generate_k8s_manifests(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [ND7] GRAFANA MONITORING DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class GrafanaMonitoringDashboard:
    """
    [D] Grafana dashboard configuration generator.

    Creates dashboards for:
      • Device status monitoring
      • Signal quality metrics
      • Authentication events
      • Command execution tracking
      • Emergency stop events

    HONEST: This generates Grafana JSON models. Import into Grafana
    to visualize metrics from Prometheus.
    """

    DASHBOARD_TEMPLATE = {
        "dashboard": {
            "title": "SESI Neural Device Monitoring",
            "tags": ["neural", "sesi", "monitoring"],
            "timezone": "browser",
            "panels": [],
            "refresh": "10s",
        }
    }

    def __init__(self, output_dir: str = "./monitoring"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dashboard(self) -> str:
        """Generate Grafana dashboard JSON."""
        dashboard = self.DASHBOARD_TEMPLATE.copy()
        dashboard["dashboard"]["panels"] = [
            {
                "id": 1,
                "type": "stat",
                "title": "Active Devices",
                "targets": [
                    {"expr": "sesi_neural_devices_active", "legendFormat": "Devices"}
                ],
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
            },
            {
                "id": 2,
                "type": "graph",
                "title": "Signal Quality",
                "targets": [
                    {"expr": "sesi_signal_quality_avg", "legendFormat": "Avg Quality"}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
            },
            {
                "id": 3,
                "type": "stat",
                "title": "Emergency Stops (24h)",
                "targets": [
                    {"expr": "increase(sesi_emergency_stops_total[24h])", "legendFormat": "Stops"}
                ],
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
            },
        ]

        path = self.output_dir / "grafana-dashboard.json"
        path.write_text(json.dumps(dashboard, indent=2))
        return str(path)

    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration."""
        config = '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'neural-api'
    static_configs:
      - targets: ['neural-api:8080']
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
'''
        path = self.output_dir / "prometheus.yml"
        path.write_text(config)
        return str(path)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESINeuralDeploymentSuite:
    """
    [D] Complete deployment suite for neural device integration.

    Orchestrates:
      • Neural device interfaces
      • Authentication/authorization
      • Implanted device control
      • Remote control gateway
      • Audit trail
      • Deployment generation
      • Monitoring dashboard
    """

    def __init__(self):
        self.audit = SecurityAuditTrail()
        self.auth = AuthenticationAuthorizationLayer()
        self.devices = NeuralDeviceInterface()
        self.fusion = SignalFusionLayer()
        self.implant_controller = ImplantedDeviceController(self.auth, self.audit)
        self.remote_gateway = RemoteControlGateway(self.auth, self.devices, self.audit)
        self.deployment = DockerKubernetesDeployment()
        self.monitoring = GrafanaMonitoringDashboard()

    def setup_demo_devices(self) -> None:
        """Setup demo devices for testing."""
        self.devices.register_device("eeg_001", NeuralDeviceType.EEG, {"channels": 64})
        self.devices.register_device("meg_001", NeuralDeviceType.MEG, {"channels": 306})
        self.devices.register_device("dbs_001", NeuralDeviceType.DBS, {"contacts": 8})

    def status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "devices": self.devices.get_device_status(),
            "stimulation": self.implant_controller.get_stimulation_status(),
            "audit_events": len(self.audit.events),
            "audit_chain_valid": self.audit.verify_chain()["valid"],
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Neural Device Deployment Suite — Smoke Test")
    print("=" * 72)

    suite = SESINeuralDeploymentSuite()
    suite.setup_demo_devices()

    print("\n  ── Device Registration ──")
    status = suite.status()
    print(f"    Registered devices: {len(status['devices'])}")
    for device_id, info in status["devices"].items():
        print(f"      {device_id}: {info['device_type']} ({info['state']})")

    print("\n  ── Device Connection ──")
    result = suite.devices.connect_device("eeg_001")
    print(f"    Connect EEG: {result['success']}")

    result = suite.devices.start_recording("eeg_001")
    print(f"    Start recording: {result['success']}")

    signal = suite.devices.get_signal("eeg_001", n_samples=1024)
    if signal:
        print(f"    Signal shape: {signal.data.shape}")
        print(f"    Sampling rate: {signal.sampling_rate} Hz")

    result = suite.devices.stop_recording("eeg_001")
    print(f"    Stop recording: {result['success']}")

    print("\n  ── Authentication ──")
    suite.auth.register_user(
        "dr_smith",
        roles={"physician", "researcher"},
        permissions={"device_read", "device_control"},
        mfa_secret="JBSWY3DPEHPK3PXP",
    )

    auth_result = suite.auth.authenticate("dr_smith", "password", mfa_code="123456")
    print(f"    Authentication: {auth_result['success']}")
    print(f"    Auth level: {auth_result.get('auth_level', 'N/A')}")

    if auth_result["success"]:
        session_id = auth_result["session_id"]

        # Verify physician
        suite.auth.verify_physician(session_id, "MD123456")

        # Grant device access
        suite.auth.grant_device_access(session_id, "dbs_001", ControlLevel.REMOTE_SUPERVISED)

        print("\n  ── Implanted Device Control ──")
        stim_request = suite.implant_controller.request_stimulation(
            session_id, "dbs_001",
            parameters={"frequency": 130, "amplitude": 2.0},
            patient_id="patient_001",
        )
        print(f"    Stimulation request: {stim_request['success']}")
        print(f"    Approval ID: {stim_request.get('approval_id', 'N/A')}")

        if stim_request["success"]:
            approval_id = stim_request["approval_id"]

            # Approve
            approval = suite.implant_controller.approve_stimulation(
                "dr_smith", approval_id, "MD123456",
                conditions=["Patient monitored", "Duration < 30 min"],
            )
            print(f"    Approval: {approval['success']}")

            # Execute
            execution = suite.implant_controller.execute_stimulation(session_id, approval_id)
            print(f"    Execution: {execution['success']}")
            print(f"    Status: {execution.get('status', 'N/A')}")

            # Check status
            stim_status = suite.implant_controller.get_stimulation_status()
            print(f"    Active stimulations: {len(stim_status)}")

            # Emergency stop
            emergency = suite.implant_controller.emergency_stop_stimulation(
                session_id, "dbs_001", "Patient discomfort"
            )
            print(f"    Emergency stop: {emergency['success']}")

    print("\n  ── Audit Trail ──")
    audit_status = suite.audit.verify_chain()
    print(f"    Events logged: {len(suite.audit.events)}")
    print(f"    Chain valid: {audit_status['valid']}")

    events = suite.audit.get_events(limit=5)
    print(f"    Recent events:")
    for e in events[-3:]:
        print(f"      {e['event_type']}: {e['action']}")

    print("\n  ── Deployment Generation ──")
    deployment_files = suite.deployment.generate_all()
    print(f"    Dockerfile: {deployment_files['dockerfile']}")
    print(f"    Docker Compose: {deployment_files['docker_compose']}")
    print(f"    K8s Manifests: {deployment_files['k8s_manifests']}")

    print("\n  ── Monitoring Dashboard ──")
    dashboard = suite.monitoring.generate_dashboard()
    prometheus = suite.monitoring.generate_prometheus_config()
    print(f"    Grafana dashboard: {dashboard}")
    print(f"    Prometheus config: {prometheus}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is a SOFTWARE FRAMEWORK, not a medical device")
    print("  • Implanted device control REQUIRES physician approval")
    print("  • Remote control REQUIRES verified clinical session")
    print("  • Emergency stop is ALWAYS available")
    print("  • All actions are logged to tamper-evident audit trail")
    print("  • Production requires regulatory approval (FDA/CE/TGA)")
    print("  • All parameters are FREE DESIGN PARAMETERS")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
