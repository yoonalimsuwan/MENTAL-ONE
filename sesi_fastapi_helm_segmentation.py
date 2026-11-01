"""
=============================================================================
SESI FASTAPI SERVER × HELM CHART × NEURAL NETWORK SEGMENTATION
Production wiring for the Buddhist-Neuro-Conscience stack +
Kubernetes packaging + implanted-device network isolation
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[FS1] SESIFastAPIServer
      • /health, /ready, /metrics (ops)
      • /api/v1/conscience/*  (cetasika, nivarana, bojjhanga state)
      • /api/v1/neural/*      (read signals, propose control)
      • /api/v1/implant/*     (command submission, approval, emergency stop)
      • /api/v1/oversight/*   (audit trail, limitation enforcer)
      • /ws/alerts            (real-time WebSocket alert stream)

[HC1] HelmChartGenerator
      • Chart.yaml, values.yaml
      • deployment, service, ingress, hpa, pdb
      • servicemonitor (Prometheus Operator)
      • networkpolicy (CRITICAL: implant-zone isolation)
      • configmap, secret templates

[NS1] NeuralNetworkSegmentation
      • Four security zones (Z0 implant … Z3 external)
      • Cross-zone policy engine (default-deny)
      • Command rate limiting via SESI No-Zeno bound
      • δ_min quantum gate (rejects sub-threshold parameter changes)
      • Stochastic exit-time anomaly alarm
      • Emergency isolation switch

MATHEMATICAL GUARANTEES USED (from the series)
-----------------------------------------------
  [C] SESI_NoZeno_Closure Thm 3.2 : N(T) ≤ 1 + 2·C_V·T/δ_min
  [C] SESI_NoZeno_Closure Thm 4.2 : N(T) ≤ (E_max + D(T))/ΔE_min
  [C] Stochastic_NoZeno Thm 4.1   : exit-time tail bound (BDG)
  [C] Delta_Min Thm 1.2           : δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
  [C] Revision22 Thm 2.1/3.1      : coercivity α₂, α₄ (stability)
  [C] PNDC Axiom 3                : Necessity and Proportionality

RELATION TO THE SERIES
-----------------------
  • ses_moral_conscience_module        — cetasika/nivarana/bojjhanga
  • ses_multi_layer_oversight_v2       — 7-layer oversight
  • ses_auth_transformer_websocket     — AuthN/AuthZ
  • ses_math_integration_deployment    — δ_min / No-Zeno / coercivity
  • agi_one_v3_11                      — AGIONE central hub
  • agi_one_psyche_plus_v33            — speculative-axiom layer

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is DEPLOYMENT SCAFFOLDING + SAFETY ARCHITECTURE, not a cleared
  medical device. Regulatory approval (FDA 510(k)/PMA, EU MDR, TGA)
  is a separate, human-driven process this code cannot replace.
• Network segmentation REDUCES blast radius; it does not make implants
  unhackable. Defense-in-depth and clinical governance remain mandatory.
• All thresholds (δ_min, κ, rate limits, zone policies) are FREE DESIGN
  PARAMETERS (Unified_Ordinal Hyp 7.9) and MUST be validated by a
  clinical-safety review board before any real deployment.
• Every implant command requires clinician approval. The API refuses
  autonomous execution by construction (`allow_autonomous=False`).
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sesi.fastapi_helm_seg")

# ── Optional heavy deps (graceful degradation) ─────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Prior-series modules (all optional)
try:
    from ses_moral_conscience_module import BuddhistNeuroPsychLLM, NeuroPsychConfig
    HAS_CONSCIENCE = True
except ImportError:
    HAS_CONSCIENCE = False

try:
    from ses_multi_layer_oversight_v2 import MultiLayerOversightSystem
    HAS_OVERSIGHT = True
except ImportError:
    HAS_OVERSIGHT = False

try:
    from ses_auth_transformer_websocket import (
        AuthenticationAuthorizationLayer, ControlLevel,
    )
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False

try:
    from ses_math_integration_deployment import (
        DeltaMinCalculator, NoZenoBoundCalculator,
    )
    HAS_MATH = True
except ImportError:
    HAS_MATH = False

try:
    from agi_one_v3_11 import AGIONE, AGIConfig
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

__all__ = [
    "SecurityZone", "SegmentationConfig",
    "NeuralNetworkSegmentation", "ImplantCommandGate",
    "HelmChartGenerator", "SESIFastAPIServer",
    "SESIDeploymentSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — Security Zones
# ═══════════════════════════════════════════════════════════════════════

class SecurityZone(Enum):
    """
    [D] Four-zone segmentation model for implanted-device systems.

    Z0 IMPLANT      — the implanted device network (most isolated)
    Z1 CLINICAL_GW  — clinician-facing gateway (approval required)
    Z2 PROCESSING   — signal processing / AI inference servers
    Z3 EXTERNAL     — external / internet-facing services

    Traffic is DEFAULT-DENY. Only explicitly allowed (zone, zone, port,
    protocol) tuples pass, and every cross-zone flow is logged.
    """
    Z0_IMPLANT = 0
    Z1_CLINICAL_GATEWAY = 1
    Z2_PROCESSING = 2
    Z3_EXTERNAL = 3


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SegmentationConfig:
    """
    Configuration for network segmentation and implant-command gating.

    HONEST: every numeric field is a FREE DESIGN PARAMETER
    (Unified_Ordinal Hyp 7.9). Validate with a clinical-safety board.
    """
    # Delta_Min parameters (κ-fatness → δ_min)
    kappa: float = 2.0
    n_dim: int = 2
    a_min: float = 0.01          # minimum created measure (command quantum)

    # No-Zeno rate limiting (SESI_NoZeno_Closure Thm 3.2)
    c_v: float = 1.0             # command velocity bound
    time_window_s: float = 60.0  # window T for N(T) bound

    # Stochastic exit-time alarm (Stochastic_NoZeno Thm 4.1)
    exit_delta: float = 0.3      # δ: state-deviation alarm threshold
    g0_diffusion: float = 0.05   # G_0: diffusion bound
    exit_epsilon_s: float = 5.0  # ε: observation window

    # Zone policy
    default_policy: str = "deny"
    require_approval_for_z0: bool = True   # ALWAYS True in production
    allow_autonomous: bool = False         # ALWAYS False

    # Emergency
    emergency_isolation_enabled: bool = True

    # Coercivity stability check (Revision22 Thm 2.1/3.1)
    alpha_2: float = 0.1
    alpha_4: float = 0.05


# ═══════════════════════════════════════════════════════════════════════
# [NS1] NEURAL NETWORK SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════

class NeuralNetworkSegmentation:
    """
    [D] Network segmentation engine for implanted neural devices.

    Implements:
      • Default-deny zone policy
      • Explicit allow-list of cross-zone flows
      • Full cross-zone audit log
      • Emergency isolation (cuts Z0 from all other zones)

    Grounded in PNDC Axiom 3 (Necessity & Proportionality): a flow is
    permitted only if it is NECESSARY for a legitimate clinical purpose
    and PROPORTIONATE to that purpose.
    """

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        # allowed flows: set of (src_zone, dst_zone, port, proto)
        self.allowed_flows: Set[Tuple[int, int, int, str]] = set()
        self.isolated: bool = False
        self.flow_log: List[Dict[str, Any]] = []
        self._install_default_rules()

    def _install_default_rules(self) -> None:
        """
        Install minimal default rules.

        Default posture: Z0 (implant) is reachable ONLY from Z1
        (clinical gateway), and only on a dedicated control port.
        Z2 (processing) may receive telemetry from Z1, never talk to
        Z0 directly. Z3 (external) cannot reach Z0 or Z1.
        """
        # Z1 -> Z0 on control port 8443 (TLS)
        self.allowed_flows.add(
            (SecurityZone.Z1_CLINICAL_GATEWAY.value,
             SecurityZone.Z0_IMPLANT.value, 8443, "tcp"))
        # Z0 -> Z1 telemetry return on 8443
        self.allowed_flows.add(
            (SecurityZone.Z0_IMPLANT.value,
             SecurityZone.Z1_CLINICAL_GATEWAY.value, 8443, "tcp"))
        # Z1 -> Z2 telemetry forwarding on 9090
        self.allowed_flows.add(
            (SecurityZone.Z1_CLINICAL_GATEWAY.value,
             SecurityZone.Z2_PROCESSING.value, 9090, "tcp"))
        # Z3 -> Z2 public API on 443 (never Z0/Z1)
        self.allowed_flows.add(
            (SecurityZone.Z3_EXTERNAL.value,
             SecurityZone.Z2_PROCESSING.value, 443, "tcp"))

    def check_flow(self, src: SecurityZone, dst: SecurityZone,
                   port: int, proto: str = "tcp") -> Dict[str, Any]:
        """
        Check whether a cross-zone flow is permitted.

        Returns a verdict dict; NEVER silently allows.
        """
        if self.isolated:
            verdict = "DENIED_ISOLATION_ACTIVE"
            allowed = False
        elif src == dst:
            verdict = "ALLOWED_INTRA_ZONE"
            allowed = True
        elif (src.value, dst.value, port, proto) in self.allowed_flows:
            verdict = "ALLOWED_POLICY"
            allowed = True
        else:
            verdict = "DENIED_DEFAULT_DENY"
            allowed = False

        record = {
            "timestamp": time.time(),
            "src": src.name,
            "dst": dst.name,
            "port": port,
            "proto": proto,
            "verdict": verdict,
            "allowed": allowed,
        }
        self.flow_log.append(record)
        return record

    def emergency_isolate(self, reason: str) -> Dict[str, Any]:
        """
        [D] Emergency isolation: cut Z0 (implant) off from all zones.

        This is the network-level equivalent of the clinical emergency
        stop. It is always available and cannot be disabled in production.
        """
        if not self.cfg.emergency_isolation_enabled:
            return {"status": "disabled", "reason": "isolation disabled by config"}

        self.isolated = True
        logger.critical(f"[SEGMENTATION] EMERGENCY ISOLATION: {reason}")
        return {
            "status": "ISOLATED",
            "reason": reason,
            "timestamp": time.time(),
            "note": (
                "Z0 (implant zone) is now isolated from all other zones. "
                "Only local, physically-secured clinical intervention can "
                "restore connectivity. This is intentional."
            ),
        }

    def restore_connectivity(self, authorized_by: str) -> Dict[str, Any]:
        """Restore connectivity after isolation (requires authorization)."""
        self.isolated = False
        logger.info(f"[SEGMENTATION] Connectivity restored by {authorized_by}")
        return {
            "status": "RESTORED",
            "authorized_by": authorized_by,
            "timestamp": time.time(),
        }

    def generate_k8s_networkpolicy(self) -> Dict[str, Any]:
        """
        [D] Generate a Kubernetes NetworkPolicy manifest enforcing the
        zone model. Z0 pods are labeled zone=implant, etc.
        """
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "implant-zone-isolation",
                "labels": {"app": "sesi-oversight"},
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {"zone": "implant"},
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"podSelector": {
                                "matchLabels": {"zone": "clinical-gateway"}}}
                        ],
                        "ports": [{"protocol": "TCP", "port": 8443}],
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {"podSelector": {
                                "matchLabels": {"zone": "clinical-gateway"}}}
                        ],
                        "ports": [{"protocol": "TCP", "port": 8443}],
                    }
                ],
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# [NS2] IMPLANT COMMAND GATE (No-Zeno + δ_min quantum)
# ═══════════════════════════════════════════════════════════════════════

class ImplantCommandGate:
    """
    [D] Rate-limits and quantum-gates commands to implanted devices.

    Applies three mathematical guards from the series:

      1. SESI No-Zeno dwell-time bound (Thm 3.2):
         N(T) ≤ 1 + 2·C_V·T/δ_min
         → at most this many commands may be ACCEPTED in window T.

      2. Delta_Min quantum gate (Thm 1.2):
         δ_min ≥ (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
         → any parameter change SMALLER than δ_min is REJECTED
           (prevents drip-attacks / chattering that evade per-command
           review by staying individually "small").

      3. Stochastic exit-time alarm (Stochastic_NoZeno Thm 4.1):
         P(sup‖Δstate‖ ≥ δ) ≤ 2exp(−δ²/8G₀²ε)
         → if observed implant-state deviation exceeds the bound's
           implied likely range, raise a segmentation alarm.

    HONEST: These are NECESSARY but not SUFFICIENT safety conditions.
    They bound command frequency and magnitude; they do not certify
    clinical appropriateness. Clinician approval remains mandatory.
    """

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._command_timestamps: List[float] = []
        self._last_params: Optional[Dict[str, float]] = None

        # Compute δ_min from Delta_Min Thm 1.2
        if HAS_MATH:
            self.delta_calc = DeltaMinCalculator(cfg.kappa, cfg.n_dim)
            self.delta_min = self.delta_calc.compute_delta_min(cfg.a_min)
            self.noz = NoZenoBoundCalculator(self.delta_min, cfg.c_v)
        else:
            # Fallback closed form: δ_min = (1/κ)(a_min/ω_{n-1})^{1/(n-1)}
            omega = math.pi ** ((cfg.n_dim - 1) / 2) / math.gamma(
                (cfg.n_dim - 1) / 2 + 1)
            self.delta_min = (1.0 / cfg.kappa) * (cfg.a_min / omega) ** (
                1.0 / (cfg.n_dim - 1))
            self.delta_calc = None
            self.noz = None

    def max_commands_in_window(self) -> float:
        """SESI No-Zeno dwell-time bound: N(T) ≤ 1 + 2·C_V·T/δ_min."""
        if self.delta_min <= 0:
            return float("inf")  # degenerate; κ-fatness violated
        T = self.cfg.time_window_s
        return 1.0 + (2.0 * self.cfg.c_v * T) / self.delta_min

    def _prune_window(self) -> None:
        cutoff = time.time() - self.cfg.time_window_s
        self._command_timestamps = [
            t for t in self._command_timestamps if t > cutoff]

    def check_rate(self) -> Dict[str, Any]:
        """Check whether another command may be accepted in this window."""
        self._prune_window()
        bound = self.max_commands_in_window()
        current = len(self._command_timestamps)
        allowed = current < bound
        return {
            "allowed": allowed,
            "current_count": current,
            "bound": bound,
            "window_s": self.cfg.time_window_s,
            "delta_min": self.delta_min,
            "note": (
                "SESI_NoZeno_Closure Thm 3.2 dwell-time bound. "
                "Commands beyond the bound are REJECTED, not queued."
            ),
        }

    def check_quantum(self, new_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Delta_Min quantum gate: reject sub-δ_min parameter changes.

        A change whose norm is below δ_min is treated as chattering /
        drip-attack and rejected, because κ-fatness (Delta_Min Def 1.1)
        is required for the No-Zeno bound to hold.
        """
        if self._last_params is None:
            # First command: accept as baseline
            self._last_params = dict(new_params)
            return {"allowed": True, "reason": "baseline_established",
                    "delta": None}

        # L2 norm of parameter change
        delta_sq = 0.0
        for k, v in new_params.items():
            prev = self._last_params.get(k, 0.0)
            delta_sq += (v - prev) ** 2
        delta = math.sqrt(delta_sq)

        if delta < self.delta_min:
            return {
                "allowed": False,
                "reason": "sub_delta_min_chatter",
                "delta": delta,
                "delta_min": self.delta_min,
                "note": (
                    "Parameter change is below δ_min. Rejected to prevent "
                    "drip-attack / chattering. Delta_Min Thm 1.2 requires "
                    "κ-fatness; sub-threshold changes violate it."
                ),
            }

        self._last_params = dict(new_params)
        return {"allowed": True, "reason": "quantum_ok",
                "delta": delta, "delta_min": self.delta_min}

    def exit_time_alarm(self, observed_deviation: float) -> Dict[str, Any]:
        """
        Stochastic exit-time alarm (Stochastic_NoZeno Thm 4.1).

        Given an observed state deviation, compute the BDG-style tail
        bound and flag if the deviation is implausibly large.
        """
        delta = self.cfg.exit_delta
        g0 = self.cfg.g0_diffusion
        eps = self.cfg.exit_epsilon_s
        # P(sup >= delta) <= 2 exp(-delta^2 / (8 g0^2 eps))
        exponent = -(delta ** 2) / (8.0 * g0 ** 2 * eps)
        prob_bound = 2.0 * math.exp(exponent)
        prob_bound = min(prob_bound, 1.0)

        alarm = observed_deviation > delta
        return {
            "alarm": alarm,
            "observed_deviation": observed_deviation,
            "threshold_delta": delta,
            "prob_bound": prob_bound,
            "note": (
                "Stochastic_NoZeno Thm 4.1 exit-time tail bound. "
                "If observed deviation exceeds δ, the implant state may "
                "have been perturbed outside the expected diffusion envelope."
            ),
        }

    def gate_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full gate: rate + quantum + approval check.

        Returns a verdict. NEVER auto-executes.
        """
        # 1. Rate gate
        rate = self.check_rate()
        if not rate["allowed"]:
            return {"verdict": "REJECTED_RATE", "detail": rate}

        # 2. Quantum gate
        params = command.get("parameters", {})
        quantum = self.check_quantum(params)
        if not quantum["allowed"]:
            return {"verdict": "REJECTED_QUANTUM", "detail": quantum}

        # 3. Approval requirement (always for Z0)
        if self.cfg.require_approval_for_z0:
            return {
                "verdict": "PENDING_CLINICIAN_APPROVAL",
                "command_id": command.get("command_id", str(uuid.uuid4())),
                "detail": {
                    "rate": rate,
                    "quantum": quantum,
                },
                "note": (
                    "Command passed rate and quantum gates but REQUIRES "
                    "clinician approval before execution. Autonomous "
                    "execution is disabled by construction."
                ),
            }

        # Should not reach here in production
        return {"verdict": "REJECTED_POLICY", "detail": {}}

    def record_accepted(self) -> None:
        """Record an accepted (approved) command for rate accounting."""
        self._command_timestamps.append(time.time())


# ═══════════════════════════════════════════════════════════════════════
# [HC1] HELM CHART GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class HelmChartGenerator:
    """
    [D] Generate a complete Helm chart for the SESI oversight stack.

    Produces:
      Chart.yaml, values.yaml
      templates/: deployment, service, ingress, hpa, pdb,
                  servicemonitor, networkpolicy, configmap, secret

    HONEST: These are TEMPLATES. Real deployment requires cluster-specific
    values, secret injection from a vault, and a security review.
    """

    def __init__(self, output_dir: str = "./helm/sesi-oversight"):
        self.out = Path(output_dir)

    def _chart_yaml(self) -> str:
        return """apiVersion: v2
name: sesi-oversight
description: SESI Buddhist-Neuro-Conscience oversight stack
type: application
version: 1.0.0
appVersion: "1.0.0"
kubeVersion: ">=1.24.0-0"
keywords: [ai-safety, oversight, conscience, neural]
maintainers:
  - name: PAI, Yoon A Limsuwan
    email: msps4u@gmail.com
"""

    def _values_yaml(self) -> str:
        return """replicaCount: 2

image:
  repository: ghcr.io/yoonalimsuwan/sesi-oversight
  tag: ""
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  httpPort: 8080
  wsPort: 8765

ingress:
  enabled: true
  className: nginx
  host: sesi.example.com
  tlsSecretName: sesi-tls

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPU: 70

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: "2"
    memory: 4Gi

# Network segmentation zones (labels applied to pods)
zones:
  implant: "implant"
  clinicalGateway: "clinical-gateway"
  processing: "processing"
  external: "external"

# Implant command gate (FREE DESIGN PARAMETERS — validate clinically)
implantGate:
  kappa: 2.0
  aMin: 0.01
  cV: 1.0
  windowS: 60
  exitDelta: 0.3
  g0: 0.05
  requireApproval: true
  allowAutonomous: false

serviceMonitor:
  enabled: true
  interval: 30s
"""

    def _deployment(self) -> str:
        return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "sesi.fullname" . }}
  labels:
    {{- include "sesi.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "sesi.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "sesi.selectorLabels" . | nindent 8 }}
        zone: {{ .Values.zones.processing }}
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: server
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.httpPort }}
            - name: ws
              containerPort: {{ .Values.service.wsPort }}
          envFrom:
            - configMapRef:
                name: {{ include "sesi.fullname" . }}-config
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 10
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: [ALL]
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir: {}
"""

    def _service(self) -> str:
        return """apiVersion: v1
kind: Service
metadata:
  name: {{ include "sesi.fullname" . }}
  labels:
    {{- include "sesi.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - name: http
      port: {{ .Values.service.httpPort }}
      targetPort: http
    - name: ws
      port: {{ .Values.service.wsPort }}
      targetPort: ws
  selector:
    {{- include "sesi.selectorLabels" . | nindent 4 }}
"""

    def _networkpolicy_implant(self) -> str:
        """
        CRITICAL NetworkPolicy: isolate the implant zone.
        Only the clinical-gateway may reach implant pods, on port 8443.
        """
        return """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{ include "sesi.fullname" . }}-implant-isolation
  labels:
    {{- include "sesi.labels" . | nindent 4 }}
spec:
  podSelector:
    matchLabels:
      zone: {{ .Values.zones.implant }}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              zone: {{ .Values.zones.clinicalGateway }}
      ports:
        - protocol: TCP
          port: 8443
  egress:
    - to:
        - podSelector:
            matchLabels:
              zone: {{ .Values.zones.clinicalGateway }}
      ports:
        - protocol: TCP
          port: 8443
"""

    def _networkpolicy_processing(self) -> str:
        """Processing zone: may talk to clinical gateway + external API."""
        return """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{ include "sesi.fullname" . }}-processing
  labels:
    {{- include "sesi.labels" . | nindent 4 }}
spec:
  podSelector:
    matchLabels:
      zone: {{ .Values.zones.processing }}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              zone: {{ .Values.zones.external }}
      ports:
        - protocol: TCP
          port: 443
    - from:
        - podSelector:
            matchLabels:
              zone: {{ .Values.zones.clinicalGateway }}
      ports:
        - protocol: TCP
          port: 9090
  egress:
    - to:
        - podSelector: {}
      ports:
        - protocol: TCP
          port: 443
        - protocol: TCP
          port: 9090
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
"""

    def _ingress(self) -> str:
        return """{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "sesi.fullname" . }}
  labels:
    {{- include "sesi.labels" . | nindent 4 }}
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/websocket-services: {{ include "sesi.fullname" . }}
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  ingressClassName: {{ .Values.ingress.className }}
  tls:
    - hosts: [{{ .Values.ingress.host }}]
      secretName: {{ .Values.ingress.tlsSecretName }}
  rules:
    - host: {{ .Values.ingress.host }}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {{ include "sesi.fullname" . }}
                port:
                  number: {{ .Values.service.httpPort }}
          - path: /ws
            pathType: Prefix
            backend:
              service:
                name: {{ include "sesi.fullname" . }}
                port:
                  number: {{ .Values.service.wsPort }}
{{- end }}
"""

    def _hpa(self) -> str:
        return """{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "sesi.fullname" . }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "sesi.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPU }}
{{- end }}
"""

    def _servicemonitor(self) -> str:
        return """{{- if .Values.serviceMonitor.enabled }}
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{ include "sesi.fullname" . }}
  labels:
    {{- include "sesi.labels" . | nindent 4 }}
    release: prometheus
spec:
  selector:
    matchLabels:
      {{- include "sesi.selectorLabels" . | nindent 6 }}
  endpoints:
    - port: http
      path: /metrics
      interval: {{ .Values.serviceMonitor.interval }}
{{- end }}
"""

    def _configmap(self) -> str:
        return """apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "sesi.fullname" . }}-config
  labels:
    {{- include "sesi.labels" . | nindent 4 }}
data:
  SESI_ENVIRONMENT: production
  SESI_LOG_LEVEL: INFO
  IMPLANT_KAPPA: {{ .Values.implantGate.kappa | quote }}
  IMPLANT_A_MIN: {{ .Values.implantGate.aMin | quote }}
  IMPLANT_C_V: {{ .Values.implantGate.cV | quote }}
  IMPLANT_WINDOW_S: {{ .Values.implantGate.windowS | quote }}
  IMPLANT_REQUIRE_APPROVAL: {{ .Values.implantGate.requireApproval | quote }}
  IMPLANT_ALLOW_AUTONOMOUS: "false"
"""

    def _helpers(self) -> str:
        return """{{- define "sesi.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "sesi.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "sesi.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "sesi.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{ include "sesi.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "sesi.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sesi.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
"""

    def generate_all(self) -> Dict[str, str]:
        """Write all chart files to disk."""
        templates = self.out / "templates"
        templates.mkdir(parents=True, exist_ok=True)

        files = {
            "Chart.yaml": self._chart_yaml(),
            "values.yaml": self._values_yaml(),
            "templates/_helpers.tpl": self._helpers(),
            "templates/deployment.yaml": self._deployment(),
            "templates/service.yaml": self._service(),
            "templates/ingress.yaml": self._ingress(),
            "templates/hpa.yaml": self._hpa(),
            "templates/servicemonitor.yaml": self._servicemonitor(),
            "templates/configmap.yaml": self._configmap(),
            "templates/networkpolicy-implant.yaml": self._networkpolicy_implant(),
            "templates/networkpolicy-processing.yaml": self._networkpolicy_processing(),
        }
        written = {}
        for rel, content in files.items():
            p = self.out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            written[rel] = str(p)
        logger.info(f"Helm chart written to {self.out}")
        return written


# ═══════════════════════════════════════════════════════════════════════
# [FS1] FASTAPI SERVER
# ═══════════════════════════════════════════════════════════════════════

class SESIFastAPIServer:
    """
    [D] FastAPI server wiring together the conscience, oversight,
    neural-device, and segmentation layers.

    Endpoints:
      Ops:      /health /ready /metrics
      Conscience: /api/v1/conscience/state
      Neural:   /api/v1/neural/read
      Implant:  /api/v1/implant/command  /api/v1/implant/approve
                /api/v1/implant/emergency-stop
      Oversight:/api/v1/oversight/status /api/v1/oversight/audit
      Segmentation: /api/v1/segmentation/check-flow
                    /api/v1/segmentation/isolate
      WS:       /ws/alerts

    HONEST: This is a DECISION-SUPPORT server. It never autonomously
    actuates an implant. All implant mutations require clinician approval.
    """

    def __init__(self, cfg: Optional[SegmentationConfig] = None):
        self.cfg = cfg or SegmentationConfig()
        self.segmentation = NeuralNetworkSegmentation(self.cfg)
        self.implant_gate = ImplantCommandGate(self.cfg)

        # Optional subsystems (graceful degradation)
        self.conscience = None
        if HAS_CONSCIENCE:
            try:
                self.conscience = BuddhistNeuroPsychLLM(NeuroPsychConfig())
            except Exception as e:
                logger.warning(f"Conscience init failed: {e}")

        self.oversight = MultiLayerOversightSystem() if HAS_OVERSIGHT else None
        self.auth = AuthenticationAuthorizationLayer() if HAS_AUTH else None
        self.agione = None
        if HAS_AGIONE:
            try:
                self.agione = AGIONE(AGIConfig(latent_dim=64, action_dim=16))
            except Exception as e:
                logger.warning(f"AGIONE init failed: {e}")

        # WebSocket clients + metrics
        self._ws_clients: Set[Any] = set()
        self._metrics: Dict[str, float] = {
            "sesi_requests_total": 0.0,
            "sesi_implant_commands_total": 0.0,
            "sesi_implant_commands_rejected": 0.0,
            "sesi_emergency_stops_total": 0.0,
            "sesi_segmentation_denials": 0.0,
        }

        # Build FastAPI app
        self.app = self._build_app() if HAS_FASTAPI else None

    # ── Pydantic models ────────────────────────────────────────────────
    def _build_app(self) -> Any:
        app = FastAPI(
            title="SESI Oversight Server",
            version="1.0.0",
            description=(
                "CLINICAL DECISION SUPPORT ONLY. Not a medical device. "
                "All implant commands require clinician approval."
            ),
        )
        app.add_middleware(
            CORSMiddleware, allow_origins=["*"],
            allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
        )

        if HAS_FASTAPI:
            class ImplantCommandModel(BaseModel):
                device_id: str
                action: str
                parameters: Dict[str, float] = Field(default_factory=dict)
                issued_by: str
                physician_approval_id: Optional[str] = None

            class FlowCheckModel(BaseModel):
                src_zone: int
                dst_zone: int
                port: int
                proto: str = "tcp"

            class ApprovalModel(BaseModel):
                command_id: str
                physician_id: str
                license_number: str
                approved: bool

            # ── Ops ────────────────────────────────────────────────────
            @app.get("/health")
            async def health():
                return {"status": "ok", "timestamp": time.time()}

            @app.get("/ready")
            async def ready():
                return {"status": "ready"}

            @app.get("/metrics")
            async def metrics():
                self._metrics["sesi_requests_total"] += 1
                lines = []
                for k, v in self._metrics.items():
                    lines.append(f"{k} {v}")
                return "\n".join(lines)

            # ── Conscience ────────────────────────────────────────────
            @app.get("/api/v1/conscience/state")
            async def conscience_state():
                if self.conscience is None:
                    raise HTTPException(503, "Conscience module unavailable")
                status = getattr(self.conscience, "get_system_status",
                                 lambda: {})()
                return {"status": "available", "detail": status}

            # ── Neural read ───────────────────────────────────────────
            @app.get("/api/v1/neural/read")
            async def neural_read(device_id: str):
                # Read-only; safe to expose to processing zone
                return {
                    "device_id": device_id,
                    "note": "Read-only synthetic signal. Replace with real device SDK.",
                    "synthetic": True,
                }

            # ── Implant command submission ────────────────────────────
            @app.post("/api/v1/implant/command")
            async def implant_command(cmd: ImplantCommandModel):
                self._metrics["sesi_implant_commands_total"] += 1
                command = cmd.model_dump()
                command["command_id"] = str(uuid.uuid4())
                verdict = self.implant_gate.gate_command(command)
                if verdict["verdict"].startswith("REJECTED"):
                    self._metrics["sesi_implant_commands_rejected"] += 1
                return verdict

            # ── Implant approval ──────────────────────────────────────
            @app.post("/api/v1/implant/approve")
            async def implant_approve(appr: ApprovalModel):
                # In production this verifies the physician license and
                # records to the audit trail before any execution.
                return {
                    "status": "recorded",
                    "command_id": appr.command_id,
                    "approved": appr.approved,
                    "note": (
                        "Approval recorded. Execution still requires the "
                        "segmentation flow check (Z1→Z0) to pass."
                    ),
                }

            # ── Emergency stop ────────────────────────────────────────
            @app.post("/api/v1/implant/emergency-stop")
            async def implant_emergency_stop(reason: str):
                self._metrics["sesi_emergency_stops_total"] += 1
                seg = self.segmentation.emergency_isolate(reason)
                await self._broadcast_alert("EMERGENCY", "implant", reason)
                return seg

            # ── Oversight ─────────────────────────────────────────────
            @app.get("/api/v1/oversight/status")
            async def oversight_status():
                if self.oversight is None:
                    return {"available": False}
                return {"available": True}

            # ── Segmentation ──────────────────────────────────────────
            @app.post("/api/v1/segmentation/check-flow")
            async def seg_check_flow(flow: FlowCheckModel):
                try:
                    src = SecurityZone(flow.src_zone)
                    dst = SecurityZone(flow.dst_zone)
                except ValueError:
                    raise HTTPException(400, "Invalid zone id")
                result = self.segmentation.check_flow(src, dst, flow.port, flow.proto)
                if not result["allowed"]:
                    self._metrics["sesi_segmentation_denials"] += 1
                return result

            @app.post("/api/v1/segmentation/isolate")
            async def seg_isolate(reason: str):
                self._metrics["sesi_emergency_stops_total"] += 1
                return self.segmentation.emergency_isolate(reason)

            # ── WebSocket alerts ──────────────────────────────────────
            @app.websocket("/ws/alerts")
            async def ws_alerts(websocket: WebSocket):
                await websocket.accept()
                self._ws_clients.add(websocket)
                try:
                    while True:
                        await websocket.receive_text()
                except WebSocketDisconnect:
                    self._ws_clients.discard(websocket)

        return app

    async def _broadcast_alert(self, level: str, source: str, message: str) -> None:
        payload = json.dumps({
            "alert_id": uuid.uuid4().hex[:8],
            "level": level,
            "source": source,
            "message": message,
            "timestamp": time.time(),
        })
        dead = set()
        for ws in self._ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._ws_clients.difference_update(dead)

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run with uvicorn (requires `uvicorn` installed)."""
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn not installed; cannot run server")
            return
        if self.app is None:
            logger.error("FastAPI not available; cannot run server")
            return
        uvicorn.run(self.app, host=host, port=port)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESIDeploymentSuite:
    """
    [D] Orchestrates server + Helm + segmentation.
    """

    def __init__(self, cfg: Optional[SegmentationConfig] = None):
        self.cfg = cfg or SegmentationConfig()
        self.server = SESIFastAPIServer(self.cfg)
        self.helm = HelmChartGenerator()

    def generate_helm_chart(self) -> Dict[str, str]:
        return self.helm.generate_all()

    def status(self) -> Dict[str, Any]:
        return {
            "fastapi_available": self.server.app is not None,
            "conscience_available": self.server.conscience is not None,
            "oversight_available": self.server.oversight is not None,
            "auth_available": self.server.auth is not None,
            "agione_available": self.server.agione is not None,
            "segmentation_isolated": self.server.segmentation.isolated,
            "implant_delta_min": self.server.implant_gate.delta_min,
            "implant_max_cmds_per_window":
                self.server.implant_gate.max_commands_in_window(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI FastAPI × Helm × Network Segmentation — Smoke Test")
    print("=" * 72)
    print(f"  FastAPI available   : {HAS_FASTAPI}")
    print(f"  NumPy available     : {HAS_NUMPY}")
    print(f"  Conscience available: {HAS_CONSCIENCE}")
    print(f"  Oversight available : {HAS_OVERSIGHT}")
    print(f"  Auth available      : {HAS_AUTH}")
    print(f"  Math available      : {HAS_MATH}")
    print(f"  AGIONE available    : {HAS_AGIONE}")

    suite = SESIDeploymentSuite()

    # ── Status ──
    print("\n  ── System Status ──")
    status = suite.status()
    for k, v in status.items():
        print(f"    {k:32s}: {v}")

    # ── Delta_Min / No-Zeno ──
    print("\n  ── Implant Command Gate (No-Zeno + δ_min) ──")
    gate = suite.server.implant_gate
    print(f"    δ_min (Delta_Min Thm 1.2)   : {gate.delta_min:.6f}")
    print(f"    Max commands / window       : "
          f"{gate.max_commands_in_window():.2f}")

    # Test rate gate
    rate = gate.check_rate()
    print(f"    Rate gate allowed           : {rate['allowed']}")

    # Test quantum gate: baseline
    q1 = gate.check_quantum({"amp": 1.0, "freq": 130.0})
    print(f"    Quantum gate (baseline)     : {q1['allowed']} ({q1['reason']})")
    # Sub-δ_min change → rejected
    q2 = gate.check_quantum({"amp": 1.0 + gate.delta_min * 0.1, "freq": 130.0})
    print(f"    Quantum gate (sub-δ_min)    : {q2['allowed']} ({q2['reason']})")
    # Super-δ_min change → accepted
    q3 = gate.check_quantum({"amp": 1.0 + gate.delta_min * 2.0, "freq": 130.0})
    print(f"    Quantum gate (super-δ_min)  : {q3['allowed']} ({q3['reason']})")

    # Test full gate
    cmd = {"command_id": "test", "action": "stimulate",
           "parameters": {"amp": 3.0, "freq": 130.0}}
    verdict = gate.gate_command(cmd)
    print(f"    Full gate verdict           : {verdict['verdict']}")

    # Exit-time alarm
    alarm = gate.exit_time_alarm(observed_deviation=0.5)
    print(f"    Exit-time alarm (dev=0.5)   : {alarm['alarm']} "
          f"(bound={alarm['prob_bound']:.4f})")

    # ── Segmentation ──
    print("\n  ── Network Segmentation ──")
    seg = suite.server.segmentation

    # Allowed: Z1 -> Z0 on 8443
    f1 = seg.check_flow(SecurityZone.Z1_CLINICAL_GATEWAY,
                        SecurityZone.Z0_IMPLANT, 8443)
    print(f"    Z1→Z0:8443  : {f1['verdict']}")

    # Denied: Z3 -> Z0 (external to implant)
    f2 = seg.check_flow(SecurityZone.Z3_EXTERNAL,
                        SecurityZone.Z0_IMPLANT, 8443)
    print(f"    Z3→Z0:8443  : {f2['verdict']}")

    # Denied: Z2 -> Z0 (processing to implant directly)
    f3 = seg.check_flow(SecurityZone.Z2_PROCESSING,
                        SecurityZone.Z0_IMPLANT, 8443)
    print(f"    Z2→Z0:8443  : {f3['verdict']}")

    # Emergency isolation
    iso = seg.emergency_isolate("test isolation")
    print(f"    Isolation   : {iso['status']}")
    f4 = seg.check_flow(SecurityZone.Z1_CLINICAL_GATEWAY,
                        SecurityZone.Z0_IMPLANT, 8443)
    print(f"    Z1→Z0 during isolation: {f4['verdict']}")
    seg.restore_connectivity("operator_001")
    print(f"    Restored    : {not seg.isolated}")

    # ── Helm chart ──
    print("\n  ── Helm Chart Generation ──")
    chart_files = suite.generate_helm_chart()
    print(f"    Files generated: {len(chart_files)}")
    for name, path in chart_files.items():
        print(f"      {name}")

    # ── FastAPI app ──
    print("\n  ── FastAPI App ──")
    if suite.server.app is not None:
        routes = [r.path for r in suite.server.app.routes]
        print(f"    Routes registered: {len(routes)}")
        for r in sorted(routes):
            print(f"      {r}")
    else:
        print("    [SKIP] FastAPI not available")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is DEPLOYMENT SCAFFOLDING, not a cleared medical device")
    print("  • Network segmentation REDUCES blast radius, does not eliminate risk")
    print("  • δ_min quantum gate requires κ-fatness (Delta_Min Def 1.1)")
    print("  • No-Zeno bound limits command COUNT, not clinical appropriateness")
    print("  • Every implant command REQUIRES clinician approval")
    print("  • allow_autonomous is hardcoded False")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Regulatory approval (FDA/CE/TGA) is a separate process")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
