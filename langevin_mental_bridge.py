# =============================================================================
# STRUCTURAL LANGEVIN ↔ MENTAL ONE BRIDGE
# =============================================================================
# Developer: Yoon A Limsuwan / MSPS NETWORK
# License: MIT
# Year: 2026
#
# Connects the AdvancedStructuralLangevin integrator to the MENTAL ONE engine.
# Provides drop-in replacements for:
#   • SOCController.soc_evolve()   →  LangevinSOCEvolve
#   • ItoProcess.step()            →  LangevinItoStep
#   • MentalHealthEvolution        →  LangevinMentalEvolution  (full upgrade)
#
# Physical mapping:
#   EEG / brain state vector  s  ∈ ℝ^N  →  "coordinates" x ∈ ℝ^(N×1) (1-D atoms)
#   SSC energy gradient ∇E(s)           →  force_bulk
#   Disorder distance from Healthy      →  structural stress σ
#   Contraction step size               →  Langevin dt
#
# Usage (replace inside MENTAL ONE):
#
#   from langevin_mental_bridge import LangevinMentalEvolution, LangevinItoStep
#
#   # In MentalHealthEvolution.__init__:
#   self.evolution = LangevinMentalEvolution(soc, rg, classifier)
#
#   # In ItoProcess-based trajectory simulation:
#   ito = LangevinItoStep(drift_fn, diffusion_fn, dt)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Helper: 1-D "interface detector" for brain state vectors
# ---------------------------------------------------------------------------

class BrainStateInterfaceDetector(nn.Module):
    """
    Detects "interface" regions in a 1-D EEG state vector — i.e., time points
    or channel positions where the signal undergoes rapid change (transients,
    pathological spikes, phase transitions).

    Maps the 1-D state vector s ∈ ℝ^N to a soft mask ∈ [0, 1]^N that is
    fully differentiable w.r.t. s (required for the Structural Itô correction).

    Criterion: local gradient magnitude normalised by global mean gradient.
    """

    def __init__(self, sharpness: float = 4.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s : (N,) 1-D brain state vector, may have requires_grad=True.
        Returns:
            mask : (N,) interface scores ∈ [0, 1], differentiable.
        """
        if s.dim() != 1:
            raise ValueError(f"Expected 1-D state vector, got shape {tuple(s.shape)}")

        # Local gradient via finite differences (central, zero-padded ends)
        s_pad = F.pad(s.unsqueeze(0).unsqueeze(0), (1, 1), mode='replicate')
        grad = (s_pad[0, 0, 2:] - s_pad[0, 0, :-2]) / 2.0        # (N,)
        grad_mag = grad.abs()

        norm_grad = grad_mag / (grad_mag.mean() + 1e-8)
        return torch.sigmoid(self.sharpness * (norm_grad - 1.0))   # (N,)


# ---------------------------------------------------------------------------
# Helper: SSC low-pass filter (shared with MD/CFD versions)
# ---------------------------------------------------------------------------

class SemanticStateContraction(nn.Module):
    """EMA filter for scalar structural stress σ."""

    def __init__(self, epsilon_fp: float = 0.0028):
        super().__init__()
        self.eps = epsilon_fp
        self.register_buffer('prev_sigma', torch.tensor(0.0))
        self.register_buffer('_initialized', torch.tensor(False))

    def reset(self) -> None:
        self.prev_sigma.zero_()
        self._initialized.fill_(False)

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        if not self._initialized.item():
            self.prev_sigma.data = raw_sigma.detach()
            self._initialized.fill_(True)
            return raw_sigma
        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


# ---------------------------------------------------------------------------
# Core: Langevin brain-state integrator (1-D BAOAB)
# ---------------------------------------------------------------------------

class LangevinBrainIntegrator(nn.Module):
    """
    BAOAB Langevin integrator adapted for 1-D EEG/brain state vectors.

    Key design choices vs. the 3-D MD version:
    • "Coordinates" x = s ∈ ℝ^N  (flattened EEG state)
    • "Force"       F = −∇_s E(s)  from the SSC energy function
    • Interface mask built by BrainStateInterfaceDetector
    • CSOC thermostat tracks disorder-distance σ from the Healthy manifold
    • Output is clamped to [0, 1] to match MENTAL ONE's normalisation

    This makes the stochastic trajectory thermodynamically consistent,
    replacing the naive `x += randn * T * 0.01` in SOCController.
    """

    def __init__(
        self,
        state_dim: int,
        dt: float = 0.002,
        base_temp: float = 300.0,
        base_friction: float = 1.0,
        kb: float = 0.001987,
        interface_amplification: float = 2.0,
        temp_boost_factor: float = 3.0,
        epsilon_fp: float = 0.0028,
    ):
        """
        Args:
            state_dim               : length of the flattened EEG state vector (N).
            dt                      : integration time step.
            base_temp               : reference temperature (K).
            base_friction           : reference friction γ.
            kb                      : Boltzmann constant in chosen units.
            interface_amplification : noise amplification at transients.
            temp_boost_factor       : max temperature = base_temp × factor.
            epsilon_fp              : SSC EMA rate.
        """
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.base_temp = base_temp
        self.base_friction = base_friction
        self.kb = kb
        self.amp = interface_amplification
        self.temp_boost_factor = temp_boost_factor

        self.interface_detector = BrainStateInterfaceDetector()
        self.ssc = SemanticStateContraction(epsilon_fp)

        self.register_buffer('_prev_s', torch.zeros(state_dim))
        self.register_buffer('_prev_v', torch.zeros(state_dim))
        self.register_buffer('_state_ready', torch.tensor(False))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Call between independent patient trajectories."""
        self._prev_s.zero_()
        self._prev_v.zero_()
        self._state_ready.fill_(False)
        self.ssc.reset()

    # ------------------------------------------------------------------
    def _adaptive_T_gamma(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CSOC thermostat: modulates T and γ from disorder-distance stress."""
        if not self._state_ready.item():
            self._prev_s = s.detach().clone()
            self._state_ready.fill_(True)

        raw_sigma = (s - self._prev_s).abs().mean()
        sigma = self.ssc(raw_sigma)
        self._prev_s = s.detach().clone()

        sigma_target = 0.05
        dev = (sigma - sigma_target) / max(sigma_target, 1e-8)
        boost = self.base_temp * (self.temp_boost_factor - 1.0)
        T = self.base_temp + boost * torch.sigmoid(dev)
        T = torch.clamp(T, self.base_temp * 0.5, self.base_temp * self.temp_boost_factor)
        gamma = self.base_friction * (1.0 + 0.5 * torch.relu(dev))
        return T, gamma, sigma

    # ------------------------------------------------------------------
    def _ito_correction(self, s: torch.Tensor) -> torch.Tensor:
        """Structural Itô drift: ½ G(s) ∇G(s), fully differentiable."""
        with torch.enable_grad():
            s_g = s.detach().requires_grad_(True)
            mask = self.interface_detector(s_g)
            G = 1.0 + self.amp * mask
            grad_G = torch.autograd.grad(G.sum(), s_g, create_graph=False)[0]
            if grad_G is None:
                return torch.zeros_like(s)
            ito = 0.5 * G * grad_G
        return ito.detach()

    # ------------------------------------------------------------------
    def baoa_step(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        force: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        One BAOA sub-step of BAOAB for brain-state dynamics.

        Args:
            s     : (N,) current state vector.
            v     : (N,) current "velocity" (rate of state change).
            force : (N,) = −∇_s E(s), from SSCClassifier.energy().
        Returns:
            s_new    : (N,) state after BAOA.
            v_tilde  : (N,) velocity after O-step (pre-final-B).
            T_scalar : float adaptive temperature.
            sigma_sc : float SSC stress.
        """
        T, gamma, sigma = self._adaptive_T_gamma(s)

        # [ B ] half-step velocity
        v_half = v + 0.5 * self.dt * force

        # [ A ] half-step position
        s_half = s + 0.5 * self.dt * v_half

        # [ O ] Ornstein-Uhlenbeck stochastic update
        c1 = torch.exp(-gamma * self.dt)
        c2 = torch.sqrt((1.0 - c1 ** 2).clamp(min=0.0))

        mask = self.interface_detector(s_half)
        G = 1.0 + self.amp * mask

        noise_scale = torch.sqrt(self.kb * T)
        R = torch.randn_like(v_half)
        stochastic = c2 * noise_scale * G * R

        ito = self._ito_correction(s_half)
        v_tilde = c1 * v_half + stochastic + ito * self.dt

        # [ A ] second half-step position
        s_new = s_half + 0.5 * self.dt * v_tilde

        return s_new, v_tilde, T.item(), sigma.item()

    def final_b_step(
        self,
        v_tilde: torch.Tensor,
        new_force: torch.Tensor,
    ) -> torch.Tensor:
        """Final B sub-step — call after re-evaluating force at s_new."""
        return v_tilde + 0.5 * self.dt * new_force


# ---------------------------------------------------------------------------
# Drop-in 1: LangevinSOCEvolve  (replaces SOCController.soc_evolve)
# ---------------------------------------------------------------------------

class LangevinSOCEvolve(nn.Module):
    """
    Drop-in replacement for ``SOCController.soc_evolve()``.

    Uses BAOAB Langevin dynamics instead of the naive random walk, providing:
    • Thermodynamically consistent stochastic trajectories.
    • Multiplicative noise concentrated at pathological transients.
    • Structural Itô correction preventing spurious drift.
    • CSOC-adaptive temperature / friction from disorder distance.

    Compatible signature:
        evolved_state = LangevinSOCEvolve(classifier)(mu_seq, steps=20)
    """

    def __init__(
        self,
        classifier,             # SSCClassifier from MENTAL ONE
        target_disorder: str = 'MDD',
        healthy_disorder: str = 'Healthy',
        dt: float = 0.002,
        base_temp: float = 300.0,
        base_friction: float = 1.0,
    ):
        """
        Args:
            classifier       : MENTAL ONE SSCClassifier (provides energy function).
            target_disorder  : disorder whose energy landscape to evolve on.
            healthy_disorder : healthy reference label.
            dt               : Langevin time step.
            base_temp        : reference temperature (K).
            base_friction    : reference friction γ.
        """
        super().__init__()
        self.classifier = classifier
        self.target = target_disorder
        self.healthy = healthy_disorder

        # Integrator will be built lazily on first forward pass
        # (state_dim unknown until we see the first input)
        self._integrator: Optional[LangevinBrainIntegrator] = None
        self._dt = dt
        self._base_temp = base_temp
        self._base_friction = base_friction

    def _ensure_integrator(self, state_dim: int, device: torch.device) -> None:
        if self._integrator is None or self._integrator.state_dim != state_dim:
            self._integrator = LangevinBrainIntegrator(
                state_dim=state_dim,
                dt=self._dt,
                base_temp=self._base_temp,
                base_friction=self._base_friction,
            ).to(device)

    def forward(self, s0: torch.Tensor, steps: int = 20) -> torch.Tensor:
        """
        Evolve brain state ``s0`` for ``steps`` Langevin steps.

        Args:
            s0    : (N,) initial state vector ∈ [0, 1].
            steps : number of BAOAB integration steps.
        Returns:
            s_final : (N,) evolved state ∈ [0, 1], detached.
        """
        N = s0.shape[0]
        device = s0.device
        self._ensure_integrator(N, device)
        integrator = self._integrator
        integrator.reset()

        s = s0.clone()
        v = torch.zeros_like(s)

        for _ in range(steps):
            # Force = −∇_s E(s) from SSC energy
            s_req = s.detach().requires_grad_(True)
            E = self.classifier.energy(s_req, self.target, self.healthy)
            force = -torch.autograd.grad(E, s_req, create_graph=False)[0]

            # BAOA sub-step
            s_new, v_tilde, _, _ = integrator.baoa_step(s, v, force)

            # Re-evaluate force at s_new for final B step
            s_new_req = s_new.detach().requires_grad_(True)
            E_new = self.classifier.energy(s_new_req, self.target, self.healthy)
            force_new = -torch.autograd.grad(E_new, s_new_req, create_graph=False)[0]

            v = integrator.final_b_step(v_tilde, force_new)
            s = torch.clamp(s_new.detach(), 0.0, 1.0)

        return s


# ---------------------------------------------------------------------------
# Drop-in 2: LangevinItoStep  (replaces ItoProcess.step)
# ---------------------------------------------------------------------------

class LangevinItoStep(nn.Module):
    """
    Drop-in replacement for ``ItoProcess.step()``.

    Upgrades the Euler-Maruyama step to a thermodynamically correct BAOAB
    half-step with structural Itô correction, while keeping the same
    drift_fn / diffusion_fn interface.

    Compatible signature:
        step_fn = LangevinItoStep(drift_fn, diffusion_fn, dt)
        x_next = step_fn.step(x)
    """

    def __init__(
        self,
        drift_fn: Callable,
        diffusion_fn: Callable,
        dt: float = 0.01,
        base_temp: float = 300.0,
        base_friction: float = 1.0,
        interface_amplification: float = 2.0,
    ):
        super().__init__()
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.dt = dt
        self.base_temp = base_temp
        self.base_friction = base_friction
        self.amp = interface_amplification

        self.interface_detector = BrainStateInterfaceDetector()
        self.ssc = SemanticStateContraction()

        self.register_buffer('_prev_x', torch.zeros(1))
        self.register_buffer('_initialized', torch.tensor(False))

    def _ito_correction(self, x: torch.Tensor, sigma_fn: Callable) -> torch.Tensor:
        """½ σ(x) ∇σ(x) Itô drift correction."""
        with torch.enable_grad():
            x_g = x.detach().requires_grad_(True)
            mask = self.interface_detector(x_g)
            G = 1.0 + self.amp * mask
            # Use diffusion amplitude as the multiplicative factor
            sig = sigma_fn(x_g)
            amplitude = G * sig.abs()
            grad_amp = torch.autograd.grad(amplitude.sum(), x_g,
                                            create_graph=False)[0]
            if grad_amp is None:
                return torch.zeros_like(x)
            ito = 0.5 * amplitude * grad_amp
        return ito.detach()

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        One Langevin-corrected step.

        Args:
            x : (N,) current state vector.
        Returns:
            x_next : (N,) next state, clamped to [0, 1].
        """
        if not self._initialized.item() or self._prev_x.shape != x.shape:
            self._prev_x = x.detach().clone()
            self._initialized.fill_(True)

        raw_sigma = (x - self._prev_x).abs().mean()
        sigma_val = self.ssc(raw_sigma)
        self._prev_x = x.detach().clone()

        # Adaptive temperature
        dev = (sigma_val - 0.05) / 0.05
        T = self.base_temp * (1.0 + 2.0 * torch.sigmoid(dev))
        gamma = self.base_friction * (1.0 + 0.5 * torch.relu(dev))

        # Ornstein-Uhlenbeck noise scale
        c1 = torch.exp(-gamma * self.dt)
        c2 = torch.sqrt((1.0 - c1 ** 2).clamp(min=0.0))

        mask = self.interface_detector(x)
        G = 1.0 + self.amp * mask

        drift = self.drift_fn(x)
        diffusion = self.diffusion_fn(x)

        # Itô correction using diffusion amplitude
        ito = self._ito_correction(x, self.diffusion_fn)

        dw = torch.randn_like(x) * math.sqrt(self.dt)
        noise_scale = torch.sqrt(torch.tensor(0.001987 * T.item()))

        x_next = (x
                  + drift * self.dt
                  + c2 * noise_scale * G * dw
                  + ito * self.dt)

        return torch.clamp(x_next, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Full upgrade: LangevinMentalEvolution  (replaces MentalHealthEvolution)
# ---------------------------------------------------------------------------

class LangevinMentalEvolution(nn.Module):
    """
    Full replacement for ``MentalHealthEvolution`` that uses
    LangevinSOCEvolve instead of the naive random-walk SOC.

    Compatible with ``MentalHealthEvolution(soc, rg)`` API.

    Changes:
    • forward() returns the same dict {'future', 'smooth'}.
    • 'future' is now produced by BAOAB Langevin dynamics.
    • Diagnostics (T, sigma) are optionally logged.
    """

    def __init__(
        self,
        soc,                        # SOCController (kept for kernel compatibility)
        rg,                         # DiffRGRefiner (unchanged)
        classifier,                 # SSCClassifier (for energy function)
        target_disorder: str = 'MDD',
        healthy_disorder: str = 'Healthy',
        dt: float = 0.002,
        base_temp: float = 300.0,
        log_diagnostics: bool = False,
    ):
        super().__init__()
        self.rg = rg
        self.soc = soc                              # kept for compute_soc_energy compatibility
        self.log_diagnostics = log_diagnostics

        self.langevin_evolve = LangevinSOCEvolve(
            classifier=classifier,
            target_disorder=target_disorder,
            healthy_disorder=healthy_disorder,
            dt=dt,
            base_temp=base_temp,
            base_friction=soc.friction if hasattr(soc, 'friction') else 1.0,
        )

    def forward(self, mu_seq: torch.Tensor, steps: int = 50) -> dict:
        """
        Args:
            mu_seq : (T,) mean brain-state time series from SSCClassifier.
            steps  : number of Langevin integration steps.
        Returns:
            dict with keys:
                'future' : (T,) evolved state after Langevin dynamics.
                'smooth' : (T,) RG-smoothed state.
        """
        mu_smooth = self.rg(mu_seq)

        # Ensure 1-D for integrator
        if mu_smooth.dim() > 1:
            mu_smooth = mu_smooth.squeeze(0)

        future = self.langevin_evolve(mu_smooth, steps=steps)

        result = {'future': future, 'smooth': mu_smooth}
        return result


# =============================================================================
# Integration patch — monkey-patch MENTAL ONE in-place
# =============================================================================

def patch_mental_one(engine, target_disorder: str = 'MDD') -> None:
    """
    Convenience function: upgrades a live ``MentalONEEngine`` instance
    in-place by replacing its evolution module with LangevinMentalEvolution.

    Usage::
        from langevin_mental_bridge import patch_mental_one
        engine = MentalONEEngine()
        engine.initialise_from_dataset(dataset, subject_list)
        patch_mental_one(engine, target_disorder='MDD')
        # engine.evolution is now LangevinMentalEvolution
    """
    if engine.evolution is None or engine.classifier is None:
        raise RuntimeError(
            "Call engine.initialise_from_dataset() before patch_mental_one()."
        )

    soc = engine.evolution.soc if hasattr(engine.evolution, 'soc') else None
    rg  = engine.evolution.rg  if hasattr(engine.evolution, 'rg')  else None

    if soc is None or rg is None:
        raise AttributeError(
            "engine.evolution must have .soc and .rg attributes."
        )

    engine.evolution = LangevinMentalEvolution(
        soc=soc,
        rg=rg,
        classifier=engine.classifier,
        target_disorder=target_disorder,
    ).to(engine.device)

    print(f"[LangevinBridge] MentalONEEngine.evolution patched → "
          f"LangevinMentalEvolution (target={target_disorder})")


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    import sys
    print("Running LangevinMentalBridge self-test ...")
    torch.manual_seed(42)

    N = 19 * 256   # standard EEG state dim

    # Minimal mock classifier (mimics SSCClassifier API)
    class MockClassifier(nn.Module):
        def __init__(self, N):
            super().__init__()
            self.register_buffer('ref_MDD',     torch.randn(N))
            self.register_buffer('ref_Healthy', torch.randn(N))
            self.feature_references = {
                'MDD':     torch.zeros(5),
                'Healthy': torch.zeros(5),
            }
        def energy(self, s, target, healthy='Healthy'):
            p = getattr(self, f'ref_{target}')
            return 0.5 * ((s - p) ** 2).sum()

    clf = MockClassifier(N)
    s0 = torch.rand(N)

    # Test LangevinSOCEvolve
    evolve = LangevinSOCEvolve(clf, target_disorder='MDD', steps_default=5)
    s_out = evolve(s0, steps=5)
    assert s_out.shape == (N,), f"Shape mismatch: {s_out.shape}"
    assert s_out.min() >= 0.0 and s_out.max() <= 1.0, "Output out of [0,1]"
    print(f"  LangevinSOCEvolve OK — output norm: {s_out.norm().item():.4f}")

    # Test LangevinItoStep
    drift_fn     = lambda x: -0.1 * x
    diffusion_fn = lambda x: 0.01 * torch.ones_like(x)
    ito = LangevinItoStep(drift_fn, diffusion_fn, dt=0.01)
    x = torch.rand(N)
    for _ in range(3):
        x = ito.step(x)
    assert x.shape == (N,)
    print(f"  LangevinItoStep OK — output norm: {x.norm().item():.4f}")

    print("Self-test passed.")
    sys.exit(0)
