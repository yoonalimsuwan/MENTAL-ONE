# =============================================================================
# Hybrid Psychopathology Engine   —  NATIVE FULL DIFFERENTIABLE 
# =============================================================================
# Developer  : PAI AND Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788


import torch
import torch.nn as nn
from typing import Tuple, Optional

class HybridPsychopathologyEngine(nn.Module):
    """
    Production-grade, native fully differentiable Hybrid Dynamical Energy-Network 
    Model for Psychopathology. Unifies a continuous Lorenz cognitive-affective-identity 
    core with a discrete, pressure-driven symptom network.
    
    This program was written with the assistance of Gemini.
    """
    def __init__(num_nodes: int, dt: float = 0.01):
        super(HybridPsychopathologyEngine, self).__init__()
        self.num_nodes = num_nodes
        self.dt = dt

        # ---------------------------------------------------------
        # Continuous Lorenz Core Parameters[span_36](start_span)[span_36](end_span)[span_37](start_span)[span_37](end_span)[span_38](start_span)[span_38](end_span)
        # ---------------------------------------------------------
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_lorenz = nn.Parameter(torch.tensor(8.0 / 3.0))

        # ---------------------------------------------------------
        # Global Pressure Coupling Weights[span_39](start_span)[span_39](end_span)[span_40](start_span)[span_40](end_span)[span_41](start_span)[span_41](end_span)
        # ---------------------------------------------------------
        self.k1 = nn.Parameter(torch.tensor(1.0))
        self.k2 = nn.Parameter(torch.tensor(1.0))
        self.k3 = nn.Parameter(torch.tensor(1.0))

        # ---------------------------------------------------------
        # Node-Specific External Driving Sensitivities[span_42](start_span)[span_42](end_span)[span_43](start_span)[span_43](end_span)[span_44](start_span)[span_44](end_span)
        # ---------------------------------------------------------
        self.gamma = nn.Parameter(torch.randn(num_nodes))
        self.delta = nn.Parameter(torch.randn(num_nodes))
        self.epsilon = nn.Parameter(torch.randn(num_nodes))

        # ---------------------------------------------------------
        # Discrete Network Parameters & Self-Loops[span_45](start_span)[span_45](end_span)[span_46](start_span)[span_46](end_span)[span_47](start_span)[span_47](end_span)
        # ---------------------------------------------------------
        self.alpha = nn.Parameter(torch.ones(num_nodes)) # Global pressure sensitivity per node[span_48](start_span)[span_48](end_span)[span_49](start_span)[span_49](end_span)[span_50](start_span)[span_50](end_span)
        self.w_ii = nn.Parameter(torch.zeros(num_nodes)) # Self-loop persistence[span_51](start_span)[span_51](end_span)[span_52](start_span)[span_52](end_span)[span_53](start_span)[span_53](end_span)
        
        # Extended variant parameters[span_54](start_span)[span_54](end_span)
        self.tau_homeostasis = nn.Parameter(torch.tensor(0.0)) # Homeostatic gain[span_55](start_span)[span_55](end_span)
        self.baseline_s = nn.Parameter(torch.zeros(num_nodes)) # Baseline state s*[span_56](start_span)[span_56](end_span)

    def _lorenz_derivatives(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates time derivatives for the continuous cognitive-affective-identity core[span_57](start_span)[span_57](end_span)[span_58](start_span)[span_58](end_span)[span_59](start_span)[span_59](end_span)."""
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta_lorenz * z
        return dx, dy, dz

    def _rk4_step(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized 4th-order Runge-Kutta (RK4) integration step for stability[span_60](start_span)[span_60](end_span)[span_61](start_span)[span_61](end_span)[span_62](start_span)[span_62](end_span)."""
        dt = self.dt
        dt2 = dt * 0.5

        k1_x, k1_y, k1_z = self._lorenz_derivatives(x, y, z)
        k2_x, k2_y, k2_z = self._lorenz_derivatives(x + dt2 * k1_x, y + dt2 * k1_y, z + dt2 * k1_z)
        k3_x, k3_y, k3_z = self._lorenz_derivatives(x + dt2 * k2_x, y + dt2 * k2_y, z + dt2 * k2_z)
        k4_x, k4_y, k4_z = self._lorenz_derivatives(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z)

        x_next = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        y_next = y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        z_next = z + (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)

        return x_next, y_next, z_next

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        z: torch.Tensor, 
        s: torch.Tensor, 
        adj_matrix: torch.Tensor, 
        steps: int = 1,
        noise_std: float = 0.0,
        autonomic_feedback: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes forward simulation over specified time steps. Fully differentiable 
        for end-to-end backpropagation and parameter optimization.

        Args:
            x, y, z (torch.Tensor): Continuous core state tensors of shape (batch_size, 1)[span_63](start_span)[span_63](end_span)[span_64](start_span)[span_64](end_span)[span_65](start_span)[span_65](end_span).
            s (torch.Tensor): Discrete network state tensor of shape (batch_size, num_nodes)[span_66](start_span)[span_66](end_span)[span_67](start_span)[span_67](end_span)[span_68](start_span)[span_68](end_span).
            adj_matrix (torch.Tensor): Weighted adjacency matrix $W$ of shape (num_nodes, num_nodes)[span_69](start_span)[span_69](end_span)[span_70](start_span)[span_70](end_span)[span_71](start_span)[span_71](end_span).
            steps (int): Number of temporal simulation steps.
            noise_std (float): Standard deviation for additive stochastic noise $\eta_i(t)$[span_72](start_span)[span_72](end_span).
            autonomic_feedback (Optional[torch.Tensor]): Autonomic feedback term $R_i(t)$[span_73](start_span)[span_73](end_span).

        Returns:
            Tuple containing final updated tensors (x, y, z, s).
        """
        # Precompute degree normalization for high-performance vectorized execution ($D^{-1}W$)[span_74](start_span)[span_74](end_span)[span_75](start_span)[span_75](end_span)[span_76](start_span)[span_76](end_span)
        degrees = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        norm_adj = adj_matrix / degrees  # Shape: (num_nodes, num_nodes)
        ext_drive_divisor = degrees.T    # Shape: (1, num_nodes)

        for _ in range(steps):
            # 1. Integrate continuous subsystem via RK4[span_77](start_span)[span_77](end_span)[span_78](start_span)[span_78](end_span)[span_79](start_span)[span_79](end_span)
            x, y, z = self._rk4_step(x, y, z)

            # 2. Compute global psychological pressure P(t)[span_80](start_span)[span_80](end_span)[span_81](start_span)[span_81](end_span)[span_82](start_span)[span_82](end_span)
            P_t = self.k1 * torch.abs(x) + self.k2 * torch.abs(y) + self.k3 * (1.0 - z)

            # 3. Compute external driving projections from continuous core[span_83](start_span)[span_83](end_span)[span_84](start_span)[span_84](end_span)[span_85](start_span)[span_85](end_span)
            ext_drive = (
                self.gamma.unsqueeze(0) * x + 
                self.delta.unsqueeze(0) * y - 
                self.epsilon.unsqueeze(0) * z
            )

            # 4. Vectorized discrete network propagation[span_86](start_span)[span_86](end_span)[span_87](start_span)[span_87](end_span)[span_88](start_span)[span_88](end_span)
            neighbor_sum = torch.matmul(s, norm_adj.T)
            self_term = self.w_ii.unsqueeze(0) * s
            pressure_term = self.alpha.unsqueeze(0) * P_t
            ext_term = ext_drive / ext_drive_divisor

            activation_input = self_term + pressure_term + neighbor_sum + ext_term

            # Optional Extended Variant Components[span_89](start_span)[span_89](end_span)
            if self.tau_homeostasis != 0.0:
                activation_input = activation_input - self.tau_homeostasis * (s - self.baseline_s.unsqueeze(0))

            if autonomic_feedback is not None:
                activation_input = activation_input + autonomic_feedback

            if noise_std > 0.0 and self.training:
                noise = torch.randn_like(s) * noise_std
                activation_input = activation_input + noise

            # Synchronous tanh activation boundary[span_90](start_span)[span_90](end_span)[span_91](start_span)[span_91](end_span)[span_92](start_span)[span_92](end_span)
            s = torch.tanh(activation_input)

        return x, y, z, s
