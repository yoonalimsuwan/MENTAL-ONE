import torch
import torch.nn as nn
from typing import Tuple, Optional

class CombinedPsychopathologyEngine(nn.Module):
    """
    A Production-Grade, Native Fully Differentiable Combined Continuous-Discrete 
    Dynamical Framework for Modeling Psychopathology.
    
    * Note: This program was written with the assistance of Gemini. *
    """
    def __init__(self, num_nodes: int, dt: float = 0.01):
        super(CombinedPsychopathologyEngine, self).__init__()
        self.num_nodes = num_nodes
        self.dt = dt

        # ---------------------------------------------------------
        # Continuous Lorenz Core Parameters[span_9](start_span)[span_9](end_span)
        # ---------------------------------------------------------
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_lorenz = nn.Parameter(torch.tensor(8.0 / 3.0))

        # ---------------------------------------------------------
        # Bridge / Coupling Parameters[span_10](start_span)[span_10](end_span)
        # ---------------------------------------------------------
        self.kappa1 = nn.Parameter(torch.tensor(1.0))
        self.kappa2 = nn.Parameter(torch.tensor(1.0))
        self.kappa3 = nn.Parameter(torch.tensor(1.0))

        self.gamma = nn.Parameter(torch.randn(num_nodes))
        self.delta = nn.Parameter(torch.randn(num_nodes))
        self.epsilon = nn.Parameter(torch.randn(num_nodes))

        # ---------------------------------------------------------
        # Discrete Network Parameters[span_11](start_span)[span_11](end_span)
        # ---------------------------------------------------------
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Global pressure sensitivity[span_12](start_span)[span_12](end_span)
        self.beta_net = nn.Parameter(torch.tensor(1.0)) # Coupling strength between nodes[span_13](start_span)[span_13](end_span)

        # Extended Variant Parameters (Homeostasis & Baseline)[span_14](start_span)[span_14](end_span)
        self.tau_homeostasis = nn.Parameter(torch.tensor(0.0))
        self.baseline_s = nn.Parameter(torch.zeros(num_nodes))

    def _lorenz_derivatives(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes derivatives for the continuous cognitive-affective-identity core[span_15](start_span)[span_15](end_span)."""
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta_lorenz * z
        return dx, dy, dz

    def _rk4_step(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized 4th-order Runge-Kutta (RK4) integration step for stability[span_16](start_span)[span_16](end_span)."""
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
        Executes forward simulation over multiple temporal steps. Fully differentiable 
        to support end-to-end backpropagation and parameter optimization[span_17](start_span)[span_17](end_span).

        Args:
            x, y, z (torch.Tensor): Continuous core variables of shape (batch_size, 1)[span_18](start_span)[span_18](end_span).
            s (torch.Tensor): Discrete network state tensor of shape (batch_size, num_nodes)[span_19](start_span)[span_19](end_span).
            adj_matrix (torch.Tensor): Network adjacency matrix $A$ of shape (num_nodes, num_nodes)[span_20](start_span)[span_20](end_span).
            steps (int): Number of time steps to simulate.
            noise_std (float): Standard deviation for optional stochastic noise $\eta_i(t)$[span_21](start_span)[span_21](end_span).
            autonomic_feedback (Optional[torch.Tensor]): Optional autonomic feedback term $R_i(t)$[span_22](start_span)[span_22](end_span).

        Returns:
            Tuple of updated tensors (x, y, z, s).
        """
        # Precompute degree normalization ($D^{-1}A$) for maximum optimization efficiency[span_23](start_span)[span_23](end_span)
        degrees = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        norm_adj = adj_matrix / degrees  # Shape: (num_nodes, num_nodes)
        ext_divisor = degrees.T          # Shape: (1, num_nodes)

        for _ in range(steps):
            # 1. Integrate continuous Lorenz core via RK4[span_24](start_span)[span_24](end_span)
            x, y, z = self._rk4_step(x, y, z)

            # 2. Compute global psychological pressure P(t)[span_25](start_span)[span_25](end_span)
            P_t = self.kappa1 * torch.abs(x) + self.kappa2 * torch.abs(y) + self.kappa3 * (1.0 - z)

            # 3. Compute external driving projections from continuous variables[span_26](start_span)[span_26](end_span)
            ext_drive = (
                self.gamma.unsqueeze(0) * x + 
                self.delta.unsqueeze(0) * y - 
                self.epsilon.unsqueeze(0) * z
            )

            # 4. Vectorized optimized discrete network propagation update[span_27](start_span)[span_27](end_span)
            neighbor_sum = torch.matmul(s, norm_adj.T)
            ext_drive_norm = ext_drive / ext_divisor

            activation_input = (
                s + 
                self.alpha * P_t + 
                self.beta_net * neighbor_sum + 
                ext_drive_norm
            )

            # Optional Extended Variant Components (Homeostasis & Noise)[span_28](start_span)[span_28](end_span)
            if self.tau_homeostasis != 0.0:
                activation_input = activation_input - self.tau_homeostasis * (s - self.baseline_s.unsqueeze(0))

            if autonomic_feedback is not None:
                activation_input = activation_input + autonomic_feedback

            if noise_std > 0.0 and self.training:
                noise = torch.randn_like(s) * noise_std
                activation_input = activation_input + noise

            # Bounded tanh activation function[span_29](start_span)[span_29](end_span)
            s = torch.tanh(activation_input)

        return x, y, z, s
