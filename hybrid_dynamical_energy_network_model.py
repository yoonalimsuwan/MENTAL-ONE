import torch
import torch.nn as nn

class PsychoDynamoNet(nn.Module):
    """
    A Native Fully Differentiable Hybrid Dynamical Energy-Network Model.
    Integrates continuous Lorenz-style cognitive-affective-identity dynamics 
    with a discrete symptom network.
    
    * Note: This program was written with the assistance of Gemini. *
    """
    def __init__(self, num_nodes: int, dt: float = 0.01):
        """
        Args:
            num_nodes (int): Number of symptom nodes in the discrete network.
            dt (float): Step size for continuous numerical integration.
        """
        super(PsychoDynamoNet, self).__init__()
        self.num_nodes = num_nodes
        self.dt = dt

        # ---------------------------------------------------------
        # Continuous Lorenz Core Parameters (Cognition, Affect, Identity)
        # ---------------------------------------------------------
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_lorenz = nn.Parameter(torch.tensor(8.0 / 3.0))

        # ---------------------------------------------------------
        # Coupling Parameters (Core to Global Pressure)
        # ---------------------------------------------------------
        self.k1 = nn.Parameter(torch.tensor(1.0))
        self.k2 = nn.Parameter(torch.tensor(1.0))
        self.k3 = nn.Parameter(torch.tensor(1.0))

        # ---------------------------------------------------------
        # External Driving Parameters (Node-Specific)
        # ---------------------------------------------------------
        self.gamma = nn.Parameter(torch.randn(num_nodes))
        self.delta = nn.Parameter(torch.randn(num_nodes))
        self.epsilon = nn.Parameter(torch.randn(num_nodes))

        # ---------------------------------------------------------
        # Discrete Network Parameters
        # ---------------------------------------------------------
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta_net = nn.Parameter(torch.tensor(1.0))

    def _lorenz_derivatives(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """Calculates the gradients for the Lorenz subsystem."""
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta_lorenz * z
        return dx, dy, dz

    def _rk4_step(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        Highly optimized 4th-order Runge-Kutta (RK4) integration step.
        Fully vectorized to avoid Python loop overhead.
        """
        dt2 = self.dt / 2.0

        k1_x, k1_y, k1_z = self._lorenz_derivatives(x, y, z)
        k2_x, k2_y, k2_z = self._lorenz_derivatives(x + dt2 * k1_x, y + dt2 * k1_y, z + dt2 * k1_z)
        k3_x, k3_y, k3_z = self._lorenz_derivatives(x + dt2 * k2_x, y + dt2 * k2_y, z + dt2 * k2_z)
        k4_x, k4_y, k4_z = self._lorenz_derivatives(x + self.dt * k3_x, y + self.dt * k3_y, z + self.dt * k3_z)

        x_next = x + (self.dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y_next = y + (self.dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        z_next = z + (self.dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z)

        return x_next, y_next, z_next

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
                s: torch.Tensor, adj_matrix: torch.Tensor, steps: int = 1):
        """
        Forward pass for the combined Continuous-Discrete framework.

        Args:
            x, y, z (torch.Tensor): Continuous states of shape (batch_size, 1).
            s (torch.Tensor): Discrete network states of shape (batch_size, num_nodes).
            adj_matrix (torch.Tensor): Graph adjacency matrix of shape (num_nodes, num_nodes).
            steps (int): Number of time steps to simulate.

        Returns:
            Tuple containing the updated states (x, y, z, s).
        """
        # Precompute normalized adjacency matrix (Random Walk matrix) D^-1 * A
        # This replaces heavy iterative division during the update step.
        degrees = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        norm_adj = adj_matrix / degrees 

        for _ in range(steps):
            # 1. Update continuous Lorenz core (Cognitive-Affective-Identity)
            x, y, z = self._rk4_step(x, y, z)

            # 2. Compute Global Psychological Pressure P(t)
            # P(t) = k1*|x| + k2*|y| + k3*(1 - z)
            P_t = self.k1 * torch.abs(x) + self.k2 * torch.abs(y) + self.k3 * (1.0 - z)

            # 3. Compute External Driving per node
            # External_i(t) = gamma_i*x + delta_i*y - epsilon_i*z
            ext_drive = (self.gamma.unsqueeze(0) * x + 
                         self.delta.unsqueeze(0) * y - 
                         self.epsilon.unsqueeze(0) * z)

            # 4. Synchronous Discrete Network Update (tanh activation)
            # Optimized algebraic formulation:
            # s(t+1) = tanh( s(t) + alpha*P(t) + beta*(norm_adj @ s(t)) + (ext_drive / degrees) )
            
            neighbor_sum = torch.matmul(s, norm_adj.T) 
            ext_drive_norm = ext_drive / degrees.T

            s = torch.tanh(
                s + 
                self.alpha * P_t + 
                self.beta_net * neighbor_sum + 
                ext_drive_norm
            )

        return x, y, z, s
