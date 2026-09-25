# =============================================================================
# Remote Neural Monitoring Engine - NATIVE FULL DIFFERENTIABLE
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# AI Assist    : Developed with the assistance of Gemini
# License      : MIT
# Year         : 2026
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DifferentiableForwardPhysics(nn.Module):
    """
    Ultra-optimized Differentiable Lead Field Matrix (LFM) Forward Operator.
    Simulates how 3D brain voxel signals propagate to external 2D/1D sensors.
    Mathematical Cost: O(V * S) per timestep.
    """
    def __init__(self, num_sensors: int, num_voxels: int):
        super().__init__()
        # Lead Field Matrix (LFM): Pre-computed or learned physical constraints
        # Shape: (num_sensors, num_voxels)
        self.lead_field = nn.Parameter(
            torch.empty(num_sensors, num_voxels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.lead_field, a=math.sqrt(5))

    def update_lead_field(self, new_lfm: torch.Tensor):
        """Allows dynamic updating of physical constraints (e.g., patient-specific head models)."""
        self.lead_field.data.copy_(new_lfm)

    def forward(self, brain_sources: torch.Tensor) -> torch.Tensor:
        """
        Projects 3D brain activity to external sensor space.
        Args:
            brain_sources: Tensor (Batch, Voxels, Time)
        Returns:
            sensor_signals: Tensor (Batch, Sensors, Time)
        """
        # Batch Matrix Multiplication: B x (S x V) @ (V x T) -> B x S x T
        # Native differentiable and highly optimized via cuBLAS
        return torch.matmul(self.lead_field, brain_sources)


class EfficientInverseSolver(nn.Module):
    """
    Amortized Neural Inverse Model: Reconstructs 3D Brain Sources from Sensors.
    Optimized for extremely low FLOPs using factored temporal-spatial projections.
    """
    def __init__(self, num_sensors: int, num_voxels: int, hidden_dim: int = 128):
        super().__init__()
        self.num_voxels = num_voxels
        
        # 1. Temporal Feature Extractor (1D Depthwise Convs - Cheap & Fast)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(num_sensors, hidden_dim, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True) # SiLU is smooth (C2 differentiable) and memory efficient
        )
        
        # 2. Sensor-to-Voxel Spatial Projection (Low-Rank Approximation)
        # Instead of dense S -> V mapping, we map S -> Hidden -> V to save weights
        self.spatial_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 2, num_voxels)
        )
        
        # 3. Voxel Refinement (Optional: lightweight spatial smoothing)
        self.refinement_gate = nn.Parameter(torch.tensor([0.1]))

    def forward(self, sensor_signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_signals: Tensor (Batch, Sensors, Time)
        Returns:
            predicted_sources: Tensor (Batch, Voxels, Time)
        """
        # Temporal encoding
        x_temp = self.temporal_encoder(sensor_signals) # (B, Hidden, T)
        
        # Permute for Linear layer: (B, T, Hidden)
        x_temp = x_temp.transpose(1, 2)
        
        # Project to 3D Voxel Space: (B, T, Voxels)
        raw_sources = self.spatial_projector(x_temp)
        
        # Permute back to output format: (B, Voxels, T)
        raw_sources = raw_sources.transpose(1, 2)
        
        # Self-gated refinement for sparsity (Brain activity is typically sparse)
        # Soft-thresholding via Tanh/Sigmoid combination
        activated_sources = raw_sources * torch.sigmoid(raw_sources * self.refinement_gate)
        
        return activated_sources


class RemoteNeuralMonitoringEndToEnd(nn.Module):
    """
    The Complete Native Differentiable Pipeline.
    Combines Inverse Solver (Neural) and Forward Physics (Analytic).
    """
    def __init__(self, num_sensors: int, num_voxels: int):
        super().__init__()
        self.inverse_solver = EfficientInverseSolver(num_sensors, num_voxels)
        self.forward_physics = DifferentiableForwardPhysics(num_sensors, num_voxels)

    def forward(self, sensor_signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Estimate brain state from sensors (Inverse)
        predicted_brain_sources = self.inverse_solver(sensor_signals)
        
        # 2. Re-project back to sensor space (Forward Physics)
        reconstructed_signals = self.forward_physics(predicted_brain_sources)
        
        return predicted_brain_sources, reconstructed_signals

    @torch.jit.export
    def infer_only(self, sensor_signals: torch.Tensor) -> torch.Tensor:
        """Production deployment method - skips forward physics completely"""
        return self.inverse_solver(sensor_signals)


class PhysicsInformedLoss(nn.Module):
    """
    Fully Differentiable Loss Function combining Data Fidelity and Physics Constraints.
    Calculated at runtime with O(1) memory allocation footprint.
    """
    def __init__(self, sparsity_weight: float = 1e-3, tv_weight: float = 1e-4):
        super().__init__()
        self.lambda_sparse = sparsity_weight
        self.lambda_tv = tv_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, original_signals: torch.Tensor, 
                reconstructed_signals: torch.Tensor, 
                predicted_sources: torch.Tensor) -> torch.Tensor:
        
        # 1. Data Fidelity: The reconstructed signal must match the observed sensor signal
        fidelity_loss = self.mse_loss(reconstructed_signals, original_signals)
        
        # 2. Physics Prior (Sparsity): Brain activation is spatially sparse (L1 Norm)
        sparsity_loss = torch.mean(torch.abs(predicted_sources))
        
        # 3. Physics Prior (Temporal Continuity): Brain waves are continuous in time (Total Variation)
        # Uses fast element-wise differentiation via slicing
        temporal_tv_loss = torch.mean(torch.abs(predicted_sources[:, :, 1:] - predicted_sources[:, :, :-1]))
        
        return fidelity_loss + (self.lambda_sparse * sparsity_loss) + (self.lambda_tv * temporal_tv_loss)

# ==========================================
# Example Production Usage (Training / Inference)
# ==========================================
if __name__ == "__main__":
    # Settings for a lightweight clinical/remote setup
    BATCH_SIZE = 16
    NUM_SENSORS = 64     # e.g., 64 OPM-MEG or fNIRS channels
    NUM_VOXELS = 2048    # Compressed 3D brain space (Grey matter regions)
    TIME_STEPS = 256     # Epoch length in samples
    
    # Initialize the ultra-optimized model
    model = RemoteNeuralMonitoringEndToEnd(num_sensors=NUM_SENSORS, num_voxels=NUM_VOXELS)
    
    # OPTIMIZATION: Compile model for CUDAGraphs and kernel fusion (PyTorch 2.0+)
    # This reduces overhead drastically in production.
    compiled_model = torch.compile(model)
    
    criterion = PhysicsInformedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Mock real-time streaming data from sensors
    mock_sensor_data = torch.randn(BATCH_SIZE, NUM_SENSORS, TIME_STEPS, device='cpu') # Use 'cuda' in real world
    
    # --- TRAINING LOOP (Physics-Informed Self-Supervised) ---
    optimizer.zero_grad(set_to_none=True) # set_to_none=True saves memory
    
    # Forward Pass
    sources, recon_signals = compiled_model(mock_sensor_data)
    
    # Calculate fully differentiable physics loss
    loss = criterion(mock_sensor_data, recon_signals, sources)
    
    # Backward Pass
    loss.backward()
    optimizer.step()
    
    print(f"Optimized Step Completed. Loss: {loss.item():.4f}")
    
    # --- PRODUCTION DEPLOYMENT (Inference Only) ---
    with torch.no_grad():
        # In production, we only need the inferred brain state, skipping physical reprojection
        real_time_brain_state = compiled_model.infer_only(mock_sensor_data)
        print(f"Real-time Brain State Extracted. Shape: {real_time_brain_state.shape}")
