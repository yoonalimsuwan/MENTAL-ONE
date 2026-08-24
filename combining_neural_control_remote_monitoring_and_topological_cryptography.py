


import torch
import torch.nn as nn
from typing import Tuple, Dict

# Importing from the provided reference files
from remote_neural_control_engine import RemoteNeuralControlEngine
from bidirectional_neuro_electromagnetic_interactions import RemoteNeuroMonitorBridge
from cryptographic_and_common_factor import PostPNPCryptography, DeepCommonFactorExtractor

class SecureBidirectionalNeuralInterface(nn.Module):
    """
    Unified Bridge Module connecting Remote Control, Monitoring, and Cryptography.
    """
    def __init__(
        self, 
        grid_shape: Tuple[int, int, int], 
        crypto_dim: int, 
        dx: float = 1.0, 
        dt: float = 0.001, 
        target_freq_hz: float = 2.4e9,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        
        # 1. Control: Synthesizes optimized EM fields
        self.controller = RemoteNeuralControlEngine(grid_shape=grid_shape, dx=dx, dt=dt, device=device)
        
        # 2. Monitor: Handles tissue EM propagation and remote scatter monitoring
        self.monitor = RemoteNeuroMonitorBridge(grid_shape=grid_shape, dx=dx, dt=dt, target_freq_hz=target_freq_hz, device=device)
        
        # 3. Cryptography & Invariance: Secures states and extracts common factors
        self.crypto = PostPNPCryptography(dim=crypto_dim)
        self.scf_extractor = DeepCommonFactorExtractor(n_dim=crypto_dim)

    def forward(
        self, 
        ambient_e: torch.Tensor, 
        ambient_b: torch.Tensor, 
        tissue_phase: torch.Tensor, 
        ion_coords: torch.Tensor, 
        ion_vel: torch.Tensor, 
        current_time: float, 
        ion_forces_fn: callable,
        delta_t: torch.Tensor, 
        delta_e_min: torch.Tensor,
        signature_a: torch.Tensor, 
        signature_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        # A. Synthesize and superimpose control fields
        control_e, control_b = self.controller._generate_control_fields()
        e_total = ambient_e + control_e
        b_total = ambient_b + control_b
        
        # B. Extract shared invariant subspace for the topological states
        shared_subspace = self.scf_extractor(signature_a, signature_b)
        
        # C. Secure the neural tissue phase using topological trapdoors
        secured_tissue_phase = self.crypto(tissue_phase, delta_t, delta_e_min)
        
        # D. Execute the coupled neuro-electromagnetic monitoring step
        e_next, b_next, u_adapted, ion_coords_next, ion_vel_next, remote_phasors = self.monitor(
            e_field=e_total,
            b_field=b_total,
            tissue_phase=secured_tissue_phase,
            ion_coords=ion_coords,
            ion_vel=ion_vel,
            current_time=current_time,
            ion_forces_fn=ion_forces_fn
        )
        
        return u_adapted, ion_coords_next, remote_phasors, shared_subspace
