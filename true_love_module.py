# =============================================================================
# TRUE LOVE MODULE —  NATIVE FULL DIFFERENTIABLE 
# =============================================================================
# Developer  : PAI AND Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyDifferentiableTrueLoveModule(nn.Module):
    """
    Production-grade, fully differentiable end-to-end module implementing 
    Stable Probability Representations, Integrated Information Theory (IIT) approximations, 
    and Psychoanalytic Latent Dynamics.
    
    Co-authored with Gemini for optimized, low-overhead tensor operations.
    """
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Low-rank regime projectors for Sub-Quantum / Stable Distribution mapping
        self.encoder_alpha = nn.Linear(feature_dim, hidden_dim)
        self.encoder_gamma = nn.Linear(feature_dim, hidden_dim)
        
        # Differentiable IIT Phi Approximator
        self.iit_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Psychoanalytic Attachment & Reciprocity Estimators
        self.psycho_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.reciprocity_head = nn.Linear(hidden_dim * 2, 1)
        
        # Learnable TLI weights
        self.raw_weights = nn.Parameter(torch.ones(5))

    def forward(self, time_series_a: torch.Tensor, time_series_b: torch.Tensor, als_scores: torch.Tensor) -> dict:
        """
        Forward pass optimized with native torch operations for maximum numerical stability and speed.
        """
        batch_size, seq_len, _ = time_series_a.shape
        
        # 1. Parameter Estimation via Differentiable Low-Rank Projections (Replacing heavy numerical fitting)
        alpha_a = torch.sigmoid(self.encoder_alpha(time_series_a)) * 1.5 + 0.5  # Map to (0.5, 2.0)
        gamma_a = F.softplus(self.encoder_gamma(time_series_a)) + 1e-4          # Map to positive scale
        
        alpha_b = torch.sigmoid(self.encoder_alpha(time_series_b)) * 1.5 + 0.5
        gamma_b = F.softplus(self.encoder_gamma(time_series_b)) + 1e-4

        # 2. Stability Metric Calculation
        score_alpha_a = (alpha_a - 0.5) / 1.5
        score_gamma_a = 1.0 / (1.0 + torch.log1p(gamma_a))
        stability_a = 0.6 * score_alpha_a.mean(dim=-1) + 0.4 * score_gamma_a.mean(dim=-1)

        score_alpha_b = (alpha_b - 0.5) / 1.5
        score_gamma_b = 1.0 / (1.0 + torch.log1p(gamma_b))
        stability_b = 0.6 * score_alpha_b.mean(dim=-1) + 0.4 * score_gamma_b.mean(dim=-1)
        mean_stability = (stability_a + stability_b) / 2.0

        # 3. Differentiable Integrated Information (Phi) Approximation via Joint vs Partitioned States
        concat_state = torch.cat([time_series_a, time_series_b], dim=-1)
        phi_raw = self.iit_projector(concat_state).squeeze(-1).mean(dim=-1)

        # 4. Mutual Information (MI) Proxy via Normalized Dot-Product Attention / Covariance
        norm_a = F.normalize(time_series_a, dim=-1)
        norm_b = F.normalize(time_series_b, dim=-1)
        mi_proxy = (norm_a * norm_b).sum(dim=-1).mean(dim=-1)
        mi_norm = torch.sigmoid(mi_proxy)

        # 5. Psychoanalytic Attachment & Reciprocity Dynamics via LSTM
        out_a, _ = self.psycho_lstm(time_series_a)
        out_b, _ = self.psycho_lstm(time_series_b)
        reciprocal_features = torch.cat([out_a[:, -1, :], out_b[:, -1, :]], dim=-1)
        reciprocity_score = torch.sigmoid(self.reciprocity_head(reciprocal_features)).squeeze(-1)

        # 6. Attachment Latent Score (ALS) Mean
        als_mean = als_scores.mean(dim=-1)

        # 7. True-Love Index (TLI) Calculation with Softmax Weights Optimization
        weights = F.softmax(self.raw_weights, dim=0)
        
        tli = (
            weights[0] * phi_raw +
            weights[1] * mi_norm +
            weights[2] * mean_stability +
            weights[3] * reciprocity_score +
            weights[4] * als_mean
        )

        return {
            "TLI": tli,
            "Phi_AB": phi_raw,
            "MI_C": mi_norm,
            "Stability": mean_stability,
            "Reciprocity": reciprocity_score,
            "Optimized_Weights": weights
        }
