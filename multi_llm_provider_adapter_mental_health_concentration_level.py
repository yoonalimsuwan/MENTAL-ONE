import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class MultiLLMProviderAdapter(nn.Module):
    """
    Unified, ultra-lightweight differentiable embedding projection layer 
    supporting multiple LLM provider embedding spaces (Gemini, GPT, Kimi, DeepSeek, Claude, Qwen, GLM, Copilot, Grok).
    """
    def __init__(self, target_dim: int = 768):
        super(MultiLLMProviderAdapter, self).__init__()
        self.target_dim = target_dim
        
        # Provider-specific hidden dimensions mapping
        self.provider_dims = {
            "gemini": 3072,     # Google Gemini Pro / Embedding API standard hidden dimension
            "gpt": 3072,        # OpenAI GPT-4o / text-embedding-3-large dimension
            "kimi": 4096,       # Moonshot/Kimi embeddings
            "deepseek": 5120,   # DeepSeek-V3/R1 latent dimension
            "claude": 1536,     # Anthropic Claude embedding interface projection
            "qwen": 3584,       # Qwen-2.5 standard hidden projection
            "glm": 4096,        # ChatGLM/GLM-4 native layer
            "copilot": 1536,    # Microsoft/OpenAI Copilot standard hidden state
            "grok": 8192        # xAI Grok high-dimensional embedding interface
        }
        
        self.projections = nn.ModuleDict({
            provider: nn.Linear(in_dim, target_dim, bias=False)
            for provider, in_dim in self.provider_dims.items()
        })

    def forward(self, hidden_states: torch.Tensor, provider: str) -> torch.Tensor:
        provider = provider.lower()
        if provider not in self.projections:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.projections.keys())}")
        
        # Linear projection to unified dimension with standard layer normalization
        projected = self.projections[provider](hidden_states)
        return F.layer_norm(projected, (self.target_dim,))


class FullyDifferentiableMindInterpreter(nn.Module):
    """
    Native Fully Differentiable Mind & Concentration Interpreter Module.
    Encapsulates 5 Core Theoretical Frameworks:
    1. Cetasika 52 (14 Akusala, 13 Annasamana, 25 Sobhana)
    2. Nivarana 5 (Five Hindrances)
    3. Yoniso Manasikara 10 Modes (Rational Thinking)
    4. Samadhi State Classifier 3 Levels (Parikamma, Upacara, Appana)
    5. Bojjhanga 7 (Seven Factors of Enlightenment)
    """
    def __init__(self, input_dim: int = 768):
        super(FullyDifferentiableMindInterpreter, self).__init__()
        
        self.input_dim = input_dim
        self.adapter = MultiLLMProviderAdapter(target_dim=input_dim)
        
        # --- Pillar 1: Cetasika 52 Framework ---
        # Categorized into standard Abhidhamma mental factors
        self.akusala_proj = nn.Linear(input_dim, 14, bias=True)     # 14 Unwholesome states (e.g., Greed, Hatred, Delusion, Sloth/Torpor)
        self.annasamana_proj = nn.Linear(input_dim, 13, bias=True)  # 13 Neutral/Common states (e.g., Feeling, Perception, Attention)
        self.sobhana_proj = nn.Linear(input_dim, 25, bias=True)     # 25 Wholesome states (e.g., Loving-kindness, Compassion, Mindfulness, Wisdom)
        
        # --- Pillar 2: Five Hindrances (Nivarana) ---
        # Mental obstacles blocking concentration and mental clarity:
        # [0]: Sensory Desire, [1]: Ill Will, [2]: Sloth & Torpor, [3]: Restlessness & Worry, [4]: Doubt
        self.nivarana_proj = nn.Linear(input_dim, 5, bias=True)
        
        # --- Pillar 3: Yoniso-Manasikara Cognition Modes (10 Modes) ---
        # Systematic and wise attention modes (e.g., Cause-and-effect reflection, Four Noble Truths framework, Present-moment awareness)
        self.yoniso_proj = nn.Linear(input_dim, 10, bias=True)
        
        # --- Pillar 4: Samadhi State Classification (3 Levels) ---
        # [0]: Preliminary Concentration (Parikamma), 
        # [1]: Access Concentration (Upacara - Hindrances suppressed), 
        # [2]: Absorption Concentration (Appana - Deep Jhanic absorption)
        self.samadhi_level_classifier = nn.Linear(input_dim, 3, bias=True)
        
        # --- Pillar 5: Bojjhanga 7 (Seven Factors of Enlightenment) ---
        # Factors leading to awakening: [0]: Mindfulness, [1]: Investigation of States, [2]: Energy, 
        # [3]: Joy, [4]: Tranquility, [5]: Concentration, [6]: Equanimity
        self.bojjhanga_proj = nn.Linear(input_dim, 7, bias=True)
        
        # Learnable temperature parameter for smooth softmax differentiability
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, 
        embeddings: torch.Tensor, 
        provider: str = "gemini",
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass processing hidden states through all 5 psychological and spiritual pillars.
        """
        # 1. Adapt provider-specific embeddings to the unified dimension space
        x = self.adapter(embeddings, provider=provider)
        
        # 2. Sequence Pooling (Mean Pooling with masking support if 3D tensor is passed)
        if x.dim() == 3:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).float()
                sum_embeddings = torch.sum(x * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(x, dim=1)
        else:
            pooled = x
            
        # --- Pillar 1: Cetasika 52 & Mental Health Index ---
        akusala_logits = torch.sigmoid(self.akusala_proj(pooled))
        annasamana_logits = torch.sigmoid(self.annasamana_proj(pooled))
        sobhana_logits = torch.sigmoid(self.sobhana_proj(pooled))
        
        unwholesome_intensity = torch.mean(akusala_logits, dim=-1, keepdim=True)
        wholesome_intensity = torch.mean(sobhana_logits, dim=-1, keepdim=True)
        neutral_intensity = torch.mean(annasamana_logits, dim=-1, keepdim=True)
        
        # Differentiable Mental Health Score based on Wholesome vs. Unwholesome ratio
        mental_health_score = wholesome_intensity / (wholesome_intensity + unwholesome_intensity + 1e-8)
        
        # --- Pillar 2: Five Hindrances (Nivarana) ---
        nivarana_vector = torch.sigmoid(self.nivarana_proj(pooled))
        nivarana_obstacle_level = torch.mean(nivarana_vector, dim=-1, keepdim=True)
        
        # --- Pillar 3: Yoniso Manasikara (10 Rational Thinking Modes) ---
        yoniso_vector = torch.sigmoid(self.yoniso_proj(pooled))
        yoniso_score = torch.mean(yoniso_vector, dim=-1, keepdim=True)
        
        # --- Pillar 5: Bojjhanga 7 (Enlightenment Factors) ---
        bojjhanga_vector = torch.sigmoid(self.bojjhanga_proj(pooled))
        bojjhanga_score = torch.mean(bojjhanga_vector, dim=-1, keepdim=True)
        
        # --- Pillar 4: Samadhi Stage Classification (3 Levels) ---
        samadhi_logits = self.samadhi_level_classifier(pooled) / torch.clamp(self.temperature, min=0.1)
        samadhi_stage_prob = F.softmax(samadhi_logits, dim=-1)
        
        # Composite Concentration Level Score (CLS)
        # Combined weight: 40% Bojjhanga + 30% Yoniso Rational Mind + 30% Inverse Nivarana Obstacles
        concentration_score = (
            0.4 * bojjhanga_score + 
            0.3 * yoniso_score + 
            0.3 * (1.0 - nivarana_obstacle_level)
        ).clamp(0.0, 1.0)
        
        return {
            "mental_health_score": mental_health_score,
            "concentration_score": concentration_score,
            "samadhi_stage_prob": samadhi_stage_prob,       # [Parikamma, Upacara, Appana]
            "nivarana_vector": nivarana_vector,             # 5 Hindrances breakdown
            "yoniso_vector": yoniso_vector,                 # 10 Rational thinking modes breakdown
            "bojjhanga_vector": bojjhanga_vector,           # 7 Enlightenment factors breakdown
            "akusala_intensity": unwholesome_intensity,     # 14 Akusala overall level
            "sobhana_intensity": wholesome_intensity        # 25 Sobhana overall level
        }


# --- Execution and Verification Test ---
if __name__ == "__main__":
    # Initialize interpreter model
    interpreter = FullyDifferentiableMindInterpreter(input_dim=768)
    interpreter.eval()

    # Test with Google Gemini Hidden States (Batch size = 2, Seq len = 32, Dim = 3072)
    gemini_input = torch.randn(2, 32, 3072)
    gemini_mask = torch.ones(2, 32)
    outputs = interpreter(gemini_input, provider="gemini", attention_mask=gemini_mask)
    
    print("=== Execution Test Results (Google Gemini Provider) ===")
    print("Mental Health Score:", outputs["mental_health_score"].detach().numpy())
    print("Concentration Score:", outputs["concentration_score"].detach().numpy())
    print("Samadhi Stage Probabilities (Parikamma, Upacara, Appana):", outputs["samadhi_stage_prob"].detach().numpy())
    print("Nivarana Vector (5 Hindrances):", outputs["nivarana_vector"].detach().numpy())
