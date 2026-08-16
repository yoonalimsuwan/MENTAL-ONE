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
        
        # Linear projection to unified dimension with standard normalization
        projected = self.projections[provider](hidden_states)
        return F.layer_norm(projected, (self.target_dim,))


class FullyDifferentiableMindInterpreter(nn.Module):
    """
    Native Fully Differentiable Mental Health and Concentration Interpreter module.
    Based on Yoniso Manasikara, 52 Cetasikas, 5 Nivarana, and 4 Meditation Models.
    Optimized with zero memory overhead and tensorized parallel linear projections.
    """
    def __init__(self, input_dim: int = 768):
        super(FullyDifferentiableMindInterpreter, self).__init__()
        
        self.input_dim = input_dim
        self.adapter = MultiLLMProviderAdapter(target_dim=input_dim)
        
        # 1. Fully Differentiable Projections for Mental Factors (Cetasika 52)
        self.akusala_proj = nn.Linear(input_dim, 14, bias=True) # 14 Unwholesome states
        self.sobhana_proj = nn.Linear(input_dim, 25, bias=True) # 25 Wholesome states
        
        # 2. Five Hindrances (Nivarana 5) Projection
        self.nivarana_proj = nn.Linear(input_dim, 5, bias=True)
        
        # 3. Ten Modes of Yoniso Manasikara Projection
        self.yoniso_proj = nn.Linear(input_dim, 10, bias=True)
        
        # 4. Concentration State Classifier (3 Primary Levels: Parikamma, Upacara, Appana)
        self.samadhi_level_classifier = nn.Linear(input_dim, 3, bias=True)
        
        # 5. Seven Factors of Enlightenment (Bojjhanga 7) Evaluator
        self.bojjhanga_proj = nn.Linear(input_dim, 7, bias=True)
        
        # Learnable Temperature parameter for smooth softmax differentiability
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, 
        embeddings: torch.Tensor, 
        provider: str = "deepseek",
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings (torch.Tensor): Input hidden states [Batch_Size, Seq_Len, Dim] or [Batch_Size, Dim]
            provider (str): LLM Platform Name ('gemini', 'gpt', 'kimi', 'deepseek', 'claude', 'qwen', 'glm', 'copilot', 'grok')
            attention_mask (torch.Tensor, optional): Mask for sequence pooling [Batch_Size, Seq_Len]
            
        Returns:
            Dict containing fully differentiable metrics and scores.
        """
        # Adapt provider-specific embeddings to unified dimension
        x = self.adapter(embeddings, provider=provider)
        
        # Pool sequence dimension if 3D tensor is passed (Mean Pooling with Masking)
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
            
        # --- A. Cetasika Activation & Mental Health Index ---
        akusala_logits = torch.sigmoid(self.akusala_proj(pooled))
        sobhana_logits = torch.sigmoid(self.sobhana_proj(pooled))
        
        unwholesome_intensity = torch.mean(akusala_logits, dim=-1, keepdim=True)
        wholesome_intensity = torch.mean(sobhana_logits, dim=-1, keepdim=True)
        
        mental_health_score = wholesome_intensity / (wholesome_intensity + unwholesome_intensity + 1e-8)
        
        # --- B. Five Hindrances (Nivarana) Detection ---
        nivarana_vector = torch.sigmoid(self.nivarana_proj(pooled))
        nivarana_obstacle_level = torch.mean(nivarana_vector, dim=-1, keepdim=True)
        
        # --- C. Yoniso Manasikara (10 Rational Thinking Modes) ---
        yoniso_vector = torch.sigmoid(self.yoniso_proj(pooled))
        yoniso_score = torch.mean(yoniso_vector, dim=-1, keepdim=True)
        
        # --- D. Concentration (Samadhi & Bojjhanga) Level ---
        bojjhanga_vector = torch.sigmoid(self.bojjhanga_proj(pooled))
        bojjhanga_score = torch.mean(bojjhanga_vector, dim=-1, keepdim=True)
        
        samadhi_logits = self.samadhi_level_classifier(pooled) / torch.clamp(self.temperature, min=0.1)
        samadhi_stage_prob = F.softmax(samadhi_logits, dim=-1)
        
        concentration_score = (
            0.4 * bojjhanga_score + 
            0.3 * yoniso_score + 
            0.3 * (1.0 - nivarana_obstacle_level)
        ).clamp(0.0, 1.0)
        
        return {
            "mental_health_score": mental_health_score,
            "concentration_score": concentration_score,
            "samadhi_stage_prob": samadhi_stage_prob,
            "nivarana_vector": nivarana_vector,
            "yoniso_score": yoniso_score,
            "akusala_intensity": unwholesome_intensity,
            "sobhana_intensity": wholesome_intensity
        }


# --- Example Execution and Verification ---
if __name__ == "__main__":
    # Initialize interpreter model
    interpreter = FullyDifferentiableMindInterpreter(input_dim=768)
    interpreter.eval()

    # 1. Test with Google Gemini Hidden States (Batch size = 2, Seq len = 32, Dim = 3072)
    gemini_input = torch.randn(2, 32, 3072)
    gemini_mask = torch.ones(2, 32)
    outputs_gemini = interpreter(gemini_input, provider="gemini", attention_mask=gemini_mask)
    
    print("--- Gemini Provider Test ---")
    print("Mental Health Score:", outputs_gemini["mental_health_score"].item())
    print("Concentration Score:", outputs_gemini["concentration_score"].item())
    print("Samadhi Stage Probabilities:", outputs_gemini["samadhi_stage_prob"].detach().numpy())

    # 2. Test with OpenAI GPT Hidden States (Batch size = 1, Dim = 3072)
    gpt_input = torch.randn(1, 3072)
    outputs_gpt = interpreter(gpt_input, provider="gpt")
    
    print("\n--- GPT Provider Test ---")
    print("Mental Health Score:", outputs_gpt["mental_health_score"].item())
    print("Concentration Score:", outputs_gpt["concentration_score"].item())
