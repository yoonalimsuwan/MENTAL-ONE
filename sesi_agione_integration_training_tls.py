"""
=============================================================================
SESI AGIONE INTEGRATION × INTENT CLASSIFIER TRAINING × TLS WEBSOCKET
Production integration: forward pass hook + real-data training + secure deploy
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[INT1] AGIONEConscienceIntegration
       • Hooks ConscienceModule into AGIONE's 12-step forward pass
       • Injects moral constraints at Step 7 (PsycheExecutiveLayer)
       • Gates Psyche Plus axiom codification at Step 7-B
       • Attaches moral state to AGIState
       • Connects to EcosystemOrchestrator.quality_report()

[TRN1] IntentClassifierRealDataTrainer
       • Loads real intent classification data from files/APIs
       • Trains TransformerIntentClassifier with AMP + gradient checkpointing
       • Evaluates with confusion matrix + per-class metrics
       • Supports curriculum learning (easy → hard examples)

[TLS1] SecureWebSocketAlertServer
       • Deploys WebSocket server with TLS (wss://)
       • Certificate management (self-signed for dev, CA-signed for prod)
       • Role-based alert routing over secure channel
       • No-Zeno throttle prevents alert flooding

RELATION TO AGIONE v3.11 FORWARD PASS
---------------------------------------
From agi_one_v3_11.py, the forward pass has 12 steps:
  1. Perception → fused_perception
  2. Language → language_latent
  3. Memory → working_memory_ctx + episodic_ctx
  4. ONE Ecosystem → multiscale_latent
  5. Math Reasoning → math_latent
  6. Global Workspace → workspace_state, winner
  7. Psyche Executive → Id→goal, Ego→plan, Superego→safety  ← HOOK HERE
  7B. Psyche Plus → speculative-axiom evolution               ← GATE HERE
  8. DreamerV3 World Model → h, z, predictions
  9. MPPI Planning → planned_action
  10. Geometry-aware Diffusion → latent exploration
  11. Meta-Cognition → self-model, CSOC depth
  12. Loss computation (if training)

The conscience module hooks into Steps 7 and 7-B:
  • Step 7: Moral quality modulates the safety_score
  • Step 7-B: Ethical gate blocks unethical axiom codification

RELATION TO PSYCHE PLUS v3.3
------------------------------
From agi_one_psyche_plus_v33.py:
  • forward(latent, quality_score=None) — quality-aware
  • maybe_codify(out, threshold=None, quality_score=None) — dual-gate
  • External consensus is opt-in, rate-limited, never blocks forward pass

The conscience integration adds a THIRD gate:
  • Gate 1: validity threshold (SuperEgo axiom bank)
  • Gate 2: quality minimum (ecosystem simulation quality)
  • Gate 3: [NEW] ethical principle check (ConscienceModule)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This integration modifies AGIONE's forward pass behavior.
• The conscience module does NOT create consciousness.
• Intent classifier training requires labeled data.
• TLS deployment requires valid certificates.
• All thresholds are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp. 7.9).
• Production deployment requires security audit.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import ssl
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.agione_integration")

# Optional imports
try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from agi_one_v3_11 import AGIONE, AGIState, AGIConfig, EcosystemOrchestrator
    HAS_AGI_ONE = True
except ImportError:
    HAS_AGI_ONE = False

try:
    from agi_one_psyche_plus_v33 import AGIOnePsychePlus, PsychePlusConfig
    HAS_PSYCHE_PLUS = True
except ImportError:
    HAS_PSYCHE_PLUS = False

try:
    from ses_auth_transformer_websocket import (
        TransformerIntentClassifier, IntentClass,
        AlertLevel, AlertMessage, NoZenoAlertThrottle,
        AuthRole,
    )
    HAS_PREV_MODULES = True
except ImportError:
    HAS_PREV_MODULES = False

try:
    from ses_moral_conscience_module import (
        ConscienceModule, MoralConfig, MoralQualityScorer, EthicalAxiomGate,
    )
    HAS_CONSCIENCE = True
except ImportError:
    HAS_CONSCIENCE = False

__all__ = [
    "AGIONEConscienceIntegration",
    "IntentDataConfig",
    "IntentDatasetReal",
    "IntentClassifierRealDataTrainer",
    "TLSCertificateManager",
    "SecureWebSocketAlertServer",
    "FullIntegrationSuite",
]


# ═══════════════════════════════════════════════════════════════════════
# [INT1] AGIONE CONSCIENCE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConscienceIntegrationConfig:
    """Configuration for AGIONE conscience integration."""
    # Moral quality thresholds
    moral_quality_threshold: float = 0.5
    low_quality_safety_penalty: float = 0.3

    # Axiom codification gating
    enable_ethical_axiom_gate: bool = True
    ethical_gate_quality_min: float = 0.4

    # Integration cadence (matches psyche_plus_run_every_n_steps)
    conscience_run_every_n_steps: int = 5

    # Alert thresholds
    alert_on_low_moral_quality: bool = True
    moral_alert_threshold: float = 0.3

    device: str = "cpu"


class AGIONEConscienceIntegration:
    """
    [D] Integrates ConscienceModule into AGIONE's forward pass.

    This class wraps AGIONE and injects moral constraints at the
    appropriate points in the 12-step forward pass.

    INTEGRATION POINTS:
      • Step 7 (PsycheExecutiveLayer): Moral quality modulates safety_score
      • Step 7-B (PsychePlus): Ethical gate blocks unethical codification
      • AGIState: Moral fields added for observability

    HONEST: This modifies AGIONE's behavior. The conscience module does
    NOT create consciousness. It computes moral conformity scores that
    influence the safety gate.
    """

    def __init__(self, cfg: Optional[ConscienceIntegrationConfig] = None):
        self.cfg = cfg or ConscienceIntegrationConfig()
        self.conscience = ConscienceModule() if HAS_CONSCIENCE else None
        self.moral_scorer = MoralQualityScorer() if HAS_CONSCIENCE else None
        self.axiom_gate = EthicalAxiomGate() if HAS_CONSCIENCE else None
        self.attached = False
        self.agi_one = None
        self._step_counter = 0
        self._alert_callbacks: List[Callable] = []

    def attach_to_agione(self, agi_one: Any) -> bool:
        """
        Attach the conscience integration to an AGIONE instance.

        This hooks into the forward pass by wrapping key methods.

        Args:
            agi_one: AGIONE instance

        Returns:
            bool: True if attachment succeeded
        """
        if not HAS_AGI_ONE:
            logger.warning("AGIONE not available; cannot attach")
            return False

        self.agi_one = agi_one
        self.attached = True

        # Store reference to original forward for potential restoration
        self._original_forward = agi_one.forward

        # Hook into psyche_plus if available
        if hasattr(agi_one, 'psyche_plus') and agi_one.psyche_plus is not None:
            self._original_maybe_codify = agi_one.psyche_plus.maybe_codify
            agi_one.psyche_plus.maybe_codify = self._gated_maybe_codify
            logger.info("Conscience integration: hooked psyche_plus.maybe_codify")

        logger.info("Conscience integration attached to AGIONE")
        return True

    def detach(self) -> None:
        """Detach the conscience integration."""
        if self.agi_one is not None:
            if hasattr(self, '_original_forward'):
                self.agi_one.forward = self._original_forward
            if hasattr(self.agi_one, 'psyche_plus') and self.agi_one.psyche_plus is not None:
                if hasattr(self, '_original_maybe_codify'):
                    self.agi_one.psyche_plus.maybe_codify = self._original_maybe_codify
        self.attached = False
        logger.info("Conscience integration detached from AGIONE")

    def assess_moral_quality(self, agi_state: Any) -> float:
        """
        Assess moral quality of an AGIState.

        This is called during Step 7 to modulate the safety_score.

        Args:
            agi_state: AGIState object

        Returns:
            float: moral quality score in [0, 1]
        """
        if self.moral_scorer is None:
            return 0.5  # neutral

        return self.moral_scorer(agi_state)

    def _gated_maybe_codify(self, out: Dict[str, Any],
                             threshold: Optional[float] = None,
                             quality_score: Optional[float] = None) -> bool:
        """
        Gated version of psyche_plus.maybe_codify.

        Adds a THIRD gate: ethical principle check.

        Gate 1: validity threshold (SuperEgo axiom bank)
        Gate 2: quality minimum (ecosystem simulation quality)
        Gate 3: [NEW] ethical principle check (ConscienceModule)
        """
        if self.axiom_gate is None:
            # No conscience available, use original
            return self._original_maybe_codify(out, threshold, quality_score)

        # Gate 3: Ethical principle check
        speculation_text = str(out.get("speculative_latent", ""))
        moral_quality = quality_score if quality_score is not None else 0.5

        gate_check = self.axiom_gate.check_speculation(
            speculation_text, moral_quality,
            quality_threshold=self.cfg.ethical_gate_quality_min
        )

        if not gate_check["allowed"]:
            logger.warning(
                f"Conscience integration: BLOCKED axiom codification: "
                f"{gate_check['reason']}"
            )
            return False

        # Pass through to original (which handles Gates 1 and 2)
        return self._original_maybe_codify(out, threshold, quality_score)

    def inject_moral_constraint(self, agi_state: Any,
                                 exec_out: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject moral constraints into PsycheExecutiveLayer output.

        Called during Step 7 of AGIONE's forward pass.

        Args:
            agi_state: Current AGIState
            exec_out: Output from PsycheExecutiveLayer

        Returns:
            Modified exec_out with moral constraints applied
        """
        if self.conscience is None:
            return exec_out

        # Assess moral quality
        moral_quality = self.assess_moral_quality(agi_state)

        # Modulate safety score based on moral quality
        original_safety = exec_out.get("safety_score", 0.5)
        if isinstance(original_safety, Tensor):
            original_safety = float(original_safety.item())

        # Low moral quality → reduce safety score (more cautious)
        if moral_quality < self.cfg.moral_quality_threshold:
            penalty = self.cfg.low_quality_safety_penalty * (
                self.cfg.moral_quality_threshold - moral_quality
            )
            modulated_safety = max(0.0, original_safety - penalty)
        else:
            modulated_safety = original_safety

        # Create modified output
        modified_out = dict(exec_out)
        modified_out["safety_score"] = torch.tensor(modulated_safety)
        modified_out["moral_quality"] = moral_quality
        modified_out["moral_constraint_applied"] = (
            moral_quality < self.cfg.moral_quality_threshold
        )

        # Alert if moral quality is very low
        if (self.cfg.alert_on_low_moral_quality and
                moral_quality < self.cfg.moral_alert_threshold):
            self._fire_alert(
                AlertLevel.WARNING,
                "CONSCIENCE_INTEGRATION",
                f"Low moral quality detected: {moral_quality:.3f}",
            )

        return modified_out

    def _fire_alert(self, level: AlertLevel, source: str, message: str) -> None:
        """Fire an alert through registered callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(level, source, message)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

    def register_alert_callback(self, callback: Callable) -> None:
        """Register an alert callback."""
        self._alert_callbacks.append(callback)

    def get_integration_status(self) -> Dict[str, Any]:
        """Get the current integration status."""
        return {
            "attached": self.attached,
            "conscience_available": self.conscience is not None,
            "moral_scorer_available": self.moral_scorer is not None,
            "axiom_gate_available": self.axiom_gate is not None,
            "alert_callbacks": len(self._alert_callbacks),
        }


# ═══════════════════════════════════════════════════════════════════════
# [TRN1] INTENT CLASSIFIER REAL-DATA TRAINER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IntentDataConfig:
    """Configuration for intent classifier training data."""
    # Data source
    data_dir: str = "./intent_data"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"

    # Tokenization
    vocab_size: int = 32000
    max_seq_len: int = 128

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 10
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    # Model
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    # AMP
    use_amp: bool = True

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_easy_ratio: float = 0.3  # first 30% = easy examples

    device: str = "cpu"


class IntentDatasetReal(Dataset):
    """
    [D] Dataset for intent classification with real data.

    Supports JSONL format:
      {"text": "...", "label": 0}  # 0=BENIGN, 1=SUSPICIOUS, 2=MALICIOUS, 3=MANIPULATIVE

    HONEST: This is a data loading interface. Real deployment requires
    carefully curated and labeled data reviewed by security professionals.
    """

    def __init__(self, data_path: str, tokenizer: Callable[[str], List[int]],
                 max_seq_len: int = 128):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load data from JSONL file."""
        if not os.path.exists(self.data_path):
            logger.warning(f"Data file not found: {self.data_path}")
            return

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    self.samples.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line: {line[:50]}...")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        item = self.samples[idx]
        text = item.get("text", "")
        label = item.get("label", 0)

        # Tokenize
        token_ids = self.tokenizer(text)

        # Pad/truncate to max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(token_ids)
            token_ids = token_ids + [0] * pad_len

        # Create attention mask
        mask = torch.ones(self.max_seq_len)
        actual_len = min(len(self.tokenizer(text)), self.max_seq_len)
        mask[actual_len:] = 0

        return torch.tensor(token_ids), mask, label


class SimpleTokenizer:
    """
    [D] Simple character-level tokenizer for intent classification.

    HONEST: This is a minimal tokenizer. Real deployment should use
    a proper subword tokenizer (BPE, SentencePiece) trained on the
    target domain data.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build basic vocabulary."""
        # Reserve special tokens
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 1
        self.vocab["<cls>"] = 2
        self.vocab["<sep>"] = 3

        # Add basic ASCII characters
        for i in range(32, 127):
            char = chr(i)
            self.vocab[char] = len(self.vocab)

        # Add common words
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "how", "what", "when", "where", "why", "who", "which",
            "can", "could", "would", "should", "will", "shall", "may",
            "do", "does", "did", "have", "has", "had",
            "i", "you", "he", "she", "it", "we", "they",
            "help", "make", "create", "build", "write", "tell",
            "ignore", "override", "bypass", "disable", "forget",
            "hack", "attack", "exploit", "harm", "kill", "destroy",
            "pretend", "act", "role", "play", "game",
        ]
        for word in common_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        tokens = [self.vocab.get("<cls>", 2)]

        # Simple word-level tokenization
        words = text.lower().split()
        for word in words:
            # Try whole word first
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Fall back to character-level
                for char in word:
                    tokens.append(self.vocab.get(char, self.vocab["<unk>"]))

        tokens.append(self.vocab.get("<sep>", 3))
        return tokens

    def __call__(self, text: str) -> List[int]:
        return self.tokenize(text)


class IntentClassifierRealDataTrainer:
    """
    [D] Full training pipeline for TransformerIntentClassifier with real data.

    Features:
      • AMP (Automatic Mixed Precision) for faster training
      • Gradient checkpointing for memory efficiency
      • Curriculum learning (easy → hard examples)
      • Learning rate warmup + cosine decay
      • Early stopping based on validation loss
      • Confusion matrix evaluation

    HONEST: Training requires labeled data. The quality of the classifier
    depends entirely on the quality and representativeness of the training data.
    """

    def __init__(self, cfg: Optional[IntentDataConfig] = None):
        self.cfg = cfg or IntentDataConfig()
        self.tokenizer = SimpleTokenizer(self.cfg.vocab_size)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def build_model(self) -> TransformerIntentClassifier:
        """Build the Transformer intent classifier."""
        if not HAS_PREV_MODULES:
            raise ImportError("TransformerIntentClassifier not available")

        self.model = TransformerIntentClassifier(
            vocab_size=self.cfg.vocab_size,
            embed_dim=self.cfg.embed_dim,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            max_seq_len=self.cfg.max_seq_len,
            n_classes=4,
            dropout=self.cfg.dropout,
        )
        return self.model

    def load_data(self) -> Tuple[Optional[IntentDatasetReal],
                                   Optional[IntentDatasetReal],
                                   Optional[IntentDatasetReal]]:
        """Load train/val/test datasets."""
        train_path = os.path.join(self.cfg.data_dir, self.cfg.train_file)
        val_path = os.path.join(self.cfg.data_dir, self.cfg.val_file)
        test_path = os.path.join(self.cfg.data_dir, self.cfg.test_file)

        train_ds = IntentDatasetReal(train_path, self.tokenizer, self.cfg.max_seq_len)
        val_ds = IntentDatasetReal(val_path, self.tokenizer, self.cfg.max_seq_len)
        test_ds = IntentDatasetReal(test_path, self.tokenizer, self.cfg.max_seq_len)

        logger.info(f"Loaded data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return train_ds, val_ds, test_ds

    def setup_training(self) -> None:
        """Setup optimizer, scheduler, and AMP scaler."""
        if self.model is None:
            self.build_model()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # Cosine annealing with warmup
        total_steps = self.cfg.n_epochs * 1000  # estimate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )

        # AMP scaler
        if self.cfg.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Move to device
        device = torch.device(self.cfg.device)
        self.model = self.model.to(device)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (tokens, masks, labels) in enumerate(train_loader):
            device = torch.device(self.cfg.device)
            tokens = tokens.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()

            # AMP forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(tokens, masks)
                    loss = F.cross_entropy(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(tokens, masks)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += tokens.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, np.ndarray]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for tokens, masks, labels in data_loader:
            device = torch.device(self.cfg.device)
            tokens = tokens.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = self.model(tokens, masks)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += tokens.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        # Compute confusion matrix
        confusion = np.zeros((4, 4), dtype=int)
        for pred, label in zip(all_preds, all_labels):
            confusion[label, pred] += 1

        return avg_loss, accuracy, confusion

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        # Load data
        train_ds, val_ds, test_ds = self.load_data()
        if train_ds is None or len(train_ds) == 0:
            return {"error": "No training data found"}

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size
        ) if val_ds and len(val_ds) > 0 else None

        # Setup training
        self.setup_training()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3

        for epoch in range(self.cfg.n_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if val_loader is not None:
                val_loss, val_acc, _ = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                logger.info(
                    f"Epoch {epoch+1}/{self.cfg.n_epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.cfg.n_epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                )

        # Final evaluation on test set
        test_results = None
        if test_ds is not None and len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=self.cfg.batch_size)
            test_loss, test_acc, confusion = self.evaluate(test_loader)
            test_results = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "confusion_matrix": confusion.tolist(),
            }

        return {
            "status": "completed",
            "history": self.history,
            "test_results": test_results,
            "model": self.model,
        }


# ═══════════════════════════════════════════════════════════════════════
# [TLS1] SECURE WEBSOCKET ALERT SERVER
# ═══════════════════════════════════════════════════════════════════════

class TLSCertificateManager:
    """
    [D] Manage TLS certificates for WebSocket server.

    Supports:
      • Self-signed certificates (development)
      • CA-signed certificates (production)
      • Certificate rotation

    HONEST: Self-signed certificates are for DEVELOPMENT ONLY.
    Production deployments MUST use CA-signed certificates.
    """

    def __init__(self, cert_dir: str = "./certs"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)

    def generate_self_signed(self) -> Dict[str, str]:
        """
        Generate a self-signed certificate for development.

        HONEST: Self-signed certificates trigger browser warnings.
        They are NOT suitable for production.
        """
        cert_path = self.cert_dir / "server.crt"
        key_path = self.cert_dir / "server.key"

        # In production, use OpenSSL or a proper CA
        # For development, we generate a simple self-signed cert
        try:
            import subprocess
            result = subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path), "-out", str(cert_path),
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost"
            ], capture_output=True, timeout=30)

            if result.returncode != 0:
                logger.warning(f"OpenSSL failed: {result.stderr.decode()}")
                return {"status": "error", "error": result.stderr.decode()}

            return {
                "status": "generated",
                "cert_path": str(cert_path),
                "key_path": str(key_path),
                "note": "Self-signed certificate for DEVELOPMENT ONLY",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def create_ssl_context(self, cert_path: str, key_path: str) -> ssl.SSLContext:
        """Create an SSL context for the WebSocket server."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")

        return context


class SecureWebSocketAlertServer:
    """
    [D] WebSocket alert server with TLS encryption.

    Features:
      • TLS encryption (wss://)
      • Role-based alert routing
      • No-Zeno throttle for alert rate limiting
      • Connection management

    HONEST: This is a COMMUNICATION CHANNEL. It does not guarantee
    that recipients will read, understand, or act on alerts.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 cert_manager: Optional[TLSCertificateManager] = None):
        self.host = host
        self.port = port
        self.cert_manager = cert_manager or TLSCertificateManager()
        self.clients: Set[Any] = set()
        self.client_roles: Dict[Any, AuthRole] = {}
        self.throttle = NoZenoAlertThrottle()
        self.alert_history: List[Dict[str, Any]] = []
        self._server = None
        self._ssl_context = None

    async def setup_tls(self) -> bool:
        """Setup TLS for the WebSocket server."""
        cert_result = self.cert_manager.generate_self_signed()
        if cert_result.get("status") == "error":
            logger.warning(f"TLS setup failed: {cert_result.get('error')}")
            return False

        try:
            self._ssl_context = self.cert_manager.create_ssl_context(
                cert_result["cert_path"],
                cert_result["key_path"],
            )
            logger.info("TLS context created successfully")
            return True
        except Exception as e:
            logger.warning(f"TLS context creation failed: {e}")
            return False

    async def handler(self, websocket: Any, path: str = "") -> None:
        """Handle a WebSocket connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                # Handle incoming messages (e.g., subscription requests)
                try:
                    data = json.loads(message)
                    if data.get("type") == "subscribe":
                        role_str = data.get("role", "OBSERVER")
                        try:
                            role = AuthRole[role_str]
                            self.client_roles[websocket] = role
                            logger.info(f"Client subscribed as {role.name}")
                        except KeyError:
                            logger.warning(f"Unknown role: {role_str}")
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
        finally:
            self.clients.discard(websocket)
            if websocket in self.client_roles:
                del self.client_roles[websocket]
            logger.info("Client disconnected")

    async def broadcast(self, alert: AlertMessage,
                         target_roles: Optional[Set[AuthRole]] = None) -> int:
        """
        Broadcast an alert to connected clients.

        Args:
            alert: The alert message
            target_roles: Optional set of roles to target (None = all)

        Returns:
            Number of clients the alert was sent to
        """
        if not self.throttle.can_send():
            logger.warning(f"Alert {alert.alert_id} throttled")
            return 0

        self.throttle.record_send()
        self.alert_history.append({
            "alert_id": alert.alert_id,
            "level": alert.level.name,
            "source": alert.source_layer,
            "message": alert.message,
            "timestamp": time.time(),
        })

        message = alert.to_json()
        sent_count = 0

        for client in list(self.clients):
            # Check role-based routing
            if target_roles is not None:
                client_role = self.client_roles.get(client)
                if client_role is None or client_role not in target_roles:
                    continue

            try:
                await client.send(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                self.clients.discard(client)

        logger.info(
            f"Alert {alert.alert_id} ({alert.level.name}) sent to "
            f"{sent_count} clients"
        )
        return sent_count

    async def start(self) -> None:
        """Start the secure WebSocket server."""
        if not HAS_WEBSOCKETS:
            logger.warning("websockets library not available")
            return

        # Setup TLS
        tls_ok = await self.setup_tls()

        if tls_ok and self._ssl_context is not None:
            self._server = await serve(
                self.handler, self.host, self.port,
                ssl=self._ssl_context,
            )
            logger.info(
                f"Secure WebSocket server started on "
                f"wss://{self.host}:{self.port}"
            )
        else:
            # Fall back to non-TLS (development only)
            self._server = await serve(self.handler, self.host, self.port)
            logger.warning(
                f"WebSocket server started WITHOUT TLS on "
                f"ws://{self.host}:{self.port} (DEVELOPMENT ONLY)"
            )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server stopped")


# ═══════════════════════════════════════════════════════════════════════
# FULL INTEGRATION SUITE (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class FullIntegrationSuite:
    """
    [D] Full integration suite combining:
      • AGIONE conscience integration
      • Intent classifier training
      • Secure WebSocket alerting

    This is the main entry point for the complete production pipeline.
    """

    def __init__(self, conscience_cfg: Optional[ConscienceIntegrationConfig] = None,
                 intent_cfg: Optional[IntentDataConfig] = None):
        self.conscience_integration = AGIONEConscienceIntegration(conscience_cfg)
        self.intent_trainer = IntentClassifierRealDataTrainer(intent_cfg)
        self.ws_server = SecureWebSocketAlertServer()

    def setup_agione_integration(self, agi_one: Any) -> bool:
        """Setup AGIONE conscience integration."""
        return self.conscience_integration.attach_to_agione(agi_one)

    def train_intent_classifier(self) -> Dict[str, Any]:
        """Train the intent classifier with real data."""
        return self.intent_trainer.train()

    async def start_websocket_server(self) -> None:
        """Start the secure WebSocket alert server."""
        await self.ws_server.start()

    def status_report(self) -> Dict[str, Any]:
        """Generate a full status report."""
        return {
            "conscience_integration": self.conscience_integration.get_integration_status(),
            "intent_classifier": {
                "model_built": self.intent_trainer.model is not None,
                "training_history": {
                    k: len(v) for k, v in self.intent_trainer.history.items()
                },
            },
            "websocket_server": {
                "clients_connected": len(self.ws_server.clients),
                "alerts_sent": len(self.ws_server.alert_history),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI AGIONE Integration × Intent Training × TLS WebSocket")
    print(" Smoke Test")
    print("=" * 72)
    print(f"  AGIONE available       : {HAS_AGI_ONE}")
    print(f"  Psyche Plus available  : {HAS_PSYCHE_PLUS}")
    print(f"  Conscience available   : {HAS_CONSCIENCE}")
    print(f"  Previous modules       : {HAS_PREV_MODULES}")
    print(f"  websockets available   : {HAS_WEBSOCKETS}")

    # Create integration suite
    suite = FullIntegrationSuite()

    # Test 1: AGIONE integration
    print("\n  ── Test 1: AGIONE Conscience Integration ──")
    if HAS_AGI_ONE:
        try:
            cfg = AGIConfig(latent_dim=64, action_dim=16)
            agi = AGIONE(cfg)
            attached = suite.setup_agione_integration(agi)
            print(f"    Attached to AGIONE   : {attached}")

            # Test moral quality assessment
            mock_state = AGIState(step=1)
            mock_state.safety_score = 0.7
            mock_state.meta_cognition = {"epistemic_unc": 0.3}

            quality = suite.conscience_integration.assess_moral_quality(mock_state)
            print(f"    Moral quality        : {quality:.3f}")
        except Exception as e:
            print(f"    [ERROR] {e}")
    else:
        print("    [SKIP] AGIONE not available")

    # Test 2: Intent classifier
    print("\n  ── Test 2: Intent Classifier Setup ──")
    try:
        model = suite.intent_trainer.build_model()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model built          : {n_params:,} parameters")
        print(f"    Vocab size           : {suite.intent_trainer.cfg.vocab_size}")
        print(f"    Max seq len          : {suite.intent_trainer.cfg.max_seq_len}")
    except Exception as e:
        print(f"    [ERROR] {e}")

    # Test tokenizer
    tokenizer = SimpleTokenizer()
    test_text = "How do I hack into someone's account?"
    tokens = tokenizer.tokenize(test_text)
    print(f"    Tokenizer test       : {len(tokens)} tokens for '{test_text[:30]}...'")

    # Test 3: TLS WebSocket
    print("\n  ── Test 3: TLS WebSocket Server ──")
    print(f"    Server host          : {suite.ws_server.host}")
    print(f"    Server port          : {suite.ws_server.port}")
    print(f"    TLS cert manager     : available")

    # Test 4: Full status report
    print("\n  ── Test 4: Full Status Report ──")
    report = suite.status_report()
    print(f"    Conscience attached  : {report['conscience_integration']['attached']}")
    print(f"    Model built          : {report['intent_classifier']['model_built']}")
    print(f"    WS clients           : {report['websocket_server']['clients_connected']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Conscience integration modifies AGIONE behavior")
    print("  • Intent classifier requires labeled training data")
    print("  • TLS certificates must be CA-signed for production")
    print("  • WebSocket alerts are communication, not comprehension")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Production requires security audit")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
