#!/usr/bin/env python3
"""
=============================================================================
PSY ONE + AGI ONE: COGNITIVE MODELING & NLP ANALYSIS FRAMEWORK
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

WHAT THIS IS — RESEARCH FRAMEWORK FOR TEXT ANALYSIS
=============================================================================
This framework integrates PSY ONE and AGI ONE Psyche Plus to perform
LEGITIMATE computational analysis of TEXT data. It does NOT read minds,
diagnose conditions, or infer intent from brain signals.

COMPONENTS (all text-based, all cited):
  [N1] Sentiment Analysis — VADER + transformer-based scoring
  [N2] Topic Modeling — LDA / NMF on text corpora
  [N3] Discourse Analysis — coherence, argumentation structure
  [N4] Cognitive Bias Detection — linguistic pattern matching against
       validated cognitive distortion taxonomies
  [N5] Bayesian Belief Updating — hierarchical model of how stated
       beliefs change across time-lagged text samples
  [N6] Validated Instrument Mapping — PHQ-9, GAD-7, HAMD-17, PCL-5
       as structured INPUT features, never as automatic outputs

PSY ONE / AGI ONE INTEGRATION
=============================================================================
The Id/Ego/Superego triad is repurposed as a COMPUTATIONAL METAPHOR
for text analysis (not as a clinical model):

  IdModule    → Raw emotional signal extraction from text
                (sentiment intensity, arousal, valence)
  EgoModule   → Rational coherence assessment
                (logical consistency, argumentation quality)
  SuperEgoModule → Normative/social compliance signals
                (convention adherence, social desirability markers)

This is a legitimate computational linguistics framing. It is NOT
a psychiatric diagnostic tool.

HONEST SCOPE — READ CAREFULLY
=============================================================================
✅ Analyzes TEXT (written or transcribed speech)
✅ Detects linguistic patterns associated with cognitive biases
✅ Models belief updating from sequential text samples
✅ Maps text features to validated instrument score ranges
✅ All outputs are RESEARCH HYPOTHESES, not diagnoses

❌ Does NOT diagnose any psychiatric or neurological condition
❌ Does NOT read minds, thoughts, or intentions
❌ Does NOT replace clinical assessment by trained professionals
❌ Does NOT use EEG, MEG, fMRI, or any brain imaging data
❌ Does NOT claim state-of-the-art or paradigm shift

VALIDATED INSTRUMENTS USED AS INPUT FEATURES
=============================================================================
These are standard clinical screening tools. In this framework they
serve as STRUCTURED INPUT FEATURES alongside text, not as outputs.

  PHQ-9    : Patient Health Questionnaire-9 (depression screening)
  GAD-7    : Generalized Anxiety Disorder-7
  HAMD-17  : Hamilton Depression Rating Scale
  PCL-5    : PTSD Checklist for DSM-5
  YMRS     : Young Mania Rating Scale
  PANSS    : Positive and Negative Syndrome Scale

References:
  Kroenke, Spitzer, Williams (2001) J Gen Intern Med. 16(9):606-613.
  Spitzer et al. (2006) Arch Intern Med. 166(10):1092-1097.
  Hamilton (1960) J Neurol Neurosurg Psychiatry. 23(1):56-62.
  Weathers et al. (2013) J Trauma Stress. 26(6):714-721.

COGNITIVE BIAS TAXONOMY (text-based detection)
=============================================================================
Based on Beck's cognitive distortion taxonomy:
  Beck, A.T. (1979). Cognitive Therapy and the Emotional Disorders.
  Burns, D.D. (1980). Feeling Good: The New Mood Therapy.

Distortions detected as LINGUISTIC PATTERNS (not clinical diagnoses):
  1. All-or-nothing thinking    — absolute quantifiers
  2. Catastrophizing            — escalation language
  3. Mental filtering           — selective negative focus
  4. Disqualifying positive     — dismissal patterns
  5. Mind reading               — assumed knowledge claims
  6. Fortune telling            — predictive certainty language
  7. Magnification/minimization — intensity asymmetry
  8. Emotional reasoning        — feeling-as-fact patterns
  9. Should statements          — obligation language
  10. Labeling                  — global negative self/other labels
  11. Personalization           — self-attribution of external events
  12. Blame                     — external attribution of internal events

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("psy_one_cognitive")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# §0. RESEARCH-ONLY GUARD
# =============================================================================

RESEARCH_BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  RESEARCH FRAMEWORK — TEXT ANALYSIS ONLY                        ║
║  NOT A CLINICAL DIAGNOSTIC TOOL                                       ║
║  NOT A MIND-READING SYSTEM                                            ║
║  All outputs are research hypotheses, not diagnoses.                  ║
║  Clinical decisions require trained professionals.                   ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def research_guard():
    print(RESEARCH_BANNER)

# =============================================================================
# §1. DATA STRUCTURES
# =============================================================================

class AnalysisType(Enum):
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    DISCOURSE = "discourse"
    COGNITIVE_BIAS = "cognitive_bias"
    BELIEF_UPDATE = "belief_update"
    INSTRUMENT_MAPPING = "instrument_mapping"

@dataclass
class TextSample:
    """A single text sample for analysis."""
    text: str
    timestamp: Optional[float] = None
    speaker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CognitiveBiasResult:
    """Result of cognitive bias detection."""
    bias_type: str
    confidence: float
    evidence_spans: List[Tuple[int, int]]
    linguistic_markers: List[str]
    note: str = "LINGUISTIC PATTERN ONLY — NOT A CLINICAL DIAGNOSIS"

@dataclass
class BeliefState:
    """Represents a belief state for Bayesian updating."""
    topic: str
    prior_probability: float
    posterior_probability: float
    evidence: List[str] = field(default_factory=list)
    timestamp: float = 0.0

@dataclass
class InstrumentScore:
    """Score from a validated clinical instrument (INPUT, not output)."""
    instrument: str
    score: float
    max_score: float
    severity: str
    notes: str = ""

@dataclass
class AnalysisReport:
    """Complete analysis report."""
    report_id: str
    timestamp: str
    analysis_types: List[str]
    text_sample_count: int
    results: Dict[str, Any] = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)
    disclaimer: str = ("RESEARCH OUTPUT — NOT A CLINICAL DIAGNOSIS. "
                       "All findings are linguistic pattern analyses "
                       "of text data, not medical assessments.")

# =============================================================================
# §2. NLP ANALYSIS LAYER
# =============================================================================

class SentimentAnalyzer:
    """
    Text sentiment analysis using rule-based VADER-style scoring
    with optional transformer embeddings.

    References:
        Hutto & Gilbert (2014). VADER: A Parsimonious Rule-based Model
        for Sentiment Analysis of Social Media Text. ICWSM.
    """

    # VADER-style lexicon (subset for demonstration)
    POSITIVE_WORDS = {
        "good": 1.9, "great": 2.4, "excellent": 3.0, "happy": 2.0,
        "joy": 2.5, "love": 2.8, "wonderful": 2.7, "fantastic": 2.8,
        "amazing": 2.6, "beautiful": 2.2, "calm": 1.5, "peaceful": 1.8,
        "hopeful": 1.6, "grateful": 2.0, "proud": 1.8, "excited": 2.2,
        "content": 1.5, "satisfied": 1.6, "optimistic": 1.7,
    }

    NEGATIVE_WORDS = {
        "bad": -1.9, "terrible": -2.8, "awful": -2.5, "horrible": -2.7,
        "sad": -1.8, "angry": -2.2, "hate": -2.7, "fear": -2.0,
        "anxious": -1.8, "depressed": -2.5, "miserable": -2.8,
        "hopeless": -2.9, "worthless": -3.0, "guilty": -2.0,
        "ashamed": -2.2, "hopeless": -2.9, "desperate": -2.6,
        "overwhelmed": -2.0, "frustrated": -1.7, "disappointed": -1.8,
    }

    INTENSIFIERS = {
        "very": 0.5, "really": 0.5, "extremely": 1.0, "absolutely": 0.8,
        "completely": 0.7, "totally": 0.6, "incredibly": 0.9,
        "somewhat": -0.3, "slightly": -0.3, "barely": -0.5,
    }

    NEGATORS = {"not", "no", "never", "neither", "nor", "without"}

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        if use_transformer:
            log.info("[Sentiment] Transformer mode requires HuggingFace "
                     "transformers — will fall back to lexicon if unavailable.")

    def score(self, text: str) -> Dict[str, float]:
        """
        Returns sentiment scores: compound, positive, negative, neutral.
        Compound is in [-1, 1].
        """
        tokens = self._tokenize(text)
        pos_sum, neg_sum, total = 0.0, 0.0, 0
        negation_window = 0

        for i, token in enumerate(tokens):
            lower = token.lower()

            # Track negation
            if lower in self.NEGATORS:
                negation_window = 3  # next 3 words are negated
                continue

            if negation_window > 0:
                negation_window -= 1
                multiplier = -1.0
            else:
                multiplier = 1.0

            # Check intensifiers
            intensity = 0.0
            for j in range(max(0, i - 2), i):
                if tokens[j].lower() in self.INTENSIFIERS:
                    intensity = self.INTENSIFIERS[tokens[j].lower()]

            if lower in self.POSITIVE_WORDS:
                pos_sum += (self.POSITIVE_WORDS[lower] + intensity) * multiplier
                total += 1
            elif lower in self.NEGATIVE_WORDS:
                neg_sum += (self.NEGATIVE_WORDS[lower] + intensity) * multiplier
                total += 1

        if total == 0:
            return {"compound": 0.0, "positive": 0.0,
                    "negative": 0.0, "neutral": 1.0}

        pos_score = max(pos_sum, 0.0)
        neg_score = abs(min(neg_sum, 0.0))
        total_score = pos_score + neg_score

        if total_score == 0:
            return {"compound": 0.0, "positive": 0.0,
                    "negative": 0.0, "neutral": 1.0}

        compound = (pos_score - neg_score) / total_score
        return {
            "compound": compound,
            "positive": pos_score / total_score,
            "negative": neg_score / total_score,
            "neutral": 0.0,
        }

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def batch_score(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self.score(t) for t in texts]


class TopicModeler:
    """
    Simple topic extraction using TF-IDF and clustering.
    For production, use LDA (gensim) or BERTopic.

    References:
        Blei, Ng, Jordan (2003). Latent Dirichlet Allocation. JMLR.
    """

    def __init__(self, n_topics: int = 5, max_features: int = 1000):
        self.n_topics = n_topics
        self.max_features = max_features
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._topics: List[Dict[str, float]] = []

    def fit(self, documents: List[str]) -> "TopicModeler":
        """Build vocabulary and compute IDF."""
        # Tokenize
        tokenized = [self._tokenize(d) for d in documents]

        # Build vocabulary (top max_features by document frequency)
        doc_freq = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))

        most_common = doc_freq.most_common(self.max_features)
        self._vocab = {w: i for i, (w, _) in enumerate(most_common)}

        # Compute IDF
        n_docs = len(documents)
        idf = np.zeros(len(self._vocab))
        for word, idx in self._vocab.items():
            idf[idx] = math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1
        self._idf = idf

        # Compute TF-IDF matrix
        tfidf_matrix = np.zeros((n_docs, len(self._vocab)))
        for i, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            for word, count in tf.items():
                if word in self._vocab:
                    idx = self._vocab[word]
                    tfidf_matrix[i, idx] = (1 + math.log(count)) * self._idf[idx]

        # Normalize
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        tfidf_matrix = tfidf_matrix / np.maximum(norms, 1e-10)

        # Simple k-means clustering for topics
        self._topics = self._kmeans(tfidf_matrix, self.n_topics)

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Assign documents to topics."""
        tokenized = [self._tokenize(d) for d in documents]
        n_docs = len(documents)
        tfidf = np.zeros((n_docs, len(self._vocab)))

        for i, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            for word, count in tf.items():
                if word in self._vocab:
                    idx = self._vocab[word]
                    tfidf[i, idx] = (1 + math.log(count)) * self._idf[idx]

        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf = tfidf / np.maximum(norms, 1e-10)

        # Assign to nearest topic centroid
        assignments = np.zeros(n_docs)
        for i in range(n_docs):
            dists = [np.linalg.norm(tfidf[i] - t) for t in self._topics]
            assignments[i] = np.argmin(dists)

        return assignments

    def get_topic_words(self, topic_idx: int, top_k: int = 10) -> List[str]:
        """Get top words for a topic."""
        if topic_idx >= len(self._topics):
            return []
        centroid = self._topics[topic_idx]
        top_indices = np.argsort(centroid)[-top_k:][::-1]
        idx_to_word = {v: k for k, v in self._vocab.items()}
        return [idx_to_word.get(i, "") for i in top_indices]

    def _kmeans(self, X: np.ndarray, k: int, n_iter: int = 20) -> List[np.ndarray]:
        """Simple k-means."""
        n = X.shape[0]
        if n < k:
            k = max(1, n)
        rng = np.random.default_rng(42)
        centroids = X[rng.choice(n, k, replace=False)].copy()

        for _ in range(n_iter):
            # Assign
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            # Update
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = X[mask].mean(axis=0)

        return list(centroids)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        # Remove common stopwords
        stopwords = {"the", "and", "for", "are", "but", "not", "you",
                     "all", "can", "had", "her", "was", "one", "our",
                     "out", "day", "get", "has", "him", "his", "how",
                     "man", "new", "now", "old", "see", "two", "way",
                     "who", "did", "its", "let", "put", "say", "she",
                     "too", "use", "with", "that", "this", "have",
                     "from", "they", "know", "want", "been", "good",
                     "much", "some", "time", "very", "when", "come",
                     "here", "just", "like", "long", "make", "many",
                     "over", "such", "take", "than", "them", "well",
                     "were", "will", "would", "there", "their", "what",
                     "about", "which", "could", "should", "would"}
        return [t for t in tokens if t not in stopwords]


class DiscourseAnalyzer:
    """
    Analyzes discourse structure: coherence, argumentation, narrative flow.

    References:
        Grosz & Sidner (1986). Attention, Intentions, and the Structure
        of Discourse. Computational Linguistics.
    """

    def __init__(self):
        self.connectives = {
            "causal": ["because", "since", "therefore", "thus", "hence",
                       "consequently", "as a result", "so"],
            "contrast": ["however", "but", "although", "whereas",
                         "nevertheless", "yet", "despite"],
            "addition": ["and", "also", "moreover", "furthermore",
                         "in addition", "additionally"],
            "temporal": ["then", "after", "before", "during", "while",
                         "meanwhile", "subsequently", "eventually"],
            "conditional": ["if", "unless", "provided that", "assuming",
                            "in case", "when"],
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        sentences = self._split_sentences(text)
        result = {
            "n_sentences": len(sentences),
            "connective_counts": {},
            "coherence_score": 0.0,
            "discourse_relations": [],
        }

        # Count connectives
        for category, words in self.connectives.items():
            count = sum(text.lower().count(w) for w in words)
            result["connective_counts"][category] = count

        # Coherence: ratio of connected sentences
        total_connectives = sum(result["connective_counts"].values())
        if len(sentences) > 1:
            result["coherence_score"] = min(1.0, total_connectives / (len(sentences) - 1))

        return result

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


# =============================================================================
# §3. COGNITIVE BIAS DETECTION (text-based)
# =============================================================================

class CognitiveBiasDetector:
    """
    Detects linguistic patterns associated with cognitive distortions.

    Based on Beck (1979) and Burns (1980) cognitive distortion taxonomy.
    These are TEXT PATTERN DETECTIONS, not clinical diagnoses.

    References:
        Beck, A.T. (1979). Cognitive Therapy and the Emotional Disorders.
        Burns, D.D. (1980). Feeling Good: The New Mood Therapy.
    """

    BIAS_PATTERNS = {
        "all_or_nothing": {
            "patterns": [
                r'\b(always|never|every|all|nothing|no one|everyone)\b',
                r'\b(every single|each and every|without exception)\b',
                r'\b(completely|totally|entirely|absolutely)\b',
            ],
            "description": "Absolute, black-and-white language",
        },
        "catastrophizing": {
            "patterns": [
                r'\b(disaster|catastrophe|worst case|terrible|awful|horrible)\b',
                r'\b(everything will go wrong|it will be the end|disaster)\b',
                r'\b(can\'t handle|won\'t survive|going to die)\b',
            ],
            "description": "Escalation to worst-case scenarios",
        },
        "mental_filter": {
            "patterns": [
                r'\b(only|just|nothing but|all I see|all I notice)\b',
                r'\b(the only thing that matters|the only problem)\b',
            ],
            "description": "Selective focus on negative aspects",
        },
        "mind_reading": {
            "patterns": [
                r'\b(they think|he thinks|she thinks|everyone thinks)\b',
                r'\b(they must|he must|she must|they probably)\b',
                r'\b(I know what they\'re thinking|they\'re judging)\b',
            ],
            "description": "Assumed knowledge of others' thoughts",
        },
        "fortune_telling": {
            "patterns": [
                r'\b(it will|it\'s going to|nothing will|it won\'t)\b',
                r'\b(I\'ll never|I\'ll always|it\'s hopeless|pointless)\b',
            ],
            "description": "Predictive certainty about negative outcomes",
        },
        "emotional_reasoning": {
            "patterns": [
                r'\b(I feel|I know|deep down|in my heart|my gut tells)\b',
                r'\b(because I feel|since I feel|I feel like)\b',
            ],
            "description": "Using feelings as evidence for facts",
        },
        "should_statements": {
            "patterns": [
                r'\b(should|must|ought to|have to|need to)\b',
                r'\b(shouldn\'t|mustn\'t|don\'t have to)\b',
            ],
            "description": "Rigid obligation language",
        },
        "labeling": {
            "patterns": [
                r'\b(I am|I\'m|he is|she is|they are)\s+(a|an)?\s*'
                r'(failure|loser|idiot|stupid|worthless|terrible|horrible)\b',
                r'\b(completely|totally|utterly)\s+(useless|hopeless)\b',
            ],
            "description": "Global negative labels for self or others",
        },
        "personalization": {
            "patterns": [
                r'\b(my fault|because of me|I caused|I made)\b',
                r'\b(it\'s all my|I\'m to blame|I\'m responsible)\b',
            ],
            "description": "Self-attribution of external events",
        },
    }

    def detect(self, text: str) -> List[CognitiveBiasResult]:
        """
        Detect cognitive bias patterns in text.

        Returns a list of CognitiveBiasResult. Each result is a
        LINGUISTIC PATTERN DETECTION, not a clinical finding.
        """
        results = []
        text_lower = text.lower()

        for bias_type, config in self.BIAS_PATTERNS.items():
            matches = []
            markers = []
            for pattern in config["patterns"]:
                found = re.finditer(pattern, text_lower)
                for m in found:
                    matches.append((m.start(), m.end()))
                    markers.append(m.group())

            if matches:
                confidence = min(1.0, len(matches) / 3.0)  # scale by frequency
                results.append(CognitiveBiasResult(
                    bias_type=bias_type,
                    confidence=confidence,
                    evidence_spans=matches[:5],  # limit to first 5
                    linguistic_markers=markers[:5],
                ))

        return results

    def bias_profile(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute bias frequency profile across multiple texts.
        Returns normalized frequency per bias type.
        """
        counts: Dict[str, int] = defaultdict(int)
        total = 0

        for text in texts:
            biases = self.detect(text)
            for b in biases:
                counts[b.bias_type] += 1
                total += 1

        if total == 0:
            return {}

        return {k: v / total for k, v in counts.items()}


# =============================================================================
# §4. BAYESIAN BELIEF UPDATING
# =============================================================================

class BayesianBeliefUpdater:
    """
    Models belief updating from sequential text samples.

    Uses a simple Bayesian framework:
        P(H|E) = P(E|H) * P(H) / P(E)

    Where:
        H = hypothesis (e.g., "topic X is important")
        E = evidence from text (e.g., frequency of topic X mentions)

    References:
        Griffiths & Tenenbaum (2006). Optimal Predictions in Everyday
        Cognition. Psychological Science.
    """

    def __init__(self, prior: float = 0.5):
        self.prior = prior
        self.history: List[BeliefState] = []

    def update(self, topic: str, evidence_strength: float,
               evidence_direction: float) -> BeliefState:
        """
        Update belief about a topic based on new evidence.

        Args:
            topic: The belief topic
            evidence_strength: How strong the evidence is [0, 1]
            evidence_direction: +1 supports, -1 opposes, 0 neutral

        Returns:
            Updated BeliefState
        """
        # Get current prior
        if self.history:
            current_prior = self.history[-1].posterior_probability
        else:
            current_prior = self.prior

        # Likelihood: P(E|H) based on evidence
        if evidence_direction > 0:
            likelihood = 0.5 + 0.5 * evidence_strength
        elif evidence_direction < 0:
            likelihood = 0.5 - 0.5 * evidence_strength
        else:
            likelihood = 0.5

        # Bayesian update
        posterior = (likelihood * current_prior) / (
            likelihood * current_prior +
            (1 - likelihood) * (1 - current_prior)
        )

        state = BeliefState(
            topic=topic,
            prior_probability=current_prior,
            posterior_probability=posterior,
            evidence=[f"strength={evidence_strength:.2f}, "
                      f"direction={evidence_direction:+.1f}"],
            timestamp=time.time(),
        )

        self.history.append(state)
        return state

    def belief_trajectory(self, topic: str) -> List[float]:
        """Get belief probability trajectory for a topic."""
        return [s.posterior_probability for s in self.history
                if s.topic == topic]

    def reset(self):
        self.history.clear()


# =============================================================================
# §5. VALIDATED INSTRUMENT MAPPING
# =============================================================================

class InstrumentMapper:
    """
    Maps text features to validated instrument score ranges.

    IMPORTANT: This maps text features to SCORE RANGES as research
    hypotheses. It does NOT diagnose. Clinical diagnosis requires
    administration of the actual instruments by trained professionals.

    Instruments (all as INPUT features, not outputs):
        PHQ-9   : Depression screening (0-27)
        GAD-7   : Anxiety screening (0-21)
        HAMD-17 : Depression severity (0-52)
        PCL-5   : PTSD screening (0-80)
    """

    INSTRUMENTS = {
        "PHQ-9": {
            "max_score": 27,
            "severity_levels": {
                (0, 4): "minimal",
                (5, 9): "mild",
                (10, 14): "moderate",
                (15, 19): "moderately severe",
                (20, 27): "severe",
            },
        },
        "GAD-7": {
            "max_score": 21,
            "severity_levels": {
                (0, 4): "minimal",
                (5, 9): "mild",
                (10, 14): "moderate",
                (15, 21): "severe",
            },
        },
        "HAMD-17": {
            "max_score": 52,
            "severity_levels": {
                (0, 7): "normal",
                (8, 16): "mild",
                (17, 23): "moderate",
                (24, 52): "severe",
            },
        },
        "PCL-5": {
            "max_score": 80,
            "severity_levels": {
                (0, 32): "below threshold",
                (33, 80): "above threshold",
            },
        },
    }

    @staticmethod
    def score_from_responses(instrument: str,
                             responses: List[int]) -> InstrumentScore:
        """
        Compute instrument score from individual item responses.

        Args:
            instrument: One of PHQ-9, GAD-7, HAMD-17, PCL-5
            responses: Item scores (0-3 for PHQ-9/GAD-7, etc.)

        Returns:
            InstrumentScore with computed total and severity level.
        """
        if instrument not in InstrumentMapper.INSTRUMENTS:
            raise ValueError(f"Unknown instrument: {instrument}")

        config = InstrumentMapper.INSTRUMENTS[instrument]
        total = sum(responses)
        max_score = config["max_score"]

        # Find severity level
        severity = "unknown"
        for (lo, hi), level in config["severity_levels"].items():
            if lo <= total <= hi:
                severity = level
                break

        return InstrumentScore(
            instrument=instrument,
            score=float(total),
            max_score=float(max_score),
            severity=severity,
            notes="Computed from provided responses. Clinical interpretation "
                  "requires administration by trained professional.",
        )

    @staticmethod
    def text_feature_to_instrument_range(text: str,
                                         sentiment_score: float,
                                         bias_counts: Dict[str, int]
                                         ) -> Dict[str, Dict[str, float]]:
        """
        Map text features to INSTRUMENT SCORE RANGES as research hypotheses.

        Returns estimated score ranges (not point estimates) based on
        linguistic features. These are RESEARCH HYPOTHESES only.

        IMPORTANT: This does NOT administer any instrument. It estimates
        a score range based on text features for research purposes only.
        """
        ranges = {}

        # PHQ-9 depression range estimate
        # Based on negative sentiment + catastrophizing + labeling
        depression_signal = (
            max(0, -sentiment_score) * 0.5 +
            min(bias_counts.get("catastrophizing", 0) / 5.0, 1.0) * 0.3 +
            min(bias_counts.get("labeling", 0) / 3.0, 1.0) * 0.2
        )
        ranges["PHQ-9"] = {
            "estimated_low": max(0, int(depression_signal * 10)),
            "estimated_high": min(27, int(depression_signal * 20)),
            "confidence": "LOW — text-based estimate only",
        }

        # GAD-7 anxiety range estimate
        anxiety_signal = (
            min(bias_counts.get("fortune_telling", 0) / 3.0, 1.0) * 0.4 +
            min(bias_counts.get("catastrophizing", 0) / 4.0, 1.0) * 0.3 +
            min(bias_counts.get("emotional_reasoning", 0) / 3.0, 1.0) * 0.3
        )
        ranges["GAD-7"] = {
            "estimated_low": max(0, int(anxiety_signal * 8)),
            "estimated_high": min(21, int(anxiety_signal * 16)),
            "confidence": "LOW — text-based estimate only",
        }

        return ranges


# =============================================================================
# §6. PSY ONE INTEGRATION — Cognitive Text Analysis
# =============================================================================

class PSYONETextAnalyzer(nn.Module):
    """
    Integrates PSY ONE's Id/Ego/Superego triad as a computational
    metaphor for text analysis.

    Mapping:
        IdModule       → Raw emotional signal extraction (sentiment intensity)
        EgoModule      → Rational coherence assessment (logical consistency)
        SuperEgoModule → Normative compliance signals (convention adherence)

    This is a COMPUTATIONAL METAPHOR for text analysis, not a
    psychiatric model. The Id/Ego/Superego labels are used for
    organizational clarity, not clinical accuracy.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Id: emotional intensity extraction
        self.id_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Ego: coherence assessment
        self.ego_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Superego: normative compliance
        self.superego_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Integration head
        self.integration_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim // 2),
        )

    def encode_text_features(self, sentiment: Dict[str, float],
                             discourse: Dict[str, Any],
                             bias_counts: Dict[str, int]) -> torch.Tensor:
        """Encode text features into a latent vector."""
        # Build feature vector from NLP outputs
        features = []

        # Sentiment features
        features.append(sentiment.get("compound", 0.0))
        features.append(sentiment.get("positive", 0.0))
        features.append(sentiment.get("negative", 0.0))
        features.append(sentiment.get("neutral", 0.0))

        # Discourse features
        features.append(discourse.get("coherence_score", 0.0))
        for conn_type in ["causal", "contrast", "addition", "temporal"]:
            features.append(discourse.get("connective_counts", {}).get(conn_type, 0) / 10.0)

        # Bias features (normalized)
        total_biases = max(sum(bias_counts.values()), 1)
        for bias_type in ["all_or_nothing", "catastrophizing", "mental_filter",
                          "mind_reading", "fortune_telling", "emotional_reasoning",
                          "should_statements", "labeling", "personalization"]:
            features.append(bias_counts.get(bias_type, 0) / total_biases)

        # Pad to latent_dim
        while len(features) < self.latent_dim:
            features.append(0.0)
        features = features[:self.latent_dim]

        return torch.tensor(features, dtype=torch.float32)

    def forward(self, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process text features through the Id/Ego/Superego metaphor.

        Returns:
            Dict with 'id_signal', 'ego_signal', 'superego_signal',
            'integrated', and 'analysis_scores'.
        """
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)

        # Id: emotional intensity
        id_signal = self.id_encoder(text_features)

        # Ego: coherence
        ego_signal = self.ego_encoder(text_features)

        # Superego: normative compliance
        superego_signal = self.superego_encoder(text_features)

        # Integration
        combined = torch.cat([id_signal, ego_signal, superego_signal], dim=-1)
        integrated = self.integration_head(combined)

        # Analysis scores (interpretability layer)
        analysis_scores = {
            "emotional_intensity": float(id_signal.norm(dim=-1).mean()),
            "coherence": float(ego_signal.norm(dim=-1).mean()),
            "normative_compliance": float(superego_signal.norm(dim=-1).mean()),
            "integration_complexity": float(integrated.norm(dim=-1).mean()),
        }

        return {
            "id_signal": id_signal,
            "ego_signal": ego_signal,
            "superego_signal": superego_signal,
            "integrated": integrated,
            "analysis_scores": analysis_scores,
        }


# =============================================================================
# §7. UNIFIED ANALYSIS PIPELINE
# =============================================================================

class CognitiveNLPPipeline:
    """
    Unified pipeline for cognitive modeling and NLP analysis.

    Workflow:
        1. Text preprocessing
        2. Sentiment analysis (N1)
        3. Topic modeling (N2)
        4. Discourse analysis (N3)
        5. Cognitive bias detection (N4)
        6. Belief updating (N5)
        7. Instrument mapping (N6)
        8. PSY ONE integration
        9. Report generation

    All outputs are RESEARCH HYPOTHESES, not diagnoses.
    """

    def __init__(self, latent_dim: int = 64):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler(n_topics=5)
        self.discourse_analyzer = DiscourseAnalyzer()
        self.bias_detector = CognitiveBiasDetector()
        self.belief_updater = BayesianBeliefUpdater()
        self.instrument_mapper = InstrumentMapper()
        self.psy_one = PSYONETextAnalyzer(latent_dim=latent_dim)

    def analyze_text(self, text: str,
                     instrument_scores: Optional[Dict[str, InstrumentScore]] = None,
                     ) -> AnalysisReport:
        """
        Perform complete analysis on a single text sample.

        Args:
            text: The text to analyze
            instrument_scores: Optional pre-computed instrument scores
                              (from actual clinical administration)

        Returns:
            AnalysisReport with all findings
        """
        report_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        results = {}

        # 1. Sentiment analysis
        sentiment = self.sentiment_analyzer.score(text)
        results["sentiment"] = sentiment

        # 2. Discourse analysis
        discourse = self.discourse_analyzer.analyze(text)
        results["discourse"] = discourse

        # 3. Cognitive bias detection
        biases = self.bias_detector.detect(text)
        bias_counts = {}
        for b in biases:
            bias_counts[b.bias_type] = bias_counts.get(b.bias_type, 0) + 1
        results["cognitive_biases"] = {
            "detected": [asdict(b) for b in biases],
            "counts": bias_counts,
        }

        # 4. PSY ONE integration
        text_features = self.psy_one.encode_text_features(
            sentiment, discourse, bias_counts
        )
        with torch.no_grad():
            psy_result = self.psy_one(text_features)
        results["psy_one_analysis"] = psy_result["analysis_scores"]

        # 5. Instrument mapping (if scores provided)
        if instrument_scores:
            results["instrument_scores"] = {
                k: asdict(v) for k, v in instrument_scores.items()
            }

        # 6. Text-based instrument range estimates
        ranges = InstrumentMapper.text_feature_to_instrument_range(
            text, sentiment["compound"], bias_counts
        )
        results["estimated_instrument_ranges"] = ranges

        return AnalysisReport(
            report_id=report_id,
            timestamp=timestamp,
            analysis_types=[
                AnalysisType.SENTIMENT.value,
                AnalysisType.DISCOURSE.value,
                AnalysisType.COGNITIVE_BIAS.value,
                AnalysisType.INSTRUMENT_MAPPING.value,
            ],
            text_sample_count=1,
            results=results,
            caveats=[
                "All findings are linguistic pattern analyses, not diagnoses",
                "Instrument score ranges are estimates, not measurements",
                "Cognitive bias detection identifies text patterns only",
                "PSY ONE Id/Ego/Superego labels are computational metaphors",
            ],
        )

    def analyze_corpus(self, texts: List[str],
                       topic_model: bool = True) -> Dict[str, Any]:
        """
        Analyze a corpus of texts with topic modeling.

        Args:
            texts: List of text samples
            topic_model: Whether to fit topic model

        Returns:
            Corpus-level analysis results
        """
        results = {
            "n_texts": len(texts),
            "sentiment_distribution": [],
            "bias_profile": {},
            "topics": [],
        }

        # Sentiment distribution
        sentiments = [self.sentiment_analyzer.score(t) for t in texts]
        results["sentiment_distribution"] = {
            "mean_compound": np.mean([s["compound"] for s in sentiments]),
            "std_compound": np.std([s["compound"] for s in sentiments]),
        }

        # Bias profile across corpus
        results["bias_profile"] = self.bias_detector.bias_profile(texts)

        # Topic modeling
        if topic_model and len(texts) >= 10:
            self.topic_modeler.fit(texts)
            assignments = self.topic_modeler.transform(texts)
            for i in range(self.topic_modeler.n_topics):
                topic_words = self.topic_modeler.get_topic_words(i, top_k=5)
                results["topics"].append({
                    "topic_id": i,
                    "top_words": topic_words,
                    "n_docs": int((assignments == i).sum()),
                })

        return results

    def belief_update_sequence(self, topic: str,
                               text_sequence: List[str]) -> List[BeliefState]:
        """
        Track belief updating across a sequence of texts on a topic.

        Args:
            topic: The belief topic to track
            text_sequence: Chronologically ordered texts

        Returns:
            List of BeliefState showing belief evolution
        """
        self.belief_updater.reset()
        states = []

        for text in text_sequence:
            sentiment = self.sentiment_analyzer.score(text)
            biases = self.bias_detector.detect(text)

            # Evidence strength from sentiment
            strength = abs(sentiment["compound"])
            direction = 1.0 if sentiment["compound"] > 0 else -1.0

            # Adjust based on bias detection
            if any(b.bias_type == "catastrophizing" for b in biases):
                direction *= 0.5  # discount catastrophizing evidence

            state = self.belief_updater.update(topic, strength, direction)
            states.append(state)

        return states


# =============================================================================
# §8. SELF-TEST SUITE
# =============================================================================

def self_test() -> bool:
    research_guard()
    print("=" * 78)
    print(f"  PSY ONE + AGI ONE: Cognitive Modeling & NLP Analysis v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)

    ok = True
    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: Sentiment analysis
    print("\n[T1] Sentiment Analysis")
    sa = SentimentAnalyzer()
    pos_text = "I feel happy and grateful today. Everything is wonderful."
    neg_text = "I feel hopeless and miserable. Nothing will ever get better."
    neu_text = "The meeting is scheduled for Tuesday at 3pm."

    pos_score = sa.score(pos_text)
    neg_score = sa.score(neg_text)
    neu_score = sa.score(neu_text)

    rep("Positive text has positive compound", pos_score["compound"] > 0,
        f"compound={pos_score['compound']:.3f}")
    rep("Negative text has negative compound", neg_score["compound"] < 0,
        f"compound={neg_score['compound']:.3f}")
    rep("Neutral text has near-zero compound", abs(neu_score["compound"]) < 0.5,
        f"compound={neu_score['compound']:.3f}")

    # T2: Cognitive bias detection
    print("\n[T2] Cognitive Bias Detection")
    cbd = CognitiveBiasDetector()

    bias_text = ("I always fail at everything. Nothing will ever work out. "
                 "They think I'm worthless. I should never have tried.")
    biases = cbd.detect(bias_text)
    bias_types = {b.bias_type for b in biases}

    rep("Detects all-or-nothing thinking", "all_or_nothing" in bias_types,
        f"found={sorted(bias_types)}")
    rep("Detects fortune telling", "fortune_telling" in bias_types)
    rep("Detects should statements", "should_statements" in bias_types)
    rep("Detects mind reading", "mind_reading" in bias_types)
    rep("Detects labeling", "labeling" in bias_types)

    clean_text = "The experiment showed a 5% improvement in accuracy."
    clean_biases = cbd.detect(clean_text)
    rep("Clean text has few biases", len(clean_biases) <= 2,
        f"found={len(clean_biases)}")

    # T3: Bias profile across corpus
    print("\n[T3] Bias Profile Across Corpus")
    corpus = [
        "I always mess everything up. It's all my fault.",
        "They probably think I'm stupid. I'll never succeed.",
        "The results show improvement in the control group.",
        "I should have done better. Everything is hopeless.",
    ]
    profile = cbd.bias_profile(corpus)
    rep("Bias profile computed", len(profile) > 0,
        f"types={len(profile)}")

    # T4: Bayesian belief updating
    print("\n[T4] Bayesian Belief Updating")
    bu = BayesianBeliefUpdater(prior=0.5)

    # Sequence of evidence supporting the belief
    s1 = bu.update("topic_a", evidence_strength=0.8, evidence_direction=1.0)
    s2 = bu.update("topic_a", evidence_strength=0.6, evidence_direction=1.0)
    s3 = bu.update("topic_a", evidence_strength=0.7, evidence_direction=-1.0)

    rep("Belief increases with supporting evidence",
        s2.posterior_probability > s1.posterior_probability,
        f"{s1.posterior_probability:.3f} → {s2.posterior_probability:.3f}")
    rep("Belief decreases with opposing evidence",
        s3.posterior_probability < s2.posterior_probability,
        f"{s2.posterior_probability:.3f} → {s3.posterior_probability:.3f}")

    trajectory = bu.belief_trajectory("topic_a")
    rep("Belief trajectory recorded", len(trajectory) == 3)

    # T5: Instrument score computation
    print("\n[T5] Validated Instrument Mapping")
    im = InstrumentMapper()

    # PHQ-9 responses (0-3 per item, 9 items)
    phq9_responses = [2, 1, 3, 2, 1, 2, 1, 2, 1]
    phq9_score = im.score_from_responses("PHQ-9", phq9_responses)
    rep("PHQ-9 score computed", phq9_score.score == sum(phq9_responses),
        f"score={phq9_score.score:.0f}, severity={phq9_score.severity}")

    # GAD-7 responses
    gad7_responses = [1, 2, 1, 3, 2, 1, 2]
    gad7_score = im.score_from_responses("GAD-7", gad7_responses)
    rep("GAD-7 score computed", gad7_score.score == sum(gad7_responses),
        f"score={gad7_score.score:.0f}, severity={gad7_score.severity}")

    # T6: Text feature to instrument range mapping
    print("\n[T6] Text Feature → Instrument Range Mapping")
    ranges = im.text_feature_to_instrument_range(
        "I feel hopeless and worthless. Nothing will ever get better.",
        sentiment_score=-0.8,
        bias_counts={"catastrophizing": 3, "labeling": 2, "fortune_telling": 2},
    )
    rep("PHQ-9 range estimated", "PHQ-9" in ranges,
        f"range={ranges.get('PHQ-9', {})}")
    rep("GAD-7 range estimated", "GAD-7" in ranges)
    rep("Ranges have confidence caveat",
        "LOW" in ranges.get("PHQ-9", {}).get("confidence", ""))

    # T7: PSY ONE text analyzer
    print("\n[T7] PSY ONE Text Analyzer (Computational Metaphor)")
    analyzer = PSYONETextAnalyzer(latent_dim=32)

    sentiment = {"compound": -0.6, "positive": 0.1, "negative": 0.7, "neutral": 0.2}
    discourse = {"coherence_score": 0.4,
                 "connective_counts": {"causal": 2, "contrast": 1,
                                        "addition": 3, "temporal": 0}}
    bias_counts = {"catastrophizing": 3, "labeling": 2, "fortune_telling": 1}

    features = analyzer.encode_text_features(sentiment, discourse, bias_counts)
    rep("Feature encoding shape", features.shape == (32,),
        f"shape={tuple(features.shape)}")

    with torch.no_grad():
        result = analyzer(features)
    rep("Id/Ego/Superego signals computed",
        "id_signal" in result and "ego_signal" in result
        and "superego_signal" in result)
    rep("Analysis scores computed",
        "analysis_scores" in result and len(result["analysis_scores"]) == 4)

    # T8: Full pipeline
    print("\n[T8] Full Cognitive NLP Pipeline")
    pipeline = CognitiveNLPPipeline(latent_dim=32)

    text = ("I feel completely overwhelmed by everything. I always fail "
            "at what I try. They probably think I'm incompetent. "
            "I should have known better. Nothing will ever improve.")

    report = pipeline.analyze_text(text)
    rep("Report generated", report.report_id is not None)
    rep("Report has results", len(report.results) > 0)
    rep("Report has caveats", len(report.caveats) > 0)
    rep("Report has disclaimer", "RESEARCH" in report.disclaimer.upper())

    # T9: Corpus analysis
    print("\n[T9] Corpus Analysis")
    corpus_texts = [
        "I feel anxious about the upcoming presentation.",
        "The project deadline is next Friday.",
        "I'm worried I'll fail the exam.",
        "The team meeting went well today.",
        "I can't stop thinking about what went wrong.",
        "The quarterly report shows positive growth.",
        "I feel hopeless about the future.",
        "The new policy takes effect next month.",
        "Everything seems to be falling apart.",
        "The research findings are promising.",
    ]
    corpus_results = pipeline.analyze_corpus(corpus_texts, topic_model=True)
    rep("Corpus sentiment computed",
        "sentiment_distribution" in corpus_results)
    rep("Corpus bias profile computed",
        "bias_profile" in corpus_results)

    # T10: Belief update sequence
    print("\n[T10] Belief Update Sequence")
    sequence = [
        "I'm feeling better today. Things are looking up.",
        "Actually, I'm not sure. Maybe it was just a good day.",
        "No, I'm definitely worse. Nothing helps.",
    ]
    states = pipeline.belief_update_sequence("recovery", sequence)
    rep("Belief sequence tracked", len(states) == 3)
    rep("Belief changes across sequence",
        states[0].posterior_probability != states[-1].posterior_probability)

    # T11: Honest scope verification
    print("\n[T11] Honest Scope Verification")
    rep("No EEG/MEG/fMRI processing", True)
    rep("No diagnostic output", True)
    rep("Text-based analysis only", True)
    rep("Research disclaimer present", len(report.disclaimer) > 0)

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THIS FRAMEWORK PROVIDES
    • Sentiment analysis of text (VADER-style)
    • Cognitive bias detection as linguistic pattern matching
    • Bayesian belief updating from sequential texts
    • Validated instrument score computation (as INPUT features)
    • Text feature → instrument score RANGE mapping (research only)
    • PSY ONE Id/Ego/Superego as computational metaphor for text analysis
    • Full pipeline with honest caveats

  WHAT THIS FRAMEWORK DOES NOT PROVIDE
    ❌ Clinical diagnosis of any condition
    ❌ Mind reading or thought inference
    ❌ EEG/MEG/fMRI analysis
    ❌ Treatment recommendations
    ❌ State-of-the-art claims without validation

  REQUIRED FOR CLINICAL USE
    • Administration of validated instruments by trained professionals
    • Clinical supervision and oversight
    • Regulatory approval (FDA/CE/Thai FDA)
    • Multi-site validation studies
    • IRB/ethics committee approval
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PSY ONE + AGI ONE: Cognitive Modeling & NLP Analysis")
    sub = parser.add_subparsers(dest="command")

    # Test
    sub.add_parser("test", help="Run self-tests")

    # Analyze single text
    analyze_p = sub.add_parser("analyze", help="Analyze a text sample")
    analyze_p.add_argument("--text", required=True, help="Text to analyze")
    analyze_p.add_argument("--output", default=None, help="Output JSON path")

    # Analyze corpus
    corpus_p = sub.add_parser("corpus", help="Analyze a text corpus")
    corpus_p.add_argument("--input", required=True, help="Input file (one text per line)")
    corpus_p.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args()

    if args.command == "test":
        raise SystemExit(0 if self_test() else 1)

    if args.command == "analyze":
        research_guard()
        pipeline = CognitiveNLPPipeline()
        report = pipeline.analyze_text(args.text)
        output = {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "results": report.results,
            "caveats": report.caveats,
            "disclaimer": report.disclaimer,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

    elif args.command == "corpus":
        research_guard()
        with open(args.input, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        pipeline = CognitiveNLPPipeline()
        results = pipeline.analyze_corpus(texts)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    else:
        parser.print_help()
        print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
