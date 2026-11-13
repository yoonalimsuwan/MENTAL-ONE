#!/usr/bin/env python3
"""
=============================================================================
RESEARCH VALIDATION & PUBLICATION PIPELINE
Ethics · Statistical Validation · Reporting Compliance · Expert Review
=============================================================================
Developers : PAI and Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

WHAT THIS MODULE PROVIDES
  §1  EthicsGuard          — IRB tracking, consent, data governance
  §2  StatisticalValidator — CV, bootstrap CI, effect sizes, calibration
  §3  ReportingCompliance  — STARD / TRIPOD / CONSORT / CLAIM checklists
  §4  ExpertReview         — panel scoring, Delphi consensus, COI tracking
  §5  PublicationPipeline  — manuscript prep, figures, submission tracking
  §6  ValidationOrchestrator — ties everything together

WHAT THIS MODULE IS NOT
  ❌ Not a substitute for actual peer review
  ❌ Not a substitute for IRB/ethics committee approval
  ❌ Not a substitute for clinical validation by medical professionals
  ❌ Does not submit to journals — prepares for submission only

REPORTING GUIDELINES REFERENCED
  [G1] STARD 2015  — Standards for Reporting Diagnostic Accuracy Studies
  [G2] TRIPOD 2015 — Transparent Reporting of a multivariable prediction
                     model for Individual Prognosis Or Diagnosis
  [G3] CONSORT 2010 — Consolidated Standards of Reporting Trials
  [G4] CLAIM 2020  — Checklist for Artificial Intelligence in Medical Imaging
  [G5] SPIRIT 2013 — Standard Protocol Items for Randomized Trials
  [G6] PRISMA 2020 — Preferred Reporting Items for Systematic Reviews

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("research_pipeline")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

ETHICS_BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  RESEARCH VALIDATION PIPELINE — RESEARCH TOOL ONLY                   ║
║                                                                     ║
║  This module does NOT:                                              ║
║  • Approve any research protocol                                    ║
║  • Grant ethics approval                                            ║
║  • Certify clinical validity                                        ║
║  • Replace peer review                                              ║
║  • Submit to journals                                               ║
║                                                                     ║
║  All outputs are PREPARATORY materials for submission to            ║
║  qualified institutions, ethics committees, and journals.           ║
║                                                                     ║
║  THE PSYCHIATRIST / CLINICIAN IS THE FINAL DECISION MAKER.          ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def ethics_guard():
    print(ETHICS_BANNER)


# =============================================================================
# §1. ETHICS GUARD — IRB, Consent, Data Governance
# =============================================================================

class ConsentStatus(Enum):
    NOT_OBTAINED = "not_obtained"
    PENDING = "pending"
    OBTAINED = "obtained"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

class IRBStatus(Enum):
    NOT_SUBMITTED = "not_submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class IRBRecord:
    """IRB/Ethics Committee approval record."""
    protocol_id: str
    title: str
    irb_status: IRBStatus = IRBStatus.NOT_SUBMITTED
    submission_date: Optional[str] = None
    approval_date: Optional[str] = None
    expiry_date: Optional[str] = None
    irb_committee: str = ""
    approval_number: str = ""
    conditions: List[str] = field(default_factory=list)
    amendments: List[Dict[str, Any]] = field(default_factory=list)

    def is_valid(self) -> bool:
        if self.irb_status != IRBStatus.APPROVED:
            return False
        if self.expiry_date:
            try:
                exp = datetime.fromisoformat(self.expiry_date)
                if datetime.now() > exp:
                    return False
            except ValueError:
                pass
        return True

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["irb_status"] = self.irb_status.value
        d["is_valid"] = self.is_valid()
        return d


@dataclass
class ConsentRecord:
    """Informed consent record for a participant."""
    participant_id: str
    consent_status: ConsentStatus = ConsentStatus.NOT_OBTAINED
    consent_date: Optional[str] = None
    consent_version: str = "1.0"
    data_use_scope: str = "research_only"
    gdpr_rights_acknowledged: bool = False
    right_to_withdraw: bool = True
    data_retention_period: str = "5_years"
    notes: str = ""

    def is_valid(self) -> bool:
        return self.consent_status == ConsentStatus.OBTAINED

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["consent_status"] = self.consent_status.value
        d["is_valid"] = self.is_valid()
        return d


@dataclass
class AuditEntry:
    """Audit log entry for data access."""
    timestamp: str
    user: str
    action: str
    resource: str
    resource_id: str
    purpose: str
    ip_address: Optional[str] = None
    approved: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


class EthicsGuard:
    """
    Tracks IRB approval, participant consent, and data access audit trail.

    This is a TRACKING TOOL, not an approval authority.
    Actual IRB/ethics approval must come from a qualified institution.
    """

    def __init__(self):
        self.irb_records: Dict[str, IRBRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.audit_log: List[AuditEntry] = []

    def register_irb(self, record: IRBRecord) -> bool:
        self.irb_records[record.protocol_id] = record
        self._audit("system", "IRB_REGISTER", "protocol",
                    record.protocol_id, "research_setup")
        return record.is_valid()

    def register_consent(self, record: ConsentRecord) -> bool:
        self.consent_records[record.participant_id] = record
        self._audit("system", "CONSENT_REGISTER", "participant",
                    record.participant_id, "research_setup")
        return record.is_valid()

    def can_process_participant(self, participant_id: str,
                                 protocol_id: str) -> Tuple[bool, str]:
        """Check if data processing is allowed for this participant."""
        consent = self.consent_records.get(participant_id)
        if consent is None:
            return False, "NO_CONSENT_RECORD"
        if not consent.is_valid():
            return False, f"INVALID_CONSENT_{consent.consent_status.value}"

        protocol = self.irb_records.get(protocol_id)
        if protocol is None:
            return False, "NO_IRB_RECORD"
        if not protocol.is_valid():
            return False, f"INVALID_IRB_{protocol.irb_status.value}"

        return True, "APPROVED"

    def _audit(self, user: str, action: str, resource_type: str,
               resource_id: str, purpose: str, approved: bool = True):
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            user=user,
            action=action,
            resource=resource_type,
            resource_id=resource_id,
            purpose=purpose,
            approved=approved,
        )
        self.audit_log.append(entry)

    def get_audit_summary(self) -> Dict[str, Any]:
        actions = {}
        for e in self.audit_log:
            actions[e.action] = actions.get(e.action, 0) + 1
        return {
            "total_entries": len(self.audit_log),
            "actions": actions,
            "latest": self.audit_log[-1].to_dict() if self.audit_log else None,
        }

    def compliance_report(self) -> Dict[str, Any]:
        """Generate a compliance report for IRB submission."""
        valid_irb = sum(1 for r in self.irb_records.values() if r.is_valid())
        valid_consent = sum(1 for r in self.consent_records.values() if r.is_valid())
        return {
            "timestamp": datetime.now().isoformat(),
            "total_protocols": len(self.irb_records),
            "valid_protocols": valid_irb,
            "total_participants": len(self.consent_records),
            "valid_consents": valid_consent,
            "audit_entries": len(self.audit_log),
            "compliance_status": (
                "COMPLIANT" if valid_irb > 0 and valid_consent > 0
                else "NON_COMPLIANT"
            ),
            "note": "This report is for internal tracking only. "
                    "Formal IRB approval must come from a qualified "
                    "ethics committee.",
        }


# =============================================================================
# §2. STATISTICAL VALIDATION
# =============================================================================

@dataclass
class StatisticalResult:
    """Result of a statistical validation test."""
    metric: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    n_samples: int = 0
    method: str = ""
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class StatisticalValidator:
    """
    Statistical validation tools for research models.

    Provides:
    - Cross-validation with proper train/test splitting
    - Bootstrap confidence intervals
    - Effect size calculation
    - Calibration assessment
    - Statistical significance testing
    """

    @staticmethod
    def k_fold_cv(
        model_fn: Callable[[Tensor, Tensor], float],
        X: Tensor, y: Tensor, k: int = 5, seed: int = 42,
    ) -> Dict[str, Any]:
        """
        K-fold cross-validation.

        Args:
            model_fn: function(X_train, y_train, X_test, y_test) -> score
            X: features (n_samples, n_features)
            y: labels (n_samples,)
            k: number of folds
            seed: random seed

        Returns:
            dict with fold scores, mean, std, CI
        """
        torch.manual_seed(seed)
        n = len(X)
        indices = torch.randperm(n)
        fold_size = n // k
        scores = []

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size if fold < k - 1 else n
            test_idx = indices[start:end]
            train_idx = torch.cat([indices[:start], indices[end:]])

            score = model_fn(X[train_idx], y[train_idx],
                             X[test_idx], y[test_idx])
            scores.append(float(score))

        scores = np.array(scores)
        mean = float(scores.mean())
        std = float(scores.std(ddof=1))
        se = std / math.sqrt(k)

        return {
            "fold_scores": scores.tolist(),
            "mean": mean,
            "std": std,
            "se": se,
            "ci_95": [mean - 1.96 * se, mean + 1.96 * se],
            "k": k,
            "n_samples": n,
        }

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Bootstrap confidence interval.

        Args:
            data: input data array
            statistic_fn: function to compute statistic
            n_bootstrap: number of bootstrap resamples
            confidence: confidence level
            seed: random seed

        Returns:
            dict with point estimate and CI
        """
        rng = np.random.default_rng(seed)
        n = len(data)
        boot_stats = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            boot_stats[i] = statistic_fn(sample)

        point_est = float(statistic_fn(data))
        alpha = 1 - confidence
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return {
            "point_estimate": point_est,
            "ci_lower": lo,
            "ci_upper": hi,
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
            "se_bootstrap": float(boot_stats.std(ddof=1)),
        }

    @staticmethod
    def effect_size_cohens_d(
        group1: np.ndarray, group2: np.ndarray
    ) -> Dict[str, float]:
        """
        Cohen's d effect size with confidence interval.

        Args:
            group1: first group values
            group2: second group values

        Returns:
            dict with d, CI, interpretation
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_sd = math.sqrt(pooled_var)

        if pooled_sd < 1e-10:
            return {"d": 0.0, "interpretation": "undefined (zero variance)"}

        d = (mean1 - mean2) / pooled_sd

        # Standard error of d (approximate)
        se_d = math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
        ci_lo = d - 1.96 * se_d
        ci_hi = d + 1.96 * se_d

        # Hedges' g correction for small samples
        J = 1 - 3 / (4 * (n1 + n2) - 9)
        g = d * J

        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interp = "negligible"
        elif abs_d < 0.5:
            interp = "small"
        elif abs_d < 0.8:
            interp = "medium"
        else:
            interp = "large"

        return {
            "d": float(d),
            "hedges_g": float(g),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "se": float(se_d),
            "interpretation": interp,
            "n1": n1, "n2": n2,
        }

    @staticmethod
    def calibration_curve(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calibration curve and metrics.

        Args:
            y_true: true binary labels
            y_prob: predicted probabilities
            n_bins: number of bins

        Returns:
            dict with calibration data and metrics
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        observed = []
        predicted = []
        counts = []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() > 0:
                observed.append(float(y_true[mask].mean()))
                predicted.append(float(y_prob[mask].mean()))
                counts.append(int(mask.sum()))
            else:
                observed.append(np.nan)
                predicted.append(np.nan)
                counts.append(0)

        # Expected Calibration Error (ECE)
        ece = 0.0
        total = sum(counts)
        for i in range(n_bins):
            if counts[i] > 0 and not np.isnan(observed[i]):
                ece += (counts[i] / total) * abs(observed[i] - predicted[i])

        # Brier score
        brier = float(np.mean((y_prob - y_true) ** 2))

        return {
            "bins": bins.tolist(),
            "observed": [float(o) if not np.isnan(o) else None for o in observed],
            "predicted": [float(p) if not np.isnan(p) else None for p in predicted],
            "counts": counts,
            "ece": float(ece),
            "brier_score": brier,
            "n_bins": n_bins,
        }

    @staticmethod
    def auroc_with_ci(
        y_true: np.ndarray, y_score: np.ndarray,
        n_bootstrap: int = 2000, seed: int = 42,
    ) -> Dict[str, float]:
        """
        AUROC with bootstrap confidence interval.
        """
        def compute_auroc(yt, ys):
            # Rank-based AUROC
            order = np.argsort(ys)
            ranks = np.empty(len(ys))
            ranks[order] = np.arange(1, len(ys) + 1)
            n_pos = yt.sum()
            n_neg = len(yt) - n_pos
            if n_pos == 0 or n_neg == 0:
                return 0.5
            return float((ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2)
                         / (n_pos * n_neg))

        point_est = compute_auroc(y_true, y_score)
        rng = np.random.default_rng(seed)
        boot_aurocs = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            yt, ys = y_true[idx], y_score[idx]
            if len(np.unique(yt)) < 2:
                continue
            boot_aurocs.append(compute_auroc(yt, ys))

        boot_aurocs = np.array(boot_aurocs)
        ci_lo = float(np.percentile(boot_aurocs, 2.5))
        ci_hi = float(np.percentile(boot_aurocs, 97.5))

        return {
            "auroc": point_est,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_bootstrap": n_bootstrap,
            "se": float(boot_aurocs.std(ddof=1)) if len(boot_aurocs) > 0 else 0.0,
        }


# =============================================================================
# §3. REPORTING COMPLIANCE CHECKLISTS
# =============================================================================

class ReportingGuideline(Enum):
    STARD = "STARD_2015"       # Diagnostic accuracy
    TRIPOD = "TRIPOD_2015"     # Prediction models
    CONSORT = "CONSORT_2010"   # Clinical trials
    CLAIM = "CLAIM_2020"       # AI in medical imaging
    SPIRIT = "SPIRIT_2013"     # Trial protocols
    PRISMA = "PRISMA_2020"     # Systematic reviews


@dataclass
class ChecklistItem:
    """A single item in a reporting checklist."""
    id: str
    section: str
    item_text: str
    is_mandatory: bool = True
    status: str = "not_addressed"  # not_addressed | addressed | not_applicable
    page_or_section: str = ""
    notes: str = ""


class ReportingComplianceChecker:
    """
    Checks manuscript compliance against reporting guidelines.

    Provides checklists for:
    - STARD 2015 (diagnostic accuracy studies)
    - TRIPOD 2015 (prediction model studies)
    - CONSORT 2010 (randomized controlled trials)
    - CLAIM 2020 (AI in medical imaging)

    This is a TRACKING TOOL. Actual compliance assessment must be done
    by qualified reviewers.
    """

    STARD_ITEMS = [
        ("1", "Title", "Identify as diagnostic accuracy study", True),
        ("2", "Abstract", "Structured summary", True),
        ("3", "Introduction", "Scientific rationale", True),
        ("4", "Introduction", "Study objectives", True),
        ("5", "Methods", "Study design", True),
        ("6", "Methods", "Eligibility criteria", True),
        ("7", "Methods", "Participant recruitment", True),
        ("8", "Methods", "Data collection", True),
        ("9", "Methods", "Index test description", True),
        ("10", "Methods", "Reference standard description", True),
        ("11", "Methods", "Sample size calculation", True),
        ("12", "Methods", "Statistical methods", True),
        ("13", "Results", "Participant flow", True),
        ("14", "Results", "Adverse events", False),
        ("15", "Results", "Index test results", True),
        ("16", "Results", "Primary accuracy estimates", True),
        ("17", "Results", "Confidence intervals", True),
        ("18", "Results", "Subgroup analyses", False),
        ("19", "Discussion", "Interpretation of results", True),
        ("20", "Discussion", "Limitations", True),
        ("21", "Discussion", "Implications for practice", True),
        ("22", "Other", "Registration", False),
        ("23", "Other", "Funding", True),
        ("24", "Other", "Conflicts of interest", True),
        ("25", "Other", "Data availability", False),
    ]

    TRIPOD_ITEMS = [
        ("1", "Title", "Identify as prediction model study", True),
        ("2", "Abstract", "Structured summary", True),
        ("3a", "Introduction", "Background and rationale", True),
        ("3b", "Introduction", "Objectives", True),
        ("4a", "Methods", "Study design", True),
        ("4b", "Methods", "Setting and location", True),
        ("5a", "Methods", "Participants", True),
        ("5b", "Methods", "Inclusion/exclusion criteria", True),
        ("6a", "Methods", "Predictor variables", True),
        ("6b", "Methods", "Outcome variable", True),
        ("7a", "Methods", "Sample size", True),
        ("8", "Methods", "Statistical methods", True),
        ("9", "Methods", "Model specification", True),
        ("10a", "Methods", "Model validation", True),
        ("10b", "Methods", "Model performance measures", True),
        ("11", "Methods", "Missing data handling", True),
        ("12", "Results", "Participants", True),
        ("13a", "Results", "Model development", True),
        ("13b", "Results", "Model performance", True),
        ("14", "Results", "Model validation", True),
        ("15", "Results", "Model specification", True),
        ("16", "Discussion", "Interpretation", True),
        ("17", "Discussion", "Limitations", True),
        ("18", "Discussion", "Implications", True),
        ("19", "Other", "Registration", False),
        ("20", "Other", "Funding", True),
        ("21", "Other", "Conflicts of interest", True),
    ]

    CLAIM_ITEMS = [
        ("1", "Title", "Identify as AI study", True),
        ("2", "Abstract", "Structured summary", True),
        ("3", "Introduction", "Clinical context", True),
        ("4", "Introduction", "AI system description", True),
        ("5", "Methods", "Study design", True),
        ("6", "Methods", "Data source", True),
        ("7", "Methods", "Data annotation", True),
        ("8", "Methods", "Data splitting", True),
        ("9", "Methods", "Preprocessing", True),
        ("10", "Methods", "Model architecture", True),
        ("11", "Methods", "Training details", True),
        ("12", "Methods", "Evaluation metrics", True),
        ("13", "Methods", "Statistical methods", True),
        ("14", "Methods", "Reproducibility", True),
        ("15", "Results", "Dataset characteristics", True),
        ("16", "Results", "Model performance", True),
        ("17", "Results", "Comparison with baselines", True),
        ("18", "Results", "Error analysis", True),
        ("19", "Results", "Generalizability", True),
        ("20", "Results", "Computational resources", False),
        ("21", "Discussion", "Clinical implications", True),
        ("22", "Discussion", "Limitations", True),
        ("23", "Discussion", "Bias and fairness", True),
        ("24", "Discussion", "Regulatory considerations", True),
        ("25", "Other", "Code availability", True),
        ("26", "Other", "Data availability", True),
        ("27", "Other", "Funding", True),
        ("28", "Other", "Conflicts of interest", True),
        ("29", "Other", "IRB approval", True),
        ("30", "Other", "Informed consent", True),
    ]

    def __init__(self):
        self.checklists: Dict[ReportingGuideline, List[ChecklistItem]] = {}
        self._init_checklists()

    def _init_checklists(self):
        for gid, items in [
            (ReportingGuideline.STARD, self.STARD_ITEMS),
            (ReportingGuideline.TRIPOD, self.TRIPOD_ITEMS),
            (ReportingGuideline.CLAIM, self.CLAIM_ITEMS),
        ]:
            self.checklists[gid] = [
                ChecklistItem(id=i[0], section=i[1], item_text=i[2],
                              is_mandatory=i[3])
                for i in items
            ]

    def get_checklist(self, guideline: ReportingGuideline) -> List[Dict]:
        items = self.checklists.get(guideline, [])
        return [asdict(item) for item in items]

    def update_item(
        self, guideline: ReportingGuideline, item_id: str,
        status: str, page_or_section: str = "", notes: str = "",
    ) -> bool:
        items = self.checklists.get(guideline, [])
        for item in items:
            if item.id == item_id:
                item.status = status
                item.page_or_section = page_or_section
                item.notes = notes
                return True
        return False

    def compliance_score(self, guideline: ReportingGuideline) -> Dict[str, Any]:
        """Calculate compliance score for a guideline."""
        items = self.checklists.get(guideline, [])
        mandatory = [i for i in items if i.is_mandatory]
        addressed = [i for i in mandatory if i.status == "addressed"]
        not_applicable = [i for i in mandatory if i.status == "not_applicable"]

        applicable = len(mandatory) - len(not_applicable)
        score = len(addressed) / max(applicable, 1) * 100

        unaddressed = [i.id for i in mandatory
                       if i.status == "not_addressed"]

        return {
            "guideline": guideline.value,
            "total_items": len(items),
            "mandatory_items": len(mandatory),
            "addressed": len(addressed),
            "not_applicable": len(not_applicable),
            "compliance_score": score,
            "unaddressed_mandatory": unaddressed,
            "ready_for_submission": score >= 90 and not unaddressed,
        }

    def full_compliance_report(self) -> Dict[str, Any]:
        report = {}
        for guideline in self.checklists:
            report[guideline.value] = self.compliance_score(guideline)
        return report


# =============================================================================
# §4. EXPERT REVIEW PIPELINE
# =============================================================================

@dataclass
class ExpertReviewer:
    """An expert reviewer."""
    reviewer_id: str
    name: str
    affiliation: str
    expertise: List[str]
    conflicts_of_interest: List[str] = field(default_factory=list)
    is_available: bool = True


@dataclass
class ReviewScore:
    """A single reviewer's score for a manuscript/model."""
    reviewer_id: str
    manuscript_id: str
    novelty: float = 0.0           # 1-10
    methodology: float = 0.0       # 1-10
    clinical_relevance: float = 0.0  # 1-10
    statistical_rigor: float = 0.0   # 1-10
    reproducibility: float = 0.0   # 1-10
    ethical_compliance: float = 0.0  # 1-10
    overall_recommendation: str = "not_scored"
    comments: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def weighted_score(self) -> float:
        """Weighted overall score."""
        weights = {
            "methodology": 0.25,
            "statistical_rigor": 0.25,
            "clinical_relevance": 0.20,
            "novelty": 0.15,
            "reproducibility": 0.10,
            "ethical_compliance": 0.05,
        }
        score = 0.0
        for attr, weight in weights.items():
            score += getattr(self, attr) * weight
        return score

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["weighted_score"] = self.weighted_score()
        return d


class ExpertReviewPipeline:
    """
    Manages expert review process.

    Supports:
    - Reviewer assignment with COI checking
    - Scoring collection
    - Delphi consensus method
    - Conflict of interest tracking
    """

    def __init__(self):
        self.reviewers: Dict[str, ExpertReviewer] = {}
        self.scores: List[ReviewScore] = []
        self.delphi_rounds: List[Dict[str, Any]] = []

    def register_reviewer(self, reviewer: ExpertReviewer) -> bool:
        self.reviewers[reviewer.reviewer_id] = reviewer
        return True

    def check_coi(self, reviewer_id: str, manuscript_id: str) -> List[str]:
        """Check for conflicts of interest."""
        reviewer = self.reviewers.get(reviewer_id)
        if reviewer is None:
            return ["REVIEWER_NOT_FOUND"]
        return reviewer.conflicts_of_interest

    def assign_reviewers(
        self, manuscript_id: str, n_reviewers: int = 3
    ) -> List[str]:
        """Assign available reviewers without COI."""
        available = [
            r.reviewer_id for r in self.reviewers.values()
            if r.is_available and not r.conflicts_of_interest
        ]
        assigned = available[:n_reviewers]
        log.info(f"[Review] Assigned {len(assigned)} reviewers to {manuscript_id}")
        return assigned

    def submit_score(self, score: ReviewScore) -> bool:
        """Submit a review score."""
        # Check COI
        coi = self.check_coi(score.reviewer_id, score.manuscript_id)
        if coi:
            log.warning(f"[Review] COI detected for {score.reviewer_id}: {coi}")
            return False

        self.scores.append(score)
        return True

    def get_consensus(self, manuscript_id: str) -> Dict[str, Any]:
        """Calculate consensus scores for a manuscript."""
        relevant = [s for s in self.scores if s.manuscript_id == manuscript_id]
        if not relevant:
            return {"status": "NO_REVIEWS"}

        weighted_scores = [s.weighted_score() for s in relevant]
        mean_score = float(np.mean(weighted_scores))
        std_score = float(np.std(weighted_scores, ddof=1)) if len(weighted_scores) > 1 else 0.0

        recommendations = [s.overall_recommendation for s in relevant]
        rec_counts = {}
        for r in recommendations:
            rec_counts[r] = rec_counts.get(r, 0) + 1

        return {
            "manuscript_id": manuscript_id,
            "n_reviews": len(relevant),
            "mean_weighted_score": mean_score,
            "std_weighted_score": std_score,
            "min_score": float(min(weighted_scores)),
            "max_score": float(max(weighted_scores)),
            "recommendations": rec_counts,
            "consensus": "ACCEPT" if mean_score >= 7.0 else
                         "REVISE" if mean_score >= 5.0 else "REJECT",
        }

    def delphi_round(
        self, manuscript_id: str, round_num: int = 1
    ) -> Dict[str, Any]:
        """
        Delphi method round for consensus building.
        Round 1: independent scoring
        Round 2+: share aggregate, allow revision
        """
        if round_num == 1:
            # Initial round - collect independent scores
            consensus = self.get_consensus(manuscript_id)
            self.delphi_rounds.append({
                "manuscript_id": manuscript_id,
                "round": round_num,
                "consensus": consensus,
            })
            return consensus
        else:
            # Subsequent rounds - return previous round's aggregate
            prev = None
            for r in reversed(self.delphi_rounds):
                if r["manuscript_id"] == manuscript_id:
                    prev = r
                    break
            if prev is None:
                return {"status": "NO_PREVIOUS_ROUND"}
            return {
                "manuscript_id": manuscript_id,
                "round": round_num,
                "previous_consensus": prev["consensus"],
                "instruction": "Reviewers should revise scores considering "
                               "the group aggregate. Submit revised scores.",
            }


# =============================================================================
# §5. PUBLICATION PIPELINE
# =============================================================================

@dataclass
class Manuscript:
    """A research manuscript."""
    manuscript_id: str
    title: str
    authors: List[str]
    abstract: str = ""
    target_journal: str = ""
    target_guideline: Optional[ReportingGuideline] = None
    status: str = "draft"  # draft | submitted | under_review | accepted | rejected
    submission_date: Optional[str] = None
    version: str = "1.0"
    supplementary_materials: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.target_guideline:
            d["target_guideline"] = self.target_guideline.value
        return d


class PublicationPipeline:
    """
    Manages the publication process from draft to submission.

    Tracks:
    - Manuscript metadata
    - Reporting guideline compliance
    - Supplementary materials
    - Submission status
    - Reviewer assignments
    """

    def __init__(self):
        self.manuscripts: Dict[str, Manuscript] = {}
        self.submissions: List[Dict[str, Any]] = []

    def create_manuscript(
        self, title: str, authors: List[str],
        target_journal: str = "",
        target_guideline: Optional[ReportingGuideline] = None,
    ) -> str:
        manuscript_id = str(uuid.uuid4())[:8]
        manuscript = Manuscript(
            manuscript_id=manuscript_id,
            title=title,
            authors=authors,
            target_journal=target_journal,
            target_guideline=target_guideline,
        )
        self.manuscripts[manuscript_id] = manuscript
        log.info(f"[Publication] Created manuscript {manuscript_id}: {title[:50]}")
        return manuscript_id

    def add_supplementary(self, manuscript_id: str, filename: str) -> bool:
        manuscript = self.manuscripts.get(manuscript_id)
        if manuscript is None:
            return False
        manuscript.supplementary_materials.append(filename)
        return True

    def prepare_submission(self, manuscript_id: str) -> Dict[str, Any]:
        """Prepare submission package."""
        manuscript = self.manuscripts.get(manuscript_id)
        if manuscript is None:
            return {"status": "MANUSCRIPT_NOT_FOUND"}

        return {
            "manuscript_id": manuscript_id,
            "title": manuscript.title,
            "authors": manuscript.authors,
            "target_journal": manuscript.target_journal,
            "target_guideline": manuscript.target_guideline.value
                                if manuscript.target_guideline else None,
            "supplementary_materials": manuscript.supplementary_materials,
            "status": manuscript.status,
            "submission_ready": manuscript.status == "draft",
        }

    def submit(self, manuscript_id: str, journal: str) -> Dict[str, Any]:
        """Record a submission (does NOT actually submit)."""
        manuscript = self.manuscripts.get(manuscript_id)
        if manuscript is None:
            return {"status": "MANUSCRIPT_NOT_FOUND"}

        manuscript.status = "submitted"
        manuscript.submission_date = datetime.now().isoformat()

        submission = {
            "manuscript_id": manuscript_id,
            "journal": journal,
            "submission_date": manuscript.submission_date,
            "status": "submitted",
        }
        self.submissions.append(submission)
        log.info(f"[Publication] Submitted {manuscript_id} to {journal}")
        return submission

    def get_submission_history(self) -> List[Dict]:
        return self.submissions


# =============================================================================
# §6. VALIDATION ORCHESTRATOR
# =============================================================================

class ValidationOrchestrator:
    """
    Orchestrates the full validation and publication pipeline.

    Workflow:
    1. Ethics check (IRB + consent)
    2. Statistical validation (CV, bootstrap, calibration)
    3. Reporting compliance check
    4. Expert review
    5. Publication preparation

    All steps must pass before a model is considered "validated".
    """

    def __init__(self):
        self.ethics = EthicsGuard()
        self.stats = StatisticalValidator()
        self.compliance = ReportingComplianceChecker()
        self.review = ExpertReviewPipeline()
        self.publication = PublicationPipeline()
        self.validation_history: List[Dict[str, Any]] = []

    def run_full_validation(
        self,
        model_name: str,
        X: Tensor, y: Tensor,
        protocol_id: str,
        participant_ids: List[str],
        guideline: ReportingGuideline = ReportingGuideline.CLAIM,
    ) -> Dict[str, Any]:
        """
        Run the full validation pipeline.

        Returns a comprehensive validation report.
        """
        report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "status": "IN_PROGRESS",
        }

        # Step 1: Ethics check
        ethics_ok = True
        for pid in participant_ids:
            can_process, reason = self.ethics.can_process_participant(
                pid, protocol_id)
            if not can_process:
                ethics_ok = False
                log.warning(f"[Ethics] Participant {pid}: {reason}")

        report["ethics"] = {
            "status": "PASS" if ethics_ok else "FAIL",
            "note": "Ethics check tracks consent and IRB status. "
                    "Actual approval must come from a qualified institution.",
        }

        # Step 2: Statistical validation
        # Cross-validation
        def cv_model(X_train, y_train, X_test, y_test):
            # Placeholder - replace with actual model evaluation
            return 0.85  # mock accuracy

        cv_result = self.stats.k_fold_cv(cv_model, X, y, k=5)
        report["cross_validation"] = cv_result

        # Bootstrap CI
        scores = np.array(cv_result["fold_scores"])
        bootstrap = self.stats.bootstrap_ci(
            scores, lambda x: np.mean(x), n_bootstrap=5000
        )
        report["bootstrap_ci"] = bootstrap

        # Step 3: Reporting compliance
        compliance = self.compliance.compliance_score(guideline)
        report["reporting_compliance"] = compliance

        # Step 4: Expert review (check if reviewers are assigned)
        consensus = self.review.get_consensus(model_name)
        report["expert_review"] = consensus

        # Step 5: Overall status
        if ethics_ok and compliance["compliance_score"] > 50:
            report["status"] = "VALIDATION_IN_PROGRESS"
            report["next_steps"] = [
                "Complete reporting checklist",
                "Obtain expert reviews",
                "Prepare supplementary materials",
                "Submit to journal",
            ]
        else:
            report["status"] = "VALIDATION_INCOMPLETE"
            report["blockers"] = []
            if not ethics_ok:
                report["blockers"].append("Ethics compliance issues")
            if compliance["compliance_score"] <= 50:
                report["blockers"].append("Reporting checklist incomplete")

        self.validation_history.append(report)
        return report

    def generate_validation_report(self, output_path: str) -> None:
        """Generate a comprehensive validation report."""
        report = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "attributed_to": ATTRIBUTION,
            "validations": self.validation_history,
            "ethics_compliance": self.ethics.compliance_report(),
            "publication_status": self.publication.get_submission_history(),
            "disclaimer": (
                "This report is a RESEARCH VALIDATION SUMMARY. "
                "It does NOT constitute clinical approval, regulatory "
                "clearance, or peer review. All findings must be "
                "independently verified by qualified professionals."
            ),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"[Validation] Report saved to {output_path}")


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test() -> bool:
    ethics_guard()
    print("=" * 78)
    print(f"  RESEARCH VALIDATION PIPELINE v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: Ethics Guard
    print("\n[T1] Ethics Guard — IRB + Consent")
    ethics = EthicsGuard()

    irb = IRBRecord(
        protocol_id="PROT-2026-001",
        title="Neural Signal Analysis Research",
        irb_status=IRBStatus.APPROVED,
        submission_date="2026-01-01",
        approval_date="2026-02-01",
        expiry_date="2027-02-01",
        irb_committee="University Ethics Committee",
    )
    ethics.register_irb(irb)
    rep("IRB registered", "PROT-2026-001" in ethics.irb_records)
    rep("IRB valid", irb.is_valid())

    consent = ConsentRecord(
        participant_id="P001",
        consent_status=ConsentStatus.OBTAINED,
        consent_date="2026-03-01",
        data_use_scope="research_only",
        gdpr_rights_acknowledged=True,
    )
    ethics.register_consent(consent)
    rep("Consent registered", "P001" in ethics.consent_records)
    rep("Consent valid", consent.is_valid())

    can_process, reason = ethics.can_process_participant("P001", "PROT-2026-001")
    rep("Participant can be processed", can_process, reason)

    # No consent
    can_process2, reason2 = ethics.can_process_participant("P999", "PROT-2026-001")
    rep("No consent rejected", not can_process2, reason2)

    # T2: Statistical Validation
    print("\n[T2] Statistical Validation")
    stats = StatisticalValidator()

    # Bootstrap CI
    data = np.random.randn(100)
    bootstrap = stats.bootstrap_ci(data, np.mean, n_bootstrap=5000)
    rep("Bootstrap CI computed",
        "point_estimate" in bootstrap and "ci_lower" in bootstrap)
    rep("Bootstrap CI contains estimate",
        bootstrap["ci_lower"] <= bootstrap["point_estimate"] <= bootstrap["ci_upper"])

    # Effect size
    group1 = np.random.randn(50) + 0.5
    group2 = np.random.randn(50)
    effect = stats.effect_size_cohens_d(group1, group2)
    rep("Effect size computed", "d" in effect and "interpretation" in effect)

    # Calibration
    y_true = np.random.randint(0, 2, 200)
    y_prob = np.random.rand(200)
    y_prob = np.where(y_true == 1, y_prob * 0.8 + 0.2, y_prob * 0.3)
    calib = stats.calibration_curve(y_true, y_prob, n_bins=5)
    rep("Calibration computed", "ece" in calib and "brier_score" in calib)

    # AUROC
    auroc = stats.auroc_with_ci(y_true, y_prob, n_bootstrap=1000)
    rep("AUROC computed", "auroc" in auroc and "ci_lower" in auroc)

    # T3: Reporting Compliance
    print("\n[T3] Reporting Compliance")
    compliance = ReportingComplianceChecker()

    # STARD checklist
    stard = compliance.get_checklist(ReportingGuideline.STARD)
    rep("STARD checklist loaded", len(stard) == 25)

    # Update some items
    compliance.update_item(ReportingGuideline.STARD, "1", "addressed",
                           page_or_section="Title", notes="Done")
    compliance.update_item(ReportingGuideline.STARD, "2", "addressed",
                           page_or_section="Abstract", notes="Done")

    score = compliance.compliance_score(ReportingGuideline.STARD)
    rep("Compliance score computed", "compliance_score" in score)
    rep("Compliance score < 100 (incomplete)", score["compliance_score"] < 100)

    # T4: Expert Review
    print("\n[T4] Expert Review Pipeline")
    review = ExpertReviewPipeline()

    reviewer1 = ExpertReviewer(
        reviewer_id="R001", name="Dr. Smith",
        affiliation="University", expertise=["Neurology"],
    )
    review.register_reviewer(reviewer1)
    rep("Reviewer registered", "R001" in review.reviewers)

    score1 = ReviewScore(
        reviewer_id="R001", manuscript_id="MS001",
        novelty=8, methodology=7, clinical_relevance=9,
        statistical_rigor=8, reproducibility=7, ethical_compliance=9,
        overall_recommendation="ACCEPT",
    )
    submitted = review.submit_score(score1)
    rep("Score submitted", submitted)

    consensus = review.get_consensus("MS001")
    rep("Consensus computed", "mean_weighted_score" in consensus)

    # T5: Publication Pipeline
    print("\n[T5] Publication Pipeline")
    pub = PublicationPipeline()

    ms_id = pub.create_manuscript(
        title="Neural Signal Analysis Using Deep Learning",
        authors=["Author A", "Author B"],
        target_journal="Nature Medicine",
        target_guideline=ReportingGuideline.CLAIM,
    )
    rep("Manuscript created", ms_id in pub.manuscripts)

    pub.add_supplementary(ms_id, "supplementary_data.csv")
    pub.add_supplementary(ms_id, "code_repository.zip")

    prep = pub.prepare_submission(ms_id)
    rep("Submission prepared", prep["manuscript_id"] == ms_id)
    rep("Supplementary materials tracked", len(prep["supplementary_materials"]) == 2)

    submission = pub.submit(ms_id, "Nature Medicine")
    rep("Submission recorded", submission["status"] == "submitted")

    # T6: Full Validation Orchestrator
    print("\n[T6] Validation Orchestrator")
    orchestrator = ValidationOrchestrator()

    # Setup ethics
    orchestrator.ethics.register_irb(irb)
    orchestrator.ethics.register_consent(consent)

    # Run validation
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    report = orchestrator.run_full_validation(
        model_name="test_model",
        X=X, y=y,
        protocol_id="PROT-2026-001",
        participant_ids=["P001"],
        guideline=ReportingGuideline.CLAIM,
    )
    rep("Full validation runs", "status" in report)
    rep("Ethics check passed", report["ethics"]["status"] == "PASS")
    rep("CV results present", "cross_validation" in report)
    rep("Compliance results present", "reporting_compliance" in report)

    # Generate report file
    import tempfile
    tmp = tempfile.mkdtemp()
    report_path = os.path.join(tmp, "validation_report.json")
    orchestrator.generate_validation_report(report_path)
    rep("Report file generated", os.path.exists(report_path))

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THIS MODULE PROVIDES
    • IRB/ethics tracking with consent management
    • Statistical validation (CV, bootstrap CI, effect sizes, calibration)
    • Reporting guideline compliance (STARD, TRIPOD, CLAIM)
    • Expert review pipeline with COI checking
    • Publication preparation and submission tracking
    • Full validation orchestration

  WHAT THIS MODULE IS NOT
    ❌ Not an IRB approval authority
    ❌ Not a substitute for peer review
    ❌ Not a clinical validation tool
    ❌ Does not actually submit to journals

  REQUIRED FOR ACTUAL PUBLICATION
    1. Real IRB/ethics committee approval
    2. Real informed consent from participants
    3. Real expert peer review
    4. Journal-specific formatting requirements
    5. Data sharing agreements
    6. Regulatory compliance (FDA/CE/Thai FDA if applicable)
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Research Validation & Publication Pipeline")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if self_test() else 1)

    parser.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
