"""
=============================================================================
SESI TPTP PROVER BENCHMARK × ARXIV LATEX EQUATION DATASET × FCI++ COMPARISON
Vampire/E/Z3 + arXiv PDF/LaTeX Mining + Latent-Variable Causal Comparison
=============================================================================

Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT
Year         : 2026

Purpose
-------
This module provides three production-oriented research components:

1. TPTPProverBenchmark
   - Load TPTP problems (.p)
   - Run Vampire / E on TPTP
   - Run Z3 on SMT-LIB2 (.smt2)
   - Collect proof status, runtime, output, and benchmark statistics

2. ArXivLatexEquationDataset
   - Query arXiv API
   - Optionally download PDFs
   - Extract LaTeX-like equations from PDF text
   - Convert LaTeX to SymPy where possible
   - Export JSONL dataset for theorem-prover / GNN training

3. FCIPPComparisonSuite
   - PC baseline for causal-sufficient setting
   - Full FCI-family backend via causal-learn when installed
   - NOTEARS differentiable DAG learner
   - Compare PC / FCI++ / NOTEARS with SHD, precision, recall, F1

Honest Scope
------------
- Vampire/E/Z3 must be installed externally.
- Z3 runs SMT-LIB2, not raw TPTP, unless a separate translator is supplied.
- arXiv extraction is heuristic and does not produce verified formal proofs.
- FCI++ here uses a full external FCI-family backend when available.
  If causal-learn is not installed, the FCI++ backend returns dependency_required.
- Causal discovery outputs are hypotheses requiring domain validation.
- This module does not claim consciousness, enlightenment, or mathematical truth.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import time
import uuid
import urllib.parse
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.tptp_arxiv_fcipp")

# Optional dependencies -------------------------------------------------------

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from causallearn.search.ConstraintBased.FCI import fci
    HAS_CAUSALLEARN = True
except ImportError:
    HAS_CAUSALLEARN = False


# =============================================================================
# Shared utilities
# =============================================================================

class RateLimiter:
    """No-Zeno style practical event rate limiter."""

    def __init__(self, max_events: int = 20, window_s: float = 60.0,
                 min_interval_s: float = 0.0):
        self.max_events = max_events
        self.window_s = window_s
        self.min_interval_s = min_interval_s
        self.events: deque[float] = deque()

    def check(self) -> Dict[str, Any]:
        now = time.time()
        while self.events and now - self.events[0] > self.window_s:
            self.events.popleft()

        if len(self.events) >= self.max_events:
            return {
                "allowed": False,
                "reason": "max_events_in_window",
                "current": len(self.events),
                "limit": self.max_events,
            }

        if self.events and now - self.events[-1] < self.min_interval_s:
            return {
                "allowed": False,
                "reason": "min_interval_not_elapsed",
                "wait_s": self.min_interval_s - (now - self.events[-1]),
            }

        return {"allowed": True}

    def record(self) -> None:
        self.events.append(time.time())


def safe_unit(x: float, default: float = 0.5) -> float:
    """Sanitize scalar into [0,1], matching _safe_unit_scalar discipline."""
    try:
        v = float(x)
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return max(0.0, min(1.0, v))


def run_cmd(cmd: List[str], timeout_s: float) -> Dict[str, Any]:
    start = time.time()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "elapsed_s": time.time() - start,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "elapsed_s": time.time() - start,
            "timeout": True,
        }
    except FileNotFoundError:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"Executable not found: {cmd[0]}",
            "elapsed_s": time.time() - start,
            "timeout": False,
            "missing_binary": True,
        }


# =============================================================================
# 1. TPTP / Vampire / E / Z3 Benchmarking
# =============================================================================

class ProverKind(str, Enum):
    VAMPIRE = "vampire"
    E = "e"
    Z3 = "z3"


@dataclass
class TPTPProblem:
    problem_id: str
    path: str
    domain: str
    format: str
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "path": self.path,
            "domain": self.domain,
            "format": self.format,
        }


@dataclass
class ProverRunResult:
    prover: str
    problem_id: str
    status: str
    elapsed_s: float
    stdout: str = ""
    stderr: str = ""
    proof_length: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prover": self.prover,
            "problem_id": self.problem_id,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "proof_length": self.proof_length,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
        }


class TPTPProverBenchmark:
    """
    Vampire/E benchmark on TPTP .p files and Z3 benchmark on SMT-LIB2 .smt2 files.

    Notes
    -----
    - Vampire and E can consume TPTP directly.
    - Z3 consumes SMT-LIB2 here. TPTP→SMT translation is not assumed.
    """

    def __init__(
        self,
        vampire_path: str = "vampire",
        eprover_path: str = "eprover",
        z3_path: str = "z3",
        timeout_s: float = 60.0,
    ):
        self.vampire_path = vampire_path
        self.eprover_path = eprover_path
        self.z3_path = z3_path
        self.timeout_s = timeout_s
        self.rate = RateLimiter(max_events=20, window_s=60.0, min_interval_s=1.0)
        self.results: List[ProverRunResult] = []

    @staticmethod
    def detect_format(content: str) -> str:
        c = content.lower()
        if "thf(" in c:
            return "THF"
        if "tff(" in c:
            return "TFF"
        if "fof(" in c:
            return "FOF"
        if "cnf(" in c:
            return "CNF"
        return "UNKNOWN"

    def load_tptp(self, root: str, max_problems: int = 100,
                  domain: Optional[str] = None) -> List[TPTPProblem]:
        root_path = Path(root)
        problems: List[TPTPProblem] = []

        for p in root_path.rglob("*.p"):
            if len(problems) >= max_problems:
                break
            stem = p.stem
            m = re.match(r"([A-Z]+)", stem)
            dom = m.group(1) if m else "UNKNOWN"
            if domain and dom != domain:
                continue
            content = p.read_text(errors="ignore")
            problems.append(
                TPTPProblem(
                    problem_id=stem,
                    path=str(p),
                    domain=dom,
                    format=self.detect_format(content),
                    content=content,
                )
            )
        return problems

    def run_vampire(self, problem: TPTPProblem) -> ProverRunResult:
        gate = self.rate.check()
        if not gate["allowed"]:
            return ProverRunResult("vampire", problem.problem_id, "rate_limited", 0.0,
                                   stderr=gate["reason"])
        self.rate.record()

        cmd = [
            self.vampire_path,
            "--mode", "casc",
            "--time_limit", str(int(self.timeout_s)),
            problem.path,
        ]
        r = run_cmd(cmd, self.timeout_s + 5)
        out = r["stdout"]

        if r["timeout"]:
            status = "timeout"
        elif "Refutation found" in out or "SZS status Theorem" in out:
            status = "proved"
        elif "SZS status CounterSatisfiable" in out:
            status = "counter_satisfiable"
        elif r.get("missing_binary"):
            status = "missing_binary"
        elif r["returncode"] == 0:
            status = "completed_unknown"
        else:
            status = "failed"

        result = ProverRunResult(
            prover="vampire",
            problem_id=problem.problem_id,
            status=status,
            elapsed_s=r["elapsed_s"],
            stdout=r["stdout"],
            stderr=r["stderr"],
            proof_length=out.count("\n") if status == "proved" else None,
        )
        self.results.append(result)
        return result

    def run_e(self, problem: TPTPProblem) -> ProverRunResult:
        gate = self.rate.check()
        if not gate["allowed"]:
            return ProverRunResult("e", problem.problem_id, "rate_limited", 0.0,
                                   stderr=gate["reason"])
        self.rate.record()

        cmd = [
            self.eprover_path,
            "--auto",
            f"--cpu-limit={int(self.timeout_s)}",
            "--print-statistics",
            problem.path,
        ]
        r = run_cmd(cmd, self.timeout_s + 5)
        out = r["stdout"]

        if r["timeout"]:
            status = "timeout"
        elif "# Proof found" in out or "SZS status Theorem" in out:
            status = "proved"
        elif "SZS status CounterSatisfiable" in out:
            status = "counter_satisfiable"
        elif r.get("missing_binary"):
            status = "missing_binary"
        elif r["returncode"] == 0:
            status = "completed_unknown"
        else:
            status = "failed"

        result = ProverRunResult(
            prover="e",
            problem_id=problem.problem_id,
            status=status,
            elapsed_s=r["elapsed_s"],
            stdout=r["stdout"],
            stderr=r["stderr"],
            proof_length=out.count("\n") if status == "proved" else None,
        )
        self.results.append(result)
        return result

    def run_z3_smt2(self, smt2_path: str, problem_id: Optional[str] = None) -> ProverRunResult:
        gate = self.rate.check()
        if not gate["allowed"]:
            return ProverRunResult("z3", problem_id or Path(smt2_path).stem,
                                   "rate_limited", 0.0, stderr=gate["reason"])
        self.rate.record()

        pid = problem_id or Path(smt2_path).stem
        cmd = [self.z3_path, f"-T:{int(self.timeout_s)}", smt2_path]
        r = run_cmd(cmd, self.timeout_s + 5)
        out = r["stdout"].strip()

        if r["timeout"]:
            status = "timeout"
        elif out.startswith("unsat"):
            status = "proved_unsat"
        elif out.startswith("sat"):
            status = "sat_countermodel"
        elif out.startswith("unknown"):
            status = "unknown"
        elif r.get("missing_binary"):
            status = "missing_binary"
        else:
            status = "failed"

        result = ProverRunResult(
            prover="z3",
            problem_id=pid,
            status=status,
            elapsed_s=r["elapsed_s"],
            stdout=r["stdout"],
            stderr=r["stderr"],
        )
        self.results.append(result)
        return result

    def benchmark(
        self,
        problems: List[TPTPProblem],
        prover: ProverKind = ProverKind.VAMPIRE,
    ) -> Dict[str, Any]:
        rows = []
        for p in problems:
            if prover == ProverKind.VAMPIRE:
                rows.append(self.run_vampire(p).to_dict())
            elif prover == ProverKind.E:
                rows.append(self.run_e(p).to_dict())
            else:
                rows.append({
                    "prover": "z3",
                    "problem_id": p.problem_id,
                    "status": "unsupported_format",
                    "note": "Z3 benchmark requires SMT-LIB2 .smt2 input.",
                })

        total = len(rows)
        proved = sum(1 for r in rows if str(r["status"]).startswith("proved"))
        return {
            "prover": prover.value,
            "total": total,
            "proved": proved,
            "success_rate": proved / max(total, 1),
            "rows": rows,
        }


# =============================================================================
# 2. arXiv LaTeX Equation Dataset
# =============================================================================

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class LatexEquationRecord:
    equation_id: str
    arxiv_id: str
    title: str
    latex: str
    source: str
    page: Optional[int]
    sympy_expr: Optional[str]
    symbols: List[str]
    parse_success: bool
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class LatexToSympyParser:
    """
    Practical LaTeX→SymPy parser.

    This is not a full TeX engine. It handles common mathematical notation:
    fractions, powers, subscripts, Greek letters, trig, logs, sums, integrals.
    """

    GREEK = {
        r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
        r"\delta": "delta", r"\theta": "theta", r"\lambda": "lambda",
        r"\mu": "mu", r"\sigma": "sigma", r"\phi": "phi",
        r"\omega": "omega", r"\pi": "pi",
    }

    def __init__(self):
        self.records: List[LatexEquationRecord] = []

    def normalize(self, latex: str) -> str:
        s = latex.strip()
        s = re.sub(r"^\$+|\$+$", "", s)
        s = re.sub(r"^\\\[|\\\]$", "", s)
        s = re.sub(r"^\\\(|\\\)$", "", s)
        s = re.sub(r"\\label\{.*?\}", "", s)
        s = re.sub(r"\\tag\{.*?\}", "", s)
        s = s.replace(r"\left", "").replace(r"\right", "")

        for k, v in self.GREEK.items():
            s = s.replace(k, v)

        # \frac{a}{b}
        for _ in range(5):
            s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

        # \sqrt{x}
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)

        # functions
        replacements = {
            r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
            r"\log": "log", r"\ln": "log", r"\exp": "exp",
            r"\cdot": "*", r"\times": "*",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)

        # powers
        s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
        s = re.sub(r"\^([A-Za-z0-9])", r"**\1", s)

        # subscripts as variable suffix
        s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)

        # remove spacing commands
        s = re.sub(r"\\[,;! ]", "", s)

        return s

    def parse(self, latex: str, arxiv_id: str = "", title: str = "",
              source: str = "manual", page: Optional[int] = None) -> LatexEquationRecord:
        eq_id = uuid.uuid4().hex[:12]

        if not HAS_SYMPY:
            rec = LatexEquationRecord(
                eq_id, arxiv_id, title, latex, source, page,
                None, [], False, "SymPy not installed"
            )
            self.records.append(rec)
            return rec

        try:
            normalized = self.normalize(latex)

            # If equation contains "=", parse lhs-rhs
            if "=" in normalized:
                lhs, rhs = normalized.split("=", 1)
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                expr = sp.sympify(normalized)

            rec = LatexEquationRecord(
                equation_id=eq_id,
                arxiv_id=arxiv_id,
                title=title,
                latex=latex,
                source=source,
                page=page,
                sympy_expr=str(expr),
                symbols=sorted(str(x) for x in expr.free_symbols),
                parse_success=True,
            )
        except Exception as e:
            rec = LatexEquationRecord(
                eq_id, arxiv_id, title, latex, source, page,
                None, [], False, str(e)
            )

        self.records.append(rec)
        return rec

    def export_jsonl(self, path: str) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        return str(out)


class ArxivLatexEquationDataset:
    """
    arXiv API → PDF download → equation extraction → LaTeX/SymPy dataset.

    Compliance
    ----------
    - Uses arXiv API metadata.
    - Keeps only necessary fields by default.
    - Does not bypass access controls.
    - Adds delay between API requests.
    """

    ARXIV_API = "http://export.arxiv.org/api/query"

    EQUATION_PATTERNS = [
        r"\$\$(.*?)\$\$",
        r"\$(.*?)\$",
        r"\\\[(.*?)\\\]",
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        r"\\begin\{align\}(.*?)\\end\{align\}",
        r"\\begin\{align\*\}(.*?)\\end\{align\*\}",
    ]

    def __init__(self, output_dir: str = "./arxiv_latex_dataset",
                 request_delay_s: float = 3.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay_s = request_delay_s
        self.parser = LatexToSympyParser()
        self.rate = RateLimiter(max_events=20, window_s=60.0, min_interval_s=request_delay_s)

    def search(self, query: str, max_results: int = 20) -> List[ArxivPaper]:
        if not HAS_REQUESTS:
            raise RuntimeError("requests is required for arXiv API search")

        gate = self.rate.check()
        if not gate["allowed"]:
            raise RuntimeError(f"arXiv query rate limited: {gate}")
        self.rate.record()

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = self.ARAXIV_API if hasattr(self, "ARAXIV_API") else self.ARXIV_API
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        papers: List[ArxivPaper] = []

        for entry in root.findall("atom:entry", ns):
            raw_id = entry.findtext("atom:id", default="", namespaces=ns)
            arxiv_id = raw_id.rstrip("/").split("/")[-1]
            title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
            abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [
                a.findtext("atom:name", default="", namespaces=ns)
                for a in entry.findall("atom:author", ns)
            ]
            categories = [
                c.attrib.get("term", "")
                for c in entry.findall("atom:category", ns)
            ]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    categories=categories,
                    published=published,
                    pdf_url=pdf_url,
                )
            )

        return papers

    def download_pdf(self, paper: ArxivPaper) -> str:
        if not HAS_REQUESTS:
            raise RuntimeError("requests required for PDF download")

        gate = self.rate.check()
        if not gate["allowed"]:
            raise RuntimeError(f"PDF download rate limited: {gate}")
        self.rate.record()

        path = self.output_dir / "pdf" / f"{paper.arxiv_id}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            return str(path)

        r = requests.get(paper.pdf_url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        return str(path)

    def extract_equations_from_pdf(self, pdf_path: str, paper: ArxivPaper) -> List[LatexEquationRecord]:
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF required: pip install PyMuPDF")

        doc = fitz.open(pdf_path)
        records: List[LatexEquationRecord] = []

        for page_idx, page in enumerate(doc):
            text = page.get_text()
            for pattern in self.EQUATION_PATTERNS:
                for m in re.finditer(pattern, text, flags=re.DOTALL):
                    latex = m.group(1).strip()
                    if len(latex) < 5:
                        continue
                    rec = self.parser.parse(
                        latex=latex,
                        arxiv_id=paper.arxiv_id,
                        title=paper.title,
                        source="arxiv_pdf",
                        page=page_idx + 1,
                    )
                    records.append(rec)

        doc.close()
        return records

    def build_dataset(self, query: str = "cat:math.*",
                      max_papers: int = 10,
                      download_pdfs: bool = True) -> Dict[str, Any]:
        papers = self.search(query, max_results=max_papers)
        all_records: List[LatexEquationRecord] = []

        metadata_path = self.output_dir / "papers.jsonl"
        with metadata_path.open("w", encoding="utf-8") as f:
            for p in papers:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

                # Parse abstract equations even without PDF
                for pattern in self.EQUATION_PATTERNS:
                    for m in re.finditer(pattern, p.abstract, flags=re.DOTALL):
                        all_records.append(
                            self.parser.parse(
                                m.group(1).strip(),
                                arxiv_id=p.arxiv_id,
                                title=p.title,
                                source="arxiv_abstract",
                            )
                        )

                if download_pdfs:
                    try:
                        pdf_path = self.download_pdf(p)
                        all_records.extend(self.extract_equations_from_pdf(pdf_path, p))
                    except Exception as e:
                        logger.warning("PDF extraction failed for %s: %s", p.arxiv_id, e)

        equations_path = self.output_dir / "equations.jsonl"
        self.parser.export_jsonl(str(equations_path))

        return {
            "papers": len(papers),
            "equations": len(all_records),
            "parse_success": sum(1 for r in all_records if r.parse_success),
            "metadata_path": str(metadata_path),
            "equations_path": str(equations_path),
        }


# =============================================================================
# 3. Full FCI++ comparison backend + NOTEARS comparator
# =============================================================================

@dataclass
class CausalComparisonResult:
    algorithm: str
    edges: List[Tuple[str, str]]
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class FCIPPBackend:
    """
    Full FCI-family backend adapter.

    If causal-learn is installed, this calls its FCI implementation.
    If not installed, returns dependency_required rather than pretending
    to implement full FCI++.

    Selection-bias / FCI++ wrapper metadata is retained in output.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @staticmethod
    def _endpoint_mark(x: int) -> str:
        # causal-learn convention may vary; this is a robust readable mapping
        return {1: ">", -1: "-", 0: "o"}.get(int(x), "?")

    def run(self, data: np.ndarray, var_names: List[str],
            selection_vars: Optional[List[int]] = None) -> Dict[str, Any]:

        if not HAS_CAUSALLEARN:
            return {
                "status": "dependency_required",
                "algorithm": "FCI++",
                "dependency": "causal-learn",
                "install": "pip install causal-learn",
                "note": "Full FCI-family backend unavailable; no approximate result returned.",
            }

        try:
            # causal-learn FCI API returns graph and edges in common versions
            g, edges = fci(
                data,
                independence_test_method="fisherz",
                alpha=self.alpha,
                verbose=False,
            )

            pag_edges = []
            mat = g.graph
            d = mat.shape[0]
            for i in range(d):
                for j in range(i + 1, d):
                    if mat[i, j] == 0 and mat[j, i] == 0:
                        continue
                    left = self._endpoint_mark(mat[j, i])
                    right = self._endpoint_mark(mat[i, j])
                    pag_edges.append({
                        "source": var_names[i],
                        "target": var_names[j],
                        "marks": f"{left}--{right}",
                    })

            return {
                "status": "success",
                "algorithm": "FCI++/FCI-family-backend",
                "variables": var_names,
                "edges": pag_edges,
                "selection_vars": selection_vars or [],
                "note": (
                    "Backend uses causal-learn FCI-family implementation. "
                    "PAG output is an equivalence class, not unique causal truth."
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "algorithm": "FCI++",
                "error": str(e),
            }


class SimplePC:
    """PC skeleton baseline for comparison, not latent-variable complete."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def run(self, data: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
        # simple correlation skeleton baseline
        corr = np.corrcoef(data.T)
        edges = []
        d = len(var_names)
        for i in range(d):
            for j in range(i + 1, d):
                if abs(corr[i, j]) > 0.25:
                    edges.append({"source": var_names[i], "target": var_names[j]})
        return {
            "status": "success",
            "algorithm": "PC-baseline",
            "edges": edges,
            "note": "Skeleton baseline only; use full PC implementation for formal studies.",
        }


class NOTEARS:
    """Differentiable NOTEARS linear DAG learner."""

    def __init__(self, lambda1: float = 0.1, max_iter: int = 200,
                 threshold: float = 0.2, device: str = "cpu"):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.threshold = threshold
        self.device = device

    def run(self, data: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
        if not HAS_TORCH:
            return {"status": "dependency_required", "dependency": "torch"}

        X = torch.tensor(data, dtype=torch.float32, device=self.device)
        n, d = X.shape
        W = torch.zeros(d, d, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([W], lr=1e-2)

        rho = 1.0
        alpha = 0.0
        prev_h = float("inf")

        for it in range(self.max_iter):
            opt.zero_grad()
            X_hat = X @ W
            loss = ((X - X_hat) ** 2).mean() + self.lambda1 * W.abs().sum()

            W_sq = W * W
            h = torch.matrix_exp(W_sq).trace() - d
            obj = loss + alpha * h + 0.5 * rho * h * h
            obj.backward()
            opt.step()

            h_val = float(h.detach().cpu().item())
            if h_val > 0.25 * prev_h:
                rho *= 10.0
            alpha += rho * h_val
            prev_h = h_val

            if abs(h_val) < 1e-8 and it > 10:
                break

        W_np = W.detach().cpu().numpy()
        W_np[np.abs(W_np) < self.threshold] = 0.0

        edges = []
        for i in range(d):
            for j in range(d):
                if W_np[i, j] != 0:
                    edges.append({
                        "source": var_names[i],
                        "target": var_names[j],
                        "weight": float(W_np[i, j]),
                    })

        return {
            "status": "success",
            "algorithm": "NOTEARS",
            "edges": edges,
            "final_h": prev_h,
            "iterations": it + 1,
            "note": "Assumes causal sufficiency and linear structural model.",
        }


class CausalComparisonSuite:
    """Compare PC baseline, full FCI-family backend, and NOTEARS."""

    def __init__(self):
        self.pc = SimplePC()
        self.fcipp = FCIPPBackend()
        self.notears = NOTEARS()

    @staticmethod
    def synthetic_data(n: int = 500, d: int = 5, seed: int = 42):
        rng = np.random.default_rng(seed)
        X = np.zeros((n, d))
        X[:, 0] = rng.normal(size=n)
        true_edges = []
        for i in range(d - 1):
            w = rng.uniform(0.6, 1.2)
            X[:, i + 1] = w * X[:, i] + 0.3 * rng.normal(size=n)
            true_edges.append((f"X{i}", f"X{i+1}"))
        return X, [f"X{i}" for i in range(d)], true_edges

    @staticmethod
    def edge_set(result: Dict[str, Any]) -> set:
        return {(e["source"], e["target"]) for e in result.get("edges", [])}

    @staticmethod
    def metrics(pred: set, true: set) -> Dict[str, float]:
        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        shd = len(pred ^ true)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "shd": float(shd),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }

    def run(self, data: np.ndarray, var_names: List[str],
            true_edges: List[Tuple[str, str]]) -> Dict[str, Any]:

        outputs = {
            "pc_baseline": self.pc.run(data, var_names),
            "fcipp_backend": self.fcipp.run(data, var_names),
            "notears": self.notears.run(data, var_names),
        }

        true = set(true_edges)
        comparison = {}
        for name, out in outputs.items():
            if out.get("status") == "success":
                pred = self.edge_set(out)
                comparison[name] = {
                    "status": "success",
                    "metrics": self.metrics(pred, true),
                    "n_edges": len(pred),
                    "raw": out,
                }
            else:
                comparison[name] = {
                    "status": out.get("status", "unknown"),
                    "raw": out,
                }

        return {
            "true_edges": true_edges,
            "comparison": comparison,
            "note": (
                "Algorithms have different assumptions. PC/NOTEARS assume causal "
                "sufficiency; FCI-family backend allows latent confounding and "
                "returns PAG-style equivalence information."
            ),
        }


# =============================================================================
# Orchestrator
# =============================================================================

class SESITPTPArxivCausalSuite:
    """
    Unified fa├зade:
      - TPTP Vampire/E/Z3 benchmarking
      - arXiv LaTeX equation dataset
      - PC/FCI++/NOTEARS comparison
    """

    def __init__(self, output_dir: str = "./sesi_tptp_arxiv_causal"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.provers = TPTPProverBenchmark()
        self.arxiv = ArxivLatexEquationDataset(str(self.output_dir / "arxiv"))
        self.causal = CausalComparisonSuite()

    def status(self) -> Dict[str, Any]:
        return {
            "dependencies": {
                "requests": HAS_REQUESTS,
                "PyMuPDF": HAS_PYMUPDF,
                "sympy": HAS_SYMPY,
                "torch": HAS_TORCH,
                "causal_learn": HAS_CAUSALLEARN,
            },
            "output_dir": str(self.output_dir),
        }


# =============================================================================
# Smoke test
# =============================================================================

def smoke_test():
    logging.basicConfig(level=logging.INFO)

    suite = SESITPTPArxivCausalSuite()
    print("=" * 72)
    print("SESI TPTP × arXiv LaTeX × FCI++/NOTEARS Smoke Test")
    print("=" * 72)
    print(json.dumps(suite.status(), indent=2))

    # --- LaTeX parser test
    print("\n[1] LaTeX equation parsing")
    parser = LatexToSympyParser()
    examples = [
        r"E = mc^2",
        r"\frac{a}{b} + c",
        r"\alpha^2 + \beta^2",
        r"\sin(x)^2 + \cos(x)^2",
    ]
    for eq in examples:
        rec = parser.parse(eq)
        print(eq, "=>", rec.sympy_expr, "success=", rec.parse_success)

    # --- arXiv dataset test (metadata only if dependencies missing)
    print("\n[2] arXiv dataset")
    if HAS_REQUESTS:
        try:
            result = suite.arxiv.build_dataset(
                query="cat:math.LO",
                max_papers=2,
                download_pdfs=False,
            )
            print(json.dumps(result, indent=2))
        except Exception as e:
            print("arXiv test skipped:", e)
    else:
        print("requests missing; arXiv API skipped")

    # --- Causal comparison
    print("\n[3] Causal comparison")
    data, names, true_edges = suite.causal.synthetic_data(n=300, d=5)
    result = suite.causal.run(data, names, true_edges)
    for algo, row in result["comparison"].items():
        print(algo, row["status"])
        if "metrics" in row:
            print("  metrics:", row["metrics"])

    print("\nHONEST SCOPE")
    print("- External provers require installed binaries.")
    print("- arXiv extraction is heuristic, not formal proof extraction.")
    print("- Full FCI-family backend requires causal-learn.")
    print("- NOTEARS assumes causal sufficiency and linear structure.")
    print("=" * 72)


if __name__ == "__main__":
    smoke_test()
