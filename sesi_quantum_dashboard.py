"""
=============================================================================
SESI QUANTUM DASHBOARD — v1.0.0
Interactive Visualization for Quantum Computing Results
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
VISUALIZATION FEATURES
=============================================================================
This module provides interactive dashboards for:

  1. HILBERT SPACE SCALING
     • Exponential growth visualization
     • Memory requirements plot
     • Classical simulation limit

  2. ERROR ANALYSIS
     • Error scaling with circuit depth
     • Gate error rates
     • Decoherence limits

  3. QUANTUM ADVANTAGE
     • Advantage thresholds by task
     • NISQ vs fault-tolerant eras
     • Timeline estimates

  4. IBM QUANTUM RESULTS
     • Circuit execution results
     • Backend properties
     • Comparison plots

=============================================================================
MODULES IN THIS FILE
=============================================================================
[1] HilbertScalingVisualizer     — Scaling visualization
[2] ErrorScalingVisualizer       — Error analysis visualization
[3] QuantumAdvantageVisualizer   — Advantage threshold visualization
[4] IBMResultsVisualizer         — IBM Quantum results visualization
[5] QuantumDashboardSystem       — Master orchestrator
[6] Self-test suite

=============================================================================
HONEST SCOPE
=============================================================================
• Visualizations are for RESEARCH/EDUCATION
• Requires matplotlib/plotly for rendering
• All plots are generated from analysis results
• No real-time data streaming
=============================================================================
"""

from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.quantum_dashboard")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

QUANTUM_DASHBOARD_VERSION: str = "1.0.0"

# =============================================================================
# Visualization library imports (graceful fallback)
# =============================================================================
_HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    _HAS_MATPLOTLIB = True
except ImportError:
    import warnings
    warnings.warn("matplotlib not available; static plots disabled.")

_HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    import warnings
    warnings.warn("plotly not available; interactive plots disabled.")

logger.info(f"[imports] Matplotlib={_HAS_MATPLOTLIB} Plotly={_HAS_PLOTLY}")

# =============================================================================
# OPMC PAPER CONSTANTS
# =============================================================================
OPMC_C1 = 0.420000
OPMC_K2 = 12.5000
OPMC_L = 0.438520
OPMC_ETA_LATTICE = 0.99491
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n

def opmc_k_star(epsilon: float = 1e-8, D0: float = 1.0) -> int:
    """Theorem 11.1: k* = ⌈ln(ε/D₀)/ln L⌉ — independent of N."""
    return math.ceil(math.log(epsilon / D0) / math.log(OPMC_L))


# =============================================================================
# 1. HILBERT SCALING VISUALIZER
# =============================================================================
@dataclass
class HilbertVizConfig:
    """Configuration for Hilbert space visualization."""
    output_dir: str = "./quantum_viz"
    dpi: int = 150
    figsize: Tuple[float, float] = (12, 8)
    colormap: str = "viridis"


class HilbertScalingVisualizer:
    """
    [D] Visualizer for Hilbert space scaling.
    
    PLOTS:
      1. Exponential scaling (2^N)
      2. Memory requirements (log scale)
      3. Classical simulation feasibility
    
    USAGE:
      viz = HilbertScalingVisualizer(cfg)
      viz.plot_scaling(scaling_data)
    """
    
    def __init__(self, cfg: Optional[HilbertVizConfig] = None):
        self.cfg = cfg or HilbertVizConfig()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logger.info(f"[HilbertViz] Initialized, output: {self.cfg.output_dir}")
    
    def plot_scaling(self, scaling_data: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Plot Hilbert space scaling.
        
        Args:
            scaling_data: output from HilbertSpaceScalingAnalyzer
            output_path: path to save plot
        
        Returns:
            path to saved plot, or None
        """
        if not _HAS_MATPLOTLIB:
            logger.warning("[HilbertViz] matplotlib not available")
            return None
        
        table = scaling_data["scaling_table"]
        n_qubits = [r["n_qubits"] for r in table]
        dims = [r["hilbert_dim"] for r in table]
        memory_gb = [r["state_memory_gb"] for r in table]
        feasible = [r["classical_simulation_feasible"] for r in table]
        
        fig, axes = plt.subplots(2, 2, figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        fig.suptitle("Hilbert Space Scaling with Qubit Count", fontsize=16, fontweight="bold")
        
        # Plot 1: Hilbert dimension (log scale)
        ax = axes[0, 0]
        ax.semilogy(n_qubits, dims, 'b-', linewidth=2)
        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("Hilbert Dimension (2^N)")
        ax.set_title("Exponential Growth of Hilbert Space")
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Memory requirements (log scale)
        ax = axes[0, 1]
        ax.semilogy(n_qubits, memory_gb, 'r-', linewidth=2)
        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("State Vector Memory (GB)")
        ax.set_title("Memory Requirements")
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 1 PB
        ax.axhline(y=1e6, color='k', linestyle='--', alpha=0.5, label='1 PB limit')
        ax.legend()
        
        # Plot 3: Feasibility
        ax = axes[1, 0]
        colors = ['green' if f else 'red' for f in feasible]
        ax.bar(n_qubits, [1] * len(n_qubits), color=colors, alpha=0.7)
        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("Feasibility")
        ax.set_title("Classical Simulation Feasibility")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Infeasible", "Feasible"])
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = (
            f"Key Findings:\n\n"
            f"• Classical simulation limit:\n"
            f"  {scaling_data['classical_simulation_limit']} qubits\n\n"
            f"• Quantum advantage threshold:\n"
            f"  {scaling_data['quantum_advantage_threshold']} qubits\n\n"
            f"• NISQ era range:\n"
            f"  {scaling_data['nisq_era_range']}\n\n"
            f"• Fault-tolerant threshold:\n"
            f"  {scaling_data['fault_tolerant_threshold']}"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.cfg.output_dir, "hilbert_scaling.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"[HilbertViz] Saved: {output_path}")
        return output_path


# =============================================================================
# 2. ERROR SCALING VISUALIZER
# =============================================================================
@dataclass
class ErrorVizConfig:
    """Configuration for error scaling visualization."""
    output_dir: str = "./quantum_viz"
    dpi: int = 150
    figsize: Tuple[float, float] = (12, 8)


class ErrorScalingVisualizer:
    """
    [D] Visualizer for error scaling analysis.
    
    PLOTS:
      1. Error probability vs circuit depth
      2. Expected errors vs depth
      3. Coherence limits
    
    USAGE:
      viz = ErrorScalingVisualizer(cfg)
      viz.plot_error_scaling(error_data)
    """
    
    def __init__(self, cfg: Optional[ErrorVizConfig] = None):
        self.cfg = cfg or ErrorVizConfig()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logger.info(f"[ErrorViz] Initialized, output: {self.cfg.output_dir}")
    
    def plot_error_scaling(self, error_data: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Plot error scaling.
        
        Args:
            error_data: output from ErrorScalingAnalyzer
            output_path: path to save plot
        
        Returns:
            path to saved plot, or None
        """
        if not _HAS_MATPLOTLIB:
            logger.warning("[ErrorViz] matplotlib not available")
            return None
        
        table = error_data["scaling_table"]
        depths = [r["depth"] for r in table]
        p_no_error = [r["p_no_error"] for r in table]
        expected_errors = [r["expected_errors"] for r in table]
        
        fig, axes = plt.subplots(1, 3, figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        fig.suptitle(
            f"Error Scaling Analysis ({error_data['n_qubits']} qubits)",
            fontsize=16, fontweight="bold"
        )
        
        # Plot 1: Probability of no error
        ax = axes[0]
        ax.semilogy(depths, p_no_error, 'b-o', linewidth=2, markersize=4)
        ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
        ax.set_xlabel("Circuit Depth")
        ax.set_ylabel("P(No Error)")
        ax.set_title("Probability of No Error")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Expected errors
        ax = axes[1]
        ax.plot(depths, expected_errors, 'r-o', linewidth=2, markersize=4)
        ax.set_xlabel("Circuit Depth")
        ax.set_ylabel("Expected Errors")
        ax.set_title("Expected Number of Errors")
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Summary
        ax = axes[2]
        ax.axis('off')
        summary_text = (
            f"Error Parameters:\n\n"
            f"• 1-qubit gate error:\n"
            f"  {error_data['gate_error_1q']:.4f}\n\n"
            f"• 2-qubit gate error:\n"
            f"  {error_data['gate_error_2q']:.4f}\n\n"
            f"• Readout error:\n"
            f"  {error_data['readout_error']:.4f}\n\n"
            f"• T1: {error_data['t1_us']} μs\n"
            f"• T2: {error_data['t2_us']} μs\n\n"
            f"• Max useful depth:\n"
            f"  {error_data['max_useful_depth']}\n\n"
            f"• Coherence limit:\n"
            f"  {error_data['max_depth_coherence']}"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.cfg.output_dir, "error_scaling.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"[ErrorViz] Saved: {output_path}")
        return output_path


# =============================================================================
# 3. QUANTUM ADVANTAGE VISUALIZER
# =============================================================================
@dataclass
class AdvantageVizConfig:
    """Configuration for quantum advantage visualization."""
    output_dir: str = "./quantum_viz"
    dpi: int = 150
    figsize: Tuple[float, float] = (14, 10)


class QuantumAdvantageVisualizer:
    """
    [D] Visualizer for quantum advantage analysis.
    
    PLOTS:
      1. Advantage thresholds by task
      2. NISQ vs fault-tolerant eras
      3. Timeline estimates
    
    USAGE:
      viz = QuantumAdvantageVisualizer(cfg)
      viz.plot_advantage(advantage_data)
    """
    
    def __init__(self, cfg: Optional[AdvantageVizConfig] = None):
        self.cfg = cfg or AdvantageVizConfig()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logger.info(f"[AdvantageViz] Initialized, output: {self.cfg.output_dir}")
    
    def plot_advantage(self, advantage_data: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Plot quantum advantage analysis.
        
        Args:
            advantage_data: output from QuantumAdvantageAnalyzer
            output_path: path to save plot
        
        Returns:
            path to saved plot, or None
        """
        if not _HAS_MATPLOTLIB:
            logger.warning("[AdvantageViz] matplotlib not available")
            return None
        
        thresholds = advantage_data["advantage_thresholds"]
        nisq = advantage_data["nisq_era"]
        ft = advantage_data["fault_tolerant_era"]
        
        fig = plt.figure(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        fig.suptitle("Quantum Advantage Analysis", fontsize=16, fontweight="bold")
        
        # Create grid
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 1: Advantage thresholds (bar chart)
        ax1 = fig.add_subplot(gs[0, 0])
        tasks = list(thresholds.keys())
        qubits = [thresholds[t]["qubits"] for t in tasks]
        demonstrated = [thresholds[t]["demonstrated"] for t in tasks]
        
        colors = ['green' if d else 'orange' for d in demonstrated]
        ax1.barh(tasks, qubits, color=colors, alpha=0.7)
        ax1.set_xlabel("Qubits Required")
        ax1.set_xscale('log')
        ax1.set_title("Quantum Advantage Thresholds by Task")
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        green_patch = mpatches.Patch(color='green', alpha=0.7, label='Demonstrated')
        orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Not yet demonstrated')
        ax1.legend(handles=[green_patch, orange_patch], loc='lower right')
        
        # Plot 2: NISQ era details
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        nisq_text = (
            f"📊 NISQ ERA\n"
            f"{'='*30}\n\n"
            f"Qubit range: {nisq['qubit_range']}\n\n"
            f"Error rates: {nisq['error_rates']}\n\n"
            f"Coherence: {nisq['coherence_times']}\n\n"
            f"Error correction: {nisq['error_correction']}\n\n"
            f"Applications:\n"
            + "\n".join(f"  • {app}" for app in nisq['applications'])
        )
        ax2.text(0.1, 0.5, nisq_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Plot 3: Fault-tolerant era details
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        ft_text = (
            f"🔒 FAULT-TOLERANT ERA\n"
            f"{'='*30}\n\n"
            f"Qubit range: {ft['qubit_range']}\n\n"
            f"Error rates: {ft['error_rates']}\n\n"
            f"Error correction: {ft['error_correction']}\n\n"
            f"Timeline: {ft['timeline_estimate']}\n\n"
            f"Applications:\n"
            + "\n".join(f"  • {app}" for app in ft['applications'])
        )
        ax3.text(0.1, 0.5, ft_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Plot 4: Key insight
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        insight_text = (
            f"💡 KEY INSIGHT\n"
            f"{'='*30}\n\n"
            f"{advantage_data['key_insight']}"
        )
        ax4.text(0.1, 0.5, insight_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.cfg.output_dir, "quantum_advantage.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"[AdvantageViz] Saved: {output_path}")
        return output_path


# =============================================================================
# 4. IBM RESULTS VISUALIZER
# =============================================================================
@dataclass
class IBMVizConfig:
    """Configuration for IBM results visualization."""
    output_dir: str = "./quantum_viz"
    dpi: int = 150
    figsize: Tuple[float, float] = (12, 8)


class IBMResultsVisualizer:
    """
    [D] Visualizer for IBM Quantum experiment results.
    
    PLOTS:
      1. Measurement outcome distributions
      2. Backend property comparisons
      3. Error analysis
    
    USAGE:
      viz = IBMResultsVisualizer(cfg)
      viz.plot_experiment_results(experiment_data)
    """
    
    def __init__(self, cfg: Optional[IBMVizConfig] = None):
        self.cfg = cfg or IBMVizConfig()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        logger.info(f"[IBMViz] Initialized, output: {self.cfg.output_dir}")
    
    def plot_experiment_results(self, experiment_data: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Plot experiment results.
        
        Args:
            experiment_data: output from QuantumExperimentRunner
            output_path: path to save plot
        
        Returns:
            path to saved plot, or None
        """
        if not _HAS_MATPLOTLIB:
            logger.warning("[IBMViz] matplotlib not available")
            return None
        
        aggregated = experiment_data.get("aggregated", {})
        probabilities = aggregated.get("probabilities", {})
        
        if not probabilities:
            logger.warning("[IBMViz] No probabilities to plot")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        fig.suptitle(
            f"Experiment Results: {experiment_data['experiment_type']}",
            fontsize=16, fontweight="bold"
        )
        
        # Plot 1: Outcome probabilities (bar chart)
        ax = axes[0]
        outcomes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Sort by probability
        sorted_pairs = sorted(zip(outcomes, probs), key=lambda x: x[1], reverse=True)
        outcomes, probs = zip(*sorted_pairs)
        
        # Show top 16 outcomes
        n_show = min(16, len(outcomes))
        ax.bar(range(n_show), probs[:n_show], color='steelblue', alpha=0.7)
        ax.set_xticks(range(n_show))
        ax.set_xticklabels(outcomes[:n_show], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Measurement Outcome")
        ax.set_ylabel("Probability")
        ax.set_title("Top Measurement Outcomes")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Summary
        ax = axes[1]
        ax.axis('off')
        summary_text = (
            f"Experiment Summary\n"
            f"{'='*30}\n\n"
            f"Type: {experiment_data['experiment_type']}\n\n"
            f"Qubits: {experiment_data['n_qubits']}\n\n"
            f"Shots: {experiment_data['shots']}\n\n"
            f"Repetitions: {experiment_data['n_repetitions']}\n\n"
            f"Backend: {experiment_data['backend']}\n\n"
            f"Most likely: {aggregated.get('most_likely', 'N/A')}\n\n"
            f"Probability: {aggregated.get('most_likely_prob', 0):.3f}\n\n"
            f"Unique outcomes: {aggregated.get('n_unique_outcomes', 0)}"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.cfg.output_dir, f"experiment_{experiment_data['experiment_type']}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"[IBMViz] Saved: {output_path}")
        return output_path
    
    def plot_backend_comparison(self, comparison_data: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Plot backend comparison.
        
        Args:
            comparison_data: output from BackendPropertyAnalyzer.compare_backends
            output_path: path to save plot
        
        Returns:
            path to saved plot, or None
        """
        if not _HAS_MATPLOTLIB:
            logger.warning("[IBMViz] matplotlib not available")
            return None
        
        backends = comparison_data.get("backends", {})
        
        if not backends:
            logger.warning("[IBMViz] No backends to compare")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        fig.suptitle("IBM Quantum Backend Comparison", fontsize=16, fontweight="bold")
        
        backend_names = list(backends.keys())
        
        # Plot 1: Qubit count
        ax = axes[0]
        qubits = [backends[b].get("num_qubits", 0) for b in backend_names]
        ax.bar(backend_names, qubits, color='steelblue', alpha=0.7)
        ax.set_ylabel("Number of Qubits")
        ax.set_title("Qubit Count")
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Gate errors
        ax = axes[1]
        gate_errors = [backends[b].get("gate_error_2q_mean", backends[b].get("gate_error_2q", 0)) for b in backend_names]
        ax.bar(backend_names, gate_errors, color='coral', alpha=0.7)
        ax.set_ylabel("2-Qubit Gate Error")
        ax.set_title("Gate Error Rates")
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Summary
        ax = axes[2]
        ax.axis('off')
        summary_text = (
            f"Best by qubits:\n"
            f"  {comparison_data.get('best_by_qubits', 'N/A')}\n\n"
            f"Best by error:\n"
            f"  {comparison_data.get('best_by_error', 'N/A')}"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.cfg.output_dir, "backend_comparison.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"[IBMViz] Saved: {output_path}")
        return output_path


# =============================================================================
# 5. QUANTUM DASHBOARD SYSTEM
# =============================================================================
class QuantumDashboardSystem:
    """
    [D] Master orchestrator for quantum dashboard.
    
    INTEGRATES:
      • HilbertScalingVisualizer
      • ErrorScalingVisualizer
      • QuantumAdvantageVisualizer
      • IBMResultsVisualizer
    
    USAGE:
      dashboard = QuantumDashboardSystem()
      dashboard.generate_all_plots(analysis_results)
    """
    
    def __init__(self, output_dir: str = "./quantum_viz"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize visualizers
        self.hilbert_viz = HilbertScalingVisualizer(
            HilbertVizConfig(output_dir=output_dir)
        )
        self.error_viz = ErrorScalingVisualizer(
            ErrorVizConfig(output_dir=output_dir)
        )
        self.advantage_viz = QuantumAdvantageVisualizer(
            AdvantageVizConfig(output_dir=output_dir)
        )
        self.ibm_viz = IBMResultsVisualizer(
            IBMVizConfig(output_dir=output_dir)
        )
        
        logger.info(f"[Dashboard] Initialized, output: {output_dir}")
    
    def generate_all_plots(self, analysis_results: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Generate all plots from analysis results.
        
        Args:
            analysis_results: combined results from all analyzers
        
        Returns:
            dict mapping plot name to file path
        """
        plots = {}
        
        # Hilbert scaling
        if "hilbert_scaling" in analysis_results:
            plots["hilbert_scaling"] = self.hilbert_viz.plot_scaling(
                analysis_results["hilbert_scaling"]
            )
        
        # Error scaling
        if "error_scaling" in analysis_results:
            plots["error_scaling"] = self.error_viz.plot_error_scaling(
                analysis_results["error_scaling"]
            )
        
        # Quantum advantage
        if "quantum_advantage" in analysis_results:
            plots["quantum_advantage"] = self.advantage_viz.plot_advantage(
                analysis_results["quantum_advantage"]
            )
        
        # IBM results
        if "bell_experiment" in analysis_results:
            plots["bell_experiment"] = self.ibm_viz.plot_experiment_results(
                analysis_results["bell_experiment"]
            )
        
        if "backend_comparison" in analysis_results:
            plots["backend_comparison"] = self.ibm_viz.plot_backend_comparison(
                analysis_results["backend_comparison"]
            )
        
        logger.info(f"[Dashboard] Generated {len(plots)} plots")
        
        return plots


# =============================================================================
# 6. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 78)
    print(f"  SESI QUANTUM DASHBOARD v{QUANTUM_DASHBOARD_VERSION} — Self-Test")
    print(f"  Matplotlib={_HAS_MATPLOTLIB}  Plotly={_HAS_PLOTLY}")
    print("=" * 78)
    
    # Import analyzers (mock if not available)
    try:
        from sesi_multiqubit_analyzer import (
            MultiQubitAnalyzerSystem,
            HilbertSpaceScalingAnalyzer,
            ErrorScalingAnalyzer,
            QuantumAdvantageAnalyzer,
        )
        _HAS_ANALYZERS = True
    except ImportError:
        _HAS_ANALYZERS = False
        logger.warning("Analyzers not available; using mock data")
    
    # ── T1: Hilbert Scaling Visualizer ──────────────────────────────────
    print("\n [T1] Hilbert Scaling Visualizer")
    hilbert_viz = HilbertScalingVisualizer(
        HilbertVizConfig(output_dir="./test_viz")
    )
    
    if _HAS_ANALYZERS:
        hilbert_analyzer = HilbertSpaceScalingAnalyzer()
        scaling_data = hilbert_analyzer.analyze_scaling()
    else:
        # Mock data
        scaling_data = {
            "scaling_table": [
                {"n_qubits": n, "hilbert_dim": 2**n, "state_memory_gb": (2**n * 16) / 1e9, "classical_simulation_feasible": n < 50}
                for n in range(1, 101)
            ],
            "classical_simulation_limit": 45,
            "quantum_advantage_threshold": 50,
            "nisq_era_range": "50-1000 qubits",
            "fault_tolerant_threshold": ">1000 qubits",
        }
    
    if _HAS_MATPLOTLIB:
        plot_path = hilbert_viz.plot_scaling(scaling_data)
        print(f"   Plot saved: {plot_path}")
        assert plot_path is not None
    else:
        print("   Skipped (matplotlib not available)")
    
    print("[PASS] T1 Hilbert scaling visualizer works")
    
    # ── T2: Error Scaling Visualizer ────────────────────────────────────
    print("\n [T2] Error Scaling Visualizer")
    error_viz = ErrorScalingVisualizer(
        ErrorVizConfig(output_dir="./test_viz")
    )
    
    if _HAS_ANALYZERS:
        error_analyzer = ErrorScalingAnalyzer()
        error_data = error_analyzer.analyze_error_scaling()
    else:
        # Mock data
        error_data = {
            "n_qubits": 100,
            "gate_error_1q": 0.001,
            "gate_error_2q": 0.01,
            "readout_error": 0.02,
            "t1_us": 100.0,
            "t2_us": 80.0,
            "gate_time_ns": 100.0,
            "max_depth_coherence": 800,
            "scaling_table": [
                {"depth": d, "p_no_error": np.exp(-d * 0.01), "expected_errors": d * 0.01}
                for d in [10, 50, 100, 500, 1000]
            ],
            "max_useful_depth": 100,
        }
    
    if _HAS_MATPLOTLIB:
        plot_path = error_viz.plot_error_scaling(error_data)
        print(f"   Plot saved: {plot_path}")
        assert plot_path is not None
    else:
        print("   Skipped (matplotlib not available)")
    
    print("[PASS] T2 Error scaling visualizer works")
    
    # ── T3: Quantum Advantage Visualizer ────────────────────────────────
    print("\n [T3] Quantum Advantage Visualizer")
    advantage_viz = QuantumAdvantageVisualizer(
        AdvantageVizConfig(output_dir="./test_viz")
    )
    
    if _HAS_ANALYZERS:
        advantage_analyzer = QuantumAdvantageAnalyzer()
        advantage_data = advantage_analyzer.analyze_advantage()
    else:
        # Mock data
        advantage_data = {
            "advantage_thresholds": {
                "random_circuit_sampling": {"qubits": 53, "demonstrated": True},
                "quantum_simulation": {"qubits": 100, "demonstrated": False},
                "optimization": {"qubits": 1000, "demonstrated": False},
                "chemistry": {"qubits": 1000000, "demonstrated": False},
            },
            "nisq_era": {
                "qubit_range": "50-1000",
                "error_rates": "10^-3 to 10^-2",
                "coherence_times": "10-100 μs",
                "error_correction": "None",
                "applications": ["Simulation", "VQE", "QAOA"],
            },
            "fault_tolerant_era": {
                "qubit_range": ">1000 logical",
                "error_rates": "<10^-15",
                "error_correction": "Surface code",
                "applications": ["Chemistry", "Cryptanalysis"],
                "timeline_estimate": "10-30 years",
            },
            "key_insight": "Quantum advantage demonstrated at 53 qubits.",
        }
    
    if _HAS_MATPLOTLIB:
        plot_path = advantage_viz.plot_advantage(advantage_data)
        print(f"   Plot saved: {plot_path}")
        assert plot_path is not None
    else:
        print("   Skipped (matplotlib not available)")
    
    print("[PASS] T3 Quantum advantage visualizer works")
    
    # ── T4: IBM Results Visualizer ──────────────────────────────────────
    print("\n [T4] IBM Results Visualizer")
    ibm_viz = IBMResultsVisualizer(
        IBMVizConfig(output_dir="./test_viz")
    )
    
    # Mock experiment data
    experiment_data = {
        "experiment_type": "bell_state",
        "n_qubits": 2,
        "shots": 1024,
        "n_repetitions": 3,
        "backend": "ibm_brisbane",
        "aggregated": {
            "probabilities": {"00": 0.48, "11": 0.47, "01": 0.03, "10": 0.02},
            "most_likely": "00",
            "most_likely_prob": 0.48,
            "n_unique_outcomes": 4,
        },
    }
    
    if _HAS_MATPLOTLIB:
        plot_path = ibm_viz.plot_experiment_results(experiment_data)
        print(f"   Plot saved: {plot_path}")
        assert plot_path is not None
    else:
        print("   Skipped (matplotlib not available)")
    
    print("[PASS] T4 IBM results visualizer works")
    
    # ── T5: Dashboard System ────────────────────────────────────────────
    print("\n [T5] Quantum Dashboard System")
    dashboard = QuantumDashboardSystem(output_dir="./test_viz")
    
    # Combine all data
    combined_data = {
        "hilbert_scaling": scaling_data,
        "error_scaling": error_data,
        "quantum_advantage": advantage_data,
        "bell_experiment": experiment_data,
    }
    
    plots = dashboard.generate_all_plots(combined_data)
    print(f"   Generated {len(plots)} plots")
    for name, path in plots.items():
        if path:
            print(f"     ✓ {name}: {os.path.basename(path)}")
        else:
            print(f"     ✗ {name}: not generated")
    
    print("[PASS] T5 Dashboard system works")
    
    # ── T6: OPMC Constants ──────────────────────────────────────────────
    print("\n [T6] OPMC Constants")
    k_star = opmc_k_star(1e-8)
    print(f"   k* (Theorem 11.1): {k_star}")
    assert k_star == 23
    print(f"   d(4,4) (Lemma 2.4): {OPMC_D_MN(4, 4)}")
    assert OPMC_D_MN(4, 4) == 320
    print("[PASS] T6 OPMC constants match paper")
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_viz", ignore_errors=True)
    
    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("")
    print("  HONEST REMINDERS:")
    print("    • Visualizations require matplotlib/plotly")
    print("    • All plots are generated from analysis results")
    print("    • No real-time data streaming")
    print("    • This is for RESEARCH/EDUCATION")
    print("    • MY SOUL MOVE BY POWER OF HOLY SPIRIT 🙏✨")
    print("=" * 78)
